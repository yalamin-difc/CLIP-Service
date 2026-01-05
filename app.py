from typing import Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi import Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.responses import Response
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io
import uuid
from datetime import datetime, timezone

import hashlib
import json
import time

from barcode_service import scan_barcodes
from match_service import build_explanation, cosine_similarity, should_return_no_match, softmax_confidences
from ocr_service import extract_ocr
from storage import MongoStore

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
except Exception:  # pragma: no cover
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    Counter = Gauge = Histogram = None
    generate_latest = None

app = FastAPI(title="CLIP Service")

# ---------------------------------------------------------
# CORS (MUST be near the top)
# ---------------------------------------------------------
ALLOWED_ORIGINS = [
    "https://ailostfound.al-amentech.io",
    "https://openailostfound-fccdb129f869.herokuapp.com",
    "http://localhost:3000",
]

# Allows all Vercel previews for this project.
ALLOW_ORIGIN_REGEX = r"^https:\/\/ai-lost-and-found-ver-2-.*\.vercel\.app$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Auth (optional)
# If CLIP_API_KEY is set, require:
#   Authorization: Bearer <CLIP_API_KEY>
# ---------------------------------------------------------
import os

CLIP_API_KEY = os.environ.get("CLIP_API_KEY", "").strip()
MONGODB_URI = os.environ.get("MONGODB_URI", "").strip()
MONGODB_DB = os.environ.get("MONGODB_DB", "clip_service").strip() or "clip_service"

# Confidence calibration / decisioning controls (env-configurable)
CONF_TEMPERATURE = float(os.environ.get("CONF_TEMPERATURE", "0.07"))
CONF_MIN_SCORE = float(os.environ.get("CONF_MIN_SCORE", "0.22"))
CONF_MIN_MARGIN = float(os.environ.get("CONF_MIN_MARGIN", "0.03"))

# Service/model governance metadata (env-configurable)
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "dev").strip() or "dev"
MODEL_ID = os.environ.get("MODEL_ID", "openai/clip-vit-base-patch32").strip() or "openai/clip-vit-base-patch32"

store = MongoStore(MONGODB_URI, db_name=MONGODB_DB)

_ALLOWED_ITEM_STATUSES = {"draft", "released", "archived"}

# ---------------------------------------------------------
# Metrics (Prometheus)
# ---------------------------------------------------------
if Counter is not None:
    METRIC_REQUESTS = Counter("clip_service_requests_total", "Requests", ["endpoint", "status"])
    METRIC_LATENCY = Histogram("clip_service_request_latency_seconds", "Request latency", ["endpoint"])
    METRIC_MODEL_LOADED = Gauge("clip_service_model_loaded", "Model loaded (1/0)")
    METRIC_OCR_OK = Counter("clip_service_ocr_ok_total", "OCR successes", ["lang"])
    METRIC_OCR_FAIL = Counter("clip_service_ocr_fail_total", "OCR failures", ["lang"])
    METRIC_BARCODE_OK = Counter("clip_service_barcode_ok_total", "Barcode scan successes")
    METRIC_BARCODE_FAIL = Counter("clip_service_barcode_fail_total", "Barcode scan failures")
    METRIC_MATCH_NO_MATCH = Counter("clip_service_match_no_match_total", "No-match decisions")
else:
    METRIC_REQUESTS = METRIC_LATENCY = METRIC_MODEL_LOADED = None
    METRIC_OCR_OK = METRIC_OCR_FAIL = None
    METRIC_BARCODE_OK = METRIC_BARCODE_FAIL = None
    METRIC_MATCH_NO_MATCH = None


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not CLIP_API_KEY:
        return  # auth disabled
    if authorization != f"Bearer {CLIP_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------------------------------------------------
# LAZY LOADING (Cloud Run Safe)
# ---------------------------------------------------------
model = None
processor = None

def load_model():
    global model, processor
    if model is None:
        print("⏳ Loading CLIP model on demand...")
        model = CLIPModel.from_pretrained(MODEL_ID)
        processor = CLIPProcessor.from_pretrained(MODEL_ID)
        print("✅ CLIP model ready!")
    return model, processor


@app.get("/metrics")
def metrics():
    if generate_latest is None:
        raise HTTPException(status_code=503, detail="Metrics not available (prometheus-client not installed)")
    if METRIC_MODEL_LOADED is not None:
        METRIC_MODEL_LOADED.set(1 if model is not None else 0)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _upload_to_pil(upload: UploadFile) -> Image.Image:
    try:
        raw = await upload.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image upload") from e


async def _read_upload_bytes(upload: UploadFile) -> bytes:
    try:
        return await upload.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Failed to read upload") from e


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _governance_meta() -> dict:
    return {
        "serviceVersion": SERVICE_VERSION,
        "modelId": MODEL_ID,
        "confidence": {"temperature": CONF_TEMPERATURE, "minScore": CONF_MIN_SCORE, "minMargin": CONF_MIN_MARGIN},
    }


def _normalize_embedding(emb: torch.Tensor) -> torch.Tensor:
    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    return emb


def _request_id_from_header(h: Optional[str]) -> str:
    return (h or "").strip() or str(uuid.uuid4())

# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/")
def home():
    return {"status": "running", "model_loaded": model is not None}


@app.get("/health")
def health():
    """
    Lightweight health probe.

    Notes:
    - Does NOT force model load (keeps probe fast).
    - Mirrors the root `/` health semantics for compatibility with common
      load balancer / uptime check expectations.
    """
    return {"status": "running", "model_loaded": model is not None}

UI_HTML = r"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>CLIP Service UI</title>
    <script src="https://cdn.tailwindcss.com"></script>
  </head>
  <body class="bg-slate-950 text-slate-100 min-h-screen">
    <div class="max-w-5xl mx-auto p-6">
      <div class="flex items-center justify-between gap-4 mb-8">
        <div>
          <h1 class="text-3xl font-semibold tracking-tight">CLIP Service</h1>
          <p class="text-slate-300 mt-1">Interactive UI for text/image embeddings + similarity.</p>
        </div>
        <div class="text-right">
          <div class="text-xs text-slate-400">Backend</div>
          <div id="baseUrl" class="font-mono text-sm text-slate-200"></div>
        </div>
      </div>

      <section class="mb-6 rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
        <div class="flex items-end justify-between gap-4">
          <div>
            <h2 class="text-lg font-semibold">Authorization (optional)</h2>
            <p class="text-sm text-slate-400 mt-1">If the server has <span class="font-mono">CLIP_API_KEY</span> set, paste it here to use the UI.</p>
          </div>
          <div class="text-xs text-slate-500">Header: <span class="font-mono">Authorization: Bearer ...</span></div>
        </div>
        <div class="mt-4 flex flex-col md:flex-row gap-3 items-stretch">
          <input id="tokenInput" type="password" class="flex-1 rounded-xl bg-slate-950/60 border border-slate-800 p-3 outline-none focus:ring-2 focus:ring-indigo-500" placeholder="Bearer token (optional)" />
          <button id="btnSaveToken" class="rounded-xl bg-slate-800 hover:bg-slate-700 px-4 py-2 text-sm font-semibold">Save</button>
          <button id="btnClearToken" class="rounded-xl bg-slate-800 hover:bg-slate-700 px-4 py-2 text-sm font-semibold">Clear</button>
        </div>
        <div id="tokenStatus" class="mt-2 text-xs text-slate-400"></div>
      </section>

      <div class="grid md:grid-cols-2 gap-6">
        <!-- Encode Text -->
        <section class="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 class="text-lg font-semibold">Encode Text</h2>
          <p class="text-sm text-slate-400 mt-1">Returns a normalized embedding vector.</p>
          <div class="mt-4">
            <label class="text-sm text-slate-300">Text</label>
            <textarea id="textInput" rows="4" class="mt-2 w-full rounded-xl bg-slate-950/60 border border-slate-800 p-3 outline-none focus:ring-2 focus:ring-indigo-500" placeholder="e.g. a photo of a cat wearing sunglasses"></textarea>
          </div>
          <div class="mt-4 flex items-center gap-3">
            <button id="btnText" class="rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-sm font-semibold">Encode</button>
            <span id="textStatus" class="text-sm text-slate-400"></span>
          </div>
          <div class="mt-4">
            <div class="text-xs text-slate-400 mb-2">Result</div>
            <pre id="textOut" class="text-xs whitespace-pre-wrap break-words rounded-xl bg-slate-950/60 border border-slate-800 p-3 min-h-[80px]"></pre>
          </div>
        </section>

        <!-- Encode Image -->
        <section class="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
          <h2 class="text-lg font-semibold">Encode Image</h2>
          <p class="text-sm text-slate-400 mt-1">Upload an image, get a normalized embedding vector.</p>
          <div class="mt-4 flex items-start gap-4">
            <div class="flex-1">
              <input id="imgFile" type="file" accept="image/*" class="block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-800 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-slate-700" />
              <div class="mt-4 flex items-center gap-3">
                <button id="btnImg" class="rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-sm font-semibold">Encode</button>
                <span id="imgStatus" class="text-sm text-slate-400"></span>
              </div>
            </div>
            <div class="w-28 h-28 rounded-xl border border-slate-800 bg-slate-950/40 overflow-hidden flex items-center justify-center">
              <img id="imgPreview" alt="" class="hidden w-full h-full object-cover" />
              <div id="imgEmpty" class="text-xs text-slate-500">preview</div>
            </div>
          </div>
          <div class="mt-4">
            <div class="text-xs text-slate-400 mb-2">Result</div>
            <pre id="imgOut" class="text-xs whitespace-pre-wrap break-words rounded-xl bg-slate-950/60 border border-slate-800 p-3 min-h-[80px]"></pre>
          </div>
        </section>
      </div>

      <!-- Similarity -->
      <section class="mt-6 rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
        <div class="flex items-end justify-between gap-4">
          <div>
            <h2 class="text-lg font-semibold">Image Similarity</h2>
            <p class="text-sm text-slate-400 mt-1">Uploads two images and returns cosine similarity.</p>
          </div>
          <div class="text-xs text-slate-400">Endpoint: <span class="font-mono">/similarity</span></div>
        </div>

        <div class="mt-4 grid md:grid-cols-3 gap-4 items-start">
          <div>
            <label class="text-sm text-slate-300">Image 1</label>
            <input id="simFile1" type="file" accept="image/*" class="mt-2 block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-800 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-slate-700" />
            <div class="mt-3 w-full h-36 rounded-xl border border-slate-800 bg-slate-950/40 overflow-hidden flex items-center justify-center">
              <img id="simPrev1" alt="" class="hidden w-full h-full object-cover" />
              <div id="simEmpty1" class="text-xs text-slate-500">preview</div>
            </div>
          </div>

          <div>
            <label class="text-sm text-slate-300">Image 2</label>
            <input id="simFile2" type="file" accept="image/*" class="mt-2 block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-800 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-slate-700" />
            <div class="mt-3 w-full h-36 rounded-xl border border-slate-800 bg-slate-950/40 overflow-hidden flex items-center justify-center">
              <img id="simPrev2" alt="" class="hidden w-full h-full object-cover" />
              <div id="simEmpty2" class="text-xs text-slate-500">preview</div>
            </div>
          </div>

          <div class="md:pt-6">
            <button id="btnSim" class="w-full rounded-xl bg-emerald-600 hover:bg-emerald-500 px-4 py-2 text-sm font-semibold">Compute similarity</button>
            <div id="simStatus" class="mt-3 text-sm text-slate-400"></div>
            <div class="mt-4 rounded-xl bg-slate-950/60 border border-slate-800 p-4">
              <div class="text-xs text-slate-400">Similarity</div>
              <div id="simScore" class="mt-1 text-2xl font-semibold">—</div>
            </div>
          </div>
        </div>
      </section>

      <!-- Analyze Image (CLIP + OCR + Barcode) -->
      <section class="mt-6 rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
        <div class="flex items-end justify-between gap-4">
          <div>
            <h2 class="text-lg font-semibold">Analyze Image (CLIP + OCR + Barcode)</h2>
            <p class="text-sm text-slate-400 mt-1">One-shot endpoint that returns embedding + OCR text + barcode scan.</p>
          </div>
          <div class="text-xs text-slate-400">Endpoint: <span class="font-mono">/analyze-image</span></div>
        </div>

        <div class="mt-4 grid md:grid-cols-3 gap-4 items-start">
          <div class="md:col-span-1">
            <label class="text-sm text-slate-300">Image</label>
            <input id="anFile" type="file" accept="image/*" class="mt-2 block w-full text-sm text-slate-300 file:mr-4 file:rounded-lg file:border-0 file:bg-slate-800 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-slate-100 hover:file:bg-slate-700" />
            <div class="mt-3 w-full h-36 rounded-xl border border-slate-800 bg-slate-950/40 overflow-hidden flex items-center justify-center">
              <img id="anPrev" alt="" class="hidden w-full h-full object-cover" />
              <div id="anEmpty" class="text-xs text-slate-500">preview</div>
            </div>
          </div>

          <div class="md:col-span-1">
            <div class="flex items-center justify-between gap-3">
              <label class="text-sm text-slate-300">Options</label>
              <div class="text-xs text-slate-500">OCR requires Tesseract on server</div>
            </div>
            <div class="mt-3 space-y-3">
              <label class="flex items-center gap-2 text-sm text-slate-300">
                <input id="anDoOcr" type="checkbox" checked class="accent-indigo-500" />
                <span>OCR</span>
              </label>
              <label class="flex items-center gap-2 text-sm text-slate-300">
                <input id="anDoBarcode" type="checkbox" checked class="accent-indigo-500" />
                <span>Barcode</span>
              </label>

              <div class="grid grid-cols-2 gap-3">
                <div>
                  <label class="text-xs text-slate-400">ocrLang</label>
                  <input id="anOcrLang" value="eng" class="mt-1 w-full rounded-xl bg-slate-950/60 border border-slate-800 p-2 text-sm outline-none focus:ring-2 focus:ring-indigo-500" />
                </div>
                <div>
                  <label class="text-xs text-slate-400">ocrPsm</label>
                  <input id="anOcrPsm" value="6" inputmode="numeric" class="mt-1 w-full rounded-xl bg-slate-950/60 border border-slate-800 p-2 text-sm outline-none focus:ring-2 focus:ring-indigo-500" />
                </div>
              </div>
            </div>
          </div>

          <div class="md:col-span-1 md:pt-6">
            <button id="btnAnalyze" class="w-full rounded-xl bg-indigo-600 hover:bg-indigo-500 px-4 py-2 text-sm font-semibold">Analyze</button>
            <div id="anStatus" class="mt-3 text-sm text-slate-400"></div>
            <div class="mt-4">
              <div class="text-xs text-slate-400 mb-2">Result</div>
              <pre id="anOut" class="text-xs whitespace-pre-wrap break-words rounded-xl bg-slate-950/60 border border-slate-800 p-3 min-h-[160px]"></pre>
            </div>
          </div>
        </div>
      </section>

      <footer class="mt-8 text-xs text-slate-500">
        Tip: first request will load the model (can take a bit).
      </footer>
    </div>

    <script>
      const base = window.location.origin;
      document.getElementById("baseUrl").textContent = base;

      // Token storage for UI requests (optional)
      const tokenInput = document.getElementById("tokenInput");
      const tokenStatus = document.getElementById("tokenStatus");
      const KEY = "clip_api_token";

      function getToken() {
        return (localStorage.getItem(KEY) || "").trim();
      }

      function setToken(v) {
        const t = (v || "").trim();
        if (!t) {
          localStorage.removeItem(KEY);
        } else {
          localStorage.setItem(KEY, t);
        }
      }

      function authHeader() {
        const t = getToken();
        return t ? { Authorization: `Bearer ${t}` } : {};
      }

      // Init token UI
      tokenInput.value = getToken();
      tokenStatus.textContent = getToken() ? "Token loaded from this browser." : "No token set (public mode).";
      document.getElementById("btnSaveToken").addEventListener("click", () => {
        setToken(tokenInput.value);
        tokenStatus.textContent = getToken() ? "Token saved." : "Token cleared.";
      });
      document.getElementById("btnClearToken").addEventListener("click", () => {
        tokenInput.value = "";
        setToken("");
        tokenStatus.textContent = "Token cleared.";
      });

      const fmt = (obj) => JSON.stringify(obj, null, 2);
      const clipVec = (v) => {
        if (!Array.isArray(v)) return v;
        const head = v.slice(0, 8).map(x => Number(x).toFixed(6));
        return { length: v.length, head };
      };

      function previewFile(inputEl, imgEl, emptyEl) {
        const f = inputEl.files && inputEl.files[0];
        if (!f) {
          imgEl.classList.add("hidden");
          emptyEl.classList.remove("hidden");
          imgEl.src = "";
          return;
        }
        const url = URL.createObjectURL(f);
        imgEl.src = url;
        imgEl.classList.remove("hidden");
        emptyEl.classList.add("hidden");
      }

      // Encode Text
      document.getElementById("btnText").addEventListener("click", async () => {
        const text = document.getElementById("textInput").value.trim();
        const status = document.getElementById("textStatus");
        const out = document.getElementById("textOut");
        out.textContent = "";
        if (!text) { status.textContent = "Please enter text."; return; }
        status.textContent = "Encoding...";
        try {
          const fd = new FormData();
          fd.append("text", text);
          const r = await fetch(`${base}/encode-text`, { method: "POST", body: fd, headers: authHeader() });
          const body = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
          out.textContent = fmt({ embedding: clipVec(body.embedding) });
          status.textContent = "Done.";
        } catch (e) {
          status.textContent = `Error: ${e.message}`;
        }
      });

      // Encode Image
      const imgFile = document.getElementById("imgFile");
      imgFile.addEventListener("change", () => previewFile(imgFile, document.getElementById("imgPreview"), document.getElementById("imgEmpty")));
      document.getElementById("btnImg").addEventListener("click", async () => {
        const status = document.getElementById("imgStatus");
        const out = document.getElementById("imgOut");
        out.textContent = "";
        const f = imgFile.files && imgFile.files[0];
        if (!f) { status.textContent = "Choose an image first."; return; }
        status.textContent = "Encoding...";
        try {
          const fd = new FormData();
          fd.append("file", f);
          const r = await fetch(`${base}/encode-image`, { method: "POST", body: fd, headers: authHeader() });
          const body = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
          out.textContent = fmt({ embedding: clipVec(body.embedding) });
          status.textContent = "Done.";
        } catch (e) {
          status.textContent = `Error: ${e.message}`;
        }
      });

      // Similarity
      const sim1 = document.getElementById("simFile1");
      const sim2 = document.getElementById("simFile2");
      sim1.addEventListener("change", () => previewFile(sim1, document.getElementById("simPrev1"), document.getElementById("simEmpty1")));
      sim2.addEventListener("change", () => previewFile(sim2, document.getElementById("simPrev2"), document.getElementById("simEmpty2")));

      document.getElementById("btnSim").addEventListener("click", async () => {
        const status = document.getElementById("simStatus");
        const f1 = sim1.files && sim1.files[0];
        const f2 = sim2.files && sim2.files[0];
        const scoreEl = document.getElementById("simScore");
        if (!f1 || !f2) { status.textContent = "Select both images."; return; }
        status.textContent = "Computing...";
        scoreEl.textContent = "—";
        try {
          const fd = new FormData();
          fd.append("file1", f1);
          fd.append("file2", f2);
          const r = await fetch(`${base}/similarity`, { method: "POST", body: fd, headers: authHeader() });
          const body = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);
          scoreEl.textContent = Number(body.similarity).toFixed(6);
          status.textContent = "Done.";
        } catch (e) {
          status.textContent = `Error: ${e.message}`;
        }
      });

      // Analyze Image (CLIP + OCR + Barcode)
      const anFile = document.getElementById("anFile");
      anFile.addEventListener("change", () => previewFile(anFile, document.getElementById("anPrev"), document.getElementById("anEmpty")));

      document.getElementById("btnAnalyze").addEventListener("click", async () => {
        const status = document.getElementById("anStatus");
        const out = document.getElementById("anOut");
        out.textContent = "";

        const f = anFile.files && anFile.files[0];
        if (!f) { status.textContent = "Choose an image first."; return; }

        const doOcr = !!document.getElementById("anDoOcr").checked;
        const doBarcode = !!document.getElementById("anDoBarcode").checked;
        const ocrLang = (document.getElementById("anOcrLang").value || "eng").trim() || "eng";
        const ocrPsm = (document.getElementById("anOcrPsm").value || "6").trim() || "6";

        status.textContent = "Analyzing...";
        try {
          const fd = new FormData();
          fd.append("file", f);
          fd.append("doOcr", String(doOcr));
          fd.append("doBarcode", String(doBarcode));
          fd.append("ocrLang", ocrLang);
          fd.append("ocrPsm", ocrPsm);

          const r = await fetch(`${base}/analyze-image`, { method: "POST", body: fd, headers: authHeader() });
          const body = await r.json().catch(() => ({}));
          if (!r.ok) throw new Error(body?.detail || `HTTP ${r.status}`);

          const ocrText = body?.ocr?.fullText || null;
          const ocrWords = (body?.ocr?.words || []);
          const barcodes = (body?.barcode?.barcodes || []);

          out.textContent = fmt({
            requestId: body.requestId,
            embedding: clipVec(body.embedding),
            ocr: {
              enabled: doOcr,
              fullText: ocrText,
              wordCount: Array.isArray(ocrWords) ? ocrWords.length : 0,
              error: body.ocrError || null,
              meta: body?.ocr?.meta || null,
            },
            barcode: {
              enabled: doBarcode,
              barcodes,
              error: body.barcodeError || null,
              meta: body?.barcode?.meta || null,
            },
            governance: body.governance || null,
          });

          status.textContent = "Done.";
        } catch (e) {
          status.textContent = `Error: ${e.message}`;
        }
      });
    </script>
  </body>
</html>
"""


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return HTMLResponse(UI_HTML)


# ---------------------------------------------------------
# Image Encoding
# ---------------------------------------------------------
@app.post("/encode-image")
async def encode_image(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    model, processor = load_model()

    upload = file or image
    if upload is None:
        raise HTTPException(status_code=400, detail="No file uploaded. Use form field 'file' (or 'image').")

    pil_image = Image.open(io.BytesIO(await upload.read())).convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        embedding = model.get_image_features(**inputs)

    embedding = _normalize_embedding(embedding)
    return {"embedding": embedding.squeeze().tolist()}


# ---------------------------------------------------------
# Text Encoding
# ---------------------------------------------------------
@app.post("/encode-text")
async def encode_text(
    text: Optional[str] = Form(default=None),
    queryText: Optional[str] = Form(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    model, processor = load_model()

    value = (text or queryText or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="No text provided. Use form field 'text' (or 'queryText').")

    inputs = processor(text=[value], return_tensors="pt", padding=True)

    with torch.no_grad():
        embedding = model.get_text_features(**inputs)

    embedding = _normalize_embedding(embedding)
    return {"embedding": embedding.squeeze().tolist()}


# ---------------------------------------------------------
# Image Similarity
# ---------------------------------------------------------
@app.post("/similarity")
async def similarity(
    file1: Optional[UploadFile] = File(default=None),
    file2: Optional[UploadFile] = File(default=None),
    image1: Optional[UploadFile] = File(default=None),
    image2: Optional[UploadFile] = File(default=None),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    model, processor = load_model()

    u1 = file1 or image1
    u2 = file2 or image2
    if u1 is None or u2 is None:
        raise HTTPException(status_code=400, detail="Two images required. Use fields file1/file2 (or image1/image2).")

    img1 = Image.open(u1.file).convert("RGB")
    img2 = Image.open(u2.file).convert("RGB")

    inputs = processor(images=[img1, img2], return_tensors="pt")

    with torch.no_grad():
        emb = model.get_image_features(**inputs)

    emb = _normalize_embedding(emb)
    sim = float(torch.mm(emb[0:1], emb[1:2].T))

    return {"similarity": sim}


# ---------------------------------------------------------
# OCR / Barcode / Match / Items / Audit Logs
# ---------------------------------------------------------
@app.post("/analyze-image")
async def analyze_image(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    doOcr: bool = Form(default=True),
    doBarcode: bool = Form(default=True),
    ocrLang: str = Form(default="eng"),
    ocrPsm: int = Form(default=6),
    authorization: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
):
    """
    One-shot analysis for a single image:
      - CLIP embedding
      - OCR (Tesseract)
      - Barcode scan (ZXing)
    """
    t0 = time.time()
    endpoint = "/analyze-image"
    require_auth(authorization)
    model, processor = load_model()

    upload = file or image
    if upload is None:
        if METRIC_REQUESTS is not None:
            METRIC_REQUESTS.labels(endpoint=endpoint, status="400").inc()
        raise HTTPException(status_code=400, detail="No file uploaded. Use form field 'file' (or 'image').")
    request_id = _request_id_from_header(x_request_id)

    raw = await _read_upload_bytes(upload)
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    input_hash = _sha256_hex(raw)
    input_meta = {"sha256": input_hash, "bytes": len(raw), "contentType": upload.content_type}

    # CLIP
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    embedding = _normalize_embedding(embedding).squeeze().tolist()

    # OCR / Barcode
    ocr = None
    ocr_error = None
    if doOcr:
        try:
            ocr = extract_ocr(pil_image, lang=ocrLang, psm=int(ocrPsm))
            if METRIC_OCR_OK is not None:
                METRIC_OCR_OK.labels(lang=str(ocrLang)).inc()
            store.add_audit_log(
                event_type="OCR_EXTRACT",
                request_id=request_id,
                payload={"meta": ocr.get("meta"), "wordCount": len(ocr.get("words") or []), "fullText": ocr.get("fullText")},
            )
        except Exception as e:
            ocr_error = str(e)
            if METRIC_OCR_FAIL is not None:
                METRIC_OCR_FAIL.labels(lang=str(ocrLang)).inc()
            store.add_audit_log(
                event_type="OCR_EXTRACT_FAILED",
                request_id=request_id,
                payload={"error": ocr_error, "ocrLang": ocrLang, "ocrPsm": int(ocrPsm)},
            )

    barcode = None
    barcode_error = None
    if doBarcode:
        try:
            barcode = scan_barcodes(pil_image)
            if METRIC_BARCODE_OK is not None:
                METRIC_BARCODE_OK.inc()
            store.add_audit_log(
                event_type="BARCODE_SCAN",
                request_id=request_id,
                payload={"meta": barcode.get("meta"), "barcodes": barcode.get("barcodes")},
            )
        except Exception as e:
            barcode_error = str(e)
            if METRIC_BARCODE_FAIL is not None:
                METRIC_BARCODE_FAIL.inc()
            store.add_audit_log(
                event_type="BARCODE_SCAN_FAILED",
                request_id=request_id,
                payload={"error": barcode_error},
            )

    store.add_audit_log(
        event_type="IMAGE_ANALYZE",
        request_id=request_id,
        payload={"input": input_meta, "governance": _governance_meta()},
    )

    if METRIC_LATENCY is not None:
        METRIC_LATENCY.labels(endpoint=endpoint).observe(max(0.0, time.time() - t0))
    if METRIC_REQUESTS is not None:
        METRIC_REQUESTS.labels(endpoint=endpoint, status="200").inc()

    return {
        "requestId": request_id,
        "governance": _governance_meta(),
        "input": input_meta,
        "embedding": embedding,
        "ocr": ocr,
        "ocrError": ocr_error,
        "barcode": barcode,
        "barcodeError": barcode_error,
    }


@app.post("/items", status_code=201)
async def create_item(
    payload: dict = Body(...),
    authorization: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
):
    """
    Create an item (store-only).

    This endpoint does NOT compute CLIP/OCR/barcodes. Use compute endpoints first:
      - /analyze-image (image -> embedding + OCR + barcodes)
      - /encode-image (image -> embedding)
    Then POST the resulting data here to persist it.
    """
    t0 = time.time()
    endpoint = "/items:create"
    require_auth(authorization)
    request_id = _request_id_from_header(x_request_id)

    if not isinstance(payload, dict):
        if METRIC_REQUESTS is not None:
            METRIC_REQUESTS.labels(endpoint=endpoint, status="422").inc()
        raise HTTPException(status_code=422, detail="Request body must be a JSON object.")

    # Required: name
    clean_name = ((payload.get("name") or "") if isinstance(payload.get("name"), str) else "").strip()
    if not clean_name:
        if METRIC_REQUESTS is not None:
            METRIC_REQUESTS.labels(endpoint=endpoint, status="422").inc()
        raise HTTPException(status_code=422, detail="Field 'name' must be a non-empty string.")

    # Required: embedding
    emb = payload.get("clipEmbedding", None)
    if emb is None:
        # Back-compat / internal name
        emb = payload.get("embedding", None)
    if not isinstance(emb, list) or not emb:
        if METRIC_REQUESTS is not None:
            METRIC_REQUESTS.labels(endpoint=endpoint, status="422").inc()
        raise HTTPException(status_code=422, detail="Field 'clipEmbedding' (array of numbers) is required.")
    if len(emb) > 8192:
        raise HTTPException(status_code=422, detail="Field 'clipEmbedding' is too large.")
    try:
        emb = [float(x) for x in emb]
    except Exception as e:
        raise HTTPException(status_code=422, detail="Field 'clipEmbedding' must contain only numbers.") from e

    # Optional: description
    description = payload.get("description")
    if description is not None and not isinstance(description, str):
        raise HTTPException(status_code=422, detail="Field 'description' must be a string.")
    description = (description.strip() if isinstance(description, str) and description.strip() else None)

    # Optional: OCR (either flattened or nested)
    ocr_text = None
    ocr_words = None
    if "ocr" in payload and isinstance(payload.get("ocr"), dict):
        ocr_text = payload["ocr"].get("fullText")
        ocr_words = payload["ocr"].get("words")
    else:
        ocr_text = payload.get("ocrText")
        ocr_words = payload.get("ocrWords")
    if ocr_text is not None and not isinstance(ocr_text, str):
        raise HTTPException(status_code=422, detail="Field 'ocr.fullText' (or 'ocrText') must be a string.")
    if ocr_words is not None and not isinstance(ocr_words, list):
        raise HTTPException(status_code=422, detail="Field 'ocr.words' (or 'ocrWords') must be an array.")

    # Optional: barcodes
    barcodes = payload.get("barcodes")
    if barcodes is not None and not isinstance(barcodes, list):
        raise HTTPException(status_code=422, detail="Field 'barcodes' must be an array.")

    item = store.create_item(
        name=clean_name,
        description=description,
        clip_embedding=emb,
        ocr_text=ocr_text,
        ocr_words=ocr_words,
        barcodes=barcodes,
        status="draft",
    )

    store.add_audit_log(
        event_type="ITEM_CREATED",
        item_id=item["id"],
        request_id=request_id,
        payload={
            "name": item["name"],
            "description": item["description"],
            "status": item["status"],
            "input": {"provided": {"hasOcr": ocr_text is not None or ocr_words is not None, "hasBarcodes": barcodes is not None}},
            "governance": _governance_meta(),
        },
    )
    if ocr_text is not None:
        store.add_audit_log(
            event_type="OCR_EXTRACT",
            item_id=item["id"],
            request_id=request_id,
            payload={"meta": {}, "wordCount": len(ocr_words or []), "fullText": ocr_text},
        )
    if barcodes is not None:
        store.add_audit_log(
            event_type="BARCODE_SCAN",
            item_id=item["id"],
            request_id=request_id,
            payload={"barcodes": barcodes},
        )

    if METRIC_LATENCY is not None:
        METRIC_LATENCY.labels(endpoint=endpoint).observe(max(0.0, time.time() - t0))
    if METRIC_REQUESTS is not None:
        METRIC_REQUESTS.labels(endpoint=endpoint, status="201").inc()

    return {"requestId": request_id, "governance": _governance_meta(), "item": item}


@app.get("/items")
def list_items(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    if status is not None:
        s = status.strip().lower()
        if s and s not in _ALLOWED_ITEM_STATUSES:
            raise HTTPException(status_code=400, detail=f"Invalid status. Use one of: {sorted(_ALLOWED_ITEM_STATUSES)}")
        status = s or None
    return {"items": store.list_items(status=status, limit=limit)}


@app.get("/items/{item_id}")
def get_item(
    item_id: str,
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    try:
        return {"item": store.get_item(item_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")


@app.post("/items/{item_id}/release")
def release_item(
    item_id: str,
    authorization: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
):
    require_auth(authorization)
    request_id = _request_id_from_header(x_request_id)
    try:
        item = store.set_item_status(item_id, status="released", released=True)
    except KeyError:
        raise HTTPException(status_code=404, detail="Item not found")

    store.add_audit_log(
        event_type="ITEM_RELEASE",
        item_id=item_id,
        request_id=request_id,
        payload={"status": "released", "releasedAt": item.get("releasedAt")},
    )
    return {"requestId": request_id, "item": item}


@app.get("/audit-logs")
def list_audit_logs(
    itemId: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    authorization: Optional[str] = Header(default=None),
):
    require_auth(authorization)
    return {"logs": store.list_audit_logs(item_id=itemId, limit=limit, offset=offset)}


@app.post("/match")
async def match_top_k(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
    queryText: Optional[str] = Form(default=None),
    doOcr: bool = Form(default=True),
    doBarcode: bool = Form(default=True),
    ocrLang: str = Form(default="eng"),
    ocrPsm: int = Form(default=6),
    k: int = Form(default=5),
    status: str = Form(default="released"),
    authorization: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-Id"),
):
    """
    Top-K matching against stored items using CLIP cosine similarity.
    Adds explainability signals from:
      - OCR token overlap (queryText vs item.ocrText)
      - Barcode intersection (query image scan vs item barcodes)
    """
    t0 = time.time()
    endpoint = "/match"
    require_auth(authorization)
    model, processor = load_model()
    request_id = _request_id_from_header(x_request_id)

    upload = file or image
    if upload is None:
        if METRIC_REQUESTS is not None:
            METRIC_REQUESTS.labels(endpoint=endpoint, status="400").inc()
        raise HTTPException(status_code=400, detail="No file uploaded. Use form field 'file' (or 'image').")
    raw = await _read_upload_bytes(upload)
    pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    input_hash = _sha256_hex(raw)

    # Query embedding
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        q_emb_t = model.get_image_features(**inputs)
    q_emb = _normalize_embedding(q_emb_t).squeeze().tolist()

    # Optional query OCR / Barcode signals
    q_ocr = None
    if doOcr:
        try:
            q_ocr = extract_ocr(pil_image, lang=ocrLang, psm=int(ocrPsm))
            if METRIC_OCR_OK is not None:
                METRIC_OCR_OK.labels(lang=str(ocrLang)).inc()
            store.add_audit_log(
                event_type="OCR_EXTRACT",
                request_id=request_id,
                payload={"scope": "query", "meta": q_ocr.get("meta"), "wordCount": len(q_ocr.get("words") or []), "fullText": q_ocr.get("fullText")},
            )
        except Exception as e:
            if METRIC_OCR_FAIL is not None:
                METRIC_OCR_FAIL.labels(lang=str(ocrLang)).inc()
            store.add_audit_log(
                event_type="OCR_EXTRACT_FAILED",
                request_id=request_id,
                payload={"scope": "query", "error": str(e), "ocrLang": ocrLang, "ocrPsm": int(ocrPsm)},
            )

    q_barcode = None
    if doBarcode:
        try:
            q_barcode = scan_barcodes(pil_image)
            if METRIC_BARCODE_OK is not None:
                METRIC_BARCODE_OK.inc()
            store.add_audit_log(
                event_type="BARCODE_SCAN",
                request_id=request_id,
                payload={"scope": "query", "barcodes": q_barcode.get("barcodes"), "meta": q_barcode.get("meta")},
            )
        except Exception as e:
            if METRIC_BARCODE_FAIL is not None:
                METRIC_BARCODE_FAIL.inc()
            store.add_audit_log(event_type="BARCODE_SCAN_FAILED", request_id=request_id, payload={"scope": "query", "error": str(e)})

    # Candidate items
    k = max(1, min(int(k), 50))
    candidates = store.list_item_embeddings(status=status, limit=5000)

    scored = []
    for it in candidates:
        s = cosine_similarity(q_emb, it.get("embedding") or [])
        scored.append((s, it))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:k]

    sims = [float(s) for s, _ in top]
    confs = softmax_confidences(sims, temperature=CONF_TEMPERATURE)

    # No-match decisioning
    top_score = sims[0] if sims else 0.0
    second_score = sims[1] if len(sims) > 1 else None
    no_match, no_match_meta = should_return_no_match(
        top_score=top_score,
        second_score=second_score,
        min_score=CONF_MIN_SCORE,
        min_margin=CONF_MIN_MARGIN,
    )
    if no_match and METRIC_MATCH_NO_MATCH is not None:
        METRIC_MATCH_NO_MATCH.inc()

    results = []
    for (s, it), conf in zip(top, confs):
        expl = build_explanation(
            similarity=float(s),
            query_text=(queryText or (q_ocr or {}).get("fullText")),
            item_ocr_text=it.get("ocrText"),
            query_barcodes=(q_barcode or {}).get("barcodes"),
            item_barcodes=it.get("barcodes"),
        )
        results.append(
            {
                "item": {
                    "id": it.get("id"),
                    "name": it.get("name"),
                    "description": it.get("description"),
                    "status": it.get("status"),
                    "ocrText": it.get("ocrText"),
                    "barcodes": it.get("barcodes") or [],
                },
                "score": float(s),
                "confidence": float(conf),
                "explanation": expl,
            }
        )

    store.add_audit_log(
        event_type="AI_MATCH_GENERATION",
        request_id=request_id,
        payload={
            "k": k,
            "statusFilter": status,
            "queryText": (queryText or None),
            "resultIds": [r["item"]["id"] for r in results],
            "input": {"sha256": input_hash, "bytes": len(raw), "contentType": upload.content_type},
            "decisioning": {"noMatch": no_match, "meta": no_match_meta},
            "governance": _governance_meta(),
        },
    )

    if METRIC_LATENCY is not None:
        METRIC_LATENCY.labels(endpoint=endpoint).observe(max(0.0, time.time() - t0))
    if METRIC_REQUESTS is not None:
        METRIC_REQUESTS.labels(endpoint=endpoint, status="200").inc()

    return {
        "requestId": request_id,
        "governance": _governance_meta(),
        "decisioning": {"noMatch": no_match, "meta": no_match_meta},
        "query": {
            "embedding": q_emb,
            "queryText": queryText,
            "ocr": q_ocr,
            "barcode": q_barcode,
        },
        "topK": ([] if no_match else results),
    }
