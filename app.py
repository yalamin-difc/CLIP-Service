from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi import Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import io

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
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("✅ CLIP model ready!")
    return model, processor

# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------
@app.get("/")
def home():
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

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
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

    embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
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

    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
    sim = float(torch.mm(emb[0:1], emb[1:2].T))

    return {"similarity": sim}
