const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

// CLIP is an external dependency (VM / service). For accurate monitoring, we treat it as
// "not configured" unless CLIP_URL is explicitly provided.
const CLIP_BASE = process.env.CLIP_URL;
const CLIP_API_KEY = process.env.CLIP_API_KEY; // optional, if VM requires Bearer auth

function clipHeaders(formData) {
  return {
    ...formData.getHeaders(),
    ...(CLIP_API_KEY ? { Authorization: `Bearer ${CLIP_API_KEY}` } : {}),
  };
}

function getClipBase() {
  return CLIP_BASE || null;
}

async function checkClipHealth({ timeoutMs = 2500 } = {}) {
  if (!CLIP_BASE) {
    return {
      configured: false,
      ok: false,
      message: "CLIP_URL not configured",
    };
  }

  // Try a lightweight health endpoint if the service provides one.
  try {
    const res = await axios.get(`${CLIP_BASE}/health`, {
      timeout: timeoutMs,
      headers: CLIP_API_KEY ? { Authorization: `Bearer ${CLIP_API_KEY}` } : undefined,
      validateStatus: () => true,
    });

    if (res.status >= 200 && res.status < 300) {
      return { configured: true, ok: true, status: res.status };
    }
    // If /health exists but is not OK, treat as unhealthy.
    if (res.status !== 404) {
      return {
        configured: true,
        ok: false,
        status: res.status,
        message: "CLIP health endpoint returned non-OK",
      };
    }
  } catch (err) {
    // network/timeouts fall through to probe call below
  }

  // Fallback probe: call encode-text with a tiny payload.
  try {
    const formData = new FormData();
    formData.append("text", "healthcheck");

    const res = await axios.post(`${CLIP_BASE}/encode-text`, formData, {
      headers: clipHeaders(formData),
      timeout: timeoutMs,
      validateStatus: () => true,
    });

    const embedding = res?.data?.embedding;
    const ok = Array.isArray(embedding) && embedding.length > 0;

    return ok
      ? { configured: true, ok: true, status: res.status }
      : {
          configured: true,
          ok: false,
          status: res.status,
          message: "CLIP probe failed",
        };
  } catch (err) {
    return {
      configured: true,
      ok: false,
      status: err?.response?.status,
      message: err?.response?.data || err.message || "CLIP unreachable",
    };
  }
}

async function getImageEmbedding(imagePath) {
  try {
    if (!CLIP_BASE) {
      console.warn("⚠️ CLIP_URL not configured; skipping image embedding");
      return null;
    }
    const formData = new FormData();
    formData.append("file", fs.createReadStream(imagePath));

    const res = await axios.post(`${CLIP_BASE}/encode-image`, formData, {
      headers: clipHeaders(formData),
      timeout: 120000,
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });

    return res.data.embedding;
  } catch (err) {
    console.error("❌ CLIP image embedding failed:", err.response?.data || err.message);
    return null;
  }
}

async function getTextEmbedding(text) {
  try {
    if (!CLIP_BASE) {
      console.warn("⚠️ CLIP_URL not configured; skipping text embedding");
      return null;
    }
    const formData = new FormData();
    formData.append("text", text);

    const res = await axios.post(`${CLIP_BASE}/encode-text`, formData, {
      headers: clipHeaders(formData),
      timeout: 120000,
    });

    return res.data.embedding;
  } catch (err) {
    console.error("❌ CLIP text embedding failed:", err.response?.data || err.message);
    return null;
  }
}

async function analyzeImage(imagePath, { doOcr = true, doBarcode = true, ocrLang = "eng", ocrPsm = 6 } = {}) {
  try {
    if (!CLIP_BASE) {
      console.warn("⚠️ CLIP_URL not configured; skipping analyze-image");
      return null;
    }
    const formData = new FormData();
    formData.append("file", fs.createReadStream(imagePath));
    formData.append("doOcr", String(doOcr));
    formData.append("doBarcode", String(doBarcode));
    formData.append("ocrLang", String(ocrLang));
    formData.append("ocrPsm", String(ocrPsm));

    const res = await axios.post(`${CLIP_BASE}/analyze-image`, formData, {
      headers: clipHeaders(formData),
      timeout: 120000,
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      validateStatus: () => true,
    });

    if (res.status < 200 || res.status >= 300) {
      return { ok: false, status: res.status, data: res.data };
    }
    return { ok: true, status: res.status, data: res.data };
  } catch (err) {
    console.error("❌ CLIP analyze-image failed:", err.response?.data || err.message);
    return { ok: false, status: err?.response?.status, error: err.message };
  }
}

module.exports = { getClipBase, checkClipHealth, getImageEmbedding, getTextEmbedding, analyzeImage };
