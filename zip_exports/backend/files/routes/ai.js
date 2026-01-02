// routes/ai.js
const express = require("express");
const fs = require("fs");
const path = require("path");
const router = express.Router();
const multer = require("multer");
const upload = multer({ storage: multer.memoryStorage() });
const { checkClipHealth, getClipBase, getImageEmbedding, getTextEmbedding } = require("../utils/clipClient");
const { findPotentialMatches } = require("../services/matchService");
const { protect, restrictTo } = require("../middleware/auth");

function cosineSimilarity(vecA = [], vecB = []) {
  if (!Array.isArray(vecA) || !Array.isArray(vecB)) return 0;
  if (!vecA.length || vecA.length !== vecB.length) return 0;

  const dot = vecA.reduce((sum, val, idx) => sum + val * vecB[idx], 0);
  const magA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
  const magB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
  if (!magA || !magB) return 0;
  return dot / (magA * magB);
}

async function getEmbeddingFromUpload(file) {
  if (!file) return null;

  const tempDir = path.join(__dirname, "..", "public", "temp");
  if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });

  const tempPath = path.join(tempDir, `${Date.now()}_${file.originalname || "upload"}`);
  try {
    fs.writeFileSync(tempPath, file.buffer);
    return await getImageEmbedding(tempPath);
  } finally {
    if (fs.existsSync(tempPath)) {
      fs.unlinkSync(tempPath);
    }
  }
}

async function handleCombinedMatch(req, res) {
  try {
    let imageEmbedding = null;
    let textEmbedding = null;

    if (req.file) {
      imageEmbedding = await getEmbeddingFromUpload(req.file);
    }

    const queryText = req.body.text || req.body.queryText;
    if (queryText) {
      textEmbedding = await getTextEmbedding(queryText);
    }

    if (!imageEmbedding && !textEmbedding) {
      return res.status(400).json({
        success: false,
        message: "Provide an image or text to search",
      });
    }

    const matches = await findPotentialMatches(
      {
        _id: null,
        type: req.body.type || "lost",
        imageEmbedding,
        textEmbedding,
      },
      { limit: 8 }
    );

    res.json({
      success: true,
      matches: matches.map(({ item, score, imageScore, textScore, signals }) => ({
        id: item._id,
        title: item.title,
        image: item.images?.[0] || null,
        score,
        imageScore,
        textScore,
        signals,
        // Avoid leaking PII; use /api/pii/reveal for audited break-glass access.
        reporter: item.reporter?._id
          ? { _id: item.reporter._id, username: item.reporter.username }
          : null,
      })),
    });
  } catch (err) {
    console.error("❌ CLIP service error:", err.message);
    res.status(500).json({ error: "CLIP match request failed", details: err.message });
  }
}

// ======================================================
// Matching (canonical)
// ======================================================
// POST /api/ai/match (canonical)
router.post("/match", upload.single("image"), handleCombinedMatch);

// POST /api/ai/combined-match (alias; kept for backwards compatibility)
router.post("/combined-match", upload.single("image"), (req, res) => {
  res.set("Deprecation", "true");
  res.set("Link", "</api/ai/match>; rel=\"canonical\"");
  return handleCombinedMatch(req, res);
});

// GET /api/ai/health (frontend compatibility)
router.get("/health", async (_req, res) => {
  const base = getClipBase();
  const health = await checkClipHealth({ timeoutMs: 2500 });

  // If CLIP isn't configured at all, return 404 so UI can show "Not Configured".
  if (!health.configured) {
    return res.status(404).json({
      success: false,
      configured: false,
      message: health.message || "CLIP not configured",
    });
  }

  // If configured but failing, return 503 so UI can show "Degraded".
  if (!health.ok) {
    return res.status(503).json({
      success: false,
      configured: true,
      clipBase: base,
      message: health.message || "CLIP unhealthy",
      status: health.status,
    });
  }

  return res.json({
    success: true,
    configured: true,
    clipBase: base,
  });
});

// POST /api/ai/chat (integration stub)
router.post("/chat", async (req, res) => {
  const { message, prompt, text } = req.body || {};
  const input = message || prompt || text || "";
  res.json({
    success: true,
    data: {
      configured: false,
      input,
      reply: "AI chat is not configured on this backend yet"
    }
  });
});

// POST /api/ai/analyze (debug/demo; admin-only)
router.post("/analyze", protect, restrictTo("admin"), upload.single("image"), async (req, res) => {
  try {
    const hasFile = Boolean(req.file);
    const text = req.body.text || req.body.queryText;

    if (!hasFile && !text) {
      return res.status(400).json({
        success: false,
        message: "Provide an image or text prompt to analyse.",
      });
    }

    const imageEmbedding = hasFile ? await getEmbeddingFromUpload(req.file) : null;
    const textEmbedding = text ? await getTextEmbedding(text) : null;

    const similarity =
      Array.isArray(imageEmbedding) && Array.isArray(textEmbedding)
        ? cosineSimilarity(imageEmbedding, textEmbedding)
        : null;

    res.json({
      success: true,
      similarity,
      imageEmbedding,
      textEmbedding,
    });
  } catch (err) {
    console.error("❌ CLIP analysis failed:", err);
    res.status(500).json({ success: false, message: "CLIP analysis failed" });
  }
});

module.exports = router;