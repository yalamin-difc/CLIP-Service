const express = require("express");
const fs = require("fs");
const path = require("path");
const router = express.Router();

const Item = require("../models/item");
const AuditLog = require("../models/auditLog");
const upload = require("../middleware/upload");
const { processImage } = require("../middleware/upload");
const { protect } = require("../middleware/auth");
const { getImageEmbedding, getTextEmbedding } = require("../utils/clipClient");
const { findPotentialMatches, attachMatchesToItem } = require("../services/matchService");

// ======================================================
// GET /api/reports
// Treat "reports" as item reports in Mongo (Item collection).
// ======================================================
router.get("/", async (req, res) => {
  try {
    const {
      type,
      category,
      search = "",
      limit = 20,
      page = 1,
      startDate,
      endDate,
    } = req.query;

    const limitNum = Math.min(parseInt(limit, 10) || 20, 100);
    const pageNum = Math.max(parseInt(page, 10) || 1, 1);
    const skip = (pageNum - 1) * limitNum;

    const match = {};
    if (type) match.type = type;
    if (category) match.category = category;

    if (search) {
      const regex = new RegExp(search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
      match.$or = [{ title: regex }, { description: regex }, { category: regex }];
    }

    if (startDate || endDate) {
      match.createdAt = {};
      if (startDate) match.createdAt.$gte = new Date(startDate);
      if (endDate) {
        const end = new Date(endDate);
        end.setHours(23, 59, 59, 999);
        match.createdAt.$lte = end;
      }
    }

    const [reports, total] = await Promise.all([
      Item.find(match)
        .populate("reporter", "username email")
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limitNum)
        .lean(),
      Item.countDocuments(match),
    ]);

    res.json({
      success: true,
      reports,
      pagination: {
        total,
        page: pageNum,
        limit: limitNum,
        totalPages: Math.max(Math.ceil(total / limitNum), 1),
      },
      filters: { type, category, search, startDate, endDate },
    });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});

async function createReport(req, res, type) {
  try {
    const {
      title,
      description,
      category,
      address,
      lat,
      lng,
      contactInfo,
      image, // optional URL string if frontend sends it
    } = req.body || {};

    if (!title) {
      return res.status(400).json({ success: false, message: "title is required" });
    }

    // Image handling:
    // - If multipart image is uploaded, we process and store it
    // - Else, if `image` string provided, store it as images[0]
    let images = [];
    let imageEmbedding = [];
    let textEmbedding = [];

    let parsedContactInfo = {};
    if (contactInfo) {
      try {
        parsedContactInfo = typeof contactInfo === "string" ? JSON.parse(contactInfo) : contactInfo;
      } catch {
        parsedContactInfo = { notes: String(contactInfo) };
      }
    }

    if (req.file?.buffer) {
      const processed = await processImage(req.file.buffer);
      images = [processed.url];

      // generate image embedding via temp file
      const tempDir = path.join(__dirname, "..", "public", "temp");
      if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });
      const tempPath = path.join(tempDir, `${Date.now()}_report_clip.jpg`);
      try {
        fs.writeFileSync(tempPath, req.file.buffer);
        const emb = await getImageEmbedding(tempPath);
        imageEmbedding = Array.isArray(emb) ? emb : [];
      } finally {
        if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
      }
    } else if (typeof image === "string" && image.trim()) {
      images = [image.trim()];
    }

    if (description) {
      const emb = await getTextEmbedding(description);
      textEmbedding = Array.isArray(emb) ? emb : [];
    }

    const item = await Item.create({
      title,
      description,
      category,
      type,
      address,
      images,
      location: {
        type: "Point",
        coordinates:
          lat !== undefined && lng !== undefined
            ? [parseFloat(lng), parseFloat(lat)]
            : [0, 0],
      },
      reporter: req.user?._id,
      contactInfo: parsedContactInfo,
      imageEmbedding,
      textEmbedding,
    });

    // audit (non-blocking)
    AuditLog.create({
      action: "report.create",
      actor: req.user?._id,
      targetType: "Item",
      targetId: item._id,
      metadata: { type, category: category || null },
    }).catch(() => {});

    // optional matching attach
    const io = req.app.get("io");
    const matches = await findPotentialMatches(item, { limit: 8 });
    await attachMatchesToItem(item, matches, io, req.user?.email);

    const populated = await Item.findById(item._id)
      .populate("reporter", "username email")
      .populate({
        path: "matchedItems.item",
        select: "title images type category reporter",
        populate: { path: "reporter", select: "username email" },
      });

    res.status(201).json({
      success: true,
      report: populated,
      item: populated,
    });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
}

// ======================================================
// POST /api/reports/lost (JWT required)
// Accepts JSON OR multipart (optional image file field: "image")
// ======================================================
router.post("/lost", protect, upload.single("image"), (req, res) =>
  createReport(req, res, "lost")
);

// ======================================================
// POST /api/reports/found (JWT required)
// Accepts JSON OR multipart (optional image file field: "image")
// ======================================================
router.post("/found", protect, upload.single("image"), (req, res) =>
  createReport(req, res, "found")
);

// ======================================================
// GET /api/reports/:id
// ======================================================
router.get("/:id", async (req, res) => {
  try {
    const report = await Item.findById(req.params.id)
      .populate("reporter", "username email")
      .populate({
        path: "matchedItems.item",
        select: "title images type category reporter",
        populate: { path: "reporter", select: "username email" },
      });

    if (!report) return res.status(404).json({ success: false, message: "Report not found" });
    res.json({ success: true, report, item: report });
  } catch (error) {
    res.status(500).json({ success: false, message: error.message });
  }
});

module.exports = router;