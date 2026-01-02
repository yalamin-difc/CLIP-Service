const express = require("express");
const fs = require("fs");
const path = require("path");
const router = express.Router();
const Item = require("../models/item");
const AuditLog = require("../models/auditLog");
const upload = require("../middleware/upload");
const { processImage } = require("../middleware/upload");
const { analyzeImage, getImageEmbedding, getTextEmbedding } = require("../utils/clipClient");
const { protect, optionalAuth } = require("../middleware/auth");
const { findPotentialMatches, attachMatchesToItem } = require("../services/matchService");

// ===============================================
// 🧩 Create new item (authenticated, with images)
// ===============================================
router.post("/", protect, upload.array("images", 4), async (req, res) => {
  try {
    const { title, description, category, type, clientReference, address, lat, lng, contactInfo } = req.body;

    if (!title || !type) {
      return res.status(400).json({
        success: false,
        message: "Title and type are required.",
      });
    }

    if (!req.files || req.files.length === 0) {
      return res.status(400).json({
        success: false,
        message: "At least one image is required.",
      });
    }

    // 1️⃣ Process uploaded images
    const processedImages = [];
    for (const file of req.files) {
      const processed = await processImage(file.buffer);
      processedImages.push(processed);
    }

    // 2️⃣ Generate CLIP embeddings
    let imageEmbedding = [];
    let textEmbedding = [];
    let parsedContactInfo = {};

    if (contactInfo) {
      try {
        parsedContactInfo = typeof contactInfo === "string" ? JSON.parse(contactInfo) : contactInfo;
      } catch (error) {
        console.warn("⚠️ Failed to parse contact info, skipping:", error.message);
      }
    }

    // Extract image embedding + optional OCR/barcodes (via CLIP-Service analyze-image)
    let ocrText = null;
    let barcodes = [];
    if (processedImages[0]) {
      const tempDir = path.join(__dirname, "..", "public", "temp");
      if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });
      const tempPath = path.join(tempDir, `${Date.now()}_clip.jpg`);
      try {
        fs.writeFileSync(tempPath, req.files[0].buffer);
        imageEmbedding = await getImageEmbedding(tempPath);

        const analysis = await analyzeImage(tempPath, { doOcr: true, doBarcode: true });
        if (analysis?.ok) {
          ocrText = analysis?.data?.ocr?.fullText || null;
          const rawBarcodes = analysis?.data?.barcode?.barcodes || [];
          barcodes = Array.isArray(rawBarcodes)
            ? rawBarcodes.map((b) => ({
                text: (b?.text || "").toString(),
                format: (b?.format || "").toString(),
                contentType: (b?.contentType || "").toString()
              }))
            : [];
        }
      } finally {
        if (fs.existsSync(tempPath)) {
          fs.unlinkSync(tempPath);
        }
      }
    }

    if (description) {
      textEmbedding = await getTextEmbedding(description);
    }

    if (!Array.isArray(imageEmbedding)) {
      imageEmbedding = [];
    }

    if (!Array.isArray(textEmbedding)) {
      textEmbedding = [];
    }

    // 3️⃣ Save to MongoDB
    const item = await Item.create({
      title,
      description,
      category,
      type,
      clientReference: clientReference || undefined,
      address,
      images: processedImages.map((img) => img.url),
      location: {
        type: "Point",
        coordinates:
          lat && lng ? [parseFloat(lng), parseFloat(lat)] : [0, 0],
      },
      imageEmbedding,
      textEmbedding,
      ocrText: ocrText || undefined,
      barcodes: Array.isArray(barcodes) && barcodes.length ? barcodes : [],
      reporter: req.user?._id, // ✅ attach logged-in user
      contactInfo: parsedContactInfo,
    });

    // Audit (non-blocking)
    AuditLog.create({
      action: "item.create",
      actor: req.user?._id,
      targetType: "Item",
      targetId: item._id,
      metadata: { type: item.type, category: item.category || null },
    }).catch(() => {});

    const io = req.app.get("io");
    const matches = await findPotentialMatches(item);
    const sanitizedMatches = matches.map(({ item: matchItem, score, imageScore, textScore, signals }) => ({
      id: matchItem._id,
      title: matchItem.title,
      type: matchItem.type,
      category: matchItem.category,
      image: matchItem.images?.[0] || null,
      score,
      imageScore,
      textScore,
      signals,
    }));

    await attachMatchesToItem(item, matches, io, req.user?.email);

    res.status(201).json({
      success: true,
      item: await Item.findById(item._id)
        // Only return minimal reporter fields on create; PII reveal is audited separately.
        .populate("reporter", "username")
        .populate({
          path: "matchedItems.item",
          select: "title images type category reporter",
          populate: { path: "reporter", select: "username" },
        }),
      matches: sanitizedMatches,
    });
  } catch (err) {
    console.error("❌ Error creating item:", err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// ===============================================
// 🧠 CLIP AI Matching Route
// ===============================================
router.post("/ai-match", upload.single("image"), async (req, res) => {
  try {
    // Backwards-compatible alias. Canonical endpoint is POST /api/ai/match.
    res.set("Deprecation", "true");
    res.set("Link", "</api/ai/match>; rel=\"canonical\"");

    const queryText = req.body.queryText || req.body.text;

    let imageEmbedding = null;
    let textEmbedding = null;

    if (req.file) {
      const tempDir = path.join(__dirname, "..", "public", "temp");
      if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });
      const tempPath = path.join(tempDir, `${Date.now()}_${req.file.originalname}`);
      try {
        fs.writeFileSync(tempPath, req.file.buffer);
        imageEmbedding = await getImageEmbedding(tempPath);
      } finally {
        if (fs.existsSync(tempPath)) {
          fs.unlinkSync(tempPath);
        }
      }
    }

    if (queryText) {
      textEmbedding = await getTextEmbedding(queryText);
    }

    if (!imageEmbedding && !textEmbedding) {
      return res.status(400).json({
        success: false,
        message: "No image or text provided for matching.",
      });
    }

    const dummyItem = {
      _id: null,
      type: req.body.type || "lost",
      imageEmbedding,
      textEmbedding,
    };

    const matches = await findPotentialMatches(dummyItem, { limit: 8 });

    res.json({
      success: true,
      deprecated: true,
      canonical: "/api/ai/match",
      matches: matches.map(({ item, score, imageScore, textScore, signals }) => ({
        id: item._id,
        title: item.title,
        image: item.images?.[0] || null,
        score,
        imageScore,
        textScore,
        signals,
        reporter: item.reporter?._id
          ? { _id: item.reporter._id, username: item.reporter.username }
          : null,
      })),
    });
  } catch (err) {
    console.error("❌ AI match failed:", err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// ===============================================
// 📜 Get paginated items (with reporter info & filters)
// ===============================================
router.get("/", optionalAuth, async (req, res) => {
  try {
    const {
      limit = 12,
      page = 1,
      search = "",
      category,
      type,
      startDate,
      endDate,
      lat,
      lng,
      radius = 10,
    } = req.query;

    const limitNum = Math.min(parseInt(limit, 10) || 12, 50);
    const pageNum = Math.max(parseInt(page, 10) || 1, 1);
    const skip = (pageNum - 1) * limitNum;
    const latNum = lat !== undefined ? parseFloat(lat) : null;
    const lngNum = lng !== undefined ? parseFloat(lng) : null;
    const radiusNum = Math.min(Math.max(parseFloat(radius) || 10, 1), 200);

    const match = {};

    if (type) match.type = type;
    if (category) match.category = category;

    if (search) {
      const regex = new RegExp(search.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
      match.$or = [
        { title: regex },
        { description: regex },
        { category: regex },
      ];
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

    const useGeo = Number.isFinite(latNum) && Number.isFinite(lngNum);
    let items;
    let total = 0;

    if (useGeo) {
      const radiusMeters = radiusNum * 1000;
      const pipeline = [
        {
          $geoNear: {
            near: {
              type: "Point",
              coordinates: [lngNum, latNum],
            },
            distanceField: "distance",
            maxDistance: radiusMeters,
            spherical: true,
            query: match,
          },
        },
        { $sort: { createdAt: -1 } },
        {
          $facet: {
            items: [
              { $skip: skip },
              { $limit: limitNum },
              {
                $lookup: {
                  from: "users",
                  let: { reporterId: "$reporter" },
                  pipeline: [
                    {
                      $match: {
                        $expr: { $eq: ["$_id", "$$reporterId"] },
                      },
                    },
                    { $project: { username: 1 } },
                  ],
                  as: "reporter",
                },
              },
              { $unwind: { path: "$reporter", preserveNullAndEmptyArrays: true } },
            ],
            totalCount: [{ $count: "count" }],
          },
        },
      ];

      const [result] = await Item.aggregate(pipeline);
      items = (result?.items || []).map((item) => ({
        ...item,
        distanceKm:
          typeof item.distance === "number"
            ? Math.round((item.distance / 1000) * 100) / 100
            : undefined,
      }));
      total = result?.totalCount?.[0]?.count || 0;
    } else {
      [items, total] = await Promise.all([
        Item.find(match)
          .populate("reporter", "username")
          .sort({ createdAt: -1 })
          .skip(skip)
          .limit(limitNum)
          .lean(),
        Item.countDocuments(match),
      ]);
    }

    // Strip PII for public list views. Only the reporter themselves should see contactInfo.
    const viewerId = req.user?._id ? String(req.user._id) : null;
    items = (items || []).map((it) => {
      const reporterId = it?.reporter?._id ? String(it.reporter._id) : (it?.reporter ? String(it.reporter) : null);
      const isOwner = Boolean(viewerId && reporterId && viewerId === reporterId);
      if (!isOwner) {
        delete it.contactInfo;
        if (it.reporter && typeof it.reporter === "object") {
          delete it.reporter.email;
          delete it.reporter.phone;
          delete it.reporter.mobile;
        }
      }
      return it;
    });

    res.json({
      success: true,
      items,
      pagination: {
        total,
        page: pageNum,
        limit: limitNum,
        totalPages: Math.max(Math.ceil(total / limitNum), 1),
      },
      filters: {
        search,
        category,
        type,
        startDate,
        endDate,
        lat: useGeo ? latNum : undefined,
        lng: useGeo ? lngNum : undefined,
        radius: radiusNum,
      },
    });
  } catch (err) {
    console.error("❌ Failed to fetch items:", err);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

// ===============================================
// ✏️ Update an item (authenticated; reporter or admin)
// ===============================================
router.patch("/:id", protect, async (req, res) => {
  try {
    const item = await Item.findById(req.params.id);
    if (!item) {
      return res.status(404).json({ success: false, message: "Item not found" });
    }

    const isOwner = String(item.reporter) === String(req.user._id);
    const isAdmin = req.user?.role === "admin";
    if (!isOwner && !isAdmin) {
      return res.status(403).json({ success: false, message: "Forbidden" });
    }

    const {
      title,
      description,
      category,
      type,
      clientReference,
      address,
      contactInfo,
      shipment,
      lat,
      lng,
    } = req.body || {};

    if (title !== undefined) item.title = title;
    if (description !== undefined) item.description = description;
    if (category !== undefined) item.category = category;
    if (type !== undefined) item.type = type;
    if (clientReference !== undefined) item.clientReference = clientReference;
    if (address !== undefined) item.address = address;

    if (contactInfo !== undefined) {
      let parsed = contactInfo;
      if (typeof contactInfo === "string") {
        try {
          parsed = JSON.parse(contactInfo);
        } catch {
          parsed = { notes: String(contactInfo) };
        }
      }
      item.contactInfo = parsed || {};
    }

    // Shipment tracking (used in ItemDetail for found items).
    if (shipment !== undefined) {
      const s = shipment && typeof shipment === "object" ? shipment : {};
      item.shipment = {
        provider: (s.provider || "").toString().trim() || undefined,
        trackingNumber: (s.trackingNumber || "").toString().trim() || undefined,
        status: (s.status || "").toString().trim() || undefined,
        updatedAt: s.updatedAt ? new Date(s.updatedAt) : new Date(),
      };
    }

    const latNum = lat !== undefined ? parseFloat(lat) : null;
    const lngNum = lng !== undefined ? parseFloat(lng) : null;
    if (Number.isFinite(latNum) && Number.isFinite(lngNum)) {
      item.location = {
        type: "Point",
        coordinates: [lngNum, latNum],
      };
    }

    await item.save();

    // Audit (non-blocking)
    AuditLog.create({
      action: "item.update",
      actor: req.user?._id,
      targetType: "Item",
      targetId: item._id,
      metadata: { fields: Object.keys(req.body || {}) },
    }).catch(() => {});

    const populated = await Item.findById(item._id)
      .populate("reporter", "username")
      .populate({
        path: "matchedItems.item",
        select: "title images type category reporter",
        populate: { path: "reporter", select: "username" },
      });

    res.json({ success: true, item: populated });
  } catch (err) {
    console.error("❌ Error updating item:", err);
    res.status(500).json({ success: false, message: err.message });
  }
});

// ===============================================
// 🔍 Single item with match metadata
// ===============================================
router.get("/:id", optionalAuth, async (req, res) => {
  try {
    const item = await Item.findById(req.params.id)
      .populate("reporter", "username")
      .populate({
        path: "matchedItems.item",
        select: "title images type category reporter",
        populate: { path: "reporter", select: "username" },
      });

    if (!item) {
      return res.status(404).json({ success: false, message: "Item not found" });
    }

    // Strip PII unless the viewer is the reporter/owner.
    const viewerId = req.user?._id ? String(req.user._id) : null;
    const reporterId = item?.reporter?._id ? String(item.reporter._id) : null;
    const isOwner = Boolean(viewerId && reporterId && viewerId === reporterId);

    const safe = item.toObject ? item.toObject() : item;
    if (!isOwner) {
      delete safe.contactInfo;
      if (safe.reporter && typeof safe.reporter === "object") {
        delete safe.reporter.email;
        delete safe.reporter.phone;
        delete safe.reporter.mobile;
      }
    }

    res.json({ success: true, item: safe });
  } catch (error) {
    console.error("❌ Failed to fetch item:", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
});

module.exports = router;