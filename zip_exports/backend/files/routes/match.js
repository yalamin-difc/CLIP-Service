const express = require("express");
const router = express.Router();
const mongoose = require("mongoose");

const { protect } = require("../middleware/auth");
const Item = require("../models/item");
const AuditLog = require("../models/auditLog");

// GET /api/matches?itemId=... (or ?reportId=...)
// Returns stored matches for an item (populated) so the frontend can render topK.
router.get("/", async (req, res) => {
  try {
    const { itemId, reportId, limit = 10 } = req.query;
    const id = itemId || reportId;
    if (!id) {
      return res.status(400).json({
        success: false,
        message: "Provide itemId (or reportId) as query param",
      });
    }
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ success: false, message: "Invalid id" });
    }

    const item = await Item.findById(id)
      .populate({
        path: "matchedItems.item",
        select: "title images type category reporter",
        populate: { path: "reporter", select: "username email" },
      })
      .lean();

    if (!item) return res.status(404).json({ success: false, message: "Item not found" });

    const top = (item.matchedItems || [])
      .slice()
      .sort((a, b) => (b.score || 0) - (a.score || 0))
      .slice(0, Math.min(parseInt(limit, 10) || 10, 50))
      .map((m) => ({
        itemId: m.item?._id || m.item,
        item: m.item,
        score: m.score,
        imageScore: m.imageScore,
        textScore: m.textScore,
        signals: m.signals,
        status: m.status,
        matchedAt: m.matchedAt,
        read: m.read,
      }));

    res.json({
      success: true,
      data: { itemId: id, matches: top },
      matches: top,
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// POST /api/matches/:matchId/accept - Accept a match
router.post("/:matchId/accept", protect, async (req, res) => {
  try {
    const { matchId } = req.params;
    const { itemId } = req.body || {};
    if (!itemId) return res.status(400).json({ success: false, message: "itemId is required" });
    if (!mongoose.Types.ObjectId.isValid(itemId) || !mongoose.Types.ObjectId.isValid(matchId)) {
      return res.status(400).json({ success: false, message: "Invalid id" });
    }

    await Item.updateOne(
      { _id: itemId, "matchedItems.item": matchId },
      { $set: { "matchedItems.$.status": "accepted" } }
    );

    AuditLog.create({
      action: "match.accept",
      actor: req.user?._id,
      targetType: "Item",
      targetId: itemId,
      metadata: { matchId },
    }).catch(() => {});
    
    // Emit socket event for real-time notification
    const io = req.app.get("io");
    io.emit("match-accepted", {
      matchId,
      message: "Match accepted successfully",
      timestamp: new Date().toISOString()
    });
    
    res.json({
      success: true,
      message: "Match accepted successfully",
      matchId
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

// POST /api/matches/:matchId/reject - Reject a match
router.post("/:matchId/reject", protect, async (req, res) => {
  try {
    const { matchId } = req.params;
    const { itemId } = req.body || {};
    if (!itemId) return res.status(400).json({ success: false, message: "itemId is required" });
    if (!mongoose.Types.ObjectId.isValid(itemId) || !mongoose.Types.ObjectId.isValid(matchId)) {
      return res.status(400).json({ success: false, message: "Invalid id" });
    }

    await Item.updateOne(
      { _id: itemId, "matchedItems.item": matchId },
      { $set: { "matchedItems.$.status": "rejected" } }
    );

    AuditLog.create({
      action: "match.reject",
      actor: req.user?._id,
      targetType: "Item",
      targetId: itemId,
      metadata: { matchId },
    }).catch(() => {});
    
    res.json({
      success: true,
      message: "Match rejected successfully",
      matchId
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: error.message
    });
  }
});

module.exports = router;