const express = require('express');
const mongoose = require('mongoose');
const router = express.Router();

const { protect } = require('../middleware/auth');
const ChatMessage = require('../models/chatMessage');

// GET /api/chat/items/:itemId
router.get('/items/:itemId', protect, async (req, res) => {
  try {
    const { itemId } = req.params;
    if (!mongoose.Types.ObjectId.isValid(itemId)) {
      return res.status(400).json({ success: false, message: 'Invalid item id' });
    }

    const messages = await ChatMessage.find({ item: itemId })
      .populate('sender', 'username email')
      .sort({ createdAt: 1 })
      .lean();

    res.json({
      success: true,
      itemId,
      messages,
      data: {
        itemId,
        messages
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// POST /api/chat/items/:itemId/messages
router.post('/items/:itemId/messages', protect, async (req, res) => {
  try {
    const { itemId } = req.params;
    const { text, message } = req.body || {};
    const finalText = (text || message || '').trim();

    if (!mongoose.Types.ObjectId.isValid(itemId)) {
      return res.status(400).json({ success: false, message: 'Invalid item id' });
    }
    if (!finalText) {
      return res.status(400).json({ success: false, message: 'text is required' });
    }

    const msg = await ChatMessage.create({
      item: itemId,
      sender: req.user._id,
      text: finalText
    });

    const populated = await ChatMessage.findById(msg._id)
      .populate('sender', 'username email')
      .lean();

    // emit real-time update
    try {
      const io = req.app.get('io');
      const room = `item:${itemId}`;
      io?.to?.(room)?.emit?.('chat:message', { room, itemId, message: populated });
    } catch {
      // ignore
    }

    res.status(201).json({
      success: true,
      message: populated,
      data: populated
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;

