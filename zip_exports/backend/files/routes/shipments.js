const express = require('express');
const mongoose = require('mongoose');
const router = express.Router();

const { protect } = require('../middleware/auth');
const Shipment = require('../models/shipment');

// GET /api/shipments/my
router.get('/my', protect, async (req, res) => {
  try {
    const items = await Shipment.find({ owner: req.user._id })
      .populate('item', 'title type category images')
      .sort({ createdAt: -1 })
      .lean();
    res.json({
      success: true,
      shipments: items,
      items,
      data: items
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// GET /api/shipments/:id
router.get('/:id', protect, async (req, res) => {
  try {
    const { id } = req.params;
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ success: false, message: 'Invalid shipment id' });
    }

    const shipment = await Shipment.findById(id)
      .populate('owner', 'username email')
      .populate('item', 'title type category images')
      .lean();
    if (!shipment) return res.status(404).json({ success: false, message: 'Shipment not found' });

    const isOwner = String(shipment.owner?._id || shipment.owner) === String(req.user._id);
    const isAdmin = req.user?.role === 'admin';
    if (!isOwner && !isAdmin) return res.status(403).json({ success: false, message: 'Forbidden' });

    res.json({
      success: true,
      shipment,
      item: shipment, // compatibility alias
      data: shipment
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// POST /api/shipments
router.post('/', protect, async (req, res) => {
  try {
    const { itemId, carrier, trackingNumber, address } = req.body || {};
    const shipment = await Shipment.create({
      owner: req.user._id,
      item: itemId || undefined,
      carrier: carrier || undefined,
      trackingNumber: trackingNumber || undefined,
      address: address || undefined
    });
    res.status(201).json({
      success: true,
      shipment,
      item: shipment, // compatibility alias
      data: shipment
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;

