const express = require('express');
const router = express.Router();

const { protect } = require('../middleware/auth');
const Appointment = require('../models/appointment');

// GET /api/appointments/my
router.get('/my', protect, async (req, res) => {
  try {
    const items = await Appointment.find({ owner: req.user._id })
      .populate('item', 'title type category images')
      .sort({ scheduledAt: 1 })
      .lean();
    res.json({
      success: true,
      appointments: items,
      items,
      data: items
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// POST /api/appointments
router.post('/', protect, async (req, res) => {
  try {
    const { itemId, claimId, appointmentRef, site, scheduledAt, slot, time, location, notes } = req.body || {};

    // Frontend sends `slot` like "YYYY-MM-DD HH:mm" (or separate date/time).
    // Accept either `scheduledAt` (ISO) or `slot`.
    const rawSlot = slot || time || null;
    const rawScheduledAt = scheduledAt || null;
    const parsedDate = rawScheduledAt
      ? new Date(rawScheduledAt)
      : rawSlot
        ? new Date(String(rawSlot).replace(' ', 'T'))
        : null;

    if (!parsedDate || Number.isNaN(parsedDate.getTime())) {
      return res.status(400).json({ success: false, message: 'scheduledAt (or slot) is required' });
    }

    const appt = await Appointment.create({
      owner: req.user._id,
      item: itemId || undefined,
      claim: claimId || undefined,
      appointmentRef: appointmentRef || undefined,
      site: site || undefined,
      scheduledAt: parsedDate,
      location: location || undefined,
      slot: rawSlot ? String(rawSlot) : undefined,
      notes: notes || undefined
    });

    res.status(201).json({
      success: true,
      appointment: appt,
      item: appt, // compatibility alias
      data: appt
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;

