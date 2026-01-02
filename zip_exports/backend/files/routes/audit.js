const express = require('express');
const router = express.Router();

const { protect, restrictTo } = require('../middleware/auth');
const AuditLog = require('../models/auditLog');

// GET /api/audit/events (alias for UI scaffolds)
router.get('/events', protect, restrictTo('admin'), async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit, 10) || 20, 100);
    const page = Math.max(parseInt(req.query.page, 10) || 1, 1);
    const skip = (page - 1) * limit;

    const [total, logs] = await Promise.all([
      AuditLog.countDocuments(),
      AuditLog.find()
        .populate('actor', 'username email role')
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .lean()
    ]);

    res.json({
      success: true,
      data: logs,
      events: logs,
      items: logs,
      pagination: {
        total,
        page,
        limit,
        totalPages: Math.max(Math.ceil(total / limit), 1)
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;

