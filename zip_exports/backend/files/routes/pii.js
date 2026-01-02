const express = require('express');
const router = express.Router();

const { protect, restrictTo } = require('../middleware/auth');
const AuditLog = require('../models/auditLog');
const Item = require('../models/item');

// POST /api/pii/reveal (admin only; audited)
router.post('/reveal', protect, restrictTo('admin'), async (req, res) => {
  try {
    const { caseId, itemId, reason } = req.body || {};
    if (!itemId || !reason) {
      return res.status(400).json({
        success: false,
        message: 'itemId and reason are required'
      });
    }

    const item = await Item.findById(itemId)
      .populate('reporter', 'username email phone')
      .lean();
    if (!item) {
      return res.status(404).json({ success: false, message: 'Item not found' });
    }

    const contact = {
      email: item.contactInfo?.email || item.reporter?.email || null,
      phone: item.contactInfo?.phone || item.reporter?.phone || null,
      notes: item.contactInfo?.notes || null
    };

    await AuditLog.create({
      action: 'pii.reveal',
      actor: req.user?._id,
      targetType: 'Item',
      targetId: item._id,
      metadata: { caseId: caseId || null, reason }
    });

    res.json({
      success: true,
      data: {
        caseId: caseId || null,
        itemId: String(item._id),
        contact
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;

