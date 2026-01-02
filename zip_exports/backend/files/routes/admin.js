const express = require('express');
const router = express.Router();

const { protect, restrictTo } = require('../middleware/auth');
const User = require('../models/user');
const Item = require('../models/item');
const AuditLog = require('../models/auditLog');
const Claim = require('../models/claim');

// GET /api/admin/dashboard
router.get('/dashboard', protect, restrictTo('admin'), async (_req, res) => {
  try {
    const now = new Date();
    const SLA_HOURS = 48;
    const slaCutoff = new Date(now.getTime() - SLA_HOURS * 60 * 60 * 1000);

    const [
      users,
      itemsTotal,
      itemsLost,
      itemsFound,
      itemsMatched,
      claimsTotal,
      claimsSubmitted,
      claimsInReview,
      claimsApproved,
      claimsRejected,
      claimsSlaRisk,
      recentClaims
    ] = await Promise.all([
      User.countDocuments(),
      Item.countDocuments(),
      Item.countDocuments({ type: 'lost' }),
      Item.countDocuments({ type: 'found' }),
      Item.countDocuments({ 'matchedItems.0': { $exists: true } }),
      Claim.countDocuments(),
      Claim.countDocuments({ status: 'submitted' }),
      Claim.countDocuments({ status: 'in_review' }),
      Claim.countDocuments({ status: 'approved' }),
      Claim.countDocuments({ status: 'rejected' }),
      Claim.countDocuments({
        status: { $in: ['submitted', 'in_review'] },
        createdAt: { $lte: slaCutoff }
      }),
      Claim.find()
        .sort({ createdAt: -1 })
        .limit(20)
        .populate('owner', 'username email')
        .populate('item', 'title type category images')
        .lean()
    ]);

    const counts = {
      users,
      itemsTotal,
      itemsLost,
      itemsFound,
      itemsMatched,
      claimsTotal,
      claimsSubmitted,
      claimsInReview,
      claimsApproved,
      claimsRejected
    };

    const queues = {
      newIntake: claimsSubmitted,
      inReview: claimsInReview,
      matched: itemsMatched,
      claimPending: claimsSubmitted,
      disputed: claimsRejected,
      slaRisk: claimsSlaRisk
    };

    const worklist = recentClaims.map((c) => ({
      id: c._id,
      type: 'claim',
      status: c.status,
      createdAt: c.createdAt,
      owner: c.owner,
      item: c.item
    }));

    const kpis = {
      intake: { count: claimsSubmitted },
      backlog: { count: claimsSubmitted + claimsInReview },
      resolutionRate: {
        percent:
          claimsTotal > 0
            ? Math.round(((claimsApproved + claimsRejected) / claimsTotal) * 1000) / 10
            : 0
      },
      slaBreaches: { count: claimsSlaRisk }
    };

    res.json({
      success: true,
      // Top-level aliases (many UIs expect these at the root)
      counts,
      queues,
      worklist,
      kpis,

      // Canonical nested payload
      data: {
        counts,
        queues,
        worklist,
        kpis,
        meta: {
          slaHours: SLA_HOURS,
          generatedAt: now.toISOString()
        }
      }
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// GET /api/admin/audit-logs?limit=&page=
router.get('/audit-logs', protect, restrictTo('admin'), async (req, res) => {
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

// GET /api/admin/items/:itemId/custody
// Placeholder: custody chain is not modeled yet; endpoint exists for frontend integration.
router.get('/items/:itemId/custody', protect, restrictTo('admin'), async (req, res) => {
  res.json({
    success: true,
    data: {
      itemId: req.params.itemId,
      custody: []
    }
  });
});

// GET /api/admin/verify/audit
router.get('/verify/audit', protect, restrictTo('admin'), async (_req, res) => {
  res.json({ success: true, data: { ok: true } });
});

module.exports = router;

