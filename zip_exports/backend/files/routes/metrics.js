const express = require('express');
const router = express.Router();

const { protect, restrictTo } = require('../middleware/auth');
const Item = require('../models/item');
const Claim = require('../models/claim');

// GET /api/metrics/home
// Executive dashboard KPIs (non-PII).
router.get('/home', protect, restrictTo('admin'), async (req, res) => {
  try {
    const now = new Date();
    const { startDate, endDate } = req.query || {};

    const start = startDate ? new Date(startDate) : new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);
    const end = endDate ? new Date(endDate) : now;

    const SLA_HOURS = 48;
    const slaCutoff = new Date(now.getTime() - SLA_HOURS * 60 * 60 * 1000);

    const [intake, backlog, resolved, slaBreaches] = await Promise.all([
      // Intake: new claims in selected period
      Claim.countDocuments({ createdAt: { $gte: start, $lte: end } }),
      // Backlog: open claims awaiting action
      Claim.countDocuments({ status: { $in: ['submitted', 'in_review'] } }),
      // Resolved within selected period (approved/rejected)
      Claim.countDocuments({
        status: { $in: ['approved', 'rejected'] },
        updatedAt: { $gte: start, $lte: end }
      }),
      // SLA breaches: open claims older than cutoff
      Claim.countDocuments({
        status: { $in: ['submitted', 'in_review'] },
        createdAt: { $lte: slaCutoff }
      })
    ]);

    const resolutionRate = intake > 0 ? Math.round((resolved / intake) * 1000) / 10 : 0; // 0.0–100.0

    // A couple of useful item KPIs for drill-downs
    const [itemsLost, itemsFound, itemsMatched] = await Promise.all([
      Item.countDocuments({ type: 'lost' }),
      Item.countDocuments({ type: 'found' }),
      Item.countDocuments({ 'matchedItems.0': { $exists: true } })
    ]);

    res.json({
      success: true,
      data: {
        period: { start: start.toISOString(), end: end.toISOString() },
        kpis: {
          intake: { count: intake },
          backlog: { count: backlog },
          resolutionRate: { percent: resolutionRate },
          slaBreaches: { count: slaBreaches }
        },
        items: { lost: itemsLost, found: itemsFound, matched: itemsMatched }
      },

      // Extra aliases for frontend scaffolds (so cards render even if shape differs)
      intake: { cases: intake },
      backlog: { cases: backlog },
      resolutionRate: { percent: resolutionRate },
      slaBreaches: { cases: slaBreaches },
      cards: [
        {
          key: 'intake',
          title: 'INTAKE',
          value: intake,
          unit: 'cases',
          description: 'New cases in selected period.'
        },
        {
          key: 'backlog',
          title: 'BACKLOG',
          value: backlog,
          unit: 'cases',
          description: 'Open cases awaiting action.'
        },
        {
          key: 'resolutionRate',
          title: 'RESOLUTION RATE',
          value: resolutionRate,
          unit: '%',
          description: 'Closed/returned within period.'
        },
        {
          key: 'slaBreaches',
          title: 'SLA BREACHES',
          value: slaBreaches,
          unit: 'cases',
          description: 'Cases past SLA thresholds.'
        }
      ]
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;

