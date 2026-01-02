const express = require('express');
const mongoose = require('mongoose');
const router = express.Router();

const { protect } = require('../middleware/auth');
const { uploadFiles, fileUrlFromDiskFilename } = require('../middleware/fileUpload');
const Claim = require('../models/claim');
const AuditLog = require('../models/auditLog');

// GET /api/claims/my
router.get('/my', protect, async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit, 10) || 20, 100);
    const page = Math.max(parseInt(req.query.page, 10) || 1, 1);
    const skip = (page - 1) * limit;

    const [total, claims] = await Promise.all([
      Claim.countDocuments({ owner: req.user._id }),
      Claim.find({ owner: req.user._id })
        .populate('item', 'title type category images')
        .sort({ createdAt: -1 })
        .skip(skip)
        .limit(limit)
        .lean()
    ]);

    res.json({
      success: true,
      // Frontend expects `claims` or `items` in some screens.
      claims,
      items: claims,
      data: claims,
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

// GET /api/claims/:claimId
router.get('/:claimId', protect, async (req, res) => {
  try {
    const { claimId } = req.params;
    if (!mongoose.Types.ObjectId.isValid(claimId)) {
      return res.status(400).json({ success: false, message: 'Invalid claim id' });
    }

    const claim = await Claim.findById(claimId)
      .populate('owner', 'username email')
      .populate('item', 'title type category images')
      .lean();

    if (!claim) return res.status(404).json({ success: false, message: 'Claim not found' });

    const isOwner = String(claim.owner?._id || claim.owner) === String(req.user._id);
    const isAdmin = req.user?.role === 'admin';
    if (!isOwner && !isAdmin) return res.status(403).json({ success: false, message: 'Forbidden' });

    res.json({
      success: true,
      claim,
      item: claim, // compatibility alias
      data: claim
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

// POST /api/claims (multipart: evidence[])
router.post('/', protect, uploadFiles.array('evidence', 6), async (req, res) => {
  try {
    const {
      itemId,
      claimReference,
      claimantName,
      claimantEmail,
      claimantPhone,
      narrative,
      notes
    } = req.body || {};

    const evidence = (req.files || []).map((f) => ({
      url: fileUrlFromDiskFilename(f.filename),
      filename: f.originalname,
      mime: f.mimetype
    }));

    const claim = await Claim.create({
      owner: req.user._id,
      item: itemId || undefined,
      claimReference: claimReference || undefined,
      claimantName: claimantName || undefined,
      claimantEmail: claimantEmail || undefined,
      claimantPhone: claimantPhone || undefined,
      narrative: narrative || undefined,
      notes: notes || undefined,
      evidence
    });

    // Audit (non-blocking)
    AuditLog.create({
      action: 'claim.create',
      actor: req.user?._id,
      targetType: 'Claim',
      targetId: claim._id,
      metadata: { itemId: itemId || null }
    }).catch(() => {});

    res.status(201).json({
      success: true,
      claim,
      item: claim, // compatibility alias
      data: claim
    });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
});

module.exports = router;

