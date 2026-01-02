const express = require('express');
const router = express.Router();

// POST /api/communications/sms
// Integration stub: accepts payload and returns "queued".
router.post('/sms', async (req, res) => {
  const { to, message } = req.body || {};
  res.json({
    success: true,
    data: {
      status: 'queued',
      to: to || null,
      message: message || null
    }
  });
});

module.exports = router;

