const express = require('express');
const {
  getUserNotifications,
  markAsRead,
  getUnreadCount
} = require('../controllers/notificationsController');
const { protect } = require('../middleware/auth');

const router = express.Router();

router.get('/', protect, getUserNotifications);
router.put('/read', protect, markAsRead);
router.get('/unread-count', protect, getUnreadCount);

module.exports = router;