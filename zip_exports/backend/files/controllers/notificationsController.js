const mongoose = require('mongoose');
const Item = require('../models/item');

exports.getUserNotifications = async (req, res) => {
  try {
    // Get items reported by this user
    const userItems = await Item.find({ reporter: req.user._id }).lean();
    const notifications = [];

    for (const userItem of userItems) {
      if (!Array.isArray(userItem.matchedItems) || userItem.matchedItems.length === 0) {
        continue;
      }

      const matchesById = new Map();
      userItem.matchedItems.forEach((matchMeta) => {
        const id = matchMeta?.item || matchMeta;
        if (!id) return;
        matchesById.set(id.toString(), matchMeta);
      });

      if (matchesById.size === 0) continue;

      const matchedIds = Array.from(matchesById.keys()).map((id) => new mongoose.Types.ObjectId(id));

      const matchedItems = await Item.find({
        _id: { $in: matchedIds },
      })
        .populate('reporter', 'username email')
        .lean();

      matchedItems.forEach((matchedItem) => {
        const meta = matchesById.get(matchedItem._id.toString()) || {};

        notifications.push({
          type: 'match',
          message: `Potential match found for your ${userItem.type} item: ${userItem.title}`,
          item: matchedItem,
          score: meta.score,
          matchedAt: meta.matchedAt,
          read: Boolean(meta.read),
        });
      });
    }

    res.json({
      success: true,
      data: notifications
    });
  } catch (error) {
    console.error('Get notifications error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to fetch notifications'
    });
  }
};

exports.markAsRead = async (req, res) => {
  try {
    const { itemId, matchItemId } = req.body;

    if (!itemId || !matchItemId) {
      return res.status(400).json({
        success: false,
        message: 'itemId and matchItemId are required',
      });
    }

    await Item.updateOne(
      { _id: itemId, reporter: req.user._id, 'matchedItems.item': matchItemId },
      { $set: { 'matchedItems.$.read': true } }
    );

    res.json({
      success: true,
      message: 'Notification marked as read'
    });
  } catch (error) {
    console.error('Mark as read error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to mark notification as read'
    });
  }
};

exports.getUnreadCount = async (req, res) => {
  try {
    const userItems = await Item.find({ reporter: req.user._id }).lean();
    let unreadCount = 0;

    for (const item of userItems) {
      if (!Array.isArray(item.matchedItems)) continue;
      unreadCount += item.matchedItems.filter((match) => !match.read).length;
    }

    res.json({
      success: true,
      data: { unreadCount }
    });
  } catch (error) {
    console.error('Get unread count error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to get unread count'
    });
  }
}