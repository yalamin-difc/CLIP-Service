const mongoose = require('mongoose');

const chatMessageSchema = new mongoose.Schema(
  {
    item: { type: mongoose.Schema.Types.ObjectId, ref: 'Item', required: true },
    sender: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    text: { type: String, required: true, trim: true, maxlength: 4000 }
  },
  { timestamps: true }
);

chatMessageSchema.index({ item: 1, createdAt: -1 });

module.exports = mongoose.model('ChatMessage', chatMessageSchema);

