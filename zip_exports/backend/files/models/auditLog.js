const mongoose = require('mongoose');

const auditLogSchema = new mongoose.Schema(
  {
    action: { type: String, required: true, trim: true }, // e.g. "pii.reveal"
    actor: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
    targetType: { type: String, trim: true }, // e.g. "Item", "Claim"
    targetId: { type: mongoose.Schema.Types.ObjectId },
    metadata: { type: Object, default: {} }
  },
  { timestamps: true }
);

module.exports = mongoose.model('AuditLog', auditLogSchema);

