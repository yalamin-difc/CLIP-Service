const mongoose = require('mongoose');

const claimSchema = new mongoose.Schema(
  {
    owner: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    item: { type: mongoose.Schema.Types.ObjectId, ref: 'Item' }, // optional
    // External/customer-facing reference (UI shows this prominently).
    claimReference: { type: String, trim: true, index: true },
    status: {
      type: String,
      enum: ['submitted', 'in_review', 'approved', 'rejected'],
      default: 'submitted'
    },
    // Claimant supplied information (used in citizen workflow).
    claimantName: { type: String, trim: true },
    claimantEmail: { type: String, trim: true },
    claimantPhone: { type: String, trim: true },
    narrative: { type: String, trim: true },
    // Backwards-compatible/freeform notes (optional)
    notes: { type: String, trim: true },
    evidence: [
      {
        url: { type: String, required: true },
        filename: { type: String, required: true },
        mime: { type: String, trim: true }
      }
    ]
  },
  { timestamps: true }
);

module.exports = mongoose.model('Claim', claimSchema);

