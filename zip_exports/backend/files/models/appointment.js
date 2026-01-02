const mongoose = require('mongoose');

const appointmentSchema = new mongoose.Schema(
  {
    owner: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    item: { type: mongoose.Schema.Types.ObjectId, ref: 'Item' }, // optional
    claim: { type: mongoose.Schema.Types.ObjectId, ref: 'Claim' }, // optional
    appointmentRef: { type: String, trim: true, index: true },
    site: { type: String, trim: true },
    scheduledAt: { type: Date, required: true },
    location: { type: String, trim: true },
    // Human-friendly slot string (frontend may submit/display this directly).
    slot: { type: String, trim: true },
    status: {
      type: String,
      enum: ['scheduled', 'completed', 'cancelled'],
      default: 'scheduled'
    },
    notes: { type: String, trim: true }
  },
  { timestamps: true }
);

module.exports = mongoose.model('Appointment', appointmentSchema);

