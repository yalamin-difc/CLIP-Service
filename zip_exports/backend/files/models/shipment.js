const mongoose = require('mongoose');

const shipmentSchema = new mongoose.Schema(
  {
    owner: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
    item: { type: mongoose.Schema.Types.ObjectId, ref: 'Item' }, // optional
    status: {
      type: String,
      enum: ['created', 'in_transit', 'delivered', 'cancelled'],
      default: 'created'
    },
    carrier: { type: String, trim: true },
    trackingNumber: { type: String, trim: true },
    address: { type: String, trim: true }
  },
  { timestamps: true }
);

module.exports = mongoose.model('Shipment', shipmentSchema);

