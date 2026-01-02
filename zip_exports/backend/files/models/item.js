const mongoose = require("mongoose");

const itemSchema = new mongoose.Schema(
  {
    title: { type: String, required: true },
    description: { type: String },
    category: { type: String },
    type: { type: String, enum: ["lost", "found"], required: true },
    // Customer-facing reference number (generated in frontend today).
    clientReference: { type: String, trim: true, index: true },
    address: { type: String },
    images: [{ type: String }],
    location: {
      type: { type: String, enum: ["Point"], default: "Point" },
      coordinates: { type: [Number], default: [0, 0] },
    },
    reporter: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true, // ensures user is linked
    },
    // Shipment tracking (used by ItemDetail for found items).
    shipment: {
      provider: { type: String, trim: true },
      trackingNumber: { type: String, trim: true },
      status: { type: String, trim: true },
      updatedAt: { type: Date }
    },
    contactInfo: {
      email: { type: String, trim: true },
      phone: { type: String, trim: true },
      notes: { type: String, trim: true },
    },
    imageEmbedding: { type: [Number], default: [] }, // CLIP image vector
    textEmbedding: { type: [Number], default: [] },  // CLIP text vector
    // Optional OCR + barcode signals (from CLIP-Service /analyze-image)
    ocrText: { type: String, trim: true },
    barcodes: [
      {
        text: { type: String, trim: true },
        format: { type: String, trim: true },
        contentType: { type: String, trim: true }
      }
    ],
    matchedItems: [
      {
        item: { type: mongoose.Schema.Types.ObjectId, ref: "Item" },
        score: { type: Number, default: 0 },
        // Match report (for explainability / UI)
        imageScore: { type: Number, default: 0 },
        textScore: { type: Number, default: 0 },
        signals: [{ type: String }], // e.g. ["image","text"]
        status: {
          type: String,
          enum: ["suggested", "accepted", "rejected"],
          default: "suggested",
        },
        matchedAt: { type: Date, default: Date.now },
        read: { type: Boolean, default: false },
      },
    ],
  },
  { timestamps: true }
);

// ✅ Add geospatial index
itemSchema.index({ location: "2dsphere" });

// ✅ Export the model only
module.exports = mongoose.model("Item", itemSchema);