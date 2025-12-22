/**
 * Item model (OCR + barcode + CLIP embedding)
 *
 * This repo’s runtime is Python/FastAPI, but many deployments store items in MongoDB.
 * This file provides a ready-to-use Mongoose schema mirroring the backend fields.
 */

const mongoose = require("mongoose");

const BBoxSchema = new mongoose.Schema(
  {
    x: { type: Number, required: true },
    y: { type: Number, required: true },
    w: { type: Number, required: true },
    h: { type: Number, required: true },
  },
  { _id: false }
);

const OcrWordSchema = new mongoose.Schema(
  {
    text: { type: String, required: true },
    conf: { type: Number, default: 0 },
    bbox: { type: BBoxSchema, required: true },
    pageNum: { type: Number, default: 0 },
    blockNum: { type: Number, default: 0 },
    parNum: { type: Number, default: 0 },
    lineNum: { type: Number, default: 0 },
    wordNum: { type: Number, default: 0 },
  },
  { _id: false }
);

const OcrSchema = new mongoose.Schema(
  {
    fullText: { type: String, default: "" },
    words: { type: [OcrWordSchema], default: [] },
    meta: {
      lang: { type: String, default: "eng" },
      psm: { type: Number, default: 6 },
    },
  },
  { _id: false }
);

const BarcodeSchema = new mongoose.Schema(
  {
    text: { type: String, default: "" },
    format: { type: String, default: "" },
    contentType: { type: String, default: "" },
    position: { type: mongoose.Schema.Types.Mixed, default: null },
  },
  { _id: false }
);

const ItemSchema = new mongoose.Schema(
  {
    name: { type: String, required: true, trim: true },
    description: { type: String, default: "", trim: true },
    status: { type: String, enum: ["draft", "released", "archived"], default: "draft", index: true },

    // Stored CLIP embedding vector (normalized)
    clipEmbedding: { type: [Number], required: true },

    // OCR results
    ocr: { type: OcrSchema, default: () => ({}) },

    // Barcode scan results
    barcodes: { type: [BarcodeSchema], default: [] },

    releasedAt: { type: Date, default: null, index: true },
  },
  { timestamps: true }
);

module.exports = mongoose.models.Item || mongoose.model("Item", ItemSchema);

