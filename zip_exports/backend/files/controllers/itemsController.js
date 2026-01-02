const Item = require('../models/item');
const upload = require('../middleware/upload');
const { processImage } = require('../middleware/upload');

// 🖼️ Middleware: handle multiple image uploads & Sharp processing
const handleImagesUpload = async (req, res, next) => {
  try {
    if (req.files && req.files.length > 0) {
      const results = [];
      for (const file of req.files) {
        const processed = await processImage(file.buffer);
        results.push(processed); // { filename, url, mime }
      }
      req.filesProcessed = results;
    }
    next();
  } catch (err) {
    console.error('❌ Image processing failed:', err);
    next(err);
  }
};

// 🆕 Controller: create a new item (assigns logged-in user as reporter)
const createItem = async (req, res) => {
  try {
    const {
      title,
      description,
      type,
      category,
      address,
      date,
      location
    } = req.body;

    const images = req.filesProcessed ? req.filesProcessed.map(f => f.url) : [];

    // ✅ Attach reporter from authenticated user (from authMiddleware)
    const item = await Item.create({
      title,
      description,
      type,
      category,
      address,
      date,
      location,
      images,
      reporter: req.user?._id, // important!
    });

    // ✅ Populate reporter info in response
    await item.populate('reporter', 'username email');

    res.status(201).json({
      success: true,
      message: 'Item created successfully',
      item,
    });
  } catch (err) {
    console.error('❌ Error creating item:', err);
    res.status(500).json({ success: false, message: err.message });
  }
};

// 📦 Controller: get all items (for Browse & Home page)
const getItems = async (req, res) => {
  try {
    const items = await Item.find()
      .populate('reporter', 'username email')
      .sort({ createdAt: -1 });

    res.json({ success: true, items });
  } catch (err) {
    console.error('❌ Error fetching items:', err);
    res.status(500).json({ success: false, message: err.message });
  }
};

// 📦 Controller: get single item by ID
const getItemById = async (req, res) => {
  try {
    const item = await Item.findById(req.params.id)
      .populate('reporter', 'username email');
    if (!item) {
      return res.status(404).json({ success: false, message: 'Item not found' });
    }
    res.json({ success: true, item });
  } catch (err) {
    console.error('❌ Error fetching item:', err);
    res.status(500).json({ success: false, message: err.message });
  }
};

module.exports = {
  upload,              // used in routes
  handleImagesUpload,  // sharp processor
  createItem,
  getItems,
  getItemById,
}