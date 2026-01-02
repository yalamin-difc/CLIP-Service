const fs = require('fs');
const path = require('path');
const Item = require('../models/item');
const { getImageEmbedding, getTextEmbedding } = require('../utils/clipClient');
const { findPotentialMatches, attachMatchesToItem } = require('../services/matchService');

exports.reportItem = async (req, res) => {
  try {
    const {
      type,
      title,
      description,
      category,
      lat,
      lng,
      address,
      contactInfo,
    } = req.body;

    if (!title || !type) {
      return res.status(400).json({
        success: false,
        message: 'Title and type are required.',
      });
    }

    if (!req.file || !req.fileProcessed) {
      return res.status(400).json({
        success: false,
        message: 'An image is required to report an item.',
      });
    }

    let parsedContactInfo = {};
    if (contactInfo) {
      try {
        parsedContactInfo =
          typeof contactInfo === 'string' ? JSON.parse(contactInfo) : contactInfo;
      } catch (err) {
        console.warn('⚠️ Failed to parse contact info:', err.message);
      }
    }

    // Generate CLIP embeddings for the uploaded image/text
    let imageEmbedding = [];
    let textEmbedding = [];

    const tempDir = path.join(__dirname, '..', 'public', 'temp');
    if (!fs.existsSync(tempDir)) fs.mkdirSync(tempDir, { recursive: true });

    const tempPath = path.join(tempDir, `${Date.now()}_${req.file.originalname}`);
    try {
      fs.writeFileSync(tempPath, req.file.buffer);
      imageEmbedding = await getImageEmbedding(tempPath);
    } catch (err) {
      console.error('❌ Failed to generate image embedding:', err);
    } finally {
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
    }

    if (description) {
      try {
        textEmbedding = await getTextEmbedding(description);
      } catch (err) {
        console.error('❌ Failed to generate text embedding:', err);
      }
    }

    const item = await Item.create({
      type,
      title,
      description,
      category,
      location: {
        type: 'Point',
        coordinates:
          lat && lng ? [parseFloat(lng), parseFloat(lat)] : [0, 0],
      },
      address,
      images: [req.fileProcessed.url],
      reporter: req.user._id,
      contactInfo: parsedContactInfo,
      imageEmbedding: Array.isArray(imageEmbedding) ? imageEmbedding : [],
      textEmbedding: Array.isArray(textEmbedding) ? textEmbedding : [],
    });

    await item.populate('reporter', 'username email');

    const matches = await findPotentialMatches(item);
    await attachMatchesToItem(item, matches, req.app.get('io'), req.user?.email);

    res.status(201).json({
      success: true,
      data: await Item.findById(item._id)
        .populate('reporter', 'username email')
        .populate({
          path: 'matchedItems.item',
          select: 'title images type category reporter',
          populate: { path: 'reporter', select: 'username email' },
        }),
      matches: matches.map(({ item: matchItem, score }) => ({
        id: matchItem._id,
        title: matchItem.title,
        type: matchItem.type,
        category: matchItem.category,
        image: matchItem.images?.[0] || null,
        score,
      })),
    });
  } catch (error) {
    console.error('Report item error:', error);
    res.status(500).json({
      success: false,
      message: error.message || 'Failed to report item'
    });
  }
};