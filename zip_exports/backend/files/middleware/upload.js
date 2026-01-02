const multer = require('multer');
const crypto = require('crypto');
const path = require('path');
const sharp = require('sharp');
const fs = require('fs');

const allowed = new Set(['.jpg', '.jpeg', '.png', '.webp']);
const uploadDir = process.env.UPLOAD_DIR || path.join(process.cwd(), 'public', 'uploads');

if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir, { recursive: true });

const storage = multer.memoryStorage();

const fileFilter = (req, file, cb) => {
  const ext = path.extname(file.originalname).toLowerCase();
  if (!allowed.has(ext)) return cb(new Error('Only image files are allowed'), false);
  cb(null, true);
};

const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 5 * 1024 * 1024 }
});

async function processImage(buffer) {
  const name = crypto.randomBytes(12).toString('hex') + '.webp';
  const outPath = path.join(uploadDir, name);
  await sharp(buffer)
    .rotate()
    .resize(1600, 1600, { fit: 'inside' })
    .toFormat('webp', { quality: 80 })
    .toFile(outPath);
  return { filename: name, url: `/uploads/${name}`, mime: 'image/webp' };
}

module.exports = upload;
module.exports.processImage = processImage