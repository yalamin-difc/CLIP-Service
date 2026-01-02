const multer = require('multer');
const crypto = require('crypto');
const path = require('path');
const fs = require('fs');

const uploadDir =
  process.env.UPLOAD_DIR || path.join(process.cwd(), 'public', 'uploads');
const filesDir = path.join(uploadDir, 'files');

if (!fs.existsSync(filesDir)) fs.mkdirSync(filesDir, { recursive: true });

const storage = multer.diskStorage({
  destination: function (_req, _file, cb) {
    cb(null, filesDir);
  },
  filename: function (_req, file, cb) {
    const ext = path.extname(file.originalname || '').toLowerCase();
    const name = crypto.randomBytes(12).toString('hex') + ext;
    cb(null, name);
  }
});

const uploadFiles = multer({
  storage,
  limits: { fileSize: 15 * 1024 * 1024 } // 15MB
});

function fileUrlFromDiskFilename(filename) {
  return `/uploads/files/${filename}`;
}

module.exports = { uploadFiles, fileUrlFromDiskFilename };

