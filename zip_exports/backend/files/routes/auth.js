const express = require('express');
const router = express.Router();
const { register, login, me, changePassword, uaePassStart, uaePassExchange } = require('../controllers/authController');
const rateLimit = require('express-rate-limit');
const { body } = require('express-validator');
const { protect } = require('../middleware/auth'); // ✅ import the protect middleware

// Rate limiter for auth routes
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,                 // limit each IP to 100 requests per windowMs
  standardHeaders: true,    // return rate limit info in the `RateLimit-*` headers
  legacyHeaders: false,     // disable the `X-RateLimit-*` headers
});

// Register
router.post(
  '/register',
  authLimiter,
  body('email').isEmail(),
  body('password').isLength({ min: 8 }),
  body('username').isLength({ min: 3 }),
  register
);

// Login
router.post(
  '/login',
  authLimiter,
  body('email').isEmail(),
  body('password').isLength({ min: 8 }),
  login
);

// Current user
router.get('/me', protect, me);

// Change password
router.post(
  '/change-password',
  protect,
  body('currentPassword').isLength({ min: 1 }),
  body('newPassword').isLength({ min: 8 }),
  changePassword
);

// UAE PASS (stubs)
router.get('/uaepass/start', uaePassStart);
router.post('/uaepass/exchange', uaePassExchange);

module.exports = router;