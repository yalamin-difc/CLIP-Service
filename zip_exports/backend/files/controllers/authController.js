const jwt = require('jsonwebtoken');
const User = require('../models/user');
const { validationResult } = require('express-validator');
const { sendWelcomeEmail } = require('../services/emailservice'); // ✅ import mailer

function signToken(id) {
  return jwt.sign(
    { id },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
}

function signTokenWithClaims(user) {
  // Include minimal claims the frontend uses for gating (role + identity).
  // Backend MUST still enforce authz regardless of these UI hints.
  return jwt.sign(
    {
      id: user.id,
      role: user.role || 'user',
      identityProvider: user.identityProvider || 'password',
      identityVerified: Boolean(user.identityVerified),
    },
    process.env.JWT_SECRET,
    { expiresIn: process.env.JWT_EXPIRES_IN || '7d' }
  );
}

exports.register = async (req, res, next) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty())
      return res.status(400).json({ success: false, errors: errors.array() });

    const { username, email, password, phone } = req.body;

    const user = await User.create({
      username,
      email,
      password,
      phone,
      identityProvider: 'password',
      identityVerified: false,
    });

    // 🔔 Send welcome email asynchronously (don’t block response)
    sendWelcomeEmail({ username, email })
      .catch(err => console.error('❌ Failed to send welcome email:', err));

    const token = signTokenWithClaims(user);
    res.status(201).json({
      success: true,
      token,
      user: user.toJSON()
    });
  } catch (err) {
    if (err.code === 11000)
      return res.status(409).json({
        success: false,
        message: 'Username or email already in use'
      });
    next(err);
  }
};

exports.login = async (req, res, next) => {
  try {
    const { email, password } = req.body;
    const user = await User.findOne({ email }).select('+password');
    if (!user)
      return res.status(401).json({ success: false, message: 'Invalid credentials' });

    const ok = await user.comparePassword(password);
    if (!ok)
      return res.status(401).json({ success: false, message: 'Invalid credentials' });

    const token = signTokenWithClaims(user);
    res.json({ success: true, token, user: user.toJSON() });
  } catch (err) {
    next(err);
  }
};

exports.me = async (req, res) => {
  const user = await User.findById(req.user.id);
  res.json({ success: true, user });
};

// POST /api/auth/change-password (requires JWT)
exports.changePassword = async (req, res) => {
  try {
    const { currentPassword, newPassword } = req.body || {};
    if (!currentPassword || !newPassword) {
      return res.status(400).json({
        success: false,
        message: 'currentPassword and newPassword are required'
      });
    }
    if (String(newPassword).length < 8) {
      return res.status(400).json({
        success: false,
        message: 'newPassword must be at least 8 characters'
      });
    }

    const user = await User.findById(req.user.id).select('+password');
    if (!user) return res.status(401).json({ success: false, message: 'User not found' });

    const ok = await user.comparePassword(currentPassword);
    if (!ok) {
      return res.status(401).json({ success: false, message: 'Current password is incorrect' });
    }

    user.password = newPassword;
    await user.save();

    res.json({ success: true, message: 'Password updated' });
  } catch (err) {
    res.status(500).json({ success: false, message: err.message });
  }
};

// UAE PASS integration stubs (endpoints exist for frontend integration)
exports.uaePassStart = async (req, res) => {
  const { returnTo } = req.query || {};
  const rt = (returnTo || '').toString();

  // Minimal end-to-end workflow support:
  // - If UAEPASS is not configured, allow a *mock* flow in dev via UAEPASS_MOCK_ENABLED=1
  // - Frontend expects redirect to /uaepass/callback with either token or code.
  const mockEnabled = String(process.env.UAEPASS_MOCK_ENABLED || '') === '1';
  if (!mockEnabled) {
    return res.json({
      success: true,
      data: {
        configured: false,
        returnTo: rt || null,
        message: 'UAE PASS is not configured on this backend yet'
      }
    });
  }

  if (!rt) {
    return res.status(400).json({ success: false, message: 'returnTo is required for mock UAEPASS flow' });
  }

  // Redirect with mock code (frontend will exchange it).
  const nextUrl = new URL(rt);
  nextUrl.searchParams.set('code', `mock_${Date.now()}`);
  nextUrl.searchParams.set('next', '/citizen');
  return res.redirect(nextUrl.toString());
};

exports.uaePassExchange = async (req, res) => {
  const { code } = req.body || {};
  const mockEnabled = String(process.env.UAEPASS_MOCK_ENABLED || '') === '1';

  if (!mockEnabled) {
    return res.json({
      success: true,
      data: {
        configured: false,
        code: code || null,
        message: 'UAE PASS is not configured on this backend yet'
      }
    });
  }

  // Mock: create/find a verified user and issue a JWT containing identity claims.
  // In production, this should validate the code with UAE Pass and map to a real identity.
  const email = `uaepass.mock.${String(code || 'user').replace(/[^a-z0-9._-]/gi, '')}@example.com`.toLowerCase();
  const username = `uaepass_${String(code || 'user').slice(-10)}`;

  let user = await User.findOne({ email }).select('-password');
  if (!user) {
    // Create with a random password (not used), mark identity verified.
    const randomPw = `mock_${Math.random().toString(36).slice(2)}_${Date.now()}`;
    user = await User.create({
      username,
      email,
      password: randomPw,
      role: 'user',
      identityProvider: 'uaepass',
      identityVerified: true,
    });
  } else {
    user.identityProvider = 'uaepass';
    user.identityVerified = true;
    await user.save();
  }

  const token = signTokenWithClaims(user);
  return res.json({
    success: true,
    token,
    user: user.toJSON ? user.toJSON() : user,
    data: { configured: false, mock: true }
  });
};