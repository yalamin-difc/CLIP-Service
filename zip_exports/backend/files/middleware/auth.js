const jwt = require('jsonwebtoken');
const User = require('../models/user');

const protect = async (req, res, next) => {
  let token;
  try {
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer')) {
      token = req.headers.authorization.split(' ')[1];
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      req.user = await User.findById(decoded.id).select('-password');
      if (!req.user) return res.status(401).json({ message: 'User not found' });
      next();
    } else {
      res.status(401).json({ message: 'Not authorized, no token' });
    }
  } catch (err) {
    console.error('Auth error:', err.message);
    res.status(401).json({ message: 'Not authorized, token failed' });
  }
};

/**
 * Optional authentication:
 * - If a Bearer token is present and valid, sets req.user
 * - If no token (or invalid token), continues without failing the request
 */
const optionalAuth = async (req, _res, next) => {
  try {
    const auth = req.headers.authorization || '';
    if (!auth || !auth.startsWith('Bearer ')) return next();
    const token = auth.split(' ')[1];
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(decoded.id).select('-password');
    if (user) req.user = user;
    return next();
  } catch {
    // ignore invalid/expired tokens for public endpoints
    return next();
  }
};

/**
 * Restrict access to specific roles.
 * Usage: router.get("/admin", protect, restrictTo("admin"), handler)
 */
const restrictTo = (...roles) => (req, res, next) => {
  const role = req.user?.role;
  if (!role || !roles.includes(role)) {
    return res.status(403).json({ message: 'Forbidden' });
  }
  next();
};

module.exports = { protect, optionalAuth, restrictTo };