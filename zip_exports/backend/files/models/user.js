const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const validator = require('validator');

const UserSchema = new mongoose.Schema({
  username: {
    type: String, required: true, unique: true, trim: true, minlength: 3, maxlength: 30
  },
  email: {
    type: String, required: true, unique: true, lowercase: true, trim: true,
    validate: { validator: validator.isEmail, message: 'Invalid email' }
  },
  phone: { type: String, trim: true },
  password: { type: String, required: true, minlength: 8, select: false },
  role: { type: String, enum: ['user','admin'], default: 'user' },
  // Identity (UAE Pass / external IdP) - used for citizen workflow gating in the UI.
  identityProvider: { type: String, trim: true, default: 'password' }, // password | uaepass | other
  identityVerified: { type: Boolean, default: false },
}, { timestamps: true });

UserSchema.index({ email: 1 }, { unique: true });
UserSchema.index({ username: 1 }, { unique: true });

UserSchema.pre('save', async function (next) {
  if (!this.isModified('password')) return next();
  this.password = await bcrypt.hash(this.password, 12);
  next();
});

UserSchema.methods.comparePassword = function (candidate) {
  return bcrypt.compare(candidate, this.password);
};

UserSchema.methods.toJSON = function () {
  const obj = this.toObject();
  delete obj.password;
  delete obj.__v;
  return obj;
};

module.exports = mongoose.model('User', UserSchema);