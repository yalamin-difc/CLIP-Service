/**
 * 🌐 Lost & Found AI Backend Server
 */

// Heroku provides config vars via `process.env` and doesn't require dotenv.
// Also defensively avoid crashing if dotenv is missing/corrupted in the runtime image.
try {
  require("dotenv").config();
} catch (err) {
  if (err && err.code !== "MODULE_NOT_FOUND") throw err;
  // eslint-disable-next-line no-console
  console.warn("⚠️ dotenv not loaded (MODULE_NOT_FOUND); continuing without .env");
}
const express = require("express");
const fs = require("fs");
const path = require("path");
const morgan = require("morgan");
const winston = require("winston");
const mongoose = require("mongoose");
const cors = require("cors");
const http = require("http");
const { Server } = require("socket.io");

// Express + HTTP + Socket.io
const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"] }
});

// =================================================
// 1️⃣ FIX: Define localIP (or Heroku will crash)
// =================================================
const localIP = process.env.LOCAL_IP || "localhost";

// =================================================
// 2️⃣ Correct CORS for Vercel + Heroku
// =================================================
// --------------------------------------
// FIX CORS — MUST be at the top
// --------------------------------------
const allowedOrigins = [
  "https://ailostfound.al-amentech.io",
  "https://openailostfound-fccdb129f869.herokuapp.com",
  /^https:\/\/ai-lost-and-found-ver-2-.*\.vercel\.app$/, // Allows all Vercel previews
  "http://localhost:3000" // For local development
];
app.use(
  cors({
    origin: function (origin, callback) {
      // Allow requests with no origin like server-to-server
      if (!origin) return callback(null, true);
      
      // Check if the origin matches any in our list
      const isAllowed = allowedOrigins.some(allowed => {
        return typeof allowed === 'string' ? origin === allowed : allowed instanceof RegExp && allowed.test(origin);
      });
      
      if (isAllowed) {
        callback(null, true);
      } else {
        console.log(`🚫 CORS blocked for origin: ${origin}`); // Log blocked origin
        callback(new Error('Not allowed by CORS'));
      }
    },
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// Preflight (OPTIONS)
app.options("*", cors());

// =================================================
// Logging
// =================================================
const logDir = path.join(__dirname, "logs");
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });

const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.File({
      filename: path.join(logDir, "error.log"),
      level: "error"
    }),
    new winston.transports.File({
      filename: path.join(logDir, "combined.log")
    }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ]
});

const accessLogStream = fs.createWriteStream(
  path.join(logDir, "access.log"),
  { flags: "a" }
);

app.use(morgan("combined", { stream: accessLogStream }));
app.use(morgan("dev"));

app.use(express.json({ limit: "10mb" }));
app.use(express.urlencoded({ extended: true, limit: "10mb" }));
app.use("/uploads", express.static(path.join(__dirname, "public", "uploads")));

// =================================================
// Dynamic Route Loader
// =================================================
const routes = [
  { path: "/api/items", file: "./routes/items" },
  { path: "/api/ai", file: "./routes/ai" },
  { path: "/api/notifications", file: "./routes/notifications" },
  { path: "/api/auth", file: "./routes/auth" },
  { path: "/api/admin", file: "./routes/admin" },
  { path: "/api/metrics", file: "./routes/metrics" },
  { path: "/api/audit", file: "./routes/audit" },
  { path: "/api/pii", file: "./routes/pii" },
  { path: "/api/claims", file: "./routes/claims" },
  { path: "/api/appointments", file: "./routes/appointments" },
  { path: "/api/shipments", file: "./routes/shipments" },
  { path: "/api/communications", file: "./routes/communications" },
  { path: "/api/chat", file: "./routes/chat" },
  { path: "/api/reports", file: "./routes/report" },
  { path: "/api/matches", file: "./routes/match" }
];

console.log("📂 Loading routes...");

routes.forEach(({ path: routePath, file }) => {
  try {
    const route = require(file);
    app.use(routePath, route);
    console.log(`✅ Route loaded: ${routePath}`);
  } catch (err) {
    console.warn(`⚠️ Skipping ${routePath}: ${err.message}`);
    const router = express.Router();
    router.get("/", (_, res) =>
      res.json({ success: false, message: `Route ${file}.js missing` })
    );
    app.use(routePath, router);
  }
});

// =================================================
// Health Check
// =================================================
app.get(["/api/health", "/health"], (req, res) => {
  const mongoUri =
    process.env.MONGO_URI ||
    process.env.MONGODB_URI ||
    process.env.MONGO_URL ||
    process.env.MONGODB_URL ||
    null;
  const mongoConfigured = Boolean(mongoUri);
  res.json({
    success: true,
    message: "Backend is running",
    uptime: process.uptime(),
    mongoConfigured,
    database:
      mongoose.connection.readyState === 1
        ? "connected"
        : mongoConfigured
          ? "disconnected"
          : "not_configured",
    timestamp: new Date().toISOString()
  });
});

// =================================================
// Root Endpoint
// =================================================
app.get("/", (req, res) => {
  res.json({
    message: "Lost & Found AI API Server",
    status: "running"
  });
});

// =================================================
// Socket.io
// =================================================
io.on("connection", (socket) => {
  logger.info(`🔌 Client connected: ${socket.id}`);

  // Rooms for chat threads (frontend uses `chat:join` / `chat:leave`).
  socket.on("chat:join", ({ room } = {}) => {
    if (!room) return;
    try {
      socket.join(room);
    } catch {
      // ignore
    }
  });

  socket.on("chat:leave", ({ room } = {}) => {
    if (!room) return;
    try {
      socket.leave(room);
    } catch {
      // ignore
    }
  });

  // Allow clients to broadcast chat messages in a room (best-effort realtime).
  // Canonical persistence is via REST `/api/chat/items/:itemId/messages`.
  socket.on("chat:message", ({ room, message } = {}) => {
    if (!room || !message) return;
    try {
      io.to(room).emit("chat:message", { room, message });
    } catch {
      // ignore
    }
  });

  socket.on("disconnect", (reason) => {
    logger.info(`❌ Client disconnected: ${socket.id} - ${reason}`);
  });
});

app.set("io", io);

// =================================================
// MongoDB Connection
// =================================================
const mongoURI =
  process.env.MONGO_URI ||
  process.env.MONGODB_URI ||
  process.env.MONGO_URL ||
  process.env.MONGODB_URL ||
  null;

if (!mongoURI) {
  logger.warn(
    "⚠️ MongoDB not configured. Set one of MONGO_URI / MONGODB_URI (or MONGO_URL / MONGODB_URL)."
  );
} else {
  mongoose.set("strictQuery", true);
  mongoose
    .connect(mongoURI, {
      maxPoolSize: 10,
      serverSelectionTimeoutMS: 10000,
      socketTimeoutMS: 45000,
      retryWrites: true,
      w: "majority",
    })
    .then(() => logger.info("✅ MongoDB connected"))
    .catch((err) => logger.error("❌ MongoDB error: " + err.message));
}

// =================================================
// Error Handling
// =================================================
app.use("*", (req, res) => {
  res.status(404).json({ success: false, message: "Route not found" });
});

// =================================================
// Start Server
// =================================================
const PORT = process.env.PORT || 5000;

server.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on http://${localIP}:${PORT}`);
  console.log(`🌐 CORS allowed: ${allowedOrigins.join(", ")}`);
});

module.exports = { app, server, io };
