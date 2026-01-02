import React, { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  Container,
  Typography,
  Paper,
  Box,
  CircularProgress,
  Grid,
  Alert,
  Button,
  Divider,
  TextField,
} from "@mui/material";
import { itemsAPI, piiAPI } from "../services/api";
import { UPLOAD_URL } from "../utils/constants";
import { useAuth } from "../contexts/AuthContext";
import ChatThread from "../components/ChatThread";
import SmsComposer from "../components/SmsComposer";
import PiiField from "../components/PiiField";
import { maskEmail, maskName, maskPhone } from "../utils/pii";
import { useTranslation } from "react-i18next";
import { downloadCsv, downloadJson, printHtmlReport } from "../utils/export";

const ItemDetails = () => {
  const { t } = useTranslation();
  const { id } = useParams();
  const [item, setItem] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [savingShipment, setSavingShipment] = useState(false);
  const [shipmentStatus, setShipmentStatus] = useState({ type: "", text: "" });
  const { user, isAuthenticated, hasRole, hasPermission } = useAuth();
  const [revealedContact, setRevealedContact] = useState(null);

  const isOwner = useMemo(() => {
    const uid = user?._id || user?.id;
    const reporterId = item?.reporter?._id || item?.reporter?.id;
    return Boolean(uid && reporterId && uid === reporterId);
  }, [item?.reporter, user]);

  const contactInfo = useMemo(() => {
    const raw = item?.contactInfo;
    if (!raw) return null;
    if (typeof raw === "string") {
      try {
        return JSON.parse(raw);
      } catch {
        return null;
      }
    }
    if (typeof raw === "object") return raw;
    return null;
  }, [item?.contactInfo]);

  const [shipmentForm, setShipmentForm] = useState({
    provider: "",
    trackingNumber: "",
    status: "",
  });

  useEffect(() => {
    const fetchItem = async () => {
      try {
        const res = await itemsAPI.getItem(id);
        setItem(res.item);
        const existing = res?.item?.shipment || {};
        setShipmentForm({
          provider: existing.provider || "",
          trackingNumber: existing.trackingNumber || "",
          status: existing.status || "",
        });
      } catch (err) {
        console.error("❌ Failed to fetch item details:", err);
        setError("Failed to fetch item details");
      } finally {
        setLoading(false);
      }
    };
    fetchItem();
  }, [id]);

  if (loading)
    return (
      <Box sx={{ display: "flex", justifyContent: "center", mt: 10 }}>
        <CircularProgress />
      </Box>
    );

  if (error)
    return (
      <Container sx={{ mt: 10 }}>
        <Alert severity="error">{error}</Alert>
      </Container>
    );

  if (!item)
    return (
      <Container sx={{ mt: 10 }}>
        <Typography variant="h6">Item not found.</Typography>
      </Container>
    );

  const firstImage = Array.isArray(item.images) && item.images.length
    ? item.images[0]
    : null;
  const imageUrl = typeof firstImage === "string"
    ? firstImage.startsWith("http")
      ? firstImage
      : `${UPLOAD_URL}${firstImage.startsWith("/") ? "" : "/"}${firstImage}`
    : null;

  const reporterPhone =
    revealedContact?.phone ||
    contactInfo?.phone ||
    item?.reporter?.phone ||
    item?.reporter?.mobile ||
    item?.phone ||
    "";

  const reporterName = item?.reporter?.username || item?.reporter?.name || "";
  const reporterEmail = revealedContact?.email || item?.reporter?.email || contactInfo?.email || "";

  const canRevealPii = Boolean(
    isOwner ||
      hasPermission?.("pii:reveal") ||
      hasRole?.(["operator", "supervisor", "admin", "auditor"])
  );

  const saveShipment = async () => {
    setShipmentStatus({ type: "", text: "" });
    if (!isAuthenticated) {
      setShipmentStatus({ type: "error", text: "Please log in to update shipment tracking." });
      return;
    }
    if (!isOwner) {
      setShipmentStatus({ type: "error", text: "Only the reporter can update shipment tracking." });
      return;
    }
    if (!shipmentForm.trackingNumber.trim()) {
      setShipmentStatus({ type: "error", text: "Tracking number is required." });
      return;
    }
    setSavingShipment(true);
    try {
      const payload = {
        shipment: {
          provider: shipmentForm.provider.trim(),
          trackingNumber: shipmentForm.trackingNumber.trim(),
          status: shipmentForm.status.trim(),
          updatedAt: new Date().toISOString(),
        },
      };
      const res = await itemsAPI.updateItem(id, payload);
      if (res?.item) setItem(res.item);
      setShipmentStatus({ type: "success", text: "Shipment tracking saved." });
    } catch (e) {
      setShipmentStatus({
        type: "error",
        text:
          e?.response?.data?.message ||
          e?.message ||
          "Failed to save shipment tracking. Backend route may not be enabled yet.",
      });
    } finally {
      setSavingShipment(false);
    }
  };

  const exportCaseCsv = () => {
    const headers = ["Field", "Value"];
    const rows = [
      ["Item ID", id],
      ["Title", item.title || ""],
      ["Type", item.type || ""],
      ["Category", item.category || ""],
      ["CreatedAt", item.createdAt ? new Date(item.createdAt).toISOString() : ""],
      ["Address", item.address || ""],
      ["Latitude", item.lat || item.location?.lat || ""],
      ["Longitude", item.lng || item.location?.lng || ""],
      ["Description", item.description || ""],
      ["Reporter name", canRevealPii ? reporterName : t("pii.hidden")],
      ["Reporter email", canRevealPii ? reporterEmail : t("pii.hidden")],
      ["Reporter phone", canRevealPii ? reporterPhone : t("pii.hidden")],
      ["Shipment provider", item.shipment?.provider || ""],
      ["Shipment tracking", item.shipment?.trackingNumber || ""],
      ["Shipment status", item.shipment?.status || ""],
    ];
    downloadCsv({
      filename: `case-${id}-${Date.now()}.csv`,
      headers,
      rows,
    });
  };

  const exportCaseJson = () => {
    const data = {
      itemId: id,
      title: item.title,
      type: item.type,
      category: item.category,
      createdAt: item.createdAt,
      address: item.address,
      lat: item.lat || item.location?.lat,
      lng: item.lng || item.location?.lng,
      description: item.description,
      reporter: canRevealPii
        ? { name: reporterName, email: reporterEmail, phone: reporterPhone }
        : { name: null, email: null, phone: null },
      shipment: item.shipment || null,
    };
    downloadJson({ filename: `case-${id}-${Date.now()}.json`, data });
  };

  const downloadCasePdf = () => {
    const now = new Date().toLocaleString();
    const piiLine = canRevealPii
      ? `<div class="muted">Reporter: ${reporterName || "—"} · ${reporterEmail || "—"} · ${reporterPhone || "—"}</div>`
      : `<div class="muted">Reporter: ${t("pii.hidden")}</div>`;
    const shipment = item.shipment?.trackingNumber
      ? `<table>
          <thead><tr><th>Provider</th><th>Tracking</th><th>Status</th></tr></thead>
          <tbody><tr><td>${item.shipment.provider || "—"}</td><td>${item.shipment.trackingNumber}</td><td>${item.shipment.status || "—"}</td></tr></tbody>
        </table>`
      : `<div class="muted">No shipment tracking.</div>`;
    const body = `
      <h1>${t("exports.caseTitle") || "Case report"}</h1>
      <div class="muted">${now}</div>
      <h2>Item</h2>
      <div><strong>ID:</strong> ${id}</div>
      <div><strong>Title:</strong> ${item.title || "—"}</div>
      <div><strong>Type:</strong> ${item.type || "—"}</div>
      <div><strong>Category:</strong> ${item.category || "—"}</div>
      <div><strong>Created:</strong> ${item.createdAt ? new Date(item.createdAt).toLocaleString() : "—"}</div>
      <div><strong>Address:</strong> ${item.address || "—"}</div>
      ${piiLine}
      <h2>Description</h2>
      <div>${(item.description || "").replace(/[<>]/g, "") || "—"}</div>
      <h2>Shipment</h2>
      ${shipment}
    `;
    printHtmlReport({ title: `case-${id}`, htmlBody: body });
  };

  return (
    <Container sx={{ mt: 6 }}>
      <Paper sx={{ p: 4, borderRadius: 3 }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", gap: 2, flexWrap: "wrap", mb: 3 }}>
          <Typography variant="h4" sx={{ fontWeight: 700 }}>
            {item.title}
          </Typography>
          <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", alignItems: "center" }}>
            <Button variant="outlined" onClick={exportCaseCsv}>
              {t("exports.csv") || "Export CSV"}
            </Button>
            <Button variant="outlined" onClick={exportCaseJson}>
              {t("exports.json") || "Export JSON"}
            </Button>
            <Button variant="contained" onClick={downloadCasePdf}>
              {t("exports.pdf") || "Download PDF"}
            </Button>
          </Box>
        </Box>
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            {imageUrl ? (
              <img
                src={imageUrl}
                alt={item.title}
                style={{
                  width: "100%",
                  height: 300,
                  objectFit: "cover",
                  borderRadius: 8,
                }}
              />
            ) : (
              <Typography variant="body2" color="text.secondary">
                No image available
              </Typography>
            )}
          </Grid>
          <Grid item xs={12} md={6}>
            <Typography variant="subtitle1">
              <strong>{t("itemDetail.category")}:</strong> {item.category}
            </Typography>
            <Typography variant="subtitle1">
              <strong>{t("itemDetail.type")}:</strong> {item.type}
            </Typography>
            <Typography variant="subtitle1">
              <strong>{t("itemDetail.date")}:</strong>{" "}
              {new Date(item.createdAt).toLocaleDateString()}
            </Typography>
            <Typography variant="subtitle1">
              <strong>{t("itemDetail.address")}:</strong> {item.address || "—"}
            </Typography>

            {isOwner ? (
              <>
                <PiiField
                  label={t("itemDetail.reporterName")}
                  value={reporterName || "—"}
                  maskedValue={maskName(reporterName)}
                  canReveal={true}
                  onReveal={async () => {}}
                />
                <PiiField
                  label={t("itemDetail.reporterEmail")}
                  value={reporterEmail || "—"}
                  maskedValue={maskEmail(reporterEmail)}
                  canReveal={true}
                  onReveal={async () => {}}
                />
              </>
            ) : (
              <>
                <PiiField
                  label={t("itemDetail.reporterName")}
                  value={reporterName || "—"}
                  maskedValue={t("pii.hidden")}
                  canReveal={canRevealPii}
                  onReveal={async ({ reason }) => {
                    // Backend enforces + audits (break-glass reveal)
                    const res = await piiAPI.reveal({ caseId: id, itemId: id, reason });
                    const contact = res?.data?.contact || res?.contact || null;
                    if (contact && typeof contact === "object") {
                      setRevealedContact((prev) => ({ ...(prev || {}), ...contact }));
                    }
                  }}
                />
                <PiiField
                  label={t("itemDetail.reporterEmail")}
                  value={reporterEmail || "—"}
                  maskedValue={t("pii.hidden")}
                  canReveal={canRevealPii}
                  onReveal={async ({ reason }) => {
                    const res = await piiAPI.reveal({ caseId: id, itemId: id, reason });
                    const contact = res?.data?.contact || res?.contact || null;
                    if (contact && typeof contact === "object") {
                      setRevealedContact((prev) => ({ ...(prev || {}), ...contact }));
                    }
                  }}
                />
              </>
            )}

            <Typography variant="subtitle1" sx={{ mt: 2 }}>
              <strong>{t("itemDetail.description")}:</strong>
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {item.description || t("itemDetail.noDescription")}
            </Typography>
          </Grid>
        </Grid>

        <Divider sx={{ my: 3 }} />

        {/* Shipment tracking for found items */}
        {item.type === "found" && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
              Shipment Tracking (Found items)
            </Typography>
            {shipmentStatus.text && (
              <Alert
                severity={shipmentStatus.type === "success" ? "success" : "error"}
                sx={{ mb: 2 }}
              >
                {shipmentStatus.text}
              </Alert>
            )}

            {item.shipment?.trackingNumber ? (
              <Paper variant="outlined" sx={{ p: 2, borderRadius: 2, mb: 2 }}>
                <Typography variant="body2">
                  <strong>Provider:</strong> {item.shipment.provider || "—"}
                </Typography>
                <Typography variant="body2">
                  <strong>Tracking #:</strong> {item.shipment.trackingNumber}
                </Typography>
                <Typography variant="body2">
                  <strong>Status:</strong> {item.shipment.status || "—"}
                </Typography>
              </Paper>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                No shipment tracking has been added yet.
              </Typography>
            )}

            {/* Self-service: allow reporter to add/update tracking */}
            <Paper sx={{ p: 2, borderRadius: 2 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                Update tracking (reporter only)
              </Typography>
              <Box sx={{ display: "grid", gap: 2 }}>
                <TextField
                  label="Shipment provider"
                  value={shipmentForm.provider}
                  onChange={(e) => setShipmentForm((p) => ({ ...p, provider: e.target.value }))}
                  placeholder="DHL, FedEx, UPS…"
                  fullWidth
                />
                <TextField
                  label="Tracking number"
                  value={shipmentForm.trackingNumber}
                  onChange={(e) => setShipmentForm((p) => ({ ...p, trackingNumber: e.target.value }))}
                  fullWidth
                />
                <TextField
                  label="Shipment status (optional)"
                  value={shipmentForm.status}
                  onChange={(e) => setShipmentForm((p) => ({ ...p, status: e.target.value }))}
                  placeholder="Label created / In transit / Delivered…"
                  fullWidth
                />
                <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
                  <Button variant="contained" onClick={saveShipment} disabled={savingShipment}>
                    Save tracking
                  </Button>
                </Box>
              </Box>
            </Paper>
          </Box>
        )}

        {/* SMS to reporter/customer phone */}
        {canRevealPii && (
          <Box sx={{ mb: 3 }}>
            <SmsComposer
              title={t("itemDetail.smsTitle")}
              defaultTo={reporterPhone}
              defaultMessage={t("itemDetail.smsDefaultMessage", { title: item.title })}
              recipientLabel={t("itemDetail.smsRecipient")}
              messageLabel={t("itemDetail.smsMessage")}
            />
            {!reporterPhone && (
              <Alert severity="info" sx={{ mt: 2 }}>
                {t("itemDetail.smsNoPhone")}
              </Alert>
            )}
            {reporterPhone && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {t("itemDetail.smsMaskedPhone")}: {maskPhone(reporterPhone)}
              </Typography>
            )}
          </Box>
        )}

        <ChatThread itemId={id} />
      </Paper>
    </Container>
  );
};

export default ItemDetails