## Frontend ↔ Backend integration mapping (CLIP + OCR + Barcode)

This repo’s backend is **FastAPI** (`app.py`) exposing a CLIP embedding + OCR + barcode pipeline and an items/matching store. This document maps **frontend requests** to **backend endpoints** and the **response shapes** you should use in the UI.

### Base URL

- **Local**: `http://localhost:<PORT>` (commonly 8080)
- **Prod**: your deployed service URL (Cloud Run / Heroku etc.)

### Auth (optional)

If `CLIP_API_KEY` is set on the server, every request must include:

- **Header**: `Authorization: Bearer <CLIP_API_KEY>`

If `CLIP_API_KEY` is empty, auth is disabled.

### Request correlation (recommended)

You can pass an optional request id:

- **Header**: `X-Request-Id: <uuid-or-any-string>`

Backend will echo it back as `requestId` in many responses and also store it in audit logs.

---

## 1) One-shot analysis: image → { CLIP embedding + OCR + Barcode }

### Endpoint

- **POST** `/analyze-image`

### Frontend request (multipart/form-data)

- **file**: `File` (preferred) *(backend also accepts `image`)*
- **doOcr**: `true|false` (string/boolean via FormData)
- **doBarcode**: `true|false`
- **ocrLang**: e.g. `eng`, `ara`, `fra`
- **ocrPsm**: e.g. `6`

### Response (JSON)

- `requestId`: string
- `governance`: `{ serviceVersion, modelId, confidence{temperature,minScore,minMargin} }`
- `input`: `{ sha256, bytes, contentType }`
- `embedding`: `number[]` (normalized CLIP image embedding)
- `ocr`: `null | { fullText, words[], meta{lang,psm} }`
- `ocrError`: `null | string`
- `barcode`: `null | { barcodes[], meta{engine} }`
- `barcodeError`: `null | string`

---

## 2) Create an item (store embedding + optional OCR/barcodes)

### Endpoint

- **POST** `/items`

### Frontend request (multipart/form-data)

- **name**: string *(required)*
- **description**: string *(optional)*
- **file**: `File` *(required; backend also accepts `image`)*
- **doOcr**, **doBarcode**, **ocrLang**, **ocrPsm**: same as `/analyze-image`

### Response (JSON)

- `requestId`: string
- `governance`: object
- `item`: object
  - `id`, `name`, `description`, `status`
  - `embedding`: `number[]`
  - `ocrText`: `string|null`
  - `ocrWords`: `array`
  - `barcodes`: `array`
  - `createdAt`, `updatedAt`, `releasedAt`

---

## 3) Release an item (make it matchable)

### Endpoint

- **POST** `/items/{item_id}/release`

### Response (JSON)

- `requestId`: string
- `item`: item object (status updated to `released`)

---

## 4) Find matches: query image (+ optional queryText/OCR/barcode) → Top-K items

### Endpoint

- **POST** `/match`

### Frontend request (multipart/form-data)

- **file**: `File` *(required; backend also accepts `image`)*
- **queryText**: string *(optional; if omitted, backend can use OCR fullText if `doOcr=true`)*
- **doOcr**: `true|false`
- **doBarcode**: `true|false`
- **ocrLang**, **ocrPsm**
- **k**: number (default `5`, max `50`)
- **status**: string (default `released`) — which items are eligible candidates

### Response (JSON)

- `requestId`: string
- `governance`: object
- `decisioning`: `{ noMatch: boolean, meta: { reason, topScore, secondScore?, minScore, minMargin } }`
- `query`:
  - `embedding`: `number[]`
  - `queryText`: `string|null`
  - `ocr`: OCR object or null
  - `barcode`: barcode object or null
- `topK`: `[] | Array<{ item, score, confidence, explanation }>`
  - `item`: `{ id, name, description, status, ocrText, barcodes[] }`
  - `explanation`:
    - `signals`: `{ clipCosine, ocrTokenJaccard, barcodeMatches }`
    - `details`: `{ ocr{...}, barcode{...} }`

If `decisioning.noMatch === true`, `topK` will be empty.

---

## 5) Audit logs (for explainability + debugging)

### Endpoint

- **GET** `/audit-logs?itemId=<optional>&limit=<optional>&offset=<optional>`

### Response (JSON)

- `logs`: `AuditLog[]`
  - `id`, `ts`, `eventType`, `itemId`, `requestId`, `payload`

---

## 6) Items list / get

- **GET** `/items?status=<optional>&limit=<optional>`
- **GET** `/items/{item_id}`

---

## Frontend types (matches `frontend/components/MatchResultCard.tsx`)

The backend’s `/match` response `topK[*]` entries match the `MatchResult` type used by `MatchResultCard`:

- `item.{id,name,description,status,ocrText,barcodes}`
- `score`
- `confidence`
- `explanation.{signals,details}`

