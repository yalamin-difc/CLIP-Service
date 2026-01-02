export type ClipServiceGovernance = {
  serviceVersion: string;
  modelId: string;
  confidence: { temperature: number; minScore: number; minMargin: number };
};

export type ClipServiceInputMeta = {
  sha256: string;
  bytes: number;
  contentType?: string | null;
};

export type OcrWord = {
  text: string;
  conf: number;
  bbox: { x: number; y: number; w: number; h: number };
  pageNum: number;
  blockNum: number;
  parNum: number;
  lineNum: number;
  wordNum: number;
};

export type OcrResult = {
  fullText: string;
  words: OcrWord[];
  meta: { lang: string; psm: number };
};

export type Barcode = {
  text?: string;
  format?: string;
  contentType?: string;
  position?: any;
};

export type BarcodeResult = {
  barcodes: Barcode[];
  meta: { engine: string };
};

export type AnalyzeImageResponse = {
  requestId: string;
  governance: ClipServiceGovernance;
  input: ClipServiceInputMeta;
  embedding: number[];
  ocr: OcrResult | null;
  ocrError: string | null;
  barcode: BarcodeResult | null;
  barcodeError: string | null;
};

export type StoredItem = {
  id: string;
  name: string;
  description?: string | null;
  status: string;
  embedding: number[];
  ocrText?: string | null;
  ocrWords?: OcrWord[];
  barcodes?: Barcode[];
  createdAt: string;
  updatedAt: string;
  releasedAt?: string | null;
};

export type CreateItemResponse = {
  requestId: string;
  governance: ClipServiceGovernance;
  item: StoredItem;
};

export type ReleaseItemResponse = {
  requestId: string;
  item: StoredItem;
};

export type MatchExplanation = {
  signals: { clipCosine: number; ocrTokenJaccard: number; barcodeMatches: number };
  details: {
    ocr?: { jaccard: number; overlap: string[]; aCount: number; bCount: number };
    barcode?: { matched: string[]; queryCount: number; itemCount: number };
  };
};

export type MatchResult = {
  item: {
    id: string;
    name: string;
    description?: string | null;
    status?: string;
    ocrText?: string | null;
    barcodes?: Barcode[];
  };
  score: number;
  confidence: number;
  explanation: MatchExplanation;
};

export type MatchResponse = {
  requestId: string;
  governance: ClipServiceGovernance;
  decisioning: { noMatch: boolean; meta: any };
  query: { embedding: number[]; queryText?: string | null; ocr?: OcrResult | null; barcode?: BarcodeResult | null };
  topK: MatchResult[];
};

export type AuditLog = {
  id: string;
  ts: string;
  eventType: string;
  itemId?: string | null;
  requestId?: string | null;
  payload: any;
};

export type ListAuditLogsResponse = { logs: AuditLog[] };
export type ListItemsResponse = { items: StoredItem[] };
export type GetItemResponse = { item: StoredItem };

export type ClipServiceClientConfig = {
  /**
   * Example: "http://localhost:8080" or your Cloud Run URL (no trailing slash recommended)
   */
  baseUrl: string;
  /**
   * If backend has CLIP_API_KEY set: provide the raw token here.
   * Client will send: Authorization: Bearer <apiKey>
   */
  apiKey?: string;
};

type RequestOptions = { requestId?: string };

function joinUrl(baseUrl: string, path: string) {
  return `${baseUrl.replace(/\/+$/, "")}/${path.replace(/^\/+/, "")}`;
}

function authHeaders(apiKey?: string): Record<string, string> {
  const k = (apiKey || "").trim();
  return k ? { Authorization: `Bearer ${k}` } : {};
}

async function readJsonOrThrow<T>(r: Response): Promise<T> {
  const body = (await r.json().catch(() => ({}))) as any;
  if (!r.ok) {
    const msg = (body && (body.detail || body.error)) || `HTTP ${r.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return body as T;
}

export function createClipServiceClient(cfg: ClipServiceClientConfig) {
  const baseUrl = cfg.baseUrl;
  const apiKey = cfg.apiKey;

  return {
    async analyzeImage(
      file: File,
      opts?: { doOcr?: boolean; doBarcode?: boolean; ocrLang?: string; ocrPsm?: number } & RequestOptions
    ): Promise<AnalyzeImageResponse> {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("doOcr", String(opts?.doOcr ?? true));
      fd.append("doBarcode", String(opts?.doBarcode ?? true));
      fd.append("ocrLang", String(opts?.ocrLang ?? "eng"));
      fd.append("ocrPsm", String(opts?.ocrPsm ?? 6));

      const headers: Record<string, string> = { ...authHeaders(apiKey) };
      if (opts?.requestId) headers["X-Request-Id"] = opts.requestId;

      const r = await fetch(joinUrl(baseUrl, "/analyze-image"), { method: "POST", body: fd, headers });
      return await readJsonOrThrow<AnalyzeImageResponse>(r);
    },

    async createItem(
      args: {
        name: string;
        description?: string;
        file: File;
        doOcr?: boolean;
        doBarcode?: boolean;
        ocrLang?: string;
        ocrPsm?: number;
      } & RequestOptions
    ): Promise<CreateItemResponse> {
      const fd = new FormData();
      fd.append("name", args.name);
      if (args.description) fd.append("description", args.description);
      fd.append("file", args.file);
      fd.append("doOcr", String(args.doOcr ?? true));
      fd.append("doBarcode", String(args.doBarcode ?? true));
      fd.append("ocrLang", String(args.ocrLang ?? "eng"));
      fd.append("ocrPsm", String(args.ocrPsm ?? 6));

      const headers: Record<string, string> = { ...authHeaders(apiKey) };
      if (args.requestId) headers["X-Request-Id"] = args.requestId;

      const r = await fetch(joinUrl(baseUrl, "/items"), { method: "POST", body: fd, headers });
      return await readJsonOrThrow<CreateItemResponse>(r);
    },

    async releaseItem(itemId: string, opts?: RequestOptions): Promise<ReleaseItemResponse> {
      const headers: Record<string, string> = { ...authHeaders(apiKey) };
      if (opts?.requestId) headers["X-Request-Id"] = opts.requestId;

      const r = await fetch(joinUrl(baseUrl, `/items/${encodeURIComponent(itemId)}/release`), {
        method: "POST",
        headers,
      });
      return await readJsonOrThrow<ReleaseItemResponse>(r);
    },

    async match(
      args: {
        file: File;
        queryText?: string;
        doOcr?: boolean;
        doBarcode?: boolean;
        ocrLang?: string;
        ocrPsm?: number;
        k?: number;
        status?: string;
      } & RequestOptions
    ): Promise<MatchResponse> {
      const fd = new FormData();
      fd.append("file", args.file);
      if (args.queryText) fd.append("queryText", args.queryText);
      fd.append("doOcr", String(args.doOcr ?? true));
      fd.append("doBarcode", String(args.doBarcode ?? true));
      fd.append("ocrLang", String(args.ocrLang ?? "eng"));
      fd.append("ocrPsm", String(args.ocrPsm ?? 6));
      fd.append("k", String(args.k ?? 5));
      fd.append("status", String(args.status ?? "released"));

      const headers: Record<string, string> = { ...authHeaders(apiKey) };
      if (args.requestId) headers["X-Request-Id"] = args.requestId;

      const r = await fetch(joinUrl(baseUrl, "/match"), { method: "POST", body: fd, headers });
      return await readJsonOrThrow<MatchResponse>(r);
    },

    async listItems(args?: { status?: string; limit?: number }): Promise<ListItemsResponse> {
      const u = new URL(joinUrl(baseUrl, "/items"));
      if (args?.status) u.searchParams.set("status", args.status);
      if (typeof args?.limit === "number") u.searchParams.set("limit", String(args.limit));

      const r = await fetch(u.toString(), { headers: { ...authHeaders(apiKey) } });
      return await readJsonOrThrow<ListItemsResponse>(r);
    },

    async getItem(itemId: string): Promise<GetItemResponse> {
      const r = await fetch(joinUrl(baseUrl, `/items/${encodeURIComponent(itemId)}`), {
        headers: { ...authHeaders(apiKey) },
      });
      return await readJsonOrThrow<GetItemResponse>(r);
    },

    async listAuditLogs(args?: { itemId?: string; limit?: number; offset?: number }): Promise<ListAuditLogsResponse> {
      const u = new URL(joinUrl(baseUrl, "/audit-logs"));
      if (args?.itemId) u.searchParams.set("itemId", args.itemId);
      if (typeof args?.limit === "number") u.searchParams.set("limit", String(args.limit));
      if (typeof args?.offset === "number") u.searchParams.set("offset", String(args.offset));

      const r = await fetch(u.toString(), { headers: { ...authHeaders(apiKey) } });
      return await readJsonOrThrow<ListAuditLogsResponse>(r);
    },
  };
}

