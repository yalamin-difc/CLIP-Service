import React from "react";

export type AuditLogEventType =
  | "OCR_EXTRACT"
  | "OCR_EXTRACT_FAILED"
  | "BARCODE_SCAN"
  | "BARCODE_SCAN_FAILED"
  | "AI_MATCH_GENERATION"
  | "ITEM_RELEASE"
  | "ITEM_CREATED";

export type AuditLog = {
  id: string;
  ts: string;
  eventType: AuditLogEventType | string;
  itemId?: string | null;
  requestId?: string | null;
  payload: any;
};

export type MatchExplanation = {
  signals: {
    clipCosine: number;
    ocrTokenJaccard: number;
    barcodeMatches: number;
  };
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
    barcodes?: Array<{ text?: string; format?: string }>;
  };
  score: number;
  confidence: number;
  explanation: MatchExplanation;
};

function fmtPct(x: number) {
  if (!Number.isFinite(x)) return "—";
  return `${(x * 100).toFixed(1)}%`;
}

function fmtNum(x: number) {
  if (!Number.isFinite(x)) return "—";
  return x.toFixed(4);
}

function eventLabel(t: string) {
  switch (t) {
    case "OCR_EXTRACT":
      return "OCR extraction";
    case "BARCODE_SCAN":
      return "Barcode scan";
    case "AI_MATCH_GENERATION":
      return "AI match generation";
    case "ITEM_RELEASE":
      return "Item release";
    case "ITEM_CREATED":
      return "Item created";
    default:
      return t;
  }
}

export function MatchResultCard(props: {
  result: MatchResult;
  auditLogs?: AuditLog[];
  onViewItem?: (itemId: string) => void;
}) {
  const { result, auditLogs, onViewItem } = props;
  const { item } = result;
  const expl = result.explanation;

  const itemLogs = (auditLogs || []).filter((l) => !l.itemId || l.itemId === item.id);

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-5">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold text-slate-100 truncate">{item.name}</h3>
            {item.status ? (
              <span className="text-xs rounded-full border border-slate-700 bg-slate-950/40 px-2 py-0.5 text-slate-300">
                {item.status}
              </span>
            ) : null}
          </div>
          {item.description ? <p className="mt-1 text-sm text-slate-300">{item.description}</p> : null}
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="rounded-lg border border-slate-800 bg-slate-950/40 px-2 py-1 text-slate-300">
              Confidence: <span className="font-mono text-slate-100">{fmtPct(result.confidence)}</span>
            </span>
            <span className="rounded-lg border border-slate-800 bg-slate-950/40 px-2 py-1 text-slate-300">
              CLIP cosine: <span className="font-mono text-slate-100">{fmtNum(expl.signals.clipCosine)}</span>
            </span>
            <span className="rounded-lg border border-slate-800 bg-slate-950/40 px-2 py-1 text-slate-300">
              OCR overlap: <span className="font-mono text-slate-100">{fmtNum(expl.signals.ocrTokenJaccard)}</span>
            </span>
            <span className="rounded-lg border border-slate-800 bg-slate-950/40 px-2 py-1 text-slate-300">
              Barcode matches: <span className="font-mono text-slate-100">{expl.signals.barcodeMatches}</span>
            </span>
          </div>
        </div>

        {onViewItem ? (
          <button
            className="shrink-0 rounded-xl bg-slate-800 hover:bg-slate-700 px-3 py-2 text-sm font-semibold text-slate-100"
            onClick={() => onViewItem(item.id)}
          >
            View item
          </button>
        ) : null}
      </div>

      {expl.details?.ocr?.overlap?.length ? (
        <div className="mt-4">
          <div className="text-xs text-slate-400 mb-2">Explainability (OCR token overlap)</div>
          <div className="flex flex-wrap gap-2">
            {expl.details.ocr.overlap.map((tok) => (
              <span
                key={tok}
                className="rounded-lg border border-slate-800 bg-slate-950/40 px-2 py-1 text-xs text-slate-200 font-mono"
              >
                {tok}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {itemLogs.length ? (
        <div className="mt-5">
          <div className="text-xs text-slate-400 mb-2">Audit logs (OCR / barcode / match / release)</div>
          <div className="space-y-2">
            {itemLogs.slice(0, 8).map((l) => (
              <div key={l.id} className="rounded-xl border border-slate-800 bg-slate-950/30 p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="text-sm text-slate-200">
                    {eventLabel(l.eventType)}{" "}
                    <span className="text-xs text-slate-500 font-mono">{l.requestId ? `rid:${l.requestId}` : ""}</span>
                  </div>
                  <div className="text-xs text-slate-500 font-mono">{new Date(l.ts).toLocaleString()}</div>
                </div>
                <pre className="mt-2 text-xs text-slate-300 whitespace-pre-wrap break-words">
                  {JSON.stringify(l.payload ?? {}, null, 2)}
                </pre>
              </div>
            ))}
          </div>
          {itemLogs.length > 8 ? <div className="mt-2 text-xs text-slate-500">Showing latest 8 events.</div> : null}
        </div>
      ) : null}
    </div>
  );
}

