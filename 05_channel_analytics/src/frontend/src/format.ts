export const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function money(v: unknown, digits?: number): string {
  if (!isNum(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1e9) return `$${(v / 1e9).toFixed(digits ?? 2)}B`;
  if (a >= 1e6) return `$${(v / 1e6).toFixed(digits ?? 1)}M`;
  if (a >= 1e3) return `$${(v / 1e3).toFixed(digits ?? 0)}K`;
  return `$${v.toFixed(0)}`;
}

/** Exact millions with one decimal, e.g. $1,139.6M (used for forecast figures). */
export function millions(v: unknown): string {
  if (!isNum(v)) return "–";
  return `$${(v / 1e6).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 })}M`;
}

export const moneyAxis = (v: number) => money(v, Math.abs(v) >= 1e9 ? 1 : 0);

export function pct(v: unknown, digits = 1): string {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : "–";
}

export function pctSigned(v: unknown, digits = 1): string {
  return isNum(v) ? `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)}%` : "–";
}

export const pctPoints = (v: unknown, digits = 1) => (isNum(v) ? `${v.toFixed(digits)}%` : "–");
export const count = (v: unknown) => (isNum(v) ? Math.round(v).toLocaleString("en-US") : "–");
export const ratio = (v: unknown) => (isNum(v) ? `${v.toFixed(2)}x` : "–");
export const decimal = (v: unknown, d = 1) => (isNum(v) ? v.toFixed(d) : "–");

export function formatValue(v: unknown, format?: string): string {
  switch (format) {
    case "money": return money(v);
    case "pct": return pct(v);
    case "pct_signed": return pctSigned(v);
    case "pct_points": return pctPoints(v);
    case "pp": return isNum(v) ? `${v.toFixed(1)} pp` : "–";
    case "count": return count(v);
    case "ratio": return ratio(v);
    case "decimal": return decimal(v);
    case "index": return decimal(v, 1);
    default: return isNum(v) ? v.toLocaleString("en-US", { maximumFractionDigits: 2 }) : String(v ?? "–");
  }
}

export function axisFormatter(format?: string): (v: number) => string {
  switch (format) {
    case "money": return moneyAxis;
    case "pct": return (v) => `${Math.round(v * 100)}%`;
    case "pct_points": return (v) => `${v}%`;
    case "ratio": return (v) => `${v}x`;
    default: return (v) => (Math.abs(v) >= 1000 ? v.toLocaleString("en-US") : String(v));
  }
}

export const truncate = (s: string | null | undefined, n = 80) =>
  !s ? "" : s.length > n ? `${s.slice(0, n - 1).trimEnd()}…` : s;
