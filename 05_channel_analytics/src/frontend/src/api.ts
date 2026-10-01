import { useCallback, useContext, useEffect, useRef, useState } from "react";
import { AppContext } from "./context";

export type Params = Record<string, string | number | boolean | string[] | undefined | null>;

export interface Filters {
  region: string[];
  tier: string[];
  product_family: string[];
  partner_type: string[];
  q_from?: string;
  q_to?: string;
}

export const EMPTY_FILTERS: Filters = { region: [], tier: [], product_family: [], partner_type: [] };

export function filterParams(f: Filters): Params {
  return {
    region: f.region.join(","), tier: f.tier.join(","), product_family: f.product_family.join(","),
    partner_type: f.partner_type.join(","), q_from: f.q_from, q_to: f.q_to,
  };
}

export function query(params?: Params): string {
  if (!params) return "";
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "") continue;
    q.set(k, Array.isArray(v) ? v.join(",") : String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function apiGet<T>(path: string, params?: Params): Promise<T> {
  const res = await fetch(`/api${path}${query(params)}`);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function apiPost<T>(path: string): Promise<T> {
  const res = await fetch(`/api${path}`, { method: "POST" });
  if (!res.ok && res.status !== 202) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/** Fetch `path` and refetch when params change or when the analysis is refreshed. */
export function useApi<T>(path: string | null, params?: Params) {
  const { dataVersion } = useContext(AppContext);
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | Error | null>(null);
  const [loading, setLoading] = useState<boolean>(!!path);
  const key = path ? `${path}${query(params)}` : null;
  const latest = useRef(0);

  const load = useCallback(async () => {
    if (!path) return;
    const id = ++latest.current;
    setLoading(true);
    setError(null);
    try {
      const d = await apiGet<T>(path, params);
      if (id === latest.current) setData(d);
    } catch (e) {
      if (id === latest.current) setError(e as Error);
    } finally {
      if (id === latest.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(() => { load(); }, [load, dataVersion]);
  return { data, error, loading, reload: load };
}

// ---------------------------------------------------------------- response types (subset used by the UI)

export interface Meta {
  regions: string[]; tiers: string[]; products: string[]; partner_types: string[]; quarters: string[];
  segments: string[]; data_from: string; data_to: string; partners: number; order_lines: number;
  outlier_lines_excluded: number; generated_at: string; errors: Record<string, string>;
  sections: { at_risk: boolean; forecast: boolean; insights: boolean };
}

export interface Kpis {
  quarter: string; revenue: number; revenue_last_year: number; yoy: number | null; margin_pct: number | null;
  active_partners: number; attainment: number | null; attainment_note: string | null; revenue_in_range: number;
  at_risk: { flagged: number; partners: number; declining: number } | null;
  forecast: { area: string; quarter: string; forecast: number; lo80: number; hi80: number } | null;
}

export interface Trend { grain: string; by: string; keys: string[]; rows: Record<string, number | string>[] }

export interface BreakdownRow {
  key: string; revenue: number; share: number; margin_pct: number; partners: number; order_lines: number;
  growth_yoy: number | null; attainment?: number | null; pct_partner_quarters_hit?: number | null;
}
export interface Breakdown { dim: string; rows: BreakdownRow[]; growth_label: string | null; attainment_label: string | null }

export interface PartnerRow {
  partner_id: string; partner_name: string; region: string; tier: string; partner_type: string; country: string;
  segment: string; health_score: number; p_leave: number; at_risk: number; risk_rank: number; revenue_range: number;
  margin_pct: number | null; revenue_q2_2026: number; yoy_h1_pct: number | null; days_since_last_order: number | null;
  reason_1: string | null;
}
export interface PartnerPage { total: number; page: number; page_size: number; rows: PartnerRow[]; available: boolean }

export interface PartnerDetail {
  profile: { partner_id: string; partner_name: string; partner_type: string; tier: string; region: string; country: string; onboarded_date: string };
  score: {
    health_score: number; p_leave: number; risk_rank: number; at_risk: number; segment: string; revenue_avg_4q: number;
    revenue_q2_2026: number; yoy_h1_pct: number | null; last_order: string | null; days_since_last_order: number | null; partners: number;
  } | null;
  reasons: { text: string; driver: string; points: number | null }[];
  signals: { key: string; label: string; value: number | null; tier_median: number | null }[] | null;
  quarterly: { quarter: string; revenue_usd: number; target_usd: number | null; attainment: number | null; margin_pct: number | null; order_lines: number; n_products: number }[];
  monthly: { month: string; revenue_usd: number; order_lines: number }[];
  product_mix: { product: string; revenue: number; share: number }[];
  mix_period: string;
}

export interface AtRiskRow extends Record<string, unknown> {
  risk_rank: number; partner_id: string; partner_name: string; region: string; tier: string; partner_type: string;
  health_score: number; p_leave: number; segment: string; revenue_avg_4q: number; reason_1: string | null;
  driver_1: string | null; reason_2: string | null; reason_3: string | null;
}
export interface AtRisk {
  available: boolean; error?: string;
  summary: {
    partners: number; flagged: number; expected_leavers: number; flagged_revenue_avg_4q: number; flagged_share_of_revenue: number;
    season_factor: number; flag_multiplier: number; backtest_auc: number; backtest_auc_heuristic: number;
    backtest_precision: number; backtest_recall: number;
  };
  drivers: { driver: string; partners: number }[];
  segments: { segment: string; partners: number; revenue_q2: number; avg_health: number }[];
  health_histogram: { from: number; to: number; partners: number }[];
  flagged: AtRiskRow[];
}

export interface ForecastData {
  available: boolean; error?: string; target_quarter: string;
  submission: { region: string; forecast: number; lo80: number; hi80: number }[];
  history: Record<string, number | string>[];
  methods: Record<string, number | string>[];
  method_labels: Record<string, string>;
  backtest_mape: Record<string, number | string>[];
  selection: Record<string, number | string>[];
  chosen: string; bias_pct: number;
  interval_pct: Record<string, { down: number; up: number }>;
  loo: { area: string; ape_bias_corrected: number; inside_80: number }[];
  backtest_quarters: string[];
  churn: {
    applied: boolean; adjustment: Record<string, number>; share_now: Record<string, number>;
    share_normal: Record<string, number>; flagged_revenue_avg_4q: Record<string, number>;
    backtest: { area: string; ape_without: number; ape_with: number }[];
  };
}

export interface ChartSpec {
  type: "line" | "bar" | "stacked_bar" | "area";
  x: string; x_format?: string; y_format?: string; highlight?: string; reference?: number;
  series: { key: string; label: string; color: string }[];
  data: Record<string, unknown>[];
}
export interface InsightCard {
  id: string; title: string; takeaway: string;
  metric: { label: string; value: number; format: string };
  chart: ChartSpec;
  table: { columns: { key: string; label: string; format?: string }[]; rows: Record<string, unknown>[] } | null;
}
export interface Insights { available: boolean; cards: InsightCard[]; error?: string | null }

export interface PipelineStatus {
  running: boolean; error: string | null; progress: string | null; generated_at: string | null;
}
