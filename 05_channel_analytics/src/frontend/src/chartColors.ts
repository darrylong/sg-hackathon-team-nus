// Chart colours. Recharts cannot read Grommet theme tokens, so every chart imports from here.
// Validated with the dataviz palette checker (lightness band, chroma, colour-blind separation, contrast):
// light mode keeps the HPE hues with amber darkened to #E5A000 (the original #FFB81C was 1.7:1 on white);
// dark mode uses the same hues stepped for a dark surface. Graph order puts purple between amber and red
// so the two never sit next to each other.

export type Mode = "light" | "dark";

interface ChartPalette {
  regions: Record<string, string>;
  status: { healthy: string; declining: string; atRisk: string };
  forecast: string;
  graph: string[];
  neutral: string;
  grid: string;
  axis: string;
  text: string;
  tooltipBg: string;
  tooltipBorder: string;
}

export const PALETTES: Record<Mode, ChartPalette> = {
  light: {
    regions: { AMS: "#0072CE", EMEA: "#01A982", APJ: "#E5A000" },
    status: { healthy: "#01A982", declining: "#E5A000", atRisk: "#FF5042" },
    forecast: "#7630EA",
    graph: ["#0072CE", "#01A982", "#E5A000", "#7630EA", "#FF5042", "#00B388"],
    neutral: "#B8B8B8",
    grid: "#EEEEEE",
    axis: "#666666",
    text: "#333333",
    tooltipBg: "#FFFFFF",
    tooltipBorder: "rgba(0,0,0,0.08)",
  },
  dark: {
    regions: { AMS: "#2A7FD4", EMEA: "#01A982", APJ: "#B08C00" },
    status: { healthy: "#01A982", declining: "#B08C00", atRisk: "#E04A3A" },
    forecast: "#9067F0",
    graph: ["#2A7FD4", "#01A982", "#B08C00", "#9067F0", "#E04A3A", "#00957A"],
    neutral: "#5A5A5A",
    grid: "#333333",
    axis: "#A8A8A8",
    text: "#E6E6E6",
    tooltipBg: "#262626",
    tooltipBorder: "rgba(255,255,255,0.12)",
  },
};

export const CHART_FONT = "'Metric', 'MetricHPE', sans-serif";

/** Resolve a colour token from the API ("region:AMS", "status:atRisk", "graph:2", "forecast", "neutral"). */
export function resolveColor(token: string, mode: Mode): string {
  const p = PALETTES[mode];
  const [kind, key] = token.split(":");
  if (kind === "region") return p.regions[key] ?? p.graph[0];
  if (kind === "status") return (p.status as Record<string, string>)[key] ?? p.neutral;
  if (kind === "graph") return p.graph[Number(key) % p.graph.length];
  if (kind === "forecast") return p.forecast;
  if (kind === "neutral") return p.neutral;
  return token;
}

/** Colour for a series key in a dimension (regions use region colours, everything else the graph order). */
export function seriesColor(dim: string, key: string, index: number, mode: Mode): string {
  const p = PALETTES[mode];
  if (dim === "region" && p.regions[key]) return p.regions[key];
  return p.graph[index % p.graph.length];
}

export const SEGMENT_COLOR: Record<string, (m: Mode) => string> = {
  "At risk": (m) => PALETTES[m].status.atRisk,
  "Quietly declining": (m) => PALETTES[m].status.declining,
  Growing: (m) => PALETTES[m].status.healthy,
  Stable: (m) => PALETTES[m].graph[0],
  "Ramping (new)": (m) => PALETTES[m].graph[3],
};
