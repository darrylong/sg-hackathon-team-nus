import {
  Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { CHART_FONT, resolveColor } from "../chartColors";
import { useMode } from "../context";
import { axisFormatter, formatValue } from "../format";

export interface Series { key: string; label: string; color: string }

/** Axis, grid and legend styling shared by every chart; re-computed when the theme mode changes. */
export function useChartTheme() {
  const { mode, palette } = useMode();
  return {
    mode,
    palette,
    axis: { stroke: palette.grid, tick: { fill: palette.axis, fontSize: 12, fontFamily: CHART_FONT }, tickLine: false },
    grid: { stroke: palette.grid, vertical: false, strokeDasharray: "0" },
    legend: { wrapperStyle: { fontFamily: CHART_FONT, fontSize: 12, color: palette.text, paddingTop: 8 } },
  };
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export function ChartTooltip({ active, payload, label, valueFormat, labelFormat }: any) {
  const { palette } = useMode();
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: palette.tooltipBg, border: `1px solid ${palette.tooltipBorder}`, borderRadius: "8px",
      boxShadow: "0 4px 16px rgba(0,0,0,0.16)", padding: "12px", fontFamily: CHART_FONT, color: palette.text,
      fontSize: 13, minWidth: 140,
    }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>{labelFormat ? labelFormat(label) : label}</div>
      {payload.filter((p: any) => p.value !== null && p.value !== undefined).map((p: any) => (
        <div key={p.dataKey} style={{ display: "flex", alignItems: "center", gap: 8, lineHeight: "20px" }}>
          <span style={{ width: 10, height: 10, borderRadius: 2, background: p.color || p.payload?.fill, flexShrink: 0 }} />
          <span style={{ flex: 1 }}>{p.name}</span>
          <span style={{ fontVariantNumeric: "tabular-nums", fontWeight: 600 }}>
            {Array.isArray(p.value) ? p.value.map((v: number) => formatValue(v, valueFormat)).join(" – ") : formatValue(p.value, valueFormat)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function SeriesChart({ type, data, x, series, yFormat, xFormat, height = 260, highlight, reference, stacked }: {
  type: "line" | "bar" | "stacked_bar" | "area"; data: Record<string, unknown>[]; x: string; series: Series[];
  yFormat?: string; xFormat?: string; height?: number; highlight?: string; reference?: number; stacked?: boolean;
}) {
  const t = useChartTheme();
  const colored = series.map((s) => ({ ...s, color: resolveColor(s.color, t.mode) }));
  const yTick = axisFormatter(yFormat);
  const xTick = xFormat ? axisFormatter(xFormat) : undefined;
  const isStacked = stacked || type === "stacked_bar";
  const common = {
    data,
    margin: { top: 8, right: 28, bottom: 0, left: 4 },
    style: { fontFamily: CHART_FONT, background: "transparent" },
  };
  const axes = [
    <CartesianGrid key="g" {...t.grid} />,
    <XAxis key="x" dataKey={x} {...t.axis} tickFormatter={xTick} minTickGap={16} />,
    <YAxis key="y" {...t.axis} tickFormatter={yTick} width={64} axisLine={false} />,
    <Tooltip key="t" content={<ChartTooltip valueFormat={yFormat} labelFormat={xTick} />}
      cursor={{ fill: t.palette.grid, fillOpacity: 0.5, stroke: t.palette.axis, strokeOpacity: 0.3 }} />,
    colored.length >= 2 ? <Legend key="l" {...t.legend} iconType="square" iconSize={10} /> : null,
    reference !== undefined ? <ReferenceLine key="r" y={reference} stroke={t.palette.axis} strokeDasharray="4 4" /> : null,
  ];

  let chart;
  if (type === "line") {
    chart = (
      <LineChart {...common}>
        {axes}
        {colored.map((s) => (
          <Line key={s.key} isAnimationActive={false} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2}
            dot={{ r: 3, strokeWidth: 0, fill: s.color }} activeDot={{ r: 5 }} connectNulls />
        ))}
      </LineChart>
    );
  } else if (type === "area") {
    chart = (
      <AreaChart {...common}>
        {axes}
        {colored.map((s) => (
          <Area key={s.key} isAnimationActive={false} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2}
            fill={s.color} fillOpacity={0.15} />
        ))}
      </AreaChart>
    );
  } else {
    chart = (
      <BarChart {...common} barGap={2} barCategoryGap="20%">
        {axes}
        {colored.map((s, i) => (
          <Bar key={s.key} isAnimationActive={false} dataKey={s.key} name={s.label} fill={s.color} maxBarSize={56}
            stackId={isStacked ? "a" : undefined}
            radius={!isStacked || i === colored.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]}
            stroke={t.palette.tooltipBg} strokeWidth={isStacked ? 1 : 0}>
            {highlight ? data.map((d, j) => <Cell key={j} fillOpacity={d[highlight] ? 1 : 0.45} />) : null}
          </Bar>
        ))}
      </BarChart>
    );
  }
  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer key={t.mode} width="100%" height="100%">{chart}</ResponsiveContainer>
    </div>
  );
}

/** Horizontal bars with one colour per row (used for segments, drivers, product mix). */
export function HBarChart({ data, labelKey, valueKey, colors, valueFormat, height }: {
  data: Record<string, unknown>[]; labelKey: string; valueKey: string; colors: string[] | string;
  valueFormat?: string; height?: number;
}) {
  const t = useChartTheme();
  const h = height ?? Math.max(120, data.length * 36 + 24);
  return (
    <div style={{ width: "100%", height: h }}>
      <ResponsiveContainer key={t.mode} width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 48, bottom: 4, left: 4 }}
          style={{ fontFamily: CHART_FONT, background: "transparent" }}>
          <CartesianGrid {...t.grid} horizontal={false} vertical />
          <XAxis type="number" {...t.axis} tickFormatter={axisFormatter(valueFormat)} />
          <YAxis type="category" dataKey={labelKey} {...t.axis} width={118} axisLine={false} />
          <Tooltip content={<ChartTooltip valueFormat={valueFormat} />} cursor={{ fill: t.palette.grid, fillOpacity: 0.5 }} />
          <Bar isAnimationActive={false} dataKey={valueKey} name={valueKey === "partners" ? "Partners" : "Value"} radius={[0, 4, 4, 0]} maxBarSize={24}
            label={{ position: "right", fill: t.palette.text, fontSize: 12, fontFamily: CHART_FONT,
              formatter: (v: unknown) => formatValue(v, valueFormat) }}>
            {data.map((_, i) => <Cell key={i} fill={Array.isArray(colors) ? colors[i % colors.length] : colors} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
