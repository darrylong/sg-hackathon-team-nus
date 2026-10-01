import { Box, DataTable, Grid, Heading, Paragraph, ResponsiveContext, Tab, Tabs, Text } from "grommet";
import { useContext, useMemo, useState } from "react";
import { Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ForecastData } from "../api";
import { useApi } from "../api";
import { CHART_FONT, PALETTES } from "../chartColors";
import { ChartCard } from "../components/ChartCard";
import { ChartTooltip, useChartTheme } from "../components/charts";
import { KpiCard } from "../components/Kpi";
import { Gate, Unavailable } from "../components/States";
import { millions, money, moneyAxis, pct, pctSigned } from "../format";

const AREAS = ["ALL", "AMS", "APJ", "EMEA"];

function ForecastChart({ data, area }: { data: ForecastData; area: string }) {
  const t = useChartTheme();
  const color = area === "ALL" ? t.palette.text : PALETTES[t.mode].regions[area];
  const rows = useMemo(() => {
    const sub = data.submission.find((s) => s.region === area)!;
    const r: Record<string, unknown>[] = data.history.map((h) => ({ quarter: h.quarter, actual: h[area] }));
    const last = r[r.length - 1];
    last.forecast = last.actual;
    last.band = [last.actual, last.actual];
    r.push({ quarter: data.target_quarter, forecast: sub.forecast, band: [sub.lo80, sub.hi80] });
    return r;
  }, [data, area]);
  return (
    <div style={{ width: "100%", height: 320 }}>
      <ResponsiveContainer key={`${t.mode}-${area}`} width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 36, bottom: 0, left: 4 }} style={{ fontFamily: CHART_FONT }}>
          <CartesianGrid {...t.grid} />
          <XAxis dataKey="quarter" {...t.axis} />
          <YAxis {...t.axis} tickFormatter={moneyAxis} width={70} axisLine={false} domain={["auto", "auto"]} />
          <Tooltip content={<ChartTooltip valueFormat="money" />} />
          <Legend {...t.legend} iconType="square" iconSize={10} />
          <Area isAnimationActive={false} dataKey="band" name="80% interval" stroke="none" fill={t.palette.forecast} fillOpacity={0.2} />
          <Line isAnimationActive={false} dataKey="actual" name="Actual" stroke={color} strokeWidth={2} dot={{ r: 3, strokeWidth: 0, fill: color }} />
          <Line isAnimationActive={false} dataKey="forecast" name="Forecast" stroke={t.palette.forecast} strokeWidth={2} strokeDasharray="6 4"
            dot={{ r: 4, strokeWidth: 0, fill: t.palette.forecast }} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Forecast() {
  const size = useContext(ResponsiveContext);
  const res = useApi<ForecastData>("/forecast");
  const [areaIdx, setAreaIdx] = useState(0);

  return (
    <Box gap="medium">
      <Box>
        <Heading level={1} size="small" margin="none">2026-Q3 revenue forecast</Heading>
        <Text color="text-weak">Forecast for July–September 2026 by region, with an 80% interval.</Text>
      </Box>
      <Gate {...res} label="Forecasting…">
        {(d) => {
          if (!d.available) return <Unavailable what="Forecast" error={d.error} />;
          const q3ly = (area: string) => Number(d.history.find((h) => h.quarter === "2025-Q3")?.[area]);
          const labels = d.method_labels;
          const methodKeys = Object.keys(labels).filter((k) => k in d.methods[0]);
          const methodRows = methodKeys.map((m) => ({
            method: m, label: labels[m], ...Object.fromEntries(d.methods.map((r) => [r.area as string, r[m]])),
          }));
          return (
            <>
              <Grid columns={{ count: size === "small" ? 1 : 4, size: "auto" }} gap="small">
                {AREAS.map((a) => {
                  const s = d.submission.find((x) => x.region === a)!;
                  return (
                    <KpiCard key={a} label={a === "ALL" ? "All regions" : a} value={millions(s.forecast)}
                      sub={`80%: ${millions(s.lo80)} – ${millions(s.hi80)} · ${pctSigned(s.forecast / q3ly(a) - 1)} vs 2025-Q3`} />
                  );
                })}
              </Grid>

              <ChartCard title="History and forecast" subtitle="Quarterly revenue, outlier order line excluded; shaded band = 80% interval">
                <Tabs activeIndex={areaIdx} onActive={setAreaIdx} justify="start">
                  {AREAS.map((a) => <Tab key={a} title={a === "ALL" ? "All regions" : a} />)}
                </Tabs>
                <ForecastChart data={d} area={AREAS[areaIdx]} />
              </ChartCard>

              <Grid columns={size === "small" ? ["1fr"] : ["1fr", "1fr"]} gap="medium">
                <ChartCard title="How the forecast is made">
                  <Paragraph margin="none" fill size="small">
                    Six methods forecast each region: seasonal ratio, YoY growth, a monthly regression, an ARIMA model, a SARIMAX model that uses the
                    partner targets, and target × usual attainment. The 2026-Q3 target is not in the data, so it is projected from earlier targets.
                  </Paragraph>
                  <Paragraph margin="none" fill size="small">
                    A rule fixed in advance picks the ensemble with the lower out-of-sample error: <b>{labels[d.chosen]}</b>. The ensemble ran
                    high in the backtest because growth is slowing, so the forecast is adjusted by <b>{pctSigned(d.bias_pct)}</b>. The 80% interval comes from the
                    spread of backtest errors (±{pct(d.interval_pct.ALL.up)} for all regions). ALL is the sum of the regions.
                  </Paragraph>
                  <Paragraph margin="none" fill size="small">
                    Churn link: the at-risk model implies an extra {money(-d.churn.adjustment.ALL)} of lost revenue. It was{" "}
                    <b>{d.churn.applied ? "applied" : "not applied"}</b>, because adjusting for it {d.churn.applied ? "lowered" : "did not lower"} the backtest error.
                  </Paragraph>
                </ChartCard>
                <ChartCard title="Out-of-sample check" subtitle={`Leave one quarter out, ${d.backtest_quarters[0]} – ${d.backtest_quarters[d.backtest_quarters.length - 1]}`}>
                  <DataTable
                    primaryKey="area"
                    data={d.loo}
                    columns={[
                      { property: "area", header: "Area" },
                      { property: "ape_bias_corrected", header: "Mean abs. % error", align: "end", render: (r) => pct(r.ape_bias_corrected) },
                      { property: "inside_80", header: "Actual inside 80% range", align: "end", render: (r) => pct(r.inside_80, 0) },
                    ]}
                  />
                  <Text size="xsmall" color="text-weak">Five backtest quarters only; treat coverage as indicative.</Text>
                </ChartCard>
              </Grid>

              <ChartCard title="2026-Q3 forecast by method" subtitle="Bold = the forecast submitted">
                <Box overflow={{ horizontal: "auto" }}>
                  <DataTable
                    primaryKey="method"
                    data={methodRows}
                    columns={[
                      { property: "label", header: "Method", render: (r) => <Text weight={r.method === "final" ? "bold" : undefined}>{r.label}</Text> },
                      ...AREAS.map((a) => ({
                        property: a, header: a, align: "end" as const,
                        render: (r: Record<string, unknown>) => <Text weight={r.method === "final" ? "bold" : undefined}>{millions(r[a] as number)}</Text>,
                      })),
                    ]}
                  />
                </Box>
              </ChartCard>

              <Grid columns={size === "small" ? ["1fr"] : ["3fr", "2fr"]} gap="medium">
                <ChartCard title="Backtest accuracy by method" subtitle="Mean absolute % error before bias correction">
                  <Box overflow={{ horizontal: "auto" }}>
                    <DataTable
                      primaryKey="method"
                      data={d.backtest_mape}
                      columns={[
                        { property: "method", header: "Method", render: (r) => labels[r.method as string] ?? r.method },
                        ...AREAS.map((a) => ({ property: a, header: a, align: "end" as const, render: (r: Record<string, unknown>) => pct(r[a]) })),
                      ]}
                    />
                  </Box>
                </ChartCard>
                <ChartCard title="Ensemble selection" subtitle="Leave-one-quarter-out error after bias correction">
                  <DataTable
                    primaryKey="ensemble"
                    data={d.selection}
                    columns={[
                      { property: "ensemble", header: "Ensemble", render: (r) => (
                        <Text weight={r.ensemble === d.chosen ? "bold" : undefined}>{labels[r.ensemble as string] ?? r.ensemble}</Text>) },
                      { property: "mean", header: "Mean error", align: "end", render: (r) => pct(r.mean) },
                    ]}
                  />
                </ChartCard>
              </Grid>
            </>
          );
        }}
      </Gate>
    </Box>
  );
}
