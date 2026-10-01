import { Anchor, Box, Grid, Heading, Paragraph, ResponsiveContext, Text } from "grommet";
import { useContext } from "react";
import { useNavigate } from "react-router-dom";
import type { AtRisk, Insights, Kpis, Trend } from "../api";
import { useApi } from "../api";
import { SEGMENT_COLOR } from "../chartColors";
import { ChartCard } from "../components/ChartCard";
import { HBarChart, SeriesChart } from "../components/charts";
import { KpiCard } from "../components/Kpi";
import { Gate, Unavailable } from "../components/States";
import { useMode } from "../context";
import { count, millions, money, pct, pctPoints, pctSigned } from "../format";

export function Overview() {
  const size = useContext(ResponsiveContext);
  const navigate = useNavigate();
  const { mode } = useMode();
  const kpis = useApi<Kpis>("/kpis");
  const trend = useApi<Trend>("/trend", { grain: "quarter", by: "region" });
  const risk = useApi<AtRisk>("/at-risk");
  const insights = useApi<Insights>("/insights");
  const cols = size === "small" ? 1 : size === "medium" ? 3 : 6;

  return (
    <Box gap="medium">
      <Box>
        <Heading level={1} size="small" margin="none">Partner channel overview</Heading>
        <Text color="text-weak">Where revenue comes from, who is at risk of leaving, and what to expect in 2026-Q3.</Text>
      </Box>

      <Gate {...kpis} label="Running the analysis…">
        {(k) => (
          <Grid columns={{ count: cols, size: "auto" }} gap="small">
            <KpiCard label={`Revenue ${k.quarter}`} value={money(k.revenue)}
              sub={`${pctSigned(k.yoy)} vs ${k.quarter.replace(/^(\d{4})/, (y) => String(+y - 1))}`}
              subColor={k.yoy !== null && k.yoy >= 0 ? "status-ok" : "status-critical"} />
            <KpiCard label="Partner margin" value={pctPoints(k.margin_pct)} sub="revenue-weighted" />
            <KpiCard label="Target attainment" value={pct(k.attainment, 0)} sub={`channel total, ${k.quarter}`} />
            <KpiCard label="Active partners" value={count(k.active_partners)} sub={`ordered in ${k.quarter}`} />
            <KpiCard label="At-risk partners" value={k.at_risk ? count(k.at_risk.flagged) : "–"}
              sub={k.at_risk ? `of ${count(k.at_risk.partners)} · ${count(k.at_risk.declining)} quietly declining` : "data unavailable"}
              subColor="text-weak" onClick={() => navigate("/at-risk")} />
            <KpiCard label="2026-Q3 forecast" value={k.forecast ? money(k.forecast.forecast) : "–"}
              sub={k.forecast ? `80%: ${millions(k.forecast.lo80)} – ${millions(k.forecast.hi80)}` : "data unavailable"}
              onClick={() => navigate("/forecast")} />
          </Grid>
        )}
      </Gate>

      <Grid columns={size === "small" ? ["1fr"] : ["2fr", "1fr"]} gap="medium">
        <ChartCard title="Revenue by region" subtitle="Quarterly, outlier order line excluded">
          <Gate {...trend}>
            {(t) => (
              <SeriesChart type="line" data={t.rows} x="period" yFormat="money" height={280}
                series={t.keys.map((k) => ({ key: k, label: k, color: `region:${k}` }))} />
            )}
          </Gate>
        </ChartCard>
        <ChartCard title="Partner segments" subtitle="2026-Q2 snapshot">
          <Gate {...risk}>
            {(r) => r.available ? (
              <HBarChart data={r.segments} labelKey="segment" valueKey="partners" valueFormat="count"
                colors={r.segments.map((s) => SEGMENT_COLOR[s.segment]?.(mode) ?? "#888")} />
            ) : <Unavailable what="Segments" error={r.error} />}
          </Gate>
        </ChartCard>
      </Grid>

      <Box gap="small">
        <Box direction="row" justify="between" align="center">
          <Heading level={2} size="small" margin="none">Key insights</Heading>
          <Anchor label="All 10 insights" onClick={() => navigate("/insights")} />
        </Box>
        <Gate {...insights}>
          {(ins) => ins.available ? (
            <Grid columns={size === "small" ? ["1fr"] : { count: 3, size: "auto" }} gap="medium">
              {ins.cards.slice(0, 3).map((c) => (
                <ChartCard key={c.id} title={c.title}>
                  <Paragraph margin="none" fill size="small">{c.takeaway}</Paragraph>
                </ChartCard>
              ))}
            </Grid>
          ) : <Unavailable what="Insights" error={ins.error} />}
        </Gate>
      </Box>
    </Box>
  );
}
