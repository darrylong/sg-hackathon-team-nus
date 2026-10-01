import { Anchor, Box, Button, DataTable, Grid, Heading, Layer, Tag, Text } from "grommet";
import { Close, ShareRounded } from "grommet-icons";
import { useNavigate } from "react-router-dom";
import { Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PartnerDetail } from "../api";
import { ApiError, useApi } from "../api";
import { CHART_FONT, SEGMENT_COLOR } from "../chartColors";
import { useMode } from "../context";
import { count, decimal, money, moneyAxis, pct, pctSigned } from "../format";
import { ChartCard } from "./ChartCard";
import { ChartTooltip, HBarChart, SeriesChart, useChartTheme } from "./charts";
import { HealthMeter } from "./Kpi";
import { Empty, ErrorState, Loading } from "./States";

const DRIVER_LABEL: Record<string, string> = {
  recency: "Recency", frequency: "Order frequency", volatility: "Volatility", size: "Account size", momentum: "Momentum",
  yoy: "Year on year", attainment: "Target attainment", breadth: "Product breadth", margin: "Margin", profile: "Profile",
};

function formatSignal(key: string, v: number | null): string {
  if (v === null || v === undefined) return "–";
  if (key === "active_month_share_6m") return `${Math.round(v * 6)} of 6`;
  if (key === "att_avg_4q") return pct(v, 0);
  if (key === "margin_pct_4q") return `${v.toFixed(1)}%`;
  if (key.includes("days") || key === "tenure_months" || key === "n_products_2q") return count(v);
  return decimal(v, key === "rev_cv_6m" ? 2 : 1);
}

function RevenueVsTarget({ rows }: { rows: PartnerDetail["quarterly"] }) {
  const t = useChartTheme();
  return (
    <div style={{ width: "100%", height: 220 }}>
      <ResponsiveContainer key={t.mode} width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 8, bottom: 0, left: 4 }} style={{ fontFamily: CHART_FONT }}>
          <CartesianGrid {...t.grid} />
          <XAxis dataKey="quarter" {...t.axis} minTickGap={12} />
          <YAxis {...t.axis} tickFormatter={moneyAxis} width={60} axisLine={false} />
          <Tooltip content={<ChartTooltip valueFormat="money" />} cursor={{ fill: t.palette.grid, fillOpacity: 0.5 }} />
          <Legend {...t.legend} iconType="square" iconSize={10} />
          <Bar isAnimationActive={false} dataKey="revenue_usd" name="Revenue" fill={t.palette.graph[0]} radius={[4, 4, 0, 0]} maxBarSize={36} />
          <Line isAnimationActive={false} dataKey="target_usd" name="Target" stroke={t.palette.forecast} strokeWidth={2} dot={{ r: 3 }} connectNulls />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function PartnerContent({ id, compact }: { id: string; compact?: boolean }) {
  const { data, error, loading } = useApi<PartnerDetail>(`/partners/${encodeURIComponent(id)}`);
  const { mode, palette } = useMode();
  if (error) {
    return (error as ApiError).status === 404 ? <Empty message={`Partner ${id} not found`} /> : <ErrorState error={error} />;
  }
  if (loading || !data) return <Loading label="Loading partner…" />;
  const { profile: p, score: s } = data;
  const segColor = s ? SEGMENT_COLOR[s.segment]?.(mode) : undefined;
  const lowRisk = !!s && s.at_risk !== 1 && s.p_leave < 0.05;
  return (
    <Box gap="medium" flex={false}>
      <Box gap="xsmall" flex={false}>
        <Heading level={2} margin="none" size="small">{p.partner_name}</Heading>
        <Box direction="row" gap="xsmall" wrap align="center">
          <Text size="small" color="text-weak">{p.partner_id}</Text>
          {[p.region, p.tier, p.partner_type, p.country].map((v) => <Tag key={v} value={v} size="small" />)}
          <Text size="small" color="text-weak">Onboarded {p.onboarded_date}</Text>
        </Box>
      </Box>

      {s && (
        <Grid columns={compact ? ["1fr", "1fr"] : { count: 4, size: "auto" }} gap="small">
          <Box gap="xxsmall">
            <Text size="small" color="text-weak">Health score</Text>
            <HealthMeter score={s.health_score} />
            <Text size="xsmall" color="text-weak">Rank {count(s.risk_rank)} of {count(s.partners)} by risk</Text>
          </Box>
          <Box gap="xxsmall">
            <Text size="small" color="text-weak">P(leave in 2026-Q3)</Text>
            <Text size="large" weight="bold">{pct(s.p_leave)}</Text>
          </Box>
          <Box gap="xxsmall">
            <Text size="small" color="text-weak">Segment</Text>
            <Box direction="row" gap="xsmall" align="center">
              <Box width="10px" height="10px" round="2px" background={segColor} />
              <Text weight="bold">{s.segment}</Text>
            </Box>
            {s.at_risk === 1 && <Text size="xsmall" color="status-critical">Flagged at risk</Text>}
          </Box>
          <Box gap="xxsmall">
            <Text size="small" color="text-weak">Avg quarterly revenue (4 q)</Text>
            <Text size="large" weight="bold">{money(s.revenue_avg_4q)}</Text>
            <Text size="xsmall" color="text-weak">H1 YoY {pctSigned(s.yoy_h1_pct)} · last order {s.last_order ?? "never"}</Text>
          </Box>
        </Grid>
      )}

      <Box gap="xsmall" flex={false}>
        <Heading level={4} margin="none">Why this score</Heading>
        {lowRisk && (
          <Box pad="small" round="small" background="background-contrast" flex={false}>
            <Text>Low risk: {s ? pct(s.p_leave) : "–"} chance of leaving in 2026-Q3. No factor adds meaningful risk.</Text>
          </Box>
        )}
        {!lowRisk && data.reasons.length === 0 && <Text color="text-weak">No reasons available.</Text>}
        {!lowRisk && data.reasons.map((r, i) => (
          <Box key={i} direction="row" gap="small" align="start" pad="small" round="small" background="background-contrast" flex={false}>
            <Text weight="bold" style={{ minWidth: 18 }}>{i + 1}.</Text>
            <Box flex>
              <Text>{r.text}</Text>
              <Text size="xsmall" color="text-weak">
                {DRIVER_LABEL[r.driver] ?? r.driver}
                {r.points !== null && r.points !== undefined && r.points > 0 ? ` · adds ${r.points.toFixed(1)} pts of risk` : ""}
              </Text>
            </Box>
          </Box>
        ))}
        {!lowRisk && (
          <Text size="xsmall" color="text-weak">
            Points = how much the risk drops if this factor were typical for a healthy {p.tier} partner.
          </Text>
        )}
      </Box>

      <Grid columns={compact ? ["1fr"] : { count: 2, size: "auto" }} gap="medium">
        <ChartCard title="Revenue vs target" subtitle="By quarter">
          <RevenueVsTarget rows={data.quarterly} />
        </ChartCard>
        <ChartCard title="Order lines per month" subtitle="Ordering rhythm">
          <SeriesChart type="bar" data={data.monthly} x="month" yFormat="count" height={220}
            series={[{ key: "order_lines", label: "Order lines", color: "graph:0" }]} />
        </ChartCard>
        <ChartCard title="Product mix" subtitle={data.mix_period}>
          {data.product_mix.length ? (
            <HBarChart data={data.product_mix} labelKey="product" valueKey="revenue" valueFormat="money" colors={palette.graph} />
          ) : <Empty message="No orders in the last 4 quarters" />}
        </ChartCard>
        {data.signals && (
          <ChartCard title="Signals vs tier median" subtitle={`Compared with the median ${p.tier} partner`}>
            <DataTable
              primaryKey="key"
              data={data.signals}
              columns={[
                { property: "label", header: "Signal" },
                { property: "value", header: "Partner", align: "end", render: (r) => formatSignal(r.key, r.value) },
                { property: "tier_median", header: "Tier median", align: "end", render: (r) => formatSignal(r.key, r.tier_median) },
              ]}
            />
          </ChartCard>
        )}
      </Grid>
    </Box>
  );
}

/** Right-hand slide-over used by every table row click. */
export function PartnerLayer({ id, onClose }: { id: string; onClose: () => void }) {
  const navigate = useNavigate();
  return (
    <Layer position="right" full="vertical" modal onEsc={onClose} onClickOutside={onClose} responsive>
      <Box width={{ min: "min(760px, 100vw)", max: "760px" }} fill="vertical" overflow="auto" pad="medium" gap="small"
        background="background-back">
        <Box direction="row" justify="between" align="center" flex={false}>
          <Anchor label="Open as page" icon={<ShareRounded size="small" />} size="small"
            onClick={() => { onClose(); navigate(`/partners/${id}`); }} />
          <Button icon={<Close />} a11yTitle="Close partner panel" onClick={onClose} />
        </Box>
        <PartnerContent id={id} compact />
      </Box>
    </Layer>
  );
}
