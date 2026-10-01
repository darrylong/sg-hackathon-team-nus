import { Box, DataTable, Grid, Heading, Pagination, Paragraph, ResponsiveContext, Tab, Tabs, Text } from "grommet";
import { useContext, useState } from "react";
import type { AtRisk as AtRiskData, AtRiskRow, PartnerPage, PartnerRow } from "../api";
import { useApi } from "../api";
import { ChartCard } from "../components/ChartCard";
import { HBarChart } from "../components/charts";
import { HealthMeter, KpiCard } from "../components/Kpi";
import { RiskReason } from "./Performance";
import { Empty, Gate, Unavailable } from "../components/States";
import { AppContext, useMode } from "../context";
import { count, money, pct, pctSigned, truncate } from "../format";

const DRIVER_LABEL: Record<string, string> = {
  recency: "Recency", frequency: "Order frequency", volatility: "Volatility", size: "Account size", momentum: "Momentum",
  yoy: "Year on year", attainment: "Target attainment", breadth: "Product breadth", margin: "Margin", profile: "Profile",
};

function PartnerCell({ name, id }: { name: string; id: string }) {
  return <Box><Text weight="bold" size="small">{name}</Text><Text size="xsmall" color="text-weak">{id}</Text></Box>;
}

function FlaggedTable({ rows }: { rows: AtRiskRow[] }) {
  const { openPartner } = useContext(AppContext);
  if (!rows.length) return <Empty message="No partners flagged" />;
  return (
    <Box overflow={{ horizontal: "auto" }}>
      <DataTable
        primaryKey="partner_id"
        data={rows}
        sortable
        step={100}
        onClickRow={({ datum }) => openPartner(datum.partner_id)}
        columns={[
          { property: "risk_rank", header: "Rank", align: "end", size: "xsmall" },
          { property: "partner_name", header: "Partner", render: (r) => <PartnerCell name={r.partner_name} id={r.partner_id} /> },
          { property: "region", header: "Region" },
          { property: "tier", header: "Tier" },
          { property: "health_score", header: "Health", render: (r) => <HealthMeter score={r.health_score} size="xsmall" /> },
          { property: "p_leave", header: "P(leave)", align: "end", render: (r) => pct(r.p_leave) },
          { property: "reason_1", header: "Main reason (click for all)", sortable: false,
            render: (r) => <Text size="small">{truncate(r.reason_1, 80)}</Text> },
        ]}
      />
    </Box>
  );
}

function SegmentTable({ segment, sort, desc }: { segment?: string; sort: string; desc: boolean }) {
  const { openPartner } = useContext(AppContext);
  const [page, setPage] = useState(1);
  const res = useApi<PartnerPage>("/partners", { segment, sort, desc, page, page_size: 20 });
  return (
    <Gate {...res}>
      {(p) => p.rows.length ? (
        <Box gap="small">
          <Box overflow={{ horizontal: "auto" }}>
            <DataTable
              primaryKey="partner_id"
              data={p.rows}
              onClickRow={({ datum }) => openPartner(datum.partner_id)}
              columns={[
                { property: "partner_name", header: "Partner", render: (r: PartnerRow) => <PartnerCell name={r.partner_name} id={r.partner_id} /> },
                { property: "region", header: "Region" },
                { property: "tier", header: "Tier" },
                { property: "revenue_q2_2026", header: "Revenue 2026-Q2", align: "end", render: (r: PartnerRow) => money(r.revenue_q2_2026) },
                { property: "yoy_h1_pct", header: "H1 YoY", align: "end", render: (r: PartnerRow) => pctSigned(r.yoy_h1_pct, 0) },
                { property: "health_score", header: "Health", render: (r: PartnerRow) => <HealthMeter score={r.health_score} size="xsmall" /> },
                { property: "segment", header: "Segment" },
                { property: "reason_1", header: "Main risk factor", render: (r: PartnerRow) => <RiskReason row={r} /> },
              ]}
            />
          </Box>
          <Box direction="row" justify="between" align="center" wrap>
            <Text size="small" color="text-weak">{count(p.total)} partners</Text>
            <Pagination numberItems={p.total} step={p.page_size} page={p.page} onChange={({ page: n }) => setPage(n)} size="small" />
          </Box>
        </Box>
      ) : <Empty message="No matching partners" />}
    </Gate>
  );
}

export function AtRisk() {
  const size = useContext(ResponsiveContext);
  const { palette } = useMode();
  const res = useApi<AtRiskData>("/at-risk");
  const [tab, setTab] = useState(0);

  return (
    <Box gap="medium">
      <Box>
        <Heading level={1} size="small" margin="none">Partners at risk of leaving</Heading>
        <Text color="text-weak">
          At risk = likely to place no order, or only a token order, in 2026-Q3. Not the same as "revenue down".
        </Text>
      </Box>
      <Gate {...res} label="Scoring partners…">
        {(d) => !d.available ? <Unavailable what="At-risk scores" error={d.error} /> : (
          <>
            <Grid columns={{ count: size === "small" ? 1 : 4, size: "auto" }} gap="small">
              <KpiCard label="Flagged at risk" value={count(d.summary.flagged)} sub={`of ${count(d.summary.partners)} partners`} />
              <KpiCard label="Expected leavers in 2026-Q3" value={count(d.summary.expected_leavers)}
                sub={`Q3 seasonality x${d.summary.season_factor.toFixed(2)} applied`} />
              <KpiCard label="Revenue exposure" value={money(d.summary.flagged_revenue_avg_4q)}
                sub={`a quarter · ${pct(d.summary.flagged_share_of_revenue, 2)} of channel revenue`} />
              <KpiCard label="Backtest ROC-AUC" value={d.summary.backtest_auc.toFixed(3)}
                sub={`precision ${pct(d.summary.backtest_precision, 0)} · recall ${pct(d.summary.backtest_recall, 0)}`} />
            </Grid>

            <Grid columns={size === "small" ? ["1fr"] : ["1fr", "1fr"]} gap="medium">
              <ChartCard title="Main driver of each flag" subtitle="Which factor adds the most risk">
                <HBarChart data={d.drivers.map((x) => ({ ...x, label: DRIVER_LABEL[x.driver] ?? x.driver }))}
                  labelKey="label" valueKey="partners" valueFormat="count" colors={palette.status.atRisk} />
              </ChartCard>
              <ChartCard title="How the health score works">
                <Paragraph margin="none" fill size="small">
                  <b>Health score = 100 × (1 − P(leave in 2026-Q3))</b>. P(leave) comes from a logistic regression and a gradient-boosting
                  model trained on six past quarters, where "left" means no orders or a token order (under 10% of normal) in the next quarter.
                  Probabilities are calibrated on out-of-sample predictions and scaled for Q3, when partners skip orders
                  {` ${d.summary.season_factor.toFixed(2)}x`} more often.
                </Paragraph>
                <Paragraph margin="none" fill size="small">
                  The top {count(d.summary.flagged)} partners ({d.summary.flag_multiplier}x the expected number of leavers, the setting with the
                  best F1 in the backtest) are flagged. Each reason is the factor whose reset to a typical healthy same-tier partner lowers the
                  risk most, written from the partner's own numbers. Backtest AUC {d.summary.backtest_auc.toFixed(3)}; most past "leavers"
                  were small, irregular buyers who skipped a quarter, so precision is modest.
                </Paragraph>
              </ChartCard>
            </Grid>

            <ChartCard title="Partner lists" subtitle="Click any row for the full partner panel">
              <Tabs activeIndex={tab} onActive={setTab} justify="start">
                <Tab title={`At risk (${count(d.summary.flagged)})`}>
                  <Box pad={{ top: "small" }}><FlaggedTable rows={d.flagged} /></Box>
                </Tab>
                <Tab title={`Quietly declining (${count(d.segments.find((s) => s.segment === "Quietly declining")?.partners)})`}>
                  <Box pad={{ top: "small" }} gap="small">
                    <Text size="small" color="text-weak">
                      Revenue down 25%+ year on year for 2+ quarters but still ordering. Historically these partners did not leave more often than
                      similar partners, so they need a growth conversation rather than a retention alarm.
                    </Text>
                    <SegmentTable segment="Quietly declining" sort="yoy_h1_pct" desc={false} />
                  </Box>
                </Tab>
                <Tab title={`All partners (${count(d.summary.partners)})`}>
                  <Box pad={{ top: "small" }}><SegmentTable sort="p_leave" desc /></Box>
                </Tab>
              </Tabs>
            </ChartCard>
          </>
        )}
      </Gate>
    </Box>
  );
}
