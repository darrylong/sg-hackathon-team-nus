import { Box, DataTable, Grid, Heading, Pagination, ResponsiveContext, Select, Text, TextInput } from "grommet";
import { Search } from "grommet-icons";
import { useContext, useEffect, useState } from "react";
import type { Breakdown, Filters, Meta, PartnerPage, Trend } from "../api";
import { EMPTY_FILTERS, filterParams, useApi } from "../api";
import { ChartCard } from "../components/ChartCard";
import { SeriesChart } from "../components/charts";
import { FilterBar } from "../components/FilterBar";
import { HealthMeter } from "../components/Kpi";
import { Empty, Gate } from "../components/States";
import { AppContext } from "../context";
import { count, money, pct, pctPoints, pctSigned, truncate } from "../format";

const DIMS = [
  { value: "region", label: "Region" }, { value: "tier", label: "Tier" },
  { value: "product_family", label: "Product" }, { value: "partner_type", label: "Partner type" },
];
const SORTS = [
  { value: "revenue_range", label: "Revenue (selected period)" }, { value: "health_score", label: "Health score (lowest first)", asc: true },
  { value: "yoy_h1_pct", label: "H1 YoY growth (lowest first)", asc: true }, { value: "margin_pct", label: "Margin %" },
  { value: "partner_name", label: "Name", asc: true },
];

/** Reasons only mean something for partners with real risk; everyone else is "Low risk". */
export function RiskReason({ row }: { row: { at_risk: number; p_leave: number; reason_1: string | null } }) {
  if (row.at_risk === 1 || row.p_leave >= 0.05) return <Text size="small">{truncate(row.reason_1)}</Text>;
  return <Text size="small" color="text-weak">Low risk ({pct(row.p_leave)} chance of leaving)</Text>;
}

function seriesFor(t: Trend, dim: string) {
  return t.keys.map((k, i) => ({ key: k, label: k, color: dim === "region" ? `region:${k}` : `graph:${i}` }));
}

export function Performance({ meta }: { meta: Meta }) {
  const size = useContext(ResponsiveContext);
  const { openPartner } = useContext(AppContext);
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [dim, setDim] = useState("region");
  const [grain, setGrain] = useState("quarter");
  const [search, setSearch] = useState("");
  const [debounced, setDebounced] = useState("");
  const [sort, setSort] = useState(SORTS[0]);
  const [page, setPage] = useState(1);
  const fp = filterParams(filters);

  useEffect(() => { const h = setTimeout(() => setDebounced(search), 300); return () => clearTimeout(h); }, [search]);
  useEffect(() => setPage(1), [debounced, sort, JSON.stringify(filters)]);

  const trend = useApi<Trend>("/trend", { ...fp, grain, by: dim });
  const breakdown = useApi<Breakdown>("/breakdown", { ...fp, dim });
  const partners = useApi<PartnerPage>("/partners", {
    ...fp, search: debounced, sort: sort.value, desc: !sort.asc, page, page_size: 15,
  });
  const dimLabel = DIMS.find((d) => d.value === dim)?.label ?? dim;

  return (
    <Box gap="medium">
      <Box>
        <Heading level={1} size="small" margin="none">Performance</Heading>
        <Text color="text-weak">Revenue, margin and target attainment by partner, region, tier and product.</Text>
      </Box>
      <FilterBar meta={meta} filters={filters} onChange={setFilters} />

      <ChartCard
        title={`Revenue by ${dimLabel.toLowerCase()}`}
        subtitle={grain === "quarter" ? "Quarterly, stacked" : "Monthly, stacked"}
        controls={
          <Box direction="row" gap="small">
            <Select options={DIMS} labelKey="label" valueKey={{ key: "value", reduce: true }} value={dim}
              onChange={({ value }: { value: string }) => setDim(value)} a11yTitle="Group by" />
            <Select options={[{ value: "quarter", label: "Quarterly" }, { value: "month", label: "Monthly" }]} labelKey="label"
              valueKey={{ key: "value", reduce: true }} value={grain} onChange={({ value }: { value: string }) => setGrain(value)}
              a11yTitle="Time grain" />
          </Box>
        }
      >
        <Gate {...trend}>
          {(t) => t.rows.length ? (
            <SeriesChart type="stacked_bar" data={t.rows} x="period" yFormat="money" height={300} series={seriesFor(t, dim)} />
          ) : <Empty message="No matching sales for these filters" />}
        </Gate>
      </ChartCard>

      <Grid columns={size === "small" ? ["1fr"] : ["3fr", "2fr"]} gap="medium">
        <ChartCard title={`Breakdown by ${dimLabel.toLowerCase()}`}
          subtitle={breakdown.data?.growth_label ? `Growth: ${breakdown.data.growth_label}` : undefined}>
          <Gate {...breakdown}>
            {(b) => b.rows.length ? (
              <Box overflow={{ horizontal: "auto" }}>
                <DataTable
                  primaryKey="key"
                  data={b.rows}
                  sortable
                  columns={[
                    { property: "key", header: dimLabel, primary: true },
                    { property: "revenue", header: "Revenue", align: "end", render: (r) => money(r.revenue) },
                    { property: "share", header: "Share", align: "end", render: (r) => pct(r.share) },
                    { property: "margin_pct", header: "Margin", align: "end", render: (r) => pctPoints(r.margin_pct) },
                    { property: "growth_yoy", header: "YoY", align: "end", render: (r) => (
                      <Text color={r.growth_yoy !== null && r.growth_yoy < 0 ? "status-critical" : undefined}>{pctSigned(r.growth_yoy)}</Text>) },
                    ...(b.attainment_label ? [
                      { property: "attainment", header: "Attainment", align: "end" as const, render: (r: typeof b.rows[0]) => pct(r.attainment, 0) },
                      { property: "pct_partner_quarters_hit", header: "At target", align: "end" as const,
                        render: (r: typeof b.rows[0]) => pct(r.pct_partner_quarters_hit, 0) },
                    ] : []),
                    { property: "partners", header: "Partners", align: "end", render: (r) => count(r.partners) },
                  ]}
                />
                {b.attainment_label && (
                  <Text size="xsmall" color="text-weak" margin={{ top: "xsmall" }}>
                    Attainment = revenue / target over {b.attainment_label}; "At target" = share of partner-quarters at or above target.
                  </Text>
                )}
              </Box>
            ) : <Empty message="No matching sales for these filters" />}
          </Gate>
        </ChartCard>
        <ChartCard title="Partner margin %" subtitle={`By ${grain}, all selected sales`}>
          <Gate {...trend}>
            {(t) => t.rows.length ? (
              <SeriesChart type="line" data={t.rows} x="period" yFormat="pct_points" height={260}
                series={[{ key: "margin_pct", label: "Margin %", color: "graph:1" }]} />
            ) : <Empty message="No matching sales for these filters" />}
          </Gate>
        </ChartCard>
      </Grid>

      <ChartCard
        title="Partners"
        subtitle="Click a partner for its health score, reasons and history"
        controls={
          <Box direction="row" gap="small" wrap>
            <Box width="medium">
              <TextInput icon={<Search />} placeholder="Search name or ID" value={search}
                onChange={(e) => setSearch(e.target.value)} a11yTitle="Search partners" />
            </Box>
            <Select options={SORTS} labelKey="label" value={sort} onChange={({ option }) => setSort(option)} a11yTitle="Sort partners" />
          </Box>
        }
      >
        <Gate {...partners}>
          {(p) => p.rows.length ? (
            <Box gap="small">
              <Box overflow={{ horizontal: "auto" }}>
                <DataTable
                  primaryKey="partner_id"
                  data={p.rows}
                  onClickRow={({ datum }) => openPartner(datum.partner_id)}
                  columns={[
                    { property: "partner_name", header: "Partner", render: (r) => (
                      <Box><Text weight="bold" size="small">{r.partner_name}</Text><Text size="xsmall" color="text-weak">{r.partner_id}</Text></Box>) },
                    { property: "region", header: "Region" },
                    { property: "tier", header: "Tier" },
                    { property: "revenue_range", header: "Revenue", align: "end", render: (r) => money(r.revenue_range) },
                    { property: "margin_pct", header: "Margin", align: "end", render: (r) => pctPoints(r.margin_pct) },
                    { property: "yoy_h1_pct", header: "H1 YoY", align: "end", render: (r) => pctSigned(r.yoy_h1_pct, 0) },
                    { property: "health_score", header: "Health", render: (r) => <HealthMeter score={r.health_score} size="xsmall" /> },
                    { property: "segment", header: "Segment" },
                    { property: "reason_1", header: "Main risk factor", render: (r) => <RiskReason row={r} /> },
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
      </ChartCard>
    </Box>
  );
}
