import { Box, DataTable, Grid, Heading, Paragraph, ResponsiveContext, Text } from "grommet";
import { useContext } from "react";
import type { InsightCard, Insights as InsightsData } from "../api";
import { useApi } from "../api";
import { ChartCard } from "../components/ChartCard";
import { SeriesChart } from "../components/charts";
import { Gate, Unavailable } from "../components/States";
import { formatValue } from "../format";

function Card({ c, index }: { c: InsightCard; index: number }) {
  return (
    <ChartCard title={`${index + 1}. ${c.title}`}>
      <Box direction="row" gap="small" align="baseline" wrap>
        <Text size="xxlarge" weight="bold" style={{ fontVariantNumeric: "tabular-nums" }}>{formatValue(c.metric.value, c.metric.format)}</Text>
        <Text size="small" color="text-weak">{c.metric.label}</Text>
      </Box>
      <SeriesChart type={c.chart.type} data={c.chart.data} x={c.chart.x} series={c.chart.series} yFormat={c.chart.y_format}
        xFormat={c.chart.x_format} highlight={c.chart.highlight} reference={c.chart.reference} height={240} />
      <Paragraph margin="none" fill size="small">{c.takeaway}</Paragraph>
      {c.table && (
        <Box overflow={{ horizontal: "auto" }}>
          <DataTable
            size="small"
            primaryKey={c.table.columns[0].key}
            data={c.table.rows}
            columns={c.table.columns.map((col, i) => ({
              property: col.key, header: <Text size="small" weight="bold">{col.label}</Text>, align: i === 0 ? "start" : "end",
              render: (r: Record<string, unknown>) => <Text size="small">{i === 0 ? String(r[col.key]) : formatValue(r[col.key], col.format)}</Text>,
            }))}
          />
        </Box>
      )}
    </ChartCard>
  );
}

export function Insights() {
  const size = useContext(ResponsiveContext);
  const { data, error, loading } = useApi<InsightsData>("/insights");
  return (
    <Box gap="medium">
      <Box>
        <Heading level={1} size="small" margin="none">Key insights</Heading>
        <Text color="text-weak">
          What the headline numbers hide. Every figure below is computed from the sales, partner and target data.
        </Text>
      </Box>
      <Gate loading={loading} error={error} data={data}>
        {(d) => d.available ? (
          <Grid columns={size === "small" ? ["1fr"] : { count: 2, size: "auto" }} gap="medium">
            {d.cards.map((c, i) => <Card key={c.id} c={c} index={i} />)}
          </Grid>
        ) : <Unavailable what="Insights" error={d.error} />}
      </Gate>
    </Box>
  );
}
