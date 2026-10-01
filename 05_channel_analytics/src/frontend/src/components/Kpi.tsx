import { Box, Card, Meter, Text } from "grommet";
import type { ReactNode } from "react";

export function KpiCard({ label, value, sub, subColor, onClick }: {
  label: string; value: ReactNode; sub?: ReactNode; subColor?: string; onClick?: () => void;
}) {
  return (
    <Card pad="medium" gap="xsmall" background="background-front" elevation="small" round="small" onClick={onClick}
      hoverIndicator={onClick ? "background-contrast" : undefined}>
      <Text size="small" color="text-weak">{label}</Text>
      <Text size="xxlarge" weight="bold" style={{ fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>{value}</Text>
      {sub && <Text size="small" color={subColor ?? "text-weak"}>{sub}</Text>}
    </Card>
  );
}

export function healthColor(score: number): string {
  if (score < 80) return "status-critical";
  if (score < 95) return "status-warning";
  return "status-ok";
}

export function HealthMeter({ score, size = "small" }: { score: number; size?: "xsmall" | "small" | "medium" }) {
  return (
    <Box direction="row" gap="small" align="center">
      <Meter type="bar" value={score} max={100} thickness="small" size={size}
        values={[{ value: score, color: healthColor(score), label: "health" }]} a11yTitle={`Health score ${score.toFixed(1)}`} />
      <Text size="small" style={{ fontVariantNumeric: "tabular-nums" }}>{score.toFixed(1)}</Text>
    </Box>
  );
}
