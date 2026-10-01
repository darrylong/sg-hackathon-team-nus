import { Box, Card, Heading, Text } from "grommet";
import type { ReactNode } from "react";

/** Every chart sits in a Grommet Card with a title, an optional subtitle and optional header controls. */
export function ChartCard({ title, subtitle, controls, children, fill }: {
  title: string; subtitle?: ReactNode; controls?: ReactNode; children: ReactNode; fill?: boolean;
}) {
  return (
    <Card pad="medium" gap="small" background="background-front" elevation="small" round="small" fill={fill ? "vertical" : undefined}>
      <Box direction="row" justify="between" align="start" gap="small" wrap>
        <Box>
          <Heading level={3} margin="none" size="small">{title}</Heading>
          {subtitle && <Text size="small" color="text-weak">{subtitle}</Text>}
        </Box>
        {controls}
      </Box>
      {children}
    </Card>
  );
}
