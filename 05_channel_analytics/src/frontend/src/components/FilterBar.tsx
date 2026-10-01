import { Box, Button, Select, SelectMultiple, Text } from "grommet";
import type { Filters, Meta } from "../api";
import { EMPTY_FILTERS } from "../api";

const LABELS: Record<string, string> = { region: "Region", tier: "Tier", product_family: "Product", partner_type: "Partner type" };

export function FilterBar({ meta, filters, onChange }: { meta: Meta; filters: Filters; onChange: (f: Filters) => void }) {
  const options: Record<string, string[]> = {
    region: meta.regions, tier: meta.tiers, product_family: meta.products, partner_type: meta.partner_types,
  };
  const active = Object.keys(LABELS).some((k) => (filters[k as keyof Filters] as string[]).length) || filters.q_from || filters.q_to;
  return (
    <Box direction="row" gap="small" wrap align="center" pad={{ vertical: "small" }}>
      {(Object.keys(LABELS) as (keyof typeof LABELS)[]).map((k) => (
        <Box key={k} width="small" margin={{ bottom: "xsmall" }}>
          <SelectMultiple
            placeholder={LABELS[k]}
            a11yTitle={`Filter by ${LABELS[k]}`}
            options={options[k]}
            value={filters[k as keyof Filters] as string[]}
            onChange={({ value }: { value: string[] }) => onChange({ ...filters, [k]: value })}
            showSelectedInline
          />
        </Box>
      ))}
      <Box width="xsmall" margin={{ bottom: "xsmall" }} style={{ minWidth: 120 }}>
        <Select placeholder="From" a11yTitle="From quarter" options={meta.quarters} value={filters.q_from ?? ""}
          onChange={({ option }: { option: string }) => onChange({ ...filters, q_from: option })} clear />
      </Box>
      <Box width="xsmall" margin={{ bottom: "xsmall" }} style={{ minWidth: 120 }}>
        <Select placeholder="To" a11yTitle="To quarter" options={meta.quarters} value={filters.q_to ?? ""}
          onChange={({ option }: { option: string }) => onChange({ ...filters, q_to: option })} clear />
      </Box>
      {active && <Button label="Reset" onClick={() => onChange(EMPTY_FILTERS)} margin={{ bottom: "xsmall" }} />}
      {filters.product_family.length > 0 && (
        <Text size="small" color="text-weak">Targets are per partner, so attainment is hidden when a product is selected.</Text>
      )}
    </Box>
  );
}
