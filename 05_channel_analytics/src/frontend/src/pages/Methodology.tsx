import { Box, Grid, Heading, Paragraph, ResponsiveContext, Text } from "grommet";
import { useContext } from "react";
import type { Meta } from "../api";
import { ChartCard } from "../components/ChartCard";
import { count } from "../format";

export function Methodology({ meta }: { meta: Meta }) {
  const size = useContext(ResponsiveContext);
  const P = ({ children }: { children: React.ReactNode }) => <Paragraph margin="none" fill size="small">{children}</Paragraph>;
  return (
    <Box gap="medium">
      <Box>
        <Heading level={1} size="small" margin="none">Methodology and limitations</Heading>
        <Text color="text-weak">How the numbers in this dashboard are produced, and where they can be wrong.</Text>
      </Box>
      <Grid columns={size === "small" ? ["1fr"] : { count: 2, size: "auto" }} gap="medium">
        <ChartCard title="Data">
          <P>
            {count(meta.order_lines)} order lines from {meta.data_from} to {meta.data_to}, {count(meta.partners)} partners and their quarterly
            targets. The three files join cleanly: no missing values, duplicates or unknown partner IDs.
          </P>
          <P>
            {meta.outlier_lines_excluded} order line is excluded everywhere: a GreenLake line with quantity 12,800 worth $37.8M, about 47x
            the next-largest line. Margin % is always revenue-weighted. Partners onboarded less than 12 months before the cutoff count as
            "ramping".
          </P>
        </ChartCard>
        <ChartCard title="At-risk label and model">
          <P>
            No churn labels are given, so past quarters are labelled with the judges' definition: a partner "left" if it placed no orders, or
            only a token order (under 10% of its average over the previous four quarters), in the next quarter.
          </P>
          <P>
            57 features per partner and quarter cover revenue trend, ordering rhythm, target attainment, margin, product breadth, year-on-year
            change and profile, all computed only from data before each cutoff. The model averages a logistic regression and gradient boosting,
            is calibrated on out-of-sample predictions, and is scaled for Q3 seasonality. XGBoost, LightGBM and reweighting were tested and
            did not beat it.
          </P>
        </ChartCard>
        <ChartCard title="Reasons for each flag">
          <P>
            For each partner, every driver group (recency, order frequency, size, momentum, year on year, attainment, product breadth,
            margin, volatility) is reset to the median of healthy partners in the same tier. The drop in risk ranks the drivers. The top
            three are written as sentences from the partner's own numbers, so no figure is invented.
          </P>
        </ChartCard>
        <ChartCard title="Forecast">
          <P>
            Six methods per region, including ARIMA and a SARIMAX model with partner targets. A rule fixed in advance chooses the ensemble by
            out-of-sample error, and the forecast is corrected for the backtest bias. The 80% interval comes from backtest errors. A
            churn adjustment from the at-risk model is applied only if it improves the backtest.
          </P>
        </ChartCard>
        <ChartCard title="Limitations">
          <P>
            The history has almost no permanent leavers. Most past "leavers" are small, irregular partners who skipped a quarter, so
            flag precision is modest (about 10% in the backtest) even though the ranking is good (AUC about 0.86).
          </P>
          <P>
            Thirty months of history leave only five backtest quarters for the forecast, so intervals and method choices are uncertain. The 2026-Q3
            target is projected. Reasons explain the model, not causation.
          </P>
        </ChartCard>
        <ChartCard title="Reproduce">
          <P>
            <code>./start.sh</code> starts the API and this dashboard. <code>python -m src.run_all</code> regenerates the submission files and
            reports. "Refresh analysis" in the header re-runs the analysis on the server without stopping the dashboard.
          </P>
        </ChartCard>
      </Grid>
    </Box>
  );
}
