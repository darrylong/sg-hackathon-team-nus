import { Box, Notification, Spinner, Text } from "grommet";
import type { ReactNode } from "react";

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <Box pad="medium" direction="row" gap="small" align="center">
      <Spinner size="small" />
      <Text color="text-weak">{label}</Text>
    </Box>
  );
}

export function ErrorState({ error, title = "Could not load data" }: { error: Error; title?: string }) {
  return <Notification status="critical" title={title} message={error.message} />;
}

export function Empty({ message = "No matching partners" }: { message?: string }) {
  return (
    <Box pad="medium" align="center" justify="center" round="small" border={{ style: "dashed", color: "border" }}>
      <Text color="text-weak">{message}</Text>
    </Box>
  );
}

export function Unavailable({ what, error }: { what: string; error?: string | null }) {
  return (
    <Notification
      status="warning"
      title={`${what}: data unavailable`}
      message={error ?? "This part of the analysis failed to compute. The rest of the dashboard still works."}
    />
  );
}

/** Render children only when data is loaded; otherwise a spinner or an error. */
export function Gate<T>({ loading, error, data, children, label }: {
  loading: boolean; error: Error | null; data: T | null; children: (d: T) => ReactNode; label?: string;
}) {
  if (error) return <ErrorState error={error} />;
  if (loading && !data) return <Loading label={label} />;
  if (!data) return null;
  return <>{children(data)}</>;
}
