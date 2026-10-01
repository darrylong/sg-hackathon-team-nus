import { Box, Button, Notification, Spinner, Text } from "grommet";
import { Refresh } from "grommet-icons";
import { useContext, useEffect, useRef, useState } from "react";
import type { PipelineStatus } from "../api";
import { apiGet, apiPost } from "../api";
import { AppContext } from "../context";

/** Re-runs the analysis on the server (202 + status polling every 2 s). The old data stays visible until it succeeds. */
export function PipelineButton() {
  const { bumpData } = useContext(AppContext);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState<string | null>(null);
  const [toast, setToast] = useState<{ status: "normal" | "critical"; title: string; message: string } | null>(null);
  const timer = useRef<number | null>(null);

  const stop = () => { if (timer.current) window.clearInterval(timer.current); timer.current = null; };
  useEffect(() => stop, []);

  const poll = () => {
    stop();
    timer.current = window.setInterval(async () => {
      try {
        const s = await apiGet<PipelineStatus>("/pipeline/status");
        setStep(s.progress);
        if (!s.running) {
          stop();
          setRunning(false);
          if (s.error) {
            setToast({ status: "critical", title: "Analysis failed - previous results kept", message: s.error });
          } else {
            bumpData();
            setToast({ status: "normal", title: "Analysis refreshed", message: `Generated ${s.generated_at?.replace("T", " ") ?? ""}` });
          }
        }
      } catch (e) {
        stop();
        setRunning(false);
        setToast({ status: "critical", title: "Lost contact with the server", message: (e as Error).message });
      }
    }, 2000);
  };

  const start = async () => {
    try {
      await apiPost("/pipeline/run");
      setRunning(true);
      setStep("starting");
      poll();
    } catch (e) {
      setToast({ status: "critical", title: "Could not start the analysis", message: (e as Error).message });
    }
  };

  return (
    <Box direction="row" align="center" gap="small">
      {running ? (
        <Box direction="row" gap="xsmall" align="center">
          <Spinner size="xsmall" />
          <Text size="small" color="text-weak">{step ?? "running"}…</Text>
        </Box>
      ) : (
        <Button icon={<Refresh />} label="Refresh analysis" onClick={start} size="small" secondary />
      )}
      {toast && (
        <Notification toast status={toast.status} title={toast.title} message={toast.message} onClose={() => setToast(null)} />
      )}
    </Box>
  );
}
