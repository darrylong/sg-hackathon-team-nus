import { Box, Button, Grommet, Header, Nav, ResponsiveContext, Text } from "grommet";
import { Analytics, Book, Clock, Dashboard, Moon, StatusWarning, Sun, Table } from "grommet-icons";
import { hpe } from "grommet-theme-hpe";
import { useCallback, useContext, useMemo, useState } from "react";
import { BrowserRouter, Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import type { Meta } from "./api";
import { useApi } from "./api";
import type { Mode } from "./chartColors";
import { PartnerLayer } from "./components/PartnerPanel";
import { PipelineButton } from "./components/PipelineButton";
import { ErrorState, Loading } from "./components/States";
import { AppContext } from "./context";
import { AtRisk } from "./pages/AtRisk";
import { Forecast } from "./pages/Forecast";
import { Insights } from "./pages/Insights";
import { Methodology } from "./pages/Methodology";
import { Overview } from "./pages/Overview";
import { PartnerPage } from "./pages/PartnerPage";
import { Performance } from "./pages/Performance";

const NAV = [
  { path: "/", label: "Overview", icon: Dashboard },
  { path: "/performance", label: "Performance", icon: Table },
  { path: "/insights", label: "Insights", icon: Analytics },
  { path: "/at-risk", label: "At-risk partners", icon: StatusWarning },
  { path: "/forecast", label: "Q3 forecast", icon: Clock },
  { path: "/methodology", label: "Methodology", icon: Book },
];

function SideNav({ horizontal }: { horizontal?: boolean }) {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  return (
    <Nav direction={horizontal ? "row" : "column"} gap="xxsmall" pad={horizontal ? { horizontal: "small" } : "small"}
      overflow={horizontal ? { horizontal: "auto" } : undefined} a11yTitle="Main navigation">
      {NAV.map(({ path, label, icon: Icon }) => {
        const active = path === "/" ? pathname === "/" : pathname.startsWith(path);
        return (
          <Button key={path} onClick={() => navigate(path)} active={active} hoverIndicator a11yTitle={label}
            plain>
            <Box direction="row" gap="small" align="center" pad={{ vertical: "small", horizontal: "medium" }} round="small"
              background={active ? "background-contrast" : undefined}>
              <Icon size="small" color={active ? "brand" : undefined} />
              <Text size="small" weight={active ? "bold" : undefined} style={{ whiteSpace: "nowrap" }}>{label}</Text>
            </Box>
          </Button>
        );
      })}
    </Nav>
  );
}

function Shell() {
  const size = useContext(ResponsiveContext);
  const { mode, toggleMode } = useContext(AppContext);
  const meta = useApi<Meta>("/meta");
  const small = size === "small";
  const generated = meta.data?.generated_at?.replace("T", " ").slice(0, 16);

  return (
    <Box fill background="background-back">
      <Header pad={{ horizontal: "medium", vertical: "small" }} background="background-front" border={{ side: "bottom", color: "border-weak" }}
        wrap gap="small">
        <Box direction="row" gap="medium" align="center">
          <img src={mode === "dark" ? "/hpe-logo-dark.png" : "/hpe-logo.png"} alt="Hewlett Packard Enterprise"
            style={{ height: 40, width: "auto", display: "block" }} />
          <Box width="1px" alignSelf="stretch" background="border" margin={{ vertical: "xsmall" }} />
          <Box>
            <Text weight="bold">Partner Channel Analytics</Text>
            {meta.data && (
              <Text size="xsmall" color="text-weak">
                {meta.data.data_from} – {meta.data.data_to} · analysis {generated}
              </Text>
            )}
          </Box>
        </Box>
        <Box direction="row" gap="small" align="center">
          <PipelineButton />
          <Button icon={mode === "light" ? <Moon /> : <Sun />} a11yTitle={`Switch to ${mode === "light" ? "dark" : "light"} mode`}
            tip={mode === "light" ? "Dark mode" : "Light mode"} onClick={toggleMode} />
        </Box>
      </Header>
      {small && <Box background="background-front" border={{ side: "bottom", color: "border-weak" }}><SideNav horizontal /></Box>}
      <Box direction="row" flex overflow="hidden">
        {!small && (
          <Box width={{ min: "220px" }} background="background-front" border={{ side: "right", color: "border-weak" }} overflow="auto">
            <SideNav />
          </Box>
        )}
        <Box flex overflow="auto" pad={small ? "medium" : "large"} id="main">
          <Box width={{ max: "1440px" }} fill="horizontal" margin={{ horizontal: "auto" }} flex={false}>
            {meta.error ? <ErrorState error={meta.error} title="The analysis server is not reachable" /> : !meta.data ? (
              <Loading label="Starting the analysis (about 15 seconds on first load)…" />
            ) : (
              <Routes>
                <Route path="/" element={<Overview />} />
                <Route path="/performance" element={<Performance meta={meta.data} />} />
                <Route path="/insights" element={<Insights />} />
                <Route path="/at-risk" element={<AtRisk />} />
                <Route path="/forecast" element={<Forecast />} />
                <Route path="/methodology" element={<Methodology meta={meta.data} />} />
                <Route path="/partners/:id" element={<PartnerPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            )}
          </Box>
        </Box>
      </Box>
    </Box>
  );
}

function readMode(): Mode {
  try {
    const saved = window.localStorage.getItem("theme-mode");
    if (saved === "light" || saved === "dark") return saved;
  } catch { /* storage unavailable */ }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function App() {
  const [mode, setMode] = useState<Mode>(readMode);
  const [dataVersion, setDataVersion] = useState(0);
  const [partner, setPartner] = useState<string | null>(null);

  const toggleMode = useCallback(() => {
    setMode((m) => {
      const next = m === "light" ? "dark" : "light";
      try { window.localStorage.setItem("theme-mode", next); } catch { /* storage unavailable */ }
      return next;
    });
  }, []);
  const ctx = useMemo(() => ({
    mode, toggleMode, dataVersion, bumpData: () => setDataVersion((v) => v + 1), openPartner: setPartner,
  }), [mode, toggleMode, dataVersion]);

  return (
    <Grommet theme={hpe} themeMode={mode} full>
      <AppContext.Provider value={ctx}>
        <BrowserRouter>
          <Shell />
          {partner && <PartnerLayer id={partner} onClose={() => setPartner(null)} />}
        </BrowserRouter>
      </AppContext.Provider>
    </Grommet>
  );
}
