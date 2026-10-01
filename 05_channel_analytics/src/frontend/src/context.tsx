import { createContext, useContext } from "react";
import type { Mode } from "./chartColors";
import { PALETTES } from "./chartColors";

export interface AppState {
  mode: Mode;
  toggleMode: () => void;
  /** Bumped after a successful "Refresh analysis" so every useApi hook refetches. */
  dataVersion: number;
  bumpData: () => void;
  openPartner: (id: string) => void;
}

export const AppContext = createContext<AppState>({
  mode: "light",
  toggleMode: () => undefined,
  dataVersion: 0,
  bumpData: () => undefined,
  openPartner: () => undefined,
});

export function useMode() {
  const { mode } = useContext(AppContext);
  return { mode, palette: PALETTES[mode] };
}
