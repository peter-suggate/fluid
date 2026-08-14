"use client";

import { useEffect } from "react";
import { useThemeStore, type ThemePreference } from "../lib/core/stores/theme-store";

/**
 * System, light, dark — in that order, because the default is first.
 *
 * Three states rather than a toggle: a toggle cannot express "whatever the
 * machine is doing", which is the setting most people actually want and the one
 * that follows a laptop from day to evening on its own.
 */
const THEMES: ReadonlyArray<{ id: ThemePreference; label: string; icon: React.ReactNode }> = [
  { id: "system", label: "Match system", icon: <><rect x="2.5" y="3" width="11" height="8" rx="1.5" /><path d="M1.5 13.5h13" /></> },
  { id: "light", label: "Light", icon: <><circle cx="8" cy="8" r="3.1" /><path d="M8 1.2v1.5M8 13.3v1.5M1.2 8h1.5M13.3 8h1.5M3.3 3.3l1.1 1.1M11.6 11.6l1.1 1.1M12.7 3.3l-1.1 1.1M4.4 11.6l-1.1 1.1" /></> },
  { id: "dark", label: "Dark", icon: <path d="M13.2 9.4A5.7 5.7 0 0 1 6.6 2.8a5.7 5.7 0 1 0 6.6 6.6Z" /> },
];

export function ThemeSwitch() {
  const theme = useThemeStore((state) => state.theme);
  const setTheme = useThemeStore((state) => state.setTheme);
  const hydrateTheme = useThemeStore((state) => state.hydrateTheme);

  useEffect(() => hydrateTheme(), [hydrateTheme]);

  return (
    <div className="theme-switch" role="group" aria-label="Colour theme">
      {THEMES.map(({ id, label, icon }) => (
        <button
          key={id}
          type="button"
          title={label}
          aria-label={label}
          aria-pressed={theme === id}
          onClick={() => setTheme(id)}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true">{icon}</svg>
        </button>
      ))}
    </div>
  );
}
