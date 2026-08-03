import { create } from "zustand";

/**
 * Which palette the product wears.
 *
 * The preference is the only state here; the palettes themselves are one
 * declaration in `app/globals.css`, resolved by `light-dark()` against the used
 * `color-scheme`. So this store sets a single attribute on the document element
 * and every surface follows — there is no theme class to thread through
 * components and no second copy of any colour.
 *
 * `system` is the default and is the *absence* of the attribute, so a reader who
 * has never chosen simply gets `color-scheme: light dark` and their own setting,
 * live, without this module running at all.
 */

export type ThemePreference = "system" | "light" | "dark";

export const THEME_STORAGE_KEY = "fluid-lab-theme";

/** Kept in sync with the boot script in `app/layout.tsx`, which runs first. */
export function applyThemePreference(theme: ThemePreference): void {
  if (typeof document === "undefined") return;
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
}

function readStoredTheme(): ThemePreference {
  try {
    const stored = window.localStorage.getItem(THEME_STORAGE_KEY);
    return stored === "light" || stored === "dark" ? stored : "system";
  } catch {
    // Storage can be denied outright; a theme is not worth failing a render for.
    return "system";
  }
}

interface ThemeStore {
  theme: ThemePreference;
  setTheme: (theme: ThemePreference) => void;
  /** Adopt the stored choice after hydration. */
  hydrateTheme: () => void;
}

export const useThemeStore = create<ThemeStore>((set) => ({
  // `system` on the server and on the first client render, so hydration cannot
  // disagree with the markup. The boot script has already painted the stored
  // choice, so the control settling a tick later costs nothing visible.
  theme: "system",
  setTheme: (theme) => {
    applyThemePreference(theme);
    try { window.localStorage.setItem(THEME_STORAGE_KEY, theme); } catch { /* see above */ }
    set({ theme });
  },
  hydrateTheme: () => set({ theme: readStoredTheme() }),
}));
