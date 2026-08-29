import { create } from "zustand";
import {
  COMPARE_ALL_LINKED,
  INITIAL_COMPARE_STATE,
  type CompareLinkGroup,
  type CompareState,
} from "../compare/compare-query";
import type { PaneId } from "../session/session";

/**
 * Session metadata shared by the library page and the scene page.
 *
 * The route now decides which page is visible. `view` remains the serializable
 * compatibility value for existing URLs, while `studioEntered` says whether
 * there is a real working document worth autosaving or retaining during Fast
 * Refresh. The persistent app shell, rather than this flag, keeps the mounted
 * WebGPU canvas alive between those routes.
 */

export type ShellView = "library" | "studio";

interface ShellStore {
  view: ShellView;
  /**
   * Whether a scene has been opened in this session.
   *
   * Escape returns to the studio, and on a cold load there is no studio to
   * return to yet — a viewport showing a scene nobody chose is not a place.
   */
  studioEntered: boolean;
  librarySearch: string;
  /**
   * A/B compare, which is a property of the *page* and not of either pane.
   *
   * It has to live outside a session by construction: it names both sessions,
   * it says which of them the keyboard belongs to, and its diff is the one
   * thing neither pane can own — pane B's whole configuration is this record
   * read against pane A. See `lib/core/compare/compare-query.ts`.
   */
  compare: CompareState;
  openLibrary: () => void;
  enterStudio: () => void;
  setLibrarySearch: (query: string) => void;
  /** Replace the compare record wholesale; the mirror is the only frequent caller. */
  setCompare: (next: CompareState | ((current: CompareState) => CompareState)) => void;
  /** Open or close the second pane. Opening always starts from an empty diff. */
  setCompareActive: (active: boolean) => void;
  /** One padlock. Re-linking a group drops that group's overrides — see the mirror. */
  setCompareLink: (group: CompareLinkGroup, linked: boolean) => void;
  /** Which pane the keyboard, the ring and the shortcuts belong to. */
  setFocusedPane: (pane: PaneId) => void;
}

export const useShellStore = create<ShellStore>((set) => ({
  // The front door. The studio itself is lazy-mounted by AppShell.
  view: "library",
  studioEntered: false,
  librarySearch: "",
  compare: INITIAL_COMPARE_STATE,
  openLibrary: () => set({ view: "library" }),
  enterStudio: () => set({ view: "studio", studioEntered: true, librarySearch: "" }),
  setLibrarySearch: (librarySearch) => set({ librarySearch }),
  setCompare: (next) => set((state) => ({
    compare: typeof next === "function" ? next(state.compare) : next,
  })),
  // Closing keeps neither the diff nor the padlocks: the mode is left, not
  // suspended, and a diff that survived it would reappear on the next open
  // over a scene it was never written against.
  setCompareActive: (active) => set((state) => ({
    compare: active
      ? { ...state.compare, active: true, focusedPane: "b" }
      : { ...INITIAL_COMPARE_STATE, links: COMPARE_ALL_LINKED },
  })),
  setCompareLink: (group, linked) => set((state) => ({
    compare: { ...state.compare, links: { ...state.compare.links, [group]: linked } },
  })),
  setFocusedPane: (focusedPane) => set((state) => (
    state.compare.focusedPane === focusedPane ? {} : { compare: { ...state.compare, focusedPane } }
  )),
}));
