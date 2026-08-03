import { create } from "zustand";
import type { SceneSectionId } from "../scene-cards";

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
  /** Sections the reader has opened past their default disclosure. */
  expandedSections: readonly SceneSectionId[];
  openLibrary: () => void;
  enterStudio: () => void;
  setLibrarySearch: (query: string) => void;
  toggleSection: (id: SceneSectionId) => void;
}

export const useShellStore = create<ShellStore>((set) => ({
  // The front door. The studio itself is lazy-mounted by AppShell.
  view: "library",
  studioEntered: false,
  librarySearch: "",
  expandedSections: [],
  openLibrary: () => set({ view: "library" }),
  enterStudio: () => set({ view: "studio", studioEntered: true, librarySearch: "" }),
  setLibrarySearch: (librarySearch) => set({ librarySearch }),
  toggleSection: (id) => set((state) => ({
    expandedSections: state.expandedSections.includes(id)
      ? state.expandedSections.filter((entry) => entry !== id)
      : [...state.expandedSections, id],
  })),
}));
