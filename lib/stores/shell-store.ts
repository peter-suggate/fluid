import { create } from "zustand";
import type { SceneSectionId } from "../scene-cards";

/**
 * Which half of the product is in front.
 *
 * Deliberately a view flag rather than a route. The WebGPU device, the compiled
 * pipelines and the retained draw loop survive Fast Refresh on purpose (see the
 * controller singleton); unmounting the viewport to show a menu would make
 * choosing a scene the most expensive navigation in the product, and would throw
 * away the one property this design is built on — that the library is what you
 * read while the platform lane is still starting.
 *
 * So the library is a layer over a live viewport, and entering the studio is a
 * state change with no teardown.
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
  // The front door. A cold load spends its first second or two in the platform
  // lane regardless; spending it on something worth reading is free.
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
