import { create, type StoreApi, type UseBoundStore } from "zustand";
import type { AdvanceStageId } from "../lib/methods/adaptive-volume/features/advance-slice/advance-work";
import {
  ADVANCE_DEFAULT_SURFACE_VIEW, ADVANCE_DEFAULT_TRANSPORT_EXPERIMENT,
  ADVANCE_TRANSPORT_EXPERIMENTS, type AdvanceSurfaceViewId,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import type { AdvanceRefinementRegion, AdvanceTransportExperiment } from "../lib/physics-wasm/advance-controller";
import { ADVANCE_DEFAULT_LENS_MODE, type SliceOverlayId } from "./lenses";
import { DEFAULT_LAB_SCENE_ID } from "./lab-scenes";
import { fitView, type SliceView, type SliceViewFraction } from "./view-transform";

/**
 * The lab's reading, as state a link can name.
 *
 * Nine `useState` hooks stood here before, inside `AdvanceSlice`, and the
 * problem with them was not that they were hooks: it was that a mirror needs to
 * *subscribe* to what it mirrors. React state is readable only from inside the
 * component that holds it, so the page's own `?scene=`/`?transport=` code had to
 * be a second writer — its own `replaceState`, its own defaults, its own two key
 * names — running beside the studio's loop and able to disagree with it. A store
 * outside React is what lets `startHostQueryStateSync` watch this page the same
 * way it watches the studio's session.
 *
 * What is here is exactly what the address bar carries, and nothing else. The
 * playback flag, the probe under the pointer, the step cost, the open folds, the
 * rubber band and the fault banner are all still `useState` in the page, because
 * none of them is a *reading*: a link that restored which cell somebody's cursor
 * was over would be restoring the cursor.
 *
 * Not `ui-store.ts`, deliberately. The shared UI store is the studio's — mode,
 * armed gesture, selection, region draft — and the lab reads all four of those
 * from it through the same session the studio uses. These nine are the lab's
 * page, and putting them in the shared store would have given every studio pane
 * a transport arm it has no solver for.
 */

/** The camera, plus the lattice shape it was framed for. */
export interface LabSliceView extends SliceView {
  /**
   * `nx`x`ny` at the moment a gesture stamped it.
   *
   * A different lattice is a different picture, not the same one looked at from
   * the old place, so a view stamped with a shape that is no longer on screen is
   * simply not the view: the page falls back to the fit and the next gesture
   * stamps the new shape on. A derivation rather than an effect, so a scene
   * change never shows one frame through the old scene's camera.
   */
  readonly framing: string;
}

export interface LabState {
  /* ---- the run ---- */
  readonly sceneId: string;
  readonly transport: AdvanceTransportExperiment;
  readonly adaptiveSdf: boolean;
  readonly budget: number;
  /* ---- the reading ---- */
  readonly lens: AdvanceStageId;
  readonly overlays: ReadonlySet<SliceOverlayId>;
  readonly surface: AdvanceSurfaceViewId;
  readonly view: LabSliceView;
  /* ---- the boxes ---- */
  readonly regions: readonly AdvanceRefinementRegion[];
  /**
   * The lattice the regions and the camera are measured against.
   *
   * Carried rather than derived because the mirror runs outside React and has to
   * turn cells into percentages without a published view in hand. Set once per
   * load, from the view the controller publishes.
   */
  readonly nx: number;
  readonly ny: number;
  /**
   * The regions a link asked for, before the lattice they measure existed.
   *
   * A link names boxes as percentages of a slice whose cell count is not known
   * until the controller has loaded the scene and published a view — which is
   * two awaits after hydration. So the raw value waits here and the boot applies
   * it against the lattice that arrives. `null` means nobody asked, and the
   * scene's own authored regions stand.
   */
  readonly linkedRegions: string | null;
  /** The camera a link asked for, waiting on the same lattice. */
  readonly linkedView: SliceViewFraction | null;
  /**
   * The scene's own boxes, encoded, or `null` before a scene has loaded.
   *
   * The address carries `regions=` only when the list differs from this, which
   * is what keeps a preset that authors boxes out of every link until somebody
   * edits them — and what makes an emptied list a *written* empty key rather
   * than an absent one, so a reload cannot restore a deliberate deletion.
   */
  readonly baselineRegions: string | null;

  setSceneId(sceneId: string): void;
  setTransport(transport: AdvanceTransportExperiment): void;
  setAdaptiveSdf(enabled: boolean): void;
  setBudget(budget: number): void;
  setLens(lens: AdvanceStageId): void;
  setOverlays(overlays: ReadonlySet<SliceOverlayId>
    | ((current: ReadonlySet<SliceOverlayId>) => ReadonlySet<SliceOverlayId>)): void;
  setSurface(surface: AdvanceSurfaceViewId): void;
  setView(view: LabSliceView | ((current: LabSliceView) => LabSliceView)): void;
  setRegions(regions: readonly AdvanceRefinementRegion[]): void;
  /** A new lattice is on screen: hand the mirror the shape it must measure in. */
  setLattice(nx: number, ny: number): void;
  /** This run's authored boxes, encoded, as the address compares against them. */
  setRegionBaseline(encoded: string | null): void;
  /** The link's boxes and camera, to be applied once a lattice exists. */
  setLinked(regions: string | null, view: SliceViewFraction | null): void;
  clearLinked(): void;
}

export type LabStore = UseBoundStore<StoreApi<LabState>>;

export function createLabStore(): LabStore {
  return create<LabState>((set) => ({
    sceneId: DEFAULT_LAB_SCENE_ID,
    transport: ADVANCE_DEFAULT_TRANSPORT_EXPERIMENT,
    adaptiveSdf: true,
    budget: ADVANCE_TRANSPORT_EXPERIMENTS[ADVANCE_DEFAULT_TRANSPORT_EXPERIMENT]
      .defaultPressureBudget,
    lens: ADVANCE_DEFAULT_LENS_MODE as AdvanceStageId,
    // Off until asked for, like every fold in the sidebar: the water is the
    // subject, and an annotation nobody turned on is chrome over it.
    overlays: new Set<SliceOverlayId>(),
    surface: ADVANCE_DEFAULT_SURFACE_VIEW,
    view: { ...fitView(1, 1), framing: "" },
    regions: [],
    nx: 1,
    ny: 1,
    linkedRegions: null,
    linkedView: null,
    baselineRegions: null,

    setSceneId: (sceneId) => set({ sceneId }),
    setTransport: (transport) => set({ transport }),
    setAdaptiveSdf: (adaptiveSdf) => set({ adaptiveSdf }),
    setBudget: (budget) => set({ budget }),
    setLens: (lens) => set({ lens }),
    setOverlays: (overlays) => set((current) => ({
      overlays: typeof overlays === "function" ? overlays(current.overlays) : overlays,
    })),
    setSurface: (surface) => set({ surface }),
    setView: (view) => set((current) => ({
      view: typeof view === "function" ? view(current.view) : view,
    })),
    setRegions: (regions) => set({ regions }),
    setLattice: (nx, ny) => set({ nx, ny }),
    setRegionBaseline: (baselineRegions) => set({ baselineRegions }),
    setLinked: (linkedRegions, linkedView) => set({ linkedRegions, linkedView }),
    clearLinked: () => set({ linkedRegions: null, linkedView: null }),
  }));
}
