import { startHostQueryStateSync } from "../lib/core/query-state-sync";
import type { PaneSession } from "../lib/core/session/session";
import { managedQueryKey } from "../lib/framework/persistence";
import {
  runtimeFeatureQuery, runtimeFeaturesChanged, type RuntimeFeatureState,
} from "../lib/features/runtime-lifecycle";
import {
  REGIONS_QUERY_KEY, regionsFromQuery, regionsToQuery,
} from "../lib/features/refinement-region/persistence";
import {
  advanceRunQuery,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import type { AdvanceStageId } from "../lib/methods/adaptive-volume/features/advance-slice/advance-work";
import type { AdvanceRefinementRegion } from "../lib/physics-wasm/advance-controller";
import { advanceFieldViewQuery, type SliceOverlayId } from "./lenses";
import { labSceneQuery } from "./lab-scenes";
import { labRegionFromRecord, labRegionSpace } from "./lab-region-space";
import type { LabState, LabStore } from "./lab-store";
import { fitView, sliceViewFraction, sliceViewQuery } from "./view-transform";

/**
 * The 2-D lab's address bar, on the studio's loop.
 *
 * Every key here is declared by the module that owns the thing it names — the
 * scene roster, the method's transport arms, the lens table, the region package,
 * the camera — and this file does nothing but say which of them this page
 * mounts and where its values live. That is the whole content of "defined via
 * plugin": the page below has no `SCENE_PARAM`, no `replaceState`, and no
 * opinion about what a link may say.
 *
 * It also fixes a real bug rather than only tidying one. The page used to run
 * its own mirror — two keys, its own defaults, its own `history.replaceState` —
 * beside the studio's loop, and `startHostQueryStateSync`'s path gate is what
 * makes exactly one of them the writer: `/scene` and `/advance-lab` are
 * different pages, and whoever owns the current path writes. A lab link and a
 * studio link can therefore share a key name — `gridMode` is the same *row* on
 * both pages — without either host having to narrow to the other's roster.
 *
 * Two values wait on a lattice. Regions are percentages of the slice and the
 * camera is a fraction of it, and the slice's cell count is not known until the
 * controller has loaded the scene and published a view — two awaits after
 * hydration. So hydration parks them in `linkedRegions`/`linkedView` and the
 * boot applies them against the lattice that arrives. Everything else lands
 * immediately.
 *
 * Not mirrored, and each for its own reason: whether the clock is running and
 * where it is (`playing`, `step`) are a session rather than a reading; the probe
 * under the pointer and the pinned one are the cursor; which sidebar folds are
 * open and which sub-seam is expanded are furniture; the metric the bottom strip
 * prices work in and the wall-clock cost of a step are readings *of this
 * machine* and mean nothing on another.
 */

export const LAB_PATH = "/advance-lab";

/** Everything this page's rows carry, as one query record. */
export interface LabQueryState {
  readonly sceneId: string;
  readonly transport: LabState["transport"];
  readonly budget: number;
  readonly lens: AdvanceStageId;
  readonly overlays: ReadonlySet<SliceOverlayId>;
  readonly surface: LabState["surface"];
  readonly runtime: RuntimeFeatureState;
  /** The raw `regions=` value, or `null` when the link named none. */
  readonly regions: string | null;
  readonly view: ReturnType<typeof sliceViewFraction>;
}

/**
 * Keys this page rewrites from scratch on every canonical write.
 *
 * Composed from the codecs it mounts rather than listed, for the same reason the
 * studio's is: a key that quietly leaves the set survives forever in every
 * address it appears in, and one that quietly joins it deletes somebody else's
 * parameter on the first store change. No prefixes — the lab has no `param.` or
 * `scene.` patch layer, and `view.zoom` and its two siblings are named keys
 * rather than a family, so a future `view.something` belonging to somebody else
 * is safe.
 */
export const isLabQueryKey = managedQueryKey(
  labSceneQuery,
  advanceRunQuery,
  advanceFieldViewQuery,
  sliceViewQuery,
  runtimeFeatureQuery,
  { keys: [REGIONS_QUERY_KEY] },
);

/**
 * The camera actually on screen.
 *
 * A view stamped for a lattice that is no longer up is not the view — render
 * falls back to the fit — so the mirror has to write the fit too, or a link
 * would carry a camera nobody is looking through.
 */
function shownView(state: LabState) {
  return state.view.framing === `${state.nx}x${state.ny}`
    ? state.view : fitView(state.nx, state.ny);
}

export function parseLabQuery(search: string): LabQueryState {
  const query = new URLSearchParams(search);
  const run = advanceRunQuery.read(query);
  const field = advanceFieldViewQuery.read(query);
  return {
    sceneId: labSceneQuery.read(query).sceneId,
    transport: run.transportExperiment,
    budget: run.pressureBudget,
    lens: field.mode as AdvanceStageId,
    overlays: new Set(field.overlays as readonly SliceOverlayId[]),
    surface: run.surfaceView,
    runtime: runtimeFeatureQuery.read(query),
    regions: query.get(REGIONS_QUERY_KEY),
    view: sliceViewQuery.read(query).sliceView,
  };
}

/** The regions a link named, against a lattice that now exists. */
export function labRegionsFromQuery(
  raw: string,
  nx: number,
  ny: number,
): readonly AdvanceRefinementRegion[] {
  return regionsFromQuery(labRegionSpace, { regions: [], nx, ny }, raw)
    .map(labRegionFromRecord);
}

/** A region list as the query value, against the lattice it sits on. */
export function labRegionsToQuery(
  regions: readonly AdvanceRefinementRegion[],
  nx: number,
  ny: number,
): string {
  return regionsToQuery(labRegionSpace, { regions, nx, ny });
}

export function serializeLabQuery(
  search: string,
  state: LabState,
  runtime: RuntimeFeatureState,
): string {
  const query = new URLSearchParams(search);
  for (const key of [...query.keys()]) if (isLabQueryKey(key)) query.delete(key);
  labSceneQuery.write(query, { sceneId: state.sceneId });
  advanceRunQuery.write(query, {
    transportExperiment: state.transport,
    pressureBudget: state.budget,
    surfaceView: state.surface,
  });
  advanceFieldViewQuery.write(query, {
    mode: state.lens, overlays: [...state.overlays],
  });
  // Same rule as the boxes below: while a link's camera is still waiting on a
  // lattice, the link's own answer is what the address says. The fit is written
  // over it only once the boot has framed the slice and cleared the wait.
  sliceViewQuery.write(query, {
    sliceView: state.linkedView
      ?? sliceViewFraction(shownView(state), state.nx, state.ny),
  });
  runtimeFeatureQuery.write(query, runtime);
  // The boxes, on the same present-key-means-removal contract the studio's
  // `regions=` carries: a scene's own authored boxes stay out of the address
  // until they are edited, and an emptied list still writes the key — as the
  // empty string — or a reload would restore them over a deliberate deletion.
  //
  // While a link's boxes are still waiting on a lattice, the link's own value is
  // what the address says. Writing this page's (empty) list over it before the
  // boot could apply it is how a reload would lose them.
  const regions = state.linkedRegions
    ?? labRegionsToQuery(state.regions, state.nx, state.ny);
  const authored = state.baselineRegions;
  const differs = authored === null
    // No scene has loaded, so there is nothing to have edited: the link's own
    // answer is the only one, and a link that named the key keeps it.
    ? state.linkedRegions !== null || regions !== ""
    : regions !== authored;
  if (differs) query.set(REGIONS_QUERY_KEY, regions);
  return query.toString();
}

/**
 * Hydrate the page from its address, then mirror it back.
 *
 * `session.ui` is a source although no key it holds is mirrored today. The
 * studio's mode, armed stroke, selection and region draft all live there and the
 * lab reads all four, so the day one of them belongs in a link the loop already
 * watches it — and a coalesced write that finds nothing changed costs one
 * string comparison.
 */
export function startLabQueryStateSync(
  session: PaneSession,
  lab: LabStore,
  options: { readonly hydrateFromUrl?: boolean } = {},
): () => void {
  return startHostQueryStateSync({
    path: LAB_PATH,
    serialize: (search) =>
      serializeLabQuery(search, lab.getState(), session.runtime.getState()),
    hydrate: (search) => {
      const state = parseLabQuery(search);
      const store = lab.getState();
      store.setSceneId(state.sceneId);
      store.setTransport(state.transport);
      store.setBudget(state.budget);
      store.setLens(state.lens);
      store.setOverlays(state.overlays);
      store.setSurface(state.surface);
      store.setLinked(state.regions, state.view);
      session.runtime.setState(state.runtime);
    },
    sources: [
      (onChange) => lab.subscribe(onChange),
      (onChange) => session.ui.subscribe(onChange),
      (onChange) => session.runtime.subscribe((next, previous) => {
        if (runtimeFeaturesChanged(previous, next)) onChange();
      }),
    ],
    ...(options.hydrateFromUrl === undefined ? {}
      : { hydrateFromUrl: options.hydrateFromUrl }),
  });
}
