import { refinementRegionsToQuery, withRefinementRegionsFromQuery } from "./editor-refinement-region";
import { sceneSeedsQuery, withSceneSeedsFromQuery } from "./initial-brick-seed-query";
import { defaultMethodId, interactiveMethodId, registeredSimulationMethods } from "./method-registry";
import type { MethodParamValue, MethodParamValues } from "./method-contract";
import { cloneScene, validateScene, type CameraState, type SceneDescription } from "./model";
import { isOctreeTechniqueOverlayMode } from "./octree-technique-debug";
import { isSparseCM12DirtyOverlayMode } from "./sparse-cm12-dirty-visualizations";
import { isPressureJournalOverlayMode } from "./webgpu-pressure-journal-overlay";
import { cameraForPreset, defaultScenePresetId, findSceneDefinition, getScenePreset, scenePresets, type ScenePreset } from "./scenes";
import { sceneDefinitionTakesLattice, sceneDocumentAtLattice } from "./scene-definition";
import { resolveSession, type PaneSession } from "./session/session";
import {
  compareQueryEntries,
  INITIAL_COMPARE_STATE,
  isCompareQueryKey,
  parseCompareQuery,
  type CompareState,
} from "./compare/compare-query";
import { useShellStore, type ShellView } from "./stores/shell-store";
import { useUIStore, type SceneOverlay } from "./stores/ui-store";
import {
  DEFAULT_SVO_RENDER_DIAGNOSTICS,
  SVO_RENDER_STAGE_VIEWS,
  type SvoRenderStageView,
} from "../svo/svo-render-diagnostics";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoConeTracingMode, type SvoPrimaryTraversalMode } from "../svo/svo-render-options";
import {
  DEFAULT_SVO_RENDER_TUNING,
  normalizeSvoRenderTuning,
  SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM,
  SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM,
  type SvoRenderTuning,
} from "../svo/svo-render-tuning";
import type { GPUQuality } from "./gpu-quality";
import { sceneStoneQuery, withSceneStoneQuery } from "./stone-look-controls";
import { sceneRimQuery, withSceneRimQuery } from "./vessel-rim-controls";
import { sceneCanopyQuery, withSceneCanopyQuery } from "./tree-canopy-controls";
import { isStageLensOverlayMode } from "./stage-lens";
import type { GridOverlayConfig, GridOverlayMode } from "./webgpu-renderer";

const qualities: ReadonlyArray<GPUQuality> = ["balanced", "high", "ultra"];
const deletedValue = "~delete";

// Scene objects and arrays are deliberately kept atomic. This keeps URLs
// readable for ordinary scalar edits while still round-tripping inflows and
// the rigid-body roster without inventing array-index patch semantics.
const sceneQueryPaths = [
  "sceneId",
  "surfaceStyle",
  "randomSeed",
  "duration_s",
  "container.width_m",
  "container.height_m",
  "container.depth_m",
  "container.fillFraction",
  "container.top",
  "container.fluidWallMode",
  "container.shape",
  "container.vessel",
  "voxelDomain",
  "fluid.density_kg_m3",
  "fluid.dynamicViscosity_Pa_s",
  "fluid.surfaceTension_N_m",
  "fluid.gravity_m_s2.x",
  "fluid.gravity_m_s2.y",
  "fluid.gravity_m_s2.z",
  "fluid.initialCondition",
  "fluid.inflow",
  // Analytic terrain round-trips as an atomic blob. A sculpted terrain grid is
  // far too large for a URL and belongs to the scene library; painted water was
  // too, until it moved to the compact `seeds` key below.
  "fluid.initialBrickSeedsAdditive",
  "fluid.initialLiquidVolumes",
  "terrain",
  "nominalResolution.length_m",
  "numerics.fixedDt_s",
  "numerics.maxDt_s",
  "numerics.pressureRelativeTolerance",
  "numerics.pressureMaxIterations",
  "rigidBodies"
] as const;

export type QueryState = {
  methodId: string;
  quality: GPUQuality;
  overrides: Record<string, MethodParamValues>;
  presetId: string;
  scene: SceneDescription;
  view?: ShellView;
  ui: UIQueryState;
};

export type UIQueryState = {
  camera: CameraState;
  /**
   * The instrument drawn over the scene, or `null` for a bare view.
   *
   * In the address for the same reason the camera is: a frame-pipeline reading
   * is something a reader arrives at and then wants to keep — across a reload,
   * a Fast Refresh, or a link sent to somebody else — and losing it on every
   * reload made the overlay a thing to reopen rather than a thing to work in.
   * Like the camera it is listed in the overrides popover but never counted:
   * having an instrument open is not an edit to the scene.
   */
  sceneOverlay: SceneOverlay | null;
  gridOverlayAxis: GridOverlayConfig["axis"];
  gridOverlaySlice: number;
  gridOverlayMode: GridOverlayMode;
  gridOverlayLensPhase: number;
  svoShadowsEnabled: boolean;
  svoAmbientOcclusionEnabled: boolean;
  silhouetteRefinementEnabled: boolean;
  svoConeTracingMode: SvoConeTracingMode;
  svoPrimaryTraversal: SvoPrimaryTraversalMode;
  svoStageView: SvoRenderStageView;
  /**
   * The sparse-presentation tuning, of which exactly two fields round-trip.
   *
   * The whole record is ~40 numbers and would dominate any link it appeared in,
   * so only the two an experiment is actually run over are addressable: the
   * refinement depth, which rebuilds the world at a finer leaf, and the
   * screen-space LOD threshold, without which a finer leaf is never descended
   * into and the depth reads as a no-op. Everything else stays a session value
   * reachable through the PROFILE strip.
   */
  svoRenderTuning: SvoRenderTuning;
};

export type SerializableMethodState = Pick<QueryState, "methodId" | "quality" | "overrides">;
export type SerializableSceneState = Pick<QueryState, "presetId" | "scene">;
/**
 * The page-level half of the address: which shell layer is in front, and pane
 * B's diff. Both are properties of the page rather than of a pane, which is
 * why they arrive from `shell-store` rather than from a session.
 */
type SerializableShellState = Pick<QueryState, "view"> & {
  readonly compare?: CompareState;
};
type SerializableUIState = UIQueryState;

export interface ShellSessionState {
  readonly view: ShellView;
  readonly studioEntered: boolean;
}

/**
 * Which shell layer a link opens.
 *
 * Only the library is spelled out. The studio is what a URL that names a scene
 * already means, and `view` carried a retired presentation mode in older links,
 * so anything unrecognised resolves to the studio rather than being preserved.
 */
export function shellViewFromQuery(search: string): ShellView {
  return new URLSearchParams(search).get("view") === "library" ? "library" : "studio";
}

/**
 * Restore the shell around a URL-hydrated scene.
 *
 * A completely bare first visit is still the library front door. Once a URL
 * names a scene, however, it is a real studio location: a page reload, React
 * Fast Refresh, or an RSC program reload must not put the library layer back in
 * front of it. An explicitly requested library remains the front door; within
 * the current session its Back to scene affordance keeps the existing studio.
 */
export function shellSessionFromQuery(
  search: string,
  current: ShellSessionState,
): ShellSessionState {
  const query = new URLSearchParams(search);
  const requestedView = shellViewFromQuery(search);
  if (requestedView === "library") {
    return { view: "library", studioEntered: current.studioEntered };
  }
  return current.studioEntered || query.has("scene")
    ? { view: "studio", studioEntered: true }
    : current;
}

function exactMethod(id: string | null) {
  return registeredSimulationMethods().find((method) => method.id === id);
}

function exactPreset(id: string | null) {
  return scenePresets.find((preset) => preset.id === id);
}

function getAtPath(value: unknown, path: string): unknown {
  let current = value;
  for (const segment of path.split(".")) {
    if (!current || typeof current !== "object") return undefined;
    current = (current as Record<string, unknown>)[segment];
  }
  return current;
}

function setAtPath(value: object, path: string, next: unknown) {
  const segments = path.split(".");
  let current = value as Record<string, unknown>;
  for (const segment of segments.slice(0, -1)) {
    const child = current[segment];
    if (!child || typeof child !== "object" || Array.isArray(child)) return;
    current = child as Record<string, unknown>;
  }
  const leaf = segments.at(-1)!;
  if (next === undefined) delete current[leaf];
  else current[leaf] = next;
}

/**
 * Address one `scene.*` query path in a document.
 *
 * Exported for the override inspector, which restores a single overridden path
 * to its preset value and has to reach it by exactly the road the query key
 * took — including the atomic-object semantics `sceneQueryPaths` documents.
 */
export const sceneQueryPathValue = getAtPath;

/** The same path, written onto a copy; the document itself stays immutable. */
export function withSceneQueryPathValue(scene: SceneDescription, path: string, value: unknown): SceneDescription {
  const next = cloneScene(scene);
  setAtPath(next, path, value);
  return next;
}

type SceneQueryEntry = readonly [key: string, value: string];

/**
 * The preset side of a scene diff is immutable and deterministic. Keep its
 * serialized path values instead of rebuilding an authored preset every time
 * the address bar changes. This matters for generated scenes: constructing the
 * hero pond bakes a 289 x 193 heightfield before a single value can be compared.
 */
interface SceneQueryBaseline {
  /** One serialized value per entry of `sceneQueryPaths`, in order. */
  readonly paths: readonly (string | undefined)[];
  /** The preset's own refinement regions, already container-relative. */
  readonly regions: string;
  /** The preset's authored canopy dials, in the same form the key carries. */
  readonly canopy: string;
  /** The preset's authored stone looks, in the same form the key carries. */
  readonly stones: string;
  /** The preset's authored coping rims, in the same form the key carries. */
  readonly rim: string;
  /** The preset's own painted water, as brick occupancy. */
  readonly seeds: string;
}

const sceneQueryBaselineCache = new WeakMap<ScenePreset, SceneQueryBaseline>();

function sceneQueryBaseline(presetId: string): SceneQueryBaseline {
  const preset = getScenePreset(presetId);
  const cached = sceneQueryBaselineCache.get(preset);
  if (cached) return cached;
  const baseScene = preset.create();
  const baseline: SceneQueryBaseline = {
    paths: sceneQueryPaths.map((path) => JSON.stringify(getAtPath(baseScene, path))),
    regions: refinementRegionsToQuery(baseScene),
    canopy: sceneCanopyQuery(baseScene),
    stones: sceneStoneQuery(baseScene),
    rim: sceneRimQuery(baseScene),
    seeds: sceneSeedsQuery(baseScene),
  };
  sceneQueryBaselineCache.set(preset, baseline);
  return baseline;
}

/**
 * Refinement regions ride their own key, not a `scene.*` path.
 *
 * Every other scene value in the query is the document's own number, which is
 * right for a metre extent and wrong for a drawn box: a region is a question
 * about a *part of the domain* ("what does it cost to stop resolving the back
 * third"), and that question is worth carrying between scenes. So the key holds
 * percentages of the container and is resolved against whatever container the
 * link lands on. See `refinementRegionsToQuery`.
 *
 * Compared as encoded strings against the preset's own regions, which is the
 * same container-relative comparison the value itself makes, so a preset that
 * authors regions stays out of the URL until they are edited — and an emptied
 * list still writes the key, as the empty string, or hydration would restore
 * the preset's boxes over a deliberate removal.
 */
const REGIONS_QUERY_KEY = "regions";

/**
 * Canopy dials ride their own key for the same reason regions do: the scenery
 * graph is atomic and far too large for a URL, but the three dials a tree is
 * art-directed by are exactly the kind of edit a shared link — and a lattice
 * re-author, which rebuilds the document from the preset factory — must not
 * lose. Compared against the preset's own dials, so an untouched tree stays
 * out of the URL.
 */
const CANOPY_QUERY_KEY = "canopy";

/** Stone-look dials and seeds, on the same contract as `canopy`. */
const STONES_QUERY_KEY = "stones";

/** Coping-rim dials, on the same contract as `canopy`. */
const RIM_QUERY_KEY = "rim";

/**
 * Painted water, as brick occupancy rather than the document's seed array.
 *
 * The array is a list of metre positions whose only readers immediately floor
 * them to the brick they land in, so the URL was spending ~81 characters per
 * seed on a small integer triple — a 256-brick paint reloaded as an HTTP 431.
 * See `initial-brick-seed-query` for the two encodings and why the brick grid
 * travels with them. On the same present-key-means-removal contract as
 * `regions`: an erased paint still writes `seeds=`, or hydration would restore
 * the preset's water over a deliberate deletion.
 */
const SEEDS_QUERY_KEY = "seeds";

function sceneQueryEntries(sceneState: SerializableSceneState): readonly SceneQueryEntry[] {
  const baseline = sceneQueryBaseline(sceneState.presetId);
  const entries: SceneQueryEntry[] = [];
  const regions = refinementRegionsToQuery(sceneState.scene);
  if (regions !== baseline.regions) entries.push([REGIONS_QUERY_KEY, regions]);
  const canopy = sceneCanopyQuery(sceneState.scene);
  if (canopy !== baseline.canopy) entries.push([CANOPY_QUERY_KEY, canopy]);
  const stones = sceneStoneQuery(sceneState.scene);
  if (stones !== baseline.stones) entries.push([STONES_QUERY_KEY, stones]);
  const rim = sceneRimQuery(sceneState.scene);
  if (rim !== baseline.rim) entries.push([RIM_QUERY_KEY, rim]);
  const seeds = sceneSeedsQuery(sceneState.scene);
  if (seeds !== baseline.seeds) entries.push([SEEDS_QUERY_KEY, seeds]);
  sceneQueryPaths.forEach((path, index) => {
    const current = getAtPath(sceneState.scene, path);
    const serialized = JSON.stringify(current);
    if (serialized === baseline.paths[index]) return;
    // A sculpted grid is a scene-library document, not a URL value. The hero
    // pond carries 55,777 samples: serializing one edited height produces a
    // roughly 1.2 MB location that works until reload, when the server rejects
    // the request headers with HTTP 431. Analytic terrain remains small and
    // shareable, and deleting a preset's terrain remains the compact ~delete
    // marker; only a value that actually carries a grid stays out of the URL.
    if (path === "terrain" && current && typeof current === "object" && "grid" in current) return;
    entries.push([`scene.${path}`, current === undefined ? deletedValue : serialized!]);
  });
  return entries;
}

/**
 * Camera motion changes only UI state. Reuse the already-derived scene layer
 * until the immutable scene document (or its preset baseline) changes, so a
 * pointer-rate camera update never scans or serializes a sculpted terrain.
 */
export function createSceneQueryLayerCache() {
  let cached: {
    readonly presetId: string;
    readonly scene: SceneDescription;
    readonly entries: readonly SceneQueryEntry[];
  } | undefined;
  return (sceneState: SerializableSceneState): readonly SceneQueryEntry[] => {
    if (cached?.presetId === sceneState.presetId && cached.scene === sceneState.scene) return cached.entries;
    const entries = sceneQueryEntries(sceneState);
    cached = { ...sceneState, entries };
    return entries;
  };
}

function parseMethodValue(methodId: string, key: string, raw: string): MethodParamValue | undefined {
  const spec = exactMethod(methodId)?.params.find((candidate) => candidate.key === key);
  if (!spec) return undefined;
  if (spec.kind === "select") return spec.options.some((option) => option.value === raw) ? raw : undefined;
  const value = Number(raw);
  return Number.isFinite(value) && value >= spec.min && value <= spec.max ? value : undefined;
}

function compatibleSceneValue(base: unknown, value: unknown) {
  if (typeof base === "number") return typeof value === "number" && Number.isFinite(value);
  if (typeof base === "string") return typeof value === "string";
  if (typeof base === "boolean") return typeof value === "boolean";
  if (Array.isArray(base)) return Array.isArray(value);
  if (base && typeof base === "object") return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  // An absent baseline admits whatever that key's schema is optional about: an
  // object like `fluid.inflow`, or a flag like `fluid.initialBrickSeedsAdditive`,
  // which a preset authoring no painted water simply does not carry. Without the
  // flag arm the object test below refused every boolean whose preset left it
  // unset, so a painted link reloaded as water that *replaced* the authored dam
  // instead of adding to it — the one thing the flag exists to say.
  if (typeof value === "boolean") return true;
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

/** The key carrying the open scene instrument; see `UIQueryState.sceneOverlay`. */
const OVERLAY_QUERY_KEY = "overlay";

/**
 * The instrument names a link may carry.
 *
 * An exhaustive record rather than a list of string comparisons: adding an
 * instrument to `SceneOverlay` then fails to compile here instead of quietly
 * becoming a value this parser drops and the serializer writes, which is the
 * one way a round-trip can disagree with itself.
 */
const SCENE_OVERLAY_KEYS: Readonly<Record<SceneOverlay, true>> = {
  "sim-pipeline": true,
  "render-pipeline": true,
  diagnostics: true,
};

/**
 * Closed is the absence of the key, so an unrecognised value — a retired
 * instrument, a hand-edited link — resolves to closed and is canonicalised out
 * of the address rather than surviving as a name nothing draws.
 */
function parseSceneOverlay(raw: string | null): SceneOverlay | null {
  return raw !== null && Object.hasOwn(SCENE_OVERLAY_KEYS, raw) ? raw as SceneOverlay : null;
}

/** The generic dense-grid views, which no predicate of their own covers. */
const DENSE_GRID_OVERLAY_MODES: Readonly<Record<string, true>> = {
  structure: true, resolution: true, optical: true, cfl: true, speed: true,
  phi: true, divergence: true, pressure: true, projection: true,
  representation: true, density: true, tracers: true, "face-velocity": true,
};

/**
 * The field view a link names, or the current one when it names nothing valid.
 *
 * Every family the serializer can write has to be readable back, or picking a
 * view and reloading silently returns a different one. The four predicates are
 * the passes' own, so a mode added beside its pass is parsed here without this
 * module learning its name — which is the whole point for the lenses, whose
 * modes are one per solver stage and none of them written down here.
 */
function parseGridOverlayMode(raw: string | null, fallback: GridOverlayMode): GridOverlayMode {
  if (raw === null) return fallback;
  return Object.hasOwn(DENSE_GRID_OVERLAY_MODES, raw) || isOctreeTechniqueOverlayMode(raw)
    || isSparseCM12DirtyOverlayMode(raw) || isPressureJournalOverlayMode(raw)
    || isStageLensOverlayMode(raw)
    ? raw as GridOverlayMode : fallback;
}

function numberParam(query: URLSearchParams, key: string, fallback: number, min = -Infinity, max = Infinity) {
  const raw = query.get(key);
  if (raw === null) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
}

/**
 * Rewrite a pre-split `method=octree` link into the method it actually named.
 *
 * The two adaptive backends used to be one registered method forked by a
 * `coarseBackend` parameter, and parameters are namespaced by method id in the
 * URL (`param.<methodId>.<key>`), so a legacy link is a whole namespace keyed
 * on a sibling key's value. Without this, `exactMethod("octree")` returns
 * undefined, `parseQueryState` falls through to `defaultMethodId()`, and every
 * shared link would quietly open on the *uniform* solver with its tuning
 * dropped — a different simulation, reported as success. There is no error to
 * notice, which is the whole reason this runs before the id is resolved.
 *
 * `serializeQueryState` deletes every `param.*` key it does not re-emit, so
 * the first canonicalisation after hydration erases the legacy namespace; this
 * only has to survive that one read.
 */
function migrateLegacyOctreeMethodQuery(query: URLSearchParams): void {
  const LEGACY_ID = "octree";
  const LEGACY_PREFIX = `param.${LEGACY_ID}.`;
  if (query.get("method") !== LEGACY_ID && ![...query.keys()].some((key) => key.startsWith(LEGACY_PREFIX))) return;
  // Only the frozen reference backend names itself; absent, empty or unknown
  // reads as the product default the picker offered, which is Losasso.
  const resolvedId = query.get(`${LEGACY_PREFIX}coarseBackend`) === "power2017" ? "power-liquids" : "losasso";
  if (query.get("method") === LEGACY_ID) query.set("method", resolvedId);
  for (const [key, value] of [...query.entries()]) {
    if (!key.startsWith(LEGACY_PREFIX)) continue;
    query.delete(key);
    const paramKey = key.slice(LEGACY_PREFIX.length);
    // The backend is the method now, so it survives as the id above and not as
    // a parameter; re-emitting it would resurrect a key no schema declares.
    if (paramKey !== "coarseBackend") query.set(`param.${resolvedId}.${paramKey}`, value);
  }
}

/**
 * The solver layer of a query, on its own.
 *
 * Split out for the compare mirror, which has to push pane A's method onto
 * pane B without rebuilding pane B's document: `parseQueryState` runs the
 * preset factory, and the hero pond bakes a 289 x 193 heightfield to produce
 * one. Reading the halves separately is what keeps a solver swap in pane B
 * from costing a scene rebuild in pane A.
 */
function methodQueryState(query: URLSearchParams, preset: ScenePreset): SerializableMethodState {
  // Profiles preserve comparison settings for their named method, but method
  // selection itself has one product-wide default. A bare scene URL must not
  // silently switch back to an adaptive solver merely because the card stores
  // its legacy comparison tuple; an explicit `method=` remains authoritative.
  const profile = preset.methodProfile;
  const methodId = exactMethod(query.get("method"))?.id ?? defaultMethodId();
  const qualityCandidate = query.get("quality") as GPUQuality | null;
  const quality = qualityCandidate && qualities.includes(qualityCandidate)
    ? qualityCandidate : "balanced";
  const overrides: Record<string, MethodParamValues> = profile
    ? { [profile.methodId]: { ...profile.overrides } } : {};

  // Explicit `param.*` keys are deliberate tuning and win per key, so an A/B
  // link can still vary one parameter of a profiled scene.
  for (const method of registeredSimulationMethods()) {
    const values: MethodParamValues = {};
    for (const spec of method.params) {
      const raw = query.get(`param.${method.id}.${spec.key}`);
      if (raw === null) continue;
      const value = parseMethodValue(method.id, spec.key, raw);
      if (value !== undefined) values[spec.key] = value;
    }
    if (Object.keys(values).length > 0) {
      overrides[method.id] = { ...overrides[method.id], ...values };
    }
  }
  return { methodId, quality, overrides };
}

/** The solver layer of an external query string. See `methodQueryState`. */
export function parseMethodQueryState(search: string): SerializableMethodState {
  const query = new URLSearchParams(search);
  migrateLegacyOctreeMethodQuery(query);
  return methodQueryState(query, exactPreset(query.get("scene")) ?? getScenePreset(defaultScenePresetId));
}

/**
 * The view layer of an external query string.
 *
 * The preset is named rather than derived, so the compare mirror can read pane
 * A's camera keys against the preset pane *B* is actually holding — the two
 * differ exactly when the diff has forked the scene, which is the case where
 * falling back to a preset camera has to fall back to the right one.
 */
export function parseUIQueryState(search: string, presetId: string): UIQueryState {
  return uiQueryState(new URLSearchParams(search), getScenePreset(presetId));
}

/** Parse external URL input into a complete, validated store snapshot. */
export function parseQueryState(search: string): QueryState {
  const query = new URLSearchParams(search);
  migrateLegacyOctreeMethodQuery(query);
  const preset = exactPreset(query.get("scene")) ?? getScenePreset(defaultScenePresetId);
  const { methodId, quality, overrides } = methodQueryState(query, preset);

  /**
   * The lattice comes back through the factory, not as a patched number.
   *
   * `voxelDomain` is one of the paths below, so a link from a re-authored
   * document carries its `detailCellSize_m`. Writing that onto a preset-built
   * scene would set the number the renderer derives its refinement depth from
   * while every generator in the document had already expanded at the preset's
   * coarse size — the same disagreement the tuning field used to carry, arriving
   * by another road. Re-authoring first makes the restored document the one that
   * lattice actually describes; the patch loop then runs over it and any other
   * edit in the link still lands.
   */
  const baseScene = ((): SceneDescription => {
    const plain = preset.create();
    const raw = query.get("scene.voxelDomain");
    const definition = findSceneDefinition(preset.id);
    if (raw === null || !definition || !sceneDefinitionTakesLattice(definition)) return plain;
    try {
      const requested = JSON.parse(raw) as SceneDescription["voxelDomain"];
      if (!compatibleSceneValue(plain.voxelDomain, requested)) return plain;
      const cellSize_m = requested.finestCellSize_m;
      const detailCellSize_m = requested.detailCellSize_m ?? cellSize_m;
      if (!(cellSize_m > 0) || !(detailCellSize_m > 0)) return plain;
      const authored = plain.voxelDomain.detailCellSize_m ?? plain.voxelDomain.finestCellSize_m;
      if (cellSize_m === plain.voxelDomain.finestCellSize_m && detailCellSize_m === authored) return plain;
      return sceneDocumentAtLattice(definition, { cellSize_m, detailCellSize_m }).scene;
    } catch {
      // A malformed or refused lattice is not a reason to fail the whole
      // restore; the preset's own document is always a legal answer.
      return plain;
    }
  })();
  const patched = cloneScene(baseScene);
  for (const path of sceneQueryPaths) {
    const raw = query.get(`scene.${path}`);
    if (raw === null) continue;
    if (raw === deletedValue && (path === "fluid.inflow" || path === "surfaceStyle")) {
      setAtPath(patched, path, undefined);
      continue;
    }
    try {
      const value = JSON.parse(raw);
      // Most presets omit the default voxel-flat style, so this optional path
      // has no baseline type for the generic compatibility check to inspect.
      if (path === "surfaceStyle") {
        if (value === "smooth" || value === "voxel-flat") setAtPath(patched, path, value);
        continue;
      }
      if (compatibleSceneValue(getAtPath(baseScene, path), value)) setAtPath(patched, path, value);
    }
    catch { /* Malformed external values are ignored and canonicalized away. */ }
  }
  // Links written before painted water moved to `seeds` carry the seed array on
  // its old `scene.*` path. Read here rather than from `sceneQueryPaths`, which
  // no longer lists it, so those links keep their water; the compact key below
  // wins when a link somehow carries both.
  const legacySeeds = query.get("scene.fluid.initialBrickSeeds_m");
  if (legacySeeds !== null) {
    try {
      const value = JSON.parse(legacySeeds);
      if (compatibleSceneValue(getAtPath(baseScene, "fluid.initialBrickSeeds_m"), value)) {
        setAtPath(patched, "fluid.initialBrickSeeds_m", value);
      }
    }
    catch { /* Malformed external values are ignored and canonicalized away. */ }
  }
  // After the loop, never inside it, and for the same reason the regions below
  // are: brick indices are addressed in a lattice the link's own container and
  // `voxelDomain` define, and the loop is what just applied them.
  const seedsQuery = query.get(SEEDS_QUERY_KEY);
  const withSeeds = seedsQuery === null
    ? patched
    : withSceneSeedsFromQuery(patched, seedsQuery);
  // After the loop, never inside it: the regions are percentages of the
  // container, and the container is one of the paths above. Resolving them
  // first would measure them against the preset's tank rather than the one the
  // link actually describes.
  const regionsQuery = query.get(REGIONS_QUERY_KEY);
  const withRegions = regionsQuery === null
    ? withSeeds
    : withRefinementRegionsFromQuery(withSeeds, regionsQuery);
  // After the lattice re-author above, deliberately: the dials are applied to
  // whatever document the link's lattice actually built, which is what lets an
  // environment-level change and a tree edit travel in the same URL.
  const canopyQuery = query.get(CANOPY_QUERY_KEY);
  const withCanopy = canopyQuery === null
    ? withRegions
    : withSceneCanopyQuery(withRegions, canopyQuery);
  const stonesQuery = query.get(STONES_QUERY_KEY);
  const withStones = stonesQuery === null
    ? withCanopy
    : withSceneStoneQuery(withCanopy, stonesQuery);
  const rimQuery = query.get(RIM_QUERY_KEY);
  const scene = rimQuery === null
    ? withStones
    : withSceneRimQuery(withStones, rimQuery);

  return {
    methodId,
    quality,
    overrides,
    presetId: preset.id,
    scene: validateScene(scene).length === 0 ? scene : baseScene,
    view: shellViewFromQuery(search),
    ui: uiQueryState(query, preset),
  };
}

/**
 * Everything a link says about how the scene is *looked at*, as opposed to what
 * it is. Reads no document, so the compare mirror can push a camera orbit or a
 * slice change across the seam at pointer rate.
 */
function uiQueryState(query: URLSearchParams, preset: ScenePreset): UIQueryState {
  // Session-invariant: every UI store instance is built by the same factory,
  // so this is the authored default rather than any pane's live state.
  const initialUI = useUIStore.getInitialState();
  const presetCamera = cameraForPreset(preset);
  const grid = query.get("grid");
  const gridMode = query.get("gridMode");
  return {
    camera: {
      azimuth_rad: numberParam(query, "camera.azimuth", presetCamera.azimuth_rad),
      elevation_rad: numberParam(query, "camera.elevation", presetCamera.elevation_rad, -1.45, 1.45),
      distance_m: numberParam(query, "camera.distance", presetCamera.distance_m, 0.65, 12),
      // Carried from the preset rather than the query: the aperture is the
      // scene's lens, not a view the user orbited to, so it has no URL key to
      // restore from and must not be dropped while rebuilding the rest.
      tanHalfFov: presetCamera.tanHalfFov,
      target_m: {
        x: numberParam(query, "camera.targetX", presetCamera.target_m.x),
        y: numberParam(query, "camera.targetY", presetCamera.target_m.y),
        z: numberParam(query, "camera.targetZ", presetCamera.target_m.z)
      }
    },
    sceneOverlay: parseSceneOverlay(query.get(OVERLAY_QUERY_KEY)),
    gridOverlayAxis: grid === "off" || grid === "x" || grid === "y" || grid === "z" || grid === "volume" ? grid : initialUI.gridOverlayAxis,
    gridOverlaySlice: grid === "volume"
      ? Math.max(0.05, numberParam(query, "gridSlice", initialUI.gridOverlaySlice, 0, 1))
      : numberParam(query, "gridSlice", initialUI.gridOverlaySlice, 0, 1),
    gridOverlayMode: parseGridOverlayMode(gridMode, initialUI.gridOverlayMode),
    // Only the lens knows how many phases it has, so the ceiling is the
    // overlay's to enforce; a link can only be stopped from naming a
    // fractional or negative one.
    gridOverlayLensPhase: Math.max(0,
      Math.floor(numberParam(query, "lensPhase", initialUI.gridOverlayLensPhase, 0))),
    svoShadowsEnabled: query.get("svoShadows") !== "0" ? DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled : false,
    svoAmbientOcclusionEnabled: query.get("svoAO") !== "0" ? DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled : false,
    silhouetteRefinementEnabled: query.get("svoPrimarySeamClosure") === "1",
    svoConeTracingMode: query.get("svoCones") === "exact" || query.get("svoCones") === "off"
      ? query.get("svoCones") as SvoConeTracingMode
      : DEFAULT_SVO_LIGHTING_OPTIONS.coneTracingMode,
    svoPrimaryTraversal: query.get("svoPrimary") === "traced" || query.get("svoPrimary") === "raster"
      ? query.get("svoPrimary") as SvoPrimaryTraversalMode
      : DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal,
    svoStageView: SVO_RENDER_STAGE_VIEWS.includes(query.get("svoStage") as SvoRenderStageView)
      ? query.get("svoStage") as SvoRenderStageView
      : DEFAULT_SVO_RENDER_DIAGNOSTICS.stageView,
    // Normalized rather than trusted: these are the tuning fields a link can
    // carry, and the numbers are clamped by the same function the store
    // applies, so an out-of-range external value lands on the ceiling instead
    // of reaching the octree as an unbounded depth request.
    svoRenderTuning: normalizeSvoRenderTuning({
      ...DEFAULT_SVO_RENDER_TUNING,
      environmentRefinementDepth: numberParam(query, "svoRefinementDepth",
        DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth, 0,
        SVO_ENVIRONMENT_REFINEMENT_DEPTH_MAXIMUM),
      environmentPlanarRefinementExemption: query.get("svoFlatExempt") === "1",
    lodScreenSpacePixels: numberParam(query, "svoLodPixels",
      initialUI.svoRenderTuning.lodScreenSpacePixels, 0, SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM),
    }),
  };
}

/**
 * Keys this module owns and therefore rewrites from scratch on every canonical
 * write. `panel`, `panelWidth` and `sceneConfig` are retired — the docked right
 * panel and the configuration popover are both gone — but they stay listed so a
 * link from before the hero-scene cut is *tolerated*: the key parses to nothing
 * and is dropped from the address rather than surviving as a stale flag nothing
 * reads. What replaced the dock is `overlay`, which is a live key and is
 * deliberately not a migration target: the retired panel names are not the
 * instruments, so an old link opens the scene bare.
 */
function isManagedKey(key: string) {
  return key === "method" || key === "scene" || key === "quality" || key === "view" || key === "diagnostics" || key === "waterdiag" || key === "panel" || key === "panelWidth" || key === OVERLAY_QUERY_KEY
    || key === "performance" || key === "validation" || key === "sceneConfig" || key === "grid" || key === "gridSlice" || key === "gridMode" || key === "lensPhase"
    || isCompareQueryKey(key)
    || key === REGIONS_QUERY_KEY || key === CANOPY_QUERY_KEY || key === STONES_QUERY_KEY || key === RIM_QUERY_KEY || key === SEEDS_QUERY_KEY || key === "render" || key === "svoLighting" || key === "svoShadows" || key === "svoAO" || key === "svoSilhouetteRefinement" || key === "svoPrimarySeamClosure" || key === "svoCones" || key === "svoPrimary" || key === "svoStage" || key === "svoRefinementDepth" || key === "svoFlatExempt" || key === "svoLodPixels" || key === "svoSurface" || key === "environment" || key === "fps" || key.startsWith("camera.") || key.startsWith("param.") || key.startsWith("scene.");
}

/** Build a canonical query string from the stores, preserving unrelated keys. */
export function serializeQueryState(
  search: string,
  sceneState: SerializableSceneState,
  methodState: SerializableMethodState,
  // As above: the authored default, not pane A's state.
  uiState: SerializableUIState = useUIStore.getInitialState(),
  shellState: SerializableShellState = { view: "studio" },
  preparedSceneEntries?: readonly SceneQueryEntry[],
): string {
  const query = new URLSearchParams(search);
  for (const key of [...query.keys()]) if (isManagedKey(key)) query.delete(key);

  query.set("scene", sceneState.presetId);
  const preset = getScenePreset(sceneState.presetId);
  const profile = preset.methodProfile;
  const baselineMethodId = defaultMethodId();
  const baselineQuality = "balanced";
  // A catalog scene's authored runtime contract is implied by its identity.
  // Writing it beside the scene made every clean card navigation look like a
  // hand-tuned override and allowed the URL to drift if that contract changed.
  if (methodState.methodId !== baselineMethodId) query.set("method", methodState.methodId);
  if (methodState.quality !== baselineQuality) query.set("quality", methodState.quality);
  // The studio is the absence of the layer, not a second value: a link to a
  // scene should not also have to say that it is not the shelf it came from.
  if (shellState.view === "library") query.set("view", "library");
  if (uiState.svoShadowsEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled) query.set("svoShadows", uiState.svoShadowsEnabled ? "1" : "0");
  if (uiState.svoAmbientOcclusionEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled) query.set("svoAO", uiState.svoAmbientOcclusionEnabled ? "1" : "0");
  if (uiState.silhouetteRefinementEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.silhouetteRefinementEnabled) {
    query.set("svoPrimarySeamClosure", uiState.silhouetteRefinementEnabled ? "1" : "0");
  }
  if (uiState.svoConeTracingMode !== "cones") query.set("svoCones", uiState.svoConeTracingMode);
  if (uiState.svoPrimaryTraversal !== DEFAULT_SVO_LIGHTING_OPTIONS.primaryTraversal) {
    query.set("svoPrimary", uiState.svoPrimaryTraversal);
  }
  if (uiState.svoStageView !== DEFAULT_SVO_RENDER_DIAGNOSTICS.stageView) {
    query.set("svoStage", uiState.svoStageView);
  }
  if (uiState.svoRenderTuning.environmentRefinementDepth
    !== DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth) {
    query.set("svoRefinementDepth", String(uiState.svoRenderTuning.environmentRefinementDepth));
  }
  if (uiState.svoRenderTuning.environmentPlanarRefinementExemption
    !== DEFAULT_SVO_RENDER_TUNING.environmentPlanarRefinementExemption) {
    query.set("svoFlatExempt", uiState.svoRenderTuning.environmentPlanarRefinementExemption ? "1" : "0");
  }
  if (uiState.svoRenderTuning.lodScreenSpacePixels !== DEFAULT_SVO_RENDER_TUNING.lodScreenSpacePixels) {
    query.set("svoLodPixels", String(uiState.svoRenderTuning.lodScreenSpacePixels));
  }
  // Only when one is up: a closed instrument is the absence of the key, so an
  // ordinary scene link does not have to say which panels it is not showing.
  if (uiState.sceneOverlay) query.set(OVERLAY_QUERY_KEY, uiState.sceneOverlay);
  if (uiState.gridOverlayAxis !== "off") query.set("grid", uiState.gridOverlayAxis);
  if (uiState.gridOverlaySlice !== 0.5) query.set("gridSlice", String(uiState.gridOverlaySlice));
  if (uiState.gridOverlayMode !== "structure") query.set("gridMode", uiState.gridOverlayMode);
  // Gated on the mode as well as the value: a scrubber position outside a lens
  // addresses nothing, and a key that rode along on every other field view
  // would be noise in every link the picker writes.
  if (uiState.gridOverlayLensPhase !== 0 && isStageLensOverlayMode(uiState.gridOverlayMode)) {
    query.set("lensPhase", String(uiState.gridOverlayLensPhase));
  }

  const presetCamera = cameraForPreset(getScenePreset(sceneState.presetId));
  const cameraValues: ReadonlyArray<[string, number, number]> = [
    ["camera.azimuth", uiState.camera.azimuth_rad, presetCamera.azimuth_rad],
    ["camera.elevation", uiState.camera.elevation_rad, presetCamera.elevation_rad],
    ["camera.distance", uiState.camera.distance_m, presetCamera.distance_m],
    ["camera.targetX", uiState.camera.target_m.x, presetCamera.target_m.x],
    ["camera.targetY", uiState.camera.target_m.y, presetCamera.target_m.y],
    ["camera.targetZ", uiState.camera.target_m.z, presetCamera.target_m.z]
  ];
  for (const [key, value, base] of cameraValues) if (value !== base) query.set(key, String(value));

  for (const [key, value] of preparedSceneEntries ?? sceneQueryEntries(sceneState)) query.set(key, value);

  for (const method of registeredSimulationMethods()) {
    const values = methodState.overrides[method.id] ?? {};
    const baselineValues = method.id === profile?.methodId ? profile.overrides : {};
    for (const spec of method.params) {
      const value = values[spec.key];
      if (value !== undefined && value !== baselineValues[spec.key]) {
        query.set(`param.${method.id}.${spec.key}`, String(value));
      }
    }
  }
  // Pane B last, and as a diff over everything above it: the address reads as
  // one scene plus the handful of keys the second pane disagrees about, which
  // is exactly what compare mode *is*. See `compare/compare-query.ts`.
  for (const [key, value] of compareQueryEntries(shellState.compare)) query.set(key, value);
  return query.toString();
}

/**
 * Write a whole query onto one pane's stores.
 *
 * The same three writes `startQueryStateSync` performs on hydration, reachable
 * from outside it because the compare mirror needs them for pane B, which has
 * no address of its own — B's address *is* pane A's plus the diff.
 */
export function applyQueryStateToSession(session: PaneSession, search: string): void {
  const state = parseQueryState(search);
  session.method.setState({
    methodId: interactiveMethodId(state.methodId),
    quality: state.quality,
    overrides: state.overrides,
  });
  session.scene.getState().setScene(state.scene, state.presetId);
  session.ui.setState(state.ui);
}

/**
 * Mirror every store into the address bar in place.
 *
 * The one writer, so that a subscriber outside this module — the shell view is
 * the first — canonicalises the whole query rather than editing one key of it
 * behind the others' backs. Replace and never push: the URL describes what is
 * on screen, and a back button that stepped through every panel toggle and
 * library visit would never reach the page the reader actually arrived from.
 */
export function replaceQueryStateUrl(
  preparedSceneEntries?: readonly SceneQueryEntry[],
  // WP3: the one writer serializes pane A from this session and pane B as a
  // `b.*` diff beside it. Today there is one pane and it is A.
  session: PaneSession = resolveSession(),
) {
  const search = serializeQueryState(window.location.search, session.scene.getState(), session.method.getState(), session.ui.getState(), useShellStore.getState(), preparedSceneEntries);
  const next = `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next !== current) window.history.replaceState(window.history.state, "", next);
}

/** The canonical location for the document currently held by the stores. */
export function currentScenePageUrl(session: PaneSession = resolveSession()): string {
  const search = serializeQueryState(window.location.search, session.scene.getState(), session.method.getState(), session.ui.getState(), { view: "studio" });
  return `/scene${search ? `?${search}` : ""}`;
}

export interface QueryStateSyncOptions {
  /** False when a client navigation or Fast Refresh already retained the document stores. */
  readonly hydrateFromUrl?: boolean;
  /** The realm the address describes. Defaults to pane A. */
  readonly session?: PaneSession;
}

/**
 * Hydrate all source-of-truth stores once, then mirror their snapshots to
 * history.replaceState. Popstate follows the same path, so back/forward,
 * reloads and development module replacement rebuild the application from one
 * coherent store snapshot.
 */
export function startQueryStateSync(onHydrated: (presetId: string) => void, options: QueryStateSyncOptions = {}) {
  const session = options.session ?? resolveSession();
  let active = true;
  let queued = false;
  let applyingUrl = false;
  const cachedSceneLayer = createSceneQueryLayerCache();
  const scenePageActive = () => window.location.pathname === "/scene" || window.location.pathname.startsWith("/scene/");

  const writeUrl = () => {
    // AppShell retains this component while the library route is visible. Its
    // URL belongs to the library and must neither mirror nor hydrate the hidden
    // studio until navigation returns to /scene.
    if (!active || applyingUrl || !scenePageActive()) return;
    replaceQueryStateUrl(cachedSceneLayer(session.scene.getState()), session);
  };

  const scheduleWrite = () => {
    if (queued || applyingUrl) return;
    queued = true;
    queueMicrotask(() => { queued = false; writeUrl(); });
  };

  const hydrate = () => {
    applyingUrl = true;
    const search = window.location.search;
    const state = parseQueryState(search);
    const shellState = useShellStore.getState();
    const restoredShell = shellSessionFromQuery(search, shellState);
    // The `b.*` block restores both panes: a compare link is one address, and
    // reloading it has to bring back the second pane and the diff it carried,
    // not just pane A with a stray flag.
    const restoredCompare = parseCompareQuery(search);
    useShellStore.setState({
      view: restoredShell.view,
      studioEntered: restoredShell.studioEntered,
      compare: restoredCompare.active
        ? { ...restoredCompare, focusedPane: shellState.compare.focusedPane }
        : INITIAL_COMPARE_STATE,
      ...(restoredShell.view === "studio" ? { librarySearch: "" } : {}),
    });
    // Offline comparison methods remain parseable and serializable, while the
    // interactive application admits only the choices exposed by its picker.
    session.method.setState({ methodId: interactiveMethodId(state.methodId), quality: state.quality, overrides: state.overrides });
    session.scene.getState().setScene(state.scene, state.presetId);
    session.ui.setState(state.ui);
    applyingUrl = false;
    onHydrated(state.presetId);
    writeUrl();
  };

  if (options.hydrateFromUrl === false) writeUrl();
  else hydrate();
  const stopMethod = session.method.subscribe(scheduleWrite);
  const stopScene = session.scene.subscribe(scheduleWrite);
  const stopUI = session.ui.subscribe(scheduleWrite);
  // Search text and section disclosure are intentionally session-only; only
  // the layer in front belongs in the address bar.
  const stopShell = useShellStore.subscribe((shell, previous) => {
    // The compare record is in the address for the same reason the view is: it
    // says which page this is, and a reload has to land on the same one.
    if (shell.view !== previous.view || shell.compare !== previous.compare) scheduleWrite();
  });
  const hydrateScenePage = () => { if (scenePageActive()) hydrate(); };
  window.addEventListener("popstate", hydrateScenePage);

  return () => {
    active = false;
    stopMethod();
    stopScene();
    stopUI();
    stopShell();
    window.removeEventListener("popstate", hydrateScenePage);
  };
}
