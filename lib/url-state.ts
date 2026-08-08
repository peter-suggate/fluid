import { refinementRegionsToQuery, withRefinementRegionsFromQuery } from "./editor-refinement-region";
import { defaultMethodId, interactiveMethodId, simulationMethods, type MethodParamValue, type MethodParamValues } from "./methods";
import { cloneScene, validateScene, type CameraState, type SceneDescription } from "./model";
import { isOctreeTechniqueOverlayMode } from "./octree-technique-debug";
import { cameraForPreset, defaultScenePresetId, findSceneDefinition, getScenePreset, scenePresets, type ScenePreset } from "./scenes";
import { sceneDefinitionTakesLattice, sceneDocumentAtLattice } from "./scene-definition";
import { useMethodStore } from "./stores/method-store";
import { useSceneStore } from "./stores/scene-store";
import { useShellStore, type ShellView } from "./stores/shell-store";
import { DEFAULT_RIGHT_PANEL_WIDTH, MAX_RIGHT_PANEL_WIDTH, MIN_RIGHT_PANEL_WIDTH, useUIStore, type RightPanel } from "./stores/ui-store";
import { DEFAULT_SVO_LIGHTING_OPTIONS, type SvoConeTracingMode, type SvoPrimaryTraversalMode } from "./svo-render-options";
import {
  DEFAULT_SVO_RENDER_TUNING,
  normalizeSvoRenderTuning,
  SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM,
  type SvoRenderTuning,
} from "./svo-render-tuning";
import type { GPUQuality } from "./tall-cell-grid";
import type { GridOverlayConfig, GridOverlayMode } from "./webgpu-renderer";

const qualities: ReadonlyArray<GPUQuality> = ["balanced", "high", "ultra"];
const deletedValue = "~delete";

// Scene objects and arrays are deliberately kept atomic. This keeps URLs
// readable for ordinary scalar edits while still round-tripping inflows and
// the rigid-body roster without inventing array-index patch semantics.
const sceneQueryPaths = [
  "sceneId",
  "randomSeed",
  "duration_s",
  "container.width_m",
  "container.height_m",
  "container.depth_m",
  "container.fillFraction",
  "container.top",
  "container.fluidWallMode",
  "voxelDomain",
  "fluid.density_kg_m3",
  "fluid.dynamicViscosity_Pa_s",
  "fluid.surfaceTension_N_m",
  "fluid.gravity_m_s2.x",
  "fluid.gravity_m_s2.y",
  "fluid.gravity_m_s2.z",
  "fluid.initialCondition",
  "fluid.inflow",
  // Painted water and analytic terrain round-trip as atomic blobs. A sculpted
  // terrain grid is far too large for a URL and belongs to the scene library.
  "fluid.initialBrickSeeds_m",
  "fluid.initialBrickSeedsAdditive",
  "terrain",
  "nominalResolution.length_m",
  "numerics.fixedDt_s",
  "numerics.maxDt_s",
  "numerics.pressureRelativeTolerance",
  "numerics.pressureMaxIterations",
  "rigidBodies"
] as const;

type QueryState = {
  methodId: string;
  quality: GPUQuality;
  overrides: Record<string, MethodParamValues>;
  presetId: string;
  scene: SceneDescription;
  view?: ShellView;
  ui: UIQueryState;
};

type UIQueryState = {
  camera: CameraState;
  sceneModalOpen: boolean;
  diagnosticsOpen: boolean;
  rightPanel: RightPanel;
  rightPanelWidth: number;
  gridOverlayAxis: GridOverlayConfig["axis"];
  gridOverlaySlice: number;
  gridOverlayMode: GridOverlayMode;
  svoShadowsEnabled: boolean;
  svoAmbientOcclusionEnabled: boolean;
  silhouetteRefinementEnabled: boolean;
  svoConeTracingMode: SvoConeTracingMode;
  svoPrimaryTraversal: SvoPrimaryTraversalMode;
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

type SerializableMethodState = Pick<QueryState, "methodId" | "quality" | "overrides">;
type SerializableSceneState = Pick<QueryState, "presetId" | "scene">;
type SerializableShellState = Pick<QueryState, "view">;
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
  return simulationMethods.find((method) => method.id === id);
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

function sceneQueryEntries(sceneState: SerializableSceneState): readonly SceneQueryEntry[] {
  const baseline = sceneQueryBaseline(sceneState.presetId);
  const entries: SceneQueryEntry[] = [];
  const regions = refinementRegionsToQuery(sceneState.scene);
  if (regions !== baseline.regions) entries.push([REGIONS_QUERY_KEY, regions]);
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
  if (Array.isArray(base)) return Array.isArray(value);
  if (base && typeof base === "object") return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function numberParam(query: URLSearchParams, key: string, fallback: number, min = -Infinity, max = Infinity) {
  const raw = query.get(key);
  if (raw === null) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) && value >= min && value <= max ? value : fallback;
}

/** Parse external URL input into a complete, validated store snapshot. */
export function parseQueryState(search: string): QueryState {
  const query = new URLSearchParams(search);
  const preset = exactPreset(query.get("scene")) ?? getScenePreset(defaultScenePresetId);
  // A validation preset authors the solver settings its scene is only valid at,
  // and the scene picker applies them through `applyProfile`. Hydration has to
  // start from that same baseline: otherwise a bare `?scene=<profiled preset>`
  // — the first load, any reload, or a shared link — silently runs the preset
  // at the method's generic quality defaults, which is a different solver
  // configuration than the one the scene was authored and validated against.
  const profile = preset.methodProfile;
  const methodId = exactMethod(query.get("method"))?.id ?? profile?.methodId ?? defaultMethodId;
  const qualityCandidate = query.get("quality") as GPUQuality | null;
  const quality = qualityCandidate && qualities.includes(qualityCandidate)
    ? qualityCandidate : profile?.quality ?? "balanced";
  const overrides: Record<string, MethodParamValues> = profile
    ? { [profile.methodId]: { ...profile.overrides } } : {};

  // Explicit `param.*` keys are deliberate tuning and win per key, so an A/B
  // link can still vary one parameter of a profiled scene.
  for (const method of simulationMethods) {
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
    if (raw === deletedValue && path === "fluid.inflow") {
      setAtPath(patched, path, undefined);
      continue;
    }
    try {
      const value = JSON.parse(raw);
      if (compatibleSceneValue(getAtPath(baseScene, path), value)) setAtPath(patched, path, value);
    }
    catch { /* Malformed external values are ignored and canonicalized away. */ }
  }
  // After the loop, never inside it: the regions are percentages of the
  // container, and the container is one of the paths above. Resolving them
  // first would measure them against the preset's tank rather than the one the
  // link actually describes.
  const regionsQuery = query.get(REGIONS_QUERY_KEY);
  const scene = regionsQuery === null
    ? patched
    : withRefinementRegionsFromQuery(patched, regionsQuery);
  const initialUI = useUIStore.getInitialState();
  const presetCamera = cameraForPreset(preset);
  const grid = query.get("grid");
  const gridMode = query.get("gridMode");
  const requestedPanel = query.get("panel");
  const rightPanel: RightPanel = requestedPanel === "visual" || requestedPanel === "bodies" || requestedPanel === "diagnostics" || requestedPanel === "performance"
    ? requestedPanel
    : initialUI.rightPanel;

  return {
    methodId,
    quality,
    overrides,
    presetId: preset.id,
    scene: validateScene(scene).length === 0 ? scene : baseScene,
    view: shellViewFromQuery(search),
    ui: {
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
      sceneModalOpen: query.get("sceneConfig") === "1",
      diagnosticsOpen: rightPanel === "diagnostics",
      rightPanel,
      rightPanelWidth: numberParam(query, "panelWidth", DEFAULT_RIGHT_PANEL_WIDTH, MIN_RIGHT_PANEL_WIDTH, MAX_RIGHT_PANEL_WIDTH),
      gridOverlayAxis: grid === "off" || grid === "x" || grid === "y" || grid === "z" || grid === "volume" ? grid : initialUI.gridOverlayAxis,
      gridOverlaySlice: grid === "volume"
        ? Math.max(0.05, numberParam(query, "gridSlice", initialUI.gridOverlaySlice, 0, 1))
        : numberParam(query, "gridSlice", initialUI.gridOverlaySlice, 0, 1),
      gridOverlayMode: gridMode === "structure" || gridMode === "resolution" || gridMode === "optical" || gridMode === "cfl" || gridMode === "speed" || gridMode === "phi" || gridMode === "divergence" || gridMode === "pressure" || gridMode === "projection" || gridMode === "representation" || (gridMode !== null && isOctreeTechniqueOverlayMode(gridMode)) ? gridMode : initialUI.gridOverlayMode,
      svoShadowsEnabled: query.get("svoShadows") !== "0" ? DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled : false,
      svoAmbientOcclusionEnabled: query.get("svoAO") !== "0" ? DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled : false,
      silhouetteRefinementEnabled: query.get("svoPrimarySeamClosure") === "1",
      svoConeTracingMode: query.get("svoCones") === "exact" || query.get("svoCones") === "off" ? query.get("svoCones") as SvoConeTracingMode : "cones",
      svoPrimaryTraversal: query.get("svoPrimary") === "traced" ? "traced" : "raster",
      // Normalized rather than trusted: these are the tuning fields a link can
      // carry, and the numbers are clamped by the same function the store
      // applies, so an out-of-range external value lands on the ceiling instead
      // of reaching the octree as an unbounded depth request.
      svoRenderTuning: normalizeSvoRenderTuning({
        ...initialUI.svoRenderTuning,
        // The refinement depth is deliberately absent. It is not a tuning value
        // any more — it is `scene.voxelDomain.detailCellSize_m`, which
        // `sceneQueryPaths` already round-trips as `scene.voxelDomain`. A second
        // copy in the query is a second thing that can disagree, which is the
        // failure this collapse removes.
        environmentPlanarRefinementExemption: query.get("svoFlatExempt") === "1",
        lodScreenSpacePixels: numberParam(query, "svoLodPixels",
          initialUI.svoRenderTuning.lodScreenSpacePixels, 0, SVO_LOD_SCREEN_SPACE_PIXELS_MAXIMUM),
      }),
    }
  };
}

function isManagedKey(key: string) {
  return key === "method" || key === "scene" || key === "quality" || key === "view" || key === "diagnostics" || key === "waterdiag" || key === "panel" || key === "panelWidth"
    || key === "performance" || key === "validation" || key === "sceneConfig" || key === "grid" || key === "gridSlice" || key === "gridMode"
    || key === REGIONS_QUERY_KEY || key === "render" || key === "svoLighting" || key === "svoShadows" || key === "svoAO" || key === "svoSilhouetteRefinement" || key === "svoPrimarySeamClosure" || key === "svoCones" || key === "svoPrimary" || key === "svoFlatExempt" || key === "svoLodPixels" || key === "svoSurface" || key === "environment" || key === "fps" || key.startsWith("camera.") || key.startsWith("param.") || key.startsWith("scene.");
}

/** Build a canonical query string from the stores, preserving unrelated keys. */
export function serializeQueryState(
  search: string,
  sceneState: SerializableSceneState,
  methodState: SerializableMethodState,
  uiState: SerializableUIState = useUIStore.getInitialState(),
  shellState: SerializableShellState = { view: "studio" },
  preparedSceneEntries?: readonly SceneQueryEntry[],
): string {
  const query = new URLSearchParams(search);
  for (const key of [...query.keys()]) if (isManagedKey(key)) query.delete(key);

  query.set("method", methodState.methodId);
  query.set("scene", sceneState.presetId);
  query.set("quality", methodState.quality);
  // The studio is the absence of the layer, not a second value: a link to a
  // scene should not also have to say that it is not the shelf it came from.
  if (shellState.view === "library") query.set("view", "library");
  if (uiState.svoShadowsEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled) query.set("svoShadows", uiState.svoShadowsEnabled ? "1" : "0");
  if (uiState.svoAmbientOcclusionEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled) query.set("svoAO", uiState.svoAmbientOcclusionEnabled ? "1" : "0");
  if (uiState.silhouetteRefinementEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.silhouetteRefinementEnabled) {
    query.set("svoPrimarySeamClosure", uiState.silhouetteRefinementEnabled ? "1" : "0");
  }
  if (uiState.svoConeTracingMode !== "cones") query.set("svoCones", uiState.svoConeTracingMode);
  if (uiState.svoPrimaryTraversal !== "raster") query.set("svoPrimary", uiState.svoPrimaryTraversal);
  // A fine-leaf experiment is two values, and a link that carries only one of
  // them reproduces the wrong frame: at the shipped threshold a 0.78 mm leaf is
  // never descended into, so the depth alone would read as having done nothing.
  // The depth itself rides on `scene.voxelDomain`; only the threshold and the
  // exemption are tuning.
  if (uiState.svoRenderTuning.environmentPlanarRefinementExemption
    !== DEFAULT_SVO_RENDER_TUNING.environmentPlanarRefinementExemption) {
    query.set("svoFlatExempt", uiState.svoRenderTuning.environmentPlanarRefinementExemption ? "1" : "0");
  }
  if (uiState.svoRenderTuning.lodScreenSpacePixels !== DEFAULT_SVO_RENDER_TUNING.lodScreenSpacePixels) {
    query.set("svoLodPixels", String(uiState.svoRenderTuning.lodScreenSpacePixels));
  }
  if (uiState.rightPanel) query.set("panel", uiState.rightPanel);
  if (uiState.rightPanelWidth !== DEFAULT_RIGHT_PANEL_WIDTH) query.set("panelWidth", String(uiState.rightPanelWidth));
  if (uiState.sceneModalOpen) query.set("sceneConfig", "1");
  if (uiState.gridOverlayAxis !== "off") query.set("grid", uiState.gridOverlayAxis);
  if (uiState.gridOverlaySlice !== 0.5) query.set("gridSlice", String(uiState.gridOverlaySlice));
  if (uiState.gridOverlayMode !== "structure") query.set("gridMode", uiState.gridOverlayMode);

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

  for (const method of simulationMethods) {
    const values = methodState.overrides[method.id] ?? {};
    for (const spec of method.params) {
      const value = values[spec.key];
      if (value !== undefined) query.set(`param.${method.id}.${spec.key}`, String(value));
    }
  }
  return query.toString();
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
export function replaceQueryStateUrl(preparedSceneEntries?: readonly SceneQueryEntry[]) {
  const search = serializeQueryState(window.location.search, useSceneStore.getState(), useMethodStore.getState(), useUIStore.getState(), useShellStore.getState(), preparedSceneEntries);
  const next = `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  if (next !== current) window.history.replaceState(window.history.state, "", next);
}

/** The canonical location for the document currently held by the stores. */
export function currentScenePageUrl(): string {
  const search = serializeQueryState(window.location.search, useSceneStore.getState(), useMethodStore.getState(), useUIStore.getState(), { view: "studio" });
  return `/scene${search ? `?${search}` : ""}`;
}

export interface QueryStateSyncOptions {
  /** False when a client navigation or Fast Refresh already retained the document stores. */
  readonly hydrateFromUrl?: boolean;
}

/**
 * Hydrate all source-of-truth stores once, then mirror their snapshots to
 * history.replaceState. Popstate follows the same path, so back/forward,
 * reloads and development module replacement rebuild the application from one
 * coherent store snapshot.
 */
export function startQueryStateSync(onHydrated: (presetId: string) => void, options: QueryStateSyncOptions = {}) {
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
    replaceQueryStateUrl(cachedSceneLayer(useSceneStore.getState()));
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
    useShellStore.setState({
      view: restoredShell.view,
      studioEntered: restoredShell.studioEntered,
      ...(restoredShell.view === "studio" ? { librarySearch: "" } : {}),
    });
    // Offline comparison methods remain parseable and serializable, while the
    // interactive application admits only the choices exposed by its picker.
    useMethodStore.setState({ methodId: interactiveMethodId(state.methodId), quality: state.quality, overrides: state.overrides });
    useSceneStore.getState().setScene(state.scene, state.presetId);
    useUIStore.setState(state.ui);
    applyingUrl = false;
    onHydrated(state.presetId);
    writeUrl();
  };

  if (options.hydrateFromUrl === false) writeUrl();
  else hydrate();
  const stopMethod = useMethodStore.subscribe(scheduleWrite);
  const stopScene = useSceneStore.subscribe(scheduleWrite);
  const stopUI = useUIStore.subscribe(scheduleWrite);
  // Search text and section disclosure are intentionally session-only; only
  // the layer in front belongs in the address bar.
  const stopShell = useShellStore.subscribe((shell, previous) => {
    if (shell.view !== previous.view) scheduleWrite();
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
