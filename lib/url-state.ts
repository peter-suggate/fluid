import { defaultMethodId, simulationMethods, type MethodParamValue, type MethodParamValues } from "./methods";
import { cloneScene, validateScene, type CameraState, type SceneDescription } from "./model";
import { isOctreeTechniqueOverlayMode } from "./octree-technique-debug";
import { cameraForPreset, defaultScenePresetId, getScenePreset, scenePresets } from "./scenes";
import { useMethodStore } from "./stores/method-store";
import { useSceneStore } from "./stores/scene-store";
import { useUIStore, type RightPanel } from "./stores/ui-store";
import {
  DEFAULT_SVO_LIGHTING_MODE,
  DEFAULT_SVO_LIGHTING_OPTIONS,
  DEFAULT_SVO_RENDER_MODE,
  isSvoLightingMode,
  isSvoRenderMode,
  type SvoLightingMode,
  type SvoRenderMode,
} from "./svo-render-mode";
import type { GPUQuality } from "./tall-cell-grid";
import type { GridOverlayConfig, GridOverlayMode } from "./webgpu-renderer";
import type { VoxelRenderMode } from "./webgpu-voxel-debug";

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
  ui: UIQueryState;
};

type UIQueryState = {
  camera: CameraState;
  sceneModalOpen: boolean;
  diagnosticsOpen: boolean;
  rightPanel: RightPanel;
  gridOverlayAxis: GridOverlayConfig["axis"];
  gridOverlaySlice: number;
  gridOverlayMode: GridOverlayMode;
  voxelRenderMode: VoxelRenderMode;
  svoRenderMode: SvoRenderMode;
  svoLightingMode: SvoLightingMode;
  svoShadowsEnabled: boolean;
  svoAmbientOcclusionEnabled: boolean;
};

type SerializableMethodState = Pick<QueryState, "methodId" | "quality" | "overrides">;
type SerializableSceneState = Pick<QueryState, "presetId" | "scene">;
type SerializableUIState = UIQueryState;

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

function sameValue(left: unknown, right: unknown) {
  return JSON.stringify(left) === JSON.stringify(right);
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
  const methodId = exactMethod(query.get("method"))?.id ?? defaultMethodId;
  const qualityCandidate = query.get("quality") as GPUQuality | null;
  const quality = qualityCandidate && qualities.includes(qualityCandidate) ? qualityCandidate : "balanced";
  const overrides: Record<string, MethodParamValues> = {};

  for (const method of simulationMethods) {
    const values: MethodParamValues = {};
    for (const spec of method.params) {
      const raw = query.get(`param.${method.id}.${spec.key}`);
      if (raw === null) continue;
      const value = parseMethodValue(method.id, spec.key, raw);
      if (value !== undefined) values[spec.key] = value;
    }
    if (Object.keys(values).length > 0) overrides[method.id] = values;
  }

  const preset = exactPreset(query.get("scene")) ?? getScenePreset(defaultScenePresetId);
  const baseScene = preset.create();
  const scene = cloneScene(baseScene);
  for (const path of sceneQueryPaths) {
    const raw = query.get(`scene.${path}`);
    if (raw === null) continue;
    if (raw === deletedValue && path === "fluid.inflow") {
      setAtPath(scene, path, undefined);
      continue;
    }
    try {
      const value = JSON.parse(raw);
      if (compatibleSceneValue(getAtPath(baseScene, path), value)) setAtPath(scene, path, value);
    }
    catch { /* Malformed external values are ignored and canonicalized away. */ }
  }
  const initialUI = useUIStore.getInitialState();
  const presetCamera = cameraForPreset(preset);
  const grid = query.get("grid");
  const gridMode = query.get("gridMode");
  const voxels = query.get("voxels");
  const render = query.get("render");
  const svoLighting = query.get("svoLighting");
  const requestedPanel = query.get("panel");
  // One-way migration for shared pre-sidebar links. Serialization always emits
  // the mutually exclusive panel state instead of restoring the old UI flag.
  const rightPanel: RightPanel = requestedPanel === "visual" || requestedPanel === "bodies" || requestedPanel === "diagnostics" || requestedPanel === "performance"
    ? requestedPanel
    : query.get("performance") === "1" ? "performance"
    : query.get("diagnostics") === "1" ? "diagnostics" : initialUI.rightPanel;

  return {
    methodId,
    quality,
    overrides,
    presetId: preset.id,
    scene: validateScene(scene).length === 0 ? scene : baseScene,
    ui: {
      camera: {
        azimuth_rad: numberParam(query, "camera.azimuth", presetCamera.azimuth_rad),
        elevation_rad: numberParam(query, "camera.elevation", presetCamera.elevation_rad, -1.45, 1.45),
        distance_m: numberParam(query, "camera.distance", presetCamera.distance_m, 0.65, 12),
        target_m: {
          x: numberParam(query, "camera.targetX", presetCamera.target_m.x),
          y: numberParam(query, "camera.targetY", presetCamera.target_m.y),
          z: numberParam(query, "camera.targetZ", presetCamera.target_m.z)
        }
      },
      sceneModalOpen: query.get("sceneConfig") === "1",
      diagnosticsOpen: rightPanel === "diagnostics",
      rightPanel,
      gridOverlayAxis: grid === "off" || grid === "x" || grid === "y" || grid === "z" || grid === "volume" ? grid : initialUI.gridOverlayAxis,
      gridOverlaySlice: grid === "volume"
        ? Math.max(0.05, numberParam(query, "gridSlice", initialUI.gridOverlaySlice, 0, 1))
        : numberParam(query, "gridSlice", initialUI.gridOverlaySlice, 0, 1),
      gridOverlayMode: gridMode === "structure" || gridMode === "resolution" || gridMode === "surface" || gridMode === "faces" || gridMode === "optical" || gridMode === "cfl" || gridMode === "speed" || gridMode === "phi" || gridMode === "divergence" || gridMode === "pressure" || gridMode === "projection" || gridMode === "representation" || (gridMode !== null && isOctreeTechniqueOverlayMode(gridMode)) ? gridMode : initialUI.gridOverlayMode,
      voxelRenderMode: voxels === "smooth" || voxels === "raw-voxels" || voxels === "surface-voxels" || voxels === "brick-grid" ? voxels : initialUI.voxelRenderMode,
      svoRenderMode: isSvoRenderMode(render) ? render : DEFAULT_SVO_RENDER_MODE,
      svoLightingMode: isSvoLightingMode(svoLighting) ? svoLighting : DEFAULT_SVO_LIGHTING_MODE,
      svoShadowsEnabled: query.get("svoShadows") !== "0" ? DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled : false,
      svoAmbientOcclusionEnabled: query.get("svoAO") !== "0" ? DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled : false,
    }
  };
}

function isManagedKey(key: string) {
  return key === "method" || key === "scene" || key === "quality" || key === "view" || key === "diagnostics" || key === "panel"
    || key === "performance" || key === "validation" || key === "sceneConfig" || key === "grid" || key === "gridSlice" || key === "gridMode"
    || key === "render" || key === "svoLighting" || key === "svoShadows" || key === "svoAO" || key === "voxels" || key === "environment" || key === "fps" || key.startsWith("camera.") || key.startsWith("param.") || key.startsWith("scene.");
}

/** Build a canonical query string from the stores, preserving unrelated keys. */
export function serializeQueryState(
  search: string,
  sceneState: SerializableSceneState,
  methodState: SerializableMethodState,
  uiState: SerializableUIState = useUIStore.getInitialState()
): string {
  const query = new URLSearchParams(search);
  for (const key of [...query.keys()]) if (isManagedKey(key)) query.delete(key);

  query.set("method", methodState.methodId);
  query.set("scene", sceneState.presetId);
  query.set("quality", methodState.quality);
  if (uiState.svoRenderMode !== DEFAULT_SVO_RENDER_MODE) query.set("render", uiState.svoRenderMode);
  if (uiState.svoLightingMode !== DEFAULT_SVO_LIGHTING_MODE) query.set("svoLighting", uiState.svoLightingMode);
  if (uiState.svoShadowsEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.shadowsEnabled) query.set("svoShadows", uiState.svoShadowsEnabled ? "1" : "0");
  if (uiState.svoAmbientOcclusionEnabled !== DEFAULT_SVO_LIGHTING_OPTIONS.ambientOcclusionEnabled) query.set("svoAO", uiState.svoAmbientOcclusionEnabled ? "1" : "0");
  if (uiState.voxelRenderMode !== "smooth") query.set("voxels", uiState.voxelRenderMode);
  const rightPanel = uiState.rightPanel ?? (uiState.diagnosticsOpen ? "diagnostics" : null);
  if (rightPanel === "diagnostics") query.set("diagnostics", "1");
  else if (rightPanel) query.set("panel", rightPanel);
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

  const baseScene = getScenePreset(sceneState.presetId).create();
  for (const path of sceneQueryPaths) {
    const base = getAtPath(baseScene, path);
    const current = getAtPath(sceneState.scene, path);
    if (sameValue(base, current)) continue;
    query.set(`scene.${path}`, current === undefined ? deletedValue : JSON.stringify(current));
  }

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
 * Hydrate both source-of-truth stores once, then mirror their snapshots to
 * history.replaceState. Popstate follows the same path, so back/forward and
 * reloads rebuild the simulation from one coherent store snapshot.
 */
export function startQueryStateSync(onHydrated: (presetId: string) => void) {
  let active = true;
  let queued = false;
  let applyingUrl = false;

  const writeUrl = () => {
    if (!active || applyingUrl) return;
    const search = serializeQueryState(window.location.search, useSceneStore.getState(), useMethodStore.getState(), useUIStore.getState());
    const next = `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`;
    const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (next !== current) window.history.replaceState(window.history.state, "", next);
  };

  const scheduleWrite = () => {
    if (queued || applyingUrl) return;
    queued = true;
    queueMicrotask(() => { queued = false; writeUrl(); });
  };

  const hydrate = () => {
    applyingUrl = true;
    const state = parseQueryState(window.location.search);
    // The interactive application has one scene authority. Legacy method IDs
    // remain parseable for offline comparison tooling, but UI hydration always
    // resolves to the unified sparse-brick octree.
    useMethodStore.setState({ methodId: "octree", quality: state.quality, overrides: state.overrides });
    useSceneStore.getState().setScene(state.scene, state.presetId);
    useUIStore.setState(state.ui);
    applyingUrl = false;
    onHydrated(state.presetId);
    writeUrl();
  };

  hydrate();
  const stopMethod = useMethodStore.subscribe(scheduleWrite);
  const stopScene = useSceneStore.subscribe(scheduleWrite);
  const stopUI = useUIStore.subscribe(scheduleWrite);
  window.addEventListener("popstate", hydrate);

  return () => {
    active = false;
    stopMethod();
    stopScene();
    stopUI();
    window.removeEventListener("popstate", hydrate);
  };
}
