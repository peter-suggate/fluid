export type GPUStartupMode = "off" | "manual" | "automatic" | "safe";

export interface GPUStartupContext {
  readonly presetId: string;
  readonly methodId: string;
}

/**
 * Resolve browser GPU startup without touching navigator.gpu.
 *
 * `gpu=off` is the safe UI-only mode, `gpu=manual` requires an explicit user
 * action, and `gpu=on` restores eager startup. The high-risk default octree
 * dam-break scene starts manually when no explicit policy is supplied.
 */
export function resolveGPUStartupMode(search: string, context: GPUStartupContext): GPUStartupMode {
  const requested = new URLSearchParams(search).get("gpu");
  if (requested === "off") return "off";
  if (requested === "safe") return "safe";
  if (requested === "manual") return "manual";
  if (requested === "on") return "automatic";
  return context.presetId === "water-box-dam-break" && context.methodId === "octree"
    ? "manual"
    : "automatic";
}

/** One-shot browser bring-up: explicit start, pinned workload, one STEP, then STOP GPU. */
export function safeBrowserGPUBringupEnabled(search: string): boolean {
  return new URLSearchParams(search).get("gpu") === "safe";
}

/** Any reset/rebuild epoch after consent invalidates the one-shot session. */
export function safeBrowserSimulationEpochChanged(
  safeMode: boolean,
  initializationStarted: boolean,
  initialEpoch: number | undefined,
  currentEpoch: number,
): boolean {
  return safeMode && initializationStarted && initialEpoch !== undefined && currentEpoch !== initialEpoch;
}

export interface SafeBrowserGPUBringupConfig {
  readonly presetId: string;
  readonly methodId: string;
  readonly quality: string;
  readonly methodValues: Readonly<Record<string, unknown>>;
  readonly canonicalMethodValues: Readonly<Record<string, unknown>>;
  readonly exactScene: boolean;
  readonly voxelRenderMode: string;
  readonly diagnosticsOpen: boolean;
  readonly rightPanel: string | null;
  readonly gridOverlayAxis: string;
  /** Armed WYSIWYG editor tool; authoring gestures mutate the pinned workload. */
  readonly activeTool: string;
  readonly search: string;
}

/** Fail closed before requesting an adapter if the bounded browser workload drifted. */
export function safeBrowserGPUBringupViolations(config: SafeBrowserGPUBringupConfig): string[] {
  const query = new URLSearchParams(config.search);
  const values = config.methodValues;
  const canonical = config.canonicalMethodValues;
  const safeVariantKeys = new Set(["globalFineLevelSetFactor", "surfaceRefinementGradingLayers"]);
  const parameterDrift = [...new Set([...Object.keys(values), ...Object.keys(canonical)])]
    .filter((key) => !safeVariantKeys.has(key))
    .filter((key) => JSON.stringify(values[key]) !== JSON.stringify(canonical[key]));
  const approvedQueryKeys = new Set([
    "gpu", "method", "scene", "quality", "voxels",
    "param.octree.coarseBackend",
    "param.octree.globalFineLevelSetFactor",
    "param.octree.maximumLeafSize",
    "param.octree.interfaceRefinementBandCells",
    "param.octree.surfaceRefinementGradingLayers",
    "param.octree.topologyCadenceAdvances",
  ]);
  const unapprovedQueryKeys = [...new Set([...query.keys()].filter((key) => !approvedQueryKeys.has(key)))];
  return [
    config.presetId !== "water-box-dam-break" && "scene must be water-box-dam-break",
    config.methodId !== "octree" && "method must be octree",
    config.quality !== "balanced" && "quality must be balanced",
    !config.exactScene && "scene parameters must match the authored dam-break preset",
    !["1", "4", "8"].includes(String(values.globalFineLevelSetFactor))
      && "global fine level set must be factor 1, 4, or 8",
    values.maximumLeafSize !== "16" && "maximum leaf size must be 16",
    parameterDrift.length > 0 && `method profile drifted: ${parameterDrift.join(", ")}`,
    config.voxelRenderMode !== "smooth" && "voxel inspection must be smooth/off",
    config.diagnosticsOpen && "diagnostics panel must remain closed",
    config.rightPanel !== null && "all right-side panels must remain closed",
    config.gridOverlayAxis !== "off" && "grid overlays must remain off",
    config.activeTool !== "select" && "editor tools must remain on select",
    unapprovedQueryKeys.length > 0 && `unapproved safe-mode query flags: ${unapprovedQueryKeys.join(", ")}`,
    query.get("gpuRecovery") === "1" && "automatic GPU recovery must be off",
  ].filter((value): value is string => typeof value === "string");
}

export const BROWSER_GPU_LOCK_NAME = "fluid-lab:webgpu-exclusive";

interface BrowserLockManager {
  request(
    name: string,
    options: { mode: "exclusive"; ifAvailable: true },
    callback: (lock: { readonly name: string } | null) => Promise<void>,
  ): Promise<void>;
}

export type BrowserGPULeaseResult =
  | { readonly status: "acquired"; readonly release: () => void }
  | { readonly status: "held" | "unsupported" | "error"; readonly message: string };

/**
 * Hold one Web Lock for the lifetime of the browser GPU device. `ifAvailable`
 * makes a second tab fail immediately instead of queuing a surprise startup.
 */
export async function acquireBrowserGPULease(
  manager: BrowserLockManager | undefined,
): Promise<BrowserGPULeaseResult> {
  if (!manager) return { status: "unsupported", message: "This browser cannot enforce the cross-tab WebGPU lock" };
  let resolveAcquisition!: (status: "acquired" | "held" | "error") => void;
  let releaseLock!: () => void;
  let settled = false;
  const acquisition = new Promise<"acquired" | "held" | "error">((resolve) => { resolveAcquisition = resolve; });
  const release = new Promise<void>((resolve) => { releaseLock = resolve; });
  try {
    void manager.request(BROWSER_GPU_LOCK_NAME, { mode: "exclusive", ifAvailable: true }, async (lock) => {
      settled = true;
      resolveAcquisition(lock ? "acquired" : "held");
      if (lock) await release;
    }).catch(() => {
      if (!settled) resolveAcquisition("error");
    });
  } catch {
    resolveAcquisition("error");
  }
  const acquisitionStatus = await acquisition;
  if (acquisitionStatus === "held") {
    return { status: "held", message: "Another Fluid Lab tab owns the WebGPU safety lock" };
  }
  if (acquisitionStatus === "error") {
    return { status: "error", message: "The browser WebGPU safety lock failed" };
  }
  let released = false;
  return { status: "acquired", release: () => {
    if (released) return;
    released = true;
    releaseLock();
  } };
}

/** Keep exclusivity until renderer work has settled and its device is gone. */
export async function shutdownBrowserGPUSession(
  renderer: { shutdown(): Promise<void> },
  pendingLease?: Promise<BrowserGPULeaseResult>,
  releaseCurrentLease?: () => void,
): Promise<void> {
  await renderer.shutdown();
  const acquiredDuringShutdown = pendingLease ? await pendingLease : undefined;
  if (acquiredDuringShutdown?.status === "acquired") acquiredDuringShutdown.release();
  releaseCurrentLease?.();
}

/** Device recreation is diagnostic-only because a deterministic fault may recur. */
export function automaticGPURecoveryEnabled(search: string): boolean {
  return new URLSearchParams(search).get("gpuRecovery") === "1";
}

/** Request the single device feature required by exhaustive GPU traces. */
export function performanceTraceDeviceFeatures(
  features: { has(feature: string): boolean },
): GPUFeatureName[] {
  return features.has("timestamp-query") ? ["timestamp-query"] : [];
}

/**
 * Request target execution features only when the adapter advertises them.
 * The production octree solver separately rejects devices without subgroups;
 * this function never asks WebGPU for an unsupported feature.
 */
export function fluidExecutionDeviceFeatures(
  features: { has(feature: string): boolean },
): GPUFeatureName[] {
  const requested = performanceTraceDeviceFeatures(features);
  for (const feature of ["shader-f16", "subgroups"] as const) {
    if (features.has(feature)) requested.push(feature as GPUFeatureName);
  }
  return requested;
}

export const GPU_MANUAL_START_EVENT = "fluid-lab:start-gpu";
export const GPU_MANUAL_STOP_EVENT = "fluid-lab:stop-gpu";

export function requestManualGPUStart(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(GPU_MANUAL_START_EVENT));
}

export function requestManualGPUStop(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(GPU_MANUAL_STOP_EVENT));
}
