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

/**
 * Explicitly request phase-by-phase sparse-authority fences for driver
 * diagnosis. Timestamp-capable devices also select those fences inside the
 * solver because their instrumented queue needs the same t=0 ordering guard;
 * this query remains useful on timestamp-free devices and in safe bring-up.
 */
export function fencedSparseAuthorityBringupEnabled(search: string): boolean {
  const query = new URLSearchParams(search);
  return query.get("gpu") === "safe" || query.get("safeBringup") === "1";
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
  readonly svoRenderMode: string;
  readonly diagnosticsOpen: boolean;
  readonly rightPanel: string | null;
  readonly gridOverlayAxis: string;
  readonly stageCapturePhase: string;
  readonly search: string;
}

/** Fail closed before requesting an adapter if the bounded browser workload drifted. */
export function safeBrowserGPUBringupViolations(config: SafeBrowserGPUBringupConfig): string[] {
  const query = new URLSearchParams(config.search);
  const values = config.methodValues;
  const canonical = config.canonicalMethodValues;
  const parameterDrift = [...new Set([...Object.keys(values), ...Object.keys(canonical)])]
    .filter((key) => JSON.stringify(values[key]) !== JSON.stringify(canonical[key]));
  const approvedQueryKeys = new Set([
    "gpu", "method", "scene", "quality", "render", "voxels",
    "param.octree.globalFineLevelSetFactor",
    "param.octree.maximumLeafSize",
    "param.octree.interfaceRefinementBandCells",
  ]);
  const unapprovedQueryKeys = [...new Set([...query.keys()].filter((key) => !approvedQueryKeys.has(key)))];
  return [
    config.presetId !== "water-box-dam-break" && "scene must be water-box-dam-break",
    config.methodId !== "octree" && "method must be octree",
    config.quality !== "balanced" && "quality must be balanced",
    !config.exactScene && "scene parameters must match the authored dam-break preset",
    values.globalFineLevelSetFactor !== "4" && "global fine level set must be factor 4",
    values.maximumLeafSize !== "16" && "maximum leaf size must be 16",
    parameterDrift.length > 0 && `method profile drifted: ${parameterDrift.join(", ")}`,
    config.voxelRenderMode !== "smooth" && "voxel inspection must be smooth/off",
    config.svoRenderMode !== "raster" && "renderer must be raster",
    config.diagnosticsOpen && "diagnostics panel must remain closed",
    config.rightPanel !== null && "all right-side panels must remain closed",
    config.gridOverlayAxis !== "off" && "grid overlays must remain off",
    config.stageCapturePhase !== "idle" && "GPU stage capture/readback must be idle",
    unapprovedQueryKeys.length > 0 && `unapproved safe-mode query flags: ${unapprovedQueryKeys.join(", ")}`,
    (query.get("diagnostics") === "1" || query.get("panel") === "diagnostics" || query.get("waterdiag") === "1")
      && "diagnostic GPU readback flags must be absent",
    query.get("gpuRecovery") === "1" && "automatic GPU recovery must be off",
    query.get("gpuTimestamps") === "1" && "GPU timestamps must be off",
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

/**
 * Keep normal browser sessions on the same timestamp-free correctness path as
 * the authored Dawn scenarios. Hardware timestamps are an explicit profiling
 * opt-in because enabling them changes sparse-authority startup scheduling on
 * timestamp-capable devices; a default instrumentation choice must not change
 * the simulation being validated.
 */
export function optionalBrowserTimestampFeatures(
  search: string,
  features: { has(feature: string): boolean },
): GPUFeatureName[] {
  const query = new URLSearchParams(search);
  if (query.get("gpu") === "safe" || query.get("gpuTimestamps") !== "1") return [];
  return features.has("timestamp-query") ? ["timestamp-query"] : [];
}

export const GPU_MANUAL_START_EVENT = "fluid-lab:start-gpu";
export const GPU_MANUAL_STOP_EVENT = "fluid-lab:stop-gpu";

export function requestManualGPUStart(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(GPU_MANUAL_START_EVENT));
}

export function requestManualGPUStop(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(GPU_MANUAL_STOP_EVENT));
}
