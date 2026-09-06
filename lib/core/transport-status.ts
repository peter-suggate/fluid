import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { WorkProgress } from "./work-progress";
import { requiresFencedInitialRasterPresentation } from "./gpu-t0-presentation";

/**
 * Readiness facts the transport and the activity tray both interpret.
 *
 * The transport consumes these as a lock (disabled controls, with the reason on
 * the control itself); the tray consumes them as work (a progress card in the
 * corner). They are derived once so the two surfaces can never disagree about
 * whether the sparse world is loading.
 */
export interface TransportReadiness {
  readonly sparseWorld: boolean;
  readonly faultCode?: string;
  readonly ready: boolean;
  readonly loading: boolean;
  readonly initialSceneReady: boolean;
  readonly simulationReady: boolean;
}

export function transportReadiness(
  gpuInfo: GPUEulerianInfo | null,
  methodId: string,
): TransportReadiness {
  const sparseWorldStatus = gpuInfo?.sparseWorldStatus;
  const sparseWorldDeviceStatus = gpuInfo?.sparseWorldDeviceStatus;
  const sparseWorld = sparseWorldStatus !== undefined || sparseWorldDeviceStatus !== undefined;
  const initialSceneReady = !requiresFencedInitialRasterPresentation(methodId)
    || (gpuInfo?.initialSparseAuthorityReady === true
      && gpuInfo?.initialRasterSurfaceReady === true);
  const fault = sparseWorldStatus?.fault ?? gpuInfo?.sparseWorldDeviceFault;
  const faultCode = fault?.code
    ?? (sparseWorldDeviceStatus === "fault" ? "device-library"
      : sparseWorldStatus?.state === "fault" ? "internal" : undefined);
  const ready = sparseWorldStatus !== undefined
    && sparseWorldDeviceStatus === "ready"
    && sparseWorldStatus.state !== "fault";
  const loading = sparseWorld && !faultCode && (!ready || !initialSceneReady);
  // Legacy solvers retain their atomic-pipeline readiness flag. Sparse worlds
  // expose only device-library readiness and semantic world status.
  const simulationReady = sparseWorld ? ready : gpuInfo?.simulationPipelinesReady !== false;
  return { sparseWorld, faultCode, ready, loading, initialSceneReady, simulationReady };
}

/** Why the transport controls are disabled, stated on the controls themselves. */
export function transportLockReason(
  readiness: TransportReadiness,
  gpuInfo: GPUEulerianInfo | null,
): string {
  if (readiness.faultCode) return `Sparse world fault: ${readiness.faultCode}`;
  if (readiness.sparseWorld) {
    return readiness.loading ? "Sparse world is loading"
      : "Simulation controls unlock after the sparse world is ready";
  }
  if (gpuInfo?.simulationPipelineError) {
    return `Simulation pipeline compilation failed: ${gpuInfo.simulationPipelineError}`;
  }
  if (!readiness.simulationReady) return "Simulation pipelines are compiling in the background";
  return "Simulation controls unlock after the initial GPU scene is ready";
}

/**
 * Transport-gating work as tray progress, never as transport chrome.
 *
 * The transport bar states only that its controls are suspended; the story of
 * what is running — loading, compiling, topology preparation, capacity — is a
 * card in the activity tray, where appearing and disappearing costs nothing.
 * Resource-plugin activities that block the transport are not restated here:
 * the tray already renders those from the readiness snapshot, with counts.
 */
export function transportWorkStatus(
  readiness: TransportReadiness,
  gpuInfo: GPUEulerianInfo | null,
): WorkProgress | undefined {
  if (readiness.faultCode) {
    return { label: "Sparse world fault", state: "error", detail: readiness.faultCode };
  }
  if (gpuInfo?.topologyGenerationPending) {
    return {
      label: "Preparing detail", state: "active",
      detail: "Preparing the next sparse resolution in the background",
    };
  }
  if (gpuInfo?.topologyGenerationError) {
    return { label: "Resolution update deferred", state: "waiting", detail: gpuInfo.topologyGenerationError };
  }
  if (gpuInfo?.topologyGenerationDeferred) {
    return {
      label: "Resolution budget reached", state: "waiting",
      detail: "The requested resolution exceeds the current topology budget",
    };
  }
  const status = gpuInfo?.sparseWorldStatus;
  if (status?.state === "saturated") {
    return {
      label: "Sparse world capacity reached", state: "waiting",
      completed: status.residentTiles, total: status.capacityTiles, unit: "tiles",
    };
  }
  if (readiness.loading) return { label: "Loading sparse world", state: "active" };
  if (!readiness.sparseWorld && gpuInfo?.simulationPipelinesReady === false) {
    return gpuInfo.simulationPipelineError
      ? { label: "Simulation compile failed", state: "error", detail: gpuInfo.simulationPipelineError }
      : { label: "Compiling simulation", state: "active" };
  }
  return undefined;
}
