import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { WorkProgress } from "./work-progress";

/**
 * Readiness facts the transport and the activity tray both interpret.
 *
 * The transport consumes these as a lock (disabled controls, with the reason on
 * the control itself); the tray consumes them as work (a progress card in the
 * corner). They are derived once so the two surfaces can never disagree about
 * whether the simulation is ready.
 */
export interface TransportReadiness {
  readonly ready: boolean;
  readonly loading: boolean;
  readonly initialSceneReady: boolean;
  readonly simulationReady: boolean;
}

export function transportReadiness(
  gpuInfo: GPUEulerianInfo | null,
  _methodId: string,
): TransportReadiness {
  const initialSceneReady = true;
  const simulationReady = gpuInfo?.simulationPipelinesReady !== false;
  const ready = initialSceneReady && simulationReady;
  return { ready, loading: !ready, initialSceneReady, simulationReady };
}

/** Why the transport controls are disabled, stated on the controls themselves. */
export function transportLockReason(
  readiness: TransportReadiness,
  gpuInfo: GPUEulerianInfo | null,
): string {
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
  if (gpuInfo?.topologyGenerationPending) {
    return {
      label: "Preparing detail", state: "active",
      detail: "Preparing the next sparse resolution in the background",
    };
  }
  if (gpuInfo?.topologyGenerationError) {
    return { label: "Resolution update failed", state: "error", detail: gpuInfo.topologyGenerationError };
  }
  if (gpuInfo?.topologyGenerationDeferred) {
    if (gpuInfo.topologyGenerationDeferred.reason === "volume-capacity") {
      return {
        label: "Resolution update cannot preserve liquid", state: "waiting",
        detail: gpuInfo.topologyGenerationDeferred.detail,
      };
    }
    return {
      label: "Resolution budget reached", state: "waiting",
      detail: "The requested resolution exceeds the current topology budget",
    };
  }
  if (gpuInfo?.simulationPipelinesReady === false) {
    return gpuInfo.simulationPipelineError
      ? { label: "Simulation compile failed", state: "error", detail: gpuInfo.simulationPipelineError }
      : { label: "Compiling simulation", state: "active" };
  }
  return undefined;
}
