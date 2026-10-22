import type { InteractionFluidDriver } from "../../interaction/runtime/physics";
import type { DiagnosticsStoreHook } from "../stores/diagnostics-store";
import { resourceInteractionGates } from "../resource-readiness";

/** The viewport's fluid adapter, shared with readiness integration tests. */
export function createInteractionFluidDriver(
  diagnostics: DiagnosticsStoreHook,
  step: InteractionFluidDriver["step"],
): InteractionFluidDriver {
  const failureReason: NonNullable<InteractionFluidDriver["failureReason"]> = () => {
    const { gpuStatus, gpuInfo } = diagnostics.getState();
    return gpuStatus.state === "lost" || gpuStatus.state === "unavailable" || gpuStatus.state === "blocked"
      ? gpuStatus.label : gpuInfo?.simulationPipelineError ?? gpuInfo?.topologyGenerationError;
  };
  return {
    isReady: () => {
      const { resourceReadiness, gpuInfo, gpuStatus } = diagnostics.getState();
      // Status is the most recent work message, not the state of the solver.
      // Presentation evidence can finish a compile without another status event.
      return !failureReason() && gpuStatus.state !== "stopping"
        && resourceInteractionGates(resourceReadiness, true).transportInteractive
        && gpuInfo?.simulationPipelinesReady !== false;
    },
    failureReason,
    subscribeReadiness: changed => diagnostics.subscribe((state, previous) => {
      if (state.gpuStatus !== previous.gpuStatus
        || state.resourceReadiness !== previous.resourceReadiness
        || state.gpuInfo?.simulationPipelinesReady !== previous.gpuInfo?.simulationPipelinesReady
        || state.gpuInfo?.simulationPipelineError !== previous.gpuInfo?.simulationPipelineError
        || state.gpuInfo?.topologyGenerationError !== previous.gpuInfo?.topologyGenerationError) changed();
    }),
    step,
  };
}
