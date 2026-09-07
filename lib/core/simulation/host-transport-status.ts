import type { PaneSession } from "../session/session";
import { effectiveSimulationStep_s } from "../simulation-step";
import { resourceInteractionGates } from "../resource-readiness";
import { transportLockReason, transportReadiness } from "../transport-status";
import { CLOCK_EPSILON_S } from "./gpu-clock";

/** Preflight the whole experiment before admitting work to either pane. */
export function hostTransportFailure(panes: readonly PaneSession[]): string | undefined {
  for (const pane of panes) {
    const { gpuStatus, gpuInfo } = pane.diagnostics.getState();
    const detail = ["lost", "unavailable", "blocked", "stopping"].includes(gpuStatus.state)
      ? gpuStatus.label
      : gpuInfo?.simulationPipelineError ?? gpuInfo?.topologyGenerationError
        ?? gpuInfo?.sparseWorldStatus?.fault?.code ?? gpuInfo?.sparseWorldDeviceFault?.code;
    if (detail) return `Pane ${pane.id.toUpperCase()}: ${detail}`;
  }
  const steps = panes.map((pane) => ({ id: pane.id, dt: effectiveSimulationStep_s(pane.scene.getState().scene, pane.method.getState()) }));
  if (steps.some(({ dt }) => Math.abs(dt - steps[0].dt) > CLOCK_EPSILON_S)) {
    return `Incompatible comparison steps: ${steps.map(({ id, dt }) => `${id.toUpperCase()} = ${(dt * 1000).toFixed(2)} ms`).join(", ")}. Choose matching step sizes before running. Uniform's paper profile requires 33.33 ms; select its scene time step to use the scene's dt.`;
  }
}

export function hostTransportBlockReason(panes: readonly PaneSession[]): string | undefined {
  const failure = hostTransportFailure(panes);
  if (failure) return failure;
  for (const pane of panes) {
    const { gpuInfo, resourceReadiness } = pane.diagnostics.getState();
    const readiness = transportReadiness(gpuInfo, pane.method.getState().methodId);
    if (!resourceInteractionGates(resourceReadiness, true).transportInteractive
      || !readiness.initialSceneReady || !readiness.simulationReady) {
      return `Pane ${pane.id.toUpperCase()}: ${transportLockReason(readiness, gpuInfo)}`;
    }
  }
}
