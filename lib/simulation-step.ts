import type { SceneDescription } from "./model";
import type { GPUQuality } from "./tall-cell-grid";
import type { MethodParamValues } from "./methods";
import { UNIFORM_PAPER_DT_S } from "./uniform-paper";
import { resolvedMethodValues } from "./stores/method-store";

export interface SimulationStepMethodState {
  readonly methodId: string;
  readonly quality: GPUQuality;
  readonly overrides: Record<string, MethodParamValues>;
}

/**
 * The clock step that the active solver actually advances.
 *
 * The uniform paper profile is a numerical contract, not merely a maximum
 * substep: all CM12 examples use one 1/30 s advance. Keeping the browser's
 * target clock on the scene-authored 4 ms step silently turned that profile
 * into a 250 Hz solve even though the solver advertised the paper step.
 */
export function effectiveSimulationStep_s(
  scene: Pick<SceneDescription, "numerics">,
  method: SimulationStepMethodState,
): number {
  const values = resolvedMethodValues(method);
  return method.methodId === "uniform" && values.timeStep !== "scene"
    ? UNIFORM_PAPER_DT_S
    : scene.numerics.fixedDt_s;
}

export function uniformPaperStepActive(method: SimulationStepMethodState): boolean {
  return method.methodId === "uniform" && resolvedMethodValues(method).timeStep !== "scene";
}
