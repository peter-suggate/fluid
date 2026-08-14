import type { SceneDescription } from "./model";
import type { GPUQuality } from "./gpu-quality";
import type { MethodParamValues } from "./method-contract";
import { getMethod } from "./method-registry";
import { resolvedMethodValues } from "./stores/method-store";

export interface SimulationStepMethodState {
  readonly methodId: string;
  readonly quality: GPUQuality;
  readonly overrides: Record<string, MethodParamValues>;
}

/**
 * The clock step that the active solver actually advances.
 *
 * A method's numerics can pin the step rather than cap it — CM12 runs every
 * example at one 1/30 s advance — and keeping the browser's target clock on
 * the scene-authored 4 ms step silently turned that profile into a 250 Hz
 * solve even though the solver advertised the paper step. Which methods pin a
 * step is the method's own business: this asks, rather than testing an id.
 */
export function effectiveSimulationStep_s(
  scene: Pick<SceneDescription, "numerics">,
  method: SimulationStepMethodState,
): number {
  return methodPinnedStep_s(scene, method) ?? scene.numerics.fixedDt_s;
}

/** True while the active method holds the clock to a step of its own. */
export function methodPinsSimulationStep(
  scene: Pick<SceneDescription, "numerics">,
  method: SimulationStepMethodState,
): boolean {
  return methodPinnedStep_s(scene, method) !== undefined;
}

function methodPinnedStep_s(
  scene: Pick<SceneDescription, "numerics">,
  method: SimulationStepMethodState,
): number | undefined {
  return getMethod(method.methodId).effectiveStep_s?.(scene, resolvedMethodValues(method));
}
