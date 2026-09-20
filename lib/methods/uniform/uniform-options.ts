import { UNIFORM_PARAMS as params } from "./parameters";
import { numberValue, type MethodParamValues } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";
import type { WebGPUUniformReferenceOptions } from "./webgpu-uniform-reference";

export function uniformDensityPostProcessingEnabled(value: unknown, _sceneId?: string): boolean {
  // Kept in the public helper signature for existing scene-aware callers.
  void _sceneId;
  return value === "on";
}

/**
 * Fixed numerical contract for the dense comparison lane.
 *
 * Keeping transport and conditioning out of the method controls makes this a
 * stable reference rather than a second experimental solver family. Its grid
 * is the scene-authored finest lattice used as the base resolution by both
 * adaptive backends.
 */
export function uniformReferenceSolverOptions(
  values: MethodParamValues,
  scene?: Pick<SceneDescription, "sceneId">,
): WebGPUUniformReferenceOptions {
  const whole = (key: string) => Math.round(numberValue(values, params, key));
  return {
    activeRegion: values.activeRegion === "on",
    densitySharpening: values.densitySharpening !== "off",
    sharpeningMassCorrection: values.sharpeningMassCorrection !== "off",
    gammaDiffusionIterations: values.gammaDiffusion === "off"
      ? 0 : whole("gammaDiffusionIterations"),
    sharpeningStrength: numberValue(values, params, "sharpeningStrength"),
    sharpeningDistance: numberValue(values, params, "sharpeningDistance"),
    solidExcessCorrection: values.solidExcessCorrection !== "off",
    rigidCoupling: values.rigidCoupling !== "off",
    extensionFrontSweeps: whole("extensionFrontSweeps"),
    pressureSchedule: {
      residualTolerance: numberValue(values, params, "pressureResidualTolerance"),
      fullCycles: whole("pressureFullCycles"),
      vCycles: whole("pressureVCycles"),
      preSweeps: whole("pressureSweeps"),
      postSweeps: whole("pressureSweeps"),
    },
    pressureCycleBudget: values.pressureCycleBudget === "fixed" ? "fixed" : "lagged",
    pressureBudgetHeadroom: whole("pressureBudgetHeadroom"),
    densityPostProcessing: uniformDensityPostProcessingEnabled(
      values.densityPostProcessing,
      scene?.sceneId,
    ),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
    velocityTransport: values.velocityTransport === "maccormack"
      ? "maccormack" : "semi-lagrangian",
    liquidOnlyVelocityAdvection: values.liquidOnlyVelocityAdvection === "on",
  };
}
