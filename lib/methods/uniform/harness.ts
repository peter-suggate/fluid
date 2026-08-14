import type {
  MethodHarnessLane,
  MethodHarnessPlugin,
} from "../../core/method-harness-contract";
import type { MethodParamValues } from "../../core/method-contract";

/**
 * The dense reference lane's node-only harness half.
 *
 * Almost all of this method's presence in the smoke executor was negative
 * space: thirty-four `method.id !== "uniform"` tests standing in for "the
 * authority is a compact adaptive publication". Answering that question here
 * is what let the executor stop naming it, and the four parameter overrides
 * below are the only positive knowledge the harness ever had about it.
 */

const UNIFORM_ENVIRONMENT_VARIABLES: readonly string[] = Object.freeze([
  "FLUID_VELOCITY_TRANSPORT",
  "FLUID_SHARPENING",
  "FLUID_UNIFORM_DENSITY_POSTPROCESSING",
  "FLUID_UNIFORM_TIME_STEP",
]);

/**
 * The dense lane: collocated 3D textures for volume and velocity, per-stage
 * audit textures for the comparison metrics, no adaptive topology and so no
 * tripwire sources and no separate fine band at any factor.
 */
const uniformHarnessLane: MethodHarnessLane = Object.freeze({
  compactAdaptivePublication: false,
  silentFailureTripwires: false,
  stagedTextureComparison: true,
  structuredGenerationAudit: false,
  nativeTerminalReceipt: false,
  separateFineLevelSetBand: () => false,
});

function applyUniformEnvironmentOverrides(
  values: MethodParamValues,
  env: Readonly<Record<string, string | undefined>>,
): void {
  if (env.FLUID_VELOCITY_TRANSPORT !== undefined) {
    values.velocityTransport = env.FLUID_VELOCITY_TRANSPORT;
  }
  if (env.FLUID_SHARPENING !== undefined) {
    values.densitySharpening = env.FLUID_SHARPENING !== "0" ? "on" : "off";
  }
  if (env.FLUID_UNIFORM_DENSITY_POSTPROCESSING !== undefined) {
    values.densityPostProcessing = env.FLUID_UNIFORM_DENSITY_POSTPROCESSING !== "0" ? "on" : "off";
  }
  const timeStep = env.FLUID_UNIFORM_TIME_STEP;
  if (timeStep !== undefined) {
    if (!["paper", "scene"].includes(timeStep)) {
      throw new Error("FLUID_UNIFORM_TIME_STEP must be paper or scene");
    }
    values.timeStep = timeStep;
  }
}

export const uniformHarnessPlugin: MethodHarnessPlugin = {
  methodId: "uniform",
  lane: uniformHarnessLane,
  environmentVariables: UNIFORM_ENVIRONMENT_VARIABLES,
  applyEnvironmentOverrides: applyUniformEnvironmentOverrides,
};

export default uniformHarnessPlugin;
