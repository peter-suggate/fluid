import { resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { uniformReferenceSolverOptions } from "./uniform-options";
import { uniformDetailSettings } from "./uniform-detail-policy";
import type { WebGPUUniformReferenceOptions } from "./webgpu-uniform-reference";
import type { MethodParamValues } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";

/** Backend-neutral parameter resolution; GPU names are confined to this adapter. */
export function uniformGeometricSolverOptions(overrides: MethodParamValues = {}, scene?: Pick<SceneDescription,"sceneId">): WebGPUUniformReferenceOptions {
  const values=resolveUniformGeometricValues(overrides);
  return {
      retainStageDiagnosticsForQA: overrides.retainStageDiagnosticsForQA === true,
      ...uniformReferenceSolverOptions(values, scene), geometricVolume: true, pressureCycleBudget:"lagged", pressureBudgetHeadroom:0,
      velocityTransport: "semi-lagrangian",
      liquidOnlyVelocityAdvection: false,
      // Interim mapping onto the existing solver (uniformDetailMixedCoarsening):
      // Requested and Full use the CPU region layout, Dynamic the GPU census.
      detail: uniformDetailSettings(values),
      phiCubicAdvection: values.phiCubicAdvection === "on",
      phiDrain: values.phiDrain === "on",
      phiPreserveSurface: values.phiPreserveSurface === "on",
      activeRegion:false,
      gammaDiffusionIterations: 0, densityPostProcessing: false,
      solidExcessCorrection: false,
  };
}
