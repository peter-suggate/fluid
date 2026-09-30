import { resolveUniformGeometricValues, uniformGeometricSharpeningSweeps, UNIFORM_GEOMETRIC_SHARPENING_STRENGTH } from "./uniform-geometric-parameters";
import { uniformReferenceSolverOptions } from "./uniform-options";
import type { WebGPUUniformReferenceOptions } from "./webgpu-uniform-reference";
import type { MethodParamValues } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";

/** Backend-neutral parameter resolution; GPU names are confined to this adapter. */
export function uniformGeometricSolverOptions(overrides: MethodParamValues = {}, scene?: Pick<SceneDescription,"sceneId">): WebGPUUniformReferenceOptions {
  const values=resolveUniformGeometricValues(overrides);
  return {
      retainStageDiagnosticsForQA: overrides.retainStageDiagnosticsForQA === true,
      ...uniformReferenceSolverOptions(values, scene), geometricVolume: true, pressureCycleBudget:"lagged", pressureBudgetHeadroom:0,
      sharpeningStrength: UNIFORM_GEOMETRIC_SHARPENING_STRENGTH,
      sharpeningDistance: Number(values.sharpeningDistance),
      sharpeningSweeps: uniformGeometricSharpeningSweeps(values.sharpeningSweeps),
      surfaceVolumeRounds: Number(values.surfaceVolumeRounds),
      velocityTransport: "semi-lagrangian",
      liquidOnlyVelocityAdvection: false,
      surfaceDeficitBalancing: values.surfaceDeficitBalancing === "on",
      totalSurfaceVolume: values.totalSurfaceVolume === "on",
      volumeDustThreshold: Number(values.volumeDustThreshold),
      orphanDustThreshold: Number(values.orphanDustThreshold),
      mixedCoarsening: values.coarsening === "regions" ? "regions" : "dynamic",
      mixedCoarseningBoundaryTravel: Number(values.coarseningBoundaryTravel),
      phiCubicAdvection: values.phiCubicAdvection === "on",
      phiDrain: values.phiDrain === "on",
      phiPreserveSurface: values.phiPreserveSurface === "on",
      activeRegion:false,
      gammaDiffusionIterations: 0, densityPostProcessing: false,
      solidExcessCorrection: false,
  };
}
