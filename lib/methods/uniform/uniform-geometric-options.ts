import { resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { uniformReferenceSolverOptions } from "./uniform-options";
import type { WebGPUUniformReferenceOptions } from "./webgpu-uniform-reference";
import type { MethodParamValues } from "../../core/method-contract";
import type { SceneDescription } from "../../core/model";

/** Backend-neutral parameter resolution; GPU names are confined to this adapter. */
export function uniformGeometricSolverOptions(overrides: MethodParamValues = {}, scene?: Pick<SceneDescription,"sceneId">): WebGPUUniformReferenceOptions {
  const values=resolveUniformGeometricValues(overrides);
  // Plans are prebuilt; only the encoded prefix follows asynchronous residual
  // evidence. Start at one cycle and reserve no speculative tail.
  return {
      ...uniformReferenceSolverOptions(values, scene), geometricVolume: true, pageDomain:true, pressureCycleBudget:"lagged", pressureBudgetHeadroom:0,
      volumePages: values.pageSize === "16" ? 16 : 32,
      sharpeningStrength: Number(values.sharpeningStrength),
      velocityTransport: values.velocityTransport === "maccormack" ? "maccormack" : "semi-lagrangian",
      surfaceDeficitBalancing: values.surfaceDeficitBalancing === "on",
      totalSurfaceVolume: values.totalSurfaceVolume === "on",
      geometricRedistance: values.redistance !== "off",
      geometricTileWork: values.sharpeningWorkMap !== "off",
      volumeDustThreshold: Number(values.volumeDustThreshold),
      twoLevelVelocity: values.twoLevelVelocity === "on",
      twoLevelFineReach: Number(values.twoLevelFineReach),
      twoLevelExtensionTiles: values.twoLevelExtension !== "dense",
      twoLevelShellReach: Number(values.twoLevelShellReach),
      twoLevelAdvectionTiles: values.twoLevelAdvection !== "dense",
      transportTiles: values.transportWorkMap !== "dense",
      transportReach: Number(values.transportReach),
      volumePressureRows: values.volumePressureRows === "all" ? "all" : values.volumePressureRows === "abandoned" ? "abandoned" : "off",
      volumeCompaction: values.volumeCompaction === "on",
      phiSeedFromVolume: values.phiSeedFromVolume === "on",
      phiAgreementGain: values.phiAgreement === "on" ? Number(values.phiAgreementGain) : 0,
      phiAgreementClamp: Number(values.phiAgreementClamp),
      activeRegion:false,
      pressureWindow:false,
      gammaDiffusionIterations: 0, densityPostProcessing: false,
      solidExcessCorrection: false,
  };
}
