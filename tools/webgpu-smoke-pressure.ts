export interface OctreeMGPCGDiagnostics {
  readonly flags: number;
  readonly converged: boolean;
  readonly iterations: number;
  readonly rows: number;
  readonly residualSquared: number;
  readonly rhsSquared: number;
  readonly relativeResidualSquared: number;
  readonly relativeResidual: number;
}

/** Decode the 64-byte GPU MGPCG control publication used by Dawn QA. */
export function decodeOctreeMGPCGDiagnostics(words: Uint32Array): OctreeMGPCGDiagnostics {
  if (words.length < 16) throw new RangeError("Octree MGPCG diagnostics require sixteen words");
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const residualSquared = floats[4];
  const rhsSquared = floats[5];
  const relativeResidualSquared = residualSquared / Math.max(rhsSquared, 1e-30);
  return {
    flags: words[0],
    converged: words[1] !== 0,
    iterations: words[2],
    rows: words[3],
    residualSquared,
    rhsSquared,
    relativeResidualSquared,
    relativeResidual: Math.sqrt(relativeResidualSquared),
  };
}

/** Float32 Dawn QA policy; the 2017 paper reports iteration counts, not this tolerance. */
export function octreeMGPCGDiagnosticsAreAcceptable(
  value: OctreeMGPCGDiagnostics | undefined,
  maximumRelativeResidualSquared = 1e-8,
): value is OctreeMGPCGDiagnostics {
  return value !== undefined
    && value.flags === 0
    && value.converged
    && value.rows > 0
    && Number.isFinite(value.residualSquared) && value.residualSquared >= 0
    && Number.isFinite(value.rhsSquared) && value.rhsSquared >= 0
    && Number.isFinite(value.relativeResidualSquared) && value.relativeResidualSquared <= maximumRelativeResidualSquared
    && Number.isFinite(value.relativeResidual) && value.relativeResidual <= Math.sqrt(maximumRelativeResidualSquared);
}

/**
 * Both selectable pressure solvers publish the same control ABI, but the
 * Galerkin lane also has the production absolute-RMS floor used for nearly
 * zero right-hand sides. Do not make Dawn QA reinterpret an accepted
 * Galerkin solve as an MGPCG relative-residual failure.
 */
export function octreePowerPressureDiagnosticsAreAcceptable(
  solverLabel: string | undefined,
  value: OctreeMGPCGDiagnostics | undefined,
  maximumRelativeResidualSquared = 1e-8,
  galerkinAbsoluteRmsTolerance = 1e-7,
): value is OctreeMGPCGDiagnostics {
  if (octreeMGPCGDiagnosticsAreAcceptable(value, maximumRelativeResidualSquared)) return true;
  const candidate = value as OctreeMGPCGDiagnostics | undefined;
  return solverLabel?.includes("fixed native-L2 Galerkin") === true
    && candidate !== undefined
    && candidate.flags === 0
    && candidate.converged
    && candidate.rows > 0
    && Number.isFinite(candidate.residualSquared)
    && candidate.residualSquared >= 0
    && candidate.residualSquared <= candidate.rows * galerkinAbsoluteRmsTolerance * galerkinAbsoluteRmsTolerance
    && Number.isFinite(candidate.rhsSquared)
    && candidate.rhsSquared >= 0;
}

/**
 * Solver-aware form of the per-step stability-envelope residual gate.
 * `maximumResidualRms` is the compact live-row RMS published by the octree
 * projection, so it uses the same 1e-7 Galerkin floor as the GPU control.
 */
export function octreePowerPressureEnvelopeIsAcceptable(
  solverLabel: string | undefined,
  maximumRelativeResidual: number | undefined,
  maximumResidualRms: number | undefined,
  relativeTolerance = 1e-4,
  galerkinAbsoluteRmsTolerance = 1.1e-7,
): boolean {
  const relativeAccepted = Number.isFinite(maximumRelativeResidual)
    && maximumRelativeResidual! >= 0
    && maximumRelativeResidual! <= relativeTolerance;
  if (relativeAccepted) return true;
  return solverLabel?.includes("fixed native-L2 Galerkin") === true
    && Number.isFinite(maximumResidualRms)
    && maximumResidualRms! >= 0
    && maximumResidualRms! <= galerkinAbsoluteRmsTolerance;
}
