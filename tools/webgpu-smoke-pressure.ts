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
  const residualSquared = floats[10] + floats[11];
  const rhsSquared = floats[8] + floats[9];
  const relativeResidualSquared = residualSquared / Math.max(rhsSquared, 1e-30);
  return {
    flags: words[0],
    converged: words[1] !== 0,
    iterations: words[2],
    rows: words[4],
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

/** The production pressure authority has one residual policy. */
export function octreePowerPressureDiagnosticsAreAcceptable(
  _solverLabel: string | undefined,
  value: OctreeMGPCGDiagnostics | undefined,
  maximumRelativeResidualSquared = 1e-8,
): value is OctreeMGPCGDiagnostics {
  return octreeMGPCGDiagnosticsAreAcceptable(value, maximumRelativeResidualSquared);
}

/** Solver-aware form of the per-step stability-envelope residual gate. */
export function octreePowerPressureEnvelopeIsAcceptable(
  _solverLabel: string | undefined,
  maximumRelativeResidual: number | undefined,
  _maximumResidualRms: number | undefined,
  relativeTolerance = 1e-4,
): boolean {
  return Number.isFinite(maximumRelativeResidual)
    && maximumRelativeResidual! >= 0
    && maximumRelativeResidual! <= relativeTolerance;
}

/**
 * Convert the algebraic pressure residual to the residual flux left by the
 * projection. Aanjaneya et al. Eq. (3)/(4) gives
 * `fluxAfter = (dt / rho) * (b - A p)` for the production sign convention.
 * This is a units conversion of existing diagnostics, not a looser solve gate.
 */
export function octreeProjectedVariationalResidualRms(
  algebraicResidualRms: number | undefined,
  dt_s: number,
  density_kg_m3: number,
): number | undefined {
  if (algebraicResidualRms === undefined
    || !Number.isFinite(algebraicResidualRms) || algebraicResidualRms < 0
    || !Number.isFinite(dt_s) || dt_s < 0
    || !Number.isFinite(density_kg_m3) || density_kg_m3 <= 0) return undefined;
  const residual = algebraicResidualRms * dt_s / density_kg_m3;
  return Number.isFinite(residual) && residual >= 0 ? residual : undefined;
}
