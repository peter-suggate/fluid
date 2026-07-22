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
