import type { SparseCM12ActivityPolicy } from "./policy";

/**
 * The accepted-surface proof and prediction region of the resident uniform ABI.
 *
 * Feature-preserving coarsening sizes material by deformation and
 * representability, so the page-wide normal proof, the curvature floor and the
 * incoming-impact retention floor no longer exist in the shader. Their lanes
 * stay in place — a vec4 is aligned whether or not every component is read —
 * and publish zero, so the struct offsets every other consumer depends on do
 * not move. Do not re-use a reserved lane without renaming it on both sides.
 */
export function packAdaptivitySurfaceParameters(
  f: Float32Array, u: Uint32Array, surfaceProofWord: number,
  policy: SparseCM12ActivityPolicy, finestCellSize_m: number, dt_s: number,
  brickFineResolution: number, previousSignature?: string,
): string {
    f[surfaceProofWord] = policy.surfaceDisplacementToleranceCells * finestCellSize_m;
    f[surfaceProofWord + 1] = 0; // reserved: retired normal-angle proof
    u[surfaceProofWord + 2] = policy.surfaceCoarseningEnabled ? 1 : 0;
    // Low bits retain the forced-rung QA ABI; high bits are independent controls.
    u[surfaceProofWord + 3] = (policy.forcedSurfaceResolutionForQA ?? 0)
      | (policy.freezeTopology ? 0x80000000 : 0)
      | (policy.legacyFaceTransportForQA ? 0x40000000 : 0);
    const velocityThresholds = new Float32Array(8);
    const finestLevel = Math.log2(brickFineResolution);
    velocityThresholds[finestLevel] = policy.finestTravelCells;
    if (finestLevel > 0) velocityThresholds[finestLevel - 1] = policy.fourTravelCells;
    if (finestLevel > 1) velocityThresholds[finestLevel - 2] = policy.twoTravelCells;
    for (let level = finestLevel - 3; level > 0; level -= 1) {
      velocityThresholds[level] = 0.5 * velocityThresholds[level + 1]!;
    }
    if (policy.coarseFirst) {
      const finestTravel = Math.sqrt(2 * policy.energyThreshold) * dt_s / finestCellSize_m;
      for (let level = 0; level <= finestLevel; level++) {
        velocityThresholds[level] = finestTravel * 2 ** (level - finestLevel);
      }
    }
    f.set(velocityThresholds, surfaceProofWord + 4);
    // enabled, finest specific kinetic energy, then two reserved lanes.
    f.set([policy.coarseFirst ? 1 : 0, policy.energyThreshold, 0, 0], surfaceProofWord + 12);
    const policySignature = JSON.stringify([policy.coarseFirst, policy.energyThreshold,
      policy.surfaceDisplacementToleranceCells, policy.surfaceQuietEpochs]);
    const changed = previousSignature !== undefined
      && previousSignature !== policySignature;
    f.set([0, policy.surfaceQuietEpochs, changed ? 1 : 0, 0], surfaceProofWord + 16);
    return policySignature;
}
