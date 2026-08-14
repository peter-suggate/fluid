/**
 * Control-word ABI of the power-liquids coarse level-set publication.
 *
 * The engine's t=0 acceptance gate reads this control before it will admit a
 * global-fine authority, and it reads it as sixteen published words: a flag
 * set, a first-error row, per-phase counters, the generation, and the validity
 * stamp. That is a description of a buffer the GPU wrote, not solver physics,
 * so it lives where the producer and the shared engine can both name it — the
 * same split the sample directory already has in
 * `lib/core/octree-power-coarse-levelset-sample-abi.ts`.
 *
 * The producer imports these back and re-exports them, so the power package
 * and its tests still name the wire format in one place.
 */

export const OCTREE_POWER_COARSE_LEVELSET_VALID = 0x8000_0000;

export const OCTREE_POWER_COARSE_LEVELSET_ERROR = Object.freeze({
  capacity: 1, invalidRow: 2, invalidVelocity: 4, invalidCatalog: 8,
  invalidFineOffsets: 16, invalidFineSample: 32, fineContributionBound: 64,
  sampleIndex: 128, invalidSource: 256,
} as const);

export interface OctreePowerCoarseLevelSetControl {
  readonly flags: number; readonly firstErrorRow: number; readonly rowCount: number;
  readonly advectedRows: number; readonly uniformUpdates: number; readonly transitionUpdates: number;
  readonly representationPasses: number; readonly correctedRows: number;
  readonly interfaceRows: number; readonly contributionCount: number; readonly generation: number;
  readonly valid: number;
}

export function unpackOctreePowerCoarseLevelSetControl(words: ArrayLike<number>): OctreePowerCoarseLevelSetControl {
  if (words.length < 16) throw new RangeError("Power coarse-phi control needs sixteen words");
  return { flags: Number(words[0]) >>> 0, firstErrorRow: Number(words[1]) >>> 0, rowCount: Number(words[2]) >>> 0,
    advectedRows: Number(words[3]) >>> 0, uniformUpdates: Number(words[4]) >>> 0, transitionUpdates: Number(words[5]) >>> 0,
    representationPasses: Number(words[6]) >>> 0, correctedRows: Number(words[7]) >>> 0,
    interfaceRows: Number(words[8]) >>> 0, contributionCount: Number(words[9]) >>> 0,
    generation: Number(words[10]) >>> 0, valid: Number(words[11]) >>> 0 };
}
