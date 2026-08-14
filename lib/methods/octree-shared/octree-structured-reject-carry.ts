/**
 * Host decoder for the structured-dynamics GPU reject carry.
 *
 * `rejectSample`/`rejectVector` in the power backend's
 * `webgpu-octree-structured-dynamics.ts` record
 * a fault into the accepted-authority words rather than aborting: word 0 takes
 * `ERROR_SAMPLE` via `atomicOr`, word 1 takes `(stage<<24)|index` via
 * `atomicMin`, and `rejectVector` additionally stores a vec4f detail into words
 * 6..9 and the workset class into word 10 when it wins that minimum.
 *
 * That carry then gates `prepareStructuredDynamics`, which publishes a zero
 * workgroup count for all nine classes. A single rejected face therefore turns
 * the remainder of the step — and every later step — into a silent no-op. This
 * module exists so that freeze is reported as a named stage and row instead of
 * being rediscovered hundreds of steps later as a step-count or volume-drift
 * failure.
 *
 * IMPORTANT: word 1 is an `atomicMin` over `(stage<<24)|index`, so the retained
 * record is the numerically smallest stage — NOT the temporally first fault. A
 * stage-1 rejection anywhere masks every higher-numbered stage in the same
 * submission. Treat `maskedStagesPossible` accordingly.
 */

import { OCTREE_STRUCTURED_GPU_ERROR } from "./octree-structured-receipt-abi";

/** Words the carry occupies in the accepted-authority buffer. */
export const OCTREE_STRUCTURED_REJECT_CARRY_WORDS = 11;

const INVALID = 0xffff_ffff;

/** Stage codes as they appear at the `rejectSample`/`rejectVector` call sites. */
export const OCTREE_STRUCTURED_REJECT_STAGES: Readonly<Record<number, string>> = Object.freeze({
  1: "advect: advecting velocity sample invalid at the face centroid",
  2: "advect: transported velocity sample invalid at the departure point",
  3: "advect: non-finite aperture, area, solid velocity, or prior face value",
  10: "force: aperture outside [0,1] or non-finite solid normal velocity",
  11: "force: gravity update produced a non-finite face value",
  20: "divergence: row slot count exceeds the catalog maximum",
  21: "divergence: row slot handle is unpublished or out of range",
  22: "divergence: non-finite sign, area, aperture, solid, or face sample",
  23: "divergence: non-finite flux, non-positive cell volume, or invalid dt",
  30: "project: owner or neighbour row outside the accepted row authority",
  31: "project: non-finite face normal",
  32: "project: aperture outside [0,1] or non-finite solid normal velocity",
  33: "project: projected face value is non-finite",
  50: "reconstruct: case header slot count exceeds the catalog maximum",
  51: "reconstruct: row slot handle is unpublished or out of range",
  52: "reconstruct: neighbour row outside the accepted row authority",
  53: "reconstruct: non-finite face sample or reconstruction coefficient",
  54: "reconstruct: reconstructed cell-centred vector is non-finite",
});

/** Stages that carry a vec4f detail and workset class in words 6..10. */
const DETAILED_STAGES = Object.freeze(new Set([1, 2]));

export interface OctreeStructuredRejectCarry {
  /** True when word 0 carries no error bits and word 1 is unclaimed. */
  readonly clean: boolean;
  /** Set error-bit names from `OCTREE_STRUCTURED_GPU_ERROR`. */
  readonly flags: readonly string[];
  readonly stage?: number;
  readonly stageDescription?: string;
  /** Face handle or row index, depending on the stage. */
  readonly index?: number;
  /** Present only for stages that call `rejectVector`. */
  readonly detail?: readonly [number, number, number, number];
  readonly worksetClass?: number;
  /**
   * Stages that could be masked by this record, because word 1 keeps the
   * numeric minimum rather than the first fault.
   */
  readonly maskedStagesPossible: readonly number[];
  /** One line suitable for a per-step log or an assertion message. */
  readonly summary: string;
}

function flagNames(word: number): string[] {
  return Object.entries(OCTREE_STRUCTURED_GPU_ERROR)
    .filter(([, bit]) => (word & bit) !== 0)
    .map(([name]) => name);
}

/**
 * Decode the carry from the accepted-authority words.
 *
 * Pass at least `OCTREE_STRUCTURED_REJECT_CARRY_WORDS` words when the detail
 * payload is wanted; shorter reads still decode the stage and index.
 */
export function decodeOctreeStructuredRejectCarry(
  words: ArrayLike<number>,
): OctreeStructuredRejectCarry {
  if (words.length < 2) {
    return Object.freeze({
      clean: false, flags: Object.freeze(["truncated"]),
      maskedStagesPossible: Object.freeze([]),
      summary: "structured reject carry is truncated (needs at least two words)",
    });
  }
  const flagWord = Number(words[0]) >>> 0;
  const packed = Number(words[1]) >>> 0;
  const flags = Object.freeze(flagNames(flagWord));
  if (flagWord === 0 && packed === INVALID) {
    return Object.freeze({
      clean: true, flags, maskedStagesPossible: Object.freeze([]),
      summary: "structured dynamics accepted every sample",
    });
  }
  if (packed === INVALID) {
    return Object.freeze({
      clean: false, flags, maskedStagesPossible: Object.freeze([]),
      summary: `structured dynamics raised ${flags.join("|") || `flags 0x${flagWord.toString(16)}`}`
        + " with no claimed stage record",
    });
  }
  const stage = packed >>> 24;
  const index = packed & 0x00ff_ffff;
  const stageDescription = OCTREE_STRUCTURED_REJECT_STAGES[stage];
  const masked = Object.freeze(Object.keys(OCTREE_STRUCTURED_REJECT_STAGES)
    .map(Number).filter((candidate) => candidate > stage));
  const detailed = DETAILED_STAGES.has(stage)
    && words.length >= OCTREE_STRUCTURED_REJECT_CARRY_WORDS;
  const detail = detailed
    ? Object.freeze(bitsToFloats([words[6]!, words[7]!, words[8]!, words[9]!]))
    : undefined;
  const worksetClass = detailed ? Number(words[10]) >>> 0 : undefined;
  return Object.freeze({
    clean: false, flags, stage, stageDescription, index, detail, worksetClass,
    maskedStagesPossible: masked,
    summary: `structured dynamics rejected at stage ${stage}`
      + ` (${stageDescription ?? "unknown stage"}) index ${index}`
      + (worksetClass === undefined ? "" : ` workset class ${worksetClass}`)
      + (detail === undefined ? "" : ` detail [${detail.map((v) => v.toPrecision(6)).join(", ")}]`)
      + (masked.length === 0 ? "" : `; stages >${stage} are masked by the atomicMin carry`),
  });
}

function bitsToFloats(words: readonly number[]): [number, number, number, number] {
  const bits = Uint32Array.from(words.map((word) => word >>> 0));
  const floats = new Float32Array(bits.buffer);
  return [floats[0]!, floats[1]!, floats[2]!, floats[3]!];
}
