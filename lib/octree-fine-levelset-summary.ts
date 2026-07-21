/** Sparse sizing summaries derived from globally keyed fine level-set bricks. */

import {
  FINE_LEVELSET_SAMPLE_FLAGS,
  type FineLevelSetBrickPlan,
  type FineLevelSetVec3,
  packFineLevelSetBrickKey,
  unpackFineLevelSetBrickKey,
} from "./octree-fine-levelset-bricks";

export const FINE_LEVELSET_SUMMARY_BYTES = 32;

export const FINE_LEVELSET_SUMMARY_FLAG = Object.freeze({
  valid: 1 << 0,
  complete: 1 << 1,
  metric: 1 << 2,
  containsInterface: 1 << 3,
} as const);

export interface FineLevelSetSummary {
  readonly level: number;
  readonly coord: FineLevelSetVec3;
  readonly key: string;
  readonly generation: number;
  readonly minimumPhi: number;
  readonly maximumPhi: number;
  readonly minimumAbsolutePhi: number;
  readonly minimumSolidFraction: number;
  readonly maximumSolidFraction: number;
  readonly validSampleCount: number;
  readonly childMask: number;
  readonly flags: number;
}

export type FineLevelSetSummaryDecision = "refine" | "coarse" | "inconclusive";

function summaryKey(level: number, coord: FineLevelSetVec3): string {
  return `${level}:${coord[0]},${coord[1]},${coord[2]}`;
}

function validateGeneration(generation: number): number {
  if (!Number.isSafeInteger(generation) || generation < 1 || generation > 0xffff_ffff) {
    throw new RangeError("Fine level-set summary generation must be a positive u32 integer");
  }
  return generation;
}

/** Build one deterministic summary without reading any nonresident brick. */
export function summarizeFineLevelSetBrick(
  plan: FineLevelSetBrickPlan,
  brickKey: number,
  generationValue: number,
  phi: ArrayLike<number>,
  sampleFlags: ArrayLike<number>,
  solidFraction?: ArrayLike<number>,
): FineLevelSetSummary {
  const generation = validateGeneration(generationValue);
  if (phi.length !== plan.samplesPerBrick || sampleFlags.length !== plan.samplesPerBrick
    || (solidFraction !== undefined && solidFraction.length !== plan.samplesPerBrick)) {
    throw new RangeError("Fine level-set summary channels must match samplesPerBrick");
  }
  const coord = unpackFineLevelSetBrickKey(plan, brickKey);
  let minimumPhi = Number.POSITIVE_INFINITY;
  let maximumPhi = Number.NEGATIVE_INFINITY;
  let minimumAbsolutePhi = Number.POSITIVE_INFINITY;
  let minimumSolidFraction = 1;
  let maximumSolidFraction = 0;
  let validSampleCount = 0;
  let allMetric = true;
  for (let index = 0; index < plan.samplesPerBrick; index += 1) {
    if ((Number(sampleFlags[index]) & FINE_LEVELSET_SAMPLE_FLAGS.valid) === 0) continue;
    const value = Number(phi[index]);
    if (!Number.isFinite(value)) throw new RangeError("Valid fine phi summary samples must be finite");
    minimumPhi = Math.min(minimumPhi, value);
    maximumPhi = Math.max(maximumPhi, value);
    minimumAbsolutePhi = Math.min(minimumAbsolutePhi, Math.abs(value));
    const solid = solidFraction === undefined ? 0 : Number(solidFraction[index]);
    if (!Number.isFinite(solid) || solid < 0 || solid > 1) {
      throw new RangeError("Fine level-set solid fractions must be in [0, 1]");
    }
    minimumSolidFraction = Math.min(minimumSolidFraction, solid);
    maximumSolidFraction = Math.max(maximumSolidFraction, solid);
    // Known samples have completed redistance and therefore support the
    // signed-distance/Lipschitz rejection proof used below.
    allMetric &&= (Number(sampleFlags[index]) & FINE_LEVELSET_SAMPLE_FLAGS.known) !== 0;
    validSampleCount += 1;
  }
  let flags = 0;
  if (validSampleCount > 0) flags |= FINE_LEVELSET_SUMMARY_FLAG.valid;
  if (validSampleCount === plan.samplesPerBrick) flags |= FINE_LEVELSET_SUMMARY_FLAG.complete;
  if (validSampleCount > 0 && allMetric) flags |= FINE_LEVELSET_SUMMARY_FLAG.metric;
  if (minimumPhi <= 0 && maximumPhi >= 0) flags |= FINE_LEVELSET_SUMMARY_FLAG.containsInterface;
  return {
    level: 0, coord, key: summaryKey(0, coord), generation, minimumPhi, maximumPhi, minimumAbsolutePhi,
    minimumSolidFraction, maximumSolidFraction, validSampleCount, childMask: 1, flags,
  };
}

/**
 * Reduce resident summaries into resident ancestors only. Missing children are
 * explicit through the complete flag; no dense mip pyramid is materialized.
 */
export function buildSparseFineLevelSetSummaryHierarchy(
  base: readonly FineLevelSetSummary[],
  maximumLevel: number,
): ReadonlyMap<string, FineLevelSetSummary> {
  if (!Number.isSafeInteger(maximumLevel) || maximumLevel < 0 || maximumLevel > 30) {
    throw new RangeError("Fine summary maximum level must be in [0, 30]");
  }
  const output = new Map<string, FineLevelSetSummary>();
  let current = [...base];
  for (const summary of current) {
    if (summary.level !== 0) throw new RangeError("Fine summary hierarchy inputs must be level zero");
    if (output.has(summary.key)) throw new RangeError(`Duplicate fine summary ${summary.key}`);
    output.set(summary.key, summary);
  }
  for (let level = 1; level <= maximumLevel && current.length > 0; level += 1) {
    const groups = new Map<string, FineLevelSetSummary[]>();
    for (const child of current) {
      const parentCoord = child.coord.map((value) => Math.floor(value / 2)) as unknown as FineLevelSetVec3;
      const key = summaryKey(level, parentCoord);
      const group = groups.get(key) ?? [];
      group.push(child);
      groups.set(key, group);
    }
    const next: FineLevelSetSummary[] = [];
    for (const [key, children] of [...groups].sort(([a], [b]) => a.localeCompare(b))) {
      const coord = children[0].coord.map((value) => Math.floor(value / 2)) as unknown as FineLevelSetVec3;
      const generation = children[0].generation;
      let childMask = 0;
      let flags = FINE_LEVELSET_SUMMARY_FLAG.valid | FINE_LEVELSET_SUMMARY_FLAG.complete
        | FINE_LEVELSET_SUMMARY_FLAG.metric;
      let minimumPhi = Number.POSITIVE_INFINITY;
      let maximumPhi = Number.NEGATIVE_INFINITY;
      let minimumAbsolutePhi = Number.POSITIVE_INFINITY;
      let minimumSolidFraction = 1;
      let maximumSolidFraction = 0;
      let validSampleCount = 0;
      for (const child of children) {
        if (child.generation !== generation) flags &= ~FINE_LEVELSET_SUMMARY_FLAG.valid;
        const localX = child.coord[0] - coord[0] * 2;
        const localY = child.coord[1] - coord[1] * 2;
        const localZ = child.coord[2] - coord[2] * 2;
        childMask |= 1 << (localX + 2 * (localY + 2 * localZ));
        flags &= child.flags | ~(FINE_LEVELSET_SUMMARY_FLAG.valid | FINE_LEVELSET_SUMMARY_FLAG.complete
          | FINE_LEVELSET_SUMMARY_FLAG.metric);
        minimumPhi = Math.min(minimumPhi, child.minimumPhi);
        maximumPhi = Math.max(maximumPhi, child.maximumPhi);
        minimumAbsolutePhi = Math.min(minimumAbsolutePhi, child.minimumAbsolutePhi);
        minimumSolidFraction = Math.min(minimumSolidFraction, child.minimumSolidFraction);
        maximumSolidFraction = Math.max(maximumSolidFraction, child.maximumSolidFraction);
        validSampleCount += child.validSampleCount;
      }
      if (childMask !== 0xff) flags &= ~FINE_LEVELSET_SUMMARY_FLAG.complete;
      if (minimumPhi <= 0 && maximumPhi >= 0) flags |= FINE_LEVELSET_SUMMARY_FLAG.containsInterface;
      const parent = { level, coord, key, generation, minimumPhi, maximumPhi, minimumAbsolutePhi,
        minimumSolidFraction, maximumSolidFraction, validSampleCount, childMask, flags };
      output.set(key, parent);
      next.push(parent);
    }
    current = next;
  }
  return output;
}

/** Conservative sizing decision; stale or incomplete summaries never coarsen. */
export function classifyFineLevelSetSummary(
  summary: FineLevelSetSummary | undefined,
  options: { generation: number; refinementDistance: number; samplingRadius: number },
): FineLevelSetSummaryDecision {
  validateGeneration(options.generation);
  if (!Number.isFinite(options.refinementDistance) || options.refinementDistance < 0
    || !Number.isFinite(options.samplingRadius) || options.samplingRadius < 0) {
    throw new RangeError("Fine summary distances must be finite and non-negative");
  }
  const required = FINE_LEVELSET_SUMMARY_FLAG.valid | FINE_LEVELSET_SUMMARY_FLAG.complete
    | FINE_LEVELSET_SUMMARY_FLAG.metric;
  if (!summary || summary.generation !== options.generation || (summary.flags & required) !== required) {
    return "inconclusive";
  }
  if (summary.minimumAbsolutePhi <= options.refinementDistance) return "refine";
  // A completed metric signed-distance summary is 1-Lipschitz. Subtracting
  // the furthest unsampled support radius makes this a proof, not a heuristic.
  if (summary.minimumAbsolutePhi - options.samplingRadius > options.refinementDistance) return "coarse";
  return "inconclusive";
}

export function fineSummaryKeyForBrick(plan: FineLevelSetBrickPlan, coord: FineLevelSetVec3): string {
  return summaryKey(0, unpackFineLevelSetBrickKey(plan, packFineLevelSetBrickKey(plan, coord)));
}
