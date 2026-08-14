/**
 * Provenance of the Section 5 closest-point flood, decoded from what the
 * redistance transform already leaves in memory.
 *
 * `webgpu-octree-fine-levelset-redistance.ts` runs a descending jump-flood: a
 * ladder of strides P, P/2, ..., 1 followed by a fixed number of stride-1
 * collar repairs. Every resident sample ends the transform holding the index of
 * the seed it inherited its distance from (`workA[index]`) and the packed
 * sub-cell closest-point code of that seed (`flags[index] >> SAMPLE_FLAG_BITS`).
 * Neither is cleared by the commit pass, so the flood's dependency graph is
 * resident after the frame with no instrumentation at all.
 *
 * This module is the ABI for reading it and the arithmetic for the one question
 * the flood cannot answer about itself: the ladder is sized from the authored
 * band, but the hops the samples actually take are a property of the flow. When
 * the two diverge the transform spends whole passes gathering from a radius no
 * sample ever needed. Nothing here knows about WebGPU; the same decode runs in
 * the histogram shader and in tests.
 */
import { FINE_LEVELSET_SAMPLE_FLAGS } from "./fine-levelset-brick-abi";
import { linearFromHex } from "./visualization-decorations";
import {
  decorationVisualization,
  hasFields,
  type Visualization,
} from "./visualization-registry";

/** Low bits of `flags` reserved for sample state; the closest-point code sits above them. */
export const FINE_FLOOD_SAMPLE_FLAG_BITS = 5;
export const FINE_FLOOD_CLOSEST_POINT_FRACTION_MASK = 0x00ff_ffff;
export const FINE_FLOOD_CLOSEST_POINT_FRACTION_SCALE = 16_777_215;
export const FINE_FLOOD_INVALID = 0xffff_ffff;

/**
 * Sample flags published by the redistance commit pass. A sample with no
 * `valid` bit carries no distance this generation and therefore no provenance.
 *
 * Aliased onto the brick layout rather than restated. These are the *same* bits
 * — the commit writes `flags[index] = closestPointCode | VALID | INTERFACE |
 * NEGATIVE` into the field `octree-fine-levelset-bricks.ts` defines — and
 * restating them here previously put `negative` at bit 2, which is `known` in
 * that layout. Nothing read it until the cell trace did, and the symptom was a
 * liquid fraction of zero for every cell in the water. Deriving them makes the
 * two unable to disagree again.
 */
export const FINE_FLOOD_SAMPLE_FLAGS = Object.freeze({
  valid: FINE_LEVELSET_SAMPLE_FLAGS.valid,
  interface: FINE_LEVELSET_SAMPLE_FLAGS.interface,
  negative: FINE_LEVELSET_SAMPLE_FLAGS.negative,
} as const);

/**
 * Face directions the seeding pass searches for a sign change, in the shader's
 * own order. A code's direction selects one of these; 6 and 7 mean the closest
 * point coincides with the sample centre and carries no offset.
 */
export const FINE_FLOOD_DIRECTION_DELTAS: readonly (readonly [number, number, number])[] = Object.freeze([
  [-1, 0, 0], [1, 0, 0],
  [0, -1, 0], [0, 1, 0],
  [0, 0, -1], [0, 0, 1],
].map((delta) => Object.freeze(delta as [number, number, number])));

export type FineFloodClosestPointKind =
  /** No code: this sample never held a crossing and can only be a follower. */
  | "absent"
  /** A sign change across one face; the closest point is a sub-cell offset. */
  | "axis"
  /** phi is exactly zero here, or the crossing quantized onto the centre. */
  | "coincident";

export interface FineFloodClosestPoint {
  readonly kind: FineFloodClosestPointKind;
  /** Index into `FINE_FLOOD_DIRECTION_DELTAS`, or undefined when not an axis crossing. */
  readonly direction?: number;
  /** Fraction of a fine cell along that direction, in [0, 1]. */
  readonly fraction: number;
}

/** Decode the packed code stored above the sample flags. */
export function decodeFineFloodClosestPointCode(code: number): FineFloodClosestPoint {
  if (!Number.isSafeInteger(code) || code < 0) {
    throw new RangeError("Fine flood closest-point code must be a non-negative integer");
  }
  if (code === 0) return { kind: "absent", fraction: 0 };
  const direction = code >>> 24;
  if (direction >= FINE_FLOOD_DIRECTION_DELTAS.length) return { kind: "coincident", fraction: 0 };
  return {
    kind: "axis",
    direction,
    fraction: (code & FINE_FLOOD_CLOSEST_POINT_FRACTION_MASK) / FINE_FLOOD_CLOSEST_POINT_FRACTION_SCALE,
  };
}

/** The closest-point code lives above the persistent state bits. */
export function fineFloodClosestPointCode(flags: number): number {
  return flags >>> FINE_FLOOD_SAMPLE_FLAG_BITS;
}

/**
 * World-free closest point of a seed, in fine-lattice units, matching
 * `FINE_LEVELSET_JFA_CLOSEST_POINT_BODY` exactly: the seed's cell centre plus
 * the sub-cell offset its crossing recorded.
 */
export function fineFloodClosestPoint(
  seedCell: readonly [number, number, number],
  code: number,
): readonly [number, number, number] {
  const closest = decodeFineFloodClosestPointCode(code);
  const centre: [number, number, number] = [seedCell[0] + 0.5, seedCell[1] + 0.5, seedCell[2] + 0.5];
  if (closest.kind !== "axis" || closest.direction === undefined) return centre;
  const delta = FINE_FLOOD_DIRECTION_DELTAS[closest.direction];
  return [
    centre[0] + delta[0] * closest.fraction,
    centre[1] + delta[1] * closest.fraction,
    centre[2] + delta[2] * closest.fraction,
  ];
}

/** Chebyshev (max-axis) hop from a sample to its seed, in fine cells. */
export function fineFloodAxisHop(
  sampleCell: readonly [number, number, number],
  seedCell: readonly [number, number, number],
): number {
  return Math.max(
    Math.abs(seedCell[0] - sampleCell[0]),
    Math.abs(seedCell[1] - sampleCell[1]),
    Math.abs(seedCell[2] - sampleCell[2]),
  );
}

/**
 * Descending passes a jump flood needs to deliver a hop of this reach.
 *
 * A ladder starting at stride P sums to P + P/2 + ... + 1 = 2P - 1, and the
 * binary decomposition of any offset below that sum is exactly what the
 * descending passes walk. So the shortest ladder covering `maximumAxisHop` is
 * the one whose first stride is the smallest power of two with 2P - 1 >= hop,
 * and its length is log2(P) + 1. This is deliberately the same rule
 * `planFineLevelSetJFAStrides` uses to size the ladder from the authored band,
 * so planned and required pass counts are the same measurement applied to a
 * different reach: one authored, one observed.
 */
export function fineFloodDescendingPassesForReach(maximumAxisHop: number): number {
  if (!Number.isSafeInteger(maximumAxisHop) || maximumAxisHop < 0) {
    throw new RangeError("Fine flood reach must be a non-negative integer");
  }
  if (maximumAxisHop === 0) return 0;
  let stride = 1;
  let passes = 1;
  while (stride * 2 - 1 < maximumAxisHop) { stride *= 2; passes += 1; }
  return passes;
}

/**
 * Passes a ladder may hold, matching `FINE_LEVELSET_JFA_MAX_PASSES`. Bins cover
 * a self-seeded sample plus every pass a legal ladder can encode.
 */
export const FINE_FLOOD_MAXIMUM_LADDER_PASSES = 13;
export const FINE_FLOOD_HISTOGRAM_PASS_BINS = FINE_FLOOD_MAXIMUM_LADDER_PASSES + 1;

export interface FineFloodLadderPlan {
  /** The encoded schedule, exactly as `planFineLevelSetJFAStrides` returns it. */
  readonly strides: readonly number[];
  /** Leading strictly-descending passes; the reach-covering part of the ladder. */
  readonly descendingPasses: number;
  /** Trailing stride-1 passes that close page-boundary turning routes. */
  readonly collarRepairPasses: number;
  /** Chebyshev reach of the descending part, 2P - 1 for a first stride of P. */
  readonly descendingReach: number;
  /** Reach of the whole schedule, collar repairs included. */
  readonly encodedReach: number;
  /** `prefixReach[k]` is the reach of the first k + 1 passes. */
  readonly prefixReach: readonly number[];
}

/**
 * Leading passes of an encoded ladder needed to cover a hop.
 *
 * Passes execute in order, so a schedule can only be shortened from the tail;
 * asking which prefix covers a reach is therefore the question that maps onto a
 * change anyone could make. Saturates at the ladder length rather than
 * reporting an impossible pass index, and the caller compares `encodedReach`
 * against the hop to learn whether the whole ladder was even sufficient.
 */
export function fineFloodLadderPrefixForReach(
  prefixReach: readonly number[],
  maximumAxisHop: number,
): number {
  if (!Number.isSafeInteger(maximumAxisHop) || maximumAxisHop < 0) {
    throw new RangeError("Fine flood reach must be a non-negative integer");
  }
  if (maximumAxisHop === 0) return 0;
  const covering = prefixReach.findIndex((reach) => reach >= maximumAxisHop);
  return covering < 0 ? prefixReach.length : covering + 1;
}

/**
 * Split an encoded ladder into its descending part and its collar repairs.
 *
 * The planner appends repairs as additional stride-1 entries, so the boundary
 * is the first index at which the stride stops decreasing. Reading it back this
 * way keeps the summary honest when the repair count is retuned: nothing here
 * hardcodes five.
 */
export function describeFineFloodLadder(strides: readonly number[]): FineFloodLadderPlan {
  if (strides.length === 0) throw new RangeError("Fine flood ladder must have at least one pass");
  for (const stride of strides) {
    if (!Number.isSafeInteger(stride) || stride < 1 || (stride & (stride - 1)) !== 0) {
      throw new RangeError("Fine flood ladder strides must be powers of two");
    }
  }
  if (strides.length > FINE_FLOOD_MAXIMUM_LADDER_PASSES) {
    throw new RangeError(`Fine flood ladder exceeds ${FINE_FLOOD_MAXIMUM_LADDER_PASSES} passes`);
  }
  let descendingPasses = 1;
  while (descendingPasses < strides.length && strides[descendingPasses] < strides[descendingPasses - 1]) {
    descendingPasses += 1;
  }
  // Each pass gathers at its own stride, so the reach of a prefix is the sum of
  // its strides. The descending part alone sums to 2P - 1; the collar repairs
  // extend that by one cell each, which is why a hop can legally exceed the
  // descending reach without the flood having failed.
  const prefixReach: number[] = [];
  let reach = 0;
  for (const stride of strides) { reach += stride; prefixReach.push(reach); }
  return {
    strides: Object.freeze([...strides]),
    descendingPasses,
    collarRepairPasses: strides.length - descendingPasses,
    descendingReach: 2 * strides[0] - 1,
    encodedReach: reach,
    prefixReach: Object.freeze(prefixReach),
  };
}

export interface FineFloodHistogram {
  /** Resident, resolved samples binned by required descending passes. */
  readonly bins: readonly number[];
  /** Resident samples the flood left with no seed. */
  readonly unresolved: number;
  /** Largest Chebyshev hop observed, in fine cells. */
  readonly maximumAxisHop: number;
}

/**
 * Readback ABI for the GPU accumulation pass. The header precedes the bins so
 * a truncated read still yields the totals, and the bins are plain counts so
 * the decode below is the only place the layout is spelled out.
 */
export const FINE_FLOOD_HISTOGRAM_HEADER = Object.freeze({
  resident: 0,
  unresolved: 1,
  maximumAxisHop: 2,
  residentPages: 3,
} as const);
export const FINE_FLOOD_HISTOGRAM_HEADER_WORDS = 4;
export const FINE_FLOOD_HISTOGRAM_WORDS =
  FINE_FLOOD_HISTOGRAM_HEADER_WORDS + FINE_FLOOD_HISTOGRAM_PASS_BINS;

export interface FineFloodHistogramReadback {
  readonly histogram: FineFloodHistogram;
  /** Resident samples carrying a valid distance this generation. */
  readonly resident: number;
  /** Physical pages that matched the published generation. */
  readonly residentPages: number;
}

export function decodeFineFloodHistogram(words: ArrayLike<number>): FineFloodHistogramReadback {
  if (words.length < FINE_FLOOD_HISTOGRAM_WORDS) {
    throw new RangeError(`Fine flood histogram needs ${FINE_FLOOD_HISTOGRAM_WORDS} words`);
  }
  const bins: number[] = [];
  for (let bin = 0; bin < FINE_FLOOD_HISTOGRAM_PASS_BINS; bin += 1) {
    bins.push(words[FINE_FLOOD_HISTOGRAM_HEADER_WORDS + bin]);
  }
  return {
    histogram: {
      bins: Object.freeze(bins),
      unresolved: words[FINE_FLOOD_HISTOGRAM_HEADER.unresolved],
      maximumAxisHop: words[FINE_FLOOD_HISTOGRAM_HEADER.maximumAxisHop],
    },
    resident: words[FINE_FLOOD_HISTOGRAM_HEADER.resident],
    residentPages: words[FINE_FLOOD_HISTOGRAM_HEADER.residentPages],
  };
}

export interface FineFloodLadderSummary {
  readonly plan: FineFloodLadderPlan;
  readonly resident: number;
  readonly resolved: number;
  readonly unresolved: number;
  /** Samples that held their own crossing; the flood carried nothing to them. */
  readonly selfSeeded: number;
  readonly maximumAxisHop: number;
  /** Leading encoded passes whose combined reach covers the deepest hop seen. */
  readonly requiredLadderPasses: number;
  /**
   * Trailing encoded passes beyond the reach anything needed.
   *
   * This is a reach surplus, not a licence to delete passes: the collar repairs
   * exist to close page-boundary turning routes, where the path a seed actually
   * travelled is longer than the straight-line hop measured here. Read it as an
   * upper bound on what could be pruned, to be confirmed against residency.
   */
  readonly surplusLadderPasses: number;
  /**
   * False when the deepest observed hop exceeds the whole encoded reach. For a
   * warm publication that is expected rather than broken: topology carries and
   * remaps the previous closest-point field, so a link can be older and deeper
   * than any single frame's ladder could have built.
   */
  readonly coveredByEncodedLadder: boolean;
  /**
   * Passes a ladder sized from scratch for the observed reach would hold. Not
   * comparable to `requiredLadderPasses`, which measures the ladder that ran.
   */
  readonly idealLadderPasses: number;
  /** Share of resolved samples closed by the first k encoded passes, k = 0..bins-1. */
  readonly cumulativeResolvedShare: readonly number[];
}

/**
 * Fold a histogram against the ladder that produced it.
 *
 * `resident` is passed rather than summed because a sample can be resident and
 * excluded from every bin (no valid distance this generation); making the
 * caller state the denominator keeps the shares from silently renormalising
 * over a subset.
 */
export function summarizeFineFloodLadder(input: {
  readonly strides: readonly number[];
  readonly resident: number;
  readonly histogram: FineFloodHistogram;
}): FineFloodLadderSummary {
  const plan = describeFineFloodLadder(input.strides);
  const { bins, unresolved, maximumAxisHop } = input.histogram;
  if (bins.length !== FINE_FLOOD_HISTOGRAM_PASS_BINS) {
    throw new RangeError(`Fine flood histogram needs ${FINE_FLOOD_HISTOGRAM_PASS_BINS} pass bins`);
  }
  const resolved = bins.reduce((sum, count) => sum + count, 0);
  const requiredLadderPasses = fineFloodLadderPrefixForReach(plan.prefixReach, maximumAxisHop);
  let running = 0;
  const cumulativeResolvedShare = bins.map((count) => {
    running += count;
    return resolved > 0 ? running / resolved : 0;
  });
  return {
    plan,
    resident: input.resident,
    resolved,
    unresolved,
    selfSeeded: bins[0],
    maximumAxisHop,
    requiredLadderPasses,
    surplusLadderPasses: plan.strides.length - requiredLadderPasses,
    coveredByEncodedLadder: plan.encodedReach >= maximumAxisHop,
    idealLadderPasses: fineFloodDescendingPassesForReach(maximumAxisHop),
    cumulativeResolvedShare: Object.freeze(cumulativeResolvedShare),
  };
}

/* ------------------------------------------------------------------------- */
/* Visualization: how far the flood reached to seed one leaf.                 */
/* ------------------------------------------------------------------------- */

/**
 * What the reach decorator needs from whatever selected a region: the leaf, the
 * lattice it sits in, and the deepest hop observed among the samples inside it.
 * Structural, so the flood model does not have to know about cell traces.
 */
export interface FineFloodReachSubject {
  readonly dimensions: readonly [number, number, number];
  readonly leafOrigin: readonly [number, number, number];
  readonly leafSize: number;
  readonly fineSamples: number;
  readonly fineMaximumHop: number;
  /** Fine samples per finest cell along one axis; hops are counted in those. */
  readonly fineFactor: number;
}

const REACH_SWATCH = "#ff168f";

function acceptsFineFloodReach(subject: unknown): subject is FineFloodReachSubject {
  if (!hasFields(subject, [
    "dimensions", "leafOrigin", "leafSize", "fineSamples", "fineMaximumHop", "fineFactor",
  ])) return false;
  const candidate = subject as FineFloodReachSubject;
  return candidate.leafSize >= 1
    && candidate.fineSamples > 0
    && candidate.fineMaximumHop > 0
    && candidate.fineFactor >= 1
    && candidate.dimensions.every((extent) => Number.isFinite(extent) && extent >= 1);
}

/**
 * The flood's reach around one leaf, as a bound rather than a path.
 *
 * The deepest hop among the samples inside the leaf dilates the leaf by that
 * many finest cells, so the box contains every seed those samples inherited.
 * Dashed, because no single sample walked it: it is the envelope of many, and
 * drawing it solid would claim a trajectory that was never measured.
 */
export const fineFloodVisualizations: readonly Visualization[] = Object.freeze([
  decorationVisualization<FineFloodReachSubject>({
    kind: "decoration",
    id: "fine-band/flood-reach",
    pass: "Fine band",
    group: "fine",
    label: "Flood reach",
    swatch: REACH_SWATCH,
    description: "How far the jump-flood had to look to seed the samples inside this leaf — the envelope of their closest points, not any one sample's route.",
    source: "Closest-point seeds left resident by the redistance commit",
    legend: [
      { mark: "dashed-box", swatch: REACH_SWATCH, label: "deepest hop observed (bound)" },
    ],
    accepts: acceptsFineFloodReach,
    key: (subject) => `${subject.leafOrigin.join("_")}.${subject.leafSize}.${subject.fineMaximumHop}.${subject.fineFactor}`,
    build(subject, _context, into) {
      const reach = subject.fineMaximumHop / subject.fineFactor;
      const { leafOrigin: origin, leafSize: size, dimensions } = subject;
      into.box(
        [
          Math.max(0, origin[0] - reach),
          Math.max(0, origin[1] - reach),
          Math.max(0, origin[2] - reach),
        ],
        [
          Math.min(dimensions[0], origin[0] + size + reach),
          Math.min(dimensions[1], origin[1] + size + reach),
          Math.min(dimensions[2], origin[2] + size + reach),
        ],
        {
          colorLinear: linearFromHex(REACH_SWATCH),
          width_px: 1.3, intensity: 0.8, dash_m: into.cellSize_m * 0.9,
        },
      );
      return [{
        label: "Track the surface", evidence: "gathered",
        value: `${subject.fineMaximumHop} fine cells`,
        detail: `the deepest hop inside this leaf reaches ${reach.toFixed(2)} finest cells beyond it`,
      }];
    },
  }),
]);
