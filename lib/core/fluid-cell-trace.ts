/**
 * Cell-trace ABI: what one selected pressure cell had done to it, and the host
 * decode that turns it into a HUD narrative.
 *
 * This is the fluid counterpart of `svo-pixel-trace.ts`, and the difference in
 * mechanism is the whole design. A pixel is one ray is one sequential shader
 * invocation, so the pixel trace re-runs the shipping traversal law and records
 * work in execution order. A pressure cell has no such property: its work is
 * spread across every pass of the frame and repeated once per smoother sweep,
 * per V-cycle level, per outer iteration. Nothing can replay it.
 *
 * So a cell trace is assembled from two sources that together are exact:
 *
 * 1. **Gathered state** — what the frame actually published about this cell.
 *    Its row identity, leaf size, operator diagonal and right-hand side, the
 *    eighteen power-stencil neighbours it couples to and their leaf sizes, and
 *    the fine-band samples inside it with the flood hop that resolved each.
 *    Written by `webgpu-fluid-cell-trace.ts`; this file is the layout contract.
 *
 * 2. **Scheduled work** — how many times the encoded solve touches a row, from
 *    `fluid-blast-radius.ts`. That is a property of the command graph, not of
 *    the cell, so it is derived on the host rather than counted on the GPU.
 *
 * The narrative below is explicit about which of the two every line comes from.
 * A count that was scheduled rather than observed is labelled as such, because
 * an encoded iteration whose residual gate zeroed the tail did not run.
 */

export const FLUID_CELL_TRACE_ABI_VERSION = 2;
/** "FCT" plus the ABI version, so a stale mapped buffer is never decoded. */
export const FLUID_CELL_TRACE_MAGIC = 0x46435402;
export const FLUID_CELL_TRACE_HEADER_WORDS = 48;
export const FLUID_CELL_TRACE_RECORD_WORDS = 10;
/** Six faces and twelve edges: the power stencil is the only record kind. */
export const FLUID_CELL_TRACE_NEIGHBOR_CAPACITY = 18;

/**
 * Distinct leaves the ray passes through, nearest first.
 *
 * A pick is only as useful as the cells it can reach, and the first owned leaf
 * along a ray is almost always a surface cell — so selecting by pixel alone can
 * never name an interior unknown. Recording the whole ordered run lets the host
 * step along it. Twenty-four is generous for a camera crossing a domain of leaves
 * at mixed sizes; a longer run is truncated and says so rather than silently
 * dropping its tail.
 */
export const FLUID_CELL_TRACE_HIT_CAPACITY = 24;
export const FLUID_CELL_TRACE_HIT_WORDS = 7;

/**
 * What each leaf along the ray is, so the run can be navigated rather than only
 * stepped through.
 *
 * A pixel can only ever name the nearest leaf, which on a liquid is a surface
 * cell, and `[`/`]` walk inward one at a time with no idea what is coming. The
 * interesting cell in a domain of thousands is almost always the one the
 * interface passes through, so the march records which leaves those are while
 * it already has the coarse record in hand.
 */
export const FLUID_CELL_TRACE_HIT_FLAGS = Object.freeze({
  /** This leaf's coarse record holds phi of both signs: the surface is inside. */
  interface: 1 << 0,
  /** Its centre phi is negative — the leaf's site sits in the liquid. */
  liquid: 1 << 1,
  /** It carries a published coarse record at all. */
  corrected: 1 << 2,
} as const);

/**
 * Fine-band probes recorded individually, rather than only counted.
 *
 * The statistics still come from the full 8³ probe lattice the gather walks.
 * These records are the strict every-other-one sub-lattice of it — 4³ = 64 —
 * because line work needs positions, not totals, and 512 links inside one leaf
 * is clutter rather than a picture. Being a regular sub-lattice of the same
 * lattice is what keeps the drawn sample unbiased with respect to the counted
 * one; taking "the first 64 that resolved" would quietly favour whichever
 * corner the loop starts in.
 */
export const FLUID_CELL_TRACE_FINE_RECORD_EDGE = 4;
export const FLUID_CELL_TRACE_FINE_RECORD_CAPACITY = FLUID_CELL_TRACE_FINE_RECORD_EDGE ** 3;
export const FLUID_CELL_TRACE_FINE_RECORD_WORDS = 10;

export const FLUID_CELL_TRACE_WORDS =
  FLUID_CELL_TRACE_HEADER_WORDS
  + FLUID_CELL_TRACE_NEIGHBOR_CAPACITY * FLUID_CELL_TRACE_RECORD_WORDS
  + FLUID_CELL_TRACE_HIT_CAPACITY * FLUID_CELL_TRACE_HIT_WORDS
  + FLUID_CELL_TRACE_FINE_RECORD_CAPACITY * FLUID_CELL_TRACE_FINE_RECORD_WORDS;

/** First word of the ray run, after the header and the neighbour records. */
export const FLUID_CELL_TRACE_HITS_OFFSET =
  FLUID_CELL_TRACE_HEADER_WORDS + FLUID_CELL_TRACE_NEIGHBOR_CAPACITY * FLUID_CELL_TRACE_RECORD_WORDS;

/** First word of the fine-probe records, after the ray run. */
export const FLUID_CELL_TRACE_FINE_RECORDS_OFFSET =
  FLUID_CELL_TRACE_HITS_OFFSET + FLUID_CELL_TRACE_HIT_CAPACITY * FLUID_CELL_TRACE_HIT_WORDS;

/** Ray-run record word offsets, relative to the record's base. */
export const FLUID_CELL_TRACE_HIT = Object.freeze({
  row: 0,
  leafSize: 1,
  /** Lower corner of the leaf in finest cells; three words. */
  leafOrigin: 2,
  /** f32: distance along the ray in metres, for ordering and for the HUD. */
  distance_m: 5,
  /** `FLUID_CELL_TRACE_HIT_FLAGS`, so the run can be navigated by content. */
  flags: 6,
} as const);

export const FLUID_CELL_TRACE_STATUS = Object.freeze({
  /** No gather has run for this request yet. */
  pending: 0,
  /** The ray found an owned cell and the gather completed. */
  resolved: 1,
  /** The ray crossed the domain without meeting an owned pressure cell. */
  miss: 2,
  /** Topology or publication was incomplete, so no cell could be named. */
  unavailable: 3,
  /** A gathered field failed a validity check; the prefix is still exact. */
  invalid: 4,
} as const);

export type FluidCellTraceStatus =
  typeof FLUID_CELL_TRACE_STATUS[keyof typeof FLUID_CELL_TRACE_STATUS];

/**
 * Header word offsets. 0..7 identify the request, 8..19 describe the row, and
 * 20.. summarise the fine band inside it.
 */
export const FLUID_CELL_TRACE_HEADER = Object.freeze({
  magic: 0,
  status: 1,
  pixelX: 2,
  pixelY: 3,
  requestToken: 4,
  neighborCount: 5,
  /** Finest-grid cell the ray resolved to; three words. */
  cell: 6,
  /** Compact pressure row owning that cell. */
  row: 9,
  /** Leaf width in finest cells: 1, 2, 4, 8, 16 or 32. */
  leafSize: 10,
  /** Lower corner of the leaf in finest cells; three words. */
  leafOrigin: 11,
  /** f32: operator diagonal for this row. */
  diagonal: 14,
  /** f32: right-hand side for this row. */
  rhs: 15,
  /** Stencil entries the assembled row actually carries. */
  entryCount: 16,
  /** f32: power-cell volume in leaf-relative units. */
  volume: 17,
  /** Packed 18-bit same-or-finer power descriptor. */
  topologyCode: 18,
  /** f32: current pressure potential. */
  pressure: 19,
  /** Fine-band samples resident inside this leaf. */
  fineSamples: 20,
  /** Of those, samples the flood resolved to a seed. */
  fineResolved: 21,
  /** Largest Chebyshev flood hop among them, in fine cells. */
  fineMaximumHop: 22,
  /** Samples inside this leaf that lie on the interface. */
  fineInterface: 23,
  /**
   * Finest-grid extent; three words. Carried in the trace so the host can
   * derive the solve schedule from the trace alone rather than having to pair
   * it with a scene description that may already have moved on.
   */
  dimensions: 24,
  /**
   * Fine samples per finest cell along one axis, or 0 with no fine band. The
   * hop counts above are in fine-lattice cells, so this is what converts them
   * into the finest-grid units the overlay draws in.
   */
  fineFactor: 27,
  /** Distinct leaves the ray crossed, capped at the hit capacity. */
  hitCount: 28,
  /** Which of them this gather described; the rest are outlines only. */
  hitIndex: 29,
  /** Leaves the ray crossed beyond the capacity, so the HUD can say "24+". */
  hitOverflow: 30,
  /**
   * Probes the gather walked. Published rather than assumed so the HUD states
   * the denominator it actually sampled instead of a constant that a later
   * lattice change would silently falsify.
   */
  fineProbes: 31,
  /** Probes whose brick had no resident page at all. */
  fineMissing: 32,
  /** Probes whose page was resident but at a different generation. */
  fineStale: 33,
  /** Of the valid probes, those inside the liquid (phi < 0). */
  fineNegative: 34,
  /** Probe records actually written, at most the record capacity. */
  fineRecordCount: 35,
  /* --- The Section 5 correction as this row received it. --- */
  /**
   * f32: coarse phi at this leaf's centre — the value the ghost-fluid free
   * surface condition is built from, and the one number the whole fine band
   * exists to deliver to this row.
   */
  coarsePhi: 36,
  /** f32: smallest coarse phi over the leaf. */
  coarsePhiMinimum: 37,
  /** f32: largest coarse phi over the leaf. */
  coarsePhiMaximum: 38,
  /** Publication flags on the coarse record; zero means never corrected. */
  coarsePhiFlags: 39,
  /** f32: smallest phi among the probes, in fine-lattice distance units. */
  probeMinimumPhi: 40,
  /** f32: largest phi among the probes. */
  probeMaximumPhi: 41,
  /** f32: smallest |phi| among the probes — how near the surface comes. */
  probeNearestPhi: 42,
} as const);

/** Neighbour record word offsets, relative to the record's base. */
export const FLUID_CELL_TRACE_RECORD = Object.freeze({
  /** Index into `FLUID_CELL_TRACE_DIRECTIONS`. */
  direction: 0,
  row: 1,
  leafSize: 2,
  flags: 3,
  /** Lower corner of the neighbour leaf in finest cells; three words. */
  leafOrigin: 4,
  /** f32: the neighbour's current pressure potential. */
  pressure: 7,
  /**
   * f32: the neighbour's coarse phi. Carried per neighbour because the
   * ghost-fluid crossing is a property of the *pair* — it is the sign change
   * along one dual edge, and neither cell alone can name it.
   */
  phi: 8,
  /** Publication flags on that neighbour's coarse record. */
  phiFlags: 9,
} as const);

/**
 * Fine-probe record word offsets, relative to the record's base.
 *
 * The seed's cell and code travel with the sample rather than being looked up
 * later, because the host has no way to address the fine lattice: page
 * residency is a GPU-side hash and the flags carrying the closest-point code
 * live beside it. Everything needed to reconstruct one closest point is here.
 */
export const FLUID_CELL_TRACE_FINE_RECORD = Object.freeze({
  /** Fine-lattice coordinate of the probe; three words. Valid even when absent. */
  cell: 0,
  flags: 3,
  /** f32: the sample's phi, in fine-lattice distance units. */
  phi: 4,
  /** Fine-lattice coordinate of the seed this sample inherited; three words. */
  seedCell: 5,
  /** Packed closest-point code of that seed, as the redistance left it. */
  seedCode: 8,
  /** Chebyshev hop from the sample to its seed, in fine cells. */
  hop: 9,
} as const);

export const FLUID_CELL_TRACE_FINE_FLAGS = Object.freeze({
  /** A page for this probe's brick was resident at the published generation. */
  resident: 1 << 0,
  /** The sample carries a distance this generation. */
  valid: 1 << 1,
  /** The sample sits on the interface — it held its own zero crossing. */
  interface: 1 << 2,
  /** phi is negative here: this probe is inside the liquid. */
  negative: 1 << 3,
  /** The flood resolved this sample to a seed. */
  resolved: 1 << 4,
  /** A page was found but published a different generation. */
  stale: 1 << 5,
  /**
   * The gather visited this slot.
   *
   * Set on every written record so that "the band had nothing here" — no
   * residency, no validity, every other bit clear — is still distinguishable
   * from a slot the walk never reached. Absence is the finding for the gaps
   * layer, so it cannot be encoded as all-zero.
   */
  probed: 1 << 6,
} as const);

export const FLUID_CELL_TRACE_RECORD_FLAGS = Object.freeze({
  /** The neighbour resolves to a live compact row. */
  present: 1 << 0,
  /** The neighbour leaf is wider than the selected one. */
  coarser: 1 << 1,
  /** The neighbour leaf is narrower than the selected one. */
  finer: 1 << 2,
  /** The direction leaves the domain. */
  boundary: 1 << 3,
} as const);

/**
 * The eighteen power-diagram neighbour directions, in the operator's own order:
 * six faces then twelve edges. Kept here so the HUD, the gather shader and the
 * tests all name a direction the same way.
 */
export const FLUID_CELL_TRACE_DIRECTIONS: readonly (readonly [number, number, number])[] =
  Object.freeze([
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0], [1, 0, 1], [1, 0, -1],
    [-1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
  ].map((direction) => Object.freeze(direction as [number, number, number])));

export function fluidCellTraceDirectionLabel(direction: number): string {
  const delta = FLUID_CELL_TRACE_DIRECTIONS[direction];
  if (!delta) return "?";
  const axes = ["x", "y", "z"];
  return delta
    .map((component, axis) => (component === 0 ? "" : `${component > 0 ? "+" : "−"}${axes[axis]}`))
    .filter((part) => part !== "")
    .join("");
}

export type FluidCellTraceVec3 = readonly [number, number, number];

export interface FluidCellTraceNeighbor {
  readonly direction: number;
  readonly row: number;
  readonly leafSize: number;
  readonly flags: number;
  readonly leafOrigin: FluidCellTraceVec3;
  readonly pressure: number;
  readonly present: boolean;
  readonly coarser: boolean;
  readonly finer: boolean;
  readonly boundary: boolean;
  /** The neighbour's coarse phi, or undefined when its record is unpublished. */
  readonly phi?: number;
}

/** One fine-band probe, as the gather found it. */
export interface FluidCellTraceFineProbe {
  /** Fine-lattice coordinate, meaningful whether or not a page was resident. */
  readonly cell: FluidCellTraceVec3;
  readonly flags: number;
  /** Only meaningful when `valid`; the sample's signed distance. */
  readonly phi: number;
  readonly seedCell: FluidCellTraceVec3;
  readonly seedCode: number;
  readonly hop: number;
  readonly resident: boolean;
  readonly valid: boolean;
  readonly interface: boolean;
  readonly negative: boolean;
  readonly resolved: boolean;
  readonly stale: boolean;
  /** No page at all covered this probe — the band does not reach here. */
  readonly missing: boolean;
}

/** One leaf the ray crossed, whether or not it is the selected one. */
export interface FluidCellTraceHit {
  readonly row: number;
  readonly leafSize: number;
  readonly leafOrigin: FluidCellTraceVec3;
  readonly distance_m: number;
  /** True for the one this trace's gathered fields describe. */
  readonly selected: boolean;
  readonly flags: number;
  /** The surface passes through this leaf — usually the one worth selecting. */
  readonly holdsInterface: boolean;
  readonly liquid: boolean;
  readonly corrected: boolean;
}

export interface FluidCellTrace {
  readonly version: number;
  readonly status: FluidCellTraceStatus;
  readonly pixel: readonly [number, number];
  readonly requestToken: number;
  readonly cell: FluidCellTraceVec3;
  readonly row: number;
  readonly leafSize: number;
  readonly leafOrigin: FluidCellTraceVec3;
  readonly diagonal: number;
  readonly rhs: number;
  readonly entryCount: number;
  readonly volume: number;
  readonly topologyCode: number;
  readonly pressure: number;
  readonly fineSamples: number;
  readonly fineResolved: number;
  readonly fineMaximumHop: number;
  readonly fineInterface: number;
  readonly dimensions: FluidCellTraceVec3;
  /** Fine samples per finest cell along one axis; 0 when no fine band exists. */
  readonly fineFactor: number;
  /** Probes the gather walked — the denominator every fine count is out of. */
  readonly fineProbes: number;
  /** Probes whose brick had no resident page. */
  readonly fineMissing: number;
  /** Probes whose page was resident at another generation. */
  readonly fineStale: number;
  /** Valid probes inside the liquid. */
  readonly fineNegative: number;
  /** Coarse phi at the leaf centre — what the free-surface condition reads. */
  readonly coarsePhi: number;
  readonly coarsePhiMinimum: number;
  readonly coarsePhiMaximum: number;
  /** Zero means this row was never corrected by the fine band. */
  readonly coarsePhiFlags: number;
  readonly probeMinimumPhi: number;
  readonly probeMaximumPhi: number;
  readonly probeNearestPhi: number;
  /** The recorded sub-lattice; positions rather than totals. */
  readonly fineProbeRecords: readonly FluidCellTraceFineProbe[];
  readonly neighbors: readonly FluidCellTraceNeighbor[];
  /** Every leaf under the pointer, nearest first — the run the host steps along. */
  readonly hits: readonly FluidCellTraceHit[];
  /** Index within `hits` this gather described. */
  readonly hitIndex: number;
  /** Leaves past the capacity, so a long run reads as truncated not complete. */
  readonly hitOverflow: number;
}

const flagged = (flags: number, bit: number) => (flags & bit) !== 0;

export function decodeFluidCellTrace(words: ArrayLike<number>): FluidCellTrace | undefined {
  if (words.length < FLUID_CELL_TRACE_HEADER_WORDS) return undefined;
  const unsigned = words instanceof Uint32Array ? words : Uint32Array.from(words);
  if (unsigned[FLUID_CELL_TRACE_HEADER.magic] !== FLUID_CELL_TRACE_MAGIC) return undefined;
  const floats = new Float32Array(unsigned.buffer, unsigned.byteOffset, unsigned.length);
  const vec3 = (offset: number): FluidCellTraceVec3 =>
    Object.freeze([unsigned[offset], unsigned[offset + 1], unsigned[offset + 2]]) as FluidCellTraceVec3;

  const neighborCount = Math.min(
    unsigned[FLUID_CELL_TRACE_HEADER.neighborCount], FLUID_CELL_TRACE_NEIGHBOR_CAPACITY);
  const neighbors: FluidCellTraceNeighbor[] = [];
  for (let index = 0; index < neighborCount; index += 1) {
    const base = FLUID_CELL_TRACE_HEADER_WORDS + index * FLUID_CELL_TRACE_RECORD_WORDS;
    if (base + FLUID_CELL_TRACE_RECORD_WORDS > unsigned.length) break;
    const flags = unsigned[base + FLUID_CELL_TRACE_RECORD.flags];
    neighbors.push({
      direction: unsigned[base + FLUID_CELL_TRACE_RECORD.direction],
      row: unsigned[base + FLUID_CELL_TRACE_RECORD.row],
      leafSize: unsigned[base + FLUID_CELL_TRACE_RECORD.leafSize],
      flags,
      leafOrigin: vec3(base + FLUID_CELL_TRACE_RECORD.leafOrigin),
      pressure: floats[base + FLUID_CELL_TRACE_RECORD.pressure],
      present: flagged(flags, FLUID_CELL_TRACE_RECORD_FLAGS.present),
      coarser: flagged(flags, FLUID_CELL_TRACE_RECORD_FLAGS.coarser),
      finer: flagged(flags, FLUID_CELL_TRACE_RECORD_FLAGS.finer),
      boundary: flagged(flags, FLUID_CELL_TRACE_RECORD_FLAGS.boundary),
      // Absent rather than zero when the record is unpublished: zero is a
      // perfectly good phi, and a crossing computed against a fabricated zero
      // would put the free surface exactly on a neighbour that has no opinion.
      ...(unsigned[base + FLUID_CELL_TRACE_RECORD.phiFlags] !== 0
        ? { phi: floats[base + FLUID_CELL_TRACE_RECORD.phi] } : {}),
    });
  }

  const fineRecordCount = Math.min(
    unsigned[FLUID_CELL_TRACE_HEADER.fineRecordCount], FLUID_CELL_TRACE_FINE_RECORD_CAPACITY);
  const fineProbeRecords: FluidCellTraceFineProbe[] = [];
  for (let index = 0; index < fineRecordCount; index += 1) {
    const base = FLUID_CELL_TRACE_FINE_RECORDS_OFFSET + index * FLUID_CELL_TRACE_FINE_RECORD_WORDS;
    if (base + FLUID_CELL_TRACE_FINE_RECORD_WORDS > unsigned.length) break;
    const flags = unsigned[base + FLUID_CELL_TRACE_FINE_RECORD.flags];
    // Slots the gather never reached stay zeroed, and a zeroed slot is not a
    // probe at the origin — it is no probe at all.
    if (flags === 0) continue;
    fineProbeRecords.push({
      cell: vec3(base + FLUID_CELL_TRACE_FINE_RECORD.cell),
      flags,
      phi: floats[base + FLUID_CELL_TRACE_FINE_RECORD.phi],
      seedCell: vec3(base + FLUID_CELL_TRACE_FINE_RECORD.seedCell),
      seedCode: unsigned[base + FLUID_CELL_TRACE_FINE_RECORD.seedCode],
      hop: unsigned[base + FLUID_CELL_TRACE_FINE_RECORD.hop],
      resident: flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.resident),
      valid: flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.valid),
      interface: flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.interface),
      negative: flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.negative),
      resolved: flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.resolved),
      stale: flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.stale),
      missing: !flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.resident)
        && !flagged(flags, FLUID_CELL_TRACE_FINE_FLAGS.stale),
    });
  }

  const hitIndex = unsigned[FLUID_CELL_TRACE_HEADER.hitIndex];
  const hitCount = Math.min(
    unsigned[FLUID_CELL_TRACE_HEADER.hitCount], FLUID_CELL_TRACE_HIT_CAPACITY);
  const hits: FluidCellTraceHit[] = [];
  for (let index = 0; index < hitCount; index += 1) {
    const base = FLUID_CELL_TRACE_HITS_OFFSET + index * FLUID_CELL_TRACE_HIT_WORDS;
    if (base + FLUID_CELL_TRACE_HIT_WORDS > unsigned.length) break;
    const hitFlags = unsigned[base + FLUID_CELL_TRACE_HIT.flags];
    hits.push({
      row: unsigned[base + FLUID_CELL_TRACE_HIT.row],
      leafSize: unsigned[base + FLUID_CELL_TRACE_HIT.leafSize],
      leafOrigin: vec3(base + FLUID_CELL_TRACE_HIT.leafOrigin),
      distance_m: floats[base + FLUID_CELL_TRACE_HIT.distance_m],
      selected: index === hitIndex,
      flags: hitFlags,
      holdsInterface: flagged(hitFlags, FLUID_CELL_TRACE_HIT_FLAGS.interface),
      liquid: flagged(hitFlags, FLUID_CELL_TRACE_HIT_FLAGS.liquid),
      corrected: flagged(hitFlags, FLUID_CELL_TRACE_HIT_FLAGS.corrected),
    });
  }

  return {
    version: FLUID_CELL_TRACE_ABI_VERSION,
    status: unsigned[FLUID_CELL_TRACE_HEADER.status] as FluidCellTraceStatus,
    pixel: Object.freeze([
      unsigned[FLUID_CELL_TRACE_HEADER.pixelX], unsigned[FLUID_CELL_TRACE_HEADER.pixelY],
    ]) as readonly [number, number],
    requestToken: unsigned[FLUID_CELL_TRACE_HEADER.requestToken],
    cell: vec3(FLUID_CELL_TRACE_HEADER.cell),
    row: unsigned[FLUID_CELL_TRACE_HEADER.row],
    leafSize: unsigned[FLUID_CELL_TRACE_HEADER.leafSize],
    leafOrigin: vec3(FLUID_CELL_TRACE_HEADER.leafOrigin),
    diagonal: floats[FLUID_CELL_TRACE_HEADER.diagonal],
    rhs: floats[FLUID_CELL_TRACE_HEADER.rhs],
    entryCount: unsigned[FLUID_CELL_TRACE_HEADER.entryCount],
    volume: floats[FLUID_CELL_TRACE_HEADER.volume],
    topologyCode: unsigned[FLUID_CELL_TRACE_HEADER.topologyCode],
    pressure: floats[FLUID_CELL_TRACE_HEADER.pressure],
    fineSamples: unsigned[FLUID_CELL_TRACE_HEADER.fineSamples],
    fineResolved: unsigned[FLUID_CELL_TRACE_HEADER.fineResolved],
    fineMaximumHop: unsigned[FLUID_CELL_TRACE_HEADER.fineMaximumHop],
    fineInterface: unsigned[FLUID_CELL_TRACE_HEADER.fineInterface],
    dimensions: vec3(FLUID_CELL_TRACE_HEADER.dimensions),
    fineFactor: unsigned[FLUID_CELL_TRACE_HEADER.fineFactor],
    fineProbes: unsigned[FLUID_CELL_TRACE_HEADER.fineProbes],
    fineMissing: unsigned[FLUID_CELL_TRACE_HEADER.fineMissing],
    fineStale: unsigned[FLUID_CELL_TRACE_HEADER.fineStale],
    fineNegative: unsigned[FLUID_CELL_TRACE_HEADER.fineNegative],
    coarsePhi: floats[FLUID_CELL_TRACE_HEADER.coarsePhi],
    coarsePhiMinimum: floats[FLUID_CELL_TRACE_HEADER.coarsePhiMinimum],
    coarsePhiMaximum: floats[FLUID_CELL_TRACE_HEADER.coarsePhiMaximum],
    coarsePhiFlags: unsigned[FLUID_CELL_TRACE_HEADER.coarsePhiFlags],
    probeMinimumPhi: floats[FLUID_CELL_TRACE_HEADER.probeMinimumPhi],
    probeMaximumPhi: floats[FLUID_CELL_TRACE_HEADER.probeMaximumPhi],
    probeNearestPhi: floats[FLUID_CELL_TRACE_HEADER.probeNearestPhi],
    fineProbeRecords: Object.freeze(fineProbeRecords),
    neighbors: Object.freeze(neighbors),
    hits: Object.freeze(hits),
    hitIndex,
    hitOverflow: unsigned[FLUID_CELL_TRACE_HEADER.hitOverflow],
  };
}

/**
 * Clamp a requested step along the ray to the run that actually exists.
 *
 * The host holds the index across frames while the ray keeps moving under it, so
 * a run that shortens must not strand the selection past its end. Wrapping
 * rather than saturating keeps `]` on the last hit useful — it returns to the
 * surface cell instead of doing nothing.
 */
export function stepFluidCellTraceHit(
  index: number, delta: number, hitCount: number,
): number {
  if (hitCount <= 0) return 0;
  return ((index + delta) % hitCount + hitCount) % hitCount;
}

/**
 * The next leaf along the run whose coarse record holds the interface.
 *
 * Stepping one leaf at a time is the wrong instrument for finding the cell
 * worth looking at: a ray crossing a dam break passes a dozen interior cells
 * before it reaches anything the surface touches. Searching forward with wrap
 * keeps this a single repeatable gesture rather than a mode — press it again
 * and you get the next one, and the run cycles rather than dead-ending.
 *
 * Returns the current index unchanged when no leaf on the run holds the
 * interface, so the caller can disable the control rather than move the
 * selection somewhere arbitrary.
 */
export function nextFluidCellTraceInterfaceHit(
  hits: readonly FluidCellTraceHit[], index: number,
): number {
  if (hits.length === 0) return 0;
  for (let step = 1; step <= hits.length; step += 1) {
    const candidate = (index + step) % hits.length;
    if (hits[candidate].holdsInterface) return candidate;
  }
  return index;
}

/** Whether any leaf on the run is worth jumping to. */
export function fluidCellTraceHasInterfaceHit(hits: readonly FluidCellTraceHit[]): boolean {
  return hits.some((hit) => hit.holdsInterface);
}

/* ------------------------------------------------------------------------- */
/* Layers: what the 3D overlay may draw, and what the HUD may filter.        */
/* ------------------------------------------------------------------------- */

export const FLUID_CELL_TRACE_LAYERS = [
  "ray", "cell", "stencil", "transition", "surface", "cone",
  "fine", "patch", "links", "gaps",
] as const;
export type FluidCellTraceLayer = typeof FLUID_CELL_TRACE_LAYERS[number];

export const FLUID_CELL_TRACE_LAYER_DEFINITIONS: Readonly<Record<FluidCellTraceLayer, {
  readonly label: string;
  readonly swatch: `#${string}`;
  readonly description: string;
  /** Overlay line width in device pixels, before any width scale. */
  readonly width_px: number;
}>> = Object.freeze({
  ray: {
    label: "Ray run", swatch: "#8a7ad6", width_px: 1.1,
    description: "Every live leaf the pointer ray crosses, nearest first — the run the selection steps along.",
  },
  cell: {
    label: "Cell", swatch: "#ffffff", width_px: 2.4,
    description: "The selected power cell and the leaf that owns it.",
  },
  stencil: {
    label: "Stencil", swatch: "#0fc7cc", width_px: 1.6,
    description: "The eighteen power-diagram neighbours this row couples to.",
  },
  transition: {
    label: "Transitions", swatch: "#fa9e14", width_px: 2.0,
    description: "Neighbours at a different leaf size — the T-junction cases the power diagram exists to handle.",
  },
  cone: {
    label: "Cone", swatch: "#1a47eb", width_px: 1.2,
    description: "Hop distance outward through the level-0 stencil.",
  },
  fine: {
    label: "Flood reach", swatch: "#ff168f", width_px: 1.3,
    description: "How far the jump-flood looked to seed the samples inside this leaf — the envelope of their closest points.",
  },
  // The four below are the fine band's own picture. They share the flood
  // reach's pass but not its colour, because each answers a different question
  // and a reader has to be able to switch one off without losing the others.
  surface: {
    label: "Free surface", swatch: "#f5ba1a", width_px: 2.2,
    description: "Where the free surface cuts each dual edge, and by how much that scales the face's operator coefficient.",
  },
  patch: {
    label: "Interface patch", swatch: "#f5f5e6", width_px: 1.8,
    description: "The surface as this cell's own fine samples reconstruct it — closest points, not a coarse classification.",
  },
  links: {
    label: "Closest points", swatch: "#6b54eb", width_px: 1.0,
    description: "Each recorded sample joined to the seed it inherited its distance from; the flood's dependency graph inside one leaf.",
  },
  gaps: {
    label: "Band gaps", swatch: "#ff1738", width_px: 1.2,
    description: "Bricks inside this leaf with no resident page, or a page at another generation — where the fine band could not answer.",
  },
});

export function fluidCellTraceLayerForNeighbor(neighbor: FluidCellTraceNeighbor): FluidCellTraceLayer {
  return neighbor.coarser || neighbor.finer ? "transition" : "stencil";
}

/* ------------------------------------------------------------------------- */
/* Narrative: the ordered story of one cell, for the HUD.                     */
/* ------------------------------------------------------------------------- */

export interface FluidCellTraceNarrativeStep {
  readonly id: string;
  readonly label: string;
  readonly detail: string;
  readonly value: string;
  readonly layer?: FluidCellTraceLayer;
  /**
   * Where the number comes from. `gathered` was published by the frame;
   * `scheduled` is what the command graph encodes, which a residual gate may
   * not have run. Never blend the two into one figure.
   */
  readonly evidence: "gathered" | "scheduled";
}

export interface FluidCellTraceSchedule {
  /** Encoded outer iterations for this scene. */
  readonly outerIterations: number;
  /** Levels in the SPGrid pyramid, down to the exact one-cell bottom. */
  readonly levels: number;
  /** Level-0 sweeps the encoded schedule spends: operator plus smoother. */
  readonly fineGridSweeps: number;
  /** Stages until this cell's dependency cone covers the domain. */
  readonly stagesToGlobal?: number;
  /** Total information-moving stages in the encoded solve. */
  readonly stageCount: number;
}

const count = (value: number) => value.toLocaleString();

function pressureText(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 || (Math.abs(value) < 0.01 && value !== 0)
    ? value.toExponential(2) : value.toFixed(3);
}

/**
 * Row updates the encoded schedule performs on this cell.
 *
 * Every level-0 sweep touches the row once, and each touch reads the row's
 * stencil entries. This is the count that makes the pass census legible: a
 * single cell is revisited a hundred-odd times inside one advance.
 */
export function fluidCellTraceScheduledRowUpdates(schedule: FluidCellTraceSchedule): number {
  return schedule.fineGridSweeps;
}

/** Stencil reads the schedule performs on this cell's row. */
export function fluidCellTraceScheduledStencilReads(
  trace: FluidCellTrace, schedule: FluidCellTraceSchedule,
): number {
  return schedule.fineGridSweeps * trace.entryCount;
}

/** Human-readable account of what the solve does to one cell. */
export function fluidCellTraceNarrative(
  trace: FluidCellTrace, schedule: FluidCellTraceSchedule,
): readonly FluidCellTraceNarrativeStep[] {
  const present = trace.neighbors.filter((neighbor) => neighbor.present);
  const transitions = present.filter((neighbor) => neighbor.coarser || neighbor.finer);
  const boundaries = trace.neighbors.filter((neighbor) => neighbor.boundary);
  const coarser = transitions.filter((neighbor) => neighbor.coarser).length;
  const finer = transitions.filter((neighbor) => neighbor.finer).length;
  const rowUpdates = fluidCellTraceScheduledRowUpdates(schedule);

  const steps: FluidCellTraceNarrativeStep[] = [
    {
      id: "cell", label: "Own the cell", layer: "cell", evidence: "gathered",
      detail: `leaf ${trace.leafSize}³ finest cells at (${trace.leafOrigin.join(", ")}), row ${trace.row}`,
      value: `${trace.leafSize}³`,
    },
    {
      id: "operator", label: "Assemble the row", layer: "stencil", evidence: "gathered",
      detail: `diagonal ${pressureText(trace.diagonal)}, rhs ${pressureText(trace.rhs)}, volume ${pressureText(trace.volume)}`,
      value: `${trace.entryCount} entries`,
    },
    {
      id: "stencil", label: "Couple to neighbours", layer: "stencil", evidence: "gathered",
      detail: boundaries.length > 0
        ? `${present.length} live of 18 directions; ${boundaries.length} leave the domain`
        : `${present.length} live of 18 directions`,
      value: `${present.length} rows`,
    },
    {
      id: "transition", label: "Cross resolution", layer: "transition", evidence: "gathered",
      detail: transitions.length === 0
        ? "every neighbour is at this leaf's size, so the stencil is the regular Cartesian case"
        : `${coarser} coarser, ${finer} finer — the T-junctions the power diagram reconstructs`,
      value: `${transitions.length} of ${present.length}`,
    },
    {
      id: "updates", label: "Update the row", layer: "cone", evidence: "scheduled",
      detail: `${schedule.outerIterations} encoded outer iterations over a ${schedule.levels}-level pyramid; a residual gate may zero the tail`,
      value: `${count(rowUpdates)}×`,
    },
    {
      id: "reads", label: "Re-read the stencil", layer: "cone", evidence: "scheduled",
      detail: `every level-0 sweep re-reads all ${trace.entryCount} entries of this row`,
      value: `${count(fluidCellTraceScheduledStencilReads(trace, schedule))} reads`,
    },
    {
      id: "cone", label: "Reach the domain", layer: "cone", evidence: "scheduled",
      detail: schedule.stagesToGlobal === undefined
        ? `the cone stays local across all ${schedule.stageCount} information-moving stages`
        : `after ${schedule.stagesToGlobal} of ${schedule.stageCount} stages this cell depends on every other cell`,
      value: schedule.stagesToGlobal === undefined ? "local" : `${schedule.stagesToGlobal} stages`,
    },
  ];

  if (trace.fineSamples > 0) {
    steps.push({
      id: "fine", label: "Track the surface", layer: "fine", evidence: "gathered",
      detail: trace.fineInterface > 0
        ? `${trace.fineInterface} on the interface; deepest flood hop ${trace.fineMaximumHop} fine cells`
        : `no interface inside this leaf; deepest flood hop ${trace.fineMaximumHop} fine cells`,
      value: `${count(trace.fineResolved)} of ${count(trace.fineSamples)}`,
    });
  }
  return steps;
}

/**
 * One headline figure about the selected cell.
 *
 * Separate from `FluidCellTraceNarrativeStep` because it answers a different
 * question. The narrative is the ordered story of what happens to a cell and is
 * read once someone has decided to study it; these are what tell them whether
 * to. Same `evidence` discipline: a scheduled figure never merges with a
 * gathered one.
 */
export interface FluidCellTraceFigure {
  readonly id: string;
  /** Two or three words. This is read at a glance or not at all. */
  readonly label: string;
  readonly value: string;
  readonly detail: string;
  readonly layer?: FluidCellTraceLayer;
  readonly evidence: "gathered" | "scheduled";
  /** Something the reader should not miss — a gap in what the frame published. */
  readonly alert?: boolean;
}

/**
 * The four figures worth reading before anything else about a cell.
 *
 * Progressive disclosure needs a first page, and it has to be one that answers
 * "is this cell interesting" rather than one that starts explaining. So: how
 * much work it costs, how connected it is, how irregular its stencil is, and
 * where it sits relative to the surface. Everything else in the HUD is a detail
 * of one of those four, which is what makes them the right things to keep
 * visible when the rest is folded away.
 */
export function fluidCellTraceKeyFigures(
  trace: FluidCellTrace, schedule: FluidCellTraceSchedule,
): readonly FluidCellTraceFigure[] {
  const present = trace.neighbors.filter((neighbor) => neighbor.present);
  const transitions = present.filter((neighbor) => neighbor.coarser || neighbor.finer);
  const coarser = transitions.filter((neighbor) => neighbor.coarser).length;
  const corrected = trace.coarsePhiFlags !== 0;
  const straddles = corrected && trace.coarsePhiMinimum <= 0 && trace.coarsePhiMaximum >= 0;
  return [
    {
      id: "work", label: "Work", layer: "cone", evidence: "scheduled",
      value: count(fluidCellTraceTotalWork(trace, schedule)),
      detail: `${count(fluidCellTraceScheduledRowUpdates(schedule))} row updates and `
        + `${count(fluidCellTraceScheduledStencilReads(trace, schedule))} stencil reads across `
        + `${schedule.outerIterations} encoded outer iterations; a residual gate may zero the tail`,
    },
    {
      id: "couples", label: "Couples", layer: "stencil", evidence: "gathered",
      value: `${present.length}/18`,
      detail: `${trace.entryCount} assembled entries, diagonal ${pressureText(trace.diagonal)}`,
    },
    {
      id: "transitions", label: "T-junctions", layer: "transition", evidence: "gathered",
      value: transitions.length === 0 ? "none" : `${transitions.length}`,
      detail: transitions.length === 0
        ? "every neighbour is at this leaf's size — the regular Cartesian case"
        : `${coarser} coarser and ${transitions.length - coarser} finer neighbours, the cases the power diagram reconstructs`,
    },
    {
      id: "surface", label: "φ", layer: "surface", evidence: "gathered",
      alert: !corrected,
      value: corrected ? `${trace.coarsePhi.toFixed(2)}` : "—",
      detail: !corrected
        ? "this row carries no coarse level-set record, so the solve sees no free surface here"
        : straddles
          ? `the surface is inside this leaf — φ runs ${trace.coarsePhiMinimum.toFixed(2)} to ${trace.coarsePhiMaximum.toFixed(2)} cells across it`
          : `${Math.abs(trace.coarsePhi).toFixed(2)} cells ${trace.coarsePhi < 0 ? "inside the liquid" : "into the air"}`,
    },
  ];
}

/**
 * Total units of work attributable to this one cell in one advance.
 *
 * Deliberately the scheduled stencil reads plus row updates and nothing else:
 * it is the figure that answers "why is there so much work per cell", and
 * padding it with gathered state would make it uncomparable between cells.
 */
export function fluidCellTraceTotalWork(
  trace: FluidCellTrace, schedule: FluidCellTraceSchedule,
): number {
  return fluidCellTraceScheduledRowUpdates(schedule)
    + fluidCellTraceScheduledStencilReads(trace, schedule);
}

/**
 * The encoded solve schedule implied by a trace's own domain.
 *
 * Derived from the trace rather than from a scene description so the HUD cannot
 * pair a cell with a schedule for a domain that has since changed. The caller
 * supplies the policy inputs, because the tail policy lives with the solver.
 */
export function fluidCellTraceScheduleFor(input: {
  readonly dimensions: FluidCellTraceVec3;
  readonly outerIterations: number;
  readonly levels: number;
  readonly smoothsPerLevel: number;
  readonly stagesToGlobal?: number;
  readonly stageCount: number;
}): FluidCellTraceSchedule {
  return {
    outerIterations: input.outerIterations,
    levels: input.levels,
    // One operator application plus the pre- and post-smoothing sweeps at
    // level 0, repeated per outer iteration.
    fineGridSweeps: input.outerIterations * (1 + 2 * input.smoothsPerLevel),
    ...(input.stagesToGlobal === undefined ? {} : { stagesToGlobal: input.stagesToGlobal }),
    stageCount: input.stageCount,
  };
}
