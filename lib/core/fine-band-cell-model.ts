/**
 * What the Section 5 narrow band did to one pressure cell, as arithmetic.
 *
 * The fine SPGrid stores phi and nothing else; velocity and pressure stay on
 * the octree. So the band's whole job is to collapse `(leafSize x m)^3` signed
 * distances into the few numbers one pressure row needs — a centre phi for the
 * ghost-fluid free-surface condition, a liquid fraction for the finite-volume
 * `V_cell`, and a crossing per dual edge. Every function here is one step of
 * that collapse, read backwards from what the frame published.
 *
 * Deliberately free of WebGPU and of React: the same functions run in the HUD,
 * in the decorators and in tests. `fluid-cell-trace.ts` owns the layout; this
 * owns the meaning.
 *
 * ## Units
 *
 * Every phi in a trace is in **finest octree cells**, converted by the gather
 * from the physical metres the fine lattice stores. That is the unit the rest
 * of the cell trace already speaks — leaf sizes are in finest cells — so a
 * reader can compare "the surface is 0.6 cells from the site" against "this
 * leaf is 32 cells wide" without a conversion in their head. Hops stay in fine
 * cells, because that is the lattice the flood walks.
 */
import { FINE_LEVELSET_CHANNELS } from "./fine-levelset-brick-abi";
import {
  describeFineFloodLadder,
  fineFloodClosestPoint,
  fineFloodLadderPrefixForReach,
  type FineFloodLadderPlan,
} from "./fine-flood-provenance";
import {
  fluidCellTraceDirectionLabel,
  type FluidCellTrace,
  type FluidCellTraceFineProbe,
  type FluidCellTraceNarrativeStep,
  type FluidCellTraceNeighbor,
  type FluidCellTraceVec3,
} from "./fluid-cell-trace";

/** Samples along one brick edge, matching `FineLevelSetBrickResolution`. */
export const FINE_BAND_BRICK_RESOLUTION = 4;

/* ------------------------------------------------------------------------- */
/* The free surface, where it meets the operator.                            */
/* ------------------------------------------------------------------------- */

export interface FineBandCrossing {
  readonly direction: number;
  /** "+x", "-yz" — the same name the stencil strip uses. */
  readonly label: string;
  readonly ownPhi: number;
  readonly neighborPhi: number;
  /**
   * Where the surface cuts the dual edge, as a fraction from this cell's site
   * towards the neighbour's. Symmetric in the pair and always in (0, 1), which
   * is what makes it safe to draw from either side.
   */
  readonly fraction: number;
  /** True when this cell is the liquid one and the neighbour is air. */
  readonly outward: boolean;
  /**
   * How much the ghost-fluid condition scales this face's coefficient: the
   * dual distance shrinks to `theta` of itself, so `A/d` grows by `1/theta`.
   *
   * Only defined when this cell is the liquid side, because that is the only
   * side the condition is imposed from (Gibou et al., via section 4.1). A cell
   * sitting in air has no row to scale.
   */
  readonly coefficientScale?: number;
}

/**
 * Every dual edge of this cell that the free surface cuts.
 *
 * A crossing needs both phis, so a neighbour whose coarse record is
 * unpublished contributes nothing rather than being compared against a
 * fabricated zero — which would place the surface exactly on a cell that has
 * no opinion about where it is.
 */
export function fineBandFreeSurfaceCrossings(trace: FluidCellTrace): readonly FineBandCrossing[] {
  if (trace.coarsePhiFlags === 0 || !Number.isFinite(trace.coarsePhi)) return [];
  const own = trace.coarsePhi;
  const crossings: FineBandCrossing[] = [];
  for (const neighbor of trace.neighbors) {
    if (!neighbor.present || neighbor.phi === undefined) continue;
    const other = neighbor.phi;
    if (!Number.isFinite(other)) continue;
    // A shared zero is not a crossing: the surface is on the site, and the
    // fraction below would be 0/0.
    if ((own < 0) === (other < 0)) continue;
    const span = Math.abs(own) + Math.abs(other);
    if (span <= 0) continue;
    const fraction = Math.abs(own) / span;
    const outward = own < 0 && other >= 0;
    crossings.push({
      direction: neighbor.direction,
      label: fluidCellTraceDirectionLabel(neighbor.direction),
      ownPhi: own,
      neighborPhi: other,
      fraction,
      outward,
      ...(outward && fraction > 0 ? { coefficientScale: 1 / fraction } : {}),
    });
  }
  // Tightest first: the near-singular face is the one that explains a large
  // diagonal, and it is what the reader is looking for.
  return Object.freeze(crossings.sort((a, b) => a.fraction - b.fraction));
}

/** The largest coefficient scale any face of this row carries, if any does. */
export function fineBandTightestCrossing(
  crossings: readonly FineBandCrossing[],
): FineBandCrossing | undefined {
  return crossings.find((crossing) => crossing.coefficientScale !== undefined);
}

/* ------------------------------------------------------------------------- */
/* Phi across the cell: what the row received, and what it could not see.    */
/* ------------------------------------------------------------------------- */

export interface FineBandPhiSpan {
  /** False when this row carries no coarse record — never corrected at all. */
  readonly published: boolean;
  /** Coarse phi at the leaf centre; the value the free-surface condition reads. */
  readonly centre: number;
  readonly minimum: number;
  readonly maximum: number;
  /** Extremes over the probes, which sample the fine band directly. */
  readonly probeMinimum: number;
  readonly probeMaximum: number;
  /** Smallest |phi| any probe saw — how near the surface actually comes. */
  readonly probeNearest: number;
  /**
   * Half the leaf's diagonal, in finest cells. The natural full scale for a
   * bar: phi cannot vary by more than this inside the cell without the field
   * failing the Lipschitz condition redistancing exists to restore.
   */
  readonly scale: number;
  /** The probes hold samples on both sides of the surface. */
  readonly probesStraddle: boolean;
  /** The published record holds both signs. */
  readonly recordStraddles: boolean;
  /**
   * The probes found an interface the published record does not carry.
   *
   * This is the paper's own limitation made specific (section 8: "sub-grid
   * droplets or air pockets may at times be overlooked by the simulation").
   * The surface tracker resolved a feature and the correction dropped it, so
   * the pressure solve is about to run as though it were not there.
   */
  readonly unresolvedInterface: boolean;
  /**
   * The surface is inside this cell but not near its site: the record straddles
   * zero and the centre is further from it than half a finest cell. Not a loss
   * — the crossing is still carried by theta — but it is where second-order
   * accuracy near the surface is resting entirely on that one number.
   */
  readonly interfaceOffCentre: boolean;
}

/**
 * The span bar's data.
 *
 * Two spans rather than one, because they answer different questions and can
 * disagree: the record is what the pressure row will read, the probes are what
 * the fine band actually holds. `unresolvedInterface` is exactly their
 * disagreement in the direction that costs something.
 */
export function fineBandPhiSpan(trace: FluidCellTrace): FineBandPhiSpan {
  const published = trace.coarsePhiFlags !== 0 && Number.isFinite(trace.coarsePhi);
  const finite = (value: number, fallback: number) => Number.isFinite(value) ? value : fallback;
  const centre = finite(trace.coarsePhi, 0);
  const minimum = published ? finite(trace.coarsePhiMinimum, centre) : centre;
  const maximum = published ? finite(trace.coarsePhiMaximum, centre) : centre;
  const hasProbes = trace.fineSamples > 0;
  const probeMinimum = hasProbes ? finite(trace.probeMinimumPhi, 0) : 0;
  const probeMaximum = hasProbes ? finite(trace.probeMaximumPhi, 0) : 0;
  const probeNearest = hasProbes ? finite(trace.probeNearestPhi, 0) : 0;
  const probesStraddle = hasProbes && probeMinimum <= 0 && probeMaximum >= 0;
  const recordStraddles = published && minimum <= 0 && maximum >= 0;
  return {
    published, centre, minimum, maximum,
    probeMinimum, probeMaximum, probeNearest,
    scale: 0.5 * trace.leafSize * Math.sqrt(3),
    probesStraddle,
    recordStraddles,
    unresolvedInterface: probesStraddle && published && !recordStraddles,
    interfaceOffCentre: recordStraddles && Math.abs(centre) > 0.5,
  };
}

/* ------------------------------------------------------------------------- */
/* The probe census: what the band could and could not answer.               */
/* ------------------------------------------------------------------------- */

export interface FineBandCensus {
  /** Probes walked — the denominator every figure below is out of. */
  readonly probes: number;
  /** Probes whose brick had no resident page. */
  readonly missing: number;
  /** Probes whose page was resident at another generation. */
  readonly stale: number;
  /** Probes that reached a live sample carrying a distance. */
  readonly valid: number;
  /** Resident and current, but carrying no distance this generation. */
  readonly residentInvalid: number;
  readonly resolved: number;
  readonly onInterface: number;
  readonly negative: number;
  /** Probes recorded individually, for the line work. */
  readonly recorded: number;
  /** Share of the leaf the fine band could answer for, in [0, 1]. */
  readonly coverage: number;
  /** Share of the valid probes inside the liquid, in [0, 1]. */
  readonly liquidFraction: number;
}

export function fineBandCensus(trace: FluidCellTrace): FineBandCensus {
  const probes = Math.max(0, trace.fineProbes);
  const missing = Math.min(probes, trace.fineMissing);
  const stale = Math.min(probes - missing, trace.fineStale);
  const valid = Math.min(probes - missing - stale, trace.fineSamples);
  return {
    probes, missing, stale, valid,
    residentInvalid: Math.max(0, probes - missing - stale - valid),
    resolved: Math.min(valid, trace.fineResolved),
    onInterface: Math.min(valid, trace.fineInterface),
    negative: Math.min(valid, trace.fineNegative),
    recorded: trace.fineProbeRecords.length,
    coverage: probes > 0 ? valid / probes : 0,
    liquidFraction: valid > 0 ? Math.min(valid, trace.fineNegative) / valid : 0,
  };
}

/* ------------------------------------------------------------------------- */
/* The liquid ledger and the cost of one unknown.                            */
/* ------------------------------------------------------------------------- */

export interface FineBandVolumeLedger {
  /** Power-cell volume the operator carries, in leaf-relative units. */
  readonly operatorVolume: number;
  /** Share of the cell the fine band says is liquid. */
  readonly liquidFraction: number;
  /** That share of the operator's own volume. */
  readonly liquidVolume: number;
  /**
   * True when the cell is neither full nor empty. Only then does the difference
   * between the two figures mean anything: an interior cell agrees trivially.
   */
  readonly partial: boolean;
  /** Probes the fraction was measured over; zero makes the rest meaningless. */
  readonly samples: number;
}

/**
 * What the fine band implies about this cell's liquid content.
 *
 * Deliberately a fraction of the operator's own volume rather than an absolute:
 * `Metric.volume` is in leaf-relative units, and multiplying it by a probe
 * fraction keeps both numbers in one system. The comparison is what matters —
 * a cell the operator treats as full while its band says it is 40% liquid is
 * over-counting divergence by the difference.
 */
export function fineBandVolumeLedger(trace: FluidCellTrace): FineBandVolumeLedger {
  const census = fineBandCensus(trace);
  const operatorVolume = Number.isFinite(trace.volume) ? trace.volume : 0;
  return {
    operatorVolume,
    liquidFraction: census.liquidFraction,
    liquidVolume: operatorVolume * census.liquidFraction,
    partial: census.valid > 0 && census.negative > 0 && census.negative < census.valid,
    samples: census.valid,
  };
}

export interface FineBandBudget {
  /** Fine samples along one edge of this leaf. */
  readonly edge: number;
  /** Fine samples backing this one pressure unknown, were the band dense. */
  readonly samples: number;
  /** Bricks those samples occupy, at the four-sample brick edge. */
  readonly bricks: number;
  /** Bytes they occupy across the band's four channels. */
  readonly bytes: number;
  /** Probes actually walked, against which the counts above are estimates. */
  readonly probes: number;
  /** The band's estimated resident samples, scaled from probe coverage. */
  readonly residentEstimate: number;
}

/**
 * What one pressure unknown costs the surface tracker.
 *
 * The point of the figure is its ratio to the scheduled row updates already in
 * the HUD: one unknown, revisited a hundred-odd times by the solve, is backed
 * by millions of samples the band moves and floods every step. `samples` is the
 * dense count for the leaf — an upper bound, since the band is sparse — and
 * `residentEstimate` scales it by measured probe coverage, which is why it is
 * named an estimate rather than a count.
 */
export function fineBandBudget(trace: FluidCellTrace): FineBandBudget {
  const factor = Math.max(1, trace.fineFactor);
  const edge = trace.leafSize * factor;
  const samples = edge ** 3;
  const bricksPerEdge = Math.max(1, Math.ceil(edge / FINE_BAND_BRICK_RESOLUTION));
  const census = fineBandCensus(trace);
  return {
    edge,
    samples,
    bricks: bricksPerEdge ** 3,
    bytes: samples * FINE_LEVELSET_CHANNELS * 4,
    probes: census.probes,
    residentEstimate: Math.round(samples * census.coverage),
  };
}

export interface FineBandScheduledWork {
  /** Semi-Lagrangian substeps the encoded transport takes per sample. */
  readonly substeps: number;
  /** Passes in the redistance ladder that ran. */
  readonly ladderPasses: number;
  /** Sample-touches the encoded schedule spends on this leaf's band. */
  readonly sampleTouches: number;
}

/**
 * Fine-band work the encoded schedule spends on this one leaf.
 *
 * Badged `scheduled` for the same reason the row-update figure is: the count
 * comes from the command graph, not from an observation, and a transport
 * generation that failed to commit still encoded its passes. Every resident
 * sample is touched once per advection substep and once per ladder pass.
 */
export function fineBandScheduledWork(input: {
  readonly residentSamples: number;
  readonly substeps: number;
  readonly ladderStrides: readonly number[];
}): FineBandScheduledWork {
  const substeps = Math.max(0, Math.floor(input.substeps));
  const ladderPasses = input.ladderStrides.length;
  return {
    substeps,
    ladderPasses,
    sampleTouches: Math.max(0, input.residentSamples) * (substeps + ladderPasses),
  };
}

/* ------------------------------------------------------------------------- */
/* Flood attribution, for this leaf rather than the domain.                  */
/* ------------------------------------------------------------------------- */

export interface FineBandLadderAttribution {
  readonly plan: FineFloodLadderPlan;
  /** Recorded probes closed by pass k, k = 0 (self-seeded) upward. */
  readonly bins: readonly number[];
  readonly resolved: number;
  /** Recorded, valid, but left with no seed. */
  readonly unresolved: number;
  readonly deepestHop: number;
  /** Leading passes needed to cover this leaf's own deepest hop. */
  readonly requiredPasses: number;
  /** False when a hop here is deeper than the whole encoded ladder could build. */
  readonly coveredByEncodedLadder: boolean;
}

/**
 * Bin this leaf's recorded probes against the ladder that actually ran.
 *
 * The same arithmetic the global provenance report applies to the whole field,
 * narrowed to one cell — which is the only way to tell whether a given cell sits
 * in the flood's cheap bulk or in the tail the schedule is really paying for.
 * Returns undefined rather than an empty attribution when no ladder was
 * published, because a histogram binned against an imagined schedule names
 * passes that never ran.
 */
export function fineBandLadderAttribution(
  records: readonly FluidCellTraceFineProbe[],
  strides: readonly number[],
): FineBandLadderAttribution | undefined {
  if (strides.length === 0) return undefined;
  let plan: FineFloodLadderPlan;
  try {
    plan = describeFineFloodLadder(strides);
  } catch {
    return undefined;
  }
  const bins = new Array<number>(plan.strides.length + 1).fill(0);
  let resolved = 0;
  let unresolved = 0;
  let deepestHop = 0;
  for (const record of records) {
    if (!record.valid) continue;
    if (!record.resolved) { unresolved += 1; continue; }
    resolved += 1;
    deepestHop = Math.max(deepestHop, record.hop);
    const bin = fineFloodLadderPrefixForReach(plan.prefixReach, record.hop);
    bins[Math.min(bin, bins.length - 1)] += 1;
  }
  return {
    plan,
    bins: Object.freeze(bins),
    resolved,
    unresolved,
    deepestHop,
    requiredPasses: fineFloodLadderPrefixForReach(plan.prefixReach, deepestHop),
    coveredByEncodedLadder: plan.encodedReach >= deepestHop,
  };
}

/**
 * Colour for a ladder bin, matching the `flood-provenance` field's legend.
 *
 * The same stops the volumetric view uses, so a reader moving between the whole
 * field and one cell is reading one scale rather than two that happen to be
 * near each other. Bin zero is the self-seeded set and is deliberately outside
 * the ramp: those samples were closed by no pass at all.
 */
export function fineBandLadderPassColor(bin: number, passes: number): `#${string}` {
  if (bin <= 0) return "#ffffff";
  const stops: readonly `#${string}`[] = ["#1a47eb", "#0fc7cc", "#fa9e14"];
  if (passes <= 1) return stops[stops.length - 1];
  const position = (bin - 1) / Math.max(1, passes - 1);
  const index = Math.min(stops.length - 1, Math.floor(position * stops.length));
  return stops[index];
}

/* ------------------------------------------------------------------------- */
/* Band membership: why this cell is refined at all.                          */
/* ------------------------------------------------------------------------- */

/** The four authored and derived half-widths, as the solver's planner set them. */
export interface FineBandWidths {
  /** Authored pressure refinement half-width, in finest cells. */
  readonly pressureBandCells: number;
  /** Authored surface-tracking half-width, in finest cells. */
  readonly surfaceBandCells: number;
  /** Width Section 5 transport actually moves, in fine cells. */
  readonly transportBandFineCells: number;
  /** Width redistance leaves valid, in fine cells. */
  readonly redistanceBandFineCells: number;
}

export interface FineBandRing {
  readonly id: "pressure" | "surface" | "transport" | "redistance";
  readonly label: string;
  readonly swatch: `#${string}`;
  /** Half-width in finest cells, so the four are directly comparable. */
  readonly radius: number;
  /** True when this cell's nearest surface sample lies inside this ring. */
  readonly contains: boolean;
}

export interface FineBandMembership {
  /** Distance from this cell to the surface, in finest cells. */
  readonly distance: number;
  /** Rings ordered innermost first. */
  readonly rings: readonly FineBandRing[];
  /** The innermost ring this cell falls inside, if any. */
  readonly innermost?: FineBandRing;
  /** True when the cell lies outside every band — the coarse authority case. */
  readonly outside: boolean;
}

/**
 * Where this cell sits in the nest of bands.
 *
 * The two transport widths arrive in fine cells and the two authored ones in
 * finest cells; converting both to finest cells here is the only place that
 * mismatch has to be handled, and it is what makes the four comparable on one
 * bar. The colours are the `band-residency` field's own, so a reader moving
 * between the volumetric view and this one is reading the same nest.
 */
export function fineBandMembership(
  distance: number, widths: FineBandWidths, fineFactor: number,
): FineBandMembership {
  const factor = Math.max(1, fineFactor);
  const near = Math.abs(Number.isFinite(distance) ? distance : Number.POSITIVE_INFINITY);
  const rings: FineBandRing[] = ([
    { id: "pressure", label: "Pressure refinement", swatch: "#ff168f", radius: widths.pressureBandCells, contains: false },
    { id: "surface", label: "Surface band", swatch: "#15c8db", radius: widths.surfaceBandCells, contains: false },
    { id: "transport", label: "Transport reach", swatch: "#6b54eb", radius: widths.transportBandFineCells / factor, contains: false },
    { id: "redistance", label: "Redistance support", swatch: "#1a664d", radius: widths.redistanceBandFineCells / factor, contains: false },
  ] as const satisfies readonly FineBandRing[])
    .map((ring) => ({ ...ring }))
    .filter((ring) => Number.isFinite(ring.radius) && ring.radius > 0)
    .sort((a, b) => a.radius - b.radius)
    .map((ring) => ({ ...ring, contains: near <= ring.radius }));
  const innermost = rings.find((ring) => ring.contains);
  return {
    distance: near,
    rings: Object.freeze(rings),
    ...(innermost ? { innermost } : {}),
    outside: rings.length > 0 && innermost === undefined,
  };
}

/* ------------------------------------------------------------------------- */
/* The narrative the HUD appends to the pressure story.                       */
/* ------------------------------------------------------------------------- */

/**
 * What the HUD needs beyond the trace itself.
 *
 * The band widths and the ladder that ran are properties of the frame's
 * publication rather than of the cell, so they arrive from the debug source
 * instead of the gather. Keeping them out of the trace ABI is deliberate: they
 * are the same for every cell in the frame, and copying them into a per-cell
 * record would invite the two to disagree.
 */
export interface FineBandCellContext {
  readonly widths: FineBandWidths;
  /** The ladder the last redistance encode emitted, not a re-derivation. */
  readonly ladderStrides: readonly number[];
}

const count = (value: number) => Math.round(value).toLocaleString();

function distanceText(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return Math.abs(value) >= 100 ? value.toFixed(0)
    : Math.abs(value) >= 1 ? value.toFixed(2) : value.toFixed(3);
}

function bytesText(value: number): string {
  if (value >= 1 << 30) return `${(value / (1 << 30)).toFixed(1)} GiB`;
  if (value >= 1 << 20) return `${(value / (1 << 20)).toFixed(1)} MiB`;
  return `${(value / 1024).toFixed(0)} KiB`;
}

/**
 * The fine band's half of one cell's story.
 *
 * Appended to `fluidCellTraceNarrative` rather than folded into it, so the
 * pressure solve's account stays readable on a scene with no band at all and
 * this module does not have to be imported to render one. Every step carries
 * the same `gathered`/`scheduled` badge the rest of the HUD does, and for the
 * same reason: the sample budget is a measurement, the passes over it are a
 * schedule, and adding them would produce a figure describing nothing.
 */
export function fineBandNarrative(
  trace: FluidCellTrace, context: FineBandCellContext,
): readonly FluidCellTraceNarrativeStep[] {
  if (trace.fineFactor < 1) return [];
  const census = fineBandCensus(trace);
  const span = fineBandPhiSpan(trace);
  const ledger = fineBandVolumeLedger(trace);
  const budget = fineBandBudget(trace);
  const steps: FluidCellTraceNarrativeStep[] = [];

  steps.push({
    id: "fine-phi",
    label: "Read the corrected φ",
    layer: "surface",
    evidence: "gathered",
    value: span.published ? `${distanceText(span.centre)} cells` : "uncorrected",
    detail: span.published
      ? `the free-surface condition reads this one number; the band spans `
        + `${distanceText(span.minimum)} to ${distanceText(span.maximum)} across the leaf`
      : "this row carries no coarse record, so Section 5 never corrected it and the "
        + "solve is running on the coarse advection alone",
  });

  if (census.valid > 0) {
    steps.push({
      id: "fine-liquid",
      label: "Hold the liquid",
      layer: "patch",
      evidence: "gathered",
      value: `${(ledger.liquidFraction * 100).toFixed(0)}% liquid`,
      detail: ledger.partial
        ? `${count(census.negative)} of ${count(census.valid)} probes are inside the surface, so `
          + `the operator's volume ${pressureText(ledger.operatorVolume)} covers about `
          + `${pressureText(ledger.liquidVolume)} of liquid`
        : `every probe is on one side of the surface, so the cell is ${
          ledger.liquidFraction > 0.5 ? "interior liquid" : "outside the liquid"} and the `
          + `operator's volume needs no cut`,
    });
  }

  if (census.missing + census.stale > 0) {
    steps.push({
      id: "fine-coverage",
      label: "Cover the leaf",
      layer: "gaps",
      evidence: "gathered",
      value: `${(census.coverage * 100).toFixed(0)}%`,
      detail: `${count(census.missing)} probes found no resident page and `
        + `${count(census.stale)} found one at another generation; the coarse field answered there`,
    });
  }

  steps.push({
    id: "fine-budget",
    label: "Back one unknown",
    layer: "fine",
    evidence: "gathered",
    value: `${count(budget.samples)} samples`,
    detail: `this single pressure unknown spans ${budget.edge}³ fine samples across `
      + `${count(budget.bricks)} bricks — ${bytesText(budget.bytes)} were the band dense here, `
      + `and about ${count(budget.residentEstimate)} of them are resident`,
  });

  const scheduled = fineBandScheduledWork({
    residentSamples: budget.residentEstimate,
    // Section 5 divides the step by m, the factor the fine lattice is finer by,
    // which is exactly what the transport encodes.
    substeps: trace.fineFactor,
    ladderStrides: context.ladderStrides,
  });
  if (scheduled.sampleTouches > 0) {
    steps.push({
      id: "fine-work",
      label: "Move and redistance",
      layer: "links",
      evidence: "scheduled",
      value: `${count(scheduled.sampleTouches)} touches`,
      detail: `${scheduled.substeps} semi-Lagrangian substeps and ${scheduled.ladderPasses} flood `
        + `passes over this leaf's resident samples, every step — against `
        + `${count(budget.residentEstimate)} samples for one unknown`,
    });
  }
  return steps;
}

/** Shared with the pressure narrative's own formatting, so figures read alike. */
function pressureText(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return Math.abs(value) >= 1000 || (Math.abs(value) < 0.01 && value !== 0)
    ? value.toExponential(2) : value.toFixed(3);
}

/* ------------------------------------------------------------------------- */
/* Geometry the decorators draw.                                              */
/* ------------------------------------------------------------------------- */

/**
 * The closest point a probe's distance came from, in fine-lattice units.
 *
 * Thin wrapper over `fineFloodClosestPoint` so the decorators do not each have
 * to know that the code belongs to the *seed* rather than to the sample. Guards
 * the decode, because a code is a packed field read from GPU memory and a
 * malformed one must not take the frame down.
 */
export function fineBandProbeClosestPoint(
  probe: FluidCellTraceFineProbe,
): FluidCellTraceVec3 | undefined {
  if (!probe.resolved) return undefined;
  try {
    return fineFloodClosestPoint(probe.seedCell, probe.seedCode) as FluidCellTraceVec3;
  } catch {
    return undefined;
  }
}

/**
 * Centre of the brick a fine-lattice coordinate belongs to, and its size, both
 * in fine-lattice units.
 *
 * Residency is a property of the brick, not of the sample, so a gap has to be
 * drawn at brick granularity or it would claim a hole the size of one sample
 * where the real hole is sixty-four.
 */
export function fineBandBrickOf(cell: FluidCellTraceVec3): {
  readonly origin: FluidCellTraceVec3; readonly size: number;
} {
  const floor = (value: number) =>
    Math.floor(value / FINE_BAND_BRICK_RESOLUTION) * FINE_BAND_BRICK_RESOLUTION;
  return {
    origin: [floor(cell[0]), floor(cell[1]), floor(cell[2])] as FluidCellTraceVec3,
    size: FINE_BAND_BRICK_RESOLUTION,
  };
}

/** Stable key for deduplicating bricks across probes. */
export function fineBandBrickKey(cell: FluidCellTraceVec3): string {
  const { origin } = fineBandBrickOf(cell);
  return origin.join("_");
}

/**
 * The point on the dual edge where a crossing cuts, in finest cells.
 *
 * The dual edge of the power diagram runs site to site, which is what makes the
 * discretization second order (section 4: "faces of the primal mesh ... should
 * be orthogonal to the edges of the dual mesh"). Placing the crossing anywhere
 * but on that segment would draw a surface the operator does not use.
 */
export function fineBandCrossingPoint(
  ownCentre: FluidCellTraceVec3,
  neighborCentre: FluidCellTraceVec3,
  fraction: number,
): FluidCellTraceVec3 {
  const t = Math.max(0, Math.min(1, fraction));
  return [
    ownCentre[0] + (neighborCentre[0] - ownCentre[0]) * t,
    ownCentre[1] + (neighborCentre[1] - ownCentre[1]) * t,
    ownCentre[2] + (neighborCentre[2] - ownCentre[2]) * t,
  ] as FluidCellTraceVec3;
}

/** Leaf centre in finest cells, shared by the crossing geometry and the HUD. */
export function fineBandLeafCentre(
  origin: FluidCellTraceVec3, size: number,
): FluidCellTraceVec3 {
  return [origin[0] + size / 2, origin[1] + size / 2, origin[2] + size / 2] as FluidCellTraceVec3;
}

/** Neighbour centres, for a decorator that needs both ends of a dual edge. */
export function fineBandNeighborCentre(
  neighbor: FluidCellTraceNeighbor,
): FluidCellTraceVec3 {
  return fineBandLeafCentre(neighbor.leafOrigin, neighbor.leafSize);
}
