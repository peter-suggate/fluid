/**
 * What the Section 5 narrow band draws about the cell under the pointer.
 *
 * Four pictures, deliberately not one. The existing `fine-band/flood-reach` box
 * is an *envelope* — the deepest hop, dilated — and an envelope is the right
 * summary and the wrong detail. These are the detail, and each answers a
 * different question, which is why each is its own toggle:
 *
 * | Layer | Question |
 * | --- | --- |
 * | `surface` | Where does the free surface cut this row's dual edges, and what does that do to its coefficients? |
 * | `patch` | What does the surface actually look like inside this cell? |
 * | `links` | Where did each sample's distance come from? |
 * | `gaps` | Where could the band not answer at all? |
 *
 * The first belongs to the pressure solve and is drawn in finest cells; the
 * other three belong to the fine band and are drawn in the fine lattice, which
 * is `DecorationSpace`'s whole reason for existing — a decorator reasons in the
 * lattice it owns and the space makes it land.
 */
import {
  FINE_FLOOD_DIRECTION_DELTAS,
  decodeFineFloodClosestPointCode,
} from "./fine-flood-provenance";
import {
  fineBandBrickKey,
  fineBandBrickOf,
  fineBandCrossingPoint,
  fineBandFreeSurfaceCrossings,
  fineBandLeafCentre,
  fineBandNeighborCentre,
  fineBandProbeClosestPoint,
  fineBandTightestCrossing,
} from "./fine-band-cell-model";
import {
  FLUID_CELL_TRACE_LAYER_DEFINITIONS,
  FLUID_CELL_TRACE_STATUS,
  type FluidCellTrace,
  type FluidCellTraceFineProbe,
} from "./fluid-cell-trace";
import {
  linearFromHex,
  type DecorationBuilder,
  type DecorationSpace,
  type DecorationVec3,
} from "./visualization-decorations";
import {
  decorationVisualization,
  hasFields,
  type DecorationContext,
  type Visualization,
  type VisualizationNote,
} from "./visualization-registry";

const SURFACE_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.surface.swatch);
const PATCH_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.swatch);
const LINK_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.links.swatch);
const GAP_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.gaps.swatch);
/** A tight crossing is a near-singular coefficient, so it grades to the hazard red. */
const SINGULAR_COLOR = linearFromHex("#ff1738");
/** The flood-provenance legend's "closed by the last passes"; deep hops grade to it. */
const DEEP_HOP_COLOR = linearFromHex("#fa9e14");
/** A page at the wrong generation is a different fault from no page at all. */
const STALE_COLOR = linearFromHex("#fa9e14");

/** Boxes a gaps layer may draw before it stops being a picture. */
const MAXIMUM_GAP_BRICKS = 64;

function mix(a: DecorationVec3, b: DecorationVec3, t: number): DecorationVec3 {
  const amount = Math.max(0, Math.min(1, t));
  return [
    a[0] + (b[0] - a[0]) * amount,
    a[1] + (b[1] - a[1]) * amount,
    a[2] + (b[2] - a[2]) * amount,
  ];
}

const vec3 = (value: readonly number[]): DecorationVec3 =>
  [value[0], value[1], value[2]] as DecorationVec3;

/** A resolved cell trace carrying fine-band records. */
function acceptsFineBand(subject: unknown): subject is FluidCellTrace {
  if (!hasFields(subject, [
    "status", "leafSize", "leafOrigin", "fineFactor", "fineProbeRecords", "dimensions",
  ])) return false;
  const trace = subject as FluidCellTrace;
  return trace.status === FLUID_CELL_TRACE_STATUS.resolved
    && trace.leafSize >= 1
    && trace.fineFactor >= 1
    && Array.isArray(trace.fineProbeRecords);
}

/** A resolved trace whose row and neighbours carry coarse phi. */
function acceptsCrossings(subject: unknown): subject is FluidCellTrace {
  if (!hasFields(subject, ["status", "coarsePhiFlags", "neighbors", "leafOrigin", "leafSize"])) {
    return false;
  }
  const trace = subject as FluidCellTrace;
  return trace.status === FLUID_CELL_TRACE_STATUS.resolved
    && trace.coarsePhiFlags !== 0
    && Array.isArray(trace.neighbors);
}

/**
 * The same container mapped onto the fine lattice.
 *
 * The fine lattice has `fineFactor` samples per finest cell over exactly the
 * same world extent, so only `dimensions` changes. Building it here rather than
 * converting coordinates by hand is what keeps a fine-band decorator from ever
 * disagreeing with the shader that wrote the samples.
 */
function fineSpace(space: DecorationSpace, factor: number): DecorationSpace {
  const scale = Math.max(1, factor);
  return {
    dimensions: [
      space.dimensions[0] * scale,
      space.dimensions[1] * scale,
      space.dimensions[2] * scale,
    ],
    origin_m: space.origin_m,
    extent_m: space.extent_m,
  };
}

/** Records that reached a live sample, which is all three fine layers care about. */
function liveRecords(trace: FluidCellTrace): readonly FluidCellTraceFineProbe[] {
  return trace.fineProbeRecords.filter((record) => record.valid);
}

/* ------------------------------------------------------------------------- */
/* Free surface: where the band meets the operator.                          */
/* ------------------------------------------------------------------------- */

/**
 * Every dual edge the free surface cuts, and how hard it cuts.
 *
 * This is the payoff view. The HUD prints a row's diagonal and right-hand side
 * as two bare floats with nothing to interrogate them by; the ghost-fluid
 * condition is what makes them what they are. The dual edge runs site to site —
 * that orthogonality is the whole reason the discretization is second order —
 * so the crossing is drawn *on that segment* and nowhere else.
 *
 * The ring is normal to the edge, so it reads as the surface plane slicing
 * through, and it is placed at the measured fraction rather than halfway. A
 * ring hard against this cell's own face is a near-singular coefficient, and it
 * looks like one.
 */
const freeSurfaceDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "pressure-solve/free-surface",
  pass: "Pressure solve",
  group: "surface",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.surface.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.surface.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.surface.description,
  source: "Coarse phi at this row and its neighbours, crossed along the power dual edge",
  legend: [
    { mark: "ring", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.surface.swatch, label: "crossing on the dual edge" },
    { mark: "ring", swatch: "#ff1738", label: "tight crossing — the coefficient it scales" },
    { mark: "ring", swatch: "#8a7ad6", label: "crossing into this cell — the neighbour's condition, not this row's" },
  ],
  accepts: acceptsCrossings,
  key: (trace) => `${trace.row}.${trace.coarsePhi.toFixed(4)}.${trace.neighbors
      .map((neighbor) => `${neighbor.direction}:${neighbor.phi?.toFixed(4) ?? "-"}`).join(",")}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    const crossings = fineBandFreeSurfaceCrossings(trace);
    if (crossings.length === 0) return [];
    const centre = fineBandLeafCentre(trace.leafOrigin, trace.leafSize);
    const byDirection = new Map(trace.neighbors.map((neighbor) => [neighbor.direction, neighbor]));

    for (const crossing of crossings) {
      const neighbor = byDirection.get(crossing.direction);
      if (!neighbor) continue;
      const other = fineBandNeighborCentre(neighbor);
      const point = fineBandCrossingPoint(centre, other, crossing.fraction);
      const axis: DecorationVec3 = [
        other[0] - centre[0], other[1] - centre[1], other[2] - centre[2],
      ];
      // Tightness is what the reader is hunting: a crossing at 7% of the edge
      // scales that face's coefficient fourteenfold. Grading the colour by it
      // makes the hazardous face findable without reading the strip.
      const tightness = 1 - Math.min(1, crossing.fraction / 0.5);
      const colorLinear = crossing.outward
        ? mix(SURFACE_COLOR, SINGULAR_COLOR, tightness)
        : linearFromHex("#8a7ad6");
      const width_px = FLUID_CELL_TRACE_LAYER_DEFINITIONS.surface.width_px
        * (crossing.outward ? 1 : 0.7);
      // The liquid part of the dual edge — the length the coefficient actually
      // sees once the condition has moved the boundary in.
      into.segment(centre, point, {
        colorLinear, width_px, intensity: crossing.outward ? 0.95 : 0.5,
      });
      // Sized to the smaller of the two leaves so a ring between a 32-cell and
      // a 1-cell leaf still reads as belonging to both.
      const radius = Math.max(0.35, 0.3 * Math.min(trace.leafSize, neighbor.leafSize));
      into.ring(point, axis, radius, {
        colorLinear, width_px, intensity: crossing.outward ? 1 : 0.55,
      }, 16);
    }

    const outward = crossings.filter((crossing) => crossing.outward);
    const tightest = fineBandTightestCrossing(crossings);
    return [{
      label: "Meet the free surface",
      value: `${crossings.length} of 18`,
      evidence: "gathered",
      detail: tightest?.coefficientScale === undefined
        ? `${crossings.length} dual edges cross the surface; none from the liquid side, so this row carries no free-surface condition`
        : `${outward.length} leave the liquid; the tightest cuts ${tightest.label} at `
          + `${(tightest.fraction * 100).toFixed(0)}% of the dual edge, scaling that face's `
          + `coefficient ${tightest.coefficientScale.toFixed(1)}x`,
    }];
  },
});

/* ------------------------------------------------------------------------- */
/* Interface patch: the surface as this cell's own samples reconstruct it.   */
/* ------------------------------------------------------------------------- */

/**
 * The surface inside this leaf, from the samples that hold their own crossing.
 *
 * An interface sample is self-seeded: the redistance found a sign change across
 * one of its faces and recorded the sub-cell offset to it. Decoding that offset
 * gives an actual point on the zero level set, measured rather than
 * interpolated — so the cross marks a place the surface really passes through.
 *
 * Each point also gets a ring normal to the axis its crossing was found along.
 * That axis is the measured surface normal to within a face, and orienting the
 * marks is the difference between reading a cloud of confetti and reading a
 * sheet.
 */
const interfacePatchDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "fine-band/patch",
  pass: "Fine band",
  group: "patch",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.description,
  source: "Sub-cell closest-point codes left resident by the redistance commit",
  legend: [
    { mark: "cross", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.swatch, label: "point on the zero level set" },
    { mark: "ring", swatch: "#f5f5e6", label: "ring normal to the axis the crossing was found along" },
  ],
  accepts: acceptsFineBand,
  key: (trace) => `${trace.fineFactor}.${liveRecords(trace)
      .filter((record) => record.interface)
      .map((record) => `${record.cell.join("_")}:${record.seedCode}`).join(",")}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    const patch = liveRecords(trace).filter((record) => record.interface);
    if (patch.length === 0) return [];
    const builder = into.inSpace(fineSpace(into.space, trace.fineFactor));
    const width_px = FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.width_px;

    for (const record of patch) {
      const point = fineBandProbeClosestPoint(record);
      if (!point) continue;
      builder.cross(vec3(point), 0.45, {
        colorLinear: PATCH_COLOR, width_px, intensity: 1,
      });
      const closest = decodeFineFloodClosestPointCode(record.seedCode);
      if (closest.kind !== "axis" || closest.direction === undefined) continue;
      const delta = FINE_FLOOD_DIRECTION_DELTAS[closest.direction];
      if (!delta) continue;
      builder.ring(vec3(point), vec3(delta), 0.5, {
        colorLinear: PATCH_COLOR, width_px: width_px * 0.7, intensity: 0.7,
      }, 10);
    }

    return [{
      label: "Reconstruct the surface",
      value: `${patch.length} points`,
      evidence: "gathered",
      detail: `${patch.length} of ${trace.fineProbeRecords.length} recorded probes hold their own zero`
        + ` crossing; each cross is a measured point on the interface, not an interpolation`,
    }];
  },
});

/* ------------------------------------------------------------------------- */
/* Closest-point links: the flood's dependency graph, inside one leaf.       */
/* ------------------------------------------------------------------------- */

/**
 * Each sample joined to the closest point it inherited its distance from.
 *
 * The flood-reach box bounds all of these at once, which is the honest summary
 * and tells you nothing about direction. These are the links themselves: they
 * show which way the band had to look, and a leaf whose links all point one way
 * is one the surface is leaving.
 *
 * Self-seeded samples are skipped — their link has zero length, and the patch
 * layer already marks them.
 */
const closestPointDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "fine-band/closest-points",
  pass: "Fine band",
  group: "links",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.links.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.links.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.links.description,
  source: "Resolved seeds left resident by the redistance commit, one per recorded probe",
  legend: [
    { mark: "arrow", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.links.swatch, label: "short hop — closed early in the ladder" },
    { mark: "arrow", swatch: "#fa9e14", label: "deep hop — the tail the ladder pays for" },
  ],
  accepts: acceptsFineBand,
  key: (trace) => `${trace.fineFactor}.${liveRecords(trace)
      .map((record) => `${record.cell.join("_")}>${record.seedCell.join("_")}:${record.seedCode}`)
      .join(",")}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    const records = liveRecords(trace).filter((record) => record.resolved && record.hop > 0);
    if (records.length === 0) return [];
    const builder = into.inSpace(fineSpace(into.space, trace.fineFactor));
    const deepest = records.reduce((maximum, record) => Math.max(maximum, record.hop), 1);
    const width_px = FLUID_CELL_TRACE_LAYER_DEFINITIONS.links.width_px;

    for (const record of records) {
      const point = fineBandProbeClosestPoint(record);
      if (!point) continue;
      const depth = record.hop / deepest;
      // The sweep really is a sequence here — a deeper hop was closed by a
      // later pass of the ladder — so `order` is used for what it is for.
      builder.segment(
        [record.cell[0] + 0.5, record.cell[1] + 0.5, record.cell[2] + 0.5],
        vec3(point),
        {
          colorLinear: mix(LINK_COLOR, DEEP_HOP_COLOR, depth),
          width_px,
          intensity: 0.45 + 0.5 * depth,
          order: depth,
          arrow: true,
        },
      );
    }

    return [{
      label: "Inherit a distance",
      value: `${records.length} links`,
      evidence: "gathered",
      detail: `each recorded sample points at the closest point it took its distance from;`
        + ` the deepest here is ${deepest} fine cells`,
    }];
  },
});

/* ------------------------------------------------------------------------- */
/* Gaps: where the band could not answer.                                     */
/* ------------------------------------------------------------------------- */

/**
 * Bricks inside this leaf the band could not answer for.
 *
 * Section 5 corrects the coarse level set "wherever we have valid phi-values",
 * and that conditional is silent everywhere else in the tool: a cell whose band
 * never reached it looks exactly like a cell the band agreed with. Drawing the
 * holes is the only way to see the difference.
 *
 * Residency is a property of the brick, not of the sample, so a gap is drawn at
 * brick granularity — claiming a hole the size of one sample where the real
 * hole is sixty-four would understate it by two orders of magnitude. Dashed,
 * because absence is inferred from a lookup that failed rather than measured.
 * Missing and stale are different faults with different fixes, so they are
 * different colours.
 */
const bandGapDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "fine-band/gaps",
  pass: "Fine band",
  group: "gaps",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.gaps.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.gaps.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.gaps.description,
  source: "Page lookups that found no brick, or a brick at another generation",
  legend: [
    { mark: "dashed-box", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.gaps.swatch, label: "no resident page — the band does not reach here" },
    { mark: "dashed-box", swatch: "#fa9e14", label: "page at another generation — stale, not absent" },
  ],
  accepts: acceptsFineBand,
  key: (trace) => `${trace.fineFactor}.${trace.fineProbeRecords
      .filter((record) => record.missing || record.stale)
      .map((record) => `${fineBandBrickKey(record.cell)}:${record.stale ? "s" : "m"}`)
      .join(",")}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    const faults = trace.fineProbeRecords.filter((record) => record.missing || record.stale);
    if (faults.length === 0) return [];
    const builder = into.inSpace(fineSpace(into.space, trace.fineFactor));
    const width_px = FLUID_CELL_TRACE_LAYER_DEFINITIONS.gaps.width_px;
    const drawn = new Set<string>();
    let missing = 0;
    let stale = 0;

    for (const record of faults) {
      if (record.stale) stale += 1; else missing += 1;
      const key = fineBandBrickKey(record.cell);
      if (drawn.has(key) || drawn.size >= MAXIMUM_GAP_BRICKS) continue;
      drawn.add(key);
      const brick = fineBandBrickOf(record.cell);
      builder.cellBox(vec3(brick.origin), brick.size, {
        colorLinear: record.stale ? STALE_COLOR : GAP_COLOR,
        width_px,
        intensity: 0.85,
        dash_m: builder.cellSize_m * 1.5,
      });
    }

    const capped = drawn.size >= MAXIMUM_GAP_BRICKS;
    return [{
      label: "Fall back to the coarse field",
      value: `${missing + stale} of ${trace.fineProbeRecords.length}`,
      evidence: "gathered",
      detail: (stale > 0
        ? `${missing} probes found no page and ${stale} found one at another generation`
        : `${missing} recorded probes found no resident page, so the coarse field answered for them`)
        + (capped ? `; the first ${MAXIMUM_GAP_BRICKS} distinct bricks are drawn` : ""),
    }];
  },
});

/** Everything the fine band draws about one selected cell. */
export const fineBandCellVisualizations: readonly Visualization[] = Object.freeze([
  freeSurfaceDecoration,
  interfacePatchDecoration,
  closestPointDecoration,
  bandGapDecoration,
]);

export { acceptsFineBand, acceptsCrossings, fineSpace };
export type { DecorationContext };
