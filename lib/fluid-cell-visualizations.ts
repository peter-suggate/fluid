/**
 * What the pressure solve draws about one selected cell.
 *
 * Colocated with `fluid-cell-trace.ts` because these are that gather's picture:
 * the leaf the ray landed in, the finest cell inside it, and the eighteen power
 * neighbours the assembled row couples to. Everything here is `gathered` — state
 * the frame published and read back — so all of it draws solid. The derived
 * views of the same cell live with the models that derive them: the dependency
 * cone in `fluid-blast-radius.ts`, the flood reach in `fine-flood-provenance.ts`.
 *
 * The subject is structural rather than the `FluidCellTrace` type, which is what
 * lets these decorators be exercised without a GPU and keeps the registry from
 * needing to know the trace ABI.
 */
import {
  FLUID_CELL_TRACE_LAYER_DEFINITIONS,
  FLUID_CELL_TRACE_STATUS,
  fluidCellTraceDirectionLabel,
  fluidCellTraceLayerForNeighbor,
  type FluidCellTrace,
  type FluidCellTraceNeighbor,
} from "./fluid-cell-trace";
import {
  linearFromHex,
  type DecorationBuilder,
  type DecorationVec3,
} from "./visualization-decorations";
import {
  decorationVisualization,
  hasFields,
  type DecorationContext,
  type Visualization,
  type VisualizationNote,
} from "./visualization-registry";

const CELL_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.swatch);
const STENCIL_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.stencil.swatch);
const TRANSITION_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.transition.swatch);

/** A resolved cell trace, recognised without importing its decode. */
function acceptsCell(subject: unknown): subject is FluidCellTrace {
  if (!hasFields(subject, ["status", "row", "leafSize", "leafOrigin", "cell", "neighbors"])) return false;
  const trace = subject as FluidCellTrace;
  return trace.status === FLUID_CELL_TRACE_STATUS.resolved
    && trace.leafSize >= 1
    && Array.isArray(trace.neighbors);
}

const vec3 = (value: readonly number[]): DecorationVec3 =>
  [value[0], value[1], value[2]] as DecorationVec3;

function leafCentre(origin: readonly number[], size: number): DecorationVec3 {
  return [origin[0] + size / 2, origin[1] + size / 2, origin[2] + size / 2];
}

/** Key fragment for the neighbour set: sizes, origins and flags, not pressures. */
function neighborKey(neighbors: readonly FluidCellTraceNeighbor[]): string {
  return neighbors
    .map((neighbor) => `${neighbor.flags}.${neighbor.leafSize}.${neighbor.leafOrigin.join("_")}`)
    .join(",");
}

/**
 * Draw one coupling.
 *
 * The arrow leaves through the face, edge or corner its direction points at
 * rather than from the leaf centre, so eighteen of them read as a stencil
 * instead of a starburst from a single point.
 */
function pushCoupling(
  into: DecorationBuilder,
  trace: FluidCellTrace,
  neighbor: FluidCellTraceNeighbor,
  colorLinear: DecorationVec3,
  width_px: number,
  boxIntensity: number,
): void {
  const origin = vec3(neighbor.leafOrigin);
  const centre = leafCentre(origin, neighbor.leafSize);
  const own = leafCentre(trace.leafOrigin, trace.leafSize);
  const exit: DecorationVec3 = [
    own[0] + Math.sign(centre[0] - own[0]) * (trace.leafSize / 2),
    own[1] + Math.sign(centre[1] - own[1]) * (trace.leafSize / 2),
    own[2] + Math.sign(centre[2] - own[2]) * (trace.leafSize / 2),
  ];
  into.segment(exit, centre, { colorLinear, width_px, intensity: 0.95, arrow: true });
  into.cellBox(origin, neighbor.leafSize, {
    colorLinear, width_px: width_px * 0.55, intensity: boxIntensity,
  });
}

const cellDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "pressure-solve/cell",
  pass: "Pressure solve",
  group: "cell",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.description,
  source: "Published owner map and compact leaf headers",
  legend: [
    { mark: "box", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.swatch, label: "owning leaf — one unknown" },
    { mark: "box", swatch: "#b9c4cc", label: "finest cell under the pointer" },
  ],
  accepts: acceptsCell,
  key: (trace) => `${trace.row}.${trace.leafSize}.${trace.leafOrigin.join("_")}.${trace.cell.join("_")}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    // The leaf is the unknown — everything inside it shares one pressure — so it
    // is drawn boldest whether the pointer is passing over or a cell is frozen.
    into.cellBox(vec3(trace.leafOrigin), trace.leafSize, {
      colorLinear: CELL_COLOR,
      width_px: FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.width_px,
      intensity: 1,
    });
    // Which part of a 32³ leaf the pointer is actually over.
    if (trace.leafSize > 1) {
      into.cellBox(vec3(trace.cell), 1, {
        colorLinear: CELL_COLOR,
        width_px: FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.width_px * 0.5,
        intensity: 0.55,
      });
    }
    return [
      {
        label: "Own the cell", value: `${trace.leafSize}³`, evidence: "gathered",
        detail: `row ${trace.row} at (${trace.leafOrigin.join(", ")})`,
      },
    ];
  },
});

const stencilDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "pressure-solve/stencil",
  pass: "Pressure solve",
  group: "stencil",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.stencil.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.stencil.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.stencil.description,
  source: "Assembled power-operator row and the owner map around it",
  legend: [
    { mark: "arrow", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.stencil.swatch, label: "coupled neighbour at this leaf size" },
  ],
  accepts: acceptsCell,
  key: (trace) => `${trace.leafSize}.${trace.leafOrigin.join("_")}.${neighborKey(trace.neighbors)}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    const present = trace.neighbors.filter((neighbor) => neighbor.present && neighbor.leafSize >= 1);
    const sameSize = present.filter((neighbor) => !neighbor.coarser && !neighbor.finer);
    for (const neighbor of sameSize) {
      pushCoupling(into, trace, neighbor, STENCIL_COLOR,
        FLUID_CELL_TRACE_LAYER_DEFINITIONS.stencil.width_px, 0.32);
    }
    const boundaries = trace.neighbors.filter((neighbor) => neighbor.boundary).length;
    return [
      {
        label: "Couple to neighbours", value: `${present.length} rows`, evidence: "gathered",
        detail: boundaries > 0
          ? `${present.length} live of 18 directions; ${boundaries} leave the domain`
          : `${present.length} live of 18 directions`,
      },
      {
        label: "Assemble the row", value: `${trace.entryCount} entries`, evidence: "gathered",
        detail: `diagonal ${trace.diagonal.toPrecision(4)}, rhs ${trace.rhs.toPrecision(4)}`,
      },
    ];
  },
});

const transitionDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "pressure-solve/transition",
  pass: "Pressure solve",
  group: "transition",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.transition.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.transition.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.transition.description,
  source: "Neighbour leaf sizes read from the same compact headers",
  legend: [
    { mark: "arrow", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.transition.swatch, label: "neighbour at a different leaf size" },
  ],
  accepts: acceptsCell,
  key: (trace) => `${trace.leafSize}.${neighborKey(trace.neighbors)}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    const transitions = trace.neighbors.filter(
      (neighbor) => neighbor.present && (neighbor.coarser || neighbor.finer));
    for (const neighbor of transitions) {
      pushCoupling(into, trace, neighbor, TRANSITION_COLOR,
        FLUID_CELL_TRACE_LAYER_DEFINITIONS.transition.width_px, 0.6);
    }
    if (transitions.length === 0) {
      return [{
        label: "Cross resolution", value: "none", evidence: "gathered",
        detail: "every neighbour is at this leaf's size, so the stencil is the regular Cartesian case",
      }];
    }
    const coarser = transitions.filter((neighbor) => neighbor.coarser);
    return [{
      label: "Cross resolution", value: `${transitions.length} of 18`, evidence: "gathered",
      // One hop crossing a whole coarse leaf is why fine-grid reach is uneven,
      // so the directions are named rather than only counted.
      detail: `${coarser.length} coarser, ${transitions.length - coarser.length} finer — ${transitions
        .map((neighbor) => `${fluidCellTraceDirectionLabel(neighbor.direction)} ${neighbor.leafSize}³`)
        .join(", ")}`,
    }];
  },
});

const RAY_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.ray.swatch);
/** Interface leaves borrow the patch colour, so the run and the patch agree. */
const INTERFACE_HIT_COLOR = linearFromHex(FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.swatch);

const rayRunDecoration = decorationVisualization<FluidCellTrace>({
  kind: "decoration",
  id: "pressure-solve/ray-run",
  pass: "Pressure solve",
  group: "ray",
  label: FLUID_CELL_TRACE_LAYER_DEFINITIONS.ray.label,
  swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.ray.swatch,
  description: FLUID_CELL_TRACE_LAYER_DEFINITIONS.ray.description,
  source: "Owner map sampled along the pointer ray, deduplicated by row",
  legend: [
    { mark: "box", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.ray.swatch, label: "leaf under the pointer, unselected" },
    { mark: "box", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.patch.swatch, label: "the surface passes through this one" },
    { mark: "box", swatch: FLUID_CELL_TRACE_LAYER_DEFINITIONS.cell.swatch, label: "the step currently selected" },
  ],
  accepts: acceptsCell,
  // Every leaf drawn, plus which one is picked out and which hold the surface:
  // stepping the selection changes the picture without the run itself changing.
  key: (trace) => `${trace.hitIndex}.${trace.hits
    .map((hit) => `${hit.leafSize}@${hit.leafOrigin.join("_")}${hit.holdsInterface ? "*" : ""}`)
    .join(",")}`,
  build(trace, _context, into): readonly VisualizationNote[] {
    // Drawn on hover as well as pinned: the run is what tells the reader there
    // is anything to step through, and it is one box per leaf either way.
    for (const hit of trace.hits) {
      if (hit.selected) continue; // The cell decoration already owns this one.
      // A leaf the surface passes through is picked out of the run, because it
      // is almost always the one worth selecting and the jump control needs
      // something visible to aim at.
      into.cellBox(vec3(hit.leafOrigin), hit.leafSize, {
        colorLinear: hit.holdsInterface ? INTERFACE_HIT_COLOR : RAY_COLOR,
        width_px: FLUID_CELL_TRACE_LAYER_DEFINITIONS.ray.width_px
          * (hit.holdsInterface ? 1.5 : 1),
        intensity: hit.holdsInterface ? 0.6 : 0.3,
      });
    }
    if (trace.hits.length === 0) return [];
    const deepest = trace.hits[trace.hits.length - 1];
    const surface = trace.hits.filter((hit) => hit.holdsInterface).length;
    return [{
      label: "Choose along the ray",
      value: `${trace.hitIndex + 1} of ${trace.hits.length}${trace.hitOverflow > 0 ? "+" : ""}`,
      evidence: "gathered",
      detail: trace.hits.length === 1
        ? "the ray crosses one live leaf, so there is nothing to step through"
        : `[ and ] step between them; the deepest is a ${deepest.leafSize}³ leaf `
          + `${deepest.distance_m.toPrecision(3)} m along the ray`
          + (surface > 0 ? `, and ${surface} hold the interface` : "")
          + (trace.hitOverflow > 0 ? `, and ${trace.hitOverflow} more were not recorded` : ""),
    }];
  },
});

/** Everything the pressure solve draws about a selected cell. */
export const fluidCellVisualizations: readonly Visualization[] = Object.freeze([
  rayRunDecoration, cellDecoration, stencilDecoration, transitionDecoration,
]);

export { acceptsCell as acceptsFluidCellSubject, fluidCellTraceLayerForNeighbor };
