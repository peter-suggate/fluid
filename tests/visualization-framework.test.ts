import assert from "node:assert/strict";
import test from "node:test";
import {
  DECORATION_SEGMENT_FLOATS,
  DecorationBuilder,
  containerDecorationSpace,
  decorationSpaceValid,
  linearFromHex,
} from "../lib/visualization-decorations";
import {
  assembleDecorations,
  decorationVisualization,
  hasFields,
  isFieldVisualization,
  type Visualization,
} from "../lib/visualization-registry";
import {
  VISUALIZATION_CATALOG,
  VISUALIZATION_FIELDS,
  duplicateVisualizationGroups,
  duplicateVisualizationIds,
  visualizationIdsForGroups,
  visualizationMark,
  visualizationsForGroups,
} from "../lib/visualization-catalog";
import { legendRows } from "../components/VisualizationLegend";
import { SVO_PIXEL_TRACE_LAYERS, svoPixelTraceLayersForMode } from "../lib/svo-pixel-trace";
import {
  FLUID_CELL_TRACE_ABI_VERSION,
  FLUID_CELL_TRACE_DIRECTIONS,
  FLUID_CELL_TRACE_LAYERS,
  FLUID_CELL_TRACE_STATUS,
  stepFluidCellTraceHit,
  type FluidCellTrace,
  type FluidCellTraceHit,
  type FluidCellTraceNeighbor,
} from "../lib/fluid-cell-trace";
import { OCTREE_TECHNIQUE_OVERLAY_CODES, isOctreeTechniqueOverlayMode } from "../lib/octree-technique-debug";

/* ------------------------------------------------------------------------- */
/* Fixtures                                                                   */
/* ------------------------------------------------------------------------- */

const DIMENSIONS = [16, 16, 16] as const;
const CONTAINER_M = [2, 2, 2] as const;

function neighbor(overrides: Partial<FluidCellTraceNeighbor> = {}): FluidCellTraceNeighbor {
  return {
    direction: 0, row: 7, leafSize: 4, flags: 1,
    leafOrigin: [12, 0, 8], pressure: 0,
    present: true, coarser: false, finer: false, boundary: false,
    ...overrides,
  };
}

/** A leaf on the ray run; the classification defaults to a plain interior one. */
function hit(overrides: Partial<FluidCellTraceHit> = {}): FluidCellTraceHit {
  return {
    row: 3, leafSize: 4, leafOrigin: [8, 0, 8], distance_m: 1, selected: true,
    flags: 0, holdsInterface: false, liquid: false, corrected: false,
    ...overrides,
  };
}

function cellTrace(overrides: Partial<FluidCellTrace> = {}): FluidCellTrace {
  return {
    version: FLUID_CELL_TRACE_ABI_VERSION,
    status: FLUID_CELL_TRACE_STATUS.resolved,
    pixel: [10, 20], requestToken: 1,
    cell: [8, 0, 8], row: 3, leafSize: 4, leafOrigin: [8, 0, 8],
    diagonal: 6, rhs: -1, entryCount: 6, volume: 1, topologyCode: 0, pressure: 0.5,
    fineSamples: 0, fineResolved: 0, fineMaximumHop: 0, fineInterface: 0,
    dimensions: [...DIMENSIONS] as unknown as FluidCellTrace["dimensions"],
    fineFactor: 4,
    fineProbes: 0, fineMissing: 0, fineStale: 0, fineNegative: 0,
    coarsePhi: 0, coarsePhiMinimum: 0, coarsePhiMaximum: 0, coarsePhiFlags: 0,
    probeMinimumPhi: 0, probeMaximumPhi: 0, probeNearestPhi: 0,
    fineProbeRecords: [],
    neighbors: [neighbor()],
    hits: [hit()],
    hitIndex: 0,
    hitOverflow: 0,
    ...overrides,
  };
}

const SPACE = containerDecorationSpace(
  DIMENSIONS as unknown as readonly [number, number, number],
  CONTAINER_M as unknown as readonly [number, number, number],
);

const ALL_IDS = new Set(VISUALIZATION_CATALOG.map((entry) => entry.id));
const everything = () => true;

function segmentStart(geometry: { segments: Float32Array }, index: number): [number, number, number] {
  const at = index * DECORATION_SEGMENT_FLOATS;
  return [geometry.segments[at], geometry.segments[at + 1], geometry.segments[at + 2]];
}

/* ------------------------------------------------------------------------- */
/* The catalog's own invariants                                               */
/* ------------------------------------------------------------------------- */

test("catalog ids are unique, because a toggle is persisted by id", () => {
  assert.deepEqual(duplicateVisualizationIds(), []);
});

test("group tags are owned by exactly one pass", () => {
  // visualizationIdsForGroups resolves a tag without asking which pass owns it,
  // so a collision would silently switch on another pass's line work.
  assert.deepEqual(duplicateVisualizationGroups(), []);
});

test("every declared field with a mode code agrees with the overlay's own table", () => {
  for (const field of VISUALIZATION_FIELDS) {
    if (field.modeCode === undefined) {
      assert.equal(isOctreeTechniqueOverlayMode(field.mode), false,
        `${field.id} names a technique mode but declares no code`);
      continue;
    }
    assert.equal(isOctreeTechniqueOverlayMode(field.mode), true, `${field.id} has a code for a non-technique mode`);
    assert.equal(field.modeCode,
      OCTREE_TECHNIQUE_OVERLAY_CODES[field.mode as keyof typeof OCTREE_TECHNIQUE_OVERLAY_CODES]);
  }
});

test("every technique overlay mode is declared exactly once", () => {
  const declared = VISUALIZATION_FIELDS
    .filter((field) => field.modeCode !== undefined)
    .map((field) => field.mode)
    .sort();
  assert.deepEqual(declared, Object.keys(OCTREE_TECHNIQUE_OVERLAY_CODES).sort());
});

test("the observatory's card set is the visible fields, in catalog order", () => {
  // Guards the hand-maintained PAPER_VIEWS list this replaced: the cards are
  // exactly the non-hidden fields, and hiding one is the only way to declare a
  // mode for the shader harness without adding a card.
  const visible = VISUALIZATION_FIELDS.filter((field) => !field.hidden);
  assert.equal(visible.length, 17);
  assert.deepEqual(visible.map((field) => field.mode), [
    "fine-band-lifecycle", "resolution", "global-fine-phi", "band-residency",
    "blast-radius", "flood-provenance", "power-cells", "power-faces",
    "octree-lifecycle", "pressure", "evaluated-velocity", "projection-update",
    "divergence-closure", "row-cost", "stencil-locality", "structured-velocity",
    "power-operator",
  ]);
  for (const field of visible) {
    assert.ok(field.legend && field.legend.length > 0, `${field.id} has no legend`);
    assert.ok(field.source, `${field.id} does not say where its numbers come from`);
  }
});

test("hidden fields are declared for the harness but offered no card", () => {
  const hidden = VISUALIZATION_FIELDS.filter((field) => field.hidden);
  assert.equal(hidden.length, 7);
  for (const field of hidden) assert.equal(field.modeCode !== undefined, true);
});

/* ------------------------------------------------------------------------- */
/* The legend                                                                 */
/* ------------------------------------------------------------------------- */

test("every layer the cell picker draws declares the shape of its own mark", () => {
  // A legend of ten identically-shaped swatches cannot say that a coupling is an
  // arrow and a bound is a dashed box, and only the decorator knows which it
  // draws — so the mark is declared beside the colour, and a view that forgot
  // would silently fall back to a line.
  for (const definition of visualizationsForGroups(FLUID_CELL_TRACE_LAYERS)) {
    assert.ok(definition.legend && definition.legend.length > 0, `${definition.id} has no legend`);
    for (const entry of definition.legend!) {
      assert.ok(entry.mark, `${definition.id} declares a legend entry with no mark`);
    }
    assert.equal(visualizationMark(definition), definition.legend![0].mark);
  }
});

test("a coupling reads as an arrow and anything derived reads as dashed", () => {
  // The two marks that carry meaning rather than decoration: direction, and the
  // solid/dashed distinction the line work itself draws with.
  const mark = (id: string) => visualizationMark(
    VISUALIZATION_CATALOG.find((definition) => definition.id === id)!);
  assert.equal(mark("pressure-solve/stencil"), "arrow");
  assert.equal(mark("pressure-solve/transition"), "arrow");
  assert.equal(mark("fine-band/closest-points"), "arrow");
  assert.equal(mark("pressure-solve/cone"), "dashed-box");
  assert.equal(mark("fine-band/flood-reach"), "dashed-box");
  assert.equal(mark("fine-band/gaps"), "dashed-box");
});

test("a switched-off layer keeps its row, because the row is also its switch", () => {
  // Hiding a view once it was off took its own switch away with it, so turning
  // it back on meant opening the disclosure first. Flicking a layer off to see
  // what is under it and straight back on is what the panel is for.
  const definitions = visualizationsForGroups(FLUID_CELL_TRACE_LAYERS);
  const counts = new Map<string, number>([["cell", 1], ["stencil", 12], ["gaps", 0]]);
  const rows = legendRows(definitions, ["cell", "stencil"], counts);
  assert.equal(rows.length, definitions.length);
  assert.deepEqual(rows.filter((row) => row.enabled).map((row) => row.group), ["cell", "stencil"]);
  assert.equal(rows.filter((row) => !row.enabled).length, definitions.length - 2);
  // Dimmed rather than dropped, and a measured zero is distinguished from a
  // layer whose count was never published at all.
  assert.equal(rows.find((row) => row.group === "gaps")?.empty, true);
  assert.equal(rows.find((row) => row.group === "cone")?.empty, false);
});

test("group lookup maps a reader's vocabulary onto catalog ids", () => {
  const ids = visualizationIdsForGroups(["cell", "cone"]);
  assert.deepEqual([...ids].sort(), ["pressure-solve/cell", "pressure-solve/cone"]);
  assert.deepEqual(visualizationIdsForGroups([]), []);
  assert.deepEqual(visualizationIdsForGroups(["not-a-group"]), []);
});

/* ------------------------------------------------------------------------- */
/* The drawing vocabulary                                                     */
/* ------------------------------------------------------------------------- */

test("a lattice box lands exactly where the shader's fineToWorld puts it", () => {
  // This is the whole accuracy claim: the overlay must draw on the pixels the
  // gather picked from. The container spans the finest grid, floor on y = 0,
  // centred on x and z.
  const builder = new DecorationBuilder(SPACE);
  builder.cellBox([8, 0, 8], 1, { colorLinear: [1, 1, 1], width_px: 1 });
  const geometry = builder.finish();
  assert.equal(geometry.segmentCount, 12);
  const fineToWorld = (point: readonly number[]) => [
    -0.5 * CONTAINER_M[0] + (point[0] / DIMENSIONS[0]) * CONTAINER_M[0],
    0 + (point[1] / DIMENSIONS[1]) * CONTAINER_M[1],
    -0.5 * CONTAINER_M[2] + (point[2] / DIMENSIONS[2]) * CONTAINER_M[2],
  ];
  assert.deepEqual(segmentStart(geometry, 0), fineToWorld([8, 0, 8]));
  assert.deepEqual(builder.toWorld([16, 16, 16]), [1, 2, 1]);
  assert.deepEqual(builder.toWorld([0, 0, 0]), [-1, 0, -1]);
});

test("an arrow emits a second instance for its screen-space head", () => {
  const shaft = new DecorationBuilder(SPACE);
  shaft.segment([0, 0, 0], [1, 1, 1], { colorLinear: [1, 1, 1], width_px: 2 });
  const headed = new DecorationBuilder(SPACE);
  headed.segment([0, 0, 0], [1, 1, 1], { colorLinear: [1, 1, 1], width_px: 2, arrow: true });
  assert.equal(shaft.segmentCount, 1);
  assert.equal(headed.segmentCount, 2);
});

test("append concatenates a producer's own buffer without reinterpreting it", () => {
  // The migration path for a producer that predates the framework.
  const source = new DecorationBuilder(SPACE);
  source.worldSegment([1, 2, 3], [4, 5, 6], { colorLinear: [0.5, 0.25, 0.125], width_px: 3 });
  const target = new DecorationBuilder(SPACE);
  const built = source.finish();
  target.append(built.segments, built.segmentCount);
  assert.equal(target.segmentCount, 1);
  assert.deepEqual([...target.finish().segments], [...built.segments]);
});

test("append refuses to read past the buffer it was handed", () => {
  const target = new DecorationBuilder(SPACE);
  target.append(new Float32Array(DECORATION_SEGMENT_FLOATS), 99);
  assert.equal(target.segmentCount, 1);
});

test("a space with a zero extent is rejected rather than dividing by it", () => {
  assert.equal(decorationSpaceValid(SPACE), true);
  assert.equal(decorationSpaceValid(containerDecorationSpace([0, 1, 1], [1, 1, 1])), false);
  assert.equal(decorationSpaceValid(containerDecorationSpace([1, 1, 1], [1, 0, 1])), false);
});

test("hex swatches decode to the linear triple the overlay composites", () => {
  assert.deepEqual(linearFromHex("#000000"), [0, 0, 0]);
  assert.deepEqual(linearFromHex("#ffffff"), [1, 1, 1]);
  const [red, green, blue] = linearFromHex("#ff0000");
  assert.equal(red, 1);
  assert.equal(green, 0);
  assert.equal(blue, 0);
});

/* ------------------------------------------------------------------------- */
/* Assembly                                                                   */
/* ------------------------------------------------------------------------- */

test("hovering draws everything pinning draws", () => {
  // Hover used to be a cheap outline and pinning the full picture, which made
  // committing to a cell the price of learning anything about it. Exploration is
  // the point of a picker, so the two states now differ only in whether the aim
  // is frozen — `emphasis` is prominence, never content.
  const subject = cellTrace();
  const hovered = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [subject],
    space: SPACE, emphasis: "hover", enabled: everything,
  });
  const pinned = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [subject],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  assert.deepEqual(hovered.entries, pinned.entries);
  assert.equal(hovered.geometry.segmentCount, pinned.geometry.segmentCount);
  assert.ok(hovered.geometry.segmentCount > 24, "the hovered picture is still only the outline");
  // The ray run contributes nothing here — the single-hit fixture's only leaf is
  // the selected one, which the cell decorator already owns.
  assert.equal(
    hovered.entries.find((entry) => entry.id === "pressure-solve/ray-run")!.segmentCount, 0);
});

test("the ray run outlines every leaf but the selected one", () => {
  // The point of the run is reaching an interior unknown: the pointer alone can
  // only ever name the nearest leaf, which on a liquid is a surface cell.
  const trace = cellTrace({
    row: 11, leafSize: 2, leafOrigin: [8, 4, 8], cell: [8, 4, 8],
    hitIndex: 1,
    hits: [
      hit({ row: 3, leafSize: 4, leafOrigin: [8, 0, 8], distance_m: 0.5, selected: false }),
      hit({ row: 11, leafSize: 2, leafOrigin: [8, 4, 8], distance_m: 0.9, selected: true }),
      hit({ row: 12, leafSize: 8, leafOrigin: [0, 8, 0], distance_m: 1.4, selected: false }),
    ],
  });
  const assembled = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [trace],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  const run = assembled.entries.find((entry) => entry.id === "pressure-solve/ray-run")!;
  // Two unselected leaves at twelve box edges each. The selected one is left to
  // the cell decorator so it is never drawn twice at two different weights.
  assert.equal(run.segmentCount, 24);
  const note = run.notes.find((entry) => entry.label === "Choose along the ray")!;
  assert.equal(note.value, "2 of 3");
  assert.equal(note.evidence, "gathered");
});

test("a truncated run says so rather than reading as complete", () => {
  const trace = cellTrace({ hitOverflow: 5 });
  const assembled = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [trace],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  const note = assembled.entries.find((entry) => entry.id === "pressure-solve/ray-run")!
    .notes.find((entry) => entry.label === "Choose along the ray")!;
  assert.equal(note.value, "1 of 1+");
});

test("stepping the run wraps at both ends against the run that exists", () => {
  // The index outlives the ray moving under it, so it has to survive a run that
  // shortens; wrapping keeps the last step useful instead of inert.
  assert.equal(stepFluidCellTraceHit(0, 1, 3), 1);
  assert.equal(stepFluidCellTraceHit(2, 1, 3), 0);
  assert.equal(stepFluidCellTraceHit(0, -1, 3), 2);
  // A run that vanished cannot be stepped into a nonsense index.
  assert.equal(stepFluidCellTraceHit(7, 1, 0), 0);
  assert.equal(stepFluidCellTraceHit(7, 1, 2), 0);
});

test("selecting draws the stencil, the transitions and their notes", () => {
  const trace = cellTrace({
    neighbors: [
      neighbor({ direction: 0, leafSize: 4 }),
      neighbor({ direction: 1, leafSize: 8, coarser: true, flags: 3, leafOrigin: [0, 0, 0] }),
      neighbor({ direction: 2, present: false, flags: 0 }),
      neighbor({ direction: 3, boundary: true, present: false, flags: 8 }),
    ],
  });
  const selected = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [trace],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  const ids = selected.entries.map((entry) => entry.id);
  assert.deepEqual(ids, [
    "pressure-solve/ray-run", "pressure-solve/cell",
    "pressure-solve/stencil", "pressure-solve/transition",
  ]);
  // One same-size neighbour: arrow + head + 12 box edges. Same for the coarser
  // one under the transition layer.
  const stencil = selected.entries.find((entry) => entry.id === "pressure-solve/stencil")!;
  assert.equal(stencil.segmentCount, 14);
  assert.equal(selected.entries.find((entry) => entry.id === "pressure-solve/transition")!.segmentCount, 14);
  // Notes are gathered state, and say so.
  assert.ok(stencil.notes.every((note) => note.evidence === "gathered"));
  assert.match(stencil.notes[0].detail!, /1 leave the domain/);
});

test("the cone is scheduled evidence, drawn only when a solve policy is supplied", () => {
  const trace = cellTrace();
  const without = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [trace],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  assert.equal(without.entries.some((entry) => entry.id === "pressure-solve/cone"), false);

  const withPolicy = assembleDecorations({
    definitions: VISUALIZATION_CATALOG,
    subjects: [{ ...trace, solvePolicy: { outerIterations: 2, levels: 5, smoothsPerLevel: 2 } }],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  const cone = withPolicy.entries.find((entry) => entry.id === "pressure-solve/cone");
  assert.ok(cone, "a cone should be drawn once a policy names the schedule");
  assert.equal(cone!.segmentCount % 12, 0);
  assert.ok(cone!.segmentCount / 12 <= 6, "the cone is thinned to a readable number of shells");
  assert.deepEqual(cone!.notes.map((note) => note.evidence), ["scheduled"]);
  // The exact-single-cell bottom is why one V-cycle reaches the whole grid.
  assert.match(cone!.notes[0].detail!, /depends on every other cell/);
});

test("the flood reach draws only where the fine band actually reported a hop", () => {
  const dry = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [cellTrace()],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  assert.equal(dry.entries.some((entry) => entry.id === "fine-band/flood-reach"), false);

  const wet = assembleDecorations({
    definitions: VISUALIZATION_CATALOG,
    subjects: [cellTrace({ fineSamples: 400, fineResolved: 400, fineMaximumHop: 16, fineFactor: 4 })],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  const reach = wet.entries.find((entry) => entry.id === "fine-band/flood-reach");
  assert.ok(reach);
  assert.equal(reach!.segmentCount, 12);
  assert.deepEqual(reach!.notes.map((note) => note.evidence), ["gathered"]);
});

test("two selections assemble into one buffer", () => {
  // The point of central assembly: a frame holding a picked cell and a traced
  // ray costs the same one upload as either alone.
  const rayLike = {
    records: [], ray: { origin_m: [0, 0, 0], direction: [0, 0, 1] },
    pixel: [4, 4], requestToken: 2,
  };
  const both = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [rayLike, cellTrace()],
    space: SPACE, emphasis: "selected", enabled: everything,
  });
  const passes = new Set(both.entries.map((entry) => entry.pass));
  assert.ok(passes.has("Pressure solve"));
  assert.ok(both.geometry.segments.length % DECORATION_SEGMENT_FLOATS === 0);
});

test("only enabled ids contribute", () => {
  const one = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [cellTrace()],
    space: SPACE, emphasis: "selected",
    enabled: (definition) => definition.id === "pressure-solve/cell",
  });
  assert.deepEqual(one.entries.map((entry) => entry.id), ["pressure-solve/cell"]);
  const none = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [cellTrace()],
    space: SPACE, enabled: () => false,
  });
  assert.equal(none.geometry.segmentCount, 0);
  assert.deepEqual(none.entries, []);
});

test("an unresolved trace draws nothing", () => {
  const missed = assembleDecorations({
    definitions: VISUALIZATION_CATALOG,
    subjects: [cellTrace({ status: FLUID_CELL_TRACE_STATUS.miss })],
    space: SPACE, enabled: everything,
  });
  assert.equal(missed.geometry.segmentCount, 0);
});

/* ------------------------------------------------------------------------- */
/* Keying: what makes the overlay cheap                                       */
/* ------------------------------------------------------------------------- */

test("a re-gathered cell whose picture did not change re-uses its upload", () => {
  // The gather re-runs every frame, so the trace's revision ticks even while the
  // pointer sits still. Keying on the drawn facts is what stops a rebuild.
  const options = { definitions: VISUALIZATION_CATALOG, space: SPACE, enabled: everything } as const;
  const first = assembleDecorations({ ...options, subjects: [cellTrace()] });
  const restOfFrame = assembleDecorations({
    ...options,
    // Pressure and token moved; nothing drawn did.
    subjects: [cellTrace({ pressure: 99, requestToken: 44, rhs: -3 })],
  });
  assert.equal(first.key, restOfFrame.key);

  const moved = assembleDecorations({ ...options, subjects: [cellTrace({ leafOrigin: [0, 0, 0] })] });
  assert.notEqual(first.key, moved.key);
});

test("the key covers the lattice and the emphasis, not just the subject", () => {
  const base = { definitions: VISUALIZATION_CATALOG, subjects: [cellTrace()], enabled: everything } as const;
  const selected = assembleDecorations({ ...base, space: SPACE, emphasis: "selected" });
  const hovered = assembleDecorations({ ...base, space: SPACE, emphasis: "hover" });
  const resized = assembleDecorations({
    ...base, emphasis: "selected",
    space: containerDecorationSpace(DIMENSIONS as unknown as readonly [number, number, number], [4, 2, 2]),
  });
  assert.notEqual(selected.key, hovered.key);
  assert.notEqual(selected.key, resized.key);
});

/* ------------------------------------------------------------------------- */
/* Failure containment                                                        */
/* ------------------------------------------------------------------------- */

test("a decorator that throws does not take the frame with it", () => {
  // A broken diagnostic must not break the thing it is diagnosing.
  const exploding = decorationVisualization<{ marker: true }>({
    kind: "decoration", id: "test/exploding", pass: "Test", label: "Boom", description: "",
    accepts: (subject): subject is { marker: true } => hasFields(subject, ["marker"]),
    key: () => "boom",
    build() { throw new Error("decorator failure"); },
  });
  const working = decorationVisualization<{ marker: true }>({
    kind: "decoration", id: "test/working", pass: "Test", label: "Fine", description: "",
    accepts: (subject): subject is { marker: true } => hasFields(subject, ["marker"]),
    key: () => "fine",
    build(_subject, _context, into) {
      into.cellBox([0, 0, 0], 1, { colorLinear: [1, 1, 1], width_px: 1 });
    },
  });
  const assembled = assembleDecorations({
    definitions: [exploding, working] as unknown as readonly Visualization[],
    subjects: [{ marker: true }], space: SPACE, enabled: everything,
  });
  assert.deepEqual(assembled.entries.map((entry) => entry.id), ["test/working"]);
  assert.equal(assembled.geometry.segmentCount, 12);
});

test("an invalid space assembles nothing rather than emitting NaN geometry", () => {
  const assembled = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [cellTrace()],
    space: containerDecorationSpace([0, 0, 0], [1, 1, 1]), enabled: everything,
  });
  assert.equal(assembled.geometry.segmentCount, 0);
  assert.equal(assembled.key, "invalid-space");
});

/* ------------------------------------------------------------------------- */
/* The wrapped producer                                                       */
/* ------------------------------------------------------------------------- */

test("every pixel-trace layer is declared, one entry each, beside the pass that produces it", () => {
  const svo = VISUALIZATION_CATALOG.filter((entry) => entry.id.startsWith("svo-traversal/"));
  assert.equal(svo.length, SVO_PIXEL_TRACE_LAYERS.length);
  for (const entry of svo) {
    assert.equal(entry.kind, "decoration");
    assert.ok(ALL_IDS.has(entry.id));
  }
  // The picker's layers span several passes now, because the raster primary
  // spread the work that one traversal megakernel used to do across a cull, a
  // background draw and an instanced brick draw. A layer naming a pass the frame
  // never encodes would send a reader to the wrong line of the profile.
  const encoded = new Set([
    "SVO primary visibility", "SVO primary brick raster", "SVO primary background and terrain",
    "SVO analytic rigid discovery", "SVO deferred dry lighting", "SVO cone-lighting prepass",
  ]);
  for (const entry of svo) assert.ok(encoded.has(entry.pass), `${entry.id} names an unencoded pass: ${entry.pass}`);
});

test("the layers offered for a mode are exactly the ones that mode can populate", () => {
  const traced = svoPixelTraceLayersForMode("traced");
  const raster = svoPixelTraceLayersForMode("raster");
  for (const layers of [traced, raster]) {
    for (const layer of layers) assert.ok(SVO_PIXEL_TRACE_LAYERS.includes(layer));
    assert.equal(new Set(layers).size, layers.length, "a mode may not offer a layer twice");
  }
  // Nothing in a raster frame corresponds to a per-pixel octree descent, and the
  // traced path never builds a proxy. Offering either across the seam would put
  // a control in the panel that can only ever draw nothing.
  for (const absent of ["hierarchy", "rejected", "bricks"] as const) {
    assert.ok(traced.includes(absent));
    assert.ok(!raster.includes(absent), `${absent} cannot exist in a raster frame`);
  }
  for (const absent of ["proxies", "proxy-losers", "winner"] as const) {
    assert.ok(raster.includes(absent));
    assert.ok(!traced.includes(absent), `${absent} cannot exist in a traced frame`);
  }
  // The union is what the registry and the persisted selection are keyed on, so
  // switching primary mode must never invalidate either.
  assert.deepEqual(
    [...new Set([...traced, ...raster])].sort(),
    [...SVO_PIXEL_TRACE_LAYERS].sort(),
  );
});

test("the eighteen power directions are what the stencil decorator draws", () => {
  // Six faces then twelve edges: the operator's own order, so a direction the
  // HUD names is the direction the arrow points.
  assert.equal(FLUID_CELL_TRACE_DIRECTIONS.length, 18);
  const trace = cellTrace({
    neighbors: FLUID_CELL_TRACE_DIRECTIONS.map((_, direction) => neighbor({ direction })),
  });
  const assembled = assembleDecorations({
    definitions: VISUALIZATION_CATALOG, subjects: [trace],
    space: SPACE, emphasis: "selected",
    enabled: (definition) => definition.id === "pressure-solve/stencil",
  });
  assert.equal(assembled.entries[0].segmentCount, 18 * 14);
});

test("fields never contribute segments", () => {
  const fields = VISUALIZATION_CATALOG.filter(isFieldVisualization);
  assert.ok(fields.length > 0);
  const assembled = assembleDecorations({
    definitions: fields, subjects: [cellTrace()], space: SPACE, enabled: everything,
  });
  assert.equal(assembled.geometry.segmentCount, 0);
});
