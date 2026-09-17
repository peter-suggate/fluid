import assert from "node:assert/strict";
import test from "node:test";
import {
  ADVANCE_LENSES, CELL_FILL_KEYS, DIRECT_LEVEL_SET_CONTOUR_KEY, DIRECT_LEVEL_SET_KEY,
  LIQUID_KEY, PALETTE, REPRESENT_LENS, SLICE_OVERLAYS, SOLID_KEY,
  drawCellFillSlice, drawDirectLevelSetSlice,
  drawSlice, interfaceSegments, markQuery,
  rdfMinorityAreaDistorted, type LensContext, type LensKey, usesCellFillSlice,
} from "./lenses";
/* The fraction view itself is shared with the 3-D dense-grid overlay and lives
 * beside it; the lab consumes it. These behaviours are pinned here because the
 * lab is what draws them. */
import {
  FRACTION_FLOOR, cellFillIsOverCapacity, cellFillOpacity, fractionReadout,
  fractionResidueRamp,
} from "../lib/core/fluid-fraction-view";
import {
  advanceCellAt,
  type AdvanceCellView, type AdvanceLattice, type AdvanceRdfView, type AdvanceView,
} from "../lib/physics-wasm/advance-view";

test("minority-area guard ignores roundoff and rejects amplified corner cuts", () => {
  assert.equal(rdfMinorityAreaDistorted(3.999774, 3.78, 4), true,
    "a 0.000226 accepted air sliver must not become a 0.22-cell corner hole");
  assert.equal(rdfMinorityAreaDistorted(3.999774, 3.9996, 4), false,
    "sub-per-mille cell-area variation remains RDF presentation smoothing");
  assert.equal(rdfMinorityAreaDistorted(0.2, 0.3, 1), false,
    "a resolved minority phase may vary without forcing a PLIC seam");
});

test("the fraction readout sizes itself to the answer it has to keep", () => {
  assert.equal(fractionReadout(0.42), ".42", "two decimals carry the ordinary range");
  assert.equal(fractionReadout(1), "1", "a full cell is one character, not 1.00");
  assert.equal(fractionReadout(0.995), "1",
    "half a percent under full rounds to full rather than drawing .99 forever");
  assert.equal(fractionReadout(1.04), "1.04",
    "overfull keeps the digit that says how far past capacity the cell is");
  assert.equal(fractionReadout(1e-4), "1e-4",
    "a resolved residue cell must not read as .00, which is what vacuum reads as");
  assert.equal(fractionReadout(FRACTION_FLOOR), "1e-6", "the floor still states itself");
});

test("the residue ramp spends a fixed share on each decade", () => {
  assert.equal(fractionResidueRamp(FRACTION_FLOOR), 0);
  assert.equal(fractionResidueRamp(0.5), 1);
  const decades = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1].map(fractionResidueRamp);
  for (let i = 1; i < decades.length; i += 1) {
    const step = decades[i]! - decades[i - 1]!;
    assert.ok(Math.abs(step - 0.1755) < 1e-3,
      `a decade must be a fixed share of the ramp, not ${step.toFixed(4)}`);
  }
  /* The claim the log ramp exists for: on a linear ramp over [0, 1/2] these two
   * cells differ by under a percent of the range and draw as the same nothing. */
  assert.ok(fractionResidueRamp(1e-3) - fractionResidueRamp(1e-4) > 0.15,
    "two residue cells a decade apart must separate");
});

test("an interface chord is the part of the plane that is not the cell's own edge", () => {
  /* Liquid above the waterline in canvas terms: the normal points down, out of
   * it, and the plane sits halfway down the cell. */
  const halved = interfaceSegments({ nx: 0, ny: 1, clipNx: 0, clipNy: 1, offset: 0.5 });
  assert.equal(halved.length, 1, "a plane through the cell cuts exactly one chord");
  const [ax, ay, bx, by] = halved[0]!;
  assert.equal(ay, 0.5);
  assert.equal(by, 0.5);
  assert.deepEqual([ax, bx].sort(), [0, 1], "the chord spans the cell");
  assert.equal(
    interfaceSegments({ nx: 0, ny: 1, clipNx: 0, clipNy: 1, offset: 2 }).length, 0,
    "a plane the cell does not reach contributes no chord, only its own box edges");
});

interface CanvasCall {
  readonly operation: string;
  readonly values: readonly number[];
  readonly alpha: number;
  readonly fillStyle: string;
  readonly strokeStyle: string;
}

function recordingContext(): { readonly g: CanvasRenderingContext2D; readonly calls: CanvasCall[] } {
  const calls: CanvasCall[] = [], stack: { alpha: number; fill: string; stroke: string }[] = [];
  const state = { alpha: 0.37, fill: "initial-fill", stroke: "initial-stroke" };
  const record = (operation: string, ...values: number[]): void => {
    calls.push({ operation, values, alpha: state.alpha,
      fillStyle: state.fill, strokeStyle: state.stroke });
  };
  const g = {
    get globalAlpha() { return state.alpha; },
    set globalAlpha(value: number) { state.alpha = value; },
    get fillStyle() { return state.fill; },
    set fillStyle(value: string | CanvasGradient | CanvasPattern) { state.fill = String(value); },
    get strokeStyle() { return state.stroke; },
    set strokeStyle(value: string | CanvasGradient | CanvasPattern) { state.stroke = String(value); },
    lineWidth: 1,
    clearRect: (...v: number[]) => record("clearRect", ...v),
    fillRect: (...v: number[]) => record("fillRect", ...v),
    strokeRect: (...v: number[]) => record("strokeRect", ...v),
    beginPath: () => record("beginPath"),
    closePath: () => record("closePath"),
    moveTo: (...v: number[]) => record("moveTo", ...v),
    lineTo: (...v: number[]) => record("lineTo", ...v),
    rect: (...v: number[]) => record("rect", ...v),
    clip: () => record("clip"),
    fill: () => record("fill"),
    stroke: () => record("stroke"),
    save() {
      stack.push({ alpha: state.alpha, fill: state.fill, stroke: state.stroke });
      record("save");
    },
    restore() {
      const saved = stack.pop();
      if (saved) {
        state.alpha = saved.alpha;
        state.fill = saved.fill;
        state.stroke = saved.stroke;
      }
      record("restore");
    },
  } as unknown as CanvasRenderingContext2D;
  return { g, calls };
}

function cell(partial: Partial<AdvanceCellView> & Pick<AdvanceCellView, "x0" | "y0">): AdvanceCellView {
  const width = partial.width ?? 1, height = partial.height ?? width;
  const capacity = partial.capacity ?? width * height;
  const volume = partial.volume ?? 0;
  return { x0: partial.x0, y0: partial.y0, width, height, size: Math.max(width, height), brick: 0,
    topologyCell: partial.topologyCell ?? 0, capacity, volume,
    fill: partial.fill ?? volume / Math.max(capacity, 1e-8), open: partial.open ?? capacity > 0,
    plane: partial.plane ?? null };
}

function cellFillFixture(): { readonly context: LensContext; readonly cells: readonly AdvanceCellView[];
  readonly rdf: AdvanceRdfView; readonly calls: CanvasCall[] } {
  const cells = [
    cell({ x0: 0, y0: 0, volume: 0, topologyCell: 0 }),
    cell({ x0: 1, y0: 0, volume: 0.5, topologyCell: 1 }),
    cell({ x0: 2, y0: 0, width: 2, height: 2, volume: 1, topologyCell: 2 }),
    cell({ x0: 4, y0: 0, capacity: 0.5, volume: 0.375, topologyCell: 3 }),
    cell({ x0: 5, y0: 0, volume: 1.2, topologyCell: 4 }),
    cell({ x0: 0, y0: 1, capacity: 0, volume: 1, fill: 1, open: false, topologyCell: 5 }),
    /* The rest of the bottom row, empty. Nothing is drawn in these, but a mark
     * probed here indexes the last row of every plane — which is the only
     * place an `nx + 1` row and an `ny + 1` one can be told apart. */
    cell({ x0: 1, y0: 1, topologyCell: 6 }),
    cell({ x0: 4, y0: 1, topologyCell: 7 }),
    cell({ x0: 5, y0: 1, topologyCell: 8 }),
  ] as const;
  const lattice: AdvanceLattice = { nx: 6, ny: 2, cells };
  const { g, calls } = recordingContext();
  /* Every plane the lenses read, at the sizes the index helpers imply: 6×2
   * cells, 7×2 vertical rows, 6×3 horizontal ones. A mark that indexes one of
   * them the way its lens does not is an out-of-range read, and this fixture
   * is what makes that a failure rather than an `undefined`. */
  const s = { nx: 6, ny: 2, bx: 1, by: 1, lattice,
    brickRung: new Int8Array([1]), previousBrickRung: new Int8Array([0]),
    brickActivity: new Float32Array([0.4]),
    capacityFine: new Float32Array(12).fill(1),
    liquidVolumeFine: new Float32Array(12).fill(0.5),
    previousLiquidVolumeFine: new Float32Array(12).fill(0.5),
    pressureFine: new Float32Array(12), divergenceFine: new Float32Array(12),
    materialFine: new Float32Array(12), extensionFine: new Uint8Array(12),
    faceVelocityXFine: new Float32Array(14), faceVelocityYFine: new Float32Array(18),
    faceVelocityXBeforePressure: new Float32Array(14),
    faceVelocityYBeforePressure: new Float32Array(18),
    limitedFluxXFine: new Float32Array(14), limitedFluxYFine: new Float32Array(18),
    fluxLimitedXFine: new Uint8Array(14), fluxLimitedYFine: new Uint8Array(18),
    markers: [{ x: 1.5, y: 0.5, alive: true }] } as unknown as AdvanceView;
  const rdf: AdvanceRdfView = { vertexPhiFine: new Float32Array(21).fill(-1),
    segmentsFine: new Float32Array([0, 1, 6, 1]) };
  Object.assign(s, { rdf });
  return { context: { g, s, lattice, scale: 10 }, cells, rdf, calls };
}

test("transport cell fill uses exact clamped V/K opacity across adaptive and cut cells", () => {
  assert.deepEqual([-1, 0, 0.25, 0.5, 0.75, 1, 1.2, Number.NaN].map(cellFillOpacity),
    [0, 0, 0.25, 0.5, 0.75, 1, 1, 0]);
  assert.equal(cellFillIsOverCapacity(1 + 1e-6), false);
  assert.equal(cellFillIsOverCapacity(1 + 2e-6), true);

  const { context, rdf, calls } = cellFillFixture();
  drawCellFillSlice(context, rdf);
  const ground = calls.find(call => call.operation === "fillRect"
    && call.fillStyle === PALETTE.ground);
  assert.equal(ground?.alpha, 1, "the diagnostic always repaints an opaque ground");
  const liquid = calls.filter(call => call.operation === "fillRect" && call.fillStyle === PALETTE.liquid);
  assert.deepEqual(liquid.map(call => ({ box: call.values, alpha: call.alpha })), [
    { box: [10, 0, 10, 10], alpha: 0.5 },
    { box: [20, 0, 20, 20], alpha: 0.25 },
    { box: [40, 0, 10, 10], alpha: 0.75 },
    { box: [50, 0, 10, 10], alpha: 1 },
  ], "zero V stays clear and each accepted adaptive footprint uses its unboosted clamped fill");
  assert.equal(context.g.globalAlpha, 1, "the renderer restores alpha for following lenses");
});

test("only the transport-stage picture defaults to the cell-fill legend", () => {
  assert.equal(usesCellFillSlice("conservative-transport", false), true);
  assert.equal(usesCellFillSlice("conservative-transport", true), false,
    "the representation view retains its own surface picture");
  assert.equal(usesCellFillSlice("pressure-solve", false), false,
    "a non-transport lens retains the geometric slice");
  assert.deepEqual(CELL_FILL_KEYS.map(mark => [mark.tone, mark.label]), [
    ["liquid", "cell fill opacity · clamp(V/K, 0, 1)"],
    ["output", "accepted surface contour"],
    ["amber", "over-capacity cell · V/K > 1"],
  ]);
  assert.ok(CELL_FILL_KEYS.every(mark => mark.note.length > 0),
    "a mark with no note cannot answer the probe, which is now the only place it is read");
});

test("transport cell fill retains only the contour over cells and marks overcapacity amber", () => {
  const { context, rdf, calls } = cellFillFixture();
  drawCellFillSlice(context, rdf);
  assert.equal(calls.some(call => call.operation === "fill" && call.fillStyle === PALETTE.liquid), false,
    "the RDF must not leave an opaque liquid polygon beneath transparent cell fill");
  assert.ok(calls.some(call => call.operation === "lineTo" && call.strokeStyle === PALETTE.output),
    "the selected RDF contour remains as a thin reference line");
  const amberBoxes = calls.filter(call => call.operation === "strokeRect"
    && call.strokeStyle === PALETTE.amber);
  assert.equal(amberBoxes.length, 1, "only the over-capacity cell receives the amber marker");
  assert.deepEqual(amberBoxes[0]!.values, [50.75, 0.75, 8.5, 8.5]);

  const plic = cellFillFixture();
  Object.assign(plic.cells[1]!, { plane: { nx: 0, ny: 1, clipNx: 0, clipNy: 1, offset: 0.5 } });
  drawCellFillSlice(plic.context);
  assert.ok(plic.calls.some(call => call.operation === "lineTo"
    && call.strokeStyle === PALETTE.output), "the selected PLIC contour is retained too");
});

test("the ordinary slice renderer keeps its geometric liquid fill", () => {
  const { context, rdf, calls } = cellFillFixture();
  drawSlice(context, rdf);
  assert.ok(calls.some(call => call.operation === "fill" && call.fillStyle === PALETTE.liquid
    && call.alpha === 0.9), "non-transport lenses retain the original RDF/PLIC liquid geometry");
  assert.equal(calls.some(call => call.operation === "strokeRect"
    && call.strokeStyle === PALETTE.amber), false, "the overcapacity marker belongs only to cell fill");
});

test("the direct level-set renderer never substitutes cell fill or PLIC", () => {
  const { context, cells, rdf, calls } = cellFillFixture();
  Object.assign(cells[1]!, { plane: { nx: 0, ny: 1, clipNx: 0, clipNy: 1, offset: 0.5 } });
  drawDirectLevelSetSlice(context, rdf);
  assert.equal(calls.some(call => call.operation === "fillRect"
    && call.fillStyle === PALETTE.liquid), false,
  "authoritative V/K must not become surface fill in direct level-set mode");
  const contour = calls.filter(call => call.operation === "lineTo"
    && call.strokeStyle === PALETTE.output);
  assert.equal(contour.length, 1, "only the published zero-set segment is stroked");
  assert.deepEqual(contour[0]!.values, [60, 10]);

  const query = markQuery(context.s, cells[1]!, 1, 0);
  assert.equal(DIRECT_LEVEL_SET_KEY.holds(query), true);
  assert.equal(DIRECT_LEVEL_SET_CONTOUR_KEY.holds(query), false,
    "an all-negative scalar cell is liquid but is not crossed by the zero set");
});

/** Every mark the picture can make, in one list, the way the probe sees them. */
const everyMark = (): readonly LensKey[] => [
  LIQUID_KEY, DIRECT_LEVEL_SET_KEY, DIRECT_LEVEL_SET_CONTOUR_KEY, SOLID_KEY,
  ...CELL_FILL_KEYS, ...REPRESENT_LENS.keys,
  ...Object.values(ADVANCE_LENSES).flatMap(lens => lens.keys),
  ...Object.values(SLICE_OVERLAYS).flatMap(overlay => overlay.keys),
];

/**
 * A plane that refuses a read it does not hold.
 *
 * A typed array hands back `undefined` for an index past its end, which every
 * comparison in a mark then quietly turns into `false` — an out-of-range read
 * would pass unnoticed as "this cell does not carry that mark". The rows are
 * the easy ones to get wrong: three of the planes are `nx + 1` wide and three
 * are `ny + 1` tall, and a mark that indexes one the way its lens indexes the
 * other lands inside the array on most cells and outside it on the last row.
 */
const guarded = <T extends Float32Array | Uint8Array | Int8Array>(field: T, label: string): T =>
  new Proxy(field, {
    get(target, property) {
      const index = typeof property === "string" && /^-?\d+$/.test(property)
        ? Number(property) : null;
      if (index !== null && (index < 0 || index >= target.length)) {
        throw new RangeError(`${label} was read at ${index}, past its ${target.length} values`);
      }
      return Reflect.get(target, property);
    },
  }) as T;

test("every mark answers for one cell without reading outside its own planes", () => {
  const { context } = cellFillFixture();
  const planes = Object.fromEntries(Object.entries(context.s)
    .filter(([, value]) => ArrayBuffer.isView(value))
    .map(([name, value]) => [name, guarded(value as Float32Array, name)]));
  const s = { ...context.s, ...planes } as AdvanceView, marks = everyMark();
  assert.ok(marks.length > 25, "the whole declared set is under test, not a corner of it");
  assert.ok(marks.every(mark => mark.note.length > 12),
    "a mark with nothing to say has no reason to appear in the probe");
  let probed = 0;
  for (let fy = 0; fy < s.ny; fy += 1) for (let fx = 0; fx < s.nx; fx += 1) {
    const cell = advanceCellAt(context.lattice, s, fx, fy);
    assert.ok(cell, `the fixture must cover ${fx},${fy}: an uncovered cell tests nothing`);
    probed += 1;
    for (const mark of marks) {
      assert.equal(typeof mark.holds(markQuery(s, cell, fx, fy)), "boolean",
        `${mark.label} could not answer for the cell at ${fx},${fy}`);
    }
  }
  assert.equal(probed, s.nx * s.ny, "every finest cell is asked, including the last row");
});

test("the transport marks classify the cell under the pointer, not the picture", () => {
  const { context, cells } = cellFillFixture();
  const s = context.s;
  /* The cut cell's own footprint is part solid, which is the one reading the
   * base picture makes that no lens owns. */
  s.capacityFine[4] = 0.5;
  const at = (cell: AdvanceCellView, fx: number, fy: number): readonly string[] =>
    [...CELL_FILL_KEYS, SOLID_KEY].filter(mark => mark.holds(markQuery(s, cell, fx, fy)))
      .map(mark => mark.label);

  assert.deepEqual(at(cells[0]!, 0, 0), [],
    "an empty cell draws nothing, so the probe must claim nothing");
  assert.deepEqual(at(cells[1]!, 1, 0),
    ["cell fill opacity · clamp(V/K, 0, 1)", "accepted surface contour"],
    "a half-full cell is washed and cut, and is not over capacity");
  assert.deepEqual(at(cells[3]!, 4, 0),
    ["cell fill opacity · clamp(V/K, 0, 1)", "accepted surface contour", "solid"],
    "the cut cell reports the solid taking half its footprint");
  assert.deepEqual(at(cells[4]!, 5, 0),
    ["cell fill opacity · clamp(V/K, 0, 1)", "over-capacity cell · V/K > 1"],
    "past capacity the wash saturates and the contour has left the cell");
  assert.deepEqual(at(cells[5]!, 0, 1), [],
    "a closed cell holds no water to describe");
});
