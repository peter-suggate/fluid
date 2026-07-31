import assert from "node:assert/strict";
import test from "node:test";
import {
  FLUID_CELL_TRACE_DIRECTIONS,
  FLUID_CELL_TRACE_HEADER,
  FLUID_CELL_TRACE_HEADER_WORDS,
  FLUID_CELL_TRACE_MAGIC,
  FLUID_CELL_TRACE_NEIGHBOR_CAPACITY,
  FLUID_CELL_TRACE_RECORD,
  FLUID_CELL_TRACE_RECORD_FLAGS,
  FLUID_CELL_TRACE_RECORD_WORDS,
  FLUID_CELL_TRACE_STATUS,
  FLUID_CELL_TRACE_WORDS,
  decodeFluidCellTrace,
  fluidCellTraceDirectionLabel,
  fluidCellTraceKeyFigures,
  fluidCellTraceLayerForNeighbor,
  fluidCellTraceNarrative,
  fluidCellTraceTotalWork,
  type FluidCellTraceSchedule,
} from "../lib/fluid-cell-trace";
import { add, dot, scale } from "../lib/math";
import { defaultCamera } from "../lib/model";
import { projectToViewport, viewportRayForPixel } from "../lib/webgpu-camera";

/** The mini dam's encoded solve, as `report-blast-radius.ts` reports it. */
const SCHEDULE: FluidCellTraceSchedule = {
  outerIterations: 10, levels: 5, fineGridSweeps: 50,
  stagesToGlobal: 24, stageCount: 260,
};

interface NeighborSpec {
  direction: number; row: number; leafSize: number; flags: number;
  origin: [number, number, number]; pressure: number;
}

/** Build the exact word array the gather shader writes, so decode is tested against the writer. */
function buildTrace(options: {
  status?: number; leafSize?: number; entryCount?: number;
  neighbors?: NeighborSpec[]; fine?: [number, number, number, number];
} = {}): Uint32Array {
  const words = new Uint32Array(FLUID_CELL_TRACE_WORDS);
  const floats = new Float32Array(words.buffer);
  words[FLUID_CELL_TRACE_HEADER.magic] = FLUID_CELL_TRACE_MAGIC;
  words[FLUID_CELL_TRACE_HEADER.status] = options.status ?? FLUID_CELL_TRACE_STATUS.resolved;
  words[FLUID_CELL_TRACE_HEADER.pixelX] = 640;
  words[FLUID_CELL_TRACE_HEADER.pixelY] = 360;
  words[FLUID_CELL_TRACE_HEADER.requestToken] = 7;
  words.set([5, 6, 7], FLUID_CELL_TRACE_HEADER.cell);
  words[FLUID_CELL_TRACE_HEADER.row] = 41;
  words[FLUID_CELL_TRACE_HEADER.leafSize] = options.leafSize ?? 4;
  words.set([4, 4, 4], FLUID_CELL_TRACE_HEADER.leafOrigin);
  floats[FLUID_CELL_TRACE_HEADER.diagonal] = 6.5;
  floats[FLUID_CELL_TRACE_HEADER.rhs] = -0.25;
  words[FLUID_CELL_TRACE_HEADER.entryCount] = options.entryCount ?? 12;
  floats[FLUID_CELL_TRACE_HEADER.volume] = 0.875;
  words[FLUID_CELL_TRACE_HEADER.topologyCode] = 0b101;
  floats[FLUID_CELL_TRACE_HEADER.pressure] = 1.5;
  const fine = options.fine ?? [64, 64, 3, 8];
  words[FLUID_CELL_TRACE_HEADER.fineSamples] = fine[0];
  words[FLUID_CELL_TRACE_HEADER.fineResolved] = fine[1];
  words[FLUID_CELL_TRACE_HEADER.fineMaximumHop] = fine[2];
  words[FLUID_CELL_TRACE_HEADER.fineInterface] = fine[3];

  const neighbors = options.neighbors ?? [];
  words[FLUID_CELL_TRACE_HEADER.neighborCount] = neighbors.length;
  neighbors.forEach((neighbor, index) => {
    const base = FLUID_CELL_TRACE_HEADER_WORDS + index * FLUID_CELL_TRACE_RECORD_WORDS;
    words[base + FLUID_CELL_TRACE_RECORD.direction] = neighbor.direction;
    words[base + FLUID_CELL_TRACE_RECORD.row] = neighbor.row;
    words[base + FLUID_CELL_TRACE_RECORD.leafSize] = neighbor.leafSize;
    words[base + FLUID_CELL_TRACE_RECORD.flags] = neighbor.flags;
    words.set(neighbor.origin, base + FLUID_CELL_TRACE_RECORD.leafOrigin);
    floats[base + FLUID_CELL_TRACE_RECORD.pressure] = neighbor.pressure;
  });
  return words;
}

const live = (direction: number, leafSize: number, extra = 0): NeighborSpec => ({
  direction, row: 100 + direction, leafSize,
  flags: FLUID_CELL_TRACE_RECORD_FLAGS.present | extra,
  origin: [0, 0, 0], pressure: 0.5,
});

test("the eighteen directions are six faces then twelve edges", () => {
  assert.equal(FLUID_CELL_TRACE_DIRECTIONS.length, FLUID_CELL_TRACE_NEIGHBOR_CAPACITY);
  const nonZero = (index: number) =>
    FLUID_CELL_TRACE_DIRECTIONS[index].filter((component) => component !== 0).length;
  for (let index = 0; index < 6; index += 1) assert.equal(nonZero(index), 1, `direction ${index}`);
  for (let index = 6; index < 18; index += 1) assert.equal(nonZero(index), 2, `direction ${index}`);
});

test("direction labels name the axes they move along", () => {
  assert.equal(fluidCellTraceDirectionLabel(0), "+x");
  assert.equal(fluidCellTraceDirectionLabel(3), "−y");
  assert.equal(fluidCellTraceDirectionLabel(6), "+x+y");
});

test("a trace without the magic word is refused rather than misread", () => {
  const words = buildTrace();
  words[FLUID_CELL_TRACE_HEADER.magic] = 0;
  assert.equal(decodeFluidCellTrace(words), undefined);
  assert.equal(decodeFluidCellTrace(new Uint32Array(4)), undefined);
});

test("the decode recovers every gathered field the shader wrote", () => {
  const trace = decodeFluidCellTrace(buildTrace({ neighbors: [live(0, 4), live(1, 8)] }));
  assert.ok(trace);
  assert.equal(trace.status, FLUID_CELL_TRACE_STATUS.resolved);
  assert.deepEqual([...trace.pixel], [640, 360]);
  assert.deepEqual([...trace.cell], [5, 6, 7]);
  assert.equal(trace.row, 41);
  assert.equal(trace.leafSize, 4);
  assert.ok(Math.abs(trace.diagonal - 6.5) < 1e-6);
  assert.ok(Math.abs(trace.rhs + 0.25) < 1e-6);
  assert.equal(trace.entryCount, 12);
  assert.equal(trace.neighbors.length, 2);
  assert.equal(trace.neighbors[1].leafSize, 8);
  assert.ok(Math.abs(trace.neighbors[1].pressure - 0.5) < 1e-6);
});

test("a neighbour count beyond the record capacity cannot overrun the buffer", () => {
  const words = buildTrace({ neighbors: [live(0, 4)] });
  words[FLUID_CELL_TRACE_HEADER.neighborCount] = 999;
  const trace = decodeFluidCellTrace(words);
  assert.ok(trace);
  assert.equal(trace.neighbors.length, FLUID_CELL_TRACE_NEIGHBOR_CAPACITY);
});

test("resolution transitions are their own layer, because they are the paper's hard case", () => {
  const same = live(0, 4);
  const coarse = live(1, 8, FLUID_CELL_TRACE_RECORD_FLAGS.coarser);
  const trace = decodeFluidCellTrace(buildTrace({ neighbors: [same, coarse] }));
  assert.ok(trace);
  assert.equal(fluidCellTraceLayerForNeighbor(trace.neighbors[0]), "stencil");
  assert.equal(fluidCellTraceLayerForNeighbor(trace.neighbors[1]), "transition");
});

test("the narrative separates what the frame published from what the graph encodes", () => {
  const trace = decodeFluidCellTrace(buildTrace({
    neighbors: [live(0, 4), live(1, 8, FLUID_CELL_TRACE_RECORD_FLAGS.coarser)],
  }));
  assert.ok(trace);
  const steps = fluidCellTraceNarrative(trace, SCHEDULE);
  const byId = new Map(steps.map((step) => [step.id, step]));
  // Row identity and the assembled operator are facts about this frame.
  assert.equal(byId.get("cell")?.evidence, "gathered");
  assert.equal(byId.get("operator")?.evidence, "gathered");
  assert.equal(byId.get("stencil")?.evidence, "gathered");
  // Iteration counts belong to the command graph, not to the cell, and a
  // residual gate may zero the tail — so they must never read as observed.
  assert.equal(byId.get("updates")?.evidence, "scheduled");
  assert.equal(byId.get("reads")?.evidence, "scheduled");
  assert.equal(byId.get("cone")?.evidence, "scheduled");
  assert.match(byId.get("updates")?.detail ?? "", /residual gate/);
});

test("the narrative reports transitions against the live neighbour count", () => {
  const trace = decodeFluidCellTrace(buildTrace({
    neighbors: [live(0, 4), live(1, 8, FLUID_CELL_TRACE_RECORD_FLAGS.coarser),
      live(2, 2, FLUID_CELL_TRACE_RECORD_FLAGS.finer)],
  }));
  assert.ok(trace);
  const step = fluidCellTraceNarrative(trace, SCHEDULE).find((entry) => entry.id === "transition");
  assert.equal(step?.value, "2 of 3");
  assert.match(step?.detail ?? "", /1 coarser, 1 finer/);
});

test("a leaf with no resolution transitions says so instead of showing an empty tally", () => {
  const trace = decodeFluidCellTrace(buildTrace({ neighbors: [live(0, 4), live(1, 4)] }));
  assert.ok(trace);
  const step = fluidCellTraceNarrative(trace, SCHEDULE).find((entry) => entry.id === "transition");
  assert.match(step?.detail ?? "", /regular Cartesian case/);
});

test("boundary directions are reported rather than counted as live neighbours", () => {
  const boundary: NeighborSpec = {
    direction: 5, row: 0xffff_ffff, leafSize: 0,
    flags: FLUID_CELL_TRACE_RECORD_FLAGS.boundary, origin: [0, 0, 0], pressure: 0,
  };
  const trace = decodeFluidCellTrace(buildTrace({ neighbors: [live(0, 4), boundary] }));
  assert.ok(trace);
  const step = fluidCellTraceNarrative(trace, SCHEDULE).find((entry) => entry.id === "stencil");
  assert.equal(step?.value, "1 rows");
  assert.match(step?.detail ?? "", /1 leave the domain/);
});

test("total work is the schedule's row updates plus its stencil re-reads", () => {
  const trace = decodeFluidCellTrace(buildTrace({ entryCount: 12 }));
  assert.ok(trace);
  // 50 level-0 sweeps, each re-reading all twelve entries of this row.
  assert.equal(fluidCellTraceTotalWork(trace, SCHEDULE), 50 + 50 * 12);
});

test("a cell with no fine band omits the surface step rather than showing zeroes", () => {
  const withBand = decodeFluidCellTrace(buildTrace({ fine: [64, 64, 3, 8] }));
  const without = decodeFluidCellTrace(buildTrace({ fine: [0, 0, 0, 0] }));
  assert.ok(withBand && without);
  assert.ok(fluidCellTraceNarrative(withBand, SCHEDULE).some((step) => step.id === "fine"));
  assert.ok(!fluidCellTraceNarrative(without, SCHEDULE).some((step) => step.id === "fine"));
});

test("a cone that never goes global is stated, not left blank", () => {
  const trace = decodeFluidCellTrace(buildTrace());
  assert.ok(trace);
  const step = fluidCellTraceNarrative(trace, { ...SCHEDULE, stagesToGlobal: undefined })
    .find((entry) => entry.id === "cone");
  assert.equal(step?.value, "local");
  assert.match(step?.detail ?? "", /stays local/);
});

/* ------------------------------------------------------------------------- */
/* The aim: what a pin actually freezes.                                      */
/* ------------------------------------------------------------------------- */

/**
 * A pixel is not a selection.
 *
 * The gather marches a camera ray to decide which cell a request means, so
 * holding the pixel still while the camera moves holds nothing still: the ray
 * behind that pixel swings, and the pin slides onto whatever moved under it.
 * This is the property the fix rests on — the ray for a pixel is the exact
 * inverse of the projection the shader draws with, so the host can record one
 * and keep answering it after the camera has moved.
 */
test("a pixel's ray is the exact inverse of the projection drawn at it", () => {
  const width = 1280, height = 720;
  for (const [pixelX, pixelY] of [[640, 360], [17, 4], [1279, 719]] as const) {
    const ray = viewportRayForPixel(defaultCamera, pixelX, pixelY, width, height);
    const point = add(ray.origin, scale(ray.direction, 2.4));
    const projected = projectToViewport(point, defaultCamera, width, height);
    assert.ok(Math.abs(projected.leftFraction * width - (pixelX + 0.5)) < 1e-6);
    assert.ok(Math.abs(projected.topFraction * height - (pixelY + 0.5)) < 1e-6);
  }
});

test("the same pixel names a different aim once the camera moves, and a frozen ray does not", () => {
  const width = 1280, height = 720;
  const moved = { ...defaultCamera, azimuth_rad: defaultCamera.azimuth_rad + 0.25 };
  const pinned = viewportRayForPixel(defaultCamera, 900, 300, width, height);
  const samePixelLater = viewportRayForPixel(moved, 900, 300, width, height);
  // The bug, stated as an assertion: re-deriving from the pixel after a camera
  // move answers a materially different ray, which is why the pin has to carry
  // the ray itself rather than the pixel it came from.
  const drift = Math.acos(Math.max(-1, Math.min(1, dot(pinned.direction, samePixelLater.direction))));
  assert.ok(drift > 0.2, `re-derived aim moved only ${drift.toFixed(4)} rad`);
  // Said the other way round: the point the pin named is now under a different
  // pixel. Freezing the pixel therefore freezes nothing, and freezing the ray —
  // which does not depend on the camera at all — is what holds the selection.
  const point = add(pinned.origin, scale(pinned.direction, 2.4));
  const later = projectToViewport(point, moved, width, height);
  assert.ok(Math.abs(later.leftFraction * width - 900.5) > 1,
    "the pinned point still lands on its original pixel, so this camera move proves nothing");
});

/* ------------------------------------------------------------------------- */
/* Key figures: the HUD's first page.                                         */
/* ------------------------------------------------------------------------- */

test("the key figures answer whether a cell is worth reading, in four numbers", () => {
  const trace = decodeFluidCellTrace(buildTrace({
    entryCount: 12,
    neighbors: [live(0, 4), live(1, 4),
      live(2, 8, FLUID_CELL_TRACE_RECORD_FLAGS.coarser),
      live(3, 2, FLUID_CELL_TRACE_RECORD_FLAGS.finer)],
  }));
  assert.ok(trace);
  const figures = fluidCellTraceKeyFigures(trace, SCHEDULE);
  assert.deepEqual(figures.map((figure) => figure.id),
    ["work", "couples", "transitions", "surface"]);
  // Scheduled and gathered are never blended: the one derived figure says so,
  // and the HUD badges it on that word alone.
  assert.deepEqual(figures.map((figure) => figure.evidence),
    ["scheduled", "gathered", "gathered", "gathered"]);
  assert.equal(figures[0].value, (50 + 50 * 12).toLocaleString());
  assert.equal(figures[1].value, "4/18");
  // One coarser and one finer neighbour of the four.
  assert.equal(figures[2].value, "2");
});

test("a regular cell says its stencil is the Cartesian case rather than showing a zero", () => {
  const trace = decodeFluidCellTrace(buildTrace({ neighbors: [live(0, 4), live(1, 4)] }));
  assert.ok(trace);
  const transitions = fluidCellTraceKeyFigures(trace, SCHEDULE)
    .find((figure) => figure.id === "transitions");
  assert.equal(transitions?.value, "none");
  assert.match(transitions?.detail ?? "", /regular Cartesian case/);
});

test("an uncorrected row is flagged rather than reported as a phi of zero", () => {
  // Zero is a perfectly good distance, and reading it off a row the fine band
  // never corrected would put the free surface exactly on a cell that has no
  // opinion — the same trap the neighbour decode avoids by omitting phi.
  const decoded = decodeFluidCellTrace(buildTrace());
  assert.ok(decoded);
  const uncorrected = fluidCellTraceKeyFigures(decoded, SCHEDULE)
    .find((figure) => figure.id === "surface");
  assert.equal(uncorrected?.value, "—");
  assert.equal(uncorrected?.alert, true);
  assert.match(uncorrected?.detail ?? "", /no coarse level-set record/);

  const straddling = fluidCellTraceKeyFigures({
    ...decoded, coarsePhiFlags: 0b11, coarsePhi: -0.25,
    coarsePhiMinimum: -1.5, coarsePhiMaximum: 0.75,
  }, SCHEDULE).find((figure) => figure.id === "surface");
  assert.equal(straddling?.value, "-0.25");
  assert.equal(straddling?.alert, false);
  assert.match(straddling?.detail ?? "", /surface is inside this leaf/);

  const interior = fluidCellTraceKeyFigures({
    ...decoded, coarsePhiFlags: 0b11, coarsePhi: -3.5,
    coarsePhiMinimum: -4, coarsePhiMaximum: -3,
  }, SCHEDULE).find((figure) => figure.id === "surface");
  assert.match(interior?.detail ?? "", /3\.50 cells inside the liquid/);
});
