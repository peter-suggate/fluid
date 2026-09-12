import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { productionSceneSliceSeedById } from "./production-scene-slice";
import {
  applySliceInjectionDose, injectionCoverage, SLICE_LIQUID_INJECTION_SOURCE,
  sliceDropFromCanvas, sliceDropIsAddressable, sliceInjectionDemandedBrickKeys,
  sliceInjectionRequestedArea, type SliceInjectionCell, type SliceLiquidDrop,
} from "./slice-liquid-injection";
import {
  advanceSlice, createAdvanceSlice, injectAdvanceSliceLiquid,
} from "./slice-solver";

const scene = () => createAdvanceSlice(
  productionSceneSliceSeedById("water-box-dam-break", { dt: 1 / 30 }));

/** A drop over the dry half of the dam break, clear of the reservoir. */
const DRY_SIDE: SliceLiquidDrop = { centreFine: [18, 4], radiusFine: 2.5 };

function cell(id: number, centre: readonly [number, number], width = 1): SliceInjectionCell {
  return { id, centerFine: centre, widthsFine: [width, width],
    volumeFineCells: width * width };
}

/** One resident function's source, from its signature to the next one's. */
function residentFunction(wgsl: string, name: string): string {
  const begin = wgsl.indexOf(`fn ${name}(`);
  assert.ok(begin >= 0, `${name} left the resident encoder`);
  const end = wgsl.indexOf("\nfn ", begin + 3);
  return wgsl.slice(begin, end < 0 ? wgsl.length : end);
}

test("the resident kernels this port mirrors still carry the terms it copies", () => {
  const wgsl = readFileSync(new URL("../webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url),
    "utf8");
  for (const name of SLICE_LIQUID_INJECTION_SOURCE.residentFunctions) {
    assert.match(wgsl, new RegExp(`fn ${name}\\b`), `${name} left the resident encoder`);
  }
  const kernel = residentFunction(wgsl, "injectLiquid");
  // The drop branch establishes occupancy. `+` is the hose, and a `max` that
  // became an add here would let a second click overfill a cell.
  assert.match(kernel, /max\(previous,\s*clippedCoverage\)/,
    "a drop must establish occupancy with max, never accumulate like a hose");
  assert.match(kernel, /coverage\*cellOpenFraction\(id\)/,
    "the dose must stay clipped by the cell's open fraction");
  assert.match(kernel, /sparseCM12TopologyLifecycleAccepted\(\)/,
    "a refused topology must leave accepted density untouched");
  // Faces belong to the hose. If this early return ever goes, a dropped ball
  // stops arriving at rest and this port is wrong without failing anywhere.
  assert.match(residentFunction(wgsl, "injectLiquidFaces"),
    /p\.injectionCenter\.w!=2\.0\)\{return;\}/,
    "injectLiquidFaces must remain hose-only, so a drop lands at rest");
  const coverage = residentFunction(wgsl, "injectionCoverageAt");
  assert.match(coverage, /let signed=length\(q\)-1\.0;/,
    "coverage is the unit-ellipsoid signed distance");
  assert.match(coverage, /clamp\(0\.5-signed\*min\(p\.injectionRadius\.x/,
    "the smoothed one-cell indicator is the coverage this port mirrors");
  // Activation is conservative on purpose: an exact disk test here would leave
  // the ball's rim on a page that is still air when the dose lands.
  assert.match(residentFunction(wgsl, "injectionReachesBrick"),
    /injectionBoundsRadius\(\)>=lower/,
    "brick demand must stay the bounding-box test this port mirrors");
});

test("coverage is the kernel's smoothed indicator, not an exact area", () => {
  const drop: SliceLiquidDrop = { centreFine: [0, 0], radiusFine: 4 };
  // On the rim the indicator is exactly a half, whatever the radius: signed
  // distance is zero there and the width scaling multiplies nothing.
  assert.equal(injectionCoverage(drop, [4, 0], 1), 0.5);
  assert.equal(injectionCoverage(drop, [0, -4], 1), 0.5);
  // Deep inside saturates and far outside is empty, with one cell of ramp.
  assert.equal(injectionCoverage(drop, [0, 0], 1), 1);
  assert.equal(injectionCoverage(drop, [8, 0], 1), 0);
  assert.equal(injectionCoverage(drop, [4.25, 0], 1), 0.25);
  // A coarse cell sees a softer rim: the ramp is one *cell* wide, so the same
  // ball reads as a blur on a rung-1 leaf and as a disk on a rung-8 one.
  assert.ok(injectionCoverage(drop, [6, 0], 8) > 0.2);
  assert.equal(injectionCoverage(drop, [6, 0], 1), 0);
});

test("a dose is clipped by the open fraction and never accumulates", () => {
  const cells = [cell(0, [0, 0]), cell(1, [0.5, 0])];
  const density = new Float32Array(2), gamma = new Float32Array(2);
  const drop: SliceLiquidDrop = { centreFine: [0, 0], radiusFine: 4 };
  // Cell 1 is half blocked by solid, so it takes half the coverage.
  const first = applySliceInjectionDose(cells, density, gamma,
    Float32Array.from([1, 0.5]), drop);
  assert.equal(density[0], 1);
  assert.equal(density[1], 0.5);
  assert.equal(first.cellsWetted, 2);
  assert.ok(Math.abs(first.areaAdmittedFine - 1.5) < 1e-6);
  assert.deepEqual([...gamma], [1, 1]);

  // The same drop again is a no-op: `max` against water already there.
  const again = applySliceInjectionDose(cells, density, gamma,
    Float32Array.from([1, 0.5]), drop);
  assert.equal(again.cellsWetted, 0);
  assert.equal(again.areaAdmittedFine, 0);
  assert.deepEqual([...density], [1, 0.5]);
});

test("a fully blocked cell takes nothing at all", () => {
  const density = new Float32Array(1), gamma = new Float32Array(1);
  const dose = applySliceInjectionDose([cell(0, [0, 0])], density, gamma,
    Float32Array.from([0]), { centreFine: [0, 0], radiusFine: 4 });
  assert.equal(dose.cellsWetted, 0);
  assert.equal(density[0], 0);
  // Gamma is untouched too: a cell with no open volume was never covered.
  assert.equal(gamma[0], 0);
});

test("brick demand is the conservative bounding box, not the disk", () => {
  const bricks = [
    { key: 0, coordinate: [0, 0] as const },
    { key: 1, coordinate: [1, 0] as const },
    { key: 2, coordinate: [1, 1] as const },
    { key: 9, coordinate: [4, 4] as const },
  ];
  // Centred just inside brick 0's far corner: the disk never enters brick 2,
  // but its bounding box shares that corner, so the rim's page is woken too.
  const demanded = sliceInjectionDemandedBrickKeys(bricks,
    { centreFine: [7, 7], radiusFine: 2 });
  assert.deepEqual([...demanded].sort((a, b) => a - b), [0, 1, 2]);
  assert.ok(!demanded.has(9));
});

test("a macro brick is demanded across its whole span", () => {
  const demanded = sliceInjectionDemandedBrickKeys(
    [{ key: 5, coordinate: [0, 0], spanBricks: 4 }],
    { centreFine: [30, 30], radiusFine: 1 });
  assert.ok(demanded.has(5), "a span-4 leaf covers 32 fine cells, not 8");
});

test("a canvas aim is reflected into the solver's own frame once", () => {
  const drop = sliceDropFromCanvas([24, 16], [18, 11.5], 2.5);
  assert.deepEqual([...drop.centreFine], [18, 4.5]);
  assert.equal(drop.radiusFine, 2.5);
});

test("an unaddressable drop is refused before the authority is touched", () => {
  const dimensions = [24, 16] as const;
  assert.ok(sliceDropIsAddressable({ centreFine: [12, 8], radiusFine: 2 }, dimensions));
  // Overlapping the lattice edge is a real experiment; wholly outside is not.
  assert.ok(sliceDropIsAddressable({ centreFine: [-1, 8], radiusFine: 2 }, dimensions));
  assert.ok(!sliceDropIsAddressable({ centreFine: [-4, 8], radiusFine: 2 }, dimensions));
  assert.ok(!sliceDropIsAddressable({ centreFine: [12, 8], radiusFine: 0 }, dimensions));
  assert.ok(!sliceDropIsAddressable({ centreFine: [NaN, 8], radiusFine: 2 }, dimensions));

  const s = scene();
  const before = Float32Array.from(s.fields.density);
  const receipt = injectAdvanceSliceLiquid(s, { centreFine: [-40, 8], radiusFine: 2 });
  assert.equal(receipt.accepted, false);
  assert.equal(receipt.cellsWetted, 0);
  assert.equal(s.injections, 0);
  assert.equal(s.topology.accepted.generation, receipt.acceptedGeneration);
  assert.deepEqual([...s.fields.density], [...before]);
});

test("a drop into air activates its bricks and refines them", () => {
  const s = scene();
  const dryBricks = s.topology.accepted.bricks.filter(brick => brick.active !== false).length;
  const generation = s.topology.accepted.generation;

  const receipt = injectAdvanceSliceLiquid(s, DRY_SIDE);
  assert.equal(receipt.accepted, true);
  assert.ok(receipt.bricksDemanded >= 1);
  assert.ok(receipt.bricksActivated >= 1, "air under the ball must become resident");
  assert.equal(receipt.candidateGeneration, generation + 1);
  assert.equal(s.topology.accepted.generation, generation + 1);
  assert.ok(s.topology.accepted.bricks.filter(brick => brick.active !== false).length
    > dryBricks);

  // Demanded leaves are pinned to the finest rung, or the ball arrives as a
  // single coarse block of water rather than as a ball.
  for (const brick of s.topology.accepted.bricks) {
    if (!receipt.bricksDemanded || brick.active === false) continue;
    const key = sliceInjectionDemandedBrickKeys([brick], DRY_SIDE);
    if (key.has(brick.key)) assert.equal(brick.resolution, 8);
  }

  // And the wet page survives the next planner: a brick retired the frame
  // after it was filled is the vanishing drop.
  advanceSlice(s, 28);
  assert.equal(s.fault, null);
  assert.ok(s.fields.density.some(value => value > 0.5));
});

test("dropped water joins the conserved total rather than reading as drift", () => {
  const s = scene();
  for (let frame = 0; frame < 3; frame += 1) advanceSlice(s, 28);
  const drift = s.drift;
  const seeded = s.seededVolume;

  const receipt = injectAdvanceSliceLiquid(s, DRY_SIDE);
  assert.equal(receipt.accepted, true);
  assert.ok(receipt.areaAdmittedFine > 0);
  assert.ok(Math.abs(s.seededVolume - (seeded + receipt.areaAdmittedFine)) < 1e-4,
    "the denominator must move by exactly what the cells took");
  assert.ok(Math.abs(s.drift - drift) < 1e-5,
    "a drop is added water, never a conservation failure");
  assert.equal(s.injections, 1);

  for (let frame = 0; frame < 12; frame += 1) advanceSlice(s, 28);
  assert.equal(s.fault, null);
  assert.ok(Math.abs(s.drift) < 1e-3, `drift after the drop was ${s.drift}`);
});

test("the admitted area tracks the disk the reader asked for", () => {
  const s = scene();
  const receipt = injectAdvanceSliceLiquid(s, DRY_SIDE);
  const requested = sliceInjectionRequestedArea(DRY_SIDE, [s.nx, s.ny]);
  assert.ok(Math.abs(receipt.areaRequestedFine - requested) < 1e-9);
  // The smoothed rim is not the analytic disk, and must not be reported as if
  // it were: within a rim's worth of area, never equal.
  assert.ok(Math.abs(receipt.areaAdmittedFine - requested)
    < 2 * Math.PI * DRY_SIDE.radiusFine);
});

test("a drop onto the reservoir displaces nothing it cannot", () => {
  const s = scene();
  const seeded = s.seededVolume;
  // Deep inside the dam, every covered cell is already full, so the `max`
  // admits nothing and the run is untouched apart from its topology clock.
  const receipt = injectAdvanceSliceLiquid(s, { centreFine: [5, 5], radiusFine: 2 });
  assert.equal(receipt.accepted, true);
  assert.equal(receipt.cellsWetted, 0);
  assert.equal(receipt.areaAdmittedFine, 0);
  assert.equal(s.seededVolume, seeded);
});
