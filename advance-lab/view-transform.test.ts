import assert from "node:assert/strict";
import test from "node:test";

import {
  cellFromClient, clampedView, clientFromCell, fitScale, fitView, originPixels,
  panned, pixelsPerCell, SLICE_ZOOM_RANGE, svgViewBox, zoomedToward,
  type SliceView, type ViewportRect,
} from "./view-transform";

/* A room that divides: 64x48 finest cells at ten pixels each is exactly the
   viewport, so every number below can be read off by hand. */
const ROOM: ViewportRect = { left: 0, top: 0, width: 640, height: 480 };
const NX = 64, NY = 48;
const FIT = fitScale(ROOM, NX, NY);

test("the fit view is the whole slice at a whole number of pixels per cell", () => {
  assert.equal(FIT, 10);
  const view = fitView(NX, NY);
  assert.equal(pixelsPerCell(view, FIT), 10);
  assert.deepEqual(originPixels(view, FIT, ROOM), [0, 0],
    "at fit the slice fills the viewport, so cell (0,0) is its corner");
});

test("pixel and cell are inverses, and at fit they are the old picture", () => {
  const view = fitView(NX, NY);
  assert.deepEqual(cellFromClient(view, FIT, ROOM, 0, 0), [0, 0]);
  assert.deepEqual(cellFromClient(view, FIT, ROOM, 640, 480), [NX, NY]);
  for (const cell of [[0, 0], [10.5, 7.25], [63.9, 47.9]] as const) {
    const [x, y] = clientFromCell(view, FIT, ROOM, cell[0], cell[1]);
    const back = cellFromClient(view, FIT, ROOM, x, y);
    assert.ok(Math.abs(back[0] - cell[0]) < 1e-9 && Math.abs(back[1] - cell[1]) < 1e-9,
      `round trip lost ${cell} -> ${[x, y]} -> ${back}`);
  }
});

test("the rect's own frame is the frame back out, so an offset canvas still reads true", () => {
  const offset: ViewportRect = { left: 120, top: 40, width: 640, height: 480 };
  const view = fitView(NX, NY);
  assert.deepEqual(cellFromClient(view, FIT, offset, 120, 40), [0, 0]);
  assert.deepEqual(clientFromCell(view, FIT, offset, 0, 0), [120, 40]);
});

test("a wheel notch keeps the cell under the cursor under the cursor", () => {
  const fit = fitView(NX, NY);
  for (const [clientX, clientY] of [[100, 80], [0, 0], [639, 479], [320, 240]] as const) {
    const held = cellFromClient(fit, FIT, ROOM, clientX, clientY);
    /* Asserted through the clamp, because the clamp is what the page shows. */
    let view: SliceView = fit;
    for (let notch = 0; notch < 4; notch += 1) {
      view = clampedView(
        zoomedToward(view, FIT, ROOM, clientX, clientY, -100), FIT, ROOM, NX, NY);
      const now = cellFromClient(view, FIT, ROOM, clientX, clientY);
      assert.ok(Math.abs(now[0] - held[0]) < 1e-9 && Math.abs(now[1] - held[1]) < 1e-9,
        `notch ${notch} at ${[clientX, clientY]} slid ${held} to ${now}`);
    }
    assert.ok(view.zoom > 1.4, "four notches in should have magnified the picture");
  }
});

test("the wheel's sign and rate are the studio's", () => {
  const inward = zoomedToward(fitView(NX, NY), FIT, ROOM, 320, 240, -100);
  assert.ok(Math.abs(inward.zoom - Math.exp(0.1)) < 1e-12,
    "100 px of wheel travel is one exponential tenth, as in lib/core/math.ts");
  const outward = zoomedToward({ zoom: 4, panFine: [32, 24] }, FIT, ROOM, 320, 240, 100);
  assert.ok(outward.zoom < 4, "wheel down zooms out");
});

test("zoom cannot go below the fit, and the fit is centred whatever the pan says", () => {
  const out = clampedView(
    zoomedToward(fitView(NX, NY), FIT, ROOM, 100, 100, 400), FIT, ROOM, NX, NY);
  assert.equal(out.zoom, SLICE_ZOOM_RANGE.minimum);
  assert.deepEqual(out.panFine, [NX / 2, NY / 2]);
  assert.deepEqual(
    clampedView(panned(fitView(NX, NY), FIT, 137, -240), FIT, ROOM, NX, NY).panFine,
    [NX / 2, NY / 2], "panning a picture that already fits is a no-op");
});

test("panning grabs the water: it slides with the pointer, not against it", () => {
  const view: SliceView = { zoom: 4, panFine: [32, 24] };
  const ppc = pixelsPerCell(view, FIT);
  assert.equal(ppc, 40);
  const held = cellFromClient(view, FIT, ROOM, 200, 150);
  const after = panned(view, FIT, 80, -40);
  assert.deepEqual(after.panFine, [32 - 80 / ppc, 24 + 40 / ppc]);
  const moved = cellFromClient(after, FIT, ROOM, 280, 110);
  assert.ok(Math.abs(moved[0] - held[0]) < 1e-9 && Math.abs(moved[1] - held[1]) < 1e-9,
    "the cell that was grabbed is the cell under the pointer at the end of the drag");
});

test("the slice cannot be thrown off screen: the viewport's centre stays in it", () => {
  const view: SliceView = { zoom: 4, panFine: [32, 24] };
  const far = clampedView(panned(view, FIT, 100_000, -100_000), FIT, ROOM, NX, NY);
  assert.deepEqual(far.panFine, [0, NY]);
  const other = clampedView(panned(view, FIT, -100_000, 100_000), FIT, ROOM, NX, NY);
  assert.deepEqual(other.panFine, [NX, 0]);
  /* Which is the half-a-viewport statement: with the centre on the slice's
     corner, exactly half the viewport still shows water. */
  const [cornerX] = clientFromCell(far, FIT, ROOM, 0, 0);
  assert.equal(cornerX, ROOM.width / 2);
  assert.equal(clampedView(far, FIT, ROOM, NX, NY).panFine[0], 0, "clamping is idempotent");
  assert.ok(clampedView({ zoom: 1e9, panFine: [0, 0] }, FIT, ROOM, NX, NY).zoom
    === SLICE_ZOOM_RANGE.maximum, "and it bounds the far end too");
});

test("pixels per cell is whole while that is legible and fractional below two", () => {
  assert.equal(pixelsPerCell({ zoom: 1, panFine: [0, 0] }, 1.5), 1.5);
  assert.equal(pixelsPerCell({ zoom: 1.4, panFine: [0, 0] }, 1.5), 2);
  assert.equal(pixelsPerCell({ zoom: 2, panFine: [0, 0] }, 1.5), 3);
  assert.equal(pixelsPerCell({ zoom: 1, panFine: [0, 0] }, 7.9), 7);
});

test("the overlay viewBox is the viewport expressed in cells", () => {
  assert.equal(svgViewBox(fitView(NX, NY), FIT, ROOM, NX, NY), "0 0 64 48");
  assert.equal(svgViewBox({ zoom: 2, panFine: [10, 10] }, FIT, ROOM, NX, NY), "-6 -2 32 24");
  /* Same aspect ratio as the element it sizes, so the default
     preserveAspectRatio cannot letterbox it a second time. */
  const box = svgViewBox({ zoom: 3, panFine: [20, 30] }, FIT, ROOM, NX, NY)
    .split(" ").map(Number);
  assert.ok(Math.abs(box[2]! / box[3]! - ROOM.width / ROOM.height) < 1e-4,
    "to within the four decimal places the string is rounded to");
  /* And it agrees with the canvas transform: the cell at the viewport's
     top-left corner is the box's own origin. */
  const view: SliceView = { zoom: 3, panFine: [20, 30] };
  const corner = cellFromClient(clampedView(view, FIT, ROOM, NX, NY), FIT, ROOM, 0, 0);
  assert.ok(Math.abs(corner[0] - box[0]!) < 1e-3 && Math.abs(corner[1] - box[1]!) < 1e-3);
});

test("a viewport with no room yet does not produce a degenerate scale", () => {
  const empty: ViewportRect = { left: 0, top: 0, width: 0, height: 0 };
  assert.equal(fitScale(empty, NX, NY), 1);
  assert.equal(fitScale(ROOM, 0, 0), 480, "a slice with no cells cannot divide by zero");
  assert.ok(Number.isFinite(pixelsPerCell({ zoom: Number.NaN, panFine: [0, 0] }, FIT)));
});
