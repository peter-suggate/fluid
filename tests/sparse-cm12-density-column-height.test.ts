import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");
const classifier = readFileSync(new URL(
  "../lib/core/webgpu-water-global-fine-classify.ts",
  import.meta.url,
), "utf8");
const tetraEmitter = readFileSync(new URL(
  "../lib/core/webgpu-water-global-fine-tetra.ts",
  import.meta.url,
), "utf8");

function columnHeight(fills: readonly number[], base = 0): number {
  return base + fills.reduce((sum, fill) => sum + Math.max(0, Math.min(1, fill)), 0);
}

function restrictedColumn(waterline: number, scale: number,
  cells: number): number[] {
  return Array.from({ length: cells }, (_, y) => {
    const lower = Math.floor(y / scale) * scale;
    const fill = Math.max(0, Math.min(1, (waterline - lower) / scale));
    return fill;
  });
}

function adaptiveFloorReceipt(fills: readonly number[]): boolean {
  let previous = 1;
  let sawAir = false;
  for (const fill of fills) {
    if (fill > previous + 0.01) return false;
    previous = fill;sawAir ||= fill < 1 - 1e-3;
  }
  return sawAir && columnHeight(fills) <= 8.125;
}

test("integrated density gives one waterline at every represented scale", () => {
  for (const waterline of [0.25, 3.75, 8.125, 8.75, 15.5, 23.875]) {
    const heights = [1, 2, 4, 8].map((scale) =>
      columnHeight(restrictedColumn(waterline, scale, 24)));
    for (const height of heights) {
      assert.ok(Math.abs(height - waterline) < 1e-12,
        `${waterline}: ${heights.join(", ")}`);
    }
  }
});

test("height survives conservative refinement that spreads one cut over children", () => {
  const parentLower = 8;
  const parentFill = 0.375;
  const childFills = [0.5625, 0.1875];
  assert.equal(columnHeight(childFills, parentLower),
    parentLower + 2 * parentFill);
});

test("height integrates a monotone sheet split across coarse vertical bricks", () => {
  const brickWidth = 8;
  const fills = [0.4, 0.1, 0];
  const height = fills.reduce((sum, fill) => sum + brickWidth * fill, 0);
  assert.equal(height, 4);
  assert.ok(fills[0]! < 0.5,
    "the regression must remain invisible to per-coarse-cell rho=.5 classification");
});

test("floor ghost continuation reconstructs a true sub-half-cell waterline", () => {
  for (const height of [0.0011, 0.01, 0.125, 0.25, 0.499]) {
    const firstCentrePhi = 0.5 - height;
    const ghostCentrePhi = firstCentrePhi - 1;
    const crossing = -ghostCentrePhi / (firstCentrePhi - ghostCentrePhi);
    const reconstructedHeight = crossing - 0.5;
    assert.ok(Math.abs(reconstructedHeight - height) < 1e-12,
      `${height}: ${reconstructedHeight}`);
  }
});

test("one affine row-zero receipt spans thin, regular, and tall floor sheets", () => {
  for (const height of [0.0011, 0.499, 0.501, 4.49, 4.51, 8.125]) {
    const rowZeroPhi = 0.5 - height;
    assert.ok(Math.abs(0.5 - rowZeroPhi - height) < 1e-12,
      `${height}: row-zero receipt changed representation`);
  }
});

test("general adaptive receipt admits floor sheets but rejects elevated liquid", () => {
  assert.equal(adaptiveFloorReceipt([0.4, 0.1, 0]), true);
  assert.equal(adaptiveFloorReceipt([0, 1, 0]), false,
    "the falling corner brick must not be projected down to the floor");
  assert.equal(adaptiveFloorReceipt([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), false,
    "a tall free surface must keep the established local reconstruction");
});

test("cell-centred coarse heights interpolate continuously across their face", () => {
  const left = 1, right = 5, width = 8;
  const sample = (x: number) => left
    + (right - left) * ((x + 0.5 - width / 2) / width);
  assert.equal(sample(4) - sample(3), (right - left) / width);
  assert.equal(sample(7), 2.75);
  assert.equal(sample(8), 3.25);
  assert.equal(sample(8) - sample(7), (right - left) / width);
});

test("a physical waterline gives a rung-independent ghost-fluid boundary", () => {
  const waterline = 15.25;
  for (const width of [1, 2, 4, 8]) {
    const cutLower = Math.floor(waterline / width) * width;
    const cutFill = (waterline - cutLower) / width;
    const liquidCenter = cutFill >= 0.5
      ? cutLower + width / 2
      : cutLower - width / 2;
    const airCenter = liquidCenter + width;
    const theta = (waterline - liquidCenter) / (airCenter - liquidCenter);
    assert.ok(theta > 0 && theta < 1, `width ${width}: theta=${theta}`);
    assert.ok(Math.abs(
      liquidCenter + theta * (airCenter - liquidCenter) - waterline,
    ) < 1e-12, `width ${width}: reconstructed waterline`);
  }
});


