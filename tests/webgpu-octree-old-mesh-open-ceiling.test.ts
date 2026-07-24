import assert from "node:assert/strict";
import test from "node:test";
import { octreeFaceBandWGSL } from "../lib/webgpu-octree-face-closest-point";

function compact(source: string) {
  return source.replace(/\s+/g, "");
}

function wgslFunction(name: string) {
  const source = compact(octreeFaceBandWGSL);
  const start = source.indexOf(`fn${name}(`);
  assert.notEqual(start, -1, `missing WGSL function ${name}`);
  const open = source.indexOf("{", start);
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    if (source[index] === "}") depth -= 1;
    if (depth === 0) return source.slice(start, index + 1);
  }
  assert.fail(`unterminated WGSL function ${name}`);
}

function extendCoordinate(
  grid: number,
  extent: number,
  maximumLeaf: number,
  kind: "closed-negative" | "closed-positive" | "open-ceiling",
) {
  if (kind === "closed-negative") {
    return grid < 0 && grid >= -maximumLeaf
      ? { extended: true, grid: -grid, flipped: true }
      : { extended: false, grid, flipped: false };
  }
  if (grid < extent || grid > extent + maximumLeaf) {
    return { extended: false, grid, flipped: false };
  }
  return kind === "closed-positive"
    ? { extended: true, grid: 2 * extent - grid, flipped: true }
    : { extended: true, grid: extent - 1e-5, flipped: false };
}

test("old-mesh backtrace extends a tangential crossing through the open ceiling", () => {
  const incident = wgslFunction("retainedBandIncidentVector");
  assert.match(incident,
    /for\(varaxis=0u;axis<3u;axis\+=1u\).*openCeiling=axis==1u&&!closed.*reflectComponents\(boundary,flips\)/s,
    "the query crossing itself authorizes the bounded product boundary extension");
  assert.doesNotMatch(incident, /extended=[^;]*normal\.y|normal\.y[^;]*extended/,
    "a -Z generalized face may cross the open +Y ceiling tangentially");
  assert.match(incident, /negativeBoundaryBit\(axis\).*positiveBoundaryBit\(axis\)/s,
    "closed side-wall exits use the same product ghost policy");

  assert.deepEqual(extendCoordinate(16.02, 16, 2, "open-ceiling"),
    { extended: true, grid: 15.99999, flipped: false },
    "the observed 0.026-cell tangential departure clamps to the final half-open sample plane");
  assert.deepEqual(extendCoordinate(16.02, 16, 2, "closed-positive"),
    { extended: true, grid: 15.98, flipped: true },
    "a closed positive wall mirrors the coordinate and flips only its normal component");
  assert.deepEqual(extendCoordinate(-0.02, 16, 2, "closed-negative"),
    { extended: true, grid: 0.02, flipped: true },
    "a closed negative wall mirrors the coordinate and flips only its normal component");
  assert.deepEqual(extendCoordinate(18.01, 16, 2, "open-ceiling"),
    { extended: false, grid: 18.01, flipped: false },
    "an excursion beyond the fixed maximum-leaf support remains invalid");
});
