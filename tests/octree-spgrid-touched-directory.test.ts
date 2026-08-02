import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { buildSPGridBrickRankDirectory, buildSPGridPhysicalPageAdjacency,
  octreeSPGridVCycleShader } from
  "../lib/webgpu-octree-spgrid-vcycle";
import { buildSPGridTouchedDirectory, stableRadixSortU32 } from
  "../lib/octree-spgrid-touched-directory";

test("four-pass u32 radix is stable and exactly ascending", () => {
  const values = [0xffff_ffff, 256, 1, 0x10001, 0, 256, 255, 0x8000_0000];
  assert.deepEqual(stableRadixSortU32(values), [...values].sort((a, b) => a - b));
});

test("touched brick masks and ranks exactly match the dense directory oracle", () => {
  const dimensions = [19, 11, 7] as const;
  const cells = [
    { coordinate: [18, 10, 6] as const, slot: 91 },
    { coordinate: [0, 0, 0] as const, slot: 7 },
    { coordinate: [8, 6, 5] as const, slot: 44 },
    { coordinate: [3, 3, 3] as const, slot: 8 },
    { coordinate: [9, 6, 5] as const, slot: 45 },
    { coordinate: [17, 0, 0] as const, slot: 72 },
  ];
  const touched = buildSPGridTouchedDirectory(dimensions, cells);
  const dense = buildSPGridBrickRankDirectory(dimensions, cells);
  const denseOccupied = dense.masks.flatMap((masks, index) =>
    masks[0] !== 0 || masks[1] !== 0 ? [index] : []);
  assert.deepEqual(touched.bricks.map((brick) => brick.dense), denseOccupied);
  assert.deepEqual(touched.bricks.map((brick) => brick.masks),
    denseOccupied.map((index) => dense.masks[index]));
  assert.deepEqual(touched.bricks.map((brick) => brick.rankedBase),
    denseOccupied.map((index) => dense.bases[index]));
  assert.deepEqual(touched.rankedSlots, dense.slots);
  assert.ok(touched.touchedBricks.length < dense.masks.length,
    "the producer enumerates only first-touched identities, not the dense brick arena");
});

test("pages derived from touched bricks equal stable dense physical page identities", () => {
  const dimensions = [24, 16, 8] as const;
  const cells = [
    { coordinate: [17, 0, 0] as const, slot: 2 },
    { coordinate: [9, 9, 5] as const, slot: 5 },
    { coordinate: [0, 0, 0] as const, slot: 1 },
    { coordinate: [9, 0, 0] as const, slot: 3 },
    { coordinate: [10, 10, 1] as const, slot: 4 },
  ];
  const touched = buildSPGridTouchedDirectory(dimensions, cells);
  const pages = buildSPGridPhysicalPageAdjacency(dimensions,
    cells.map((cell) => cell.coordinate));
  const pageDims = [3, 2, 2] as const;
  const densePages = pages.origins.map(([x, y, z]) =>
    x / 8 + pageDims[0] * (y / 8 + pageDims[1] * (z / 4)));
  assert.deepEqual(touched.pages, densePages);
});

test("a rejected attempt can be rebuilt from its published identities alone", () => {
  const first = buildSPGridTouchedDirectory([16, 16, 8], [
    { coordinate: [15, 15, 7], slot: 3 }, { coordinate: [0, 0, 0], slot: 1 },
  ]);
  const retry = buildSPGridTouchedDirectory([16, 16, 8], [
    { coordinate: [4, 4, 4], slot: 2 },
  ]);
  assert.deepEqual(first.touchedBricks, [31, 0]);
  assert.deepEqual(retry.touchedBricks, [21]);
  assert.equal(retry.bricks.some((brick) => first.touchedBricks.includes(brick.dense)), false,
    "retry state contains no retired brick identity from the rejected attempt");
});

test("occupied and independently queried-empty identities round-trip through runs", () => {
  const dimensions = [20, 12, 8] as const;
  const cells = [
    { coordinate: [1, 1, 1] as const, slot: 9 },
    { coordinate: [5, 1, 1] as const, slot: 3 },
    { coordinate: [18, 10, 7] as const, slot: 12 },
  ];
  const queriedEmpty = [[9, 1, 1], [13, 5, 1], [9, 1, 1]] as const;
  const directory = buildSPGridTouchedDirectory(dimensions, cells, queriedEmpty);
  const brickDims = dimensions.map((value) => Math.ceil(value / 4));
  const identity = ([x, y, z]: readonly [number, number, number]) =>
    (x >> 2) + brickDims[0]! * ((y >> 2) + brickDims[1]! * (z >> 2));
  const sorted = stableRadixSortU32([
    ...cells.map(({ coordinate }) => identity(coordinate)),
    ...queriedEmpty.map(identity),
  ]);
  const runs = sorted.filter((value, index) => index === 0 || sorted[index - 1] !== value);
  assert.deepEqual(runs, directory.bricks.map(({ dense }) => dense));
  const emptyRuns = directory.bricks.filter(({ masks }) => masks[0] === 0 && masks[1] === 0);
  assert.deepEqual(emptyRuns.map(({ dense }) => dense), [identity([9, 1, 1]), identity([13, 5, 1])]);
  assert.deepEqual(directory.pages, [0, 11],
    "queried-empty bricks stamp misses but never materialize physical pages");
});

test("radix cutover enumerates exact claim-neighbour identities without dense launches", () => {
  const append = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn appendCandidateDirectoryIdentities"),
    octreeSPGridVCycleShader.indexOf("fn brickOfIndex"),
  );
  assert.match(octreeSPGridVCycleShader,
    /@compute @workgroup_size\(1\) fn appendCandidateDirectoryIdentities/);
  assert.match(append, /let n=cCount\(l\)/);
  assert.match(append, /ordinal<7u/,
    "each claimed identity contributes itself and the six directory queries");
  assert.doesNotMatch(append, /totalBrickCount|brickCount\(l\)|logicalPageCount/,
    "identity publication must walk live claims, never dense directory capacity");
  const source = readFileSync(new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
  const setup = source.slice(source.indexOf("private encodeSetupCandidate"),
    source.indexOf("encodeReadySetupCommit"));
  assert.match(setup, /brickSort\.encode\(broker\)/);
  assert.match(setup, /pageSort\.encode\(broker\)/);
  assert.match(setup, /runExternalIndirect[\s\S]*brickSort\.liveDispatch/);
  assert.match(setup, /runExternalIndirect[\s\S]*pageSort\.liveDispatch/);
  assert.match(setup,
    /if \(!this\.touchedDirectoryEnabled \|\| this\.touchedDirectoryTripwire\)[\s\S]*markCandidateBrickOccupancy/,
    "the dense builder is encoded only for the legacy path or explicit differential run");
  assert.match(source, /FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE === "1"/);
  assert.match(octreeSPGridVCycleShader, /select\(0u,4u\*runs,touchedTripwire\(\)\)/,
    "a nonzero comparison count proves that the differential tripwire ran");
  const commit = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn commitCandidateTouchedBricks"),
    octreeSPGridVCycleShader.indexOf("fn commitCandidateSlot"),
  );
  assert.match(commit, /let runs=touchedBrickControl\[4\][\s\S]*run<runs/);
  assert.doesNotMatch(commit, /brick<brickCount\(l\)/,
    "compact commit copies only sorted touched runs");
});
