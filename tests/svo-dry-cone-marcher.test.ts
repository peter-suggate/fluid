import assert from "node:assert/strict";
import test from "node:test";

import { encodeSvoNodeMipMorton, planSvoNodeMipPyramid, type SvoNodeMipCoordinate } from "../lib/svo-node-mip-pyramid";
import { createSvoDryConeMarcherWGSL, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

/** CPU mirror of the WGSL dryNodeMipSpreadMortonBits bit expansion. */
function spreadMortonBits(value: number): number {
  let x = value >>> 0;
  x = (x | (x << 16)) & 0xff0000ff;
  x = (x | (x << 8)) & 0x0f00f00f;
  x = (x | (x << 4)) & 0xc30c30c3;
  x = (x | (x << 2)) & 0x49249249;
  return x >>> 0;
}

/** CPU mirror of the branchless WGSL dryNodeMipMorton encode. */
function branchlessMorton(coordinate: SvoNodeMipCoordinate): [number, number] {
  const [x, y, z] = coordinate.map((component) => (component & 0x1fffff) >>> 0);
  const low = (spreadMortonBits(x & 0x7ff) | (spreadMortonBits(y & 0x7ff) << 1) | (spreadMortonBits(z & 0x3ff) << 2)) >>> 0;
  const high = ((spreadMortonBits(x >>> 11) << 1) | (spreadMortonBits(y >>> 11) << 2) | spreadMortonBits(z >>> 10)) >>> 0;
  return [low, high];
}

/** CPU mirror of the WGSL reference-loop dryNodeMipMorton encode. */
function referenceLoopMorton(coordinate: SvoNodeMipCoordinate): [number, number] {
  let low = 0, high = 0;
  for (let bit = 0; bit < 21; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const outputBit = bit * 3 + axis;
      const value = (coordinate[axis] >>> bit) & 1;
      if (outputBit < 32) low = (low | (value << outputBit)) >>> 0;
      else high = (high | (value << (outputBit - 32))) >>> 0;
    }
  }
  return [low, high];
}

test("branchless Morton spread equals the reference loop and the authoritative bigint encoder", () => {
  // Axis-edge values around every word/mask boundary the expansion depends on:
  // 11-bit low/high split for x/y, the 10-bit split for z, and the 21-bit cap.
  const edges = [0, 1, 2, 3, 7, 8, 0x3ff, 0x400, 0x7fe, 0x7ff, 0x800, 0x801, 0xfffff, 0x100000, 0x1ffffe, 0x1fffff];
  for (const x of edges) for (const y of edges) for (const z of edges) {
    const coordinate = [x, y, z] as const;
    const viaSpread = branchlessMorton(coordinate);
    assert.deepEqual(viaSpread, referenceLoopMorton(coordinate), `loop parity at ${coordinate}`);
    const authoritative = encodeSvoNodeMipMorton(coordinate);
    assert.equal(BigInt(viaSpread[0]) | (BigInt(viaSpread[1]) << 32n), authoritative, `bigint parity at ${coordinate}`);
  }
  // Deterministic pseudo-random sweep over the full 21-bit range.
  let state = 0x12345678;
  const next = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state & 0x1fffff;
  };
  for (let index = 0; index < 4096; index += 1) {
    const coordinate = [next(), next(), next()] as const;
    assert.deepEqual(branchlessMorton(coordinate), referenceLoopMorton(coordinate), `random parity at ${coordinate}`);
  }
});

interface DirectoryRow { level: number; mortonLow: number; mortonHigh: number }

function compareRow(row: DirectoryRow, level: number, morton: [number, number]): number {
  if (row.level !== level) return row.level < level ? -1 : 1;
  if (row.mortonHigh !== morton[1]) return row.mortonHigh < morton[1] ? -1 : 1;
  if (row.mortonLow !== morton[0]) return row.mortonLow < morton[0] ? -1 : 1;
  return 0;
}

/** Mirrors the WGSL 24-iteration lower_bound over [low, high). */
function lowerBound(rows: DirectoryRow[], level: number, morton: [number, number], low: number, high: number): number {
  for (let iteration = 0; iteration < 24 && low < high; iteration += 1) {
    const middle = low + ((high - low) >> 1);
    if (compareRow(rows[middle], level, morton) < 0) low = middle + 1;
    else high = middle;
  }
  return low;
}

test("per-level windowed directory search returns the identical row as the full-range search", () => {
  const occupied: SvoNodeMipCoordinate[] = [];
  for (let x = 0; x < 6; x += 1) for (let z = 0; z < 6; z += 1) occupied.push([x * 3, 0, z * 2]);
  occupied.push([100, 40, 7], [1023, 511, 255], [2047, 2047, 2047]);
  const plan = planSvoNodeMipPyramid({ generation: 9, occupiedPages: occupied, levelCount: 12 });
  const rows: DirectoryRow[] = plan.pages.map(({ key }) => {
    const morton = encodeSvoNodeMipMorton(key.coordinate);
    return { level: key.level, mortonLow: Number(morton & 0xffffffffn), mortonHigh: Number(morton >> 32n) };
  });
  // The directory row index is the sorted (level, morton) position, so boundary
  // i (count of pages with level < i) brackets each level's contiguous run.
  plan.pages.forEach((page, index) => assert.equal(page.slot, index));
  const levelStart = new Uint32Array(12);
  for (const page of plan.pages) if (page.key.level < 11) levelStart[page.key.level + 1] += 1;
  for (let boundary = 1; boundary < levelStart.length; boundary += 1) levelStart[boundary] += levelStart[boundary - 1];

  const queries: Array<{ level: number; coordinate: SvoNodeMipCoordinate }> = [];
  for (const { key } of plan.pages) {
    queries.push({ level: key.level, coordinate: key.coordinate });
    queries.push({ level: key.level, coordinate: [key.coordinate[0] + 1, key.coordinate[1], key.coordinate[2]] });
    queries.push({ level: Math.min(key.level + 1, 11), coordinate: key.coordinate });
  }
  queries.push({ level: 0, coordinate: [0x1fffff, 0x1fffff, 0x1fffff] }, { level: 11, coordinate: [0, 0, 0] }, { level: 13, coordinate: [1, 1, 1] });
  for (const { level, coordinate } of queries) {
    const morton = branchlessMorton(coordinate);
    const full = lowerBound(rows, level, morton, 0, rows.length);
    const low = levelStart[Math.min(level, 11)];
    const high = level < 11 ? levelStart[level + 1] : rows.length;
    assert.equal(lowerBound(rows, level, morton, low, high), full, `windowed lower_bound diverged at level ${level} ${coordinate}`);
  }
});

test("the production shader embeds the optimized marcher; the baseline variant keeps the reference text", () => {
  // Production ships branchless Morton + ranged directory search. Empty-space
  // elision stays benchmark-only: tools/benchmark-svo-cone-gpu.ts measured it
  // slower than the subset (cone LOD samples empty space through
  // ancestor-resident coarse pages where the non-residency proof never fires).
  const optimized = createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true });
  assert.ok(svoDrySceneShader.includes(optimized), "production dry shader must embed the morton+ranged marcher block");
  assert.match(optimized, /0xff0000ffu[^]*0x0f00f00fu[^]*0xc30c30c3u[^]*0x49249249u/);
  assert.match(optimized, /dryNodeMipLevelStart\(level\)/);
  assert.doesNotMatch(optimized, /nodeMipLevelStart\[clamped>>2u\]/,
    "a dynamically indexed uniform array trips a slow-path Metal transform; keep constant vector indexing");
  const elision = createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true, emptySpaceElision: true });
  assert.match(elision, /dryConeZeroRegionAt\(/);
  const baseline = createSvoDryConeMarcherWGSL();
  assert.match(baseline, /for\(var bit=0u;bit<21u;bit\+=1u\)/);
  assert.match(baseline, /var low=0u;var high=dry\.nodeMip\.y;/);
  assert.doesNotMatch(baseline, /dryConeZeroRegionAt/);
  for (const variant of [optimized, elision, baseline]) {
    assert.match(variant, /fn dryNodeMipAt\(position_m:vec3f,lodIn:f32,pageCache:ptr<function,DryNodeMipPageCache>\)->DryNodeMipLookup\{/);
    assert.match(variant, /fn dryNodeMipReady\(\)->bool\{return dry\.nodeMip\.w!=0u&&dry\.nodeMip\.x!=0u&&dry\.nodeMip\.x==publicationState\[2\]&&dry\.nodeMip\.y>0u&&dry\.nodeMip\.z>0u;\}/);
    assert.match(variant, /dryMipSteps\+=1u/);
  }
});
