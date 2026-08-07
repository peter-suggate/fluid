import assert from "node:assert/strict";
import test from "node:test";

import { OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import {
  auditOctreeInterfaceBand,
  decodeOctreeInterfaceBandRows,
} from "../lib/octree-interface-band-audit";

/**
 * The audit is the objective statement of Losasso section 6's refinement rule,
 * so it has to be exactly as strict as that rule and no stricter.
 *
 * The rule has two halves. A band about the interface must be refined, which
 * makes "a leaf coarser than one cell holds the zero set" a defect no matter
 * how the leaf got there; and coarsening must increase with distance, which is
 * what the section's fast-marching redistancing assumes when it discards
 * incomplete T-junction directions. Both are read off a published directory,
 * so the risk here is a decode that mislabels rows -- a sign convention flipped
 * on the interval, an origin unpacked in the wrong axis order -- and every case
 * below is written to fail on exactly that.
 */

const HEADER_WORDS = 8;
const ROW_WORDS = 8;

function bits(value: number): number {
  return new Uint32Array(new Float32Array([value]).buffer)[0]!;
}

interface Row {
  readonly origin: readonly [number, number, number];
  readonly size: number;
  readonly minimumPhi: number;
  readonly maximumPhi: number;
  readonly flags?: number;
}

/** Build a PowerCoarseSampleDirectory holding exactly these rows. */
function directoryOf(
  dimensions: readonly [number, number, number],
  cellWidth: number,
  rows: readonly Row[],
): Uint32Array {
  const words = new Uint32Array(HEADER_WORDS + rows.length * ROW_WORDS);
  words[0] = 0x8000_0000;
  words[1] = 1;
  words[2] = rows.length;
  words[4] = dimensions[0];
  words[5] = dimensions[1];
  words[6] = dimensions[2];
  words[7] = bits(cellWidth);
  rows.forEach((row, slot) => {
    const base = HEADER_WORDS + slot * ROW_WORDS;
    const cell = row.origin[0] + dimensions[0] * (row.origin[1] + dimensions[1] * row.origin[2]);
    words[base] = cell + 1;
    words[base + 1] = row.size;
    words[base + 2] = bits(0.5 * (row.minimumPhi + row.maximumPhi));
    words[base + 3] = bits(row.minimumPhi);
    words[base + 4] = bits(row.maximumPhi);
    words[base + 5] = row.flags
      ?? (OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite);
    words[base + 6] = slot;
    words[base + 7] = bits(row.size ** 3);
  });
  return words;
}

const DIMENSIONS = [24, 18, 16] as const;
// A dyadic cell width, so every phi below is exact in f32 and the distance
// buckets are decided by the audit rather than by rounding.
const H = 0.0625;

test("a row's distance to the interface is the smallest magnitude it attains", () => {
  const decoded = decodeOctreeInterfaceBandRows(directoryOf(DIMENSIONS, H, [
    // Wholly submerged: the surface is 2 cells above the row's shallowest phi.
    { origin: [0, 0, 0], size: 1, minimumPhi: -6 * H, maximumPhi: -2 * H },
    // Wholly dry: same distance, other side.
    { origin: [1, 0, 0], size: 1, minimumPhi: 2 * H, maximumPhi: 6 * H },
    // Straddling: distance is zero however deep the interval runs.
    { origin: [2, 0, 0], size: 1, minimumPhi: -9 * H, maximumPhi: 1 * H },
  ]), DIMENSIONS);

  assert.equal(decoded.length, 3);
  assert.deepEqual(decoded.map((row) => row.side), ["water", "air", "interface"]);
  assert.deepEqual(decoded.map((row) => Number(row.distanceCells.toFixed(6))), [2, 2, 0]);
  assert.deepEqual(decoded.map((row) => row.straddles), [false, false, true]);
});

test("row origins unpack in x-fastest order", () => {
  const decoded = decodeOctreeInterfaceBandRows(directoryOf(DIMENSIONS, H, [
    { origin: [3, 5, 7], size: 2, minimumPhi: 1, maximumPhi: 2 },
  ]), DIMENSIONS);
  assert.deepEqual(decoded[0]!.origin, [3, 5, 7]);
});

test("only leaves coarser than one cell count as interface facets", () => {
  const audit = auditOctreeInterfaceBand(directoryOf(DIMENSIONS, H, [
    // Unit leaves are allowed to hold the interface -- that is the point.
    { origin: [0, 4, 0], size: 1, minimumPhi: -H, maximumPhi: H },
    { origin: [1, 4, 0], size: 1, minimumPhi: -H, maximumPhi: H },
    // These are the facets: one interval spread over 8 and 64 cells.
    { origin: [2, 4, 0], size: 2, minimumPhi: -H, maximumPhi: H },
    { origin: [4, 8, 0], size: 4, minimumPhi: -2 * H, maximumPhi: 2 * H },
  ]), DIMENSIONS, 3);

  assert.deepEqual(audit.straddlingRowsBySize, { "1": 2, "2": 1, "4": 1 });
  assert.equal(audit.straddlingCoarseRows, 2);
  assert.equal(audit.straddlingCoarseCells, 2 ** 3 + 4 ** 3);
  assert.equal(audit.straddlingCoarseRowsByOriginY[4], 1);
  assert.equal(audit.straddlingCoarseRowsByOriginY[8], 1);
  assert.equal(audit.worstStraddlingRows[0]!.size, 4);
});

test("band reach is reported per side, as the paper's asymmetry requires", () => {
  const audit = auditOctreeInterfaceBand(directoryOf(DIMENSIONS, H, [
    // Two cells under the surface, coarse: inside a 3-cell band, water side.
    { origin: [0, 0, 0], size: 2, minimumPhi: -5 * H, maximumPhi: -2 * H },
    // Two cells above, coarse: inside the band, air side.
    { origin: [2, 0, 0], size: 2, minimumPhi: 2 * H, maximumPhi: 5 * H },
    // Ten cells under: outside the band, counted nowhere.
    { origin: [4, 0, 0], size: 4, minimumPhi: -20 * H, maximumPhi: -10 * H },
    // Inside the band but already unit size: not a reach failure.
    { origin: [8, 0, 0], size: 1, minimumPhi: -2 * H, maximumPhi: -H },
  ]), DIMENSIONS, 3);

  assert.equal(audit.bandCoarseRowsWaterSide, 1);
  assert.equal(audit.bandCoarseRowsAirSide, 1);
  assert.equal(audit.straddlingCoarseRows, 0);
  assert.equal(audit.bandCoarseRows, 2);
});

test("a coarse row holding the interface counts once, in the band total", () => {
  const audit = auditOctreeInterfaceBand(directoryOf(DIMENSIONS, H, [
    { origin: [0, 0, 0], size: 2, minimumPhi: -H, maximumPhi: H },
    { origin: [2, 0, 0], size: 2, minimumPhi: 2 * H, maximumPhi: 5 * H },
  ]), DIMENSIONS, 3);
  assert.equal(audit.straddlingCoarseRows, 1);
  assert.equal(audit.bandCoarseRowsAirSide, 1);
  assert.equal(audit.bandCoarseRows, 2, "straddling rows must not be double counted");
});

test("coarsening that shrinks again with distance breaks the redistancing assumption", () => {
  const monotone = auditOctreeInterfaceBand(directoryOf(DIMENSIONS, H, [
    { origin: [0, 0, 0], size: 1, minimumPhi: -H, maximumPhi: H },
    { origin: [2, 0, 0], size: 2, minimumPhi: 4 * H, maximumPhi: 6 * H },
    { origin: [4, 0, 0], size: 4, minimumPhi: 9 * H, maximumPhi: 12 * H },
  ]), DIMENSIONS, 1);
  assert.equal(monotone.monotonicityBreaks, 0);
  assert.deepEqual(monotone.maximumSizeByDistanceCells.slice(0, 10),
    [1, 0, 0, 0, 2, 0, 0, 0, 0, 4]);

  const inverted = auditOctreeInterfaceBand(directoryOf(DIMENSIONS, H, [
    { origin: [0, 0, 0], size: 4, minimumPhi: -H, maximumPhi: H },
    { origin: [4, 0, 0], size: 1, minimumPhi: 9 * H, maximumPhi: 12 * H },
  ]), DIMENSIONS, 1);
  assert.equal(inverted.monotonicityBreaks, 1);
});

test("invalid and non-finite rows are skipped, and the flag is cross-checked", () => {
  const audit = auditOctreeInterfaceBand(directoryOf(DIMENSIONS, H, [
    { origin: [0, 0, 0], size: 2, minimumPhi: -H, maximumPhi: H, flags: 0 },
    {
      origin: [2, 0, 0], size: 2, minimumPhi: -H, maximumPhi: H,
      flags: OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite
        | OCTREE_COARSE_PHI_FLAG.containsInterface,
    },
  ]), DIMENSIONS, 3);
  assert.equal(audit.rows, 1, "a row without valid|finite carries no evidence");
  assert.equal(audit.straddlingCoarseRows, 1);
  assert.equal(audit.flaggedInterfaceRows, 1);
});

test("a malformed directory is rejected rather than scored", () => {
  assert.throws(() => decodeOctreeInterfaceBandRows(new Uint32Array(4), DIMENSIONS),
    /truncated/);
  const badWidth = directoryOf(DIMENSIONS, H, []);
  badWidth[7] = bits(0);
  assert.throws(() => decodeOctreeInterfaceBandRows(badWidth, DIMENSIONS), /cell width/);
  const badCount = directoryOf(DIMENSIONS, H, []);
  badCount[2] = 3;
  assert.throws(() => decodeOctreeInterfaceBandRows(badCount, DIMENSIONS), /row count/);
});
