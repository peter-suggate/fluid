import assert from "node:assert/strict";
import test from "node:test";

import {
  buildOctreeAirSupportCompactAuthority,
  lookupOctreeAirSupportCompactAuthority,
  planOctreeAirSupportCompactAuthority,
  stableRadixOrderCompactAuthorityRecords,
} from "../lib/octree-air-support-compact-authority";
import { planOctreeAirVelocitySupportGPU } from "../lib/webgpu-octree-air-velocity-support-gpu";
import { octreeAirSupportCompactAuthorityWGSL } from "../lib/octree-air-support-compact-authority.wgsl";

test("compact authority census has no domain-shaped allocation or cold launch", () => {
  const plan = planOctreeAirSupportCompactAuthority(65_536, 98_304);
  assert.equal(plan.identityCapacity, 163_840);
  assert.equal(plan.hashCapacity, 327_680);
  assert.equal(plan.rowCandidateCapacity, 4_128_768);
  assert.equal(plan.fineCandidateCapacity, 98_304);
  assert.equal(plan.candidateCapacity, 4_227_072);
  assert.equal(plan.radixBlockCapacity, 640);
  assert.equal(plan.radixHistogramWords, 163_840);
  assert.equal(plan.scratchWords, 19_235_168);
  assert.equal(plan.scratchBytes, 76_940_672);
  assert.equal(plan.coldClearItems, 0);
  assert.equal(plan.maximumLiveClearItems, 163_840);
  assert.equal(plan.domainShapedRecords, 0);
  assert.equal(Object.hasOwn(plan, "domainVolume"), false);
  const productionOcean = planOctreeAirVelocitySupportGPU(65_536, 65_536, [320, 96, 80], 256, 98_304);
  assert.equal(productionOcean.scratchBytes, plan.scratchBytes);
  const precedingDenseScratchBytes = 154_739_996;
  assert.ok(precedingDenseScratchBytes / productionOcean.scratchBytes >= 2,
    `compact authority scratch improved only ${(precedingDenseScratchBytes / productionOcean.scratchBytes).toFixed(6)}x`);
});

test("canonical construction is input-order independent and lookup exact", () => {
  const entries = [
    { cell: 91, size: 0, flags: 0x10 },
    { cell: 7, size: 4, flags: 0x20, winner: 31 },
    { cell: 91, size: 8, flags: 0x40, winner: 17 },
    { cell: 7, size: 4, flags: 0x80, winner: 12 },
    { cell: 3, size: 2, flags: 1, directRow: 5 },
    { cell: 20, size: 4, flags: 4 },
  ] as const;
  const forward = buildOctreeAirSupportCompactAuthority(entries, 8);
  const reverse = buildOctreeAirSupportCompactAuthority([...entries].reverse(), 8);
  assert.deepEqual(reverse, forward);
  assert.equal(forward.ok, true);
  if (!forward.ok) return;
  assert.deepEqual(forward.records.map(({ cell, size, flags, winner, directRow }) =>
    ({ cell, size, flags, winner, directRow })), [
    { cell: 3, size: 2, flags: 1, winner: undefined, directRow: 5 },
    { cell: 7, size: 4, flags: 0xa0, winner: 12, directRow: undefined },
    { cell: 20, size: 4, flags: 4, winner: undefined, directRow: undefined },
    { cell: 91, size: 8, flags: 0x50, winner: 17, directRow: undefined },
  ]);
  assert.equal(lookupOctreeAirSupportCompactAuthority(forward, 7)?.winner, 12);
  assert.equal(lookupOctreeAirSupportCompactAuthority(forward, 8), undefined);
  assert.deepEqual(forward.candidates.map(({ cell, winner }) => ({ cell, winner })), [
    { cell: 7, winner: 12 }, { cell: 91, winner: 17 }, { cell: 20, winner: undefined },
  ]);
  assert.deepEqual(stableRadixOrderCompactAuthorityRecords([
    forward.records[2]!, forward.records[0]!, forward.records[3]!, forward.records[1]!,
  ]).map((record) => record.cell), [3, 7, 20, 91]);
});

test("conflicts and bounded-probe exhaustion reject the transaction", () => {
  assert.deepEqual(buildOctreeAirSupportCompactAuthority([
    { cell: 4, size: 2, flags: 1 }, { cell: 4, size: 4, flags: 2 },
  ], 2), { ok: false, reason: "size-mismatch", cell: 4 });
  assert.deepEqual(buildOctreeAirSupportCompactAuthority([
    { cell: 4, size: 2, flags: 1, directRow: 0 },
    { cell: 4, size: 2, flags: 2, directRow: 1 },
  ], 2), { ok: false, reason: "row-mismatch", cell: 4 });
  assert.deepEqual(buildOctreeAirSupportCompactAuthority([
    { cell: 0, size: 1, flags: 1 }, { cell: 4, size: 1, flags: 1 },
  ], 2, 4, 1), { ok: false, reason: "probe-overflow", cell: 4 });
  assert.deepEqual(buildOctreeAirSupportCompactAuthority([
    { cell: 0, size: 1, flags: 1 }, { cell: 1, size: 1, flags: 1 },
  ], 1), { ok: false, reason: "capacity", cell: 1 });
});

test("zero-default winner and direct-row merges are order independent and conflicts fail closed", () => {
  const mergeWinner = (encoded: number, winner: number): number =>
    Math.max(encoded, (0xffff_ffff - winner) >>> 0) >>> 0;
  const decodeWinner = (encoded: number): number | undefined =>
    encoded === 0 ? undefined : (0xffff_ffff - encoded) >>> 0;
  for (const order of [[31, 12, 17], [17, 31, 12], [12, 17, 31]]) {
    assert.equal(decodeWinner(order.reduce(mergeWinner, 0)), 12);
  }
  const mergeDirect = (encoded: number, row: number): { encoded: number; conflict: boolean } => {
    const incoming = row + 1;
    return { encoded: Math.max(encoded, incoming), conflict: encoded !== 0 && encoded !== incoming };
  };
  assert.deepEqual(mergeDirect(0, 5), { encoded: 6, conflict: false });
  assert.deepEqual(mergeDirect(6, 5), { encoded: 6, conflict: false });
  assert.deepEqual(mergeDirect(6, 7), { encoded: 8, conflict: true });
});

test("GPU authority core clears live slots and radix-orders independently of claim order", () => {
  const shader = octreeAirSupportCompactAuthorityWGSL.replace(/\s+/g, "");
  assert.match(shader, /if\(item>=p\.previousLiveCount\)\{return;\}/);
  assert.match(shader, /for\(varprobe=0u;probe<min\(p\.hashCapacity,64u\);probe\+=1u\)/);
  assert.match(shader, /if\(rank>=p\.identityCapacity\)\{fail\(item,ERROR_CAPACITY\)/);
  assert.match(shader, /atomicCompareExchangeWeak\(&scratch\[at\],0u,wanted\)/);
  assert.match(shader, /atomicMax\(&scratch\[at\+3u\],0u-winnerPlusOne\)/);
  assert.doesNotMatch(shader, /waitKey|0u,INVALID/);
  assert.match(shader, /for\(varprior=0u;prior<lane;prior\+=1u\)\{localRank\+=select\(0u,1u,digits\[prior\]==digit\);\}/);
  assert.doesNotMatch(shader, /domainVolume|dimensions|dispatchWorkgroups/);
});
