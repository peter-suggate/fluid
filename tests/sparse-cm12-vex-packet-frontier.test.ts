import assert from "node:assert/strict";
import test from "node:test";
import {
  createSparseCM12VexPacketFrontierLayout,
  executeSparseCM12VexDirectPacketRecurrence,
  executeSparseCM12VexPacketFrontierOracle,
  reconcileSparseCM12VexPacketRootBatches,
  type SparseCM12VexPacketAddress,
  type SparseCM12VexPacketRootBatch,
} from "../lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier";
import { createSparseCM12VexPacketFrontierWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier.wgsl";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";

test("VXP1 fits its checked overlay and never silently allocates a fallback", () => {
  const layout = createSparseCM12VexPacketFrontierLayout({
    baseWords: 128, availableWords: 3 * 1024, packetCapacity: 128,
  });
  assert.equal(layout.requiredWords, 32 + 17 * 128);
  assert.equal(layout.rootStampBaseWords, layout.frontierA.generationStampBaseWords);
  assert.notEqual(layout.rootPacketListBaseWords, layout.frontierA.packetListBaseWords);
  assert.equal(layout.rootCauseMaskLowBaseWords.length, 6);
  assert.equal(layout.rootCauseMaskHighBaseWords.length, 6);
  assert.throws(() => createSparseCM12VexPacketFrontierLayout({
    baseWords: 0, availableWords: 100, packetCapacity: 128,
  }), /explicit capacity replan/);
});

test("VXP1 three-list overlay fits the measured ocean, mini64, and symmetric capacities", () => {
  for (const [scene, cellCapacity, packetCapacity] of [
    ["ocean", 537_220, 600 * 64],
    ["mini64", 512 * 512, 512 * 64],
    ["symmetric", 32 * 512, 32 * 64],
  ] as const) {
    const layout = createSparseCM12VexPacketFrontierLayout({
      baseWords: 0, availableWords: 3 * cellCapacity, packetCapacity,
    });
    assert.ok(layout.requiredWords <= layout.availableWords, scene);
  }
});

const graph = [
  [1], [0, 2, 4], [1, 3], [2, 5], [1, 5], [3, 4],
] as const;
const address = (cell: number): SparseCM12VexPacketAddress => ({
  packetId: Math.floor(cell / 3), lane: cell % 3,
});
const packetCell = (packet: number, lane: number): number | undefined => {
  const cell = 3 * packet + lane;
  return lane < 3 && cell < graph.length ? cell : undefined;
};
const masks = (...cells: number[]): ReadonlyMap<number, readonly [number, number]> => {
  const result = new Map<number, [number, number]>();
  for (const cell of cells) {
    const at = address(cell);const mask = result.get(at.packetId) ?? [0, 0];
    mask[0] = (mask[0] | (1 << at.lane)) >>> 0;result.set(at.packetId, mask);
  }
  return result;
};
const batch = (cause: number, cells: number[]): SparseCM12VexPacketRootBatch => ({
  generation: 9, topologyGeneration: 4, topologySlot: 1, cause,
  packetMasks: masks(...cells),
});

test("VXP1 merges producer causes per lane and preserves exact depth-zero/blast sets", () => {
  const batches = [batch(1 << 2, [0, 4]), batch(1 << 4, [0, 3])];
  const result = executeSparseCM12VexPacketFrontierOracle({
    cellCapacity: graph.length, packetCapacity: 2,
    batches,
    packetCell: (packet, lane) => packetCell(packet, lane),
    cellAddress: (cell) => address(cell), cellActive: () => true,
    neighbors: (cell) => graph[cell]!, maximumDepth: 2,
  });
  assert.equal(result.rootCount, 3);
  assert.deepEqual([...result.rootStamp], [9, 0, 0, 9, 9, 0]);
  assert.equal(result.rootCause[0], (1 << 2) | (1 << 4));
  assert.equal(result.rootCause[3], 1 << 4);
  assert.equal(result.rootCause[4], 1 << 2);
  assert.deepEqual([...result.blastDepth], [0, 1, 1, 0, 0, 1]);
  assert.equal(result.blastCount, 6);
  assert.deepEqual(result.rootPacketMasks.get(0), [1, 0]);
  assert.deepEqual(result.rootPacketMasks.get(1), [3, 0]);
  assert.deepEqual(result.blastPacketMasks.get(0), [7, 0]);
  assert.deepEqual(result.blastPacketMasks.get(1), [7, 0]);
  const direct = executeSparseCM12VexDirectPacketRecurrence({
    cellCapacity: graph.length, packetCapacity: 2, batches,
    packetCell: (packet, lane) => packetCell(packet, lane), cellActive: () => true,
    expandPacketMask: (sourcePacket, low, high) => {
      const targets = new Set<number>();
      for (let lane = 0; lane < 64; lane += 1) {
        const word = lane < 32 ? low : high;
        if (((word >>> (lane & 31)) & 1) === 0) continue;
        const cell = packetCell(sourcePacket, lane);if (cell === undefined) continue;
        targets.add(cell);for (const neighbor of graph[cell]!) targets.add(neighbor);
      }
      return masks(...targets);
    }, maximumDepth: 2,
  });
  assert.deepEqual([...direct.rootStamp], [...result.rootStamp]);
  assert.deepEqual([...direct.rootCause], [...result.rootCause]);
  assert.deepEqual([...direct.blastStamp], [...result.blastStamp]);
  assert.deepEqual([...direct.blastDepth], [...result.blastDepth]);
  assert.deepEqual([...direct.blastPacketMasks], [...result.blastPacketMasks]);
});

test("VXP1 rejects producer batches from a different TEI transaction", () => {
  const incompatible = { ...batch(1, [0]), topologySlot: 0 as const };
  assert.throws(() => executeSparseCM12VexPacketFrontierOracle({
    cellCapacity: graph.length, packetCapacity: 2,
    batches: [batch(1, [0]), incompatible],
    packetCell: (packet, lane) => packetCell(packet, lane),
    cellAddress: (cell) => address(cell), cellActive: () => true,
    neighbors: (cell) => graph[cell]!,
  }), /different generation\/TEI slots/);
});

test("VXP1 invalidates changed-leaf old lanes and requires post-flip lifecycle roots", () => {
  const pre: SparseCM12VexPacketRootBatch = {
    generation: 11, topologyGeneration: 6, topologySlot: 0, cause: 1 << 2,
    packetMasks: new Map([[0, [1, 0]], [1, [1, 0]]]),
  };
  const post: SparseCM12VexPacketRootBatch = {
    generation: 11, topologyGeneration: 7, topologySlot: 1, cause: 1 << 3,
    packetMasks: new Map([[0, [2, 0]], [1, [1, 0]]]),
  };
  const packetCellBySlot = (packet: number, lane: number, slot: 0 | 1) => {
    if (packet === 0 && lane < 2) return lane;
    if (packet === 1 && lane === 0) return slot === 0 ? 2 : 3;
    return undefined;
  };
  const reconciled = reconcileSparseCM12VexPacketRootBatches({
    targetTopologyGeneration: 7, targetTopologySlot: 1,
    preFlipBatches: [pre], postCommitBatches: [post],
    packetMappingUnchanged: (packet) => packet === 0,
    requiredReplacementCells: [1, 3], topologyReplacementCause: 1 << 3,
    packetCell: packetCellBySlot,
  });
  assert.equal(reconciled.retainedPreFlipPacketCount, 1);
  assert.equal(reconciled.invalidatedPreFlipPacketCount, 1);
  const result = executeSparseCM12VexPacketFrontierOracle({
    cellCapacity: 4, packetCapacity: 2, batches: reconciled.batches,
    packetCell: packetCellBySlot,
    cellAddress: (cell) => cell < 2 ? { packetId: 0, lane: cell }
      : { packetId: 1, lane: 0 },
    cellActive: () => true, neighbors: () => [], maximumDepth: 0,
  });
  assert.equal(result.rootCause[0], 1 << 2);
  // The old slot's packet1/lane0 density cause must never be reinterpreted as cell 3.
  assert.equal(result.rootCause[3], 1 << 3);
  assert.equal(result.rootCause[1], 1 << 3);
  assert.equal(result.rootCause[2], 0);
  assert.throws(() => reconcileSparseCM12VexPacketRootBatches({
    targetTopologyGeneration: 7, targetTopologySlot: 1,
    preFlipBatches: [pre], postCommitBatches: [],
    packetMappingUnchanged: (packet) => packet === 0,
    requiredReplacementCells: [1, 3], topologyReplacementCause: 1 << 3,
    packetCell: packetCellBySlot,
  }), /lacks post-commit replacement root/);
});

test("VXP1 retirement drops the retired packet and roots only final-slot endpoints", () => {
  const pre = { ...batch(1 << 2, [1]), topologySlot: 0 as const };
  const endpoint = { ...batch(1 << 3, [0]), topologyGeneration: 5,
    topologySlot: 1 as const };
  const reconciled = reconcileSparseCM12VexPacketRootBatches({
    targetTopologyGeneration: 5, targetTopologySlot: 1,
    preFlipBatches: [pre], postCommitBatches: [endpoint],
    packetMappingUnchanged: () => false,
    requiredReplacementCells: [0], topologyReplacementCause: 1 << 3,
    packetCell: (packet, lane) => packetCell(packet, lane),
  });
  assert.equal(reconciled.invalidatedPreFlipPacketCount, 1);
  assert.equal(reconciled.replacementRootCount, 1);
});

test("VXP1 WGSL exposes packet producer, frontier, and direct blast-consumer hooks", () => {
  const velocity = createSparseCM12VelocityExtensionLayout({ cellCapacity: 1024 });
  const packet = createSparseCM12VexPacketFrontierLayout({
    baseWords: velocity.frontierABaseWords,
    availableWords: 3 * velocity.cellCapacity,
    packetCapacity: 128,
  });
  const wgsl = createSparseCM12VexPacketFrontierWGSL({
    layout: packet, velocityExtensionLayout: velocity,
    generationExpression: "frameGeneration()",
    topologyGenerationExpression: "acceptedTopologyGeneration()",
    topologySlotExpression: "acceptedTopologySlot()",
    packetCellFunction: "teiPacketCell", cellPacketLaneFunction: "teiCellPacketLane",
    targetCellActiveFunction: "cellActiveInSlot",
    currentCellActiveFunction: "cellActive", rootReceiptFunction: "recordRootReceipt",
    closureReceiptFunction: "recordClosureReceipt",
  });
  assert.match(wgsl, /fn vxp1RecordRootPacketMask/);
  assert.match(wgsl, /fn vxp1RecordCellRoot/);
  assert.match(wgsl, /fn vxp1RecordPendingCellRoot/);
  assert.match(wgsl, /fn vxp1InvalidateChangedRootPacket/);
  assert.match(wgsl, /bridgeSparseCM12LegacyVexRoots/);
  assert.match(wgsl, /fn vxp1MergeFrontierTarget/);
  assert.match(wgsl, /fn vxp1CellInBlast/);
  assert.match(wgsl, /materializeSparseCM12VexPacketRoots/);
  assert.match(wgsl, /compactSparseCM12VexPacketRoots/);
  assert.match(wgsl, /finalizeSparseCM12VexPacketFrontier/);
  assert.doesNotMatch(wgsl, /incidenceBegin|rowTermOffset|global fallback/i);
  assert.match(wgsl, /const vxp1Invalid:u32=0xffffffffu/);
});
