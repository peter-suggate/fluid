import assert from "node:assert/strict";
import test from "node:test";
import {
  SPARSE_CM12_DYNAMIC_CLOSURE_ADDRESSING,
  SPARSE_CM12_DYNAMIC_CLOSURE_HEADER,
  SPARSE_CM12_DYNAMIC_CLOSURE_DISPATCH_ORDER,
  SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION,
  SPARSE_CM12_DYNAMIC_CLOSURE_MAGIC,
  compileSparseCM12DynamicClosureReference,
  createSparseCM12DynamicClosureInitialWords,
  createSparseCM12DynamicClosureLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-dynamic-closure-authority";
import { createSparseCM12DynamicClosureAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-dynamic-closure-authority.wgsl";

test("DCA1 sizes append-only sparse lists and generation-stamped target planes", () => {
  const layout = createSparseCM12DynamicClosureLayout({
    sourcePacketCapacity: 130, targetPacketCapacity: 257, baseWords: 3,
  });
  for (const offset of [layout.baseWords, layout.surfaceListBaseWords,
    layout.densityListBaseWords, layout.rowStampBaseWords, layout.rowMaskBaseWords,
    layout.rowTouchedBaseWords, layout.cellStampBaseWords, layout.cellMaskBaseWords,
    layout.cellTouchedBaseWords, layout.indirectBaseWords, layout.totalWords]) {
    assert.equal(offset % 64, 0);
  }
  assert.ok(layout.rowTouchedBaseWords >= layout.rowMaskBaseWords + 6 * 257);
  assert.ok(layout.cellTouchedBaseWords >= layout.cellMaskBaseWords + 2 * 257);
  const words = createSparseCM12DynamicClosureInitialWords(layout);
  const h = SPARSE_CM12_DYNAMIC_CLOSURE_HEADER;
  assert.equal(words[h.magic], SPARSE_CM12_DYNAMIC_CLOSURE_MAGIC);
  assert.equal(words[h.sourceCapacity], 130);
  assert.equal(words[h.targetCapacity], 257);
  assert.equal(words[h.rowMaskBase], layout.rowMaskBaseWords);
  assert.equal(words.length * 4, layout.totalBytes);
});

test("DCA1 oracle appends only nonzero sources and ORs exact touched masks", () => {
  const receipt = compileSparseCM12DynamicClosureReference({
    producers: [
      { packetId: 1, surfaceLow: 0b11, surfaceHigh: 0,
        densityLow: 0, densityHigh: 0 },
      { packetId: 2, surfaceLow: 0, surfaceHigh: 0,
        densityLow: 1, densityHigh: 0 },
      { packetId: 3, surfaceLow: 4, surfaceHigh: 0,
        densityLow: 2, densityHigh: 0 },
    ],
    tra: (packet, low) => packet === 1 ? [
      { packetId: 8, axis: 0, low, high: 0 },
      { packetId: 9, axis: 2, low: 8, high: 0 },
    ] : [{ packetId: 8, axis: 0, low: low << 1, high: 0 }],
    vex: (packet, low) => [{ packetId: 20, low: low << (packet - 2), high: 0 }],
  });
  assert.deepEqual(receipt.surfaceSources, [1, 3]);
  assert.deepEqual(receipt.densitySources, [2, 3]);
  assert.deepEqual(receipt.rowTouched, [8, 9]);
  assert.deepEqual(receipt.rowMasks.get(8), [0b1011, 0, 0, 0, 0, 0]);
  assert.deepEqual(receipt.rowMasks.get(9), [0, 0, 0, 0, 8, 0]);
  assert.deepEqual(receipt.cellTouched, [20]);
  assert.deepEqual(receipt.cellMasks.get(20), [5, 0]);
});

test("DCA1 WGSL is sparse, indirect and spatial-row-packet based", () => {
  const layout = createSparseCM12DynamicClosureLayout({ sourcePacketCapacity: 512 });
  const source = createSparseCM12DynamicClosureAuthorityWGSL({
    layout, arenaName: "activity", hookPrefix: "testResident",
  });
  assert.match(source, /fn cm12DynamicClosurePublishSourcePacket/);
  assert.match(source, /if\(\(surface\.x\|surface\.y\)!=0u\)/);
  assert.match(source, /fn dca1MarkRowMask\(packet:u32,axis:u32,low:u32,high:u32\)/);
  assert.match(source, /atomicCompareExchangeWeak\(&activity\[DCA1_ROW_STAMP\+packet\]/);
  assert.match(source, /fn clearSparseCM12DynamicRows/);
  assert.match(source, /if\(stamp!=0u\).*targetNotCleared|if\(stamp!=0u\)/s);
  assert.match(source, /compileSparseCM12DynamicTRA/);
  assert.match(source, /4u\*wid\.x\+lane\/16u/);
  assert.match(source, /scatterSparseCM12DynamicGammaSnapshotRows/);
  assert.match(source, /scatterSparseCM12DynamicGammaRefinementRows/);
  assert.match(source, /@workgroup_size\(256\)/);
  assert.match(source, /var<workgroup>dca1ScatterMasks:array<u32,24>/);
  assert.match(source, /cellLane<6u.*DCA1_ROW_MASK\+cellLane\*DCA1_TARGET_CAP/s);
  assert.match(source, /let low=dca1ScatterMasks\[6u\*group\+2u\*axis\]/);
  assert.match(source,
    /testResidentDynamicClosureScatterGammaSnapshotRow\(packet,axis,cellLane\)/);
  assert.match(source,
    /testResidentDynamicClosureScatterGammaRefinementRow\(packet,axis,cellLane\)/);
  assert.match(source, /seedSparseCM12DynamicVEXFrontier/);
  assert.match(source, /testResidentDynamicClosureSeedVEXFrontier\(packet,vec2u\(low,high\),worker\)/);
  assert.doesNotMatch(source, /TRA1|MarkRow\(row|ExtensionRecordRoot/);
  assert.doesNotMatch(source, /IncidenceBegin|IncidenceRow|for\(var row=0u|fallback/i);
});

test("DCA1 clears only prior touched packets before reusing generation stamps", () => {
  assert.deepEqual(SPARSE_CM12_DYNAMIC_CLOSURE_DISPATCH_ORDER, [
    "clearSparseCM12DynamicRows + clearSparseCM12DynamicCells (prior target indirects)",
    "beginSparseCM12DynamicClosure",
    "gather publishes nonzero source packets",
    "sealSparseCM12DynamicClosureSources",
    "compileSparseCM12DynamicTRA + compileSparseCM12DynamicVEX (source indirects)",
    "sealSparseCM12DynamicClosureTargets",
    "scatterSparseCM12DynamicGammaSnapshotRows",
    "resident finalizeGammaSnapshot",
    "scatterSparseCM12DynamicGammaRefinementRows",
    "resident finalizeGammaRefinement + seedSparseCM12DynamicVEXFrontier",
  ]);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION.intra, /axisBase/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION.equalFace, /interface base/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION.mixedSeam, /canonical interface patch/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION.publication, /gamma scatter/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION.snapshot, /destination/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION.refinement, /refinement input/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ROW_RESOLUTION.lifetime,
    /persist through both ordered scatters and clear next frame/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ADDRESSING.packet, /stable TEI packet id/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ADDRESSING.gather, /never the compact gather work rank/);
  assert.match(SPARSE_CM12_DYNAMIC_CLOSURE_ADDRESSING.gather, /staged stable packet id/);
});

test("DCA1 rejects zero and invalid capacities", () => {
  assert.throws(() => createSparseCM12DynamicClosureLayout({ sourcePacketCapacity: 0 }),
    /positive/);
  assert.throws(() => createSparseCM12DynamicClosureLayout({
    sourcePacketCapacity: 64, targetPacketCapacity: 0,
  }), /positive/);
});
