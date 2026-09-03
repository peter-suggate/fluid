import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createSparseCM12TransportPacketAuthorityLayout,
  sparseCM12TransportPacketLaneCell,
  SPARSE_CM12_TRANSPORT_PACKET_INVALID,
} from "../lib/methods/adaptive-mass/sparse-cm12-transport-packet-authority";
import { createSparseCM12TransportPacketAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-packet-authority.wgsl";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { createSparseCM12VelocityExtensionWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl";

const wgsl = createSparseCM12TransportPacketAuthorityWGSL({
  layout: createSparseCM12TransportPacketAuthorityLayout({
    baseWords: 256,
    packetCapacity: 512,
    dispatchPacketsPerLeaf: 8,
    dispatchPacketCount: 64,
  }),
});
const packedWGSL = createSparseCM12TransportPacketAuthorityWGSL({
  layout: createSparseCM12TransportPacketAuthorityLayout({
    baseWords: 256,
    packetCapacity: 512,
    dispatchPacketsPerLeaf: 8,
    dispatchPacketCount: 64,
  }),
  packCoarseTransportCells: true,
});

test("TPA1 stages one sealed descriptor for packet-lane execution", () => {
  assert.match(wgsl, /var<workgroup>cm12TransportStagedPacket:CM12TransportPacket;/);
  assert.match(wgsl, /var<workgroup>cm12TransportStagedPacketMask:vec2u;/);
  const begin = wgsl.indexOf("fn cm12StageTransportPacket");
  const end = wgsl.indexOf("fn cm12TransportStagedExecutionCell", begin);
  assert.ok(begin >= 0 && end > begin);
  const stage = wgsl.slice(begin, end);
  assert.match(stage, /if\(lane==0u\)/);
  assert.equal(stage.match(/cm12TransportPacketOrdinal\(/g)?.length, 1);
  assert.equal(stage.match(/cm12TransportPacketMaskAt\(/g)?.length, 1);
  assert.equal(stage.match(/cm12TeiPacket\(/g)?.length, 1);
  assert.equal(stage.match(/cm12TeiPacketFineOrigin\(/g)?.length, 1);
  assert.equal(stage.match(/acceptedTopologySlot\(/g)?.length, 1);
  assert.match(stage, /cm12TransportStagedPacketOriginFine=cm12TeiPacketFineOrigin/);
  assert.match(stage, /workgroupBarrier\(\)/);

  const cellEnd = wgsl.indexOf("// Compile the prior frame", end);
  const cell = wgsl.slice(end, cellEnd);
  assert.match(cell, /packet\.first\+q\.x\+packet\.strideY\*q\.y\+packet\.strideZ\*q\.z/);
  assert.doesNotMatch(cell,
    /atomicLoad|cm12TransportPacketId|cm12TransportPacketMask|cm12TeiPacket\(/,
    "hot lanes must consume only the staged descriptor");
  assert.doesNotMatch(cell, /owner|OpenVolume|cellOpen/i);
});

test("accepted conservative passes consume the staged packet prologue", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf("fn stageSparseCM12TransportExecutionImage");
  const end = source.indexOf("fn sharpeningSourceCell", begin);
  assert.ok(begin >= 0 && end > begin);
  const stage = source.slice(begin, end);
  assert.match(stage, /cm12StageTransportPacket\(rank,lane\)/);
  assert.match(stage, /cm12TransportStagedExecutionCell\(lane\)/);
  assert.match(stage, /cm12TransportStagedPacketOriginFine/);
  assert.match(stage, /cm12TransportStagedTopologySlot/);
  assert.doesNotMatch(stage,
    /cm12TransportPacketId\(|cm12TeiPacketFineOrigin\(|acceptedTopologySlot\(\)|cm12TransportExecutionCell\(/,
    "the hot prologue must not reload sealed packet authority per lane");
  assert.doesNotMatch(stage, /ownerCellAt|cellOpenVolume/);

  const gatherBegin = source.indexOf("fn gatherConservativeDensity");
  const gatherEnd = source.indexOf("fn tracerCount", gatherBegin);
  const gather = source.slice(gatherBegin, gatherEnd);
  assert.doesNotMatch(gather, /cm12TransportPublish.*Mask/,
    "gather must not publish a second topology representation");
  assert.doesNotMatch(gather, /cm12TransportPacketId\(/);
});

test("accepted VEX packet compilation scales with accepted leaves and their rungs", () => {
  const begin = wgsl.indexOf("fn compileSparseCM12AcceptedVelocityExtensionPackets");
  const end = wgsl.indexOf("fn cm12TransportPacketMaskAt", begin);
  assert.ok(begin >= 0 && end > begin);
  const compile = wgsl.slice(begin, end);
  assert.match(compile, /acceptedLeafInvocation\(gid\.x\)/);
  assert.match(compile, /resolution=descriptor\.flags&31u/);
  assert.match(compile, /packetAxis\*packetAxis\*packetAxis/);
  assert.match(compile, /atomicAdd\(&activity\[CM12_TPA_INDIRECT\],1u\)/);
  assert.doesNotMatch(compile, /CM12_TPA_DIRECT_PACKET_COUNT\)\{return;/,
    "accepted packet compilation must not enumerate direct packet capacity");

  const vex = createSparseCM12VelocityExtensionWGSL({
    layout: createSparseCM12VelocityExtensionLayout({
      cellCapacity: 4096, packetCapacity: 512, brickFineResolution: 8,
    }),
    compactAcceptedPacketsForQA: true,
  });
  assert.match(vex, /cm12TransportPacketOrdinal\(dispatchOrdinal\)/);
  assert.match(vex, /workgroupUniformLoad\(&cm12ExtensionDispatchPacket\)/);
  assert.match(vex, /cm12ExtensionPublishFrameReceipt\(dispatchOrdinal,lane\)/);
});

test("coarse transport selection is GPU-authored from dirty packet work", () => {
  assert.match(packedWGSL, /CM12_TPA_PACK_COARSE_CAPABLE:bool=true/);
  assert.match(packedWGSL, /fn finalizeSparseCM12CoarseTransportSchedule\(\)/);
  assert.match(packedWGSL, /dirtyCoarse>=minimumPackets/);
  assert.match(packedWGSL, /acceptedTemplateCellWorkgroups\(\),enabled/);
  assert.match(packedWGSL, /fn compileSparseCM12TransportPacketLists/);
  assert.match(packedWGSL, /CM12_TPA_SHARPENING_PACKET_LIST/);

  const lists = packedWGSL.slice(
    packedWGSL.indexOf("fn compileSparseCM12TransportPacketLists"),
    packedWGSL.indexOf("fn cm12TransportExecutionCell"),
  );
  assert.match(lists,
    /cm12CoarseTransportPackingEnabled\(\)&&\(descriptor\.flags&31u\)<=4u\)\{return;\}/,
    "packing may remove only B1/B2/B4 packets from conservative transport");
  assert.match(lists,
    /atomicStore\(&activity\[CM12_TPA_PACKET_LIST\+transportAt\],ordinal\)/,
    "B8 and fallback packets must retain their staged TEI execution list");
});

test("packed coarse transport preserves stable accepted-cell order and B8 locality", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf("fn cm12PackedCoarseCell");
  const end = source.indexOf("fn cm12PublishPackedCoarseSharpening", begin);
  assert.ok(begin >= 0 && end > begin);
  const resolve = source.slice(begin, end);
  assert.match(resolve, /acceptedTemplateCellInvocation\(invocation\)/,
    "packed lanes must be a deterministic projection of accepted-cell order");
  assert.match(resolve, /if\(resolution>4u\)/,
    "B8 cells must not enter the cross-leaf execution path");
  assert.match(resolve, /cm12TransportPacketLaneSelected/,
    "coarse cells must still consume the sealed packet dirty mask");

  for (const entryPoint of [
    "traceGammaAndBetaPackedCoarse",
    "scatterDensityDeficitPackedCoarse",
    "gatherConservativeDensityPackedCoarse",
  ]) {
    assert.match(source, new RegExp(`@compute @workgroup_size\\(64\\)\\nfn ${entryPoint}`));
  }
});

test("staged packet arithmetic is equivalent to direct packet decode", () => {
  const masks = [
    [0xffff_ffff, 0xffff_ffff],
    [0xaaaa_aaaa, 0x5555_5555],
    [0x0000_0001, 0x8000_0000],
  ] as const;
  for (let xCount = 1; xCount <= 4; xCount += 1) {
    for (let yCount = 1; yCount <= 4; yCount += 1) {
      for (let zCount = 1; zCount <= 4; zCount += 1) {
        const strideY = xCount + 3;
        const strideZ = strideY * (yCount + 2);
        for (const [maskLow, maskHigh] of masks) {
          for (let lane = 0; lane < 64; lane += 1) {
            const mask = lane < 32 ? maskLow : maskHigh;
            const selected = ((mask >>> (lane & 31)) & 1) !== 0;
            const x = lane & 3;
            const y = (lane >>> 2) & 3;
            const z = lane >>> 4;
            const expected = selected && x < xCount && y < yCount && z < zCount
              ? 700 + x + strideY * y + strideZ * z
              : SPARSE_CM12_TRANSPORT_PACKET_INVALID;
            assert.equal(sparseCM12TransportPacketLaneCell({
              first: 700,
              counts: [xCount, yCount, zCount],
              strideY, strideZ, maskLow, maskHigh, lane,
            }), expected);
          }
        }
      }
    }
  }
  assert.equal(sparseCM12TransportPacketLaneCell({
    first: SPARSE_CM12_TRANSPORT_PACKET_INVALID,
    counts: [4, 4, 4], strideY: 4, strideZ: 16,
    maskLow: 0xffff_ffff, maskHigh: 0xffff_ffff, lane: 0,
  }), SPARSE_CM12_TRANSPORT_PACKET_INVALID);
});
