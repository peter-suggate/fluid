import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createSparseCM12VelocityExtensionInitialWords,
  createSparseCM12VelocityExtensionLayout,
  sparseCM12VelocityExtensionDispatchShape,
  SPARSE_CM12_VELOCITY_EXTENSION_DISPATCH_WIDTH,
} from "../lib/methods/adaptive-volume/sparse-cm12-velocity-extension";
import { createSparseCM12VelocityExtensionWGSL } from
  "../lib/methods/adaptive-volume/sparse-cm12-velocity-extension.wgsl";
import { createSparseCM12TransportPacketAuthorityLayout } from
  "../lib/methods/adaptive-volume/sparse-cm12-transport-packet-authority";

test("VEX cache is resident-bounded and disjoint from masks, depth, and transport scratch", () => {
  for (const brickFineResolution of [4, 8, 16] as const) {
    const layout = createSparseCM12VelocityExtensionLayout({
      baseWords: 129, cellCapacity: 513, packetCapacity: 192,
      brickFineResolution,
    });
    const initial = createSparseCM12VelocityExtensionInitialWords(layout);
    const at = (address: number) => address - layout.headerBaseWords;
    assert.equal(layout.scheduleBaseWords, layout.acceptedDepthBaseWords + 513);
    assert.equal(layout.packetListBaseWords, layout.scheduleBaseWords + 11);
    assert.equal(layout.totalWords, layout.packetListBaseWords + layout.dispatchPacketCount);
    assert.ok(initial.subarray(at(layout.acceptedDepthBaseWords),
      at(layout.scheduleBaseWords)).every(value => value === 0xffff_ffff));
    assert.equal(initial[at(layout.scheduleBaseWords)], 0xffff_ffff);
    assert.equal(initial[at(layout.scheduleBaseWords) + 2], 0);
    assert.equal(initial[at(layout.scheduleBaseWords) + 10], 0xffff_ffff);
    const transport = createSparseCM12TransportPacketAuthorityLayout({
      baseWords: Math.ceil(layout.totalWords / 64) * 64,
      packetCapacity: layout.packetCapacity,
      dispatchPacketCount: layout.dispatchPacketCount,
      dispatchPacketsPerLeaf: layout.dispatchPacketsPerLeaf,
    });
    assert.ok(transport.indirectBaseWords >= layout.totalWords);
    assert.ok(transport.packetListBaseWords >= layout.totalWords);
  }
});

test("VEX indirect rectangles cover large B16 schedules without losing or duplicating packets", () => {
  const width = SPARSE_CM12_VELOCITY_EXTENSION_DISPATCH_WIDTH;
  for (const count of [0, 1, 3050, 24576, 65535, 65536, 262144]) {
    const [x, y, z] = sparseCM12VelocityExtensionDispatchShape(count);
    assert.ok(x >= 1 && x <= width && y >= 1 && y <= width);
    assert.equal(z, 1);
    const visited = new Uint8Array(count);
    for (let row = 0; row < y; row++) for (let column = 0; column < x; column++) {
      const ordinal = column + width * row;
      if (ordinal < count) visited[ordinal]++;
    }
    assert.ok(visited.every(value => value === 1), `coverage for ${count}`);
    assert.ok(x * y >= Math.max(1, count));
  }
  assert.deepEqual(sparseCM12VelocityExtensionDispatchShape(0), [1, 1, 1]);
  assert.throws(() => sparseCM12VelocityExtensionDispatchShape(width * width + 1), RangeError);
});

test("cached VEX rebuilds at topology changes and preserves empty and retired-packet completion", () => {
  const layout = createSparseCM12VelocityExtensionLayout({
    cellCapacity: 1024, packetCapacity: 192, brickFineResolution: 8,
  });
  const source = createSparseCM12VelocityExtensionWGSL({ layout, cacheAcceptedPackets: true });
  const cache = source.slice(source.indexOf("fn beginSparseCM12VelocityExtensionSchedule"),
    source.indexOf("fn cm12ExtensionExpectedMask"));
  assert.match(cache, /CM12_VEX_SCHEDULE\+10u\)!=acceptedTopologySlot\(\)/);
  assert.match(cache, /CM12_VEX_SCHEDULE\+1u\)==0u\)\{return;/);
  assert.match(cache, /acceptedLeafInvocation\(gid.x\)/);
  assert.match(cache, /initialCompact=compact&&cm12ExtensionLoad\(CM12_VEX_SCHEDULE\+1u\)==0u/);
  assert.match(cache, /sweepGroups=max\(1u,select\(cm12ExtensionDispatchPacketCount,count,compact\)\)/);
  assert.match(cache, /cm12ExtensionDispatchPacketCount-cm12ExtensionDispatchPacketCount\/4u/);
  assert.doesNotMatch(cache, /CM12_TPA_|cm12TransportPacketOrdinal/);
  assert.match(source, /dispatchOrdinal=wid.x\+cm12ExtensionDispatchWidth\*wid.y/);
  const sweep = source.slice(source.indexOf("fn advanceVelocityExtensionPackets"));
  assert.match(sweep, /if\(packet==cm12ExtensionInvalid\)\{\s*cm12ExtensionPublishFrameReceipt\(dispatchOrdinal,lane\);return;/);
  const resident = readFileSync(new URL(
    "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.ts", import.meta.url), "utf8");
  assert.match(resident, /scheduleBaseWords \+ 4\),\s*this.transportPacketIndirectArguments!, 0, 24/);
  assert.match(resident, /dispatchWorkgroupsIndirect\(this.transportPacketIndirectArguments!, offset\)/);
});
