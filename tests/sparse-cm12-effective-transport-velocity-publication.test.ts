import assert from "node:assert/strict";
import test from "node:test";

import {
  createSparseCM12EffectiveTransportVelocityLayout,
  publishSparseCM12CollocatedWetVelocity,
  publishSparseCM12VexAcceptedVelocity,
  seedSparseCM12EffectiveTransportVelocity,
} from "../lib/methods/adaptive-mass/sparse-cm12-effective-transport-velocity";
import { createSparseCM12EffectiveTransportVelocityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-effective-transport-velocity.wgsl";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import { createSparseCM12VelocityExtensionWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl";

const bits = (values: Float32Array) => new Uint32Array(
  values.buffer, values.byteOffset, values.length,
);

test("effective velocity layout is one tightly packed vec4f per cell", () => {
  assert.deepEqual(createSparseCM12EffectiveTransportVelocityLayout(7), {
    cellCapacity: 7, vectorStrideBytes: 16, floatCount: 28, byteLength: 112,
  });
  assert.throws(() => createSparseCM12EffectiveTransportVelocityLayout(0),
    /capacity/);
});

test("construction, VEX acceptance, and wet collocation preserve exact f32 bits", () => {
  const layout = createSparseCM12EffectiveTransportVelocityLayout(4);
  const source = new Uint32Array([
    0x8000_0000, 0x7fc0_1234, 0x3f80_0001, 0x3f80_0000,
    0x3f00_0000, 0xbf00_0000, 0x0000_0001, 0x0000_0000,
  ]);
  const sourceFloats = new Float32Array(source.buffer);
  const plane = seedSparseCM12EffectiveTransportVelocity(layout, [0, 2], (cell) => {
    const at = cell === 0 ? 0 : 4;
    return [sourceFloats[at]!, sourceFloats[at + 1]!,
      sourceFloats[at + 2]!, sourceFloats[at + 3]!] as const;
  });
  assert.deepEqual([...bits(plane).slice(0, 4)], [...source.slice(0, 4)]);
  assert.deepEqual([...bits(plane).slice(8, 12)], [...source.slice(4, 8)]);

  const priorDry = bits(plane).slice(8, 12);
  publishSparseCM12CollocatedWetVelocity(plane, 2, [9, 8, 7], false);
  assert.deepEqual([...bits(plane).slice(8, 12)], [...priorDry],
    "dry collocation must retain the prior accepted effective value");

  publishSparseCM12CollocatedWetVelocity(plane, 2, [-0, 8, 7], true);
  assert.deepEqual([...bits(plane).slice(8, 12)],
    [0x8000_0000, 0x4100_0000, 0x40e0_0000, 0x3f80_0000]);
  publishSparseCM12VexAcceptedVelocity(plane, 1, [1.25, -2.5, 3.75, 0]);
  assert.deepEqual([...plane.slice(4, 8)], [1.25, -2.5, 3.75, 0]);
});

test("WGSL plane helpers use native vec4 loads and producer-only stores", () => {
  const wgsl = createSparseCM12EffectiveTransportVelocityWGSL({
    layout: createSparseCM12EffectiveTransportVelocityLayout(32),
    planeName: "partials",
  });
  assert.match(wgsl,
    /fn cm12EffectiveTransportVelocity\(cell:u32\)->vec4f\{\s*return partials\[cell\];/);
  assert.match(wgsl,
    /fn cm12PublishVexAcceptedEffectiveVelocity[\s\S]*partials\[cell\]=value/);
  assert.match(wgsl,
    /fn cm12PublishCollocatedWetEffectiveVelocity[\s\S]*if\(wet[\s\S]*vec4f\(velocity,1\.0\)/);
  assert.match(wgsl, /fn seedSparseCM12EffectiveTransportVelocity/);
  assert.doesNotMatch(wgsl, /state\[/,
    "the hot read/write helpers must address only the dedicated vec4 plane");
});

test("VEX acceptance hook publishes only after accepted cache metadata", () => {
  const layout = createSparseCM12VelocityExtensionLayout({
    baseWords: 16, cellCapacity: 8,
  });
  const wgsl = createSparseCM12VelocityExtensionWGSL({
    layout, acceptedVelocityFloatBase: 64,
    effectiveVelocityHookPrefix: "cm12",
  });
  const commit = wgsl.slice(wgsl.indexOf("fn commitVelocityExtensionCandidates"),
    wgsl.indexOf("fn finalizeVelocityExtensionCandidate"));
  const ownerAt = commit.indexOf("cm12ExtensionAcceptedOwner+cell");
  const publishAt = commit.indexOf("cm12PublishVexAcceptedEffectiveVelocity");
  const receiptAt = commit.indexOf("atomicAdd", publishAt);
  assert.ok(ownerAt >= 0 && publishAt > ownerAt && receiptAt > publishAt);
  assert.match(commit,
    /cm12PublishVexAcceptedEffectiveVelocity\(cell,vec4f\([\s\S]*state\[output\+3u\]\)\)/);

  const baseline = createSparseCM12VelocityExtensionWGSL({
    layout, acceptedVelocityFloatBase: 64,
  });
  assert.doesNotMatch(baseline, /PublishVexAcceptedEffectiveVelocity/,
    "baseline WGSL must not contain the Phase-1 publication hook");
});
