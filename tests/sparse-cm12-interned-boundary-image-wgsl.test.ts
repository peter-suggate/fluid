import assert from "node:assert/strict";
import test from "node:test";
import type { SparseCM12InternedBoundaryLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { createSparseCM12InternedBoundaryImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-image.wgsl";

const layout: SparseCM12InternedBoundaryLayout = {
  leafCapacity: 8, canonicalCapacity: 40, templateCount: 4,
  templatePayloadWords: 256, canonicalBaseWords: 32,
  templateDirectoryBaseWords: 672, templatePayloadBaseWords: 688,
  immutableWords: 960, immutableBytes: 3840,
  slotBaseWords: [960, 1664], slotLeafBaseWords: [992, 1696],
  slotRefBaseWords: [1056, 1760], wordsPerSlot: 704, bytesPerSlot: 2816,
  totalWords: 2368, totalBytes: 9472,
};

test("IBO1 WGSL exposes frozen canonical/template/slot accessors", () => {
  const source = createSparseCM12InternedBoundaryImageWGSL({ layout,
    arenaName: "fixtureArena", hookPrefix: "fixture", packetsPerLeaf: 64,
    acceptedSlotHook: "sharedAcceptedSlot",
    acceptedGenerationHook: "sharedAcceptedGeneration" });
  assert.match(source, /fn fixtureIBOAcceptedSlot\(\)->u32/);
  assert.match(source, /fn fixtureIBOAcceptedGeneration\(\)->u32/);
  assert.match(source, /fn fixtureIBOCanonicalWord/);
  assert.match(source, /fn fixtureIBOCanonicalRowBase/);
  assert.match(source, /fn fixtureIBOFaceRefCount/);
  assert.match(source, /fn fixtureIBORef.*->vec3u/);
  assert.match(source, /fn fixtureIBOTemplateDirectory.*->vec4u/);
  assert.match(source, /fn fixtureIBOTemplateRowWord/);
  assert.match(source, /fn fixtureIBOTemplateTermWord/);
  assert.match(source, /fn fixtureIBOValidateDeltaLeaf/);
  assert.match(source, /fn fixtureIBOBeginDeltaLeaf/);
  assert.match(source, /fn fixtureIBOWriteDeltaRef/);
  assert.match(source, /fn fixtureIBOSealDeltaLeaf/);
  assert.match(source, /fn fixtureIBOValidateScheduledDeltaLeaf/);
  assert.match(source, /fn fixtureIBOReplayDeltaLeaf/);
  assert.match(source, /fn fixtureIBOAcceptedSlot\(\)->u32\{return sharedAcceptedSlot\(\)&1u;/);
  assert.match(source, /fn fixtureIBOAcceptedGeneration\(\)->u32\{return sharedAcceptedGeneration\(\);/);
  assert.doesNotMatch(source,
    /fn fixtureIBOAcceptedSlot\(\)->u32\{return fixtureIBOLoad\(IBO1_BASE\+2u\)/);
  assert.doesNotMatch(source, /for\(var leaf=0u;leaf<IBO1_LEAF_CAPACITY/);
  assert.doesNotMatch(source, /incidence|fallback/i);
});

test("IBO1 WGSL relocates B16 packet-compatible addressing", () => {
  const baseWords = 65536;
  const source = createSparseCM12InternedBoundaryImageWGSL({ layout,
    baseWords, packetsPerLeaf: 64, acceptedSlotHook: "sharedAcceptedSlot",
    acceptedGenerationHook: "sharedAcceptedGeneration" });
  assert.match(source, new RegExp(`const IBO1_BASE:u32=${baseWords}u`));
  assert.match(source, new RegExp(
    `const IBO1_CANONICAL_BASE:u32=${baseWords + layout.canonicalBaseWords}u`,
  ));
  assert.match(source, /const IBO1_PACKETS_PER_LEAF:u32=64u/);
  assert.match(source, /packet\/IBO1_PACKETS_PER_LEAF/);
  assert.throws(() => createSparseCM12InternedBoundaryImageWGSL({ layout,
    arenaName: "bad-name", packetsPerLeaf: 64,
    acceptedSlotHook: "sharedAcceptedSlot",
    acceptedGenerationHook: "sharedAcceptedGeneration" }), /identifier/);
});
