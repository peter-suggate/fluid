import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  OCTREE_OWNER_ARENA_MAGIC,
  OCTREE_OWNER_PAGE_CONTROL_WORDS,
  OCTREE_OWNER_PAGE_LOOKUP_STATUS,
  OCTREE_OWNER_PAGE_PUBLICATION_STATUS,
  OCTREE_OWNER_PAGE_WORD_VALID,
  WebGPUOctreeSimulationOwnerPages,
  OctreeOwnerPageLifecycleMirror,
  canonicalMissingAirOwner,
  decodeOctreeOwnerPageWord,
  findOctreeOwnerPageRecord,
  lookupOctreeOwnerPage,
  octreeDeterministicOwnerPageLifecycleShader,
  octreeOwnerPageLookupWgsl,
  packOctreeOwnerPageWord,
  planOctreeOwnerPages,
  unpackOctreeOwnerPageControl,
  unpackOctreeOwnerPageWord,
  type OctreeOwnerLeafSize,
} from "../lib/webgpu-octree-owner-pages";

const projectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const ownerPageSource = readFileSync(new URL("../lib/webgpu-octree-owner-pages.ts", import.meta.url), "utf8");

test("recurring owner lifecycle batches its three ordered singleton commits", () => {
  const encode = WebGPUOctreeSimulationOwnerPages.prototype.encode.toString();
  assert.equal((encode.match(/broker\.compute/g) ?? []).length, 1);
  assert.equal((encode.match(/broker\.fence/g) ?? []).length, 2,
    "one normal publication fence plus the exceptional cleanup fence");
  assert.match(encode, /broker\.fence\("owner-page generation publication"\)/);
  assert.equal((encode.match(/dispatchWorkgroups\(1\)/g) ?? []).length, 3,
    "owner publication remains one persistent candidate build plus two ordered commits");
  assert.doesNotMatch(encode, /dispatchWorkgroupsIndirect|this\.indirect|sortCandidates/);
  assert.doesNotMatch(encode, /FLUID_BRICK_|hash|compatibility/);
});

test("recurring GPU owner publication keeps stable physical IDs and is atomic-free", () => {
  const shader = octreeDeterministicOwnerPageLifecycleShader;
  assert.match(shader, /fn buildOwnerPageCandidate/);
  assert.match(shader,
    /for\(var width=2u;width<=SORT_CAPACITY;width<<=1u\)[\s\S]*let partner=index\^stride/);
  assert.match(shader, /let key=sortedKey\(row\);let old=lowerBoundOld\(key,oldCount\)/);
  assert.match(shader, /let carried=old<oldCount&&arena\[params\.offsets\.x\+old\]==key/);
  assert.match(shader, /if\(carried\)\{oldPage=arena\[params\.offsets\.y\+old\];\}/);
  assert.match(shader, /if\(rank<retired\)\{page=scratch\[retiredPageBase\(\)\+rank\];\}/);
  assert.match(shader, /page=scratch\[freeQueueBase\(\)\+\(head\+queueRank\)%params\.counts\.y\]/);
  assert.match(shader, /arena\[params\.offsets\.x\+row\]=scratch\[candidateKeyBase\(\)\+row\]/);
  assert.match(shader, /arena\[params\.offsets\.y\+row\]=scratch\[candidatePageBase\(\)\+row\]/);
  assert.match(shader, /params\.offsets\.z\+\(encodedPage-1u\)\*PAGE_VOXELS/);
  assert.match(shader,
    /fn ownerPageWord\(cell: vec3u, origin: vec3u, size: u32\)[\s\S]*u32\(delta\.x \+ 32\)[\s\S]*firstTrailingBit\(size\)[\s\S]*word=ownerPageWord\(cell,origin,size\)/,
    "every published page payload must use the sole brick-relative owner ABI");
  assert.doesNotMatch(shader, /analyticOwnerWord|word = u32\(firstTrailingBit\(size\)\)/,
    "the legacy exponent-only page payload must stay deleted");
  assert.doesNotMatch(shader, /atomic|compareExchange|hash|tombstone|freeList|popFree|prepareCandidate|lowestFree|usedPhysicalBase/);
  assert.doesNotMatch(ownerPageSource,
    /atomic|compareExchange|pageHash|freeList|beginComputePass|GPUOctreeOwnerPageArena|WebGPUSvoOwnerPageAllocator/,
    "retired allocator source and backing must stay deleted");
  assert.doesNotMatch(ownerPageSource,
    /octreeOwnerPageDispatchShader|beginRecurringOwnerPages|beginAnalyticOwnerPages|sortCandidatePages|scanCandidateDelta|prefixCandidateDelta|dispatchWorkgroupsIndirect|SORT_WIDTH|sortBBase/,
    "the retired indirect and per-sort-width schedule must not retain host or shader backing");
});

test("generation commits only after candidate validation and payload publication", () => {
  const shader = octreeDeterministicOwnerPageLifecycleShader;
  assert.match(shader,
    /let eligible=generation!=0u&&generation>arena\[CONTROL_ACCEPTED_GENERATION\]/);
  assert.match(shader,
    /scratch\[META_VALID\]=0u;scratch\[META_GENERATION\]=0u;[\s\S]*let generation=worklist\[15\]/,
    "an unchanged producer invalidates shared bootstrap scratch before its zero-work return");
  assert.match(shader,
    /arena\[params\.offsets\.z\+\(encodedPage-1u\)\*PAGE_VOXELS\+local\]=word[\s\S]*scratch\[META_VALID\]=CANDIDATE_VALID/);
  assert.match(shader,
    /fn commitOwnerPageGeneration\(\)[\s\S]*arena\[CONTROL_ACCEPTED_GENERATION\] = scratch\[META_GENERATION\]/);
  const encode = WebGPUOctreeSimulationOwnerPages.prototype.encode.toString();
  assert.ok(encode.indexOf("this.buildCandidate") < encode.indexOf("this.commitCandidate"));
  assert.ok(encode.indexOf("this.commitCandidate") < encode.lastIndexOf("this.commit"));
});

test("persistent owner transaction keeps storage authority uniform across every barrier", () => {
  const shader = octreeDeterministicOwnerPageLifecycleShader;
  const build = shader.match(
    /fn buildOwnerPageCandidate[\s\S]*?\n}\n\nvar<workgroup> commitState/,
  )?.[0] ?? "";
  for (const index of [0, 1, 2, 3, 4, 5, 6, 7]) {
    assert.match(build, new RegExp(`workgroupUniformLoad\\(&transactionState\\[${index}\\]\\)`));
  }
  assert.match(build, /storageBarrier\(\);workgroupBarrier\(\)/);
  assert.doesNotMatch(build,
    /if\(scratch\[(?:META_GENERATION|META_VALID)\][^)]*\)\{[\s\S]*workgroupBarrier/,
    "storage-backed authority must never control barrier reachability");
});

test("owner-page storage is two sorted record words plus one exclusive payload", () => {
  const ocean = planOctreeOwnerPages([288, 96, 64]);
  assert.deepEqual(ocean.brickDimensions, [36, 12, 8]);
  assert.equal(ocean.logicalBrickCount, 3456);
  assert.equal(ocean.ownerRecordCapacity, 3456);
  assert.equal(ocean.ownerRecordKeyOffsetWords, 16);
  assert.equal(ocean.ownerRecordPageOffsetWords, 16 + 3456);
  assert.equal(ocean.ownerPagesOffsetWords, 16 + 3456 * 2);
  assert.equal(ocean.bytesPerPage, 2056);
  assert.equal(ocean.allocatedBytes, 7_105_600);
  assert.ok(ocean.allocatedBytes < ocean.denseOwnerBytes * 0.51);
});

test("compact capacity still follows pressure and surface bounds", () => {
  const ocean = planOctreeOwnerPages([320, 96, 80], {
    adaptiveBounds: { pressureRowCapacity: 384_768, fineSeedLeafCapacity: 123_126 },
  });
  assert.equal(ocean.adaptiveCapacity, 4_014);
  assert.equal(ocean.capacity, 4_014);
  assert.equal(ocean.allocatedBytes, 8_252_848);
  const target = planOctreeOwnerPages([640, 192, 160], {
    adaptiveBounds: { pressureRowCapacity: 1_540_864, fineSeedLeafCapacity: 493_077 },
  });
  assert.equal(target.adaptiveCapacity, 16_073);
  assert.equal(target.allocatedBytes, 33_046_152);
});

test("sorted CPU lookup finds exact records by binary search", () => {
  const plan = planOctreeOwnerPages([32, 8, 8], { maximumPages: 3 });
  const arena = new Uint32Array(plan.allocatedWords);
  arena[1] = 3;
  arena.set([1, 3, 4], plan.ownerRecordKeyOffsetWords);
  arena.set([1, 2, 3], plan.ownerRecordPageOffsetWords);
  assert.equal(findOctreeOwnerPageRecord(arena, plan, 0), 0);
  assert.equal(findOctreeOwnerPageRecord(arena, plan, 1), -1);
  assert.equal(findOctreeOwnerPageRecord(arena, plan, 2), 1);
  assert.equal(findOctreeOwnerPageRecord(arena, plan, 3), 2);
});

test("sorted owner lookup decodes a stable physical page independent of record rank", () => {
  const plan = planOctreeOwnerPages([32, 8, 8], { maximumPages: 2 });
  const arena = new Uint32Array(plan.allocatedWords);
  arena[1] = 2;
  arena.set([1, 3], plan.ownerRecordKeyOffsetWords);
  arena.set([2, 1], plan.ownerRecordPageOffsetWords);
  const cell = [16, 0, 0] as const;
  arena[plan.ownerPagesOffsetWords] =
    packOctreeOwnerPageWord(cell, cell, 1);
  assert.deepEqual(lookupOctreeOwnerPage(arena, plan, cell, 16), {
    origin: [...cell], size: 1, missing: false, status: 0,
  });
  assert.ok((lookupOctreeOwnerPage(arena, plan, [8, 0, 0], 16).status
    & OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing) !== 0);
});

test("CPU generation oracle fails closed before replacing an accepted exact set", () => {
  const mirror = new OctreeOwnerPageLifecycleMirror(4, 2);
  assert.equal(mirror.publish(0, [0], []).status, OCTREE_OWNER_PAGE_PUBLICATION_STATUS.unpublished);
  assert.equal(mirror.publish(5, [0, 2], []).status, OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready);
  assert.equal(mirror.slot(0), 0);
  assert.equal(mirror.slot(2), 1);
  const overflow = mirror.publish(6, [0, 1, 2], []);
  assert.ok((overflow.status & OCTREE_OWNER_PAGE_PUBLICATION_STATUS.overflow) !== 0);
  assert.equal(overflow.stats.generation, 5);
  assert.equal(mirror.slot(2), 1);
  const next = mirror.publish(7, [1, 3], [0, 2]);
  assert.equal(next.status, OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready);
  assert.equal(mirror.slot(1), 0);
  assert.equal(mirror.slot(3), 1);
});

test("CPU lifecycle retains carried physical IDs across sorted reorder, add, and remove", () => {
  const mirror = new OctreeOwnerPageLifecycleMirror(8, 4);
  assert.equal(mirror.publish(1, [2, 4, 6], []).status,
    OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready);
  assert.deepEqual([mirror.slot(2), mirror.slot(4), mirror.slot(6)], [0, 1, 2]);

  const reordered = mirror.publish(2, [6, 1, 4], [2]);
  assert.equal(reordered.status, OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready);
  assert.equal(mirror.slot(4), 1, "carried key keeps its physical page after its sorted rank moves");
  assert.equal(mirror.slot(6), 2, "carried key keeps its physical page after its sorted rank moves");
  assert.equal(mirror.slot(1), 0, "activation deterministically reuses the lowest retired page");
  assert.equal(mirror.slot(2), undefined);

  mirror.publish(3, [0, 1, 6], [4]);
  assert.equal(mirror.slot(1), 0);
  assert.equal(mirror.slot(6), 2);
  assert.equal(mirror.slot(0), 1, "the next activation takes the lowest remaining free page");
  assert.equal(mirror.slot(4), undefined);
});

test("CPU lifecycle mirrors retired-first FIFO reuse without a physical-ID search", () => {
  const mirror = new OctreeOwnerPageLifecycleMirror(8, 4);
  mirror.publish(1, [0, 1, 2], []);
  assert.deepEqual([mirror.slot(0), mirror.slot(1), mirror.slot(2)], [0, 1, 2]);
  mirror.publish(2, [0], [1, 2]);
  mirror.publish(3, [0, 6], []);
  assert.equal(mirror.slot(6), 3,
    "net shrink appends retired IDs after the already-free FIFO instead of searching for the lowest ID");
  mirror.publish(4, [0, 5, 6], []);
  assert.equal(mirror.slot(5), 1, "the first retired ID follows the prior free prefix");
});

test("WGSL initializes canonical payload only for newly activated physical pages", () => {
  const publish = octreeDeterministicOwnerPageLifecycleShader.match(
    /for\(var activation=0u;activation<added;activation\+=1u\)\{[\s\S]*?storageBarrier\(\);workgroupBarrier\(\);/,
  )?.[0] ?? "";
  assert.match(publish,
    /let row=scratch\[activationRowBase\(\)\+activation\][\s\S]*let encodedPage=scratch\[candidatePageBase\(\)\+row\]/);
  assert.match(publish,
    /arena\[params\.offsets\.z\+\(encodedPage-1u\)\*PAGE_VOXELS\+local\]=word/);
  assert.doesNotMatch(publish, /params\.offsets\.x|params\.offsets\.y/,
    "payload publication must not overwrite the accepted record table");
});

test("owner control ABI names every word by its current deterministic-publication meaning", () => {
  assert.deepEqual(OCTREE_OWNER_PAGE_CONTROL_WORDS, {
    freeCount: 0,
    residentCount: 1,
    candidateError: 2,
    capacity: 3,
    logicalBrickCount: 4,
    ownerRecordPageOffsetWords: 5,
    ownerPagesOffsetWords: 6,
    acceptedGeneration: 7,
    activatedCount: 8,
    retiredCount: 9,
    status: 10,
    observedGeneration: 11,
    invalidEntryCount: 12,
    tileListCapacity: 13,
    tileSizeCells: 14,
    magic: 15,
  });
  const words = Uint32Array.from([
    7, 5, 0, 12, 72, 28, 40, 9,
    3, 2, OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready, 9, 0, 48, 16, OCTREE_OWNER_ARENA_MAGIC,
  ]);
  assert.deepEqual(unpackOctreeOwnerPageControl(words), {
    freeCount: 7,
    residentCount: 5,
    candidateError: 0,
    capacity: 12,
    logicalBrickCount: 72,
    ownerRecordPageOffsetWords: 28,
    ownerPagesOffsetWords: 40,
    acceptedGeneration: 9,
    activatedCount: 3,
    retiredCount: 2,
    status: OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready,
    observedGeneration: 9,
    invalidEntryCount: 0,
    tileListCapacity: 48,
    tileSizeCells: 16,
    magic: OCTREE_OWNER_ARENA_MAGIC,
  });
  assert.throws(() => unpackOctreeOwnerPageControl(words.subarray(0, 15)), /needs 16 words/);
  assert.doesNotMatch(ownerPageSource,
    /SVO_OWNER_PAGE_|peakResidentCount|overflowCount|requiredCount|ownerMismatchCount|comparedOwnerCount|stalePublicationCount|unchangedPublicationCount|unpublishedPublicationCount/,
    "obsolete allocator and parity-diagnostic aliases must stay deleted from the owner ABI");
});

test("packed owner words round-trip every supported leaf size", () => {
  for (const size of [1, 2, 4, 8, 16, 32] as OctreeOwnerLeafSize[]) {
    const origin = [32, 32, 32] as const;
    const cell = [32 + size - 1, 32 + Math.floor(size / 2), 32] as const;
    const packed = packOctreeOwnerPageWord(cell, origin, size);
    assert.ok((packed & OCTREE_OWNER_PAGE_WORD_VALID) !== 0);
    assert.deepEqual(unpackOctreeOwnerPageWord(packed, cell),
      { origin: [...origin], size, missing: false });
  }
  assert.deepEqual(decodeOctreeOwnerPageWord(0, [63, 31, 15], [64, 32, 16], 32),
    canonicalMissingAirOwner([63, 31, 15], [64, 32, 16], 32));
});

test("lookup WGSL uses the sorted record ABI without mutable allocator operations", () => {
  assert.match(octreeOwnerPageLookupWgsl, /var low = 0u/);
  assert.match(octreeOwnerPageLookupWgsl, /while \(low < high\)/);
  assert.match(octreeOwnerPageLookupWgsl, /recordKeyOffset \+ capacity \+ low/);
  assert.doesNotMatch(octreeOwnerPageLookupWgsl,
    /atomic|compareExchange|0x9e3779b1|probe|tombstone|freeList/);
  assert.match(projectionSource, /let recordCapacity = pageIndexOffset - 16u/);
  assert.match(projectionSource,
    /fn encodePagedOwner\(cell: vec3u, origin: vec3u, size: u32\)[\s\S]*fn decodePagedOwner\(word: u32, cell: vec3u\)/);
  assert.match(projectionSource, /encodePagedOwner\(cell, origin, size\)/);
  assert.doesNotMatch(projectionSource, /ownerPagesEnabled|decodeDenseOwner|encodeDenseOwner/,
    "production has one sparse owner-page ABI with no runtime format switch");
});
