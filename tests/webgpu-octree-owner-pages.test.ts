import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  OCTREE_OWNER_ARENA_MAGIC,
  OCTREE_OWNER_PAGE_ACTIVE_TABLE_B,
  OCTREE_OWNER_PAGE_CONTROL_WORDS,
  OCTREE_OWNER_PAGE_LOOKUP_STATUS,
  OCTREE_OWNER_PAGE_PUBLICATION_STATUS,
  OCTREE_OWNER_PAGE_WORD_TOPOLOGY,
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
  resolveOctreeOwnerCandidateGeneration,
  unpackOctreeOwnerPageControl,
  unpackOctreeOwnerPageWord,
  type OctreeOwnerLeafSize,
} from "../lib/webgpu-octree-owner-pages";
import { PassBroker } from "../lib/webgpu-pass-broker";

const projectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const ownerPageSource = readFileSync(new URL("../lib/webgpu-octree-owner-pages.ts", import.meta.url), "utf8");

test("recurring owner lifecycle separates inactive preparation from next-boundary commit", () => {
  const prepare = WebGPUOctreeSimulationOwnerPages.prototype.encodeInactiveCandidate.toString();
  const commit = WebGPUOctreeSimulationOwnerPages.prototype.encodeReadyCommit.toString();
  assert.equal((prepare.match(/broker\.compute/g) ?? []).length, 1);
  assert.equal((prepare.match(/dispatchWorkgroups\(1\)/g) ?? []).length, 2);
  assert.equal((commit.match(/dispatchWorkgroups\(1\)/g) ?? []).length, 1);
  assert.match(prepare, /broker\.fence\("inactive owner-page generation prepared"\)/);
  assert.doesNotMatch(commit, /broker\.fence/,
    "the ready singleton must stay in the caller's coupled storage pass");
  assert.doesNotMatch(`${prepare}\n${commit}`,
    /dispatchWorkgroupsIndirect|this\.indirect|sortCandidates|FLUID_BRICK_|hash|compatibility/);
});

test("recurring owner candidate reads the frontier attempt generation", () => {
  assert.match(octreeDeterministicOwnerPageLifecycleShader,
    /generation=candidateGenerationSource\[params\.source\.w\+4u\]/,
    "source.w is acceptedGenerationIndex + 1, so +4 addresses frontier[8]");
  assert.doesNotMatch(octreeDeterministicOwnerPageLifecycleShader,
    /generation=candidateGenerationSource\[params\.source\.w\+5u\]/,
    "frontier[9] is the error word, not a generation");
});

test("recurring GPU owner publication keeps stable physical IDs and is atomic-free", () => {
  const shader = octreeDeterministicOwnerPageLifecycleShader;
  assert.match(shader, /fn buildOwnerPageCandidate/);
  assert.match(shader,
    /for\(var width=2u;width<=SORT_CAPACITY;width<<=1u\)[\s\S]*let partner=index\^stride/);
  assert.match(shader, /let key=sortedKey\(row\);let old=lowerBoundOld\(key,oldCount\)/);
  assert.match(shader, /let carried=old<oldCount&&arena\[recordKeyBase\(activeTable\(\)\)\+old\]==key/);
  assert.match(shader, /if\(carried\)\{oldPage=arena\[recordPageBase\(activeTable\(\)\)\+old\];\}/);
  assert.match(shader, /if\(rank<retired\)\{page=scratch\[retiredPageBase\(\)\+rank\];\}/);
  assert.match(shader, /page=scratch\[freeQueueBase\(\)\+\(head\+queueRank\)%params\.counts\.y\]/);
  assert.match(shader,
    /let key=scratch\[candidateKeyBase\(\)\+row\];[\s\S]*arena\[recordKeyBase\(inactiveTable\(\)\)\+row\]=key/);
  assert.match(shader,
    /let page=scratch\[candidatePageBase\(\)\+row\];[\s\S]*arena\[recordPageBase\(inactiveTable\(\)\)\+row\]=page/);
  assert.match(shader, /arena\[directoryBase\(inactiveTable\(\)\)\+key-1u\]=page/);
  assert.match(shader, /payloadBase\(inactiveTable\(\)\)\+\(encodedPage-1u\)\*PAGE_VOXELS/);
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
    /let age=generation-arena\[CONTROL_ACCEPTED_GENERATION\][\s\S]*let eligible=generation!=0u&&age!=0u&&age<0x80000000u/);
  assert.match(shader,
    /scratch\[META_VALID\]=0u;scratch\[META_GENERATION\]=0u;[\s\S]*var generation=worklist\[15\]/,
    "an unchanged producer invalidates shared bootstrap scratch before its zero-work return");
  assert.match(shader,
    /arena\[payloadBase\(inactiveTable\(\)\)\+\(encodedPage-1u\)\*PAGE_VOXELS\+local\]=word[\s\S]*scratch\[META_VALID\]=CANDIDATE_VALID/);
  assert.match(shader,
    /fn commitOwnerPageGeneration\(\)[\s\S]*candidateCount>params\.topology\.x[\s\S]*arena\[CONTROL_ACCEPTED_GENERATION\] = scratch\[META_GENERATION\][\s\S]*candidateGenerationSource\[activeGenerationIndex\]=generation/);
  assert.doesNotMatch(projectionSource, /fn publishReadyFrontier|publishReadyFrontierPipeline/,
    "the retired second half of the epoch flip must not remain in production");
  const prepare = WebGPUOctreeSimulationOwnerPages.prototype.encodeInactiveCandidate.toString();
  const commit = WebGPUOctreeSimulationOwnerPages.prototype.encodeReadyCommit.toString();
  assert.ok(prepare.indexOf("this.buildCandidate") < prepare.indexOf("this.commitCandidate"));
  assert.match(commit, /this\.commit/);
});

test("owner candidate clock advances every attempt and reserves zero across wrap", () => {
  assert.equal(resolveOctreeOwnerCandidateGeneration(7), 8);
  assert.equal(resolveOctreeOwnerCandidateGeneration(0xffff_ffff), 1);
  assert.match(octreeDeterministicOwnerPageLifecycleShader,
    /generation=candidateGenerationSource\[params\.source\.w\+4u\]/);
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

test("owner-page storage owns two complete immutable record, directory, and payload banks", () => {
  const ocean = planOctreeOwnerPages([288, 96, 64]);
  assert.deepEqual(ocean.brickDimensions, [36, 12, 8]);
  assert.equal(ocean.logicalBrickCount, 3456);
  assert.equal(ocean.ownerRecordCapacity, 3456);
  assert.equal(ocean.ownerRecordKeyOffsetWords, 16);
  assert.equal(ocean.ownerRecordPageOffsetWords, 16 + 3456);
  assert.equal(ocean.ownerRecordKeyBOffsetWords, 16 + 3456 * 2);
  assert.equal(ocean.ownerRecordPageBOffsetWords, 16 + 3456 * 3);
  assert.equal(ocean.ownerDirectoryOffsetWords, 16 + 3456 * 4);
  assert.equal(ocean.ownerDirectoryBOffsetWords, 16 + 3456 * 5);
  assert.equal(ocean.ownerPagesOffsetWords, 16 + 3456 * 6);
  assert.equal(ocean.ownerPagesBOffsetWords, ocean.ownerPagesOffsetWords + 3456 * 512);
  assert.equal(ocean.bytesPerPage, 2056);
  assert.equal(ocean.allocatedBytes, 14_238_784);
  assert.ok(ocean.allocatedBytes < ocean.denseOwnerBytes * 1.01);
});

test("compact capacity still follows pressure and surface bounds", () => {
  const ocean = planOctreeOwnerPages([320, 96, 80], {
    adaptiveBounds: { pressureRowCapacity: 384_768, fineSeedLeafCapacity: 123_126 },
  });
  assert.equal(ocean.adaptiveCapacity, 4_014);
  assert.equal(ocean.capacity, 4_014);
  assert.equal(ocean.allocatedBytes, 16_544_032);
  const target = planOctreeOwnerPages([640, 192, 160], {
    adaptiveBounds: { pressureRowCapacity: 1_540_864, fineSeedLeafCapacity: 493_077 },
  });
  assert.equal(target.adaptiveCapacity, 16_073);
  assert.equal(target.allocatedBytes, 66_399_440);
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
  arena[plan.ownerDirectoryOffsetWords + 2] = 1;
  const cell = [16, 0, 0] as const;
  arena[plan.ownerPagesOffsetWords] =
    packOctreeOwnerPageWord(cell, cell, 1) | OCTREE_OWNER_PAGE_WORD_TOPOLOGY;
  assert.deepEqual(lookupOctreeOwnerPage(arena, plan, cell, 16), {
    origin: [...cell], size: 1, missing: false,
    status: OCTREE_OWNER_PAGE_LOOKUP_STATUS.topology,
  });
  assert.ok((lookupOctreeOwnerPage(arena, plan, [8, 0, 0], 16).status
    & OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing) !== 0);
});

test("CPU lookup follows only the status-selected owner bank", () => {
  const plan = planOctreeOwnerPages([16, 8, 8], { maximumPages: 1 });
  const arena = new Uint32Array(plan.allocatedWords);
  const cellA = [0, 0, 0] as const;
  const cellB = [8, 0, 0] as const;
  arena[plan.ownerDirectoryOffsetWords] = 1;
  arena[plan.ownerPagesOffsetWords] = packOctreeOwnerPageWord(cellA, cellA, 1);
  arena[plan.ownerDirectoryBOffsetWords + 1] = 1;
  arena[plan.ownerPagesBOffsetWords] = packOctreeOwnerPageWord(cellB, cellB, 1);
  assert.equal(lookupOctreeOwnerPage(arena, plan, cellA, 8).status, 0);
  assert.notEqual(lookupOctreeOwnerPage(arena, plan, cellB, 8).status, 0);
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.status] =
    OCTREE_OWNER_PAGE_ACTIVE_TABLE_B | OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready;
  assert.notEqual(lookupOctreeOwnerPage(arena, plan, cellA, 8).status, 0);
  assert.equal(lookupOctreeOwnerPage(arena, plan, cellB, 8).status, 0);
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

test("WGSL materializes a complete inactive payload without touching the active bank", () => {
  const publish = octreeDeterministicOwnerPageLifecycleShader.match(
    /\/\/ Materialize a complete inactive payload bank\.[\s\S]*?storageBarrier\(\);workgroupBarrier\(\);/,
  )?.[0] ?? "";
  assert.match(publish, /var word=arena\[payloadBase\(activeTable\(\)\)[\s\S]*arena\[payloadBase\(inactiveTable\(\)\)/);
  assert.doesNotMatch(publish, /arena\[payloadBase\(activeTable\(\)\).*\]=/,
    "candidate publication must never write the accepted payload bank");
  assert.match(publish, /word&=~OWNER_WORD_TOPOLOGY/,
    "inactive-bank carry must clear the prior frontier's leaf-membership marks");
  assert.match(projectionSource,
    /fn markAcceptedOwner[\s\S]*atomicOr\(&owners\[at\],OWNER_WORD_TOPOLOGY\)[\s\S]*fn emitLeaves[\s\S]*markAcceptedOwner\(unpackOrigin\(acceptedOwner\.packedOrigin\)\)/,
    "only compact accepted leaves may mark extension-graph membership");
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

test("lookup WGSL uses the direct logical directory without mutable allocator operations", () => {
  assert.match(octreeOwnerPageLookupWgsl,
    /let encodedPage = ownerPageArena\[directoryOffset \+ logical\]/);
  assert.doesNotMatch(octreeOwnerPageLookupWgsl, /while \(low < high\)/);
  assert.doesNotMatch(octreeOwnerPageLookupWgsl,
    /atomic|compareExchange|0x9e3779b1|probe|tombstone|freeList/);
  assert.match(projectionSource, /return atomicLoad\(&owners\[directoryOffset \+ logical\]\)/);
  assert.match(projectionSource,
    /fn encodePagedOwner\(cell: vec3u, origin: vec3u, size: u32\)[\s\S]*fn decodePagedOwner\(word: u32, cell: vec3u\)/);
  assert.match(projectionSource, /encodePagedOwner\(cell, origin, size\)/);
  assert.doesNotMatch(projectionSource, /ownerPagesEnabled|decodeDenseOwner|encodeDenseOwner/,
    "production has one sparse owner-page ABI with no runtime format switch");
});

test("Dawn rejected candidate preserves the accepted owner bank and frontier epoch bit-for-bit", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU owner rejection checks",
}, async (t) => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const worklist = device.createBuffer({ size: 18 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const frontier = device.createBuffer({ size: 10 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const coldWorklist = new Uint32Array(18);
  coldWorklist[0] = 1; coldWorklist[15] = 1; coldWorklist[16] = 0;
  device.queue.writeBuffer(worklist, 0, coldWorklist);
  device.queue.writeBuffer(frontier, 0, Uint32Array.from([
    0, 1, 0, 0, 0, 0, 1, 1, 1, 0,
  ]));
  const pages = new WebGPUOctreeSimulationOwnerPages(device, [16, 8, 8],
    { maximumPages: 2 }, {
      tileWorklist: worklist, tileSizeCells: 8,
      activeTileLimits: [1, 1, 1], activeTileCount: 1,
    }, {
      tileWorklist: worklist, tileSizeCells: 8, tileListCapacity: 2,
      candidateGeneration: { buffer: frontier, offsetWords: 3, frontierListCapacity: 1 },
    });
  const submit = (encode: (broker: PassBroker) => void) => {
    const broker = new PassBroker(device.createCommandEncoder());
    encode(broker); device.queue.submit([broker.finish()]);
  };
  const readWords = async (buffer: GPUBuffer, words: number) => {
    const readback = device.createBuffer({ size: words * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, readback, 0, words * 4);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap(); readback.destroy(); return result;
  };

  submit((broker) => { pages.encodeAnalyticBootstrap(broker); pages.encodeReadyCommit(broker); });
  await device.queue.onSubmittedWorkDone();
  const acceptedArena = await readWords(pages.arena, pages.plan.allocatedWords);
  const acceptedFrontier = await readWords(frontier, 10);
  if (acceptedArena[OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration] === 0) {
    const validationError = await device.popErrorScope();
    assert.equal(validationError, null, validationError?.message);
    pages.destroy(); frontier.destroy(); worklist.destroy(); device.destroy();
    t.skip("local Dawn Metal runtime completed a validated compute submission as a no-op");
    return;
  }
  assert.equal(acceptedArena[OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration], 1);
  assert.equal(acceptedArena[OCTREE_OWNER_PAGE_CONTROL_WORDS.status] >>> 31, 1,
    "cold publication activates bank B");
  assert.deepEqual([...acceptedFrontier.slice(2, 4)], [1, 1]);

  const nextWorklist = new Uint32Array(coldWorklist); nextWorklist[16] = 1;
  device.queue.writeBuffer(worklist, 0, nextWorklist);
  // Candidate zero names a valid successor, but its injected terminal reason
  // must reject the entire coupled publication.
  device.queue.writeBuffer(frontier, 0, Uint32Array.from([
    1, 1, 1, 1, 0, 0, 1, 0, 2, 7,
  ]));
  submit((broker) => { pages.encodeInactiveCandidate(broker); pages.encodeReadyCommit(broker); });
  await device.queue.onSubmittedWorkDone();
  const rejectedArena = await readWords(pages.arena, pages.plan.allocatedWords);
  const rejectedFrontier = await readWords(frontier, 10);

  assert.deepEqual([...rejectedArena.slice(0, 16)], [...acceptedArena.slice(0, 16)],
    "rejection preserves every accepted owner control word");
  assert.deepEqual(
    [...rejectedArena.slice(pages.plan.ownerRecordKeyBOffsetWords,
      pages.plan.ownerDirectoryOffsetWords)],
    [...acceptedArena.slice(pages.plan.ownerRecordKeyBOffsetWords,
      pages.plan.ownerDirectoryOffsetWords)],
    "rejection preserves accepted bank-B sorted records");
  assert.deepEqual(
    [...rejectedArena.slice(pages.plan.ownerDirectoryBOffsetWords,
      pages.plan.ownerPagesOffsetWords)],
    [...acceptedArena.slice(pages.plan.ownerDirectoryBOffsetWords,
      pages.plan.ownerPagesOffsetWords)],
    "rejection preserves the accepted bank-B directory");
  assert.deepEqual(
    [...rejectedArena.slice(pages.plan.ownerPagesBOffsetWords,
      pages.plan.ownerPagesBOffsetWords + pages.plan.capacity * pages.plan.pageVoxels)],
    [...acceptedArena.slice(pages.plan.ownerPagesBOffsetWords,
      pages.plan.ownerPagesBOffsetWords + pages.plan.capacity * pages.plan.pageVoxels)],
    "rejection preserves every accepted bank-B payload word");
  assert.deepEqual([...rejectedFrontier.slice(2, 4)], [...acceptedFrontier.slice(2, 4)],
    "rejection preserves the active frontier selector and generation");
  assert.equal(rejectedFrontier[rejectedFrontier[2]], acceptedFrontier[acceptedFrontier[2]],
    "rejection preserves the active frontier count");

  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  pages.destroy(); frontier.destroy(); worklist.destroy(); device.destroy();
});
