import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  decodeGeneratedOctreePowerCatalog,
} from "../lib/generated/octree-power-catalog";
import { PassBroker } from "../lib/webgpu-pass-broker";
import {
  OCTREE_OWNER_ARENA_MAGIC,
  OCTREE_OWNER_PAGE_CONTROL_WORDS,
  OCTREE_OWNER_PAGE_PUBLICATION_STATUS,
  packOctreeOwnerPageWord,
  planOctreeOwnerPages,
} from "../lib/webgpu-octree-owner-pages";
import {
  OCTREE_POWER_SAME_OR_COARSER_FLAG,
  enumerateCanonicalSameOrFinerPowerDescriptors,
  encodeSameOrCoarserPowerDescriptor,
  sitesForSameOrCoarserPowerDescriptor,
  sitesForSameOrFinerPowerDescriptor,
} from "../lib/octree-power-descriptor";
import {
  OCTREE_POWER_DESCRIPTOR_ERROR,
  OCTREE_POWER_DESCRIPTOR_BOUNDARY_MASK,
  OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT,
  OCTREE_POWER_DESCRIPTOR_INVALID,
  WebGPUOctreePowerDescriptor,
  assertOctreePowerRowDeltaLayout,
  decodePagedOctreePowerOwner,
  describeOctreePowerRow,
  octreePowerDescriptorShader,
  octreePowerOwnerArenaPublicationIsValid,
  planOctreePowerDescriptors,
  unpackOctreePowerDescriptorControl,
  type OctreePowerOwner,
} from "../lib/webgpu-octree-power-descriptor";
import { createColdPowerRowPublication } from "./webgpu-octree-power-row-delta-fixture";

const dimensions = [32, 32, 32] as const;
const linear = (p: readonly [number, number, number], d: readonly [number, number, number] = dimensions) =>
  p[0] + d[0] * (p[1] + d[1] * p[2]);

function singletonOwnerPageArena(d: readonly [number, number, number]): Uint32Array<ArrayBuffer> {
  const plan = planOctreeOwnerPages(d);
  assert.equal(plan.capacity, 1);
  const arena = new Uint32Array(new ArrayBuffer(plan.allocatedBytes));
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.freeCount] = 0;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.residentCount] = 1;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.capacity] = 1;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.logicalBrickCount] = 1;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerRecordPageOffsetWords] = plan.ownerRecordPageOffsetWords;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.ownerPagesOffsetWords] = plan.ownerPagesOffsetWords;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration] = 1;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.status] = OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.observedGeneration] = 1;
  arena[OCTREE_OWNER_PAGE_CONTROL_WORDS.magic] = OCTREE_OWNER_ARENA_MAGIC;
  arena[plan.ownerRecordKeyOffsetWords] = 1;
  arena[plan.ownerRecordPageOffsetWords] = 1;
  arena[plan.ownerDirectoryOffsetWords] = 1;
  for (let z = 0; z < 8; z += 1) {
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 8; x += 1) {
        const cell = [x, y, z] as const;
        arena[plan.ownerPagesOffsetWords + x + 8 * (y + 8 * z)] =
          packOctreeOwnerPageWord(cell, cell, 1);
      }
    }
  }
  return arena;
}

function catalogViews() {
  const bytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  return decodeGeneratedOctreePowerCatalog(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

test("power descriptor planner is compact-row proportional and exposes the exact ABIs", () => {
  assert.deepEqual(planOctreePowerDescriptors(100), {
    rowCapacity: 100,
    descriptorBytes: 400,
    controlBytes: 32,
    dispatchBytes: 12,
    allocatedBytes: 2_232,
  });
  assert.deepEqual(unpackOctreePowerDescriptorControl([3, 2, 1, 2, 4, 2, 0, 19]), {
    rowCount: 3, validCount: 2, errorCount: 1, firstInvalid: 2,
    flags: 4, sameOrFinerCount: 2, sameOrCoarserCount: 0, generation: 19,
  });
});

test("exact row-delta ABI rejects truncated or mismatched publications before encoding", () => {
  const exact = {
    rows: { size: 96 } as GPUBuffer,
    rowCapacity: 2,
    controlOffsetWords: 0,
    newToOldOffsetWords: 16,
    oldToNewOffsetWords: 18,
    dirtyRowsOffsetWords: 20,
    affectedRowsOffsetWords: 22,
  };
  assert.doesNotThrow(() => assertOctreePowerRowDeltaLayout(exact, 2, "Fixture"));
  assert.throws(() => assertOctreePowerRowDeltaLayout(
    { ...exact, rows: { size: 92 } as GPUBuffer }, 2, "Fixture",
  ), /row-delta layout is invalid/);
  assert.throws(() => assertOctreePowerRowDeltaLayout(
    { ...exact, rowCapacity: 1 }, 2, "Fixture",
  ), /row-delta layout is invalid/);
});

test("CPU descriptor generation reproduces every uniquely graded immutable catalog key", () => {
  const offset = [10, 10, 10] as const;
  const descriptors = [
    ...enumerateCanonicalSameOrFinerPowerDescriptors(),
    ...Array.from({ length: 512 }, (_, low) => {
      const mask = low >>> 3;
      const coarseNeighbors = Array.from({ length: 6 }, (_, bit) => (mask & (1 << bit)) !== 0);
      return encodeSameOrCoarserPowerDescriptor({
        child: [low & 1, (low >> 1) & 1, (low >> 2) & 1] as [0 | 1, 0 | 1, 0 | 1],
        coarseNeighbors: coarseNeighbors as [boolean, boolean, boolean, boolean, boolean, boolean],
      });
    }),
  ];
  let redundantUniformCoarser = 0, checked = 0;
  for (const descriptor of descriptors) {
    const coarser = (descriptor & OCTREE_POWER_SAME_OR_COARSER_FLAG) !== 0;
    // With no coarse-neighbor bits the physical neighborhood is uniform and
    // has eight parity-dependent 9-bit spellings plus one 18-bit spelling.
    // Runtime deterministically selects the 18-bit same/finer spelling.
    if (coarser && (descriptor & 0x1f8) === 0) { redundantUniformCoarser += 1; continue; }
    const sites = coarser
      ? sitesForSameOrCoarserPowerDescriptor(descriptor)
      : sitesForSameOrFinerPowerDescriptor(descriptor);
    const anchor = sites.find((site) => site.key === (coarser ? "anchor" : "0,0,0/2"));
    assert.ok(anchor);
    const translated = sites.map((site) => ({
      origin: site.origin.map((value, axis) => value + offset[axis]) as [number, number, number],
      size: site.size,
    }));
    const ownerAt = (cell: readonly [number, number, number]): OctreePowerOwner => {
      const owner = translated.find((candidate) => candidate.origin.every((origin, axis) =>
        cell[axis] >= origin && cell[axis] < origin + candidate.size));
      return owner ?? { origin: [...cell] as [number, number, number], size: 1, invalid: true };
    };
    const origin = anchor.origin.map((value, axis) => value + offset[axis]) as [number, number, number];
    const result = describeOctreePowerRow({ cell: linear(origin), size: anchor.size }, dimensions, 32, ownerAt);
    assert.equal(result.descriptor, descriptor, `runtime descriptor ${descriptor}`);
    assert.equal(result.flags, 0, `runtime descriptor ${descriptor}`);
    checked += 1;
  }
  assert.equal(redundantUniformCoarser, 8);
  assert.ok(checked > 1_000, "the exhaustive symmetry quotient must remain substantial");
});

test("CPU oracle reports grading/owner errors and encodes domain boundaries as valid metadata", () => {
  const origin = [4, 4, 4] as const;
  const mixedOwner = (cell: readonly [number, number, number]): OctreePowerOwner => {
    if (cell.every((value, axis) => value >= origin[axis] && value < origin[axis] + 2)) return { origin, size: 2 };
    if (cell[0] < 4 && cell[1] >= 4 && cell[1] < 8 && cell[2] >= 4 && cell[2] < 8) return { origin: [0, 4, 4], size: 4 };
    if (cell[0] >= 6) return { origin: [...cell] as [number, number, number], size: 1 };
    return { origin: cell.map((value) => Math.floor(value / 2) * 2) as [number, number, number], size: 2 };
  };
  const mixed = describeOctreePowerRow({ cell: linear(origin), size: 2 }, dimensions, 32, mixedOwner);
  assert.equal(mixed.descriptor, OCTREE_POWER_DESCRIPTOR_INVALID);
  assert.ok((mixed.flags & OCTREE_POWER_DESCRIPTOR_ERROR.mixedGrading) !== 0);

  const ratioOwner = (cell: readonly [number, number, number]): OctreePowerOwner => {
    if (cell.every((value) => value >= 8 && value < 10)) return { origin: [8, 8, 8], size: 2 };
    if (cell[0] < 8) return { origin: [0, 0, 0], size: 8 };
    return { origin: cell.map((value) => Math.floor(value / 2) * 2) as [number, number, number], size: 2 };
  };
  const ratio = describeOctreePowerRow({ cell: linear([8, 8, 8]), size: 2 }, dimensions, 32, ratioOwner);
  assert.ok((ratio.flags & OCTREE_POWER_DESCRIPTOR_ERROR.gradingRatio) !== 0);

  const malformed = describeOctreePowerRow({ cell: linear(origin), size: 2 }, dimensions, 32,
    () => ({ origin: [0, 0, 0], size: 2, invalid: true }));
  assert.equal(malformed.flags, OCTREE_POWER_DESCRIPTOR_ERROR.malformedOwner);

  const boundary = describeOctreePowerRow({ cell: 0, size: 1 }, dimensions, 32,
    (cell) => ({ origin: [...cell] as [number, number, number], size: 1 }));
  assert.equal(boundary.flags, 0);
  assert.equal((boundary.descriptor & OCTREE_POWER_DESCRIPTOR_BOUNDARY_MASK) >>> OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT,
    (1 << 0) | (1 << 1) | (1 << 2));
});

test("CPU and GPU descriptor publication admit every co-spherical coarse mask", () => {
  const offset = [10, 10, 10] as const;
  const describe = (mask: number) => {
    const descriptor = (OCTREE_POWER_SAME_OR_COARSER_FLAG | (mask << 3)) >>> 0;
    const sites = sitesForSameOrCoarserPowerDescriptor(descriptor);
    const anchor = sites.find((site) => site.key === "anchor")!;
    const translated = sites.map((site) => ({
      origin: site.origin.map((value, axis) => value + offset[axis]) as [number, number, number],
      size: site.size,
    }));
    const ownerAt = (cell: readonly [number, number, number]): OctreePowerOwner => translated.find((candidate) =>
      candidate.origin.every((origin, axis) => cell[axis] >= origin && cell[axis] < origin + candidate.size))
      ?? { origin: [...cell] as [number, number, number], size: 1, invalid: true };
    const origin = anchor.origin.map((value, axis) => value + offset[axis]) as [number, number, number];
    return describeOctreePowerRow({ cell: linear(origin), size: anchor.size }, dimensions, 32, ownerAt);
  };

  for (const mask of [25, 42, 52, 57, 58, 60]) {
    const described = describe(mask);
    assert.equal(described.flags, 0, `coarse mask ${mask}`);
    assert.equal(described.kind, "same-or-coarser", `coarse mask ${mask}`);
    assert.notEqual(described.descriptor, OCTREE_POWER_DESCRIPTOR_INVALID, `coarse mask ${mask}`);
  }
  const catalog = catalogViews();
  for (const mask of [25, 42, 52, 57, 58, 60]) {
    const packed = catalog.sameOrCoarserDirect[mask << 3];
    const entry = packed & 0xffff;
    assert.notEqual(packed, 0xffff_ffff, `co-spherical mask ${mask} must be directly addressable`);
    assert.ok(catalog.tetrahedronHeaders[entry * 3 + 1] > 0,
      `co-spherical mask ${mask} must retain its local tetrahedra`);
    assert.equal(catalog.tetrahedronHeaders[entry * 3 + 2] & 1, 0,
      `co-spherical mask ${mask} remains a transition entry`);
  }
  assert.doesNotMatch(octreePowerDescriptorShader,
    /coarseMask==25u|failRow\(row,ACUTE_GRADING/,
    "descriptor publication must not refine away a catalog-valid co-spherical link");
});

test("uniform descriptor constrains face and edge owners but not refined corner-only cells", () => {
  const origin = [4, 4, 4] as const;
  const ownerAt = (cell: readonly [number, number, number]): OctreePowerOwner => {
    const positiveCorner = cell.every((value, axis) => value >= origin[axis] + 2);
    if (positiveCorner) return { origin: [...cell] as [number, number, number], size: 1 };
    return {
      origin: cell.map((value) => Math.floor(value / 2) * 2) as [number, number, number],
      size: 2,
    };
  };
  assert.deepEqual(describeOctreePowerRow({ cell: linear(origin), size: 2 }, dimensions, 32, ownerAt), {
    descriptor: 0x0003_ffff,
    flags: 0,
    kind: "same-or-finer",
  });
  assert.equal(ownerAt([6, 6, 6]).size, 1,
    "the eight corner-only cells remain outside the paper's 18-bit neighborhood contract");
});

test("paged descriptor decoding fails closed on missing and reserved owner words", () => {
  const rowCell = [0, 8, 10] as const;
  const rowWord = packOctreeOwnerPageWord(rowCell, rowCell, 2);
  assert.deepEqual(decodePagedOctreePowerOwner(rowWord, rowCell, [60, 45, 40], 4), {
    origin: rowCell, size: 2,
  });
  for (const word of [0, 0xffff_ffff, 0x801c_0000, 6, 0x0000_0102]) {
    assert.equal(decodePagedOctreePowerOwner(word, rowCell, [60, 45, 40], 4).invalid, true,
      `reserved owner word ${word.toString(16)}`);
  }
  assert.deepEqual(decodePagedOctreePowerOwner(packOctreeOwnerPageWord(rowCell, rowCell, 1),
    rowCell, [60, 45, 40], 4), {
    origin: rowCell, size: 1,
  });
  assert.match(octreePowerDescriptorShader, /\(word&OWNER_VALID\)==0u/);
  assert.match(octreePowerDescriptorShader,
    /encoded==0u\|\|encoded==INVALID\|\|encoded>capacity/,
    "a missing, reserved, or out-of-range page must invalidate descriptor publication");
  assert.match(octreePowerDescriptorShader,
    /fn pagedOwnerPublicationValid\(\)->bool\{[\s\S]*accepted!=0u&&\(owners\[OWNER_PUBLICATION_STATUS\]&0x7fffffffu\)==OWNER_PUBLICATION_READY[\s\S]*owners\[OWNER_OBSERVED_GENERATION\]==accepted&&owners\[OWNER_CANDIDATE_ERROR\]==0u[\s\S]*owners\[OWNER_INVALID_ENTRY_COUNT\]==0u&&resident<=capacity[\s\S]*owners\[OWNER_FREE_COUNT\]==capacity-resident/,
    "paged lookup admits only one internally complete immutable owner publication");
  assert.doesNotMatch(octreePowerDescriptorShader,
    /owners\[OWNER_ACCEPTED_GENERATION\]\s*[!=]=\s*generation\(\)|generation\(\)\s*[!=]=\s*owners\[OWNER_ACCEPTED_GENERATION\]|owners\[OWNER_ACCEPTED_GENERATION\]\s*[+-]\s*1u/,
    "owner-topology and power-descriptor generations are independent namespaces with no adjacent-generation fallback");
  assert.doesNotMatch(octreePowerDescriptorShader, /fn canonicalOwner/,
    "descriptor probes must not synthesize owners for incomplete topology pages");
  assert.doesNotMatch(octreePowerDescriptorShader, /indexedOwner|hashSite|hashCapacity/,
    "legacy hash and live-index owner lookups are deleted");
  assert.match(octreePowerDescriptorShader, /return decodePagedOwner\(owners\[at\],cell\)/);
});

test("paged descriptor authority is the owner arena's exact self-publication", () => {
  const control = new Uint32Array(16);
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.freeCount] = 9;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.residentCount] = 3;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.candidateError] = 0;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.capacity] = 12;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration] = 41;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.status] = OCTREE_OWNER_PAGE_PUBLICATION_STATUS.ready;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.observedGeneration] = 41;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.invalidEntryCount] = 0;
  control[OCTREE_OWNER_PAGE_CONTROL_WORDS.magic] = OCTREE_OWNER_ARENA_MAGIC;
  assert.equal(octreePowerOwnerArenaPublicationIsValid(control), true,
    "a clean topology generation remains authoritative when a later power generation consumes it");

  for (const [word, value] of [
    [OCTREE_OWNER_PAGE_CONTROL_WORDS.freeCount, 8], // free + resident no longer equals capacity
    [OCTREE_OWNER_PAGE_CONTROL_WORDS.candidateError, 1], // a newer candidate was rejected
    [OCTREE_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration, 0], // no committed topology generation
    [OCTREE_OWNER_PAGE_CONTROL_WORDS.status, 0], // publication is not ready
    [OCTREE_OWNER_PAGE_CONTROL_WORDS.observedGeneration, 40], // observed/accepted transaction is torn
    [OCTREE_OWNER_PAGE_CONTROL_WORDS.invalidEntryCount, 1], // owner entry validation failed
    [OCTREE_OWNER_PAGE_CONTROL_WORDS.magic, 0], // not an owner arena
  ] as const) {
    const malformed = new Uint32Array(control);
    malformed[word] = value;
    assert.equal(octreePowerOwnerArenaPublicationIsValid(malformed), false,
      `owner control word ${word} must fail the self-publication gate`);
  }
  const overflowing = new Uint32Array(control);
  overflowing[1] = 13;
  assert.equal(octreePowerOwnerArenaPublicationIsValid(overflowing), false);
  assert.equal(octreePowerOwnerArenaPublicationIsValid(control.subarray(0, 15)), false,
    "truncated control cannot fall back to dense or a prior generation");
});

test("descriptor WGSL has bounded direct-owner queries and exact delta publication", () => {
  assert.match(octreePowerDescriptorShader,
    /let requested=rowDelta\[params\.delta\.x\]/,
    "candidate rows must come from the immutable row-delta transaction");
  assert.doesNotMatch(octreePowerDescriptorShader, /rowCountSource|@binding\(6\)/,
    "the reused compaction arena must not remain a second candidate-count authority");
  assert.match(octreePowerDescriptorShader, /const DIRECTIONS:array<vec3i,18>/);
  assert.match(octreePowerDescriptorShader, /for\(var bit=0u;bit<18u;bit\+=1u\)/);
  assert.match(octreePowerDescriptorShader,
    /let encoded=owners\[directoryOffset\+logical\]/);
  assert.doesNotMatch(octreePowerDescriptorShader, /while\(low<high\)/);
  assert.doesNotMatch(octreePowerDescriptorShader, /fn structuralChange\(\)->bool/);
  assert.match(octreePowerDescriptorShader, /fn stagePowerDescriptorDelta/);
  assert.match(octreePowerDescriptorShader, /fn prefixPowerDescriptorDelta/);
  assert.match(octreePowerDescriptorShader, /fn scatterPowerDescriptorDelta/);
  assert.match(octreePowerDescriptorShader,
    /descriptor=committedDescriptors\[old\];[\s\S]*descriptors\[row\]=descriptor;[\s\S]*if\(old!=row\)\{status\|=STATUS_PUBLISH;\}/,
    "same-index carry must refresh complete candidate scratch without entering the sparse commit list");
  assert.match(octreePowerDescriptorShader,
    /commitDispatch=dispatchFor\(published\)/);
  assert.doesNotMatch(octreePowerDescriptorShader,
    /carryPowerDescriptors|summarizePowerDescriptors|publishPowerDescriptors|controlArena\.moved/);
  assert.doesNotMatch(octreePowerDescriptorShader,
    /for\(var row=0u;row<requested/);
  assert.doesNotMatch(octreePowerDescriptorShader, /atomic|compareExchange/);
  assert.match(octreePowerDescriptorShader, /indirectDispatch\[0\]=0u/);
  assert.doesNotMatch(octreePowerDescriptorShader, /texture_/);
});

test("descriptor production source has only exact delta work and deterministic publication", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree-power-descriptor.ts", import.meta.url), "utf8");
  assert.doesNotMatch(source, /\batomic(?:Add|Load|Min|Or|Store)?\b|compareExchange|clearBuffer|workControl/);
  assert.doesNotMatch(source, /live-index|indexedOwner|hashSite|hashCapacity|ownerMode.*auto/);
  assert.match(source, /copyBufferToBuffer\(delta\.rows, \(delta\.controlOffsetWords \+ 9\) \* 4/,
    "descriptor work must consume the structural dirty dispatch, not the wider wet affected set");
  assert.match(source, /delta\.dirtyRowsOffsetWords/);
  assert.match(source, /dispatchWorkgroupsIndirect\(this\.workDispatch, 0\)/);
  assert.match(source, /controlArena\.authority=candidate/);
  assert.doesNotMatch(source,
    /rowStatusBytes|summaryBytes|carryPipeline|summarizePipeline|dispatchDirect\(pass,\s*this\.plan\.rowCapacity/);
  assert.match(source, /Stage exact power descriptor carry and affected rows/);
  assert.match(source, /scatterPowerDescriptorDelta/);
});

test("Dawn matches the CPU descriptor, preserves boundary metadata, and fails capacity/malformed arenas closed", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU power-descriptor checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  // Retain the native GPU wrapper for the full test lifetime. Letting it be
  // collected before the Metal device has drained can tear down Dawn early.
  const gpu = dawn.create(["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const shaderModule = device.createShaderModule({ code: octreePowerDescriptorShader });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);

  const d = [4, 4, 4] as const;
  const ownerArena = singletonOwnerPageArena(d);
  const owners = device.createBuffer({ size: ownerArena.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(owners, 0, ownerArena);
  const headerWords = new Uint32Array(24);
  headerWords.set([linear([1, 1, 1], d), 0, 0, 1], 0);
  headerWords.set([linear([0, 0, 0], d), 0, 0, 1], 12);
  const headers = device.createBuffer({ size: headerWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(headers, 0, headerWords);
  const generator = new WebGPUOctreePowerDescriptor(device, 2);
  const readback = device.createBuffer({ size: 52, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  const cold = createColdPowerRowPublication(device, 2, 2, 7);
  generator.encode(broker, headers, owners,
    {
      dimensions: d, maximumLeafSize: 4,
      generation: 7, rowDelta: cold.rowDelta,
    });
  broker.fence();
  encoder.copyBufferToBuffer(generator.descriptors, 0, readback, 0, 8);
  encoder.copyBufferToBuffer(generator.control, 0, readback, 8, 32);
  encoder.copyBufferToBuffer(generator.dispatch, 0, readback, 40, 12);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
  assert.equal(words[0], 0x3ffff);
  assert.equal(words[1], (0x3ffff | (((1 << 0) | (1 << 1) | (1 << 2)) << OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT)) >>> 0);
  assert.deepEqual([...words.slice(2, 10)], [2, 2, 0, 0xffff_ffff, 0, 2, 0, 7]);
  assert.deepEqual([...words.slice(10, 13)], [1, 1, 1]);

  // Poison the current header after a valid publication. The exact row-delta
  // carries row zero from the immutable generation and resolves no dirty row:
  // the reused descriptor must remain immutable while its generation advances.
  const deltaWords = new Uint32Array(24);
  deltaWords.set([1, 2, 1, 0, 1, 0, 0, 9, 0x5244_4c54, 0, 1, 1, 0, 1, 1, 0], 0);
  deltaWords[16] = 1;
  const rowDelta = device.createBuffer({ size: deltaWords.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(rowDelta, 0, deltaWords);
  const malformedHeader = new Uint32Array(headerWords); malformedHeader[3] = 3;
  device.queue.writeBuffer(headers, 0, malformedHeader);
  const reuseReadback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const reuseEncoder = device.createCommandEncoder();
  const reuseBroker = new PassBroker(reuseEncoder);
  generator.encode(reuseBroker, headers, owners, {
    dimensions: d, maximumLeafSize: 4, generation: 9,
    rowDelta: {
      rows: rowDelta, rowCapacity: 2, controlOffsetWords: 0, newToOldOffsetWords: 16,
      oldToNewOffsetWords: 18, dirtyRowsOffsetWords: 20, affectedRowsOffsetWords: 22,
    },
  });
  reuseBroker.fence();
  reuseEncoder.copyBufferToBuffer(generator.descriptors, 0, reuseReadback, 0, 4);
  reuseEncoder.copyBufferToBuffer(generator.control, 0, reuseReadback, 4, 32);
  reuseEncoder.copyBufferToBuffer(generator.dispatch, 0, reuseReadback, 36, 12);
  device.queue.submit([reuseEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await reuseReadback.mapAsync(GPUMapMode.READ);
  const reused = new Uint32Array(reuseReadback.getMappedRange().slice(0)); reuseReadback.unmap();
  assert.equal(reused[0], 0x3ffff);
  assert.deepEqual([...reused.slice(1, 9)], [1, 1, 0, 0xffff_ffff, 0, 1, 0, 9]);
  assert.deepEqual([...reused.slice(9, 12)], [1, 1, 1]);
  device.queue.writeBuffer(headers, 0, headerWords);

  const capacityGenerator = new WebGPUOctreePowerDescriptor(device, 1);
  const capacityReadback = device.createBuffer({ size: 44, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const capacityEncoder = device.createCommandEncoder();
  const capacityBroker = new PassBroker(capacityEncoder);
  const overflowingCold = createColdPowerRowPublication(device, 1, 1, 0);
  device.queue.writeBuffer(overflowingCold.rowDelta.rows,
    overflowingCold.rowDelta.controlOffsetWords * 4, new Uint32Array([2]));
  capacityGenerator.encode(capacityBroker, headers, owners,
    {
      dimensions: d, maximumLeafSize: 4,
      rowDelta: overflowingCold.rowDelta,
    });
  capacityBroker.fence();
  capacityEncoder.copyBufferToBuffer(capacityGenerator.control, 0, capacityReadback, 0, 32);
  capacityEncoder.copyBufferToBuffer(capacityGenerator.dispatch, 0, capacityReadback, 32, 12);
  device.queue.submit([capacityEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await capacityReadback.mapAsync(GPUMapMode.READ);
  const capacity = new Uint32Array(capacityReadback.getMappedRange().slice(0)); capacityReadback.unmap();
  assert.ok(capacity[2] >= 1,
    "every detected capacity/publication violation is counted; the candidate must fail closed");
  assert.ok(capacity[3] < 2,
    "the first rejected row must identify a row in the requested publication");
  assert.ok((capacity[4] & OCTREE_POWER_DESCRIPTOR_ERROR.capacity) !== 0);
  assert.deepEqual([...capacity.slice(8, 11)], [0, 1, 1]);

  const malformedArena = new Uint32Array(16); malformedArena[15] = 0x4f57_4e52;
  const paged = device.createBuffer({ size: malformedArena.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paged, 0, malformedArena);
  const malformedReadback = device.createBuffer({ size: 44, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const malformedEncoder = device.createCommandEncoder();
  const malformedBroker = new PassBroker(malformedEncoder);
  const malformedCold = createColdPowerRowPublication(device, 1, 1, 0);
  capacityGenerator.encode(malformedBroker, headers, paged,
    {
      dimensions: d, maximumLeafSize: 4,
      rowDelta: malformedCold.rowDelta,
    });
  malformedBroker.fence();
  malformedEncoder.copyBufferToBuffer(capacityGenerator.control, 0, malformedReadback, 0, 32);
  malformedEncoder.copyBufferToBuffer(capacityGenerator.dispatch, 0, malformedReadback, 32, 12);
  device.queue.submit([malformedEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await malformedReadback.mapAsync(GPUMapMode.READ);
  const malformedWords = new Uint32Array(malformedReadback.getMappedRange().slice(0)); malformedReadback.unmap();
  assert.ok((malformedWords[4] & OCTREE_POWER_DESCRIPTOR_ERROR.malformedOwner) !== 0);
  assert.deepEqual([...malformedWords.slice(8, 11)], [0, 1, 1]);
  cold.destroy(); overflowingCold.destroy(); malformedCold.destroy();
});
