import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  decodeGeneratedOctreePowerCatalog,
} from "../lib/generated/octree-power-catalog";
import {
  OCTREE_POWER_SAME_OR_COARSER_FLAG,
  sitesForSameOrCoarserPowerDescriptor,
  sitesForSameOrFinerPowerDescriptor,
} from "../lib/octree-power-descriptor";
import {
  OCTREE_POWER_DESCRIPTOR_ERROR,
  OCTREE_POWER_DESCRIPTOR_BOUNDARY_MASK,
  OCTREE_POWER_DESCRIPTOR_BOUNDARY_SHIFT,
  OCTREE_POWER_DESCRIPTOR_INVALID,
  WebGPUOctreePowerDescriptor,
  decodeDenseOctreePowerOwner,
  decodePagedOctreePowerOwner,
  describeOctreePowerRow,
  octreePowerDescriptorShader,
  packDenseOctreePowerOwner,
  planOctreePowerDescriptors,
  unpackOctreePowerDescriptorControl,
  type OctreePowerOwner,
} from "../lib/webgpu-octree-power-descriptor";

const dimensions = [32, 32, 32] as const;
const linear = (p: readonly [number, number, number], d: readonly [number, number, number] = dimensions) =>
  p[0] + d[0] * (p[1] + d[1] * p[2]);

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
    allocatedBytes: 480,
  });
  assert.deepEqual(unpackOctreePowerDescriptorControl([3, 2, 1, 2, 4, 2, 0, 19]), {
    rowCount: 3, validCount: 2, errorCount: 1, firstInvalid: 2,
    flags: 4, sameOrFinerCount: 2, sameOrCoarserCount: 0, generation: 19,
  });
});

test("CPU descriptor generation reproduces every uniquely graded immutable catalog key", () => {
  const lookup = catalogViews().lookup;
  const offset = [10, 10, 10] as const;
  let redundantUniformCoarser = 0;
  for (let index = 0; index < lookup.length; index += 3) {
    const descriptor = lookup[index] >>> 0;
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
    assert.equal(result.descriptor, descriptor, `catalog lookup ${index / 3}`);
    assert.equal(result.flags, 0, `catalog lookup ${index / 3}`);
  }
  assert.equal(redundantUniformCoarser, 8);
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

test("dense owner helpers match the production owner word and reject malformed words", () => {
  const word = packDenseOctreePowerOwner([8, 12, 16], 4);
  assert.deepEqual(decodeDenseOctreePowerOwner(word, [9, 14, 18], dimensions, 32), {
    origin: [8, 12, 16], size: 4,
  });
  assert.deepEqual(decodeDenseOctreePowerOwner(0x8000_0000, [3, 2, 1], dimensions, 32), {
    origin: [3, 2, 1], size: 1,
  });
  assert.equal(decodeDenseOctreePowerOwner(0, [3, 2, 1], dimensions, 32).invalid, true);
  assert.equal(decodeDenseOctreePowerOwner(0xffff_ffff, [3, 2, 1], dimensions, 32).invalid, true);
});

test("paged descriptor decoding matches projection generation sentinels on the captured transition row", () => {
  const rowCell = [0, 8, 10] as const;
  const rowWord = packDenseOctreePowerOwner(rowCell, 2);
  assert.deepEqual(decodePagedOctreePowerOwner(rowWord, rowCell, [60, 45, 40], 4), {
    origin: rowCell, size: 2,
  });
  assert.deepEqual(decodePagedOctreePowerOwner(0xffff_ffff, rowCell, [60, 45, 40], 4), {
    origin: [0, 8, 8], size: 4,
  });
  assert.deepEqual(decodePagedOctreePowerOwner(0, rowCell, [60, 45, 40], 4), {
    origin: [0, 8, 8], size: 4,
  });
  assert.deepEqual(decodePagedOctreePowerOwner(0x8000_0123, rowCell, [60, 45, 40], 4), {
    origin: rowCell, size: 1,
  });
  assert.match(octreePowerDescriptorShader, /word==0u\|\|word==INVALID/);
  assert.match(octreePowerDescriptorShader,
    /!found\|\|encoded==0u\|\|encoded==INVALID\|\|encoded>capacity\)\{return canonicalOwner\(cell\)/,
    "generation-transition page-table values must resolve exactly like projection ownerAt");
  assert.match(octreePowerDescriptorShader, /return decodePagedOwner\(word,cell\)/);
});

test("descriptor WGSL has bounded 18-slot queries and fail-closed indirect publication", () => {
  assert.match(octreePowerDescriptorShader, /const DIRECTIONS:array<vec3i,18>/);
  assert.match(octreePowerDescriptorShader, /for\(var bit=0u;bit<18u;bit\+=1u\)/);
  assert.match(octreePowerDescriptorShader, /probe<hashCapacity/);
  assert.match(octreePowerDescriptorShader, /atomicOr\(&control\.flags,CAPACITY\)/);
  assert.match(octreePowerDescriptorShader, /indirectDispatch\[0\]=0u/);
  assert.doesNotMatch(octreePowerDescriptorShader, /texture_/);
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
  const device = await adapter.requestDevice();
  const shaderModule = device.createShaderModule({ code: octreePowerDescriptorShader });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);

  const d = [4, 4, 4] as const;
  const owners = device.createBuffer({ size: 4 * 4 * 4 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(owners, 0, new Uint32Array(64).fill(0x8000_0000));
  const headerWords = new Uint32Array(24);
  headerWords.set([linear([1, 1, 1], d), 0, 0, 1], 0);
  headerWords.set([linear([0, 0, 0], d), 0, 0, 1], 12);
  const headers = device.createBuffer({ size: headerWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(headers, 0, headerWords);
  const generator = new WebGPUOctreePowerDescriptor(device, 2);
  const readback = device.createBuffer({ size: 52, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  generator.encode(encoder, headers, owners, { dimensions: d, maximumLeafSize: 4, rowCount: 2, generation: 7, ownerMode: "dense" });
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

  const gpuCount = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(gpuCount, 0, new Uint32Array([1]));
  const gpuCountReadback = device.createBuffer({ size: 44, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const gpuCountEncoder = device.createCommandEncoder();
  generator.encode(gpuCountEncoder, headers, owners, {
    dimensions: d, maximumLeafSize: 4, rowCountBuffer: gpuCount, generation: 8, ownerMode: "dense",
  });
  gpuCountEncoder.copyBufferToBuffer(generator.control, 0, gpuCountReadback, 0, 32);
  gpuCountEncoder.copyBufferToBuffer(generator.dispatch, 0, gpuCountReadback, 32, 12);
  device.queue.submit([gpuCountEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await gpuCountReadback.mapAsync(GPUMapMode.READ);
  const gpuCountWords = new Uint32Array(gpuCountReadback.getMappedRange().slice(0)); gpuCountReadback.unmap();
  assert.deepEqual([...gpuCountWords.slice(0, 8)], [1, 1, 0, 0xffff_ffff, 0, 1, 0, 8]);
  assert.deepEqual([...gpuCountWords.slice(8, 11)], [1, 1, 1]);

  const capacityGenerator = new WebGPUOctreePowerDescriptor(device, 1);
  const capacityReadback = device.createBuffer({ size: 44, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const capacityEncoder = device.createCommandEncoder();
  capacityGenerator.encode(capacityEncoder, headers, owners, { dimensions: d, maximumLeafSize: 4, rowCount: 2, ownerMode: "dense" });
  capacityEncoder.copyBufferToBuffer(capacityGenerator.control, 0, capacityReadback, 0, 32);
  capacityEncoder.copyBufferToBuffer(capacityGenerator.dispatch, 0, capacityReadback, 32, 12);
  device.queue.submit([capacityEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await capacityReadback.mapAsync(GPUMapMode.READ);
  const capacity = new Uint32Array(capacityReadback.getMappedRange().slice(0)); capacityReadback.unmap();
  assert.equal(capacity[2], 1);
  assert.equal(capacity[3], 1);
  assert.ok((capacity[4] & OCTREE_POWER_DESCRIPTOR_ERROR.capacity) !== 0);
  assert.deepEqual([...capacity.slice(8, 11)], [0, 1, 1]);

  const malformedArena = new Uint32Array(16); malformedArena[15] = 0x4f57_4e52;
  const paged = device.createBuffer({ size: malformedArena.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paged, 0, malformedArena);
  const malformedReadback = device.createBuffer({ size: 44, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const malformedEncoder = device.createCommandEncoder();
  capacityGenerator.encode(malformedEncoder, headers, paged, { dimensions: d, maximumLeafSize: 4, rowCount: 1, ownerMode: "auto" });
  malformedEncoder.copyBufferToBuffer(capacityGenerator.control, 0, malformedReadback, 0, 32);
  malformedEncoder.copyBufferToBuffer(capacityGenerator.dispatch, 0, malformedReadback, 32, 12);
  device.queue.submit([malformedEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  await malformedReadback.mapAsync(GPUMapMode.READ);
  const malformedWords = new Uint32Array(malformedReadback.getMappedRange().slice(0)); malformedReadback.unmap();
  assert.ok((malformedWords[4] & OCTREE_POWER_DESCRIPTOR_ERROR.malformedOwner) !== 0);
  assert.deepEqual([...malformedWords.slice(8, 11)], [0, 1, 1]);
});
