import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { packMaterialOwner, unpackMaterialOwner } from "../lib/sparse-brick-octree";
import { SVO_FIELD_PROGRAM_BLOCK_WORDS, type SvoFieldProgram } from "../lib/svo-field-program";
import {
  SPARSE_SCENE_FIELD_PROGRAM_CAPACITY,
  SPARSE_SCENE_MAINTENANCE_OVERFLOW,
  SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES,
  SPARSE_SCENE_PRIMITIVE_TYPES,
  SparseSceneProxyVoxelizer,
  sparseSceneRevisionIncomplete,
  packSparseSceneFieldProgramArena,
  packSparseScenePrimitives,
  sparseSceneFieldProgramSlots,
  sparseSceneFieldPrograms,
  sampleSparseScenePrimitiveCell,
  sparseScenePrimitiveBounds,
  sparseScenePrimitiveSignedDistance,
  sparseSceneProxyVoxelizationShader,
  type SparseScenePrimitive,
} from "../lib/webgpu-sparse-scene-proxies";

const close = (actual: number, expected: number, tolerance = 1e-6) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

test("scene primitive ABI is compact, integer-exact, and normalizes orientations", () => {
  const primitives: SparseScenePrimitive[] = [
    { kind: "box", center: [1, 2, 3], halfExtents: [4, 5, 6], materialId: 7, ownerId: 8 },
    { kind: "cylinder", center: [-1, -2, -3], radius: 0.5, halfHeight: 2, orientation: [0, 0, 2, 2], materialId: 9 },
    { kind: "ellipsoid", center: [0, 1, 0], radii: [1, 2, 3], materialId: 10, ownerId: 11 },
  ];
  const packed = packSparseScenePrimitives(primitives);
  const floats = new Float32Array(packed.buffer);
  assert.equal(packed.byteLength, primitives.length * SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES);
  assert.deepEqual([...floats.slice(0, 3)], [1, 2, 3]);
  assert.equal(packed[3], SPARSE_SCENE_PRIMITIVE_TYPES.box);
  assert.deepEqual([...floats.slice(4, 7)], [4, 5, 6]);
  assert.deepEqual(unpackMaterialOwner(packed[7]), { materialId: 7, ownerId: 8 });
  assert.deepEqual([...floats.slice(8, 12)], [0, 0, 0, 1], "axis-aligned boxes have canonical identity rotation");
  assert.equal(packed[15], SPARSE_SCENE_PRIMITIVE_TYPES.cylinder);
  assert.deepEqual(unpackMaterialOwner(packed[19]), { materialId: 9, ownerId: 0xffff });
  close(floats[22], Math.SQRT1_2);
  close(floats[23], Math.SQRT1_2);
});

test("CPU mirrors evaluate boxes, oriented capped cylinders, and ellipsoids", () => {
  const box: SparseScenePrimitive = {
    kind: "box", center: [0, 0, 0], halfExtents: [1, 2, 3], materialId: 1,
  };
  close(sparseScenePrimitiveSignedDistance(box, [0, 0, 0]), -1);
  close(sparseScenePrimitiveSignedDistance(box, [2, 0, 0]), 1);
  close(sparseScenePrimitiveSignedDistance(box, [2, 3, 3]), Math.SQRT2);

  const cylinder: SparseScenePrimitive = {
    kind: "cylinder", center: [0, 0, 0], radius: 1, halfHeight: 2, materialId: 2,
  };
  close(sparseScenePrimitiveSignedDistance(cylinder, [0, 0, 0]), -1);
  close(sparseScenePrimitiveSignedDistance(cylinder, [0, 3, 0]), 1);
  close(sparseScenePrimitiveSignedDistance(cylinder, [2, 0, 0]), 1);

  const horizontalCylinder: SparseScenePrimitive = {
    ...cylinder, orientation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
  };
  close(sparseScenePrimitiveSignedDistance(horizontalCylinder, [2, 0, 0]), 0);

  const ellipsoid: SparseScenePrimitive = {
    kind: "ellipsoid", center: [0, 0, 0], radii: [2, 1, 0.5], materialId: 3,
  };
  close(sparseScenePrimitiveSignedDistance(ellipsoid, [0, 0, 0]), -0.5);
  close(sparseScenePrimitiveSignedDistance(ellipsoid, [2, 0, 0]), 0);
});

// The identity this mirror reports is the material half alone. The GPU pass's
// identity word carries a baked oct8 normal in its high half now, and this
// oracle has no way to evaluate one — see `sampleSparseScenePrimitiveCell`.
test("conservative cell occupancy catches surface intersections and selects the union material", () => {
  const primitives: SparseScenePrimitive[] = [
    { kind: "box", center: [0, 0, 0], halfExtents: [1, 1, 1], materialId: 20, ownerId: 3 },
    { kind: "ellipsoid", center: [4, 0, 0], radii: [1, 2, 1], materialId: 21, ownerId: 4 },
  ];
  const surface = sampleSparseScenePrimitiveCell(primitives, [1.5, 0, 0], [1, 1, 1]);
  close(surface.solidSignedDistance, 0.5);
  assert.ok(surface.solidFraction > 0 && surface.solidFraction < 0.5,
    "half-diagonal conservative support includes a cell whose volume crosses the surface");
  assert.equal(surface.materialOwner, 20);

  const interior = sampleSparseScenePrimitiveCell(primitives, [4, 0, 0], [0.25, 0.25, 0.25]);
  assert.equal(interior.solidFraction, 1);
  assert.equal(interior.materialOwner, 21);
  const far = sampleSparseScenePrimitiveCell(primitives, [40, 0, 0], [1, 1, 1]);
  assert.equal(far.solidFraction, 0);
  assert.equal(far.materialOwner, 0);
  const empty = sampleSparseScenePrimitiveCell([], [0, 0, 0], [1, 1, 1]);
  assert.equal(empty.solidFraction, 0);
  assert.equal(empty.solidSignedDistance, Number.POSITIVE_INFINITY);
});

test("noise foliage voxel occupancy follows the density threshold without SDF dilation", () => {
  const foliage: SparseScenePrimitive = {
    kind: "smooth-union-cluster",
    center: [0, 0, 0],
    lobeRadii: [0.18, 0.11, 0.15],
    packing: {
      field: "noise-foliage",
      smoothRadius_m: 0,
      seed: 0x51a9,
      clusterPeriod_m: 0.075,
      detailPeriod_m: 0.018,
      threshold: 0.55,
      clusterWeight: 0.62,
      detailWeight: 0.38,
      interiorBias: 0.20,
    },
    materialId: 22,
  };
  const cellSize = [0.004, 0.004, 0.004] as const;
  const cellRadius = 0.5 * Math.hypot(...cellSize);
  let justOutside: readonly [number, number, number] | undefined;
  let inside: readonly [number, number, number] | undefined;
  for (let z = -0.14; z <= 0.14 && (!justOutside || !inside); z += 0.01) {
    for (let y = -0.10; y <= 0.10 && (!justOutside || !inside); y += 0.01) {
      for (let x = -0.17; x <= 0.17 && (!justOutside || !inside); x += 0.01) {
        const point = [x, y, z] as const;
        const residual = sparseScenePrimitiveSignedDistance(foliage, point);
        if (residual < 0) inside ??= point;
        // This is the regression case: the residual is outside by sign but so
        // small after its conservative gradient scaling that the generic planar
        // SDF law would publish a partially solid voxel.
        if (residual > 0 && residual < cellRadius) justOutside ??= point;
      }
    }
  }
  assert.ok(justOutside, "fixture must include an outside residual inside one planar coverage radius");
  assert.ok(inside, "fixture must include occupied density");

  const outsideSample = sampleSparseScenePrimitiveCell([foliage], justOutside, cellSize);
  assert.ok(outsideSample.solidSignedDistance > 0);
  assert.equal(outsideSample.solidFraction, 0, "positive density residual must remain an empty voxel");
  assert.equal(outsideSample.materialOwner, 0);

  const insideSample = sampleSparseScenePrimitiveCell([foliage], inside, cellSize);
  assert.ok(insideSample.solidSignedDistance < 0);
  assert.equal(insideSample.solidFraction, 1, "negative density residual defines a solid voxel");
  assert.equal(insideSample.materialOwner, foliage.materialId);
});

test("primitive input validation rejects lossy IDs and degenerate geometry", () => {
  assert.throws(() => packSparseScenePrimitives([
    { kind: "box", center: [0, 0, 0], halfExtents: [1, 0, 1], materialId: 1 },
  ]), /positive/);
  assert.throws(() => packSparseScenePrimitives([
    { kind: "cylinder", center: [0, 0, 0], radius: 1, halfHeight: 1, orientation: [0, 0, 0, 0], materialId: 1 },
  ]), /nonzero length/);
  assert.throws(() => packSparseScenePrimitives([
    { kind: "ellipsoid", center: [0, 0, 0], radii: [1, 1, 1], materialId: 0 },
  ]), /nonzero uint16/);
});

test("field-program records share one arena block per distinct tape", () => {
  // A recursive crown in miniature: many masses, each with its own centre, shade
  // and envelope, drawn from a handful of quantised tapes. The arena's capacity
  // is a ceiling on *shapes*, so a crown far past it must still pack.
  const tape = (radius: number): SvoFieldProgram => ({
    ops: [{ op: "ellipsoid", out: 1, point: 0, parameters: { a: radius, b: radius * 0.78, c: radius } }],
    result: 1,
  });
  const distinct = 3;
  const records = SPARSE_SCENE_FIELD_PROGRAM_CAPACITY * 4;
  const primitives: SparseScenePrimitive[] = Array.from({ length: records }, (_, index) => ({
    kind: "field-program",
    center: [0.01 * index, 0.5, 0],
    envelopeRadii: [0.05, 0.04, 0.05],
    program: tape(0.02 + 0.01 * (index % distinct)),
    materialId: 1 + index,
    ownerId: 1 + index,
  }));

  const slots = sparseSceneFieldProgramSlots(primitives);
  const programs = sparseSceneFieldPrograms(primitives);
  assert.equal(programs.length, distinct, "one block per distinct tape, not one per record");
  assert.deepEqual(slots.slice(0, 2 * distinct), [0, 1, 2, 0, 1, 2], "slots repeat with the tapes");
  primitives.forEach((primitive, index) => {
    assert.deepEqual(programs[slots[index]], (primitive as { program: SvoFieldProgram }).program,
      `record ${index} must name the block holding its own tape`);
  });

  // The record packer and the arena packer derive the numbering the same way,
  // so neither may throw and the slot in the type word must be the shared one.
  const packed = packSparseScenePrimitives(primitives);
  const words = packed.length / (SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES / 4);
  assert.equal(words, records);
  for (let index = 0; index < 2 * distinct; index += 1) {
    const type = packed[index * (SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES / 4) + 3];
    assert.equal(type & 0xff, SPARSE_SCENE_PRIMITIVE_TYPES["field-program"]);
    assert.equal(type >>> 8, index % distinct);
  }
  assert.equal(
    packSparseSceneFieldProgramArena(primitives).length,
    SPARSE_SCENE_FIELD_PROGRAM_CAPACITY * SVO_FIELD_PROGRAM_BLOCK_WORDS,
  );
});

test("primitive bounds conservatively rotate the complete authored shape", () => {
  const box = sparseScenePrimitiveBounds({
    kind: "box", center: [1, 2, 3], halfExtents: [2, 1, 0.5],
    orientation: [0, 0, Math.SQRT1_2, Math.SQRT1_2], materialId: 1,
  });
  close(box.minimum[0], 0); close(box.maximum[0], 2);
  close(box.minimum[1], 0); close(box.maximum[1], 4);
  const torus = sparseScenePrimitiveBounds({
    kind: "torus", center: [0, 0, 0], majorRadius: 3, minorRadius: 1, materialId: 2,
  });
  assert.deepEqual(torus, { minimum: [-4, -1, -4], maximum: [4, 1, 4] });
  const capsule = sparseScenePrimitiveBounds({
    kind: "capsule", center: [0, 0, 0], radius: 0.5, halfLength: 2, materialId: 3,
  });
  assert.deepEqual(capsule, { minimum: [-0.5, -2.5, -0.5], maximum: [0.5, 2.5, 0.5] });
});

test("GPU maintenance invalidates before rebuild, bins brick candidates, clears old scene payload, and finalizes", () => {
  assert.match(sparseSceneProxyVoxelizationShader, /fn invalidateDirtyBricks/);
  assert.match(sparseSceneProxyVoxelizationShader, /topologyAnd\(nodeIndex\*8u\+7u,~OCCUPANCY_READY\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /topologyOr\(nodeIndex\*8u\+7u,BRICK_DIRTY\|BRICK_QUEUED\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /fn binDirtyBrickCandidates/);
  assert.match(sparseSceneProxyVoxelizationShader, /boundsOverlap\(leafBounds\(leafIndex\),arenaBounds\(primitiveBoundsOffset\(\)\+primitiveIndex\*8u\)\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /let candidateCount=min\(atomicLoad\(&maintenance\[record\+1u\]\),candidatesPerBrick\(\)\)/);
  assert.doesNotMatch(sparseSceneProxyVoxelizationShader, /for \(var primitiveIndex = 0u; primitiveIndex < params\.primitiveCount/);
  assert.match(sparseSceneProxyVoxelizationShader, /payload\[geometryBase\+1u\]=bitcast<u32>\(bestDistance\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /payload\[geometryBase\+2u\]=bitcast<u32>\(primitiveFraction\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /fn finalizeDirtyBricks/);
  assert.match(sparseSceneProxyVoxelizationShader, /topologyStore\(nodeIndex\*8u\+7u,packed\|\(lifecycle&BRICK_ACTIVE\)\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /fn cylinderDistance/);
  assert.match(sparseSceneProxyVoxelizationShader, /fn ellipsoidDistance/);
  assert.match(sparseSceneProxyVoxelizationShader, /fn primitiveUsesThresholdOccupancy/);
  assert.match(sparseSceneProxyVoxelizationShader,
    /select\(cellRadius, -cellRadius, candidate < 0\.0\),\s*primitiveUsesThresholdOccupancy\(primitive\)/);
  assert.equal((sparseSceneProxyVoxelizationShader.match(/var<storage,/g) ?? []).length, 4,
    "proxy voxelization respects the four-storage-buffer design ceiling");
  assert.doesNotMatch(sparseSceneProxyVoxelizationShader, /texture_|mapAsync|getMappedRange/);
});

test("a per-brick candidate overflow degrades that brick's detail and does not withhold the revision", () => {
  const { dirtyBricks, candidates, topology } = SPARSE_SCENE_MAINTENANCE_OVERFLOW;
  // A truncated dirty list and a skipped relocating brick both leave some brick
  // holding the previous revision's voxels, so neither may publish. An
  // over-subscribed brick was rebuilt from the first candidatesPerBrick()
  // primitives, so it is the new revision with less detail in it.
  assert.equal(sparseSceneRevisionIncomplete(dirtyBricks), true);
  assert.equal(sparseSceneRevisionIncomplete(topology), true);
  assert.equal(sparseSceneRevisionIncomplete(candidates), false);
  assert.equal(sparseSceneRevisionIncomplete(candidates | topology), true);
  assert.equal(sparseSceneRevisionIncomplete(0), false);

  // Every store of completedRevision consults the mask rather than the raw flag
  // word, and an over-subscribed brick still reaches its occupancy summary --
  // the early-out that used to skip it left the payload current and the summary
  // stale, and its record flag is now read only where the candidates are.
  const stores = sparseSceneProxyVoxelizationShader.match(/atomicStore\(&maintenance\[stateOffset\(\)\+3u\],sceneRevision\(\)\)/g);
  assert.equal(stores?.length, 3);
  assert.equal((sparseSceneProxyVoxelizationShader
    .match(/atomicLoad\(&maintenance\[stateOffset\(\)\+1u\]\)&INCOMPLETE_OVERFLOW\)==0u/g) ?? []).length, 3);
  assert.doesNotMatch(sparseSceneProxyVoxelizationShader, /atomicLoad\(&maintenance\[stateOffset\(\)\+1u\]\)==0u/);
  assert.doesNotMatch(sparseSceneProxyVoxelizationShader, /if\(atomicLoad\(&maintenance\[record\+2u\]\)!=0u\)\{return;\}/);
});

test("live GPU resources are fixed-capacity, hot-written, and encode invalidate-to-finalize ordering", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { STORAGE: 1, COPY_DST: 2, UNIFORM: 4, INDIRECT: 8, COPY_SRC: 16 },
  });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { COMPUTE: 1 } });
  class BufferMock {
    destroyCount = 0;
    constructor(readonly descriptor: GPUBufferDescriptor) {}
    destroy() { this.destroyCount += 1; }
  }
  const buffers: BufferMock[] = [];
  const writes: Array<{ buffer: unknown; offset: number; data: AllowSharedBufferSource }> = [];
  const bindGroups: GPUBindGroupDescriptor[] = [];
  const dispatches: number[][] = [];
  const indirectDispatches: Array<{ buffer: unknown; offset: number }> = [];
  const passLabels: string[] = [];
  const clears: Array<{ buffer: unknown; offset?: number; size?: number }> = [];
  const copies: Array<{ source: unknown; sourceOffset: number; destination: unknown; destinationOffset: number; size: number }> = [];
  const device = {
    queue: { writeBuffer: (buffer: unknown, offset: number, data: AllowSharedBufferSource) => writes.push({ buffer, offset, data }) },
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const buffer = new BufferMock(descriptor); buffers.push(buffer); return buffer;
    },
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipeline: (descriptor: GPUComputePipelineDescriptor) => ({ descriptor }),
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => { bindGroups.push(descriptor); return descriptor; },
  } as unknown as GPUDevice;
  const structure = { name: "structural arena" };
  const treeBuffers = { structure, control: structure, topology: structure, payload: { name: "payload arena" } };
  const tree = {
    ...treeBuffers,
    brickSize: 8,
    nodeCapacity: 128,
    leafCapacity: 100,
    voxelCapacity: 51_200,
    leafOffsetBytes: 768,
    topologyOffsetBytes: 512,
    materialOwnerOffsetBytes: 4096,
    // The banded lanes a `dense` tree still declares. Zero offsets are inert here:
    // the codec that would read them is not compiled into a `dense` shader at all.
    leafPayloadMode: "dense", bandedLaneWordOffsets: [0, 0, 0, 0, 0], bandedRecordCapacity: 0,
    payloadLayout: { bandedBlobWords: 0 },
  } as unknown as import("../lib/sparse-brick-octree").SparseBrickOctreeGPU;
  const pass = {
    setPipeline: () => undefined,
    setBindGroup: () => undefined,
    dispatchWorkgroups: (...dimensions: number[]) => dispatches.push(dimensions),
    dispatchWorkgroupsIndirect: (buffer: GPUBuffer, offset: number) => indirectDispatches.push({ buffer, offset }),
    end: () => undefined,
  };
  const encoder = {
    clearBuffer: (buffer: unknown, offset?: number, size?: number) => clears.push({ buffer, offset, size }),
    copyBufferToBuffer: (source: unknown, sourceOffset: number, destination: unknown, destinationOffset: number, size: number) =>
      copies.push({ source, sourceOffset, destination, destinationOffset, size }),
    beginComputePass: (descriptor: GPUComputePassDescriptor) => { passLabels.push(descriptor.label ?? ""); return pass; },
  } as unknown as GPUCommandEncoder;
  try {
    const voxelizer = new SparseSceneProxyVoxelizer(device, tree, {
      cellSize: [0.1, 0.2, 0.3], worldOrigin: [-1, -2, -3], finestLevel: 6,
      primitiveCapacity: 2, dirtyRegionCapacity: 2, dirtyBrickCapacity: 3,
      candidatesPerDirtyBrick: 4,
    });
    assert.equal(voxelizer.primitiveCount, 0);
    // 592, not 532, and every one of the 60 bytes is named: the parameter uniform
    // grew 32 bytes for the banded lane offsets and their two capacities, the
    // maintenance state block grew 16 for a fourth indirect-dispatch triple, and the
    // dispatch-argument buffer grew 12 to carry it. `dense` allocates all three
    // regardless — the arena is fixed-capacity, and a size that depends on a lever
    // is a size no test can pin.
    assert.equal(voxelizer.allocatedBytes, 592);
    assert.deepEqual(buffers.map((buffer) => buffer.descriptor.size), [96, 320, 48, 128]);
    assert.equal(writes.length, 0, "construction allocates capacity but does not bake scene content");
    voxelizer.publish({
      revision: 1,
      primitives: [{ kind: "box", center: [1, 2, 3], halfExtents: [4, 5, 6], materialId: 32, ownerId: 13 }],
      dirtyRegions: [{ minimum: [-3, -3, -3], maximum: [5, 7, 9] }],
    });
    assert.equal(voxelizer.primitiveCount, 1);
    assert.equal(voxelizer.sceneRevision, 1);
    assert.equal(writes.length, 4);
    const params = new Float32Array(writes[3].data as ArrayBuffer);
    const paramUints = new Uint32Array(writes[3].data as ArrayBuffer);
    assert.deepEqual([...params.slice(0, 3)], [-1, -2, -3]);
    close(params[4], 0.1); close(params[5], 0.2); close(params[6], 0.3);
    assert.equal(paramUints[8], 1);
    assert.equal(paramUints[9], 1);
    assert.equal(paramUints[10], 6);
    assert.equal(paramUints[11], 1);
    assert.equal(paramUints[22], 128, "the one arena binding carries a topology-relative base word");
    const entries = Array.from(bindGroups[0].entries);
    assert.equal((entries[0].resource as GPUBufferBinding).buffer, treeBuffers.structure);
    assert.equal((entries[0].resource as GPUBufferBinding).offset, undefined,
      "control and topology share one whole structural-arena binding");
    assert.equal((entries[1].resource as GPUBufferBinding).buffer, treeBuffers.payload);
    assert.equal((entries[1].resource as GPUBufferBinding).offset, undefined,
      "material offset comes from control word 18 in the shared payload arena");
    assert.equal(voxelizer.encodeMaintenance(encoder), true);
    assert.equal(voxelizer.encodeMaintenance(encoder), false, "a revision is maintained exactly once");
    assert.deepEqual(clears.map(({ offset, size }) => [offset, size]), [[224, 96]]);
    assert.deepEqual(passLabels, [
      "Invalidate live scene dirty bricks",
      "Prepare live scene maintenance dispatches",
      "Bin live scene primitives into dirty bricks",
      "Rebuild live scene dirty brick payloads",
      "Finalize live scene dirty bricks",
    ]);
    assert.deepEqual(dispatches, [[2, 1, 1], [1, 1, 1]]);
    assert.deepEqual(copies.map(({ sourceOffset, destinationOffset, size }) => [sourceOffset, destinationOffset, size]), [[256, 0, 48]]);
    assert.equal(copies[0].source, buffers[1]);
    assert.equal(copies[0].destination, buffers[2]);
    assert.deepEqual(indirectDispatches.map(({ offset }) => offset), [0, 12, 24]);
    assert.ok(indirectDispatches.every(({ buffer }) => buffer === buffers[2]),
      "writable maintenance storage is never also consumed as indirect arguments");
    const allocationCount = buffers.length;
    voxelizer.publish({
      revision: 2,
      primitives: [{ kind: "box", center: [2, 2, 3], halfExtents: [4, 5, 6], materialId: 32, ownerId: 13 }],
      dirtyRegions: [{ minimum: [-3, -3, -3], maximum: [6, 7, 9] }],
    });
    assert.equal(buffers.length, allocationCount, "hot publication never allocates a GPU resource");
    voxelizer.destroy(); voxelizer.destroy();
    assert.ok(buffers.every((buffer) => buffer.destroyCount === 1));
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
    if (previousStage) Object.defineProperty(globalThis, "GPUShaderStage", previousStage);
    else delete (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage;
  }
});

test("live publication rejects overflow, missing invalidation coverage, and revision replacement", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1, COPY_DST: 2, UNIFORM: 4, INDIRECT: 8, COPY_SRC: 16 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { COMPUTE: 1 } });
  const device = {
    queue: { writeBuffer: () => undefined }, createBuffer: () => ({ destroy: () => undefined }),
    createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createComputePipeline: () => ({}), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const structure = {};
  const tree = {
    brickSize: 8, leafCapacity: 2, structure, control: structure, topology: structure,
    topologyOffsetBytes: 512, payload: {},
    // The banded lanes a `dense` tree still declares. Zero offsets are inert here:
    // the codec that would read them is not compiled into a `dense` shader at all.
    leafPayloadMode: "dense", bandedLaneWordOffsets: [0, 0, 0, 0, 0], bandedRecordCapacity: 0,
    payloadLayout: { bandedBlobWords: 0 },
  } as unknown as import("../lib/sparse-brick-octree").SparseBrickOctreeGPU;
  try {
    const voxelizer = new SparseSceneProxyVoxelizer(device, tree, {
      cellSize: [1, 1, 1], primitiveCapacity: 1, dirtyRegionCapacity: 1,
      dirtyBrickCapacity: 1, candidatesPerDirtyBrick: 1,
    });
    const primitive: SparseScenePrimitive = { kind: "box", center: [0, 0, 0], halfExtents: [1, 1, 1], materialId: 1 };
    assert.throws(() => voxelizer.publish({ revision: 1, primitives: [primitive], dirtyRegions: [] }), /old\/new dirty bounds/);
    assert.throws(() => voxelizer.publish({ revision: 1, primitives: [primitive, primitive], dirtyRegions: [{ minimum: [-1, -1, -1], maximum: [1, 1, 1] }] }), /primitive capacity/);
    voxelizer.publish({ revision: 1, primitives: [primitive], dirtyRegions: [{ minimum: [-1, -1, -1], maximum: [1, 1, 1] }] });
    assert.throws(() => voxelizer.publish({ revision: 2, primitives: [primitive], dirtyRegions: [{ minimum: [-1, -1, -1], maximum: [1, 1, 1] }] }), /pending scene revision/);
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
    if (previousStage) Object.defineProperty(globalThis, "GPUShaderStage", previousStage);
    else delete (globalThis as { GPUShaderStage?: unknown }).GPUShaderStage;
  }
});

test("Dawn accepts live maintenance storage and indirect dispatches in disjoint buffers", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for Dawn validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const structure = device.createBuffer({ size: 4608, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const payload = device.createBuffer({ size: 8192, usage: GPUBufferUsage.STORAGE });
  const tree = {
    structure, control: structure, topology: structure, topologyOffsetBytes: 512,
    payload, brickSize: 8, leafCapacity: 1,
    sceneGeometryOffsetBytes: 0, sceneMaterialOwnerOffsetBytes: 4096,
    // The banded lanes a `dense` tree still declares. Zero offsets are inert here:
    // the codec that would read them is not compiled into a `dense` shader at all.
    leafPayloadMode: "dense", bandedLaneWordOffsets: [0, 0, 0, 0, 0], bandedRecordCapacity: 0,
    payloadLayout: { bandedBlobWords: 0 },
  } as unknown as import("../lib/sparse-brick-octree").SparseBrickOctreeGPU;
  device.pushErrorScope("validation");
  const voxelizer = new SparseSceneProxyVoxelizer(device, tree, {
    cellSize: [1, 1, 1], primitiveCapacity: 1, dirtyRegionCapacity: 1,
    dirtyBrickCapacity: 1, candidatesPerDirtyBrick: 1,
  });
  voxelizer.publish({
    revision: 1,
    primitives: [{ kind: "box", center: [0, 0, 0], halfExtents: [1, 1, 1], materialId: 1 }],
    dirtyRegions: [{ minimum: [-1, -1, -1], maximum: [1, 1, 1] }],
  });
  const encoder = device.createCommandEncoder();
  assert.equal(voxelizer.encodeMaintenance(encoder), true);
  encoder.finish();
  const error = await device.popErrorScope();
  assert.equal(error, null, error?.message);
  voxelizer.destroy();
  structure.destroy();
  payload.destroy();
  device.destroy();
});
