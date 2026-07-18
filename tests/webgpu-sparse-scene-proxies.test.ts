import assert from "node:assert/strict";
import test from "node:test";
import { packMaterialOwner, unpackMaterialOwner } from "../lib/sparse-brick-octree";
import {
  SPARSE_SCENE_PRIMITIVE_STRIDE_BYTES,
  SPARSE_SCENE_PRIMITIVE_TYPES,
  SparseSceneProxyVoxelizer,
  packSparseScenePrimitives,
  sampleSparseScenePrimitiveCell,
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

test("conservative cell occupancy catches surface intersections and selects the union identity", () => {
  const primitives: SparseScenePrimitive[] = [
    { kind: "box", center: [0, 0, 0], halfExtents: [1, 1, 1], materialId: 20, ownerId: 3 },
    { kind: "ellipsoid", center: [4, 0, 0], radii: [1, 2, 1], materialId: 21, ownerId: 4 },
  ];
  const surface = sampleSparseScenePrimitiveCell(primitives, [1.5, 0, 0], [1, 1, 1]);
  close(surface.solidSignedDistance, 0.5);
  assert.ok(surface.solidFraction > 0 && surface.solidFraction < 0.5,
    "half-diagonal conservative support includes a cell whose volume crosses the surface");
  assert.equal(surface.materialOwner, packMaterialOwner(20, 3));

  const interior = sampleSparseScenePrimitiveCell(primitives, [4, 0, 0], [0.25, 0.25, 0.25]);
  assert.equal(interior.solidFraction, 1);
  assert.equal(interior.materialOwner, packMaterialOwner(21, 4));
  const far = sampleSparseScenePrimitiveCell(primitives, [40, 0, 0], [1, 1, 1]);
  assert.equal(far.solidFraction, 0);
  assert.equal(far.materialOwner, packMaterialOwner(0));
  const empty = sampleSparseScenePrimitiveCell([], [0, 0, 0], [1, 1, 1]);
  assert.equal(empty.solidFraction, 0);
  assert.equal(empty.solidSignedDistance, Number.POSITIVE_INFINITY);
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

test("GPU shader walks published leaves, unions geometry, and preserves authored materials", () => {
  assert.match(sparseSceneProxyVoxelizationShader, /let leafBase = control\[16\] \+ leafIndex \* 4u/);
  assert.match(sparseSceneProxyVoxelizationShader, /let materialOffset = control\[18\] \+ output/);
  assert.match(sparseSceneProxyVoxelizationShader, /min\(previousDistance, bestDistance\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /max\(previousFraction, primitiveFraction\)/);
  assert.match(sparseSceneProxyVoxelizationShader, /previousMaterial == 0u/);
  assert.match(sparseSceneProxyVoxelizationShader, /primitiveFraction > 0\.0/);
  assert.match(sparseSceneProxyVoxelizationShader, /fn cylinderDistance/);
  assert.match(sparseSceneProxyVoxelizationShader, /fn ellipsoidDistance/);
  assert.equal((sparseSceneProxyVoxelizationShader.match(/var<storage,/g) ?? []).length, 4,
    "proxy voxelization stays well below the portable eight-storage-binding limit");
  assert.doesNotMatch(sparseSceneProxyVoxelizationShader, /texture_|mapAsync|getMappedRange/);
});

test("GPU resource uploads static primitives, binds whole offset arenas, and dispatches portably", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { STORAGE: 1, COPY_DST: 2, UNIFORM: 4 },
  });
  class BufferMock {
    destroyCount = 0;
    constructor(readonly descriptor: GPUBufferDescriptor) {}
    destroy() { this.destroyCount += 1; }
  }
  const buffers: BufferMock[] = [];
  const writes: Array<{ buffer: unknown; offset: number; data: AllowSharedBufferSource }> = [];
  const bindGroups: GPUBindGroupDescriptor[] = [];
  const dispatches: number[][] = [];
  const pipeline = { getBindGroupLayout: () => ({}) };
  const device = {
    queue: { writeBuffer: (buffer: unknown, offset: number, data: AllowSharedBufferSource) => writes.push({ buffer, offset, data }) },
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const buffer = new BufferMock(descriptor); buffers.push(buffer); return buffer;
    },
    createShaderModule: () => ({}),
    createComputePipeline: () => pipeline,
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => { bindGroups.push(descriptor); return descriptor; },
  } as unknown as GPUDevice;
  const treeBuffers = {
    control: { name: "control" }, topology: { name: "topology arena" }, payload: { name: "payload arena" },
  };
  const tree = {
    ...treeBuffers,
    voxelCapacity: 256 * 65_535 + 1,
    leafOffsetBytes: 768,
    materialOwnerOffsetBytes: 4096,
  } as unknown as import("../lib/sparse-brick-octree").SparseBrickOctreeGPU;
  const pass = {
    setPipeline: () => undefined,
    setBindGroup: () => undefined,
    dispatchWorkgroups: (...dimensions: number[]) => dispatches.push(dimensions),
    end: () => undefined,
  };
  const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  try {
    const voxelizer = new SparseSceneProxyVoxelizer(device, tree, [{
      kind: "box", center: [1, 2, 3], halfExtents: [4, 5, 6], materialId: 12, ownerId: 13,
    }], { cellSize: [0.1, 0.2, 0.3], worldOrigin: [-1, -2, -3] });
    assert.equal(voxelizer.primitiveCount, 1);
    assert.equal(voxelizer.allocatedBytes, 96);
    assert.deepEqual(buffers.map((buffer) => buffer.descriptor.size), [48, 48]);
    assert.equal(writes.length, 2);
    const params = new Float32Array(writes[1].data as ArrayBuffer);
    const paramUints = new Uint32Array(writes[1].data as ArrayBuffer);
    assert.deepEqual([...params.slice(0, 3)], [-1, -2, -3]);
    close(params[4], 0.1); close(params[5], 0.2); close(params[6], 0.3);
    assert.equal(paramUints[8], 1);
    const entries = Array.from(bindGroups[0].entries);
    assert.equal((entries[1].resource as GPUBufferBinding).buffer, treeBuffers.topology);
    assert.equal((entries[1].resource as GPUBufferBinding).offset, undefined,
      "leaf offset comes from control word 16, not a potentially unaligned binding offset");
    assert.equal((entries[2].resource as GPUBufferBinding).buffer, treeBuffers.payload);
    assert.equal((entries[2].resource as GPUBufferBinding).offset, undefined,
      "material offset comes from control word 18 in the shared payload arena");
    voxelizer.encode(encoder);
    assert.deepEqual(dispatches, [[65_535, 2, 1]]);
    voxelizer.destroy(); voxelizer.destroy();
    assert.ok(buffers.every((buffer) => buffer.destroyCount === 1));
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  }
});
