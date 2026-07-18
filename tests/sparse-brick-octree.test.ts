import assert from "node:assert/strict";
import test from "node:test";
import {
  SPARSE_BRICK_GPU_LAYOUT,
  SPARSE_BRICK_INVALID_INDEX,
  SparseBrickOctreeGPU,
  mortonChild,
  mortonDecode3D,
  mortonEncode3D,
  mortonParent,
  packMaterialOwner,
  packSparseBrickPlan,
  planSparseBrickOctree,
  sparseBrickDenseFieldShader,
  sparseBrickDispatchDimensions,
  sparseBrickPublicationShader,
  unpackMaterialOwner,
} from "../lib/sparse-brick-octree";

test("Morton addressing is exact, reversible, and uses xyz octant order", () => {
  assert.equal(mortonEncode3D(1, 0, 0), 1n);
  assert.equal(mortonEncode3D(0, 1, 0), 2n);
  assert.equal(mortonEncode3D(0, 0, 1), 4n);
  assert.equal(mortonEncode3D(1, 1, 1), 7n);
  for (const coordinate of [
    { x: 0, y: 0, z: 0 },
    { x: 3, y: 5, z: 6 },
    { x: 65_535, y: 1_024, z: 999 },
    { x: 2 ** 21 - 1, y: 2 ** 21 - 2, z: 2 ** 20 + 3 },
  ]) assert.deepEqual(mortonDecode3D(mortonEncode3D(coordinate.x, coordinate.y, coordinate.z)), coordinate);

  const parent = mortonEncode3D(3, 5, 2);
  for (let child = 0; child < 8; child += 1) {
    const coordinate = mortonDecode3D(mortonChild(parent, child));
    assert.deepEqual(coordinate, {
      x: 6 + (child & 1),
      y: 10 + ((child >>> 1) & 1),
      z: 4 + ((child >>> 2) & 1),
    });
    assert.equal(mortonParent(mortonChild(parent, child)), parent);
  }
});

test("Morton and compact payload validation reject lossy addresses", () => {
  assert.throws(() => mortonEncode3D(-1, 0, 0), /non-negative/);
  assert.throws(() => mortonEncode3D(2 ** 21, 0, 0), /below/);
  assert.throws(() => mortonEncode3D(1.5, 0, 0), /integer/);
  assert.throws(() => mortonChild(0n, 8), /0\.\.7/);
  const packed = packMaterialOwner(0x1234, 0xabcd);
  assert.equal(packed, 0xabcd1234);
  assert.deepEqual(unpackMaterialOwner(packed), { materialId: 0x1234, ownerId: 0xabcd });
  assert.throws(() => packMaterialOwner(0x10000), /uint16/);
});

test("brick planning is canonical under permutation and duplicates", () => {
  const coordinates = [
    { x: 3, y: 3, z: 3 },
    { x: 1, y: 0, z: 0 },
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 0, z: 0 },
  ];
  const plan = planSparseBrickOctree(coordinates, { brickSize: 4 });
  const reverse = planSparseBrickOctree([...coordinates].reverse(), { brickSize: 4 });
  assert.equal(plan.maximumDepth, 2);
  assert.deepEqual(plan.levelOffsets, [0, 1, 3, 6]);
  assert.equal(plan.nodes.length, 6);
  assert.equal(plan.leaves.length, 3);
  assert.equal(plan.voxelCount, 3 * 4 ** 3);
  assert.deepEqual(plan.nodes.map((node) => node.morton), reverse.nodes.map((node) => node.morton));
  assert.deepEqual(plan.leaves.map((leaf) => leaf.morton), [0n, 1n, 63n]);

  const root = plan.nodes[0];
  assert.equal(root.childMask, 0b10000001);
  assert.equal(root.firstChild, 1);
  assert.equal(root.childCount, 2);
  assert.equal(root.leafIndex, SPARSE_BRICK_INVALID_INDEX);
  assert.deepEqual(plan.leaves.map((leaf) => leaf.nodeIndex), [3, 4, 5]);
  assert.deepEqual(plan.leaves.map((leaf) => leaf.voxelOffset), [0, 64, 128]);
  assert.deepEqual(plan.nodes.slice(3).map((node) => node.leafIndex), [0, 1, 2]);
});

test("explicit depth and empty plans retain stable level offsets", () => {
  const empty = planSparseBrickOctree([], { brickSize: 8, maximumDepth: 3 });
  assert.deepEqual(empty.levelOffsets, [0, 0, 0, 0, 0]);
  assert.equal(empty.voxelCount, 0);
  assert.throws(
    () => planSparseBrickOctree([{ x: 4, y: 0, z: 0 }], { brickSize: 4, maximumDepth: 2 }),
    /cannot contain/,
  );
});

test("packed topology matches the documented pointerless GPU ABI", () => {
  const plan = planSparseBrickOctree([{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }], { brickSize: 8 });
  const packed = packSparseBrickPlan(plan, 17);
  assert.equal(packed.nodes.byteLength, plan.nodes.length * SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes);
  assert.equal(packed.leaves.byteLength, plan.leaves.length * SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes);
  assert.deepEqual([...packed.counts], [plan.nodes.length, 2, 2 * 8 ** 3, 17, 0, plan.nodes.length * 8]);
  assert.equal(packed.topology.length, packed.nodes.length + packed.leaves.length);
  assert.deepEqual([...packed.nodes.slice(0, 8)], [0, 0, 0, 3, 1, 2, SPARSE_BRICK_INVALID_INDEX, 0]);
  assert.deepEqual([...packed.leaves.slice(0, 8)], [1, 0, 0, 0, 2, 512, 1, 0]);
  assert.deepEqual(SPARSE_BRICK_GPU_LAYOUT.geometryChannels, ["fluidSignedDistance", "solidSignedDistance", "solidFraction", "pressure"]);
});

test("portable stream dispatch crosses the one-dimensional WebGPU boundary", () => {
  assert.deepEqual(sparseBrickDispatchDimensions(0), [0, 1, 1]);
  assert.deepEqual(sparseBrickDispatchDimensions(1), [1, 1, 1]);
  assert.deepEqual(sparseBrickDispatchDimensions(256 * 65_535), [65_535, 1, 1]);
  assert.deepEqual(sparseBrickDispatchDimensions(256 * 65_535 + 1), [65_535, 2, 1]);
});

test("GPU publication is fail-closed on overflow and writes downstream indirect arguments", () => {
  assert.match(sparseBrickPublicationShader, /let overflow = requested > capacities/);
  assert.match(sparseBrickPublicationShader, /let valid = !any\(overflow\)/);
  assert.match(sparseBrickPublicationShader, /atomicStore\(&control\[0\], select\(0u, requested\.x, valid\)\)/);
  assert.match(sparseBrickPublicationShader, /atomicStore\(&control\[12\], flags\)/);
  assert.match(sparseBrickPublicationShader, /atomicStore\(&control\[20\], dispatchX\)/);
  assert.match(sparseBrickPublicationShader, /atomicStore\(&control\[25\], select\(0u, requested\.y, valid\)\)/);
  assert.equal((sparseBrickPublicationShader.match(/var<storage,/g) ?? []).length, 8,
    "publication must fit WebGPU's portable per-stage storage-buffer limit");
  assert.doesNotMatch(sparseBrickPublicationShader, /mapAsync|getMappedRange/);
  assert.doesNotMatch(`${SparseBrickOctreeGPU.prototype.encodePublish}\n${SparseBrickOctreeGPU.prototype.encodeFromDenseFields}`, /mapAsync|getMappedRange/);
});

test("dense publication preserves f32 physics fields and compact scene identity", () => {
  assert.match(sparseBrickDenseFieldShader, /payload\[geometryBase\] = bitcast<u32>\(phi\)/);
  assert.match(sparseBrickDenseFieldShader, /payload\[velocityBase \+ 3u\] = bitcast<u32>\(liquidFraction\)/);
  assert.match(sparseBrickDenseFieldShader, /payload\[materialOffset\] = select\(\(owner << 16u\) \| \(material & 0xffffu\), previousIdentity, preserveStatic && material == 0u\)/);
  assert.match(sparseBrickDenseFieldShader, /params\.origin\.w > 0 && previousMaterial >= u32\(params\.origin\.w\)/);
  assert.match(sparseBrickDenseFieldShader, /solid\.fraction > 0\.0 && solid\.owner >= 0/);
  assert.match(sparseBrickDenseFieldShader, /let brick = decodeMorton/);
  assert.match(sparseBrickDenseFieldShader, /let output = voxelOffset \+ localIndex/);
});

test("GPU resource class allocates explicit capacities, encodes both passes, and destroys cleanly", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", {
    configurable: true,
    value: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, INDIRECT: 8, UNIFORM: 16 },
  });
  class BufferMock {
    destroyCount = 0;
    constructor(readonly descriptor: GPUBufferDescriptor) {}
    destroy() { this.destroyCount += 1; }
  }
  const buffers: BufferMock[] = [];
  const writes: Array<{ offset: number; data: unknown }> = [];
  const dispatches: number[][] = [];
  const clears: unknown[] = [];
  const pipeline = { getBindGroupLayout: () => ({}) };
  const device = {
    queue: { writeBuffer: (_buffer: unknown, offset: number, data: unknown) => writes.push({ offset, data }) },
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const buffer = new BufferMock(descriptor); buffers.push(buffer); return buffer;
    },
    createShaderModule: () => ({}),
    createComputePipeline: () => pipeline,
    createBindGroup: (descriptor: unknown) => descriptor,
  } as unknown as GPUDevice;
  const pass = {
    setPipeline: () => undefined,
    setBindGroup: () => undefined,
    dispatchWorkgroups: (...dimensions: number[]) => dispatches.push(dimensions),
    end: () => undefined,
  };
  const encoder = {
    clearBuffer: (buffer: unknown) => clears.push(buffer),
    beginComputePass: () => pass,
  } as unknown as GPUCommandEncoder;
  try {
    const tree = new SparseBrickOctreeGPU(device, { brickSize: 4, nodeCapacity: 9, leafCapacity: 3 });
    assert.equal(tree.voxelCapacity, 192);
    assert.equal(tree.leafOffsetBytes, 512);
    assert.equal((tree.nodes as unknown as BufferMock).descriptor.size, 512 + 3 * 16);
    assert.equal((tree.leaves as unknown as BufferMock).descriptor.size, 512 + 3 * 16);
    assert.equal(tree.velocityOffsetBytes, 192 * 16);
    assert.equal(tree.materialOwnerOffsetBytes, 192 * 32);
    assert.equal((tree.geometry as unknown as BufferMock).descriptor.size, 192 * (16 + 16 + 4));
    assert.equal(tree.allocatedBytes, 512 + 3 * 16 + 192 * (16 + 16 + 4) + 128 + 64);
    assert.ok(((tree.dispatchIndirect as unknown as BufferMock).descriptor.usage & 8) !== 0);
    const sourceBuffer = buffers[0] as unknown as GPUBuffer;
    tree.encodePublish(encoder, {
      counts: sourceBuffer, topology: sourceBuffer,
      geometry: sourceBuffer, velocity: sourceBuffer, materialOwners: sourceBuffer,
      capacities: { nodes: 9, leaves: 3, voxels: 192 },
    });
    assert.equal(clears.length, 4);
    assert.deepEqual(dispatches[0], [1, 1, 1]);
    assert.equal(writes[0].offset, 32, "capacity words are initialized without a readback");
    assert.equal(writes[1].offset, 64, "arena offsets are resident alongside capacity counters");
    tree.encodeFromDenseFields(encoder, {
      levelSet: {} as GPUTextureView,
      velocity: {} as GPUTextureView,
      solidCells: sourceBuffer,
      dimensions: [12, 8, 4], cellSize: [0.1, 0.1, 0.1],
      fluidMaterialId: 4, solidMaterialId: 9,
    });
    assert.deepEqual(dispatches[1], [1, 1, 1]);
    tree.destroy(); tree.destroy();
    assert.ok(buffers.every((buffer) => buffer.destroyCount === 1));
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  }
});
