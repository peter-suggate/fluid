import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { SparseBrickOctreeGPU } from "../lib/sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE, decodeSvoBrickLifecycle } from "../lib/svo-brick-occupancy";
import {
  SPARSE_BRICK_TOPOLOGY_MUTATION,
  WebGpuSparseBrickTopologyMutator,
  packSparseBrickTopologyMutationWorklist,
  sparseBrickTopologyMutationNodeReserve,
  webgpuSparseBrickTopologyMutationWGSL,
} from "../lib/webgpu-sparse-brick-topology-mutation";

test("mutation worklists deduplicate coordinates and reserve explicit receipts", () => {
  const packed = packSparseBrickTopologyMutationWorklist([
    { x: 3, y: 2, z: 1 }, { x: 3, y: 2, z: 1 }, { x: 0, y: 0, z: 0 },
  ], 19, 4);
  assert.equal(packed.length, SPARSE_BRICK_TOPOLOGY_MUTATION.headerWords
    + 4 * SPARSE_BRICK_TOPOLOGY_MUTATION.recordWords);
  assert.deepEqual([...packed.slice(0, 8)], [2, 19, 4, 0, 0, 0, 0, 0]);
  assert.deepEqual([...packed.slice(8, 16)], [3, 2, 1, 1, 0, 0, 0, 1]);
  assert.throws(() => packSparseBrickTopologyMutationWorklist([{ x: -1, y: 0, z: 0 }], 1), /fit uint32/);
  assert.throws(() => packSparseBrickTopologyMutationWorklist([{ x: 0, y: 0, z: 0 }], 1, 0), /contain/);
  assert.equal(sparseBrickTopologyMutationNodeReserve(5, 100), 4_001);
  assert.equal(sparseBrickTopologyMutationNodeReserve(0, 100), 1);
});

test("GPU mutation is bounded, copy-on-write, and publishes parent links last", () => {
  assert.match(webgpuSparseBrickTopologyMutationWGSL, /if\(first>capacity\|\|newCount>capacity-first\)/);
  assert.match(webgpuSparseBrickTopologyMutationWGSL,
    /atomicOr\(&structure\[TOPOLOGY_BASE\+parent\*8u\+7u\],RELOCATING\)/);
  const copyAt = webgpuSparseBrickTopologyMutationWGSL.indexOf("copyNode(oldFirst");
  const firstChildAt = webgpuSparseBrickTopologyMutationWGSL.indexOf("storeNode(parent,4u,first)");
  const clearAt = webgpuSparseBrickTopologyMutationWGSL.indexOf("~RELOCATING");
  assert.ok(copyAt >= 0 && firstChildAt > copyAt && clearAt > firstChildAt);
  assert.match(webgpuSparseBrickTopologyMutationWGSL, /atomicOr\(&requests\[3\],flag\)/);
  assert.match(webgpuSparseBrickTopologyMutationWGSL, /atomicAdd\(&structure\[13\],newCount\)/);
  assert.match(webgpuSparseBrickTopologyMutationWGSL, /storeNode\(node,7u,ACTIVE\|DIRTY\|QUEUED\)/);
  assert.doesNotMatch(webgpuSparseBrickTopologyMutationWGSL, /mapAsync|getMappedRange/);
});

test("mutation encode rejects domains that cannot fit the declared root", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2 } });
  const buffer = { destroy: () => undefined } as unknown as GPUBuffer;
  const pipeline = { getBindGroupLayout: () => ({}) };
  const device = {
    queue: { writeBuffer: () => undefined },
    createBuffer: () => buffer,
    createShaderModule: () => ({}),
    createComputePipeline: () => pipeline,
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  try {
    const mutator = new WebGpuSparseBrickTopologyMutator(device);
    assert.throws(() => mutator.encode({} as GPUCommandEncoder, {
      control: buffer, topology: buffer,
    } as SparseBrickOctreeGPU, { buffer, capacity: 1 }, {
      maximumDepth: 2, brickDimensions: [5, 4, 4], generation: 1,
    }), /exceeds the declared topology depth/);
    mutator.destroy();
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  }
});

const webgpuModulePath = process.env.WEBGPU_NODE_MODULE;
test("GPU mutation inserts an absent path and reactivation is allocation-free", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const tree = new SparseBrickOctreeGPU(device, { brickSize: 8, nodeCapacity: 64, leafCapacity: 8 });
  const mutator = new WebGpuSparseBrickTopologyMutator(device);
  const packed = packSparseBrickTopologyMutationWorklist([{ x: 3, y: 2, z: 1 }], 7, 4);
  const worklist = device.createBuffer({
    size: packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(worklist, 0, packed);

  const read = async (): Promise<{ control: Uint32Array; topology: Uint32Array; receipt: Uint32Array }> => {
    const controlRead = device.createBuffer({ size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const topologyRead = device.createBuffer({ size: tree.topology.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const receiptRead = device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(tree.control, 0, controlRead, 0, 128);
    encoder.copyBufferToBuffer(tree.topology, 0, topologyRead, 0, tree.topology.size);
    encoder.copyBufferToBuffer(worklist, 0, receiptRead, 0, packed.byteLength);
    device.queue.submit([encoder.finish()]);
    await Promise.all([controlRead.mapAsync(GPUMapMode.READ), topologyRead.mapAsync(GPUMapMode.READ), receiptRead.mapAsync(GPUMapMode.READ)]);
    const result = {
      control: new Uint32Array(controlRead.getMappedRange()).slice(),
      topology: new Uint32Array(topologyRead.getMappedRange()).slice(),
      receipt: new Uint32Array(receiptRead.getMappedRange()).slice(),
    };
    controlRead.unmap(); topologyRead.unmap(); receiptRead.unmap();
    controlRead.destroy(); topologyRead.destroy(); receiptRead.destroy();
    return result;
  };

  try {
    let encoder = device.createCommandEncoder();
    mutator.encode(encoder, tree, { buffer: worklist, capacity: 4 }, {
      maximumDepth: 2, brickDimensions: [4, 4, 4], generation: 7,
    });
    device.queue.submit([encoder.finish()]);
    let view = await read();
    assert.equal(view.control[0], 3);
    assert.equal(view.control[1], 1);
    assert.equal(view.control[19], 3);
    assert.equal(view.control[23], 1);
    assert.equal(view.control[31], 7);
    assert.equal(view.receipt[3], 0);
    assert.equal(view.receipt[4], 1);
    assert.equal(view.receipt[5], 3);
    assert.equal(view.receipt[6], 1);
    const topologyBase = tree.topologyOffsetBytes / Uint32Array.BYTES_PER_ELEMENT;
    assert.deepEqual(decodeSvoBrickLifecycle(view.topology[topologyBase + 2 * 8 + 7]), {
      active: true, dirty: true, queued: true, relocating: false,
    });
    const leafBase = tree.leafOffsetBytes / 4;
    assert.deepEqual([...view.topology.slice(leafBase, leafBase + 4)], [2, 0, 29, 0]);

    encoder = device.createCommandEncoder();
    mutator.encode(encoder, tree, { buffer: worklist, capacity: 4 }, {
      maximumDepth: 2, brickDimensions: [4, 4, 4], generation: 8,
    });
    device.queue.submit([encoder.finish()]);
    view = await read();
    assert.equal(view.control[0], 3);
    assert.equal(view.control[1], 1);
    assert.equal(view.receipt[5], 0);
    assert.equal(view.receipt[6], 0);
    assert.equal(view.control[31], 8);
    assert.equal(view.topology[topologyBase + 2 * 8 + 7] & SVO_BRICK_LIFECYCLE.relocatingBit, 0);
  } finally {
    worklist.destroy(); mutator.destroy(); tree.destroy(); device.destroy();
  }
});

test("GPU mutation reports fixed-arena overflow without publishing a relocating link", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const tree = new SparseBrickOctreeGPU(device, { brickSize: 8, nodeCapacity: 2, leafCapacity: 1 });
  const mutator = new WebGpuSparseBrickTopologyMutator(device);
  const packed = packSparseBrickTopologyMutationWorklist([{ x: 3, y: 2, z: 1 }], 3);
  const worklist = device.createBuffer({
    size: packed.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  const controlRead = device.createBuffer({ size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const topologyRead = device.createBuffer({ size: tree.topology.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const receiptRead = device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(worklist, 0, packed);
  try {
    const encoder = device.createCommandEncoder();
    mutator.encode(encoder, tree, { buffer: worklist, capacity: 1 }, {
      maximumDepth: 2, brickDimensions: [4, 4, 4], generation: 3,
    });
    encoder.copyBufferToBuffer(tree.control, 0, controlRead, 0, 128);
    encoder.copyBufferToBuffer(tree.topology, 0, topologyRead, 0, tree.topology.size);
    encoder.copyBufferToBuffer(worklist, 0, receiptRead, 0, packed.byteLength);
    device.queue.submit([encoder.finish()]);
    await Promise.all([controlRead.mapAsync(GPUMapMode.READ), topologyRead.mapAsync(GPUMapMode.READ), receiptRead.mapAsync(GPUMapMode.READ)]);
    const control = new Uint32Array(controlRead.getMappedRange());
    const topology = new Uint32Array(topologyRead.getMappedRange());
    const receipt = new Uint32Array(receiptRead.getMappedRange());
    assert.equal(receipt[3] & SPARSE_BRICK_TOPOLOGY_MUTATION.overflowNodeCapacity,
      SPARSE_BRICK_TOPOLOGY_MUTATION.overflowNodeCapacity);
    assert.equal(receipt[7], 1);
    assert.equal(control[12] & SPARSE_BRICK_TOPOLOGY_MUTATION.overflowNodeCapacity,
      SPARSE_BRICK_TOPOLOGY_MUTATION.overflowNodeCapacity);
    assert.equal(control[1], 0, "an incomplete path never publishes a terminal leaf");
    for (let node = 0; node < control[0]; node += 1) {
      const topologyBase = tree.topologyOffsetBytes / Uint32Array.BYTES_PER_ELEMENT;
      assert.equal(topology[topologyBase + node * 8 + 7] & SVO_BRICK_LIFECYCLE.relocatingBit, 0);
    }
  } finally {
    controlRead.unmap(); topologyRead.unmap(); receiptRead.unmap();
    controlRead.destroy(); topologyRead.destroy(); receiptRead.destroy();
    worklist.destroy(); mutator.destroy(); tree.destroy(); device.destroy();
  }
});
