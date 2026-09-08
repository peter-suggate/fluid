import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { SPARSE_BRICK_GPU_LAYOUT, SPARSE_BRICK_INVALID_INDEX as INVALID, type SparseBrickOctreeGPU } from "../lib/svo/features/construction/sparse-brick-octree";
import { SVO_BRICK_LIFECYCLE } from "../lib/svo/features/construction/svo-brick-occupancy";
import { traversePackedSvo } from "../lib/svo/features/primary-visibility/webgpu-svo-traversal";
import { WebGpuSparseBrickTopologyMutator, packSparseBrickTopologyMutationWorklist,
  SPARSE_BRICK_TOPOLOGY_MUTATION as RECEIPT } from "../lib/core/webgpu-sparse-brick-topology-mutation";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const topologyBase = SPARSE_BRICK_GPU_LAYOUT.topologyOffsetBytes / 4;
const nodeCapacity = 128, leafCapacity = 64, leafOffset = nodeCapacity * 8;

function fixture(overrides: { nodes?: number; leaves?: number; voxels?: number; kind?: number } = {}) {
  const words = new Uint32Array(topologyBase + leafOffset + leafCapacity * 4);
  words.set([1, 1, 64, 10]);
  words.set([overrides.nodes ?? nodeCapacity, overrides.leaves ?? leafCapacity, overrides.voxels ?? leafCapacity * 64, 4], 8);
  words[16] = leafOffset; words[19] = 1; words[23] = 1; words[28] = 1; words[31] = 10;
  words.set([0, 0, 0, 0, INVALID, 0, 0, SVO_BRICK_LIFECYCLE.activeBit], topologyBase);
  words.set([0, 0, overrides.kind ?? 1, overrides.kind === 0 ? INVALID : 37], topologyBase + leafOffset);
  return words;
}

test("Dawn terminal splits preserve planar siblings, expose edited descendants and reject atomically", {
  skip: !dawnModule && "set WEBGPU_NODE_MODULE to run native topology mutation acceptance",
  timeout: 30_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "tests/sparse-brick-terminal-split-dawn.test.ts");
  let device: GPUDevice | undefined;
  let mutator: WebGpuSparseBrickTopologyMutator | undefined;
  try {
    const dawn = await import(pathToFileURL(dawnModule!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    mutator = new WebGpuSparseBrickTopologyMutator(device);
    await mutator.initializePipelines();

    async function mutate(initial: Uint32Array, coordinates = [{ x: 0, y: 0, z: 0 }], generation = 11, maximumRequests = coordinates.length) {
      const work = packSparseBrickTopologyMutationWorklist(coordinates, generation);
      const structure = device!.createBuffer({ size: initial.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const requests = device!.createBuffer({ size: work.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const readback = device!.createBuffer({ size: initial.byteLength + work.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        device!.queue.writeBuffer(structure, 0, initial as Uint32Array<ArrayBuffer>);
        device!.queue.writeBuffer(requests, 0, work);
        const encoder = device!.createCommandEncoder();
        mutator!.encode(encoder, { structure } as SparseBrickOctreeGPU, { buffer: requests, capacity: coordinates.length }, {
          maximumDepth: 2, brickDimensions: [4, 4, 4], generation, maximumRequests,
        });
        encoder.copyBufferToBuffer(structure, 0, readback, 0, initial.byteLength);
        encoder.copyBufferToBuffer(requests, 0, readback, initial.byteLength, work.byteLength);
        device!.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        return { words: result.slice(0, initial.length), receipt: result.slice(initial.length) };
      } finally { structure.destroy(); requests.destroy(); readback.destroy(); }
    }

    const first = await mutate(fixture());
    assert.equal(first.receipt[3], 0);
    assert.equal(first.words[0], 17); assert.equal(first.words[1], 15);
    assert.equal(first.words[23], 15, "each two-level split adds fourteen leaves and reuses the parent slot");
    assert.equal(first.words[topologyBase + 6], INVALID, "root must no longer hide descendants");
    const nodes = first.words.slice(topologyBase, topologyBase + first.words[0]! * 8);
    const leaves = first.words.slice(topologyBase + leafOffset, topologyBase + leafOffset + first.words[1]! * 4);
    for (let leaf = 0; leaf < first.words[1]!; leaf++) {
      const node = leaves[leaf * 4]!;
      assert.equal(nodes[node * 8 + 6], leaf, `leaf ${leaf} backlink`);
      assert.equal(leaves[leaf * 4 + 1], leaf * 64);
      assert.equal(leaves[leaf * 4 + 2], leaf === 0 ? 0 : 1);
      assert.equal(leaves[leaf * 4 + 3], leaf === 0 ? INVALID : 37);
    }
    const target = traversePackedSvo({ origin: [-1, .5, .5], direction: [1, 0, 0] }, { nodes, leaves }, {
      origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 2,
    });
    assert.equal(target.status, "hit");
    if (target.status === "hit") { assert.equal(target.hit.level, 2); assert.equal(target.hit.terminalKind, 0); }
    const sibling = traversePackedSvo({ origin: [-1, 12, 12], direction: [1, 0, 0] }, { nodes, leaves }, {
      origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 2,
    });
    assert.equal(sibling.status, "hit");
    if (sibling.status === "hit") { assert.equal(sibling.hit.level, 1); assert.equal(sibling.hit.terminalKind, 1); assert.equal(sibling.hit.terminalIndex, 37); }
    assert.equal(first.words[28], 15); assert.equal(first.words[29], 15); assert.equal(first.words[30], 15);

    const repeated = await mutate(first.words, undefined, 12);
    assert.equal(repeated.receipt[3], 0);
    assert.equal(repeated.words[19], first.words[19]); assert.equal(repeated.words[23], first.words[23]);
    assert.equal(repeated.receipt[6], 0, "repeated edit must not leak leaves");

    const finest = await mutate(first.words, [{ x: 1, y: 0, z: 0 }], 13);
    assert.equal(finest.receipt[3], 0);
    assert.equal(finest.words[23], 15, "an existing finest planar terminal needs no extra leaf");
    const finestKinds = Array.from({ length: 15 }, (_, i) => finest.words[topologyBase + leafOffset + i * 4 + 2]);
    assert.equal(finestKinds.filter(kind => kind === 0).length, 2);

    const shared = await mutate(fixture({ leaves: 22 }), [{ x: 3, y: 3, z: 3 }, { x: 0, y: 0, z: 0 }]);
    assert.equal(shared.receipt[3], 0);
    assert.equal(shared.words[19], 25); assert.equal(shared.words[23], 22, "shared ancestors split once");
    for (let leaf = 0; leaf < 22; leaf++) {
      const node = shared.words[topologyBase + leafOffset + leaf * 4]!;
      assert.equal(shared.words[topologyBase + node * 8 + 6], leaf);
    }

    for (const [initial, coordinates, flag] of [
      [fixture({ nodes: 16 }), [{ x: 0, y: 0, z: 0 }], RECEIPT.overflowNodeCapacity],
      [fixture({ leaves: 14 }), [{ x: 0, y: 0, z: 0 }], RECEIPT.overflowLeafCapacity],
      [fixture({ voxels: 14 * 64 }), [{ x: 0, y: 0, z: 0 }], RECEIPT.overflowVoxelCapacity],
      [fixture({ kind: 0 }), [{ x: 0, y: 0, z: 0 }], RECEIPT.overflowUnsupportedTerminal],
      [fixture(), [{ x: 0, y: 0, z: 0 }, { x: 4, y: 0, z: 0 }], RECEIPT.overflowMalformedRequest],
    ] as const) {
      const rejected = await mutate(initial, [...coordinates]);
      assert.ok(rejected.receipt[3]! & flag);
      assert.equal(rejected.words[3], 10); assert.equal(rejected.words[31], 10);
      assert.deepEqual(rejected.words, initial, "a rejected worklist preserves the entire accepted source, including overflow state");
      for (const word of [0, 1, 2, 19, 23, 28, 29, 30]) assert.equal(rejected.words[word], initial[word]);
    }
    const budgetBase = fixture();
    const budget = await mutate(budgetBase, [{ x: 0, y: 0, z: 0 }, { x: 3, y: 3, z: 3 }], 11, 1);
    assert.ok(budget.receipt[3]! & RECEIPT.overflowRequestBudget);
    assert.deepEqual(budget.words, budgetBase, "request-budget overflow cannot partially accept the first request");
    assert.deepEqual(errors, []);
  } finally {
    mutator?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  }
});
