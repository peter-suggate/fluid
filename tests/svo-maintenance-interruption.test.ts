import assert from "node:assert/strict";
import test from "node:test";
import { SparseSceneProxyVoxelizer, type SparseSceneAxisAlignedBounds } from "../lib/core/webgpu-sparse-scene-proxies";

function fixture(capacity = 2, halo = false) {
  const params = {}, arena = {}, writes = new Map<object, ArrayBuffer>();
  const primitives: never[] = [];
  const voxelizer = Object.assign(Object.create(SparseSceneProxyVoxelizer.prototype), {
    options: { primitiveCapacity: 1, dirtyRegionCapacity: capacity, dirtyBrickCapacity: 64,
      candidatesPerDirtyBrick: 1, finestLevel: 2, cellSize: [1, 1, 1], surfaceDualMarchingCubes: halo },
    tree: { brickSize: 8, leafCapacity: 64, sceneGeometryOffsetBytes: 0, sceneMaterialOwnerOffsetBytes: 0,
      topologyOffsetBytes: 0, bandedLaneWordOffsets: [0, 0, 0, 0, 0], payloadLayout: { bandedBlobWords: 0 } },
    device: { queue: { writeBuffer(buffer: object, _offset: number, data: ArrayBuffer | ArrayBufferView) {
      writes.set(buffer, data instanceof ArrayBuffer ? data.slice(0) :
        data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength) as ArrayBuffer);
    } } },
    paramsBuffer: params, maintenanceArena: arena, publishedPrimitives: primitives, publishedBounds: [],
    unfinishedDirtyRegions: [], chunksRemaining: 0, sceneRevision: 0,
    brickBudget: 1, candidateScratchBricks: 64, dualOffsetWords: 0, encodedFrames: 0,
    dirtyRegionOffsetWords: 0, clusterCapacity: 0, fieldProgramCapacity: 0,
  });
  const encoder = { clearBuffer() {}, beginComputePass() {
    return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
  } } as unknown as GPUCommandEncoder;
  const publish = (revision: number, dirtyRegions: SparseSceneAxisAlignedBounds[], budgeted = true) => {
    voxelizer.publish({ revision, primitives, dirtyRegions, budgeted });
    const count = new Uint32Array(writes.get(params)!)[9]!;
    const data = new Float32Array(writes.get(arena)!);
    return Array.from({ length: count }, (_, i) => ({ minimum: Array.from(data.slice(i * 8, i * 8 + 3)),
      maximum: Array.from(data.slice(i * 8 + 4, i * 8 + 7)) }));
  };
  return { voxelizer, encoder, publish };
}

const region = (low: number, high: number): SparseSceneAxisAlignedBounds =>
  ({ minimum: [low, 0, 0], maximum: [high, 8, 8] });

test("interrupted budgeted revisions retain old dirty bounds and retire them only after repair", () => {
  const { voxelizer, encoder, publish } = fixture();
  publish(1, [region(0, 16)]);
  voxelizer.encodeMaintenance(encoder);
  assert.ok(voxelizer.chunksRemaining > 0);
  const oldChunks = voxelizer.plannedChunks;
  assert.deepEqual(publish(2, [region(24, 32)]), [region(0, 16), region(24, 32)]);
  assert.ok(voxelizer.plannedChunks > oldChunks, "the new revision budgets both old and new bricks");
  voxelizer.encodeMaintenance(encoder);
  assert.deepEqual(publish(3, [region(8, 16)]), [region(0, 32)], "overflow coalesces without losing any owed brick");
  while (voxelizer.encodeMaintenance(encoder)) { /* finish all queued chunks */ }
  assert.deepEqual(publish(4, [region(24, 32)]), [region(24, 32)], "completed revisions do not accumulate history");
});

test("repeated interruption expands the DMC halo once and unbudgeted draining releases carried work", () => {
  const { voxelizer, encoder, publish } = fixture(1, true);
  const expanded = [{ minimum: [-1, -1, -1], maximum: [17, 9, 9] }];
  assert.deepEqual(publish(1, [region(0, 16)]), expanded);
  voxelizer.encodeMaintenance(encoder);
  assert.deepEqual(publish(2, [region(0, 16)]), expanded);
  voxelizer.encodeMaintenance(encoder);
  assert.deepEqual(publish(3, [region(0, 16)], false), expanded);
  voxelizer.encodeMaintenance(encoder);
  assert.equal(voxelizer.chunksRemaining, 0);
  assert.deepEqual(publish(4, [region(24, 32)]), [{ minimum: [23, -1, -1], maximum: [33, 9, 9] }]);
});

test("interrupted material-only bounds retain their purpose until mixed-region coalescing requires geometry", () => {
  const { voxelizer, encoder, publish } = fixture();
  const identityRegion = { ...region(0, 16), identityOnly: true };
  publish(1, [identityRegion]);
  voxelizer.encodeMaintenance(encoder);
  publish(2, [region(24, 32)]);
  assert.equal(voxelizer.unfinishedDirtyRegions[0].identityOnly, true,
    "a carried import palette transition need not repeat its geometry work");
  voxelizer.encodeMaintenance(encoder);
  publish(3, [region(8, 16)]);
  assert.equal(voxelizer.unfinishedDirtyRegions.length, 1);
  assert.equal(voxelizer.unfinishedDirtyRegions[0].identityOnly, undefined,
    "a coalesced region containing geometric edits must perform a full rebuild");
});

test("refused superseding publication leaves unfinished work and its revision intact", () => {
  const { voxelizer, encoder, publish } = fixture(1);
  publish(1, [region(0, 16)]);
  voxelizer.encodeMaintenance(encoder);
  const remaining = voxelizer.chunksRemaining;
  assert.throws(() => publish(2, [region(32, 24)]), /bounds must be ordered/);
  assert.equal(voxelizer.sceneRevision, 1);
  assert.equal(voxelizer.chunksRemaining, remaining);
  assert.deepEqual(voxelizer.unfinishedDirtyRegions, [region(0, 16)]);
  assert.deepEqual(publish(2, [region(24, 32)]), [region(0, 32)]);
});
