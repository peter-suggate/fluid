import assert from "node:assert/strict";
import test from "node:test";
import { SVO_LATTICE_VISIBILITY_CONTRACT as C, svoLatticeVisibilitySizing } from "../lib/svo/features/lighting-visibility/webgpu-svo-cone-fanout";
import { StagedReadback } from "../lib/core/staged-readback";
import { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

test("fragmented voxel faces can grow beyond the reduced-prepass estimate", () => {
  const initial = svoLatticeVisibilitySizing(400_000, 256 * 1024 ** 2);
  const grown = svoLatticeVisibilitySizing(400_000, 256 * 1024 ** 2, initial.slots * 2);
  assert.equal(grown.slots, initial.slots * 2);
  assert.equal(grown.bytes, grown.slots * C.slotBytes + 4 * C.headerWords + C.recordHeaderBytes);
});

test("growth respects the record binding and packed miss-list slot limits", () => {
  const bindingLimit = 128 * 1024 ** 2;
  const capped = svoLatticeVisibilitySizing(400_000, bindingLimit, C.maximumSlots * 2);
  assert.ok(capped.slots * C.recordBytes + C.recordHeaderBytes <= bindingLimit);
  assert.ok((capped.slots + C.bucketWays) * C.recordBytes + C.recordHeaderBytes > bindingLimit);
  assert.equal(svoLatticeVisibilitySizing(400_000, 1024 ** 3, C.maximumSlots * 2).slots, C.maximumSlots);
});

/** A lattice overflow readback copied this poll period, answering `overflow` when read. */
function copiedOverflow(overflow: number): StagedReadback {
  Object.assign(globalThis, { GPUBufferUsage: { COPY_DST: 8, MAP_READ: 1 }, GPUMapMode: { READ: 1 } });
  const device = { createBuffer: () => ({
    mapAsync: async () => {}, getMappedRange: () => new Uint32Array([overflow]).buffer, unmap() {}, destroy() {},
  }) } as unknown as GPUDevice;
  const readback = new StagedReadback(device, { label: "lattice overflow", size: 4 });
  readback.encode({} as GPUCommandEncoder, () => {});
  return readback;
}

/** The frame before a read phase: the overflow is polled every 30 frames and read at half-period. */
const READ_FRAME = 44;
const settle = () => new Promise(resolve => setTimeout(resolve, 0));

test("an overflow readback grows the live store without marking presentation failed", async t => {
  const initial = svoLatticeVisibilitySizing(400_000, 256 * 1024 ** 2);
  let rebuilt = 0;
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    device: { limits: { maxStorageBufferBindingSize: 256 * 1024 ** 2, maxBufferSize: 1024 ** 3 } },
    conePrepassWidth: 800, conePrepassHeight: 500,
    latticeRecords: { size: initial.slots * C.recordBytes + C.recordHeaderBytes },
    latticeOverflowReadback: copiedOverflow(163134), latticeOverflowFrame: READ_FRAME, latticeMinimumSlots: 0,
    ensureLatticeTargets() { rebuilt++; },
  });
  renderer.pollLatticeOverflow({});
  await settle();
  assert.equal(rebuilt, 1);
  assert.equal(renderer.latticeMinimumSlots, initial.slots * 2);
  assert.equal(renderer.latticeStoreFailure, undefined);
  assert.equal(renderer.latticeOverflowReadback.idle, true);

  // A resize can retire the store while its readback is in flight.
  renderer.latticeOverflowReadback = copiedOverflow(163134);
  renderer.latticeOverflowFrame = READ_FRAME;
  renderer.pollLatticeOverflow({});
  renderer.latticeOverflowReadback.destroy();
  await settle();
  assert.equal(rebuilt, 1, "a retired store's overflow must not resize its replacement");

  t.mock.method(console, "error", () => {});
  renderer.latticeOverflowReadback = copiedOverflow(163134);
  renderer.latticeOverflowFrame = READ_FRAME;
  renderer.ensureLatticeTargets = () => {
    renderer.latticeOverflowReadback.destroy();
    throw new Error("allocation rejected");
  };
  renderer.pollLatticeOverflow({});
  await settle();
  assert.match(renderer.latticeStoreFailure, /allocation failed: allocation rejected/);
});

test("overflow at the device limit remains an explicit capacity failure", async t => {
  t.mock.method(console, "error", () => {});
  const bindingLimit = 128 * 1024 ** 2;
  const maximum = svoLatticeVisibilitySizing(400_000, bindingLimit, C.maximumSlots);
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    device: { limits: { maxStorageBufferBindingSize: bindingLimit, maxBufferSize: 1024 ** 3 } },
    conePrepassWidth: 800, conePrepassHeight: 500,
    latticeRecords: { size: maximum.slots * C.recordBytes + C.recordHeaderBytes },
    latticeOverflowReadback: copiedOverflow(7), latticeOverflowFrame: READ_FRAME,
    ensureLatticeTargets() { assert.fail("must not allocate beyond the binding limit"); },
  });
  renderer.pollLatticeOverflow({});
  await settle();
  assert.match(renderer.latticeStoreFailure, /Lattice visibility store overflowed: 7/);
});
