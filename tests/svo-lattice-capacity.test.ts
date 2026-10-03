import assert from "node:assert/strict";
import test from "node:test";
import { SVO_LATTICE_VISIBILITY_CONTRACT as C, svoLatticeVisibilitySizing } from "../lib/svo/features/lighting-visibility/webgpu-svo-cone-fanout";
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

test("an overflow readback grows the live store without marking presentation failed", async t => {
  Object.assign(globalThis, { GPUMapMode: { READ: 1 } });
  const initial = svoLatticeVisibilitySizing(400_000, 256 * 1024 ** 2);
  let rebuilt = 0;
  const staging = {
    mapAsync: async () => {}, getMappedRange: () => new Uint32Array([163134]).buffer, unmap() {},
  };
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    device: { limits: { maxStorageBufferBindingSize: 256 * 1024 ** 2, maxBufferSize: 1024 ** 3 } },
    conePrepassWidth: 800, conePrepassHeight: 500,
    latticeRecords: { size: initial.slots * C.recordBytes + C.recordHeaderBytes },
    latticeOverflowStaging: staging, latticeOverflowCopied: true, latticeOverflowFrame: 44,
    latticeOverflowReading: false, latticeMinimumSlots: 0,
    ensureLatticeTargets() { rebuilt++; },
  });
  renderer.pollLatticeOverflow({});
  await Promise.resolve();
  assert.equal(rebuilt, 1);
  assert.equal(renderer.latticeMinimumSlots, initial.slots * 2);
  assert.equal(renderer.latticeStoreFailure, undefined);
  assert.equal(renderer.latticeOverflowReading, false);

  // A resize can retire the store while its readback is in flight.
  renderer.latticeOverflowCopied = true;
  renderer.latticeOverflowFrame = 44;
  renderer.pollLatticeOverflow({});
  renderer.latticeOverflowStaging = undefined;
  await Promise.resolve();
  assert.equal(rebuilt, 1, "a retired store's overflow must not resize its replacement");

  t.mock.method(console, "error", () => {});
  renderer.latticeOverflowStaging = staging;
  renderer.latticeOverflowCopied = true;
  renderer.latticeOverflowFrame = 44;
  renderer.ensureLatticeTargets = () => {
    renderer.latticeOverflowStaging = undefined;
    throw new Error("allocation rejected");
  };
  renderer.pollLatticeOverflow({});
  await Promise.resolve();
  assert.match(renderer.latticeStoreFailure, /allocation failed: allocation rejected/);
});

test("overflow at the device limit remains an explicit capacity failure", async t => {
  Object.assign(globalThis, { GPUMapMode: { READ: 1 } });
  t.mock.method(console, "error", () => {});
  const bindingLimit = 128 * 1024 ** 2;
  const maximum = svoLatticeVisibilitySizing(400_000, bindingLimit, C.maximumSlots);
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    device: { limits: { maxStorageBufferBindingSize: bindingLimit, maxBufferSize: 1024 ** 3 } },
    conePrepassWidth: 800, conePrepassHeight: 500,
    latticeRecords: { size: maximum.slots * C.recordBytes + C.recordHeaderBytes },
    latticeOverflowStaging: {
      mapAsync: async () => {}, getMappedRange: () => new Uint32Array([7]).buffer, unmap() {},
    },
    latticeOverflowCopied: true, latticeOverflowFrame: 44, latticeOverflowReading: false,
    ensureLatticeTargets() { assert.fail("must not allocate beyond the binding limit"); },
  });
  renderer.pollLatticeOverflow({});
  await Promise.resolve();
  assert.match(renderer.latticeStoreFailure, /Lattice visibility store overflowed: 7/);
});
