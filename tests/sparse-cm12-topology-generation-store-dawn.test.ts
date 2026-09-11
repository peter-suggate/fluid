import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from
  "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createSparseAdaptiveMassAtlas, type SparseBrickResolution } from
  "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-volume/sparse-atlas-composite-projection";
import { prepareSparseCM12TopologyWorkingSet } from
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";
import { SparseCM12TopologyGenerationStore } from
  "../lib/methods/adaptive-volume/sparse-cm12-topology-generation-store";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function packet(generation: number, accepted: SparseBrickResolution, candidate: SparseBrickResolution) {
  const atlas = createSparseAdaptiveMassAtlas([13, 8, 8], [0, 1].map((x) => ({
    key: x, coordinate: [x, 0, 0] as const, resolution: accepted,
    density: new Float64Array(accepted ** 3).fill(0.5),
    gamma: new Float64Array(accepted ** 3).fill(1),
  })), generation, 8);
  const result = prepareSparseCM12TopologyWorkingSet(buildSparseAtlasCompositeGrid(atlas),
    new Map(atlas.bricks.map((b) => [b.key, candidate])), {
      maximumCells: 10_000, maximumRows: 50_000, maximumBytes: 16 * 1024 ** 2,
    });
  if (result.status !== "ready") assert.fail(JSON.stringify(result));
  return result;
}
const imageBytes = (p: ReturnType<typeof packet>, candidate: boolean) => p.words.byteLength
  + 4 * (4 + (candidate ? p.candidateCellWorklist.length + p.candidateRowWorklist.length
    : p.acceptedCellWorklist.length + p.acceptedRowWorklist.length));

dawnTest("bounded GPU topology generations defer, cancel and reclaim only after leases finish",
  { timeout: 30_000 }, async (t) => {
    await acquireWebGPUExclusiveLock("dawn-test", "tests/sparse-cm12-topology-generation-store-dawn.test.ts");
    let device: GPUDevice | undefined;
    let store: SparseCM12TopologyGenerationStore | undefined;
    const leases: Array<{ releaseAfterSubmission(): Promise<void> }> = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as NodeDawnProvider;
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn,
        [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", (event) => errors.push(event.error.message));
      const p0 = packet(11, 2, 2), p1 = packet(11, 2, 4), p2 = packet(12, 4, 2);
      const initialBytes = imageBytes(p0, false), firstBytes = imageBytes(p1, true);
      const secondBytes = imageBytes(p2, true);
      const maximumBytes = firstBytes + Math.max(initialBytes, secondBytes);
      store = await SparseCM12TopologyGenerationStore.create(device, p0, maximumBytes);
      const old = store.acquire(); leases.push(old);
      const prepared = await store.prepare(p1);
      assert.equal(prepared.status, "ready");
      if (prepared.status !== "ready") return;
      await assert.rejects(store.prepare(p1), /pending/);
      const validate = store.acquire(prepared.candidate); leases.push(validate);
      store.cancel(prepared.candidate);
      assert.equal(store.receipt.generation, 11);
      assert.equal(store.receipt.retiredBytes, firstBytes,
        "cancelled storage remains resident while validation commands can still use it");
      const cancelledBudget = await store.prepare(p1);
      assert.equal(cancelledBudget.status, "deferred");
      await validate.releaseAfterSubmission();
      assert.equal(store.receipt.reservedBytes, initialBytes);
      assert.throws(() => store!.commit(prepared.candidate), /not owned/);
      const retried = await store.prepare(p1);
      assert.equal(retried.status, "ready");
      if (retried.status !== "ready") return;

      // Encode an old-generation consumer before publication, but submit it
      // afterwards. The explicit lease must keep its buffers alive throughout.
      const readback = device.createBuffer({ size: old.buffers.membership.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(old.buffers.membership, 0, readback, 0, readback.size);
      store.commit(retried.candidate);
      assert.equal(store.receipt.generation, 12);
      assert.equal(store.receipt.retiredBytes, initialBytes);
      const deferred = await store.prepare(p2);
      assert.equal(deferred.status, "deferred", "leased retired bytes must count against capacity");
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const oldWords = new Uint32Array(readback.getMappedRange());
      assert.equal(oldWords[0], 11);
      assert.deepEqual(Array.from(oldWords.slice(4, oldWords[3])), Array.from(p0.acceptedCellWorklist));
      readback.unmap(); readback.destroy();
      await old.releaseAfterSubmission();
      await old.releaseAfterSubmission(); // release is idempotent
      assert.equal(store.receipt.retiredBytes, 0);
      assert.equal(store.receipt.reservedBytes, firstBytes);
      await assert.rejects(store.prepare(p1), /stale/);
      const next = await store.prepare(p2);
      assert.equal(next.status, "ready", "reclamation must allow the deferred generation to progress");
      if (next.status !== "ready") return;
      store.commit(next.candidate);
      assert.equal(store.receipt.generation, 13);
      assert.equal(store.receipt.reservedBytes, secondBytes);

      const current = store.acquire(); leases.push(current);
      const topologyReadback = device.createBuffer({ size: current.buffers.topology.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const copy = device.createCommandEncoder();
      copy.copyBufferToBuffer(current.buffers.topology, 0, topologyReadback, 0, topologyReadback.size);
      store.destroy();
      assert.equal(store.receipt.reservedBytes, secondBytes,
        "destroy cannot invalidate an encoded command protected by a lease");
      device.queue.submit([copy.finish()]);
      await topologyReadback.mapAsync(GPUMapMode.READ);
      assert.deepEqual(new Uint32Array(topologyReadback.getMappedRange()), p2.words);
      topologyReadback.unmap(); topologyReadback.destroy();
      await current.releaseAfterSubmission();
      assert.equal(store.receipt.reservedBytes, 0);
      assert.equal(store.receipt.peakReservedBytes, maximumBytes);
      assert.equal(store.receipt.deferredRequests, 2);
      await assert.rejects(store.prepare(p2), /destroyed/);

      // A synchronous failure after the first allocation must roll back that
      // buffer and its reservation without changing the accepted generation.
      let allocations = 0, failAt = Infinity;
      const allocationDevice = {
        queue: device.queue,
        limits: device.limits,
        pushErrorScope: device.pushErrorScope.bind(device),
        popErrorScope: device.popErrorScope.bind(device),
        createBuffer: (descriptor: GPUBufferDescriptor) => {
          if (++allocations === failAt) throw new Error("injected second-buffer failure");
          return device!.createBuffer(descriptor);
        },
      } as unknown as GPUDevice;
      const rollback = await SparseCM12TopologyGenerationStore.create(allocationDevice, p0, maximumBytes);
      try {
        failAt = allocations + 2;
        await assert.rejects(rollback.prepare(p1), /second-buffer failure/);
        assert.equal(rollback.receipt.generation, 11);
        assert.equal(rollback.receipt.reservedBytes, initialBytes);
        assert.equal(rollback.receipt.allocationFailures, 1);
        failAt = Infinity;
        const recovery = await rollback.prepare(p1);
        assert.equal(recovery.status, "ready");
        if (recovery.status === "ready") rollback.cancel(recovery.candidate);
        assert.equal(rollback.receipt.reservedBytes, initialBytes);
        const invalid = { ...p1, candidateCellWorklist: p1.candidateCellWorklist.slice(1) };
        await assert.rejects(rollback.prepare(invalid), /unselected cell/);
        assert.equal(rollback.receipt.reservedBytes, initialBytes);
        const inFlight = rollback.prepare(p1);
        assert.equal(rollback.receipt.preparingBytes, firstBytes);
        rollback.destroy();
        await assert.rejects(inFlight, /destroyed/);
        assert.equal(rollback.receipt.reservedBytes, 0);
      } finally { rollback.destroy(); }
      assert.deepEqual(errors, []);
      t.diagnostic(JSON.stringify({ maximumBytes, initialBytes, firstBytes, secondBytes,
        finalReservedBytes: store.receipt.reservedBytes }));
    } finally {
      for (const lease of leases) await lease.releaseAfterSubmission();
      store?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
    }
  });
