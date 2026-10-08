import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { MacMultigrid } from "../lib/methods/mac-shared/multigrid";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("APIC pressure obeys the absolute ceiling, relative reduction and small-residual floor", { timeout: 30_000 }, async () => {
  const owned: GPUBuffer[] = []; let mg: MacMultigrid | undefined; let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const buffer = (size: number, uniform = false) => {
      const b = device!.createBuffer({ size, usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); owned.push(b); return b;
    };
    const fine = { matrix: buffer(16), rhs: buffer(4), cg: buffer(16), partial: buffer(32), scalars: buffer(144), params: buffer(80, true) };
    device!.queue.writeBuffer(fine.matrix, 0, new Float32Array([0, 0, 0, 1]));
    mg = await MacMultigrid.create(device, [1, 1, 1], fine);
    const read = async () => {
      const b = device!.createBuffer({ size: 144, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const encoder = device!.createCommandEncoder(); encoder.copyBufferToBuffer(fine.scalars, 0, b, 0, 144); device!.queue.submit([encoder.finish()]);
      await b.mapAsync(GPUMapMode.READ); const out = new Float32Array(b.getMappedRange().slice(0)); b.unmap(); b.destroy(); return out;
    };
    // Includes a warm start: the reference is b-Ap, not the RHS norm.
    for (const [rhs, warm, relative, expected, accepted] of [
      [2, 0, 0.1, 0.2, 0], [100, 0, 0.1, 5, 0], [10, 9, 0.1, 0.1, 0],
      [0.0002, 0, 0.1, 0.0001, 0], [0.00005, 0, 0.1, 0.0001, 1], [2, 0, 0, 5, 1],
    ]) {
      const params = new Float32Array(20); params[3] = relative; params[14] = 5;
      const scalars = new Float32Array(32); scalars[21] = 1;
      device!.queue.writeBuffer(fine.params, 0, params); device!.queue.writeBuffer(fine.scalars, 0, scalars);
      device!.queue.writeBuffer(fine.rhs, 0, new Float32Array([rhs])); device!.queue.writeBuffer(fine.cg, 0, new Float32Array([warm, 0, 0, 0]));
      const initial = device!.createCommandEncoder(); mg.encode(initial, 0); device!.queue.submit([initial.finish()]);
      const receipt = await read();
      assert.ok(Math.abs(receipt[33] - expected) < 1e-6, `target ${receipt[33]}, expected ${expected}`);
      assert.equal(receipt[20], accepted, `initial acceptance for rhs ${rhs}, warm ${warm}`);
      if (!accepted) {
        const solve = device!.createCommandEncoder(); mg.encode(solve, 8); device!.queue.submit([solve.finish()]);
        const result = await read();
        assert.equal(result[20], 1); assert.ok(result[10] > 0);
        assert.ok(result[4] <= expected); assert.ok(Math.abs(result[33] - expected) < 1e-6, "target stays anchored to the initial residual");
      }
    }
    assert.deepEqual(errors, []);
  } finally { mg?.destroy(); owned.forEach(b => b.destroy()); device?.destroy(); }
});
