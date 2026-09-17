import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createSymmetricExpansionScene } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { gridOverlayShader } from "../lib/core/webgpu-grid-overlay";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
(dawnModule ? test : test.skip)("structure overlay resolves dilute symmetric corners at four paper steps",
  { timeout: 120_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "symmetric-support-overlay");
    let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      assert.ok(device);
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const scene = createSymmetricExpansionScene();
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", { selectorMode: "coarse-first", timeStep: "paper" });
      solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values,
        undefined, () => {}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      for (let step = 1; step <= 4; step++) {
        while (!solver.advanceTo(step / 30, [])) await new Promise(setImmediate);
        await solver.awaitFrameCompletion?.();
      }
      const [activity, fields, stats] = await Promise.all([
        solver.readGPUActivityPolicy(), solver.readDiagnosticFields(true), solver.readStats(),
      ]);
      assert.ok(Math.abs(stats.simulatedTime_s! - 4 / 30) < 1e-9);
      assert.equal(activity.faultFlags, 0);
      const corner = (q: readonly number[]) => (q[0] === 0 || q[0] === 3)
        && q[1] === 0 && (q[2] === 0 || q[2] === 3);
      const corners = activity.bricks.filter(b => corner(b.coordinate));
      assert.equal(corners.length, 4);
      assert.ok(corners.every(b => b.active));
      assert.ok(activity.bricks.some(b => b.active && b.acceptedResolution < 8));
      let cornerMass = 0, maximumCornerDensity = 0;
      for (let z = 0; z < 32; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 32; x++) {
        if (!corner([Math.floor(x / 8), 0, Math.floor(z / 8)])) continue;
        const rho = fields.density[x + 32 * (y + 16 * z)]!;
        cornerMass += rho; maximumCornerDensity = Math.max(maximumCornerDensity, rho);
      }
      assert.ok(cornerMass > 1, "diagonal support must already accept transported mass");
      assert.ok(maximumCornerDensity < 0.5, "exercise the dilute front hidden by occupied-only display");
      const mass = fields.density.reduce((sum, rho) => sum + rho, 0);
      assert.ok(Math.abs(mass / 2048 - 1) < 0.001);

      // Execute the renderer's actual owner lookup against live producer buffers,
      // including the dynamically allocated corner leaves absent at reset.
      const source = solver.sparseAdaptiveGridSource!;
      const module = device.createShaderModule({ code: `${gridOverlayShader}
@group(1) @binding(0) var<storage,read_write> result:array<vec2u>;
@compute @workgroup_size(4)
fn probeCorner(@builtin(local_invocation_index)i:u32){
  let q=vec3i(select(0,31,(i&1u)!=0u),0,select(0,31,(i&2u)!=0u));
  let owner=sparseOwner(q);
  result[i]=vec2u(owner.y,select(0u,1u,sparseBrickOccupied(owner.y)));
}` });
      const compilation = await module.getCompilationInfo();
      assert.deepEqual(compilation.messages.filter(m => m.type === "error"), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probeCorner" } });
      const overlay = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(overlay, 0, new Uint32Array([
        source.worldDirectoryBaseWords!, 1, source.worldDirectoryInitialLeaves!, source.activityRecordWords,
      ]));
      const uniform = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const uniforms = new Float32Array(36); uniforms.set([32, 16, 32, 0], 20);
      device.queue.writeBuffer(uniform, 0, uniforms);
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 10, resource: source.params }, { binding: 11, resource: source.topology },
        { binding: 13, resource: source.activity }, { binding: 17, resource: source.topologyArena! },
        { binding: 18, resource: { buffer: overlay } },
      ] });
      const output = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      const resultGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.setBindGroup(1, resultGroup); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, 32); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Uint32Array(readback.getMappedRange());
      for (let i = 0; i < 4; i++) {
        const q = [i & 1 ? 3 : 0, 0, i & 2 ? 3 : 0];
        assert.equal(result[2 * i], corners.find(b => b.coordinate.join() === q.join())!.leafId);
        assert.equal(result[2 * i + 1], 0, "these active receivers need the support outline");
      }
      readback.unmap();
      assert.deepEqual(errors, []);
      console.log({ cornerMass, maximumCornerDensity, activeCells: stats.activeSampleCount });
    } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
  });
