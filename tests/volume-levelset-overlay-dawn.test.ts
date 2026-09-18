import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { gridOverlayLevelSetVolumeUniform } from "../lib/core/grid-overlay-levelset-volume.wgsl";
import { GridOverlayPipeline, gridOverlayShader } from "../lib/core/webgpu-grid-overlay";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("slice overlay samples committed volume and the accepted vertex level set", { timeout: 90_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "volume-levelset-overlay");
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal", "enable-dawn-features=disable_blob_cache"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const renderUniform = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM });
    const bodies = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
    const overlayPipeline = new GridOverlayPipeline(device, "rgba8unorm", renderUniform, bodies);
    await overlayPipeline.initialize();
    overlayPipeline.destroy(); renderUniform.destroy(); bodies.destroy();
    solver = await adaptiveMassMethod.createSolverAsync!(device, getScenePreset("water-box-dam-break").create(), "balanced",
      resolveMethodValues(adaptiveMassMethod, "balanced", { timeStep: "paper" }), undefined, () => {}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    for (const frame of [0, 30]) {
      for (let step = 1; step <= frame; step++) {
        while (!solver.advanceTo(step / 30, [])) await new Promise(setImmediate);
        await solver.awaitFrameCompletion?.();
      }
      await solver.awaitFrameCompletion?.();
      const [levelSet, fields] = await Promise.all([solver.readAdaptiveLevelSetQA(true), solver.readDiagnosticFields(true)]);
      const supported = (levelSet.vertices ?? []).filter(v => v.support === 3
        && v.positionFine.every((q, axis) => q > 0 && q < [24, 16, 16][axis]!));
      const stride = Math.max(1, Math.floor(supported.length / 128));
      const vertices = supported.filter((_, i) => i % stride === 0).slice(0, 128);
      assert.ok(vertices.length >= 16);
      const source = solver.sparseAdaptiveGridSource!;
      const queries = new Float32Array(vertices.length * 4);
      vertices.forEach((v, i) => queries.set([...v.positionFine, 0], 4 * i));
      const module: GPUShaderModule = device.createShaderModule({ code: `${gridOverlayShader}
@group(1) @binding(0) var<storage,read> queries:array<vec4f>;
@group(1) @binding(1) var<storage,read_write> results:array<vec4f>;
@compute @workgroup_size(64) fn probeSlice(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&queries)){return;}
 let p=queries[i].xyz;results[i]=vec4f(sliceLevelSetPhi(p),sliceVolumeFill(vec3i(floor(p))));
}` });
      const compilation: GPUCompilationInfo = await module.getCompilationInfo();
      assert.deepEqual(compilation.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probeSlice" } });
      // Filled with the same producer-owned addresses as the visible overlay.
      const overlay = device.createBuffer({ size: 256, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(overlay, 0, new Uint32Array([source.worldDirectoryBaseWords!, 1, source.worldDirectoryInitialLeaves!, source.activityRecordWords]));
      const lsv = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(lsv, 0, gridOverlayLevelSetVolumeUniform(source.levelSetVolume));
      const uniform = device.createBuffer({ size: 144, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const uniforms = new Float32Array(36); uniforms.set([24, 16, 16, 0], 20);
      device.queue.writeBuffer(uniform, 0, uniforms);
      const dummy = device.createTexture({size:[1,1,1],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING});
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 10, resource: source.params }, { binding: 11, resource: source.topology },
        { binding: 12, resource: source.state }, { binding: 13, resource: source.activity },
        { binding: 17, resource: source.topologyArena }, { binding: 18, resource: { buffer: overlay } },
        { binding: 20, resource: { buffer: lsv } },
        ...[9,21,22].map(binding=>({binding,resource:dummy.createView()})),
      ] });
      const input = device.createBuffer({ size: queries.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(input, 0, queries);
      const output = device.createBuffer({ size: queries.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: queries.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const resultGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: [
        { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output } },
      ] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.setBindGroup(1, resultGroup); pass.dispatchWorkgroups(Math.ceil(vertices.length / 64)); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, queries.byteLength); device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange());
      vertices.forEach((v, i) => {
        assert.equal(result[4 * i + 1], 1, `frame ${frame}, vertex ${i} phi missing`);
        assert.ok(Math.abs(result[4 * i]! - v.phiFine) < 1e-5, `frame ${frame}, vertex ${i}: phi ${result[4*i]} vs ${v.phiFine}`);
        const [x,y,z] = v.positionFine, rho = fields.density[x + 24 * (y + 16 * z)]!;
        assert.equal(result[4 * i + 3], 1, `frame ${frame}, vertex ${i} capacity missing`);
        assert.ok(Math.abs(result[4 * i + 2]! - rho) < 1e-5, `frame ${frame}, vertex ${i}: fill ${result[4*i+2]} vs ${rho}`);
      });
      dummy.destroy(); readback.unmap(); for (const buffer of [overlay, lsv, uniform, input, output, readback]) buffer.destroy();
    }
    assert.deepEqual(errors, []);
  } finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
