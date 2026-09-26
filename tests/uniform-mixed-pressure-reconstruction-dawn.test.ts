import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedPressureReconstructionWGSL } from "../lib/methods/uniform/uniform-mixed-pressure-reconstruction.wgsl";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { mixedPressureFixture, faceGradient, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("mixed pressure reconstruction matches independent physical-face oracle on GPU", { timeout: 180000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform mixed pressure reconstruction");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const layouts = [seamLayout(0, "fine"), seamLayout(0, "coarse"), ...mixedPressureLayouts()];
    for (const layout of layouts) {
      const fixture = mixedPressureFixture(layout), ownership = new UniformMixedOwnership(device, layout);
      const size = layout.cellCount * 24 * 8;
      const pressure = device.createBuffer({ size: layout.cellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      // Test-only resources. Production must borrow the existing pressure arena.
      const slopes = device.createBuffer({ size: layout.cellCount * 16, usage: GPUBufferUsage.STORAGE });
      const output = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback: GPUBuffer = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      try {
        const module: GPUShaderModule = device.createShaderModule({ code: uniformMixedTopologyWGSL(layout, 0) + /* wgsl */ `
const UM_H=vec3f(${layout.lattice.cellSize_m.map(n => `${n.toFixed(8)}`).join(",")});
@group(1) @binding(0) var<storage,read> pressures:array<f32>;
@group(1) @binding(1) var<storage,read_write> slopes:array<vec4f>;
@group(1) @binding(2) var<storage,read_write> result:array<vec2u>;
fn umPressure(owner:UMOwner)->f32{return pressures[owner.index];}
fn umPressureSlope(owner:UMOwner)->vec3f{return slopes[owner.index].xyz;}
${uniformMixedPressureReconstructionWGSL}
@compute @workgroup_size(64) fn reconstruct(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u){return;}
 slopes[owner.index]=vec4f(umReconstructPressureSlope(owner),0.0);
}
@compute @workgroup_size(64) fn gradients(@builtin(global_invocation_id) gid:vec3u){
 let owner=umOwner(gid);if(owner.width==0u){return;}
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){for(var k=0u;k<4u;k++){
  let face=umFace(owner,axis,select(-1,1,side==1u),k);let at=owner.index*24u+axis*8u+side*4u+k;
  result[at]=vec2u(0u,0xfffffffeu);if(face.width==0u){continue;}
  result[at]=vec2u(bitcast<u32>(umReconstructedPressureGradient(owner,face)),select(face.neighbor.index,0xffffffffu,face.neighbor.width==0u));
 }}}
}
` });
        const compilation: GPUCompilationInfo = await module.getCompilationInfo();
        assert.deepEqual(compilation.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
        const resources = device.createBindGroupLayout({ entries: [0, 1, 2].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: binding === 0 ? "read-only-storage" as const : "storage" as const } })) });
        const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, resources] });
        const pipelines = await Promise.all(["reconstruct", "gradients"].map(entryPoint => Promise.all([1, 2, 4].map(umCellWidth =>
          device!.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint, constants: { umCellWidth, umDispatchX: ownership.dispatchX } } })))));
        const group = device.createBindGroup({ layout: resources, entries: [pressure, slopes, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        for (const mode of ["constant", "affine", "random"] as const) {
          const p = Float32Array.from(fixture.cells, (c, i) => mode === "constant" ? 3 : mode === "random" ? Math.sin(i * 7)
            : 3 + 2 * c.center[0] - 3 * c.center[1] + .25 * c.center[2]);
          device.queue.writeBuffer(pressure, 0, p);
          const expected = faceGradient(fixture, p);
          const faceIndices = new Map(fixture.faces.map((f, k) => [`${f.left}:${f.right}:${f.axis}`, k]));
          const encoder = device.createCommandEncoder();
          for (const stage of pipelines) {
            const pass = encoder.beginComputePass(); pass.setBindGroup(0, ownership.bindGroup); pass.setBindGroup(1, group);
            ownership.dispatch(pass, stage); pass.end();
          }
          encoder.copyBufferToBuffer(output, 0, readback, 0, size); device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
          const bytes: ArrayBuffer = readback.getMappedRange();
          const words: Uint32Array = new Uint32Array(bytes), values = new Float32Array(bytes);
          let count = 0;
          for (let i = 0; i < layout.cellCount; i++) for (let slot = 0; slot < 24; slot++) {
            const at = (i * 24 + slot) * 2, neighbor = words[at + 1]!;
            if (neighbor >= 0xfffffffe) continue;
            const axis = Math.floor(slot / 8), positive = slot % 8 >= 4;
            const key = positive ? `${i}:${neighbor}:${axis}` : `${neighbor}:${i}:${axis}`;
            const k = faceIndices.get(key); assert.notEqual(k, undefined, key);
            const actual = values[at]!;
            assert.ok(Math.abs(actual - expected[k!]!) < 3e-5, `${mode}, face ${key}: ${actual} != ${expected[k!]}`);
            if (mode === "affine") assert.ok(Math.abs(actual - [2, -3, .25][axis]!) < 3e-5);
            if (mode === "constant") assert.equal(Math.abs(actual), 0);
            count++;
          }
          assert.equal(count, 2 * fixture.faces.length); readback.unmap();
        }
      } finally { if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); output.destroy(); slopes.destroy(); pressure.destroy(); ownership.destroy(); }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
