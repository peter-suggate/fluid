import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createGeometricVolumeResidentWGSL, type SparseGeometricVolumeLayout } from "../lib/methods/adaptive-volume/resident-volume.wgsl";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("phi feedback is bounded, bidirectional, and preserves phase-only support", { timeout: 60_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "bounded volume-to-phi feedback");
  let device: GPUDevice | undefined, gpu: GPU | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals); gpu = dawn.create(["backend=metal"]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
    const production = createGeometricVolumeResidentWGSL({} as SparseGeometricVolumeLayout);
    const code = `
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<storage,read_write> conditioning:array<atomic<i32>>;
const GV_WHOLE_FRAME_CONTROL=0u; const LSV_SUPPORT_METRIC=3u;
fn gvFailed()->bool{return false;}
fn surfaceSharpeningEnabled()->bool{return true;}
fn surfaceSharpeningStrength()->f32{return bitcast<f32>(atomicLoad(&conditioning[24u]));}
fn lsvAccepted()->bool{return true;}
fn lsvAcceptedSlot()->u32{return 0u;}
fn lsvHeader(s:u32,w:u32)->u32{_=s;return w;}
fn lsvLoad(w:u32)->u32{return select(0u,8u,w==3u);}
fn lsvConstraintCount(s:u32,v:u32)->u32{_=s;return u32(state[48u+v]);}
fn lsvVertexSupport(s:u32,b:u32,v:u32)->u32{_=s;_=b;return u32(state[32u+v]);}
fn lsvVertexPhi(s:u32,b:u32,v:u32)->f32{_=s;return state[8u*b+v];}
fn lsvPhiBase(s:u32,b:u32)->u32{_=s;return 8u*b;}
fn lsvStoreFloat(i:u32,v:f32){state[i]=v;}
` + production.slice(production.indexOf("// One bounded global volume feedback step."));
    const module = device.createShaderModule({ code });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "correctWholeFrameVolumePhi" } });
    const initial = [0.05, -0.05, 1.5, -1.5, 3, 0.001, -0.001, 0.05];
    for (const fixture of [
      { residual: 100, area: 1, strength: 1, offset: 0.1 },
      { residual: -100, area: 1, strength: 1, offset: -0.1 },
      { residual: 0.04, area: 2, strength: 1, offset: 0.005 },
      { residual: 0, area: 1, strength: 1, offset: 0 },
      { residual: 100, area: 0, strength: 1, offset: 0 },
      { residual: 100, area: 1, strength: 0, offset: 0 },
    ]) {
      const values = new Float32Array(64); values.set(initial); values.set(initial, 8);
      values.fill(3, 32, 40); values[37] = 1; values[38] = 2; values[55] = 1;
      const controls = new Float32Array(32); controls[22] = fixture.residual;
      controls[23] = fixture.area; controls[24] = fixture.strength;
      const storage = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
      const control = device.createBuffer({ size: controls.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const read: GPUBuffer = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(storage, 0, values); device.queue.writeBuffer(control, 0, controls);
      const binding = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: storage } }, { binding: 1, resource: { buffer: control } },
      ] });
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, binding); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(storage, 0, read, 0, values.byteLength); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const actual: Float32Array = new Float32Array(read.getMappedRange());
      for (let bank = 0; bank < 2; bank++) initial.forEach((phi, i) => {
        const weight = i >= 5 ? 0 : Math.max(0, Math.min(1, 2 - Math.abs(phi)));
        assert.ok(Math.abs(actual[8 * bank + i]! - (phi - fixture.offset * weight)) < 1e-6,
          `${JSON.stringify(fixture)}: vertex ${i}, bank ${bank}`);
      });
      assert.deepEqual(Array.from(actual.slice(32)), Array.from(values.slice(32)), "support and constraint metadata remain intact");
      read.unmap(); storage.destroy(); control.destroy(); read.destroy();
    }
  } finally { device?.destroy(); gpu = undefined; await releaseWebGPUExclusiveLock(); }
});
