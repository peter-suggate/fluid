import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createGeometricVolumeResidentWGSL, type SparseGeometricVolumeLayout } from "../lib/methods/adaptive-volume/resident-volume.wgsl";
import { referenceSharpenVolume } from "../lib/methods/adaptive-volume/whole-frame-volume-coupling-reference";

const entries = ["proposeWholeFrameVolumeSharpening", "gatherWholeFrameVolumeSharpening", "commitWholeFrameVolumeSharpening"];
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("GPU sharpening spends eligible budgets conservatively without crossing an air gap", { timeout: 60_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "geometric sharpening budget fixtures");
  let device: GPUDevice | undefined, gpu: GPU | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create(["backend=metal"]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const production = createGeometricVolumeResidentWGSL({} as SparseGeometricVolumeLayout);
    const kernels = production.slice(production.indexOf("fn gvSharpeningFaceCentre"), production.indexOf("// Delete only whole dilute pages"));
    const code = `
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<storage,read> topology:array<u32>;
@group(0) @binding(2) var<storage,read_write> conditioning:array<atomic<i32>>;
struct Params { cells:u32, faces:u32, phi:f32, pad:u32 }
@group(0) @binding(3) var<uniform> p:Params;
const INVALID=0xffffffffu; const GV_CURRENT=0u; const GV_PLUS=32u;
const GV_MINUS=64u; const GV_LOW=96u; const GV_FLUX=512u;
const GV_WHOLE_FRAME_CONTROL=0u;
fn gvFailed()->bool{return atomicLoad(&conditioning[31])!=0;}
fn gvFault(a:u32,b:u32,c:f32,d:f32,e:f32){_=a;_=b;_=c;_=d;_=e;atomicStore(&conditioning[31],1);}
fn gvRoundoff(c:f32)->f32{return 9.5367431640625e-7*max(1.0,c);}
fn acceptedTemplateCellInvocation(i:u32)->u32{return select(INVALID,i,i<p.cells);}
fn acceptedTemplateRowInvocation(i:u32)->u32{return select(INVALID,i,i<p.faces);}
fn surfaceSharpeningEnabled()->bool{return true;}
fn surfaceSharpeningStrength()->f32{return 1.0;}
fn gvAcceptedPhysicalRow(r:u32)->bool{return r<p.faces;}
fn cnxPhysicalFaceRangeUnchecked(r:u32)->vec2u{return vec2u(r,r+1u);}
fn gvCells(f:u32)->vec2u{return vec2u(topology[512u+3u*f],topology[513u+3u*f]);}
fn gvRow(f:u32)->u32{return f;}
fn rowAxis(r:u32)->u32{return topology[514u+3u*r];}
fn gvArea(f:u32)->f32{_=f;return 1.0;}
fn rowOpenFraction(r:u32)->f32{_=r;return 1.0;}
fn cellWidths(c:u32)->vec3f{_=c;return vec3f(1.0);}
fn cellCenter(c:u32)->vec3f{return vec3f(state[256u+3u*c],state[257u+3u*c],state[258u+3u*c]);}
fn cellMinimumWidth(c:u32)->f32{_=c;return 1.0;}
fn cellVolume(c:u32)->f32{_=c;return 1.0;}
fn gvReceiverCapacity(c:u32)->f32{return state[200u+c];}
fn gvCellFaceRange(c:u32)->vec2u{return vec2u(topology[2u*c],topology[2u*c+1u]);}
fn gvCellFace(i:u32)->u32{return topology[64u+i];}
fn destinationDensity()->u32{return 128u;}
fn destinationGamma()->u32{return 160u;}
fn incrementalActivityMarkCellClosure(c:u32){_=c;}
struct PhiSample { phi:f32, metric:bool }
fn lsvSampleAt(q:vec3f)->PhiSample{return PhiSample(select(p.phi,0.4-(q.x-1.5),p.pad==1u),true);}
` + kernels;
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const pipelines = entries.map(entryPoint => device!.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } }));
    const star = Array.from({ length: 6 }, (_, face) => (face % 2 ? [0, face + 1] : [face + 1, 0]) as [number, number]);
    const fixtures = [
      { name: "one eligible face", amounts: [0.6, 0, 0, 0, 0, 0, 0], targets: [0, 0.6, 0, 0, 0, 0, 0] },
      { name: "six receivers", amounts: [1, 0, 0, 0, 0, 0, 0], targets: [0, 1, 1, 1, 1, 1, 1] },
      { name: "six donors", amounts: [0, 1, 1, 1, 1, 1, 1], targets: [1, 0, 0, 0, 0, 0, 0] },
      { name: "inward air-side return", amounts: [0.6, 0, 0, 0, 0, 0, 0], targets: [0, 0.6, 0, 0, 0, 0, 0], phi: 0.1, centres: [0.6, -0.4, 0.6, 0.6, 0.6, 0.6, 0.6], connected: true },
      { name: "air crest", amounts: [0.6, 0, 0, 0, 0, 0, 0], targets: [0, 0.6, 0, 0, 0, 0, 0], phi: 0.8, centres: [0.6, -0.4, 0.6, 0.6, 0.6, 0.6, 0.6] },
      { name: "one-hop relay", amounts: [0.3, 0.6, 0, 0, 0, 0, 0], targets: [0, 0, 1, 0, 0, 0, 0], centres: [0.4, 1.4, -0.6, 0.4, 0.4, 0.4, 0.4], receive: [0.7, 0.4, 1, 0, 0, 0, 0], linear: true, expected: [0.6, 0, 0.3, 0, 0, 0, 0] },
      { name: "relay must not draw outward", amounts: [0, 0.6, 0, 0, 0, 0, 0], targets: [0, 0, 0, 0, 0, 0, 0], centres: [0.4, -0.6, 0.4, 0.4, 0.4, 0.4, 0.4], receive: [1, 0, 0, 0, 0, 0, 0], phi: -0.1, expected: [0, 0.6, 0, 0, 0, 0, 0] },
      { name: "air gap", amounts: [1, 0, 0, 0, 0, 0, 0], targets: [0, 1, 1, 1, 1, 1, 1], phi: 0.01 },
    ];
    for (const fixture of fixtures) {
      const values = new Float32Array(1024), metadata = new Uint32Array(1024);
      const adjacency: number[][] = Array.from({ length: 7 }, () => []);
      star.forEach(([a, b], face) => { metadata.set([a, b, Math.floor(face / 2)], 512 + 3 * face); adjacency[a]!.push(2 * face + 1); adjacency[b]!.push(2 * face); });
      let offset = 0;
      for (let c = 0; c < 7; c++) {
        values[96 + c] = fixture.centres?.[c] ?? 0;
        values[c] = fixture.amounts[c]!; values[32 + c] = Math.max(0, fixture.amounts[c]! - fixture.targets[c]!);
        values[64 + c] = fixture.receive?.[c] ?? Math.max(0, fixture.targets[c]! - fixture.amounts[c]!); values[200 + c] = 1;
        const centre = [1.5, 1.5, 1.5]; if (c > 0) centre[Math.floor((c - 1) / 2)]! += c % 2 ? -1 : 1;
        values.set(centre, 256 + 3 * c);
        metadata.set([offset, offset + adjacency[c]!.length], 2 * c); metadata.set(adjacency[c]!, 64 + offset); offset += adjacency[c]!.length;
      }
      const make = (size: number, usage: number) => device!.createBuffer({ size, usage });
      const state = make(values.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const meta = make(metadata.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const control = make(128, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const params = make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const read = make(values.byteLength + 128, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const words = new Uint32Array([7, 6, 0, fixture.linear ? 1 : 0]); new Float32Array(words.buffer)[2] = fixture.phi ?? -0.1;
      device.queue.writeBuffer(state, 0, values); device.queue.writeBuffer(meta, 0, metadata); device.queue.writeBuffer(params, 0, words);
      const bindings = device.createBindGroup({ layout, entries: [state, meta, control, params].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder();
      for (const pipeline of pipelines) { const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(1); pass.end(); }
      encoder.copyBufferToBuffer(state, 0, read, 0, values.byteLength); encoder.copyBufferToBuffer(control, 0, read, values.byteLength, 128);
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const mapped = read.getMappedRange();
      const actual = new Float32Array(mapped).slice(0, 7);
      assert.equal(new Int32Array(mapped)[1024 + 31], 0, fixture.name);
      const expected = fixture.expected ?? ((fixture.phi && !fixture.connected) ? fixture.amounts : referenceSharpenVolume(fixture.amounts, fixture.targets, Array(7).fill(1), star));
      actual.forEach((value, c) => assert.ok(Math.abs(value - expected[c]!) < 2e-6, `${fixture.name}: cell ${c}, ${value}/${expected[c]}`));
      assert.ok(Math.abs(actual.reduce((a, b) => a + b, 0) - fixture.amounts.reduce((a, b) => a + b, 0)) < 2e-6, fixture.name);
      read.unmap(); for (const buffer of [state, meta, control, params, read]) buffer.destroy();
    }
  } finally { device?.destroy(); gpu = undefined; await releaseWebGPUExclusiveLock(); }
});
