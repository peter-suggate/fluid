/** Compare the compiled extension neighbor against its frozen loop reference
 * on identical GPU inputs, including all side masks, candidate counts, tie
 * cases and skipped planes. Also reports alternating-order kernel timings.
 * node --import tsx tools/probe-compiled-extension.ts [--out=FILE] [--repeats=64]
 * Run alone: another GPU benchmark invalidates these diagnostic timings. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { uniformCompiledExtensionNeighborWGSL as candidate } from "../lib/methods/uniform/uniform-compiled-extension.wgsl";
import { uniformCompiledExtensionNeighborWGSL as reference } from "../tests/fixtures/uniform-compiled-extension-loop.wgsl";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { GPUPassProfile } from "./gpu-pass-profile";

const probes: number[][] = [];
const repeats = Number(process.argv.find(a => a.startsWith("--repeats="))?.slice(10) ?? 64);
assert.ok(Number.isInteger(repeats) && repeats >= 1 && repeats <= 1024);
for (let mask = 0; mask < 64; mask++) for (let component = 0; component < 3; component++) {
  for (let step = 0; step < 3; step++) for (let branch = 0; branch < 3; branch++) {
    for (const offset of [-4, -1, 0, 1, 4]) for (let mode = 0; mode < 4; mode++) {
      for (const spacing of [1, .037]) {
        const width = branch === 0 ? 4 : 1;
        const point = [12 + (width === 1 ? .5 : 0), 16 + (width === 1 ? .5 : 0), 20 + (width === 1 ? .5 : 0)];
        point[component] = branch === 2 ? 14 : 16;
        const center = [...point]; center[step]! += offset;
        probes.push([...point, width, ...center, spacing, component, step, mask, mode]);
      }
    }
  }
}
const packed = new ArrayBuffer(probes.length * 48), floats = new Float32Array(packed), words = new Uint32Array(packed);
for (let i = 0; i < probes.length; i++) {
  const probe = probes[i]!;
  for (let k = 0; k < 8; k++) floats[12 * i + k] = probe[k]!;
  for (let k = 8; k < 12; k++) words[12 * i + k] = probe[k]!;
}
const common = /* wgsl */ `
struct UMNeighbor{value:f32,distance:f32,spatial:f32}
struct Probe{point:vec4f,center:vec4f,options:vec4u}
const UM_D=vec3u(32);const UM_INF=1e20;
@group(0) @binding(0) var<storage,read> inputs:array<Probe>;
@group(0) @binding(1) var<storage,read_write> outputs:array<vec4u>;
var<private> h:vec4f;
var<private> mask:u32;
var<private> mode:u32;
fn umTileAt(tile:vec3u)->u32{return tile.x+8u*(tile.y+8u*tile.z);}
fn umFineFaceSides(tile:u32)->u32{return mask;}
fn umSlotState(anchor:vec3i,component:u32,width:u32)->vec2f{
 let k=u32(anchor.x+37*anchor.y+101*anchor.z)+17u*component+23u*width;
 // Finite velocities differ by patch. Distances include exact ties, near
 // ties on either side of the tolerance, and the unknown-distance sentinel.
 let value=f32(i32((k*1664525u+1013904223u)&65535u)-32768)*0.0037;
 var distance=f32(k%5u)*0.5;
 if(mode==1u){distance=2.0;}else if(mode==2u){distance=2.0+f32(k%4u)*0.000001;}
 else if(mode==3u){distance=select(UM_INF,1.0,(k&1u)!=0u);}
 return vec2f(value,distance);
}
`;
const entry = /* wgsl */ `
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=arrayLength(&inputs)){return;}
 let p=inputs[gid.x];h=vec4f(p.center.w,p.center.w*1.3,p.center.w*0.7,0);mask=p.options.z;mode=p.options.w;
 let r=ueCompiledNeighbor(p.point.xyz,p.center.xyz,p.options.x,p.options.y,u32(p.point.w));
 outputs[gid.x]=vec4u(bitcast<u32>(r.value),bitcast<u32>(r.distance),bitcast<u32>(r.spatial),0u);
}
`;
let device: GPUDevice | undefined, profile: GPUPassProfile | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  profile = new GPUPassProfile(device);
  const input = device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(input, 0, packed);
  const size = probes.length * 16;
  const output = [0, 1].map(() => device!.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }));
  const read = device.createBuffer({ size: size * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const pipelines: GPUComputePipeline[] = [], groups: GPUBindGroup[] = [], timingGroups: GPUBindGroup[] = [];
  for (const [index, source] of [reference, candidate].entries()) {
    const module = device.createShaderModule({ code: common + source + entry });
    const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    assert.deepEqual(messages, [], `shader ${index}`);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    pipelines.push(pipeline);
    groups.push(device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output[index]! } },
    ] }));
    // Both timed variants write the same resource so the backend must retain
    // their WAW dependency. Separate outputs can execute concurrently.
    timingGroups.push(device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: output[0]! } },
    ] }));
  }
  const dispatch = (encoder: GPUCommandEncoder, index: number, timing = false, count = repeats) => {
    const pass = encoder.beginComputePass({ label: index ? "candidate" : "reference" });
    pass.setPipeline(pipelines[index]!); pass.setBindGroup(0, (timing ? timingGroups : groups)[index]!);
    for (let repeat = 0; repeat < count; repeat++) pass.dispatchWorkgroups(Math.ceil(probes.length / 64));
    pass.end();
  };
  const warm = device.createCommandEncoder();
  for (let i = 0; i < 4; i++) { dispatch(warm, 0); dispatch(warm, 1); }
  device.queue.submit([warm.finish()]); await device.queue.onSubmittedWorkDone();
  const passes: Awaited<ReturnType<GPUPassProfile["finish"]>> = [];
  for (let round = 0; round < 8; round++) for (const index of round % 2 ? [1, 0] : [0, 1]) {
    profile.start(); const encoder = profile.device.createCommandEncoder(); dispatch(encoder, index, true);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    passes.push(...await profile.finish());
  }
  const parity = device.createCommandEncoder(); dispatch(parity, 0, false, 1); dispatch(parity, 1, false, 1);
  parity.copyBufferToBuffer(output[0]!, 0, read, 0, size); parity.copyBufferToBuffer(output[1]!, 0, read, size, size);
  device.queue.submit([parity.finish()]); await read.mapAsync(GPUMapMode.READ);
  const all = new Uint32Array(read.getMappedRange());
  let mismatches = 0; const examples: object[] = [];
  for (let i = 0; i < size / 4; i++) if (all[i] !== all[size / 4 + i]) {
    mismatches++;
    if (examples.length < 10) examples.push({ probe: probes[Math.floor(i / 4)], component: i % 4, reference: all[i], candidate: all[size / 4 + i] });
  }
  read.unmap();
  const means = Object.fromEntries(["reference", "candidate"].map(label => [label,
    passes.filter(p => p.label === label).reduce((sum, p) => sum + p.ms, 0) / (8 * repeats)]));
  const result = { adapter: adapter.info, probes: probes.length, components: probes.length * 3, repeats,
    timingIsolation: "shared output binding; each variant separately submitted and fenced", mismatches, examples, means_ms: means, passes, errors };
  const path = process.argv.find(a => a.startsWith("--out="))?.slice(6) ?? "artifacts/nbflip-ui-perf/compiled-extension-probe.json";
  mkdirSync(dirname(path), { recursive: true }); writeFileSync(path, JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify({ ...result, passes: undefined }));
  assert.deepEqual(errors, []); assert.equal(mismatches, 0, "All neighbor fields must match as raw f32 bits");
  input.destroy(); output.forEach(b => b.destroy()); read.destroy();
} finally { await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
