/** Small real-GPU regression for the production native-asset selector.
 * node --import tsx tools/check-imported-surface-batching.ts
 * Uses the shipped compute WGSL with only camera/primitive accessors stubbed;
 * no forest assets, renderer startup, or fluid solve is needed. */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { importedSurfaceWGSL } from "../lib/svo/features/primary-visibility/imported-surface.wgsl";
import { planImportedSurfaceGroups } from "../lib/svo/features/primary-visibility/imported-surface-groups";

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
assert.ok(adapter);
// In particular, do not request indirect-first-instance.
const device = await adapter.requestDevice();
const errors: string[] = [];
device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
const instances = Array.from({ length: 153 }, (_, i) => ({ asset: i % 3 === 0 ? "rock" : "tree", index: i * 3 + 7 }));
const levelsFor = (asset: string) => Array.from({ length: asset === "tree" ? 7 : 3 }, (_, level) => ({
  first: level * 300, count: 297 - level * 30, error: [0, .002, .01, .03, .1, .3, 1.5][level]!,
}));
const plan = planImportedSurfaceGroups(instances, levelsFor);
const records = new Uint32Array(instances.length * 32), recordFloats = new Float32Array(records.buffer);
const centres = new Float32Array((instances.at(-1)!.index + 1) * 4);
instances.forEach(({ asset, index }, i) => {
  const levels = levelsFor(asset), at = i * 32;
  records[at] = index; records[at + 1] = levels.length; recordFloats[at + 2] = 1;
  records[at + 3] = plan.firstGroupByAsset.get(asset)!;
  levels.forEach((level, j) => { records.set([level.first, level.count], at + 4 + j * 4); recordFloats[at + 6 + j * 4] = level.error; });
  const z = i % 11 === 0 ? -10 : 2 ** (1 + i % 11);
  centres.set([i % 5 === 0 ? z * 3 : (i % 7 - 3) * .1, i % 13 === 0 ? z * 4 : 0, z, 0], index * 4);
});
const groupWords = new Uint32Array(plan.groups.length * 4);
plan.groups.forEach((group, i) => groupWords.set([group.count, group.first, group.instanceOffset, group.capacity], i * 4));
const upload = (data: Uint32Array | Float32Array, usage = GPUBufferUsage.STORAGE) => {
  const buffer = device.createBuffer({ size: data.byteLength, usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
  new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)); buffer.unmap(); return buffer;
};
const positions = upload(centres), lods = upload(records), groups = upload(groupWords);
const camera = upload(new Float32Array([1200, 800, 0, 0, 0, 0, 0, 0]), GPUBufferUsage.UNIFORM);
const source = /* wgsl */ `
struct TestCamera{viewport:vec4f,eye:vec4f}
@group(0) @binding(0) var<storage,read> centres:array<vec4f>;
@group(0) @binding(1) var<uniform> uniforms:TestCamera;
const DRY_REVERSED_Z_NEAR_M:f32=.01;
fn dryPrimitive(index:u32)->vec4f{return centres[index];}
fn svoPrimitiveCenter_m(record:vec4f)->vec3f{return record.xyz;}
fn cameraTanHalfFov()->f32{return 1.;}
fn dryRasterPrimaryCamera()->array<vec3f,4>{return array<vec3f,4>(uniforms.eye.xyz,vec3f(0,0,1),vec3f(1,0,0),vec3f(0,1,0));}
${importedSurfaceWGSL.slice(0, importedSurfaceWGSL.indexOf("struct ImportedSurfaceVertex"))}`;
const module = device.createShaderModule({ code: source });
assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error"), []);
const sceneLayout = device.createBindGroupLayout({ entries: [
  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
] });
const selectionLayout = device.createBindGroupLayout({ entries: [80, 81, 82, 83].map(binding => ({
  binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 80 || binding === 83 ? "read-only-storage" as const : "storage" as const },
})) });
const layout = device.createPipelineLayout({ bindGroupLayouts: [sceneLayout, selectionLayout] });
const [reset, batched, reference] = await Promise.all(["importedSurfaceResetGroups", "importedSurfaceSelectBatched", "importedSurfaceSelect"].map(entryPoint =>
  device.createComputePipelineAsync({ layout, compute: { module, entryPoint } })));
const scene = device.createBindGroup({ layout: sceneLayout, entries: [{ binding: 0, resource: { buffer: positions } }, { binding: 1, resource: { buffer: camera } }] });
const make = (groupCount: number, capacity: number) => {
  const args = device.createBuffer({ size: groupCount * 20, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const ids = device.createBuffer({ size: capacity * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const read = device.createBuffer({ size: args.size + ids.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: selectionLayout, entries: [
    { binding: 80, resource: { buffer: lods } }, { binding: 81, resource: { buffer: args } },
    { binding: 82, resource: { buffer: ids } }, { binding: 83, resource: { buffer: groups } },
  ] });
  return { args, ids, read, group };
};
const batch = make(plan.groups.length, plan.instanceCapacity), old = make(instances.length, instances.length);
try {
  let frame = 0;
  for (const eye of [[0, 0, 0], [100_000, 0, 0], [0, 0, -100_000]]) {
    device.queue.writeBuffer(camera, 16, new Float32Array([...eye, 0]));
    for (const result of [batch, old]) {
      device.queue.writeBuffer(result.args, 0, new Uint32Array(result.args.size / 4).fill(0xabcdef));
      device.queue.writeBuffer(result.ids, 0, new Uint32Array(result.ids.size / 4).fill(0xdeadbeef));
    }
    const encoder = device.createCommandEncoder();
    for (const [pipeline, group, count] of [[reset!, batch.group, plan.groups.length], [batched!, batch.group, instances.length], [reference!, old.group, instances.length]] as const) {
      const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, scene); pass.setBindGroup(1, group);
      pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
    }
    for (const result of [batch, old]) { encoder.copyBufferToBuffer(result.args, 0, result.read, 0, result.args.size); encoder.copyBufferToBuffer(result.ids, 0, result.read, result.args.size, result.ids.size); }
    device.queue.submit([encoder.finish()]);
    await Promise.all([batch.read.mapAsync(GPUMapMode.READ), old.read.mapAsync(GPUMapMode.READ)]);
    const actual = new Uint32Array(batch.read.getMappedRange()), expected = new Uint32Array(old.read.getMappedRange());
    let visible = 0;
    plan.groups.forEach((group, i) => {
      const at = i * 5, count = actual[at + 1]!;
      assert.deepEqual(Array.from(actual.slice(at, at + 5)), [group.count, count, group.first, 0, 0], "base vertex and firstInstance remain zero on all devices");
      const selected = instances.filter(({ asset }, instance) => asset === group.asset && expected[instance * 5]! > 0 && expected[instance * 5 + 2] === group.first).map(instance => instance.index).sort((a, b) => a - b);
      assert.equal(count, selected.length); assert.ok(count <= group.capacity); visible += count;
      const offset = batch.args.size / 4 + group.instanceOffset;
      assert.deepEqual(Array.from(actual.slice(offset, offset + count)).sort((a, b) => a - b), selected, "compaction preserves primitive identity and exact reference LOD/frustum selection");
      assert.ok(Array.from(actual.slice(offset + count, offset + group.capacity)).every(value => value === 0xdeadbeef), "compaction writes only within its group's allocated segment");
    });
    if (frame === 0) assert.ok(visible > 0 && visible < instances.length);
    if (frame === 1) assert.equal(visible, 0, "reset removes every prior visible instance");
    if (frame === 2) assert.equal(visible, instances.length, "all instances fit even when each asset chooses one LOD");
    batch.read.unmap(); old.read.unmap(); console.log(JSON.stringify({ frame: frame++, visible, groups: plan.groups.length, parity: true }));
  }
  assert.deepEqual(errors, []);
} finally { device.destroy(); }
process.exit(0);
