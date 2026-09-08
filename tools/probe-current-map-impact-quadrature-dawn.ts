/** Replay one refused candidate support at explicit refinement depths. No
 * production state, velocities, map coefficients or tolerances are changed. */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { gunzipSync } from "node:zlib";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12CurrentMapWGSL, type SparseCM12CurrentMapLayout } from "../lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl";
import { createSparseCM12CurrentMapMeasureWGSL } from "../lib/methods/adaptive-mass/sparse-cm12-current-map-measure.wgsl";

const directory = resolve(process.argv[2] ?? "artifacts/current-map/quarter/visual-v19/coarse");
const cell = (process.argv.find(v => v.startsWith("--cell="))?.split("=")[1] ?? "15,8,15").split(",").map(Number);
const depths = (process.argv.find(v => v.startsWith("--depths="))?.split("=")[1] ?? "3,4,5").split(",").map(Number);
const metadata = JSON.parse(await readFile(join(directory, "current-map-failure.json"), "utf8"));
const config = JSON.parse(await readFile(join(directory, "configuration.json"), "utf8"));
const compressed = await readFile(join(directory, "current-map-failure.bin.gz"));
const unpacked = gunzipSync(compressed), captured = new Float32Array(unpacked.buffer, unpacked.byteOffset, unpacked.byteLength / 4);
const sourceBase = metadata.baseWords as number;
const sourceMap = metadata.map as SparseCM12CurrentMapLayout;
const shifted = Object.fromEntries(Object.entries(sourceMap).map(([key, value]) => [key,
  key === "coefficientBaseWords" ? (value as number[]).map(n => n - sourceBase)
    : key.endsWith("BaseWords") || key === "baseWords" || key === "endWords" ? (value as number) - sourceBase : value])) as unknown as SparseCM12CurrentMapLayout;
const measureBase = metadata.measure.baseWords - sourceBase;
const count = config.dimensions.reduce((a: number, b: number) => a * b, 1);
const ordinal = cell[0]! + config.dimensions[0] * (cell[1]! + config.dimensions[1] * cell[2]!);
assert.ok(cell.length === 3 && cell.every((n, axis) => Number.isInteger(n) && n >= 0 && n < config.dimensions[axis]));
const seedBase = captured.length, bank = 1 - metadata.retainedControl[1];
const source = new Float32Array(seedBase + 64); source.set(captured);
const scene = config.scene, sphere = scene.fluid.initialLiquidVolumes.find((v: { shape: string }) => v.shape === "sphere");
assert.ok(sphere && scene.fluid.initialCondition === "tank-fill");
const seed = source.subarray(seedBase), lower = config.origin as number[], upper = lower.map((v, axis) => v + config.h * config.dimensions[axis]);
seed.set([1, 2, 1, scene.voxelDomain.finestCellSize_m]);
seed.set([...lower, config.dimensions[0]], 4); seed.set([...upper, config.dimensions[1]], 8);
seed.set([...lower, config.dimensions[2]], 12);
seed[16] = 2; seed.set([sphere.center_m.x, sphere.center_m.y, sphere.center_m.z], 20); seed.set([sphere.radius_m, sphere.radius_m, sphere.radius_m], 24);
seed[32] = 3; seed.set([0, scene.container.height_m * scene.container.fillFraction, 0], 36);
const prefix = /* wgsl */ `
struct Parameters{frame:vec4f}
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<uniform> p:Parameters;
@group(0) @binding(2) var<storage,read_write> failures:array<atomic<u32>>;
const CM12_RETAINED_FIELD_BASE:u32=${seedBase}u;
fn cm12RetainedDensityAcceptedBank()->u32{return ${metadata.retainedControl[1]}u;}
fn cm12CurrentMapFailed()->bool{return atomicLoad(&failures[0])!=0u;}
fn cm12CurrentMapFail(code:u32,id:u32){atomicStore(&failures[0],code);atomicStore(&failures[1],id);}
fn cm12CurrentMapMeasureFailure(id:u32,error:vec4f,tolerance:vec4f,value:vec4f){
  cm12CurrentMapFail(21u,id);
  for(var i=0u;i<4u;i++){atomicStore(&failures[4u+i],bitcast<u32>(error[i]));
    atomicStore(&failures[8u+i],bitcast<u32>(tolerance[i]));atomicStore(&failures[12u+i],bitcast<u32>(value[i]));}
}
fn cm12CurrentMapMeasureCompleted(_id:u32){atomicAdd(&failures[2],1u);}
fn cm12CurrentMapNativeVelocity(_point:vec3f)->vec4f{return vec4f(0.0);}
fn cm12CurrentMapInitializeVelocity(_id:u32,_value:vec4f){}
fn cm12CurrentMapCoefficientBoundaryPoint(point:vec3f)->vec3f{return point;}
fn cm12CurrentMapCoefficientBoundaryValue(_point:vec3f,value:vec3f)->vec3f{return value;}
fn cm12CurrentMapVelocityBoundary(point:vec3f,value:vec3f)->vec3f{
  let dimensions=vec3f(${config.dimensions.map((v: number) => `${v}.0`).join(",")});var velocity=value;
  velocity.x*=clamp(2.0*min(point.x,dimensions.x-point.x),-1.0,1.0);
  velocity.z*=clamp(2.0*min(point.z,dimensions.z-point.z),-1.0,1.0);
  velocity.y*=clamp(2.0*point.y,-1.0,1.0);
  let outside=max(vec3f(0.0),max(-point,point-dimensions));
  let t=clamp((max(outside.x,max(outside.y,outside.z))-4.0)/8.0,0.0,1.0);
  return velocity*(1.0-t*t*(3.0-2.0*t));
}
fn cm12RetainedDensityVector(at:u32)->vec3f{return vec3f(state[at],state[at+1u],state[at+2u]);}
fn cm12SolidVoxelFractionQ8(_q:vec3i)->u32{return 0u;}
fn cm12RetainedDensityRigidPointOpen(_point:vec3f)->bool{return true;}
fn cm12RetainedDensityPhiMetres(point:vec3f)->f32{
  let lower=cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+4u);
  let upper=lower+p.frame.y*vec3f(${config.dimensions.map((v: number) => `${v}.0`).join(",")});
  if(any(point<lower)||any(point>upper)){return state[CM12_RETAINED_FIELD_BASE+3u];}
  let delta=(point-cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+20u))/cm12RetainedDensityVector(CM12_RETAINED_FIELD_BASE+24u);
  let sphere=0.5*state[CM12_RETAINED_FIELD_BASE+24u]*(dot(delta,delta)-1.0);
  return min(sphere,point.y-state[CM12_RETAINED_FIELD_BASE+37u]);
}
${createSparseCM12CurrentMapWGSL(shifted)}
`;
const dawnModule = process.env.WEBGPU_NODE_MODULE;
assert.ok(dawnModule, "Set WEBGPU_NODE_MODULE to the Dawn module");
const live = new Set<GPU>();
await acquireWebGPUExclusiveLock("dawn-test", "current-map-impact-support");
let device: GPUDevice | undefined, gpu: GPU | undefined;
const buffers: GPUBuffer[] = [], report: unknown[] = [];
try {
  const dawn = await import(pathToFileURL(dawnModule).href); Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: source.byteLength, maxBufferSize: source.byteLength } });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const allocate = (size: number, usage: GPUBufferUsageFlags) => {
    const buffer = device!.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
  };
  const state = allocate(source.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const parameters = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const failures = allocate(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const readback = allocate(80, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  device.queue.writeBuffer(state, 0, source); device.queue.writeBuffer(parameters, 0, new Float32Array([config.dt, config.h, 0, 0]));
  const bindingLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindingLayout] });
  const bindings = device.createBindGroup({ layout: bindingLayout,
    entries: [state, parameters, failures].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  for (const depth of depths) {
    const code = prefix + createSparseCM12CurrentMapMeasureWGSL({ baseWords: measureBase,
      dimensions: config.dimensions, maximumRefinementDepth: depth }).replace("let ordinal=group.x;", `let ordinal=${ordinal}u+group.x;`);
    const module = device.createShaderModule({ code });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => `${m.lineNum}:${m.linePos} ${m.message}`), []);
    const pipeline = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "integrateCurrentMapFineMeasureCooperative" } });
    device.queue.writeBuffer(failures, 0, new Uint32Array(16));
    device.queue.writeBuffer(state, 4 * (measureBase + 4 * (ordinal + bank * count)), new Float32Array(4).fill(-777));
    const start = performance.now(), encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(state, 4 * (measureBase + 4 * (ordinal + bank * count)), readback, 0, 16);
    encoder.copyBufferToBuffer(failures, 0, readback, 16, 64); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ); const mapped = readback.getMappedRange();
    const value = Array.from(new Float32Array(mapped, 0, 4)), receipt = Array.from(new Uint32Array(mapped, 16, 16));
    const floats = new Float32Array(mapped, 16, 16);
    const row = { depth, cell, ordinal, milliseconds: performance.now() - start, value, receipt: receipt.slice(0, 4),
      error: Array.from(floats.slice(4, 8)), tolerance: Array.from(floats.slice(8, 12)), unresolvedValue: Array.from(floats.slice(12, 16)) };
    readback.unmap(); report.push(row); console.log(JSON.stringify(row));
  }
  assert.deepEqual(errors, []);
  await writeFile(join(directory, `impact-cell-${ordinal}-depth-replay.json`), JSON.stringify(report, null, 2) + "\n");
} finally {
  for (const buffer of buffers) buffer.destroy(); device?.destroy();
  if (gpu) live.delete(gpu);
  await releaseWebGPUExclusiveLock();
}
