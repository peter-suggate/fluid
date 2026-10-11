/** Diagnostic: replace materialized old extension face reads in the unchanged
 * FLIP cache/tap/snapshot kernels with direct slot/hierarchy resolution. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { uniformDetailField, type UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { GPUPassProfile } from "./gpu-pass-profile";

export function fn(source: string, name: string): string {
 const start = source.indexOf(`fn ${name}(`); assert.ok(start >= 0, name);
 let end = source.indexOf("{", start) + 1, depth = 1;
 for (; depth && end < source.length; end++) { if (source[end] === "{") depth++; else if (source[end] === "}") depth--; }
 assert.equal(depth, 0); return source.slice(start, end);
}

export class Capture {
 readonly device: GPUDevice;
 readonly sources = new WeakMap<GPUShaderModule, string>();
 readonly modules = new Map<string, string>();
 readonly pipelines = new WeakMap<GPUComputePipeline, GPUComputePipelineDescriptor>();
 readonly layouts = new WeakMap<GPUBindGroupLayout, GPUBindGroupLayoutDescriptor>();
 readonly pipelineLayouts = new WeakMap<GPUPipelineLayout, GPUPipelineLayoutDescriptor>();
 readonly groups = new WeakMap<GPUBindGroup, GPUBindGroupDescriptor>();
 constructor(readonly raw: GPUDevice) {
  this.device = new Proxy(raw, { get: (target, key) => {
   if (key === "createShaderModule") return (d: GPUShaderModuleDescriptor) => { const m = target.createShaderModule(d); this.sources.set(m, d.code); this.modules.set(d.label ?? "", d.code); return m; };
   if (key === "createBindGroupLayout") return (d: GPUBindGroupLayoutDescriptor) => { const l = target.createBindGroupLayout(d); this.layouts.set(l, { ...d, entries: [...d.entries] }); return l; };
   if (key === "createPipelineLayout") return (d: GPUPipelineLayoutDescriptor) => { const l = target.createPipelineLayout(d); this.pipelineLayouts.set(l, { ...d, bindGroupLayouts: [...d.bindGroupLayouts] }); return l; };
   if (key === "createBindGroup") return (d: GPUBindGroupDescriptor) => { const g = target.createBindGroup(d); this.groups.set(g, { ...d, entries: [...d.entries] }); return g; };
   if (key === "createComputePipelineAsync") return async (d: GPUComputePipelineDescriptor) => { const p = await target.createComputePipelineAsync(d); this.pipelines.set(p, d); return p; };
   const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
 }
}

function directSource(extension: string, dimensions: readonly number[], physical: string, staged = false, finalizeFine = false, finalizeAll = false): string {
 const [nx, ny, nz] = dimensions as readonly [number, number, number], tiles = nx * ny * nz / 64;
 const wall = 3 * tiles, coarseWall = wall + nx * ny + nx * nz + ny * nz;
 const flags = coarseWall + (nx * ny + nx * nz + ny * nz) / 16;
 const constants: Record<string, number> = { UE_TILES: tiles, UE_COARSE: 0, UE_UNIT_WALL: wall, UE_COARSE_WALL: coarseWall, UE_FLAGS: flags, UE_NONE: flags + 2 * tiles, UE_UNIT: flags + 2 * tiles + 1 };
 const names = ["umClosedWall", "ueFarTop", "ueUnitSlotIn", "ueUnitSlot", "ueCoarseWallIndex", "umSlot", "umSourcePoint", "umFarValue"];
 let helpers = names.map(name => fn(extension, name)).join("\n");
 for (const name of [...names, "ueTileWord"]) helpers = helpers.replaceAll(new RegExp(`\\b${name}\\b`, "g"), `ds_${name}`);
 for (const [name, value] of Object.entries(constants)) helpers = helpers.replaceAll(new RegExp(`\\b${name}\\b(?:\\(\\))?`, "g"), `${value}u`);
 helpers = helpers.replaceAll(/\bUM_INF\b/g, "1e20").replaceAll(/\bh\b/g, "dsH").replaceAll(/\bcoarse\b/g, "dsCoarse").replaceAll(/\borigins\b/g, "dsOrigins");
 return `
@group(3) @binding(0) var<storage,read> dsState:array<vec2f>;
@group(3) @binding(1) var<storage,${finalizeFine ? "read_write" : "read"}> dsOld:array<f32>;
@group(3) @binding(2) var dsCoarse:texture_3d<f32>;
@group(3) @binding(3) var dsOrigins:texture_3d<u32>;
@group(3) @binding(4) var<uniform> dsH:vec4f;
fn ds_ueTileWord(tile:vec3u)->u32{return umTopology[umTileAt(tile)];}
${helpers}
${physical}
fn ${finalizeAll ? "dsEvaluateFace" : "dsKnownFace"}(anchor:vec3i,axis:u32,width:u32)->f32{
 if(anchor[axis]<0){return dsPhysical(anchor,axis);}
 if(any(anchor<vec3i(0))||any(anchor>=vec3i(UM_D))){return 0.0;}
 if(width==0u){return 0.0;}
 let face=UMFace(UMOwner(),anchor,width,1u,axis,1);
 if(ds_umClosedWall(face)){return dsPhysical(anchor,axis);}
 let slot=ds_umSlot(anchor,axis,width);let tile=umTileAt(vec3u(anchor)/4u);
 ${finalizeFine ? "if(width==1u&&umTileWidth(tile)==1u){return dsOld[slot];}" : ""}
 if((umTileSupport(tile)&2u)!=0u&&dsState[slot].y<0.5e20){return dsOld[slot];}
 ${staged ? `// A width-4 face centre has exactly one nonzero far interpolation tap.
 if(width==4u){let p=clamp(anchor/4,vec3i(0),ds_ueFarTop(axis));let state=textureLoad(dsCoarse,p,0);let lower=textureLoad(dsOrigins,p,0)[axis];
  return select(0.0,0.0+state[axis],(u32(round(state.w))&(1u<<axis))!=0u&&lower!=0u);}` : ""}
 return ds_umFarValue(face);
}
${finalizeAll ? `fn dsKnownFace(anchor:vec3i,axis:u32,width:u32)->f32{
 if(anchor[axis]<0){return dsPhysical(anchor,axis);}
 if(any(anchor<vec3i(0))||any(anchor>=vec3i(UM_D))||width==0u){return 0.0;}
 return dsOld[ds_umSlot(anchor,axis,width)];
}` : ""}
fn dsFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return dsPhysical(anchor,axis);}
 if(any(anchor<vec3i(0))||any(anchor>=vec3i(UM_D))){return 0.0;}
 ${finalizeFine ? "if(umTileWidth(umTileAt(vec3u(anchor)/4u))==1u){return dsKnownFace(anchor,axis,1u);}" : ""}
 let owner=umOwnerAt(anchor);let face=umPositiveFaceAtAnchor(owner,axis,anchor);
 return dsKnownFace(anchor,axis,face.width);
}
// These callers have already certified the source tile's unit ownership.
fn dsTexel(anchor:vec3i)->vec4f{return vec4f(dsKnownFace(anchor,0u,1u),dsKnownFace(anchor,1u,1u),dsKnownFace(anchor,2u,1u),0.0);}
`.replaceAll(/\bUM_D\b(?!\()/g, extension.includes("fn UM_D()") ? "UM_D()" : "UM_D");
}

/** Keep the publisher's shared 27-cell far-source staging, but write the
 * snapshot's fine taps directly. A separate 64-lane dispatch leaves the
 * unchanged 256-lane coarse hanging-tap fill alone. */
function stagedFineSource(extension: string, dimensions: readonly number[], finalizeFine = false): string {
 const counts = extension.includes("struct UMSceneCounts") ? "umCounts.work" : "umCounts";
 const tilesRef = extension.includes("fn UM_TILES()") ? "UM_TILES()" : "UM_TILES";
 const names = ["ueOwnSlot", "umUnitExtensionFace", "umFarSlot", "umFarDistance", "umFarValueStaged", "umFarValueUniform", "uePublishFine"];
 let code = names.map(name => fn(extension, name)).join("\n");
 code = code.replace(/if\(origin\[axis\]==0u\)\{let face=umUnitExtensionFace\(owner,axis,-1\);boundary\[umNegativeBoundaryIndex\(origin,axis\)\]=umPhysical\(face\);\}/, "");
 assert.ok(!code.includes("boundary["), "fine tap publisher must not write negative boundary storage");
 const stores = [...code.matchAll(/udStore_output\(/g)].length; assert.equal(stores, 2);
 const flags = [...code.matchAll(/value\.w=udLoad_physical\([^;]+;/g)].length; assert.equal(flags, 2);
 code = code.replaceAll("udStore_output(", "udStore_unitVelocity(").replaceAll(/value\.w=udLoad_physical\([^;]+;/g, "value.w=0.0;");
 if (finalizeFine) code = code.replaceAll("udStore_unitVelocity(", "dsFinalizeFine(owner,value);udStore_unitVelocity(");
 assert.ok(!code.includes("udLoad_physical"));
 code = code.replaceAll("umPhysical(face)", "dsPhysical(face.anchor,face.axis)").replaceAll("ueFinishState(", "dsFinishState(");
 for (const name of [...names, "umClosedWall", "ueFarTop", "umSourcePoint", "farTile", "farState", "farLo", "farHi", "farMixed", "farRequired"]) code = code.replaceAll(new RegExp(`\\b${name}\\b`, "g"), `ds_${name}`);
 const [nx, ny, nz] = dimensions as readonly [number, number, number], tiles = nx * ny * nz / 64;
 const unit = 5 * tiles + nx * ny + nx * nz + ny * nz + (nx * ny + nx * nz + ny * nz) / 16 + 1;
 code = code.replaceAll(/\bUE_UNIT\b(?:\(\))?/g, `${unit}u`).replaceAll(/\bUM_INF\b/g, "1e20").replaceAll(/\bh\b/g, "dsH").replaceAll(/\bcoarse\b/g, "dsCoarse").replaceAll(/\borigins\b/g, "dsOrigins");
 return `
var<workgroup> ds_farTile:u32;
var<workgroup> ds_farState:array<vec4f,27>;
var<workgroup> ds_farLo:array<vec4f,81>;
var<workgroup> ds_farHi:array<vec3f,81>;
var<workgroup> ds_farMixed:atomic<u32>;
var<workgroup> ds_farRequired:u32;
fn dsFinishState(slot:u32)->vec2f{return vec2f(dsOld[slot],dsState[slot].y);}
${finalizeFine ? "fn dsFinalizeFine(owner:UMOwner,value:vec4f){for(var axis=0u;axis<3u;axis++){dsOld[ds_ueOwnSlot(owner,axis)]=value[axis];}}" : ""}
${code}
@compute @workgroup_size(64) fn directFineTaps(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var job=group.x;job<${counts}.x;job+=groups.x){
  let cell=vec3i(umTileCoord(umTopology[${tilesRef}+job])*4u+umCorner(lane,4u));
  ds_uePublishFine(umOwnerAt(cell),lane);workgroupBarrier();
 }
}
`;
}

function coarseFinalSource(extension: string): string {
 const counts = extension.includes("struct UMSceneCounts") ? "umCounts.work" : "umCounts";
 const tilesRef = extension.includes("fn UM_TILES()") ? "UM_TILES()" : "UM_TILES";
 return `
// One lane per coarse owner/component. Its positive patches have disjoint
// slots (fine-owned positives and coarse-to-fine plane extras are separate).
@compute @workgroup_size(64) fn directCoarseSlots(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x+65535u*64u*gid.y;let job=index/3u;let axis=index%3u;if(job>=${counts}.y){return;}
 let tile=umTopology[${tilesRef}+${counts}.x+job];let owner=umOwnerAt(vec3i(umTileCoord(tile)*4u));
 let first=umFace(owner,axis,1,0u);
 for(var part=0u;part<first.count;part++){
  let face=umFacePatch(first,part);let value=dsEvaluateFace(face.anchor,axis,face.width);
  dsOld[ds_umSlot(face.anchor,axis,face.width)]=value;
 }
}
`;
}

function variantSource(capture: Capture, descriptor: GPUComputePipelineDescriptor, name: string, loader: string, extCode: string, dimensions: readonly number[], staged: boolean, finalizeFine: boolean, finalizeAll: boolean): string {
   let code = capture.sources.get(descriptor.compute.module)!;
   const originalLoader = fn(code, loader), physical = originalLoader.replace(`fn ${loader}(`, "fn dsPhysical(");
   code = code.replace(originalLoader, `fn ${loader}(anchor:vec3i,axis:u32)->f32{return dsFace(anchor,axis);}`);
   if (name === "cache") {
    const old = fn(code, "cache"), changed = old.replace(/(?:udLoad4_extended\((?:vec3i\()?anchor\)?\)|textureLoad\(extended,anchor,0\))\[axis\]/g, "dsKnownFace(anchor,axis,4u)");
    assert.notEqual(changed, old); code = code.replace(old, changed);
   } else if (name === "taps") {
    const old = fn(code, code.includes("fn unitVelocityTapsJob(") ? "unitVelocityTapsJob" : "unitVelocityTaps"), changed = old.replace(/udLoad_extended\((?:vec3i\()?cell\)?\)|textureLoad\(extended,cell,0\)/g, "dsTexel(cell)");
    assert.notEqual(changed, old); code = code.replace(old, changed);
    if (staged) code = code.replace(fn(code, "umUnitJobs"), "fn umUnitJobs()->u32{return 0u;}");
   } else {
    for (const entry of ["umSampleVelocityFine", "umSampleVelocity1"]) {
     const old = fn(code, entry), changed = old.replace(/(?:udLoadH_velocity\((?:vec3i\()?base\+bit\)?\)|textureLoad\(velocity,base\+bit,0\))\[axis\]/g, "dsKnownFace(base+bit,axis,1u)");
     assert.notEqual(changed, old); code = code.replace(old, changed);
    }
   }
   code += directSource(extCode, dimensions, physical, staged, finalizeFine, finalizeAll);
   if (staged && name === "taps") code += stagedFineSource(extCode, dimensions, finalizeFine);
   if (finalizeAll && name === "taps") code += coarseFinalSource(extCode);
 return code;
}

async function run(): Promise<void> {
 const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
 const steps = Number(option("steps") ?? 96), rounds = Number(option("rounds") ?? 12);
 const integrated = process.argv.includes("--integrated");
 const finalizeAll = process.argv.includes("--finalize-all") || integrated, finalizeFine = process.argv.includes("--finalize-fine") || finalizeAll, staged = process.argv.includes("--staged") || finalizeFine;
 const split = process.argv.includes("--split");
 const definition = getSceneDefinition(option("scene") ?? "nbflip-figure-9-teaser"), quality = definition.methodProfile!.quality;
 const scene = structuredClone(sceneDocument(definition));
 const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile!.overrides });
 const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
 const out = option("out") ?? "artifacts/nbflip-ui-perf/extension-direct-snapshot.json";
 let device: GPUDevice | undefined, solver: WebGPUUniformReferenceSolver | undefined, profile: GPUPassProfile | undefined;
 const errors: string[] = [], buffers: GPUBuffer[] = [];
 try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(adapter.features.has("subgroups") ? ["subgroups" as const] : [])], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  profile = new GPUPassProfile(raw); const capture = new Capture(profile.device);
  device = managedGPUDevice(capture.device, { requireWorkerRealm: false });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  // Source adaptation is checked before the expensive 96-frame fixture.
  // Capacity growth may rebind resources during advancing, so final bind
  // groups are still created from the completed fixture below.
  const initialFrame = (solver as any).mixedFrame, initialFlip = initialFrame.narrowBandFlip;
  if(!integrated) for (const [name, pipeline, loader] of [
   ["cache", initialFlip.cache.pipeline, "umCacheExtended"],
   ["taps", initialFlip.stageTaps.builder.pipeline, "umLoadMixedFace"],
   ["snapshot", initialFlip.pipelines.get("snapshot"), "umLoadMixedFace"],
  ] as const) {
   const descriptor = capture.pipelines.get(pipeline)!;
   const code = variantSource(capture, descriptor, name, loader, capture.modules.get("Uniform mixed paired extension")!, initialFrame.ownership.capacity.lattice.dimensions, staged, finalizeFine, finalizeAll);
   const module = raw.createShaderModule({ label: `Direct snapshot preflight ${name}`, code });
   const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
   if (messages.length) { writeFileSync(`/tmp/fluid-direct-${name}.wgsl`, code); throw new Error(messages.map(message => `${message.lineNum}: ${message.message}`).join("\n")); }
  }
  console.log(JSON.stringify({ phase: "source-preflight", staged }));
  for (let n = 1; n <= steps; n++) { solver.advanceTo(n * dt, []); await solver.awaitFrameCompletion(); assert.equal(solver.info.simulationPipelineError, undefined); if (n % 24 === 0 || n === steps) console.log(JSON.stringify({ phase: "advance", frame: n })); }
  const frame = (solver as any).mixedFrame, ext = frame.extension, flip = frame.narrowBandFlip, ownership = frame.ownership, f = frame.fields;
  const allocate = (size: number, read = false) => { const b = raw.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | (read ? GPUBufferUsage.MAP_READ : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC) }); buffers.push(b); return b; };
  // Completed-frame parity points at the live bank. Snapshot's working bank
  // is binding 1 (the opposite index), so restore the preceding stage parity.
  const particles = flip.activeParticles as GPUBuffer; flip.parity ^= 1;
  const countRead = allocate(8, true); let e = raw.createCommandEncoder(); e.copyBufferToBuffer(flip.state, 0, countRead, 0, 8); raw.queue.submit([e.finish()]);
  await countRead.mapAsync(GPUMapMode.READ); const counts = new Uint32Array(countRead.getMappedRange().slice(0)); countRead.unmap();
  const count = counts[0]!; assert.ok(count > 0); raw.queue.writeBuffer(flip.state, 4, new Uint32Array([count]));
  const original = allocate(count * 48), reference = allocate(count * 16), stats = allocate(16), readStats = allocate(16, true);
  e = raw.createCommandEncoder(); e.copyBufferToBuffer(particles, 0, original, 0, original.size); raw.queue.submit([e.finish()]);
  const sweeps = Number(option("sweeps") ?? values.extensionFrontSweeps ?? 2), companion = allocate(ext.companionBytesAt(ownership.capacity.fineTiles));
  const common = { phase: f.phase, output: f.velocityScratch, outputNegative: f.negativeScratch, scratch: { buffer: frame.stageScratch }, params: frame.params.extension };
  const companionField = { scratch: companion, physical: f.departure, negative: f.negativeDeparture };
  const pairCurrent = ext.bind({ ...common, physical: f.velocity, negative: f.negative, companion: companionField });
  const pairOld = ext.bind({ ...common, physical: f.departure, negative: f.negativeDeparture, companion: companionField }) as readonly [UniformDetailGroup, UniformDetailGroup];
  const integratedOld=integrated?ext.bind({...common,physical:f.departure,negative:f.negativeDeparture,companion:companionField,snapshot:flip.snapshotUnitVelocity}):undefined;
  const integratedSnapshots=integrated?ext.companionBindings(companion).map((slots:GPUBufferBinding)=>flip.bindSnapshotSlots(slots)):undefined;
  const extCode = capture.modules.get("Uniform mixed paired extension")!; assert.ok(extCode);
  const dsLayout = raw.createBindGroupLayout({ entries: [
   ...[0, 1].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: (binding === 1 && finalizeFine ? "storage" : "read-only-storage") as GPUBufferBindingType } })),
   { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: "3d", sampleType: "unfilterable-float" } },
   { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { viewDimension: "3d", sampleType: "uint" } },
   { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ] });
  const dsGroups = pairOld.map(epoch => {
  const epochEntries = [...capture.groups.get(epoch.group)!.entries];
  const resource = (binding: number) => epochEntries.find(e => e.binding === binding)!.resource;
  return raw.createBindGroup({ layout: dsLayout, entries: [
   { binding: 0, resource: resource(0) }, { binding: 1, resource: resource(2) },
   { binding: 2, resource: ext.hierarchy.output.createView() }, { binding: 3, resource: ext.hierarchy.outputOrigins.createView() },
   { binding: 4, resource: { buffer: frame.params.extension, size: 16 } },
  ] });
  });
  const emptyLayout = raw.createBindGroupLayout({ entries: [] }), emptyGroup = raw.createBindGroup({ layout: emptyLayout, entries: [] });
  const rawView = (texture: GPUTexture, base: boolean) => { const field = uniformDetailField(texture); return (field ? base ? field.storage.baseOf(field.handle)! : field.storage.physical(field.handle) : texture).createView(); };
  const variants = new Map<string, { pipeline: GPUComputePipeline; group: GPUBindGroup }>();
  const variantsInput: [string, GPUComputePipeline, GPUBindGroup, number, number, string][] = [
   ["cache", flip.cache.pipeline, flip.cacheGroups.snapshot.group, 0, 1, "umCacheExtended"],
   ["taps", flip.stageTaps.builder.pipeline, flip.stageTaps.group.group, 0, 1, "umLoadMixedFace"],
   ["snapshot", flip.pipelines.get("snapshot"), flip.groups.snapshot[flip.parity].group, 8, 5, "umLoadMixedFace"],
  ];
  if(!integrated) for (const [name, pipeline, group, physicalBinding, negativeBinding, loader] of variantsInput) {
   const descriptor = capture.pipelines.get(pipeline)!; assert.ok(descriptor, name);
   const code = variantSource(capture, descriptor, name, loader, extCode, ownership.capacity.lattice.dimensions, staged, finalizeFine, finalizeAll);
   const oldLayout = capture.pipelineLayouts.get(descriptor.layout as GPUPipelineLayout)!, layouts = [...oldLayout.bindGroupLayouts];
   while (layouts.length < 3) layouts.push(emptyLayout); layouts.push(dsLayout);
   const oldGroup = capture.groups.get(group)!, groupEntries = [...oldGroup.entries];
   let entries = [...capture.layouts.get(oldGroup.layout)!.entries];
   if (name === "snapshot") {
    const keep = new Set([1, 4, 5, 8, 10, 11, 72, 74, 75]); entries = entries.filter(e => keep.has(e.binding));
    layouts[1] = raw.createBindGroupLayout({ entries }); layouts[2] = emptyLayout;
   }
   const bound = raw.createBindGroup({ layout: layouts[1]!, entries: groupEntries.filter(e => entries.some(d => d.binding === e.binding)).map(entry => {
    if (entry.binding === physicalBinding) return { ...entry, resource: rawView(f.departure, false) };
    if (entry.binding === physicalBinding + 64) return { ...entry, resource: rawView(f.departure, true) };
    if (entry.binding === negativeBinding) return { ...entry, resource: { buffer: f.negativeDeparture } };
    return entry;
   }) });
   const module = raw.createShaderModule({ label: `Direct old extension ${name}`, code });
   const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
   if (messages.length) { writeFileSync(`/tmp/fluid-direct-${name}.wgsl`, code); throw new Error(messages.map(m => `${m.lineNum}: ${m.message}`).join("\n")); }
   const variantLayout = raw.createPipelineLayout({ bindGroupLayouts: layouts });
   variants.set(name, { group: bound, pipeline: await raw.createComputePipelineAsync({ ...descriptor, layout: variantLayout, compute: { ...descriptor.compute, module } }) });
   if (staged && name === "taps") variants.set("fine", { group: bound, pipeline: await raw.createComputePipelineAsync({ ...descriptor, layout: variantLayout, compute: { ...descriptor.compute, module, entryPoint: "directFineTaps" } }) });
   if (finalizeAll && name === "taps") variants.set("coarse", { group: bound, pipeline: await raw.createComputePipelineAsync({ ...descriptor, layout: variantLayout, compute: { ...descriptor.compute, module, entryPoint: "directCoarseSlots" } }) });
  }
  const compareModule = raw.createShaderModule({ code: `
struct Particle{position:vec4f,velocity:vec4f,before:vec4u}
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<storage,read_write> reference:array<vec4u>;
@group(0) @binding(2) var<storage,read_write> result:array<atomic<u32>>;
override capture:bool=false;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=arrayLength(&reference)){return;}let value=particles[i].before;
 if(capture){reference[i]=value;return;}for(var c=0u;c<4u;c++){if(value[c]!=reference[i][c]){
  atomicAdd(&result[0],1u);atomicMin(&result[1],i);
  let a=value[c];let b=reference[i][c];let oa=select(a|0x80000000u,~a,(a&0x80000000u)!=0u);let ob=select(b|0x80000000u,~b,(b&0x80000000u)!=0u);
  atomicMax(&result[2],max(oa,ob)-min(oa,ob));atomicMax(&result[3],bitcast<u32>(abs(bitcast<f32>(a)-bitcast<f32>(b))));
 }}
}` });
  const comparison = await raw.createComputePipelineAsync({ layout: "auto", compute: { module: compareModule, entryPoint: "main" } });
  const captureBefore = await raw.createComputePipelineAsync({ layout: "auto", compute: { module: compareModule, entryPoint: "main", constants: { capture: 1 } } });
  const compareGroups = [comparison, captureBefore].map(pipeline => raw.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: particles } }, { binding: 1, resource: { buffer: reference } }, { binding: 2, resource: { buffer: stats } }] }));
  let activeSweeps = sweeps;
  const prepare = (encoder: GPUCommandEncoder) => { encoder.copyBufferToBuffer(original, 0, particles, 0, original.size); ext.preparePair(encoder, pairCurrent, activeSweeps); };
  const encode = (encoder: GPUCommandEncoder, direct: boolean) => {
   if (!direct) {
    ext.finishPair(encoder, pairOld, activeSweeps, true);
    if (!split) flip.snapshot(encoder);
    else {
     flip.cache.encode(encoder, flip.cacheGroups.snapshot); flip.encodeStageTaps(encoder);
     const pass = encoder.beginComputePass({ label: "Materialized snapshot sample" }); flip.encodeParticleWork(encoder, pass);
     flip.dispatchBatch(encoder, ["snapshot"], undefined, "snapshot", pass); pass.end();
    }
    return;
   }
   if(integrated){ext.finishSnapshot(encoder,integratedOld,activeSweeps);flip.snapshot(encoder,integratedSnapshots![activeSweeps%2]);return;}
   const pass = encoder.beginComputePass({ label: "Direct old extension restriction and hierarchy" });
   pass.setBindGroup(0, ownership.bindGroup); pass.setBindGroup(1, pairOld[activeSweeps % 2]!.group); pass.setPipeline(ext.companionRestrict);
   pass.dispatchWorkgroups(...ownership.capacity.tileDimensions.map((n: number) => Math.ceil(n / 4)) as [number, number, number]); ext.hierarchy.encode(pass, false); pass.end();
   let sample = split ? undefined : encoder.beginComputePass({ label: "Direct old extension cache taps and snapshot" });
   if (sample) flip.encodeParticleWork(encoder, sample);
   for (const name of finalizeAll ? ["fine", "coarse", "cache", "taps", "snapshot"] : staged ? ["fine", "cache", "taps", "snapshot"] : ["cache", "taps", "snapshot"]) {
    sample ??= encoder.beginComputePass({ label: `Direct snapshot ${name}` });
    if (split && name === "snapshot") flip.encodeParticleWork(encoder, sample);
    const variant = variants.get(name)!; sample.setBindGroup(0, ownership.bindGroup); sample.setBindGroup(1, variant.group); sample.setBindGroup(2, name === "taps" || name === "fine" || name === "coarse" ? ownership.hangingGroup : emptyGroup); sample.setBindGroup(3, dsGroups[activeSweeps % 2]!); sample.setPipeline(variant.pipeline);
    if (name === "cache") sample.dispatchWorkgroups(...ownership.capacity.lattice.dimensions.map((n: number) => Math.ceil((n / 4 + 2) / 4)) as [number, number, number]);
    else if (name === "fine") { if (ownership.capacity.fineTiles > 0) sample.dispatchWorkgroups(Math.min(65535, ownership.capacity.fineTiles)); }
    else if (name === "coarse") { const groups = Math.ceil(3 * ownership.capacity.tiles / 64); sample.dispatchWorkgroups(Math.min(65535, groups), Math.ceil(groups / 65535)); }
    else if (name === "taps") { if (ownership.capacity.fineTiles > 0) ownership.dispatchCounted(sample, variant.pipeline, ownership.capacity.tiles, "fineAndSeams"); }
    else sample.dispatchWorkgroupsIndirect(flip.particleDispatch, 0);
    if (split) { sample.end(); sample = undefined; }
   }
   sample?.end();
  };
  const compareSnapshot = async (value: number) => {
  assert.ok(Number.isSafeInteger(value) && value >= 0); activeSweeps = value;
  raw.queue.writeBuffer(stats, 0, new Uint32Array([0, 0xffffffff, 0, 0]));
  for (const direct of [false, true]) {
   e = raw.createCommandEncoder(); prepare(e); encode(e, direct); const pass = e.beginComputePass(); pass.setPipeline(direct ? comparison : captureBefore); pass.setBindGroup(0, compareGroups[direct ? 0 : 1]!); pass.dispatchWorkgroups(Math.ceil(count / 256)); pass.end();
   if (direct) e.copyBufferToBuffer(stats, 0, readStats, 0, 16); raw.queue.submit([e.finish()]);
  }
  await readStats.mapAsync(GPUMapMode.READ); const result = [...new Uint32Array(readStats.getMappedRange())]; readStats.unmap();
  const maximumAbsolute = new Float32Array(Uint32Array.from([result[3]!]).buffer)[0]!;
  return { sweeps: value, words: 4 * count, mismatches: result[0]!, firstMismatch: result[1]!, maximumULP: result[2]!, maximumAbsolute };
  };
  const parity = await compareSnapshot(sweeps);
  console.log(JSON.stringify({ phase: "parity", count, ...parity, errors }));
  const snapshotSweeps = [];
  for (const value of (option("snapshot-sweeps") ?? (finalizeAll ? "0,1,3" : "")).split(",").filter(Boolean).map(Number)) {
   const proof = await compareSnapshot(value); snapshotSweeps.push(proof); console.log(JSON.stringify({ phase: "snapshot-sweeps", ...proof }));
  }
  activeSweeps = sweeps;
  const waitPid = Number(option("wait-for-pid") ?? 0);
  if (waitPid) {
   assert.ok(Number.isSafeInteger(waitPid) && waitPid > 0);
   let seconds = 0;
   for (;;) {
    try { process.kill(waitPid, 0); } catch (error) { if ((error as NodeJS.ErrnoException).code === "ESRCH") break; throw error; }
    if (seconds++ % 60 === 0) console.log(JSON.stringify({ phase: "waiting-for-external-process-before-timing", pid: waitPid }));
    await new Promise(resolve => setTimeout(resolve, 1000));
   }
  }
  if (process.argv.includes("--wait-for-gpu-quiet")) {
   let seconds = 0;
   for (;;) {
    const processes = execFileSync("ps", ["-Ao", "pid=,comm=,args="], { encoding: "utf8" }).split("\n").filter(line => {
     const row = line.match(/^\s*(\d+)\s+(\S+)\s+(.*)$/);
     return row && Number(row[1]) !== process.pid && /(?:^|\/)node$/.test(row[2]!) && /tools\/(?:probe|benchmark)[^\s]*\.ts/.test(row[3]!);
    });
    if (!processes.length) break;
    if (seconds++ % 60 === 0) console.log(JSON.stringify({ phase: "waiting-for-external-benchmarks-before-timing", processes }));
    await new Promise(resolve => setTimeout(resolve, 1000));
   }
  }
  const trials: { direct: boolean; ms: number; passes: Awaited<ReturnType<GPUPassProfile["finish"]>> }[] = [];
  for (let round = 0; round < rounds + 2; round++) for (const direct of round % 2 ? [true, false] : [false, true]) {
   e = raw.createCommandEncoder(); prepare(e); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone();
   profile.start(); e = device.createCommandEncoder(); encode(e, direct); raw.queue.submit([e.finish()]); await raw.queue.onSubmittedWorkDone(); const passes = await profile.finish(); if (round >= 2) trials.push({ direct, ms: passes.reduce((s, p) => s + p.ms, 0), passes });
  }
  const distribution = (values: number[]) => {
   const sorted = [...values].sort((a, b) => a - b), mean = values.reduce((sum, value) => sum + value, 0) / values.length;
   return { mean, median: (sorted[Math.floor((sorted.length - 1) / 2)]! + sorted[Math.ceil((sorted.length - 1) / 2)]!) / 2,
    min: sorted[0]!, max: sorted.at(-1)!, stddev: Math.sqrt(values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length) };
  };
  const materialized = distribution(trials.filter(t => !t.direct).map(t => t.ms)), direct = distribution(trials.filter(t => t.direct).map(t => t.ms));
  const saving = distribution(Array.from({ length: rounds }, (_, round) => {
   const pair = trials.slice(2 * round, 2 * round + 2); return pair.find(t => !t.direct)!.ms - pair.find(t => t.direct)!.ms;
  }));
  const generalSweeps: { sweeps: number; words: number; mismatches: number }[] = [];
  const sweepCases = (option("general-sweeps") ?? "0,1,3").split(",").filter(Boolean).map(Number);
  if (sweepCases.length) {
   const oldScalar = ext.bind({ ...common, physical: f.departure, negative: f.negativeDeparture });
   const field = uniformDetailField(f.velocityScratch);
   const textures = [...new Set(field ? [field.storage.physical(field.handle), field.storage.baseOf(field.handle)!] : [f.velocityScratch as GPUTexture])];
   const records = textures.map(texture => { const bytesPerRow = Math.ceil(texture.width * 16 / 256) * 256; return { texture, bytesPerRow, rowsPerImage: texture.height, size: bytesPerRow * texture.height * texture.depthOrArrayLayers, extent: [texture.width, texture.height, texture.depthOrArrayLayers] as [number, number, number] }; });
   const sizes = [...records.map(r => r.size), (f.negativeScratch as GPUBuffer).size];
   const originals = sizes.map(size => allocate(size)), actual = sizes.map(size => allocate(size));
   const expected = [sizes.map(size => allocate(size)), sizes.map(size => allocate(size))];
   const allWords = 2 * sizes.reduce((sum, size) => sum + size / 4, 0);
   const comparisonModule = raw.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> reference:array<u32>;
@group(0) @binding(1) var<storage,read> actual:array<u32>;
@group(0) @binding(2) var<storage,read_write> stats:array<atomic<u32>>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<arrayLength(&reference);i+=65536u){if(reference[i]!=actual[i]){atomicAdd(&stats[0],1u);}}
}` });
   const comparisonPipeline = await raw.createComputePipelineAsync({ layout: "auto", compute: { module: comparisonModule, entryPoint: "main" } });
   const comparisonGroups = expected.map(stage => stage.map((reference, i) => raw.createBindGroup({ layout: comparisonPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: reference } }, { binding: 1, resource: { buffer: actual[i]! } }, { binding: 2, resource: { buffer: stats } }] })));
   const copy = (encoder: GPUCommandEncoder, destination: GPUBuffer[]) => {
    records.forEach((r, i) => encoder.copyTextureToBuffer({ texture: r.texture }, { buffer: destination[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, r.extent));
    encoder.copyBufferToBuffer(f.negativeScratch, 0, destination.at(-1)!, 0, sizes.at(-1)!);
   };
   const restore = (encoder: GPUCommandEncoder) => {
    records.forEach((r, i) => encoder.copyBufferToTexture({ buffer: originals[i]!, bytesPerRow: r.bytesPerRow, rowsPerImage: r.rowsPerImage }, { texture: r.texture }, r.extent));
    encoder.copyBufferToBuffer(originals.at(-1)!, 0, f.negativeScratch, 0, sizes.at(-1)!);
   };
   e = raw.createCommandEncoder(); copy(e, originals); raw.queue.submit([e.finish()]);
   for (const value of sweepCases) {
    assert.ok(Number.isSafeInteger(value) && value >= 0);
    e = raw.createCommandEncoder(); e.clearBuffer(stats); restore(e);
    ext.encode(e, oldScalar, value, false); copy(e, expected[0]!);
    ext.encode(e, frame.extensionGroups, value, true); copy(e, expected[1]!);
    restore(e); ext.preparePair(e, pairCurrent, value);
    for (let stage = 0; stage < 2; stage++) {
     ext.finishPair(e, stage ? frame.extensionGroups : pairOld, value, stage === 0); copy(e, actual);
     const pass = e.beginComputePass(); pass.setPipeline(comparisonPipeline);
     for (const group of comparisonGroups[stage]!) { pass.setBindGroup(0, group); pass.dispatchWorkgroups(256); }
     pass.end();
    }
    e.copyBufferToBuffer(stats, 0, readStats, 0, 16); raw.queue.submit([e.finish()]);
    await readStats.mapAsync(GPUMapMode.READ); const mismatches = new Uint32Array(readStats.getMappedRange())[0]!; readStats.unmap();
    const proof = { sweeps: value, words: allWords, mismatches }; generalSweeps.push(proof); console.log(JSON.stringify({ phase: "general-sweeps", ...proof }));
   }
  }
  const report = { date: new Date().toISOString(), sceneId: definition.id, values, dt, steps, rounds, integrated, staged, finalizeFine, finalizeAll, split, count, ...parity, means_ms: { materialized: materialized.mean, direct: direct.mean, saving: saving.mean }, distributions_ms: { materialized, direct, saving }, snapshotSweeps, generalSweeps, trials, errors };
  mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(report, null, 2) + "\n"); console.log(JSON.stringify(report));
  assert.deepEqual(errors, []); assert.equal(parity.mismatches, 0); assert.ok(snapshotSweeps.every(proof => proof.mismatches === 0)); assert.ok(generalSweeps.every(proof => proof.mismatches === 0));
 } finally { buffers.forEach(b => b.destroy()); solver?.destroy(); await device?.queue.onSubmittedWorkDone(); profile?.destroy(); device?.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
