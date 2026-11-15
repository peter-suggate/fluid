/** A frozen census of the old NB-FLIP engine ("uniform-narrow-band-flip") on the dam, for
 * lib/methods/narrow-band/brick-census-dam.test.ts: the "Uniform dynamic ownership census" pass's
 * inputs and outputs, decoded to the layout-neutral fixture of
 * lib/methods/narrow-band/parity/census-fixture.ts.
 *
 * No hook in the engine: the device is wrapped (as tools/nbflip-remap-capture.ts does) and the
 * census pass is found by its label. Just before the pass, in its encoder (after encode's copy of
 * the region heat's deep COLD flags into the bound cubes), the tool records:
 * - every lattice tile's owner width, page kind and tile key, through the old engine's own
 *   topology readers on the live ownership;
 * - the census's φ and velocity as the census loads them: a kernel compiled as the census module is
 *   (the detail rewrite, the capacity topology) and bound to the census's own group 1, loading each
 *   vertex and face texel both plainly (an h owner's load) and canonically (UNIFORM_DETAIL_4H_LOAD,
 *   a 4h owner's); and the velocity's low-face walls. (UniformDetailStorage.capture, a host-side
 *   placement, came out page-permuted against the census's own loads: whole pages misread.) The kernel is compiled on the detail storage's own device object: the
 *   live layouts the rewrite needs are keyed by it, and an unrewritten load reads raw texels.
 * - the census words (for CC_DECIDED, the last census's band bits), the deep COLD flags, the join
 *   plane and the policy.
 * Just after the pass: the census words again (decided and band bits, importance words, keys,
 * crossing cells, gaps, census[18]).
 *
 * Probe (counts per census, every step, no fixture):
 *   node --import tsx tools/nbflip-census-capture.ts --steps=45
 * Capture (writes lib/methods/narrow-band/parity/dam-census-<step>.{bin.gz,json}, the .bin gzipped):
 *   node --import tsx tools/nbflip-census-capture.ts --capture=14,43 --window=128 [--steps=N]
 * (the test's files). A window under the lattice holds the lattice's low corner (lo = 0).
 * Run through tools/gpu/fluid-gpu-run.sh check with WEBGPU_NODE_MODULE set. */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import "../lib/methods";
import { getMethod } from "../lib/core/method-registry";
import { resolveMethodValues, type GPUSolverInstance } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { managedGPUDevice, whenGPUCompilationIdle } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformDetailModule, uniformDetailPipeline, uniformDetailShader, type UniformDetailStorage } from "../lib/methods/uniform/uniform-detail-fields";
import { UNIFORM_DETAIL_4H_LOAD } from "../lib/core/uniform-detail-abi";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { ownerOf, storedFaces, storedVertices, type V3 } from "../lib/methods/narrow-band/parity/remap-fixture";
import { encodeCensusFixture, type CensusFixtureMeta } from "../lib/methods/narrow-band/parity/census-fixture";

const METHOD = "uniform-narrow-band-flip", SCENE = "high-resolution-dam-break";
const args = process.argv.slice(2);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const captures = new Set((option("capture") ?? "").split(",").filter(Boolean).map(Number));
const steps = Number(option("steps") ?? Math.max(0, ...captures)) || 8, W = Number(option("window") ?? 64);
assert.ok(W % 32 === 0 && W > 0, "the window is whole 32^3 pages");

const method = getMethod(METHOD), definition = getSceneDefinition(SCENE);
const scene = sceneDocument(definition);
const profile = definition.methodProfile?.methodId === METHOD ? definition.methodProfile : undefined;
const quality = profile?.quality ?? "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...profile?.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter);
const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable", "timestamp-query"] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });

// ---- The hook: the census pass, by label.
const LABEL = "Uniform dynamic ownership census";
type Hook = { pre(e: GPUCommandEncoder): void; post(e: GPUCommandEncoder): void };
let hook: Hook | undefined;
const intercepted = new Proxy(raw, { get(target, key) {
  if (key === "createCommandEncoder") return (d?: GPUCommandEncoderDescriptor) => {
    const encoder = target.createCommandEncoder(d);
    return new Proxy(encoder, { get(e, key) {
      if (key === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
        const census = descriptor?.label === LABEL && hook ? hook : undefined;
        census?.pre(e);
        const pass = e.beginComputePass(descriptor);
        if (!census) return pass;
        return new Proxy(pass, { get(p, key) {
          if (key === "end") return () => { p.end(); census.post(e); };
          const v = Reflect.get(p, key, p); return typeof v === "function" ? v.bind(p) : v;
        } });
      };
      const v = Reflect.get(e, key, e); return typeof v === "function" ? v.bind(e) : v;
    } });
  };
  const v = Reflect.get(target, key, target); return typeof v === "function" ? v.bind(target) : v;
} });
const device = managedGPUDevice(intercepted, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
const solver: GPUSolverInstance = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
await solver.waitForSimulationReady?.();
await whenGPUCompilationIdle();

// ---- The engine's internals this tool reads (nothing is written).
interface Classifier { regionOffset: number; resources: GPUBindGroupLayout; group: { group: GPUBindGroup }; work: GPUBuffer; bounds: GPUBuffer; joinTiles: GPUBuffer; policyU: Uint32Array; words: number; joinWords: number; pages: number; replayOffset: number; decidedOffset: number; importanceOffset: number; heatTiers?: GPUBuffer }
type Frame = { ownership: UniformMixedOwnership; fields: { velocityScratch: GPUTexture; phi: GPUTexture; negativeScratch: GPUBuffer; detail: UniformDetailStorage } };
const internals = solver as unknown as { mixedFrame: Frame; mixedDynamic: Classifier; velocityB: GPUTexture; vertexPhiField: GPUTexture };
const frame = internals.mixedFrame, census = internals.mixedDynamic, fields = frame.fields, detail = fields.detail;
const ownership = frame.ownership;
const velocity = internals.velocityB, phiField = internals.vertexPhiField;
const velocityIsScratch = velocity === fields.velocityScratch;
assert.ok(phiField === fields.phi, "the census's φ is the frame's phi field");
const dims = ownership.layout.lattice.dimensions as readonly number[], tiles = dims.map(d => d / 4) as V3;
const tileCount = tiles[0] * tiles[1] * tiles[2];
const N_KEYS = census.work.size / 4;
const KEYS_AT = 20 + census.words, GAP_AT = 20 + census.words + 6 * 512 * (census.pages / 8);
const TILE_KEYS = 512 * (census.pages / 8), CROSS_AT = census.importanceOffset - 2 * TILE_KEYS;
const latticeInfo = ownership.layout.lattice as unknown as Record<string, unknown>;
const H_M = (latticeInfo.cellSize_m as number[])[0]!;
assert.ok((latticeInfo.cellSize_m as number[]).every(v => v === H_M), "a cubic lattice");
console.log(JSON.stringify({ kind: "scene", scene: SCENE, quality, dt, lattice: dims, latticeKeys: Object.keys(latticeInfo), h: H_M, tileKeys: TILE_KEYS, censusWords: N_KEYS, heat: !!census.heatTiers, velocityIsScratch, layoutTiles: ownership.layout.tiles.length, capacityTiles: ownership.capacity.tiles }));

// The decode kernel: per lattice tile, owner width | page kind << 8, and its tile key (0xffffffff where none).
// Compiled on the ownership's capacity, as the census module is (uniform-mixed-dynamic.ts): its tile keys are the census's.
const code = uniformMixedTopologyWGSL(ownership.capacity, 0, "") + /* wgsl */`
@group(1) @binding(0) var<storage,read_write> capOut:array<u32>;
const CAP_T:vec3u=vec3u(${tiles[0]}u,${tiles[1]}u,${tiles[2]}u);
@compute @workgroup_size(64) fn capOwners(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+65535u*64u*gid.y;if(i>=CAP_T.x*CAP_T.y*CAP_T.z){return;}
 let t=vec3i(vec3u(i%CAP_T.x,(i/CAP_T.x)%CAP_T.y,i/(CAP_T.x*CAP_T.y)));
 let key=umTileKey(t);var w=0u;var k=0xffffffffu;var f=0u;if(umKeyStored(key)){w=umTileWidth(key);k=key;f=select(0u,1u,umTileLive(key))|select(0u,2u,all(vec3i(umTileCoord(key))==t));}
 capOut[2u*i]=w|(umTileKind(t)<<8u)|(f<<16u);capOut[2u*i+1u]=k;
}`;
const module = uniformDetailModule(device, { label: "census capture", code });
const capLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, capLayout] });
const ownersPipe = await uniformDetailPipeline(device, ownership, { layout: pipelineLayout, compute: { module, entryPoint: "capOwners", constants: { umDispatchX: ownership.dispatchX } } });
// The image kernels: the census's φ (binding 1) and velocity (binding 2) through the census's own
// group 1 and its detail loads, plain (umLoadFine; an h owner's faces) and canonical (umLoadCorner;
// a 4h owner's wide patches). Out: plain φ at every lattice vertex, plain velocity at every cell
// (xyz), canonical φ at every tile corner, canonical velocity at every tile's three +face anchors.
const [LX, LY, LZ] = dims as [number, number, number];
const NV = (LX + 1) * (LY + 1) * (LZ + 1), NC = LX * LY * LZ, NV4 = (tiles[0] + 1) * (tiles[1] + 1) * (tiles[2] + 1);
const IMG = { phi: 0, us: NV, phi4: NV + 3 * NC, us4: NV + 3 * NC + NV4, words: NV + 3 * NC + NV4 + 3 * tileCount };
const imageCode = uniformMixedTopologyWGSL(ownership.capacity, 0, "") + /* wgsl */`
@group(1) @binding(1) var phi:texture_3d<f32>;
@group(1) @binding(2) var velocity:texture_3d<f32>;
@group(2) @binding(0) var<storage,read_write> img:array<f32>;
const IL:vec3u=vec3u(${LX}u,${LY}u,${LZ}u);
const IT:vec3u=vec3u(${tiles[0]}u,${tiles[1]}u,${tiles[2]}u);
fn ldFine(p:vec3i)->f32{return textureLoad(phi,p,0).x;}
fn ldCorner(p:vec3i)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,p,0).x;}
fn ldFace(p:vec3i)->vec4f{return textureLoad(velocity,p,0);}
fn ldWide(p:vec3i,a:u32)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,p,0)[a];}
fn lane(gid:vec3u)->u32{return gid.x+65535u*64u*gid.y;}
@compute @workgroup_size(64) fn imgPhi(@builtin(global_invocation_id) gid:vec3u){
 let i=lane(gid);let e=IL+1u;if(i>=e.x*e.y*e.z){return;}
 img[${IMG.phi}u+i]=ldFine(vec3i(vec3u(i%e.x,(i/e.x)%e.y,i/(e.x*e.y))));
}
@compute @workgroup_size(64) fn imgUS(@builtin(global_invocation_id) gid:vec3u){
 let i=lane(gid);if(i>=IL.x*IL.y*IL.z){return;}
 let u=ldFace(vec3i(vec3u(i%IL.x,(i/IL.x)%IL.y,i/(IL.x*IL.y))));
 img[${IMG.us}u+3u*i]=u.x;img[${IMG.us}u+3u*i+1u]=u.y;img[${IMG.us}u+3u*i+2u]=u.z;
}
@compute @workgroup_size(64) fn imgPhi4(@builtin(global_invocation_id) gid:vec3u){
 let i=lane(gid);let e=IT+1u;if(i>=e.x*e.y*e.z){return;}
 img[${IMG.phi4}u+i]=ldCorner(vec3i(4u*vec3u(i%e.x,(i/e.x)%e.y,i/(e.x*e.y))));
}
@compute @workgroup_size(64) fn imgUS4(@builtin(global_invocation_id) gid:vec3u){
 let i=lane(gid);if(i>=IT.x*IT.y*IT.z){return;}
 let t=vec3i(4u*vec3u(i%IT.x,(i/IT.x)%IT.y,i/(IT.x*IT.y)));
 for(var a=0u;a<3u;a++){var p=t;p[a]+=3;img[${IMG.us4}u+3u*i+a]=ldWide(p,a);}
}`;
// The detail rewrite must apply (a live layout of this tile count on this device): an unrewritten load reads raw physical texels.
// The storage's own device object (the live layouts are keyed by it; the solver may hold another wrapper than ours).
const detailDevice = (detail as unknown as { device: GPUDevice }).device;
{ const rewritten = uniformDetailShader(imageCode, detailDevice); assert.ok(rewritten !== imageCode && rewritten.includes("udb_phi") && rewritten.includes("udb_velocity"), "the image kernel takes the census's detail rewrite"); }
const imageModule = uniformDetailModule(detailDevice, { label: "census capture images", code: imageCode });
const imageOut = detailDevice.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
const imageLayout = detailDevice.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, census.resources, imageOut] });
const imagePipes = await Promise.all((["imgPhi", "imgUS", "imgPhi4", "imgUS4"] as const).map(entryPoint => uniformDetailPipeline(detailDevice, ownership, { layout: imageLayout, compute: { module: imageModule, entryPoint, constants: { umDispatchX: ownership.dispatchX } } })));
const imageCounts = [NV, NC, NV4, tileCount];
const U = GPUBufferUsage;
const buf = (size: number, usage: number) => device.createBuffer({ size: Math.max(16, size), usage });
const groups = (n: number): [number, number] => { const g = Math.ceil(n / 64); return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)]; };
const copyOf = (e: GPUCommandEncoder, b: GPUBuffer, offset = 0, size = b.size - offset) => { const r = buf(size, U.COPY_DST | U.MAP_READ); e.copyBufferToBuffer(b, offset, r, 0, size); return r; };
interface Pre { owners: GPUBuffer; work: GPUBuffer; deep: GPUBuffer; join: GPUBuffer; params: Uint32Array; images?: GPUBuffer; wall?: GPUBuffer }
const encodePre = (e: GPUCommandEncoder, images: boolean): Pre => {
  const out = buf(8 * tileCount, U.STORAGE | U.COPY_SRC);
  const group = device.createBindGroup({ layout: capLayout, entries: [{ binding: 0, resource: { buffer: out } }] });
  const pass = e.beginComputePass({ label: "census capture" });
  pass.setBindGroup(0, ownership.bindGroup); pass.setBindGroup(1, group); pass.setPipeline(ownersPipe); pass.dispatchWorkgroups(...groups(tileCount)); pass.end();
  const pre: Pre = { owners: copyOf(e, out), work: copyOf(e, census.work), deep: copyOf(e, census.bounds, 4 * census.replayOffset, 4 * census.pages), join: copyOf(e, census.joinTiles, 0, 4 * (census.joinWords + 1)), params: census.policyU.slice(0, 28) };
  if (images) {
    const img = buf(4 * IMG.words, U.STORAGE | U.COPY_SRC), g = device.createBindGroup({ layout: imageOut, entries: [{ binding: 0, resource: { buffer: img } }] });
    const ip = e.beginComputePass({ label: "census capture images" });
    ip.setBindGroup(0, ownership.bindGroup); ip.setBindGroup(1, census.group.group); ip.setBindGroup(2, g);
    imagePipes.forEach((pipe, j) => { ip.setPipeline(pipe); ip.dispatchWorkgroups(...groups(imageCounts[j]!)); });
    ip.end();
    pre.images = copyOf(e, img);
    if (velocityIsScratch) pre.wall = copyOf(e, fields.negativeScratch);
  }
  return pre;
};
const mapped = async (b: GPUBuffer) => { await b.mapAsync(GPUMapMode.READ); const v = b.getMappedRange().slice(0); b.unmap(); b.destroy(); return new Uint32Array(v); };
const f32 = (u: Uint32Array) => new Float32Array(u.buffer, u.byteOffset, u.length);
const bitAt = (w: Uint32Array, base: number, k: number) => (w[base + (k >>> 5)]! >>> (k & 31)) & 1;
const regionOf = (k: number) => 8 * (k >>> 9) + (((k >>> 2) & 1) | (((k >>> 5) & 1) << 1) | (((k >>> 8) & 1) << 2));
const coordOf = (i: number): V3 => [i % tiles[0], Math.floor(i / tiles[0]) % tiles[1], Math.floor(i / (tiles[0] * tiles[1]))];

// The region words after the pass: CC_REGION (2 a region), CC_MODE (1 replay, 2 classify), CC_LISTS (classified, replayed), the two lists.
const REGION_WORDS = 5 * census.pages + 4;
// ---- Step; report every census, capture at the chosen steps.
let printedPolicy = false;
for (let step = 1; step <= steps; step++) {
  const pending: { pre: Pre; post: GPUBuffer; regions: GPUBuffer }[] = [];
  let pre: Pre | undefined;
  hook = { pre: e => { pre = encodePre(e, captures.has(step)); }, post: e => { pending.push({ pre: pre!, post: copyOf(e, census.work), regions: copyOf(e, census.bounds, 4 * census.regionOffset, 4 * REGION_WORDS) }); pre = undefined; } };
  assert.ok(solver.advanceTo(step * dt, []), `step ${step} refused`);
  await (solver.awaitFrameCompletion ? solver.awaitFrameCompletion() : device.queue.onSubmittedWorkDone());
  hook = undefined;
  if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
  if (errors.length) throw new Error(`GPU error at step ${step}: ${errors[0]}`);
  for (const [pass, { pre: p, post, regions }] of pending.entries()) {
    const params = p.params, [owners, before, deepWords, join, after, regionWords] = await Promise.all([mapped(p.owners), mapped(p.work), mapped(p.deep), mapped(p.join), mapped(post), mapped(regions)]);
    // Per region (16^3 cells): 0 not listed this census (the census never visited it), 1 replayed, 2 classified (CC_MODE, replay's demotions included).
    const P = census.pages, LISTS = 3 * P, nClass = regionWords[LISTS]!, nReplay = regionWords[LISTS + 1]!;
    const regionMode = new Uint8Array(P);
    for (let j = 0; j < nClass; j++) regionMode[regionWords[LISTS + 4 + j]!] = 2;
    for (let j = 0; j < nReplay; j++) { const r = regionWords[LISTS + 4 + P + j]!; regionMode[r] = regionWords[2 * P + r]! === 2 ? 2 : 1; }
    if (!printedPolicy) { printedPolicy = true; console.log(JSON.stringify({ kind: "policy", u32: [...params.subarray(0, 28)], f32: [...f32(params).subarray(0, 28)] })); }
    const hostJoins = join[census.joinWords]!;
    // Per lattice tile.
    const pageKinds = [0, 0, 0, 0], windowNonResident = new Map<number, number>();
    let unvisited = 0, replayedT = 0, notLive = 0, badCoord = 0, unvisitedH = 0, unvisitedChanged = 0, hTiles = 0, deepTiles = 0, deepRegions = 0, joined = 0, decided = 0, changes = 0, refine = 0, coarsen = 0, triggered = 0, deepUsedPrev = 0, deepPrevBand = 0;
    const deepList: string[] = [];
    for (let i = 0; i < tileCount; i++) {
      const word = owners[2 * i]!, width = word & 255, kind = (word >>> 8) & 255, k = owners[2 * i + 1]!, t = coordOf(i);
      if (t.every(v => (v & 7) === 0)) pageKinds[kind]!++;
      for (const w of [64, 96, 128]) if (kind !== 0 && t.every(v => 4 * v < w)) windowNonResident.set(w, (windowNonResident.get(w) ?? 0) + 1);
      if (k === 0xffffffff) continue;
      if (width === 1) hTiles++;
      const flags = word >>> 16, m = regionMode[regionOf(k)]!;
      if (!(flags & 1)) notLive++;
      if (!(flags & 2)) badCoord++;
      if (m === 0) { unvisited++; if (width === 1) unvisitedH++; if ((bitAt(after, census.decidedOffset, k) === 1) !== (bitAt(before, census.decidedOffset, k) === 1)) unvisitedChanged++; }
      if (m === 1) replayedT++;
      const deep = deepWords[regionOf(k) - 0]! !== 0, j = bitAt(join, 0, k) === 1;
      const d = bitAt(after, census.decidedOffset, k) === 1;
      if (j) joined++;
      if (d) decided++;
      if (d !== (width === 1)) { changes++; if (d) refine++; else coarsen++; }
      if (((after[census.importanceOffset + 2 * k + 1]! >>> 16) & 63) !== 0) triggered++;
      if (deep) {
        deepTiles++;
        if (!j) { deepUsedPrev++; if (bitAt(before, census.decidedOffset, k)) deepPrevBand++; }
        if (deepList.length < 8) deepList.push(t.join(","));
      }
    }
    for (let r = 0; r < census.pages; r++) if (deepWords[r]) deepRegions++;
    const reach = f32(after)[18]!;
    console.log(JSON.stringify({ kind: "census", step, pass, pageKinds: { resident: pageKinds[0], air: pageKinds[1], solid: pageKinds[2], bulk16: pageKinds[3] }, windowNonResidentTiles: Object.fromEntries(windowNonResident), hTiles, decided, changes, refine, coarsen, triggered, joined, hostJoins, reach, regions: { classified: nClass, replayed: nReplay, unvisitedTiles: unvisited, unvisitedH, unvisitedDecidedChanged: unvisitedChanged, replayedTiles: replayedT, notLive, badCoord }, deep: { regions: deepRegions, tiles: deepTiles, decidedFromPrev: deepUsedPrev, prevBand: deepPrevBand, first: deepList } }));
    if (!p.images) continue;

    // ---- Decode the capture to the fixture over the window.
    const T = W / 4, n = T * T * T, wi = (t: V3) => t[0] + T * (t[1] + T * t[2]);
    const owner = new Uint8Array(n), joinedT = new Uint8Array(n), deepT = new Uint8Array(n), prevT = new Uint8Array(n), decidedT = new Uint8Array(n), bandT = new Uint8Array(n);
    const modeT = new Uint8Array(n), w0 = new Uint32Array(n), w1 = new Uint32Array(n), keys = new Uint32Array(6 * n), cross = new Uint32Array(2 * n), gap = new Uint32Array(n);
    const keyOf = new Map<number, number>();
    for (let z = 0; z < T; z++) for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
      const i = x + tiles[0] * (y + tiles[1] * z), word = owners[2 * i]!, width = word & 255, kind = (word >>> 8) & 255, k = owners[2 * i + 1]!, o = wi([x, y, z]);
      if (kind !== 0 || (width !== 1 && width !== 4) || k === 0xffffffff) throw new Error(`step ${step}: window tile ${[x, y, z]} has width ${width}, page kind ${kind} (0 resident, 1 AIR, 2 SOLID, 3 BULK16): the brick engine holds h and 4h tiles of resident pages only`);
      keyOf.set(o, k); modeT[o] = regionMode[regionOf(k)]!;
      owner[o] = width; joinedT[o] = bitAt(join, 0, k); deepT[o] = deepWords[regionOf(k)]! !== 0 ? 1 : 0; prevT[o] = bitAt(before, census.decidedOffset, k);
      decidedT[o] = bitAt(after, census.decidedOffset, k); bandT[o] = bitAt(after, 20, k);
      w0[o] = after[census.importanceOffset + 2 * k]!; w1[o] = after[census.importanceOffset + 2 * k + 1]!;
      for (let q = 0; q < 6; q++) keys[6 * o + q] = after[KEYS_AT + 6 * k + q]!;
      cross[2 * o] = after[CROSS_AT + 2 * k]!; cross[2 * o + 1] = after[CROSS_AT + 2 * k + 1]!; gap[o] = after[GAP_AT + k]!;
    }
    const img = f32(await mapped(p.images));
    const wall = p.wall ? f32(await mapped(p.wall)) : undefined;
    const h = ownerOf(owner, T), verts = storedVertices(h, W), faces = storedFaces(h, W);
    const [lx, ly] = dims as [number, number, number];
    const vIndex = (q: V3) => q[0] + (lx + 1) * (q[1] + (ly + 1) * q[2]);
    const cIndex = (c: V3) => c[0] + lx * (c[1] + ly * c[2]);
    const blockOf = (c: V3) => keyOf.get(wi([c[0] >> 2, c[1] >> 2, c[2] >> 2]))! >>> 9;
    const lowIndex = (q: V3, a: number) => { const l = q.map(v => v & 31), uv = a === 0 ? [l[1]!, l[2]!] : a === 1 ? [l[0]!, l[2]!] : [l[0]!, l[1]!]; return (((blockOf(q) * 3 + a) << 10) | uv[0]! | (uv[1]! << 5)) >>> 0; };
    // A vertex: the plain load where an h tile's closure holds it (an h owner reads it), else (a
    // 4h corner) the canonical load. The two differ at a 4-aligned vertex of an h closure: counted.
    const lt = (q: V3) => q[0] + tiles[0] * (q[1] + tiles[1] * q[2]);
    const hTileAt = (q: V3) => q.every((v, a) => v >= 0 && v < T && v < tiles[a]!) && h(q);
    const inHClosure = (q: V3) => { for (let dz = -1; dz <= 0; dz++) for (let dy = -1; dy <= 0; dy++) for (let dx = -1; dx <= 0; dx++) { const t: V3 = [(q[0] + 4 * dx) >> 2, (q[1] + 4 * dy) >> 2, (q[2] + 4 * dz) >> 2]; if ([dx, dy, dz].every((d, a) => d === 0 || (q[a]! & 3) === 0) && hTileAt(t)) return true; } return false; };
    let alignedDiffer = 0, wideDiffer = 0; const split: number[] = [], splitPhi: number[] = [];
    const phi = Float32Array.from(verts, ({ p: q }, vi) => {
      const plain = img[IMG.phi + vIndex(q)]!, aligned = q.every(v => (v & 3) === 0);
      const wide = aligned ? img[IMG.phi4 + q[0] / 4 + (tiles[0] + 1) * (q[1] / 4 + (tiles[1] + 1) * (q[2] / 4))]! : NaN;
      if (inHClosure(q)) { if (aligned && !Object.is(wide, plain)) { alignedDiffer++; split.push(vi); splitPhi.push(wide); } return plain; }
      assert.ok(aligned, `vertex ${q}: neither an h closure's nor a tile corner`); return wide;
    });
    // A face: its lower cell b's +face texel; plain where either side is an h tile, else the 4h
    // tiles' canonical anchor (b at local 3 on the axis, 0 across). A low wall (never read by the
    // census) from the walls, or 0 without them.
    const US = Float32Array.from(faces, ({ c, a, kind }) => {
      if (c[a] === 0) { if (!wall) return 0; const origin: V3 = kind === "F" && h([c[0] >> 2, c[1] >> 2, c[2] >> 2]) ? c : [c[0] & ~3, c[1] & ~3, c[2] & ~3]; return wall[lowIndex(origin, a)]!; }
      const b: V3 = [...c]; b[a]!--;
      const tb: V3 = [b[0] >> 2, b[1] >> 2, b[2] >> 2], tc: V3 = [c[0] >> 2, c[1] >> 2, c[2] >> 2];
      const plain = img[IMG.us + 3 * cIndex(b) + a]!;
      if (hTileAt(tb) || hTileAt(tc)) return plain;
      assert.ok((b[a]! & 3) === 3 && b.every((v, q) => q === a || (v & 3) === 0), `face ${c}/${a}: a 4h face off its canonical anchor`);
      const wide = img[IMG.us4 + 3 * lt(tb) + a]!; if (!Object.is(wide, plain)) wideDiffer++; return wide;
    });
    console.log(JSON.stringify({ kind: "images", step, pass, alignedHVerticesPlainVsCanonical: alignedDiffer, wideFacesCanonicalVsPlain: wideDiffer }));
    let winDeep = 0, winDecided = 0, winChanges = 0, winTriggered = 0, winJoined = 0;
    for (let o = 0; o < n; o++) { winDeep += deepT[o]!; winDecided += decidedT[o]!; if (decidedT[o] !== (owner[o] === 1 ? 1 : 0)) winChanges++; if ((w1[o]! >>> 16) & 63) winTriggered++; winJoined += joinedT[o]!; }
    const { bin, sections } = encodeCensusFixture({ split: Uint32Array.from(split), splitPhi: Float32Array.from(splitPhi), owner, mode: modeT, joined: joinedT, deep: deepT, decidedPrev: prevT, phi, US, decided: decidedT, band: bandT, w0, w1, keys, cross, gap });
    const pf = f32(params);
    const meta: CensusFixtureMeta = { scene: SCENE, step, pass, window: W, lattice: dims, h: H_M, dt: pf[0]!, gravity: [pf[12]! / pf[0]!, pf[13]! / pf[0]!, pf[14]! / pf[0]!], reachBits: after[18]!, header: [...after.subarray(0, 20)], policy: [...params.subarray(0, 28)],
      counts: { latticeHTiles: hTiles, latticeDecided: decided, latticeChanges: changes, latticeRefine: refine, latticeCoarsen: coarsen, latticeTriggered: triggered, latticeJoined: joined, latticeDeepTiles: deepTiles, latticeDeepRegions: deepRegions, latticeDeepFromPrev: deepUsedPrev, hostJoins, windowDeep: winDeep, windowDecided: winDecided, windowChanges: winChanges, windowTriggered: winTriggered, windowJoined: winJoined, vertices: verts.length, faces: faces.length },
      sections, notes: [
        `${METHOD} (${quality}), dt ${dt}: frame ${step}'s census pass ${pass} ("${LABEL}"), captured by tools/nbflip-census-capture.ts.`,
        "pre: just before the census pass, after encode's copy of the deep COLD flags; post: just after it.",
        `US: the census's velocity texture (the solver's velocityB${velocityIsScratch ? ", the frame's velocityScratch" : ""}); its low walls ${wall ? "from negativeScratch" : "are 0"} (the census never reads them).`,
        "φ in metres, as the census loads it: an h closure's vertex the plain (h store) load, a 4h corner the canonical (base) load. split/splitPhi: the 4-aligned h-closure vertices (indices into phi) whose canonical load differs from the plain one, and the canonical value (umLoadCorner reads it there, umLoadFine the plain one).",
        "gravity: policy flow.xyz / dt.",
      ] };
    const base = resolve(`lib/methods/narrow-band/parity/dam-census-${step}`);
    writeFileSync(`${base}.bin.gz`, gzipSync(bin, { level: 9 })); writeFileSync(`${base}.json`, JSON.stringify(meta, null, 1) + "\n");
    console.log(JSON.stringify({ kind: "fixture", step, pass, file: `${base}.{bin.gz,json}`, bytes: bin.length, counts: meta.counts }));
  }
  if (pending.length === 0) console.log(JSON.stringify({ kind: "census", step, none: true }));
}
solver.destroy();
process.exit(0);
