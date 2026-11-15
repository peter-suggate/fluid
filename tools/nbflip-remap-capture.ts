/** A frozen tier change of the old NB-FLIP engine ("uniform-narrow-band-flip") on the dam, for
 * lib/methods/narrow-band/brick-transfer-dam.test.ts: the frame-head remap's pre and post state,
 * decoded to the layout-neutral fixture of lib/methods/narrow-band/parity/remap-fixture.ts.
 *
 * No hook in the engine: the device is wrapped (as tools/probe-nbflip-copy-faces.ts does), and the
 * remap's two passes are found by their labels ("Uniform mixed remap", "Uniform mixed remap
 * publish"). Just before the remap pass, in its encoder, the tool records:
 * - every lattice tile's owner width and page kind, through the old engine's own topology readers
 *   (a kernel bound to the live ownership, which is still the old one);
 * - the logical velocity (U), scratch velocity (the census extension, the brick engine's US) and
 *   phi images (UniformDetailStorage.capture, which loads as the accessors do), and both low-face
 *   wall buffers (negative and negativeScratch).
 * Just after the publish pass, the same again (the live ownership has adopted the new
 * generation), and the remap's worklist, decoded to tile coordinates.
 *
 * Probe (listing counts per frame, no capture):
 *   node --import tsx tools/nbflip-remap-capture.ts --steps=8
 * Capture (writes lib/methods/narrow-band/parity/dam-remap-<step>.{bin,json}):
 *   node --import tsx tools/nbflip-remap-capture.ts --capture=STEP [--window=64]
 * Run through tools/gpu/fluid-gpu-run.sh check with WEBGPU_NODE_MODULE set. */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
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
import { uniformDetailModule, uniformDetailPipeline, type UniformDetailStorage } from "../lib/methods/uniform/uniform-detail-fields";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { encodeFixture, ownerOf, storedFaces, storedVertices, type RemapFixtureMeta, type V3 } from "../lib/methods/narrow-band/parity/remap-fixture";

const METHOD = "uniform-narrow-band-flip", SCENE = "high-resolution-dam-break";
const args = process.argv.slice(2);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const captureStep = option("capture") ? Number(option("capture")) : undefined, capturePass = Number(option("pass") ?? 0);
const steps = captureStep ?? Number(option("steps") ?? 8), W = Number(option("window") ?? 64);
assert.ok(W % 32 === 0 && W > 0, "the window is whole 32^3 pages");

const method = getMethod(METHOD), definition = getSceneDefinition(SCENE);
assert.equal(definition.id, SCENE);
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

// ---- The hook: the remap's pass and publish pass, by label, in the frame being stepped.
type Hook = { pre(e: GPUCommandEncoder): void; post(e: GPUCommandEncoder): void };
let hook: Hook | undefined, remapPasses = 0;
const intercepted = new Proxy(raw, { get(target, key) {
  if (key === "createCommandEncoder") return (d?: GPUCommandEncoderDescriptor) => {
    const encoder = target.createCommandEncoder(d);
    return new Proxy(encoder, { get(e, key) {
      if (key === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
        if (descriptor?.label === "Uniform mixed remap") { remapPasses++; hook?.pre(e); }
        const pass = e.beginComputePass(descriptor);
        if (descriptor?.label !== "Uniform mixed remap publish" || !hook) return pass;
        const h = hook;
        return new Proxy(pass, { get(p, key) {
          if (key === "end") return () => { p.end(); h.post(e); };
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

// ---- The engine's internals this tool reads (no field is written).
type Frame = { remap: { ownership: UniformMixedOwnership; worklist: GPUBuffer }; fields: { velocity: GPUTexture; velocityScratch: GPUTexture; phi: GPUTexture; negative: GPUBuffer; negativeScratch: GPUBuffer; detail: UniformDetailStorage } };
const frame = (solver as unknown as { mixedFrame: Frame }).mixedFrame;
const ownership = frame.remap.ownership, fields = frame.fields, detail = fields.detail;
const dims = ownership.layout.lattice.dimensions as readonly number[], tiles = dims.map(d => d / 4) as V3;
const tileCount = tiles[0] * tiles[1] * tiles[2], listWords = frame.remap.worklist.size / 4;
console.log(JSON.stringify({ kind: "scene", scene: SCENE, quality, dt, lattice: dims, listWords }));

// The decode kernels: the old engine's topology readers on the live ownership (group 0).
const OUT_LIST = 2 * tileCount;
const code = uniformMixedTopologyWGSL(ownership.layout, 0, "") + /* wgsl */`
@group(1) @binding(0) var<storage,read_write> capOut:array<u32>;
@group(1) @binding(1) var<storage,read> capWork:array<u32>;
const CAP_T:vec3u=vec3u(${tiles[0]}u,${tiles[1]}u,${tiles[2]}u);
// Per lattice tile: owner width (0 where no key) | page kind << 8; the block of its page (umLowFaceIndex's).
@compute @workgroup_size(64) fn capOwners(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+65535u*64u*gid.y;if(i>=CAP_T.x*CAP_T.y*CAP_T.z){return;}
 let t=vec3i(vec3u(i%CAP_T.x,(i/CAP_T.x)%CAP_T.y,i/(CAP_T.x*CAP_T.y)));
 let key=umTileKey(t);var w=0u;if(umKeyStored(key)){w=umTileWidth(key);}
 capOut[2u*i]=w|(umTileKind(t)<<8u);capOut[2u*i+1u]=umCellBlock(vec3u(t*4));
}
// The worklist (a copy of the remap's: [planes, -, -, count, entries]): tile coordinate x | y<<10 | z<<20, entry bits 26..31.
@compute @workgroup_size(64) fn capList(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+65535u*64u*gid.y;if(i==0u){capOut[${OUT_LIST}u]=capWork[3];}
 if(i>=min(capWork[3],${listWords - 4}u)){return;}
 let e=capWork[4u+i];let c=umTileCoord(e&0x03ffffffu);
 capOut[${OUT_LIST}u+1u+2u*i]=c.x|(c.y<<10u)|(c.z<<20u);capOut[${OUT_LIST}u+2u+2u*i]=e&0xfc000000u;
}`;
const module = uniformDetailModule(device, { label: "remap capture", specializeWorldGeometry: true, code });
const capLayout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }] });
const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, capLayout] });
const pipe = (entryPoint: string) => uniformDetailPipeline(device, ownership, { layout: pipelineLayout, compute: { module, entryPoint, constants: { umDispatchX: ownership.dispatchX } } });
const [ownersPipe, listPipe] = await Promise.all([pipe("capOwners"), pipe("capList")]);
const U = GPUBufferUsage;
const outWords = OUT_LIST + 1 + 2 * listWords;
const buf = (size: number, usage: number) => device.createBuffer({ size: Math.max(16, size), usage });
const workCopy = buf(4 * listWords, U.STORAGE | U.COPY_DST);
const groups = (n: number): [number, number] => { const g = Math.ceil(n / 64); return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)]; };
interface State { out: GPUBuffer; read: GPUBuffer; images?: { U: () => Promise<Float32Array>; US: () => Promise<Float32Array>; phi: () => Promise<Float32Array> }; walls?: { U: GPUBuffer; US: GPUBuffer } }
const encodeState = (e: GPUCommandEncoder, images: boolean, list: boolean): State => {
  const out = buf(4 * outWords, U.STORAGE | U.COPY_SRC), read = buf(4 * outWords, U.COPY_DST | U.MAP_READ);
  if (list) e.copyBufferToBuffer(frame.remap.worklist, 0, workCopy, 0, 4 * listWords);
  const group = device.createBindGroup({ layout: capLayout, entries: [{ binding: 0, resource: { buffer: out } }, { binding: 1, resource: { buffer: workCopy } }] });
  const pass = e.beginComputePass({ label: "remap capture" });
  pass.setBindGroup(0, ownership.bindGroup); pass.setBindGroup(1, group);
  pass.setPipeline(ownersPipe); pass.dispatchWorkgroups(...groups(tileCount));
  if (list) { pass.setPipeline(listPipe); pass.dispatchWorkgroups(...groups(listWords)); }
  pass.end();
  e.copyBufferToBuffer(out, 0, read, 0, 4 * outWords);
  if (!images) return { out, read };
  const wall = (b: GPUBuffer) => { const r = buf(b.size, U.COPY_DST | U.MAP_READ); e.copyBufferToBuffer(b, 0, r, 0, b.size); return r; };
  return { out, read, images: { U: detail.capture(e, fields.velocity), US: detail.capture(e, fields.velocityScratch), phi: detail.capture(e, fields.phi) }, walls: { U: wall(fields.negative), US: wall(fields.negativeScratch) } };
};
const mapped = async (b: GPUBuffer, kind: "u32" | "f32") => { await b.mapAsync(GPUMapMode.READ); const v = b.getMappedRange().slice(0); b.unmap(); b.destroy(); return kind === "u32" ? new Uint32Array(v) : new Float32Array(v); };

// ---- Step; probe every frame's listing, capture at captureStep.
const T = W / 4, inWindow = (c: V3) => c.every((v, a) => v < T && v < tiles[a]!);
for (let step = 1; step <= steps; step++) {
  // A frame may remap more than once (a host layout, then the frame head): each pass is reported, and --pass picks one.
  let seen = 0, pre: State | undefined;
  const posts: { state: State; pre?: State }[] = [];
  hook = { pre: e => { if (step === captureStep && seen === capturePass) pre = encodeState(e, true, false); }, post: e => { const capture = step === captureStep && seen === capturePass; posts.push({ state: encodeState(e, capture, true), pre: capture ? pre : undefined }); seen++; } };
  assert.ok(solver.advanceTo(step * dt, []), `step ${step} refused`);
  await (solver.awaitFrameCompletion ? solver.awaitFrameCompletion() : device.queue.onSubmittedWorkDone());
  hook = undefined;
  if (solver.info.simulationPipelineError) throw new Error(String(solver.info.simulationPipelineError));
  if (errors.length) throw new Error(`GPU error at step ${step}: ${errors[0]}`);
  for (const [pass, { state: post, pre }] of posts.entries()) {
  const capture = pre !== undefined;
  const words = await mapped(post.read, "u32") as Uint32Array, n = words[OUT_LIST]!;
  let refine = 0, coarsen = 0, neighbour = 0, win = 0, winRefine = 0, winCoarsen = 0;
  for (let i = 0; i < n; i++) {
    const c = words[OUT_LIST + 1 + 2 * i]!, f = words[OUT_LIST + 2 + 2 * i]!, t: V3 = [c & 1023, (c >>> 10) & 1023, (c >>> 20) & 1023];
    const changed = (f >>> 31) & 1, refined = (f >>> 30) & 1;
    if (refined) refine++; else if (changed) coarsen++; else neighbour++;
    if (inWindow(t)) { win++; if (refined) winRefine++; else if (changed) winCoarsen++; }
  }
  const owners = words.subarray(0, 2 * tileCount);
  let h = 0, bulk = 0, air = 0, winAir = 0, winBulk = 0;
  for (let i = 0; i < tileCount; i++) {
    const w = owners[2 * i]! & 255, kind = owners[2 * i]! >>> 8, t: V3 = [i % tiles[0], Math.floor(i / tiles[0]) % tiles[1], Math.floor(i / (tiles[0] * tiles[1]))];
    if (w === 1) h++; if (kind === 3) bulk++; if (kind === 1) air++;
    if (inWindow(t)) { if (kind === 1) winAir++; if (kind === 3) winBulk++; }
  }
  // Where the listing is: per 32^3 page, listed tiles (refine/coarsen/neighbour); and the BULK16 and AIR pages.
  const perPage = new Map<string, number[]>(), kinds = new Map<string, number>();
  for (let i = 0; i < n; i++) {
    const c = words[OUT_LIST + 1 + 2 * i]!, f = words[OUT_LIST + 2 + 2 * i]!, k = [c & 1023, (c >>> 10) & 1023, (c >>> 20) & 1023].map(v => v >> 3).join(",");
    const r = perPage.get(k) ?? [0, 0, 0]; r[(f >>> 30) & 1 ? 0 : (f >>> 31) ? 1 : 2]!++; perPage.set(k, r);
  }
  for (let i = 0; i < tileCount; i += 1) { const kind = owners[2 * i]! >>> 8; if (kind === 0) continue; const k = [i % tiles[0], Math.floor(i / tiles[0]) % tiles[1], Math.floor(i / (tiles[0] * tiles[1]))].map(v => v >> 3).join(","); kinds.set(k, kind); }
  if (option("pages")) console.log(JSON.stringify({ kind: "pages", step, pass, listed: Object.fromEntries(perPage), nonResident: Object.fromEntries(kinds) }));
  console.log(JSON.stringify({ kind: "frame", step, pass, listed: n, refine, coarsen, neighbour, window: { W, listed: win, refine: winRefine, coarsen: winCoarsen, airTiles: winAir, bulkTiles: winBulk }, hTilesAfter: h, airTiles: air, bulkTiles: bulk }));
  post.out.destroy();
  if (!capture) continue;

  // ---- Decode the capture to the fixture.
  const [preWords, preU, preUS, prePhi, preWallU, preWallUS] = await Promise.all([mapped(pre.read, "u32"), pre.images!.U(), pre.images!.US(), pre.images!.phi(), mapped(pre.walls!.U, "f32"), mapped(pre.walls!.US, "f32")]);
  const [postU, postUS, postPhi, postWallU, postWallUS] = await Promise.all([post.images!.U(), post.images!.US(), post.images!.phi(), mapped(post.walls!.U, "f32"), mapped(post.walls!.US, "f32")]);
  const widths = (w: Uint32Array) => {
    const out = new Uint8Array(T * T * T);
    for (let z = 0; z < T; z++) for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
      const i = x + tiles[0] * (y + tiles[1] * z), word = w[2 * i]!, width = word & 255, kind = word >>> 8;
      if (kind !== 0 || (width !== 1 && width !== 4)) throw new Error(`step ${step}: window tile ${[x, y, z]} has width ${width}, page kind ${kind} (0 resident, 1 AIR, 2 SOLID, 3 BULK16): the brick engine holds h and 4h tiles of resident pages only`);
      out[x + T * (y + T * z)] = width;
    }
    return out;
  };
  const oldOwner = widths(preWords as Uint32Array), newOwner = widths(words);
  const blockOf = (w: Uint32Array, c: V3) => w[2 * ((c[0] >> 2) + tiles[0] * ((c[1] >> 2) + tiles[1] * (c[2] >> 2))) + 1]!;
  // umLowFaceIndex of the cell p on its page's low plane along a.
  const lowIndex = (w: Uint32Array, p: V3, a: number) => {
    const l = p.map(v => v & 31), uv = a === 0 ? [l[1]!, l[2]!] : a === 1 ? [l[0]!, l[2]!] : [l[0]!, l[1]!];
    return (((blockOf(w, p) * 3 + a) << 10) | uv[0]! | (uv[1]! << 5)) >>> 0;
  };
  const [lx, ly] = dims as [number, number, number];
  const vIndex = (p: V3) => p[0] + (lx + 1) * (p[1] + (ly + 1) * p[2]);
  const cIndex = (c: V3) => c[0] + lx * (c[1] + ly * c[2]);
  const decode = (owner: Uint8Array, w: Uint32Array, phi: Float32Array, u: Float32Array, us: Float32Array, wallU: Float32Array, wallUS: Float32Array) => {
    const h = ownerOf(owner, T);
    const verts = storedVertices(h, W), faces = storedFaces(h, W);
    const phiOut = Float32Array.from(verts, ({ p }) => phi[vIndex(p)]!);
    // A face: a low wall from its owner's low-face row (the h cell, or the tile's origin), else its lower cell's +face texel.
    const face = (img: Float32Array, wall: Float32Array) => Float32Array.from(faces, ({ c, a, kind }) => {
      if (c[a] === 0) { const origin: V3 = kind === "F" && h([c[0] >> 2, c[1] >> 2, c[2] >> 2]) ? c : [c[0] & ~3, c[1] & ~3, c[2] & ~3]; return wall[lowIndex(w, origin, a)]!; }
      const b: V3 = [...c]; b[a]!--; return img[4 * cIndex(b) + a]!;
    });
    return { phi: phiOut, U: face(u, wallU), US: face(us, wallUS), verts: verts.length, faces: faces.length };
  };
  const f32 = (x: Float32Array | Uint32Array) => x as Float32Array;
  const a = decode(oldOwner, preWords as Uint32Array, prePhi, preU, preUS, f32(preWallU), f32(preWallUS));
  const b = decode(newOwner, words, postPhi, postU, postUS, f32(postWallU), f32(postWallUS));
  const listed: number[] = [];
  for (let i = 0; i < n; i++) {
    const c = words[OUT_LIST + 1 + 2 * i]!, t: V3 = [c & 1023, (c >>> 10) & 1023, (c >>> 20) & 1023];
    if (inWindow(t)) listed.push(t[0] + T * (t[1] + T * t[2]), words[OUT_LIST + 2 + 2 * i]!);
  }
  const { bin, sections } = encodeFixture({ oldOwner, newOwner, listed: Uint32Array.from(listed), prePhi: a.phi, preU: a.U, preUS: a.US, postPhi: b.phi, postU: b.U, postUS: b.US });
  const meta: RemapFixtureMeta = { scene: SCENE, step, window: W, lattice: dims, listed: { total: n, refine, coarsen, neighbour, window: win }, sections, notes: [
    `${METHOD} (${quality}), dt ${dt}: frame ${step}'s remap pass ${pass} (UniformMixedRemap.encodePass), captured by tools/nbflip-remap-capture.ts.`,
    "pre: just before the remap pass (live ownership = old); post: just after the publish pass (live ownership = new).",
    "U: the velocity field and its low-face walls (negative); US: the scratch velocity, the census extension the remap carries, and its walls (negativeScratch).",
    `stored locations: pre ${a.verts} vertices and ${a.faces} faces, post ${b.verts} and ${b.faces}.`,
  ] };
  const base = resolve(`lib/methods/narrow-band/parity/dam-remap-${step}`);
  writeFileSync(`${base}.bin`, bin); writeFileSync(`${base}.json`, JSON.stringify(meta, null, 1) + "\n");
  console.log(JSON.stringify({ kind: "fixture", pass, file: `${base}.{bin,json}`, bytes: bin.length, ...meta.listed, windowRefine: winRefine, windowCoarsen: winCoarsen }));
  }
}
console.log(JSON.stringify({ kind: "done", remapPasses }));
solver.destroy();
process.exit(0);
