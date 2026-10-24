/** Dawn proof of the block-grid nearest-source pyramid
 * (lib/methods/uniform/webgpu-uniform-velocity-extrapolation.ts) and of the
 * far lookup that reads its 4h level (umFarValue) against their host twin
 * (uniform-velocity-extrapolation-twin.ts): known masks, source bounds and
 * tap selections bit for bit, values to the rounding of a mean (the count of
 * values equal in every bit is reported).
 *
 * A pool of 72 blocks (a 4 x 4 x 5 block grid) holds a tank of 4 x 3 x 3
 * pages under an AIR top, 32 pages up the window's z so that source bounds
 * cross cell 1024, with AIR pages over one end, one SOLID page inside and the
 * resident pages in shuffled blocks with free ones between. The 4h root gets
 * known values and source bounds over a body of liquid and stale values
 * everywhere else, free blocks and sentinels included. The solver's own
 * extrapolator then runs its launches (restrict, block sweeps, prolong) and
 * the root it filled is read back: values, known masks, source bounds and tap
 * selections of every held block, and umFarValue at 8192 faces in and around
 * the tank. New root values are replayed through the recorded taps. A page is
 * then retired and another claimed in a free block, and all of it runs again
 * over the textures as they stand.
 *
 * node --import tsx tools/probe-uniform-extension-pyramid.ts
 * DUMP_WGSL=<dir>: write the probe's two modules and the kernels there and
 * stop before any device (for naga and Tint). */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { UNIFORM_PAGE_TABLE_WORDS as TABLE, uniformBlockNeighbourWords, uniformLatticeAddress, uniformLatticeAddressWGSL } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformExtensionFarValueWGSL, uniformExtensionSourceBoundsWGSL } from "../lib/methods/uniform/uniform-mixed-extension";
import {
  uniformExtrapolationFarValue, uniformExtrapolationLevel, uniformExtrapolationReplay, uniformExtrapolationRestrict, uniformExtrapolationRun, uniformExtrapolationSweep,
  uniformSourceWord, type UniformExtrapolationLevel,
} from "../lib/methods/uniform/uniform-velocity-extrapolation-twin";
import { uniformExtrapolationWorkBytes, WebGPUUniformVelocityExtrapolator } from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation";
import { UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS, uniformVelocityExtrapolationShader } from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation.wgsl";

type Vec3 = readonly [number, number, number];
const errors: string[] = [], started = performance.now();
const fail = (message: string) => { if (errors.push(message) <= 12) console.error(`FAIL ${message}`); };
const BLOCKS = 72, FREE = 0xffffffff, SOLID = 0xffffffff, FACES = 8192, SCALE = [1, 1, 1] as const;
const PAGES: Vec3 = [4, 3, 3], ORIGIN: Vec3 = [0, 0, 30];
const entry = (x: number, y: number, z: number) => x | (y << 6) | (z << 12);
/** A value is a mean of tied taps, a division the device may round an ulp
 * from the twin's; each level and the lookup compound it. */
const near = (got: number, expected: number) => Math.abs(got - expected) <= 2 ** -20 * Math.max(1, Math.abs(expected));
function random(s: number): () => number { return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 2 ** 32; }; }
const address = uniformLatticeAddress([32 * PAGES[0], 32 * PAGES[1], 32 * PAGES[2]], true, { blocks: BLOCKS });
const [gx, gy, gz] = address.blockGrid, W = 8 * gx, H = 8 * gy, D = 8 * gz, TEXELS = W * H * D, mask = (1 << address.blockBits) - 1;
assert.equal(address.blockBits, 2);assert.equal((16 * W) % 256, 0);

/** The tank's pages by box coordinate; outside it SOLID, AIR above. */
const kinds = new Map<number, "air" | "solid">([[entry(1, 0, 2), "solid"], [entry(3, 1, 0), "air"]]);
for (let z = 0; z < PAGES[2]; z++) for (let x = 2; x < PAGES[0]; x++) kinds.set(entry(x, 2, z), "air");
const table = new Uint32Array(TABLE).fill(SOLID), records = new Uint32Array(BLOCKS).fill(FREE), order = Array.from({ length: BLOCKS - 2 }, (_, i) => i + 1);
{
  const r = random(41);
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1));[order[i], order[j]] = [order[j]!, order[i]!]; }
  let held = 0;
  for (let z = 0; z < PAGES[2]; z++) for (let x = 0; x < PAGES[0]; x++) for (let y = 0; y < 34 - ORIGIN[1]; y++) {
    const kind = y < PAGES[1] ? kinds.get(entry(x, y, z)) ?? "resident" : "air", at = entry(x + ORIGIN[0], y + ORIGIN[1], z + ORIGIN[2]);
    if (kind === "air") table[at] = 0;
    else if (kind === "resident") { table[at] = order[held]!;records[order[held++]!] = at; }
  }
  assert.equal(held, 28);
}
const rows = () => uniformBlockNeighbourWords(table, records).subarray(0, 27 * BLOCKS);
const held = (b: number) => b >= 1 && b < BLOCKS - 1 && records[b] !== FREE;
const liquid = (x: number, y: number, z: number) => (y < 11 && x < 20) || (x > 9 && x < 15 && z > 6 && z < 13);

/** A 4h root by block: sources over the liquid, stale values everywhere else. */
function makeRoot(seed: number): UniformExtrapolationLevel {
  const r = random(seed), root = uniformExtrapolationLevel(BLOCKS, 8);
  for (let b = 0; b < BLOCKS; b++) for (let l = 0; l < 512; l++) {
    const cell = 512 * b + l, page = [records[b]! & 63, (records[b]! >>> 6) & 63, (records[b]! >>> 12) & 63], t = [8 * page[0]! + (l & 7), 8 * page[1]! + ((l >> 3) & 7), 8 * page[2]! + (l >> 6)];
    const wet = held(b) && liquid(t[0]! - 8 * ORIGIN[0], t[1]! - 8 * ORIGIN[1], t[2]! - 8 * ORIGIN[2]);
    for (let c = 0; c < 3; c++) {
      root.value[3 * cell + c] = Math.fround(8 * r() - 4);
      if (held(b) ? wet && r() < 0.9 : r() < 0.5) root.mask[cell] = root.mask[cell]! | (1 << c);
      for (let a = 0; a < 3; a++) {
        const low = held(b) ? 4 * t[a]! + (a === c ? 3 : Math.floor(4 * r())) : Math.floor(2048 * r());
        root.lower[9 * cell + 3 * c + a] = low;root.upper[9 * cell + 3 * c + a] = held(b) ? Math.min(low + (a === c ? 0 : Math.floor(3 * r())), 4 * t[a]! + 3) : Math.floor(2048 * r());
      }
    }
  }
  return root;
}
const texel = (b: number, l: number) => 8 * (b & mask) + (l & 7) + W * (8 * ((b >> address.blockBits) & mask) + ((l >> 3) & 7) + H * (8 * (b >> (2 * address.blockBits)) + (l >> 6)));
/** The root's value and origin texels, as the mixed owners write them. */
function texels(root: UniformExtrapolationLevel): { values: Float32Array<ArrayBuffer>; lower: Uint32Array<ArrayBuffer>; upper: Uint32Array<ArrayBuffer> } {
  const values = new Float32Array(4 * TEXELS), lower = new Uint32Array(4 * TEXELS), upper = new Uint32Array(4 * TEXELS);
  for (let b = 0; b < BLOCKS; b++) for (let l = 0; l < 512; l++) {
    const cell = 512 * b + l, i = 4 * texel(b, l);
    values[i + 3] = root.mask[cell]!;
    for (let c = 0; c < 3; c++) {
      const at = 9 * cell + 3 * c, lo = uniformSourceWord(root.lower[at]!, root.lower[at + 1]!, root.lower[at + 2]!), hi = uniformSourceWord(root.upper[at]!, root.upper[at + 1]!, root.upper[at + 2]!);
      values[i + c] = root.value[3 * cell + c]!;lower[i + c] = lo.word;upper[i + c] = hi.word;lower[i + 3] = (lower[i + 3]! | (lo.high << (24 + c))) >>> 0;upper[i + 3] = (upper[i + 3]! | (hi.high << (24 + c))) >>> 0;
    }
  }
  return { values, lower, upper };
}

const uploadWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> values:array<vec4f>;
@group(0) @binding(1) var<storage,read> lower:array<vec4u>;
@group(0) @binding(2) var<storage,read> upper:array<vec4u>;
@group(0) @binding(3) var valuesOut:texture_storage_3d<rgba32float,write>;
@group(0) @binding(4) var originsOut:texture_storage_3d<rgba32uint,write>;
@compute @workgroup_size(4,4,4) fn upload(@builtin(global_invocation_id) gid:vec3u){
 let d=textureDimensions(valuesOut);if(any(gid>=d)){return;}
 let i=gid.x+d.x*(gid.y+d.y*gid.z);let p=vec3i(gid);
 textureStore(valuesOut,p,values[i]);textureStore(originsOut,p,lower[i]);textureStore(originsOut,p+vec3i(0,0,i32(d.z)),upper[i]);
}`;
/** umFarValue as the extension builds it, over the least it reads: the
 * address, a face and its centre, and the filled root's two textures. */
const farWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> tailWords:array<u32>;
@group(0) @binding(1) var coarse:texture_3d<f32>;
@group(0) @binding(2) var origins:texture_3d<u32>;
@group(0) @binding(3) var<storage,read> faces:array<vec4i>;
@group(0) @binding(4) var<storage,read_write> far:array<f32>;
override umDispatchX:u32=100u;
const UM_D=vec3u(2048u);const UM_T=UM_D/4u;const UM_TILES:u32=${512 * BLOCKS}u;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return tailWords[e.x|(e.y<<6u)|(e.z<<12u)];}
fn umBlockWord(b:u32)->u32{return tailWords[${TABLE}u+b];}
fn umBlockNeighbourWord(b:u32,i:u32)->u32{return tailWords[${TABLE + BLOCKS}u+27u*b+i];}
${uniformLatticeAddressWGSL({ cells: "UM_D", tiles: "UM_T", constant: true, paged: { blocks: `${BLOCKS}u` } })}
struct UMFace{anchor:vec3i,width:u32,axis:u32}
fn umFaceCenter(face:UMFace)->vec3f{var center=vec3f(face.anchor)+vec3f(0.5*f32(face.width));center[face.axis]=f32(face.anchor[face.axis]+1);return center;}
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
const UM_INF=1e20;const h=vec4f(1.0,1.0,1.0,0.0);
${uniformExtensionSourceBoundsWGSL}
${uniformExtensionFarValueWGSL("umFarValue")}
@compute @workgroup_size(64) fn lookup(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x>=arrayLength(&faces)){return;}
 let f=faces[gid.x];far[gid.x]=umFarValue(UMFace(f.xyz,u32(f.w)>>2u,u32(f.w)&3u));
}`;
if (process.env.DUMP_WGSL) {
  const out = resolve(process.env.DUMP_WGSL);mkdirSync(out, { recursive: true });
  writeFileSync(resolve(out, "pyramid-upload.wgsl"), uploadWGSL);writeFileSync(resolve(out, "pyramid-far.wgsl"), farWGSL);
  writeFileSync(resolve(out, "pyramid-kernels.wgsl"), uniformVelocityExtrapolationShader(address.blocks, address.blockBits));
  console.log(`wrote 3 modules to ${out}`);process.exit(0);
}

/** Faces in and around the tank: anchor, then width << 2 | axis. */
const faces = new Int32Array(4 * FACES);
{
  const r = random(7);
  for (let i = 0; i < FACES; i++) {
    const width = r() < 0.25 ? 4 : 1, anchor = PAGES.map((n, a) => 32 * ORIGIN[a]! - 8 + Math.floor(r() * (32 * n + 16))).map(v => width === 4 ? v & ~3 : v);
    faces.set([...anchor, (width << 2) | Math.floor(3 * r())], 4 * i);
  }
}

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault();fail(`device: ${e.error.message}`); });
const storage = (label: string, bytes: number, data?: ArrayBufferView<ArrayBuffer>) => {
  const buffer = device.createBuffer({ label, size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};
async function mapped(source: GPUBuffer | GPUTexture, size: number, depth = D): Promise<ArrayBuffer> {
  const target = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), encoder = device.createCommandEncoder();
  if (source instanceof GPUBuffer) encoder.copyBufferToBuffer(source, 0, target, 0, size);
  else encoder.copyTextureToBuffer({ texture: source }, { buffer: target, bytesPerRow: 16 * W, rowsPerImage: H }, [W, H, depth]);
  device.queue.submit([encoder.finish()]);await target.mapAsync(GPUMapMode.READ);
  const bytes = target.getMappedRange().slice(0);target.unmap();target.destroy();return bytes;
}

const params = device.createBuffer({ label: "params", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(params, 0, new Float32Array([...address.cells, 1 / 60, 0.02, 0.02, 0.02, -9.81]));
const extrapolator = new WebGPUUniformVelocityExtrapolator(device, address, params);
const built = performance.now();await extrapolator.initialize();
const hierarchy = extrapolator.prepareMixedContinuation(), tail = storage("page pool tail", 4 * (TABLE + 28 * BLOCKS));
// Every gate runs; the held blocks follow the gate words and their count.
const work = storage("work words", uniformExtrapolationWorkBytes(BLOCKS));
hierarchy.attach(tail, 0, work);
const valuesBuffer = storage("root values", 16 * TEXELS), lowerBuffer = storage("root lower bounds", 16 * TEXELS), upperBuffer = storage("root upper bounds", 16 * TEXELS);
const facesBuffer = storage("faces", 16 * FACES, faces), farBuffer = storage("far values", 4 * FACES);
async function pipeline(label: string, code: string, entryPoint: string, entries: GPUBindGroupLayoutEntry[]): Promise<{ pipeline: GPUComputePipeline; layout: GPUBindGroupLayout }> {
  const module = device.createShaderModule({ label, code }), layout = device.createBindGroupLayout({ entries });
  for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") fail(`${label}: ${m.lineNum}: ${m.message}`);
  return { layout, pipeline: await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint } }) };
}
const buffer = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
const upload = await pipeline("upload", uploadWGSL, "upload", [buffer(0, "read-only-storage"), buffer(1, "read-only-storage"), buffer(2, "read-only-storage"),
  { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
  { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32uint", viewDimension: "3d" } }]);
const far = await pipeline("far lookup", farWGSL, "lookup", [buffer(0, "read-only-storage"),
  { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } }, { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
  buffer(3, "read-only-storage"), buffer(4, "storage")]);
if (errors.length) process.exit(1);
console.log(`pipelines built in ${((performance.now() - built) / 1000).toFixed(1)} s; hierarchy ${extrapolator.scratchBytes} B over a ${address.blockGrid.join("x")} block grid`);
const uploadGroup = device.createBindGroup({ layout: upload.layout, entries: [{ binding: 0, resource: { buffer: valuesBuffer } }, { binding: 1, resource: { buffer: lowerBuffer } }, { binding: 2, resource: { buffer: upperBuffer } },
  { binding: 3, resource: hierarchy.input.createView() }, { binding: 4, resource: hierarchy.inputOrigins.createView() }] });
const farGroup = device.createBindGroup({ layout: far.layout, entries: [{ binding: 0, resource: { buffer: tail } }, { binding: 1, resource: hierarchy.output.createView() }, { binding: 2, resource: hierarchy.outputOrigins.createView() },
  { binding: 3, resource: { buffer: facesBuffer } }, { binding: 4, resource: { buffer: farBuffer } }] });

/** Writes the root, runs the hierarchy's launches and the far lookup; returns the filled root's texels and the far values. */
async function launch(root: UniformExtrapolationLevel, replay: boolean) {
  const data = texels(root), encoder = device.createCommandEncoder();
  device.queue.writeBuffer(valuesBuffer, 0, data.values);device.queue.writeBuffer(lowerBuffer, 0, data.lower);device.queue.writeBuffer(upperBuffer, 0, data.upper);
  let pass = encoder.beginComputePass();pass.setPipeline(upload.pipeline);pass.setBindGroup(0, uploadGroup);pass.dispatchWorkgroups(W / 4, H / 4, D / 4);pass.end();
  pass = encoder.beginComputePass();hierarchy.encode(pass, replay);pass.end();
  pass = encoder.beginComputePass();pass.setPipeline(far.pipeline);pass.setBindGroup(0, farGroup);pass.dispatchWorkgroups(FACES / 64);pass.end();
  device.queue.submit([encoder.finish()]);
  const values = await mapped(hierarchy.output, 16 * TEXELS), origins = new Uint32Array(await mapped(hierarchy.outputOrigins, 32 * TEXELS, 2 * D));
  return { values: new Float32Array(values), bits: new Uint32Array(values), origins, far: new Uint32Array(await mapped(farBuffer, 4 * FACES)) };
}
/** The filled root of the device against the twin's, over every held block. */
function compare(label: string, got: Awaited<ReturnType<typeof launch>>, twin: UniformExtrapolationLevel, bounds: boolean): { cells: number; known: number; filled: number; equal: number; error: number } {
  const expected = new Uint32Array(twin.value.buffer), count = { cells: 0, known: 0, filled: 0, equal: 0, error: 0 };
  for (let b = 0; b < BLOCKS; b++) if (held(b)) for (let l = 0; l < 512; l++) {
    const cell = 512 * b + l, i = 4 * texel(b, l), where = `${label}: block ${b} cell ${l}`;
    count.cells++;
    if (got.values[i + 3] !== twin.mask[cell]) fail(`${where} mask is ${got.values[i + 3]}, expected ${twin.mask[cell]}`);
    for (let c = 0; c < 3; c++) {
      if (!near(got.values[i + c]!, twin.value[3 * cell + c]!)) fail(`${where} component ${c} is ${got.values[i + c]}, expected ${twin.value[3 * cell + c]}`);
      if (twin.mask[cell]! & (1 << c)) { count.known++;count.error = Math.max(count.error, Math.abs(got.values[i + c]! - twin.value[3 * cell + c]!));if (got.bits[i + c] === expected[3 * cell + c]) count.equal++; }
      if (!bounds) continue;
      if ((twin.selection[cell]! >>> (8 * c)) & 255) count.filled++;
      const at = 9 * cell + 3 * c, lo = uniformSourceWord(twin.lower[at]!, twin.lower[at + 1]!, twin.lower[at + 2]!), hi = uniformSourceWord(twin.upper[at]!, twin.upper[at + 1]!, twin.upper[at + 2]!), up = i + 4 * TEXELS;
      if (got.origins[i + c] !== lo.word || ((got.origins[i + 3]! >>> (24 + c)) & 1) !== lo.high) fail(`${where} component ${c} lower bound is ${got.origins[i + c]} high ${got.origins[i + 3]! >>> 24}, expected ${lo.word} high ${lo.high}`);
      if (got.origins[up + c] !== hi.word || ((got.origins[up + 3]! >>> (24 + c)) & 1) !== hi.high) fail(`${where} component ${c} upper bound is ${got.origins[up + c]} high ${got.origins[up + 3]! >>> 24}, expected ${hi.word} high ${hi.high}`);
    }
    if (bounds && (got.origins[i + 3]! & 0xffffff) !== twin.selection[cell]) fail(`${where} selection is ${got.origins[i + 3]! & 0xffffff}, expected ${twin.selection[cell]}`);
  }
  return count;
}
function compareFar(label: string, got: Uint32Array, pool: Parameters<typeof uniformExtrapolationFarValue>[0], root: UniformExtrapolationLevel): { faces: number; nonzero: number; equal: number } {
  const one = new Float32Array(1), bits = new Uint32Array(one.buffer), values = new Float32Array(got.buffer);
  let nonzero = 0, equal = 0;
  for (let i = 0; i < FACES; i++) {
    const anchor = faces.subarray(4 * i, 4 * i + 3), width = faces[4 * i + 3]! >> 2, axis = faces[4 * i + 3]! & 3;
    one[0] = uniformExtrapolationFarValue(pool, root, [0, 1, 2].map(a => a === axis ? anchor[a]! + 1 : anchor[a]! + 0.5 * width) as unknown as Vec3, axis);
    // A face no known cell reaches is zero on both sides, whatever the rounding.
    if (!near(values[i]!, one[0]!) || (one[0] === 0) !== (values[i] === 0)) fail(`${label}: far value of face ${[...anchor]} axis ${axis} width ${width} is ${values[i]}, expected ${one[0]}`);
    if (one[0] !== 0) nonzero++;
    if (bits[0] === got[i]) equal++;
  }
  return { faces: FACES, nonzero, equal };
}
async function round(label: string, seed: number): Promise<void> {
  const pool = { table, records, rows: rows() };
  device.queue.writeBuffer(tail, 0, table);device.queue.writeBuffer(tail, 4 * TABLE, records);device.queue.writeBuffer(tail, 4 * (TABLE + BLOCKS), pool.rows);
  const heldBlocks = order.filter(held);device.queue.writeBuffer(work, 0, Uint32Array.from([1, 1, heldBlocks.length, 0, ...heldBlocks]));
  const first = makeRoot(seed), twin = uniformExtrapolationRun(pool, first, SCALE), got = await launch(first, false);
  const filled = compare(`${label} prolong`, got, twin.up[0]!, true), looked = compareFar(`${label} prolong`, got.far, pool, twin.up[0]!);
  // New values at the same sources: the replay re-reads the taps the prolongation recorded.
  const second = makeRoot(seed);
  { const r = random(seed + 1);for (let i = 0; i < second.value.length; i++) second.value[i] = Math.fround(8 * r() - 4); }
  const down = [second];
  for (let level = 1; level < 4; level++) { down.push(uniformExtrapolationLevel(BLOCKS, 8 >> level));uniformExtrapolationRestrict(pool, down[level - 1]!, down[level]!, SCALE); }
  let top = down[3]!, other = uniformExtrapolationLevel(BLOCKS, 1);
  for (let sweep = 0; sweep < UNIFORM_EXTRAPOLATION_BLOCK_SWEEPS; sweep++) { uniformExtrapolationSweep(pool, top, other, SCALE);[top, other] = [other, top]; }
  const up = [...twin.up];up[3] = top;
  for (let level = 2; level >= 0; level--) uniformExtrapolationReplay(pool, up[level + 1]!, down[level]!, up[level]!);
  const again = await launch(second, true), replayed = compare(`${label} replay`, again, up[0]!, false);
  compareFar(`${label} replay`, again.far, pool, up[0]!);
  console.log(JSON.stringify({ round: label, heldBlocks: filled.cells / 512, cells: filled.cells, knownAfterProlong: filled.known, valuesEqualInBits: filled.equal, largestValueDifference: filled.error, filledByProlong: filled.filled,
    knownAfterReplay: replayed.known, replayValuesEqualInBits: replayed.equal, largestReplayDifference: replayed.error, farFaces: looked.faces, farNonzero: looked.nonzero, farEqualInBits: looked.equal, failures: errors.length }));
  if (filled.known !== replayed.known) fail(`${label}: the replay knows ${replayed.known} values, the prolongation ${filled.known}`);
}
await round("seed", 3);
// Retire a tank page (its block goes free with its texels as they are) and claim an AIR page of the tank in a free block.
{
  const gone = entry(ORIGIN[0], 2 + ORIGIN[1], ORIGIN[2]), claimed = entry(2 + ORIGIN[0], 2 + ORIGIN[1], 1 + ORIGIN[2]), block = order[40]!;
  assert.ok(held(table[gone]!) && table[claimed] === 0 && records[block] === FREE);
  records[table[gone]!] = FREE;table[gone] = 0;table[claimed] = block;records[block] = claimed;
}
await round("after a retire and a claim", 19);
console.log(`extension pyramid: ${errors.length} failure(s), ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(errors.length ? 1 : 0);
