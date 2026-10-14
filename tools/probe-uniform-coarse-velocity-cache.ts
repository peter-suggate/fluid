/** Dawn proof of the 4h velocity cache by block (lib/methods/uniform/
 * uniform-coarse-velocity-cache.ts): UniformMixedMomentumCache's kernel and
 * the readers' shader text against their host twins.
 *
 * An ownership of 4h tiles on a 96 x 64 x 64 box under an open top, its
 * pages in permuted blocks with free ones between and one page of the box
 * AIR. A dense velocity texture and a low-face table hold a value a face
 * (unstored tiles too, so a load the kernel must not make shows). The cache
 * texture starts at a sentinel. After one launch it is read back: every texel
 * of every stored block is the face the host twin names
 * (uniformCoarseCacheSource), and no texel of a block that holds no page was
 * written. The launch is repeated into a texture that starts at zero, as the
 * solver's does, and a second kernel evaluates the readers' text over it:
 * umSampleVelocity4 and umSampleVelocity (uniformMixedVelocitySamplingSource
 * with a cache) at points across the box, on its walls and page faces, in the
 * AIR page and above the open top, against uniformCoarseCacheSample; and
 * umCompiledVelocityTap at h sites of tiles on and off the page faces of
 * every stored block.
 *
 * node --import tsx tools/probe-uniform-coarse-velocity-cache.ts
 * DUMP_WGSL=<dir>: write the reader module there and stop before any device
 * (for naga and Tint).
 * TIME=<launches>: also time that many cache launches on a letters-shaped box
 * of 4h tiles (256 x 192 x 128, 194 blocks), wall clock a launch. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import {
  UNIFORM_COARSE_CACHE_EDGE as EDGE, uniformCoarseCacheBlock, uniformCoarseCacheBytes, uniformCoarseCacheExtent, uniformCoarseCacheGroups, uniformCoarseCacheOrigin, uniformCoarseCacheSample,
  uniformCoarseCacheSource,
} from "../lib/methods/uniform/uniform-coarse-velocity-cache";
import { uniformCompiledVelocityTapWGSL } from "../lib/methods/uniform/uniform-compiled-velocity-taps.wgsl";
import { uniformBlockNeighbourWords, uniformLatticeAddress, type UniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedMomentumCache } from "../lib/methods/uniform/uniform-mixed-momentum-cache";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";
import { uniformMixedVelocitySamplingSource } from "../lib/methods/uniform/uniform-mixed-velocity-sampling.wgsl";
import { UNIFORM_PAGE_POOL_FREE as FREE } from "../lib/methods/uniform/uniform-page-pool";

type Triple = [number, number, number];
const errors: string[] = [], started = performance.now(), f = Math.fround;
const fail = (message: string) => { if (errors.push(message) <= 12) console.error(`FAIL ${message}`); };
const SENTINEL = 7, TOLERANCE = 2e-6;

const lattice = { dimensions: [96, 64, 64] as Triple, cellSize_m: [1, 1, 1] as Triple, origin_m: { x: 0, y: 0, z: 0 } };
const address = uniformLatticeAddress(lattice.dimensions, true, { blocks: 20, permute: 7, air: [[1, 1, 0]] });
const [dx, dy, dz] = lattice.dimensions, table = address.seed.table, records = address.seed.blocks, rows = uniformBlockNeighbourWords(table, records);
const wordAt = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < 64 && y < 64 && z < 64 ? table[x | (y << 6) | (z << 12)]! : 0xffffffff;
const stored = (t: readonly number[]) => { const w = wordAt(t[0]! >> 3, t[1]! >> 3, t[2]! >> 3);return w >= 1 && w < address.blocks - 1; };
const hash = (i: number, k: number) => { let h = Math.imul(i + 977, 0x9e3779b1) ^ Math.imul(k + 1, 0x27d4eb2f);h ^= h >>> 15;h = Math.imul(h, 0x2c1b3c6d);h ^= h >>> 12;return f((h >>> 8) / 2 ** 23 - 1); };
/** The velocity texture: a value a channel of every cell of the box. */
const dense = new Float32Array(4 * dx * dy * dz);
for (let i = 0; i < dx * dy * dz; i++) for (let c = 0; c < 3; c++) dense[4 * i + c] = hash(i, c);
/** The low-face table (umLowFaceIndex): three 32 x 32 planes a block. */
const negative = new Float32Array(address.lowFaceKeys);
for (let i = 0; i < negative.length; i++) negative[i] = hash(i, 5);
/** The kernel's value for the +face along axis of tile s, on 4h tiles. */
function face(s: readonly number[], axis: number): number {
  const anchor = [4 * s[0]!, 4 * s[1]!, 4 * s[2]!];anchor[axis]! += 3;
  const load = () => dense[4 * (anchor[0]! + dx * (anchor[1]! + dy * anchor[2]!)) + axis]!;
  if (stored(s)) return load();
  const above = [...s];above[axis]!++;
  if (!stored(above)) return 0;
  if (anchor[axis]! >= 0) return load();
  const p = [...anchor];p[axis]!++;
  const l = p.map(n => n & 31), uv = axis === 0 ? [l[1]!, l[2]!] : axis === 1 ? [l[0]!, l[2]!] : [l[0]!, l[1]!];
  return negative[((wordAt(p[0]! >> 5, p[1]! >> 5, p[2]! >> 5) * 3 + axis) << 10) | uv[0]! | (uv[1]! << 5)]!;
}

// The readers' text: the sampler with a cache and the compiled h sites, over a script.
const POINTS = 16384, TAPS = 65536;
const readerWGSL = (shape: Parameters<typeof uniformMixedTopologyWGSL>[0]) => `${uniformMixedTopologyWGSL(shape, 0)}
@group(1) @binding(0) var velocity:texture_3d<f32>;
@group(1) @binding(1) var<storage,read> negative:array<f32>;
@group(1) @binding(2) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(3) var coarse:texture_3d<f32>;
@group(1) @binding(4) var<storage,read> script:array<u32>;
@group(1) @binding(5) var<storage,read_write> result:array<f32>;
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(umFaceLow(anchor,axis)){var above=anchor;above[axis]+=1;return negative[umLowFaceIndex(vec3u(above),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
${uniformMixedVelocitySamplingSource(false, "coarseVelocity", "velocity")}
${uniformCompiledVelocityTapWGSL}
@compute @workgroup_size(64) fn points(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=script[0]){return;}
 let p=vec3f(bitcast<f32>(script[4u+4u*i]),bitcast<f32>(script[5u+4u*i]),bitcast<f32>(script[6u+4u*i]));
 let whole=umSampleVelocity(p);
 for(var axis=0u;axis<3u;axis++){result[6u*i+axis]=umSampleVelocity4(p,axis);result[6u*i+3u+axis]=whole[axis];}
}
@compute @workgroup_size(64) fn taps(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=script[1]){return;}
 let a=4u+4u*${POINTS}u+2u*i;let w=script[a+1u];
 result[6u*${POINTS}u+i]=umCompiledVelocityTap(script[a],vec3u(w&3u,(w>>2u)&3u,(w>>4u)&3u),(w>>6u)&3u,(w>>8u)!=0u);
}`;
if (process.env.DUMP_WGSL) {
  const out = process.env.DUMP_WGSL;mkdirSync(out, { recursive: true });
  writeFileSync(`${out}/000.wgsl`, `// label: 4h cache probe readers\n${readerWGSL({ lattice, tiles: address.tileKeys })}`);
  console.log(`wrote 1 module to ${out}`);process.exit(0);
}

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();assert.ok(adapter);
const device = managedGPUDevice(await adapter.requestDevice(), { requireWorkerRealm: false });
device.addEventListener("uncapturederror", e => { e.preventDefault();fail(`device: ${e.error.message}`); });

/** A cache texture's texels, four floats each, rows unpadded. */
async function readCache(texture: GPUTexture, size: Triple): Promise<Float32Array> {
  const row = Math.ceil(16 * size[0] / 256) * 256, bytes = row * size[1] * size[2];
  const target = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer: target, bytesPerRow: row, rowsPerImage: size[1] }, size);device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const padded = new Float32Array(target.getMappedRange().slice(0)), out = new Float32Array(4 * size[0] * size[1] * size[2]);target.unmap();target.destroy();
  for (let r = 0; r < size[1] * size[2]; r++) out.set(padded.subarray(r * row / 4, r * row / 4 + 4 * size[0]), 4 * size[0] * r);
  return out;
}
function cacheTexture(a: UniformLatticeAddress, label: string): { texture: GPUTexture; size: Triple } {
  const size = uniformCoarseCacheExtent(a);
  return { size, texture: device.createTexture({ label, size, dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST }) };
}

{
  const layout = createUniformMixedLayout(lattice, [], 4, undefined, address), ownership = new UniformMixedOwnership(device, layout, false, 0);
  const extended = device.createTexture({ label: "probe velocity", size: lattice.dimensions, dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: extended }, dense, { bytesPerRow: 16 * dx, rowsPerImage: dy }, lattice.dimensions);
  const low = device.createBuffer({ label: "probe low faces", size: negative.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });device.queue.writeBuffer(low, 0, negative);
  const { texture, size } = cacheTexture(address, "probe 4h cache");
  device.queue.writeTexture({ texture }, new Float32Array(4 * size[0] * size[1] * size[2]).fill(SENTINEL), { bytesPerRow: 16 * size[0], rowsPerImage: size[1] }, size);
  const cache = new UniformMixedMomentumCache(device, ownership);
  assert.throws(() => cache.bind({ extended, negative: low, coarseExtended: extended }), /10 texels a block/);
  await cache.initialize();
  const built = Math.round(performance.now() - started);
  const encoder = device.createCommandEncoder();cache.encode(encoder, cache.bind({ extended, negative: low, coarseExtended: texture }));device.queue.submit([encoder.finish()]);
  const marked = await readCache(texture, size), at = (x: number, y: number, z: number) => 4 * (x + size[0] * (y + size[1] * z));

  // The kernel against the twin: every texel of the texture.
  const origin: Triple = [0, 0, 0], s: Triple = [0, 0, 0];
  let storedBlocks = 0, written = 0, untouched = 0, halo = 0, nonzero = 0;
  for (let z = 0; z < size[2]; z++) for (let y = 0; y < size[1]; y++) for (let x = 0; x < size[0]; x++) {
    const b = uniformCoarseCacheBlock(address.blockBits, x, y, z), r = b < address.blocks ? records[b]! : FREE, i = at(x, y, z);
    if (b < 1 || b >= address.blocks - 1 || r === FREE) {
      for (let c = 0; c < 4; c++) if (marked[i + c] !== SENTINEL) fail(`texel ${x},${y},${z} of block ${b}, which holds no page, was written: ${marked[i + c]}`);
      untouched++;continue;
    }
    const page = [r & 63, (r >>> 6) & 63, (r >>> 12) & 63], l = [x % EDGE - 1, y % EDGE - 1, z % EDGE - 1], tile = l.map((n, a) => 8 * page[a]! + n);
    for (let axis = 0; axis < 3; axis++) {
      const want = face(uniformCoarseCacheSource(rows, b, page, tile, axis, s), axis);
      if (marked[i + axis] !== want) fail(`block ${b} page ${page.join(",")} local ${l.join(",")} axis ${axis}: ${marked[i + axis]}, expected ${want} (the face of tile ${s.join(",")})`);
      if (want !== 0) nonzero++;
    }
    if (marked[i + 3] !== 0) fail(`block ${b} local ${l.join(",")}: w is ${marked[i + 3]}`);
    written++;if (l.some(n => n < 0 || n > 7)) halo++;
  }
  for (let b = 1; b < address.blocks - 1; b++) if (records[b] !== FREE) storedBlocks++;
  if (written !== 1000 * storedBlocks) fail(`${written} texels of stored blocks, expected ${1000 * storedBlocks}`);
  console.log(JSON.stringify({ kernel: lattice.dimensions.join("x"), blocks: address.blocks, stored: storedBlocks, texture: size.join("x"), bytes: uniformCoarseCacheBytes(address), groups: uniformCoarseCacheGroups(address).join("x"), written, halo, nonzero, untouched, buildMs: built, errors: errors.length }));

  // The solver's cache starts at zero and stays zero wherever no block holds a page.
  texture.destroy();
  const { texture: zeroed } = cacheTexture(address, "probe 4h cache from zero");
  { const encoder = device.createCommandEncoder();cache.encode(encoder, cache.bind({ extended, negative: low, coarseExtended: zeroed }));device.queue.submit([encoder.finish()]); }
  const got = await readCache(zeroed, size);
  for (let i = 0; i < got.length; i++) if (got[i] !== (marked[i] === SENTINEL ? 0 : marked[i])) { fail(`float ${i} of the cache from zero is ${got[i]}, and ${marked[i]} over the sentinel`);break; }

  // The readers against the twin, over the cache the kernel wrote.
  const script = new Uint32Array(4 + 4 * POINTS + 2 * TAPS), floats = new Float32Array(script.buffer), points: Triple[] = [], taps: { tile: number; local: Triple; axis: number; plane: boolean }[] = [];
  let seed = 4242;
  const random = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  for (let n = 0; n < 8000; n++) points.push([f(random() * dx), f(random() * dy), f(random() * dz)]);
  // On the walls and page faces, within a tile of them, and above the open top.
  const edge = (d: number) => [0, 0.25, 1.5, 2, 3.5, 30.5, 31.75, 32, 32.5, 34, d - 3.5, d - 2, d - 0.25, d];
  for (const x of edge(dx)) for (const y of [...edge(dy), dy + 1, dy + 3.75, dy + 9]) for (const z of edge(dz)) points.push([x, y, z]);
  // In the AIR page of the box.
  for (let n = 0; n < 400; n++) points.push([f(32 + random() * 32), f(32 + random() * 32), f(random() * 32)]);
  assert.ok(points.length <= POINTS);
  for (const [i, p] of points.entries()) floats.set(p, 4 + 4 * i);
  for (let b = 1; b < address.blocks - 1; b++) {
    if (records[b] === FREE) continue;
    for (const tz of [0, 4, 7]) for (const ty of [0, 3, 7]) for (const tx of [0, 5, 7]) for (let axis = 0; axis < 3; axis++) {
      const tile = 512 * b + tx + 8 * ty + 64 * tz;
      for (let c = 0; c < 64; c++) taps.push({ tile, local: [c & 3, (c >> 2) & 3, c >> 4], axis, plane: false });
      // A negative plane is asked only of a tile on its page's low plane.
      if ([tx, ty, tz][axis] === 0) for (let c = 0; c < 16; c++) { const local: Triple = [0, 0, 0];local[(axis + 1) % 3] = c & 3;local[(axis + 2) % 3] = c >> 2;taps.push({ tile, local, axis, plane: true }); }
    }
  }
  assert.ok(taps.length <= TAPS);
  for (const [i, t] of taps.entries()) { script[4 + 4 * POINTS + 2 * i] = t.tile;script[5 + 4 * POINTS + 2 * i] = t.local[0] | (t.local[1] << 2) | (t.local[2] << 4) | (t.axis << 6) | (Number(t.plane) << 8); }
  script[0] = points.length;script[1] = taps.length;
  const scriptBuffer = device.createBuffer({ size: script.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });device.queue.writeBuffer(scriptBuffer, 0, script);
  const resultBuffer = device.createBuffer({ size: 4 * (6 * POINTS + TAPS), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const sampled: GPUBindGroupLayoutEntry["texture"] = { sampleType: "unfilterable-float", viewDimension: "3d" };
  const readerLayout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: sampled }, { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: sampled }, { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: sampled },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const module = device.createShaderModule({ label: "4h cache probe readers", code: readerWGSL(ownership.capacity) });
  const messages = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
  if (messages.length) throw new Error(messages.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [ownership.bindLayout, readerLayout] });
  const [pointsPipeline, tapsPipeline] = await Promise.all(["points", "taps"].map(entryPoint => device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint } })));
  const group = device.createBindGroup({ layout: readerLayout, entries: [
    { binding: 0, resource: extended.createView() }, { binding: 1, resource: { buffer: low } }, { binding: 2, resource: zeroed.createView() }, { binding: 3, resource: zeroed.createView() },
    { binding: 4, resource: { buffer: scriptBuffer } }, { binding: 5, resource: { buffer: resultBuffer } },
  ] });
  {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setBindGroup(0, ownership.bindGroup);pass.setBindGroup(1, group);
    pass.setPipeline(pointsPipeline!);pass.dispatchWorkgroups(POINTS / 64);pass.setPipeline(tapsPipeline!);pass.dispatchWorkgroups(TAPS / 64);pass.end();
    const target = device.createBuffer({ size: resultBuffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    encoder.copyBufferToBuffer(resultBuffer, 0, target, 0, resultBuffer.size);device.queue.submit([encoder.finish()]);
    await target.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(target.getMappedRange().slice(0));target.unmap();target.destroy();
    const load = (x: number, y: number, z: number, axis: number) => got[at(x, y, z) + axis]!;
    let unstoredHome = 0, above = 0, pointNonzero = 0, worst = 0;
    for (const [i, p] of points.entries()) {
      // umSampleTile, umCoarseHome: the tile of p held to the box, its page and that page's block.
      const tile = p.map((n, a) => Math.min(Math.max(Math.floor(Math.min(Math.max(n, 0), lattice.dimensions[a]!) / 4), 0), address.tiles[a]! - 1)), page = tile.map(n => n >> 3);
      const block = Math.min(wordAt(page[0]!, page[1]!, page[2]!), address.blocks - 1);uniformCoarseCacheOrigin(address.blockBits, block, origin);
      if (block === 0 || block === address.blocks - 1) unstoredHome++;
      if (p[1] > dy) above++;
      for (let axis = 0; axis < 3; axis++) {
        const want = uniformCoarseCacheSample(load, origin, page, p, axis), four = result[6 * i + axis]!, whole = result[6 * i + 3 + axis]!;
        worst = Math.max(worst, Math.abs(four - want));if (want !== 0) pointNonzero++;
        if (!(Math.abs(four - want) <= TOLERANCE)) fail(`umSampleVelocity4 at ${p.join(",")} axis ${axis}: ${four}, expected ${want}`);
        if (!(Math.abs(whole - want) <= TOLERANCE)) fail(`umSampleVelocity at ${p.join(",")} axis ${axis}: ${whole}, expected ${want}`);
        if ((block === 0 || block === address.blocks - 1) && four !== 0) fail(`a sample at ${p.join(",")}, whose page is not stored, is ${four}`);
      }
    }
    let planes = 0, tapNonzero = 0, tapWorst = 0;
    for (const [i, t] of taps.entries()) {
      // umCompiledVelocityTap: eighths from the page's low corner, the block from the key.
      const b = t.tile >> 9, k = t.tile & 511, within = [k & 7, (k >> 3) & 7, k >> 6], q = within.map((n, a) => 8 * n + 2 * t.local[a]! - 3);q[t.axis]! -= 3;
      if (t.plane) { q[t.axis] = -8;planes++; }
      const base = q.map(n => n >> 3), fraction = q.map((n, a) => f((n - 8 * base[a]!) / 8)), terms: number[] = [];uniformCoarseCacheOrigin(address.blockBits, b, origin);
      for (let c = 0; c < 8; c++) {
        const bit = [c & 1, (c >> 1) & 1, c >> 2], weight = f(f((bit[0] ? fraction[0]! : f(1 - fraction[0]!)) * (bit[1] ? fraction[1]! : f(1 - fraction[1]!))) * (bit[2] ? fraction[2]! : f(1 - fraction[2]!)));
        terms.push(weight > 0 ? f(weight * load(origin[0] + base[0]! + bit[0]!, origin[1] + base[1]! + bit[1]!, origin[2] + base[2]! + bit[2]!, t.axis)) : 0);
      }
      const want = f(f(f(terms[0]! + terms[5]!) + f(terms[1]! + terms[4]!)) + f(f(terms[2]! + terms[7]!) + f(terms[3]! + terms[6]!))), value = result[6 * POINTS + i]!;
      tapWorst = Math.max(tapWorst, Math.abs(value - want));if (want !== 0) tapNonzero++;
      if (!(Math.abs(value - want) <= TOLERANCE)) fail(`umCompiledVelocityTap of tile ${t.tile} local ${t.local.join(",")} axis ${t.axis}${t.plane ? " on its low plane" : ""}: ${value}, expected ${want}`);
    }
    if (!unstoredHome || !above || pointNonzero < points.length || tapNonzero < taps.length / 2) fail(`the points do not cover the cases: ${unstoredHome} in an unstored page, ${above} above the top, ${pointNonzero} nonzero`);
    console.log(JSON.stringify({ readers: "sampler and compiled taps", points: points.length, unstoredHome, aboveTop: above, nonzero: pointNonzero, worst, taps: taps.length, lowPlaneTaps: planes, tapsNonzero: tapNonzero, tapWorst, errors: errors.length }));
  }
  extended.destroy();low.destroy();zeroed.destroy();scriptBuffer.destroy();resultBuffer.destroy();
}

// Wall clock of the cache launch on a letters-shaped box of 4h tiles.
if (process.env.TIME) {
  const launches = Number(process.env.TIME), box = { dimensions: [256, 192, 128] as Triple, cellSize_m: [1, 1, 1] as Triple, origin_m: { x: 0, y: 0, z: 0 } }, a = uniformLatticeAddress(box.dimensions);
  const ownership = new UniformMixedOwnership(device, createUniformMixedLayout(box, [], 4, undefined, a), false, 0);
  const extended = device.createTexture({ size: box.dimensions, dimension: "3d", format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const low = device.createBuffer({ size: 4 * a.lowFaceKeys, usage: GPUBufferUsage.STORAGE }), { texture, size } = cacheTexture(a, "timed 4h cache");
  const cache = new UniformMixedMomentumCache(device, ownership);await cache.initialize();
  const group = cache.bind({ extended, negative: low, coarseExtended: texture }), times: number[] = [];
  for (let run = 0; run < 4; run++) {
    const encoder = device.createCommandEncoder();
    for (let n = 0; n < launches; n++) cache.encode(encoder, group);
    await device.queue.onSubmittedWorkDone();
    const from = performance.now();device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();
    times.push((performance.now() - from) / launches);
  }
  console.log(JSON.stringify({ timed: box.dimensions.join("x"), blocks: a.blocks, texture: size.join("x"), groups: uniformCoarseCacheGroups(a).join("x"), launches, msPerLaunch: times.map(t => +t.toFixed(3)) }));
  extended.destroy();low.destroy();texture.destroy();
}
await device.queue.onSubmittedWorkDone();
if (errors.length) { console.error(`uniform coarse velocity cache: ${errors.length} failure(s)`);process.exit(1); }
device.destroy();
console.log(`uniform coarse velocity cache: ok (${Math.round(performance.now() - started)} ms)`);
