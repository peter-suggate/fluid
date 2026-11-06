import assert from "node:assert/strict";
import test from "node:test";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { narrowBandSeedRankShape, narrowBandSeedRankWGSL, type NarrowBandSeedRank } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { NB_SLAB_SEEDS_WORD } from "../lib/methods/uniform/uniform-particle-slab.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

// The seed budget admits overlap reseeds by rank, not by the order the seed
// list was filled in: a shuffled list admits the same sites, the first
// `room` of them by (rank key, site index).
const RANK = narrowBandSeedRankShape(32768, 512);
const BUDGET = 100000;

// seedCandidate's stand-in: an entry's site j is a candidate where bit j of
// its mask is set (masks[slot]); a source entry (bit 31) is not an overlap candidate.
const harness = (rank: NarrowBandSeedRank) => /* wgsl */`
struct Particle{position:vec4f}
struct Params{live:vec4u}
const NB_BAND:u32=0u;const NB_BAND_SEAM:u32=8u;const NB_NO_BIN:u32=0xffffffffu;
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> links:array<u32>;
@group(0) @binding(3) var<storage,read> particles:array<vec4f>;
@group(0) @binding(4) var<uniform> params:Params;
@group(0) @binding(5) var<storage,read> masks:array<u32>;
@group(0) @binding(6) var<storage,read_write> admitted:array<u32>;
@group(0) @binding(7) var<storage,read> ids:array<u32>;
fn nbSeedList()->u32{return 0u;}
fn seedCandidate(group:vec3u,lane:u32,sourcePass:bool,counting:bool)->Particle{
 var p=Particle();p.position.x=-1.0;
 let slot=8u*group.x+lane/8u;if(slot>=atomicLoad(&bins[NB_BAND+3u])){return p;}
 if(((links[slot]&0x80000000u)!=0u)!=sourcePass){return p;}
 if(((masks[slot]>>(lane%8u))&1u)!=0u){p.position.x=1.0;}
 return p;
}
${narrowBandSeedRankWGSL(rank, BUDGET)}
// seed's admission, filed by the entry's id.
@compute @workgroup_size(64) fn admit(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let p=seedCandidate(group,lane,false,true);
 nbSeedKept[lane]=select(0u,1u,p.position.x>=0.0);workgroupBarrier();
 let slot=8u*group.x+lane/8u;
 if(p.position.x>=0.0&&nbSeedAdmitted(nbSeedRankKey(group,lane),lane)){admitted[8u*ids[slot]+lane%8u]=1u;}
}`;

interface Entry { id: number; word: number; mask: number }

async function admit(device: GPUDevice, list: readonly Entry[], used: { records: number; sources: number }, frame: number): Promise<Uint32Array> {
  const manager = gpuCompilationManagerFor(device);
  const storage = (type: GPUBufferBindingType): GPUBindGroupLayoutEntry["buffer"] => ({ type });
  const layout = manager.createBindGroupLayout({ entries: [
    ...(["storage", "storage", "read-only-storage", "read-only-storage", "uniform", "read-only-storage", "storage", "read-only-storage"] as const)
      .map((type, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: storage(type) })),
  ] });
  const module = manager.createShaderModule({ code: harness(RANK) });
  const pipelineLayout = manager.createPipelineLayout({ bindGroupLayouts: [layout] });
  const compile = (entryPoint: string, level?: number) => manager.compileComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint, ...(level === undefined ? {} : { constants: { nbSeedLevel: level } }) } });
  const steps = await Promise.all([...Array.from({ length: RANK.levels }, (_, l) => [compile("seedRank", l), compile("seedCut", l)]).flat(), compile("admit")]);
  const n = list.length, words = (w: ArrayLike<number>) => Uint32Array.from(w);
  const make = (data: Uint32Array<ArrayBuffer>, usage: number) => { const b = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data); return b; };
  const bins = make(new Uint32Array(8 + RANK.levels * (1 << RANK.bits) + 3).fill(0).map((v, i) => i === 3 ? n : v), GPUBufferUsage.STORAGE);
  const state = new Uint32Array(32);state[1] = used.records;state[NB_SLAB_SEEDS_WORD] = used.sources;
  const buffers = [bins, make(state, GPUBufferUsage.STORAGE), make(words(list.map(e => e.word)), GPUBufferUsage.STORAGE), make(new Uint32Array(4 * n), GPUBufferUsage.STORAGE),
    make(words([0, 0, frame, 0]), GPUBufferUsage.UNIFORM), make(words(list.map(e => e.mask)), GPUBufferUsage.STORAGE),
    make(new Uint32Array(8 * n), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC), make(words(list.map(e => e.id)), GPUBufferUsage.STORAGE)];
  const read = device.createBuffer({ size: 32 * n, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const group = device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setBindGroup(0, group);
    steps.forEach((pipeline, i) => { pass.setPipeline(pipeline); pass.dispatchWorkgroups(i % 2 === 1 && i < 2 * RANK.levels ? 1 : Math.ceil(n / 8)); });
    pass.end();encoder.copyBufferToBuffer(buffers[6]!, 0, read, 0, 32 * n);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);return new Uint32Array(read.getMappedRange()).slice();
  } finally { if (read.mapState === "mapped") read.unmap(); for (const b of buffers) b.destroy(); read.destroy(); }
}

// nbSeedRankKey on the CPU.
function rankKey(word: number, frame: number): number {
  let h = Math.imul(frame, 0x9e3779b9) >>> 0;h = (h ^ (h >>> 16)) >>> 0;h = Math.imul(h, 0x7feb352d) >>> 0;h = (h ^ (h >>> 15)) >>> 0;
  return ((word & 0x3fffffff) ^ h) & (2 ** RANK.keyBits - 1);
}

test("seed rank: a 32768-cell lattice in 512 tile keys takes three 7-bit digits", () => {
  assert.deepEqual(narrowBandSeedRankShape(32768, 512), { bits: 7, levels: 3, keyBits: 15 });
  assert.deepEqual(narrowBandSeedRankShape(32768 * 64, 512 * 64), { bits: 10, levels: 3, keyBits: 21 });
  assert.throws(() => narrowBandSeedRankShape(32768, 4), /cannot hold/);
});

test("the seed budget admits the same overlap reseeds in any seed-list order", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("seed rank", async device => {
    // 400 distinct cells; one in five a source cell; random site masks.
    let s = 2024;const rnd = () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0);
    const cells = new Set<number>();while (cells.size < 400) cells.add(rnd() % 32768);
    const list: Entry[] = [...cells].map((cell, id) => ({ id, word: (cell | (id % 5 === 0 ? 0x80000000 : 0)) >>> 0, mask: rnd() & 0xff }));
    const shuffled = [...list];for (let i = shuffled.length - 1; i > 0; i--) { const j = rnd() % (i + 1); [shuffled[i], shuffled[j]] = [shuffled[j]!, shuffled[i]!]; }
    const candidates = (frame: number) => list.filter(e => (e.word & 0x80000000) === 0).flatMap(e => [...Array(8).keys()].filter(j => (e.mask >> j) & 1).map(j => ({ key: rankKey(e.word, frame), j, at: 8 * e.id + j })))
      .sort((a, b) => a.key - b.key || a.j - b.j);
    const total = candidates(7).length;
    const sites = (a: Uint32Array) => [...a.keys()].filter(i => a[i] === 1);
    for (const [frame, room] of [[7, Math.floor(0.4 * total)], [7, 1], [8, Math.floor(0.4 * total)]] as const) {
      const used = { records: BUDGET - room - 50, sources: 50 };
      const a = await admit(device, list, used, frame), b = await admit(device, shuffled, used, frame);
      assert.deepEqual(sites(b), sites(a), `frame ${frame} room ${room}: the same sites in either order`);
      assert.deepEqual(sites(a), candidates(frame).slice(0, room).map(c => c.at).sort((x, y) => x - y), `frame ${frame} room ${room}: the first ${room} of ${total} by rank`);
    }
    // No room: none; room for all: every candidate.
    assert.equal(sites(await admit(device, shuffled, { records: BUDGET, sources: 3 }, 7)).length, 0);
    assert.equal(sites(await admit(device, shuffled, { records: 0, sources: 0 }, 7)).length, total);
    // Another frame defers other cells.
    const deferred = (frame: number) => new Set(candidates(frame).slice(Math.floor(0.4 * total)).map(c => c.at));
    const d7 = deferred(7), d8 = deferred(8);
    const moved = [...d7].filter(x => !d8.has(x)).length;
    assert.ok(moved > 0.05 * d7.size, `the frame mask moves the cut (${moved} of ${d7.size} deferred sites change)`);
  });
});
