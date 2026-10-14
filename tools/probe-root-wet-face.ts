/** Holds the pressure root's cut-tile kernels to their host emulation
 * (tools/root-wet-face-reference.ts) on fixtures, three pipelines and no
 * solver: the cooperative cut fit and its ordered reference (phi and the
 * sides word of the cut slot), and umWetFace (V and mask of a 4h face).
 * The kernels are the production text, sliced from the authority's source;
 * the fixture stands in for the solid record and the vertex phi.
 *   node --import tsx tools/probe-root-wet-face.ts */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformMixedPressureCutWGSL } from "../lib/methods/uniform/uniform-mixed-pressure-cut.wgsl";
import { cutFit, cutLive, cutSide, wetFace, type OpenCells, type WetTile } from "./root-wet-face-reference";

const f = Math.fround;
const cells = (open: (x: number, y: number, z: number) => boolean): [number, number] => {
  const bits: [number, number] = [0, 0];
  for (let i = 0; i < 64; i++) if (open(i % 4, (i >> 2) % 4, i >> 4)) bits[i >>> 5] = (bits[i >>> 5]! | (1 << (i & 31))) >>> 0;
  return bits;
};
// Single cells, single voxels, empty, full, walls and floors through the tile, and scattered cells on several sides.
const MASKS: [number, number][] = [];
for (let c = 0; c < 64; c++) MASKS.push(cells((x, y, z) => x + 4 * y + 16 * z === c));
for (let c = 0; c < 64; c++) MASKS.push(cells((x, y, z) => x + 4 * y + 16 * z !== c));
MASKS.push([0, 0], [0xffffffff, 0xffffffff]);
for (let k = 0; k < 4; k++) MASKS.push(cells(x => x !== k), cells((_x, y) => y !== k), cells((_x, _y, z) => z !== k), cells((_x, y) => y >= k), cells((x, y) => x !== k && y >= 2));
while (MASKS.length < 256) { const m = MASKS.length; MASKS.push(cells((x, y, z) => ((x + 4 * y + 16 * z) * 73 + m * 19) % (m % 13 + 2) === 0)); }
const MODES = 8, FIT_TILES = MASKS.length * MODES, PAIRS = 4096;

/** The fit's vertex phi, as fixturePhi sums it. */
function fitPhi(t: number, v: number): number {
  const q = [v % 5 - 2, Math.floor(v / 5) % 5 - 2, Math.floor(v / 25) - 2], mode = Math.floor(t / MASKS.length);
  let level = f((t % 7 - 3) / 16);
  if (mode === 1) level = 0; if (mode === 2) level = f(-0.004); if (mode === 3) level = f(1e-7);
  const plane = f(f(f(level + f(0.125 * q[0]!)) - f(0.25 * q[1]!)) + f(0.5 * q[2]!));
  if (mode < 4) return plane;
  if (mode === 4) return f(plane + f(0.0625 * (q[0]! * q[1]! + q[2]! * q[2]!)));
  if (mode === 5) return ((v * 173 + t * 59) % 1024 - 512) / 128;
  if (mode === 6) return f(f(1e-4) * plane);
  return f(16 + plane);
}
/** The face's vertex phi, as facePhi hashes it: every value a multiple of 1/64, so a corner sum is exact. */
function facePhi(t: number, p: readonly number[]): number {
  const mode = (t >>> 1) % 4;
  if (mode === 1) return 2; if (mode === 2) return -2;
  if (mode === 3) return p[1]! - 2 - ((t >>> 1) % 29) / 16;
  const h = ((Math.imul(t, 0x9e3779b1) + p[0]! * 40503 + p[1]! * 69069 + p[2]! * 12345) >>> 0) >>> 7;
  return (h % 65 - 32) / 64;
}
const hash = (i: number, salt: number) => ((Math.imul(i + 1, 0x85ebca6b) ^ Math.imul(salt + 1, 0xc2b2ae35)) >>> 0) >>> 9;

const source = readFileSync(resolve("lib/methods/uniform/uniform-mixed-pressure-authority.ts"), "utf8").replace("${uniformMixedPressureCutWGSL}", uniformMixedPressureCutWGSL);
const slice = (from: string, to: string) => { const begin = source.indexOf(from), end = source.indexOf(to, begin); assert.ok(begin >= 0 && end > begin, `${from} .. ${to}`); return source.slice(begin, end); };
const kernels = `${slice("// Cooperative cut fit;", "fn umCutPhi(")}\n${slice("fn umWetOpen(", "fn umWetTile(")}`
  .replaceAll("textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x", "fixturePhi(t,v)")
  .replaceAll("textureLoad(vertexPhi,origin+vec3i(g),0).x", "fixturePhi(t,g.x+5u*g.y+25u*g.z)")
  .replaceAll("textureLoad(vertexPhi,p,0).x", "facePhi(t,p)").replaceAll("umTopology[t]&0x3fffffffu", "t");
assert.ok(!kernels.includes("textureLoad") && !kernels.includes("umTopology") && !kernels.includes("${"));
const code = /* wgsl */`
const UM_TILES=${FIT_TILES}u;const UM_CUT_BASE=0u;const UM_HMIN=1.0;const pcHeld=false;
// (open.x, open.y, flags: 1 cut, 2 held at h, 4 a row).
@group(0) @binding(0) var<storage,read> tiles:array<vec4u>;
@group(0) @binding(1) var<storage,read_write> balance:array<vec2f>;
@group(0) @binding(2) var<storage,read_write> faces:array<vec2f>;
fn umCorner(i:u32,n:u32)->vec3u{return vec3u(i%n,(i/n)%n,i/(n*n));}
fn umTileCoord(t:u32)->vec3u{return vec3u(0u);}
fn umTileLive(t:u32)->bool{return true;}
fn umSolidOpenCells(t:u32)->vec2u{return tiles[t].xy;}
fn umSolidCut(t:u32)->bool{return (tiles[t].z&1u)!=0u;}
fn umSolidFineTile(t:u32)->bool{return (tiles[t].z&2u)!=0u;}
fn umWetRow(t:u32)->bool{return (tiles[t].z&4u)!=0u;}
fn umPatchFaceV(t:u32,id:vec3i,axis:u32)->f32{
 var l=vec3u(id);let low=umWetOpen(tiles[t].xy,l);l[axis]=0u;let high=umWetOpen(tiles[t+1u].xy,l);
 return 0.5*(f32(low)+f32(high));
}
fn fixturePhi(t:u32,v:u32)->f32{
 let q=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0;let mode=t/${MASKS.length}u;
 var level=f32(i32(t%7u)-3)/16.0;
 if(mode==1u){level=0.0;}if(mode==2u){level=-0.004;}if(mode==3u){level=1e-7;}
 let plane=level+0.125*q.x-0.25*q.y+0.5*q.z;
 if(mode<4u){return plane;}
 if(mode==4u){return plane+0.0625*(q.x*q.y+q.z*q.z);}
 if(mode==5u){return f32(i32((v*173u+t*59u)%1024u)-512)/128.0;}
 if(mode==6u){return 1e-4*plane;}
 return 16.0+plane;
}
fn facePhi(t:u32,p:vec3i)->f32{
 let mode=(t>>1u)%4u;
 if(mode==1u){return 2.0;}if(mode==2u){return -2.0;}
 if(mode==3u){return f32(p.y)-2.0-f32((t>>1u)%29u)/16.0;}
 let h=(t*0x9e3779b1u+u32(p.x)*40503u+u32(p.y)*69069u+u32(p.z)*12345u)>>7u;
 return f32(i32(h%65u)-32)/64.0;
}
${kernels}
@compute @workgroup_size(64) fn reference(@builtin(global_invocation_id) id:vec3u){if(id.x<UM_TILES){cutTile(id.x);}}
@compute @workgroup_size(64) fn face(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${PAIRS}u){return;}
 for(var axis=0u;axis<3u;axis++){faces[3u*id.x+axis]=umWetFace(2u*id.x,2u*id.x+1u,vec3i(0),axis);}
}
`;

async function run() {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice(), errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const module = device.createShaderModule({ label: "Root wet face probe", code });
  assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => `${m.lineNum}:${m.linePos} ${m.message}`), []);
  const layout = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
  /** One entry point over `tiles` and `balance`: the balance and face words it leaves. */
  const dispatch = async (entryPoint: string, tiles: Uint32Array<ArrayBuffer>, balance: Float32Array<ArrayBuffer>, groups: number) => {
    const buffers = [tiles, balance, new Float32Array(6 * PAIRS)].map((data, i) => {
      const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (i > 0 ? GPUBufferUsage.COPY_SRC : 0) });
      device.queue.writeBuffer(buffer, 0, data); return buffer;
    });
    const pipeline = await device.createComputePipelineAsync({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint } });
    const group = device.createBindGroup({ layout, entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const reads = [1, 2].map(i => device.createBuffer({ size: buffers[i]!.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    const e = device.createCommandEncoder(), p = e.beginComputePass(); p.setPipeline(pipeline); p.setBindGroup(0, group); p.dispatchWorkgroups(groups); p.end();
    reads.forEach((read, i) => e.copyBufferToBuffer(buffers[i + 1]!, 0, read, 0, read.size)); device.queue.submit([e.finish()]);
    const out: Float32Array[] = [];
    for (const read of reads) { await read.mapAsync(GPUMapMode.READ); out.push(new Float32Array(read.getMappedRange().slice(0))); read.unmap(); read.destroy(); }
    for (const buffer of buffers) buffer.destroy();
    return { balance: out[0]!, faces: out[1]! };
  };
  try {
    // The fit: every tile cut, the cooperative entry and the ordered reference against the emulation.
    const fitTiles = new Uint32Array(4 * FIT_TILES);
    for (let t = 0; t < FIT_TILES; t++) fitTiles.set([...MASKS[t % MASKS.length]!, 3, 0], 4 * t);
    const fits = { cooperative: await dispatch("cut", fitTiles, new Float32Array(2 * FIT_TILES), Math.ceil(FIT_TILES / 64)), reference: await dispatch("reference", fitTiles, new Float32Array(2 * FIT_TILES), Math.ceil(FIT_TILES / 64)) };
    for (const [name, got] of Object.entries(fits)) {
      let ties = 0, worst = 0, sided = [0, 0, 0, 0];
      for (let t = 0; t < FIT_TILES; t++) {
        const open = MASKS[t % MASKS.length]!, vertex = (v: number) => fitPhi(t, v), want = cutFit(open, vertex, 1), phi = got.balance[2 * t]!, sides = got.balance[2 * t + 1]!;
        // Liquid on neither side and the two means within rounding of each other: either side is the nearer.
        const side = cutSide(open); let wet = false, sum = 0, sumRest = 0, n = 0, nRest = 0, scale = 1;
        for (let v = 0; v < 125; v++) if (cutLive(open, v)) { const value = vertex(v); wet ||= value < 0; scale = Math.max(scale, Math.abs(value)); if (cutLive(side, v)) { sum += value; n++; } else { sumRest += value; nRest++; } }
        if (!wet && nRest > 0 && Math.abs(sum * nRest - sumRest * n) <= 1e-4 * scale * n * nRest) { ties++; assert.ok(sides === 1 || sides === 2, `${name} tile ${t}: a tie is through one side, not ${sides}`); continue; }
        assert.equal(sides, want.sides, `${name} tile ${t} (mask ${t % MASKS.length}, mode ${Math.floor(t / MASKS.length)}): sides`);
        assert.ok(Number.isFinite(phi), `${name} tile ${t}: phi ${phi}`);
        const error = Math.abs(phi - want.phi) / scale; worst = Math.max(worst, error); sided[sides]!++;
        assert.ok(error <= 2e-3, `${name} tile ${t} (mask ${t % MASKS.length}, mode ${Math.floor(t / MASKS.length)}): phi ${phi}, emulation ${want.phi}`);
      }
      console.log(JSON.stringify({ kernel: `cut fit, ${name}`, tiles: FIT_TILES, ties, sides: { none: sided[0], grown: sided[1], rest: sided[2], both: sided[3] }, worstPhiError: worst }));
    }
    // The faces: pairs of tiles under every flag, the sides word as the fit would leave it.
    const pairTiles = new Uint32Array(8 * PAIRS), pairBalance = new Float32Array(4 * PAIRS), record: (WetTile & { open: OpenCells })[] = [];
    for (let t = 0; t < 2 * PAIRS; t++) {
      const open = MASKS[hash(t, 0) % 3 === 0 ? 129 : hash(t, 1) % MASKS.length]!, empty = open[0] === 0 && open[1] === 0;
      const cut = hash(t, 2) % 4 !== 0, fine = cut || hash(t, 3) % 2 === 0, row = hash(t, 4) % 2 === 0, sides = !cut || empty ? 0 : 1 + hash(t, 5) % 3;
      pairTiles.set([...open, (cut ? 1 : 0) | (fine ? 2 : 0) | (row ? 4 : 0), 0], 4 * t); pairBalance[2 * t + 1] = sides; record.push({ open, cut, fine, row, sides });
    }
    const got = await dispatch("face", pairTiles, pairBalance, Math.ceil(PAIRS / 64));
    let kept = 0, surface = 0;
    for (let pair = 0; pair < PAIRS; pair++) for (let axis = 0; axis < 3; axis++) {
      const low = record[2 * pair]!, high = record[2 * pair + 1]!, a = (axis + 1) % 3, b = (axis + 2) % 3;
      const cell = (open: OpenCells, i: number, j: number, along: number) => { const l = [0, 0, 0]; l[a] = i; l[b] = j; l[axis] = along; const k = l[0]! + 4 * l[1]! + 16 * l[2]!; return (open[k >>> 5]! >>> (k & 31)) & 1; };
      const want = wetFace(axis, low, high, (i, j) => 0.5 * (cell(low.open, i, j, 3) + cell(high.open, i, j, 0)),
        (plane, i, j) => { const p = [0, 0, 0]; p[a] = i; p[b] = j; p[axis] = 3 + plane; return facePhi(2 * pair, p); });
      assert.deepEqual({ v: got.faces[2 * (3 * pair + axis)], mask: got.faces[2 * (3 * pair + axis) + 1] }, want, `pair ${pair} axis ${axis}: ${JSON.stringify({ low, high })}`);
      if (want.mask !== 0) kept++; if (low.row !== high.row && want.mask !== 0) surface++;
    }
    console.log(JSON.stringify({ kernel: "umWetFace", faces: 3 * PAIRS, kept, betweenRowAndAir: surface }));
    assert.deepEqual(errors, []);
    console.log("root wet face: the kernels hold their emulation");
  } finally { device.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
