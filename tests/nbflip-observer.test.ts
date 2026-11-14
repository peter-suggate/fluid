import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { levelSetFill } from "../tools/seep-observables";
import {
  BOX_SCENES, boxAdd, boxAddRegion, boxGeometry, boxObservation, boxObserverWGSL, boxOracle, boxWords, damColumn, decodeBoxWords, emptyBoxSums,
  expectedCells, scheduledDropVolumes, type BoxGeometry,
} from "../tools/nbflip-observables";
import { withUniformDevice } from "./helpers/uniform-geometric";

// The NB-FLIP port's box observables (tools/nbflip-observables.ts) on a small
// synthetic tank: 32 x 24 x 16 cells, bins of 8, a 4 x 8 column at the back
// wall. Its left half is h owners, its right half 4h owners. The Float64
// oracle is checked against closed forms; the WGSL accumulator (GPU,
// WEBGPU_NODE_MODULE only) against the oracle on the same owners.
const g: BoxGeometry = { nx: 32, ny: 24, nz: 16, cell_m: 0.01, gravity_m_s2: 9.81, density_kg_m3: 1000, binCells: [8, 8], bins: [4, 2], column: { x: [0, 4], z: [0, 8] } };
type Owner = { q: [number, number, number]; w: number; phi: (x: number, y: number, z: number) => number; u: [number, number, number] };
/** h owners at x < 16, 4h owners at x >= 16, over the whole tank. */
function owners(phi: Owner["phi"], u: (x: number, y: number, z: number) => [number, number, number]): Owner[] {
  const list: Owner[] = [];
  for (let z = 0; z < g.nz; z++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < 16; x++) list.push({ q: [x, y, z], w: 1, phi, u: u(x, y, z) });
  for (let z = 0; z < g.nz; z += 4) for (let y = 0; y < g.ny; y += 4) for (let x = 16; x < g.nx; x += 4) list.push({ q: [x, y, z], w: 4, phi, u: u(x, y, z) });
  return list;
}
const LEVEL = 10.3;
const level = (x: number, y: number) => y - LEVEL + 0 * x;
const still = (): [number, number, number] => [0.3, -0.4, 0];

test("box oracle matches closed forms for a level pool", () => {
  const s = boxOracle(g, owners(level, still), 5);
  const columns = g.nx * g.nz;
  assert.ok(Math.abs(s.total - LEVEL * columns) < 1e-9, `total ${s.total}`);
  // A level surface's bottom-filled centre is exact at h and 4h: each column's y moment is LEVEL^2 / 2.
  assert.ok(Math.abs(s.yMoment - columns * LEVEL ** 2 / 2) < 1e-9, `y moment ${s.yMoment}`);
  assert.ok(Math.abs(s.xMoment / s.total - g.nx / 2) < 1e-9 && Math.abs(s.zMoment / s.total - g.nz / 2) < 1e-9);
  assert.ok(Math.abs(s.ke - s.total * 0.25) < 1e-9, `ke ${s.ke}`);
  assert.ok(s.tops.every(t => Math.abs(t - LEVEL) < 1e-9), `tops ${s.tops}`);
  assert.ok(s.bins.every(b => Math.abs(b - LEVEL * 64) < 1e-9));
  assert.equal(s.front, g.nx); assert.equal(s.wetFloor, columns);
  assert.ok(Math.abs(s.column - LEVEL * 32) < 1e-9 && Math.abs(s.columnTop - LEVEL) < 1e-9);
  const row = boxObservation(g, "dam", 30, 1 / 60, s, s.total, { dam: { width_cells: 16, height_cells: 20 } });
  assert.equal(row.t_s, 0.5); assert.equal(row.drift, 0); assert.equal(row.particles, 5);
  assert.ok(Math.abs(row.total_L - LEVEL * columns * 1e-3) < 1e-9);
  // Energies: KE = rho V |u|^2 / 2, PE = rho g h sum(liquid y).
  assert.ok(Math.abs(row.ke_J - 0.5 * 1000 * 1e-6 * s.total * 0.25) < 1e-9);
  assert.ok(Math.abs(row.pe_J - 1000 * 1e-6 * 9.81 * 0.01 * columns * LEVEL ** 2 / 2) < 1e-9);
  assert.equal(row.frontZ, 2); assert.ok(Math.abs(row.columnEta! - LEVEL / 20) < 1e-12 && Math.abs(row.columnDepth_cells! - LEVEL) < 1e-9);
  assert.ok(row.profileDepth_cells.every(d => Math.abs(d - LEVEL) < 1e-9) && row.profileTop_cells.every(t => Math.abs(t - LEVEL) < 1e-9));
  const short = boxObservation(g, "dam", 30, 1 / 60, s, s.total / 0.98);
  assert.ok(Math.abs(short.drift + 0.02) < 1e-12);
});

/** A dam: liquid at x < 10.5, z < 6, y < 18 (the box's signed distance, cells). */
function dam(x: number, y: number, z: number): number {
  const d = [x - 10.5, y - 18, z - 6], outside = Math.hypot(...d.map(v => Math.max(v, 0)));
  return outside > 0 ? outside : Math.max(...d);
}
test("box oracle reads a dam's front, column and bins", () => {
  const s = boxOracle(g, owners(dam, () => [0, 0, 0]));
  // Exact but for the 23 h cells along the x = 10.5 edges (with y = 18 and with z = 6), where the box's
  // distance is not linear across the cell and the six-tetrahedron fill reads under 0.5 (by about 0.13 each).
  assert.ok(s.total < 10.5 * 18 * 6 && s.total > 10.5 * 18 * 6 - 0.15 * 23, `total ${s.total}`);
  // The furthest wet floor owner is the h cell x = 10 (it holds half a cell): the front is its far edge, 11.
  assert.equal(s.front, 11); assert.equal(s.wetFloor, 11 * 6);
  assert.ok(Math.abs(s.columnTop - 18) < 1e-9);
  assert.ok(Math.abs(s.column - 4 * 6 * 18) < 1e-6, `column ${s.column}`);
  // Bins: x bin 0 (x 0-8) holds 8 * 18 * 6, bin 1 (8-16) 2.5 * 18 * 6, the rest nothing; all at z bin 0.
  assert.ok(Math.abs(s.bins[0]! - 8 * 18 * 6) < 1e-6 && Math.abs(s.bins[1]! - (s.total - 8 * 18 * 6)) < 1e-6 && s.bins.slice(2).every(b => Math.abs(b) < 1e-9));
  assert.ok(Math.abs(s.tops[1]! - 18) < 1e-9 && s.tops[2] === 0 && s.tops[4] === 0);
});

test("a BULK16 region adds as its 64 full 4h owners", () => {
  const big: BoxGeometry = { ...g, nx: 64, ny: 32, nz: 32, bins: [8, 4], column: { x: [0, 4], z: [0, 8] } };
  const s = emptyBoxSums(big);
  boxAddRegion(big, s, [0, 0, 16], [1, 0, 0]);
  assert.equal(s.total, 4096); assert.equal(s.ke, 4096);
  assert.ok(Math.abs(s.yMoment - 4096 * 8) < 1e-9 && Math.abs(s.xMoment - 4096 * 8) < 1e-9 && Math.abs(s.zMoment - 4096 * 24) < 1e-9);
  assert.equal(s.front, 16); assert.equal(s.wetFloor, 256);
  // The region covers x bins 0-1 and z bins 2-3: 1024 cells and a top of 16 in each.
  for (const b of [16, 17, 24, 25]) { assert.equal(s.bins[b], 1024); assert.equal(s.tops[b], 16); }
  assert.equal(s.bins.reduce((a, v) => a + v, 0), 4096);
  assert.equal(s.column, 0);
});

test("scene geometry and the letters' drop schedule", () => {
  const dam = sceneDocument(getSceneDefinition(BOX_SCENES.dam)), gd = boxGeometry(dam, "dam");
  assert.deepEqual([gd.nx, gd.ny, gd.nz, gd.bins], [128, 128, 128, [16, 16]]);
  const c = damColumn(dam);
  assert.ok(Math.abs(c.width_cells - 80) < 1e-9 && Math.abs(c.depth_cells - 80) < 1e-9 && Math.abs(c.height_cells - 0.92 * 128) < 1e-9);
  assert.deepEqual(gd.column, { x: [0, 4], z: [0, 80] });
  const letters = sceneDocument(getSceneDefinition(BOX_SCENES.letters)), gl = boxGeometry(letters, "letters");
  assert.deepEqual([gl.nx, gl.ny, gl.nz, gl.bins], [256, 192, 128, [16, 8]]);
  // A letter every 10 frames at 24 fps, the first at frame 1: drop i lands in step 25 i + 3 at 1/60 s.
  const drops = scheduledDropVolumes(letters, 1 / 60);
  assert.deepEqual(drops.map(d => d.step), Array.from({ length: 9 }, (_, i) => 25 * i + 3));
  assert.ok(drops.every(d => d.cells > 5000 && d.cells < 30000), `drop cells ${drops.map(d => d.cells.toFixed(0))}`);
  assert.equal(expectedCells(100, drops, 2), 100);
  assert.ok(Math.abs(expectedCells(100, drops, 28) - (100 + drops[0]!.cells + drops[1]!.cells)) < 1e-9);
  const gp = boxGeometry(sceneDocument(getSceneDefinition(BOX_SCENES.paddleboats)), "paddleboats");
  assert.deepEqual([gp.nx, gp.ny, gp.nz, gp.bins, gp.column], [512, 32, 192, [16, 6], undefined]);
});

test("box words decode their own layout", () => {
  const u = new Uint32Array(boxWords(g));
  // total: 2^32 + 4096 fixed (1048576 + 1 cells); front 17; the last bin's top 5.5 cells.
  u[0] = 4096; u[1] = 1; u[2 * (6 + 8)] = 17; u[boxWords(g) - 1] = 88;
  const s = decodeBoxWords(g, u);
  assert.equal(s.total, (4294967296 + 4096) / 4096); assert.equal(s.front, 17); assert.equal(s.tops[7], 5.5);
});

/** The box accumulator alone, over owners from a storage buffer: (q, w, liquid, u) in eight words. */
test("box observer WGSL matches the Float64 oracle", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const tilted = (x: number, y: number, z: number) => Math.min(y - 9 - 0.2 * x - 0.1 * z, dam(x, y, z));
  const list = owners(tilted, (x, y, z) => [0.05 * x, -0.1 * y, 0.02 * z]);
  const cpu = boxOracle(g, list);
  const data = new ArrayBuffer(32 * list.length), f = new Float32Array(data), w = new Uint32Array(data);
  list.forEach((o, i) => {
    const c = Array.from({ length: 8 }, (_, k) => o.phi(o.q[0] + o.w * (k & 1), o.q[1] + o.w * ((k >> 1) & 1), o.q[2] + o.w * (k >> 2)));
    const lo = Math.min(...c), hi = Math.max(...c), cells = o.w ** 3;
    w.set([...o.q, o.w], 8 * i); f[8 * i + 4] = hi < 0 ? cells : lo >= 0 ? 0 : levelSetFill(c) * cells; f.set(o.u, 8 * i + 5);
  });
  // The oracle on the f32-rounded inputs the GPU reads.
  const rounded = emptyBoxSums(g);
  list.forEach((o, i) => boxAdd(g, rounded, o.q, o.w, f[8 * i + 4]!, [f[8 * i + 5]!, f[8 * i + 6]!, f[8 * i + 7]!]));
  assert.ok(Math.abs(rounded.total - cpu.total) < 1e-3);
  await withUniformDevice("box observer", async device => {
    const WG = 64, words = boxWords(g);
    const module = device.createShaderModule({ code: /* wgsl */`
@group(0) @binding(0) var<storage,read> owners:array<u32>;
@group(0) @binding(1) var<storage,read_write> boxWords:array<atomic<u32>>;
${boxObserverWGSL(g, WG)}
@compute @workgroup_size(${WG}) fn observe(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 boxBegin(lane);
 if(gid.x<${list.length}u){let b=8u*gid.x;
  boxAdd(vec3u(owners[b],owners[b+1u],owners[b+2u]),owners[b+3u],bitcast<f32>(owners[b+4u]),vec3f(bitcast<f32>(owners[b+5u]),bitcast<f32>(owners[b+6u]),bitcast<f32>(owners[b+7u])));}
 boxEnd(lane);
}` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "observe" } });
    const input = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, data);
    const out = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const e = device.createCommandEncoder(), pass = e.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: input } }, { binding: 1, resource: { buffer: out } }] }));
    pass.dispatchWorkgroups(Math.ceil(list.length / WG)); pass.end();
    e.copyBufferToBuffer(out, 0, read, 0, 4 * words); device.queue.submit([e.finish()]);
    await read.mapAsync(GPUMapMode.READ); const gpu = decodeBoxWords(g, new Uint32Array(read.getMappedRange().slice(0))); read.unmap();
    input.destroy(); out.destroy(); read.destroy();
    // Fixed point rounds each owner's term by half a unit: bound by the owner count.
    const n = list.length, near = (a: number, b: number, tol: number, what: string) => assert.ok(Math.abs(a - b) <= tol, `${what}: gpu ${a} cpu ${b}`);
    near(gpu.total, rounded.total, n / 8192 + 1e-6 * rounded.total, "total"); near(gpu.column, rounded.column, n / 8192 + 1e-6 * rounded.column, "column");
    for (const k of ["xMoment", "yMoment", "zMoment"] as const) near(gpu[k], rounded[k], n / 128 + 1e-6 * rounded[k], k);
    near(gpu.ke, rounded.ke, n / 2048 + 1e-5 * rounded.ke, "ke");
    gpu.bins.forEach((b, i) => near(b, rounded.bins[i]!, n / 8192 + 1e-6 * b, `bin ${i}`));
    gpu.tops.forEach((t, i) => near(t, rounded.tops[i]!, 1 / 32 + 1e-6, `top ${i}`));
    near(gpu.columnTop, rounded.columnTop, 1 / 32 + 1e-6, "column top");
    assert.equal(gpu.front, rounded.front); assert.equal(gpu.wetFloor, rounded.wetFloor);
  });
});
