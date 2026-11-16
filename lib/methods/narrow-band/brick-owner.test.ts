// The tier-resolving readers (brick-owner.wgsl.ts, brick-abi.md §2.5) on the
// GPU against a CPU oracle of the same rules, on an engine whose CMASK makes
// a block of tiles 4h-owned, with known C corners and C faces. GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { BRICK_CELLS, BRICK_STRIDE, CMASK_WORDS, PHI_AIR_C, brickAbiWGSL, phiFar } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { BrickEngine } from "./brick-engine";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { cellSlot, readChannel, snapshot } from "./brick-readback";
import { bringUpStages } from "./brick-stages";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type Vec3 = [number, number, number];
const N = 32;
const FAR = phiFar([N, N, N]);
const tileIndex = (t: Vec3) => (t[0] & 7) | ((t[1] & 7) << 3) | ((t[2] & 7) << 6);
/** The block made 4h-owned: tiles x 2..3, y 0..1, every z. */
const coarse = (t: Vec3) => t[0] >= 2 && t[0] <= 3 && t[1] <= 1;
const cPhi = (T: Vec3) => 0.25 * T[0] - 0.5 * T[1] + 0.125 * T[2] - 1;
const cFace = (T: Vec3, a: number) => 0.01 * (a + 1) + 0.003 * T[0] - 0.002 * T[1] + 0.001 * T[2];

test("brick owner (GPU): vertex authority, hanging vertices and faces resolve as NB-FLIP's rules", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick owner", async device => {
    const e = await BrickEngine.create(device, { lattice: [N, N, N], h_m: 1 / 32, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 200, c: 8, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [33, 10.3, 33] }], solver: L => bringUpCG(L), stages: bringUpStages });
    const s = await snapshot(e), L = e.layout;
    const cs = s.pages.get(0)!;
    assert.ok(cs !== undefined, "page 0 is resident");
    // CMASK: clear the block's bits (every tile of the page was h-owned at boot where its brick lives).
    const mask = (await e.read("dir", L.dir.cmask + CMASK_WORDS * cs, CMASK_WORDS)).slice();
    const tileH = (t: Vec3) => t.every((v, a) => v >= 0 && 4 * v < N) && !coarse(t) && ((mask[tileIndex(t) >> 5]! >>> (tileIndex(t) & 31)) & 1) === 1;
    for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) if (coarse([x, y, z])) { const i = tileIndex([x, y, z]); mask[i >> 5]! &= ~(1 << (i & 31)); }
    e.write("dir", L.dir.cmask + CMASK_WORDS * cs, mask);
    // C corners and C faces, everywhere in the page.
    const cphi = new Float32Array(BRICK_CELLS), cu = [0, 1, 2].map(() => new Float32Array(BRICK_CELLS));
    for (let z = 0; z < 8; z++) for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
      const i = tileIndex([x, y, z]); cphi[i] = cPhi([x, y, z]); for (let a = 0; a < 3; a++) cu[a]![i] = cFace([x, y, z], a);
    }
    e.write("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, cphi);
    for (let a = 0; a < 3; a++) e.write("vel", L.base("vel", "C", `U${a}`) + BRICK_STRIDE * cs, cu[a]!);
    // F faces: a known field on every brick.
    const fFace = (c: Vec3, a: number) => -0.02 * (a + 1) + 0.001 * c[0] + 0.0005 * c[1] - 0.0003 * c[2];
    for (const [key, slot] of s.bricks) {
      const b = [key & 1023, (key >>> 10) & 1023, (key >>> 20) & 1023];
      for (let a = 0; a < 3; a++) {
        const w = new Float32Array(BRICK_CELLS);
        for (let l = 0; l < BRICK_CELLS; l++) w[l] = fFace([8 * b[0]! + (l & 7), 8 * b[1]! + ((l >> 3) & 7), 8 * b[2]! + (l >> 6)], a);
        e.write("vel", L.base("vel", "F", `U${a}`) + BRICK_STRIDE * slot, w);
      }
    }
    const fphi = await readChannel(e, "phi", "PHI");
    // The CPU oracle.
    const inLattice = (c: Vec3) => c.every(v => v >= 0 && v < N);
    // A vertex no slot stores reads PHI_AIR_C (16 h, the old stored far air); outside the lattice, FAR.
    const fVertex = (v: Vec3) => { const at = cellSlot(s, v); return at ? fphi(at.slot)[at.local]! : PHI_AIR_C; };
    const cVertex = (T: Vec3) => T.some(v => v < 0 || 4 * v > N) ? FAR : T.some(v => v >= 8) ? PHI_AIR_C : Math.fround(cPhi(T));
    const vertexCoarse = (v: Vec3): Vec3 | undefined => {
      const tA = v.map(x => x >> 2) as Vec3, inside = v.map((x, a) => x - 4 * tA[a]! !== 0);
      for (let k = 0; k < 8; k++) {
        const corner = [k & 1, (k >> 1) & 1, k >> 2];
        if (corner.some((c, a) => inside[a] && c === 0)) continue;
        const t = tA.map((x, a) => x + corner[a]! - 1) as Vec3;
        // page 0 holds every in-lattice tile here (a non-resident page's tile is no candidate: the next test).
        if (t.every(x => x >= 0 && 4 * x < N) && !tileH(t)) return t;
      }
      return undefined;
    };
    const vertexPhi = (v: Vec3) => {
      if (v.some(x => x < 0 || x > N)) return FAR;
      const T = vertexCoarse(v);
      if (!T) return fVertex(v);
      const t = v.map((x, a) => Math.fround((x - 4 * T[a]!) / 4)), values = new Array<number>(8).fill(0);
      for (let k = 0; k < 8; k++) {
        const corner = [k & 1, (k >> 1) & 1, k >> 2], w = corner.map((c, a) => c ? t[a]! : Math.fround(1 - t[a]!));
        const weight = Math.fround(Math.fround(w[0]! * w[1]!) * w[2]!);
        if (weight > 0) values[k] = Math.fround(weight * cVertex(T.map((x, a) => x + corner[a]!) as Vec3));
      }
      const f = Math.fround, v_ = values;
      return f(f(f(v_[0]! + v_[5]!) + f(v_[1]! + v_[4]!)) + f(f(v_[2]! + v_[7]!) + f(v_[3]! + v_[6]!)));
    };
    const cellH = (c: Vec3) => tileH(c.map(x => x >> 2) as Vec3);
    const cF = (T: Vec3, a: number) => T.every(x => x >= 0 && 4 * x < N) ? Math.fround(cFace(T, a)) : 0; // only page 0's C faces are written
    const fF = (c: Vec3, a: number) => cellSlot(s, c) ? Math.fround(fFace(c, a)) : 0;
    const faceInLattice = (c: Vec3, a: number) => c.every((v, i) => v >= 0 && v < (i === a ? N + 1 : N));
    const plane = (c: Vec3, a: number) => {
      if (!faceInLattice(c, a)) return 0;
      const below = [...c] as Vec3; below[a]! -= 1;
      return cellH(c) || cellH(below) ? fF(c, a) : cF(c.map(x => x >> 2) as Vec3, a);
    };
    const faceU = (c: Vec3, a: number) => {
      if (!faceInLattice(c, a)) return 0;
      const below = [...c] as Vec3; below[a]! -= 1;
      if (cellH(c) || cellH(below)) return fF(c, a);
      const T = c.map(x => x >> 2) as Vec3, k = c[a]! - 4 * T[a]!;
      if (k === 0) return cF(T, a);
      const up = [...c] as Vec3; up[a] = 4 * T[a]! + 4;
      const lo = cF(T, a), hi = plane(up, a);
      return Math.fround(lo + Math.fround(Math.fround(hi - lo) * Math.fround(k / 4)));
    };
    // Probes: every vertex and face of the region x 0..20, y 0..12, z 0..6 (block, its h|4h planes, the walls).
    const probes: Vec3[] = [];
    for (let z = 0; z <= 6; z++) for (let y = 0; y <= 12; y++) for (let x = 0; x <= 20; x++) probes.push([x, y, z]);
    const code = `${brickAbiWGSL(L, { dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> dir:array<u32>;
@group(0) @binding(1) var<storage,read> phi:array<u32>;
@group(0) @binding(2) var<storage,read> vel:array<f32>;
@group(0) @binding(3) var<storage,read> probe:array<i32>;
@group(0) @binding(4) var<storage,read_write> outp:array<f32>;
${brickOwnerWGSL()}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
  let i=gid.x;if(i>=${probes.length}u){return;}
  let v=vec3i(probe[3u*i],probe[3u*i+1u],probe[3u*i+2u]);
  outp[5u*i]=nbVertexPhi(v);outp[5u*i+1u]=select(0.0,1.0,nbVertexOwnerH(v));
  for(var a=0u;a<3u;a++){outp[5u*i+2u+a]=nbFaceU(v,a);}
}`;
    const module = device.createShaderModule({ code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    assert.deepEqual(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const buf = (data: Int32Array<ArrayBuffer>) => { const b = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device.queue.writeBuffer(b, 0, data); return b; };
    const pb = buf(new Int32Array(probes.flat())), out = device.createBuffer({ size: 20 * probes.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rd = device.createBuffer({ size: 20 * probes.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: e.buffer("dir") } }, { binding: 1, resource: { buffer: e.buffer("phi") } }, { binding: 2, resource: { buffer: e.buffer("vel") } },
      { binding: 3, resource: { buffer: pb } }, { binding: 4, resource: { buffer: out } }] });
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(probes.length / 64)); pass.end();
    enc.copyBufferToBuffer(out, 0, rd, 0, 20 * probes.length); device.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ); const got = new Float32Array(rd.getMappedRange().slice(0)); rd.unmap();
    let hanging = 0, corners = 0, hPlane = 0, walls = 0;
    probes.forEach((v, i) => {
      const T = vertexCoarse(v);
      assert.equal(got[5 * i + 1], T ? 0 : 1, `owner at ${v}`);
      assert.equal(got[5 * i], vertexPhi(v), `phi at ${v} (coarse ${T})`);
      if (T && v.some((x, a) => x !== 4 * T[a]! && x !== 4 * T[a]! + 4)) hanging++; else if (T) corners++;
      for (let a = 0; a < 3; a++) {
        if (!inLattice(v) && v[a]! < N) continue;
        // Inside a 4h tile nbFaceU is mix(), whose rounding the GPU chooses (WGSL fixes no form): a few ulps of the
        // operands (|u| < 0.06 here, so 1e-8). Every stored face is exact.
        const want = faceU(v, a), interior = !cellH(v) && v[a]! % 4 !== 0;
        if (interior) assert.ok(Math.abs(got[5 * i + 2 + a]! - want) <= 1e-8, `face ${a} at ${v}: ${got[5 * i + 2 + a]} against ${want}`);
        else assert.equal(got[5 * i + 2 + a], want, `face ${a} at ${v}`);
        const below = [...v] as Vec3; below[a]! -= 1;
        if (cellH(v) !== cellH(below) && v[a]! > 0 && v[a]! < N) hPlane++;
        if (v[a] === 0 && got[5 * i + 2 + a] !== 0) walls++;
      }
    });
    assert.ok(hanging > 0 && corners > 0 && hPlane > 0 && walls > 0, `hanging ${hanging}, corners ${corners}, interface faces ${hPlane}, wall faces ${walls}`);
    for (const b of [pb, out, rd]) b.destroy();
    e.destroy();
  });
});

// A tile of a non-resident page holds no corners: it is air, never a vertex's authority (brick-abi.md §2.5). All-h,
// the vertices on an absent page's upper face (x = 32 here, the absent page below at x < 32) are F's. Taking the
// absent tile as 4h read the resident page's C corners, which all-h hold bootPhi's stale φ: lane P's absent-page
// guard hits on the all-h dam (2026-11-15).
test("brick owner (GPU): a non-resident page's tile is no vertex authority (all-h: no stale C)", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick owner absent page", async device => {
    const M = 64;
    const e = await BrickEngine.create(device, { lattice: [M, M, M], h_m: 1 / 64, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 500, c: 32, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [40, -1, -1], hi: [65, 20.3, 65] }], solver: L => bringUpCG(L), stages: bringUpStages });
    const s = await snapshot(e), L = e.layout;
    const pageKey = (p: Vec3) => (p[0] | (p[1] << 10) | (p[2] << 20)) >>> 0;
    assert.equal(s.pages.get(pageKey([0, 0, 0])), undefined, "page (0,0,0) is not resident");
    assert.ok(s.pages.get(pageKey([1, 0, 0])) !== undefined, "page (1,0,0) is resident");
    // Stale C φ: every resident page's C vertices deep liquid, which no all-h reader may take.
    for (const cs of s.pages.values()) e.write("phi", L.base("phi", "C", "PHI") + BRICK_STRIDE * cs, new Float32Array(BRICK_CELLS).fill(-5));
    const fphi = await readChannel(e, "phi", "PHI");
    const fVertex = (v: Vec3) => { const at = cellSlot(s, v); return at ? fphi(at.slot)[at.local]! : PHI_AIR_C; };
    const probes: Vec3[] = [];
    for (let z = 0; z <= 12; z++) for (let y = 0; y <= 12; y++) for (const x of [31, 32, 33]) probes.push([x, y, z]);
    const code = `${brickAbiWGSL(L, { dir: { name: "dir", atomic: false } })}
@group(0) @binding(0) var<storage,read> dir:array<u32>;
@group(0) @binding(1) var<storage,read> phi:array<u32>;
@group(0) @binding(2) var<storage,read> vel:array<f32>;
@group(0) @binding(3) var<storage,read> probe:array<i32>;
@group(0) @binding(4) var<storage,read_write> outp:array<f32>;
${brickOwnerWGSL()}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
  let i=gid.x;if(i>=${probes.length}u){return;}
  let v=vec3i(probe[3u*i],probe[3u*i+1u],probe[3u*i+2u]);
  outp[2u*i]=nbVertexPhi(v);outp[2u*i+1u]=select(0.0,1.0,nbVertexOwnerH(v));
  _=vel[0];
}`;
    const module = device.createShaderModule({ code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    assert.deepEqual(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const pb = device.createBuffer({ size: 12 * probes.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(pb, 0, new Int32Array(probes.flat()));
    const out = device.createBuffer({ size: 8 * probes.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const rd = device.createBuffer({ size: 8 * probes.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: e.buffer("dir") } }, { binding: 1, resource: { buffer: e.buffer("phi") } }, { binding: 2, resource: { buffer: e.buffer("vel") } },
      { binding: 3, resource: { buffer: pb } }, { binding: 4, resource: { buffer: out } }] });
    const enc = device.createCommandEncoder(); const pass = enc.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(probes.length / 64)); pass.end();
    enc.copyBufferToBuffer(out, 0, rd, 0, 8 * probes.length); device.queue.submit([enc.finish()]);
    await rd.mapAsync(GPUMapMode.READ); const got = new Float32Array(rd.getMappedRange().slice(0)); rd.unmap();
    let onFace = 0;
    probes.forEach((v, i) => {
      assert.equal(got[2 * i + 1], 1, `vertex ${v} is F's (all-h: no 4h authority)`);
      assert.equal(got[2 * i], fVertex(v), `phi at ${v}`);
      if (v[0] === 32 && fVertex(v) > 0) onFace++;
    });
    // The absent page's upper face, where the old candidate (tile x = 7) would have read the stale -5.
    assert.ok(onFace > 0, `probes on the absent page's face: ${onFace}`);
    for (const b of [pb, out, rd]) b.destroy();
    e.destroy();
  });
});
