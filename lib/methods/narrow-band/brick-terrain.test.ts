/** The brick engine's terrain (brick-terrain.wgsl.ts): the old engine's bake, the mirror's exactness on a planar
 * ground, the stamp's cut bound against the mirror's cut cells, the setup's terrain and its explicit refusals (a sculpt
 * cut), and on the GPU the WGSL against the old engine's own surface text bit for bit, the hooks against the mirror and
 * the particle contact on a planar ground (skips without WEBGPU_NODE_MODULE). */
import test from "node:test";
import assert from "node:assert/strict";
import { brickTerrainColumns, brickTerrainCutBricks, brickTerrainMirror, brickTerrainWGSL, planeBoxFraction, type BrickTerrainColumns } from "./brick-terrain.wgsl";
import { brickStaticMask, brickStaticMirror, brickStaticWGSL } from "./brick-static.wgsl";
import { BRICK_ENGINE_SCENES, brickSceneSetup } from "./brick-scene";
import { parityGPU, prng, runParity, withParityDevice } from "./parity/harness";
import { uniformTerrainSamplingWGSL } from "../uniform/uniform-terrain-sampling.wgsl";
import { geometricPlaneBoxWGSL } from "../../core/geometric-plane-box.wgsl";
import { terrainColumnHeights } from "../../core/terrain";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { getSceneDefinition } from "../../core/scenes";
import { sceneDocument } from "../../core/scene-definition";
import type { SceneDescription } from "../../core/model";

const sceneOf = (id: string): SceneDescription => {
  const d = getSceneDefinition(id) as unknown as { create?: () => SceneDescription };
  return d.create ? d.create() : sceneDocument(d as never);
};
const SLOPE = "nbflip-slope-clearance";

/** A planar ground h(x, z) = a + b x + c z sampled at column centres (cells). */
function planar(nx: number, nz: number, a: number, b: number, c: number): BrickTerrainColumns {
  const heights = new Float32Array(nx * nz);
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) heights[x + nx * z] = a + b * (x + 0.5) + c * (z + 0.5);
  return { columns: [nx, nz], heights };
}

test("terrain: the columns are the old engine's bake (terrainColumnHeights over the container, in cells)", () => {
  const scene = sceneOf(SLOPE), t = brickTerrainColumns(scene)!;
  const [nx, ny, nz] = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER);
  assert.deepEqual(t.columns, [nx, nz]);
  const metres = terrainColumnHeights(scene, nx, nz), cell = scene.container.height_m / ny;
  for (let i = 0; i < nx * nz; i++) assert.equal(t.heights[i], Math.fround(metres[i]! / cell));
  assert.equal(brickTerrainColumns({ ...scene, terrain: undefined } as SceneDescription), undefined);
});

test("terrain: the mirror is exact on a planar ground (four planes, one plane)", () => {
  const t = planar(12, 10, 3.25, 0.37, -0.21), m = brickTerrainMirror(t), r = prng(3);
  for (let i = 0; i < 400; i++) {
    // Interior points (the planar patch continues one plane): every sample's plane is the ground's.
    const q = [1 + 10 * r(), 8 * r(), 1 + 8 * r()];
    const s = m.surface(q);
    // The columns are f32: the ground's plane to their rounding.
    assert.ok(Math.abs(s[0] - (3.25 + 0.37 * q[0]! - 0.21 * q[2]!)) < 1e-5);
    assert.ok(Math.abs(s[1] - 0.37) < 1e-5 && Math.abs(s[2] + 0.21) < 1e-5);
    const open = m.boxOpen(q, [1, 1, 1]), exact = planeBoxFraction([-0.37, 1, 0.21], q[1]! - s[0], [1, 1, 1]);
    assert.ok(Math.abs(open - exact) < 1e-5, `${q}: ${open} against ${exact}`);
  }
  // Flat ground at 2.5: the cell [.., 2, ..] is half open, [.., 1, ..] closed, [.., 3, ..] open.
  const flat = brickTerrainMirror(planar(4, 4, 2.5, 0, 0));
  assert.equal(flat.boxOpen([1.5, 2.5, 1.5], [1, 1, 1]), 0.5);
  assert.equal(flat.boxOpen([1.5, 1.5, 1.5], [1, 1, 1]), 0);
  assert.equal(flat.boxOpen([1.5, 3.5, 1.5], [1, 1, 1]), 1);
  assert.ok(flat.columnCut([1, 2, 1]) && !flat.columnCut([1, 3, 1]) && !flat.columnCut([4, 0, 1]));
});

test("terrain: the stamp's cut bound holds every brick with a cut cell or lower face (the scene's own ground)", () => {
  const scene = sceneOf(SLOPE), t = brickTerrainColumns(scene)!;
  const box = sceneLatticeDimensions(scene) as unknown as [number, number, number];
  const m = brickStaticMirror({ box, openTop: false, frame: [0, 0, 0], h: [1, 1, 1], mask: brickStaticMask(scene, { always: true })!.words, terrain: t });
  const bound = new Set<number>();
  brickTerrainCutBricks(t, box, (x, y, z) => bound.add(x + 4096 * (y + 4096 * z)));
  let cut = 0, missed = 0;
  for (let z = 0; z < box[2]; z++) for (let y = 0; y < box[1]; y++) for (let x = 0; x < box[0]; x++) {
    const c = [x, y, z];
    let hit = m.cellOpen(c) < 1;
    for (let a = 0; a < 3 && !hit; a++) {
      const b = c.map((v, i) => v - (i === a ? 1 : 0));
      if (b[a]! >= 0) hit = m.faceOpen(b, a) < 1;
    }
    if (!hit) continue;
    cut++;
    if (!bound.has((x >> 3) + 4096 * ((y >> 3) + 4096 * (z >> 3)))) missed++;
  }
  assert.ok(cut > 0, "the ground cuts cells");
  assert.equal(missed, 0, `${missed} of ${cut} cut cells lie in a brick the bound leaves out`);
});

test("terrain: the setup binds the ground, never defers it, and refuses a sculpt cut explicitly", () => {
  const scene = sceneOf(SLOPE);
  (BRICK_ENGINE_SCENES as string[]).push(scene.sceneId!);
  try {
    const r = brickSceneSetup(scene, { volumeControlSeconds: 0 });
    assert.deepEqual(r.gaps, []);
    assert.ok(r.setup?.terrain && r.setup.staticsDeferred === undefined);
    assert.ok(r.setup.capacities.k >= 64);
    const sculpted: SceneDescription = { ...scene, solidVoxels: [...scene.solidVoxels, { operation: "clear", kind: "ellipsoid", minimum: [10, 4, 10], maximumExclusive: [14, 8, 14], affectsScenery: true } as never] };
    const g = brickSceneSetup(sculpted, { volumeControlSeconds: 0 });
    assert.equal(g.setup, undefined);
    assert.ok(g.gaps.some(s => s.includes("sculpt")), g.gaps.join(" | "));
  } finally { (BRICK_ENGINE_SCENES as string[]).pop(); }
});

// ---- GPU ----

const columnsOf = (t: BrickTerrainColumns) => new Uint32Array(t.heights.buffer.slice(0));

test("terrain: nbTerrainSurface is the old engine's umTerrainSurface bit for bit; the plane box is its copy's", { skip: !parityGPU }, async () => {
  const scene = sceneOf(SLOPE), t = brickTerrainColumns(scene)!, [nx, nz] = t.columns;
  // The old text with its texture read as the same columns (the harness binds buffers): the two texture calls replaced.
  const dims = "textureDimensions(umSolidTerrain)", load = "textureLoad(umSolidTerrain,clamp(world,vec2i(0),d-1),0).x";
  assert.equal(uniformTerrainSamplingWGSL.split(dims).length, 3);
  assert.equal(uniformTerrainSamplingWGSL.split(load).length, 2);
  const old = uniformTerrainSamplingWGSL.replaceAll(dims, `vec2u(${nx}u,${nz}u)`)
    .replace(load, "umOldColumn(clamp(world,vec2i(0),d-1))");
  const prelude = `${brickTerrainWGSL({ buffer: "ext3", base: 0, columns: t.columns })}
@group(0) @binding(3) var<storage,read> ext3:array<u32>;
fn umOldColumn(c:vec2i)->f32{return bitcast<f32>(ext3[u32(c.x)+${nx}u*u32(c.y)]);}
fn umFramePage()->vec3i{return vec3i(0);}
struct UmAnalytic{ground:vec4f,frame:vec4f}
const umAnalyticTerrain=UmAnalytic(vec4f(0),vec4f(0));
fn owHeight(x:f32,z:f32)->f32{return 0.0;}
${old}
${geometricPlaneBoxWGSL}`;
  const r = prng(11), n = 4096, inp = new Float32Array(8 * n);
  for (let i = 0; i < n; i++) {
    inp.set([(nx + 4) * r() - 2, 40 * r() - 4, (nz + 4) * r() - 2], 8 * i);
    inp.set([2 * r() - 1, 2 * r() - 1 + 0.5, 2 * r() - 1, 4 * r() - 2], 8 * i + 3);
    inp[8 * i + 7] = [1, 0.5, 2, 0.25][i & 3]!;
  }
  const body = `let q=v3(8u*t);let o=8u*t;
 let a=nbTerrainSurface(q);let b=umTerrainSurface(q);
 put(o,a.x);put(o+1u,a.y);put(o+2u,a.z);put(o+3u,b.x);put(o+4u,b.y);put(o+5u,b.z);
 let nrm=v3(8u*t+3u);let off=bitcast<f32>(inp[8u*t+6u]);let w=vec3f(bitcast<f32>(inp[8u*t+7u]),1.0,0.5);
 put(o+6u,nbTerrainPlaneBoxFraction(nrm,off,w));put(o+7u,geometricPlaneBoxFraction(nrm,off,w));`;
  await withParityDevice(async device => {
    const got = await runParity(device, { prelude, body }, { inp, threads: n, outWords: 8 * n, extra: { 3: columnsOf(t) } });
    let bad = 0;
    for (let i = 0; i < n; i++) {
      const o = 8 * i;
      if (got[o] !== got[o + 3] || got[o + 1] !== got[o + 4] || got[o + 2] !== got[o + 5] || got[o + 6] !== got[o + 7]) if (bad++ < 3) console.error(i, Array.from(got.subarray(o, o + 8)));
    }
    assert.equal(bad, 0, `${bad} of ${n} differ`);
  });
});

test("terrain: the hooks with the ground against the mirror (cell and face fractions, solid points)", { skip: !parityGPU }, async () => {
  const scene = sceneOf(SLOPE), t = brickTerrainColumns(scene)!;
  const box = sceneLatticeDimensions(scene) as unknown as [number, number, number], mask = brickStaticMask(scene, { always: true })!;
  const walls = { ...mask.walls, terrain: { base: mask.words.length, columns: t.columns } };
  const prelude = `${brickStaticWGSL({ box, openTop: false, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "ext3", walls } })}
@group(0) @binding(3) var<storage,read> ext3:array<u32>;`;
  const m = brickStaticMirror({ box, openTop: false, frame: [0, 0, 0], h: [1, 1, 1], mask: mask.words, terrain: t }), ground = brickTerrainMirror(t);
  const r = prng(5), cells: number[][] = [];
  for (let i = 0; i < 8192; i++) cells.push([Math.floor((box[0] + 2) * r()) - 1, Math.floor(24 * r()) - 1, Math.floor((box[2] + 2) * r()) - 1]);
  const n = cells.length, inp = new Uint32Array(6 * n), fw = new Float32Array(inp.buffer), pts: number[][] = [];
  cells.forEach((c, i) => { const p = c.map(v => Math.fround(v + r())); pts.push(p); inp.set(c.map(v => v >>> 0), 6 * i); fw.set(p, 6 * i + 3); });
  const words = new Uint32Array(mask.words.length + t.heights.length);
  words.set(mask.words); words.set(columnsOf(t), mask.words.length);
  const body = `let c=vec3i(bitcast<i32>(inp[6u*t]),bitcast<i32>(inp[6u*t+1u]),bitcast<i32>(inp[6u*t+2u]));let o=5u*t;
 put(o,nbStaticCellOpen(c));for(var a=0u;a<3u;a++){put(o+1u+a,nbStaticFaceOpen(c,a));}outp[o+4u]=select(0u,1u,nbStaticSolidAt(v3(6u*t+3u)));`;
  await withParityDevice(async device => {
    const got = await runParity(device, { prelude, body }, { inp, threads: n, outWords: 5 * n, extra: { 3: words } });
    const gf = new Float32Array(got.buffer);
    let bad = 0, worst = 0, solid = 0;
    cells.forEach((c, i) => {
      const o = 5 * i, want = [m.cellOpen(c), m.faceOpen(c, 0), m.faceOpen(c, 1), m.faceOpen(c, 2)];
      want.forEach((v, j) => { const d = Math.abs(v - gf[o + j]!); worst = Math.max(worst, d); if (d > 2e-5 && bad++ < 3) console.error(c, j, v, gf[o + j]); });
      // A point within float rounding of the ground may read either side: only a clear one counts.
      const depth = ground.boundary(pts[i]!)[3];
      if (Math.abs(depth) > 1e-4 && (got[o + 4] === 1) !== m.solidAt(pts[i]!)) { if (bad++ < 3) console.error("solid", pts[i], depth); }
      solid += got[o + 4]!;
    });
    assert.ok(solid > 0 && solid < n, "the points straddle the ground");
    assert.equal(bad, 0, `${bad} differ (worst fraction ${worst})`);
  });
});

test("terrain: the particle contact lands a fall on a planar ground at its radius and keeps the tangential velocity", { skip: !parityGPU }, async () => {
  const t = planar(16, 16, 4, 0.25, 0), m = brickTerrainMirror(t);
  const prelude = `${brickTerrainWGSL({ buffer: "ext3", base: 0, columns: t.columns })}
@group(0) @binding(3) var<storage,read> ext3:array<u32>;`;
  // From 1 cell above the ground to 1 cell into it (the walk's stride is at most half a cell an axis; the contact holds
  // for any segment), a velocity down the normal and along it.
  const r = prng(9), n = 256, inp = new Float32Array(10 * n), inset = 0.3;
  for (let i = 0; i < n; i++) {
    const x = 3 + 10 * r(), z = 3 + 10 * r(), g = 4 + 0.25 * x;
    inp.set([x, g + 1, z, x + 0.4 * r(), g - 1 + 0.5 * r(), z, 0.5, -3, 0.25, inset], 10 * i);
  }
  const body = `let a=v3(10u*t);let b=v3(10u*t+3u);let v=v3(10u*t+6u);let inset=bitcast<f32>(inp[10u*t+9u]);
 let c=nbTerrainContact(a,b,v,inset);let o=6u*t;put(o,c.position.x);put(o+1u,c.position.y);put(o+2u,c.position.z);put(o+3u,c.velocity.x);put(o+4u,c.velocity.y);put(o+5u,c.velocity.z);`;
  await withParityDevice(async device => {
    const got = new Float32Array((await runParity(device, { prelude, body }, { inp, threads: n, outWords: 6 * n, extra: { 3: columnsOf(t) } })).buffer);
    const nrm = [-0.25, 1, 0].map(v => v / Math.hypot(0.25, 1));
    for (let i = 0; i < n; i++) {
      const p = [got[6 * i]!, got[6 * i + 1]!, got[6 * i + 2]!], v = [got[6 * i + 3]!, got[6 * i + 4]!, got[6 * i + 5]!];
      const distance = m.boundary(p)[3] / Math.hypot(0.25, 1);
      // At the radius, or past it by the bisection's last interval (twelve halvings of the segment: under 1e-3 cells).
      assert.ok(distance > inset - 1e-5 && distance < inset + 1e-3, `${i}: ${distance} from the ground, the radius ${inset}`);
      // The normal component into the ground is gone; the tangential part is the input's.
      const vn = v[0]! * nrm[0]! + v[1]! * nrm[1]! + v[2]! * nrm[2]!;
      assert.ok(Math.abs(vn) < 1e-5, `${i}: vn ${vn}`);
      const v0 = [0.5, -3, 0.25], v0n = v0[0]! * nrm[0]! + v0[1]! * nrm[1]!;
      for (let a = 0; a < 3; a++) assert.ok(Math.abs(v[a]! - (v0[a]! - v0n * nrm[a]!)) < 1e-5);
    }
  });
});
