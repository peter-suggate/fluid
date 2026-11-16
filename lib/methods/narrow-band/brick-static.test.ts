/** The brick engine's static solid queries (brick-static.wgsl.ts, M3): the box rule (stage 1) and the static voxel
 * mask (stage 2a) in the Float64 mirror; the AGENTS.md regressions for the mask (hit testing, local edits, untouched
 * fidelity, save/load/undo, unknown kinds); and the WGSL against the mirror bit for bit (GPU; skips without
 * WEBGPU_NODE_MODULE). */
import test from "node:test";
import assert from "node:assert/strict";
import { brickStaticMask, brickStaticMirror, brickStaticWGSL, NB_STATIC } from "./brick-static.wgsl";
import { brickSceneSetup } from "./brick-scene";
import { parityGPU, prng, runParity, withParityDevice } from "./parity/harness";
import { createReservoirSeepScene } from "../../core/reservoir-seep-scene";
import { sceneWithSolidStroke, solidWorldForFluidLattice, type SolidWorldVoxelPatch } from "../../core/solid-world";
import { solidOccupancyMaskVoxel, type SolidOccupancyMask } from "../../core/solid-occupancy-mask";
import { voxelReaderForScene } from "../../core/voxel-editor/voxel-reader";
import { pickSolidVoxel } from "../../core/editor-solid-voxel";
import { sceneCellSizes_m, sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import type { SceneDescription } from "../../core/model";

const BOX = [12, 8, 4] as const, FRAME = [0.1, -0.2, 0.3] as const, H = [0.05, 0.05, 0.05] as const;

test("statics: the box rule, closed and open top", () => {
  for (const openTop of [false, true]) {
    const m = brickStaticMirror({ box: BOX, openTop, frame: FRAME, h: H });
    assert.equal(m.cellKind([0, 0, 0]), NB_STATIC.PLAIN);
    assert.equal(m.cellKind([11, 7, 3]), NB_STATIC.PLAIN);
    for (const c of [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [12, 0, 0], [0, 0, 4]]) {
      assert.equal(m.cellKind(c), NB_STATIC.VOXEL, `${c}`); assert.equal(m.cellOpen(c), 0);
    }
    // Above the box: solid under a closed top, air under an open one.
    assert.equal(m.cellOpen([3, 8, 1]), openTop ? 1 : 0);
    assert.equal(m.cellOpen([3, 40, 1]), openTop ? 1 : 0);
    // Faces: the +axis face of c; a wall face is closed.
    assert.equal(m.faceOpen([10, 3, 1], 0), 1);
    assert.equal(m.faceOpen([11, 3, 1], 0), 0);
    assert.equal(m.faceOpen([3, 7, 1], 1), openTop ? 1 : 0);
    assert.equal(m.faceOpen([-1, 3, 1], 0), 0);
    assert.equal(m.faceBoxed([3, 3, 1], 2), false);
    // World points: cell (0,0,0) spans frame .. frame + h.
    assert.equal(m.solidAt([0.1 + 0.025, -0.2 + 0.025, 0.3 + 0.025]), false);
    assert.equal(m.solidAt([0.1 - 0.001, -0.2 + 0.025, 0.3 + 0.025]), true);
    assert.equal(m.solidAt([0.1 + 0.025, -0.2 + 8 * 0.05 + 0.01, 0.3 + 0.025]), !openTop);
  }
});

test("statics: WGSL against the mirror", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    for (const openTop of [false, true]) {
      const m = brickStaticMirror({ box: BOX, openTop, frame: FRAME.map(Math.fround) as unknown as [number, number, number], h: H.map(Math.fround) as unknown as [number, number, number] });
      const prelude = brickStaticWGSL({ box: BOX, openTop, frame: `vec3f(${FRAME.join(",")})`, h: `vec3f(${H.join(",")})` });
      // Every cell of the box grown by 2 on each side (and 30 above), each axis; then random world points.
      const cells: number[][] = [];
      for (let z = -2; z < BOX[2] + 2; z++) for (let y = -2; y < BOX[1] + 30; y++) for (let x = -2; x < BOX[0] + 2; x++) cells.push([x, y, z]);
      const r = prng(7), worlds: number[][] = [];
      for (let i = 0; i < cells.length; i++) worlds.push(FRAME.map((f, a) => Math.fround(f + (r() * (BOX[a]! + (a === 1 ? 34 : 4)) - 2) * H[a]!)));
      const n = cells.length, inp = new Uint32Array(6 * n), fw = new Float32Array(inp.buffer);
      cells.forEach((c, i) => { inp[6 * i] = c[0]! >>> 0; inp[6 * i + 1] = c[1]! >>> 0; inp[6 * i + 2] = c[2]! >>> 0; fw.set(worlds[i]!, 6 * i + 3); });
      const OUT = 9;
      const body = `let c=vec3i(bitcast<i32>(inp[6u*t]),bitcast<i32>(inp[6u*t+1u]),bitcast<i32>(inp[6u*t+2u]));let w=v3(6u*t+3u);let o=${OUT}u*t;
 outp[o]=nbStaticCellKind(c);put(o+1u,nbStaticCellOpen(c));
 for(var a=0u;a<3u;a++){put(o+2u+a,nbStaticFaceOpen(c,a));outp[o+5u+a]=select(0u,1u,nbStaticFaceBoxed(c,a));}
 outp[o+8u]=select(0u,1u,nbStaticSolidAt(w));`;
      const got = await runParity(device, { prelude, body }, { inp, threads: n, outWords: OUT * n });
      const gf = new Float32Array(got.buffer);
      let bad = 0;
      cells.forEach((c, i) => {
        const o = OUT * i, want = [m.cellKind(c), m.cellOpen(c), m.faceOpen(c, 0), m.faceOpen(c, 1), m.faceOpen(c, 2), 0, 0, 0, m.solidAt(worlds[i]!) ? 1 : 0];
        const have = [got[o]!, gf[o + 1]!, gf[o + 2]!, gf[o + 3]!, gf[o + 4]!, got[o + 5]!, got[o + 6]!, got[o + 7]!, got[o + 8]!];
        if (want.some((v, j) => v !== have[j])) { if (bad++ < 3) console.error(c, worlds[i], want, have); }
      });
      assert.equal(bad, 0, `openTop ${openTop}: ${bad} of ${n} differ`);
    }
  });
});

// ---- Stage 2a: the static voxel mask over the seep (512 x 256 x 64, an open top, a wall with a hole). ----

const seep = createReservoirSeepScene();
const SEEP = sceneLatticeDimensions(seep) as unknown as [number, number, number];
const wall = seep.solidVoxels.find(p => p.operation === "fill")!, hole = seep.solidVoxels.find(p => p.operation === "clear")!;
const frameOf = (s: SceneDescription) => [-0.5 * s.container.width_m, 0, -0.5 * s.container.depth_m] as [number, number, number];
const hOf = (s: SceneDescription) => sceneCellSizes_m(s) as unknown as [number, number, number];
const mirrorOf = (s: SceneDescription, mask: SolidOccupancyMask | undefined) =>
  brickStaticMirror({ box: SEEP, openTop: s.container.top === "open", frame: frameOf(s), h: hOf(s), mask: mask?.words });
// The edit: a round clear through the wall's face (partial cells: its kind's own coverage), and a box filled in the
// left chamber's floor.
const lo = wall.minimum[0], hi = wall.maximumExclusive[0];
const EDIT: SolidWorldVoxelPatch[] = [
  { operation: "clear", kind: "ellipsoid", minimum: [lo - 3, 150, 20], maximumExclusive: [lo + 5, 162, 32] },
  { operation: "fill", minimum: [40, 0, 10], maximumExclusive: [52, 6, 18], materialId: 2 },
];
const editBox = { lo: [lo - 4, 0, 9], hi: [lo + 6, 163, 33] }, fillBox = { lo: [39, 0, 9], hi: [53, 7, 19] };
const near = (c: readonly number[], b: { lo: number[]; hi: number[] }) => c.every((v, a) => v >= b.lo[a]! && v < b.hi[a]!);
const eachCell = (visit: (x: number, y: number, z: number) => void) => {
  for (let z = 0; z < SEEP[2]; z++) for (let y = 0; y < SEEP[1]; y++) for (let x = 0; x < SEEP[0]; x++) visit(x, y, z);
};

test("statics 2a: the mask is the shared SolidWorld the editor's hit test reads, cell for cell", () => {
  const mask = brickStaticMask(seep)!, reader = voxelReaderForScene(seep), m = mirrorOf(seep, mask);
  assert.ok(mask && !mask.empty);
  assert.ok(mask.words.length * 4 < 3 * 2 ** 20, `the image is the tables and the solid pages: ${mask.words.length * 4} bytes`);
  let bad = 0, solid = 0;
  eachCell((x, y, z) => {
    const want = reader.solid([x, y, z]), have = solidOccupancyMaskVoxel(mask.words, x, y, z);
    if (want) solid++;
    if (want !== have && bad++ < 3) console.error([x, y, z], want, have);
  });
  assert.equal(bad, 0); assert.ok(solid > 0);
  // The hooks: the wall is VOXEL, its hole and both chambers PLAIN; a wall face is the voxel-wall face.
  const yz = [10, 10] as const;
  assert.equal(m.cellKind([lo, yz[0], yz[1]]), NB_STATIC.VOXEL); assert.equal(m.cellKind([lo - 1, yz[0], yz[1]]), NB_STATIC.PLAIN);
  assert.equal(m.cellKind([lo, hole.minimum[1], hole.minimum[2]]), NB_STATIC.PLAIN);
  assert.equal(m.faceOpen([lo - 1, yz[0], yz[1]], 0), 0.5); assert.equal(m.faceOpen([lo, yz[0], yz[1]], 0), 0);
  assert.equal(m.faceOpen([lo - 2, yz[0], yz[1]], 0), 1);
  // Open top: air above the box reads no mask.
  assert.equal(m.cellKind([lo, SEEP[1] + 3, 5]), NB_STATIC.PLAIN);
  // The editor's nearest-surface pick along +x meets the wall where the hooks first say solid.
  const frame = frameOf(seep), h = hOf(seep), at = (c: number[]) => c.map((v, a) => frame[a]! + (v + 0.5) * h[a]!);
  for (const [y, z] of [[10, 10], [hole.minimum[1] - 1, hole.minimum[2] + 1], [200, 60]] as const) {
    const o = at([5, y, z]), hit = pickSolidVoxel(seep, { origin: { x: o[0]!, y: o[1]!, z: o[2]! }, direction: { x: 1, y: 0, z: 0 } })!;
    assert.ok(hit, `a pick at y ${y} z ${z}`);
    assert.deepEqual([...hit.coordinate], [lo, y, z]);
    let first = 5; while (m.cellKind([first, y, z]) !== NB_STATIC.VOXEL) first++;
    assert.equal(first, hit.coordinate[0]);
    assert.equal(m.solidAt(at([...hit.coordinate])), true); assert.equal(m.solidAt(at([lo - 1, y, z])), false);
  }
});

test("statics 2a: a local edit moves only its cells and their faces; the rest keeps every bit", () => {
  const base = brickStaticMask(seep)!, edited = sceneWithSolidStroke(seep, EDIT);
  // The edit keeps the scene's authored source: every base patch, in order, then the stroke.
  assert.deepEqual(edited.solidVoxels.slice(0, seep.solidVoxels.length), seep.solidVoxels);
  const after = brickStaticMask(edited)!, live = brickStaticMask(seep)!;
  const moved = new Set<string>();
  const ranges = live.update(solidWorldForFluidLattice(edited), (x, y, z) => { moved.add(`${x},${y},${z}`); });
  assert.ok(ranges && ranges.length > 0);
  const mb = mirrorOf(seep, base), ma = mirrorOf(edited, after), reader = voxelReaderForScene(edited);
  let differ = 0, outside = 0, reads = 0, incremental = 0;
  eachCell((x, y, z) => {
    const c = [x, y, z], was = solidOccupancyMaskVoxel(base.words, x, y, z), is = solidOccupancyMaskVoxel(after.words, x, y, z);
    if (is !== reader.solid([x, y, z])) reads++;
    if (is !== solidOccupancyMaskVoxel(live.words, x, y, z)) incremental++;
    if (was !== is) { differ++; if (!moved.has(`${x},${y},${z}`)) outside++; if (!near(c, editBox) && !near(c, fillBox)) outside++; }
  });
  assert.equal(reads, 0, "the edited mask is the edited world the hit test reads");
  assert.equal(incremental, 0, "the mask's update is the rebuild");
  assert.ok(differ > 0); assert.equal(differ, moved.size, "update names each moved cell once"); assert.equal(outside, 0);
  // Hooks: around the edit, a cell's kind and open fraction move only where its bit moved, a face only where one side did.
  let hooks = 0;
  for (const b of [editBox, fillBox]) for (let z = b.lo[2]! - 2; z < b.hi[2]! + 2; z++) for (let y = Math.max(0, b.lo[1]! - 2); y < b.hi[1]! + 2; y++) for (let x = b.lo[0]! - 2; x < b.hi[0]! + 2; x++) {
    const c = [x, y, z], hit = (q: number[]) => moved.has(q.join(","));
    if ((mb.cellKind(c) !== ma.cellKind(c) || mb.cellOpen(c) !== ma.cellOpen(c)) !== hit(c)) hooks++;
    for (let a = 0; a < 3; a++) {
      const n = c.map((v, k) => v + (k === a ? 1 : 0));
      if (mb.faceOpen(c, a) !== ma.faceOpen(c, a) && !hit(c) && !hit(n)) hooks++;
    }
  }
  assert.equal(hooks, 0);
});

test("statics 2a: save/load is bitwise, undo restores every cell, unknown kinds and unread solids fail explicitly", () => {
  const edited = sceneWithSolidStroke(seep, EDIT), loaded = JSON.parse(JSON.stringify(edited)) as SceneDescription;
  assert.deepEqual(brickStaticMask(loaded)!.words, brickStaticMask(edited)!.words, "a saved and loaded scene builds the same image");
  // Undo: the document returns to the base scene; the live mask is told so and reads as the base, cell for cell.
  const base = brickStaticMask(seep)!, live = brickStaticMask(seep)!;
  live.update(solidWorldForFluidLattice(edited));
  let back = 0;
  live.update(solidWorldForFluidLattice(seep), () => { back++; });
  let bad = 0;
  eachCell((x, y, z) => { if (solidOccupancyMaskVoxel(live.words, x, y, z) !== solidOccupancyMaskVoxel(base.words, x, y, z)) bad++; });
  assert.equal(bad, 0); assert.ok(back > 0);
  assert.equal(live.slotsHeld, base.slotsHeld, "undo gives its slots back");
  // An unknown patch kind is fatal, never drawn or stamped as something else.
  assert.throws(() => brickStaticMask(sceneWithSolidStroke(seep, [{ operation: "fill", kind: "no-such-kind", minimum: [40, 0, 10], maximumExclusive: [44, 4, 14] }])), /Unknown solid patch kind/);
  // A scene without voxels beyond the shell binds no mask.
  const plain = { ...seep, solidVoxels: [] } as SceneDescription;
  assert.equal(brickStaticMask(plain), undefined);
  // Solids the hooks do not read stay named gaps: terrain, and a procedural boundary sampled as a field.
  const gaps = (s: SceneDescription) => brickSceneSetup(s, {}).gaps.join(" | ");
  assert.match(gaps({ ...plain, terrain: {} as SceneDescription["terrain"] }), /terrain needs/);
  assert.match(gaps({ ...plain, solidBoundary: {} as SceneDescription["solidBoundary"] }), /procedural boundary sampled as a field/);
  assert.doesNotMatch(gaps({ ...plain, solidBoundary: { collision: "voxels" } as SceneDescription["solidBoundary"] }), /procedural boundary/);
});

test("statics 2a: WGSL with the mask against the mirror, base and edited", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    for (const scene of [seep, sceneWithSolidStroke(seep, EDIT)]) {
      // The mirror in the shader's f32 frame and cell size, so a world point floors to the same cell.
      const mask = brickStaticMask(scene)!, frame = frameOf(scene).map(Math.fround), h = hOf(scene).map(Math.fround);
      const mf = brickStaticMirror({ box: SEEP, openTop: true, frame: frame as [number, number, number], h: h as [number, number, number], mask: mask.words });
      const prelude = brickStaticWGSL({ box: SEEP, openTop: true, frame: `vec3f(${frame.join(",")})`, h: `vec3f(${h.join(",")})`, mask: { buffer: "aux", walls: mask.walls } });
      // Every cell around the wall's hole and the edits, and random cells in and past the box; random world points.
      const cells: number[][] = [];
      for (const b of [{ lo: [lo - 3, hole.minimum[1] - 3, hole.minimum[2] - 3], hi: [hi + 3, hole.maximumExclusive[1] + 3, hole.maximumExclusive[2] + 3] }, editBox, fillBox])
        for (let z = b.lo[2]! - 1; z < b.hi[2]! + 1; z++) for (let y = b.lo[1]! - 1; y < b.hi[1]! + 1; y++) for (let x = b.lo[0]! - 1; x < b.hi[0]! + 1; x++) cells.push([x, y, z]);
      const r = prng(11);
      for (let i = 0; i < 50000; i++) cells.push(SEEP.map(n => Math.floor((r() * 1.1 - 0.05) * n)));
      const worlds = cells.map(() => frame.map((f, a) => Math.fround(f + (r() * 1.1 - 0.05) * SEEP[a]! * h[a]!)));
      const n = cells.length, inp = new Uint32Array(6 * n), fw = new Float32Array(inp.buffer);
      cells.forEach((c, i) => { inp[6 * i] = c[0]! >>> 0; inp[6 * i + 1] = c[1]! >>> 0; inp[6 * i + 2] = c[2]! >>> 0; fw.set(worlds[i]!, 6 * i + 3); });
      const OUT = 9;
      const body = `let c=vec3i(bitcast<i32>(inp[6u*t]),bitcast<i32>(inp[6u*t+1u]),bitcast<i32>(inp[6u*t+2u]));let w=v3(6u*t+3u);let o=${OUT}u*t;
 outp[o]=nbStaticCellKind(c);put(o+1u,nbStaticCellOpen(c));
 for(var a=0u;a<3u;a++){put(o+2u+a,nbStaticFaceOpen(c,a));outp[o+5u+a]=select(0u,1u,nbStaticFaceBoxed(c,a));}
 outp[o+8u]=select(0u,1u,nbStaticSolidAt(w));`;
      const got = await runParity(device, { prelude, body }, { inp, aux: mask.words, threads: n, outWords: OUT * n });
      const gf = new Float32Array(got.buffer);
      let bad = 0, voxels = 0;
      cells.forEach((c, i) => {
        const o = OUT * i, want = [mf.cellKind(c), mf.cellOpen(c), mf.faceOpen(c, 0), mf.faceOpen(c, 1), mf.faceOpen(c, 2), 0, 0, 0, mf.solidAt(worlds[i]!) ? 1 : 0];
        const have = [got[o]!, gf[o + 1]!, gf[o + 2]!, gf[o + 3]!, gf[o + 4]!, got[o + 5]!, got[o + 6]!, got[o + 7]!, got[o + 8]!];
        if (want[0] === NB_STATIC.VOXEL) voxels++;
        if (want.some((v, j) => v !== have[j])) { if (bad++ < 3) console.error(c, worlds[i], want, have); }
      });
      assert.ok(voxels > 1000);
      assert.equal(bad, 0, `${bad} of ${n} differ`);
    }
  });
});
