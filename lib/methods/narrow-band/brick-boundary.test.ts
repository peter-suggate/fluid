/** The brick engine's procedural boundary (brick-boundary.wgsl.ts), AGENTS.md's regressions for a new representation:
 * the mirror against the old engine's own field (records, samples, cell fractions: untouched fidelity), the stamp's
 * cut bound, a local voxel edit (its cells lose their records, every other record kept) and its undo (the image
 * restored word for word), the setup's image, and on the GPU the hooks against the mirror, the hit test (the segment's
 * first crossing) and the particle contact (skips without WEBGPU_NODE_MODULE). */
import test from "node:test";
import assert from "node:assert/strict";
import { brickBoundaryChangedCells, brickBoundaryCutBricks, brickBoundaryImage, brickBoundaryMirror } from "./brick-boundary.wgsl";
import { NB_STATIC, brickStaticMask, brickStaticMirror, brickStaticWGSL } from "./brick-static.wgsl";
import { brickSceneSetup } from "./brick-scene";
import { parityGPU, prng, runParity, withParityDevice } from "./parity/harness";
import { proceduralSolidField } from "../../core/procedural-solid";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { getSceneDefinition } from "../../core/scenes";
import { sceneDocument } from "../../core/scene-definition";
import { resolveMethodValues } from "../../core/method-contract";
import { uniformNarrowBandMethod as method } from "../uniform/uniform-narrow-band-method";
import type { SceneDescription } from "../../core/model";
import type { SolidWorldVoxelPatch } from "../../core/solid-world";

const SLIDE = "nbflip-water-slide";
const sceneOf = (id: string): SceneDescription => {
  const d = getSceneDefinition(id) as unknown as { create?: () => SceneDescription };
  return structuredClone(d.create ? d.create() : sceneDocument(d as never));
};
type V = [number, number, number];

/** The field's record cells (the old engine's at()), sorted for sampling. */
function recordCells(scene: SceneDescription): V[] {
  const field = proceduralSolidField(scene)!, out: V[] = [];
  for (const b of scene.solidBoundary!.bricks) for (let i = 0; i < 512; i++) {
    const c: V = [b.origin[0] + (i & 7), b.origin[1] + ((i >> 3) & 7), b.origin[2] + (i >> 6)];
    if (field.at(c)) out.push(c);
  }
  return out;
}

function mirrorOf(scene: SceneDescription, image = brickBoundaryImage(scene)!) {
  const box = sceneLatticeDimensions(scene) as unknown as V, mask = brickStaticMask(scene, { always: true })!;
  return { box, mask, image, m: brickStaticMirror({ box, openTop: scene.container.top === "open", frame: [0, 0, 0], h: [1, 1, 1], mask: mask.words, boundary: image }) };
}

test("boundary: the mirror's records, samples and cell fractions are the old engine's field (to float32)", () => {
  const scene = sceneOf(SLIDE), field = proceduralSolidField(scene)!, { box, image, m } = mirrorOf(scene);
  const f = brickBoundaryMirror(image, c => c.every((v, a) => v >= 0 && v < box[a]!), () => false);
  const r = prng(3), cells = recordCells(scene);
  assert.ok(cells.length > 1000);
  let worst = 0;
  for (let i = 0; i < 4096; i++) {
    const c = cells[Math.floor(r() * cells.length)]!, q: V = [c[0] + r(), c[1] + r(), c[2] + r()];
    assert.notEqual(f.record(c), 0, `${c}: a record`);
    assert.equal(m.cellKind(c), NB_STATIC.BOUNDARY);
    const want = field.sample(q)!, got = f.sample(q);
    for (let a = 0; a < 4; a++) worst = Math.max(worst, Math.abs(want[a]! - got[a]!) / Math.max(1, Math.abs(want[a]!)));
    // umCellOpenGeometry with no terrain: the field's unit box (solidNodesOpen, which clamps to [0, 1]).
    assert.ok(Math.abs(Math.min(1, Math.max(0, m.cellOpen(c))) - field.open(c)!) < 1e-5, `${c}: open ${m.cellOpen(c)} vs ${field.open(c)}`);
  }
  assert.ok(worst < 1e-5, `worst relative sample difference ${worst}`);
  // A cell the image holds no record for: a voxel 0, else open (the mask's own rule).
  for (let i = 0; i < 4096; i++) {
    const c: V = [Math.floor(box[0] * r()), Math.floor(box[1] * r()), Math.floor(box[2] * r())];
    if (field.at(c)) continue;
    assert.equal(f.record(c), 0);
    assert.ok(m.cellKind(c) !== NB_STATIC.BOUNDARY);
  }
});

test("boundary: the stamp's cut bound holds every brick a record cuts (a record cell, or a lower face it touches)", () => {
  const scene = sceneOf(SLIDE), { box, image, m } = mirrorOf(scene), bricks = new Set<string>();
  brickBoundaryCutBricks(image, box, (x, y, z) => bricks.add(`${x},${y},${z}`));
  for (const c of recordCells(scene)) {
    for (let a = 0; a < 3; a++) {
      assert.ok(m.faceBoxed(c, a), `${c}: its +face ${a} is boxed`);
      const n = c.map((v, b) => v + (a === b ? 1 : 0));
      if (n.every((v, b) => v < box[b]!)) assert.ok(bricks.has(n.map(v => v >> 3).join()), `${n}: its lower face touches ${c}`);
    }
    assert.ok(bricks.has(c.map(v => v >> 3).join()), `${c}: its brick`);
  }
});

test("boundary: a local voxel edit drops exactly its cells' records and keeps the rest; undo restores the image word for word", () => {
  const scene = sceneOf(SLIDE), before = brickBoundaryImage(scene)!, cells = recordCells(scene);
  // A small block over a record cell, appended after the generated prefix (the editor's stroke).
  const at = cells[Math.floor(cells.length / 2)]!, lo: V = [at[0] - 1, at[1] - 1, at[2] - 1], hi: V = [at[0] + 2, at[1] + 2, at[2] + 2];
  const edit: SolidWorldVoxelPatch = { operation: "clear", minimum: lo, maximumExclusive: hi } as SolidWorldVoxelPatch;
  const edited = { ...scene, solidVoxels: [...scene.solidVoxels, edit] } as SceneDescription;
  const after = brickBoundaryImage(edited)!;
  assert.equal(after.length, before.length, "the same bricks");
  const moved: V[] = [];
  brickBoundaryChangedCells(before, after, (x, y, z) => moved.push([x, y, z]));
  assert.ok(moved.length > 0, "the edit drops records");
  for (const c of moved) assert.ok(c.every((v, a) => v >= lo[a]! && v < hi[a]!), `${c}: inside the edit`);
  // Fidelity outside the edit: every word but the edited cells' validity bits is unchanged.
  let differ = 0;
  for (let i = 0; i < before.length; i++) if (before[i] !== after[i]) differ++;
  assert.ok(differ > 0 && differ <= moved.length);
  // Undo: the stroke removed, the image is the original's.
  assert.deepEqual(brickBoundaryImage({ ...edited, solidVoxels: scene.solidVoxels } as SceneDescription), before);
});

test("boundary: the setup carries the image for a field-mode scene, and refuses a field off the container's lattice", () => {
  const scene = sceneOf(SLIDE);
  const profile = getSceneDefinition(SLIDE).methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
  const { setup, gaps } = brickSceneSetup(scene, values);
  assert.deepEqual(gaps, []);
  assert.deepEqual(setup!.boundary, brickBoundaryImage(scene));
  assert.equal(setup!.staticsDeferred, undefined, "the field is solid at the build");
  const b = scene.solidBoundary!;
  assert.throws(() => brickBoundaryImage({ ...scene, solidBoundary: { ...b, dimensions: [b.dimensions[0] + 8, b.dimensions[1], b.dimensions[2]] } } as SceneDescription), /lattice/);
  assert.equal(brickBoundaryImage({ ...scene, solidBoundary: { ...b, collision: "voxels" } } as SceneDescription), undefined);
});

/** Cells about the field's surface: record cells and their neighbours, and some far away. */
function probeCells(scene: SceneDescription, box: V, n: number, seed: number): V[] {
  const r = prng(seed), cells = recordCells(scene), out: V[] = [];
  for (let i = 0; i < n; i++) {
    if (i % 4 === 3) { out.push([Math.floor((box[0] + 2) * r()) - 1, Math.floor((box[1] + 2) * r()) - 1, Math.floor((box[2] + 2) * r()) - 1]); continue; }
    const c = cells[Math.floor(r() * cells.length)]!;
    out.push([c[0] + Math.floor(3 * r()) - 1, c[1] + Math.floor(3 * r()) - 1, c[2] + Math.floor(3 * r()) - 1]);
  }
  return out;
}

test("boundary: the hooks with the field against the mirror (kind, cell and face fractions, boxed faces, solid points)", { skip: !parityGPU }, async () => {
  const scene = sceneOf(SLIDE), { box, mask, image, m } = mirrorOf(scene);
  const walls = { ...mask.walls, boundary: { base: mask.words.length } };
  const prelude = `${brickStaticWGSL({ box, openTop: false, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "ext3", walls } })}
@group(0) @binding(3) var<storage,read> ext3:array<u32>;`;
  const r = prng(7), cells = probeCells(scene, box, 8192, 11);
  const n = cells.length, inp = new Uint32Array(6 * n), fw = new Float32Array(inp.buffer), pts: number[][] = [];
  cells.forEach((c, i) => { const p = c.map(v => Math.fround(v + r())); pts.push(p); inp.set(c.map(v => v >>> 0), 6 * i); fw.set(p, 6 * i + 3); });
  const words = new Uint32Array(mask.words.length + image.length);
  words.set(mask.words); words.set(image, mask.words.length);
  const body = `let c=vec3i(bitcast<i32>(inp[6u*t]),bitcast<i32>(inp[6u*t+1u]),bitcast<i32>(inp[6u*t+2u]));let o=9u*t;
 outp[o]=nbStaticCellKind(c);put(o+1u,nbStaticCellOpen(c));
 for(var a=0u;a<3u;a++){put(o+2u+a,nbStaticFaceOpen(c,a));outp[o+5u+a]=select(0u,1u,nbStaticFaceBoxed(c,a));}
 outp[o+8u]=select(0u,1u,nbStaticSolidAt(v3(6u*t+3u)));`;
  await withParityDevice(async device => {
    const got = await runParity(device, { prelude, body }, { inp, threads: n, outWords: 9 * n, extra: { 3: words } });
    const gf = new Float32Array(got.buffer);
    let bad = 0, worst = 0, field = 0, solid = 0;
    cells.forEach((c, i) => {
      const o = 9 * i;
      if (got[o] !== m.cellKind(c) && bad++ < 3) console.error("kind", c, got[o], m.cellKind(c));
      field += got[o] === NB_STATIC.BOUNDARY ? 1 : 0;
      const want = [m.cellOpen(c), m.faceOpen(c, 0), m.faceOpen(c, 1), m.faceOpen(c, 2)];
      want.forEach((v, j) => { const d = Math.abs(v - gf[o + 1 + j]!); worst = Math.max(worst, d); if (d > 2e-5 && bad++ < 3) console.error(c, j, v, gf[o + 1 + j]); });
      for (let a = 0; a < 3; a++) if ((got[o + 5 + a] === 1) !== m.faceBoxed(c, a) && bad++ < 3) console.error("boxed", c, a);
      // A point within float rounding of the surface may read either side: only a clear one counts.
      const s = brickBoundaryMirror(image, q => q.every((v, a) => v >= 0 && v < box[a]!), () => false).sample(pts[i]!);
      if (Math.abs(s[3]!) > 1e-4 && (got[o + 8] === 1) !== m.solidAt(pts[i]!) && bad++ < 3) console.error("solid", pts[i], s[3]);
      solid += got[o + 8]!;
    });
    assert.ok(field > n / 4, `${field} record cells probed`);
    assert.ok(solid > 0 && solid < n, "the points straddle the surface");
    assert.equal(bad, 0, `${bad} differ (worst fraction ${worst})`);
  });
});

test("boundary: the hit test finds a segment's first crossing, and the contact leaves the sample outside the field at its radius", { skip: !parityGPU }, async () => {
  const scene = sceneOf(SLIDE), { box, mask, image } = mirrorOf(scene);
  const f = brickBoundaryMirror(image, q => q.every((v, a) => v >= 0 && v < box[a]!), () => false);
  const walls = { ...mask.walls, boundary: { base: mask.words.length } };
  const prelude = `${brickStaticWGSL({ box, openTop: false, frame: "vec3f(0.0)", h: "vec3f(1.0)", mask: { buffer: "ext3", walls } })}
@group(0) @binding(3) var<storage,read> ext3:array<u32>;`;
  // Segments from a point outside the field (value > 0.5 cell) half a cell toward and into it along its normal.
  const r = prng(13), cells = recordCells(scene), segs: number[][] = [], inset = 0.3;
  while (segs.length < 2048) {
    const c = cells[Math.floor(r() * cells.length)]!, q = [c[0] + r(), c[1] + r(), c[2] + r()], s = f.sample(q);
    const g = Math.hypot(s[0]!, s[1]!, s[2]!); if (g < 0.5) continue;
    const d = s[3]! / g; if (d < 0.35 || d > 0.6) continue;
    const nrm = [s[0]! / g, s[1]! / g, s[2]! / g], end = q.map((v, a) => v - (d + 0.3) * nrm[a]!);
    segs.push([...q, ...end, -nrm[0]!, -nrm[1]!, -nrm[2]!, d]);
  }
  const n = segs.length, inp = new Float32Array(10 * n);
  segs.forEach((s, i) => inp.set(s, 10 * i));
  const body = `let a=v3(10u*t);let b=v3(10u*t+3u);let v=v3(10u*t+6u);let o=8u*t;
 let hit=nbBoundaryHit(a,b);put(o,hit);
 let c=nbProceduralContact(a,b,v,${inset});put(o+1u,c.position.x);put(o+2u,c.position.y);put(o+3u,c.position.z);
 put(o+4u,c.velocity.x);put(o+5u,c.velocity.y);put(o+6u,c.velocity.z);let s=nbBoundarySample(c.position);put(o+7u,s.w/max(length(s.xyz),1e-12));`;
  await withParityDevice(async device => {
    const got = new Float32Array((await runParity(device, { prelude, body }, { inp: new Uint32Array(inp.buffer), threads: n, outWords: 8 * n, extra: { 3: new Uint32Array([...mask.words, ...image]) } })).buffer);
    let missed = 0, inside = 0, worstV = 0, offSurface = 0;
    for (let i = 0; i < n; i++) {
      const o = 8 * i, s = segs[i]!, hit = got[o]!;
      // The crossing lies where the field changes sign along the segment: before it positive, at it about zero.
      if (!(hit >= 0)) { missed++; continue; }
      const at = [0, 1, 2].map(a => s[a]! + hit * (s[3 + a]! - s[a]!)), w = f.sample(at), g = Math.hypot(w[0]!, w[1]!, w[2]!);
      // The walk's step into the next cell (floor(p + sign(delta) 1e-6)) is below float32's resolution past 16 cells, so
      // a crossing just past a face may be the cell before's trilinear, extrapolated (the old engine's text, kept
      // verbatim; probed 2026-11-18: 0.03-0.18 cell off). Most hits are exact; none is far from the surface.
      const off = Math.abs(w[3]! / g);
      assert.ok(off < 0.5, `${i}: the hit's distance ${off}`);
      if (off > 0.02) offSurface++;
      if (got[o + 7]! < inset - 1e-2) inside++;
      // The velocity into the surface is gone: what is left points along it or out.
      const v = [got[o + 4]!, got[o + 5]!, got[o + 6]!];
      worstV = Math.max(worstV, Math.max(0, -(v[0]! * w[0]! + v[1]! * w[1]! + v[2]! * w[2]!) / g));
    }
    assert.ok(offSurface < n / 20, `${offSurface} of ${n} hits more than 0.02 cell off the surface`);
    assert.ok(missed < n / 100, `${missed} of ${n} segments into the field found no crossing`);
    assert.ok(inside < n / 50, `${inside} of ${n} contacts end inside the radius`);
    assert.ok(worstV < 0.05, `the worst inward velocity left ${worstV}`);
  });
});
