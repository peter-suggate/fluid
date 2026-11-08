import assert from 'node:assert/strict';
import test from 'node:test';
import { createEmptyScene } from '../lib/core/empty-scene';
import type { SceneDescription } from '../lib/core/model';
import { VOXEL_MATERIAL_IDS } from '../lib/core/voxel-scene';
import { CLOTH_CONTAINER_SHELL_FRICTION, clothSolidBand, clothSolidSourceForScene, sceneLatticeOrigin_m } from '../lib/cloth/contact/scene-solids';
import { SOLID_BRICK_CELLS, buildClothSolidField, sampleClothSolidField, type ClothSolidSource, type V3 } from '../lib/cloth/contact/solid-field';
import { bridson2002Curtain, bridson2002RotatingSphere, bridson2002Tablecloth, bridsonSolidPose, type Bridson2002Scene } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription, bridson2002SceneLayout } from '../lib/cloth/scenes/bridson-2002/scene-description';

const BAND = 0.06;
const sources = [bridson2002Curtain(), bridson2002RotatingSphere(), bridson2002Tablecloth()];
const scenes = new Map<string, SceneDescription>();
const sceneOf = (s: Bridson2002Scene) => { let d = scenes.get(s.id); if (!d) scenes.set(s.id, d = bridson2002SceneDescription(s)); return d; };
const sourcesOf = new Map<string, ClothSolidSource>();
const solidsOf = (s: Bridson2002Scene) => { let d = sourcesOf.get(s.id); if (!d) sourcesOf.set(s.id, d = clothSolidSourceForScene(sceneOf(s), { band_m: BAND })); return d; };
const toScene = (s: Bridson2002Scene, p: { x: number; y: number; z: number }): V3 => {
  const t = bridson2002SceneLayout(s).translation_m;
  return [p.x + t.x, p.y + t.y, p.z + t.z];
};
/** The field over a sub-box of the scene source, on the same lattice. */
const fieldNear = (src: ClothSolidSource, c: V3, half: number) => buildClothSolidField({ ...src,
  lo: c.map((v, a) => Math.max(src.lo[a]!, v - half)) as unknown as V3, hi: c.map((v, a) => Math.min(src.hi[a]!, v + half)) as unknown as V3 });

test('Bridson scene solids: shell and ground as lattice boxes, held shaped solids as sdf parts with their friction', () => {
  for (const s of sources) {
    const scene = sceneOf(s), src = solidsOf(s), layout = bridson2002SceneLayout(s);
    const boxes = src.parts!.filter(p => p.kind === 'lattice-box'), sdfs = src.parts!.filter(p => p.kind === 'sdf');
    assert.deepEqual(src.origin, sceneLatticeOrigin_m(scene), `${s.id} field on the scene lattice`);
    assert.equal(src.cell, layout.cell_m);
    const ground = s.solids.find(b => b.id === 'ground')!;
    assert.ok(boxes.filter(b => b.mu === CLOTH_CONTAINER_SHELL_FRICTION).length >= 6, `${s.id} shell`);
    assert.ok(ground.friction > 0 && boxes.filter(b => b.mu === ground.friction).length === 1, `${s.id} ground`);
    const shaped = s.solids.filter(b => b.id !== 'ground' && layout.held.includes(b.id));
    assert.deepEqual(sdfs.map(p => p.mu), shaped.map(b => b.friction), `${s.id} one sdf part per held shaped solid`);
    // A solid moving from t = 0 (the tablecloth sphere) is pending, not baked: no part for it.
    assert.equal(src.parts!.length, boxes.length + shaped.length);
  }
});

test('Bridson scene solids: the field reads the ground and each held solid at its surface, with its friction', () => {
  for (const s of sources) {
    const src = solidsOf(s), h = src.cell;
    const ground = s.solids.find(b => b.id === 'ground')!, top = bridsonSolidPose(ground, 0).position_m;
    // Above the ground away from every other solid: phi is the height, exact (affine, on lattice planes).
    const layout = bridson2002SceneLayout(s);
    const far = s.solids.filter(b => b.id !== 'ground' && layout.held.includes(b.id)).map(b => bridsonSolidPose(b, 0).position_m);
    const probe = { x: Math.min(...far.map(c => c.x), 0) - 0.4, y: 0, z: top.z };
    const floorY = toScene(s, { x: 0, y: 0, z: 0 })[1];
    for (const dy of [0.013, 0.031]) {
      const p = toScene(s, { ...probe, y: dy }), f = fieldNear(src, p, 0.05), q = sampleClothSolidField(f, p);
      assert.ok(q.stored && Math.abs(q.phi - (p[1] - floorY)) < 1e-6, `${s.id} ground phi ${q.phi} at ${dy}`);
      assert.ok(Math.abs(q.grad[1] - 1) < 1e-5, `${s.id} ground normal`);
      assert.equal(q.mu, ground.friction, `${s.id} ground friction`);
    }
    for (const solid of s.solids.filter(b => b.id !== 'ground' && layout.held.includes(b.id))) {
      // Straight above the solid's highest point, 2.5 cells clear of it.
      const c = bridsonSolidPose(solid, 0).position_m;
      const sdf = src.parts!.filter(p => p.kind === 'sdf');
      const at = (yy: number) => Math.min(...sdf.map(p => p.kind === 'sdf' ? p.sdf(toScene(s, { x: c.x, y: yy, z: c.z })) : Infinity));
      let lo = c.y, hi = c.y + 2;
      for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (at(m) > 0) hi = m; else lo = m; }
      const y = hi + 2.5 * h;
      const p = toScene(s, { x: c.x, y, z: c.z }), f = fieldNear(src, p, 0.05), q = sampleClothSolidField(f, p);
      assert.ok(q.stored && Math.abs(q.phi - 2.5 * h) < 0.1 * h, `${s.id} ${solid.id} phi ${q.phi} above its top`);
      assert.equal(q.mu, solid.friction, `${s.id} ${solid.id} friction`);
    }
  }
});

test('Bridson scene solids: the contact field and the visible procedural solid share their lattice nodes', () => {
  const s = sources[2]!, scene = sceneOf(s), src = solidsOf(s);
  const b = scene.solidBoundary!, h = src.cell, o = src.origin!;
  // A leg's bricks: build the field over them and compare node for node.
  const leg = bridsonSolidPose(s.solids.find(x => x.id === 'leg-0')!, 0).position_m, c = toScene(s, leg);
  const f = fieldNear(src, [c[0], c[1] + 0.3, c[2]], 0.15);
  let compared = 0;
  for (const brick of b.bricks) {
    const key = brick.origin.map(v => v / SOLID_BRICK_CELLS).join(','), mine = f.bricks.get(key);
    const centre = brick.origin.map((v, a) => o[a]! + (v + 4) * h);
    if (Math.hypot(centre[0]! - c[0], centre[2]! - c[2]) > 0.1 || !mine) continue;
    for (let i = 0; i < 729; i++) {
      // The bake samples the shaped solids alone; where the ground is nearer the union differs by design.
      if (Math.abs(mine.values[i]! - brick.values[i]! * h) > 1e-6 * h) {
        const q = [i % 9, Math.floor(i / 9) % 9, Math.floor(i / 81)], y = o[1] + (brick.origin[1] + q[1]!) * h;
        assert.ok(mine.values[i]! < brick.values[i]! * h && y < c[1] + 0.1, `node ${i} of brick ${key}: ${mine.values[i]} vs ${brick.values[i]! * h}`);
      } else compared++;
    }
  }
  assert.ok(compared > 2000, `compared ${compared} nodes`);
});

test('Scene solids: solid edits, foreign boxes and unknown kinds fail explicitly', () => {
  const s = sources[0]!, scene = sceneOf(s);
  const at = (patch: SceneDescription['solidVoxels'][number]) => ({ ...scene, solidVoxels: [...scene.solidVoxels, patch] });
  assert.throws(() => clothSolidSourceForScene(at({ operation: 'clear', minimum: [10, 0, 10], maximumExclusive: [12, 8, 12], materialId: 0 }), { band_m: BAND }),
    /cannot follow solid patch \d+ \(clear box\)/);
  assert.throws(() => clothSolidSourceForScene(at({ operation: 'fill', minimum: [10, 8, 10], maximumExclusive: [12, 9, 12], materialId: VOXEL_MATERIAL_IDS.terrain }), { band_m: BAND }),
    /no solid for the box fill/);
  assert.throws(() => clothSolidSourceForScene(at({ operation: 'fill', kind: 'sphere', minimum: [10, 8, 10], maximumExclusive: [12, 10, 12], materialId: 2 }), { band_m: BAND }),
    /cannot follow solid patch \d+ \(fill sphere\)/);
  // A boundary that no longer matches its analytic solids.
  const b = scene.solidBoundary!, bricks = b.bricks.map((x, i) => i ? x : { ...x, values: x.values.map((v, j) => j === 300 ? v + 0.5 : v) });
  assert.throws(() => clothSolidSourceForScene({ ...scene, solidBoundary: { ...b, bricks } }, { band_m: BAND }), /procedural solid at .* reads/);
  assert.throws(() => clothSolidSourceForScene({ ...scene, sceneId: 'bridson-2002-unknown' }, { band_m: BAND }), /no Bridson 2002 scene/);
  // A scene without authored cloth solids may hold only its shell.
  const empty = createEmptyScene({ sceneId: 'cloth-empty' }), shell = clothSolidSourceForScene(empty, { band_m: BAND });
  assert.ok(shell.parts!.length >= 5 && shell.parts!.every(p => p.kind === 'lattice-box' && p.mu === CLOTH_CONTAINER_SHELL_FRICTION));
  assert.throws(() => clothSolidSourceForScene({ ...empty, solidVoxels: [...empty.solidVoxels,
    { operation: 'fill', minimum: [0, 0, 0], maximumExclusive: [8, 1, 8], materialId: VOXEL_MATERIAL_IDS.terrain }] }, { band_m: BAND }), /no authored cloth friction/);
  assert.ok(Math.abs(clothSolidBand(1e-3, 0.02) - (1e-3 + 1.5 * Math.sqrt(3) * 0.02)) < 1e-15);
});

test('Scene solids in the cloth frame: the same field, translated', () => {
  const s = sources[0]!, src = solidsOf(s), o: V3 = [0.37, 0.21, -0.13];
  const local = clothSolidSourceForScene(sceneOf(s), { band_m: BAND, frameOrigin_m: o });
  const ball = toScene(s, bridsonSolidPose(s.solids.find(b => b.id === 'ball')!, 0).position_m);
  const p: V3 = [ball[0] + 0.05, ball[1] + 0.29, ball[2] - 0.04];
  const a = sampleClothSolidField(fieldNear(src, p, 0.05), p);
  const b = sampleClothSolidField(fieldNear(local, [p[0] - o[0], p[1] - o[1], p[2] - o[2]], 0.05), [p[0] - o[0], p[1] - o[1], p[2] - o[2]]);
  assert.ok(a.stored && b.stored && Math.abs(a.phi - b.phi) < 1e-6 && a.mu === b.mu, `${a.phi} vs ${b.phi}`);
});
