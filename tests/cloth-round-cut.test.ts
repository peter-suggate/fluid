import test from 'node:test';
import assert from 'node:assert/strict';
import '../lib/methods';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { pickClothSurface } from '../lib/cloth/geometry';
import { cutClothShellBox } from '../lib/cloth/edit';
import { clothCutForSolidPatch, clothRoundCutHalfSpaces } from '../lib/cloth/round-cut';
import { sceneWithClothStroke } from '../lib/cloth/scene-edit';
import { subcellEllipsoidPatch, subcellPatch } from '../lib/core/solid-patch/ellipsoid-bounds';
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneCellSizes_m } from '../lib/core/scene-lattice-dimensions';
import type { ClothDocument, ClothTopologyData, ClothVector } from '../lib/cloth/types';
import type { SolidWorldCoordinate as Cell } from '../lib/core/solid-world';

// Two large triangles: a pen nib is far smaller than either, so the hole must
// be opened inside a face, not by deleting whole faces.
const data: ClothTopologyData = {restPositions_m: [-1, 1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1],
  triangles: [0, 2, 1, 1, 2, 3], triangleIds: [10, 11], triangleMaterials: [0, 0], uv: [0, 0, 1, 0, 0, 1, 1, 1], rodOffsets: [0], rodIndices: []};
const doc: ClothDocument = {schemaVersion: '1.0.0', id: 'sheet', geometry: {kind: 'inline', data}, materials: [DEFAULT_CLOTH_MATERIAL]};

test('a rounded cut stays within its tolerance of the inscribed ellipsoid and cylinder', () => {
  const centre: ClothVector = [0.1, 1, -0.2], tolerance = 0.002;
  for (const radii of [[0.05, 0.05, 0.05], [0.12, 0.03, 0.06], [0.04, Infinity, 0.09]] as ClothVector[]) {
    const planes = clothRoundCutHalfSpaces(centre, radii, tolerance);
    assert.ok(planes.length >= 8);
    for (let s = 0; s < 400; s++) {
      // Fibonacci directions on the quadric; a flat axis is sampled at the centre.
      const z = 1 - 2 * (s + 0.5) / 400, r = Math.sqrt(1 - z * z), phi = s * Math.PI * (3 - Math.sqrt(5));
      const u = [r * Math.cos(phi), z, r * Math.sin(phi)].map((v, a) => Number.isFinite(radii[a]!) ? v : 0);
      const scale = Math.hypot(...u.map((v, a) => Number.isFinite(radii[a]!) ? v : 0));
      const d = u.map(v => v / scale);
      // The quadric's surface along d, and the polytope's along the same ray.
      const quadric = 1 / Math.hypot(...d.map((v, a) => Number.isFinite(radii[a]!) ? v / radii[a]! : 0));
      let polytope = Infinity;
      for (const h of planes) {
        const along = h.normal[0] * d[0]! + h.normal[1] * d[1]! + h.normal[2] * d[2]!;
        if (along > 0) polytope = Math.min(polytope, (h.offset_m - (h.normal[0] * centre[0] + h.normal[1] * centre[1] + h.normal[2] * centre[2])) / along);
      }
      assert.ok(Math.abs(polytope - quadric) <= tolerance, `radii ${radii}: ${polytope} vs ${quadric}`);
    }
  }
  assert.deepEqual(clothRoundCutHalfSpaces(centre, [0.05, Infinity, Infinity], tolerance), []);
  assert.throws(() => clothRoundCutHalfSpaces(centre, [0.05, 0.05, 0], tolerance), /positive radii/);
});

test('a sphere smaller than a face opens a round hole in it and keeps the sheet conforming', () => {
  const radius = 0.1, centre: ClothVector = [0.3, 1, 0.2], tolerance = 0.001;
  const halfSpaces = clothRoundCutHalfSpaces(centre, [radius, radius, radius], tolerance);
  const box = {minimum_m: centre.map(v => v - radius) as unknown as ClothVector, maximum_m: centre.map(v => v + radius) as unknown as ClothVector};
  const cut = cutClothShellBox(doc, data, data.restPositions_m, box, halfSpaces);
  assert.ok(cut.changed);
  // The plane through the centre cuts a disc; its polygon is within tolerance of it.
  assert.ok(Math.abs(cut.removedArea_m2 - Math.PI * radius ** 2) < 2 * Math.PI * radius * tolerance, `${cut.removedArea_m2}`);
  const down = (x: number, z: number) => pickClothSurface(cut.document, cut.topology, cut.positions_m, {origin: [x, 2, z], direction: [0, -1, 0]});
  assert.equal(down(centre[0], centre[2]), undefined, 'the hole is open at its centre');
  assert.equal(down(centre[0] + 0.9 * radius, centre[2]), undefined, 'and out to its rim');
  assert.ok(down(centre[0] + 1.1 * radius, centre[2]), 'the sheet is kept just outside it');
  assert.ok(down(-0.8, -0.8), 'and everywhere else');
  // Every interior edge is shared by exactly two faces: no T-junction cracks.
  const edges = new Map<string, number>(), t = cut.topology.triangles;
  for (let f = 0; f < t.length / 3; f++) for (let k = 0; k < 3; k++) {
    const a = t[3 * f + k]!, b = t[3 * f + (k + 1) % 3]!, key = a < b ? `${a},${b}` : `${b},${a}`;
    edges.set(key, (edges.get(key) ?? 0) + 1);
  }
  assert.ok([...edges.values()].every(n => n <= 2));
  const boundary = [...edges.values()].filter(n => n === 1).length;
  // Outer square (at least 4 edges, split where the planes crossed them) plus the hole's rim.
  assert.ok(boundary >= 4 + 8, `${boundary}`);
  // An open edge is on the sheet's outline or the hole's rim, never inside a
  // face, where it would be a T-junction the membrane could part along.
  const p = cut.topology.restPositions_m;
  for (const [key, n] of edges) {
    if (n !== 1) continue;
    const [a, b] = key.split(',').map(Number) as [number, number];
    const x = (p[3 * a]! + p[3 * b]!) / 2, z = (p[3 * a + 2]! + p[3 * b + 2]!) / 2;
    const outline = Math.abs(Math.abs(x) - 1) < 1e-12 || Math.abs(Math.abs(z) - 1) < 1e-12;
    const rim = Math.abs(Math.hypot(x - centre[0], z - centre[2]) - radius) <= 2 * tolerance;
    assert.ok(outline || rim, `open edge ${key} at ${x},${z}`);
  }
  // Rest shape and UVs are interpolated in source coordinates.
  for (let i = 0; i < cut.topology.restPositions_m.length / 3; i++)
    assert.ok(Math.abs(cut.topology.uv![2 * i]! - (cut.topology.restPositions_m[3 * i]! + 1) / 2) < 1e-12);
});

test('the pen nib and the shape tool cut cloth through the shared patch, and unknown kinds still refuse', () => {
  const base = cloneScene(defaultScene), cell = sceneCellSizes_m(base), origin = [-base.container.width_m / 2, 0, -base.container.depth_m / 2];
  const sheet: ClothTopologyData = {...data, restPositions_m: data.restPositions_m.map((v, i) => i % 3 === 1 ? 0.3 : v * 0.2)};
  const scene = {...base, cloth: [{...doc, geometry: {kind: 'inline' as const, data: sheet}}]};
  const at = [0.01, 0.3, 0.02].map((v, a) => (v - origin[a]!) / cell[a]!);
  const nib = subcellEllipsoidPatch(at.map(v => v - 2.5) as unknown as Cell, at.map(v => v + 2.5) as unknown as Cell, 'clear');
  const cut = clothCutForSolidPatch(scene, nib);
  assert.equal(cut.kind, 'ellipsoid');
  assert.ok(cut.halfSpaces!.length >= 12);
  const edited = sceneWithClothStroke(scene, [cut]);
  assert.notEqual(edited, scene);
  const topology = edited.cloth![0]!.geometry.kind === 'inline' ? edited.cloth![0]!.geometry.data : undefined;
  assert.ok(topology && topology.triangles.length > sheet.triangles.length);
  assert.equal(pickClothSurface(edited.cloth![0]!, topology, topology.restPositions_m, {origin: [0.01, 1, 0.02], direction: [0, -1, 0]}), undefined);
  // A subcell cuboid is an exact box.
  const cuboid = clothCutForSolidPatch(scene, subcellPatch(at.map(v => v - 1.25) as unknown as Cell, at.map(v => v + 1.25) as unknown as Cell, 'clear', 'cuboid'));
  assert.equal(cuboid.kind, 'box');
  assert.ok(Math.abs(cuboid.box.maximum_m[0] - cuboid.box.minimum_m[0] - 2.5 * cell[0]!) < 1e-12);
  assert.throws(() => sceneWithClothStroke(scene, [{kind: 'block', box: cut.box}]), /block cuts need a matching surface operation/);
});
