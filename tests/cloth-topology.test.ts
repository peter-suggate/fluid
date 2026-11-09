/**
 * Cloth topology data contracts that the solver relies on: authored per-vertex
 * masses (Bridson 2002's equal particle masses) replace area lumping, are
 * validated, and survive a local cut as a per-vertex areal density.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { clothGridData } from '../lib/cloth/grid';
import { clothTopologyErrors, clothVertexNeighbours, colourClothVertices, prepareClothTopology } from '../lib/cloth/topology';
import { cutClothShellBox } from '../lib/cloth/edit';
import { bridson2002Tablecloth } from '../lib/cloth/scenes/bridson-2002';
import type { ClothDocument, ClothMaterial, ClothTopologyData } from '../lib/cloth/types';

const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0, thickness_m: 5e-4 };
const sum = (a: ArrayLike<number>) => Array.from(a).reduce((s, x) => s + x, 0);

function equalMassSheet(n: number, nodeMass: number): ClothTopologyData {
  const data = clothGridData({ columns: n, rows: n, width_m: 1, height_m: 1, origin_m: [0, 1, 0] });
  return { ...data, vertexMass_kg: new Array(n * n).fill(nodeMass) };
}

test('the 150 x 150 Bridson tablecloth colours into five balanced classes (greedy left a one-vertex sixth class costing a dispatch per iteration)', () => {
  const { cloth } = bridson2002Tablecloth();
  const neighbours = clothVertexNeighbours(prepareClothTopology(cloth.geometry.data, cloth.materials));
  const colour = colourClothVertices(neighbours), sizes: number[] = [];
  for (const c of colour) sizes[c] = (sizes[c] ?? 0) + 1;
  assert.equal(sizes.length, 5, `classes ${sizes.join(',')}`);
  assert.ok(Math.min(...sizes) > 0.15 * colour.length, `classes ${sizes.join(',')}`);
  for (let v = 0; v < colour.length; v++)
    for (let j = neighbours.offsets[v]!; j < neighbours.offsets[v + 1]!; j++) assert.notEqual(colour[neighbours.values[j]!], colour[v]);
});

test('authored vertex masses replace area lumping', () => {
  const lumped = prepareClothTopology(clothGridData({ columns: 9, rows: 9, width_m: 1, height_m: 1, origin_m: [0, 1, 0] }), [cotton]);
  assert.ok(Math.abs(sum(lumped.mass_kg) - 0.2) < 1e-12);
  assert.ok(lumped.mass_kg[0]! < lumped.mass_kg[40]!, 'lumping makes corners lighter than the interior');
  const equal = prepareClothTopology(equalMassSheet(9, 0.2 / 81), [cotton]);
  assert.ok(equal.mass_kg.every(m => Math.abs(m - 0.2 / 81) < 1e-15));
});

test('vertex masses must be one positive finite value per vertex', () => {
  const good = equalMassSheet(4, 1e-3);
  for (const bad of [[...good.vertexMass_kg!.slice(1)], good.vertexMass_kg!.map((m, i) => i === 3 ? 0 : m), good.vertexMass_kg!.map((m, i) => i === 5 ? NaN : m)])
    assert.ok(clothTopologyErrors({ ...good, vertexMass_kg: bad }, [cotton]).some(e => e.includes('vertex masses')));
  assert.deepEqual(clothTopologyErrors(good, [cotton]), []);
});

test('a cut keeps authored masses outside the cut and removes mass with the cut area', () => {
  const n = 11, data = equalMassSheet(n, 1e-3);
  const document: ClothDocument = { schemaVersion: '2.0.0', id: 'sheet', geometry: { kind: 'inline', data }, materials: [cotton] };
  // A box that misses the sheet changes nothing.
  const miss = cutClothShellBox(document, data, data.restPositions_m, { minimum_m: [2, 2, 2], maximum_m: [3, 3, 3] });
  assert.deepEqual(miss.topology.vertexMass_kg, data.vertexMass_kg);
  // A hole in the middle: vertices far from it keep their mass exactly; the total falls with the removed area.
  const cut = cutClothShellBox(document, data, data.restPositions_m, { minimum_m: [0.42, 0.9, 0.42], maximum_m: [0.58, 1.1, 0.58] });
  assert.ok(cut.changed);
  const masses = cut.topology.vertexMass_kg!, rest = cut.topology.restPositions_m;
  for (let v = 0; v < masses.length; v++) {
    const x = rest[3 * v]!, z = rest[3 * v + 2]!;
    const interior = x > 0.05 && x < 0.95 && z > 0.05 && z < 0.95, farFromHole = Math.max(Math.abs(x - 0.5), Math.abs(z - 0.5)) > 0.25;
    if (interior && farFromHole) assert.ok(Math.abs(masses[v]! - 1e-3) < 1e-15, `vertex ${v} keeps its mass`);
  }
  const total = sum(masses), before = n * n * 1e-3;
  assert.ok(total < before && total > before * (1 - 2 * cut.removedArea_m2), `total ${total} vs ${before}, removed ${cut.removedArea_m2} m^2`);
  assert.doesNotThrow(() => prepareClothTopology(cut.topology, [cotton]));
});

test('a cut through a curved sheet of finite thickness splits each shared edge at one point, in both faces: no slit of unwelded coincident vertices, no T-junction', () => {
  // A bowl: the faces either side of an edge have different normals, so a per-face thickness support moved the plane
  // differently for each and cut a shared edge twice (the GPU contact then saw coincident vertices at distance 0).
  const data = clothGridData({ columns: 8, rows: 8, width_m: 0.4, height_m: 0.4, origin_m: [-0.2, 0.9, -0.2] });
  const bowl = data.restPositions_m.map((v, i) => i % 3 === 1 ? v - 0.8 * (data.restPositions_m[i - 1]! ** 2 + data.restPositions_m[i + 1]! ** 2) : v);
  const box = { minimum_m: [-0.06, -1, -0.06] as const, maximum_m: [0.06, 2, 0.06] as const };
  const shape = (thickness_m: number) => {
    const document: ClothDocument = { schemaVersion: '2.0.0', id: 'bowl', geometry: { kind: 'inline', data }, materials: [{ ...cotton, thickness_m }] };
    const cut = cutClothShellBox(document, data, bowl, box), p = cut.positions_m, n = p.length / 3, t = cut.topology.triangles;
    let nearest = Infinity;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) nearest = Math.min(nearest, Math.hypot(p[3 * i]! - p[3 * j]!, p[3 * i + 1]! - p[3 * j + 1]!, p[3 * i + 2]! - p[3 * j + 2]!));
    const uses = new Map<number, number>();
    for (let f = 0; f < t.length / 3; f++) for (let k = 0; k < 3; k++) { const a = t[3 * f + k]!, b = t[3 * f + (k + 1) % 3]!, key = Math.min(a, b) * n + Math.max(a, b); uses.set(key, (uses.get(key) ?? 0) + 1); }
    // T-junctions: a vertex on the inside of an edge it is not an end of (the face across kept the whole side).
    let junctions = 0;
    for (const key of uses.keys()) {
      const a = Math.floor(key / n), b = key % n, d = [0, 1, 2].map(k => p[3 * b + k]! - p[3 * a + k]!), l2 = d[0]! ** 2 + d[1]! ** 2 + d[2]! ** 2;
      for (let v = 0; v < n; v++) {
        if (v === a || v === b) continue;
        const s = [0, 1, 2].reduce((q, k) => q + (p[3 * v + k]! - p[3 * a + k]!) * d[k]!, 0) / l2;
        if (s > 1e-6 && s < 1 - 1e-6 && Math.hypot(...[0, 1, 2].map(k => p[3 * v + k]! - p[3 * a + k]! - s * d[k]!)) < 1e-9) junctions++;
      }
    }
    return { n, faces: t.length / 3, nearest, boundary: [...uses.values()].filter(c => c === 1).length, junctions };
  };
  const thin = shape(0), thick = shape(5e-3);
  // The thick shell's hole is larger, but its mesh is the thin one's: same vertices, faces and boundary, no near pair.
  assert.deepEqual([thick.n, thick.faces, thick.boundary], [thin.n, thin.faces, thin.boundary]);
  assert.ok(thick.nearest > 1e-3, `nearest vertex pair ${thick.nearest} m`);
  // A plane also crosses a face's side outside the box (along another axis), where the face across may be disjoint:
  // it takes the point too, so no vertex lies on an edge it does not end.
  assert.deepEqual([thin.junctions, thick.junctions], [0, 0]);
});
