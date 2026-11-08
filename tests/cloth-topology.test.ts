/**
 * Cloth topology data contracts that the solver relies on: authored per-vertex
 * masses (Bridson 2002's equal particle masses) replace area lumping, are
 * validated, and survive a local cut as a per-vertex areal density.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { clothGridData } from '../lib/cloth/grid';
import { clothTopologyErrors, prepareClothTopology } from '../lib/cloth/topology';
import { cutClothShellBox } from '../lib/cloth/edit';
import type { ClothDocument, ClothMaterial, ClothTopologyData } from '../lib/cloth/types';

const cotton: ClothMaterial = { id: 'cotton', arealDensity_kg_m2: 0.2, stretchWarp_N_m: 400, stretchWeft_N_m: 250, shear_N_m: 40, poisson: 0.3,
  compressionRatio: 0.05, bending_N_m: 2e-5, damping_s: 0, thickness_m: 5e-4 };
const sum = (a: ArrayLike<number>) => Array.from(a).reduce((s, x) => s + x, 0);

function equalMassSheet(n: number, nodeMass: number): ClothTopologyData {
  const data = clothGridData({ columns: n, rows: n, width_m: 1, height_m: 1, origin_m: [0, 1, 0] });
  return { ...data, vertexMass_kg: new Array(n * n).fill(nodeMass) };
}

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
