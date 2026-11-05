import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_CLOTH_MATERIAL } from '../lib/cloth/material';
import { createClothState, prepareClothTopology } from '../lib/cloth/topology';
import { stepCloth } from '../lib/cloth/dynamics';
import { solveClothIncrement } from '../lib/cloth/nonlinear-solve';
import { cutClothShellBox } from '../lib/cloth/edit';
import type { ClothDocument, ClothMaterial, ClothTopologyData } from '../lib/cloth/types';

/** An n x n sheet of spacing h in the y = 1 plane, two corners pinned. */
function sheet(n: number, h: number): ClothTopologyData {
  const restPositions_m: number[] = [], triangles: number[] = [], fixed: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {restPositions_m.push(i*h, 1, j*h); fixed.push(j === 0 && (i === 0 || i === n - 1) ? 1 : 0);}
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < n; i++) {const a = j*n + i, b = a + 1, c = a + n, d = c + 1; triangles.push(a, c, b, b, c, d);}
  return {restPositions_m, triangles, rodOffsets: [0], rodIndices: [], fixed};
}
const maximumDifference = (a: ArrayLike<number>, b: ArrayLike<number>) => {let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k]! - b[k]!)); return m;};

test('Newton-PCG reaches the same implicit-Euler minimizer as converged Gauss-Seidel', () => {
  const topology = prepareClothTopology(sheet(5, .01), [DEFAULT_CLOTH_MATERIAL]);
  const a = createClothState(topology), b = createClothState(topology), dt = .001;
  for (let step = 0; step < 6; step++) {
    const newton = solveClothIncrement(topology, a, dt, {iterations: 64, tolerance_m: 1e-12});
    const gauss = stepCloth(topology, b, dt, {iterations: 20000, tolerance_m: 1e-14});
    assert.ok(newton.converged && gauss.converged, `step ${step}: ${newton.maximumUpdate_m} / ${gauss.maximumUpdate_m}`);
    assert.ok(newton.iterations <= 8, `Newton took ${newton.iterations} iterations`);
    assert.ok(maximumDifference(a.positions_m, b.positions_m) < 1e-9, `positions differ by ${maximumDifference(a.positions_m, b.positions_m)} m`);
    assert.ok(maximumDifference(a.velocities_m_s, b.velocities_m_s) < 1e-6);
  }
  assert.ok(a.positions_m[3*12 + 1]! < 1, 'the sheet falls under gravity');
});

test('a stiff sheet that coordinate sweeps cannot finish converges in a few Newton iterations', () => {
  const stiff: ClothMaterial = {...DEFAULT_CLOTH_MATERIAL, id: 'plastic', youngModulus_Pa: 3e9, thickness_m: .0005};
  const topology = prepareClothTopology(sheet(6, .01), [stiff]), a = createClothState(topology), b = createClothState(topology);
  const gauss = stepCloth(topology, b, .001, {iterations: 64, tolerance_m: 1e-8});
  assert.equal(gauss.converged, false, 'the coordinate solver needs more than its budget here');
  for (let step = 0; step < 4; step++) {
    const newton = solveClothIncrement(topology, a, .001, {iterations: 64, tolerance_m: 1e-8});
    assert.ok(newton.converged, `step ${step}: ${newton.maximumUpdate_m} m`);
    assert.ok(newton.iterations <= 10, `Newton took ${newton.iterations} iterations`);
  }
});

test('Newton-PCG rods (stretch, bending, twist) match converged Gauss-Seidel', () => {
  const n = 7, restPositions_m: number[] = [];
  for (let i = 0; i < n; i++) restPositions_m.push(i*.005, 1, .001*Math.sin(i));
  const data: ClothTopologyData = {restPositions_m, triangles: [], rodOffsets: [0, n], rodIndices: [...Array(n).keys()], fixed: [3, 3, ...Array(n - 2).fill(0)]};
  const topology = prepareClothTopology(data, [DEFAULT_CLOTH_MATERIAL]), a = createClothState(topology), b = createClothState(topology);
  for (let step = 0; step < 4; step++) {
    const newton = solveClothIncrement(topology, a, .001, {iterations: 64, tolerance_m: 1e-11});
    const gauss = stepCloth(topology, b, .001, {iterations: 20000, tolerance_m: 1e-13});
    assert.ok(newton.converged && gauss.converged, `step ${step}: ${newton.maximumUpdate_m} / ${gauss.maximumUpdate_m}`);
    assert.ok(maximumDifference(a.positions_m, b.positions_m) < 1e-8, `positions differ by ${maximumDifference(a.positions_m, b.positions_m)} m`);
    assert.ok(maximumDifference(a.twist_rad, b.twist_rad) < 1e-5, `twist differs by ${maximumDifference(a.twist_rad, b.twist_rad)} rad`);
  }
});

test('a cut plane a Float32 rounding away from authored vertices emits no degenerate edge or triangle', () => {
  const data = sheet(5, .005), document: ClothDocument = {schemaVersion: '1.0.0', id: 'sheet', geometry: {kind: 'inline', data}, materials: [DEFAULT_CLOTH_MATERIAL]};
  // Box faces at x = 0.005 + 2.2e-10 and z = 0.01 - 2.2e-10: through vertex columns, as Float32 cells give them.
  const cut = cutClothShellBox(document, data, data.restPositions_m, {minimum_m: [.005 + 2.2e-10, .99, -.001], maximum_m: [.0125, 1.01, .01 - 2.2e-10]});
  assert.ok(cut.changed);
  const P = cut.topology.restPositions_m, T = cut.topology.triangles;
  let shortest = Infinity, smallest = Infinity;
  for (let f = 0; f < T.length/3; f++) {
    const [a, b, c] = [T[3*f]!, T[3*f+1]!, T[3*f+2]!], p = (i: number) => [P[3*i]!, P[3*i+1]!, P[3*i+2]!];
    for (const [i, j] of [[a, b], [b, c], [c, a]] as const) shortest = Math.min(shortest, Math.hypot(...p(i).map((v, k) => v - p(j)[k]!)));
    const u = p(b).map((v, k) => v - p(a)[k]!), v = p(c).map((w, k) => w - p(a)[k]!);
    smallest = Math.min(smallest, Math.hypot(u[1]!*v[2]! - u[2]!*v[1]!, u[2]!*v[0]! - u[0]!*v[2]!, u[0]!*v[1]! - u[1]!*v[0]!)/2);
  }
  assert.ok(shortest > 1e-4, `shortest edge ${shortest} m`);
  assert.ok(smallest > 1e-7, `smallest triangle ${smallest} m^2`);
  // The cut still removes the box's area of the sheet (2.5 x 2 cells of 5 mm).
  assert.ok(Math.abs(cut.removedArea_m2 - .0075*.01) < 1e-9, `removed ${cut.removedArea_m2} m^2`);
  // And the cut sheet steps.
  const topology = prepareClothTopology(cut.topology, document.materials), state = createClothState(topology);
  assert.ok(solveClothIncrement(topology, state, .001, {iterations: 64, tolerance_m: 1e-8}).converged);
});
