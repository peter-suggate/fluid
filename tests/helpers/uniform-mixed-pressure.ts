import { createUniformMixedLayout, type UniformMixedLayout } from "../../lib/methods/uniform/uniform-mixed-layout";
import { geometricSeamRows, gradedSeamLayout, type Triple } from "./uniform-geometric-seam";

export interface PressureCell { center: Triple; volume: number; width: number }
export interface PressureFace { left: number; right: number; axis: number; center: Triple; area: number; distance: number }
export type PressureRow = ReadonlyMap<number, number>;
export interface PressureFixture { cells: PressureCell[]; faces: PressureFace[]; gradients: PressureRow[] }

/** Independent geometric oracle, deliberately not a runtime graph or solver.
 * Construct faces by intersecting cell boxes, then test a candidate seam
 * reconstruction before altering the native pressure operator.
 */
export function mixedPressureFixture(layout: UniformMixedLayout): PressureFixture {
  const boxes = geometricSeamRows(layout, () => [0, 0, 0]).cells, h = layout.lattice.cellSize_m;
  const cells = boxes.map(c => ({ center: c.min.map((v, a) => (v + c.width / 2) * h[a]!) as unknown as Triple,
    volume: c.capacity * h[0] * h[1] * h[2], width: c.width }));
  const faces: PressureFace[] = [];
  const neighbors: { cell: number; area: number; axis: number; sign: number }[][] = cells.map(() => []);
  for (let left = 0; left < boxes.length; left++) for (let right = 0; right < boxes.length; right++) {
    if (left === right) continue;
    const l = boxes[left]!, r = boxes[right]!;
    for (let axis = 0; axis < 3; axis++) {
      if (l.min[axis]! + l.width !== r.min[axis]) continue;
      const center: number[] = [], extents: number[] = [];
      for (let a = 0; a < 3; a++) {
        if (a === axis) { center[a] = r.min[a]! * h[a]!; extents[a] = 1; }
        else {
          const lo = Math.max(l.min[a]!, r.min[a]!), hi = Math.min(l.min[a]! + l.width, r.min[a]! + r.width);
          center[a] = (lo + hi) * h[a]! / 2; extents[a] = Math.max(0, hi - lo) * h[a]!;
        }
      }
      const area = extents.reduce((p, v) => p * v, 1);
      if (!area) continue;
      faces.push({ left, right, axis, center: center as unknown as Triple, area, distance: cells[right]!.center[axis]! - cells[left]!.center[axis]! });
      neighbors[left]!.push({ cell: right, area, axis, sign: 1 }); neighbors[right]!.push({ cell: left, area, axis, sign: -1 });
    }
  }
  const fits = new Map<number, Map<number, number[]> >();
  const fit = (i: number) => {
    const existing = fits.get(i); if (existing) return existing;
    const result = new Map<number, number[]>(), self = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
      // Average a complete finer face into one aligned neighbour sample.
      // Ignore a coarser neighbour whose centre is tangentially offset. Each
      // 2h child has an aligned sibling along every axis; a 4h owner has no
      // coarser neighbours. Thus a one-sided sample always exists at walls.
      const sides = [-1, 1].map(sign => neighbors[i]!.filter(n => n.axis === axis && n.sign === sign
        && cells[n.cell]!.width <= cells[i]!.width)).filter(side => side.length);
      if (!sides.length) {
        // A domain only one coarse cell thick has no aligned sample on this
        // axis. The quadrants of an incident finer face resolve its slope.
        const taps = neighbors[i]!.filter(n => cells[n.cell]!.width < cells[i]!.width);
        const variance = taps.reduce((sum, n) => sum + (cells[n.cell]!.center[axis]! - cells[i]!.center[axis]!) ** 2, 0);
        if (!variance) throw new Error("Missing pressure slope samples");
        for (const tap of taps) {
          const coefficient = (cells[tap.cell]!.center[axis]! - cells[i]!.center[axis]!) / variance;
          const previous = result.get(tap.cell) ?? [0, 0, 0];
          previous[axis]! += coefficient; self[axis]! -= coefficient; result.set(tap.cell, previous);
        }
        continue;
      }
      const samples = sides.map(side => {
        const area = side.reduce((sum, n) => sum + n.area, 0);
        const delta = side.reduce((sum, n) => sum + n.area * (cells[n.cell]!.center[axis]! - cells[i]!.center[axis]!), 0) / area;
        return { side, area, delta };
      });
      const span = samples.reduce((sum, sample) => sum + Math.abs(sample.delta), 0);
      for (const { side, area, delta } of samples) for (const tap of side) {
        const coefficient = Math.sign(delta) * tap.area / area / span;
        const previous = result.get(tap.cell) ?? [0, 0, 0];
        previous[axis]! += coefficient; self[axis]! -= coefficient;
        result.set(tap.cell, previous);
      }
    }
    result.set(i, self); fits.set(i, result); return result;
  };
  const gradients = faces.map(face => {
    const row = new Map<number, number>([[face.left, -1 / face.distance], [face.right, 1 / face.distance]]);
    for (const [i, other, sign] of [[face.left, face.right, -1], [face.right, face.left, 1]]) {
      if (cells[i!]!.width <= cells[other!]!.width) continue;
      // Reconstruct only sideways: the normal centre separation already
      // appears in the two-point difference. The fine centre is aligned.
      for (const [donor, coefficient] of fit(i!)) {
        let correction = 0;
        for (let a = 0; a < 3; a++) if (a !== face.axis) correction += coefficient[a]! * (face.center[a]! - cells[i!]!.center[a]!);
        row.set(donor, (row.get(donor) ?? 0) + sign! * correction / face.distance);
      }
    }
    return row;
  });
  return { cells, faces, gradients };
}
export function faceGradient(fixture: PressureFixture, pressure: ArrayLike<number>): Float64Array {
  return Float64Array.from(fixture.gradients, row => [...row].reduce((sum, [i, w]) => sum + w * pressure[i]!, 0));
}
/** Native geometric divergence: one physical flux per shared face. */
export function geometricDivergence(fixture: PressureFixture, velocity: ArrayLike<number>): Float64Array {
  const result = new Float64Array(fixture.cells.length);
  fixture.faces.forEach((f, k) => {
    const flux = f.area * velocity[k]!;
    result[f.left]! += flux / fixture.cells[f.left]!.volume;
    result[f.right]! -= flux / fixture.cells[f.right]!.volume;
  });
  return result;
}
/** Candidate variational divergence. Deliberately separate from geometric
 * divergence: defining a transpose does not prove the physical operator agrees. */
export function adjointDivergence(fixture: PressureFixture, velocity: ArrayLike<number>): Float64Array {
  const result = new Float64Array(fixture.cells.length);
  fixture.faces.forEach((f, k) => {
    const flux = f.area * f.distance * velocity[k]!;
    for (const [i, weight] of fixture.gradients[k]!) result[i]! -= weight * flux / fixture.cells[i]!.volume;
  });
  return result;
}

/** Exact inverse of the symmetric two-point core, for small test fixtures only.
 * Pin the last pressure to remove the closed-box constant nullspace. This is
 * deliberately a dense CPU oracle, not a replacement runtime pressure solver.
 */
export function factorPressureCore(fixture: PressureFixture): (rhs: ArrayLike<number>) => Float64Array {
  const n = fixture.cells.length - 1;
  const lower = Array.from({ length: n }, () => new Float64Array(n));
  for (const face of fixture.faces) {
    const w = face.area / face.distance, { left, right } = face;
    if (left < n) lower[left]![left]! += w;
    if (right < n) lower[right]![right]! += w;
    if (left < n && right < n) { lower[left]![right]! -= w; lower[right]![left]! -= w; }
  }
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let v = lower[i]![j]!;
    for (let k = 0; k < j; k++) v -= lower[i]![k]! * lower[j]![k]!;
    if (i === j && v <= 0) throw new Error("Pressure core is not positive definite after gauge fixing");
    lower[i]![j] = i === j ? Math.sqrt(v) : v / lower[j]![j]!;
  }
  return rhs => {
    const x = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) {
      let v = rhs[i]!;
      for (let k = 0; k < i; k++) v -= lower[i]![k]! * x[k]!;
      x[i] = v / lower[i]![i]!;
    }
    for (let i = n - 1; i >= 0; i--) {
      let v = x[i]!;
      for (let k = i + 1; k < n; k++) v -= lower[k]![i]! * x[k]!;
      x[i] = v / lower[i]![i]!;
    }
    return x;
  };
}

/** Deferred nonorthogonal correction: freeze just the tangential term on the
 * RHS, preserving the symmetric native-style core and geometric divergence.
 * The caller supplies the core inverse so this oracle cannot be mistaken for
 * a GPU multigrid convergence or performance test.
 */
export function deferredPressureStep(fixture: PressureFixture, pressure: ArrayLike<number>, rhs: ArrayLike<number>,
  solveCore: (rhs: ArrayLike<number>) => Float64Array): Float64Array {
  const correction = faceGradient(fixture, pressure).map((value, k) => {
    const f = fixture.faces[k]!;
    return value - (pressure[f.right]! - pressure[f.left]!) / f.distance;
  });
  const divergence = geometricDivergence(fixture, correction);
  return solveCore(divergence.map((v, i) => rhs[i]! + v * fixture.cells[i]!.volume));
}


export function mixedPressureLayouts(): UniformMixedLayout[] {
  const layouts = [0, 1, 2].map(gradedSeamLayout);
  for (const dimensions of [[16, 16, 16], [4, 16, 16], [16, 4, 16], [16, 16, 4]] as const) {
    for (const start of dimensions.some(n => n === 4) ? [0] : [0, 4]) layouts.push(createUniformMixedLayout({
      dimensions, cellSize_m: [1, 2, .5], origin_m: { x: 0, y: 0, z: 0 },
    }, [{ id: "island", rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
      min_m: { x: start, y: start * 2, z: start / 2 },
      max_m: { x: start + 4, y: (start + 4) * 2, z: (start + 4) / 2 },
    }], true, 4));
  }
  return layouts;
}
