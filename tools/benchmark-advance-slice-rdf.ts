/**
 * Offline comparison of the accepted PLIC cache with Scheufler/Roenby's
 * reconstructed-distance-function (RDF) construction.  This deliberately
 * does not participate in transport or presentation; it is a decision aid
 * for choosing a published interface representation.
 *
 * Primary algorithm source:
 * https://doi.org/10.1016/j.jcp.2019.01.009 (sections 3.3, 3.6 and 3.8)
 * Reference implementation:
 * https://api.openfoam.com/2512/reconstructedDistanceFunction_8C_source.html
 */
import { productionSceneSliceSeedById } from "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { advanceSlice, createAdvanceSlice } from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import type { AdvanceSlice } from "../lib/methods/adaptive-volume/advance-slice/slice-solver";

type V2 = readonly [number, number];
interface Plane { cell: number; n: V2; d: number; centre: V2; fill: number }

const partial = (slice: AdvanceSlice, cell: number): boolean => {
  const k = slice.fields.capacity[cell]!, a = k > 1e-8 ? slice.fields.density[cell]! / k : 0;
  return k >= 0.999999 && a > 1e-6 && a < 1 - 1e-6;
};

function interfaceSegment(slice: AdvanceSlice, cellId: number, n: V2, d: number): V2[] {
  const cell = slice.numericalTopology.cells[cellId]!, c = cell.center;
  const lo = cell.minimum, hi = cell.maximum, out: V2[] = [];
  const add = (x: number, y: number) => {
    if (x < lo[0] - 1e-8 || x > hi[0] + 1e-8 || y < lo[1] - 1e-8 || y > hi[1] + 1e-8) return;
    if (!out.some(q => Math.hypot(q[0] - x, q[1] - y) < 1e-7)) out.push([x, y]);
  };
  if (Math.abs(n[1]) > 1e-12) {
    add(lo[0], c[1] + (d - n[0] * (lo[0] - c[0])) / n[1]);
    add(hi[0], c[1] + (d - n[0] * (hi[0] - c[0])) / n[1]);
  }
  if (Math.abs(n[0]) > 1e-12) {
    add(c[0] + (d - n[1] * (lo[1] - c[1])) / n[0], lo[1]);
    add(c[0] + (d - n[1] * (hi[1] - c[1])) / n[0], hi[1]);
  }
  return out.slice(0, 2);
}

function planes(slice: AdvanceSlice, normals = slice.fields.interfaceNormal,
  offsets = slice.fields.interfaceOffset): Plane[] {
  const out: Plane[] = [];
  for (const cell of slice.numericalTopology.cells) {
    if (!partial(slice, cell.id)) continue;
    const n: V2 = [normals[2 * cell.id]!, normals[2 * cell.id + 1]!];
    if (!(Math.hypot(...n) > 0.5)) continue;
    const segment = interfaceSegment(slice, cell.id, n, offsets[cell.id]!);
    if (segment.length !== 2) continue;
    out.push({ cell: cell.id, n, d: offsets[cell.id]!,
      centre: [(segment[0]![0] + segment[1]![0]) / 2,
        (segment[0]![1] + segment[1]![1]) / 2],
      fill: slice.fields.density[cell.id]! / slice.fields.capacity[cell.id]! });
  }
  return out;
}

function vertexCells(slice: AdvanceSlice): Map<string, number[]> {
  const result = new Map<string, number[]>();
  for (const cell of slice.numericalTopology.cells) {
    // The accepted row graph splits a coarse face at every fine-side
    // T-junction. Include every integer sample covered by the coarse cell so
    // those shared graph vertices receive one scalar value as well.
    for (let x = cell.minimum[0]; x <= cell.maximum[0]; x += 1)
      for (let y = cell.minimum[1]; y <= cell.maximum[1]; y += 1) {
        const key = `${x}:${y}`, list = result.get(key) ?? [];
        list.push(cell.id); result.set(key, list);
      }
  }
  return result;
}

function pointNeighbours(slice: AdvanceSlice, cellId: number,
  attached: ReadonlyMap<string, readonly number[]>): number[] {
  const cell = slice.numericalTopology.cells[cellId]!, result = new Set<number>();
  for (const x of [cell.minimum[0], cell.maximum[0]])
    for (const y of [cell.minimum[1], cell.maximum[1]])
      for (const other of attached.get(`${x}:${y}`) ?? []) result.add(other);
  return [...result];
}

/** Equation (9)/(10): weighted signed distances to point-neighbour segments. */
function constructRdf(slice: AdvanceSlice, plic: readonly Plane[]): Float64Array {
  const byCell = new Map(plic.map(p => [p.cell, p]));
  const attached = vertexCells(slice);
  const phi = new Float64Array(slice.numericalTopology.cells.length).fill(Number.NaN);
  for (const cell of slice.numericalTopology.cells) {
    const own = byCell.get(cell.id);
    if (own) {
      phi[cell.id] = own.n[0] * (cell.center[0] - own.centre[0])
        + own.n[1] * (cell.center[1] - own.centre[1]);
      continue;
    }
    let weighted = 0, weights = 0;
    for (const neighbour of pointNeighbours(slice, cell.id, attached)) {
      const q = byCell.get(neighbour); if (!q) continue;
      const dx = cell.center[0] - q.centre[0], dy = cell.center[1] - q.centre[1];
      const r2 = dx * dx + dy * dy, signed = q.n[0] * dx + q.n[1] * dy;
      const weight = r2 > 0 ? signed * signed / r2 : 1;
      weighted += weight * signed; weights += weight;
    }
    if (weights > 0) phi[cell.id] = weighted / weights;
  }
  return phi;
}

function solve3(a: number[][], b: number[]): number[] | null {
  const m = a.map((row, i) => [...row, b[i]!]);
  for (let col = 0; col < 3; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < 3; row += 1) if (Math.abs(m[row]![col]!) > Math.abs(m[pivot]![col]!)) pivot = row;
    if (Math.abs(m[pivot]![col]!) < 1e-10) return null;
    [m[col], m[pivot]] = [m[pivot]!, m[col]!];
    const inv = 1 / m[col]![col]!;
    for (let j = col; j < 4; j += 1) m[col]![j] *= inv;
    for (let row = 0; row < 3; row += 1) if (row !== col) {
      const q = m[row]![col]!;
      for (let j = col; j < 4; j += 1) m[row]![j] -= q * m[col]![j]!;
    }
  }
  return [m[0]![3]!, m[1]![3]!, m[2]![3]!];
}

function affineFit(samples: readonly { p: V2; value: number }[], origin: V2): number[] | null {
  const a = Array.from({ length: 3 }, () => [0, 0, 0]), b = [0, 0, 0];
  for (const sample of samples) {
    const row = [sample.p[0] - origin[0], sample.p[1] - origin[1], 1];
    for (let i = 0; i < 3; i += 1) {
      b[i]! += row[i]! * sample.value;
      for (let j = 0; j < 3; j += 1) a[i]![j]! += row[i]! * row[j]!;
    }
  }
  return solve3(a, b);
}

function vertexRdf(slice: AdvanceSlice, rdf: Float64Array, plic: readonly Plane[]): Float64Array {
  const { nx, ny } = slice, values = new Float64Array((nx + 1) * (ny + 1)).fill(Number.NaN);
  const attached = vertexCells(slice);
  const byCell = new Map(plic.map(plane => [plane.cell, plane]));
  const actual = new Set<string>();
  for (const cell of slice.numericalTopology.cells) for (const x of [cell.minimum[0], cell.maximum[0]])
    for (const y of [cell.minimum[1], cell.maximum[1]]) actual.add(`${x}:${y}`);
  for (let y = 0; y <= ny; y += 1) for (let x = 0; x <= nx; x += 1) {
    const ids = attached.get(`${x}:${y}`) ?? [];
    if (!actual.has(`${x}:${y}`)) {
      const direct = ids.map(id => byCell.get(id)).filter((plane): plane is Plane => Boolean(plane))
        .map(plane => plane.n[0] * (x - plane.centre[0]) + plane.n[1] * (y - plane.centre[1]));
      if (direct.length) {
        values[x + (nx + 1) * y] = direct.reduce((a, b) => a + b, 0) / direct.length;
        continue;
      }
    }
    const samples: { p: V2; value: number }[] = [];
    for (const id of ids) {
      const cell = slice.numericalTopology.cells[id]!;
      if (!Number.isFinite(rdf[cell.id]!)) continue;
      samples.push({ p: cell.center, value: rdf[cell.id]! });
    }
    // Exact cell-face interfaces have no 0<alpha<1 owner, so RDF has no
    // segment to seed from. isoAlpha's shared scalar is the established
    // fallback for that case; it also keeps dry/full general scenes visible.
    if (!samples.length) for (const id of attached.get(`${x}:${y}`) ?? []) {
      const cell = slice.numericalTopology.cells[id]!, k = slice.fields.capacity[id]!;
      if (!(k > 1e-8)) continue;
      const fill = Math.max(0, Math.min(1, slice.fields.density[id]! / k));
      samples.push({ p: cell.center, value: (0.5 - fill) * 4 * Math.max(...cell.widths) });
    }
    const fit = samples.length >= 3 ? affineFit(samples, [x, y]) : null;
    if (fit) values[x + (nx + 1) * y] = fit[2]!;
    else if (samples.length) values[x + (nx + 1) * y] = samples.reduce((s, q) => s + q.value, 0) / samples.length;
  }
  return values;
}

function clippedTriangleArea(points: readonly [number, number, number][]): number {
  let polygon = points.map(p => [...p] as [number, number, number]);
  const out: typeof polygon = [];
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i]!, b = polygon[(i + 1) % polygon.length]!;
    if (a[2] <= 0) out.push(a);
    if ((a[2] < 0) !== (b[2] < 0)) {
      const t = a[2] / (a[2] - b[2]);
      out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0]);
    }
  }
  polygon = out;
  let twice = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i]!, b = polygon[(i + 1) % polygon.length]!;
    twice += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(twice) / 2;
}

function volumeError(slice: AdvanceSlice, values: Float64Array) {
  const stride = slice.nx + 1, errors: number[] = [];
  const byWidth = new Map<number, number[]>();
  let exact = 0, represented = 0;
  for (const cell of slice.numericalTopology.cells) {
    const target = slice.fields.density[cell.id]! * cell.area;
    exact += target;
    if (!partial(slice, cell.id)) { represented += target; continue; }
    let area = 0, valid = true;
    for (let y = cell.minimum[1]; y < cell.maximum[1]; y += 1)
      for (let x = cell.minimum[0]; x < cell.maximum[0]; x += 1) {
        const q00 = values[x + stride * y]!, q10 = values[x + 1 + stride * y]!;
        const q11 = values[x + 1 + stride * (y + 1)]!, q01 = values[x + stride * (y + 1)]!;
        if (![q00, q10, q11, q01].every(Number.isFinite)) { valid = false; continue; }
        area += clippedTriangleArea([[x, y, q00], [x + 1, y, q10], [x + 1, y + 1, q11]])
          + clippedTriangleArea([[x, y, q00], [x + 1, y + 1, q11], [x, y + 1, q01]]);
      }
    if (!valid) continue;
    represented += area; errors.push(area - target);
    const bucket = byWidth.get(cell.widths[0]) ?? [];
    bucket.push(area - target); byWidth.set(cell.widths[0], bucket);
  }
  return { exactTotalFineArea: exact, representedTotalFineArea: represented,
    signedTotalFineArea: represented - exact,
    relativeTotal: exact > 0 ? (represented - exact) / exact : 0,
    meanAbsPerPartialCell: errors.reduce((s, e) => s + Math.abs(e), 0) / Math.max(1, errors.length),
    maxAbsPerPartialCell: Math.max(0, ...errors.map(Math.abs)), comparedPartialCells: errors.length,
    byCellWidth: Object.fromEntries([...byWidth].map(([width, values]) => [width, {
      count: values.length, signed: values.reduce((a, b) => a + b, 0),
      meanAbs: values.reduce((a, b) => a + Math.abs(b), 0) / values.length,
      maxAbs: Math.max(...values.map(Math.abs)), meanAbsFraction: values.reduce((a, b) => a + Math.abs(b), 0) / values.length / (width * width),
    }])) };
}

function contour(slice: AdvanceSlice, values: Float64Array): { points: V2[]; ambiguous: number; unresolved: number } {
  const stride = slice.nx + 1, points: V2[] = []; let ambiguous = 0, unresolved = 0;
  for (let y = 0; y < slice.ny; y += 1) for (let x = 0; x < slice.nx; x += 1) {
    const corners: readonly [number, number, number][] = [
      [x, y, values[x + stride * y]!], [x + 1, y, values[x + 1 + stride * y]!],
      [x + 1, y + 1, values[x + 1 + stride * (y + 1)]!], [x, y + 1, values[x + stride * (y + 1)]!],
    ];
    const cuts: V2[] = [];
    for (let edge = 0; edge < 4; edge += 1) {
      const a = corners[edge]!, b = corners[(edge + 1) % 4]!;
      if (!Number.isFinite(a[2]) || !Number.isFinite(b[2])) continue;
      if ((a[2] < 0) === (b[2] < 0) || a[2] === b[2]) continue;
      const t = a[2] / (a[2] - b[2]); cuts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    if (cuts.length === 2) points.push(...cuts);
    else if (cuts.length === 4) { ambiguous += 1; points.push(...cuts); }
    else if (cuts.length !== 0) unresolved += 1;
  }
  return { points, ambiguous, unresolved };
}

function plicJoinStats(slice: AdvanceSlice, plic: readonly Plane[]) {
  const intersections = new Map<string, number[]>();
  for (const p of plic) {
    const cell = slice.numericalTopology.cells[p.cell]!;
    const segment = interfaceSegment(slice, p.cell, p.n, p.d);
    for (const q of segment) {
      const onVertical = Math.abs(q[0] - cell.minimum[0]) < 1e-6 || Math.abs(q[0] - cell.maximum[0]) < 1e-6;
      const key = onVertical ? `x:${q[0].toFixed(6)}:${Math.floor(q[1] + 1e-6)}`
        : `y:${q[1].toFixed(6)}:${Math.floor(q[0] + 1e-6)}`;
      const coordinate = onVertical ? q[1] : q[0];
      const list = intersections.get(key) ?? []; list.push(coordinate); intersections.set(key, list);
    }
  }
  const gaps = [...intersections.values()].filter(v => v.length === 2).map(v => Math.abs(v[0]! - v[1]!));
  return { joins: gaps.length, mismatched: gaps.filter(g => g > 1e-6).length,
    meanGap: gaps.reduce((a, b) => a + b, 0) / Math.max(1, gaps.length), maxGap: Math.max(0, ...gaps) };
}

function sphereRadial(points: readonly V2[], slice: AdvanceSlice) {
  const authored = slice.scene.production?.scene.fluid.initialLiquidVolumes?.find(v => v.shape === "sphere");
  if (!authored || authored.shape !== "sphere") return null;
  const h = slice.scene.viewport.sourceCellSize;
  const cx = (authored.center_m.x - slice.scene.viewport.originX) / h;
  const cy = (authored.center_m.y - slice.scene.viewport.originY) / h;
  const dz = slice.scene.viewport.sourceCellCenterZ - authored.center_m.z;
  const radius = Math.sqrt(Math.max(0, authored.radius_m ** 2 - dz ** 2)) / h;
  const errors = points.map(p => Math.hypot(p[0] - cx, p[1] - cy) - radius).filter(e => Math.abs(e) < 3);
  return { count: errors.length, meanAbs: errors.reduce((s, e) => s + Math.abs(e), 0) / Math.max(1, errors.length),
    rms: Math.sqrt(errors.reduce((s, e) => s + e * e, 0) / Math.max(1, errors.length)),
    min: Math.min(...errors), max: Math.max(...errors) };
}

function benchmark(id: string, steps: number) {
  const slice = createAdvanceSlice(productionSceneSliceSeedById(id));
  for (let i = 0; i < steps; i += 1) advanceSlice(slice, { pressureIterations: 4 });
  const plic = planes(slice), rdf = constructRdf(slice, plic), vertices = vertexRdf(slice, rdf, plic), iso = contour(slice, vertices);
  const cutPartialCells = slice.numericalTopology.cells.filter(cell => {
    const k = slice.fields.capacity[cell.id]!, fill = k > 1e-8 ? slice.fields.density[cell.id]! / k : 0;
    return k < 0.999999 && fill > 1e-6 && fill < 1 - 1e-6;
  }).length;
  return { id, frame: slice.frame, generation: slice.topology.accepted.generation,
    cells: slice.numericalTopology.cells.length, interfaceCells: plic.length,
    unsupportedCutPartialCells: cutPartialCells,
    plicJoins: plicJoinStats(slice, plic), rdfContour: { segments: iso.points.length / 2,
      ambiguousCells: iso.ambiguous, unresolvedCells: iso.unresolved,
      sharedEdgeGap: 0 }, rdfVolumeError: volumeError(slice, vertices),
    sphereRadialFineCells: sphereRadial(iso.points, slice) };
}

const cases: readonly [string, number][] = [
  ["coarse-first-pool-impact-half", 0], ["coarse-first-pool-impact-half", 1],
  ["falling-water-torus", 0], ["twin-dam-collision", 0],
  ["cm12-figure-3", 0], ["cm12-figure-7", 0],
  ["sparse-cm12-ladder-cut-mixed-seam-16", 0],
];
console.log(JSON.stringify(cases.map(([id, steps]) => benchmark(id, steps)), null, 2));
