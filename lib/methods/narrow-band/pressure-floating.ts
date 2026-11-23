/**
 * The Float64 mirror of pressure-floating.wgsl.ts (a detail engine's floating components) on level-0 rows keyed by
 * packed cell (pressure-model.ts FineRows): the rows' connected components by their couplings, a component's anchor
 * (a row whose A 1 is above FLOAT_EXCESS of its diagonal), and on each component without one its data made compatible
 * (its mean RHS taken from each row) and its gauge pinned at its root (its least row) to the parent's pressure g: the
 * root's diagonal doubles and its RHS gains diagonal x g. With compatible data the pin is exact: x_root = g and A x = b.
 */
import { FLOAT_EXCESS } from "./pressure-floating.wgsl";
import { packCell, unpackCell, type FineRows } from "./pressure-model";

export interface FloatingRows extends FineRows { readonly rhs: ArrayLike<number> }
export interface FloatingComponent {
  /** The root: the component's least row index. */
  readonly root: number;
  readonly rows: number;
  readonly anchored: boolean;
  /** The component's RHS sum. */
  readonly sum: number;
}
export interface FloatingResult {
  readonly diag: Float64Array;
  readonly rhs: Float64Array;
  /** Each row's component root. */
  readonly root: Int32Array;
  readonly components: readonly FloatingComponent[];
}

const AX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;

/** The rows' components and, on each floating one, the compatible data and the pin to gauge(root). */
export function anchorFloating(rows: FloatingRows, gauge: (row: number) => number, excess = FLOAT_EXCESS): FloatingResult {
  const n = rows.keys.length, index = new Map<number, number>();
  rows.keys.forEach((k, i) => { if (rows.diag[i]! > 0) index.set(k, i); });
  const parent = Int32Array.from({ length: n }, (_, i) => i);
  const find = (x: number) => { while (parent[x] !== x) { parent[x] = parent[parent[x]!]!; x = parent[x]!; } return x; };
  const lower = (i: number, a: number) => { const [x, y, z] = unpackCell(rows.keys[i]!), d = AX[a]!; return index.get(packCell(x - d[0], y - d[1], z - d[2])); };
  const upper = (i: number, a: number) => { const [x, y, z] = unpackCell(rows.keys[i]!), d = AX[a]!; return index.get(packCell(x + d[0], y + d[1], z + d[2])); };
  for (const i of index.values()) for (let a = 0; a < 3; a++) {
    if (!rows.k[a]![i]) continue;
    const j = lower(i, a); if (j === undefined) continue;
    const ri = find(i), rj = find(j); if (ri !== rj) parent[Math.max(ri, rj)] = Math.min(ri, rj);
  }
  const root = new Int32Array(n).fill(-1), acc = new Map<number, { rows: number; anchored: boolean; sum: number }>();
  for (const i of index.values()) {
    const r = find(i); root[i] = r;
    let e = rows.diag[i]!;
    for (let a = 0; a < 3; a++) {
      if (rows.k[a]![i] && lower(i, a) !== undefined) e -= rows.k[a]![i]!;
      const u = upper(i, a); if (u !== undefined) e -= rows.k[a]![u]!;
    }
    const c = acc.get(r) ?? { rows: 0, anchored: false, sum: 0 };
    c.rows++; c.anchored ||= e > excess * rows.diag[i]!; c.sum += rows.rhs[i]!;
    acc.set(r, c);
  }
  const diag = Float64Array.from(rows.diag as ArrayLike<number>), rhs = Float64Array.from(rows.rhs as ArrayLike<number>);
  for (const i of index.values()) {
    const r = root[i]!, c = acc.get(r)!;
    if (c.anchored) continue;
    rhs[i]! -= c.sum / c.rows;
    if (i === r) { rhs[i]! += diag[i]! * gauge(i); diag[i]! *= 2; }
  }
  return { diag, rhs, root, components: [...acc].map(([r, c]) => ({ root: r, ...c })) };
}
