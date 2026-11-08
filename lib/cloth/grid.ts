import type { ClothTopologyData, ClothVector } from './types';

export interface ClothGridOptions {
  /** Vertices along warp (u) and weft (v). */
  readonly columns: number;
  readonly rows: number;
  readonly width_m: number;
  readonly height_m: number;
  /** World position of vertex (0, 0). */
  readonly origin_m: ClothVector;
  /** World directions of the warp and weft axes (unit, orthogonal). */
  readonly warp?: ClothVector;
  readonly weft?: ClothVector;
  /** Attachment group of vertex (column, row); 0 is free. */
  readonly group?: (column: number, row: number) => number;
}

/** A rectangular sheet with alternating diagonals (no preferred shear
 * direction) and material coordinates on the warp/weft lattice. */
export function clothGridData(o: ClothGridOptions): ClothTopologyData {
  const { columns: nu, rows: nv } = o;
  if (!(Number.isInteger(nu) && Number.isInteger(nv) && nu >= 2 && nv >= 2)) throw new Error('Cloth grid needs at least 2 x 2 vertices');
  const warp = o.warp ?? [1, 0, 0], weft = o.weft ?? [0, 0, 1];
  const rest: number[] = [], uv: number[] = [], triangles: number[] = [], groups: number[] = [];
  for (let j = 0; j < nv; j++) for (let i = 0; i < nu; i++) {
    const u = o.width_m * i / (nu - 1), v = o.height_m * j / (nv - 1);
    rest.push(o.origin_m[0] + u * warp[0] + v * weft[0], o.origin_m[1] + u * warp[1] + v * weft[1], o.origin_m[2] + u * warp[2] + v * weft[2]);
    uv.push(u, v); groups.push(o.group?.(i, j) ?? 0);
  }
  for (let j = 0; j + 1 < nv; j++) for (let i = 0; i + 1 < nu; i++) {
    const a = j * nu + i, b = a + 1, c = a + nu, d = c + 1;
    if ((i + j) % 2 === 0) triangles.push(a, b, d, a, d, c); else triangles.push(a, b, c, b, d, c);
  }
  return { restPositions_m: rest, triangles, uv, groups };
}
