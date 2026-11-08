import type { ClothTopologyData, ClothVector } from "../../types";

export interface HerringBoneSheet {
  /** Vertices along x (warp) and z (weft). Vertex (i, j) is index i + columns * j. */
  readonly columns: number;
  readonly rows: number;
  readonly width_m: number;
  readonly depth_m: number;
  /** World position of vertex (0, 0); the sheet lies in the plane y = origin y. */
  readonly origin_m: ClothVector;
  /** Attachment group of vertex (i, j); 0 is free. */
  readonly group?: (i: number, j: number) => number;
}

/**
 * PhysBAM's `Initialize_Herring_Bone_Mesh`, the 2002 cloth panel: each cell's
 * diagonal flips with its column only, so every interior vertex has six
 * triangles. (`clothGridData` alternates per cell, which gives interior
 * vertices four or eight triangles and so a 2:1 area-lumped mass checkerboard.)
 * Winding gives +y normals for the flat sheet.
 */
export function herringBoneSheet(o: HerringBoneSheet): ClothTopologyData {
  const { columns: m, rows: n } = o;
  if (!(Number.isInteger(m) && Number.isInteger(n) && m >= 2 && n >= 2)) throw new RangeError("Herring-bone sheet needs at least 2 x 2 vertices");
  if (!(o.width_m > 0 && o.depth_m > 0)) throw new RangeError("Herring-bone sheet needs positive extents");
  const rest: number[] = [], uv: number[] = [], groups: number[] = [], triangles: number[] = [];
  for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) {
    const u = o.width_m * i / (m - 1), v = o.depth_m * j / (n - 1);
    rest.push(o.origin_m[0] + u, o.origin_m[1], o.origin_m[2] + v);
    uv.push(u, v); groups.push(o.group?.(i, j) ?? 0);
  }
  for (let j = 0; j + 1 < n; j++) for (let i = 0; i + 1 < m; i++) {
    const a = i + m * j, b = a + 1, c = a + m, d = c + 1;
    if (i % 2 === 0) triangles.push(a, c, b, b, c, d); else triangles.push(a, d, b, a, c, d);
  }
  return { restPositions_m: rest, triangles, uv, groups };
}

/**
 * Paper section 3: "giving all particles the same mass instead of a mass
 * proportional to the area of the surrounding cloth", so edges and corners
 * are heavier, like a hem. Total mass is the sheet's area times its density.
 */
export function equalNodeMass_kg(columns: number, rows: number, width_m: number, depth_m: number, arealDensity_kg_m2: number): number {
  return arealDensity_kg_m2 * width_m * depth_m / (columns * rows);
}
