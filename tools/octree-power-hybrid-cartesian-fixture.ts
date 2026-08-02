import { createOctreePowerSite } from "../lib/octree-power-geometry";
import {
  OCTREE_POWER_INVALID_ROW,
  type OctreePowerCompactRow,
  type OctreePowerFaceRecord,
  type OctreePowerIncidence,
  type OctreePowerOperator,
} from "../lib/octree-power-operator";

export interface CartesianPowerFixtureOptions {
  readonly openFraction?: (a: readonly [number, number, number], b: readonly [number, number, number]) => number;
}

/** Cheap exact regular-grid fixture; avoids invoking the general O(n^4) cell oracle in work censuses. */
export function buildCartesianPowerFixture(
  dimensions: readonly [number, number, number],
  options: CartesianPowerFixtureOptions = {},
): OctreePowerOperator {
  if (dimensions.some((value) => !Number.isSafeInteger(value) || value < 2)) {
    throw new RangeError("Cartesian power fixture dimensions must be integers >= 2");
  }
  const [nx, ny, nz] = dimensions;
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  const coordinate = (row: number): [number, number, number] => [
    row % nx, Math.floor(row / nx) % ny, Math.floor(row / (nx * ny)),
  ];
  const sites = Array.from({ length: nx * ny * nz }, (_, row) => {
    const [x, y, z] = coordinate(row);
    return createOctreePowerSite(`${z.toString().padStart(4, "0")}:${y.toString().padStart(4, "0")}:${x.toString().padStart(4, "0")}`, [x, y, z], 1);
  });
  // The keys above preserve linear row order under the operator's key sort.
  const incidence = sites.map(() => [] as OctreePowerIncidence[]);
  const faces: OctreePowerFaceRecord[] = [];
  const addFace = (negativeRow: number, positiveRow: number, normal: readonly [number, number, number],
    centroid: readonly [number, number, number], openFraction = 1, boundaryKey?: string) => {
    const id = faces.length;
    faces.push({
      id, negativeRow, positiveRow, key: `${negativeRow}|${positiveRow}:${boundaryKey ?? "site"}`,
      centroid, normal, area: 1, inverseDistance: positiveRow === OCTREE_POWER_INVALID_ROW ? 1 : 1,
      openFraction, normalVelocity: 0, ...(boundaryKey ? { boundaryKey } : {}),
    });
    incidence[negativeRow]!.push({ face: id, sign: 1 });
    if (positiveRow !== OCTREE_POWER_INVALID_ROW) incidence[positiveRow]!.push({ face: id, sign: -1 });
  };
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const row = index(x, y, z);
    for (const [axis, normal] of [[0, [1, 0, 0]], [1, [0, 1, 0]], [2, [0, 0, 1]]] as const) {
      const next = [x, y, z] as [number, number, number]; next[axis] += 1;
      const center = [x + 0.5, y + 0.5, z + 0.5] as [number, number, number]; center[axis] += 0.5;
      if (next[axis] < dimensions[axis]) {
        const neighbor = index(next[0], next[1], next[2]);
        addFace(row, neighbor, normal, center, options.openFraction?.([x, y, z], next) ?? 1);
      } else addFace(row, OCTREE_POWER_INVALID_ROW, normal, center, 1, `${axis}+`);
    }
    for (const [axis, normal] of [[0, [-1, 0, 0]], [1, [0, -1, 0]], [2, [0, 0, -1]]] as const) {
      if ([x, y, z][axis] !== 0) continue;
      const center = [x + 0.5, y + 0.5, z + 0.5] as [number, number, number]; center[axis] -= 0.5;
      addFace(row, OCTREE_POWER_INVALID_ROW, normal, center, 1, `${axis}-`);
    }
  }
  for (const row of incidence) row.sort((a, b) => a.face - b.face);
  const rows: OctreePowerCompactRow[] = sites.map((site, row) => {
    const entries = new Map<number, number>(); let diagonal = 0;
    for (const item of incidence[row]!) {
      const face = faces[item.face]!;
      const coefficient = face.openFraction * face.area * face.inverseDistance;
      diagonal += coefficient;
      const neighbor = face.negativeRow === row ? face.positiveRow : face.negativeRow;
      if (neighbor !== OCTREE_POWER_INVALID_ROW) entries.set(neighbor, coefficient);
    }
    return { row, siteKey: site.key, diagonal, rhs: 0, volume: 1,
      entries: [...entries].sort((a, b) => a[0] - b[0]).map(([neighbor, coefficient]) => ({ row: neighbor, coefficient })) };
  });
  return {
    sites, faces, incidence, rows,
    maximumIncidence: Math.max(...incidence.map((row) => row.length)),
    maximumNeighborRows: Math.max(...rows.map((row) => row.entries.length)),
  };
}
