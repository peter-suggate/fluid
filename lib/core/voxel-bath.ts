import type { SolidWorldVoxelPatch } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

export const BATH_FLOOR_HEIGHT_M = 0.05;

/** Rounded rectangular cavity, widening smoothly from the base to the rim. */
export function bathInteriorContains(x: number, y: number, z: number): boolean {
  return y >= BATH_FLOOR_HEIGHT_M && y < 1.2 && bathDistance(x, y, z) < 0;
}

function bathDistance(x: number, y: number, z: number): number {
  const t = Math.max(0, Math.min(1, (y - BATH_FLOOR_HEIGHT_M) / (1.2 - BATH_FLOOR_HEIGHT_M)));
  const flare = Math.sin(t * Math.PI / 2);
  const radius = 0.22;
  const qx = Math.abs(x) - (1.25 + 0.3 * flare - radius);
  const qz = Math.abs(z) - (0.28 + 0.27 * flare - radius);
  return Math.hypot(Math.max(qx, 0), Math.max(qz, 0))
    + Math.min(Math.max(qx, qz), 0) - radius;
}

/** Compact voxel boxes: merge X runs through Y, then merge matching Z rows. */
export function bathVoxelBoxes(
  dimensions: readonly [number, number, number],
  contains: (x: number, y: number, z: number) => boolean,
): SolidWorldVoxelPatch[] {
  const [nx, ny, nz] = dimensions;
  const rows: SolidWorldVoxelPatch[] = [];
  for (let z = 0; z < nz; z++) {
    let previous = new Map<string, SolidWorldVoxelPatch>();
    for (let y = 0; y < ny; y++) {
      const current = new Map<string, SolidWorldVoxelPatch>();
      for (let x = 0; x < nx;) {
        if (!contains(x, y, z)) { x++; continue; }
        const start = x++;
        while (x < nx && contains(x, y, z)) x++;
        const key = `${start}:${x}`;
        const prior = previous.get(key);
        const patch: SolidWorldVoxelPatch = {
          operation: "fill", minimum: prior?.minimum ?? [start, y, z],
          maximumExclusive: [x, y + 1, z + 1], materialId: VOXEL_MATERIAL_IDS.box,
        };
        if (prior) rows[rows.indexOf(prior)] = patch;
        else rows.push(patch);
        current.set(key, patch);
      }
      previous = current;
    }
  }
  const merged: SolidWorldVoxelPatch[] = [];
  const last = new Map<string, number>();
  for (const patch of rows) {
    const key = [patch.minimum[0], patch.minimum[1], patch.maximumExclusive[0], patch.maximumExclusive[1]].join(":");
    const index = last.get(key);
    const prior = index === undefined ? undefined : merged[index];
    if (prior && prior.maximumExclusive[2] === patch.minimum[2]) {
      merged[index!] = { ...prior, maximumExclusive: patch.maximumExclusive };
    } else {
      last.set(key, merged.length);
      merged.push(patch);
    }
  }
  return merged;
}

export function bathSolidContains(x: number, y: number, z: number): boolean {
  if (y < 0 || y >= 1.2) return false;
  const distance = bathDistance(x, y, z);
  return distance < 0.05 && (y < BATH_FLOOR_HEIGHT_M || distance >= 0);
}
