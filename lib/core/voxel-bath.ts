import type { SolidWorldVoxelPatch } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";

export const BATH_FLOOR_HEIGHT_M = 0.05;
const BATH_HEIGHT_M = 1.2;
const BATH_WALL_M = 0.05;
const BATH_CORNER_RADIUS_M = 0.22;
/** Half the cavity's length and width at the base, and what each gains by the rim. */
const BATH_BASE_HALF_M = [1.25, 0.28] as const;
const BATH_FLARE_M = [0.3, 0.27] as const;

/** Rounded rectangular cavity, widening smoothly from the base to the rim. */
export function bathInteriorContains(x: number, y: number, z: number): boolean {
  return y >= BATH_FLOOR_HEIGHT_M && y < BATH_HEIGHT_M && bathDistance(x, y, z) < 0;
}

function bathDistance(x: number, y: number, z: number): number {
  const t = Math.max(0, Math.min(1, (y - BATH_FLOOR_HEIGHT_M) / (BATH_HEIGHT_M - BATH_FLOOR_HEIGHT_M)));
  const flare = Math.sin(t * Math.PI / 2);
  const radius = BATH_CORNER_RADIUS_M;
  const qx = Math.abs(x) - (BATH_BASE_HALF_M[0] + BATH_FLARE_M[0] * flare - radius);
  const qz = Math.abs(z) - (BATH_BASE_HALF_M[1] + BATH_FLARE_M[1] * flare - radius);
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
  if (y < 0 || y >= BATH_HEIGHT_M) return false;
  const distance = bathDistance(x, y, z);
  return distance < BATH_WALL_M && (y < BATH_FLOOR_HEIGHT_M || distance >= 0);
}

/**
 * The solid `bathSolidContains` rasterizes, as a field: negative inside the
 * shell and its base, zero on every face, and close to the distance from one.
 */
export function bathSolidDistance(x: number, y: number, z: number): number {
  const distance = bathDistance(x, y, z);
  return Math.max(distance - BATH_WALL_M, -y, y - BATH_HEIGHT_M,
    Math.min(-distance, y - BATH_FLOOR_HEIGHT_M));
}

const f = (value: number) => Number.isInteger(value) ? `${value}.0` : String(value);

/** `bathSolidDistance` in world metres, as the voxeliser's `swAuthoredDistance`. */
export const BATH_SOLID_DISTANCE_WGSL = /* wgsl */ `
fn swAuthoredDistance(world:vec3f)->f32{
  let t=clamp((world.y-${f(BATH_FLOOR_HEIGHT_M)})/${f(BATH_HEIGHT_M - BATH_FLOOR_HEIGHT_M)},0.0,1.0);
  let q=abs(world.xz)-(vec2f(${BATH_BASE_HALF_M.map(f).join(",")})
    +vec2f(${BATH_FLARE_M.map(f).join(",")})*sin(t*${f(Math.PI / 2)})-vec2f(${f(BATH_CORNER_RADIUS_M)}));
  let distance=length(max(q,vec2f(0.0)))+min(max(q.x,q.y),0.0)-${f(BATH_CORNER_RADIUS_M)};
  return max(max(distance-${f(BATH_WALL_M)},-world.y),
    max(world.y-${f(BATH_HEIGHT_M)},min(-distance,world.y-${f(BATH_FLOOR_HEIGHT_M)})));
}`;
