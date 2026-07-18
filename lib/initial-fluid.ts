import type { SceneDescription, Vec3 } from "./model";

export interface DamBreakFractions {
  width: number;
  height: number;
  depth: number;
}

const DAM_BREAK_HEIGHT = 0.92;

/**
 * Builds a square-footprint reservoir in the tank's lower (-x, -z) corner.
 * The footprint grows with requested fill so width * height * depth remains
 * equal to the scene fill fraction (up to a completely full tank).
 */
export function damBreakFractions(fillFraction: number): DamBreakFractions {
  const fill = Math.max(0, Math.min(1, fillFraction));
  if (fill === 0) return { width: 0, height: 0, depth: 0 };
  const height = Math.max(DAM_BREAK_HEIGHT, fill);
  const footprint = Math.sqrt(fill / height);
  return { width: footprint, height, depth: footprint };
}

export const INITIAL_FLUID_BRICK_SIZE = 8;

function seedCell(scene: SceneDescription, seed: Vec3, dimensions: readonly [number, number, number]) {
  const c = scene.container;
  return {
    x: Math.min(dimensions[0] - 1, Math.max(0, Math.floor((seed.x / c.width_m + 0.5) * dimensions[0]))),
    y: Math.min(dimensions[1] - 1, Math.max(0, Math.floor(seed.y / c.height_m * dimensions[1]))),
    z: Math.min(dimensions[2] - 1, Math.max(0, Math.floor((seed.z / c.depth_m + 0.5) * dimensions[2]))),
  };
}

/** True when a finest cell belongs to any explicitly seeded initial brick. */
export function initialFluidBrickContainsCell(
  scene: SceneDescription,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): boolean | undefined {
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds) return undefined;
  const bx = Math.floor(x / brickSize), by = Math.floor(y / brickSize), bz = Math.floor(z / brickSize);
  return seeds.some((seed) => {
    const cell = seedCell(scene, seed, dimensions);
    return Math.floor(cell.x / brickSize) === bx && Math.floor(cell.y / brickSize) === by && Math.floor(cell.z / brickSize) === bz;
  });
}

/**
 * Resolves the seeded-brick occupancy against the scene's base initial
 * condition. Seeds ordinarily replace the base fill entirely (disconnected
 * initial bodies); with `initialBrickSeedsAdditive` they union with it so a
 * seeded slab can sit on top of a settled pool.
 */
export function combineInitialBrickWet(
  scene: SceneDescription,
  brickWet: boolean | undefined,
  baseWet: boolean,
): boolean {
  if (brickWet === undefined) return baseWet;
  return scene.fluid.initialBrickSeedsAdditive ? brickWet || baseWet : brickWet;
}

function boxSignedDistance(point: Vec3, minimum: Vec3, maximum: Vec3): number {
  const center = { x: 0.5 * (minimum.x + maximum.x), y: 0.5 * (minimum.y + maximum.y), z: 0.5 * (minimum.z + maximum.z) };
  const half = { x: 0.5 * (maximum.x - minimum.x), y: 0.5 * (maximum.y - minimum.y), z: 0.5 * (maximum.z - minimum.z) };
  const qx = Math.abs(point.x - center.x) - half.x;
  const qy = Math.abs(point.y - center.y) - half.y;
  const qz = Math.abs(point.z - center.z) - half.z;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
}

/** Signed distance to the union of explicitly seeded initial bricks. */
export function initialFluidBrickSignedDistance(
  scene: SceneDescription,
  point: Vec3,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): number | undefined {
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds) return undefined;
  const c = scene.container;
  const h = [c.width_m / dimensions[0], c.height_m / dimensions[1], c.depth_m / dimensions[2]] as const;
  let result = Number.POSITIVE_INFINITY;
  for (const seed of seeds) {
    const cell = seedCell(scene, seed, dimensions);
    const origin = [Math.floor(cell.x / brickSize) * brickSize, Math.floor(cell.y / brickSize) * brickSize, Math.floor(cell.z / brickSize) * brickSize] as const;
    const end = [Math.min(dimensions[0], origin[0] + brickSize), Math.min(dimensions[1], origin[1] + brickSize), Math.min(dimensions[2], origin[2] + brickSize)] as const;
    result = Math.min(result, boxSignedDistance(point,
      { x: -0.5 * c.width_m + origin[0] * h[0], y: origin[1] * h[1], z: -0.5 * c.depth_m + origin[2] * h[2] },
      { x: -0.5 * c.width_m + end[0] * h[0], y: end[1] * h[1], z: -0.5 * c.depth_m + end[2] * h[2] }));
  }
  return result;
}

export function inflowStrength(time_s: number, start_s: number, end_s: number, ramp_s: number): number {
  if (time_s < start_s || time_s >= end_s) return 0;
  if (ramp_s <= 0) return 1;
  return Math.min(1, (time_s - start_s) / ramp_s, (end_s - time_s) / ramp_s);
}
