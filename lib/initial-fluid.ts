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

/** Resolves an authored absolute reservoir size, falling back to the legacy
 * fill-derived normalized dam shape. Size only: callers that must honour a
 * reservoir dragged off the container corner want `sceneDamBreakBox`. */
export function sceneDamBreakFractions(scene: SceneDescription): DamBreakFractions {
  const dimensions = scene.fluid.initialDamBreakDimensions_m;
  if (!dimensions) return damBreakFractions(scene.container.fillFraction);
  return {
    width: dimensions.x / scene.container.width_m,
    height: dimensions.y / scene.container.height_m,
    depth: dimensions.z / scene.container.depth_m,
  };
}

/** Normalized reservoir bounds on each container axis, in [0, 1]. */
export interface DamBreakBox {
  readonly min: { x: number; y: number; z: number };
  readonly max: { x: number; y: number; z: number };
}

/**
 * The dam reservoir as a box rather than a size.
 *
 * Every seeding path tests cell centres against this. Without an authored
 * origin the minimum corner is the container's own, which reproduces the
 * corner-anchored test exactly — `min` is zero, so `fraction > 0` admits every
 * cell the legacy `fraction <= size` test admitted.
 */
export function sceneDamBreakBox(scene: SceneDescription): DamBreakBox {
  const size = sceneDamBreakFractions(scene);
  const origin = scene.fluid.initialDamBreakOrigin_m;
  const c = scene.container;
  const min = origin
    ? { x: origin.x / c.width_m, y: origin.y / c.height_m, z: origin.z / c.depth_m }
    : { x: 0, y: 0, z: 0 };
  return {
    min,
    max: { x: min.x + size.width, y: min.y + size.height, z: min.z + size.depth },
  };
}

/** Whether a normalized cell centre lies inside the reservoir. */
export function damBreakBoxContains(box: DamBreakBox, x: number, y: number, z: number): boolean {
  return x > box.min.x && x <= box.max.x
    && y > box.min.y && y <= box.max.y
    && z > box.min.z && z <= box.max.z;
}

/**
 * True when the reservoir has been moved off the container's minimum corner.
 *
 * The GPU's closed-form t=0 phi (`analyticInitialPhi`) hard-codes the corner
 * anchor, so an offset reservoir must fall back to the host-rasterized seed
 * — the same path terrain, rigid bodies, and brick seeds already take.
 */
export function sceneDamBreakIsOffsetFromCorner(scene: SceneDescription): boolean {
  const origin = scene.fluid.initialDamBreakOrigin_m;
  if (!origin || !scene.fluid.initialDamBreakDimensions_m) return false;
  return Math.abs(origin.x) > 1e-9 || Math.abs(origin.y) > 1e-9 || Math.abs(origin.z) > 1e-9;
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

export interface InitialFluidBrickUnionBounds {
  readonly minimum: Vec3;
  readonly maximum: Vec3;
}

function initialFluidBrickCoordinates(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize: number,
): readonly [number, number, number][] | undefined {
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds?.length || scene.fluid.initialBrickSeedsAdditive) return undefined;
  const unique = new Map<string, [number, number, number]>();
  for (const seed of seeds) {
    const cell = seedCell(scene, seed, dimensions);
    const brick: [number, number, number] = [Math.floor(cell.x / brickSize),
      Math.floor(cell.y / brickSize), Math.floor(cell.z / brickSize)];
    unique.set(brick.join(","), brick);
  }
  return [...unique.values()];
}

function brickBounds(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize: number,
  coordinates: readonly [number, number, number][],
): InitialFluidBrickUnionBounds {
  const minimumBrick = [0, 1, 2].map((axis) =>
    Math.min(...coordinates.map((q) => q[axis]!))) as [number, number, number];
  const maximumBrick = [0, 1, 2].map((axis) =>
    Math.max(...coordinates.map((q) => q[axis]!))) as [number, number, number];
  const c = scene.container;
  const h = [c.width_m / dimensions[0], c.height_m / dimensions[1], c.depth_m / dimensions[2]] as const;
  const origin = [minimumBrick[0] * brickSize, minimumBrick[1] * brickSize,
    minimumBrick[2] * brickSize] as const;
  const end = [Math.min(dimensions[0], (maximumBrick[0] + 1) * brickSize),
    Math.min(dimensions[1], (maximumBrick[1] + 1) * brickSize),
    Math.min(dimensions[2], (maximumBrick[2] + 1) * brickSize)] as const;
  return {
    minimum: { x: -0.5 * c.width_m + origin[0] * h[0], y: origin[1] * h[1],
      z: -0.5 * c.depth_m + origin[2] * h[2] },
    maximum: { x: -0.5 * c.width_m + end[0] * h[0], y: end[1] * h[1],
      z: -0.5 * c.depth_m + end[2] * h[2] },
  };
}

function filledBrickBox(coordinates: readonly [number, number, number][]): boolean {
  const keys = new Set(coordinates.map((brick) => brick.join(",")));
  const minimum = [0, 1, 2].map((axis) =>
    Math.min(...coordinates.map((q) => q[axis]!))) as [number, number, number];
  const maximum = [0, 1, 2].map((axis) =>
    Math.max(...coordinates.map((q) => q[axis]!))) as [number, number, number];
  const expected = (maximum[0] - minimum[0] + 1)
    * (maximum[1] - minimum[1] + 1) * (maximum[2] - minimum[2] + 1);
  if (keys.size !== expected) return false;
  for (let bz = minimum[2]; bz <= maximum[2]; bz += 1)
    for (let by = minimum[1]; by <= maximum[1]; by += 1)
      for (let bx = minimum[0]; bx <= maximum[0]; bx += 1)
        if (!keys.has(`${bx},${by},${bz}`)) return false;
  return true;
}

/** Exact bounds for disconnected rectangular components of authored brick
 * seeds. Components that are L-shaped or otherwise non-rectangular return
 * undefined: replacing them with bounding boxes would invent liquid. */
export function initialFluidBrickComponentBounds(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): readonly InitialFluidBrickUnionBounds[] | undefined {
  const coordinates = initialFluidBrickCoordinates(scene, dimensions, brickSize);
  if (!coordinates) return undefined;
  const remaining = new Map(coordinates.map((brick) => [brick.join(","), brick]));
  const components: [number, number, number][][] = [];
  while (remaining.size > 0) {
    const first = remaining.values().next().value as [number, number, number];
    const component: [number, number, number][] = [];
    const pending = [first];
    remaining.delete(first.join(","));
    while (pending.length > 0) {
      const brick = pending.pop()!;
      component.push(brick);
      for (let axis = 0; axis < 3; axis += 1) for (const direction of [-1, 1]) {
        const neighbor = [...brick] as [number, number, number];
        neighbor[axis] += direction;
        const key = neighbor.join(",");
        const present = remaining.get(key);
        if (present) { remaining.delete(key); pending.push(present); }
      }
    }
    if (!filledBrickBox(component)) return undefined;
    components.push(component);
  }
  return components.map((component) => brickBounds(scene, dimensions, brickSize, component));
}

/** Returns exact world-space bounds when the authored brick seeds form one
 * completely filled axis-aligned box. Disconnected or L-shaped unions return
 * undefined rather than being silently replaced by their bounding box. */
export function initialFluidBrickUnionBounds(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): InitialFluidBrickUnionBounds | undefined {
  const coordinates = initialFluidBrickCoordinates(scene, dimensions, brickSize);
  if (!coordinates || !filledBrickBox(coordinates)) return undefined;
  return brickBounds(scene, dimensions, brickSize, coordinates);
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

/** Signed distance to seeded bricks at a finest-cell centre, evaluated from
 * integer brick bounds. This is the canonical bootstrap form: reflected cell
 * centres reach identical half-cell distances before the single conversion
 * to metres, avoiding world-origin cancellation in exact symmetry gates. */
export function initialFluidBrickSignedDistanceAtCell(
  scene: SceneDescription,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): number | undefined {
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds) return undefined;
  const c = scene.container;
  const h = [c.width_m / dimensions[0], c.height_m / dimensions[1],
    c.depth_m / dimensions[2]] as const;
  const point = [x + 0.5, y + 0.5, z + 0.5] as const;
  let result = Number.POSITIVE_INFINITY;
  for (const seed of seeds) {
    const cell = seedCell(scene, seed, dimensions);
    const origin = [Math.floor(cell.x / brickSize) * brickSize,
      Math.floor(cell.y / brickSize) * brickSize,
      Math.floor(cell.z / brickSize) * brickSize] as const;
    const end = [Math.min(dimensions[0], origin[0] + brickSize),
      Math.min(dimensions[1], origin[1] + brickSize),
      Math.min(dimensions[2], origin[2] + brickSize)] as const;
    const qx = Math.max(origin[0] - point[0], point[0] - end[0]) * h[0];
    const qy = Math.max(origin[1] - point[1], point[1] - end[1]) * h[1];
    const qz = Math.max(origin[2] - point[2], point[2] - end[2]) * h[2];
    result = Math.min(result,
      Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0))
        + Math.min(Math.max(qx, qy, qz), 0));
  }
  return result;
}

export function inflowStrength(time_s: number, start_s: number, end_s: number, ramp_s: number): number {
  if (time_s < start_s || time_s >= end_s) return 0;
  if (ramp_s <= 0) return 1;
  return Math.min(1, (time_s - start_s) / ramp_s, (end_s - time_s) / ramp_s);
}
