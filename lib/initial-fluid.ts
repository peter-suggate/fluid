import type { InitialLiquidVolume, SceneDescription, Vec3 } from "./model";

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

/** The scene's canonical analytic t=0 volumes. */
export function sceneInitialLiquidVolumes(scene: SceneDescription): readonly InitialLiquidVolume[] {
  return scene.fluid.initialLiquidVolumes ?? [];
}

export function sceneHasInitialLiquidVolumes(scene: SceneDescription): boolean {
  return sceneInitialLiquidVolumes(scene).length > 0;
}

function normalizedNormal(normal: Vec3): Vec3 {
  const length = Math.hypot(normal.x, normal.y, normal.z);
  return length > 1e-12
    ? { x: normal.x / length, y: normal.y / length, z: normal.z / length }
    : { x: 1, y: 0, z: 0 };
}

export function initialLiquidVolumeContainsPoint(volume: InitialLiquidVolume, point: Vec3): boolean {
  if (volume.shape === "box") return point.x >= volume.min_m.x && point.x <= volume.max_m.x
    && point.y >= volume.min_m.y && point.y <= volume.max_m.y
    && point.z >= volume.min_m.z && point.z <= volume.max_m.z;
  const dx = point.x - volume.center_m.x;
  const dy = point.y - volume.center_m.y;
  const dz = point.z - volume.center_m.z;
  if (Math.hypot(dx, dy, dz) > volume.radius_m) return false;
  if (volume.shape === "sphere") return true;
  const normal = normalizedNormal(volume.outwardNormal);
  return dx * normal.x + dy * normal.y + dz * normal.z <= 0;
}

function analyticBoxSignedDistance(point: Vec3, min: Vec3, max: Vec3): number {
  const center = { x: 0.5 * (min.x + max.x), y: 0.5 * (min.y + max.y), z: 0.5 * (min.z + max.z) };
  const half = { x: 0.5 * (max.x - min.x), y: 0.5 * (max.y - min.y), z: 0.5 * (max.z - min.z) };
  const qx = Math.abs(point.x - center.x) - half.x;
  const qy = Math.abs(point.y - center.y) - half.y;
  const qz = Math.abs(point.z - center.z) - half.z;
  return Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0))
    + Math.min(Math.max(qx, qy, qz), 0);
}

export function initialLiquidVolumeSignedDistance(volume: InitialLiquidVolume, point: Vec3): number {
  if (volume.shape === "box") return analyticBoxSignedDistance(point, volume.min_m, volume.max_m);
  const delta = {
    x: point.x - volume.center_m.x,
    y: point.y - volume.center_m.y,
    z: point.z - volume.center_m.z,
  };
  const sphere = Math.hypot(delta.x, delta.y, delta.z) - volume.radius_m;
  if (volume.shape === "sphere") return sphere;
  const normal = normalizedNormal(volume.outwardNormal);
  const plane = delta.x * normal.x + delta.y * normal.y + delta.z * normal.z;
  // Intersection of the ball and its retained half-space. This is an exact
  // sign field (and exact at either smooth face), which is what phi seeding
  // needs; the max is intentionally sharp at the circular rim.
  return Math.max(sphere, plane);
}

export function initialLiquidVolumesSignedDistance(
  scene: SceneDescription,
  point: Vec3,
): number | undefined {
  const volumes = sceneInitialLiquidVolumes(scene);
  if (volumes.length === 0) return undefined;
  return Math.min(...volumes.map((volume) => initialLiquidVolumeSignedDistance(volume, point)));
}

/** Whether a finest-cell centre lies inside any authored analytic volume. */
export function initialLiquidVolumeContainsCell(
  scene: SceneDescription,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
): boolean {
  const volumes = sceneInitialLiquidVolumes(scene);
  if (volumes.length === 0) return false;
  const c = scene.container;
  const point = {
    x: -0.5 * c.width_m + (x + 0.5) * c.width_m / dimensions[0],
    y: (y + 0.5) * c.height_m / dimensions[1],
    z: -0.5 * c.depth_m + (z + 0.5) * c.depth_m / dimensions[2],
  };
  return volumes.some((volume) => initialLiquidVolumeContainsPoint(volume, point));
}

/**
 * Fractional t=0 occupancy at a cell using the same eight samples as spherical
 * cut cells. Base fills and painted bricks remain exact whole-cell inputs;
 * analytic volumes receive the fractional boundary they were authored for.
 */
export function initialLiquidFractionAtCell(
  scene: SceneDescription,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
  baseWet: boolean,
): number {
  const brick = initialFluidBrickContainsCell(scene, x, y, z, dimensions);
  if (combineInitialBrickWet(scene, brick, baseWet)) return 1;
  const volumes = sceneInitialLiquidVolumes(scene);
  if (volumes.length === 0) return 0;
  const c = scene.container;
  const h = [c.width_m / dimensions[0], c.height_m / dimensions[1], c.depth_m / dimensions[2]] as const;
  const center = {
    x: -0.5 * c.width_m + (x + 0.5) * h[0],
    y: (y + 0.5) * h[1],
    z: -0.5 * c.depth_m + (z + 0.5) * h[2],
  };
  let wetSamples = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const point = {
      x: center.x + ((corner & 1) ? 0.4 : -0.4) * h[0],
      y: center.y + ((corner & 2) ? 0.4 : -0.4) * h[1],
      z: center.z + ((corner & 4) ? 0.4 : -0.4) * h[2],
    };
    if (volumes.some((volume) => initialLiquidVolumeContainsPoint(volume, point))) wetSamples += 1;
  }
  return wetSamples / 8;
}

/**
 * Resolves the whole t = 0 liquid occupancy at a finest cell.
 *
 * The base condition, seeded bricks and analytic volumes in the one order every
 * seeding site has to agree on. Volumes union with the brick/base resolution;
 * the terrain gate stays with the caller because only it knows whether it owns
 * a baked column table or a live sample.
 */
export function initialLiquidContainsCell(
  scene: SceneDescription,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
  baseWet: boolean,
): boolean {
  const brick = initialFluidBrickContainsCell(scene, x, y, z, dimensions);
  return combineInitialBrickWet(scene, brick, baseWet)
    || initialLiquidVolumeContainsCell(scene, x, y, z, dimensions);
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

/**
 * Which brick each seed wets, in document order, deduplicated.
 *
 * Independent of the base initial condition, because "which bricks did the
 * author seed" is a fact about the seed list alone. The *solver* entry points
 * below add the additive gate — an additive scene has no exact analytic box,
 * since the base fill is present too — but the editor needs the decomposition
 * either way, so the gate lives with the callers that need it rather than here.
 */
function seedBrickCoordinates(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize: number,
): Map<string, { readonly brick: [number, number, number]; readonly seedIndices: number[] }> {
  const unique = new Map<string, { brick: [number, number, number]; seedIndices: number[] }>();
  (scene.fluid.initialBrickSeeds_m ?? []).forEach((seed, index) => {
    const cell = seedCell(scene, seed, dimensions);
    const brick: [number, number, number] = [Math.floor(cell.x / brickSize),
      Math.floor(cell.y / brickSize), Math.floor(cell.z / brickSize)];
    const key = brick.join(",");
    const existing = unique.get(key);
    if (existing) existing.seedIndices.push(index);
    else unique.set(key, { brick, seedIndices: [index] });
  });
  return unique;
}

function initialFluidBrickCoordinates(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize: number,
): readonly [number, number, number][] | undefined {
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (!seeds?.length || scene.fluid.initialBrickSeedsAdditive) return undefined;
  return [...seedBrickCoordinates(scene, dimensions, brickSize).values()].map((entry) => entry.brick);
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

/**
 * One connected body of seeded bricks: where it is, which seeds made it, and
 * whether it is a box.
 *
 * A body rather than a bounding box, because the two questions downstream are
 * different. The solver's analytic bootstrap may only use a component that
 * *fills* its bounds — a bounding box around an L would invent liquid. The
 * editor wants every component regardless, so a painted blob is still something
 * you can point at and move; it just may not be something you can resize.
 */
export interface InitialFluidBrickComponent {
  readonly bounds: InitialFluidBrickUnionBounds;
  /** Indices into `fluid.initialBrickSeeds_m`, ascending. */
  readonly seedIndices: readonly number[];
  /**
   * Distinct bricks the component wets — its volume, in bricks. Not the seed
   * count: several seeds may land in one brick, and each wets it once.
   */
  readonly brickCount: number;
  /** True when the component fills its own bounding box exactly. */
  readonly rectangular: boolean;
}

/**
 * Every connected component of the authored brick seeds, in document order.
 *
 * Unlike `initialFluidBrickComponentBounds` this answers for an additive scene
 * too, and reports non-rectangular components rather than refusing the whole
 * set: it describes what the author wrote, and each caller decides what it may
 * do with a component that is not a box.
 */
export function initialFluidBrickComponents(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): readonly InitialFluidBrickComponent[] {
  const remaining = seedBrickCoordinates(scene, dimensions, brickSize);
  const components: InitialFluidBrickComponent[] = [];
  while (remaining.size > 0) {
    const first = remaining.values().next().value!;
    const bricks: [number, number, number][] = [];
    const seedIndices: number[] = [];
    const pending = [first];
    remaining.delete(first.brick.join(","));
    while (pending.length > 0) {
      const entry = pending.pop()!;
      bricks.push(entry.brick);
      seedIndices.push(...entry.seedIndices);
      for (let axis = 0; axis < 3; axis += 1) for (const direction of [-1, 1]) {
        const neighbor = [...entry.brick] as [number, number, number];
        neighbor[axis] += direction;
        const key = neighbor.join(",");
        const present = remaining.get(key);
        if (present) { remaining.delete(key); pending.push(present); }
      }
    }
    components.push({
      bounds: brickBounds(scene, dimensions, brickSize, bricks),
      seedIndices: seedIndices.sort((left, right) => left - right),
      brickCount: bricks.length,
      rectangular: filledBrickBox(bricks),
    });
  }
  return components;
}

/** Exact bounds for disconnected rectangular components of authored brick
 * seeds. Components that are L-shaped or otherwise non-rectangular return
 * undefined: replacing them with bounding boxes would invent liquid. */
export function initialFluidBrickComponentBounds(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): readonly InitialFluidBrickUnionBounds[] | undefined {
  if (!initialFluidBrickCoordinates(scene, dimensions, brickSize)) return undefined;
  const components = initialFluidBrickComponents(scene, dimensions, brickSize);
  if (components.some((component) => !component.rectangular)) return undefined;
  return components.map((component) => component.bounds);
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

/** Exact signed distance to the procedural dam reservoir at an integer
 * finest-lattice node. This is construction-only source data for adaptive
 * nodal phi; it does not add a recurring analytic simulation path. */
export function damBreakSignedDistanceAtNode(
  scene: SceneDescription,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
): number | undefined {
  if (scene.fluid.initialCondition !== "dam-break") return undefined;
  const dam = sceneDamBreakBox(scene), c = scene.container;
  const minimum = { x: -0.5 * c.width_m + dam.min.x * c.width_m,
    y: dam.min.y * c.height_m,
    z: -0.5 * c.depth_m + dam.min.z * c.depth_m };
  const maximum = { x: -0.5 * c.width_m + dam.max.x * c.width_m,
    y: dam.max.y * c.height_m,
    z: -0.5 * c.depth_m + dam.max.z * c.depth_m };
  const point = { x: -0.5 * c.width_m + x * c.width_m / dimensions[0],
    y: y * c.height_m / dimensions[1],
    z: -0.5 * c.depth_m + z * c.depth_m / dimensions[2] };

  // A tank-contact face is solid, not a liquid-air interface. Continue the
  // liquid through closed walls when constructing phi so the smoothed rho
  // bootstrap does not lose half of its interface kernel outside the domain.
  // Offset reservoir faces remain exposed, as does a reservoir touching an
  // open top. This is the host-nodal equivalent of analyticInitialPhi's
  // exposed-maximum formulation for the default corner-anchored dam.
  const domainMinimum = { x: -0.5 * c.width_m, y: 0, z: -0.5 * c.depth_m };
  const domainMaximum = { x: 0.5 * c.width_m, y: c.height_m, z: 0.5 * c.depth_m };
  const axes = ["x", "y", "z"] as const;
  const constraints: number[] = [];
  for (const axis of axes) {
    const scale = Math.max(1, Math.abs(domainMaximum[axis] - domainMinimum[axis]));
    const tolerance = 1e-9 * scale;
    const lowerIsClosedWall = minimum[axis] <= domainMinimum[axis] + tolerance;
    const upperIsClosedWall = maximum[axis] >= domainMaximum[axis] - tolerance
      && (axis !== "y" || c.top === "closed");
    if (!lowerIsClosedWall) constraints.push(minimum[axis] - point[axis]);
    if (!upperIsClosedWall) constraints.push(point[axis] - maximum[axis]);
  }
  if (constraints.length === 0) return -Math.max(c.width_m, c.height_m, c.depth_m);
  return Math.hypot(...constraints.map((value) => Math.max(value, 0)))
    + Math.min(Math.max(...constraints), 0);
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

/** Signed distance to seeded bricks at an integer finest-lattice node.
 *
 * Adaptive LoSasso stores phi on these nodes.  Evaluating the authored box
 * there directly avoids the cell-centre -> node -> renderer double
 * interpolation that rounds an exact brick edge or corner before the first
 * advance.  Integer coordinates also retain the reflection-bit identity used
 * by the symmetric-expansion oracle. */
export function initialFluidBrickSignedDistanceAtNode(
  scene: SceneDescription,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
  brickSize = INITIAL_FLUID_BRICK_SIZE,
): number | undefined {
  const components = initialFluidBrickComponentBounds(scene, dimensions, brickSize);
  if (!components) return undefined;
  const c = scene.container;
  const point = { x: -0.5 * c.width_m + x * c.width_m / dimensions[0],
    y: y * c.height_m / dimensions[1],
    z: -0.5 * c.depth_m + z * c.depth_m / dimensions[2] };
  let result = Number.POSITIVE_INFINITY;
  for (const component of components) {
    result = Math.min(result, boxSignedDistance(point,
      component.minimum, component.maximum));
  }
  return result;
}

export function inflowStrength(time_s: number, start_s: number, end_s: number, ramp_s: number): number {
  if (time_s < start_s || time_s >= end_s) return 0;
  if (ramp_s <= 0) return 1;
  return Math.min(1, (time_s - start_s) / ramp_s, (end_s - time_s) / ramp_s);
}
