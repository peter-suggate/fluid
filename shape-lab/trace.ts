/**
 * A CPU picture of a set of SVO records — either the exact surface, or the
 * voxels the octree would put there at a given leaf.
 *
 * ## The two modes, and why both are needed
 *
 * **Analytic** is the shape as authored: the render ABI's own hit oracle,
 * `intersectSvoPrimitive`, with its own hard-feature normals. It is what the
 * geometry *is*, with no resampling in front of it.
 *
 * **Voxel** is the shape as the picture will actually carry it. A voxel is solid
 * when the record's signed distance at its **centre** is negative — the
 * voxelizer's own occupancy test — and the surface is shaded from the *face*
 * the ray crossed to enter it. That second half is not decoration: the chunky
 * terracing on every hero frame is six-axis face normals, not a coarse mesh, so
 * a lab that shaded voxels smoothly would draw a picture the renderer never
 * produces and hide the exact artefact a shape has to survive.
 *
 * Both are one function of the leaf, and the leaf is the only difference between
 * them: analytic is the limit as the leaf goes to zero.
 *
 * ## Why the marched kinds do not go through the ABI's intersector
 *
 * `intersectMarchedLocal` recomputes the **normal at every march step** — six
 * more tape evaluations for a field program, four more neighbourhood sums for a
 * cluster — and `evaluateSvoFieldProgram` re-proves the tape legal on each of
 * them. Measured on the canopy's own tape that is 9.2 us a sample where the
 * evaluation alone is 6.5, and it is what made a 640 x 400 CPU trace of seven
 * pads take three and three quarter minutes.
 *
 * So marched kinds are traced here with a distance-only step and one normal at
 * the hit. The distance is `fieldProgramDistance_m`'s exact composition —
 * evaluate, divide by the Lipschitz constant — and the normal is the same
 * central difference the ABI takes, so the surface and its shading are the ones
 * the ABI would have reported. Nothing about the *geometry* is restated here.
 */
import type { Quaternion, Vec3 } from "../lib/core/model";
import type { TerrainGrid } from "../lib/core/terrain";
import {
  canonicalSvoPrimitive,
  intersectSvoPrimitive,
  sampleSvoPrimitive,
  svoPrimitiveBoundingRadius_m,
  SVO_PRIMITIVE_MARCH_ITERATIONS,
  type SvoFinitePrimitiveDescriptor,
  type SvoPrimitiveDescriptor,
} from "../lib/svo/svo-primitive-abi";
import { evaluateValidatedSvoFieldProgram, validateSvoFieldProgram } from "../lib/svo/svo-field-program";

// ---------------------------------------------------------------------------
// Small vector helpers. Local because the ABI keeps its own private.
// ---------------------------------------------------------------------------

const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const scale = (a: Vec3, k: number): Vec3 => ({ x: a.x * k, y: a.y * k, z: a.z * k });
const dot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
const length = (a: Vec3): number => Math.hypot(a.x, a.y, a.z);
const normalize = (a: Vec3): Vec3 => scale(a, 1 / Math.max(1e-12, length(a)));

/** q * (0, v) * q^-1, expanded — the same rotation the primitive ABI applies. */
function rotate(q: Quaternion, v: Vec3): Vec3 {
  const tx = 2 * (q.y * v.z - q.z * v.y);
  const ty = 2 * (q.z * v.x - q.x * v.z);
  const tz = 2 * (q.x * v.y - q.y * v.x);
  return {
    x: v.x + q.w * tx + (q.y * tz - q.z * ty),
    y: v.y + q.w * ty + (q.z * tx - q.x * tz),
    z: v.z + q.w * tz + (q.x * ty - q.y * tx),
  };
}

const inverseRotate = (q: Quaternion, v: Vec3): Vec3 => rotate({ w: q.w, x: -q.x, y: -q.y, z: -q.z }, v);

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

/**
 * An orbit, which is the only camera a shape lab needs.
 *
 * `reach_m` is the half-height of the view *at the target*, so framing is stated
 * in the same metres the shapes are, and a zoom does not change what "half a
 * pixel" means anywhere except through this one number.
 */
export interface ShapeLabCamera {
  readonly target_m: Vec3;
  readonly azimuth_rad: number;
  readonly elevation_rad: number;
  readonly distance_m: number;
  readonly reach_m: number;
}

export function shapeLabEye_m(camera: ShapeLabCamera): Vec3 {
  const horizontal = Math.cos(camera.elevation_rad) * camera.distance_m;
  return {
    x: camera.target_m.x + Math.sin(camera.azimuth_rad) * horizontal,
    y: camera.target_m.y + Math.sin(camera.elevation_rad) * camera.distance_m,
    z: camera.target_m.z + Math.cos(camera.azimuth_rad) * horizontal,
  };
}

interface CameraFrame {
  readonly eye: Vec3;
  readonly forward: Vec3;
  readonly right: Vec3;
  readonly up: Vec3;
  readonly distance: number;
  readonly reach: number;
}

function cameraFrame(camera: ShapeLabCamera): CameraFrame {
  const eye = shapeLabEye_m(camera);
  const forward = normalize(sub(camera.target_m, eye));
  const worldUp: Vec3 = Math.abs(forward.y) > 0.999 ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 };
  const right = normalize(cross(forward, worldUp));
  return {
    eye, forward, right,
    up: cross(right, forward),
    distance: Math.max(1e-6, length(sub(camera.target_m, eye))),
    reach: camera.reach_m,
  };
}

function rayDirection(frame: CameraFrame, u: number, v: number, aspect: number): Vec3 {
  return normalize(add(frame.forward, add(
    scale(frame.right, u * aspect * frame.reach / frame.distance),
    scale(frame.up, v * frame.reach / frame.distance),
  )));
}

// ---------------------------------------------------------------------------
// Prepared records
// ---------------------------------------------------------------------------

/**
 * Kinds whose exact hit is closed form and whose normal is authored rather than
 * differenced. These keep the ABI's own intersector: it is cheap, and its
 * hard-feature policy is the reason a box in this picture has edges.
 */
const EXACT_KINDS = new Set([
  "sphere", "box", "capsule", "cylinder", "ellipsoid",
  "torus", "cone", "round-cone", "rounded-cylinder",
]);

export interface PreparedRecord {
  readonly descriptor: SvoFinitePrimitiveDescriptor;
  readonly centre: Vec3;
  readonly radius: number;
  readonly colorLinear: readonly [number, number, number];
  readonly exact: boolean;
  /** Signed distance in world metres; negative inside. */
  readonly distance: (point: Vec3) => number;
  /** Central-difference step for the marched kinds' normal. */
  readonly normalStep: number;
}

function prepare(
  descriptor: SvoPrimitiveDescriptor,
  colorLinear: readonly [number, number, number],
): PreparedRecord | undefined {
  if (descriptor.kind === "terrain-heightfield") return undefined;
  const canonical = canonicalSvoPrimitive(descriptor);
  if (canonical.kind === "terrain-heightfield") return undefined;
  const centre = canonical.center_m;
  const radius = svoPrimitiveBoundingRadius_m(canonical);
  const exact = EXACT_KINDS.has(canonical.kind);
  const normalStep = Math.max(1e-6, 1e-3 * radius);
  if (canonical.kind === "field-program" && canonical.program) {
    const program = canonical.program;
    // Proved once, here, so the march does not re-prove it at every step.
    validateSvoFieldProgram(program);
    const orientation = canonical.orientation ?? { w: 1, x: 0, y: 0, z: 0 };
    const distance = (point: Vec3): number => {
      const local = inverseRotate(orientation, {
        x: point.x - centre.x, y: point.y - centre.y, z: point.z - centre.z,
      });
      const value = evaluateValidatedSvoFieldProgram(program, local);
      // `fieldProgramDistance_m`'s contract, restated because it is load-bearing:
      // a warped field is L-Lipschitz, so the raw value overestimates clearance
      // by up to L and a trace that stepped by it would walk through the surface.
      return value.distance_m / Math.max(1, value.lipschitz);
    };
    return { descriptor: canonical, centre, radius, colorLinear, exact, distance, normalStep };
  }
  const distance = (point: Vec3): number => sampleSvoPrimitive(canonical, point).signedDistance_m;
  return { descriptor: canonical, centre, radius, colorLinear, exact, distance, normalStep };
}

function marchedNormal(record: PreparedRecord, point: Vec3): Vec3 {
  const h = record.normalStep;
  return normalize({
    x: record.distance({ ...point, x: point.x + h }) - record.distance({ ...point, x: point.x - h }),
    y: record.distance({ ...point, y: point.y + h }) - record.distance({ ...point, y: point.y - h }),
    z: record.distance({ ...point, z: point.z + h }) - record.distance({ ...point, z: point.z - h }),
  });
}

/** Entry and exit parameters of the record's bounding sphere, or null on a miss. */
function boundsAlongRay(record: PreparedRecord, origin: Vec3, direction: Vec3): [number, number] | null {
  const offset = sub(origin, record.centre);
  const b = dot(offset, direction);
  const c = dot(offset, offset) - record.radius * record.radius;
  const discriminant = b * b - c;
  if (discriminant <= 0) return null;
  const root = Math.sqrt(discriminant);
  const exit = -b + root;
  if (exit <= 0) return null;
  return [Math.max(0, -b - root), exit];
}

// ---------------------------------------------------------------------------
// The ground
// ---------------------------------------------------------------------------

/**
 * A bilinear fetch of the baked heightfield, inlined.
 *
 * `terrainHeightAt` would be the shared route and it is 230 ns a call because it
 * re-resolves which representation the description holds; a heightfield march
 * asks a hundred times a ray. The arithmetic below is `sampleTerrainGrid`'s,
 * with the description already resolved to a grid by the caller.
 */
function groundHeight(grid: TerrainGrid, x: number, z: number): number {
  const fx = (x - grid.origin_m.x) / grid.spacing_m;
  const fz = (z - grid.origin_m.z) / grid.spacing_m;
  const i = Math.min(grid.size.nx - 2, Math.max(0, Math.floor(fx)));
  const j = Math.min(grid.size.nz - 2, Math.max(0, Math.floor(fz)));
  const tx = Math.min(1, Math.max(0, fx - i));
  const tz = Math.min(1, Math.max(0, fz - j));
  const h = grid.heights_m;
  const nx = grid.size.nx;
  const h00 = h[i + nx * j], h10 = h[i + 1 + nx * j];
  const h01 = h[i + nx * (j + 1)], h11 = h[i + 1 + nx * (j + 1)];
  return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
}

function groundNormal(grid: TerrainGrid, x: number, z: number): Vec3 {
  const h = grid.spacing_m;
  return normalize({
    x: groundHeight(grid, x - h, z) - groundHeight(grid, x + h, z),
    y: 2 * h,
    z: groundHeight(grid, x, z - h) - groundHeight(grid, x, z + h),
  });
}

interface GroundHit {
  readonly t: number;
  readonly normal: Vec3;
}

/** Analytic heightfield march: fixed-fraction steps on `y - h`, then a bisection. */
function traceGround(grid: TerrainGrid, origin: Vec3, direction: Vec3, tMax: number, ceiling: number): GroundHit | null {
  let t = 0;
  // Nothing below the highest column can be hit before the ray gets under it.
  if (direction.y < -1e-6 && origin.y > ceiling) t = (ceiling - origin.y) / direction.y;
  if (t > tMax) return null;
  let above = origin.y + direction.y * t - groundHeight(grid, origin.x + direction.x * t, origin.z + direction.z * t);
  if (above <= 0) return { t, normal: { x: 0, y: 1, z: 0 } };
  const step = Math.max(grid.spacing_m, 0.002);
  for (let guard = 0; guard < 4096 && t < tMax; guard += 1) {
    const advance = Math.max(step, 0.5 * above);
    const next = Math.min(tMax, t + advance);
    const point = add(origin, scale(direction, next));
    const nextAbove = point.y - groundHeight(grid, point.x, point.z);
    if (nextAbove <= 0) {
      let lo = t, hi = next;
      for (let iteration = 0; iteration < 24; iteration += 1) {
        const mid = 0.5 * (lo + hi);
        const at = add(origin, scale(direction, mid));
        if (at.y - groundHeight(grid, at.x, at.z) > 0) lo = mid; else hi = mid;
      }
      const hit = add(origin, scale(direction, hi));
      return { t: hi, normal: groundNormal(grid, hit.x, hit.z) };
    }
    if (next >= tMax) return null;
    t = next;
    above = nextAbove;
  }
  return null;
}

/**
 * The same ground, as the voxels the octree would put there.
 *
 * A heightfield does not need a 3-D walk, and the first version of this tried
 * one: a full DDA at a 0.78 mm leaf ran out of steps on every shallow ray and
 * stippled the plaster with the misses. What voxelizing a heightfield *is* is
 * two quantizations — the column, and the top of the topmost solid voxel in it —
 * and both are closed form:
 *
 *   - the column is the leaf cell containing (x, z), sampled at its centre,
 *     which is exactly the one height the live voxeliser takes per column;
 *   - voxel `j` is solid when `(j + 0.5) * leaf < h`, so the topmost solid one
 *     ends at `ceil(h / leaf - 0.5) * leaf`.
 *
 * So the voxelized ground is `y < stepHeight(x, z)` — a piecewise-constant
 * heightfield — and it can be marched exactly like the smooth one. The steps are
 * taken against the *smooth* height, which is never below the quantized one, and
 * floored so the march still resolves a single terrace; the crossing is then
 * bisected and the face read off whichever column boundary it fell on.
 */
function traceGroundVoxels(grid: TerrainGrid, origin: Vec3, direction: Vec3, tMax: number, ceiling: number, leaf: number): GroundHit | null {
  const column = (x: number, z: number): number => Math.ceil(
    groundHeight(grid, (Math.floor(x / leaf) + 0.5) * leaf, (Math.floor(z / leaf) + 0.5) * leaf) / leaf - 0.5,
  ) * leaf;
  const above = (t: number): number => {
    const point = add(origin, scale(direction, t));
    return point.y - column(point.x, point.z);
  };
  let t = 0;
  if (direction.y < -1e-6 && origin.y > ceiling + leaf) t = (ceiling + leaf - origin.y) / direction.y;
  if (t > tMax) return null;
  if (above(t) <= 0) return { t, normal: { x: 0, y: 1, z: 0 } };
  // Fine enough to resolve one terrace, and never finer than the grid the
  // terrace was quantized from — below that there is nothing new to find.
  const floorStep = Math.max(0.35 * leaf, 0.4 * grid.spacing_m);
  for (let guard = 0; guard < 6000 && t < tMax; guard += 1) {
    const point = add(origin, scale(direction, t));
    const smooth = point.y - groundHeight(grid, point.x, point.z);
    const next = Math.min(tMax, t + Math.max(floorStep, 0.5 * smooth));
    if (above(next) <= 0) {
      let lo = t, hi = next;
      for (let iteration = 0; iteration < 28; iteration += 1) {
        const mid = 0.5 * (lo + hi);
        if (above(mid) > 0) lo = mid; else hi = mid;
      }
      // Same column on both sides of the crossing means the ray came down
      // through a terrace's top; a different column means it walked into the
      // side of one, and the axis that changed is the face it hit.
      const before = add(origin, scale(direction, lo));
      const after = add(origin, scale(direction, hi));
      const dx = Math.floor(before.x / leaf) !== Math.floor(after.x / leaf);
      const dz = Math.floor(before.z / leaf) !== Math.floor(after.z / leaf);
      const normal: Vec3 = dx && (!dz || Math.abs(direction.x) > Math.abs(direction.z))
        ? { x: direction.x > 0 ? -1 : 1, y: 0, z: 0 }
        : dz
          ? { x: 0, y: 0, z: direction.z > 0 ? -1 : 1 }
          : { x: 0, y: 1, z: 0 };
      return { t: hi, normal };
    }
    if (next >= tMax) return null;
    t = next;
  }
  // Out of steps rather than out of ground: report the smooth surface instead of
  // a hole, which is what the first version's stipple actually was.
  return traceGround(grid, origin, direction, tMax, ceiling);
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

interface SurfaceHit {
  t: number;
  normal: Vec3;
  colorLinear: readonly [number, number, number];
}

/** Sphere-trace a marched kind on distance alone; the normal is taken once, at the hit. */
function traceMarched(record: PreparedRecord, origin: Vec3, direction: Vec3, from: number, to: number, epsilon: (t: number) => number): number | null {
  let t = from;
  for (let iteration = 0; iteration <= 3 * SVO_PRIMITIVE_MARCH_ITERATIONS; iteration += 1) {
    const d = Math.abs(record.distance(add(origin, scale(direction, t))));
    if (d <= epsilon(t)) return t;
    t += Math.max(d, 1e-7);
    if (t > to) return null;
  }
  return null;
}

/**
 * The first voxel of this record the ray enters, and the face it entered by.
 *
 * ## One evaluation a step, and the skip it buys
 *
 * The only question at each step is *is the voxel under the ray solid*, which is
 * one distance at that voxel's **centre**. That same number also bounds how far
 * the ray may jump, and the bound is worth stating because getting it wrong
 * punches holes in the surface.
 *
 * Let the current voxel's centre be `c` with `distance(c) = dc > 0`. Every solid
 * voxel's centre `c'` satisfies `distance(c') < 0`, so `|c' - c| > dc`. A ray
 * point `q` can only be inside a solid voxel if that voxel's centre is within
 * the half-diagonal `h` of it. Chaining the two, `q` is safe while
 * `|q - c| < dc - h`, and since `|q - c| <= |q - p| + h` for a start point `p` in
 * this voxel, the ray may advance `dc - 2h` before anything can change.
 *
 * Only the last two voxels before a surface fail that test and get stepped one
 * at a time, which is what keeps a 280-voxel-deep canopy pad affordable at a
 * 0.78 mm leaf — and it matters most exactly where the field is least helpful:
 * a `scatter` saturates its distance at the fold's clearance (1.26 mm on the
 * canopy's tape), so inside a pad there is no long jump to be had at any leaf.
 */
function traceVoxels(record: PreparedRecord, origin: Vec3, direction: Vec3, from: number, to: number, leaf: number): { t: number; normal: Vec3 } | null {
  const halfDiagonal = 0.5 * Math.sqrt(3) * leaf;
  let t = from;
  let axis = -1;
  let sign = 1;
  for (let guard = 0; guard < 8192 && t <= to; guard += 1) {
    const point = add(origin, scale(direction, t + 1e-9));
    const ix = Math.floor(point.x / leaf), iy = Math.floor(point.y / leaf), iz = Math.floor(point.z / leaf);
    const centre = { x: (ix + 0.5) * leaf, y: (iy + 0.5) * leaf, z: (iz + 0.5) * leaf };
    const dc = record.distance(centre);
    if (dc < 0) {
      const normal: Vec3 = axis < 0
        ? scale(direction, -1)
        : { x: axis === 0 ? -sign : 0, y: axis === 1 ? -sign : 0, z: axis === 2 ? -sign : 0 };
      return { t, normal };
    }
    // Next lattice plane, and the face the ray will cross to reach it.
    let step = Infinity;
    for (let a = 0; a < 3; a += 1) {
      const dir = a === 0 ? direction.x : a === 1 ? direction.y : direction.z;
      if (Math.abs(dir) < 1e-12) continue;
      const index = a === 0 ? ix : a === 1 ? iy : iz;
      const plane = (index + (dir > 0 ? 1 : 0)) * leaf;
      const p = a === 0 ? point.x : a === 1 ? point.y : point.z;
      const candidate = (plane - p) / dir;
      if (candidate < step) { step = candidate; axis = a; sign = dir > 0 ? 1 : -1; }
    }
    if (!Number.isFinite(step)) return null;
    const jump = dc - 2 * halfDiagonal;
    if (jump > step) {
      // A jump lands mid-voxel, so the face it entered by is unknown until the
      // next boundary; the loop re-derives one before any hit can be reported.
      t += jump;
      axis = -1;
      continue;
    }
    t += Math.max(1e-9, step) + 1e-9;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Shading
// ---------------------------------------------------------------------------

export type ShapeLabShading = "clay" | "material" | "normal";

/**
 * Flat porcelain: one warm key, one cool sky fill, no shadow.
 *
 * Deliberately not the renderer's grade. The lab's job is to make *form* legible
 * — the hero frame's own grade is tuned to a plate and was, at one point,
 * over-exposed enough to flatten the thing being judged.
 */
const KEY: Vec3 = normalize({ x: 0.48, y: 0.78, z: 0.40 });

function shade(shading: ShapeLabShading, normal: Vec3, colorLinear: readonly [number, number, number]): [number, number, number] {
  if (shading === "normal") {
    return [
      Math.round(255 * (0.5 + 0.5 * normal.x)),
      Math.round(255 * (0.5 + 0.5 * normal.y)),
      Math.round(255 * (0.5 + 0.5 * normal.z)),
    ];
  }
  const lambert = Math.max(0, dot(normal, KEY));
  const sky = 0.5 + 0.5 * normal.y;
  const level = 0.30 * sky + 0.78 * lambert;
  const tint = shading === "material"
    ? [
      Math.max(0.02, colorLinear[0]),
      Math.max(0.02, colorLinear[1]),
      Math.max(0.02, colorLinear[2]),
    ]
    : [1, 0.985, 0.955];
  // Material colours are linear reflectances and clay is near white, so both go
  // through the same one-line grade: no exposure, sRGB out.
  const channel = (value: number): number => Math.round(255 * Math.min(1, Math.max(0, level * value)) ** (1 / 2.2));
  return [channel(tint[0]), channel(tint[1]), channel(tint[2])];
}

const BACKGROUND: [number, number, number] = [206, 210, 214];

// ---------------------------------------------------------------------------
// The frame
// ---------------------------------------------------------------------------

/**
 * A specimen, canonicalised and proved once.
 *
 * Preparation is per record and not per pixel on purpose: canonicalising a
 * descriptor allocates, and a field program's tape has to be validated before it
 * can be evaluated without validation. A tile that re-prepared would pay both a
 * hundred thousand times.
 */
export interface ShapeLabScene {
  readonly records: readonly PreparedRecord[];
  readonly ground?: TerrainGrid;
  /** Highest column in the ground, so a ray above it can skip straight down to it. */
  readonly ceiling: number;
}

export function prepareShapeLabScene(
  descriptors: readonly SvoPrimitiveDescriptor[],
  colors: readonly (readonly [number, number, number])[],
  ground?: TerrainGrid,
): ShapeLabScene {
  const records: PreparedRecord[] = [];
  for (let index = 0; index < descriptors.length; index += 1) {
    const record = prepare(descriptors[index], colors[index] ?? [0.8, 0.8, 0.8]);
    if (record) records.push(record);
  }
  let ceiling = 0;
  if (ground) for (const height of ground.heights_m) if (height > ceiling) ceiling = height;
  return { records, ground, ceiling };
}

export interface ShapeLabTraceOptions {
  readonly width: number;
  readonly height: number;
  readonly camera: ShapeLabCamera;
  /** Omit for the analytic surface; supply the leaf to see the voxels. */
  readonly leaf_m?: number;
  readonly shading: ShapeLabShading;
  readonly showGround: boolean;
  /**
   * The pixel rectangle to render, in frame coordinates.
   *
   * The buffer that comes back is the **tile's** own size, not the frame's, so a
   * worker returning one moves a few kilobytes rather than a megabyte. The
   * compositor knows where to put it from `tile.x`, `tile.y`.
   */
  readonly tile?: { readonly x: number; readonly y: number; readonly width: number; readonly height: number };
}

export interface ShapeLabTraceOutput {
  readonly rgba: Uint8ClampedArray;
  readonly hits: number;
  readonly pixels: number;
}

/**
 * Render one tile.
 *
 * Records are culled to the tile's own cone before any pixel is traced, which is
 * what makes a 1 338-record stone set affordable: a pebble covers a hundred
 * pixels and is asked about only in the tiles it reaches.
 */
export function traceShapeLabTile(scene: ShapeLabScene, input: ShapeLabTraceOptions, into?: Uint8ClampedArray): ShapeLabTraceOutput {
  const { width, height } = input;
  const tile = input.tile ?? { x: 0, y: 0, width, height };
  const rgba = into ?? new Uint8ClampedArray(tile.width * tile.height * 4);
  const frame = cameraFrame(input.camera);
  const aspect = width / height;
  const prepared = scene.records;
  const ground = input.showGround ? scene.ground : undefined;

  // The tile's cone: axis through its centre, half-angle out to its corners.
  const corner = (px: number, py: number): Vec3 => rayDirection(
    frame,
    (px + 0.5) / width * 2 - 1,
    1 - (py + 0.5) / height * 2,
    aspect,
  );
  const corners = [
    corner(tile.x, tile.y),
    corner(tile.x + tile.width - 1, tile.y),
    corner(tile.x, tile.y + tile.height - 1),
    corner(tile.x + tile.width - 1, tile.y + tile.height - 1),
  ];
  const axis = normalize(corners.reduce(add, { x: 0, y: 0, z: 0 }));
  const halfAngle = Math.acos(Math.min(1, Math.max(-1, Math.min(...corners.map((c) => dot(c, axis))))));
  // Front to back, so the first record a ray hits closes `tMax` for every record
  // behind it. On the canopy that is the difference between marching one pad and
  // marching all seven, and the pads overlap enough that it is most of the frame.
  const visible = prepared
    .filter((record) => {
      const offset = sub(record.centre, frame.eye);
      const distance = length(offset);
      if (distance <= record.radius) return true;
      const angular = Math.asin(Math.min(1, record.radius / distance));
      return Math.acos(Math.min(1, Math.max(-1, dot(offset, axis) / distance))) - angular <= halfAngle + 1e-3;
    })
    .map((record) => ({ record, near: length(sub(record.centre, frame.eye)) - record.radius }))
    .sort((a, b) => a.near - b.near)
    .map((entry) => entry.record);

  const ceiling = scene.ceiling;
  // Far enough that a grazing ray still finds the ground it is about to cross,
  // and bounded so one that never will gives up.
  const far = 3 * frame.distance + 8 * frame.reach;
  // Half a pixel at the target, which is as tight as a picture can ever show.
  const pixelAtTarget = 2 * frame.reach / height;
  const epsilon = (t: number): number => Math.max(1e-6, 0.35 * pixelAtTarget * t / frame.distance);

  let hits = 0;
  let pixels = 0;
  for (let py = tile.y; py < Math.min(height, tile.y + tile.height); py += 1) {
    for (let px = tile.x; px < Math.min(width, tile.x + tile.width); px += 1) {
      pixels += 1;
      const direction = corner(px, py);
      let best: SurfaceHit | null = null;
      for (const record of visible) {
        const bounds = boundsAlongRay(record, frame.eye, direction);
        if (!bounds) continue;
        const from = bounds[0];
        const to = Math.min(bounds[1], best ? best.t : Infinity);
        if (from > to) continue;
        if (input.leaf_m === undefined) {
          if (record.exact) {
            const hit = intersectSvoPrimitive(record.descriptor, {
              origin_m: frame.eye, direction, tMin_m: from, tMax_m: to,
            });
            if (hit && (!best || hit.t_m < best.t)) best = { t: hit.t_m, normal: hit.normal, colorLinear: record.colorLinear };
            continue;
          }
          const t = traceMarched(record, frame.eye, direction, from, to, epsilon);
          if (t !== null && (!best || t < best.t)) {
            best = { t, normal: marchedNormal(record, add(frame.eye, scale(direction, t))), colorLinear: record.colorLinear };
          }
          continue;
        }
        const voxel = traceVoxels(record, frame.eye, direction, from, to, input.leaf_m);
        if (voxel && (!best || voxel.t < best.t)) {
          best = { t: voxel.t, normal: voxel.normal, colorLinear: record.colorLinear };
        }
      }
      if (ground) {
        const limit = Math.min(far, best ? best.t : Infinity);
        const hit = input.leaf_m === undefined
          ? traceGround(ground, frame.eye, direction, limit, ceiling)
          : traceGroundVoxels(ground, frame.eye, direction, limit, ceiling, input.leaf_m);
        if (hit && (!best || hit.t < best.t)) best = { t: hit.t, normal: hit.normal, colorLinear: [0.92, 0.9, 0.86] };
      }
      const offset = 4 * ((py - tile.y) * tile.width + (px - tile.x));
      const colour = best ? shade(input.shading, best.normal, best.colorLinear) : BACKGROUND;
      if (best) hits += 1;
      rgba[offset] = colour[0];
      rgba[offset + 1] = colour[1];
      rgba[offset + 2] = colour[2];
      rgba[offset + 3] = 255;
    }
  }
  return { rgba, hits, pixels };
}

/** World-space bounds of a record set, for framing a specimen without being told where it is. */
export function shapeLabBounds(
  descriptors: readonly SvoPrimitiveDescriptor[],
): { readonly centre: Vec3; readonly radius: number } | undefined {
  let min: Vec3 | undefined;
  let max: Vec3 | undefined;
  for (const descriptor of descriptors) {
    if (descriptor.kind === "terrain-heightfield") continue;
    const radius = svoPrimitiveBoundingRadius_m(descriptor);
    const centre = descriptor.center_m;
    const low = { x: centre.x - radius, y: centre.y - radius, z: centre.z - radius };
    const high = { x: centre.x + radius, y: centre.y + radius, z: centre.z + radius };
    min = min ? { x: Math.min(min.x, low.x), y: Math.min(min.y, low.y), z: Math.min(min.z, low.z) } : low;
    max = max ? { x: Math.max(max.x, high.x), y: Math.max(max.y, high.y), z: Math.max(max.z, high.z) } : high;
  }
  if (!min || !max) return undefined;
  return {
    centre: scale(add(min, max), 0.5),
    radius: Math.max(1e-3, 0.5 * length(sub(max, min))),
  };
}
