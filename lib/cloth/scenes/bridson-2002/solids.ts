import type { Quaternion, RigidPoseTrack, Vec3 } from "../../../core/model";
import { bakeProceduralSolid, sampleSolidNodes, type ProceduralSolidBoundary } from "../../../core/procedural-solid";
import { sampleRigidPoseTrack } from "../../../core/prescribed-rigid-motion";
import { quaternionInverseRotate } from "../../../core/rigid-body";
import type { SolidWorldVoxelPatch } from "../../../core/solid-world";

/**
 * Bridson 2002 solids as voxel solids. Each shape is an implicit function in
 * its body frame that is only ever used to bake the shared 8^3-brick,
 * 9^3-node field (`bakeProceduralSolid`, or the cloth contact field for a
 * moving body). Nothing here is an analytic collider: contact and editing see
 * the sampled field alone.
 */
export type BridsonSolidShape =
  /** Box centred on the body origin (the ground slab). */
  | { readonly kind: "box"; readonly halfExtents_m: Vec3 }
  | { readonly kind: "sphere"; readonly radius_m: number }
  /** Square frustum, base on body y = 0, axis +y. */
  | { readonly kind: "square-frustum"; readonly base_m: number; readonly top_m: number; readonly height_m: number }
  /** Square post, base on body y = 0, its four top edges bevelled at 45 degrees. */
  | { readonly kind: "chamfered-post"; readonly width_m: number; readonly height_m: number; readonly chamfer_m: number };

export type BridsonSolidMotion =
  | { readonly kind: "static"; readonly position_m: Vec3; readonly orientation?: Quaternion }
  /** Prescribed one-way rigid motion in absolute scene seconds (`sampleRigidPoseTrack`). */
  | { readonly kind: "pose-track"; readonly track: RigidPoseTrack };

export interface BridsonSolid {
  readonly id: string;
  readonly shape: BridsonSolidShape;
  readonly motion: BridsonSolidMotion;
  /** Coulomb coefficient against cloth (one value for static and kinetic, paper section 7.3). */
  readonly friction: number;
}

const IDENTITY: Quaternion = { w: 1, x: 0, y: 0, z: 0 };
const SQRT2 = Math.SQRT2;

function unknownKind(kind: never, what: string): never {
  throw new RangeError(`Unknown Bridson 2002 ${what} kind ${String((kind as { kind?: unknown })?.kind ?? kind)}`);
}

/** Signed distance (m) in the body frame, negative inside. Exact for box and
 * sphere; for the faceted shapes the largest plane distance, which is exact
 * inside, 1-Lipschitz, and has the exact zero set. */
export function bridsonShapeDistance(shape: BridsonSolidShape, p: Vec3): number {
  switch (shape.kind) {
    case "box": {
      const q = [Math.abs(p.x) - shape.halfExtents_m.x, Math.abs(p.y) - shape.halfExtents_m.y, Math.abs(p.z) - shape.halfExtents_m.z];
      return Math.hypot(...q.map(v => Math.max(v, 0))) + Math.min(Math.max(...q), 0);
    }
    case "sphere": return Math.hypot(p.x, p.y, p.z) - shape.radius_m;
    case "square-frustum": {
      const k = (shape.base_m - shape.top_m) / (2 * shape.height_m), n = Math.hypot(1, k);
      const side = (u: number) => (Math.abs(u) + k * p.y - shape.base_m / 2) / n;
      return Math.max(-p.y, p.y - shape.height_m, side(p.x), side(p.z));
    }
    case "chamfered-post": {
      const h = shape.width_m / 2, top = shape.height_m, c = shape.chamfer_m;
      const bevel = (u: number) => (Math.abs(u) + p.y - (h + top - c)) / SQRT2;
      return Math.max(-p.y, p.y - top, Math.abs(p.x) - h, Math.abs(p.z) - h, bevel(p.x), bevel(p.z));
    }
    default: return unknownKind(shape, "solid shape");
  }
}

/** Radius of the smallest body-origin ball holding the shape (for bounds). */
export function bridsonShapeReach(shape: BridsonSolidShape): number {
  switch (shape.kind) {
    case "box": return Math.hypot(shape.halfExtents_m.x, shape.halfExtents_m.y, shape.halfExtents_m.z);
    case "sphere": return shape.radius_m;
    case "square-frustum": return Math.hypot(shape.base_m / SQRT2, shape.height_m);
    case "chamfered-post": return Math.hypot(shape.width_m / SQRT2, shape.height_m);
    default: return unknownKind(shape, "solid shape");
  }
}

/** Body-frame axis-aligned half extents about the body origin, and the y range. */
export function bridsonShapeBox(shape: BridsonSolidShape): { halfX: number; halfZ: number; yMin: number; yMax: number } {
  switch (shape.kind) {
    case "box": return { halfX: shape.halfExtents_m.x, halfZ: shape.halfExtents_m.z, yMin: -shape.halfExtents_m.y, yMax: shape.halfExtents_m.y };
    case "sphere": return { halfX: shape.radius_m, halfZ: shape.radius_m, yMin: -shape.radius_m, yMax: shape.radius_m };
    case "square-frustum": return { halfX: shape.base_m / 2, halfZ: shape.base_m / 2, yMin: 0, yMax: shape.height_m };
    case "chamfered-post": return { halfX: shape.width_m / 2, halfZ: shape.width_m / 2, yMin: 0, yMax: shape.height_m };
    default: return unknownKind(shape, "solid shape");
  }
}

/** The narrowest dimension the lattice must resolve, metres. */
export function bridsonShapeSmallestFeature(shape: BridsonSolidShape): number {
  switch (shape.kind) {
    case "box": return 2 * Math.min(shape.halfExtents_m.x, shape.halfExtents_m.y, shape.halfExtents_m.z);
    case "sphere": return shape.radius_m;
    case "square-frustum": return Math.min(shape.top_m, shape.height_m);
    case "chamfered-post": return Math.min(shape.chamfer_m, shape.width_m - 2 * shape.chamfer_m);
    default: return unknownKind(shape, "solid shape");
  }
}

export function bridsonSolidPose(solid: BridsonSolid, time_s: number): { position_m: Vec3; orientation: Quaternion } {
  const m = solid.motion;
  switch (m.kind) {
    case "static": return { position_m: m.position_m, orientation: m.orientation ?? IDENTITY };
    case "pose-track": return sampleRigidPoseTrack(m.track, time_s);
    default: return unknownKind(m, "solid motion");
  }
}

/** World signed distance of a solid at a scene time. */
export function bridsonSolidDistance(solid: BridsonSolid, p: Vec3, time_s = 0): number {
  const pose = bridsonSolidPose(solid, time_s);
  const local = quaternionInverseRotate(pose.orientation, { x: p.x - pose.position_m.x, y: p.y - pose.position_m.y, z: p.z - pose.position_m.z });
  return bridsonShapeDistance(solid.shape, local);
}

/** Refuse a lattice that cannot hold a solid's smallest feature. */
export function assertBridsonSolidResolved(solid: BridsonSolid, cell_m: number, minimumCells: number): void {
  const feature = bridsonShapeSmallestFeature(solid.shape);
  if (!(feature >= minimumCells * cell_m))
    throw new RangeError(`Bridson solid ${solid.id}: its ${feature} m feature is narrower than ${minimumCells} cells of ${cell_m} m`);
}

/**
 * Worst disagreement between the trilinear field the lattice stores and the
 * shape: over probes in every cell the surface can cross, the largest |phi|
 * where the interpolant has the wrong sign, in metres. Below half a cell the
 * stored surface is the shape to within the lattice's own rounding of edges.
 */
export function bridsonLatticeSurfaceError(shape: BridsonSolidShape, cell_m: number, probes = 3): number {
  const reach = bridsonShapeReach(shape) + 2 * cell_m;
  const lo = Math.floor(-reach / cell_m), count = Math.ceil(2 * reach / cell_m) + 1;
  let worst = 0;
  const values = new Float64Array(9 ** 3);
  // Walk the lattice in 8^3 bricks of 9^3 nodes, exactly as the baked field stores it.
  for (let bz = 0; bz < count; bz += 8) for (let by = 0; by < count; by += 8) for (let bx = 0; bx < count; bx += 8) {
    let near = false;
    for (let i = 0; i < values.length; i++) {
      const x = lo + bx + i % 9, y = lo + by + Math.floor(i / 9) % 9, z = lo + bz + Math.floor(i / 81);
      values[i] = bridsonShapeDistance(shape, { x: x * cell_m, y: y * cell_m, z: z * cell_m }) / cell_m;
      near ||= Math.abs(values[i]!) < Math.sqrt(3);
    }
    if (!near) continue;
    for (let c = 0; c < 512; c++) {
      const cx = c % 8, cy = Math.floor(c / 8) % 8, cz = Math.floor(c / 64);
      for (let k = 0; k < probes ** 3; k++) {
        const q: [number, number, number] = [cx + (k % probes + 0.5) / probes, cy + (Math.floor(k / probes) % probes + 0.5) / probes, cz + (Math.floor(k / probes ** 2) + 0.5) / probes];
        const exact = bridsonShapeDistance(shape, { x: (lo + bx + q[0]) * cell_m, y: (lo + by + q[1]) * cell_m, z: (lo + bz + q[2]) * cell_m });
        const interp = sampleSolidNodes(values, q)[3];
        if (exact * interp < 0) worst = Math.max(worst, Math.abs(exact));
      }
    }
  }
  return worst;
}

export interface BridsonStaticLattice {
  /** World position of lattice node (0, 0, 0). */
  readonly origin_m: Vec3;
  readonly cell_m: number;
  /** Cells per axis, each a multiple of eight. */
  readonly dimensions: readonly [number, number, number];
}

/** Brick-aligned lattice holding every static solid with a two-cell margin. */
export function bridsonStaticLattice(solids: readonly BridsonSolid[], cell_m: number): BridsonStaticLattice {
  const fixed = solids.filter(s => s.motion.kind === "static");
  if (!fixed.length) throw new RangeError("Bridson scene has no static solid to bake");
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const s of fixed) {
    const { position_m: c, orientation: q } = bridsonSolidPose(s, 0), r = bridsonShapeReach(s.shape);
    const rotated = Math.abs(q.w) < 1 - 1e-12, box = bridsonShapeBox(s.shape);
    const min = rotated ? [-r, -r, -r] : [-box.halfX, box.yMin, -box.halfZ], max = rotated ? [r, r, r] : [box.halfX, box.yMax, box.halfZ];
    [c.x, c.y, c.z].forEach((v, a) => { lo[a] = Math.min(lo[a]!, v + min[a]!); hi[a] = Math.max(hi[a]!, v + max[a]!); });
  }
  const brick = 8 * cell_m;
  const start = lo.map(v => Math.floor((v - 2 * cell_m) / brick + 1e-9) * brick);
  const dimensions = hi.map((v, a) => 8 * Math.ceil((v + 2 * cell_m - start[a]!) / brick - 1e-9)) as unknown as [number, number, number];
  return { origin_m: { x: start[0]!, y: start[1]!, z: start[2]! }, cell_m, dimensions };
}

/** Bake the static solids (ground, legs, frustum) into the shared voxel solid. */
export function bakeBridsonStaticSolids(solids: readonly BridsonSolid[], lattice: BridsonStaticLattice):
  { solidVoxels: SolidWorldVoxelPatch[]; solidBoundary: ProceduralSolidBoundary } {
  const fixed = solids.filter(s => s.motion.kind === "static");
  const { origin_m: o, cell_m: h } = lattice;
  return bakeProceduralSolid(lattice.dimensions, [h, h, h], q => {
    const p = { x: o.x + q[0] * h, y: o.y + q[1] * h, z: o.z + q[2] * h };
    return Math.min(...fixed.map(s => bridsonSolidDistance(s, p))) / h;
  });
}

/** Rotation about +y by angle (rad). */
export function yawQuaternion(angle_rad: number): Quaternion {
  return { w: Math.cos(angle_rad / 2), x: 0, y: Math.sin(angle_rad / 2), z: 0 };
}
