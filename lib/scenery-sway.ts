import type { Quaternion, Vec3 } from "./model";
import type { SvoPrimitiveDescriptor } from "./svo-primitive-abi";

/**
 * Authored scenery motion — the gust that moves a tree, not a rigid body.
 *
 * A swaying prop never enters the solve and is not re-voxelized: it stays the
 * same immutable analytic primitive the sparse world already owns, re-posed in
 * the render ABI every presented frame. That is only sound while the re-posed
 * surface stays inside the cell ownership the proxy voxelizer wrote once at
 * bring-up. That pass gives a cell the nearest primitive's identity whenever
 * the cell centre lies within half a cell diagonal of its surface, so motion
 * held inside that margin can never carry a surface into a cell that names some
 * other primitive — and no brick, occupancy mip, or static publication has to
 * be rebuilt to follow it. `sceneryOwnershipMargin_m` is that budget, and
 * The garden's specimen tree is held to it by tests/scenery-sway.test.ts.
 *
 * What the sway therefore does not move is the baked cone-lighting pyramid: a
 * swaying canopy's soft shadow stays at its reference pose. Everything the
 * frame evaluates against the live surface — the silhouette, the exact shadow
 * path, normals, and every distance-dependent light term — follows the motion.
 */

/** Peak surface excursion is held to this fraction of the ownership margin. */
export const SCENERY_SWAY_MARGIN_FRACTION = 0.8;

/**
 * A cell centre within half a cell diagonal of a surface takes that surface's
 * identity, which is what the proxy voxelizer's coverage fraction resolves to.
 * The finest cell is the conservative choice: coarsened environment levels only
 * ever leave a larger margin.
 */
export function sceneryOwnershipMargin_m(finestCellSize_m: number): number {
  if (!(finestCellSize_m > 0) || !Number.isFinite(finestCellSize_m)) {
    throw new RangeError("Scenery ownership margin needs a positive finite cell size");
  }
  return 0.5 * Math.sqrt(3) * finestCellSize_m;
}

/** The excursion an authored sway may spend, in metres, at the given lattice. */
export function sceneryMaximumSwayExcursion_m(finestCellSize_m: number): number {
  return SCENERY_SWAY_MARGIN_FRACTION * sceneryOwnershipMargin_m(finestCellSize_m);
}

/**
 * One gust shared by every swaying prop in the scene, so a garden moves as one
 * body of air rather than as a set of independently wobbling objects. Two
 * incommensurate periods keep it off a metronome without needing noise.
 */
export const SCENERY_WIND = Object.freeze({
  /** Compass direction the gust pushes toward, in the XZ plane. */
  directionXZ: Object.freeze([0.82, 0.57] as const),
  primaryPeriod_s: 4.7,
  secondaryPeriod_s: 1.9,
  /** Share of the peak carried by the faster component. The two sum to one. */
  secondaryWeight: 0.28,
});

/**
 * Authored motion for one primitive. Both amplitudes are peaks: the wave below
 * spans [-1, 1], so a prop passes through its authored rest pose twice a cycle
 * and the catalog's static geometry stays the honest centre of the motion.
 */
export interface EnvironmentProxySway {
  /** World point the bend swings this primitive about — a trunk base, a stem root. */
  readonly pivot_m: Vec3;
  /**
   * Peak bend about the horizontal axis across the wind, in radians. This is
   * what actually moves the body through the world, so it is what the
   * ownership budget is spent on.
   */
  readonly bendAmplitude_rad: number;
  /**
   * Peak roll about the wind axis through the primitive's own centre. It turns
   * normals — which is the whole point, since that is what the lighting reads —
   * while moving the surface only by the body's own anisotropy.
   */
  readonly twistAmplitude_rad: number;
  /** Offset in the shared wave so neighbouring limbs lag one another. */
  readonly phase_rad: number;
}

const TWO_PI = 2 * Math.PI;

function normalizedWindDirection(): { direction: Vec3; bendAxis: Vec3 } {
  const [x, z] = SCENERY_WIND.directionXZ;
  const length = Math.hypot(x, z);
  const direction = { x: x / length, y: 0, z: z / length };
  // A positive rotation about (dz, 0, -dx) carries +Y toward the wind.
  return { direction, bendAxis: { x: direction.z, y: 0, z: -direction.x } };
}

const { direction: WIND_DIRECTION, bendAxis: WIND_BEND_AXIS } = normalizedWindDirection();

/** The shared gust, in [-1, 1]. */
export function sceneryWindWave(time_s: number, phase_rad: number): number {
  if (!Number.isFinite(time_s) || !Number.isFinite(phase_rad)) throw new RangeError("Scenery wind wave needs finite time and phase");
  const primary = Math.sin(TWO_PI * time_s / SCENERY_WIND.primaryPeriod_s + phase_rad);
  const secondary = Math.sin(TWO_PI * time_s / SCENERY_WIND.secondaryPeriod_s + 1.7 * phase_rad + 0.9);
  return (1 - SCENERY_WIND.secondaryWeight) * primary + SCENERY_WIND.secondaryWeight * secondary;
}

function axisAngleQuaternion(axis: Vec3, angle_rad: number): Quaternion {
  const half = 0.5 * angle_rad, sine = Math.sin(half);
  return { w: Math.cos(half), x: axis.x * sine, y: axis.y * sine, z: axis.z * sine };
}

function multiplyQuaternion(left: Quaternion, right: Quaternion): Quaternion {
  return {
    w: left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
    x: left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
    y: left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
    z: left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
  };
}

function rotateVector(rotation: Quaternion, value: Vec3): Vec3 {
  const { w, x, y, z } = rotation;
  const tx = 2 * (y * value.z - z * value.y);
  const ty = 2 * (z * value.x - x * value.z);
  const tz = 2 * (x * value.y - y * value.x);
  return {
    x: value.x + w * tx + (y * tz - z * ty),
    y: value.y + w * ty + (z * tx - x * tz),
    z: value.z + w * tz + (x * ty - y * tx),
  };
}

/** Half-extents of the descriptor along its own local axes, ignoring rotation. */
function descriptorExtents(descriptor: SvoPrimitiveDescriptor): readonly number[] {
  if (descriptor.kind === "sphere") return [descriptor.radius_m, descriptor.radius_m, descriptor.radius_m];
  if (descriptor.kind === "box") return [descriptor.halfExtents_m.x, descriptor.halfExtents_m.y, descriptor.halfExtents_m.z];
  if (descriptor.kind === "ellipsoid") return [descriptor.radii_m.x, descriptor.radii_m.y, descriptor.radii_m.z];
  if (descriptor.kind === "capsule") return [descriptor.radius_m, descriptor.segmentHalfLength_m + descriptor.radius_m, descriptor.radius_m];
  if (descriptor.kind === "cylinder") return [descriptor.radius_m, descriptor.halfHeight_m, descriptor.radius_m];
  if (descriptor.kind === "torus") {
    const outer = descriptor.majorRadius_m + descriptor.minorRadius_m;
    return [outer, descriptor.minorRadius_m, outer];
  }
  if (descriptor.kind === "cone") {
    const widest = Math.max(descriptor.baseRadius_m, descriptor.topRadius_m);
    return [widest, descriptor.halfHeight_m, widest];
  }
  return [0, 0, 0];
}

/**
 * Conservative peak surface excursion, in metres.
 *
 * The bend moves the whole body, so it costs the full lever arm. The twist
 * costs only the body's anisotropy: rotating a sphere moves no surface at all,
 * and for any convex body the support function moves by at most
 * `(widest - narrowest) * angle`, which is what a rotated ellipsoid's own
 * `(a² - b²) / sqrt(2(a² + b²))` bound stays under.
 */
export function scenerySwayExcursion_m(descriptor: SvoPrimitiveDescriptor, sway: EnvironmentProxySway): number {
  if (descriptor.kind === "terrain-heightfield") return 0;
  const lever = Math.hypot(
    descriptor.center_m.x - sway.pivot_m.x,
    descriptor.center_m.y - sway.pivot_m.y,
    descriptor.center_m.z - sway.pivot_m.z,
  );
  const extents = descriptorExtents(descriptor);
  const anisotropy = Math.max(...extents) - Math.min(...extents);
  return Math.abs(sway.bendAmplitude_rad) * lever + Math.abs(sway.twistAmplitude_rad) * anisotropy;
}

/**
 * Re-pose one primitive for the given presentation time. Identity, material,
 * and every dimension are untouched: only the transform moves, which is what
 * keeps the published sparse world valid underneath it.
 */
export function swayedPrimitiveDescriptor(
  descriptor: SvoPrimitiveDescriptor,
  sway: EnvironmentProxySway,
  time_s: number,
): SvoPrimitiveDescriptor {
  if (descriptor.kind === "terrain-heightfield") return descriptor;
  const bendWave = sceneryWindWave(time_s, sway.phase_rad);
  // Offset so a limb's roll leads its swing rather than peaking with it, the
  // way a real branch twists as it reaches the end of its travel.
  const twistWave = sceneryWindWave(time_s, sway.phase_rad + 1.3);
  const bend = axisAngleQuaternion(WIND_BEND_AXIS, sway.bendAmplitude_rad * bendWave);
  const twist = axisAngleQuaternion(WIND_DIRECTION, sway.twistAmplitude_rad * twistWave);
  const offset = rotateVector(bend, {
    x: descriptor.center_m.x - sway.pivot_m.x,
    y: descriptor.center_m.y - sway.pivot_m.y,
    z: descriptor.center_m.z - sway.pivot_m.z,
  });
  const center_m = { x: sway.pivot_m.x + offset.x, y: sway.pivot_m.y + offset.y, z: sway.pivot_m.z + offset.z };
  // A sphere has no orientation lane in the record, so it can only be carried
  // by the bend. Nothing that needs to twist is authored as one.
  if (descriptor.kind === "sphere") return { ...descriptor, center_m };
  const rest = descriptor.orientation ?? { w: 1, x: 0, y: 0, z: 0 };
  return { ...descriptor, center_m, orientation: multiplyQuaternion(bend, multiplyQuaternion(twist, rest)) };
}
