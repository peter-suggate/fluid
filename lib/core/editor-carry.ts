import { add, cameraBasis, dot, length, scale, sub } from "./math";
import type { CameraState, Quaternion, RigidBodyDescription, SceneDescription, Vec3 } from "./model";
import { boundingRadius } from "./rigid-body";

/**
 * Carrying something.
 *
 * A drag is a gesture: the button is down, the object is on a plane facing the
 * camera, and letting go ends it. Carrying is a *mode* — the object is in your
 * hand until you put it down — and the difference matters for the thing this
 * exists for, which is dipping a cup into water and lifting it out again. That
 * takes both hands' worth of control: a long, slow descent, a pause at the
 * bottom, a tilt, a lift. A held button cannot do the pause and a camera-facing
 * plane cannot do the descent, because on a plane that tilts with the camera,
 * "down" is only down when the camera is level.
 *
 * So the carry plane is vertical, always: it contains world up and the camera's
 * right, and faces the camera along the ground. Pointer Y is world Y and
 * nothing else, which is what makes a dip a dip. The wheel walks the plane
 * along the view, which is the one degree of freedom a single pointer cannot
 * carry, and Q and E tilt about the camera's right axis, which is what pours.
 *
 * Everything here is pure. The mode itself lives in the UI store and the
 * pointer wiring in the viewport; this file is the arithmetic both agree on,
 * and is the reason the behaviour can be tested without a canvas.
 */

/** Held-shift multiplier: enough to be a different gear, not so little it feels stuck. */
export const CARRY_FINE_SCALE = 0.25;
/** One press of Q or E. Twenty-four steps from upright to inverted. */
export const CARRY_TILT_STEP_RAD = Math.PI / 24;
/**
 * How far a carried thing can be tilted, each way.
 *
 * Past horizontal on purpose: a cup only empties completely once its rim is
 * below its base, and stopping at 90° would leave a puddle in the bottom that
 * no amount of tilting could pour out.
 */
export const CARRY_TILT_LIMIT_RAD = (5 / 6) * Math.PI;
/** Metres the wheel walks the carry plane per notch of deltaY. */
export const CARRY_DEPTH_PER_WHEEL_PX = 0.0025;
/** How far above the tank rim a carried thing may be lifted. */
export const CARRY_HEADROOM_M = 0.8;

export interface CarryPlane {
  readonly origin_m: Vec3;
  /** Horizontal, facing the camera. Never has a Y component. */
  readonly normal: Vec3;
}

/**
 * The vertical plane a carried body slides on.
 *
 * The normal is the camera's forward flattened onto the ground, so the plane
 * stands up whatever the camera's pitch. Looking straight down degenerates it —
 * there is no horizontal forward — and the fallback is world -Z, which is a
 * plane the pointer can still address rather than a division by zero.
 */
export function carryPlane(camera: CameraState, through_m: Vec3): CarryPlane {
  const { forward } = cameraBasis(camera);
  const flattened = { x: forward.x, y: 0, z: forward.z };
  const magnitude = length(flattened);
  return {
    origin_m: { ...through_m },
    normal: magnitude > 1e-6 ? scale(flattened, 1 / magnitude) : { x: 0, y: 0, z: -1 },
  };
}

/** Where a pointer ray meets the carry plane, or undefined when it runs parallel. */
export function carryPlaneHit(
  plane: CarryPlane,
  ray: { origin: Vec3; direction: Vec3 },
): Vec3 | undefined {
  const denominator = dot(ray.direction, plane.normal);
  if (Math.abs(denominator) < 1e-6) return undefined;
  const t = dot(sub(plane.origin_m, ray.origin), plane.normal) / denominator;
  if (!Number.isFinite(t)) return undefined;
  return add(ray.origin, scale(ray.direction, t));
}

/** Walk the plane along its own normal — the wheel's one degree of freedom. */
export function advanceCarryPlane(plane: CarryPlane, wheelDeltaY: number): CarryPlane {
  return { ...plane, origin_m: add(plane.origin_m, scale(plane.normal, -wheelDeltaY * CARRY_DEPTH_PER_WHEEL_PX)) };
}

/**
 * Where the body goes for a pointer that has reached `hit_m`.
 *
 * `anchor` is the pair recorded when the gear last changed: the plane point the
 * pointer was at and the position the body was at. Fine motion is the same
 * arithmetic with a smaller multiplier, which is why holding shift does not
 * teleport the object — it only changes how far it travels from where it
 * already is.
 */
export function carryPositionFor(
  anchor: { readonly pointer_m: Vec3; readonly body_m: Vec3 },
  hit_m: Vec3,
  fine: boolean,
): Vec3 {
  return add(anchor.body_m, scale(sub(hit_m, anchor.pointer_m), fine ? CARRY_FINE_SCALE : 1));
}

/** Tilt clamped to what a carried thing can be turned to. */
export function clampCarryTilt(tilt_rad: number): number {
  return Math.max(-CARRY_TILT_LIMIT_RAD, Math.min(CARRY_TILT_LIMIT_RAD, tilt_rad));
}

/**
 * Upright unless tilted, about the axis that makes a tilt read as pouring.
 *
 * The camera's right, so a tilt always tips the near rim down and the far rim
 * up regardless of which way the object is being carried around the tank. An
 * object-local axis would mean the same keypress poured toward the viewer from
 * one side of the tank and away from them on the other.
 */
export function carryOrientation(camera: CameraState, tilt_rad: number): Quaternion {
  const axis = cameraBasis(camera).right;
  const magnitude = length(axis);
  if (!(magnitude > 1e-6) || tilt_rad === 0) return { w: 1, x: 0, y: 0, z: 0 };
  const unit = scale(axis, 1 / magnitude);
  const half = tilt_rad / 2;
  const sine = Math.sin(half);
  return { w: Math.cos(half), x: unit.x * sine, y: unit.y * sine, z: unit.z * sine };
}

/**
 * Keep a carried body somewhere the scene can hold it.
 *
 * The same box `addBodyAt` clamps a placed body into, for the same reason: the
 * solver only couples what is inside the lattice, and a body carried out
 * through a wall would stop interacting with the water without anything saying
 * so. The headroom above the rim is what makes lifting a full cup *out* of the
 * tank a thing you can do rather than a limit you hit.
 */
export function clampCarryPosition(
  scene: SceneDescription,
  description: RigidBodyDescription,
  position_m: Vec3,
): Vec3 {
  const radius = boundingRadius(description);
  const { width_m, height_m, depth_m } = scene.container;
  return {
    x: Math.min(width_m / 2 - radius, Math.max(-width_m / 2 + radius, position_m.x)),
    y: Math.min(height_m + CARRY_HEADROOM_M, Math.max(radius, position_m.y)),
    z: Math.min(depth_m / 2 - radius, Math.max(-depth_m / 2 + radius, position_m.z)),
  };
}

/**
 * Velocity to hand the solver for a carried body.
 *
 * A carried body is kinematic — the solver is told where it is, not pushed — but
 * the water still has to see it *move*, because the whole point of dipping a cup
 * is the water it displaces. This is the finite difference the coupling reads,
 * clamped so a wheel notch or a plane flip cannot report a body travelling at a
 * speed the pressure solve would have to answer for.
 */
export function carryVelocity(
  previous_m: Vec3,
  next_m: Vec3,
  elapsed_s: number,
  limit_m_s = 6,
): Vec3 {
  if (!(elapsed_s > 1e-4)) return { x: 0, y: 0, z: 0 };
  const velocity = scale(sub(next_m, previous_m), 1 / elapsed_s);
  const speed = length(velocity);
  return speed > limit_m_s ? scale(velocity, limit_m_s / speed) : velocity;
}
