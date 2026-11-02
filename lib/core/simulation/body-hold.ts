import {
  advanceCarryPlane, CARRY_TILT_STEP_RAD, carryPlane, carryPlaneHit, carryPositionFor, carryVelocity,
  clampCarryPosition, clampCarryTilt, type CarryPlane,
} from "../editor-carry";
import type { CameraState, RigidBodyDescription, SceneDescription, Vec3 } from "../model";

/**
 * The one kinematic hold a pane may have on a body.
 *
 * Carrying, throwing and the gizmo all hold a body the same way — the solver is
 * told where it is and it ignores gravity until let go — and they used to keep
 * that in three places: a carry ref in the viewport, the store's carry, and the
 * controller's drag constraint, synchronised by an effect. A carry whose pose
 * lookup failed left the store saying "carrying" with nothing held, and picking
 * up a second body left the first one weightless until a reset.
 *
 * Now the controller holds at most one of these per pane. Beginning any hold
 * lets go of the previous one first, a move or an end for a hold that is no
 * longer the current one does nothing, and the store's `carry` is only what
 * the controller publishes about a carry for the chip to read.
 */
export type BodyHold =
  | { readonly kind: "throw" | "gizmo"; readonly bodyId: string; readonly position_m: Vec3; readonly velocity_m_s: Vec3 }
  | CarryHold;

export interface CarryHold {
  readonly kind: "carry";
  readonly bodyId: string;
  readonly label?: string;
  readonly position_m: Vec3;
  readonly velocity_m_s: Vec3;
  readonly grip: CarryGrip;
}

/**
 * Everything a carry needs between pointer events. `plane` is the vertical
 * sheet the body slides on, `anchor` the pointer/body pair the current gear
 * was engaged at, and `origin_m` where the body was when the carry began,
 * which is where Escape puts it back.
 */
export interface CarryGrip {
  readonly plane: CarryPlane;
  readonly anchor: { readonly pointer_m: Vec3; readonly body_m: Vec3 };
  readonly origin_m: Vec3;
  readonly lastPointer_m: Vec3;
  readonly lastTime_ms: number;
  readonly tilt_rad: number;
  readonly fine: boolean;
}

const REST: Vec3 = Object.freeze({ x: 0, y: 0, z: 0 });

/** Pick a body up where it stands. */
export function carryHold(bodyId: string, label: string | undefined, camera: CameraState, pose_m: Vec3, now_ms: number): CarryHold {
  return {
    kind: "carry", bodyId, label, position_m: pose_m, velocity_m_s: REST,
    grip: {
      plane: carryPlane(camera, pose_m),
      anchor: { pointer_m: pose_m, body_m: pose_m },
      origin_m: pose_m,
      lastPointer_m: pose_m,
      lastTime_ms: now_ms,
      tilt_rad: 0,
      fine: false,
    },
  };
}

/**
 * Where a carried body goes for a pointer ray, with the finite-difference
 * velocity the water has to see it move at. Undefined when the ray runs
 * parallel to the carry plane.
 */
export function carryAlong(
  hold: CarryHold,
  ray: { origin: Vec3; direction: Vec3 },
  timeStamp_ms: number,
  scene: SceneDescription,
  description: RigidBodyDescription,
): CarryHold | undefined {
  const hit = carryPlaneHit(hold.grip.plane, ray);
  if (!hit) return undefined;
  const position_m = clampCarryPosition(scene, description, carryPositionFor(hold.grip.anchor, hit, hold.grip.fine));
  const velocity_m_s = carryVelocity(hold.position_m, position_m, (timeStamp_ms - hold.grip.lastTime_ms) / 1000);
  return { ...hold, position_m, velocity_m_s, grip: { ...hold.grip, lastPointer_m: hit, lastTime_ms: timeStamp_ms } };
}

/** Change gear without moving anything: re-anchor where the pointer and body already are. */
export function carryGear(hold: CarryHold, fine: boolean): CarryHold {
  if (hold.grip.fine === fine) return hold;
  return { ...hold, grip: { ...hold.grip, fine, anchor: { pointer_m: hold.grip.lastPointer_m, body_m: hold.position_m } } };
}

/** Tip by whole Q/E steps; undefined when already at the limit. */
export function carryTilt(hold: CarryHold, steps: number): CarryHold | undefined {
  const tilt_rad = clampCarryTilt(hold.grip.tilt_rad + steps * CARRY_TILT_STEP_RAD);
  if (tilt_rad === hold.grip.tilt_rad) return undefined;
  return { ...hold, velocity_m_s: REST, grip: { ...hold.grip, tilt_rad } };
}

/** Walk the carry plane along the view — the wheel's one degree of freedom. */
export function carryWalk(hold: CarryHold, wheelDeltaY: number): CarryHold {
  return { ...hold, grip: { ...hold.grip, plane: advanceCarryPlane(hold.grip.plane, wheelDeltaY) } };
}

/** The tilt the chip shows, in whole degrees. */
export function carryTiltDegrees(hold: CarryHold): number {
  return Math.round((hold.grip.tilt_rad * 180) / Math.PI);
}
