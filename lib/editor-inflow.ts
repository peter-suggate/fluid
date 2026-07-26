import { add, length, normalize, scale, sub } from "./math";
import type { FluidInflow, SceneDescription, Vec3 } from "./model";

/**
 * Placing and aiming the hose.
 *
 * The schema carries exactly one `fluid.inflow`. Promoting it to an array is
 * not a schema edit in practice: `inflowBoundaryWGSL` resolves a single
 * dominant axis (`inflowAxis()`), and three solvers pack the nozzle into fixed
 * `inflowPositionRadius` / `inflowVelocityLength` params lanes. Until that is
 * reworked, the editor authors one nozzle well rather than pretending to
 * author several.
 */

export type InflowHandleKind = "center" | "nozzle";

export interface InflowHandle {
  readonly kind: InflowHandleKind;
  readonly position_m: Vec3;
}

export const INFLOW_SELECTION_ID = "inflow";
export const INFLOW_MINIMUM_SPEED_M_S = 0.05;
export const INFLOW_MAXIMUM_SPEED_M_S = 12;
export const INFLOW_MINIMUM_RADIUS_M = 0.01;

/** Metres of arrow per m/s, so the arrow length reads as the jet speed. */
export const INFLOW_ARROW_SECONDS = 0.12;

export function inflowSpeed_m_s(inflow: FluidInflow): number {
  return length(inflow.velocity_m_s);
}

export function inflowDirection(inflow: FluidInflow): Vec3 {
  const speed = inflowSpeed_m_s(inflow);
  return speed > 1e-9 ? scale(inflow.velocity_m_s, 1 / speed) : { x: 0, y: -1, z: 0 };
}

/** Centre plus the arrow tip whose offset encodes direction and speed. */
export function inflowHandles(inflow: FluidInflow): InflowHandle[] {
  const tip = add(inflow.center_m, scale(inflowDirection(inflow), INFLOW_ARROW_SECONDS * inflowSpeed_m_s(inflow)));
  return [
    { kind: "center", position_m: { ...inflow.center_m } },
    { kind: "nozzle", position_m: tip },
  ];
}

function clampToContainer(point: Vec3, container: SceneDescription["container"]): Vec3 {
  return {
    x: Math.min(container.width_m / 2, Math.max(-container.width_m / 2, point.x)),
    y: Math.min(container.height_m, Math.max(0, point.y)),
    z: Math.min(container.depth_m / 2, Math.max(-container.depth_m / 2, point.z)),
  };
}

/**
 * A nozzle on a picked surface, aimed along its normal so a hose placed on a
 * wall sprays away from it and one placed in open air sprays down.
 */
export function createInflowAt(
  point: Vec3,
  normal: Vec3,
  scene: SceneDescription,
): FluidInflow {
  const container = scene.container;
  const span = Math.min(container.width_m, container.depth_m);
  const direction = length(normal) > 1e-6 ? normalize(normal) : { x: 0, y: -1, z: 0 };
  const radius_m = Math.max(INFLOW_MINIMUM_RADIUS_M, 0.06 * span);
  const length_m = Math.max(2 * radius_m, 0.12 * span);
  // Lift the nozzle off the surface so its channel is not buried in solid.
  const center_m = clampToContainer(add(point, scale(direction, length_m)), container);
  return {
    center_m,
    radius_m,
    length_m,
    velocity_m_s: scale(direction, -1.5),
    start_s: 0,
    end_s: Math.max(1, scene.duration_s),
    ramp_s: 0.1,
  };
}

/** Move the nozzle body, keeping it inside the container as validateScene requires. */
export function moveInflow(inflow: FluidInflow, point: Vec3, container: SceneDescription["container"]): FluidInflow {
  return { ...inflow, center_m: clampToContainer(point, container) };
}

/**
 * Aim and throttle from the arrow tip: its direction sets the jet axis and its
 * distance from the nozzle sets the speed. A tip dragged onto the nozzle would
 * be a zero-length velocity, which `validateScene` rejects, so speed floors.
 */
export function aimInflow(inflow: FluidInflow, tip_m: Vec3): FluidInflow {
  const offset = sub(tip_m, inflow.center_m);
  const distance = length(offset);
  if (!(distance > 1e-6)) return inflow;
  const speed = Math.min(INFLOW_MAXIMUM_SPEED_M_S, Math.max(INFLOW_MINIMUM_SPEED_M_S, distance / INFLOW_ARROW_SECONDS));
  return { ...inflow, velocity_m_s: scale(scale(offset, 1 / distance), speed) };
}

export function setInflowRadius(inflow: FluidInflow, radius_m: number, container: SceneDescription["container"]): FluidInflow {
  const ceiling = 0.5 * Math.min(container.width_m, container.depth_m);
  return { ...inflow, radius_m: Math.min(ceiling, Math.max(INFLOW_MINIMUM_RADIUS_M, radius_m)) };
}
