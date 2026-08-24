import { add, cross, dot, length, normalize, scale, sub } from "./math";
import {
  boxHandles,
  boxSize,
  frameRayToLocal,
  moveHandles,
  pickSolidBox,
  positionFields,
  resizeBox,
  pickExcluded,
  type BoxExtent,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
  type EditorFrame,
  type EditorHandle,
} from "./editor-entity";
import type { FluidInflow, Quaternion, SceneDescription, Vec3 } from "./model";

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

export const INFLOW_SELECTION_ID = "inflow";
export const INFLOW_MINIMUM_SPEED_M_S = 0.05;
export const INFLOW_MAXIMUM_SPEED_M_S = 12;
export const INFLOW_MINIMUM_RADIUS_M = 0.01;
export const INFLOW_MINIMUM_LENGTH_M = 0.02;

/** Metres of arrow per m/s, so the arrow length reads as the jet speed. */
export const INFLOW_ARROW_SECONDS = 0.12;

export function inflowSpeed_m_s(inflow: FluidInflow): number {
  return length(inflow.velocity_m_s);
}

/** The authored circular outlet's volume flux, in the unit shown by the UI. */
export function inflowFlowRate_L_s(inflow: FluidInflow): number {
  return 1000 * Math.PI * inflow.radius_m ** 2 * inflowSpeed_m_s(inflow);
}

export function inflowDirection(inflow: FluidInflow): Vec3 {
  const speed = inflowSpeed_m_s(inflow);
  return speed > 1e-9 ? scale(inflow.velocity_m_s, 1 / speed) : { x: 0, y: -1, z: 0 };
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

/**
 * Set volume flux without changing the launch velocity that shapes the arc.
 * For a circular outlet Q = pi r^2 |u|, so the amount control changes only
 * the bore area and then uses the same geometry bounds as direct resizing.
 */
export function setInflowFlowRate_L_s(
  inflow: FluidInflow,
  flowRate_L_s: number,
  container: SceneDescription["container"],
): FluidInflow {
  const speed = inflowSpeed_m_s(inflow);
  if (!(speed > 0)) return inflow;
  const radius_m = Math.sqrt(Math.max(0, flowRate_L_s) / (1000 * Math.PI * speed));
  return setInflowRadius(inflow, radius_m, container);
}

/**
 * The channel's length, floored so the nozzle never becomes a disc.
 *
 * The floor is absolute rather than a multiple of the bore: `validateScene` asks
 * only that both are positive, and tying them would mean widening a hose
 * silently lengthened it.
 */
export function setInflowLength(inflow: FluidInflow, length_m: number): FluidInflow {
  return { ...inflow, length_m: Math.max(INFLOW_MINIMUM_LENGTH_M, length_m) };
}

// ---- entity ---------------------------------------------------------------

/**
 * The rotation that carries the entity's +y onto the jet direction.
 *
 * The hose is a cylinder about the axis it sprays along, so aiming it *is*
 * orienting it — and once the frame carries the aim, the nozzle is an ordinary
 * box entity: local x and z are its radius, local y its length. Its two bespoke
 * handles stop being bespoke.
 */
function inflowOrientation(direction: Vec3): Quaternion {
  const up = { x: 0, y: 1, z: 0 };
  const aim = normalize(direction);
  const alignment = dot(up, aim);
  // Anti-parallel is the singular case: any axis perpendicular to y will do, and
  // x is as good as another.
  if (alignment < -1 + 1e-9) return { w: 0, x: 1, y: 0, z: 0 };
  const axis = cross(up, aim);
  return normalize4({ w: 1 + alignment, x: axis.x, y: axis.y, z: axis.z });
}

function normalize4(q: Quaternion): Quaternion {
  const magnitude = Math.hypot(q.w, q.x, q.y, q.z);
  return magnitude > 1e-12
    ? { w: q.w / magnitude, x: q.x / magnitude, y: q.y / magnitude, z: q.z / magnitude }
    : { w: 1, x: 0, y: 0, z: 0 };
}

export function inflowBox(inflow: FluidInflow): BoxExtent {
  const half = 0.5 * inflow.length_m;
  return {
    min: { x: -inflow.radius_m, y: -half, z: -inflow.radius_m },
    max: { x: inflow.radius_m, y: half, z: inflow.radius_m },
  };
}

function inflowEntityFor(context: EditorEntityContext): EditorEntity | undefined {
  const scene = context.scene;
  const inflow = scene.fluid.inflow;
  if (!inflow) return undefined;
  const direction = inflowDirection(inflow);
  const frame: EditorFrame = { origin_m: inflow.center_m, orientation: inflowOrientation(direction) };
  const box = inflowBox(inflow);
  const patch = (next: FluidInflow) => ({ fluid: { ...scene.fluid, inflow: next } });
  const speed = inflowSpeed_m_s(inflow);
  const tip: EditorHandle = {
    id: "aim",
    kind: "tip",
    space: "world",
    label: "aim · direction and speed",
    position_m: add(inflow.center_m, scale(direction, INFLOW_ARROW_SECONDS * speed)),
    // The arrow points anywhere, so it owns all three axes and the lock narrows
    // it exactly as it narrows a corner.
    axes: ["x", "y", "z"],
    drag: (point_m) => patch(aimInflow(inflow, point_m)),
  };
  return {
    selection: { kind: "inflow", id: INFLOW_SELECTION_ID },
    label: "HOSE",
    tone: "inflow",
    frame,
    box,
    sizeLabel: `⌀${(2 * inflow.radius_m).toFixed(3)} m · ${speed.toFixed(2)} m/s · ${inflowFlowRate_L_s(inflow).toFixed(1)} L/s`,
    handles: [
      ...boxHandles(box, {
        drag: (sides, point_m) => {
          const next = resizeBox(box, sides, point_m, {
            minimum_m: [2 * INFLOW_MINIMUM_RADIUS_M, INFLOW_MINIMUM_LENGTH_M, 2 * INFLOW_MINIMUM_RADIUS_M],
            links: [["x", "z"]],
          });
          const size = boxSize(next);
          return patch(setInflowLength(
            setInflowRadius(inflow, 0.5 * size.x, scene.container), size.y));
        },
      }),
      ...moveHandles(inflow.center_m,
        (center_m) => patch(moveInflow(inflow, center_m, scene.container))),
      tip,
    ],
    draftSubject: "inflow",
    editLabel: (handle) => handle.kind === "tip" ? "Aimed the hose"
      : handle.space === "world" ? "Moved the hose" : "Resized the hose",
    fields: [
      ...positionFields(inflow.center_m,
        (center_m) => patch(moveInflow(inflow, center_m, scene.container))),
      {
        id: "radius",
        label: "⌀",
        unit: "m",
        value: 2 * inflow.radius_m,
        step: 0.005,
        min: 2 * INFLOW_MINIMUM_RADIUS_M,
        apply: (value) => patch(setInflowRadius(inflow, 0.5 * value, scene.container)),
      },
      {
        id: "speed",
        label: "Speed",
        unit: "m/s",
        value: speed,
        step: 0.1,
        min: INFLOW_MINIMUM_SPEED_M_S,
        max: INFLOW_MAXIMUM_SPEED_M_S,
        apply: (value) => patch({ ...inflow, velocity_m_s: scale(direction, value) }),
      },
      {
        id: "flow",
        label: "Flow",
        unit: "L/s",
        value: inflowFlowRate_L_s(inflow),
        step: 1,
        min: 0.1,
        apply: (value) => patch(setInflowFlowRate_L_s(inflow, value, scene.container)),
      },
    ],
    // When the jet runs, which the arrow cannot say. The nozzle's own length is
    // here beside them rather than beside the bore: both are the channel, and
    // the box handles are already the direct way to reshape it.
    groups: [{
      id: "schedule",
      label: "Jet",
      hint: "The channel's length, and the window the jet injects over",
      fields: [
        {
          id: "length",
          label: "Length",
          unit: "m",
          value: inflow.length_m,
          step: 0.01,
          min: INFLOW_MINIMUM_LENGTH_M,
          apply: (value) => patch(setInflowLength(inflow, value)),
        },
        {
          id: "ramp",
          label: "Ramp",
          unit: "s",
          value: inflow.ramp_s,
          step: 0.05,
          min: 0,
          apply: (value) => patch({ ...inflow, ramp_s: Math.max(0, value) }),
        },
        {
          id: "start",
          label: "Start",
          unit: "s",
          value: inflow.start_s,
          step: 0.5,
          min: 0,
          apply: (value) => patch({ ...inflow, start_s: Math.max(0, value) }),
        },
        {
          id: "end",
          label: "End",
          unit: "s",
          value: inflow.end_s,
          step: 0.5,
          min: 0,
          apply: (value) => patch({ ...inflow, end_s: Math.max(0, value) }),
        },
      ],
    }],
    remove: () => {
      const { inflow: _dropped, ...fluid } = scene.fluid;
      void _dropped;
      return { ...scene, fluid };
    },
  };
}

/** The nozzle is clicked on its own channel, like any other solid. */
export const inflowEntity: EditorEntityDefinition = {
  kind: "inflow",
  surfacedBy: (tool) => tool === "select" || tool === "inflow",
  instances: (context) => {
    const entity = inflowEntityFor(context);
    return entity ? [entity] : [];
  },
  find: (context, id) => id === INFLOW_SELECTION_ID ? inflowEntityFor(context) : undefined,
  pick: (context, ray, exclude) => {
    if (pickExcluded(exclude, "inflow", INFLOW_SELECTION_ID)) return undefined;
    const entity = inflowEntityFor(context);
    if (!entity?.box) return undefined;
    // Against the channel's own bounds, in its own frame — the same transform
    // the handles use, so what is clickable is exactly what is drawn.
    const distance_m = pickSolidBox(frameRayToLocal(entity.frame, ray), entity.box);
    return distance_m === undefined
      ? undefined
      : { selection: { kind: "inflow", id: INFLOW_SELECTION_ID }, distance_m };
  },
};
