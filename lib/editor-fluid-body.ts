import { damBreakFractions } from "./initial-fluid";
import {
  boxCenter,
  boxHandles,
  boxResizeDrag,
  moveBoxWithinLimits,
  moveHandles,
  pickSolidBox,
  positionFields,
  resizeBox,
  sceneContainerBox,
  WORLD_FRAME,
  type BoxExtent,
  type BoxResizePolicy,
  type BoxSides,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
} from "./editor-entity";
import { sceneCellSizes_m } from "./scene-lattice";
import type { SceneDescription, Vec3 } from "./model";

/**
 * Direct manipulation of the initial water body.
 *
 * The body is an axis-aligned box in metres with the full complement of
 * handles — six faces, twelve edges, eight corners — so it can be grown,
 * shrunk, and reshaped from whichever side is facing the camera.
 *
 * It is authored as `fluid.initialDamBreakDimensions_m` plus the optional
 * `fluid.initialDamBreakOrigin_m`, which is a continuous box rather than the
 * brick-quantized `initialBrickSeeds_m` the paint tool writes: at a typical
 * 0.4 m brick a painted box could only resize in three steps across the whole
 * tank. A tank fill is the same box spanning the footprint, so grabbing a
 * handle on a filled tank reshapes the pool it already had instead of
 * replacing it — the conversion is exact, and the water never jumps.
 *
 * Every edit lands in the solver's seed tier, so a drag re-seeds the existing
 * solver rather than building a new one.
 */

/** A water-body box is an ordinary box extent; the name survives its callers. */
export type FluidBodyBox = BoxExtent;

export const FLUID_BODY_SELECTION_ID = "fluid-body";
/** Cells of thickness the body may never drop below on any axis. */
export const FLUID_BODY_MINIMUM_CELLS = 1;

const AXES: readonly (keyof Vec3)[] = Object.freeze(["x", "y", "z"]);

/** World-space bounds the body may occupy: the container interior. */
export function fluidBodyLimits(scene: SceneDescription): FluidBodyBox {
  return sceneContainerBox(scene);
}

/**
 * The initial water body as a world-space box, or undefined when the scene has
 * no shapeable body — a fluid-less render scene, or an empty tank.
 *
 * Painted brick seeds are deliberately not folded in: they are a separate,
 * additive authoring surface with its own tool, and a box gizmo that silently
 * swallowed a painted blob into a rectangle would destroy work.
 */
export function fluidBodyBox(scene: SceneDescription): FluidBodyBox | undefined {
  if (scene.systems?.fluid === false) return undefined;
  const c = scene.container;
  const limits = fluidBodyLimits(scene);
  if (scene.fluid.initialCondition === "tank-fill") {
    const height_m = Math.max(0, Math.min(1, c.fillFraction)) * c.height_m;
    if (!(height_m > 0)) return undefined;
    return { min: limits.min, max: { x: limits.max.x, y: height_m, z: limits.max.z } };
  }
  const authored = scene.fluid.initialDamBreakDimensions_m;
  const size = authored ?? (() => {
    const fractions = damBreakFractions(c.fillFraction);
    return { x: fractions.width * c.width_m, y: fractions.height * c.height_m, z: fractions.depth * c.depth_m };
  })();
  if (!(size.x > 0) || !(size.y > 0) || !(size.z > 0)) return undefined;
  const origin = scene.fluid.initialDamBreakOrigin_m ?? { x: 0, y: 0, z: 0 };
  const min = { x: limits.min.x + origin.x, y: limits.min.y + origin.y, z: limits.min.z + origin.z };
  return { min, max: { x: min.x + size.x, y: min.y + size.y, z: min.z + size.z } };
}

/**
 * How the water body reshapes: onto the finest lattice, inside the container,
 * never thinner than a cell.
 *
 * The lattice is the resolution at which the seed actually changes — an
 * unsnapped box edge lands mid-cell, where it wets nothing new and the handle
 * appears to do nothing.
 */
export function fluidBodyResizePolicy(scene: SceneDescription): BoxResizePolicy {
  const cell = sceneCellSizes_m(scene);
  return {
    snap_m: [cell[0]!, cell[1]!, cell[2]!],
    limits: fluidBodyLimits(scene),
    minimum_m: [
      FLUID_BODY_MINIMUM_CELLS * cell[0]!,
      FLUID_BODY_MINIMUM_CELLS * cell[1]!,
      FLUID_BODY_MINIMUM_CELLS * cell[2]!,
    ],
  };
}

/** Move the sides a handle owns to the dragged point. */
export function dragFluidBodyBox(
  box: FluidBodyBox,
  sides: BoxSides,
  point: Vec3,
  scene: SceneDescription,
): FluidBodyBox {
  return resizeBox(box, sides, point, fluidBodyResizePolicy(scene));
}

/**
 * Scale the body about its own centre, held inside the container.
 *
 * `factor` is per-axis, not a volume: `scaleFluidBodyVolume` is what the
 * grow/shrink control calls, because a button beside a reading in litres has
 * to mean twice the water rather than eight times it.
 */
export function scaleFluidBodyBox(
  box: FluidBodyBox,
  factor: number,
  scene: SceneDescription,
): FluidBodyBox {
  const limits = fluidBodyLimits(scene);
  const cell = sceneCellSizes_m(scene);
  const min = { ...box.min }, max = { ...box.max };
  const snap = (value: number, step: number, origin: number) =>
    step > 0 ? origin + Math.round((value - origin) / step) * step : value;
  AXES.forEach((axis, index) => {
    const step = cell[index]!;
    const minimum = FLUID_BODY_MINIMUM_CELLS * step;
    const available = limits.max[axis] - limits.min[axis];
    const centre = 0.5 * (box.min[axis] + box.max[axis]);
    const half = Math.min(0.5 * available,
      Math.max(0.5 * minimum, 0.5 * (box.max[axis] - box.min[axis]) * factor));
    // Keep the requested size and slide the body back inside rather than
    // clipping it against the wall it was pushed through.
    const shift = Math.min(limits.max[axis] - (centre + half), Math.max(limits.min[axis] - (centre - half), 0));
    min[axis] = snap(centre - half + shift, step, limits.min[axis]);
    max[axis] = snap(centre + half + shift, step, limits.min[axis]);
    if (max[axis] - min[axis] < minimum) max[axis] = min[axis] + minimum;
  });
  return { min, max };
}

/**
 * Slide the body without reshaping it, held inside the container.
 *
 * A box already touching a wall stops there rather than being squashed against
 * it: the gesture is a move, so it must never silently change the volume.
 */
export function moveFluidBodyBox(
  box: FluidBodyBox,
  centre_m: Vec3,
  scene: SceneDescription,
): FluidBodyBox {
  return moveBoxWithinLimits(box, centre_m, fluidBodyLimits(scene));
}

/**
 * Grow or shrink the body by a factor of its *volume*, which is what the
 * control's litres readout promises. The container clamps each axis, so a body
 * already against a wall grows only where it still can.
 */
export function scaleFluidBodyVolume(
  box: FluidBodyBox,
  volumeFactor: number,
  scene: SceneDescription,
): FluidBodyBox {
  return scaleFluidBodyBox(box, Math.cbrt(volumeFactor), scene);
}

export function fluidBodyBoxVolume_m3(box: FluidBodyBox): number {
  return Math.max(0, box.max.x - box.min.x)
    * Math.max(0, box.max.y - box.min.y)
    * Math.max(0, box.max.z - box.min.z);
}

/**
 * Author a box back into the document.
 *
 * The reservoir stays anchor-free only when it has to: a box still sitting in
 * the container's minimum corner omits the origin entirely, which keeps legacy
 * scenes byte-identical and keeps the GPU's closed-form t=0 bootstrap — an
 * authored origin costs the analytic path and rasterizes the seed on the host.
 *
 * `fillFraction` is not an independent knob here. `validateScene` requires it
 * to equal the reservoir's share of the container whenever the dimensions are
 * authored, so it is derived, and a tank fill that has just been reshaped into
 * a box carries the same water it had.
 */
export function fluidBodyBoxPatch(
  scene: SceneDescription,
  box: FluidBodyBox,
): Pick<SceneDescription, "container" | "fluid"> {
  const c = scene.container;
  const limits = fluidBodyLimits(scene);
  const size = {
    x: Math.max(0, box.max.x - box.min.x),
    y: Math.max(0, box.max.y - box.min.y),
    z: Math.max(0, box.max.z - box.min.z),
  };
  const origin = {
    x: box.min.x - limits.min.x,
    y: box.min.y - limits.min.y,
    z: box.min.z - limits.min.z,
  };
  const anchored = Math.abs(origin.x) < 1e-9 && Math.abs(origin.y) < 1e-9 && Math.abs(origin.z) < 1e-9;
  const { initialDamBreakOrigin_m: _dropped, ...fluid } = scene.fluid;
  return {
    container: {
      ...c,
      fillFraction: Math.max(0, Math.min(1,
        (size.x * size.y * size.z) / (c.width_m * c.height_m * c.depth_m))),
    },
    fluid: {
      ...fluid,
      initialCondition: "dam-break",
      initialDamBreakDimensions_m: size,
      ...(anchored ? {} : { initialDamBreakOrigin_m: origin }),
    },
  };
}

// ---- entity ---------------------------------------------------------------

function fluidBodyEntityFor(context: EditorEntityContext): EditorEntity | undefined {
  const scene = context.scene;
  const box = fluidBodyBox(scene);
  if (!box) return undefined;
  const size = [box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z];
  return {
    selection: { kind: "fluid-body", id: FLUID_BODY_SELECTION_ID },
    label: "WATER",
    tone: "fluid",
    frame: WORLD_FRAME,
    box,
    sizeLabel: `${size.map((value) => value.toFixed(2)).join(" × ")} m`,
    handles: [
      ...boxHandles(box, {
        drag: boxResizeDrag(box, fluidBodyResizePolicy(scene),
          (next) => fluidBodyBoxPatch(scene, next)),
      }),
      // The reservoir moves because `initialDamBreakOrigin_m` exists to let it:
      // that field is the whole reason the body can leave the container's corner.
      ...moveHandles(boxCenter(box),
        (centre) => fluidBodyBoxPatch(scene, moveFluidBodyBox(box, centre, scene))),
    ],
    draftSubject: "fluid-body",
    editLabel: (handle) => handle.space === "world"
      ? "Moved the water body" : "Reshaped the water body",
    fields: positionFields(boxCenter(box),
      (centre) => fluidBodyBoxPatch(scene, moveFluidBodyBox(box, centre, scene))),
  };
}

/**
 * The water body is clicked on its seed box.
 *
 * That box is where the water will be at t=0 rather than where the solver has
 * since carried it, so a click late in a run selects the reservoir from a place
 * the water has already left. That is the honest target: the box is the thing
 * the handles move, and picking what is drawn instead would mean picking a
 * simulation result that no edit can reach.
 */
export const fluidBodyEntity: EditorEntityDefinition = {
  kind: "fluid-body",
  surfacedBy: (tool) => tool === "select",
  instances: (context) => {
    const entity = fluidBodyEntityFor(context);
    return entity ? [entity] : [];
  },
  find: (context, id) => {
    if (id !== FLUID_BODY_SELECTION_ID) return undefined;
    return fluidBodyEntityFor(context);
  },
  pick: (context, ray) => {
    const box = fluidBodyBox(context.scene);
    if (!box) return undefined;
    const distance_m = pickSolidBox(ray, box);
    return distance_m === undefined
      ? undefined
      : { selection: { kind: "fluid-body", id: FLUID_BODY_SELECTION_ID }, distance_m };
  },
};
