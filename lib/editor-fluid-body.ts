import { damBreakFractions } from "./initial-fluid";
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

export interface FluidBodyBox {
  readonly min: Vec3;
  readonly max: Vec3;
}

export type FluidBodyHandleKind = "face" | "edge" | "corner";

export interface FluidBodyHandle {
  /** Stable sign triple, e.g. `+00` for the +x face, `+-+` for a corner. */
  readonly id: string;
  readonly kind: FluidBodyHandleKind;
  readonly position_m: Vec3;
  /** Which side of which axis the handle moves; absent axes are untouched. */
  readonly sides: Readonly<Partial<Record<keyof Vec3, "min" | "max">>>;
}

export const FLUID_BODY_HANDLE_TOLERANCE_PX = 9;
/** Cells of thickness the body may never drop below on any axis. */
export const FLUID_BODY_MINIMUM_CELLS = 1;

const AXES: readonly (keyof Vec3)[] = Object.freeze(["x", "y", "z"]);
const SIGNS = Object.freeze([-1, 0, 1] as const);
const SIGN_CHARACTERS: Readonly<Record<string, string>> = Object.freeze({ "-1": "-", "0": "0", "1": "+" });

/** World-space bounds the body may occupy: the container interior. */
export function fluidBodyLimits(scene: SceneDescription): FluidBodyBox {
  const c = scene.container;
  return {
    min: { x: -0.5 * c.width_m, y: 0, z: -0.5 * c.depth_m },
    max: { x: 0.5 * c.width_m, y: c.height_m, z: 0.5 * c.depth_m },
  };
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

/** The 26 handles of the box, in a stable order. */
export function fluidBodyHandles(box: FluidBodyBox): FluidBodyHandle[] {
  const handles: FluidBodyHandle[] = [];
  for (const sx of SIGNS) for (const sy of SIGNS) for (const sz of SIGNS) {
    const signs = { x: sx, y: sy, z: sz };
    const active = AXES.filter((axis) => signs[axis] !== 0);
    if (active.length === 0) continue;
    const sides: Partial<Record<keyof Vec3, "min" | "max">> = {};
    for (const axis of active) sides[axis] = signs[axis] < 0 ? "min" : "max";
    handles.push({
      id: AXES.map((axis) => SIGN_CHARACTERS[String(signs[axis])]).join(""),
      kind: active.length === 1 ? "face" : active.length === 2 ? "edge" : "corner",
      position_m: Object.fromEntries(AXES.map((axis) => [axis, signs[axis] === 0
        ? 0.5 * (box.min[axis] + box.max[axis])
        : signs[axis] < 0 ? box.min[axis] : box.max[axis]])) as unknown as Vec3,
      sides,
    });
  }
  return handles;
}

export function fluidBodyHandleById(box: FluidBodyBox, id: string): FluidBodyHandle | undefined {
  return fluidBodyHandles(box).find((handle) => handle.id === id);
}

/**
 * A face handle's outward normal. Faces drag along it rather than in the
 * camera plane, because a face has one degree of freedom and constraining the
 * pointer to it is what keeps the drag from sliding sideways.
 */
export function fluidBodyHandleAxis(handle: FluidBodyHandle): Vec3 | undefined {
  if (handle.kind !== "face") return undefined;
  const axis = AXES.find((candidate) => handle.sides[candidate] !== undefined);
  if (!axis) return undefined;
  return {
    x: axis === "x" ? 1 : 0,
    y: axis === "y" ? 1 : 0,
    z: axis === "z" ? 1 : 0,
  };
}

function snap(value: number, step: number, origin: number): number {
  if (!(step > 0)) return value;
  return origin + Math.round((value - origin) / step) * step;
}

/**
 * Move the sides a handle owns to the dragged point.
 *
 * Each moved side snaps to the finest lattice, because that is the resolution
 * at which the seed actually changes: an unsnapped box edge lands mid-cell,
 * where it wets nothing new and the handle appears to do nothing. Sides are
 * clamped against the container and against their opposite side, so pushing a
 * face through the body stops at a minimum thickness instead of inverting it.
 */
export function dragFluidBodyBox(
  box: FluidBodyBox,
  handle: FluidBodyHandle,
  point: Vec3,
  scene: SceneDescription,
): FluidBodyBox {
  const limits = fluidBodyLimits(scene);
  const cell = sceneCellSizes_m(scene);
  const min = { ...box.min }, max = { ...box.max };
  AXES.forEach((axis, index) => {
    const side = handle.sides[axis];
    if (!side) return;
    const step = cell[index]!;
    const minimum = FLUID_BODY_MINIMUM_CELLS * step;
    const target = snap(point[axis], step, limits.min[axis]);
    if (side === "max") max[axis] = Math.min(limits.max[axis], Math.max(min[axis] + minimum, target));
    else min[axis] = Math.max(limits.min[axis], Math.min(max[axis] - minimum, target));
  });
  return { min, max };
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

/** The eight corners, indexed so bit 0/1/2 selects the max side of x/y/z. */
export function fluidBodyBoxCorners(box: FluidBodyBox): Vec3[] {
  return Array.from({ length: 8 }, (_unused, index) => ({
    x: index & 1 ? box.max.x : box.min.x,
    y: index & 2 ? box.max.y : box.min.y,
    z: index & 4 ? box.max.z : box.min.z,
  }));
}

/** Corner index pairs of the twelve edges, for drawing the box outline. */
export const FLUID_BODY_BOX_EDGES: ReadonlyArray<readonly [number, number]> = Object.freeze(
  [0, 1, 2, 3, 4, 5, 6, 7].flatMap((index) => [1, 2, 4]
    .map((bit) => [index, index ^ bit] as const)
    .filter(([from, to]) => from < to)));

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
