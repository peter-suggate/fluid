import { cloneScene, type SceneDescription, type Vec3 } from "./model";
import {
  DEFAULT_MAXIMUM_LATTICE_DIMENSION,
  MINIMUM_LATTICE_DIMENSION,
  sceneLatticeDimensions,
} from "./scene-lattice";

/**
 * Scaling a scene by factors of two.
 *
 * Two independent axes, because they cost completely different things:
 *
 * - **World** multiplies the container extents and the finest cell size by the
 *   same factor. The lattice is untouched, so the solver keeps its arenas and
 *   its compiled pipelines and only re-seeds: the scene gets physically bigger
 *   or smaller at a constant price. (`gpuSceneStructuralKey` keys the lattice
 *   in cells precisely so this stays a re-seed.)
 * - **Detail** divides the finest cell size at a fixed world size. That is
 *   eight times the cells per step, so it reallocates every arena — but the
 *   pipeline cache is keyed on shader capabilities and pipeline constants, not
 *   on dimensions, so nothing recompiles.
 *
 * Nothing else scales. A world scale grows the room, not its contents: bodies,
 * props, painted water, the hose, and the camera all stay exactly where and
 * what they were. The only edits below beyond the two lengths are the repairs
 * that keep the document *valid* — a reservoir that no longer fits, a fill
 * fraction that must equal the reservoir's volume share, seeds or a nozzle
 * left outside a shrunken tank. Those are not scaling; they are the alternative
 * to emitting a scene `validateScene` rejects.
 */

export type SceneScaleAxis = "world" | "detail";
export type SceneScaleFactor = 2 | 0.5;

/** Guards against a step that would exhaust the device or stop resolving. */
export const MAXIMUM_SCALED_LATTICE_CELLS = 1 << 26;

export interface SceneScaleOption {
  readonly axis: SceneScaleAxis;
  readonly factor: SceneScaleFactor;
  readonly available: boolean;
  /** Why the step is refused, for the control's tooltip. */
  readonly blocked?: string;
  /** Lattice the step would produce, whether or not it is available. */
  readonly dimensions: readonly [number, number, number];
}

export interface SceneScaleSummary {
  readonly extents_m: readonly [number, number, number];
  readonly cellSize_m: number;
  readonly dimensions: readonly [number, number, number];
  readonly cells: number;
  readonly options: readonly SceneScaleOption[];
}

function scaleVec3(value: Vec3, factor: number): Vec3 {
  return { x: value.x * factor, y: value.y * factor, z: value.z * factor };
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/**
 * Re-establish the invariants `validateScene` enforces after the container has
 * moved underneath the authored content. Every clamp here is a repair, not a
 * scale: content keeps its size wherever the new container still admits it.
 */
export function repairSceneForContainer(scene: SceneDescription): SceneDescription {
  const c = scene.container;
  const dimensions = scene.fluid.initialDamBreakDimensions_m;
  if (dimensions) {
    // Keep the reservoir's size where it fits, and never let it invert.
    const size: Vec3 = {
      x: clamp(dimensions.x, 0, c.width_m),
      y: clamp(dimensions.y, 0, c.height_m),
      z: clamp(dimensions.z, 0, c.depth_m),
    };
    const origin = scene.fluid.initialDamBreakOrigin_m;
    const placed: Vec3 | undefined = origin && {
      x: clamp(origin.x, 0, c.width_m - size.x),
      y: clamp(origin.y, 0, c.height_m - size.y),
      z: clamp(origin.z, 0, c.depth_m - size.z),
    };
    scene.fluid.initialDamBreakDimensions_m = size;
    if (placed) scene.fluid.initialDamBreakOrigin_m = placed;
    // validateScene requires the fill fraction to equal the reservoir's share
    // of the container, so a container that moved has to be answered here.
    scene.container.fillFraction = clamp(
      (size.x * size.y * size.z) / (c.width_m * c.height_m * c.depth_m), 0, 1);
  }
  const seeds = scene.fluid.initialBrickSeeds_m;
  if (seeds) {
    const inside = seeds.filter((seed) =>
      seed.x >= -c.width_m / 2 && seed.x < c.width_m / 2
      && seed.y >= 0 && seed.y < c.height_m
      && seed.z >= -c.depth_m / 2 && seed.z < c.depth_m / 2);
    // An emptied list is invalid; the field has to be dropped instead.
    if (inside.length === 0) delete scene.fluid.initialBrickSeeds_m;
    else if (inside.length !== seeds.length) scene.fluid.initialBrickSeeds_m = inside;
  }
  const inflow = scene.fluid.inflow;
  if (inflow) {
    inflow.center_m = {
      x: clamp(inflow.center_m.x, -c.width_m / 2, c.width_m / 2),
      y: clamp(inflow.center_m.y, 0, c.height_m),
      z: clamp(inflow.center_m.z, -c.depth_m / 2, c.depth_m / 2),
    };
  }
  // A ground plane at or above the ceiling leaves no container at all.
  if (scene.terrain && scene.terrain.baseHeight_m >= c.height_m) {
    scene.terrain.baseHeight_m = Math.max(0, c.height_m * 0.5);
  }
  return scene;
}

function scaledScene(scene: SceneDescription, axis: SceneScaleAxis, factor: SceneScaleFactor): SceneDescription {
  const next = cloneScene(scene);
  if (axis === "world") {
    next.container.width_m = scene.container.width_m * factor;
    next.container.height_m = scene.container.height_m * factor;
    next.container.depth_m = scene.container.depth_m * factor;
    next.voxelDomain.finestCellSize_m = scene.voxelDomain.finestCellSize_m * factor;
    // Metre lengths that describe the domain rather than its contents: the
    // oracle's cell and the authored address-space bounds are the same lattice
    // measured in different units, so they ride the world scale.
    next.nominalResolution = { length_m: scene.nominalResolution.length_m * factor };
    if (next.voxelDomain.bounds_m) {
      next.voxelDomain.bounds_m = {
        min: scaleVec3(next.voxelDomain.bounds_m.min, factor),
        max: scaleVec3(next.voxelDomain.bounds_m.max, factor),
      };
    }
  } else {
    next.voxelDomain.finestCellSize_m = scene.voxelDomain.finestCellSize_m / factor;
  }
  return repairSceneForContainer(next);
}

/** Why a step is refused, or undefined when it is available. */
function blockedReason(
  scene: SceneDescription,
  axis: SceneScaleAxis,
  candidate: SceneDescription,
): string | undefined {
  const dimensions = sceneLatticeDimensions(candidate);
  if (axis === "world") {
    // A world scale that moved the lattice is not the free step it claims to
    // be: the rounding drifted, and the solver would rebuild rather than
    // re-seed. Refusing is better than silently costing what we advertised.
    const current = sceneLatticeDimensions(scene);
    if (dimensions.some((value, axisIndex) => value !== current[axisIndex])) {
      return "the container no longer resolves to the same lattice at this size";
    }
    return undefined;
  }
  if (dimensions.some((value) => value >= DEFAULT_MAXIMUM_LATTICE_DIMENSION)) {
    return `an axis would reach the ${DEFAULT_MAXIMUM_LATTICE_DIMENSION}-cell device limit`;
  }
  if (dimensions.some((value) => value <= MINIMUM_LATTICE_DIMENSION)) {
    return `an axis would fall to the ${MINIMUM_LATTICE_DIMENSION}-cell floor and distort the container`;
  }
  const cells = dimensions[0] * dimensions[1] * dimensions[2];
  if (cells > MAXIMUM_SCALED_LATTICE_CELLS) {
    return `${(cells / 1e6).toFixed(0)}M cells is past the ${(MAXIMUM_SCALED_LATTICE_CELLS / 1e6).toFixed(0)}M budget`;
  }
  return undefined;
}

/**
 * The scene one step produces, or undefined when the step is refused. The
 * refusal reasons are reported by `sceneScaleSummary` so the control can
 * disable the button and say why rather than failing on click.
 */
export function scaleScene(
  scene: SceneDescription,
  axis: SceneScaleAxis,
  factor: SceneScaleFactor,
): SceneDescription | undefined {
  const candidate = scaledScene(scene, axis, factor);
  return blockedReason(scene, axis, candidate) ? undefined : candidate;
}

export function sceneScaleSummary(scene: SceneDescription): SceneScaleSummary {
  const dimensions = sceneLatticeDimensions(scene);
  const options: SceneScaleOption[] = [];
  for (const axis of ["world", "detail"] as const) {
    for (const factor of [0.5, 2] as const) {
      const candidate = scaledScene(scene, axis, factor);
      const blocked = blockedReason(scene, axis, candidate);
      options.push({
        axis,
        factor,
        available: blocked === undefined,
        blocked,
        dimensions: sceneLatticeDimensions(candidate),
      });
    }
  }
  return {
    extents_m: [scene.container.width_m, scene.container.height_m, scene.container.depth_m],
    cellSize_m: scene.voxelDomain.finestCellSize_m,
    dimensions,
    cells: dimensions[0] * dimensions[1] * dimensions[2],
    options,
  };
}

export function sceneScaleOption(
  summary: SceneScaleSummary,
  axis: SceneScaleAxis,
  factor: SceneScaleFactor,
): SceneScaleOption {
  const option = summary.options.find((entry) => entry.axis === axis && entry.factor === factor);
  if (!option) throw new Error(`No ${axis} scale option for factor ${factor}`);
  return option;
}
