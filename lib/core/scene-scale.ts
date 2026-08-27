import { cloneScene, type InitialLiquidVolume, type SceneDescription, type Vec3 } from "./model";
import {
  DEFAULT_MAXIMUM_LATTICE_DIMENSION,
  MINIMUM_LATTICE_DIMENSION,
  sceneLatticeDimensions,
  solidVoxelEditsForScene,
  solidVoxelShellForScene,
} from "./scene-lattice";
import { applyInitialFluidLayout, initialFluidLayout } from "./initial-fluid-layout";

/**
 * Scaling a scene by factors of two.
 *
 * Two independent axes, because they cost completely different things:
 *
 * - **World** multiplies the container extents and the finest cell size by the
 *   same factor. The lattice dimensions are untouched, but metre-mapped GPU
 *   state is rebuilt so seeded fluid changes physical size with the tank.
 *   Compiled pipelines remain cached.
 * - **Detail** divides the finest cell size at a fixed world size. That is
 *   eight times the cells per step, so it reallocates every arena — but the
 *   pipeline cache is keyed on shader capabilities and pipeline constants, not
 *   on dimensions, so nothing recompiles.
 *
 * Fluid scales with the tank: tank fills retain their fill fraction, authored
 * reservoirs scale their dimensions and placement, and painted water scales
 * its seed positions. Other contents — bodies, props, the hose, and the camera
 * — stay exactly where and what they were. The only other edits below are repairs
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

export interface SceneContainerExtents {
  readonly width_m: number;
  readonly height_m: number;
  readonly depth_m: number;
}

function scaleVec3(value: Vec3, factor: number): Vec3 {
  return { x: value.x * factor, y: value.y * factor, z: value.z * factor };
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}

/**
 * Move a finished scene onto an arbitrary solver lattice without changing the
 * initial water geometry it describes.
 *
 * Seed points address storage bricks, not metric volumes. Patching only
 * `finestCellSize_m` therefore changes the physical size of every seeded body:
 * one 8^3 brick at 50 mm becomes one 8^3 brick at 12.5 mm. Capture the water as
 * container-relative regions before moving the lattice, then rasterize those
 * same regions onto the new brick grid.
 */
export function sceneAtFinestCellSize(
  scene: SceneDescription,
  finestCellSize_m: number,
): SceneDescription {
  if (!(finestCellSize_m > 0) || !Number.isFinite(finestCellSize_m)) {
    throw new RangeError("Finest cell size must be positive and finite");
  }
  if (finestCellSize_m === scene.voxelDomain.finestCellSize_m) return cloneScene(scene);
  const water = initialFluidLayout(scene);
  const next = cloneScene(scene);
  next.voxelDomain.finestCellSize_m = finestCellSize_m;
  applyInitialFluidLayout(next, water);
  return repairSceneForContainer(next);
}

/**
 * Move the container walls onto new metric extents as one structural edit.
 *
 * A finished scene stores its tank as ordinary SolidWorld voxel patches. Raw
 * mutation of `scene.container` therefore does not resize the physical walls:
 * it strands the old shell inside the new lattice and leaves the enlarged
 * exterior open. Capture independent voxel edits while the old lattice still
 * identifies its derived shell, then compile that shell on the new lattice.
 */
export function sceneAtContainerExtents(
  scene: SceneDescription,
  extents: SceneContainerExtents,
): SceneDescription {
  for (const [axis, extent] of Object.entries(extents)) {
    if (!(extent > 0) || !Number.isFinite(extent)) {
      throw new RangeError(`Container ${axis} must be positive and finite`);
    }
  }
  const next = cloneScene(scene);
  const authoredEdits = solidVoxelEditsForScene(next);
  next.container = { ...next.container, ...extents };
  next.solidVoxels = [...solidVoxelShellForScene(next), ...authoredEdits];
  return repairSceneForContainer(next);
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
  const volumes = scene.fluid.initialLiquidVolumes;
  if (volumes) {
    scene.fluid.initialLiquidVolumes = volumes.flatMap<InitialLiquidVolume>((volume) => {
      if (volume.shape === "box") {
        const min_m = {
          x: clamp(volume.min_m.x, -c.width_m / 2, c.width_m / 2),
          y: clamp(volume.min_m.y, 0, c.height_m),
          z: clamp(volume.min_m.z, -c.depth_m / 2, c.depth_m / 2),
        };
        const max_m = {
          x: clamp(volume.max_m.x, -c.width_m / 2, c.width_m / 2),
          y: clamp(volume.max_m.y, 0, c.height_m),
          z: clamp(volume.max_m.z, -c.depth_m / 2, c.depth_m / 2),
        };
        return min_m.x < max_m.x && min_m.y < max_m.y && min_m.z < max_m.z
          ? [{ ...volume, min_m, max_m }] : [];
      }
      return [{
        ...volume,
        center_m: {
          x: clamp(volume.center_m.x, -c.width_m / 2, c.width_m / 2),
          y: clamp(volume.center_m.y, 0, c.height_m),
          z: clamp(volume.center_m.z, -c.depth_m / 2, c.depth_m / 2),
        },
      }];
    });
    if (scene.fluid.initialLiquidVolumes.length === 0) delete scene.fluid.initialLiquidVolumes;
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
  const grid = scene.terrain?.grid;
  if (grid && grid.heights_m.some((height) => !(height >= 0) || height > c.height_m)) {
    // A ceiling that dropped below the ground truncates it; the alternative is
    // an unloadable document. Rebuilt only when a sample is actually out of
    // range, so the common repair leaves the array shared.
    scene.terrain!.grid = {
      ...grid,
      heights_m: grid.heights_m.map((height) =>
        Number.isFinite(height) ? clamp(height, 0, c.height_m) : 0),
    };
  }
  return scene;
}

function scaledScene(scene: SceneDescription, axis: SceneScaleAxis, factor: SceneScaleFactor): SceneDescription {
  if (axis === "detail") {
    return sceneAtFinestCellSize(scene, scene.voxelDomain.finestCellSize_m / factor);
  }
  // Capture once into the multi-region representation. Every legacy initial
  // condition follows the same geometry path from here on.
  const water = initialFluidLayout(scene);
  const next = cloneScene(scene);
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
  // A baked terrain grid is metric through and through: where its samples
  // sit, how far apart they are, and how high they stand are all lengths in
  // the container's frame, so all three ride the scale together. A procedural
  // terrain needs none of this — it is re-derived against the new container.
  const grid = next.terrain?.grid;
  if (grid) {
    next.terrain!.grid = {
      ...grid,
      origin_m: { x: grid.origin_m.x * factor, z: grid.origin_m.z * factor },
      spacing_m: grid.spacing_m * factor,
      heights_m: grid.heights_m.map((height) => height * factor),
    };
  }
  // Analytic liquid volumes are absolute metres, so they ride world scale
  // whole and retain their share of the container and their radius in cells.
  if (next.fluid.initialLiquidVolumes) {
    next.fluid.initialLiquidVolumes = next.fluid.initialLiquidVolumes.map((volume) => volume.shape === "box"
      ? { ...volume, min_m: scaleVec3(volume.min_m, factor), max_m: scaleVec3(volume.max_m, factor) }
      : volume.shape === "cylinder"
        ? { ...volume, center_m: scaleVec3(volume.center_m, factor), radius_m: volume.radius_m * factor,
          halfHeight_m: volume.halfHeight_m * factor }
        : { ...volume, center_m: scaleVec3(volume.center_m, factor), radius_m: volume.radius_m * factor });
  }
  // WORLD changes metres per cell, so re-materialize the same canonical water
  // regions against the resulting metric lattice.
  applyInitialFluidLayout(next, water);
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
