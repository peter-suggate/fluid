import { INITIAL_FLUID_BRICK_SIZE } from "./initial-fluid";
import type { SceneDescription, Vec3 } from "./model";

/**
 * Painting water as initial brick seeds.
 *
 * `fluid.initialBrickSeeds_m` is already the only way to author arbitrary
 * water bodies, is already in the solver rebuild key, and each seed wets
 * exactly the one 8³ brick containing it. Painting therefore needs no schema
 * change and no new solver path: it snaps a pointer hit to a brick centre and
 * edits the seed list.
 *
 * The lattice mirrors the solver's rounding (`Math.round(extent / cellSize)`,
 * floored at 8 — see `createTallCellLayout`). Seeds are placed at brick
 * centres so a small disagreement between this mirror and a given solver's
 * dimensions still resolves to the intended brick.
 */

export interface EditorFluidLattice {
  readonly dimensions: readonly [number, number, number];
  readonly origin_m: Vec3;
  readonly brickSize_m: Vec3;
  readonly bricks: { readonly x: number; readonly y: number; readonly z: number };
}

export interface FluidBrickIndex {
  readonly x: number;
  readonly y: number;
  readonly z: number;
}

export function editorFluidLattice(scene: SceneDescription): EditorFluidLattice {
  const c = scene.container;
  const cellSize_m = scene.voxelDomain.finestCellSize_m;
  const axis = (extent_m: number) => Math.max(8, Math.round(extent_m / cellSize_m));
  const dimensions = [axis(c.width_m), axis(c.height_m), axis(c.depth_m)] as const;
  const brick = INITIAL_FLUID_BRICK_SIZE;
  return {
    dimensions,
    origin_m: { x: -0.5 * c.width_m, y: 0, z: -0.5 * c.depth_m },
    brickSize_m: {
      x: (c.width_m / dimensions[0]) * brick,
      y: (c.height_m / dimensions[1]) * brick,
      z: (c.depth_m / dimensions[2]) * brick,
    },
    bricks: {
      x: Math.ceil(dimensions[0] / brick),
      y: Math.ceil(dimensions[1] / brick),
      z: Math.ceil(dimensions[2] / brick),
    },
  };
}

/** Brick containing a world point, or undefined when it is outside the tank. */
export function fluidBrickIndexAt(lattice: EditorFluidLattice, point: Vec3): FluidBrickIndex | undefined {
  const x = Math.floor((point.x - lattice.origin_m.x) / lattice.brickSize_m.x);
  const y = Math.floor((point.y - lattice.origin_m.y) / lattice.brickSize_m.y);
  const z = Math.floor((point.z - lattice.origin_m.z) / lattice.brickSize_m.z);
  if (x < 0 || y < 0 || z < 0 || x >= lattice.bricks.x || y >= lattice.bricks.y || z >= lattice.bricks.z) return undefined;
  return { x, y, z };
}

export function fluidBrickCenter(lattice: EditorFluidLattice, index: FluidBrickIndex): Vec3 {
  return {
    x: lattice.origin_m.x + (index.x + 0.5) * lattice.brickSize_m.x,
    y: lattice.origin_m.y + (index.y + 0.5) * lattice.brickSize_m.y,
    z: lattice.origin_m.z + (index.z + 0.5) * lattice.brickSize_m.z,
  };
}

export function fluidBrickKey(index: FluidBrickIndex): string {
  return `${index.x}:${index.y}:${index.z}`;
}

/** Existing seeds keyed by the brick they wet, so repeats collapse. */
function seedsByBrick(scene: SceneDescription, lattice: EditorFluidLattice): Map<string, Vec3> {
  const byBrick = new Map<string, Vec3>();
  for (const seed of scene.fluid.initialBrickSeeds_m ?? []) {
    const index = fluidBrickIndexAt(lattice, seed);
    if (index) byBrick.set(fluidBrickKey(index), seed);
  }
  return byBrick;
}

export interface FluidPaintResult {
  readonly seeds: Vec3[];
  /** Absent when erasing emptied the list — the field must be dropped, not `[]`. */
  readonly empty: boolean;
}

/**
 * Add the brick under a point. Returns undefined when nothing changed, so a
 * drag across one brick does not churn the document.
 */
export function paintFluidBrick(scene: SceneDescription, point: Vec3): FluidPaintResult | undefined {
  const lattice = editorFluidLattice(scene);
  const index = fluidBrickIndexAt(lattice, point);
  if (!index) return undefined;
  const byBrick = seedsByBrick(scene, lattice);
  const key = fluidBrickKey(index);
  if (byBrick.has(key)) return undefined;
  byBrick.set(key, fluidBrickCenter(lattice, index));
  return { seeds: [...byBrick.values()], empty: false };
}

/** Remove the brick under a point, if one is seeded there. */
export function eraseFluidBrick(scene: SceneDescription, point: Vec3): FluidPaintResult | undefined {
  const lattice = editorFluidLattice(scene);
  const index = fluidBrickIndexAt(lattice, point);
  if (!index) return undefined;
  const byBrick = seedsByBrick(scene, lattice);
  if (!byBrick.delete(fluidBrickKey(index))) return undefined;
  const seeds = [...byBrick.values()];
  return { seeds, empty: seeds.length === 0 };
}

/**
 * Fold a paint result into a fluid patch. Painted water is always additive so
 * a brush stroke adds to the authored dam or tank fill rather than replacing
 * it, and an emptied list drops the field entirely because `validateScene`
 * rejects an empty `initialBrickSeeds_m`.
 */
export function fluidPaintPatch(scene: SceneDescription, result: FluidPaintResult): SceneDescription["fluid"] {
  const { initialBrickSeeds_m: _dropped, ...rest } = scene.fluid;
  if (result.empty) return { ...rest };
  return { ...rest, initialBrickSeeds_m: result.seeds, initialBrickSeedsAdditive: true };
}

/**
 * The bounds `validateScene` holds initial brick seeds to.
 *
 * `editorFluidLattice` rounds its brick count up, so the last brick on an axis
 * whose cell count is not a multiple of eight has its centre outside the tank —
 * a 24 x 18 x 16 cell tank has three brick rows in y and only two and a quarter
 * fit. Painting there used to author a seed the schema rejects, which surfaced
 * as a document that would not save rather than as a brush that would not
 * paint.
 */
function seedInsideSolverBounds(scene: SceneDescription, point: Vec3): boolean {
  const c = scene.container;
  return point.x >= -c.width_m / 2 && point.x < c.width_m / 2
    && point.y >= 0 && point.y < c.height_m
    && point.z >= -c.depth_m / 2 && point.z < c.depth_m / 2;
}

export interface FluidBrushSample {
  /** The brick this sample landed in, so a drag only revises on a crossing. */
  readonly brickKey: string;
  /** Absent when the sample changed nothing — a repeat, or an erase over dry brick. */
  readonly patch?: Partial<SceneDescription>;
}

/**
 * One sample of a brush stroke, as a patch over the committed document.
 *
 * Two scenes, because a stroke is a proposal being built up: `proposed` is what
 * the pointer has painted so far and is what each new brick unions with, while
 * `committed` is what the patch will land on. Folding them into one would make
 * a stroke either forget its own earlier bricks or reopen the flip below on
 * every sample.
 *
 * The patch is whole rather than incremental because `updateDraft` replaces the
 * draft's patch outright: a sample that emitted only the fields it changed
 * would silently retract the ones it did not.
 */
export function fluidBrushSample(
  committed: SceneDescription,
  proposed: SceneDescription,
  point: Vec3,
  erase: boolean,
): FluidBrushSample | undefined {
  const lattice = editorFluidLattice(proposed);
  const index = fluidBrickIndexAt(lattice, point);
  if (!index) return undefined;
  // A surface hit sits on the boundary of the brick behind it; nudging into the
  // brick centre keeps a stroke on the surface from seeding solids.
  const target = erase ? point : fluidBrickCenter(lattice, index);
  const brickKey = fluidBrickKey(index);
  if (!erase && !seedInsideSolverBounds(proposed, target)) return { brickKey };
  const result = erase ? eraseFluidBrick(proposed, target) : paintFluidBrick(proposed, target);
  if (!result) return { brickKey };
  return {
    brickKey,
    patch: {
      fluid: fluidPaintPatch(proposed, result),
      // The first brick of water in a scene built without a solver is the one
      // expensive transition in this editor, and it is meant to be exactly
      // this: user-initiated, named, and once per scene. Erasing never flips it
      // back — a scene that has had water is a fluid scene with none in it.
      ...(committed.systems?.fluid === false && !result.empty
        ? { systems: { ...committed.systems, fluid: true } }
        : {}),
    },
  };
}

/** World height of the tank-fill surface for the current fill fraction. */
export function fillLevelHeight_m(scene: SceneDescription): number {
  return Math.max(0, Math.min(1, scene.container.fillFraction)) * scene.container.height_m;
}

/** Fill fraction for a dragged surface height, clamped to the authorable range. */
export function fillFractionForHeight(scene: SceneDescription, height_m: number): number {
  return Math.max(0, Math.min(1, height_m / Math.max(scene.container.height_m, 1e-6)));
}

/** Corner post the fill-level handle rides, kept clear of painting targets. */
export function fillLevelHandlePosition(scene: SceneDescription): Vec3 {
  const c = scene.container;
  return { x: -0.5 * c.width_m, y: fillLevelHeight_m(scene), z: -0.5 * c.depth_m };
}
