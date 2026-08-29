/**
 * Authored refinement boxes as the scene document carries them.
 *
 * A region is a box the author draws over a quiet part of the domain to
 * declare the smallest pressure cell it may hold. The *rule* is an octree
 * instruction, but the box, its id, its dyadic cell-size ladder and the
 * lattice its corners are measured against are all document facts: the scene
 * validator rejects a malformed region, the editor snaps a drag to the same
 * ladder, and the viewport counts how many boxes remain. None of those callers
 * has any business constructing a solver to ask what a region is.
 *
 * The packing of these boxes into a projection params buffer, and the
 * containment tests the topology gate runs against that packing, stay with the
 * octree — they are the half that only means something to a solver.
 *
 * Containment is *full* containment: a candidate leaf is bounded only when the
 * whole leaf lies inside the box. A leaf straddling the edge keeps its ordinary
 * refinement behaviour, so a region cannot affect anything outside itself.
 * Because dyadic leaves of edge S are aligned to multiples of S in cell space,
 * a box whose bounds are multiples of its own smallest-cell bound contains
 * exactly the leaves it covers — which is why the editor snaps its handles to
 * that lattice.
 */
import type { FluidRefinementRegion, SceneDescription, Vec3 } from "./model";
import { sceneCellSizes_m, sceneLatticeDimensions } from "./scene-lattice";

/**
 * Regions carried per scene.
 *
 * They live in a fixed tail of the projection's uniform buffer rather than in a
 * storage binding: the topology shader's layout is already at the device's
 * storage-buffer budget (see `losasso-resident-solve`), and an experiment
 * surface that cost a binding would not be worth its own plumbing. Eight is
 * what fits comfortably and is far past what a hand-drawn experiment uses.
 */
export const OCTREE_REFINEMENT_REGION_CAPACITY = 8;

/**
 * Cell-size bounds a region may name, as an edge length in finest cells.
 *
 * Powers of two, because a leaf edge is a power of two: an octree has no cell
 * of edge 3, so offering one would be a control that silently rounds. The list
 * stops at 32 for the same reason `maximumLeafSize` does — that is the largest
 * leaf the topology can hold.
 */
export const OCTREE_REFINEMENT_REGION_CELL_SIZES: readonly number[] =
  Object.freeze([1, 2, 4, 8, 16, 32]);

export const DEFAULT_REFINEMENT_REGION_CELL_SIZE = 8;

/** The one rule a region can express today. */
export const REFINEMENT_REGION_RULES = Object.freeze([
  {
    id: "minimum-cell-size" as const,
    // The label is the value shown beside MEANS on a narrow column, so it is a
    // word rather than a phrase; the sentence it stands for is the hint.
    label: "Bounds",
    hint: "Choose the smallest allowed pressure cell and, optionally, the largest. Equal bounds hold fully contained leaves at one tier.",
  },
]);

export type RefinementRegionRule = (typeof REFINEMENT_REGION_RULES)[number]["id"];

/** Round a requested cell-size bound down onto the dyadic ladder above. */
export function clampRefinementRegionCellSize(requested: number): number {
  if (!Number.isFinite(requested)) return DEFAULT_REFINEMENT_REGION_CELL_SIZE;
  let chosen = OCTREE_REFINEMENT_REGION_CELL_SIZES[0]!;
  for (const size of OCTREE_REFINEMENT_REGION_CELL_SIZES) {
    if (size <= requested + 1e-9) chosen = size;
  }
  return chosen;
}

/** The scene's regions, bounded by what the uniform tail can carry. */
export function sceneRefinementRegions(
  scene: SceneDescription,
): readonly FluidRefinementRegion[] {
  const authored = scene.fluid?.refinementRegions;
  if (!authored || authored.length === 0) return [];
  return authored.slice(0, OCTREE_REFINEMENT_REGION_CAPACITY);
}

/** An id no existing region uses, stable and readable in a saved document. */
export function nextRefinementRegionId(scene: SceneDescription): string {
  const taken = new Set(sceneRefinementRegions(scene).map((region) => region.id));
  for (let index = 1; ; index += 1) {
    const id = `region-${index}`;
    if (!taken.has(id)) return id;
  }
}

/** Per-axis world size of a finest cell, and where cell (0,0,0) starts. */
export interface RefinementRegionLattice {
  readonly dimensions: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  readonly origin_m: Vec3;
}

/**
 * The lattice a region's bounds are measured against.
 *
 * The same rounding the solver performs, through `scene-lattice.ts`, because a
 * region that spoke a different cell size than the octree would name boxes the
 * refinement gate does not agree with.
 */
export function refinementRegionLattice(scene: SceneDescription): RefinementRegionLattice {
  const dimensions = sceneLatticeDimensions(scene);
  const c = scene.container;
  return {
    dimensions,
    cellSize_m: sceneCellSizes_m(scene),
    origin_m: { x: -0.5 * c.width_m, y: 0, z: -0.5 * c.depth_m },
  };
}

/** A region's world box in cell coordinates, clamped to the lattice. */
export function refinementRegionCellBounds(
  region: FluidRefinementRegion,
  lattice: RefinementRegionLattice,
): { readonly min: readonly [number, number, number]; readonly max: readonly [number, number, number] } {
  const axes = ["x", "y", "z"] as const;
  const toCells = (point: Vec3, axis: 0 | 1 | 2) =>
    (point[axes[axis]!] - lattice.origin_m[axes[axis]!]) / lattice.cellSize_m[axis]!;
  const clamp = (value: number, axis: 0 | 1 | 2) =>
    Math.max(0, Math.min(lattice.dimensions[axis]!, value));
  return {
    min: [clamp(toCells(region.min_m, 0), 0), clamp(toCells(region.min_m, 1), 1), clamp(toCells(region.min_m, 2), 2)],
    max: [clamp(toCells(region.max_m, 0), 0), clamp(toCells(region.max_m, 1), 1), clamp(toCells(region.max_m, 2), 2)],
  };
}

/** Errors `validateScene` reports for the authored regions. */
export function validateRefinementRegions(
  regions: readonly FluidRefinementRegion[] | undefined,
  container: SceneDescription["container"],
): string[] {
  if (!regions) return [];
  const errors: string[] = [];
  if (!Array.isArray(regions)) return ["Fluid refinement regions must be an array"];
  if (regions.length > OCTREE_REFINEMENT_REGION_CAPACITY) {
    errors.push(`At most ${OCTREE_REFINEMENT_REGION_CAPACITY} fluid refinement regions are supported`);
  }
  const seen = new Set<string>();
  for (const [index, region] of regions.entries()) {
    const where = `Fluid refinement region ${region?.id ?? index}`;
    if (!region?.id?.trim()) errors.push(`${where} requires an id`);
    else if (seen.has(region.id)) errors.push(`${where} has a duplicate id`);
    else seen.add(region.id);
    if (region?.rule !== "minimum-cell-size") errors.push(`${where} has an unsupported rule`);
    const min = region?.min_m, max = region?.max_m;
    if (![min?.x, min?.y, min?.z, max?.x, max?.y, max?.z].every(Number.isFinite)) {
      errors.push(`${where} bounds must be finite`);
    } else if (!(min!.x < max!.x && min!.y < max!.y && min!.z < max!.z)) {
      errors.push(`${where} must have positive extent`);
    } else if (max!.x < -container.width_m / 2 || min!.x > container.width_m / 2
      || max!.y < 0 || min!.y > container.height_m
      || max!.z < -container.depth_m / 2 || min!.z > container.depth_m / 2) {
      errors.push(`${where} must overlap the container`);
    }
    if (!OCTREE_REFINEMENT_REGION_CELL_SIZES.includes(region?.minimumCellSize_cells)) {
      errors.push(`${where} minimum cell size must be one of ${OCTREE_REFINEMENT_REGION_CELL_SIZES.join(", ")} cells`);
    }
    if (region?.maximumCellSize_cells !== undefined
      && !OCTREE_REFINEMENT_REGION_CELL_SIZES.includes(region.maximumCellSize_cells)) {
      errors.push(`${where} maximum cell size must be one of ${OCTREE_REFINEMENT_REGION_CELL_SIZES.join(", ")} cells`);
    } else if (region?.maximumCellSize_cells !== undefined
      && region.maximumCellSize_cells < region.minimumCellSize_cells) {
      errors.push(`${where} maximum cell size must not be smaller than its minimum cell size`);
    }
  }
  return errors;
}
