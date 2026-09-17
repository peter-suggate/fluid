import {
  boxCenter,
  boxHandles,
  boxResizeDrag,
  boxSize,
  moveBoxWithinLimits,
  moveHandles,
  pickSolidBox,
  positionFields,
  sceneContainerBox,
  pickExcluded,
  type BoxExtent,
  type BoxResizePolicy,
  type EditorChoiceGroup,
  type EditorEntity,
  type EditorEntityContext,
  type EditorEntityDefinition,
} from "./editor-entity";
import type { FluidRefinementRegion, SceneDescription, Vec3 } from "./model";
import {
  regionChoices,
  regionDraftCellSize,
  regionDrawIsDegenerate,
  regionEntity,
  regionFromDraw,
  regionSnapStep_cells,
  snapRegionBox,
} from "../features/refinement-region/policy";
import {
  refinementRegionIdFromSelection,
  refinementRegionSelectionId,
  type RefinementRegionRecord,
  type RegionSpace,
} from "../features/refinement-region/definition";
import {
  regionsFromQuery,
  regionsToQuery,
} from "../features/refinement-region/persistence";
import { BRICK_FINE_CELLS } from "./sparse-brick-geometry";
import {
  clampRefinementRegionCellSize,
  DEFAULT_REFINEMENT_REGION_CELL_SIZE,
  DEFAULT_REGION_DRAFT,
  nextRefinementRegionId,
  OCTREE_REFINEMENT_REGION_CAPACITY,
  OCTREE_REFINEMENT_REGION_CELL_SIZES,
  refinementRegionLattice,
  sceneRefinementRegions,
  type RegionDraft,
} from "./refinement-regions";

/**
 * Editing a refinement region.
 *
 * A region is a box, and this file deliberately adds nothing to what a box
 * already is in this editor: `boxHandles`, `boxResizeDrag`, `moveHandles` and
 * `positionFields` do the manipulation, exactly as they do for the water body
 * and for scenery. What is specific to a region is only the *policy* — the
 * lattice its sides land on, and the meaning it carries — and that is all this
 * file contains.
 *
 * The one thing worth knowing: the resize snap is **the brick**, or the
 * region's own smallest allowed cell where that is coarser — see
 * `regionSnapStep_cells`. A dyadic leaf of edge S is aligned to multiples of S
 * in cell space, so a box on that lattice contains whole leaves of that size
 * and the region holds exactly the cells it covers; and the solver binds a
 * region brick by brick, so an edge inside a brick is a distinction it cannot
 * honour. With the studio default `MIN = 8` the two are the same number, so
 * nothing about a default region moved.
 *
 * Since the region package landed, this file is an **adapter**. Every rule —
 * the outward snap, the two ladders and how they follow each other, the rows a
 * region offers — lives in `lib/features/refinement-region` in finest cells and
 * N dimensions. What is here is the three things that are genuinely the
 * studio's: metres and `Vec3`, the scene document, and the address bar.
 */

/*
 * The selection id is the package's, not this file's.
 *
 * It was restated by hand in `advance-lab/slice-regions.ts` and pinned against
 * this constant by a test, which is the clearest single instance of the problem
 * the package exists to end. Re-exported here so every studio call site keeps
 * its import.
 */
export {
  refinementRegionIdFromSelection,
  refinementRegionSelectionId,
  REFINEMENT_REGION_SELECTION_PREFIX,
} from "../features/refinement-region/definition";

export function refinementRegionBox(region: FluidRefinementRegion): BoxExtent {
  return { min: region.min_m, max: region.max_m };
}

/** World size of one region-floor cell on each axis. */
export function refinementRegionCellExtent_m(
  scene: SceneDescription,
  cells: number,
): readonly [number, number, number] {
  const { cellSize_m } = refinementRegionLattice(scene);
  return [cellSize_m[0]! * cells, cellSize_m[1]! * cells, cellSize_m[2]! * cells];
}

/**
 * How a region reshapes: onto its own floor lattice, inside the container,
 * never thinner than one of the cells it is asking for.
 */
export function refinementRegionResizePolicy(
  scene: SceneDescription,
  region: FluidRefinementRegion,
): BoxResizePolicy {
  const step = refinementRegionCellExtent_m(scene,
    regionSnapStep_cells(region, BRICK_FINE_CELLS));
  return {
    snap_m: [step[0]!, step[1]!, step[2]!],
    limits: sceneContainerBox(scene),
    minimum_m: [step[0]!, step[1]!, step[2]!],
  };
}

/** The scene with one region replaced, added, or — with `undefined` — dropped. */
export function withRefinementRegion(
  scene: SceneDescription,
  id: string,
  next: FluidRefinementRegion | undefined,
): SceneDescription {
  const current = sceneRefinementRegions(scene);
  const replaced = current.some((region) => region.id === id);
  const regions = replaced
    ? current.flatMap((region) => region.id !== id ? [region] : next ? [next] : [])
    : next ? [...current, next] : [...current];
  const { refinementRegions: _dropped, ...fluid } = scene.fluid;
  // Dropped rather than written as `[]`: an empty list and an absent field mean
  // the same thing, and only one of them leaves an untouched document byte-
  // identical to how it was authored.
  return {
    ...scene,
    fluid: regions.length === 0 ? fluid : { ...fluid, refinementRegions: regions },
  };
}

/**
 * Snap a drawn box outward onto a cell lattice, so the region covers every cell
 * the drag touched rather than the largest lattice box inside it.
 *
 * Outward and not nearest: a region is an instruction about an area the user
 * indicated, and rounding a 1.4-cell drag down to one cell would produce a box
 * visibly smaller than the rectangle they let go of.
 *
 * The container wins over the lattice. A domain whose cell count is not a
 * multiple of the floor has no lattice line on its far wall, so a box grown out
 * to that wall stops at the wall rather than at the last aligned line inside it.
 * That is the right way round: the leaves are aligned to the *domain*, so a
 * region reaching the wall still contains every aligned leaf that fits, while
 * pulling back to the previous line would drop a row the user asked for.
 *
 * Idempotent within a tolerance, which the outward rounding is not for free: a
 * box already on the lattice arrives here as 3.0000000001 steps after any
 * round-trip through metres — a URL, a saved document, a container resize — and
 * a bare `ceil` would grow it by a whole cell each time.
 */
export function snapRefinementRegionBox(
  scene: SceneDescription,
  box: BoxExtent,
  cells: number,
): BoxExtent {
  // `cells` names the region's floor, and the *step* is the brick or that floor,
  // whichever is coarser. Callers keep passing the floor because that is what a
  // region carries; deciding what it means is the package's job.
  const snapped = snapRegionBox(
    boxToCells(scene, box.min), boxToCells(scene, box.max),
    regionSnapStep_cells({ minimumCellSize_cells: cells }, BRICK_FINE_CELLS),
    { dimensions: refinementRegionLattice(scene).dimensions });
  return { min: cellsToMetres(scene, snapped.min), max: cellsToMetres(scene, snapped.max) };
}

// ---- the metre / Vec3 adapter --------------------------------------------

const AXES = ["x", "y", "z"] as const;

/** A world point as finest cells from the container corner. */
function boxToCells(scene: SceneDescription, point: Vec3): number[] {
  const { cellSize_m, origin_m } = refinementRegionLattice(scene);
  return AXES.map((axis, index) => (point[axis] - origin_m[axis]) / cellSize_m[index]!);
}

/** The same point back in metres. Exactly inverse, so a round trip is identity. */
function cellsToMetres(scene: SceneDescription, cells: readonly number[]): Vec3 {
  const { cellSize_m, origin_m } = refinementRegionLattice(scene);
  return {
    x: origin_m.x + (cells[0] ?? 0) * cellSize_m[0]!,
    y: origin_m.y + (cells[1] ?? 0) * cellSize_m[1]!,
    z: origin_m.z + (cells[2] ?? 0) * cellSize_m[2]!,
  };
}

/** The document's region as the shared record: cells, and dyadic bounds. */
export function refinementRegionRecord(
  scene: SceneDescription,
  region: FluidRefinementRegion,
): RefinementRegionRecord {
  const floor = clampRefinementRegionCellSize(region.minimumCellSize_cells);
  const ceiling = region.maximumCellSize_cells === undefined ? undefined
    : Math.max(floor, clampRefinementRegionCellSize(region.maximumCellSize_cells));
  return {
    id: region.id,
    rule: region.rule,
    minimumCellSize_cells: floor,
    ...(ceiling === undefined ? {} : { maximumCellSize_cells: ceiling }),
    min_cells: boxToCells(scene, region.min_m),
    max_cells: boxToCells(scene, region.max_m),
  };
}

/** The shared record back as the document carries it. */
export function refinementRegionFromRecord(
  scene: SceneDescription,
  record: RefinementRegionRecord,
): FluidRefinementRegion {
  return {
    id: record.id,
    rule: record.rule,
    minimumCellSize_cells: record.minimumCellSize_cells,
    ...(record.maximumCellSize_cells === undefined ? {}
      : { maximumCellSize_cells: record.maximumCellSize_cells }),
    min_m: cellsToMetres(scene, record.min_cells),
    max_m: cellsToMetres(scene, record.max_cells),
  };
}

/**
 * The studio as a `RegionSpace`: three axes, metres behind them.
 *
 * `write` hands back a whole scene, which is a legal `Partial<SceneDescription>`
 * and is what removal needs — dropping the last region drops the
 * `refinementRegions` key, and a merge patch cannot express an absence.
 */
export const studioRegionSpace: RegionSpace<SceneDescription, Partial<SceneDescription>> = {
  axes: 3,
  lattice: (scene) => ({ dimensions: refinementRegionLattice(scene).dimensions }),
  list: (scene) => sceneRefinementRegions(scene)
    .map((region) => refinementRegionRecord(scene, region)),
  write: (scene, id, next) => withRefinementRegion(scene, id,
    next && refinementRegionFromRecord(scene, next)),
  nextId: (scene) => nextRefinementRegionId(scene),
  capacity: OCTREE_REFINEMENT_REGION_CAPACITY,
  cellSizes: OCTREE_REFINEMENT_REGION_CELL_SIZES,
  defaultCellSize_cells: DEFAULT_REFINEMENT_REGION_CELL_SIZE,
  brick_cells: BRICK_FINE_CELLS,
  cellEdge_mm: (scene) => refinementRegionLattice(scene).cellSize_m[0]! * 1000,
};

/**
 * The region a rubber-band drag describes.
 *
 * The drag names a footprint on a horizontal plane, because that is the one
 * plane a single screen-space drag can resolve unambiguously against a
 * perspective camera. The height is seeded to the footprint's shorter side —
 * enough box to see and to grab — and is then an ordinary face drag like every
 * other extent in this editor.
 *
 * `undefined` for a press that has not travelled. The test is on the *drag*,
 * before the height is seeded, because the seed is what would hide it: a box a
 * step tall over a zero footprint is not a box anybody drew, and
 * `regionFromDraw` — reading the seeded corners — could no longer tell.
 */
export function refinementRegionFromDrag(
  scene: SceneDescription,
  anchor_m: Vec3,
  drag_m: Vec3,
  options: {
    readonly id?: string;
    /**
     * What the reader chose for the next box, from the shared ui-store draft.
     *
     * Absent means nobody has chosen, and the studio's own default rung stands
     * — which is the behaviour this had before there was anywhere to state a
     * choice. The explicit bounds below still win, because the URL reader and
     * the tests describe a *particular* region rather than the pending one.
     */
    readonly draft?: RegionDraft;
    readonly minimumCellSize_cells?: number;
    readonly maximumCellSize_cells?: number;
  } = {},
): FluidRefinementRegion | undefined {
  // A click, not a drag: the horizontal plane is where this gesture names its
  // area, so that is where the question is asked. See `regionDrawIsDegenerate`.
  if (regionDrawIsDegenerate([anchor_m.x, anchor_m.z], [drag_m.x, drag_m.z])) return undefined;
  const draft = options.draft ?? DEFAULT_REGION_DRAFT;
  const cells = clampRefinementRegionCellSize(options.minimumCellSize_cells
    ?? regionDraftCellSize(studioRegionSpace, draft));
  const maximumCells = options.maximumCellSize_cells !== undefined
    ? Math.max(cells, clampRefinementRegionCellSize(options.maximumCellSize_cells))
    : draft.holdAtOneTier ? cells : undefined;
  const limits = sceneContainerBox(scene);
  const footprint = {
    x: Math.abs(drag_m.x - anchor_m.x),
    z: Math.abs(drag_m.z - anchor_m.z),
  };
  // At least one *step* tall, which is the brick unless the floor is coarser.
  // Seeding a thinner box only to have the snap grow it back was the old
  // behaviour by accident; naming the step here says what the minimum is.
  const height = Math.max(
    refinementRegionCellExtent_m(scene,
      regionSnapStep_cells({ minimumCellSize_cells: cells }, BRICK_FINE_CELLS))[1]!,
    Math.min(footprint.x, footprint.z, limits.max.y - anchor_m.y));
  const drawn: BoxExtent = {
    min: {
      x: Math.min(anchor_m.x, drag_m.x),
      y: anchor_m.y,
      z: Math.min(anchor_m.z, drag_m.z),
    },
    max: {
      x: Math.max(anchor_m.x, drag_m.x),
      y: anchor_m.y + height,
      z: Math.max(anchor_m.z, drag_m.z),
    },
  };
  // The snap, the id and the record itself are the package's: `regionFromDraw`
  // is what the lab's release calls too, so a box drawn in either host lands on
  // the same lattice by the same arithmetic rather than by two transcriptions
  // of it. Only the metres on either side of this call are the studio's.
  const record = regionFromDraw(studioRegionSpace, scene,
    boxToCells(scene, drawn.min), boxToCells(scene, drawn.max),
    { cellSize_cells: cells, rule: draft.rule, holdAtOneTier: maximumCells !== undefined },
    { id: options.id });
  // Unreachable for a travelled drag — the footprint above is non-empty on at
  // least one axis, so the corners handed over differ — and stated rather than
  // asserted, because the package owns the rule and this adapter only relays it.
  if (record === undefined) return undefined;
  return refinementRegionFromRecord(scene, maximumCells === undefined ? record
    : { ...record, maximumCellSize_cells: maximumCells });
}

/** Whether another region can be drawn, or the uniform tail is already full. */
export function refinementRegionCapacityRemaining(scene: SceneDescription): number {
  return Math.max(0, OCTREE_REFINEMENT_REGION_CAPACITY - sceneRefinementRegions(scene).length);
}

// ---- the address bar ------------------------------------------------------

/**
 * Regions as a query value — the *encoding* is the package's.
 *
 * `lib/features/refinement-region/persistence.ts` states the whole wire form
 * once, in percentages of the lattice and N axes, so the 2-D lab writes and
 * reads the same `regions=` value against its own two-axis lattice. This pair is
 * the metre adapter over it, and nothing else: the percentage is of the
 * container either way, because `refinementRegionLattice`'s origin *is* the
 * container's minimum corner and each axis' cell size is its extent over its
 * cell count.
 */
export function refinementRegionsToQuery(scene: SceneDescription): string {
  return regionsToQuery(studioRegionSpace, scene);
}

/**
 * Read regions back against *this* scene's container, in metres.
 *
 * Every rule the reader applies — the clamp to 0..100, the re-snap onto the
 * region's own step, the regenerated ids, the dropped malformed record, the
 * capacity break — is `regionsFromQuery`'s, stated once in finest cells for
 * both hosts. What is left here is the two things that are genuinely the
 * studio's: `RefinementRegionRecord` becoming a `FluidRefinementRegion`, and the
 * cells becoming metres against this scene's lattice.
 */
export function refinementRegionsFromQuery(
  scene: SceneDescription,
  raw: string,
): FluidRefinementRegion[] {
  return regionsFromQuery(studioRegionSpace, scene, raw)
    .map((record) => refinementRegionFromRecord(scene, record));
}

/** The scene carrying exactly the regions a query value describes. */
export function withRefinementRegionsFromQuery(
  scene: SceneDescription,
  raw: string,
): SceneDescription {
  const regions = refinementRegionsFromQuery(scene, raw);
  const { refinementRegions: _dropped, ...fluid } = scene.fluid;
  return {
    ...scene,
    fluid: regions.length === 0 ? fluid : { ...fluid, refinementRegions: regions },
  };
}

// ---- entity ---------------------------------------------------------------

/**
 * The region's three enumerations, composed by the shared package.
 *
 * `regionChoices` is the whole of what used to stand here — the rule group, the
 * two dyadic ladders and the re-snap each one forces — with the metres lifted
 * out into `studioRegionSpace`. Deleting the copy is the point of the exercise:
 * the lab renders these same three groups through `EntityOptionRows` and cannot
 * drift from them, because there is nothing left to drift from.
 */
export function refinementRegionChoices(
  scene: SceneDescription,
  region: FluidRefinementRegion,
): EditorChoiceGroup[] {
  return regionChoices(studioRegionSpace, scene, refinementRegionRecord(scene, region));
}

function refinementRegionEntityFor(
  context: EditorEntityContext,
  region: FluidRefinementRegion,
): EditorEntity {
  const { scene } = context;
  const box = refinementRegionBox(region);
  const size = boxSize(box);
  const cells = clampRefinementRegionCellSize(region.minimumCellSize_cells);
  const maximumCells = region.maximumCellSize_cells === undefined ? undefined
    : Math.max(cells, clampRefinementRegionCellSize(region.maximumCellSize_cells));
  const record = refinementRegionRecord(scene, region);
  const write = (next: Partial<FluidRefinementRegion>) =>
    withRefinementRegion(scene, region.id, { ...region, ...next });
  const move = (centre_m: Vec3) => {
    const moved = moveBoxWithinLimits(box, centre_m, sceneContainerBox(scene));
    const snapped = snapRefinementRegionBox(scene, moved, cells);
    return write({ min_m: snapped.min, max_m: snapped.max });
  };
  // The rows, the label, the tone, the summary and the removal are the shared
  // entity — the *same* object the lab renders its strip from. What the studio
  // adds on top is the gizmo half: a world box, its eight handles, the move,
  // and the position fields behind them, none of which the lab's SVG rectangles
  // in canvas cells could use. Spread rather than duplicated, so a row added to
  // the region appears in both hosts without this file changing.
  return {
    ...regionEntity(studioRegionSpace, scene, record),
    box,
    sizeLabel: `${[size.x, size.y, size.z].map((value) => value.toFixed(2)).join(" \u00d7 ")} m \u00b7 ${maximumCells === undefined
      ? `\u2265 ${cells}\u00b3 cells`
      : cells === maximumCells ? `${cells}\u00b3 cells` : `${cells}\u00b3\u2013${maximumCells}\u00b3 cells`}`,
    handles: [
      ...boxHandles(box, {
        drag: boxResizeDrag(box, refinementRegionResizePolicy(scene, region),
          (next) => write({ min_m: next.min, max_m: next.max })),
      }),
      ...moveHandles(boxCenter(box), move),
    ],
    fields: positionFields(boxCenter(box), move),
  };
}

/**
 * Regions are surfaced by their own tool as well as by SELECT.
 *
 * They have no rendered surface of their own — nothing in the frame is a
 * region — so the tool that draws them is also the mode in which the ones
 * already drawn are visible and grabbable. Under SELECT they behave like any
 * other box: click the wireframe to pick it.
 */
export const refinementRegionEntity: EditorEntityDefinition = {
  kind: "refinement-region",
  instances: (context) => sceneRefinementRegions(context.scene)
    .map((region) => refinementRegionEntityFor(context, region)),
  find: (context, id) => {
    const regionId = refinementRegionIdFromSelection(id);
    const region = regionId === undefined
      ? undefined
      : sceneRefinementRegions(context.scene).find((candidate) => candidate.id === regionId);
    return region && refinementRegionEntityFor(context, region);
  },
  pick: (context, ray, exclude) => {
    let nearest: { id: string; distance_m: number } | undefined;
    for (const region of sceneRefinementRegions(context.scene)) {
      if (pickExcluded(exclude, "refinement-region", refinementRegionSelectionId(region.id))) continue;
      const distance_m = pickSolidBox(ray, refinementRegionBox(region));
      if (distance_m !== undefined && (!nearest || distance_m < nearest.distance_m)) {
        nearest = { id: region.id, distance_m };
      }
    }
    return nearest && {
      selection: { kind: "refinement-region", id: refinementRegionSelectionId(nearest.id) },
      distance_m: nearest.distance_m,
    };
  },
};
