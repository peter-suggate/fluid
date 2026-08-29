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
  WORLD_FRAME,
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
  clampRefinementRegionCellSize,
  DEFAULT_REFINEMENT_REGION_CELL_SIZE,
  nextRefinementRegionId,
  OCTREE_REFINEMENT_REGION_CAPACITY,
  OCTREE_REFINEMENT_REGION_CELL_SIZES,
  refinementRegionLattice,
  REFINEMENT_REGION_RULES,
  sceneRefinementRegions,
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
 * The one thing worth knowing: the resize snap is the region's own smallest
 * allowed cell, not always the finest cell. A dyadic leaf of edge S is aligned
 * to multiples of S in cell space, so a box on that lattice contains whole
 * leaves of that size and the region holds exactly the cells it covers.
 * Snapping to the finest cell instead would lose a shell of cells all the way
 * around every region, which reads as the lower bound not being respected.
 */

export const REFINEMENT_REGION_SELECTION_PREFIX = "refinement-region-";

export function refinementRegionSelectionId(regionId: string): string {
  return `${REFINEMENT_REGION_SELECTION_PREFIX}${regionId}`;
}

export function refinementRegionIdFromSelection(selectionId: string): string | undefined {
  return selectionId.startsWith(REFINEMENT_REGION_SELECTION_PREFIX)
    ? selectionId.slice(REFINEMENT_REGION_SELECTION_PREFIX.length)
    : undefined;
}

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
  const step = refinementRegionCellExtent_m(scene, clampRefinementRegionCellSize(region.minimumCellSize_cells));
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
  const limits = sceneContainerBox(scene);
  const step = refinementRegionCellExtent_m(scene, cells);
  const tolerance = 1e-6;
  const min = { ...box.min }, max = { ...box.max };
  (["x", "y", "z"] as const).forEach((axis, index) => {
    const size = step[index]!;
    const lo = Math.min(box.min[axis], box.max[axis]);
    const hi = Math.max(box.min[axis], box.max[axis]);
    const snapped = {
      min: limits.min[axis] + Math.floor((lo - limits.min[axis]) / size + tolerance) * size,
      max: limits.min[axis] + Math.ceil((hi - limits.min[axis]) / size - tolerance) * size,
    };
    min[axis] = Math.max(limits.min[axis], snapped.min);
    max[axis] = Math.min(limits.max[axis], Math.max(snapped.max, snapped.min + size));
    if (max[axis] - min[axis] < size) min[axis] = Math.max(limits.min[axis], max[axis] - size);
  });
  return { min, max };
}

/**
 * The region a rubber-band drag describes.
 *
 * The drag names a footprint on a horizontal plane, because that is the one
 * plane a single screen-space drag can resolve unambiguously against a
 * perspective camera. The height is seeded to the footprint's shorter side —
 * enough box to see and to grab — and is then an ordinary face drag like every
 * other extent in this editor.
 */
export function refinementRegionFromDrag(
  scene: SceneDescription,
  anchor_m: Vec3,
  drag_m: Vec3,
  options: {
    readonly id?: string;
    readonly minimumCellSize_cells?: number;
    readonly maximumCellSize_cells?: number;
  } = {},
): FluidRefinementRegion {
  const cells = clampRefinementRegionCellSize(
    options.minimumCellSize_cells ?? DEFAULT_REFINEMENT_REGION_CELL_SIZE);
  const maximumCells = options.maximumCellSize_cells === undefined ? undefined
    : Math.max(cells, clampRefinementRegionCellSize(options.maximumCellSize_cells));
  const limits = sceneContainerBox(scene);
  const footprint = {
    x: Math.abs(drag_m.x - anchor_m.x),
    z: Math.abs(drag_m.z - anchor_m.z),
  };
  const height = Math.max(refinementRegionCellExtent_m(scene, cells)[1]!,
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
  const box = snapRefinementRegionBox(scene, drawn, cells);
  return {
    id: options.id ?? nextRefinementRegionId(scene),
    rule: "minimum-cell-size",
    minimumCellSize_cells: cells,
    ...(maximumCells === undefined ? {} : { maximumCellSize_cells: maximumCells }),
    min_m: box.min,
    max_m: box.max,
  };
}

/** Whether another region can be drawn, or the uniform tail is already full. */
export function refinementRegionCapacityRemaining(scene: SceneDescription): number {
  return Math.max(0, OCTREE_REFINEMENT_REGION_CAPACITY - sceneRefinementRegions(scene).length);
}

// ---- the address bar ------------------------------------------------------

/**
 * Regions as a query value, in **percentages of the container**.
 *
 * Metres would make a link scene-specific, which is the opposite of what a
 * region is for: the whole point of the tool is to ask "what does it cost to
 * stop resolving *that part* of the domain", and that question travels — the
 * back third of a tank is the back third of any tank. A percentage is also the
 * form that survives the world-scale controls, which move every extent and the
 * cell size together and would otherwise strand every box.
 *
 * The cell bounds are NOT percentages. They are counts of finest cells, and
 * mean the same thing on any lattice. Scaling either with the container would
 * silently change the experiment.
 *
 * Format, chosen so the value survives `URLSearchParams` unescaped (the
 * urlencoded serializer keeps alphanumerics and `*-._`, and turns the obvious
 * `,` into `%2C`):
 *
 *     regions=<minX>_<minY>_<minZ>_<maxX>_<maxY>_<maxZ>_<minCells>[_<maxCells>][_<rule>]*<next>
 *
 * The optional largest cell follows the smallest one. The rule is omitted
 * while it is the only one. The reader also accepts the old seven-field form
 * and the old eight-field form whose last field is a rule.
 */
const REGION_FIELD_SEPARATOR = "_";
const REGION_RECORD_SEPARATOR = "*";
const DEFAULT_REGION_RULE: FluidRefinementRegion["rule"] = "minimum-cell-size";

/** Trim a percentage to the precision a sub-millimetre box actually needs. */
function percentText(value: number): string {
  return String(Number(value.toFixed(4)));
}

export function refinementRegionsToQuery(scene: SceneDescription): string {
  const limits = sceneContainerBox(scene);
  const span = boxSize(limits);
  const percent = (point: Vec3) => (["x", "y", "z"] as const)
    .map((axis) => percentText(span[axis] > 0 ? (100 * (point[axis] - limits.min[axis])) / span[axis] : 0));
  return sceneRefinementRegions(scene)
    .map((region) => [
      ...percent(region.min_m),
      ...percent(region.max_m),
      String(clampRefinementRegionCellSize(region.minimumCellSize_cells)),
      ...(region.maximumCellSize_cells === undefined ? []
        : [String(clampRefinementRegionCellSize(region.maximumCellSize_cells))]),
      ...(region.rule === DEFAULT_REGION_RULE ? [] : [region.rule]),
    ].join(REGION_FIELD_SEPARATOR))
    .join(REGION_RECORD_SEPARATOR);
}

/**
 * Read regions back against *this* scene's container.
 *
 * Re-snapped onto each region's own floor lattice rather than trusted: the
 * percentages land wherever the arithmetic puts them, and a box a fraction of a
 * cell off the lattice loses a shell of cells to partial containment — the
 * floor would quietly stop being the floor. On the scene the link was written
 * from this is the identity; on any other it is what makes the box mean the
 * same instruction there.
 *
 * Ids are regenerated by position rather than carried. They exist to address a
 * region while editing it, and spending URL characters on `region-1` to
 * reconstruct `region-1` is a cost with no reader.
 *
 * Malformed records are dropped individually. A link is external input, and one
 * bad record must not cost the reader the rest of the boxes.
 */
export function refinementRegionsFromQuery(
  scene: SceneDescription,
  raw: string,
): FluidRefinementRegion[] {
  const limits = sceneContainerBox(scene);
  const span = boxSize(limits);
  const regions: FluidRefinementRegion[] = [];
  for (const record of raw.split(REGION_RECORD_SEPARATOR)) {
    if (regions.length >= OCTREE_REFINEMENT_REGION_CAPACITY) break;
    const fields = record.split(REGION_FIELD_SEPARATOR);
    if (fields.length < 7 || fields.length > 9) continue;
    const numbers = fields.slice(0, 7).map(Number);
    if (!numbers.every(Number.isFinite)) continue;
    const eighth = fields[7];
    const eighthNumber = eighth === undefined || eighth === "" ? undefined : Number(eighth);
    const hasMaximum = eighthNumber !== undefined && Number.isFinite(eighthNumber);
    const rule = hasMaximum ? (fields[8] ?? DEFAULT_REGION_RULE)
      : (eighth ?? DEFAULT_REGION_RULE);
    if (rule !== DEFAULT_REGION_RULE) continue;
    if (fields.length === 9 && !hasMaximum) continue;
    const metres = (offset: number): Vec3 => ({
      x: limits.min.x + (Math.max(0, Math.min(100, numbers[offset]!)) / 100) * span.x,
      y: limits.min.y + (Math.max(0, Math.min(100, numbers[offset + 1]!)) / 100) * span.y,
      z: limits.min.z + (Math.max(0, Math.min(100, numbers[offset + 2]!)) / 100) * span.z,
    });
    const cells = clampRefinementRegionCellSize(numbers[6]!);
    const maximumCells = hasMaximum
      ? Math.max(cells, clampRefinementRegionCellSize(eighthNumber))
      : undefined;
    const box = snapRefinementRegionBox(scene, { min: metres(0), max: metres(3) }, cells);
    if (!(box.max.x > box.min.x) || !(box.max.y > box.min.y) || !(box.max.z > box.min.z)) continue;
    regions.push({
      id: `region-${regions.length + 1}`,
      rule: DEFAULT_REGION_RULE,
      minimumCellSize_cells: cells,
      ...(maximumCells === undefined ? {} : { maximumCellSize_cells: maximumCells }),
      min_m: box.min,
      max_m: box.max,
    });
  }
  return regions;
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

function refinementRegionChoices(
  scene: SceneDescription,
  region: FluidRefinementRegion,
): EditorChoiceGroup[] {
  const write = (next: Partial<FluidRefinementRegion>) =>
    withRefinementRegion(scene, region.id, { ...region, ...next });
  const cellSize_m = refinementRegionLattice(scene).cellSize_m;
  const minimumCells = clampRefinementRegionCellSize(region.minimumCellSize_cells);
  const maximumCells = region.maximumCellSize_cells === undefined ? undefined
    : clampRefinementRegionCellSize(region.maximumCellSize_cells);
  return [
    {
      id: "rule",
      label: "This box means",
      // Short tags throughout, because a region's column is the narrow one and
      // "THIS BOX MEANS" is a sentence where the reader needs a label. The full
      // phrasing is still on each row's tip, which is where a first reader
      // meets it.
      tag: "Means",
      value: region.rule,
      options: REFINEMENT_REGION_RULES.map((rule) => ({
        id: rule.id,
        label: rule.label,
        hint: rule.hint,
        enabled: true,
        apply: () => write({ rule: rule.id }),
      })),
    },
    {
      id: "minimumCellSize",
      label: "Smallest cell",
      tag: "Min",
      value: String(minimumCells),
      options: OCTREE_REFINEMENT_REGION_CELL_SIZES.map((cells) => ({
        id: String(cells),
        label: `${cells}³`,
        hint: `${cells}³ finest cells · ${(cells * cellSize_m[0]! * 1000).toFixed(0)} mm edge`,
        enabled: true,
        // Re-snap onto the new lattice: the box was aligned to the old floor,
        // and an unaligned box loses a shell of cells to partial containment.
        apply: () => {
          const box = snapRefinementRegionBox(scene, refinementRegionBox(region), cells);
          return write({
            minimumCellSize_cells: cells,
            ...(maximumCells !== undefined && maximumCells < cells
              ? { maximumCellSize_cells: cells } : {}),
            min_m: box.min,
            max_m: box.max,
          });
        },
      })),
    },
    {
      id: "maximumCellSize",
      label: "Largest cell",
      tag: "Max",
      value: maximumCells === undefined ? "auto" : String(maximumCells),
      options: [
        {
          id: "auto",
          label: "AUTO",
          hint: "Evidence decides how far quiet fluid may coarsen",
          enabled: true,
          apply: () => write({ maximumCellSize_cells: undefined }),
        },
        ...OCTREE_REFINEMENT_REGION_CELL_SIZES.map((cells) => ({
          id: String(cells),
          label: `${cells}³`,
          hint: `No cell larger than ${cells}³ finest cells · ${(cells * cellSize_m[0]! * 1000).toFixed(0)} mm edge`,
          enabled: true,
          apply: () => {
            // A ceiling below the current floor means the user wants the whole
            // interval to move down. Keep the bounds valid and re-snap to the
            // newly selected smallest cell in the same edit.
            const nextMinimum = Math.min(minimumCells, cells);
            const box = snapRefinementRegionBox(
              scene, refinementRegionBox(region), nextMinimum);
            return write({
              minimumCellSize_cells: nextMinimum,
              maximumCellSize_cells: cells,
              min_m: box.min,
              max_m: box.max,
            });
          },
        })),
      ],
    },
  ];
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
  const cellSize_m = refinementRegionLattice(scene).cellSize_m;
  const write = (next: Partial<FluidRefinementRegion>) =>
    withRefinementRegion(scene, region.id, { ...region, ...next });
  const move = (centre_m: Vec3) => {
    const moved = moveBoxWithinLimits(box, centre_m, sceneContainerBox(scene));
    const snapped = snapRefinementRegionBox(scene, moved, cells);
    return write({ min_m: snapped.min, max_m: snapped.max });
  };
  return {
    selection: { kind: "refinement-region", id: refinementRegionSelectionId(region.id) },
    label: region.id.toUpperCase(),
    tone: "region",
    frame: WORLD_FRAME,
    box,
    sizeLabel: `${[size.x, size.y, size.z].map((value) => value.toFixed(2)).join(" × ")} m · ${maximumCells === undefined
      ? `≥ ${cells}³ cells`
      : cells === maximumCells ? `${cells}³ cells` : `${cells}³–${maximumCells}³ cells`}`,
    handles: [
      ...boxHandles(box, {
        drag: boxResizeDrag(box, refinementRegionResizePolicy(scene, region),
          (next) => write({ min_m: next.min, max_m: next.max })),
      }),
      ...moveHandles(boxCenter(box), move),
    ],
    draftSubject: "refinement-region",
    editLabel: (handle) => handle.space === "world"
      ? `Moved ${region.id}` : `Resized ${region.id}`,
    choices: refinementRegionChoices(scene, region),
    fields: positionFields(boxCenter(box), move),
    summary: maximumCells === undefined
      ? `No pressure cell smaller than ${(cells * cellSize_m[0]! * 1000).toFixed(0)} mm inside this box. Grading still splits leaves on its boundary.`
      : cells === maximumCells
        ? `Fully contained pressure cells are held at ${(cells * cellSize_m[0]! * 1000).toFixed(0)} mm inside this box.`
        : `Fully contained pressure cells stay between ${(cells * cellSize_m[0]! * 1000).toFixed(0)} and ${(maximumCells * cellSize_m[0]! * 1000).toFixed(0)} mm inside this box.`,
    remove: () => withRefinementRegion(scene, region.id, undefined),
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
