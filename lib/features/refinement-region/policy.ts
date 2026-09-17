import { WORLD_FRAME, type EditorChoiceGroup, type EditorEntity } from "../../core/editor-entity";
import {
  REFINEMENT_REGION_RULES,
  type RegionDraft,
} from "../../core/refinement-regions";
import {
  refinementRegionSelectionId,
  type RefinementRegionRecord,
  type RegionBox,
  type RegionLattice,
  type RegionSpace,
} from "./definition";

/**
 * Everything a refinement box does, in finest cells and in N dimensions.
 *
 * No metres, no `SceneDescription`, no canvas y-flip: those are the two
 * adapters' business (`lib/core/editor-refinement-region.ts` and the lab's own).
 * What is here is the part that was mirrored by hand — the snap, the resize,
 * the move, the two cell-size ladders and how they follow each other — written
 * once.
 *
 * The one rule worth reading before the code is `regionSnapStep_cells`.
 */

/** A box already on the lattice arrives back as 3.0000000001 after a metres round trip. */
const TOLERANCE = 1e-6;

/**
 * The largest power of two at or below `value`, floored at one cell.
 *
 * The ladder itself belongs to the host (`RegionSpace.cellSizes`), but the
 * *shape* of it does not: a leaf edge is a power of two in both worlds, because
 * an octree has no cell of edge 3. Rounding here rather than against a host's
 * list is what lets the snap rule be stated without a `RegionSpace`.
 */
function dyadicFloor(value: number): number {
  if (!Number.isFinite(value) || value < 1) return 1;
  return 2 ** Math.floor(Math.log2(value) + TOLERANCE);
}

/**
 * The lattice a region's sides land on: **the brick**, not the region's own
 * floor cell.
 *
 * The old step was the region's smallest allowed cell, so a region asking for
 * `MIN = 1` snapped to one finest cell — which is why resizing a box in 3-D
 * felt smooth rather than deliberate. The solver does not bind at that
 * granularity. `cellSizeBoundsForBrick`
 * (`lib/methods/adaptive-volume/sparse-cm12-refinement-regions.ts`) raises the
 * **floor** for every brick the region *intersects* and lowers the **ceiling**
 * only for bricks it fully *contains*; `region_bounds` in
 * `rust/crates/fluid-core/src/resolution.rs` runs the same test with `B = 8`.
 * So an edge inside a brick over-applies the floor and does nothing at all for
 * the ceiling — it is not a distinction the solver can honour.
 *
 * `max` of the two, because both constraints are powers of two: a floor coarser
 * than the brick still needs its own coarser lattice, and one finer than the
 * brick has nothing finer than the brick to land on. With the studio default
 * `MIN = 8` the step is unchanged, so the shipped 3-D behaviour moves only for
 * `MIN` of 1, 2 or 4 — which is the case Peter asked to fix.
 */
export function regionSnapStep_cells(
  record: Pick<RefinementRegionRecord, "minimumCellSize_cells">,
  brick_cells: number,
): number {
  return Math.max(dyadicFloor(brick_cells), dyadicFloor(record.minimumCellSize_cells));
}

/**
 * A drawn box snapped **outward** onto the step, inside the lattice.
 *
 * Outward and not nearest: a region is an instruction about an area the reader
 * indicated, and rounding a 1.4-step drag down would hand back a box visibly
 * smaller than the rectangle they let go of.
 *
 * The lattice wins over the step at the walls. A domain whose cell count is not
 * a multiple of the step has no lattice line on its far wall, so a box grown
 * out to that wall stops at the wall rather than at the last aligned line
 * inside it — the leaves are aligned to the *domain*, so a region reaching the
 * wall still contains every aligned leaf that fits, while pulling back would
 * drop a row the reader asked for.
 *
 * Idempotent within `TOLERANCE`, which outward rounding is not for free: a bare
 * `ceil` would grow an already-aligned box by a whole step on every round trip
 * through a document, a URL or a container resize.
 */
export function snapRegionBox(
  min: readonly number[],
  max: readonly number[],
  step: number,
  lattice: RegionLattice,
): { min: number[]; max: number[] } {
  const outMin: number[] = [];
  const outMax: number[] = [];
  for (let axis = 0; axis < Math.max(min.length, max.length); axis += 1) {
    const limit = lattice.dimensions[axis] ?? Infinity;
    const lo = Math.min(min[axis] ?? 0, max[axis] ?? 0);
    const hi = Math.max(min[axis] ?? 0, max[axis] ?? 0);
    const snappedLow = Math.floor(lo / step + TOLERANCE) * step;
    const snappedHigh = Math.ceil(hi / step - TOLERANCE) * step;
    let low = Math.max(0, snappedLow);
    const high = Math.min(limit, Math.max(snappedHigh, snappedLow + step));
    if (high - low < step) low = Math.max(0, high - step);
    outMin.push(low);
    outMax.push(high);
  }
  return { min: outMin, max: outMax };
}

/**
 * The box after one side has been dragged to `to`.
 *
 * The opposite side is held: a resize moves the edge that was taken hold of and
 * nothing else, which is what makes a corner drag two of these rather than a
 * scale about the centre. The moved edge snaps to the step, stays inside the
 * lattice, and is never allowed past the held one — one step of thickness is
 * the floor, because a region thinner than the cell it is asking for contains
 * none of them.
 */
export function resizeRegionBox(
  box: RegionBox,
  side: { readonly axis: number; readonly end: "min" | "max" },
  to: number,
  step: number,
  lattice: RegionLattice,
): { min: number[]; max: number[] } {
  const min = [...box.min];
  const max = [...box.max];
  const axis = side.axis;
  const limit = lattice.dimensions[axis] ?? Infinity;
  const snapped = Math.max(0, Math.min(limit, Math.round(to / step) * step));
  if (side.end === "min") min[axis] = Math.min(snapped, (max[axis] ?? 0) - step);
  else max[axis] = Math.max(snapped, (min[axis] ?? 0) + step);
  return { min, max };
}

/**
 * The box after its body has been dragged by `delta` cells.
 *
 * Translated, never reshaped: the travel is snapped to the step and the whole
 * box is then pushed back inside the lattice, so a box dragged into a wall
 * stops against it at its own size instead of being squashed against it.
 */
export function moveRegionBox(
  box: RegionBox,
  delta: readonly number[],
  step: number,
  lattice: RegionLattice,
): { min: number[]; max: number[] } {
  const min: number[] = [];
  const max: number[] = [];
  for (let axis = 0; axis < box.min.length; axis += 1) {
    const size = (box.max[axis] ?? 0) - (box.min[axis] ?? 0);
    const limit = lattice.dimensions[axis] ?? Infinity;
    const room = Math.max(0, limit - size);
    const moved = Math.round(((box.min[axis] ?? 0) + (delta[axis] ?? 0)) / step) * step;
    const low = Math.max(0, Math.min(room, moved));
    min.push(low);
    max.push(low + size);
  }
  return { min, max };
}

/**
 * The region with a new floor cell, and the ceiling that follows it.
 *
 * A ceiling equal to the floor is a region *held at one tier*, and it follows
 * the floor wherever the floor goes — up or down. A wider authored ceiling is
 * kept, and only ever lifted to stay above the floor it now has to be above.
 * That is the rule the lab stated and the studio only half implemented (it
 * lifted, but never followed a floor downwards); one implementation is the
 * point of this package.
 *
 * `lattice` is optional and, when given, re-snaps the box: changing the floor
 * can change `regionSnapStep_cells`, and a box left on the old lattice loses a
 * shell of cells to partial containment — the floor would quietly stop being
 * the floor.
 */
export function regionWithFloor(
  record: RefinementRegionRecord,
  cells: number,
  lattice?: RegionLattice,
  brick_cells?: number,
): RefinementRegionRecord {
  const ceiling = record.maximumCellSize_cells;
  const next: RefinementRegionRecord = {
    ...record,
    minimumCellSize_cells: cells,
    ...(ceiling === undefined ? {} : {
      maximumCellSize_cells: ceiling === record.minimumCellSize_cells
        ? cells : Math.max(cells, ceiling),
    }),
  };
  return lattice === undefined ? next : resnapped(next, lattice, brick_cells ?? cells);
}

/** The region with a ceiling, or with none — AUTO is the *absence* of one. */
export function regionWithCeiling(
  record: RefinementRegionRecord,
  cells: number | undefined,
  lattice?: RegionLattice,
  brick_cells?: number,
): RefinementRegionRecord {
  if (cells === undefined) {
    // Deleted rather than set to `undefined`: a consumer reads the field's
    // presence, and a key that is there holding nothing is not the same
    // instruction as a region that never named a ceiling.
    const { maximumCellSize_cells: _dropped, ...rest } = record;
    return rest;
  }
  // A ceiling below the current floor means the whole interval moves down. Keep
  // the bounds valid and re-snap to the newly implied floor in the same edit.
  const floor = Math.min(record.minimumCellSize_cells, cells);
  const next: RefinementRegionRecord = {
    ...record,
    minimumCellSize_cells: floor,
    maximumCellSize_cells: cells,
  };
  return lattice === undefined ? next : resnapped(next, lattice, brick_cells ?? floor);
}

function resnapped(
  record: RefinementRegionRecord,
  lattice: RegionLattice,
  brick_cells: number,
): RefinementRegionRecord {
  const box = snapRegionBox(record.min_cells, record.max_cells,
    regionSnapStep_cells(record, brick_cells), lattice);
  return { ...record, min_cells: box.min, max_cells: box.max };
}

/**
 * What a box enforces, in the lattice's own words.
 *
 * Two readings of one fact: with a metre scale to quote it is the studio's
 * sentence about millimetres, and without one it is the lab's phrase in cells.
 * The *branching* — no ceiling, a ceiling equal to the floor, a range — is the
 * part that was written twice, and it is what this shares.
 */
export function regionCaption(
  record: RefinementRegionRecord,
  cellEdge_mm?: number,
): string {
  const floor = record.minimumCellSize_cells;
  const ceiling = record.maximumCellSize_cells;
  if (cellEdge_mm === undefined) {
    return ceiling === floor
      ? `held at ${floor}`
      : `≥ ${floor} cell${floor === 1 ? "" : "s"}`;
  }
  const mm = (cells: number) => (cells * cellEdge_mm).toFixed(0);
  if (ceiling === undefined) {
    return `No pressure cell smaller than ${mm(floor)} mm inside this box. `
      + "Grading still splits leaves on its boundary.";
  }
  return ceiling === floor
    ? `Fully contained pressure cells are held at ${mm(floor)} mm inside this box.`
    : `Fully contained pressure cells stay between ${mm(floor)} and ${mm(ceiling)} mm inside this box.`;
}

/**
 * The three enumerations a region carries, as the shared row protocol.
 *
 * `EditorChoiceGroup` is already host-agnostic data — an id, a current value,
 * and options each describing the patch they would produce — so the *only*
 * thing that made this studio-specific was that `apply` returned a
 * `SceneDescription` patch. Handed a `RegionSpace`, it returns whatever that
 * host calls a patch, and `EntityOptionRows` renders the identical rows in
 * either one.
 */
export function regionChoices<Doc, Patch>(
  space: RegionSpace<Doc, Patch>,
  doc: Doc,
  record: RefinementRegionRecord,
): EditorChoiceGroup<Patch>[] {
  const lattice = space.lattice(doc);
  const edge_mm = space.cellEdge_mm?.(doc);
  const write = (next: RefinementRegionRecord) => space.write(doc, record.id, next);
  const floor = record.minimumCellSize_cells;
  const ceiling = record.maximumCellSize_cells;
  const edgeClause = (cells: number) => edge_mm === undefined
    ? "" : ` · ${(cells * edge_mm).toFixed(0)} mm edge`;
  return [
    {
      id: "rule",
      label: "This box means",
      // Short tags throughout, because a region's column is the narrow one and
      // "THIS BOX MEANS" is a sentence where the reader needs a label. The full
      // phrasing is still on each row's tip, which is where a first reader
      // meets it.
      tag: "Means",
      value: record.rule,
      options: REFINEMENT_REGION_RULES.map((rule) => ({
        id: rule.id,
        label: rule.label,
        hint: rule.hint,
        enabled: true,
        apply: () => write({ ...record, rule: rule.id }),
      })),
    },
    {
      id: "minimumCellSize",
      label: "Smallest cell",
      tag: "Min",
      value: String(floor),
      options: space.cellSizes.map((cells) => ({
        id: String(cells),
        label: `${cells}³`,
        hint: `${cells}³ finest cells${edgeClause(cells)}`,
        enabled: true,
        apply: () => write(regionWithFloor(record, cells, lattice, space.brick_cells)),
      })),
    },
    {
      id: "maximumCellSize",
      label: "Largest cell",
      tag: "Max",
      value: ceiling === undefined ? "auto" : String(ceiling),
      options: [
        {
          id: "auto",
          label: "AUTO",
          hint: "Evidence decides how far quiet fluid may coarsen",
          enabled: true,
          apply: () => write(regionWithCeiling(record, undefined)),
        },
        ...space.cellSizes.map((cells) => ({
          id: String(cells),
          label: `${cells}³`,
          hint: `No cell larger than ${cells}³ finest cells${edgeClause(cells)}`,
          enabled: true,
          apply: () => write(regionWithCeiling(record, cells, lattice, space.brick_cells)),
        })),
      ],
    },
  ];
}

/**
 * The rung a box drawn next will carry on *this* host.
 *
 * The draft is shared and the ladders are not — the studio offers six rungs to
 * the lab's four — so a reader who chose 32 in the studio must not hand the lab
 * a bound it has no leaf for. Rounding down onto the host's own ladder is the
 * same bargain `clampRefinementRegionCellSize` already makes for a URL: a bound
 * that cannot be honoured becomes the nearest one that can, rather than a
 * silent no-op or an empty menu.
 */
export function regionDraftCellSize<Doc, Patch>(
  space: Pick<RegionSpace<Doc, Patch>, "cellSizes" | "defaultCellSize_cells">,
  draft: Pick<RegionDraft, "cellSize_cells">,
): number {
  const asked = draft.cellSize_cells;
  if (asked === undefined) return space.defaultCellSize_cells;
  let chosen = space.cellSizes[0] ?? space.defaultCellSize_cells;
  for (const size of space.cellSizes) if (size <= asked + TOLERANCE) chosen = size;
  return chosen;
}

/**
 * Whether two corners name no area at all — a press, released where it began.
 *
 * This is the one thing the snap cannot decide for itself. `snapRegionBox`
 * guarantees a step of thickness on every axis, because a box thinner than the
 * cell it asks for contains none of them — and that guarantee turns a *click*
 * into a one-brick region, which is what Wave 2 saw on the lab and what the
 * studio did too (its press seeds a zero-footprint draft before the pointer has
 * travelled). A minimum thickness is right for a box somebody drew and wrong
 * for a box nobody drew, so the two are separated here: a click is a selection,
 * never a draw.
 *
 * Every axis, not any: a drag along one axis only is a real instruction — a wide
 * thin region the snap then makes one step deep — while a drag along none is not
 * an instruction at all.
 */
export function regionDrawIsDegenerate(
  anchor_cells: readonly number[],
  at_cells: readonly number[],
): boolean {
  const axes = Math.max(anchor_cells.length, at_cells.length);
  for (let axis = 0; axis < axes; axis += 1) {
    if (Math.abs((at_cells[axis] ?? 0) - (anchor_cells[axis] ?? 0)) > TOLERANCE) return false;
  }
  return true;
}

/**
 * The record a rubber-band drag describes, on either host.
 *
 * Takes both corners already in finest cells and in the host's own axis count,
 * because *resolving* a drag is the one part of drawing a region that is
 * genuinely each host's: the studio turns two screen points into a footprint on
 * a horizontal plane and seeds a height for the axis the drag cannot name,
 * while the lab's pointer is already in the lattice. What happens next — the
 * outward snap onto the step the floor implies, and the ceiling the draft asks
 * for — is the same in both, and is here.
 *
 * The ceiling is written only for a held box. A floor-only region is the
 * *absence* of a ceiling and not a ceiling that happens to be wide, which is
 * the distinction `regionWithCeiling` keeps and a consumer reads.
 *
 * `undefined` for a press that named no area — see `regionDrawIsDegenerate`.
 * The decision is here rather than in each host's release handler because it is
 * the same decision, and because it belongs beside the minimum thickness that
 * makes it necessary.
 */
export function regionFromDraw<Doc, Patch>(
  space: RegionSpace<Doc, Patch>,
  doc: Doc,
  anchor_cells: readonly number[],
  at_cells: readonly number[],
  draft: RegionDraft,
  options: { readonly id?: string } = {},
): RefinementRegionRecord | undefined {
  if (regionDrawIsDegenerate(anchor_cells, at_cells)) return undefined;
  const cells = regionDraftCellSize(space, draft);
  const box = snapRegionBox(anchor_cells, at_cells,
    regionSnapStep_cells({ minimumCellSize_cells: cells }, space.brick_cells),
    space.lattice(doc));
  return {
    id: options.id ?? space.nextId(doc),
    rule: draft.rule,
    minimumCellSize_cells: cells,
    ...(draft.holdAtOneTier ? { maximumCellSize_cells: cells } : {}),
    min_cells: box.min,
    max_cells: box.max,
  };
}

/**
 * A region as the shared row protocol sees it: three choices and a removal.
 *
 * `EntityOptionRows` and `EntityDeleteRow` render an `EditorEntity`, and until
 * this existed only the studio could build one — so the lab hand-wrote its
 * MIN/MAX/Remove strip against the same three enumerations. What those two
 * components actually read is `label`, `choices`, `fields`, `groups` and
 * `remove`; the rest of the interface is the *gizmo's* half, which the studio
 * fills in around this and the lab has no use for (its handles are SVG rects in
 * canvas cells, not world-space `EditorHandle`s).
 *
 * So this is the entity minus the gizmo, and the studio composes its own on top
 * rather than beside it — which is what makes "both hosts render the same rows
 * for one definition" a fact about one function rather than a coincidence
 * between two.
 */
export function regionEntity<Doc, Patch>(
  space: RegionSpace<Doc, Patch>,
  doc: Doc,
  record: RefinementRegionRecord,
): EditorEntity<Patch, Doc> {
  return {
    selection: { kind: "refinement-region", id: refinementRegionSelectionId(record.id) },
    label: record.id.toUpperCase(),
    tone: "region",
    frame: WORLD_FRAME,
    handles: [],
    draftSubject: "refinement-region",
    editLabel: (handle) => handle.space === "world"
      ? `Moved ${record.id}` : `Resized ${record.id}`,
    choices: regionChoices(space, doc, record),
    summary: regionCaption(record, space.cellEdge_mm?.(doc)),
    // A whole document rather than a patch, because removal is an absence: the
    // studio's last region leaving drops the `refinementRegions` key, and the
    // lab's list is a whole-list command either way.
    remove: () => space.write(doc, record.id, undefined) as unknown as Doc,
  };
}
