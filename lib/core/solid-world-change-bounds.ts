import type { SceneDescription } from "./model";
import { appendedSolidPatches } from "./solid-patch-list";
import { SOLID_WORLD_BRICK_CELLS, solidWorldChangedPages, solidWorldPageAt as pageAt,
  solidWorldPageOrder as pageOrder, solidWorldVoxelPatchBounds_m, type SolidGround, type SolidWorld, type SolidWorldCoordinate,
  type SolidWorldPage, type SolidWorldVoxelPatchBounds } from "./solid-world";

export interface SolidWorldChangeBounds {
  dirtyBounds: SolidWorldVoxelPatchBounds[];
  addedBounds: SolidWorldVoxelPatchBounds[];
}

/**
 * Tight per-page edit regions. Removed cells invalidate their old samples but
 * never request new presentation topology. Unchanged immutable pages are free.
 *
 * The bounds are read from the pages, so they hold for every patch kind and
 * operation: a sphere reports the cells it covered rather than its box, and a
 * repaint reports the cells whose material moved.
 *
 * A world composed from the previous one by strokes names the pages those
 * strokes copied, made and dropped (`solidWorldChangedPages`), and only those
 * are compared: the cost is the stroke's and not the world's, whatever ground
 * either stands on.
 *
 * `sameTerrain` is the caller's statement that both worlds were composed over
 * the same ground. It serves a pair of worlds that does not carry that record
 * (a later world composed again from its document) but whose patch lists
 * still differ by an appended stroke: those are compared over the pages under
 * the stroke's own boxes. Without either, as for an undo or a branch, the
 * pages patches wrote in either world are visited. The ground is never
 * walked: two worlds over one ground agree wherever neither wrote a page, and
 * two grounds are compared by their column heights.
 */
export function solidWorldChangeBounds(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
  previous: SolidWorld,
  next: SolidWorld,
  options: { readonly sameTerrain?: boolean } = {},
): SolidWorldChangeBounds {
  const dirtyBounds: SolidWorldVoxelPatchBounds[] = [], addedBounds: SolidWorldVoxelPatchBounds[] = [];
  if (previous === next) return { dirtyBounds, addedBounds };
  const compare = (oldPage: SolidWorldPage | undefined, newPage: SolidWorldPage | undefined): void => {
    if (oldPage === newPage) return;
    const coordinate = (newPage ?? oldPage)!.coordinate;
    const B = SOLID_WORLD_BRICK_CELLS;
    // Local extents, inclusive. This loop is every cell of every page a stroke
    // touched, on the path that accepts an edit.
    let dirtyX0 = B, dirtyY0 = B, dirtyZ0 = B, dirtyX1 = -1, dirtyY1 = -1, dirtyZ1 = -1;
    let addedX0 = B, addedY0 = B, addedZ0 = B, addedX1 = -1, addedY1 = -1, addedZ1 = -1;
    const oldFraction = oldPage?.solidFraction, newFraction = newPage?.solidFraction;
    const oldMaterial = oldPage?.materialId, newMaterial = newPage?.materialId;
    const oldDistance = oldPage?.signedDistanceQ8, newDistance = newPage?.signedDistanceQ8;
    let local = 0;
    for (let z = 0; z < B; z++) for (let y = 0; y < B; y++) for (let x = 0; x < B; x++, local++) {
      const fraction = newFraction ? newFraction[local]! : 0;
      if ((oldFraction ? oldFraction[local]! : 0) === fraction
        && (oldMaterial ? oldMaterial[local]! : 0) === (newMaterial ? newMaterial[local]! : 0)
        && (oldDistance ? oldDistance[local]! : 0x7fff) === (newDistance ? newDistance[local]! : 0x7fff)) continue;
      if (x < dirtyX0) dirtyX0 = x;
      if (x > dirtyX1) dirtyX1 = x;
      if (y < dirtyY0) dirtyY0 = y;
      if (y > dirtyY1) dirtyY1 = y;
      if (z < dirtyZ0) dirtyZ0 = z;
      if (z > dirtyZ1) dirtyZ1 = z;
      if (fraction > 0) {
        if (x < addedX0) addedX0 = x;
        if (x > addedX1) addedX1 = x;
        if (y < addedY0) addedY0 = y;
        if (y > addedY1) addedY1 = y;
        if (z < addedZ0) addedZ0 = z;
        if (z > addedZ1) addedZ1 = z;
      }
    }
    const origin = [coordinate[0] * B, coordinate[1] * B, coordinate[2] * B];
    const append = (low: number[], high: number[], output: SolidWorldVoxelPatchBounds[]) => {
      if (high[0]! < 0) return;
      output.push(solidWorldVoxelPatchBounds_m(scene, { operation: "fill",
        minimum: low.map((value, axis) => value + origin[axis]!) as unknown as SolidWorldCoordinate,
        maximumExclusive: high.map((value, axis) => value + 1 + origin[axis]!) as unknown as SolidWorldCoordinate }));
    };
    append([dirtyX0, dirtyY0, dirtyZ0], [dirtyX1, dirtyY1, dirtyZ1], dirtyBounds);
    append([addedX0, addedY0, addedZ0], [addedX1, addedY1, addedZ1], addedBounds);
  };
  // Pages are reported in the previous world's order, then the pages only the
  // next world holds in its own: the order a whole-world comparison visits them.
  // A page a clear emptied is held only to keep the ground out of it, and
  // compares as no page: it is reported where a world without it would.
  const nextOnly: SolidWorldPage[] = [];
  const holdsSolid = (page: SolidWorldPage | undefined): page is SolidWorldPage => {
    if (page === undefined) return false;
    const fractions = page.solidFraction;
    for (let local = 0; local < fractions.length; local++) if (fractions[local]! > 0) return true;
    return false;
  };
  const report = (oldPage: SolidWorldPage | undefined, newPage: SolidWorldPage | undefined): void => {
    if (holdsSolid(oldPage)) compare(oldPage, newPage);
    else if (holdsSolid(newPage)) nextOnly.push(newPage);
  };
  // A world composed from the previous one by strokes names the pages it
  // copied, made and dropped, whatever ground either stands on.
  const changed = solidWorldChangedPages(previous, next);
  const sameGround = previous.ground?.key === next.ground?.key;
  const named = changed ?? (options.sameTerrain && sameGround ? appendedPageCoordinates(previous, next) : undefined);
  if (named) {
    for (const coordinate of named) report(pageAt(previous, coordinate), pageAt(next, coordinate));
    for (const page of nextOnly) compare(undefined, page);
    return { dirtyBounds, addedBounds };
  }
  // Every SolidWorld lists its written pages ascending in (z, y, x), so the
  // two lists merge in one pass. A page only one world wrote is compared with
  // what the other holds there: its ground, or nothing.
  assertPagesAscend(previous);
  assertPagesAscend(next);
  let cursor = 0;
  for (const page of previous.pages) {
    while (cursor < next.pages.length
      && pageOrder(next.pages[cursor]!.coordinate, page.coordinate) < 0) {
      const written = next.pages[cursor++]!;
      report(previous.ground?.page(written.coordinate), written);
    }
    const match = cursor < next.pages.length
      && pageOrder(next.pages[cursor]!.coordinate, page.coordinate) === 0;
    report(page, match ? next.pages[cursor++]! : next.ground?.page(page.coordinate));
  }
  while (cursor < next.pages.length) {
    const written = next.pages[cursor++]!;
    report(previous.ground?.page(written.coordinate), written);
  }
  for (const page of nextOnly) compare(undefined, page);
  if (!sameGround) groundChangeBounds(scene, previous.ground, next.ground, dirtyBounds, addedBounds);
  return { dirtyBounds, addedBounds };
}

/**
 * What a different ground changes, a page column at a time, from the column
 * heights of the two and without a page of either.
 *
 * A column whose height moved changes the distance of every cell under it and
 * the fraction of the cells at its top, so a moved column is dirty from the
 * floor to the taller of its two surfaces, and what it adds is the new ground.
 * The bounds cover written pages too; those are also compared cell by cell.
 */
function groundChangeBounds(scene: Pick<SceneDescription, "container" | "voxelDomain">,
  previous: SolidGround | undefined, next: SolidGround | undefined,
  dirtyBounds: SolidWorldVoxelPatchBounds[], addedBounds: SolidWorldVoxelPatchBounds[]): void {
  const B = SOLID_WORLD_BRICK_CELLS, grounds = [previous, next].filter((ground) => ground !== undefined);
  if (grounds.length === 0) return;
  const lo = [0, 2].map((axis) => Math.min(...grounds.map((ground) => ground.minimum[axis]!)));
  const hi = [0, 2].map((axis) => Math.max(...grounds.map((ground) => ground.maximumExclusive[axis]!)));
  const push = (output: SolidWorldVoxelPatchBounds[], minimum: SolidWorldCoordinate, maximumExclusive: SolidWorldCoordinate) =>
    output.push(solidWorldVoxelPatchBounds_m(scene, { operation: "fill", minimum, maximumExclusive }));
  for (let pz = Math.floor(lo[1]! / B); B * pz < hi[1]!; pz += 1) for (let px = Math.floor(lo[0]! / B); B * px < hi[0]!; px += 1) {
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity, dirtyTop = 0, addedTop = 0;
    for (let z = B * pz; z < B * pz + B; z += 1) for (let x = B * px; x < B * px + B; x += 1) {
      if ((previous?.height_m(x, z) ?? 0) === (next?.height_m(x, z) ?? 0)) continue;
      x0 = Math.min(x0, x); x1 = Math.max(x1, x + 1); z0 = Math.min(z0, z); z1 = Math.max(z1, z + 1);
      const cells = next?.solidCells(x, z) ?? 0;
      dirtyTop = Math.max(dirtyTop, cells, previous?.solidCells(x, z) ?? 0);
      addedTop = Math.max(addedTop, cells);
    }
    if (dirtyTop > 0) push(dirtyBounds, [x0, 0, z0], [x1, dirtyTop, z1]);
    if (addedTop > 0) push(addedBounds, [x0, 0, z0], [x1, addedTop, z1]);
  }
}

/** The merge above is only a comparison of the two worlds if both are in order. */
function assertPagesAscend(world: SolidWorld): void {
  for (let index = 1; index < world.pages.length; index += 1) {
    if (pageOrder(world.pages[index - 1]!.coordinate, world.pages[index]!.coordinate) >= 0) {
      throw new Error("SolidWorld pages must ascend in z, y, x");
    }
  }
}

/**
 * The page coordinates an appended stroke can have changed, ascending in
 * (z, y, x).
 *
 * Every patch stays inside its box, so the pages under the appended boxes are
 * the only ones that can differ, and over one ground a coordinate neither
 * world wrote holds the same ground in both. Undefined when `next` is not
 * `previous` with patches appended, or when those boxes span more pages than
 * the two worlds wrote, where visiting every written page is the cheaper walk.
 */
function appendedPageCoordinates(previous: SolidWorld, next: SolidWorld): SolidWorldCoordinate[] | undefined {
  const appended = appendedSolidPatches(previous.patches, next.patches);
  if (!appended) return undefined;
  const B = SOLID_WORLD_BRICK_CELLS;
  const heldPages = previous.pages.length + next.pages.length;
  let spannedPages = 0;
  for (const patch of appended) {
    spannedPages += patch.minimum.reduce((pages, minimum, axis) => pages
      * (Math.floor((patch.maximumExclusive[axis]! - 1) / B) - Math.floor(minimum / B) + 1), 1);
    if (spannedPages > heldPages) return undefined;
  }
  const written = new Set<SolidWorldPage>();
  for (const patch of appended) {
    const first = patch.minimum.map((value) => Math.floor(value / B));
    const last = patch.maximumExclusive.map((value) => Math.floor((value - 1) / B));
    for (let z = first[2]!; z <= last[2]!; z += 1)
      for (let y = first[1]!; y <= last[1]!; y += 1)
        for (let x = first[0]!; x <= last[0]!; x += 1) {
          const coordinate: SolidWorldCoordinate = [x, y, z];
          const nextIndex = next.directory.lookup(coordinate);
          if (nextIndex !== undefined) { written.add(next.pages[nextIndex]!); continue; }
          const previousIndex = previous.directory.lookup(coordinate);
          if (previousIndex !== undefined) written.add(previous.pages[previousIndex]!);
        }
  }
  return [...written].map((page) => page.coordinate).sort(pageOrder);
}
