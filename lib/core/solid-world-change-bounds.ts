import type { SceneDescription } from "./model";
import { SOLID_WORLD_BRICK_CELLS, SOLID_WORLD_VOXELS_PER_PAGE, solidWorldVoxelPatchBounds_m,
  type SolidWorld, type SolidWorldCoordinate, type SolidWorldVoxelPatchBounds } from "./solid-world";

/** Tight per-page edit regions. Removed cells invalidate their old samples but
 * never request new presentation topology. Unchanged immutable pages are free. */
export function solidWorldChangeBounds(
  scene: Pick<SceneDescription, "container" | "voxelDomain">,
  previous: SolidWorld,
  next: SolidWorld,
): { dirtyBounds: SolidWorldVoxelPatchBounds[]; addedBounds: SolidWorldVoxelPatchBounds[] } {
  const oldPages = new Map(previous.pages.map(page => [page.coordinate.join(","), page]));
  const newPages = new Map(next.pages.map(page => [page.coordinate.join(","), page]));
  const dirtyBounds: SolidWorldVoxelPatchBounds[] = [], addedBounds: SolidWorldVoxelPatchBounds[] = [];
  for (const key of new Set([...oldPages.keys(), ...newPages.keys()])) {
    const oldPage = oldPages.get(key), newPage = newPages.get(key);
    if (oldPage === newPage) continue;
    const coordinate = (newPage ?? oldPage)!.coordinate;
    const dirtyMin = [Infinity, Infinity, Infinity], dirtyMax = [-Infinity, -Infinity, -Infinity];
    const addedMin = [Infinity, Infinity, Infinity], addedMax = [-Infinity, -Infinity, -Infinity];
    for (let local = 0; local < SOLID_WORLD_VOXELS_PER_PAGE; local++) {
      const fraction = newPage?.solidFraction[local] ?? 0;
      if ((oldPage?.solidFraction[local] ?? 0) === fraction
        && (oldPage?.materialId[local] ?? 0) === (newPage?.materialId[local] ?? 0)
        && (oldPage?.signedDistanceQ8[local] ?? 0x7fff) === (newPage?.signedDistanceQ8[local] ?? 0x7fff)) continue;
      const cell = [local % SOLID_WORLD_BRICK_CELLS,
        Math.floor(local / SOLID_WORLD_BRICK_CELLS) % SOLID_WORLD_BRICK_CELLS,
        Math.floor(local / (SOLID_WORLD_BRICK_CELLS ** 2))]
        .map((v, axis) => v + coordinate[axis]! * SOLID_WORLD_BRICK_CELLS);
      for (let axis = 0; axis < 3; axis++) {
        dirtyMin[axis] = Math.min(dirtyMin[axis]!, cell[axis]!);
        dirtyMax[axis] = Math.max(dirtyMax[axis]!, cell[axis]! + 1);
        if (fraction > 0) {
          addedMin[axis] = Math.min(addedMin[axis]!, cell[axis]!);
          addedMax[axis] = Math.max(addedMax[axis]!, cell[axis]! + 1);
        }
      }
    }
    const append = (minimum: number[], maximumExclusive: number[], output: SolidWorldVoxelPatchBounds[]) => {
      if (!Number.isFinite(minimum[0])) return;
      output.push(solidWorldVoxelPatchBounds_m(scene, { operation: "fill",
        minimum: minimum as unknown as SolidWorldCoordinate,
        maximumExclusive: maximumExclusive as unknown as SolidWorldCoordinate }));
    };
    append(dirtyMin, dirtyMax, dirtyBounds);
    append(addedMin, addedMax, addedBounds);
  }
  return { dirtyBounds, addedBounds };
}
