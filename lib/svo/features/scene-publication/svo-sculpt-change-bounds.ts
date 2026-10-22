import type { SceneDescription } from "../../../core/model";
import { sceneCellSizes_m } from "../../../core/scene-lattice";
import { sameSolidPatchList } from "../../../core/solid-patch-list";
import { solidWorldVoxelPatchBounds_m, type SolidWorldVoxelPatch } from "../../../core/solid-world";
import type { EnvironmentProxyPrimitive } from "../../../core/voxel-environments";
import { intersectsSvoSculptCuts, intersectsSvoSculptPrimitiveBounds, packSvoSculptCuts } from "./svo-sculpt-cuts";

type SceneLattice = Pick<SceneDescription, "container" | "voxelDomain">;
export const sceneSculptCuts = (scene: SceneLattice, patches: readonly SolidWorldVoxelPatch[]) =>
  packSvoSculptCuts(patches, { origin_m: [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2],
    cellSize_m: sceneCellSizes_m(scene) });

export const sculptPrimitiveBounds = (primitive: EnvironmentProxyPrimitive) => ({
  minimum: [primitive.aabb_m.min.x, primitive.aabb_m.min.y, primitive.aabb_m.min.z] as const,
  maximum: [primitive.aabb_m.max.x, primitive.aabb_m.max.y, primitive.aabb_m.max.z] as const,
});

/** Match native-import ownership in both renderer and voxelizer exactly. */
export function sceneryPrimitiveAffectedBySculptCuts(cuts: Uint32Array, primitive: EnvironmentProxyPrimitive): boolean {
  if (primitive.kind === "field-program" && primitive.program.sampledVolume) {
    const center = primitive.center_m, extent = primitive.halfExtent_m, q = primitive.orientation;
    return intersectsSvoSculptPrimitiveBounds(cuts, [center.x, center.y, center.z], [extent.x, extent.y, extent.z],
      q ? [q.x, q.y, q.z, q.w] : undefined);
  }
  const bounds = sculptPrimitiveBounds(primitive);
  return intersectsSvoSculptCuts(cuts, bounds.minimum, bounds.maximum);
}

/** Scenery clears need publication even when their voxel pages were already empty.
 * A native import changes representation as a whole on its first/last cut, so
 * those transitions dirty its whole bound. Subsequent stamps dirty only their
 * own volumes. The same rule restores planar proxies on undo. */
export function svoSculptChangeBounds(scene: SceneLattice, previous: readonly SolidWorldVoxelPatch[],
  next: readonly SolidWorldVoxelPatch[], primitives: readonly EnvironmentProxyPrimitive[]) {
  const before = previous.filter(p => p.operation === "clear" && p.affectsScenery === true);
  const after = next.filter(p => p.operation === "clear" && p.affectsScenery === true);
  const dirtyBounds: (ReturnType<typeof sculptPrimitiveBounds> & { identityOnly?: boolean })[] = [];
  const addedPrimitiveBounds: ReturnType<typeof sculptPrimitiveBounds>[] = [];
  const cutBounds: ReturnType<typeof sculptPrimitiveBounds>[] = [];
  let relistPlanar = false;
  if (sameSolidPatchList(before, after)) return { changed: false, dirtyBounds, addedPrimitiveBounds, cutBounds, relistPlanar };
  let start = 0, beforeEnd = before.length, afterEnd = after.length;
  const same = (a: SolidWorldVoxelPatch, b: SolidWorldVoxelPatch) => a === b || sameSolidPatchList([a], [b]);
  while (start < beforeEnd && start < afterEnd && same(before[start]!, after[start]!)) start++;
  while (beforeEnd > start && afterEnd > start && same(before[beforeEnd - 1]!, after[afterEnd - 1]!)) {
    beforeEnd--; afterEnd--;
  }
  for (const patch of [...before.slice(start, beforeEnd), ...after.slice(start, afterEnd)]) {
    cutBounds.push(solidWorldVoxelPatchBounds_m(scene, patch));
  }
  dirtyBounds.push(...cutBounds);
  const oldCuts = sceneSculptCuts(scene, before), newCuts = sceneSculptCuts(scene, after);
  for (const primitive of primitives) {
    const wasAffected = sceneryPrimitiveAffectedBySculptCuts(oldCuts, primitive);
    const isAffected = sceneryPrimitiveAffectedBySculptCuts(newCuts, primitive);
    if (wasAffected === isAffected) continue;
    const bounds = sculptPrimitiveBounds(primitive);
    relistPlanar ||= primitive.kind === "box";
    dirtyBounds.push(primitive.kind === "field-program" && primitive.program.sampledVolume
      ? { ...bounds, identityOnly: true } : bounds);
    if (isAffected) addedPrimitiveBounds.push(bounds);
  }
  return { changed: true, dirtyBounds, addedPrimitiveBounds, cutBounds, relistPlanar };
}
