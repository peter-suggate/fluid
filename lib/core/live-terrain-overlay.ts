import { solidWorldContentStamp, type SolidWorld, type SolidWorldVoxelPatch } from "./solid-world";
import type { SceneDescription } from "./model";

/** Immutable heightfield/lattice identity, excluding the mutable ordered overlay. */
export function terrainFieldStamp(scene: SceneDescription): string {
  return scene.terrain ? solidWorldContentStamp({ ...scene, solidVoxels: [] }) : "no-terrain";
}

/** Reserved at construction, never grown or recompiled by a live terrain edit. */
export const LIVE_TERRAIN_PATCH_RESERVE = 4096;
export interface TerrainOverlayPatch {
  readonly operation: "fill" | "clear";
  readonly minimum_m: readonly [number, number, number];
  readonly maximum_m: readonly [number, number, number];
  readonly materialId: number;
}
export function terrainOverlayPatches(world: Pick<SolidWorld, "patches">, lattice: {
  readonly origin_m: readonly [number, number, number]; readonly cellSize_m: readonly [number, number, number];
}): TerrainOverlayPatch[] {
  return world.patches.map((patch: SolidWorldVoxelPatch) => ({
    operation: patch.operation,
    minimum_m: patch.minimum.map((value, axis) => lattice.origin_m[axis]! + value * lattice.cellSize_m[axis]!) as [number, number, number],
    maximum_m: patch.maximumExclusive.map((value, axis) => lattice.origin_m[axis]! + value * lattice.cellSize_m[axis]!) as [number, number, number],
    materialId: patch.materialId ?? 1,
  }));
}
/** Same ordered fill/clear program for initial upload, live replacement and Undo. */
export function packTerrainOverlay(patches: readonly TerrainOverlayPatch[], capacity: number): ArrayBuffer {
  if (patches.length > capacity) throw new RangeError(`Live terrain edit capacity reached (${capacity} patches). Undo or open a new scene.`);
  const data = new ArrayBuffer(patches.length * 8 * 4);
  const words = new Uint32Array(data), floats = new Float32Array(data);
  patches.forEach((patch, index) => {
    const base = index * 8;
    floats.set(patch.minimum_m, base); words[base + 3] = patch.operation === "fill" ? 1 : 0;
    floats.set(patch.maximum_m, base + 4); words[base + 7] = patch.materialId;
  });
  return data;
}
