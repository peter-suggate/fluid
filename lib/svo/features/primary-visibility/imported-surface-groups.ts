/** Static draw topology; poses, visibility and screen-error LOD stay on the GPU. */
export interface ImportedSurfaceGroup {
  readonly asset: string;
  readonly level: number;
  readonly first: number;
  readonly count: number;
  readonly instanceOffset: number;
  readonly capacity: number;
}
export interface ImportedSurfaceGroupPlan {
  readonly groups: readonly ImportedSurfaceGroup[];
  readonly firstGroupByAsset: ReadonlyMap<string, number>;
  readonly instanceCapacity: number;
}

/** Each asset/LOD has room for every instance of that asset. Selection emits
 * each visible instance exactly once, so no readback, resize, or overflow is
 * needed when the camera changes and all instances choose the same level. */
export function planImportedSurfaceGroups(
  instances: readonly { readonly asset: string }[],
  levelsFor: (asset: string) => readonly { readonly first: number; readonly count: number }[],
): ImportedSurfaceGroupPlan {
  const counts = new Map<string, number>();
  for (const { asset } of instances) counts.set(asset, (counts.get(asset) ?? 0) + 1);
  const groups: ImportedSurfaceGroup[] = [], firstGroupByAsset = new Map<string, number>();
  let instanceCapacity = 0;
  for (const [asset, capacity] of counts) {
    const levels = levelsFor(asset);
    if (levels.length < 1 || levels.length > 7) throw new RangeError(`Native asset ${asset} needs 1–7 surface levels`);
    firstGroupByAsset.set(asset, groups.length);
    levels.forEach(({ first, count }, level) => {
      if (![first, count].every(value => Number.isSafeInteger(value) && value >= 0 && value <= 0xffffffff) || first + count > 0x100000000)
        throw new RangeError(`Native asset ${asset} has an invalid index range`);
      groups.push({ asset, level, first, count, instanceOffset: instanceCapacity, capacity });
      instanceCapacity += capacity;
    });
  }
  return { groups, firstGroupByAsset, instanceCapacity };
}
