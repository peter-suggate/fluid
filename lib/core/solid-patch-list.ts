import { solidVoxelPatchesKey, type SolidWorldVoxelPatch } from "./solid-world";

/**
 * Comparisons between two authored patch lists, priced for a document that
 * holds tens of thousands of patches and is edited continuously.
 *
 * A stroke appends to an immutable list, so the two lists a live edit compares
 * share their patch objects: identity over the shorter length is then the
 * exact answer and costs a pointer compare per patch. Lists that crossed a JSON
 * round trip or a structured clone share nothing; those are compared by the
 * order-sensitive fold `solidVoxelPatchesKey` keeps per array, never by
 * serializing either list.
 */

/** Patches the two lists hold as the very same objects, counted from the front. */
function sharedPrefixLength(left: readonly SolidWorldVoxelPatch[],
  right: readonly SolidWorldVoxelPatch[]): number {
  const length = Math.min(left.length, right.length);
  let index = 0;
  while (index < length && left[index] === right[index]) index += 1;
  return index;
}

/** Whether the two lists hold the same patches in the same order. */
export function sameSolidPatchList(left: readonly SolidWorldVoxelPatch[],
  right: readonly SolidWorldVoxelPatch[]): boolean {
  if (left === right) return true;
  if (left.length !== right.length) return false;
  return sharedPrefixLength(left, right) === left.length
    || solidVoxelPatchesKey(left) === solidVoxelPatchesKey(right);
}

/**
 * Whether `next` is `base` with zero or more patches appended.
 *
 * Undo and a branched history replace patches rather than append them, and a
 * replaced prefix must never be read as a stroke suffix: this is the test that
 * separates the two.
 */
export function solidPatchListExtends(base: readonly SolidWorldVoxelPatch[],
  next: readonly SolidWorldVoxelPatch[]): boolean {
  if (base.length > next.length) return false;
  if (base === next || sharedPrefixLength(base, next) === base.length) return true;
  return solidVoxelPatchesKey(base) === solidVoxelPatchesKey(
    base.length === next.length ? next : next.slice(0, base.length));
}

/**
 * The patches `next` appends to `base`, or undefined when `next` is not `base`
 * extended — an undo, a branch, another document.
 */
export function appendedSolidPatches(base: readonly SolidWorldVoxelPatch[],
  next: readonly SolidWorldVoxelPatch[]): readonly SolidWorldVoxelPatch[] | undefined {
  if (!solidPatchListExtends(base, next)) return undefined;
  return next.slice(base.length);
}
