import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../solid-world";

// Optional subcell insets from the integer coverage box. Keeping these local
// to the box lets translated strokes retain their shape across serialization.
const decoded = new WeakMap<Patch, readonly number[]>();
export function ellipsoidInsets(patch: Patch): readonly number[] {
  if (patch.payload === undefined) return [0, 0, 0, 0, 0, 0];
  const held = decoded.get(patch);
  if (held) return held;
  const values: unknown = JSON.parse(patch.payload);
  if (!Array.isArray(values) || values.length !== 6 || !values.every(v => Number.isFinite(v) && v >= 0)
    || [0, 1, 2].some(a => values[a] + values[a + 3] >= patch.maximumExclusive[a]! - patch.minimum[a]!)) {
    throw new RangeError("Invalid ellipsoid subcell bounds");
  }
  decoded.set(patch, values);
  return values;
}

export function ellipsoidBounds(patch: Patch): { minimum: Cell; maximumExclusive: Cell } {
  const inset = ellipsoidInsets(patch);
  return {
    minimum: patch.minimum.map((v, a) => v + inset[a]!) as unknown as Cell,
    maximumExclusive: patch.maximumExclusive.map((v, a) => v - inset[a + 3]!) as unknown as Cell,
  };
}

export function subcellEllipsoidPatch(minimum: Cell, maximumExclusive: Cell, operation: Patch["operation"]): Patch {
  const low = minimum.map(Math.floor) as unknown as Cell;
  const high = maximumExclusive.map(Math.ceil) as unknown as Cell;
  return { operation, kind: "ellipsoid", minimum: low, maximumExclusive: high,
    payload: JSON.stringify([...minimum.map((v, a) => v - low[a]!), ...maximumExclusive.map((v, a) => high[a]! - v)]) };
}
