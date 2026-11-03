import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../solid-world";

/**
 * Subcell bounds for the analytic kinds.
 *
 * A patch's box is whole cells, because pages are. A shape positioned to a
 * fraction of a cell — a pen nib, a sphere, box or cylinder centred on the
 * exact point pressed — keeps that box as its *coverage* box and carries how
 * far its true bounds stand inside it as six insets in `payload`: low x, y, z,
 * then high x, y, z, each in cells, non-negative, and leaving the shape
 * non-empty on every axis. Keeping them local to the box lets a translated
 * stroke keep its shape across serialization and a resampled lattice.
 *
 * The ellipsoid has always read them; the cylinder and the cuboid (the subcell
 * box) read the same six numbers, so one decoder serves every kind that has
 * subcell bounds. A kind without them never calls this.
 */
const decoded = new WeakMap<Patch, readonly number[]>();
export function subcellInsets(patch: Patch): readonly number[] {
  if (patch.payload === undefined) return [0, 0, 0, 0, 0, 0];
  const held = decoded.get(patch);
  if (held) return held;
  const values: unknown = JSON.parse(patch.payload);
  if (!Array.isArray(values) || values.length !== 6 || !values.every(v => Number.isFinite(v) && v >= 0)
    || [0, 1, 2].some(a => values[a] + values[a + 3] >= patch.maximumExclusive[a]! - patch.minimum[a]!)) {
    throw new RangeError(`Invalid ${patch.kind ?? "box"} subcell bounds`);
  }
  decoded.set(patch, values);
  return values;
}
/** @deprecated The ellipsoid's name for `subcellInsets`, which every subcell kind now shares. */
export const ellipsoidInsets = subcellInsets;

/** The shape's true bounds, in cells and fractions of one: its box, less its insets. */
export function subcellBounds(patch: Patch): { minimum: Cell; maximumExclusive: Cell } {
  const inset = subcellInsets(patch);
  return {
    minimum: patch.minimum.map((v, a) => v + inset[a]!) as unknown as Cell,
    maximumExclusive: patch.maximumExclusive.map((v, a) => v - inset[a + 3]!) as unknown as Cell,
  };
}
/** @deprecated See `subcellBounds`. */
export const ellipsoidBounds = subcellBounds;

/**
 * A patch of `kind` whose shape spans the fractional bounds given: its box is
 * those bounds rounded out to whole cells, and the difference is its insets.
 * Bounds that are already whole carry no payload, so they are the plain kind.
 */
export function subcellPatch(minimum: Cell, maximumExclusive: Cell, operation: Patch["operation"],
  kind: "ellipsoid" | "cylinder" | "cuboid"): Patch {
  const low = minimum.map(Math.floor) as unknown as Cell;
  const high = maximumExclusive.map(Math.ceil) as unknown as Cell;
  const insets = [...minimum.map((v, a) => v - low[a]!), ...maximumExclusive.map((v, a) => high[a]! - v)];
  return { operation, kind, minimum: low, maximumExclusive: high,
    ...(kind === "ellipsoid" || insets.some((v) => v !== 0) ? { payload: JSON.stringify(insets) } : {}) };
}

export const subcellEllipsoidPatch = (minimum: Cell, maximumExclusive: Cell, operation: Patch["operation"]): Patch =>
  subcellPatch(minimum, maximumExclusive, operation, "ellipsoid");

/**
 * Rescale a subcell patch's insets onto a new box, for a lattice that moved
 * under it: the shape keeps the same share of its box.
 */
export function resizedSubcellPatch(patch: Patch, minimum: Cell, maximumExclusive: Cell): Patch {
  if (patch.payload === undefined) return { ...patch, minimum, maximumExclusive };
  const insets = subcellInsets(patch).map((v, i) => {
    const a = i % 3;
    return v * (maximumExclusive[a]! - minimum[a]!) / (patch.maximumExclusive[a]! - patch.minimum[a]!);
  });
  return { ...patch, minimum, maximumExclusive, payload: JSON.stringify(insets) };
}
