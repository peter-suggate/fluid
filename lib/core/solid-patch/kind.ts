import type { SolidWorldCoordinate, SolidWorldVoxelPatch } from "../solid-world";

/**
 * The plugin contract for what a solid patch *is*.
 *
 * A patch in the scene document is a box, an operation and a kind id. The kind
 * is a plugin: it says how much of each cell inside the box it covers, and
 * nothing else. What that coverage then does to the world — fill it, clear it,
 * repaint it — is the operation, and it is composed in exactly one place
 * (`applySolidWorldPatches`), by the rule below, for every kind alike. So a new
 * shape is one file in this directory and one line in the registry, and no
 * consumer of the solid world learns its name.
 *
 * Composition, per cell, with `c` the kind's coverage (0..255), `d` its signed
 * distance and `e` the cell's existing value:
 *
 * - `fill`:  fraction = max(e.fraction, c); distance = min(e.distance, d);
 *            material = the patch's wherever c > 0 and c >= e.fraction.
 * - `clear`: fraction = min(e.fraction, 255 - c); distance = max(e.distance, -d);
 *            material = 0 once the fraction reaches 0.
 * - `paint`: material = the patch's wherever c > 0 and e.fraction > 0.
 *            Fraction and distance are never touched.
 *
 * A cell with c = 0 is left alone by every operation. A wholly covered cell
 * (c = 255) is not blended but written: a fill makes it 255 / d / material and
 * a clear makes it an empty cell again, 0 / 0x7fff / 0, as does any clear that
 * takes a fraction to 0. That is exactly what a fill or a clear of a whole box
 * has always written, over terrain included, which is why old documents need
 * no migration.
 */

/** Dense voxels over a box: the three page channels, x fastest, then y, then z. */
export interface VoxelBlock {
  readonly size: SolidWorldCoordinate;
  /** 0 is empty and 255 is a fully covered solid voxel. */
  readonly solidFraction: Uint8Array;
  /** Signed distance in 1/256-cell units; negative is inside solid. */
  readonly signedDistanceQ8: Int16Array;
  /** 0 where the cell is empty. */
  readonly materialId: Uint16Array;
}

/** One row of cells a kind is asked to describe; index 0 is the row's first cell. */
export interface SolidPatchRowBuffer {
  /** Covered share of each cell, 0..255. Zero means the patch leaves the cell alone. */
  readonly coverage: Uint8Array;
  /** Signed distance to the patch's own surface, 1/256-cell units, negative inside. */
  readonly signedDistanceQ8: Int16Array;
  /** Per-cell material, or 0 to take the patch's own `materialId`. */
  readonly materialId: Uint16Array;
}

/** A row buffer for rows of up to `cells` cells. */
export const createSolidPatchRowBuffer = (cells: number): SolidPatchRowBuffer => ({
  coverage: new Uint8Array(cells),
  signedDistanceQ8: new Int16Array(cells),
  materialId: new Uint16Array(cells),
});

/**
 * What a row turned out to be.
 *
 * `"empty"` and `"full"` let the compositor skip the buffer entirely — a large
 * box costs its pages, not its voxels — and the kind need not have written it.
 * `"cells"` means the buffer holds the answer for every cell of the row.
 *
 * `"full"` is exactly a row of cells with coverage 255, distance -128 and the
 * patch's own material, so how a kind splits its answer between the three is
 * never visible in what is baked.
 */
export type SolidPatchRowResult = "empty" | "full" | "cells";

/** A kind's per-patch evaluator, built once per bake so a payload is decoded once. */
export interface SolidPatchRasterizer {
  /** Describe cells `x0 <= x < x1` of row (`y`, `z`), in lattice coordinates inside the patch box. */
  row(y: number, z: number, x0: number, x1: number, out: SolidPatchRowBuffer): SolidPatchRowResult;
}

/**
 * Closed-form shapes the GPU can evaluate from a patch's box alone.
 *
 * The terrain render path draws authored edits as an ordered list of analytic
 * records rather than as pages, so a kind that has a closed form names it here
 * and both paths agree. A kind without one (`analyticShape` undefined) has no
 * such record and must reach the renderer as pages.
 */
export const SOLID_PATCH_ANALYTIC_SHAPE = { box: 0, ellipsoid: 1, cylinder: 2 } as const;
export type SolidPatchAnalyticShape = typeof SOLID_PATCH_ANALYTIC_SHAPE[keyof typeof SOLID_PATCH_ANALYTIC_SHAPE];

export interface SolidPatchKind {
  /** The id a patch names in `kind`. */
  readonly id: string;
  /** The closed form the GPU evaluates for this kind, when it has one. */
  readonly analyticShape?: SolidPatchAnalyticShape;
  /**
   * Reject a malformed patch of this kind with a RangeError.
   *
   * The checks every kind shares — an integer, non-empty, i32 box and a valid
   * material — are made before this is called.
   */
  validate(patch: SolidWorldVoxelPatch): void;
  /**
   * Fold whatever identifies this patch beyond its operation, box and
   * material, which are folded for it. A kind with nothing more does nothing.
   */
  fold(patch: SolidWorldVoxelPatch, word: (value: number) => void): void;
  rasterizer(patch: SolidWorldVoxelPatch): SolidPatchRasterizer;
  /**
   * The same patch over a different box, for a lattice that moved under it.
   *
   * A kind described by its box alone needs none: the box is replaced and the
   * shape follows. A kind that carries cells must resample them, or its
   * payload no longer describes its box.
   */
  resized?(patch: SolidWorldVoxelPatch, minimum: SolidWorldCoordinate,
    maximumExclusive: SolidWorldCoordinate): SolidWorldVoxelPatch;
}

export const SOLID_PATCH_BOX_KIND = "box";
/** The ellipsoid inscribed in the patch box. */
export const SOLID_PATCH_ELLIPSOID_KIND = "ellipsoid";
/** The elliptic cylinder inscribed in the patch box, running the box's full length along `axis`. */
export const SOLID_PATCH_CYLINDER_KIND = "cylinder";
/** Dense cells carried in `payload`: whatever shape, coverage and materials were captured. */
export const SOLID_PATCH_BLOCK_KIND = "block";

/**
 * The one rule for whether a cell is solid to anything that needs a yes or no
 * — the solver's occupancy mask, a pick, a flood fill. Display reads the
 * fraction itself; nothing else may invent its own threshold.
 */
export const solidFractionOccupies = (fraction: number): boolean => fraction > 0;

/** The kind id a patch names, with the pre-kind default applied. */
export const solidPatchKindId = (patch: Pick<SolidWorldVoxelPatch, "kind">): string =>
  patch.kind ?? SOLID_PATCH_BOX_KIND;

/**
 * A static catalog of kinds, composed like the voxel tools' and the resource
 * plugins' registries. An unknown id is fatal at the first patch that names
 * it: a document this build cannot bake must not be drawn as something else.
 */
export function createSolidPatchKindRegistry(kinds: readonly SolidPatchKind[]) {
  const byId = new Map<string, SolidPatchKind>();
  for (const kind of kinds) {
    if (!kind.id.trim() || byId.has(kind.id)) throw new Error(`Duplicate or empty solid patch kind id: ${kind.id}`);
    byId.set(kind.id, kind);
  }
  return Object.freeze({
    kinds: Object.freeze([...kinds]),
    get(patch: Pick<SolidWorldVoxelPatch, "kind">): SolidPatchKind {
      const kind = byId.get(solidPatchKindId(patch));
      if (!kind) throw new RangeError(`Unknown solid patch kind "${solidPatchKindId(patch)}"`);
      return kind;
    },
  });
}
