import type { SolidWorldCoordinate, SolidWorldPage, SolidWorldVoxelPatch } from "../solid-world";
import { createSolidPatchRowBuffer } from "./kind";
import { solidPatchKinds } from "./registry";

/**
 * Where a composed patch lands.
 *
 * The bake hands in its copy-on-write page map; the editor's voxel index hands
 * in its own sparse pages. Both therefore hold exactly the same cells, because
 * both were written by the one function below.
 */
export interface SolidPatchPageStore {
  /**
   * The writable page at a page coordinate (lattice coordinate / 8, floored).
   *
   * With `create` false a missing page is `undefined` and stays missing: a
   * clear or a paint has nothing to do where nothing is solid.
   */
  page(coordinate: SolidWorldCoordinate, create: boolean): SolidWorldPage | undefined;
}

/**
 * Cells along a page edge. `../solid-world` owns the number and refuses to
 * load if this one differs; it is restated so that this module needs nothing
 * from that one at run time.
 */
export const SOLID_PATCH_PAGE_CELLS = 8;

const EMPTY_DISTANCE_Q8 = 0x7fff;
/** Half a cell inside: the distance of a wholly covered cell. */
const WHOLE_DISTANCE_Q8 = -128;

const isCoordinate = (value: unknown): value is SolidWorldCoordinate =>
  Array.isArray(value) && value.length === 3 && value.every(Number.isSafeInteger);

/** Reject a malformed patch: the checks every kind shares, then the kind's own. */
export function validateSolidPatch(patch: SolidWorldVoxelPatch): void {
  if (patch.operation !== "fill" && patch.operation !== "clear" && patch.operation !== "paint") {
    throw new RangeError(`Unsupported solid patch operation "${String(patch.operation)}"`);
  }
  if (patch.affectsScenery !== undefined && typeof patch.affectsScenery !== "boolean") {
    throw new RangeError("SolidWorld scenery-cut flag must be boolean");
  }
  if (patch.affectsScenery === true && patch.operation !== "clear") {
    throw new RangeError("Only clear patches can affect scenery");
  }
  if (!isCoordinate(patch.minimum) || !isCoordinate(patch.maximumExclusive)
    || patch.minimum.some((value, axis) => value >= patch.maximumExclusive[axis]!)) {
    throw new RangeError("SolidWorld voxel patch must be a non-empty integer box");
  }
  const materialId = patch.materialId ?? 1;
  if (patch.operation !== "clear"
    && (!Number.isSafeInteger(materialId) || materialId < 1 || materialId > 0xffff)) {
    throw new RangeError("SolidWorld material ID must fit non-zero u16");
  }
  if (![...patch.minimum, ...patch.maximumExclusive].every((value) =>
    value >= -0x8000_0000 && value <= 0x7fff_ffff)) {
    throw new RangeError("SolidWorld voxel coordinate must fit signed i32");
  }
  solidPatchKinds.get(patch).validate(patch);
}

const rowBuffer = createSolidPatchRowBuffer(SOLID_PATCH_PAGE_CELLS);

/** A box of pages, in page coordinates. */
export interface SolidPatchPageRange {
  readonly minimum: SolidWorldCoordinate;
  readonly maximumExclusive: SolidWorldCoordinate;
}

/**
 * Apply one validated patch to a page store by the composition rule in
 * `./kind`. This is the only place an operation meets a kind.
 *
 * With `pages`, only the pages of that range are written. A page composes from
 * its own cells and the patch alone, so the pages of a range hold what they
 * would after the whole patch: a store may compose a region when it is asked
 * about it rather than the document when it arrives.
 */
export function composeSolidPatch(patch: SolidWorldVoxelPatch, store: SolidPatchPageStore,
  pages?: SolidPatchPageRange): void {
  const B = SOLID_PATCH_PAGE_CELLS, { operation } = patch, ownMaterial = patch.materialId ?? 1;
  const first = patch.minimum.map((value) => Math.floor(value / B));
  const last = patch.maximumExclusive.map((value) => Math.floor((value - 1) / B));
  if (pages) for (let axis = 0; axis < 3; axis += 1) {
    first[axis] = Math.max(first[axis]!, pages.minimum[axis]!);
    last[axis] = Math.min(last[axis]!, pages.maximumExclusive[axis]! - 1);
    if (first[axis]! > last[axis]!) return;
  }
  const rasterizer = solidPatchKinds.get(patch).rasterizer(patch);
  const { coverage, signedDistanceQ8: distance, materialId: material } = rowBuffer;
  for (let pz = first[2]!; pz <= last[2]!; pz += 1)
    for (let py = first[1]!; py <= last[1]!; py += 1)
      for (let px = first[0]!; px <= last[0]!; px += 1) {
        const coordinate: SolidWorldCoordinate = [px, py, pz];
        const lo = coordinate.map((p, axis) => Math.max(0, patch.minimum[axis]! - p * B));
        const hi = coordinate.map((p, axis) => Math.min(B, patch.maximumExclusive[axis]! - p * B));
        const count = hi[0]! - lo[0]!;
        // Asked for at the first row that has anything to say, so the corners
        // of a round patch's box neither create nor copy a page.
        let page: SolidWorldPage | undefined;
        rows: for (let z = lo[2]!; z < hi[2]!; z += 1)
          for (let y = lo[1]!; y < hi[1]!; y += 1) {
            const result = rasterizer.row(py * B + y, pz * B + z, px * B + lo[0]!, px * B + hi[0]!, rowBuffer);
            if (result === "empty") continue;
            if (!page) {
              page = store.page(coordinate, operation === "fill");
              // Nothing is solid here, and only a fill makes it so.
              if (!page) break rows;
            }
            const from = B * (y + B * z) + lo[0]!, to = from + count;
            const authored = operation === "paint" ? undefined : page.authored ??= new Uint8Array(B * B * B / 8);
            // Whole rows per page: the cost of a large box is its pages, not its voxels.
            if (result === "full" && operation !== "paint") {
              const fill = operation === "fill";
              page.solidFraction.fill(fill ? 255 : 0, from, to);
              page.signedDistanceQ8.fill(fill ? WHOLE_DISTANCE_Q8 : EMPTY_DISTANCE_Q8, from, to);
              page.materialId.fill(fill ? ownMaterial : 0, from, to);
              for (let at = from; at < to; at += 1) authored![at >> 3]! |= 1 << (at & 7);
              continue;
            }
            for (let index = 0; index < count; index += 1) {
              const c = result === "full" ? 255 : coverage[index]!;
              if (c === 0) continue;
              const at = from + index, existing = page.solidFraction[at]!;
              if (operation === "paint") {
                if (existing > 0) page.materialId[at] = result === "full" ? ownMaterial : material[index]! || ownMaterial;
              } else if (operation === "fill") {
                authored![at >> 3]! |= 1 << (at & 7);
                if (c >= existing) {
                  page.solidFraction[at] = c;
                  page.materialId[at] = material[index]! || ownMaterial;
                }
                page.signedDistanceQ8[at] = c === 255 ? distance[index]!
                  : Math.min(page.signedDistanceQ8[at]!, distance[index]!);
              } else {
                // Authored whether or not anything was solid: the cell is the
                // clear's, and ground drawn from other samples stays out of it.
                authored![at >> 3]! |= 1 << (at & 7);
                if (existing === 0) continue;
                const remaining = Math.min(existing, 255 - c);
                page.solidFraction[at] = remaining;
                if (remaining === 0) {
                  page.signedDistanceQ8[at] = EMPTY_DISTANCE_Q8;
                  page.materialId[at] = 0;
                } else {
                  page.signedDistanceQ8[at] = Math.min(0x7fff, Math.max(page.signedDistanceQ8[at]!, -distance[index]!));
                }
              }
            }
          }
      }
}
