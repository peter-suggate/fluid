/**
 * Compile-time proof that the shared octree catalog extents still describe the
 * generated catalog.
 *
 * `octree-shared/octree-catalog-extents.ts` has to hold literals: the technique
 * overlays bake them into WGSL at module scope, and the air-support ABI derives
 * its selector stride from one of them, so neither can wait for a backend to
 * hand the numbers over. The shared engine may not import the generated catalog
 * to check itself, so the check lives on this side, where both are visible.
 *
 * The manifest is declared `as const`, so each field below is a literal type.
 * Regenerating the catalog with different maxima fails `tsc` here rather than
 * silently under-bounding an overlay loop or rejecting every published
 * air-support layout as the wrong stride.
 */

import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import { OCTREE_CATALOG_EXTENTS } from "../octree-shared/octree-catalog-extents";

export const OCTREE_POWER_CATALOG_EXTENT_AGREEMENT: Readonly<{
  tetrahedronVertexCount: typeof OCTREE_GENERATED_POWER_CATALOG_MANIFEST.tetrahedronVertexCount;
  maximumTetrahedra: typeof OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra;
  maximumFaceIncidence: typeof OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence;
}> = OCTREE_CATALOG_EXTENTS;
