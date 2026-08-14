/**
 * The projection's bind-group layouts, declared as data rather than as calls.
 *
 * Two things read them. The projection builds its layouts from these tables,
 * and the reachability audit checks each entry point's reflected bindings
 * against the same tables -- which is why the texture entries are declared
 * beside the buffer entries instead of being spliced in at the call site: an
 * audit that could see only the buffers reported the two legitimate textures
 * as uncovered bindings.
 */

export const OCTREE_PROJECTION_CORE_BUFFER_LAYOUT = Object.freeze([
  { binding: 2, type: "storage" },
  { binding: 3, type: "storage" },
  { binding: 4, type: "storage" },
  { binding: 5, type: "storage" },
  { binding: 6, type: "uniform" },
  { binding: 8, type: "storage" },
  { binding: 10, type: "storage" },
  { binding: 11, type: "storage" },
  { binding: 13, type: "storage" },
  { binding: 15, type: "read-only-storage" },
] as const);

/** The core family's non-buffer bindings. Exported beside the buffer layout so
 * the reachability audit can check an entry point against the complete family
 * rather than against the buffers alone, which reads a legitimate texture as an
 * uncovered binding. */
export const OCTREE_PROJECTION_CORE_TEXTURE_LAYOUT = Object.freeze([
  { binding: 12, viewDimension: "2d" },
  { binding: 14, viewDimension: "3d" },
] as const);

export const OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT = Object.freeze([
  { binding: 2, type: "storage" },
  { binding: 3, type: "storage" },
  { binding: 6, type: "uniform" },
  { binding: 7, type: "uniform" },
  { binding: 9, type: "storage" },
  { binding: 13, type: "storage" },
] as const);

export function projectionBufferLayoutEntries(
  entries: readonly { readonly binding: number;
    readonly type: "uniform" | "storage" | "read-only-storage" }[],
): GPUBindGroupLayoutEntry[] {
  return entries.map(({ binding, type }) => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
  }));
}
