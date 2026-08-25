import type { SolidWorld } from "../../core/solid-world";
import {
  WEBGPU_SOLID_WORLD_ENTRY_WORDS,
  WEBGPU_SOLID_WORLD_FRACTION_WORDS,
  WEBGPU_SOLID_WORLD_HEADER_WORDS,
  WEBGPU_SOLID_WORLD_MAGIC,
  WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS,
  WEBGPU_SOLID_WORLD_SDF_WORDS,
  WEBGPU_SOLID_WORLD_VERSION,
  createWebgpuSolidWorldPageLayout,
  packWebgpuSolidWorldPages,
  writeWebgpuSolidWorldPages,
  type WebgpuSolidWorldPageLayout,
} from "../../core/webgpu-solid-world-pages";

/** Compatibility names for the Sparse CM12 consumer of the generic image. */
export const SPARSE_CM12_SOLID_OCCUPANCY_MAGIC = WEBGPU_SOLID_WORLD_MAGIC;
export const SPARSE_CM12_SOLID_OCCUPANCY_VERSION = WEBGPU_SOLID_WORLD_VERSION;
export const SPARSE_CM12_SOLID_OCCUPANCY_HEADER_WORDS = WEBGPU_SOLID_WORLD_HEADER_WORDS;
export const SPARSE_CM12_SOLID_OCCUPANCY_ENTRY_WORDS = WEBGPU_SOLID_WORLD_ENTRY_WORDS;
export const SPARSE_CM12_SOLID_FRACTION_PAGE_WORDS = WEBGPU_SOLID_WORLD_FRACTION_WORDS;
export const SPARSE_CM12_SOLID_SDF_PAGE_WORDS = WEBGPU_SOLID_WORLD_SDF_WORDS;
export const SPARSE_CM12_SOLID_OCCUPANCY_PAGE_WORDS = WEBGPU_SOLID_WORLD_PHYSICS_PAGE_WORDS;
export const SPARSE_CM12_SOLID_OCCUPANCY_MAX_BYTES = 8 * 1024 * 1024;

export type SparseCM12SolidOccupancyLayout = WebgpuSolidWorldPageLayout;

export function createSparseCM12SolidOccupancyLayout(options: {
  readonly baseWords: number;
  readonly authoredPageCount: number;
}): SparseCM12SolidOccupancyLayout {
  return createWebgpuSolidWorldPageLayout({ ...options, includesMaterial: false,
    maximumBytes: SPARSE_CM12_SOLID_OCCUPANCY_MAX_BYTES });
}

export function packSparseCM12SolidOccupancy(layout: SparseCM12SolidOccupancyLayout,
  world: SolidWorld, originFine: readonly [number, number, number]): Uint32Array {
  return packWebgpuSolidWorldPages(layout, world, originFine);
}

export function writeSparseCM12SolidOccupancy(queue: GPUQueue, destination: GPUBuffer,
  layout: SparseCM12SolidOccupancyLayout, world: SolidWorld,
  originFine: readonly [number, number, number]): void {
  writeWebgpuSolidWorldPages(queue, destination, layout, world, originFine);
}
