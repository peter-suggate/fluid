/**
 * The residency world a simulation engine builds its bricks in, as core hands
 * it out.
 *
 * The SVO stack encodes and traces voxels; which solver filled them is not its
 * business, and the boundary that says so runs both ways — a method may not
 * reach the SVO stack either. The octree engine nonetheless needs a brick
 * residency scheduler and a scene-presentation source, and the only production
 * implementation of both is the SVO sparse-brick world. Composing the two is
 * core's job, so core states the shape the engine depends on here and returns
 * the SVO-backed implementation behind it.
 *
 * The interface is deliberately the members the engine actually calls and no
 * more: it is a description of what a solver may ask of a residency world, not
 * a mirror of `OctreeSparseBrickWorld`. Anything the engine does not name here
 * stays private to the render layer, which is what keeps a future non-SVO
 * residency provider a drop-in rather than a rewrite.
 */
import type { RenderFrameSeam } from "./render-frame-stages";
import type { SceneDescription } from "./model";
import type { FluidBrickResidencyStats, GPUFluidBrickResidency } from "./webgpu-fluid-brick-residency";
import type { SparseScenePrimitiveUpdate } from "./webgpu-sparse-scene-proxies";
import type { SparseVoxelSceneRenderSource } from "./webgpu-voxel-debug";
import {
  OctreeSparseBrickWorld,
  type OctreeSparseBrickWorldOptions,
} from "../svo/webgpu-svo-sparse-bricks";

export type OctreeSparseResidencyWorldOptions = OctreeSparseBrickWorldOptions;

export interface OctreeSparseResidencyWorld {
  /** The residency the solver schedules pressure topology against. */
  readonly topologyResidency: GPUFluidBrickResidency;
  /** Full wet-domain residency when the world keeps one apart from the band. */
  readonly bulkResidency?: GPUFluidBrickResidency;
  readonly bulkResidencyWorklist: GPUBuffer | undefined;
  /** What the renderer draws; the solver only forwards it. */
  readonly sceneSource: SparseVoxelSceneRenderSource;
  readonly allocatedBytes: number;
  initializePipelines(): Promise<void>;
  rescaleRenderDomain(scene: SceneDescription): void;
  stageSceneUpdate(scene: SceneDescription): boolean;
  stageLivePrimitiveUpdates(updates: readonly SparseScenePrimitiveUpdate[]): boolean;
  encodeSceneMaintenance(encoder: GPUCommandEncoder, deferDerived?: boolean, seam?: RenderFrameSeam<"world">): boolean;
  readBulkResidencyStats(): Promise<FluidBrickResidencyStats> | undefined;
  destroy(): void;
}

/** Build the production residency world. Synchronous, as the solver's own
 * construction is; the render worker drives the sliced build instead. */
export function createOctreeSparseResidencyWorld(
  device: GPUDevice,
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  options: OctreeSparseResidencyWorldOptions = {},
): OctreeSparseResidencyWorld {
  return new OctreeSparseBrickWorld(device, scene, dimensions, options);
}
