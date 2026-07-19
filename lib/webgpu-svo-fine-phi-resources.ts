import { WebGPUSvoOwnerPageAllocator } from "./webgpu-octree-owner-pages";
import {
  WebGPUSvoFinePhiStager,
  type SvoFineFluidGpuCapability,
} from "./webgpu-svo-fine-phi-stager";
import { WebGPUSvoRenderResidencyConsumer } from "./webgpu-svo-render-residency-consumer";
import type { SparseSurfaceBandGPUSource } from "./webgpu-sparse-surface-band";
import type {
  SparseVoxelSceneRenderSource,
  SparseVoxelStructuralRenderSource,
} from "./webgpu-voxel-debug";

export type SvoFinePhiResourceEncodeStatus =
  | "encoded"
  | "unchanged"
  | "unpublished"
  | "source-changed"
  | "destroyed";

export interface SvoFinePhiResourceStages {
  readonly residency: Pick<WebGPUSvoRenderResidencyConsumer, "encode" | "destroy" | "allocatedBytes">;
  readonly ownerPages: Pick<WebGPUSvoOwnerPageAllocator, "encode" | "destroy" | "allocatedBytes">;
  readonly finePhi: Pick<WebGPUSvoFinePhiStager, "encode" | "destroy" | "capability" | "allocatedBytes"> & Partial<Pick<WebGPUSvoFinePhiStager, "mirrorPublication">>;
}

export interface SvoFinePhiOwnerDomain {
  /** Full padded structural scene. Consumer worklists are remapped into it. */
  readonly ownerDimensionsCells: readonly [number, number, number];
  readonly ownerDimensionsBricks: readonly [number, number, number];
  /** Solver-local source extent represented by sparse fine-phi pages. */
  readonly sourceDimensionsCells: readonly [number, number, number];
  readonly sourceDimensionsBricks: readonly [number, number, number];
  /** Solver origin in the padded structural scene-cell lattice. */
  readonly sourceOriginCells: readonly [number, number, number];
  readonly refinementFactor: 1 | 2 | 4;
}

/**
 * Resolve the renderer owner lattice shared by the solver residency list and
 * sparse surface field. The structural SVO domain may include large, aligned
 * environment padding; residency indices and fine-phi samples deliberately do
 * not. Every relationship is checked before any renderer resource is created.
 */
export function resolveSvoFinePhiOwnerDomain(
  structural: SparseVoxelStructuralRenderSource,
  fineSource: Pick<SparseSurfaceBandGPUSource, "fineDimensions" | "refinementFactor">,
): SvoFinePhiOwnerDomain {
  const residency = structural.fluidResidency;
  if (!residency) throw new RangeError("Fine-phi owner domain requires structural fluid residency");
  if (structural.domain.brickSize !== 8) throw new RangeError("Fine-phi owner domain requires 8-cubed structural bricks");
  const factor = fineSource.refinementFactor;
  if (factor !== 1 && factor !== 2 && factor !== 4) throw new RangeError("Fine-phi refinement must be 1, 2, or 4");
  const sourceDimensionsCells = fineSource.fineDimensions.map((fine, axis) => {
    if (!Number.isSafeInteger(fine) || fine < 1 || fine % factor !== 0) {
      throw new RangeError(`Sparse-surface fine dimension ${axis} must be a positive multiple of refinement`);
    }
    return fine / factor;
  }) as [number, number, number];
  const sourceDimensionsBricks = sourceDimensionsCells.map((value) => Math.ceil(value / 8)) as [number, number, number];
  if (residency.domain.dimensionsBricks.some((value, axis) => value !== sourceDimensionsBricks[axis])) {
    throw new RangeError("Sparse-surface coarse domain does not match the solver residency brick domain");
  }
  const sourceOriginCells = residency.domain.originBricks.map((value, axis) => {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new RangeError(`Solver residency brick origin ${axis} must be a non-negative integer`);
    }
    return value * 8;
  }) as [number, number, number];
  const ownerDimensionsCells = structural.domain.dimensionsCells;
  if (ownerDimensionsCells.some((value) => !Number.isSafeInteger(value) || value < 1)) {
    throw new RangeError("Fine-phi structural owner dimensions must be positive integers");
  }
  if (sourceOriginCells.some((origin, axis) => origin + sourceDimensionsCells[axis] > ownerDimensionsCells[axis])) {
    throw new RangeError("Solver-local fine-phi owner domain exceeds the padded structural scene domain");
  }
  const ownerDimensionsBricks = ownerDimensionsCells.map((value) => Math.ceil(value / 8)) as [number, number, number];
  return {
    ownerDimensionsCells: [...ownerDimensionsCells], ownerDimensionsBricks,
    sourceDimensionsCells, sourceDimensionsBricks, sourceOriginCells,
    refinementFactor: factor,
  };
}

interface FineSourceIdentity {
  readonly mode: SparseSurfaceBandGPUSource["mode"];
  readonly pageTable: GPUBuffer;
  readonly control: GPUBuffer;
  readonly phi: GPUBuffer;
  readonly params: GPUBuffer;
  readonly fineDimensions: readonly [number, number, number];
  readonly brickDimensions: readonly [number, number, number];
  readonly brickSize: number;
  readonly refinementFactor: number;
  readonly pageCapacity: number;
}

function sourceIdentity(source: SparseSurfaceBandGPUSource): FineSourceIdentity {
  return {
    mode: source.mode,
    pageTable: source.pageTable.buffer,
    control: source.control.buffer,
    phi: source.phi.buffer,
    params: source.params.buffer,
    fineDimensions: [...source.fineDimensions],
    brickDimensions: [...source.brickDimensions],
    brickSize: source.brickSize,
    refinementFactor: source.refinementFactor,
    pageCapacity: source.pageCapacity,
  };
}

function sameTuple(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function sameSource(identity: FineSourceIdentity, source: SparseSurfaceBandGPUSource): boolean {
  return identity.mode === source.mode
    && identity.pageTable === source.pageTable.buffer
    && identity.control === source.control.buffer
    && identity.phi === source.phi.buffer
    && identity.params === source.params.buffer
    && sameTuple(identity.fineDimensions, source.fineDimensions)
    && sameTuple(identity.brickDimensions, source.brickDimensions)
    && identity.brickSize === source.brickSize
    && identity.refinementFactor === source.refinementFactor
    && identity.pageCapacity === source.pageCapacity;
}

/**
 * Renderer-owned production chain. Its command order is the publication
 * contract: residency validation/compaction, owner activation/retirement, then
 * fine staging/scrubbing. The exposed capability is read-only and cannot take
 * water presentation ownership.
 */
export class WebGPUSvoFinePhiResources {
  readonly allocatedBytes: number;
  private readonly identity: FineSourceIdentity;
  private lastQueuedFineGeneration = 0;
  private destroyed = false;

  constructor(
    private readonly stages: SvoFinePhiResourceStages,
    source: SparseSurfaceBandGPUSource,
  ) {
    if (source.mode !== "authoritative") throw new RangeError("Production fine-phi resources require an authoritative source");
    this.identity = sourceIdentity(source);
    this.allocatedBytes = stages.residency.allocatedBytes + stages.ownerPages.allocatedBytes + stages.finePhi.allocatedBytes;
  }

  /** Encode at most once per CPU-published fine generation. GPU fences remain authoritative. */
  encode(encoder: GPUCommandEncoder, source: SparseSurfaceBandGPUSource): SvoFinePhiResourceEncodeStatus {
    if (this.destroyed) return "destroyed";
    if (!sameSource(this.identity, source)) return "source-changed";
    this.stages.finePhi.mirrorPublication?.(encoder);
    const generation = source.revision;
    if (!Number.isSafeInteger(generation) || generation < 1 || generation > 0xffff_ffff) return "unpublished";
    if (generation === this.lastQueuedFineGeneration) return "unchanged";
    this.stages.residency.encode(encoder);
    this.stages.ownerPages.encode(encoder);
    this.stages.finePhi.encode(encoder, generation);
    this.lastQueuedFineGeneration = generation;
    return "encoded";
  }

  capability(): SvoFineFluidGpuCapability | undefined {
    if (this.destroyed) return undefined;
    const capability = this.stages.finePhi.capability();
    if (capability.coarseFallbackRequired !== true || capability.directWaterOwnership !== false) {
      throw new Error("Fine-phi capability must preserve coarse fallback and legacy water ownership");
    }
    return capability;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    // Bind groups reference upstream buffers, so tear down in reverse order.
    this.stages.finePhi.destroy();
    this.stages.ownerPages.destroy();
    this.stages.residency.destroy();
  }
}

/** Instantiate the complete optional chain without weakening solver startup. */
export function createWebgpuSvoFinePhiResources(
  device: GPUDevice,
  sceneSource: SparseVoxelSceneRenderSource | undefined,
  fineSource: SparseSurfaceBandGPUSource | undefined,
): WebGPUSvoFinePhiResources | undefined {
  const structural = sceneSource?.structural;
  const residencySource = structural?.fluidResidency;
  if (!structural || !residencySource || !fineSource || fineSource.mode !== "authoritative") return undefined;
  const ownerDomain = resolveSvoFinePhiOwnerDomain(structural, fineSource);
  const maximumArenaBytes = Math.min(Number(device.limits.maxStorageBufferBindingSize), Number(device.limits.maxBufferSize));
  const residency = new WebGPUSvoRenderResidencyConsumer(device, structural, {
    capacity: residencySource.active.capacity,
    coarseCoverageComplete: true,
  });
  let ownerPages: WebGPUSvoOwnerPageAllocator | undefined;
  let finePhi: WebGPUSvoFinePhiStager | undefined;
  try {
    ownerPages = new WebGPUSvoOwnerPageAllocator(device, ownerDomain.ownerDimensionsCells, residency, {
      maximumPages: Math.min(residencySource.active.capacity, fineSource.pageCapacity),
      maximumArenaBytes,
    });
    finePhi = new WebGPUSvoFinePhiStager(device, ownerPages, residency, fineSource, structural.domain.cellSize_m, {
      maximumArenaBytes,
      sourceOriginBricks: residencySource.domain.originBricks,
      structuralPublication: structural.publication.state,
    });
    return new WebGPUSvoFinePhiResources({ residency, ownerPages, finePhi }, fineSource);
  } catch (error) {
    finePhi?.destroy();
    ownerPages?.destroy();
    residency.destroy();
    throw error;
  }
}
