import type { GPUVelocityTransport } from "./webgpu-eulerian";
import { planOctreeHostVelocityAllocation } from "./octree-host-allocation";
import { planOctreeLeafFrontierAllocation, planOctreeOwnerAllocation, planOctreePhiSnapshotAllocation, planOctreePressureCapacity } from "./webgpu-octree";
import { planOctreeOwnerPages } from "./webgpu-octree-owner-pages";
import { planOctreeFaceMirror } from "./webgpu-octree-face-mirror";
import { planOctreeFaceTopologyTransfer } from "./webgpu-octree-face-transfer";
import { planOctreeFaceTransport } from "./webgpu-octree-face-transport";
import { planOctreeSurfaceAdapter } from "./webgpu-octree-surface-adapter";
import { planOctreeSurfacePages } from "./webgpu-octree-surface-pages";
import { planOctreeSurfaceStateAllocation } from "./octree-surface-allocation";
import { fluidBrickAtlasAllocatedBytes, planFluidBrickAtlas } from "./webgpu-brick-atlas";
import { planFineLevelSetBricks, type FineLevelSetBrickResolution, type FineLevelSetFactor } from "./octree-fine-levelset-bricks";
import { planOctreeCoarsePhi } from "./webgpu-octree-coarse-levelset";

export interface OctreeCompactAllocationPlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly denseHostBytesRemoved: number;
  readonly denseSurfaceBytesRemoved: number;
  readonly denseSnapshotBytesRemoved: number;
  readonly denseOwnerBytesRemoved: number;
  readonly denseFrontierBytesRemoved: number;
  readonly denseAtlasBytesRemoved: number;
  readonly denseBytesRemoved: number;
  readonly faceMirrorBytes: number;
  readonly topologyTransferBytes: number;
  readonly faceTransportBytes: number;
  readonly surfaceAdapterBytes: number;
  readonly surfacePageBytes: number;
  readonly ownerPageBytes: number;
  readonly frontierHashBytes: number;
  readonly compactAuxiliaryBytes: number;
  /** Negative is a net reduction relative to the existing dense host path. */
  readonly netBytes: number;
}

export interface OctreeTwoResolutionLevelSetAllocation {
  readonly fineFactor: FineLevelSetFactor;
  readonly brickResolution: FineLevelSetBrickResolution;
  readonly maximumResidentBricks: number;
  readonly finePayloadCapacityBytes: number;
  /** Dedicated signed-phi publication rollback snapshot, shared by A/B. */
  readonly fineRollbackCapacityBytes: number;
  readonly fineHashAndMetadataBytes: number;
  readonly fineWorklistBytes: number;
  readonly fineParameterBytes: number;
  readonly coarsePhiBytes: number;
  readonly allocatedBytes: number;
}

/** Exact capacity accounting for the Section 18 global fine/coarse phi pair. */
export function planOctreeTwoResolutionLevelSetAllocation(options: {
  dimensions: readonly [number, number, number];
  physicalCellWidth: number;
  rowCapacity: number;
  fineFactor: FineLevelSetFactor;
  brickResolution: FineLevelSetBrickResolution;
  maximumResidentBricks: number;
}): OctreeTwoResolutionLevelSetAllocation {
  const fine = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0], finestCellDimensions: options.dimensions,
    finestCellWidth: options.physicalCellWidth, fineFactor: options.fineFactor,
    brickResolution: options.brickResolution, maximumResidentBricks: options.maximumResidentBricks,
  });
  const coarse = planOctreeCoarsePhi(options.rowCapacity);
  const fineHashAndMetadataBytes = fine.pageTableBytes + fine.metadataCapacityBytes;
  const fineRollbackCapacityBytes = fine.maximumResidentBricks * fine.samplesPerBrick * 4;
  const fineParameterBytes = 2 * 80;
  return {
    fineFactor: options.fineFactor, brickResolution: options.brickResolution,
    maximumResidentBricks: options.maximumResidentBricks,
    finePayloadCapacityBytes: fine.payloadCapacityBytes, fineRollbackCapacityBytes,
    fineHashAndMetadataBytes, fineWorklistBytes: fine.worklistBytes, fineParameterBytes,
    coarsePhiBytes: coarse.allocatedBytes,
    allocatedBytes: fine.allocatedBytes + fineRollbackCapacityBytes + fineParameterBytes
      + coarse.allocatedBytes,
  };
}

/** Exact capacity accounting for the buffers introduced by compact transport. */
export function planOctreeCompactAllocation(
  dimensions: readonly [number, number, number],
  maximumLeafSize: number,
  interfaceBandCells: number,
  transport: GPUVelocityTransport = "maccormack",
  // Matches the measured target-scene authority budget (27.2% live + margin).
  surfaceResidentFraction = 0.32,
  surfacePageResolution: 2 | 4 = 2,
): OctreeCompactAllocationPlan {
  const [nx, ny, nz] = dimensions;
  const rowCapacity = planOctreePressureCapacity({ nx, ny, nz }, maximumLeafSize, interfaceBandCells).rowCapacity;
  const mirror = planOctreeFaceMirror(rowCapacity);
  const transfer = planOctreeFaceTopologyTransfer(mirror.faceCapacity);
  const faceTransport = planOctreeFaceTransport(mirror.faceCapacity);
  const surfaceAdapter = planOctreeSurfaceAdapter(rowCapacity);
  const surfacePages = planOctreeSurfacePages(rowCapacity, dimensions, {
    maximumResidentFraction: surfaceResidentFraction,
    pageResolution: surfacePageResolution,
  });
  const ownerPages = planOctreeOwnerPages(dimensions, {
    adaptiveBounds: {
      pressureRowCapacity: rowCapacity,
      surfacePageCapacity: surfacePages.pageCapacity,
    },
  });
  const frontier = planOctreeLeafFrontierAllocation(nx * ny * nz, rowCapacity, true);
  const denseHostBytesRemoved = planOctreeHostVelocityAllocation(nx, ny, nz, transport, true).savedBytes;
  const denseSurfaceBytesRemoved = planOctreeSurfaceStateAllocation(dimensions, true).savedBytes;
  const denseSnapshotBytesRemoved = planOctreePhiSnapshotAllocation({ nx, ny, nz }, true).savedBytes;
  const denseOwnerBytesRemoved = planOctreeOwnerAllocation(nx * ny * nz).legacyDenseBytes;
  const denseFrontierBytesRemoved = frontier.denseOriginMapBytes;
  const denseAtlasBytesRemoved = fluidBrickAtlasAllocatedBytes(planFluidBrickAtlas(dimensions, { brickSize: 8 }));
  const denseBytesRemoved = denseHostBytesRemoved + denseSurfaceBytesRemoved + denseSnapshotBytesRemoved
    + denseOwnerBytesRemoved + denseFrontierBytesRemoved + denseAtlasBytesRemoved;
  const compactAuxiliaryBytes = mirror.allocatedBytes + transfer.allocatedBytes
    + faceTransport.allocatedBytes + surfaceAdapter.allocatedBytes + surfacePages.allocatedBytes
    + ownerPages.allocatedBytes + 32 + frontier.rowMapBytes;
  return {
    rowCapacity,
    faceCapacity: mirror.faceCapacity,
    denseHostBytesRemoved, denseSurfaceBytesRemoved, denseSnapshotBytesRemoved, denseOwnerBytesRemoved,
    denseFrontierBytesRemoved, denseBytesRemoved,
    denseAtlasBytesRemoved,
    faceMirrorBytes: mirror.allocatedBytes,
    topologyTransferBytes: transfer.allocatedBytes,
    faceTransportBytes: faceTransport.allocatedBytes,
    surfaceAdapterBytes: surfaceAdapter.allocatedBytes,
    surfacePageBytes: surfacePages.allocatedBytes,
    ownerPageBytes: ownerPages.allocatedBytes + 32,
    frontierHashBytes: frontier.rowMapBytes,
    compactAuxiliaryBytes,
    netBytes: compactAuxiliaryBytes - denseBytesRemoved,
  };
}
