import type { GPUSolverInstance } from "../lib/methods/types";
import { damBreakFractions } from "../lib/initial-fluid";
import { boundingRadius, initializeRigidBodies, type RigidBodyState } from "../lib/rigid-body";
import type { SceneDescription } from "../lib/model";
import { VOXEL_MATERIAL_IDS } from "../lib/voxel-scene";
import {
  SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
  SparseVoxelDebugRenderer,
  type SparseVoxelRenderSource,
} from "../lib/webgpu-voxel-debug";
import {
  activeCubeCapacity,
  RasterWaterPipeline,
  surfaceVertexCapacity,
  type WaterSurfaceGeometrySource,
} from "../lib/webgpu-water-pipeline";
import { createGlobalFineLevelSetConsumerSource } from "../lib/octree-consumer-sampling";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import {
  FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
  unpackFineLevelSetGPURedistanceControl,
} from "../lib/webgpu-octree-fine-levelset-redistance";
import { ENVIRONMENT_VOXEL_MATERIAL_BASE } from "../lib/webgpu-octree-sparse-bricks";
import { environmentIndex } from "../lib/environments";
import {
  MAX_TERRAIN_FEATURES,
  TERRAIN_DEFAULT_FLAT,
  TERRAIN_UNION_EXPONENT,
  sceneHasTerrain,
} from "../lib/terrain";
import {
  compactOctreePublicationHeaderEvidence,
  reconstructCoarseOnlyOctreeOccupancyField,
  reconstructCompactOctreeOccupancyField,
  type CompactOctreeFieldEvidence,
} from "./webgpu-smoke-compact-field";
import { narrowVerticalSlitMetrics, type NarrowVerticalSlitMetrics } from "./raster-slit-metrics";
import {
  enclosedSurfaceHoleMetrics,
  surfaceStepMetrics,
  type EnclosedSurfaceHoleMetrics,
  type SurfaceStepMetrics,
} from "./raster-surface-metrics";
import {
  summarizeScalarField,
  summarizeTallCellActivity,
  type ScalarFieldSummary,
  type TallCellActivitySummary,
} from "./webgpu-smoke-scenarios";
import { unpackStructuredVelocityControl } from "./webgpu-smoke-structured-audit";
import {
  rasterizeStructuredCellVelocities,
  type CompactVelocityRaster,
} from "./webgpu-smoke-velocity-parity";

export async function readFloatTexture3D(device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height * depth);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(mapped.buffer, mapped.byteOffset + bytesPerRow * (y + height * z), width);
    output.set(row, width * (y + height * z));
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

export async function readBufferBinding(device: GPUDevice, binding: GPUBufferBinding, byteLength: number) {
  const alignedLength = Math.max(4, Math.ceil(byteLength / 4) * 4);
  const readback = device.createBuffer({ size: alignedLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Sparse voxel smoke readback" });
  encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0, readback, 0, alignedLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(byteLength);
  bytes.set(new Uint8Array(readback.getMappedRange(0, alignedLength)).subarray(0, byteLength));
  readback.unmap(); readback.destroy();
  return bytes;
}

/** Read the authoritative sparse fine phi and locate its upper zero crossing
 * in every coarse x/z column. Heights are returned in coarse-cell units. */
export async function readFineUpperSurfaceField(
  device: GPUDevice,
  solver: GPUSolverInstance,
  coarseDimensions: readonly [number, number, number],
): Promise<Float32Array | undefined> {
  const source = solver.globalFineLevelSetSource;
  if (!source) return undefined;
  const { plan } = source;
  if (plan.finestCellDimensions.some((value, axis) => value !== coarseDimensions[axis])) return undefined;
  const pageCapacity = plan.maximumResidentBricks;
  const payloadWords = pageCapacity * plan.samplesPerBrick;
  const [worklistBytes, metadataBytes, flagsBytes, phiBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, (7 + pageCapacity) * 4),
    readBufferBinding(device, { buffer: source.metadata }, pageCapacity * 40),
    readBufferBinding(device, { buffer: source.flags }, payloadWords * 4),
    readBufferBinding(device, { buffer: source.phi }, payloadWords * 4),
  ]);
  const worklist = new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4);
  const flags = new Uint32Array(flagsBytes.buffer, flagsBytes.byteOffset, flagsBytes.byteLength / 4);
  const phi = new Float32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4);
  const pages = new Map<number, number>();
  const activePages = Math.min(worklist[1] ?? 0, pageCapacity);
  for (let work = 0; work < activePages; work += 1) {
    const id = worklist[7 + work] ?? 0xffff_ffff;
    if (id >= pageCapacity || metadata[10 * id + 2] !== source.generation) continue;
    pages.set(metadata[10 * id + 1]!, id);
  }
  const [fineNx, fineNy] = plan.sampleDimensions;
  const brickResolution = plan.brickResolution;
  const sample = (qx: number, qy: number, qz: number): number | undefined => {
    const bx = Math.floor(qx / brickResolution), by = Math.floor(qy / brickResolution);
    const bz = Math.floor(qz / brickResolution);
    const key = bx + plan.brickDimensions[0] * (by + plan.brickDimensions[1] * bz);
    const id = pages.get(key);
    if (id === undefined) return undefined;
    const local = qx % brickResolution + brickResolution * ((qy % brickResolution)
      + brickResolution * (qz % brickResolution));
    const index = id * plan.samplesPerBrick + local;
    const value = phi[index];
    return (flags[index]! & 1) !== 0 && Number.isFinite(value) ? value : undefined;
  };
  const fineColumnHeight = (qx: number, qz: number): number | undefined => {
    let previousValue: number | undefined, previousY = -1;
    let highest: number | undefined;
    for (let qy = 0; qy < fineNy; qy += 1) {
      const value = sample(qx, qy, qz);
      if (value === undefined) { previousValue = undefined; previousY = -1; continue; }
      if (previousValue !== undefined && previousY + 1 === qy && previousValue < 0 && value >= 0) {
        const denominator = value - previousValue;
        const fraction = denominator > 0 ? -previousValue / denominator : 0;
        highest = previousY + 0.5 + fraction;
      }
      previousValue = value; previousY = qy;
    }
    if (previousY === fineNy - 1 && previousValue !== undefined && previousValue < 0) highest = fineNy;
    return highest;
  };
  const [nx, , nz] = coarseDimensions;
  const factor = plan.fineFactor;
  const result = new Float32Array(nx * nz); result.fill(Number.NaN);
  for (let k = 0; k < nz; k += 1) for (let i = 0; i < nx; i += 1) {
    let total = 0, count = 0;
    for (let oz = 0; oz < factor; oz += 1) for (let ox = 0; ox < factor; ox += 1) {
      const qx = factor * i + ox, qz = factor * k + oz;
      if (qx >= fineNx) continue;
      const height = fineColumnHeight(qx, qz);
      if (height === undefined) continue;
      total += height / factor; count += 1;
    }
    if (count > 0) result[i + nx * k] = total / count;
  }
  return result;
}

export async function readBufferBindingsPacked(
  device: GPUDevice,
  bindings: readonly { binding: GPUBufferBinding; byteLength: number }[],
): Promise<readonly Uint8Array[]> {
  const offsets: number[] = [];
  let packedLength = 0;
  for (const item of bindings) {
    packedLength = Math.ceil(packedLength / 4) * 4;
    offsets.push(packedLength);
    packedLength += Math.ceil(item.byteLength / 4) * 4;
  }
  const readback = device.createBuffer({
    label: "Final performance authority packed readback",
    size: Math.max(4, packedLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder({ label: "Final performance authority readback" });
    bindings.forEach((item, index) => {
      encoder.copyBufferToBuffer(
        item.binding.buffer,
        item.binding.offset ?? 0,
        readback,
        offsets[index],
        Math.ceil(item.byteLength / 4) * 4,
      );
    });
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(readback.getMappedRange());
    return bindings.map((item, index) => mapped.slice(offsets[index], offsets[index] + item.byteLength));
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

export interface SparseVoxelSmokeStats {
  voxelCount: number;
  brickCount: number;
  activeVoxelCount: number;
  activeBrickCount: number;
  fluidVoxelCount: number;
  environmentVoxelCount: number;
  materialVoxelCounts: Record<string, number>;
  nonFiniteRecordCount: number;
  invalidMaterialCount: number;
  fluidColorLinear: number[];
  uiRawVoxelRenderWall_ms: number;
  uiBrickGridRenderWall_ms: number;
  fluidBrickCapacity?: number;
  fluidBrickResidentCount?: number;
  fluidBrickCoreCount?: number;
  fluidBrickHaloCount?: number;
  fluidBrickActivatedCount?: number;
  fluidBrickRetiredCount?: number;
  fluidBrickGeneration?: number;
  fluidBrickCoreOrigins_m?: number[][];
  fluidBrickHaloOrigins_m?: number[][];
  sourceBrickFluidVoxelCount?: number;
  sourceBrickResidency?: "core" | "halo" | "vacant";
}

export interface FluidBrickSnapshot { resident: number; core: number; halo: number; generation: number }
export interface WorldBounds { min: [number, number, number]; max: [number, number, number] }

export function initialSeedBrickBounds(scene: SceneDescription, dimensions: readonly [number, number, number], brickSize = 8): WorldBounds | undefined {
  const seed = scene.fluid.initialBrickSeeds_m?.[0];
  if (!seed) return undefined;
  const minimum: [number, number, number] = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const extent: [number, number, number] = [scene.container.width_m, scene.container.height_m, scene.container.depth_m];
  const point = [seed.x, seed.y, seed.z];
  const start = point.map((value, axis) => {
    const cell = Math.max(0, Math.min(dimensions[axis] - 1, Math.floor((value - minimum[axis]) * dimensions[axis] / extent[axis])));
    return Math.floor(cell / brickSize) * brickSize;
  });
  return {
    min: start.map((cell, axis) => minimum[axis] + cell * extent[axis] / dimensions[axis]) as [number, number, number],
    max: start.map((cell, axis) => minimum[axis] + Math.min(dimensions[axis], cell + brickSize) * extent[axis] / dimensions[axis]) as [number, number, number]
  };
}

export async function readFluidBrickSnapshot(device: GPUDevice, source: SparseVoxelRenderSource): Promise<FluidBrickSnapshot | undefined> {
  if (!source.fluidBrickStats) return undefined;
  const words = new Uint32Array((await readBufferBinding(device, source.fluidBrickStats, 64)).buffer);
  return { resident: words[0], core: words[8], halo: words[9], generation: words[15] };
}

export async function smokeRenderSparseVoxelDebugModes(device: GPUDevice, source: SparseVoxelRenderSource) {
  const main = new SparseVoxelDebugRenderer(device, { colorFormat: "rgba8unorm" });
  await main.initialize();
  main.setSource(source);
  const color = device.createTexture({ size: [320, 180], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const depth = device.createTexture({ size: [320, 180], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const matrix = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  const renderMode = async (mode: "raw-voxels" | "brick-grid") => {
    const started = performance.now();
    const encoder = device.createCommandEncoder({ label: `Sparse voxel ${mode} WebGPU smoke` });
    main.encode(encoder, {
      mode,
      colorTarget: color.createView(), depthTarget: depth.createView(),
      colorLoadOp: "clear", depthLoadOp: "clear",
      viewProjection: matrix, cameraPosition: [0, 0, 4],
      containerBounds: { min: [-1, 0, -1], max: [1, 2, 1] },
      containerClosedTop: false
    });
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return performance.now() - started;
  };
  const uiRawVoxelRenderWall_ms = await renderMode("raw-voxels");
  const uiBrickGridRenderWall_ms = await renderMode("brick-grid");
  main.destroy(); color.destroy(); depth.destroy();
  return { uiRawVoxelRenderWall_ms, uiBrickGridRenderWall_ms };
}

export async function readSparseVoxelStats(device: GPUDevice, source: SparseVoxelRenderSource, sourceBrick?: WorldBounds): Promise<SparseVoxelSmokeStats> {
  const voxelCount = Math.min(new Uint32Array((await readBufferBinding(device, source.voxelCount, 4)).buffer)[0], source.voxelCapacity);
  const brickCount = Math.min(new Uint32Array((await readBufferBinding(device, source.brickCount, 4)).buffer)[0], source.brickCapacity);
  const voxelBytes = await readBufferBinding(device, source.voxelRecords, voxelCount * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
  const brickBytes = await readBufferBinding(device, source.brickRecords, brickCount * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
  const materialBytes = await readBufferBinding(device, source.materials, source.materialCount * 32);
  const voxelFloats = new Float32Array(voxelBytes.buffer), voxelWords = new Uint32Array(voxelBytes.buffer);
  const brickWords = new Uint32Array(brickBytes.buffer), brickFloats = new Float32Array(brickBytes.buffer), materialFloats = new Float32Array(materialBytes.buffer);
  let activeVoxelCount = 0, activeBrickCount = 0, fluidVoxelCount = 0, environmentVoxelCount = 0, nonFiniteRecordCount = 0, invalidMaterialCount = 0;
  let sourceBrickFluidVoxelCount = 0;
  const fluidBrickCoreOrigins_m: number[][] = [], fluidBrickHaloOrigins_m: number[][] = [];
  const materialVoxelCounts: Record<string, number> = {};
  for (let index = 0; index < voxelCount; index += 1) {
    const word = index * 12, material = voxelWords[word + 8], flags = voxelWords[word + 9];
    if ((flags & 1) === 0) continue;
    activeVoxelCount += 1;
    materialVoxelCounts[String(material)] = (materialVoxelCounts[String(material)] ?? 0) + 1;
    if (material === VOXEL_MATERIAL_IDS.fluid) {
      fluidVoxelCount += 1;
      const centre = [voxelFloats[word] + 0.5 * voxelFloats[word + 4], voxelFloats[word + 1] + 0.5 * voxelFloats[word + 5], voxelFloats[word + 2] + 0.5 * voxelFloats[word + 6]];
      if (sourceBrick && centre.every((value, axis) => value >= sourceBrick.min[axis] - 1e-6 && value < sourceBrick.max[axis] - 1e-6)) sourceBrickFluidVoxelCount += 1;
    }
    if (material >= ENVIRONMENT_VOXEL_MATERIAL_BASE) environmentVoxelCount += 1;
    if (material >= source.materialCount) invalidMaterialCount += 1;
    if (![...voxelFloats.slice(word, word + 3), ...voxelFloats.slice(word + 4, word + 7)].every(Number.isFinite)
      || voxelFloats[word + 4] <= 0 || voxelFloats[word + 5] <= 0 || voxelFloats[word + 6] <= 0) nonFiniteRecordCount += 1;
  }
  for (let index = 0; index < brickCount; index += 1) {
    const word = index * 12, flags = brickWords[word + 9];
    if ((flags & 1) !== 0) activeBrickCount += 1;
    const origin = () => Array.from(brickFloats.slice(word, word + 3));
    if ((flags & 2) !== 0) fluidBrickCoreOrigins_m.push(origin());
    else if ((flags & 4) !== 0) fluidBrickHaloOrigins_m.push(origin());
  }
  const colorOffset = VOXEL_MATERIAL_IDS.fluid * 8;
  const debugRenderTimings = await smokeRenderSparseVoxelDebugModes(device, source);
  const fluidBrickWords = source.fluidBrickStats
    ? new Uint32Array((await readBufferBinding(device, source.fluidBrickStats, 64)).buffer)
    : undefined;
  return {
    voxelCount, brickCount, activeVoxelCount, activeBrickCount, fluidVoxelCount, environmentVoxelCount, materialVoxelCounts,
    nonFiniteRecordCount, invalidMaterialCount,
    fluidColorLinear: Array.from(materialFloats.slice(colorOffset, colorOffset + 3)),
    ...debugRenderTimings,
    ...(fluidBrickWords ? {
      fluidBrickCapacity: source.fluidBrickCapacity,
      fluidBrickResidentCount: fluidBrickWords[0], fluidBrickCoreCount: fluidBrickWords[8], fluidBrickHaloCount: fluidBrickWords[9],
      fluidBrickActivatedCount: fluidBrickWords[10], fluidBrickRetiredCount: fluidBrickWords[11], fluidBrickGeneration: fluidBrickWords[15],
      fluidBrickCoreOrigins_m, fluidBrickHaloOrigins_m,
      sourceBrickFluidVoxelCount,
      sourceBrickResidency: fluidBrickCoreOrigins_m.some((origin) => origin.every((value, axis) => Math.abs(value - (sourceBrick?.min[axis] ?? Infinity)) <= 1e-5))
        ? "core" as const
        : fluidBrickHaloOrigins_m.some((origin) => origin.every((value, axis) => Math.abs(value - (sourceBrick?.min[axis] ?? Infinity)) <= 1e-5))
          ? "halo" as const
          : "vacant" as const,
    } : {})
  };
}

export interface HybridPresentationSmokeStats {
  initializeWall_ms: number;
  frameWall_ms: number;
  bodyCount: number;
  width: number;
  height: number;
  frontInterfacePixels: number;
  backInterfacePixels: number;
  pairedInterfacePixels: number;
  frontOnlyInterfacePixels: number;
  backOnlyInterfacePixels: number;
  /** First isolated screen-space witnesses where a back crossing has no front crossing. */
  backOnlyInterfaceLocations?: readonly (readonly [number, number])[];
  /** World-space back-interface positions for the corresponding witnesses. */
  backOnlyInterfacePositions_m?: readonly (readonly [number, number, number])[];
  frontInterfaceHash: number;
  backInterfaceHash: number;
  rendererValidationErrorCount: number;
  rendererUncapturedErrorCount: number;
  surfaceGeometrySource?: WaterSurfaceGeometrySource;
  globalFineAuthorityLatch?: number;
  globalFineCrossingPublished?: boolean;
  presentationFallbackActive?: boolean;
  vertexCount?: number;
  activeCubeCount?: number;
  vertexAllocator?: number;
  vertexCapacity?: number;
  activeCubeCapacity?: number;
  narrowVerticalSlits: NarrowVerticalSlitMetrics;
  enclosedSurfaceHoles: {
    front: EnclosedSurfaceHoleMetrics;
    back: EnclosedSurfaceHoleMetrics;
  };
  surfaceSteps: {
    front: SurfaceStepMetrics;
    back: SurfaceStepMetrics;
  };
  /** Visible fine-interface pixels lying on the authored ceiling plane. */
  ceilingContactPixels?: { front: number; back: number };
  /** Front-facing pixels lying on a side-wall cap within 0.4 fine cells of each x/z corner. */
  wallCornerCapPixels?: readonly [number, number, number, number];
  /** Highest visible interface point in each vertical x/z wall corner. */
  wallCornerMaximumY_m?: readonly [number, number, number, number];
  /** Pixels on the two exposed vertical dam faces next to their shared +x/+z corner. */
  damExposedCornerCapPixels?: readonly [number, number];
  frontInterfaceBounds_m?: readonly [readonly [number, number, number], readonly [number, number, number]];
  reverseView?: {
    frontInterfacePixels: number;
    backInterfacePixels: number;
    pairedInterfacePixels: number;
    frontOnlyInterfacePixels: number;
    backOnlyInterfacePixels: number;
    backOnlyInterfaceLocations?: readonly (readonly [number, number])[];
    backOnlyInterfacePositions_m?: readonly (readonly [number, number, number])[];
    frontInterfaceHash: number;
    backInterfaceHash: number;
    narrowVerticalSlits: NarrowVerticalSlitMetrics;
    enclosedSurfaceHoles: {
      front: EnclosedSurfaceHoleMetrics;
      back: EnclosedSurfaceHoleMetrics;
    };
    surfaceSteps: {
      front: SurfaceStepMetrics;
      back: SurfaceStepMetrics;
    };
    ceilingContactPixels?: { front: number; back: number };
    wallCornerCapPixels?: readonly [number, number, number, number];
    wallCornerMaximumY_m?: readonly [number, number, number, number];
    damExposedCornerCapPixels?: readonly [number, number];
    frontInterfaceBounds_m?: readonly [readonly [number, number, number], readonly [number, number, number]];
  };
  globalFineAuthorityTransition?: {
    validGeneration: number;
    unpublishedGeneration: number;
    cleanFineCoarseRequired: true;
    retainedGeometrySource?: WaterSurfaceGeometrySource;
    retainedFrontInterfacePixels: number;
    retainedBackInterfacePixels: number;
    retainedFrontInterfaceHash: number;
    retainedBackInterfaceHash: number;
    retainedFrontInterfaceBounds_m?: readonly [readonly [number, number, number], readonly [number, number, number]];
  };
}

export interface GlobalFineGenerationDiagnostics {
  generation: number;
  worklistGeneration?: number;
  generationSlot: number;
  activePages: number;
  configuredBrickCapacity: number;
  taggedMetadataPages: number;
  malformedActivePages: number;
  validSamples: number;
  finiteValidSamples: number;
  negativeValidSamples: number;
  positiveValidSamples: number;
  publicationValid: boolean;
  residentPayloadBytes: number;
  payloadCapacityBytes: number;
  payloadFragmentationBytes: number;
  pageMetadataBytes: number;
  pageWorklistBytes: number;
  diagnosticReadbackBytes: number;
  coarseState?: number;
  coarseGeneration?: number;
  coarseRowCount?: number;
  coarseMaximumLeafSize?: number;
  coarseEntryCount?: number;
  coarseNegativeEntries?: number;
  coarsePositiveEntries?: number;
  coarseInterfaceEntries?: number;
  coarseMalformedEntries?: number;
  seedCount?: number;
  seedFlags?: number;
  topologyFlags?: number;
  topologyInterfaceBricks?: number;
  topologyDesiredBricks?: number;
  topologyRequiredDesiredBricks?: number;
  topologyRequiredDesiredBricksExact?: boolean;
  topologyDilationBrickRings?: number;
  topologyActivatedBricks?: number;
  topologyPublished?: boolean;
  topologyRolledBack?: boolean;
  /** Downstream finalization reason mask: topology=1, redistance=2, volume=4, transport=8. */
  topologyFinalizeReason?: number;
  /** Sticky downstream reason mask across recovered/retained publications. */
  topologyLatchedFinalizeReason?: number;
  topologyFirstRejectedGeneration?: number;
  topologyRejectionCount?: number;
  phiBitXor: number;
  phiBitSum: number;
  phiSum: number;
  phiAbsSum: number;
  transportDepartureOutsideBand?: number;
  transportNonfiniteVelocity?: number;
  transportProcessed?: number;
  transportCommitted?: boolean;
  transportExtrapolatedVelocity?: number;
  transportMaximumDisplacementFineCells?: number;
  transportFaceBandUnavailable?: number;
  transportVelocityUnavailable?: number;
  redistanceUnresolvedCells?: number;
  redistanceMaximumResidualScaled?: number;
  redistanceSeedCount?: number;
  redistanceCommitted?: boolean;
  redistanceFlags?: number;
  redistanceFirstError?: number;
  redistanceAcceptedCells?: number;
  redistanceInitialPages?: number;
  redistanceFinalPages?: number;
  redistanceFrontierFloodPages?: number;
  redistanceFrontierSeedPages?: number;
  redistanceFrontierResolvePages?: number;
  redistanceFallbackPages?: number;
  redistanceFrontierMeasuredDisplacement?: number;
  redistanceFrontierFirstEnabledFloodPass?: number;
  topologyDirectValueChangePages?: number;
  topologyDirtyHaloPages?: number;
  topologySupportOnlyPages?: number;
  topologyGenerationRemapPages?: number;
  topologyMissingCurrentPages?: number;
  topologyConservativeFallbackPages?: number;
  volumeFlags?: number;
  volumeInitialized?: boolean;
  volumeSamples?: number;
  volumeReference?: number;
  volumeCurrent?: number;
  volumeInterfaceArea?: number;
  volumeCorrection?: number;
  volumeCorrected?: boolean;
  volumeCoarse?: number;
  volumeFine?: number;
  volumeReplacedCoarse?: number;
  volumeCoarseRows?: number;
  volumeUnowned?: number;
  volumeExpectedAir?: number;
  volumeLookupFailures?: number;
  volumeStaleOwners?: number;
  volumeGeneration?: number;
  probedPages?: Array<{
    key: number;
    directoryPhysicalId?: number;
    directoryFound: boolean;
    metadataKey?: number;
    metadataGeneration?: number;
    metadataMatchesGeneration: boolean;
    inPublishedWorklist: boolean;
    validSamples: number;
    finiteValidSamples: number;
    requiredCenterSamples?: Array<{ local: number; flags: number; phi: number | null }>;
    requiredCenterValid?: boolean;
    requiredCenterFinite?: boolean;
  }>;
  probedCoarseRecords?: Array<{
    cell: number; found: boolean; lookupSize: number; phi: number;
    minimumPhi: number; maximumPhi: number; flags: number;
  }>;
}

export function fineTrilinearBrickKeysAtPosition(
  source: WebGPUFineLevelSetBrickSource,
  position: readonly [number, number, number],
): number[] {
  const plan = source.plan;
  const lattice = position.map((value, axis) =>
    (value - plan.domainOrigin[axis]) / plan.fineCellWidth - 0.5) as [number, number, number];
  if (lattice.some((value) => !Number.isFinite(value))) return [];
  const base = lattice.map(Math.floor) as [number, number, number];
  const keys = new Set<number>();
  for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
    const q: [number, number, number] = [base[0] + x, base[1] + y, base[2] + z];
    if (q.some((value, axis) => value < 0 || value >= plan.sampleDimensions[axis])) continue;
    const brick = q.map((value) => Math.floor(value / plan.brickResolution)) as [number, number, number];
    keys.add(brick[0] + plan.brickDimensions[0] * (brick[1] + plan.brickDimensions[1] * brick[2]));
  }
  return [...keys].sort((a, b) => a - b);
}

export async function readGlobalFineGenerationDiagnostics(
  device: GPUDevice,
  solver: GPUSolverInstance,
  probeBrickKeys: readonly number[] = [],
  sourceOverride?: WebGPUFineLevelSetBrickSource,
): Promise<GlobalFineGenerationDiagnostics | undefined> {
  const source = sourceOverride ?? solver.globalFineLevelSetSource;
  if (!source) return undefined;
  const transportControl = (solver as GPUSolverInstance & { globalFineTransportControl?: GPUBuffer })
    .globalFineTransportControl;
  const redistanceControl = solver.globalFineRedistanceControl;
  const volumeControl = solver.globalFineVolumeControl;
  const pageDeltaDebug = solver.globalFinePageDeltaDebug;
  const pageCapacity = source.plan.maximumResidentBricks;
  const samplesPerBrick = source.plan.samplesPerBrick;
  const [worklistBytes, metadataBytes, flagBytes, phiBytes, coarseBytes, seedBytes, topologyBytes,
    transportBytes, redistanceBytes, volumeBytes, promotionBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, source.worklist.size),
    readBufferBinding(device, { buffer: source.metadata }, pageCapacity * 40),
    readBufferBinding(device, { buffer: source.flags }, pageCapacity * samplesPerBrick * 4),
    readBufferBinding(device, { buffer: source.phi }, pageCapacity * samplesPerBrick * 4),
    source.coarsePhiDirectory
      ? readBufferBinding(device, { buffer: source.coarsePhiDirectory },
        32 + (source.coarsePhiRowCapacity ?? 0) * 32)
      : Promise.resolve(undefined),
    source.seedControl ? readBufferBinding(device, { buffer: source.seedControl }, 8) : Promise.resolve(undefined),
    source.topologyControl
      // Words 10..11 are sticky rejection evidence; the first eight words are
      // cleared when the next generation begins, so a 32-byte read can hide a
      // rejected-but-recovered update.
      ? readBufferBinding(device, { buffer: source.topologyControl }, 48)
      : Promise.resolve(undefined),
    transportControl ? readBufferBinding(device, { buffer: transportControl }, 64) : Promise.resolve(undefined),
    redistanceControl
      ? readBufferBinding(device, { buffer: redistanceControl }, FINE_LEVELSET_REDISTANCE_CONTROL_BYTES)
      : Promise.resolve(undefined),
    volumeControl ? readBufferBinding(device, { buffer: volumeControl }, 64) : Promise.resolve(undefined),
    pageDeltaDebug ? readBufferBinding(device, {
      buffer: pageDeltaDebug.buffer, offset: pageDeltaDebug.promotionCountsOffsetWords * 4,
    }, 24) : Promise.resolve(undefined),
  ]);
  const worklist = new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4);
  const flags = new Uint32Array(flagBytes.buffer, flagBytes.byteOffset, flagBytes.byteLength / 4);
  const phi = new Float32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4);
  const phiBits = new Uint32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4);
  const activePages = Math.min(worklist[1], pageCapacity);
  let taggedMetadataPages = 0, malformedActivePages = 0;
  let validSamples = 0, finiteValidSamples = 0, negativeValidSamples = 0, positiveValidSamples = 0;
  let phiBitXor = 0, phiBitSum = 0, phiSum = 0, phiAbsSum = 0;
  for (let id = 0; id < pageCapacity; id += 1) if (metadata[id * 10 + 2] === source.generation) taggedMetadataPages += 1;
  for (let work = 0; work < activePages; work += 1) {
    const id = worklist[7 + work];
    if (id >= pageCapacity || metadata[id * 10 + 2] !== source.generation || metadata[id * 10] !== id) {
      malformedActivePages += 1; continue;
    }
    for (let local = 0; local < samplesPerBrick; local += 1) {
      const index = id * samplesPerBrick + local;
      if ((flags[index] & 1) === 0) continue;
      validSamples += 1;
      const value = phi[index];
      if (!Number.isFinite(value)) continue;
      finiteValidSamples += 1;
      const logicalSample = (Math.imul(metadata[id * 10 + 1], samplesPerBrick) + local) >>> 0;
      let mixed = Math.imul((phiBits[index] ^ logicalSample) >>> 0, 0x7feb_352d) >>> 0;
      mixed = Math.imul((mixed ^ (mixed >>> 15)) >>> 0, 0x846c_a68b) >>> 0;
      mixed = (mixed ^ (mixed >>> 16)) >>> 0;
      phiBitXor = (phiBitXor ^ mixed) >>> 0; phiBitSum = (phiBitSum + mixed) >>> 0;
      phiSum += value; phiAbsSum += Math.abs(value);
      if (value < 0) negativeValidSamples += 1; else positiveValidSamples += 1;
    }
  }
  const probedPages: GlobalFineGenerationDiagnostics["probedPages"] = [];
  if (probeBrickKeys.length > 0) {
    const publishedIds = new Set<number>();
    for (let work = 0; work < activePages; work += 1) publishedIds.add(worklist[7 + work]);
    const lookup = (key: number) => {
      if (worklist.length < 7 || worklist[0] !== source.generation
        || worklist[2] !== pageCapacity || (worklist[3] & 3) !== 3
        || worklist[5] !== 1 || worklist[6] !== 1) return undefined;
      const directoryBase = 7 + pageCapacity;
      if (key >= source.plan.logicalBrickCount || directoryBase + key >= worklist.length) return undefined;
      const id = worklist[directoryBase + key], base = id * 10;
      return id < pageCapacity && base + 2 < metadata.length && metadata[base] === id
        && metadata[base + 1] === key && metadata[base + 2] === source.generation ? id : undefined;
    };
    const requiredLocals = source.plan.fineFactor === 4 && source.plan.brickResolution === 4
      ? [21, 22, 25, 26, 37, 38, 41, 42] : undefined;
    for (const key of probeBrickKeys) {
      const id = lookup(key); const validId = id !== undefined && id < pageCapacity;
      let pageValid = 0, pageFinite = 0;
      if (validId) for (let local = 0; local < samplesPerBrick; local += 1) {
        const index = id * samplesPerBrick + local;
        if ((flags[index] & 1) === 0) continue;
        pageValid += 1; if (Number.isFinite(phi[index])) pageFinite += 1;
      }
      const requiredCenterSamples = validId && requiredLocals
        ? requiredLocals.map((local) => {
          const index = id * samplesPerBrick + local, value = phi[index];
          return { local, flags: flags[index], phi: Number.isFinite(value) ? value : null };
        }) : undefined;
      probedPages.push({
        key, directoryPhysicalId: id, directoryFound: validId,
        metadataKey: validId ? metadata[id * 10 + 1] : undefined,
        metadataGeneration: validId ? metadata[id * 10 + 2] : undefined,
        metadataMatchesGeneration: validId && metadata[id * 10] === id
          && metadata[id * 10 + 1] === key && metadata[id * 10 + 2] === source.generation,
        inPublishedWorklist: validId && publishedIds.has(id),
        validSamples: pageValid, finiteValidSamples: pageFinite,
        requiredCenterSamples,
        requiredCenterValid: requiredCenterSamples?.every((sample) => (sample.flags & 1) !== 0),
        requiredCenterFinite: requiredCenterSamples?.every((sample) => sample.phi !== null),
      });
    }
  }
  const coarse = coarseBytes
    ? new Uint32Array(coarseBytes.buffer, coarseBytes.byteOffset, coarseBytes.byteLength / 4)
    : undefined;
  let coarseEntryCount = 0, coarseNegativeEntries = 0, coarsePositiveEntries = 0;
  let coarseInterfaceEntries = 0, coarseMalformedEntries = 0;
  if (coarse) for (let slot = 0; slot < Math.min(coarse[2], (coarse.length - 8) / 8); slot += 1) {
    const base = 8 + slot * 8;
    if (coarse[base] === 0) continue;
    coarseEntryCount += 1;
    const values = new Float32Array(coarse.buffer, coarse.byteOffset + (base + 2) * 4, 3);
    const [phiValue, minimumPhi, maximumPhi] = values;
    if (!Number.isFinite(phiValue) || !Number.isFinite(minimumPhi) || !Number.isFinite(maximumPhi)
      || minimumPhi > phiValue || phiValue > maximumPhi || (coarse[base + 5] & 9) !== 9) {
      coarseMalformedEntries += 1; continue;
    }
    if (minimumPhi < 0) coarseNegativeEntries += 1; else coarsePositiveEntries += 1;
    if (minimumPhi <= 0 && maximumPhi >= 0) coarseInterfaceEntries += 1;
  }
  const probedCoarseRecords: GlobalFineGenerationDiagnostics["probedCoarseRecords"] = [];
  if (coarse && probeBrickKeys.length > 0) {
    const rowCount = Math.min(coarse[2], (coarse.length - 8) / 8);
    const maximumLeaf = coarse[3], dimensions = [coarse[4], coarse[5], coarse[6]];
    const physicalCellSize = new Float32Array(coarse.buffer, coarse.byteOffset + 7 * 4, 1)[0];
    const mortonPart = (input: number) => {
      let value = input & 1023;
      value = (value | (value << 16)) & 0x030000ff; value = (value | (value << 8)) & 0x0300f00f;
      value = (value | (value << 4)) & 0x030c30c3; value = (value | (value << 2)) & 0x09249249;
      return value >>> 0;
    };
    const morton = (cell: number) => {
      const x = cell % dimensions[0], y = Math.floor(cell / dimensions[0]) % dimensions[1];
      const z = Math.floor(cell / (dimensions[0] * dimensions[1]));
      return (mortonPart(x) | (mortonPart(y) << 1) | (mortonPart(z) << 2)) >>> 0;
    };
    for (const requested of probeBrickKeys) {
      const q = [requested % dimensions[0], Math.floor(requested / dimensions[0]) % dimensions[1],
        Math.floor(requested / (dimensions[0] * dimensions[1]))];
      let foundBase = -1, lookupSize = 0;
      for (let size = 1; size <= maximumLeaf; size *= 2) {
        const origin = q.map((value) => Math.floor(value / size) * size);
        const cell = origin[0] + dimensions[0] * (origin[1] + dimensions[1] * origin[2]);
        const wantedLevel = 31 - Math.clz32(size), wantedMorton = morton(cell);
        let low = 0, high = rowCount;
        while (low < high) {
          const middle = low + Math.floor((high - low) / 2), base = 8 + middle * 8;
          const entryLevel = 31 - Math.clz32(coarse[base + 1]), entryMorton = morton(coarse[base] - 1);
          if (entryLevel < wantedLevel || (entryLevel === wantedLevel && entryMorton < wantedMorton)) low = middle + 1;
          else high = middle;
        }
        if (low < rowCount) {
          const base = 8 + low * 8;
          if (coarse[base] === cell + 1 && coarse[base + 1] === size) {
            foundBase = base; lookupSize = size;
          }
        }
        if (foundBase >= 0) break;
      }
      if (foundBase >= 0) {
        const values = new Float32Array(coarse.buffer, coarse.byteOffset + (foundBase + 2) * 4, 3);
        probedCoarseRecords.push({ cell: requested, found: true, lookupSize,
          phi: values[0], minimumPhi: values[1], maximumPhi: values[2], flags: coarse[foundBase + 5] });
      } else {
        const air = physicalCellSize * maximumLeaf;
        probedCoarseRecords.push({ cell: requested, found: false, lookupSize: 0,
          phi: air, minimumPhi: air, maximumPhi: air, flags: 0 });
      }
    }
  }
  const seed = seedBytes
    ? new Uint32Array(seedBytes.buffer, seedBytes.byteOffset, seedBytes.byteLength / 4)
    : undefined;
  const topology = topologyBytes
    ? new Uint32Array(topologyBytes.buffer, topologyBytes.byteOffset, topologyBytes.byteLength / 4)
    : undefined;
  const transport = transportBytes
    ? new Uint32Array(transportBytes.buffer, transportBytes.byteOffset, transportBytes.byteLength / 4)
    : undefined;
  const redistance = redistanceBytes
    ? new Uint32Array(redistanceBytes.buffer, redistanceBytes.byteOffset, redistanceBytes.byteLength / 4)
    : undefined;
  const redistanceDiagnostics = redistance ? unpackFineLevelSetGPURedistanceControl(redistance) : undefined;
  const volume = volumeBytes
    ? new Uint32Array(volumeBytes.buffer, volumeBytes.byteOffset, volumeBytes.byteLength / 4)
    : undefined;
  const volumeFloats = volumeBytes
    ? new Float32Array(volumeBytes.buffer, volumeBytes.byteOffset, volumeBytes.byteLength / 4)
    : undefined;
  const promotion = promotionBytes
    ? new Uint32Array(promotionBytes.buffer, promotionBytes.byteOffset, 6) : undefined;
  return {
    generation: source.generation, worklistGeneration: worklist[0], generationSlot: source.generationSlot, activePages,
    configuredBrickCapacity: pageCapacity,
    taggedMetadataPages, malformedActivePages, validSamples, finiteValidSamples,
    negativeValidSamples, positiveValidSamples, phiBitXor, phiBitSum, phiSum, phiAbsSum,
    residentPayloadBytes: activePages * source.plan.payloadBytesPerBrick,
    payloadCapacityBytes: source.plan.payloadCapacityBytes,
    payloadFragmentationBytes: (pageCapacity - activePages) * source.plan.payloadBytesPerBrick,
    pageMetadataBytes: pageCapacity * 40,
    pageWorklistBytes: source.worklist.size,
    diagnosticReadbackBytes: [worklistBytes, metadataBytes, flagBytes, phiBytes, coarseBytes, seedBytes,
      topologyBytes, transportBytes, redistanceBytes, volumeBytes]
      .concat(promotionBytes ? [promotionBytes] : [])
      .reduce((sum, bytes) => sum + (bytes?.byteLength ?? 0), 0),
    publicationValid: worklist[0] === source.generation && worklist[2] === pageCapacity
      && (worklist[3] & 3) === 3 && worklist[5] === 1 && worklist[6] === 1
      && activePages > 0 && taggedMetadataPages >= activePages && malformedActivePages === 0
      && validSamples > 0 && finiteValidSamples === validSamples,
    ...(probedPages.length > 0 ? { probedPages } : {}),
    ...(probedCoarseRecords.length > 0 ? { probedCoarseRecords } : {}),
    ...(coarse ? { coarseState: coarse[0], coarseGeneration: coarse[1],
      coarseRowCount: coarse[2], coarseMaximumLeafSize: coarse[3], coarseEntryCount,
      coarseNegativeEntries, coarsePositiveEntries, coarseInterfaceEntries, coarseMalformedEntries } : {}),
    ...(seed ? { seedCount: seed[0], seedFlags: seed[1] } : {}),
    ...(topology ? { topologyFlags: topology[0], topologyInterfaceBricks: topology[1],
      topologyDesiredBricks: topology[2], topologyActivatedBricks: topology[3],
      topologyPublished: topology[4] !== 0, topologyRolledBack: topology[5] !== 0,
      topologyFinalizeReason: topology[7],
      topologyLatchedFinalizeReason: topology[10],
      topologyFirstRejectedGeneration: topology[11] >>> 16,
      topologyRejectionCount: topology[11] & 0xffff,
      topologyRequiredDesiredBricks: (topology[0] & 1) !== 0 ? topology[6] : topology[2],
      topologyRequiredDesiredBricksExact: (topology[0] & 1) === 0,
      topologyDilationBrickRings: topology[0] === 0 ? topology[6] : 0 } : {}),
    ...(transport ? { transportDepartureOutsideBand: transport[0], transportNonfiniteVelocity: transport[1],
      transportProcessed: transport[2], transportCommitted: transport[3] !== 0,
      transportExtrapolatedVelocity: transport[4],
      transportMaximumDisplacementFineCells: transport[5], transportFaceBandUnavailable: transport[6],
      transportVelocityUnavailable: transport[7] } : {}),
    ...(redistanceDiagnostics ? {
      redistanceUnresolvedCells: redistanceDiagnostics.unresolvedCells,
      redistanceResolveMissingCells: redistanceDiagnostics.resolveMissingCells,
      redistanceResidualViolationCells: redistanceDiagnostics.residualViolationCells,
      redistanceMaximumResidualScaled: redistanceDiagnostics.maximumResidualScaled,
      redistanceSeedCount: redistanceDiagnostics.seedCount,
      redistanceCommitted: redistanceDiagnostics.committed,
      redistanceFlags: redistanceDiagnostics.flags,
      redistanceFirstError: redistanceDiagnostics.firstError,
      redistanceAcceptedCells: redistanceDiagnostics.acceptedCells,
      redistanceInitialPages: redistanceDiagnostics.initialPages,
      redistanceFinalPages: redistanceDiagnostics.finalPages,
      redistanceFrontierFloodPages: redistanceDiagnostics.frontierFloodPages,
      redistanceFrontierSeedPages: redistanceDiagnostics.frontierSeedPages,
      redistanceFrontierResolvePages: redistanceDiagnostics.frontierResolvePages,
      redistanceFallbackPages: redistanceDiagnostics.fallbackPages,
      redistanceFrontierMeasuredDisplacement: redistanceDiagnostics.frontierMeasuredDisplacement,
      redistanceFrontierFirstEnabledFloodPass: redistanceDiagnostics.frontierFirstEnabledFloodPass,
    } : {}),
    ...(promotion ? { topologyDirectValueChangePages: promotion[0],
      topologyDirtyHaloPages: promotion[1], topologySupportOnlyPages: promotion[2],
      topologyGenerationRemapPages: promotion[3], topologyMissingCurrentPages: promotion[4],
      topologyConservativeFallbackPages: promotion[5] } : {}),
    ...(volume && volumeFloats ? { volumeFlags: volume[0], volumeInitialized: volume[1] !== 0,
      volumeSamples: volume[2], volumeReference: volumeFloats[3], volumeCurrent: volumeFloats[4],
      volumeInterfaceArea: volumeFloats[5], volumeCorrection: volumeFloats[6],
      volumeCorrected: volume[7] !== 0, volumeCoarse: volumeFloats[8], volumeFine: volumeFloats[9],
      volumeReplacedCoarse: volumeFloats[10], volumeCoarseRows: volume[11], volumeUnowned: volume[12],
      volumeExpectedAir: volume[12], volumeGeneration: volume[13],
      volumeLookupFailures: volume[14], volumeStaleOwners: volume[15] } : {}),
  };
}

export function decodeFloat16(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Infinity : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export function hybridPresentationBodies(scene: SceneDescription, bodies: RigidBodyState[]): RigidBodyState[] {
  if (bodies.length > 0) return bodies;
  const scale = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  return initializeRigidBodies([{
    id: "hybrid-render-smoke-body", name: "Hybrid render smoke body", shape: "box",
    dimensions_m: { x: 0.18 * scale, y: 0.22 * scale, z: 0.16 * scale }, density_kg_m3: 700,
    position_m: { x: 0.18 * scene.container.width_m, y: 0.36 * scene.container.height_m, z: 0 },
    orientation: { w: 1, x: 0, y: 0, z: 0 },
    linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    restitution: 0.2, friction: 0.5, motion: "static"
  }]);
}

export async function smokeRenderHybridPresentation(
  device: GPUDevice,
  solver: GPUSolverInstance,
  scene: SceneDescription,
  bodies: RigidBodyState[],
  verifyGlobalFineAuthorityTransition = false,
): Promise<HybridPresentationSmokeStats> {
  // Match the UI's practical pixel density closely enough that a one-pixel
  // slit there cannot disappear through smoke-test undersampling.
  const width = 640, height = 360;
  const uniformBuffer = device.createBuffer({ label: "Hybrid presentation smoke uniforms", size: 400, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bodyBuffer = device.createBuffer({ label: "Hybrid presentation smoke bodies", size: 12 * 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const output = device.createTexture({ label: "Hybrid presentation smoke output", size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT });
  const columnFallback = device.createTexture({
    label: "Hybrid presentation non-column fallback",
    size: [1, 1],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING,
  });
  const presentationBodies = hybridPresentationBodies(scene, bodies).slice(0, 12);
  const span = Math.max(scene.container.width_m, scene.container.height_m, scene.container.depth_m);
  const packed = new Float32Array(100);
  packed.set([width, height, solver.info.submittedTime_s ?? 0, 0], 0);
  packed.set([1.55 * span, 1.12 * span, 1.72 * span, 0], 4);
  packed.set([0, 0.38 * scene.container.height_m, 0, 0], 8);
  packed.set([scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction], 12);
  packed.set([0, scene.voxelDomain.finestCellSize_m, presentationBodies.length, 0], 16);
  packed.set([solver.info.nx, solver.info.ny, solver.info.nz, solver.info.gridKind === "restricted-tall-cell" ? 2 : solver.info.gridKind === "quadtree-tall-cell" || solver.info.gridKind === "octree" ? 3 : 1], 20);
  packed.set([0, 0.5, 0, 0], 24);
  packed.set([environmentIndex(scene.environment ?? "default"), solver.info.lastDt_s ?? 0, solver.info.maxSpeed_m_s ?? 0, 0], 28);
  if (sceneHasTerrain(scene) && scene.terrain) {
    const features = scene.terrain.features.slice(0, MAX_TERRAIN_FEATURES);
    packed.set([1, scene.terrain.baseHeight_m, features.length, TERRAIN_UNION_EXPONENT], 32);
    features.forEach((feature, index) => {
      packed.set([feature.center_m.x, feature.center_m.z, feature.radius_m.x, feature.radius_m.z], 36 + index * 8);
      packed.set([(feature.kind === "mound" ? 1 : -1) * feature.amount_m, feature.rotation_rad ?? 0, feature.flat ?? TERRAIN_DEFAULT_FLAT, 0], 40 + index * 8);
    });
  }
  device.queue.writeBuffer(uniformBuffer, 0, packed);
  const bodyData = new Float32Array(12 * 16);
  const shapeIndex = { sphere: 0, box: 1, capsule: 2, cylinder: 3 } as const;
  const palette = [[0.95, 0.63, 0.29], [0.48, 0.66, 0.96], [0.84, 0.42, 0.48], [0.66, 0.52, 0.92]];
  presentationBodies.forEach((body, index) => {
    const offset = index * 16, d = body.description.dimensions_m;
    const half = body.description.shape === "box" ? [d.x / 2, d.y / 2, d.z / 2] : body.description.shape === "sphere" ? [d.x, d.x, d.x] : [d.x, d.y / 2, d.x];
    const color = palette[shapeIndex[body.description.shape]];
    bodyData.set([body.position_m.x, body.position_m.y, body.position_m.z, boundingRadius(body)], offset);
    bodyData.set([half[0], half[1], half[2], shapeIndex[body.description.shape]], offset + 4);
    bodyData.set([body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z], offset + 8);
    bodyData.set([color[0], color[1], color[2], 0], offset + 12);
  });
  device.queue.writeBuffer(bodyBuffer, 0, bodyData);
  const pipeline = new RasterWaterPipeline(device, "rgba8unorm", uniformBuffer, bodyBuffer);
  const uncapturedRendererErrors: string[] = [];
  const onUncapturedRendererError = (event: Event) => {
    const error = (event as Event & { error?: GPUError }).error;
    uncapturedRendererErrors.push(error?.message ?? "unknown uncaptured WebGPU renderer error");
  };
  device.addEventListener("uncapturederror", onUncapturedRendererError);
  device.pushErrorScope("validation");
  let rendererValidationScopeActive = true;
  try {
    const initializeStarted = performance.now();
    await pipeline.initialize();
    const initializeWall_ms = performance.now() - initializeStarted;
    const globalFineLevelSet = solver.globalFineLevelSetSource
      ? createGlobalFineLevelSetConsumerSource(solver.globalFineLevelSetSource)
      : undefined;
    if (verifyGlobalFineAuthorityTransition && !globalFineLevelSet) {
      throw new Error("Global-fine authority transition requested without a published source");
    }
    pipeline.setVolume(solver.surfaceFieldTexture ?? solver.volumeTexture,
      solver.columnBaseTexture ?? columnFallback);
    pipeline.setGlobalFineLevelSet(globalFineLevelSet);
    pipeline.setCoarseLevelSet(solver.coarseLevelSetSource);
    pipeline.ensureSize(width, height);
    const capture = async (label: string, revision: number) => {
      const frameStarted = performance.now();
      const encoder = device.createCommandEncoder({ label });
      const encoded = pipeline.encode(
        encoder, output.createView(), solver.info.nx, solver.info.ny, solver.info.nz,
        solver.info.gridKind === "restricted-tall-cell", solver.info.maximumNeighborDelta ?? 0,
        revision
      );
      if (!encoded) throw new Error("Hybrid presentation pipeline did not encode a frame");
      const interfaceCapture = pipeline.diagnosticCaptureTexture("interface-positions");
      if (!interfaceCapture) throw new Error("Hybrid presentation did not expose its front interface target");
      const backInterfaceCapture = pipeline.diagnosticCaptureTexture("back-interface-positions");
      if (!backInterfaceCapture) throw new Error("Hybrid presentation did not expose its back interface target");
      const interfaceBytesPerRow = Math.ceil(width * 8 / 256) * 256;
      const interfacePlaneBytes = interfaceBytesPerRow * height;
      const interfaceReadback = device.createBuffer({ size: 2 * interfacePlaneBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        encoder.copyTextureToBuffer({ texture: interfaceCapture.texture }, { buffer: interfaceReadback, bytesPerRow: interfaceBytesPerRow, rowsPerImage: height }, [width, height]);
        encoder.copyTextureToBuffer({ texture: backInterfaceCapture.texture }, { buffer: interfaceReadback, offset: interfacePlaneBytes, bytesPerRow: interfaceBytesPerRow, rowsPerImage: height }, [width, height]);
        device.queue.submit([encoder.finish()]);
        const presentationDiagnostics = await pipeline.completeSurfaceDiagnostics();
        await device.queue.onSubmittedWorkDone();
        await interfaceReadback.mapAsync(GPUMapMode.READ);
        const interfaceWords = new Uint16Array(interfaceReadback.getMappedRange());
        const interfaceRowWords = interfaceBytesPerRow / 2;
        let frontInterfacePixels = 0, backInterfacePixels = 0, pairedInterfacePixels = 0;
        let frontCeilingContactPixels = 0, backCeilingContactPixels = 0;
        let frontOnlyInterfacePixels = 0, backOnlyInterfacePixels = 0;
        const backOnlyInterfaceLocations: [number, number][] = [];
        const backOnlyInterfacePositions_m: [number, number, number][] = [];
        let frontInterfaceHash = 0x811c_9dc5, backInterfaceHash = 0x811c_9dc5;
        const fold = (hash: number, value: number) => Math.imul((hash ^ value) >>> 0, 0x0100_0193) >>> 0;
        const frontMinimum: [number, number, number] = [Infinity, Infinity, Infinity];
        const frontMaximum: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        const wallCornerCapPixels: [number, number, number, number] = [0, 0, 0, 0];
        const wallCornerMaximumY_m: [number, number, number, number] = [0, 0, 0, 0];
        const damExposedCornerCapPixels: [number, number] = [0, 0];
        const fineCellWidth = globalFineLevelSet?.fineCellWidth
          ?? solver.coarseLevelSetSource?.physicalCellSize
          ?? scene.voxelDomain.finestCellSize_m;
        const wallPlaneTolerance = Math.max(5e-4, 0.08 * fineCellWidth);
        const cornerTangentialBand = 0.4 * fineCellWidth;
        const dam = damBreakFractions(scene.container.fillFraction);
        const damMaximum = [
          -0.5 * scene.container.width_m + dam.width * scene.container.width_m,
          dam.height * scene.container.height_m,
          -0.5 * scene.container.depth_m + dam.depth * scene.container.depth_m,
        ] as const;
        const wallCorners = [
          [-0.5 * scene.container.width_m, -0.5 * scene.container.depth_m],
          [-0.5 * scene.container.width_m, 0.5 * scene.container.depth_m],
          [0.5 * scene.container.width_m, -0.5 * scene.container.depth_m],
          [0.5 * scene.container.width_m, 0.5 * scene.container.depth_m],
        ] as const;
        const frontMask = new Uint8Array(width * height);
        const backMask = new Uint8Array(width * height);
        const frontPositions = new Float32Array(width * height * 3);
        const backPositions = new Float32Array(width * height * 3);
        frontPositions.fill(Number.NaN);
        backPositions.fill(Number.NaN);
        for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
          const at = y * interfaceRowWords + x * 4;
          const backAt = interfacePlaneBytes / 2 + at;
          for (let channel = 0; channel < 4; channel += 1) {
            frontInterfaceHash = fold(frontInterfaceHash, interfaceWords[at + channel]);
            backInterfaceHash = fold(backInterfaceHash, interfaceWords[backAt + channel]);
          }
          const frontPresent = interfaceWords[at + 3] !== 0;
          const backPresent = interfaceWords[backAt + 3] !== 0;
          if (frontPresent) {
            frontInterfacePixels += 1;
            frontMask[x + y * width] = 1;
            const px = decodeFloat16(interfaceWords[at]);
            const py = decodeFloat16(interfaceWords[at + 1]);
            const pz = decodeFloat16(interfaceWords[at + 2]);
            if (Math.abs(py - scene.container.height_m) <= wallPlaneTolerance) {
              frontCeilingContactPixels += 1;
            }
            frontPositions.set([px, py, pz], (x + y * width) * 3);
            for (let corner = 0; corner < wallCorners.length; corner += 1) {
              const dx = Math.abs(px - wallCorners[corner][0]);
              const dz = Math.abs(pz - wallCorners[corner][1]);
              if ((dx <= wallPlaneTolerance && dz <= cornerTangentialBand)
                || (dz <= wallPlaneTolerance && dx <= cornerTangentialBand)) {
                wallCornerCapPixels[corner] += 1;
                wallCornerMaximumY_m[corner] = Math.max(wallCornerMaximumY_m[corner], py);
              }
            }
            if (scene.fluid.initialCondition === "dam-break"
              && py >= fineCellWidth && py <= damMaximum[1] - fineCellWidth) {
              const damDx = Math.abs(px - damMaximum[0]);
              const damDz = Math.abs(pz - damMaximum[2]);
              if (damDx <= wallPlaneTolerance && damDz <= cornerTangentialBand) damExposedCornerCapPixels[0] += 1;
              if (damDz <= wallPlaneTolerance && damDx <= cornerTangentialBand) damExposedCornerCapPixels[1] += 1;
            }
            for (let axis = 0; axis < 3; axis += 1) {
              const value = decodeFloat16(interfaceWords[at + axis]);
              frontMinimum[axis] = Math.min(frontMinimum[axis], value);
              frontMaximum[axis] = Math.max(frontMaximum[axis], value);
            }
          }
          if (backPresent) {
            backInterfacePixels += 1;
            backMask[x + y * width] = 1;
            const backY = decodeFloat16(interfaceWords[backAt + 1]);
            if (Math.abs(backY - scene.container.height_m) <= wallPlaneTolerance) {
              backCeilingContactPixels += 1;
            }
            backPositions.set([
              decodeFloat16(interfaceWords[backAt]),
              backY,
              decodeFloat16(interfaceWords[backAt + 2]),
            ], (x + y * width) * 3);
          }
          if (frontPresent && backPresent) pairedInterfacePixels += 1;
          else if (frontPresent) frontOnlyInterfacePixels += 1;
          else if (backPresent) {
            backOnlyInterfacePixels += 1;
            if (backOnlyInterfaceLocations.length < 16) {
              backOnlyInterfaceLocations.push([x, y]);
              backOnlyInterfacePositions_m.push([
                decodeFloat16(interfaceWords[backAt]),
                decodeFloat16(interfaceWords[backAt + 1]),
                decodeFloat16(interfaceWords[backAt + 2]),
              ]);
            }
          }
        }
        const narrowVerticalSlits = narrowVerticalSlitMetrics(frontMask, width, height);
        const enclosedSurfaceHoles = {
          front: enclosedSurfaceHoleMetrics(frontMask, width, height),
          back: enclosedSurfaceHoleMetrics(backMask, width, height),
        };
        const surfaceSteps = {
          front: surfaceStepMetrics(frontMask, frontPositions, width, height, fineCellWidth),
          back: surfaceStepMetrics(backMask, backPositions, width, height, fineCellWidth),
        };
        interfaceReadback.unmap();
        return { initializeWall_ms, frameWall_ms: performance.now() - frameStarted,
          bodyCount: presentationBodies.length, width, height, frontInterfacePixels, backInterfacePixels,
          pairedInterfacePixels, frontOnlyInterfacePixels, backOnlyInterfacePixels,
          backOnlyInterfaceLocations, backOnlyInterfacePositions_m,
          frontInterfaceHash, backInterfaceHash,
          narrowVerticalSlits,
          enclosedSurfaceHoles,
          surfaceSteps,
          ceilingContactPixels: { front: frontCeilingContactPixels, back: backCeilingContactPixels },
          wallCornerCapPixels,
          wallCornerMaximumY_m,
          damExposedCornerCapPixels,
          ...(presentationDiagnostics ? {
            surfaceGeometrySource: presentationDiagnostics.surfaceGeometrySource,
            globalFineAuthorityLatch: presentationDiagnostics.globalFineAuthorityLatch,
            globalFineCrossingPublished: presentationDiagnostics.globalFineCrossingPublished,
            presentationFallbackActive: presentationDiagnostics.presentationFallbackActive,
            vertexCount: presentationDiagnostics.vertexCount,
            activeCubeCount: presentationDiagnostics.activeCubeCount,
            vertexAllocator: presentationDiagnostics.vertexAllocator,
            vertexCapacity: surfaceVertexCapacity(...(globalFineLevelSet?.sampleDimensions
              ?? [solver.info.nx, solver.info.ny, solver.info.nz])),
            activeCubeCapacity: activeCubeCapacity(surfaceVertexCapacity(...(globalFineLevelSet?.sampleDimensions
              ?? [solver.info.nx, solver.info.ny, solver.info.nz]))),
          } : {}),
          ...(frontInterfacePixels > 0 ? { frontInterfaceBounds_m: [frontMinimum, frontMaximum] as const } : {}) };
      } finally {
        interfaceReadback.destroy();
      }
    };
    const revision = solver.info.encodedSteps ?? 0;
    const validA = await capture("Hybrid smooth WebGPU smoke", revision);
    let globalFineAuthorityTransition: HybridPresentationSmokeStats["globalFineAuthorityTransition"];
    if (verifyGlobalFineAuthorityTransition && globalFineLevelSet) {
      const unpublishedGeneration = globalFineLevelSet.generation + 1;
      pipeline.setGlobalFineLevelSet({ ...globalFineLevelSet, generation: unpublishedGeneration });
      const invalidB = await capture("Unpublished global-fine generation retention smoke", revision + 1);
      globalFineAuthorityTransition = {
        validGeneration: globalFineLevelSet.generation, unpublishedGeneration, cleanFineCoarseRequired: true,
        retainedGeometrySource: invalidB.surfaceGeometrySource,
        retainedFrontInterfacePixels: invalidB.frontInterfacePixels,
        retainedBackInterfacePixels: invalidB.backInterfacePixels,
        retainedFrontInterfaceHash: invalidB.frontInterfaceHash,
        retainedBackInterfaceHash: invalidB.backInterfaceHash,
        ...(invalidB.frontInterfaceBounds_m ? { retainedFrontInterfaceBounds_m: invalidB.frontInterfaceBounds_m } : {}),
      };
      pipeline.setGlobalFineLevelSet(globalFineLevelSet);
    }
    // Exercise the opposite camera hemisphere as a distinct closure oracle.
    // Missing rear sheets and one-sided winding can be invisible from the
    // default camera even when the scalar and pressure publications are valid.
    packed.set([-1.55 * span, 1.12 * span, -1.72 * span, 0], 4);
    device.queue.writeBuffer(uniformBuffer, 0, packed);
    const reverse = await capture("Hybrid reverse-view closure smoke", revision + 2);
    if (process.env.FLUID_WATER_DIAGNOSTICS === "1") {
      await new Promise((resolve) => setTimeout(resolve, 25));
      console.info(JSON.stringify({ phase: "hybrid-water-diagnostics", ...pipeline.surfaceRenderDiagnostics }));
    }
    await device.queue.onSubmittedWorkDone();
    const rendererValidationError = await device.popErrorScope();
    rendererValidationScopeActive = false;
    await Promise.resolve();
    const rendererErrors = [
      ...(rendererValidationError ? [rendererValidationError.message] : []),
      ...uncapturedRendererErrors,
    ];
    if (rendererErrors.length > 0) {
      throw new Error(`RasterWaterPipeline production validation failed:\n${rendererErrors.join("\n")}`);
    }
    return { ...validA, reverseView: {
      frontInterfacePixels: reverse.frontInterfacePixels,
      backInterfacePixels: reverse.backInterfacePixels,
      pairedInterfacePixels: reverse.pairedInterfacePixels,
      frontOnlyInterfacePixels: reverse.frontOnlyInterfacePixels,
      backOnlyInterfacePixels: reverse.backOnlyInterfacePixels,
      backOnlyInterfaceLocations: reverse.backOnlyInterfaceLocations,
      backOnlyInterfacePositions_m: reverse.backOnlyInterfacePositions_m,
      frontInterfaceHash: reverse.frontInterfaceHash,
      backInterfaceHash: reverse.backInterfaceHash,
      narrowVerticalSlits: reverse.narrowVerticalSlits,
      enclosedSurfaceHoles: reverse.enclosedSurfaceHoles,
      surfaceSteps: reverse.surfaceSteps,
      ceilingContactPixels: reverse.ceilingContactPixels,
      wallCornerCapPixels: reverse.wallCornerCapPixels,
      wallCornerMaximumY_m: reverse.wallCornerMaximumY_m,
      damExposedCornerCapPixels: reverse.damExposedCornerCapPixels,
      ...(reverse.frontInterfaceBounds_m ? { frontInterfaceBounds_m: reverse.frontInterfaceBounds_m } : {}),
    }, rendererValidationErrorCount: 0, rendererUncapturedErrorCount: 0,
      ...(globalFineAuthorityTransition ? { globalFineAuthorityTransition } : {}) };
  } finally {
    if (rendererValidationScopeActive) await device.popErrorScope().catch(() => null);
    device.removeEventListener("uncapturederror", onUncapturedRendererError);
    pipeline.destroy(); output.destroy(); columnFallback.destroy(); uniformBuffer.destroy(); bodyBuffer.destroy();
  }
}

export interface VelocityStageSummary {
  maximum: number;
  liquidMaximum: number;
  location: number[];
  component: number;
  nonFiniteCount: number;
  kineticEnergyProxy: number;
  maximumComponentCfl: number;
  maximumLiquidDivergence_s: number;
  rmsLiquidDivergence_s: number;
}

export function gravitationalPotentialEnergyProxy(
  volume: ArrayLike<number>,
  width: number,
  height: number,
  depth: number,
  spacing: { x: number; y: number; z: number },
  gravity: { x: number; y: number; z: number }
) {
  const cellVolume = spacing.x * spacing.y * spacing.z;
  let energy = 0;
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const alpha = Math.max(0, Math.min(1, volume[x + width * (y + height * z)]));
    const position = {
      x: (x + 0.5 - width / 2) * spacing.x,
      y: (y + 0.5) * spacing.y,
      z: (z + 0.5 - depth / 2) * spacing.z
    };
    energy -= alpha * (gravity.x * position.x + gravity.y * position.y + gravity.z * position.z) * cellVolume;
  }
  return energy;
}

export async function readRgbaTexture3D(device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 16 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });
  device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height * depth * 4);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + height * z), width * 4);
    output.set(row, width * 4 * (y + height * z));
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

export function summarizeVelocityField(
  velocity: Float32Array,
  width: number,
  height: number,
  depth: number,
  volume: ArrayLike<number>,
  spacing: { x: number; y: number; z: number },
  dt_s: number,
  divergenceStencil: "backward" | "centered"
): VelocityStageSummary {
  let maximum = 0, liquidMaximum = 0, location = [0, 0, 0], component = 0, nonFiniteCount = 0;
  let kineticEnergyProxy = 0, maximumComponentCfl = 0;
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const index = x + width * (y + height * z), value = velocity[3 * index + axis];
      if (!Number.isFinite(value)) { nonFiniteCount += 1; continue; }
      const speed = Math.abs(value); if (speed > maximum) { maximum = speed; location = [x, y, z]; component = axis; }
      if (volume[index] > 0 && speed > liquidMaximum) liquidMaximum = speed;
      maximumComponentCfl = Math.max(maximumComponentCfl, speed * dt_s / [spacing.x, spacing.y, spacing.z][axis]);
      kineticEnergyProxy += 0.5 * Math.max(0, Math.min(1, volume[index])) * value * value * spacing.x * spacing.y * spacing.z;
    }
  }
  let maximumLiquidDivergence_s = 0, divergenceSquared = 0, liquidCells = 0;
  const at = (x: number, y: number, z: number, axis: number) => velocity[3 * (x + width * (y + height * z)) + axis];
  // Mirror the collocated solver's `centeredFaceVelocity`: the face value is
  // the average of the two adjacent cell centers, and a face whose neighbor
  // is outside the domain carries zero velocity.
  const centered = (x: number, y: number, z: number, axis: number) => {
    const limit = [width, height, depth][axis];
    const coordinate = [x, y, z][axis];
    const own = at(x, y, z, axis);
    const facePlus = coordinate + 1 < limit ? 0.5 * (own + at(axis === 0 ? x + 1 : x, axis === 1 ? y + 1 : y, axis === 2 ? z + 1 : z, axis)) : 0;
    const faceMinus = coordinate > 0 ? 0.5 * (own + at(axis === 0 ? x - 1 : x, axis === 1 ? y - 1 : y, axis === 2 ? z - 1 : z, axis)) : 0;
    return (facePlus - faceMinus) / [spacing.x, spacing.y, spacing.z][axis];
  };
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const index = x + width * (y + height * z);
    if (!(volume[index] > 1e-4)) continue;
    const divergence = divergenceStencil === "centered"
      ? centered(x, y, z, 0) + centered(x, y, z, 1) + centered(x, y, z, 2)
      : (at(x, y, z, 0) - (x > 0 ? at(x - 1, y, z, 0) : 0)) / spacing.x
        + (at(x, y, z, 1) - (y > 0 ? at(x, y - 1, z, 1) : 0)) / spacing.y
        + (at(x, y, z, 2) - (z > 0 ? at(x, y, z - 1, 2) : 0)) / spacing.z;
    if (!Number.isFinite(divergence)) { nonFiniteCount += 1; continue; }
    maximumLiquidDivergence_s = Math.max(maximumLiquidDivergence_s, Math.abs(divergence));
    divergenceSquared += divergence * divergence; liquidCells += 1;
  }
  return {
    maximum, liquidMaximum, location, component, nonFiniteCount, kineticEnergyProxy, maximumComponentCfl,
    maximumLiquidDivergence_s,
    rmsLiquidDivergence_s: Math.sqrt(divergenceSquared / Math.max(1, liquidCells))
  };
}

export async function readVelocityTexture3D(
  device: GPUDevice,
  texture: GPUTexture,
  width: number,
  height: number,
  depth: number,
  volume: ArrayLike<number>,
  spacing = { x: 1, y: 1, z: 1 },
  dt_s = 0
): Promise<VelocityStageSummary> {
  const velocity = await readVelocityField3D(device, texture, width, height, depth);
  return summarizeVelocityField(velocity, width, height, depth, volume, spacing, dt_s, "backward");
}

export async function readVelocityField3D(
  device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number,
) {
  const raw = await readRgbaTexture3D(device, texture, width, height, depth);
  const velocity = new Float32Array(width * height * depth * 3);
  for (let index = 0; index < width * height * depth; index += 1) {
    velocity[3 * index] = raw[4 * index];
    velocity[3 * index + 1] = raw[4 * index + 1];
    velocity[3 * index + 2] = raw[4 * index + 2];
  }
  return velocity;
}

/**
 * Reconstruct the cubic velocity field from a packed restricted tall-cell
 * texture (rows 0/1 are the tall endpoint samples; interior rows interpolate
 * linearly between them per paper Eq 5, mirroring `validVelocityCell`) and
 * summarize it with the solver's own centered collocated divergence.
 */
export async function readTallVelocityTexture3D(
  device: GPUDevice,
  texture: GPUTexture,
  nx: number,
  storedNy: number,
  nz: number,
  fineNy: number,
  bases: ArrayLike<number>,
  volume: ArrayLike<number>,
  spacing: { x: number; y: number; z: number },
  dt_s: number
): Promise<VelocityStageSummary> {
  const velocity = await readTallVelocityField3D(device, texture, nx, storedNy, nz, fineNy, bases);
  return summarizeVelocityField(velocity, nx, fineNy, nz, volume, spacing, dt_s, "backward");
}

export async function readTallVelocityField3D(
  device: GPUDevice,
  texture: GPUTexture,
  nx: number,
  storedNy: number,
  nz: number,
  fineNy: number,
  bases: ArrayLike<number>
) {
  const raw = await readRgbaTexture3D(device, texture, nx, storedNy, nz);
  const velocity = new Float32Array(nx * fineNy * nz * 3);
  const packedAt = (x: number, packedY: number, z: number, axis: number) => raw[4 * (x + nx * (packedY + storedNy * z)) + axis];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    for (let y = 0; y < fineNy; y += 1) {
      const index = 3 * (x + nx * (y + fineNy * z));
      if (y < base && base > 0) {
        const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
        for (let axis = 0; axis < 3; axis += 1) velocity[index + axis] = packedAt(x, 0, z, axis) * (1 - t) + packedAt(x, 1, z, axis) * t;
      } else {
        const packedY = 2 + y - base;
        if (packedY >= 2 && packedY < storedNy) for (let axis = 0; axis < 3; axis += 1) velocity[index + axis] = packedAt(x, packedY, z, axis);
      }
    }
  }
  return velocity;
}

export async function readCompactOctreeVelocityField3D(
  device: GPUDevice,
  solver: GPUSolverInstance,
  dimensions: readonly [number, number, number],
): Promise<(CompactVelocityRaster & {
  publicationValid: boolean;
  rowCount: number;
  reconstructedRows: number;
}) | undefined> {
  const structured = solver as GPUSolverInstance & {
    structuredVelocityControl?: GPUBuffer;
    structuredRowVelocities?: GPUBuffer;
  };
  const controlBuffer = structured.structuredVelocityControl;
  const headerBuffer = solver.powerLeafHeaders;
  const velocityBuffer = structured.structuredRowVelocities;
  if (!controlBuffer || !headerBuffer || !velocityBuffer) return undefined;
  const controlBytes = await readBufferBinding(device, { buffer: controlBuffer }, 24);
  const control = unpackStructuredVelocityControl(new Uint32Array(
    controlBytes.buffer, controlBytes.byteOffset, controlBytes.byteLength / 4));
  const rowCount = control.rowCount, reconstructedRows = control.rowCount;
  const bankStrideBytes = velocityBuffer.size / 2;
  if (rowCount === 0 || rowCount * 48 > headerBuffer.size
    || !Number.isSafeInteger(bankStrideBytes) || rowCount * 16 > bankStrideBytes) return undefined;
  const [headerBytes, velocityBytes] = await Promise.all([
    readBufferBinding(device, { buffer: headerBuffer }, rowCount * 48),
    readBufferBinding(device, { buffer: velocityBuffer, offset: control.activeBank * bankStrideBytes }, rowCount * 16),
  ]);
  const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const velocities = new Float32Array(velocityBytes.buffer, velocityBytes.byteOffset, rowCount * 4);
  return {
    ...rasterizeStructuredCellVelocities(headers, velocities, rowCount, dimensions),
    publicationValid: control.flags === 0 && control.firstError === 0xffff_ffff
      && control.epoch > 0 && control.activeBank < 2,
    rowCount,
    reconstructedRows,
  };
}

export function velocityDifferenceMagnitude(left: Float32Array, right: Float32Array) {
  if (left.length !== right.length || left.length % 3 !== 0) throw new Error("Velocity fields must share xyz dimensions");
  const difference = new Float32Array(left.length / 3);
  for (let index = 0; index < difference.length; index += 1) {
    difference[index] = Math.hypot(left[3 * index] - right[3 * index], left[3 * index + 1] - right[3 * index + 1], left[3 * index + 2] - right[3 * index + 2]);
  }
  return difference;
}

export async function readFloatTexture2D(device: GPUDevice, texture: GPUTexture, width: number, height: number) {
  const components = texture.format === "rg32float" ? 2 : 1;
  const bytesPerRow = Math.ceil(width * components * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height });
  device.queue.submit([encoder.finish()]);
  await buffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(mapped.buffer, mapped.byteOffset + bytesPerRow * y, width * components);
    for (let x = 0; x < width; x += 1) output[x + width * y] = row[components * x];
  }
  buffer.unmap(); buffer.destroy();
  return output;
}

export function inspectColumnBases(bases: ArrayLike<number>, nx: number, nz: number, fineNy: number, regularLayers: number, maximumDelta: number) {
  const histogram: Record<string, number> = {}, violations: Array<{ a: [number, number, number]; b: [number, number, number]; delta: number }> = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const height = Math.round(bases[x + nx * z]);
    histogram[height] = (histogram[height] ?? 0) + 1;
    for (const [otherX, otherZ] of [[x + 1, z], [x, z + 1]] as const) {
      if (otherX >= nx || otherZ >= nz) continue;
      const otherHeight = Math.round(bases[otherX + nx * otherZ]), delta = Math.abs(height - otherHeight);
      if (delta > maximumDelta && violations.length < 12) violations.push({ a: [x, z, height], b: [otherX, otherZ, otherHeight], delta });
    }
  }
  return { ...summarizeTallCellActivity(bases, fineNy, regularLayers, nx, nz), histogram, violations };
}

export function inspectTallVolumeGaps(packed: ArrayLike<number>, bases: ArrayLike<number>, nx: number, storedNy: number, nz: number, fineNy: number, maximumDelta = Infinity) {
  let dryTallColumns = 0, dryTallWithWetRegularAbove = 0, mixedEndpointColumns = 0, wetBandCeilingColumns = 0, unexcusedDeltaViolations = 0;
  const phiAt = (x: number, y: number, z: number) => {
    const base = Math.round(bases[x + nx * z]);
    if (y < base && base > 0) {
      const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
      const bottom = packed[x + nx * storedNy * z];
      const top = packed[x + nx * (1 + storedNy * z)];
      return bottom + (top - bottom) * t;
    }
    const packedY = 2 + y - base;
    return packedY >= 2 && packedY < storedNy ? packed[x + nx * (packedY + storedNy * z)] : Infinity;
  };
  // Eq. 10 is an unconditional restriction on neighboring band bases now
  // that the signed-distance remap can move the interface without VOF
  // representability floors.
  if (Number.isFinite(maximumDelta)) for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    for (const [otherX, otherZ] of [[x + 1, z], [x, z + 1]] as const) {
      if (otherX >= nx || otherZ >= nz) continue;
      const otherBase = Math.round(bases[otherX + nx * otherZ]);
      if (Math.abs(base - otherBase) > maximumDelta) unexcusedDeltaViolations += 1;
    }
  }
  const examples: Array<{ x: number; z: number; base: number; bottom: number; top: number; lowestWetWorldY: number }> = [];
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const base = Math.round(bases[x + nx * z]);
    const ceilingWorldY = base + storedNy - 3;
    if (ceilingWorldY < fineNy - 1 && phiAt(x, ceilingWorldY, z) <= 0) wetBandCeilingColumns += 1;
    if (base < 2) continue;
    const bottom = packed[x + nx * storedNy * z];
    const top = packed[x + nx * (1 + storedNy * z)];
    if ((bottom <= 0) !== (top <= 0)) mixedEndpointColumns += 1;
    if (bottom <= 0 || top <= 0) continue;
    dryTallColumns += 1;
    let lowestWetWorldY = -1;
    for (let y = base; y < fineNy; y += 1) if (phiAt(x, y, z) <= 0) {
      lowestWetWorldY = y;
      break;
    }
    if (lowestWetWorldY < 0) continue;
    dryTallWithWetRegularAbove += 1;
    if (examples.length < 12) examples.push({ x, z, base, bottom, top, lowestWetWorldY });
  }
  return { dryTallColumns, dryTallWithWetRegularAbove, mixedEndpointColumns, wetBandCeilingColumns, unexcusedDeltaViolations, examples };
}

export interface CubicVolumeFieldReadback {
  field: Float32Array;
  summary: ScalarFieldSummary;
  compactFieldEvidence?: CompactOctreeFieldEvidence;
  tallCellActivity?: TallCellActivitySummary;
  tallVolumeGaps?: ReturnType<typeof inspectTallVolumeGaps>;
}

/**
 * QA forensics: verify the sparse owner-page arena encodes a partition. Every
 * decoded leaf's cells must all decode to that same leaf; a zero word inside
 * a paged block that also holds written words is an overlap by construction.
 */
export async function auditOwnerLatticeConsistency(
  device: GPUDevice, solver: GPUSolverInstance, dims: readonly [number, number, number],
): Promise<Record<string, unknown>[]> {
  const debug = solver.ownerLatticeDebug;
  if (!debug) return [];
  const [nx, ny, nz] = dims;
  const words = new Uint32Array((await readBufferBinding(device, { buffer: debug.buffer }, debug.buffer.size)).buffer);
  if (words.length <= 15 || words[15] !== 0x4f57_4e52) {
    return [{ issue: "invalid-owner-page-arena", arenaHeader: Array.from(words.slice(0, 16)) }];
  }
  const rawWord = (q: readonly number[]): number | undefined => {
    const capacity = words[3], pageOffset = words[5], resident = Math.min(words[1], capacity);
    if (pageOffset !== 16 + capacity || words[6] !== pageOffset + capacity) return undefined;
    const bd = [Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 8)];
    const b = q.map((v) => Math.floor(v / 8));
    const logical = b[0] + b[1] * bd[0] + b[2] * bd[0] * bd[1];
    const key = logical + 1; let low = 0, high = resident;
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2);
      if (words[16 + middle] < key) low = middle + 1;
      else high = middle;
    }
    if (low >= resident || words[16 + low] !== key) return undefined;
    const encoded = words[pageOffset + low];
    if (encoded === 0 || encoded === 0xffff_ffff) return undefined;
    const local = q.map((v) => v % 8);
    return words[words[6] + (encoded - 1) * 512 + local[0] + local[1] * 8 + local[2] * 64] ?? 0;
  };
  const leafOf = (word: number, q: readonly number[]) => {
    if ((word & 0x8000_0000) !== 0) {
      const exponent = (word >>> 18) & 7, size = 1 << exponent;
      const brickOrigin = q.map((value) => Math.floor(value / 8) * 8);
      const delta = [word & 63, (word >>> 6) & 63, (word >>> 12) & 63]
        .map((value) => value - 32);
      return { origin: brickOrigin.map((value, axis) => value + delta[axis]), size };
    }
    // Zero or malformed page payloads decode to the canonical coarse owner.
    let size = Math.min(debug.maximumLeafSize, 8);
    for (;;) {
      const origin = q.map((v) => Math.floor(v / size) * size);
      if (origin.every((v, a) => v + size <= dims[a])) return { origin, size };
      size >>= 1;
    }
  };
  const issues: Record<string, unknown>[] = [];
  for (let z = 0; z < nz && issues.length < 64; z += 1) for (let y = 0; y < ny && issues.length < 64; y += 1) for (let x = 0; x < nx; x += 1) {
    const q = [x, y, z];
    const word = rawWord(q);
    if (word === undefined) continue;
    const leaf = leafOf(word, q);
    if (leaf.origin.some((v, a) => v + leaf.size > dims[a])) {
      issues.push({ q, word: word >>> 0, issue: "leaf-exceeds-domain", leaf }); continue;
    }
    if (leaf.size > 1 && (x !== leaf.origin[0] || y !== leaf.origin[1] || z !== leaf.origin[2])) continue;
    for (let dz = 0; dz < leaf.size; dz += 1) for (let dy = 0; dy < leaf.size; dy += 1) for (let dx = 0; dx < leaf.size; dx += 1) {
      const partner = [leaf.origin[0] + dx, leaf.origin[1] + dy, leaf.origin[2] + dz];
      const partnerWord = rawWord(partner);
      const partnerLeaf = partnerWord === undefined
        ? leafOf(0, partner) : leafOf(partnerWord, partner);
      if (partnerLeaf.size !== leaf.size || partnerLeaf.origin.some((v, a) => v !== leaf.origin[a])) {
        issues.push({ q, word: word >>> 0, leaf, partner, partnerWord: partnerWord === undefined ? "absent" : partnerWord >>> 0, partnerLeaf });
        if (issues.length >= 64) return issues;
      }
    }
  }
  return issues;
}

export async function readCubicVolumeField(
  device: GPUDevice,
  solver: GPUSolverInstance,
  requireSpatialField = false,
): Promise<CubicVolumeFieldReadback> {
  const { nx, ny, nz, storedNy, gridKind } = solver.info;
  const coarseOnly = gridKind === "octree" && !solver.globalFineLevelSetSource
    ? solver.coarseLevelSetSource : undefined;
  if (coarseOnly) {
    const [directoryBytes, controlBytes] = await Promise.all([
      readBufferBinding(device, coarseOnly.directory, 32 + coarseOnly.rowCapacity * 32),
      readBufferBinding(device, coarseOnly.control, 64),
    ]);
    const directoryWords = new Uint32Array(directoryBytes.buffer,
      directoryBytes.byteOffset, directoryBytes.byteLength / 4);
    if (directoryWords[0] !== 0x8000_0000) {
      const controlWords = new Uint32Array(controlBytes.buffer,
        controlBytes.byteOffset, controlBytes.byteLength / 4);
      throw new Error(`Coarse-only octree QA publication rejected: directory=${JSON.stringify(
        Array.from(directoryWords.slice(0, 8)))}, control=${JSON.stringify(Array.from(controlWords))}`);
    }
    const reconstructed = reconstructCoarseOnlyOctreeOccupancyField(
      directoryWords,
      coarseOnly.generation,
      [nx, ny, nz],
    );
    return {
      field: reconstructed.field,
      summary: summarizeScalarField(reconstructed.field, nx, ny, nz),
    };
  }
  const compactPaged = solver.info.gridKind === "octree" && Boolean(solver.globalFineLevelSetSource);
  if (compactPaged) {
    const source = solver.globalFineLevelSetSource;
    if (!source?.coarsePhiDirectory || !source.coarsePhiRowCapacity) {
      if (requireSpatialField) {
        throw new Error("Compact octree QA field requires a published global-fine source and compact-coarse fallback");
      }
      // Legacy compact-only smoke cases do not request cross-method spatial
      // acceptance. Keep their reduction summary while making the exact
      // comparison harness fail closed via FLUID_REQUIRE_SPATIAL_FIELD=1.
      const cellSum = solver.info.volumeCellSum ?? solver.info.initialVolumeCellSum ?? 0;
      const occupied = Math.max(0, Math.min(nx * ny * nz, Math.round(cellSum)));
      return { field: new Float32Array(0), summary: {
        minimum: 0, maximum: 1, cellSum, wetCells: occupied, mixedCells: solver.info.phiInterfaceCellCount ?? 0,
        excessCells: 0, meanColumnAmount: cellSum / Math.max(1, nx * nz), columnAmountStdDev: 0,
        componentCount: occupied > 0 ? 1 : 0, largestComponent: occupied, interfaceFaceCount: 0,
        enclosedAirComponentCount: 0, enclosedAirCells: 0, centroidCells: null,
      } };
    }
    const sampleWords = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    const [metadataBytes, flagBytes, phiBytes, worklistBytes, coarseBytes, coarseControlBytes,
      fineRestrictionBytes,
      topologyBytes, transportBytes, redistanceBytes, volumeBytes, mgpcgBytes] = await Promise.all([
      readBufferBinding(device, { buffer: source.metadata }, source.plan.maximumResidentBricks * 40),
      readBufferBinding(device, { buffer: source.flags }, sampleWords * 4),
      readBufferBinding(device, { buffer: source.phi }, sampleWords * 4),
      readBufferBinding(device, { buffer: source.worklist }, source.worklist.size),
      readBufferBinding(device, { buffer: source.coarsePhiDirectory }, 32 + source.coarsePhiRowCapacity * 32),
      solver.globalFineCoarseLevelSetControl
        ? readBufferBinding(device, { buffer: solver.globalFineCoarseLevelSetControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineRestrictionControl
        ? readBufferBinding(device, { buffer: solver.globalFineRestrictionControl }, 32)
        : Promise.resolve(undefined),
      source.topologyControl
        // 48 bytes, not 32: words 9..11 latch the finalize rejection that
        // clearDesiredGeneration wipes from words 0..8 every generation.
        ? readBufferBinding(device, { buffer: source.topologyControl }, 48)
        : Promise.resolve(undefined),
      solver.globalFineTransportControl
        ? readBufferBinding(device, { buffer: solver.globalFineTransportControl }, 64)
        : Promise.resolve(undefined),
      solver.globalFineRedistanceControl
        ? readBufferBinding(device, { buffer: solver.globalFineRedistanceControl },
          FINE_LEVELSET_REDISTANCE_CONTROL_BYTES)
        : Promise.resolve(undefined),
      solver.globalFineVolumeControl
        ? readBufferBinding(device, { buffer: solver.globalFineVolumeControl }, 64)
        : Promise.resolve(undefined),
      (solver as GPUSolverInstance & { mgpcgControl?: GPUBuffer }).mgpcgControl
        ? readBufferBinding(device, { buffer: (solver as GPUSolverInstance & { mgpcgControl: GPUBuffer }).mgpcgControl }, 64)
        : Promise.resolve(undefined),
    ]);
    const compactSnapshot = {
      plan: source.plan,
      generation: source.generation,
      metadata: new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4),
      flags: new Uint32Array(flagBytes.buffer, flagBytes.byteOffset, flagBytes.byteLength / 4),
      phi: new Float32Array(phiBytes.buffer, phiBytes.byteOffset, phiBytes.byteLength / 4),
      worklist: new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4),
      coarseDirectory: new Uint32Array(coarseBytes.buffer, coarseBytes.byteOffset, coarseBytes.byteLength / 4),
      ...(coarseControlBytes ? { coarseControl: new Uint32Array(coarseControlBytes.buffer,
        coarseControlBytes.byteOffset, coarseControlBytes.byteLength / 4) } : {}),
      ...(fineRestrictionBytes ? { fineRestrictionControl: new Uint32Array(fineRestrictionBytes.buffer,
        fineRestrictionBytes.byteOffset, fineRestrictionBytes.byteLength / 4) } : {}),
      ...(topologyBytes ? { topologyControl: new Uint32Array(topologyBytes.buffer, topologyBytes.byteOffset,
        topologyBytes.byteLength / 4) } : {}),
      ...(transportBytes ? { transportControl: new Uint32Array(transportBytes.buffer, transportBytes.byteOffset,
        transportBytes.byteLength / 4) } : {}),
      ...(redistanceBytes ? { redistanceControl: new Uint32Array(redistanceBytes.buffer, redistanceBytes.byteOffset,
        redistanceBytes.byteLength / 4) } : {}),
      ...(volumeBytes ? { volumeControl: new Uint32Array(volumeBytes.buffer, volumeBytes.byteOffset,
        volumeBytes.byteLength / 4) } : {}),
      ...(mgpcgBytes ? { mgpcgControl: new Uint32Array(mgpcgBytes.buffer,
        mgpcgBytes.byteOffset, mgpcgBytes.byteLength / 4) } : {}),
    };
    let reconstructed: ReturnType<typeof reconstructCompactOctreeOccupancyField>;
    try {
      reconstructed = reconstructCompactOctreeOccupancyField(compactSnapshot, [nx, ny, nz]);
    } catch (error) {
      const failureProjection = (solver as GPUSolverInstance & { octreeProjection?: {
        readPowerFrontierFailure(): Promise<unknown>;
        readGlobalFineLevelSetDiagnostics(): Promise<unknown>;
      } }).octreeProjection;
      const [candidateFailure, fineFailure] = await Promise.all([
        failureProjection?.readPowerFrontierFailure(),
        failureProjection?.readGlobalFineLevelSetDiagnostics(),
      ]);
      const accurateClassDispatchBinding = (solver as GPUSolverInstance & {
        workAccountingBuffers?: { accurateClassDispatch?: GPUBufferBinding };
      }).workAccountingBuffers?.accurateClassDispatch;
      const accurateClassDispatchBytes = accurateClassDispatchBinding
        ? await readBufferBinding(device, accurateClassDispatchBinding, 29 * 4)
        : undefined;
      const pressureBuffers = (solver as GPUSolverInstance & { workAccountingBuffers?: {
        mgpcgPreconditioned?: GPUBufferBinding;
        mgpcgPreconditionedImage?: GPUBufferBinding;
        mgpcgResidual?: GPUBufferBinding;
        acceptedRows?: GPUBufferBinding;
        section63Coefficients?: GPUBufferBinding;
      } }).workAccountingBuffers;
      const [preconditionedBytes, preconditionedImageBytes, residualBytes,
        acceptedRowsBytes] = await Promise.all([
        pressureBuffers?.mgpcgPreconditioned
          ? readBufferBinding(device, pressureBuffers.mgpcgPreconditioned,
            pressureBuffers.mgpcgPreconditioned.size!) : undefined,
        pressureBuffers?.mgpcgPreconditionedImage
          ? readBufferBinding(device, pressureBuffers.mgpcgPreconditionedImage,
            pressureBuffers.mgpcgPreconditionedImage.size!) : undefined,
        pressureBuffers?.mgpcgResidual
          ? readBufferBinding(device, pressureBuffers.mgpcgResidual,
            pressureBuffers.mgpcgResidual.size!) : undefined,
        pressureBuffers?.acceptedRows
          ? readBufferBinding(device, pressureBuffers.acceptedRows, 44) : undefined,
      ]);
      const vectorSummary = (bytes: Uint8Array | undefined) => {
        if (!bytes) return undefined;
        const values = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
        let minimum = Infinity, maximum = -Infinity, minimumRow = 0, maximumRow = 0;
        let sum = 0, squared = 0, nonzero = 0;
        for (let row = 0; row < values.length; row += 1) {
          const value = values[row]!;
          if (value < minimum) { minimum = value; minimumRow = row; }
          if (value > maximum) { maximum = value; maximumRow = row; }
          sum += value; squared += value * value; if (value !== 0) nonzero += 1;
        }
        return { minimum, minimumRow, maximum, maximumRow,
          sum, squared, nonzero, count: values.length };
      };
      const preconditionedSummary = vectorSummary(preconditionedBytes);
      const residualSummary = vectorSummary(residualBytes);
      const preconditionedDotResidual = preconditionedBytes && residualBytes
        ? (() => {
          const preconditioned = new Float32Array(preconditionedBytes.buffer,
            preconditionedBytes.byteOffset, preconditionedBytes.byteLength / 4);
          const residual = new Float32Array(residualBytes.buffer,
            residualBytes.byteOffset, residualBytes.byteLength / 4);
          let value = 0;
          for (let row = 0; row < Math.min(preconditioned.length, residual.length); row += 1) {
            value += preconditioned[row]! * residual[row]!;
          }
          return value;
        })() : undefined;
      const acceptedRows = acceptedRowsBytes
        ? new Uint32Array(acceptedRowsBytes.buffer, acceptedRowsBytes.byteOffset, 11) : undefined;
      const coefficientSource = pressureBuffers?.section63Coefficients;
      const coefficientSamples = coefficientSource && acceptedRows && preconditionedSummary
        ? await Promise.all(Array.from(new Set([0, 1, preconditionedSummary.minimumRow,
          preconditionedSummary.maximumRow, Math.max(0, acceptedRows[2]! - 1)])).map(async (row) => {
          const bankBytes = coefficientSource.size!;
          const bytes = await readBufferBinding(device, {
            buffer: coefficientSource.buffer,
            offset: (acceptedRows[4]! & 1) * bankBytes + row * 19 * 4,
          }, 19 * 4);
          const values = Array.from(new Float32Array(bytes.buffer, bytes.byteOffset, 19));
          return { row, diagonal: values[0], couplingSum: values.slice(1).reduce(
            (sum, value) => sum + value, 0), values };
        })) : undefined;
      console.error(JSON.stringify({ phase: "compact-octree-field-publication-rejected", grid: [nx, ny, nz],
        ...compactOctreePublicationHeaderEvidence(compactSnapshot),
        ...(accurateClassDispatchBytes ? { accurateClassDispatch: Array.from(new Uint32Array(
          accurateClassDispatchBytes.buffer, accurateClassDispatchBytes.byteOffset, 29)) } : {}),
        preconditioned: preconditionedSummary,
        preconditionedImage: vectorSummary(preconditionedImageBytes),
        residual: residualSummary,
        preconditionedDotResidual,
        acceptedRows: acceptedRows ? Array.from(acceptedRows) : undefined,
        coefficientSamples,
        candidateFailure,
        fineFailure,
        error: error instanceof Error ? error.message : String(error) }));
      await dumpFineRedistancePageDeltaForensics(device, solver, source, compactSnapshot);
      throw error;
    }
    const { field, ...reconstructionEvidence } = reconstructed;
    // Preserve the controls already read for reconstruction in the returned
    // evidence.  Short, non-raster Dawn reproductions deliberately do not run
    // the much larger presentation-transition audit, but their final
    // topology gates still need the exact same transaction words.
    // This adds no readback: compactOctreePublicationHeaderEvidence only
    // decodes the buffers above.
    const publicationEvidence = compactOctreePublicationHeaderEvidence(compactSnapshot);
    const compactFieldEvidence: CompactOctreeFieldEvidence = {
      ...reconstructionEvidence,
      ...(publicationEvidence.transportControl
        ? { transportControl: publicationEvidence.transportControl } : {}),
    };
    console.log(JSON.stringify({ phase: "compact-octree-field-readback", grid: [nx, ny, nz],
      ...compactOctreePublicationHeaderEvidence(compactSnapshot),
      ...compactFieldEvidence }));
    return { field, summary: summarizeScalarField(field, nx, ny, nz), compactFieldEvidence };
  }
  const levelSet = solver.info.surfaceField === "levelset";
  const packed = await readFloatTexture3D(device, levelSet ? solver.surfaceFieldTexture ?? solver.volumeTexture : solver.volumeTexture, nx, storedNy, nz);
  let bases = new Float32Array(nx * nz);
  if (gridKind === "restricted-tall-cell") bases = await readFloatTexture2D(device, solver.columnBaseTexture!, nx, nz);
  const field = new Float32Array(nx * ny * nz);
  const h = solver.info.cellSize_m;
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    if (gridKind !== "restricted-tall-cell") {
      const value = packed[index(x, y, z)];
      field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / (4 * h))) : value;
    }
    else {
      const base = Math.round(bases[x + nx * z]);
      if (y < base && base > 0) {
        const t = Math.min(1, Math.max(0, y / Math.max(base - 1, 1)));
        const bottom = packed[x + nx * storedNy * z];
        const top = packed[x + nx * (1 + storedNy * z)];
        const value = bottom + (top - bottom) * t;
        field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / h)) : Math.max(0, value);
      } else {
        const packedY = 2 + y - base;
        const value = packedY >= 2 && packedY < storedNy ? packed[x + nx * (packedY + storedNy * z)] : 5 * h;
        field[index(x, y, z)] = levelSet ? Math.min(1, Math.max(0, 0.5 - value / h)) : value;
      }
    }
  }
  return {
    field,
    summary: summarizeScalarField(field, nx, ny, nz),
    tallCellActivity: gridKind === "restricted-tall-cell" ? summarizeTallCellActivity(bases, ny, solver.info.regularLayers, nx, nz) : undefined,
    tallVolumeGaps: gridKind === "restricted-tall-cell" ? inspectTallVolumeGaps(packed, bases, nx, storedNy, nz, ny, solver.info.maximumNeighborDelta) : undefined
  };
}

export async function dumpFineRedistancePageDeltaForensics(
  device: GPUDevice,
  solver: GPUSolverInstance,
  source: WebGPUFineLevelSetBrickSource,
  snapshot: {
    metadata: Uint32Array;
    worklist: Uint32Array;
    flags: Uint32Array;
    phi: Float32Array;
    transportControl?: Uint32Array;
    redistanceControl?: Uint32Array;
  },
): Promise<void> {
  const debug = solver.globalFinePageDeltaDebug;
  const control = snapshot.redistanceControl;
  if (!debug || !control || control.length < 10) return;
  const [headerBytes, parameterBytes] = await Promise.all([
    readBufferBinding(device, { buffer: debug.buffer }, 64),
    readBufferBinding(device, { buffer: debug.params }, 96),
  ]);
  const header = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, 16);
  const topologyParameters = new Uint32Array(parameterBytes.buffer, parameterBytes.byteOffset, 24);
  const dirtyCount = Math.min(header[2], debug.pageCapacity);
  const supportCount = Math.min(header[3], debug.pageCapacity);
  const changedCount = Math.min(header[0], 2 * debug.pageCapacity);
  const readPageStream = async (offsetWords: number, count: number) => {
    if (count === 0) return new Uint32Array(0);
    const bytes = await readBufferBinding(device,
      { buffer: debug.buffer, offset: offsetWords * 4 }, count * 4);
    return new Uint32Array(bytes.buffer, bytes.byteOffset, count);
  };
  const [changedKeys, dirty, support] = await Promise.all([
    readPageStream(debug.changedKeysOffsetWords, changedCount),
    readPageStream(debug.dirtyPagesOffsetWords, dirtyCount),
    readPageStream(debug.supportPagesOffsetWords, supportCount),
  ]);
  const supportSet = new Set<number>(support);
  const dirtySet = new Set<number>(dirty);
  const dirtyMissingFromSupport = Array.from(dirty).filter((id) => !supportSet.has(id));
  const supportOutsideDirty = Array.from(support).filter((id) => !dirtySet.has(id));
  const firstError = control[5] >>> 0;
  const errorPage = firstError === 0xffff_ffff
    ? 0xffff_ffff : Math.floor(firstError / source.plan.samplesPerBrick);
  let errorPageScratch: Record<string, unknown> | undefined;
  if (errorPage < debug.pageCapacity) {
    const sampleOffset = errorPage * source.plan.samplesPerBrick;
    const sampleBytes = source.plan.samplesPerBrick * 4;
    const [aBytes, bBytes] = await Promise.all([
      readBufferBinding(device, { buffer: source.workA, offset: sampleOffset * 4 }, sampleBytes),
      readBufferBinding(device, { buffer: source.workB, offset: sampleOffset * 4 }, sampleBytes),
    ]);
    const a = new Uint32Array(aBytes.buffer, aBytes.byteOffset, source.plan.samplesPerBrick);
    const b = new Uint32Array(bBytes.buffer, bBytes.byteOffset, source.plan.samplesPerBrick);
    const bf = new Float32Array(bBytes.buffer, bBytes.byteOffset, source.plan.samplesPerBrick);
    let validSamples = 0, finiteDistances = 0, invalidSeeds = 0;
    for (let local = 0; local < source.plan.samplesPerBrick; local += 1) {
      if ((snapshot.flags[sampleOffset + local] & 1) !== 0) validSamples += 1;
      if (Number.isFinite(bf[local])) finiteDistances += 1;
      if (a[local] === 0xffff_ffff) invalidSeeds += 1;
    }
    errorPageScratch = {
      page: errorPage,
      generation: snapshot.metadata[errorPage * 10 + 2],
      dirtyRank: Array.from(dirty).indexOf(errorPage),
      supportRank: Array.from(support).indexOf(errorPage),
      validSamples,
      finiteDistances,
      invalidSeeds,
      workAFirst8: Array.from(a.slice(0, 8)),
      workBFirst8: Array.from(b.slice(0, 8)),
      phiFirst8: Array.from(snapshot.phi.slice(sampleOffset, sampleOffset + 8)),
      flagsFirst8: Array.from(snapshot.flags.slice(sampleOffset, sampleOffset + 8)),
    };
  }
  const transportFirstError = snapshot.transportControl?.[12] ?? 0xffff_ffff;
  let transportErrorSample: Record<string, unknown> | undefined;
  if (transportFirstError !== 0xffff_ffff) {
    const page = Math.floor(transportFirstError / source.plan.samplesPerBrick);
    const local = transportFirstError % source.plan.samplesPerBrick;
    if (page < debug.pageCapacity) {
      const key = snapshot.metadata[page * 10 + 1];
      const r = source.plan.brickResolution;
      const localZ = Math.floor(local / (r * r));
      const localRem = local - localZ * r * r;
      const localY = Math.floor(localRem / r), localX = localRem - localY * r;
      const bx = key % source.plan.brickDimensions[0];
      const by = Math.floor(key / source.plan.brickDimensions[0])
        % source.plan.brickDimensions[1];
      const bz = Math.floor(key / (source.plan.brickDimensions[0]
        * source.plan.brickDimensions[1]));
      const nextBytes = await readBufferBinding(device,
        { buffer: source.workA, offset: transportFirstError * 4 }, 4);
      const next = new Float32Array(nextBytes.buffer, nextBytes.byteOffset, 1)[0];
      transportErrorSample = {
        index: transportFirstError, page, key, local,
        fineSample: [bx * r + localX, by * r + localY, bz * r + localZ],
        phi: Number.isFinite(snapshot.phi[transportFirstError])
          ? snapshot.phi[transportFirstError] : null,
        nextPhi: Number.isFinite(next) ? next : null,
        flags: snapshot.flags[transportFirstError],
      };
    }
  }
  const [bx, by, bz] = source.plan.brickDimensions;
  const logicalCount = bx * by * bz;
  const changedMask = new Uint8Array(logicalCount);
  for (const key of changedKeys) if (key < logicalCount) changedMask[key] = 1;
  const distance = new Uint8Array(logicalCount).fill(0xff);
  let frontier = Array.from(changedKeys).filter((key) => key < logicalCount);
  for (const key of frontier) distance[key] = 0;
  for (let radius = 1; radius <= 16 && frontier.length > 0; radius += 1) {
    const next: number[] = [];
    for (const key of frontier) {
      const x = key % bx, y = Math.floor(key / bx) % by, z = Math.floor(key / (bx * by));
      for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx, ny = y + dy, nz = z + dz;
          if (nx < 0 || nx >= bx || ny < 0 || ny >= by || nz < 0 || nz >= bz) continue;
          const neighbor = nx + bx * (ny + by * nz);
          if (distance[neighbor] !== 0xff) continue;
          distance[neighbor] = radius;
          next.push(neighbor);
        }
      }
    }
    frontier = next;
  }
  const activeCount = Math.min(snapshot.worklist[1], debug.pageCapacity);
  const activeKeys: number[] = [];
  for (let rank = 0; rank < activeCount; rank += 1) {
    const id = snapshot.worklist[7 + rank];
    if (id < debug.pageCapacity) activeKeys.push(snapshot.metadata[id * 10 + 1]);
  }
  const activeCountByChangedChebyshevRadius = Array.from({ length: 17 }, (_, radius) =>
    activeKeys.reduce((count, key) => count + (key < logicalCount && distance[key] <= radius ? 1 : 0), 0));
  const pageDistanceHistogram = (pages: Uint32Array) => {
    const histogram = new Array<number>(18).fill(0);
    for (const id of pages) {
      const key = id < debug.pageCapacity ? snapshot.metadata[id * 10 + 1] : 0xffff_ffff;
      const d = key < logicalCount ? distance[key] : 0xff;
      histogram[Math.min(d, 17)] += 1;
    }
    return histogram;
  };
  console.error(JSON.stringify({
    phase: "fine-redistance-page-delta-forensics",
    generation: source.generation,
    topologyParameters: Array.from(topologyParameters),
    header: Array.from(header),
    dirtyCount,
    supportCount,
    dirtyUnique: dirtySet.size,
    supportUnique: supportSet.size,
    dirtyMissingFromSupportCount: dirtyMissingFromSupport.length,
    dirtyMissingFromSupportFirst16: dirtyMissingFromSupport.slice(0, 16),
    supportOutsideDirtyCount: supportOutsideDirty.length,
    supportOutsideDirtyFirst16: supportOutsideDirty.slice(0, 16),
    changedCount,
    changedUnique: new Set(changedKeys).size,
    firstChangedKeys: Array.from(changedKeys.slice(0, 16)),
    activeCountByChangedChebyshevRadius,
    terminalSparseTopologyCount: activeKeys.length,
    terminalSparseTopologyFirst16: activeKeys.slice(0, 16),
    dirtyDistanceHistogram: pageDistanceHistogram(dirty),
    supportDistanceHistogram: pageDistanceHistogram(support),
    firstDirtyPages: Array.from(dirty.slice(0, 16)),
    firstSupportPages: Array.from(support.slice(0, 16)),
    firstError,
    errorPageScratch,
    transportErrorSample,
  }));
}
