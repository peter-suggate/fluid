import type { GPUSolverInstance } from "../lib/methods/types";
import { damBreakFractions } from "../lib/initial-fluid";
import { boundingRadius, initializeRigidBodies, type RigidBodyState } from "../lib/rigid-body";
import type { SceneDescription } from "../lib/model";
import { VOXEL_MATERIAL_IDS } from "../lib/voxel-scene";
import type { SparseVoxelSceneRenderSource } from "../lib/webgpu-voxel-debug";
import {
  activeCubeCapacity,
  RasterWaterPipeline,
  surfaceVertexCapacity,
  type WaterSurfaceGeometrySource,
} from "../lib/webgpu-water-pipeline";
import { createGlobalFineLevelSetConsumerSource } from "../lib/octree-consumer-sampling";
import type { WebGPUFineLevelSetBrickSource } from "../lib/webgpu-octree-fine-levelset-bricks";
import {
  unpackFineLevelSetPackedFlags,
  unpackFineLevelSetPackedPhi,
} from "../lib/fine-levelset-packed-sample";
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
import { rasterCubeSymmetryMetrics, rasterMeshSymmetryMetrics, sharpPatchRasterMetrics,
  type RasterCubeSymmetryMetrics, type RasterMeshSymmetryMetrics,
  type SharpPatchRasterMetrics } from "./raster-mesh-symmetry";

function exactWordFingerprint(values: ArrayLike<number>, integer = false): Readonly<{
  length: number; hashA: string; hashB: string;
}> {
  const scratch = new ArrayBuffer(4), f32 = new Float32Array(scratch), u32 = new Uint32Array(scratch);
  let a = 0x811c9dc5, b = 0x9e3779b9;
  for (let index = 0; index < values.length; index += 1) {
    if (integer) u32[0] = Number(values[index]) >>> 0;
    else f32[0] = Number(values[index]);
    const word = u32[0]!;
    a = Math.imul(a ^ word, 0x01000193) >>> 0;
    b = (Math.imul(b ^ (word + index), 0x85ebca6b) + 0xc2b2ae35) >>> 0;
  }
  return Object.freeze({ length: values.length,
    hashA: a.toString(16).padStart(8, "0"), hashB: b.toString(16).padStart(8, "0") });
}

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

function decodePackedFineSamples(words: Uint32Array): { flags: Uint32Array; phi: Float32Array } {
  const flags = new Uint32Array(words.length), phi = new Float32Array(words.length);
  for (let index = 0; index < words.length; index += 1) {
    flags[index] = unpackFineLevelSetPackedFlags(words[index]!);
    phi[index] = unpackFineLevelSetPackedPhi(words[index]!);
  }
  return { flags, phi };
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
  const [worklistBytes, metadataBytes, sampleBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, (7 + pageCapacity) * 4),
    readBufferBinding(device, { buffer: source.metadata }, pageCapacity * 16),
    readBufferBinding(device, { buffer: source.samples }, payloadWords * 4),
  ]);
  const worklist = new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4);
  const packed = new Uint32Array(sampleBytes.buffer, sampleBytes.byteOffset, sampleBytes.byteLength / 4);
  const { flags, phi } = decodePackedFineSamples(packed);
  const pages = new Map<number, number>();
  const activePages = Math.min(worklist[1] ?? 0, pageCapacity);
  for (let work = 0; work < activePages; work += 1) {
    const id = worklist[7 + work] ?? 0xffff_ffff;
    if (id >= pageCapacity || metadata[4 * id + 2] !== source.generation) continue;
    pages.set(metadata[4 * id + 1]!, id);
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

export interface FluidBrickSnapshot { resident: number; core: number; halo: number; generation: number }
export async function readFluidBrickSnapshot(device: GPUDevice, source: SparseVoxelSceneRenderSource): Promise<FluidBrickSnapshot | undefined> {
  if (!source.fluidBrickStats) return undefined;
  const words = new Uint32Array((await readBufferBinding(device, source.fluidBrickStats, 64)).buffer);
  return { resident: words[0], core: words[8], halo: words[9], generation: words[15] };
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
  /** Generation stamped by the scalar source whose cubes produced this mesh. */
  meshPublicationGeneration?: number;
  globalFineCrossingPublished?: boolean;
  presentationFallbackActive?: boolean;
  vertexCount?: number;
  activeCubeCount?: number;
  vertexAllocator?: number;
  vertexCapacity?: number;
  activeCubeCapacity?: number;
  /** Exact unordered D4 audit of emitted position+normal records. */
  surfaceMeshSymmetry?: RasterMeshSymmetryMetrics;
  /** Exact D4 audit of classified cube origins and spans before polygonization. */
  activeCubeSymmetry?: RasterCubeSymmetryMetrics;
  /** Production classified-cube/offset/vertex audit for explicit sharp rectangles. */
  sharpPatchRaster?: SharpPatchRasterMetrics;
  /** Exact D4 audit of every valid sparse fine-phi sample feeding extraction. */
  finePhiSymmetry?: FinePhiSymmetryMetrics;
  narrowVerticalSlits: NarrowVerticalSlitMetrics;
  enclosedSurfaceHoles: {
    front: EnclosedSurfaceHoleMetrics;
    back: EnclosedSurfaceHoleMetrics;
  };
  /** Holes remaining when front- and back-facing triangle coverage is united. */
  unionSurfaceHoles?: EnclosedSurfaceHoleMetrics;
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
    unionSurfaceHoles?: EnclosedSurfaceHoleMetrics;
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

export interface FinePhiSymmetryMetrics {
  readonly validSamples: number;
  readonly comparedSamples: number;
  readonly supportMismatchCount: number;
  readonly exactValueMismatchCount: number;
  readonly nonFiniteCount: number;
  readonly maximumAbsoluteError: number;
  readonly firstMismatch?: Readonly<Record<string, unknown>>;
  readonly worstMismatch?: Readonly<Record<string, unknown>>;
}

export async function readFinePhiSymmetrySource(
  device: GPUDevice,
  source: Pick<WebGPUFineLevelSetBrickSource, "plan" | "generation" | "worklist" | "metadata" | "samples">,
): Promise<FinePhiSymmetryMetrics> {
  const { plan } = source, capacity = plan.maximumResidentBricks;
  const payloadWords = capacity * plan.samplesPerBrick;
  const [worklistBytes, metadataBytes, sampleBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, (7 + capacity) * 4),
    readBufferBinding(device, { buffer: source.metadata }, capacity * 16),
    readBufferBinding(device, { buffer: source.samples }, payloadWords * 4),
  ]);
  const worklist = new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset, worklistBytes.byteLength / 4);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4);
  const packed = new Uint32Array(sampleBytes.buffer, sampleBytes.byteOffset, sampleBytes.byteLength / 4);
  const [nx, ny, nz] = plan.sampleDimensions;
  if (nx !== nz) throw new Error("Horizontal fine-phi D4 audit requires equal x/z dimensions");
  const denseWords = new Uint32Array(nx * ny * nz), denseValues = new Float32Array(nx * ny * nz);
  const valid = new Uint8Array(nx * ny * nz);
  const r = plan.brickResolution, active = Math.min(worklist[1] ?? 0, capacity);
  let validSamples = 0, nonFiniteCount = 0;
  for (let work = 0; work < active; work += 1) {
    const id = worklist[7 + work] ?? 0xffff_ffff;
    if (id >= capacity || metadata[id * 4 + 2] !== source.generation) continue;
    const key = metadata[id * 4 + 1]!;
    const bz = Math.floor(key / (plan.brickDimensions[0] * plan.brickDimensions[1]));
    const rem = key - bz * plan.brickDimensions[0] * plan.brickDimensions[1];
    const by = Math.floor(rem / plan.brickDimensions[0]), bx = rem - by * plan.brickDimensions[0];
    for (let local = 0; local < plan.samplesPerBrick; local += 1) {
      const qx = bx * r + local % r, qy = by * r + Math.floor(local / r) % r;
      const qz = bz * r + Math.floor(local / (r * r));
      const payload = id * plan.samplesPerBrick + local;
      if (qx >= nx || qy >= ny || qz >= nz || (unpackFineLevelSetPackedFlags(packed[payload]!) & 1) === 0) continue;
      const bits = packed[payload]! & 0xffff, value = unpackFineLevelSetPackedPhi(packed[payload]!);
      if (!Number.isFinite(value)) { nonFiniteCount += 1; continue; }
      const at = qx + nx * (qy + ny * qz);
      valid[at] = 1; denseWords[at] = bits; denseValues[at] = value; validSamples += 1;
    }
  }
  const transforms = [
    (x: number, z: number) => [nx - 1 - x, z] as const,
    (x: number, z: number) => [x, nz - 1 - z] as const,
    (x: number, z: number) => [z, x] as const,
  ];
  let comparedSamples = 0, supportMismatchCount = 0, exactValueMismatchCount = 0, maximumAbsoluteError = 0;
  let firstMismatch: Record<string, unknown> | undefined;
  let worstMismatch: Record<string, unknown> | undefined;
  for (const [transformName, transform] of [
    ["reflect-x", transforms[0]], ["reflect-z", transforms[1]], ["swap-xz", transforms[2]],
  ] as const) for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const at = x + nx * (y + ny * z);
    if (!valid[at]) continue;
    comparedSamples += 1;
    const [tx, tz] = transform(x, z), target = tx + nx * (y + ny * tz);
    if (!valid[target]) { supportMismatchCount += 1; continue; }
    const sourceBits = (denseWords[at]! & 0x7fff) === 0 ? 0 : denseWords[at]!;
    const targetBits = (denseWords[target]! & 0x7fff) === 0 ? 0 : denseWords[target]!;
    const absoluteError = Math.abs(denseValues[at]! - denseValues[target]!);
    if (sourceBits !== targetBits) {
      exactValueMismatchCount += 1;
      const detail = { transform: transformName, source: [x, y, z], target: [tx, y, tz],
        sourceBits, targetBits, sourceValue: denseValues[at], targetValue: denseValues[target], absoluteError };
      firstMismatch ??= detail;
      if (!worstMismatch || absoluteError > Number(worstMismatch.absoluteError)) worstMismatch = detail;
    }
    maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
  }
  return { validSamples, comparedSamples, supportMismatchCount, exactValueMismatchCount,
    nonFiniteCount, maximumAbsoluteError, firstMismatch, worstMismatch };
}

async function readFinePhiSymmetry(device: GPUDevice, solver: GPUSolverInstance): Promise<FinePhiSymmetryMetrics | undefined> {
  const source = solver.globalFineLevelSetSource;
  return source ? readFinePhiSymmetrySource(device, source) : undefined;
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
  const [worklistBytes, metadataBytes, sampleBytes, coarseBytes, seedBytes, topologyBytes,
    transportBytes, redistanceBytes, volumeBytes, promotionBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, source.worklist.size),
    readBufferBinding(device, { buffer: source.metadata }, pageCapacity * 16),
    readBufferBinding(device, { buffer: source.samples }, pageCapacity * samplesPerBrick * 4),
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
  const packed = new Uint32Array(sampleBytes.buffer, sampleBytes.byteOffset, sampleBytes.byteLength / 4);
  const { flags, phi } = decodePackedFineSamples(packed);
  const phiBits = packed;
  const activePages = Math.min(worklist[1], pageCapacity);
  let taggedMetadataPages = 0, malformedActivePages = 0;
  let validSamples = 0, finiteValidSamples = 0, negativeValidSamples = 0, positiveValidSamples = 0;
  let phiBitXor = 0, phiBitSum = 0, phiSum = 0, phiAbsSum = 0;
  for (let id = 0; id < pageCapacity; id += 1) if (metadata[id * 4 + 2] === source.generation) taggedMetadataPages += 1;
  for (let work = 0; work < activePages; work += 1) {
    const id = worklist[7 + work];
    if (id >= pageCapacity || metadata[id * 4 + 2] !== source.generation || metadata[id * 4] !== id) {
      malformedActivePages += 1; continue;
    }
    for (let local = 0; local < samplesPerBrick; local += 1) {
      const index = id * samplesPerBrick + local;
      if ((flags[index] & 1) === 0) continue;
      validSamples += 1;
      const value = phi[index];
      if (!Number.isFinite(value)) continue;
      finiteValidSamples += 1;
      const logicalSample = (Math.imul(metadata[id * 4 + 1], samplesPerBrick) + local) >>> 0;
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
      const id = worklist[directoryBase + key], base = id * 4;
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
        metadataKey: validId ? metadata[id * 4 + 1] : undefined,
        metadataGeneration: validId ? metadata[id * 4 + 2] : undefined,
        metadataMatchesGeneration: validId && metadata[id * 4] === id
          && metadata[id * 4 + 1] === key && metadata[id * 4 + 2] === source.generation,
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
    pageMetadataBytes: pageCapacity * 16,
    pageWorklistBytes: source.worklist.size,
    diagnosticReadbackBytes: [worklistBytes, metadataBytes, sampleBytes, coarseBytes, seedBytes,
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
  // Match FluidLabRenderer's shared presentation metadata so Dawn exercises
  // the closed-ceiling film path instead of silently treating every top as
  // open exterior air.
  packed.set([0, 0.38 * scene.container.height_m, 0,
    scene.container.top === "closed" ? 1 : 0], 8);
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
    const meshAuditRequested = process.env.FLUID_RASTER_MESH_SYMMETRY === "1"
      || process.env.FLUID_RASTER_MESH_AUDIT === "1";
    const finePhiSymmetry = meshAuditRequested
      ? await readFinePhiSymmetry(device, solver)
      : undefined;
    if (verifyGlobalFineAuthorityTransition && !globalFineLevelSet) {
      throw new Error("Global-fine authority transition requested without a published source");
    }
    // The same statement of the document the renderer makes every frame, so the
    // Dawn lane exercises the scene's own optics, key and caustic receiver
    // rather than the pipeline's clean-water constructor defaults.
    pipeline.setSceneOptics({
      optics: scene.fluid.optics,
      directional: scene.lighting?.directional,
      grade: scene.lighting?.grade,
      terrain: scene.terrain,
      container: { width_m: scene.container.width_m, depth_m: scene.container.depth_m },
    });
    pipeline.setVolume(solver.surfaceFieldTexture ?? solver.volumeTexture,
      solver.columnBaseTexture ?? columnFallback);
    pipeline.setGlobalFineLevelSet(globalFineLevelSet);
    pipeline.setCoarseLevelSet(solver.coarseLevelSetSource);
    pipeline.ensureSize(width, height);
    // Browser/manual stepping keeps one water pipeline alive while the compact
    // source advances on stable buffers. Reproduce the receipt lifecycle that
    // exposed the UI bug: a receipt is captured, another extraction completes
    // while telemetry is throttled, and an unchanged later frame must still
    // publish the new receipt without rebuilding the mesh.
    const verifyCoarseReceiptRecovery = process.env.FLUID_COARSE_SURFACE_RECEIPT_RECOVERY === "1";
    if (verifyCoarseReceiptRecovery) {
      const coarse = solver.coarseLevelSetSource;
      if (!coarse || coarse.generation < 2) {
        throw new Error("Compact-coarse receipt recovery requires a generation-2 source");
      }
      const encodeReceipt = async (label: string, revision: number, force: boolean) => {
        const encoder = device.createCommandEncoder({ label });
        const encoded = pipeline.encode(
          encoder, output.createView(), solver.info.nx, solver.info.ny, solver.info.nz,
          solver.info.gridKind === "restricted-tall-cell", solver.info.maximumNeighborDelta ?? 0,
          revision, undefined, undefined, undefined, force,
        );
        if (!encoded) throw new Error(`${label} did not encode`);
        device.queue.submit([encoder.finish()]);
        const diagnostics = encoded.surfaceDiagnosticsCaptured
          ? await pipeline.completeSurfaceDiagnostics() : undefined;
        await device.queue.onSubmittedWorkDone();
        return { encoded, diagnostics };
      };
      const baseline = await encodeReceipt("Baseline compact-coarse receipt", 100, true);
      if (baseline.diagnostics?.surfaceGeometrySource !== "compact-coarse"
        || baseline.diagnostics.meshPublicationGeneration !== coarse.generation) {
        throw new Error(`Baseline compact-coarse generation was not published: ${JSON.stringify(baseline.diagnostics)}`);
      }
      const immediate = await encodeReceipt("Throttled compact-coarse recovery", 101, false);
      if (!immediate.encoded.surfaceUpdated || immediate.encoded.surfaceDiagnosticsCaptured) {
        throw new Error("Compact-coarse recovery did not exercise the throttled persistent-pipeline handoff");
      }
      await new Promise((resolve) => setTimeout(resolve, 260));
      const recovered = await encodeReceipt("Deferred compact-coarse recovery receipt", 101, false);
      if (recovered.encoded.surfaceUpdated
        || recovered.diagnostics?.surfaceGeometrySource !== "compact-coarse"
        || recovered.diagnostics.meshPublicationGeneration !== coarse.generation
        || (recovered.diagnostics.vertexCount ?? 0) === 0) {
        throw new Error(`Deferred compact-coarse receipt did not recover: ${JSON.stringify(recovered)}`);
      }
      console.info(JSON.stringify({ phase: "compact-coarse-receipt-recovery",
        generation: coarse.generation, vertexCount: recovered.diagnostics.vertexCount,
        activeCubeCount: recovered.diagnostics.activeCubeCount }));
    }
    const capture = async (label: string, revision: number) => {
      const frameStarted = performance.now();
      const encoder = device.createCommandEncoder({ label });
      const encoded = pipeline.encode(
        encoder, output.createView(), solver.info.nx, solver.info.ny, solver.info.nz,
        solver.info.gridKind === "restricted-tall-cell", solver.info.maximumNeighborDelta ?? 0,
        revision, undefined, undefined, undefined,
        verifyGlobalFineAuthorityTransition || verifyCoarseReceiptRecovery,
      );
      if (!encoded) throw new Error("Hybrid presentation pipeline did not encode a frame");
      const interfaceCapture = pipeline.diagnosticCaptureTexture("interface-positions");
      if (!interfaceCapture) throw new Error("Hybrid presentation did not expose its front interface target");
      const backInterfaceCapture = pipeline.diagnosticCaptureTexture("back-interface-positions");
      if (!backInterfaceCapture) throw new Error("Hybrid presentation did not expose its back interface target");
      // Production interface positions are rgba32float: thin wall films need
      // the full precision when the smoke compares independently rasterized
      // front/back surfaces, just as the optical compositor does.
      const interfaceBytesPerRow = Math.ceil(width * 16 / 256) * 256;
      const interfacePlaneBytes = interfaceBytesPerRow * height;
      const interfaceReadback = device.createBuffer({ size: 2 * interfacePlaneBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        encoder.copyTextureToBuffer({ texture: interfaceCapture.texture }, { buffer: interfaceReadback, bytesPerRow: interfaceBytesPerRow, rowsPerImage: height }, [width, height]);
        encoder.copyTextureToBuffer({ texture: backInterfaceCapture.texture }, { buffer: interfaceReadback, offset: interfacePlaneBytes, bytesPerRow: interfaceBytesPerRow, rowsPerImage: height }, [width, height]);
        device.queue.submit([encoder.finish()]);
        const presentationDiagnostics = await pipeline.completeSurfaceDiagnostics();
        await device.queue.onSubmittedWorkDone();
        let surfaceMeshSymmetry: RasterMeshSymmetryMetrics | undefined;
        let activeCubeSymmetry: RasterCubeSymmetryMetrics | undefined;
        let sharpPatchRaster: SharpPatchRasterMetrics | undefined;
        if (meshAuditRequested && presentationDiagnostics?.vertexCount) {
          const source = pipeline.diagnosticSurfaceVertexSource();
          if (!source) throw new Error("Raster mesh symmetry requested without an emitted vertex source");
          const vertexBytes = presentationDiagnostics.vertexCount * source.strideBytes;
          const cubeBytes = presentationDiagnostics.activeCubeCount * 8;
          const offsetBytes = presentationDiagnostics.activeCubeCount * 6 * 4;
          const byteLength = vertexBytes + cubeBytes + offsetBytes;
          const meshReadback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          try {
            const meshEncoder = device.createCommandEncoder({ label: "Read exact raster mesh symmetry" });
            meshEncoder.copyBufferToBuffer(source.buffer, 0, meshReadback, 0, vertexBytes);
            meshEncoder.copyBufferToBuffer(source.classifiedCubes, 0, meshReadback, vertexBytes, cubeBytes);
            meshEncoder.copyBufferToBuffer(source.classifiedOffsets, 0, meshReadback,
              vertexBytes + cubeBytes, offsetBytes);
            device.queue.submit([meshEncoder.finish()]);
            await meshReadback.mapAsync(GPUMapMode.READ);
            const mapped = meshReadback.getMappedRange();
            const vertices = new Float32Array(mapped, 0, presentationDiagnostics.vertexCount * 8);
            const cubes = new Uint32Array(mapped, vertexBytes, presentationDiagnostics.activeCubeCount * 2);
            const offsets = new Uint32Array(mapped, vertexBytes + cubeBytes,
              presentationDiagnostics.activeCubeCount * 6);
            surfaceMeshSymmetry = rasterMeshSymmetryMetrics(vertices, presentationDiagnostics.vertexCount, {
              minimum: [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m],
              maximum: [0.5 * scene.container.width_m, scene.container.height_m,
                0.5 * scene.container.depth_m],
              tolerance: Math.max(1e-6, 1e-4 * scene.voxelDomain.finestCellSize_m),
            }, { cubes, offsets, cubeCount: presentationDiagnostics.activeCubeCount });
            activeCubeSymmetry = rasterCubeSymmetryMetrics(cubes,
              presentationDiagnostics.activeCubeCount,
              globalFineLevelSet?.sampleDimensions
                ?? [solver.info.nx, solver.info.ny, solver.info.nz]);
            sharpPatchRaster = sharpPatchRasterMetrics(vertices, presentationDiagnostics.vertexCount,
              cubes, offsets, presentationDiagnostics.activeCubeCount);
            meshReadback.unmap();
          } finally {
            meshReadback.destroy();
          }
        }
        await interfaceReadback.mapAsync(GPUMapMode.READ);
        const interfaceRange = interfaceReadback.getMappedRange();
        const interfaceValues = new Float32Array(interfaceRange);
        const interfaceWords = new Uint32Array(interfaceRange);
        const interfaceRowWords = interfaceBytesPerRow / 4;
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
          const backAt = interfacePlaneBytes / 4 + at;
          for (let channel = 0; channel < 4; channel += 1) {
            frontInterfaceHash = fold(frontInterfaceHash, interfaceWords[at + channel]);
            backInterfaceHash = fold(backInterfaceHash, interfaceWords[backAt + channel]);
          }
          const frontPresent = interfaceValues[at + 3] !== 0;
          const backPresent = interfaceValues[backAt + 3] !== 0;
          if (frontPresent) {
            frontInterfacePixels += 1;
            frontMask[x + y * width] = 1;
            const px = interfaceValues[at];
            const py = interfaceValues[at + 1];
            const pz = interfaceValues[at + 2];
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
              const value = interfaceValues[at + axis];
              frontMinimum[axis] = Math.min(frontMinimum[axis], value);
              frontMaximum[axis] = Math.max(frontMaximum[axis], value);
            }
          }
          if (backPresent) {
            backInterfacePixels += 1;
            backMask[x + y * width] = 1;
            const backY = interfaceValues[backAt + 1];
            if (Math.abs(backY - scene.container.height_m) <= wallPlaneTolerance) {
              backCeilingContactPixels += 1;
            }
            backPositions.set([
              interfaceValues[backAt],
              backY,
              interfaceValues[backAt + 2],
            ], (x + y * width) * 3);
          }
          if (frontPresent && backPresent) pairedInterfacePixels += 1;
          else if (frontPresent) frontOnlyInterfacePixels += 1;
          else if (backPresent) {
            backOnlyInterfacePixels += 1;
            if (backOnlyInterfaceLocations.length < 16) {
              backOnlyInterfaceLocations.push([x, y]);
              backOnlyInterfacePositions_m.push([
                interfaceValues[backAt],
                interfaceValues[backAt + 1],
                interfaceValues[backAt + 2],
              ]);
            }
          }
        }
        const narrowVerticalSlits = narrowVerticalSlitMetrics(frontMask, width, height);
        const enclosedSurfaceHoles = {
          front: enclosedSurfaceHoleMetrics(frontMask, width, height),
          back: enclosedSurfaceHoleMetrics(backMask, width, height),
        };
        const unionMask = new Uint8Array(frontMask.length);
        for (let index = 0; index < unionMask.length; index += 1) {
          unionMask[index] = Number(frontMask[index] !== 0 || backMask[index] !== 0);
        }
        const unionSurfaceHoles = enclosedSurfaceHoleMetrics(unionMask, width, height);
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
          unionSurfaceHoles,
          surfaceSteps,
          ...(surfaceMeshSymmetry ? { surfaceMeshSymmetry } : {}),
          ...(activeCubeSymmetry ? { activeCubeSymmetry } : {}),
          ...(sharpPatchRaster ? { sharpPatchRaster } : {}),
          ceilingContactPixels: { front: frontCeilingContactPixels, back: backCeilingContactPixels },
          wallCornerCapPixels,
          wallCornerMaximumY_m,
          damExposedCornerCapPixels,
          ...(presentationDiagnostics ? {
            surfaceGeometrySource: presentationDiagnostics.surfaceGeometrySource,
            globalFineAuthorityLatch: presentationDiagnostics.globalFineAuthorityLatch,
            meshPublicationGeneration: presentationDiagnostics.meshPublicationGeneration,
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
      unionSurfaceHoles: reverse.unionSurfaceHoles,
      surfaceSteps: reverse.surfaceSteps,
      ceilingContactPixels: reverse.ceilingContactPixels,
      wallCornerCapPixels: reverse.wallCornerCapPixels,
      wallCornerMaximumY_m: reverse.wallCornerMaximumY_m,
      damExposedCornerCapPixels: reverse.damExposedCornerCapPixels,
      ...(reverse.frontInterfaceBounds_m ? { frontInterfaceBounds_m: reverse.frontInterfaceBounds_m } : {}),
    }, rendererValidationErrorCount: 0, rendererUncapturedErrorCount: 0,
      ...(finePhiSymmetry ? { finePhiSymmetry } : {}),
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
  locationDensity: number;
  positiveNeighborDensity: number | null;
  nonFiniteCount: number;
  kineticEnergyProxy: number;
  maximumComponentCfl: number;
  maximumLiquidDivergence_s: number;
  maximumLiquidDivergenceLocation: number[];
  maximumLiquidDivergenceSigned_s: number;
  maximumLiquidDivergenceDensity: number;
  maximumLiquidDivergenceFaces: number[];
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

/** Read an rgba16float 3D texture without format conversion. */
export async function readRgba16Texture3D(device: GPUDevice, texture: GPUTexture, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 8 / 256) * 256;
  const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, { width, height, depthOrArrayLayers: depth });
  device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(buffer.getMappedRange());
  const output = new Float32Array(width * height * depth * 4);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Uint16Array(bytes.buffer, bytes.byteOffset + bytesPerRow * (y + height * z), width * 4);
    const outputOffset = width * 4 * (y + height * z);
    for (let index = 0; index < row.length; index += 1) output[outputOffset + index] = decodeFloat16(row[index]!);
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
  // The uniform pressure system owns exactly rho > 0.5 cells. Including the
  // conservative transport halo (0 < rho <= 0.5) falsely attributes its
  // intentionally unprojected divergence to the liquid pressure solve.
  const pressureLiquid = (rho: number) => rho > 0.5;
  let maximum = 0, liquidMaximum = 0, location = [0, 0, 0], component = 0, nonFiniteCount = 0;
  let kineticEnergyProxy = 0, maximumComponentCfl = 0;
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const index = x + width * (y + height * z), value = velocity[3 * index + axis];
      if (!Number.isFinite(value)) { nonFiniteCount += 1; continue; }
      const speed = Math.abs(value); if (speed > maximum) { maximum = speed; location = [x, y, z]; component = axis; }
      if (pressureLiquid(volume[index]) && speed > liquidMaximum) liquidMaximum = speed;
      maximumComponentCfl = Math.max(maximumComponentCfl, speed * dt_s / [spacing.x, spacing.y, spacing.z][axis]);
      kineticEnergyProxy += 0.5 * Math.max(0, Math.min(1, volume[index])) * value * value * spacing.x * spacing.y * spacing.z;
    }
  }
  const [maximumX, maximumY, maximumZ] = location;
  const maximumCell = maximumX + width * (maximumY + height * maximumZ);
  const neighborCoordinates = [maximumX, maximumY, maximumZ];
  neighborCoordinates[component] += 1;
  const neighborInDomain = neighborCoordinates[0] < width
    && neighborCoordinates[1] < height && neighborCoordinates[2] < depth;
  const neighborCell = neighborCoordinates[0]
    + width * (neighborCoordinates[1] + height * neighborCoordinates[2]);
  let maximumLiquidDivergence_s = 0, maximumLiquidDivergenceSigned_s = 0;
  let maximumLiquidDivergenceLocation = [0, 0, 0], maximumLiquidDivergenceDensity = 0;
  let maximumLiquidDivergenceFaces = [0, 0, 0, 0, 0, 0];
  let divergenceSquared = 0, liquidCells = 0;
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
    if (!pressureLiquid(volume[index])) continue;
    const divergence = divergenceStencil === "centered"
      ? centered(x, y, z, 0) + centered(x, y, z, 1) + centered(x, y, z, 2)
      : (at(x, y, z, 0) - (x > 0 ? at(x - 1, y, z, 0) : 0)) / spacing.x
        + (at(x, y, z, 1) - (y > 0 ? at(x, y - 1, z, 1) : 0)) / spacing.y
        + (at(x, y, z, 2) - (z > 0 ? at(x, y, z - 1, 2) : 0)) / spacing.z;
    if (!Number.isFinite(divergence)) { nonFiniteCount += 1; continue; }
    if (Math.abs(divergence) > maximumLiquidDivergence_s) {
      maximumLiquidDivergence_s = Math.abs(divergence);
      maximumLiquidDivergenceSigned_s = divergence;
      maximumLiquidDivergenceLocation = [x, y, z];
      maximumLiquidDivergenceDensity = Number(volume[index]);
      maximumLiquidDivergenceFaces = [
        x > 0 ? at(x - 1, y, z, 0) : 0, at(x, y, z, 0),
        y > 0 ? at(x, y - 1, z, 1) : 0, at(x, y, z, 1),
        z > 0 ? at(x, y, z - 1, 2) : 0, at(x, y, z, 2),
      ];
    }
    divergenceSquared += divergence * divergence; liquidCells += 1;
  }
  return {
    maximum, liquidMaximum, location, component,
    locationDensity: Number(volume[maximumCell]),
    positiveNeighborDensity: neighborInDomain ? Number(volume[neighborCell]) : null,
    nonFiniteCount, kineticEnergyProxy, maximumComponentCfl,
    maximumLiquidDivergence_s,
    maximumLiquidDivergenceLocation,
    maximumLiquidDivergenceSigned_s,
    maximumLiquidDivergenceDensity,
    maximumLiquidDivergenceFaces,
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
  if (!controlBuffer || !headerBuffer || !velocityBuffer) {
    const losasso = solver.losassoVelocityDebug;
    if (!losasso || losasso.dimensions.some((value, axis) => value !== dimensions[axis])) {
      return undefined;
    }
    const controlBytes = await readBufferBinding(device, { buffer: losasso.control }, 32);
    const control = new Uint32Array(controlBytes.buffer, controlBytes.byteOffset, 8);
    const faceCount = control[2] ?? 0;
    if (control[3] !== 1 || faceCount === 0
      || faceCount * 16 > losasso.faceGeometry.size
      || faceCount * 4 > losasso.extendedVelocity.size) return undefined;
    const [geometryBytes, velocityBytes] = await Promise.all([
      readBufferBinding(device, { buffer: losasso.faceGeometry }, faceCount * 16),
      readBufferBinding(device, { buffer: losasso.extendedVelocity }, faceCount * 4),
    ]);
    const geometry = new Uint32Array(geometryBytes.buffer, geometryBytes.byteOffset, 4 * faceCount);
    const values = new Float32Array(velocityBytes.buffer, velocityBytes.byteOffset, faceCount);
    const faces = new Map<string, number>();
    const key = (packed: number, x: number, y: number, z: number) => `${packed}|${x}|${y}|${z}`;
    for (let face = 0; face < faceCount; face += 1) {
      const at = 4 * face;
      faces.set(key(geometry[at]!, geometry[at + 1]!, geometry[at + 2]!, geometry[at + 3]!),
        values[face]!);
    }
    const [nx, ny, nz] = dimensions;
    const sample = (axis: number, x: number, y: number, z: number): number | undefined => {
      const q = [x, y, z];
      for (let span = 1, logSpan = 0; span <= losasso.maximumLeafSize;
        span *= 2, logSpan += 1) {
        const origin = [...q];
        for (let component = 0; component < 3; component += 1) {
          if (component !== axis) origin[component] = Math.floor(origin[component]! / span) * span;
        }
        const found = faces.get(key(axis | (logSpan << 2), origin[0]!, origin[1]!, origin[2]!));
        if (found !== undefined) return found;
      }
      // The compact W7 authority materializes live wet/extension support, not
      // the quiescent air complement. Reconstruct an omitted QA face as the
      // represented zero value; production consumers still fail closed when a
      // demanded wet/transport face is absent.
      return 0;
    };
    const field = new Float32Array(3 * nx * ny * nz);
    field.fill(Number.NaN);
    let coveredCells = 0;
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const cell = x + nx * (y + ny * z);
        let complete = true;
        for (let axis = 0; axis < 3; axis += 1) {
          const lower = sample(axis, x, y, z);
          const upper = sample(axis, x + (axis === 0 ? 1 : 0),
            y + (axis === 1 ? 1 : 0), z + (axis === 2 ? 1 : 0));
          if (lower === undefined || upper === undefined
            || !Number.isFinite(lower) || !Number.isFinite(upper)) complete = false;
          else field[3 * cell + axis] = 0.5 * (lower + upper);
        }
        if (complete) coveredCells += 1;
      }
    }
    return {
      field, coveredCells, overlapCells: 0, invalidRows: 0,
      publicationValid: control[3] === 1 && (control[4] ?? 0) === 0,
      rowCount: faceCount, reconstructedRows: faceCount,
    };
  }
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

export interface CompactOctreePressureStateRaster {
  readonly pressure: Float32Array;
  readonly rhs: Float32Array;
  readonly diagonal: Float32Array;
  readonly section63Diagonal?: Float32Array;
  readonly section63CaseId?: Uint32Array;
  /** Diagnostic-only accepted row-major diagonal + eighteen coefficients. */
  readonly section63CoefficientRows?: Float32Array;
  readonly initialResidual?: Float32Array;
  readonly initialPreconditioned?: Float32Array;
  readonly initialPreconditionedImage?: Float32Array;
  readonly preconditionerPreSmoothed?: Float32Array;
  readonly preconditionerZeroSmoothed?: Float32Array;
  readonly preconditionerFirstOperatorImage?: Float32Array;
  readonly preconditionerFirstSmoothed?: Float32Array;
  readonly preconditionerInnerResidual?: Float32Array;
  readonly preconditionerInnerCorrection?: Float32Array;
  readonly preconditionerPostCorrected?: Float32Array;
  readonly topology: Uint32Array;
  readonly coveredCells: number;
  readonly overlapCells: number;
  readonly invalidRows: number;
  readonly rowCount: number;
  readonly publicationValid: boolean;
}

/**
 * Expand the compact pressure unknown and its adaptive leaf size onto the
 * finest QA lattice. This is diagnostic-only: production continues to consume
 * the compact row buffers directly.
 */
export async function readCompactOctreePressureState3D(
  device: GPUDevice,
  solver: GPUSolverInstance,
  dimensions: readonly [number, number, number],
): Promise<CompactOctreePressureStateRaster | undefined> {
  const structured = solver as GPUSolverInstance & { structuredVelocityControl?: GPUBuffer };
  const controlBuffer = structured.structuredVelocityControl;
  const headerBuffer = solver.powerLeafHeaders;
  const pressureBuffer = solver.powerPressureBuffer;
  if (!controlBuffer && headerBuffer && pressureBuffer && solver.losassoPressureDebug) {
    const debug = solver.losassoPressureDebug;
    const controlBytes = await readBufferBinding(device, { buffer: debug.control }, 32);
    const control = new Uint32Array(controlBytes.buffer, controlBytes.byteOffset, 8);
    const rowCount = control[1] ?? 0;
    if (control[3] !== 1 || rowCount === 0 || rowCount * 48 > headerBuffer.size
      || rowCount * 4 > pressureBuffer.size || rowCount * 4 > debug.rightHandSide.size
      || rowCount * 4 > debug.diagonal.size) return undefined;
    const stageBuffers = (solver as GPUSolverInstance & { workAccountingBuffers?: {
      symmetryInitialResidual?: GPUBufferBinding;
      symmetryInitialPreconditioned?: GPUBufferBinding;
      symmetryInitialPreconditionedImage?: GPUBufferBinding;
      symmetryPreconditionerPreSmoothed?: GPUBufferBinding;
      symmetryPreconditionerZeroSmoothed?: GPUBufferBinding;
      symmetryPreconditionerFirstOperatorImage?: GPUBufferBinding;
      symmetryPreconditionerFirstSmoothed?: GPUBufferBinding;
      symmetryPreconditionerInnerResidual?: GPUBufferBinding;
      symmetryPreconditionerInnerCorrection?: GPUBufferBinding;
      symmetryPreconditionerPostCorrected?: GPUBufferBinding;
    } }).workAccountingBuffers;
    const stageBindings = [stageBuffers?.symmetryInitialResidual,
      stageBuffers?.symmetryInitialPreconditioned,
      stageBuffers?.symmetryInitialPreconditionedImage,
      stageBuffers?.symmetryPreconditionerPreSmoothed,
      stageBuffers?.symmetryPreconditionerZeroSmoothed,
      stageBuffers?.symmetryPreconditionerFirstOperatorImage,
      stageBuffers?.symmetryPreconditionerFirstSmoothed,
      stageBuffers?.symmetryPreconditionerInnerResidual,
      stageBuffers?.symmetryPreconditionerInnerCorrection,
      stageBuffers?.symmetryPreconditionerPostCorrected] as const;
    const [headerBytes, pressureBytes, rhsBytes, diagonalBytes, ...stageBytes] = await Promise.all([
      readBufferBinding(device, { buffer: headerBuffer }, rowCount * 48),
      readBufferBinding(device, { buffer: pressureBuffer }, rowCount * 4),
      readBufferBinding(device, { buffer: debug.rightHandSide }, rowCount * 4),
      readBufferBinding(device, { buffer: debug.diagonal }, rowCount * 4),
      ...stageBindings.map((binding) => binding
        ? readBufferBinding(device, binding, rowCount * 4) : undefined),
    ]);
    const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
    const rows = new Float32Array(pressureBytes.buffer, pressureBytes.byteOffset, rowCount);
    const rhsRows = new Float32Array(rhsBytes.buffer, rhsBytes.byteOffset, rowCount);
    const diagonalRows = new Float32Array(diagonalBytes.buffer, diagonalBytes.byteOffset, rowCount);
    const stageRows = stageBytes.map((bytes) => bytes
      ? new Float32Array(bytes.buffer, bytes.byteOffset, rowCount) : undefined);
    const [nx, ny, nz] = dimensions, cellCount = nx * ny * nz;
    const pressure = new Float32Array(cellCount), rhs = new Float32Array(cellCount);
    const diagonal = new Float32Array(cellCount), topology = new Uint32Array(cellCount);
    pressure.fill(Number.NaN); rhs.fill(Number.NaN); diagonal.fill(Number.NaN);
    const stageFields = stageRows.map((rows) => {
      if (!rows) return undefined;
      const field = new Float32Array(cellCount); field.fill(Number.NaN); return field;
    });
    const owners = new Int32Array(cellCount); owners.fill(-1);
    let coveredCells = 0, overlapCells = 0, invalidRows = 0;
    for (let row = 0; row < rowCount; row += 1) {
      const cell = headers[12 * row] >>> 0, size = headers[12 * row + 3] >>> 0;
      const origin = [cell % nx, Math.floor(cell / nx) % ny,
        Math.floor(cell / (nx * ny))] as const;
      const valid = size > 0 && cell < cellCount
        && origin[0] + size <= nx && origin[1] + size <= ny && origin[2] + size <= nz
        && Number.isFinite(rows[row]) && Number.isFinite(rhsRows[row])
        && Number.isFinite(diagonalRows[row]);
      if (!valid) invalidRows += 1;
      if (size === 0 || cell >= cellCount || origin[0] + size > nx
        || origin[1] + size > ny || origin[2] + size > nz) continue;
      for (let z = origin[2]; z < origin[2] + size; z += 1) {
        for (let y = origin[1]; y < origin[1] + size; y += 1) {
          for (let x = origin[0]; x < origin[0] + size; x += 1) {
            const index = x + nx * (y + ny * z);
            if (owners[index] >= 0) { overlapCells += 1; pressure[index] = Number.NaN;
              rhs[index] = Number.NaN; diagonal[index] = Number.NaN; topology[index] = 0; continue; }
            owners[index] = row; coveredCells += 1; topology[index] = size;
            if (valid) { pressure[index] = rows[row]!; rhs[index] = rhsRows[row]!;
              diagonal[index] = diagonalRows[row]!;
              for (let stage = 0; stage < stageFields.length; stage += 1) {
                if (stageFields[stage] && stageRows[stage]) {
                  stageFields[stage]![index] = stageRows[stage]![row]!;
                }
              }
            }
          }
        }
      }
    }
    const [initialResidual, initialPreconditioned, initialPreconditionedImage,
      preconditionerPreSmoothed, preconditionerZeroSmoothed,
      preconditionerFirstOperatorImage, preconditionerFirstSmoothed,
      preconditionerInnerResidual, preconditionerInnerCorrection,
      preconditionerPostCorrected] = stageFields;
    return { pressure, rhs, diagonal, topology, coveredCells, overlapCells, invalidRows, rowCount,
      ...(initialResidual ? { initialResidual } : {}),
      ...(initialPreconditioned ? { initialPreconditioned } : {}),
      ...(initialPreconditionedImage ? { initialPreconditionedImage } : {}),
      ...(preconditionerPreSmoothed ? { preconditionerPreSmoothed } : {}),
      ...(preconditionerZeroSmoothed ? { preconditionerZeroSmoothed } : {}),
      ...(preconditionerFirstOperatorImage ? { preconditionerFirstOperatorImage } : {}),
      ...(preconditionerFirstSmoothed ? { preconditionerFirstSmoothed } : {}),
      ...(preconditionerInnerResidual ? { preconditionerInnerResidual } : {}),
      ...(preconditionerInnerCorrection ? { preconditionerInnerCorrection } : {}),
      ...(preconditionerPostCorrected ? { preconditionerPostCorrected } : {}),
      publicationValid: control[3] === 1 && (control[4] ?? 0) === 0
        && coveredCells === cellCount && overlapCells === 0 && invalidRows === 0 };
  }
  if (!controlBuffer || !headerBuffer || !pressureBuffer) return undefined;
  const controlBytes = await readBufferBinding(device, { buffer: controlBuffer }, 24);
  const control = unpackStructuredVelocityControl(new Uint32Array(
    controlBytes.buffer, controlBytes.byteOffset, controlBytes.byteLength / 4));
  const rowCount = control.rowCount;
  if (rowCount === 0 || rowCount * 48 > headerBuffer.size || rowCount * 4 > pressureBuffer.size) return undefined;
  const stageBuffers = (solver as GPUSolverInstance & { workAccountingBuffers?: {
    pressureRhs?: GPUBufferBinding;
    symmetryInitialResidual?: GPUBufferBinding;
    symmetryInitialPreconditioned?: GPUBufferBinding;
    symmetryInitialPreconditionedImage?: GPUBufferBinding;
    symmetryPreconditionerPreSmoothed?: GPUBufferBinding;
    symmetryPreconditionerZeroSmoothed?: GPUBufferBinding;
    symmetryPreconditionerFirstOperatorImage?: GPUBufferBinding;
    symmetryPreconditionerFirstSmoothed?: GPUBufferBinding;
    symmetryPreconditionerInnerResidual?: GPUBufferBinding;
    symmetryPreconditionerInnerCorrection?: GPUBufferBinding;
    symmetryPreconditionerPostCorrected?: GPUBufferBinding;
    section63Coefficients?: GPUBufferBinding;
  } }).workAccountingBuffers;
  const topologyMetricsBuffer = (solver as GPUSolverInstance & {
    powerTopologyMetrics?: GPUBuffer;
  }).powerTopologyMetrics;
  const [headerBytes, pressureBytes, rhsBytes, initialResidualBytes, initialPreconditionedBytes,
    initialPreconditionedImageBytes, preSmoothedBytes, zeroSmoothedBytes,
    firstOperatorImageBytes, firstSmoothedBytes, innerResidualBytes,
    innerCorrectionBytes, postCorrectedBytes, section63Bytes, topologyMetricBytes] = await Promise.all([
    readBufferBinding(device, { buffer: headerBuffer }, rowCount * 48),
    readBufferBinding(device, { buffer: pressureBuffer }, rowCount * 4),
    stageBuffers?.pressureRhs
      ? readBufferBinding(device, stageBuffers.pressureRhs, rowCount * 4) : undefined,
    stageBuffers?.symmetryInitialResidual
      ? readBufferBinding(device, stageBuffers.symmetryInitialResidual, rowCount * 4) : undefined,
    stageBuffers?.symmetryInitialPreconditioned
      ? readBufferBinding(device, stageBuffers.symmetryInitialPreconditioned, rowCount * 4) : undefined,
    stageBuffers?.symmetryInitialPreconditionedImage
      ? readBufferBinding(device, stageBuffers.symmetryInitialPreconditionedImage, rowCount * 4) : undefined,
    stageBuffers?.symmetryPreconditionerPreSmoothed
      ? readBufferBinding(device, stageBuffers.symmetryPreconditionerPreSmoothed, rowCount * 4) : undefined,
    stageBuffers?.symmetryPreconditionerZeroSmoothed
      ? readBufferBinding(device, stageBuffers.symmetryPreconditionerZeroSmoothed, rowCount * 4) : undefined,
    stageBuffers?.symmetryPreconditionerFirstOperatorImage
      ? readBufferBinding(device, stageBuffers.symmetryPreconditionerFirstOperatorImage, rowCount * 4) : undefined,
    stageBuffers?.symmetryPreconditionerFirstSmoothed
      ? readBufferBinding(device, stageBuffers.symmetryPreconditionerFirstSmoothed, rowCount * 4) : undefined,
    stageBuffers?.symmetryPreconditionerInnerResidual
      ? readBufferBinding(device, stageBuffers.symmetryPreconditionerInnerResidual, rowCount * 4) : undefined,
    stageBuffers?.symmetryPreconditionerInnerCorrection
      ? readBufferBinding(device, stageBuffers.symmetryPreconditionerInnerCorrection, rowCount * 4) : undefined,
    stageBuffers?.symmetryPreconditionerPostCorrected
      ? readBufferBinding(device, stageBuffers.symmetryPreconditionerPostCorrected, rowCount * 4) : undefined,
    stageBuffers?.section63Coefficients
      ? readBufferBinding(device, {
        buffer: stageBuffers.section63Coefficients.buffer,
        offset: control.activeBank * Number(stageBuffers.section63Coefficients.size),
      }, rowCount * 19 * 4) : undefined,
    topologyMetricsBuffer
      ? readBufferBinding(device, { buffer: topologyMetricsBuffer }, rowCount * 16) : undefined,
  ]);
  const headers = new Uint32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const headerFloats = new Float32Array(headerBytes.buffer, headerBytes.byteOffset, rowCount * 12);
  const rows = new Float32Array(pressureBytes.buffer, pressureBytes.byteOffset, rowCount);
  const rhsRows = rhsBytes
    ? new Float32Array(rhsBytes.buffer, rhsBytes.byteOffset, rowCount) : undefined;
  const stageRows = [initialResidualBytes, initialPreconditionedBytes,
    initialPreconditionedImageBytes, preSmoothedBytes, zeroSmoothedBytes,
    firstOperatorImageBytes, firstSmoothedBytes, innerResidualBytes,
    innerCorrectionBytes, postCorrectedBytes].map((bytes) => bytes
    ? new Float32Array(bytes.buffer, bytes.byteOffset, rowCount) : undefined);
  const section63Rows = section63Bytes
    ? new Float32Array(section63Bytes.buffer, section63Bytes.byteOffset, rowCount * 19) : undefined;
  const topologyMetricWords = topologyMetricBytes
    ? new Uint32Array(topologyMetricBytes.buffer, topologyMetricBytes.byteOffset, rowCount * 4) : undefined;
  const boundaryDebug = (solver as GPUSolverInstance & { structuredBoundarySymmetryDebug?: {
    control: GPUBuffer; candidateControl: GPUBuffer; epochState: GPUBuffer;
    structuredControl: GPUBuffer; readyEpochAudit?: GPUBuffer; readyFrontierAudit?: GPUBuffer;
    readyCompactionAudit?: GPUBuffer; candidates: GPUBuffer;
    authority: GPUBuffer; rowGeometry: GPUBuffer;
    rowVelocities: GPUBuffer;
    selectorRows: GPUBuffer; selectorOffsetWords: number; selectorStride: number;
    supportVectorOffsetWords: number; ownerDirectoryOffsetWords: number; supportCapacity: number;
    supportRecordArena: GPUBuffer; supportRecordOffsetWords: number;
    supportFaces: GPUBuffer; supportScratch: GPUBuffer; supportFaceAdjacency: GPUBuffer;
    supportFaceAdjacencyStride: number;
    topologyTransferAudit?: GPUBuffer;
    advectionSymmetryAudit?: GPUBuffer;
    plan: { authorityWords: number; maximumCaseSlots: number; slotCapacity: number; rowCapacity: number; offsets: {
      values: number; ownerRows: number; neighborRows: number; metadata: number;
      areas: number; inverseDistances: number; fractions: number; pressureScales: number;
      normals: number; centroids: number; rowSlotHandles: number; rowSlotSigns: number;
    } };
  } }).structuredBoundarySymmetryDebug;
  const boundaryDebugBytes = typeof process !== "undefined"
    && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1" && boundaryDebug
    ? await Promise.all([
      readBufferBinding(device, { buffer: boundaryDebug.control }, 64),
      readBufferBinding(device, { buffer: boundaryDebug.authority }, boundaryDebug.authority.size),
      readBufferBinding(device, { buffer: boundaryDebug.candidates }, boundaryDebug.candidates.size),
      readBufferBinding(device, { buffer: boundaryDebug.selectorRows,
        offset: boundaryDebug.selectorOffsetWords * 4 }, rowCount * boundaryDebug.selectorStride * 4),
      readBufferBinding(device, { buffer: boundaryDebug.selectorRows,
        offset: boundaryDebug.supportVectorOffsetWords * 4 }, boundaryDebug.supportCapacity * 16),
      readBufferBinding(device, { buffer: boundaryDebug.supportRecordArena,
        offset: boundaryDebug.supportRecordOffsetWords * 4 }, boundaryDebug.supportCapacity * 8 * 4),
      readBufferBinding(device, { buffer: boundaryDebug.supportFaces }, boundaryDebug.supportFaces.size),
      readBufferBinding(device, { buffer: boundaryDebug.supportFaceAdjacency },
        boundaryDebug.supportFaceAdjacency.size),
      readBufferBinding(device, { buffer: boundaryDebug.selectorRows },
        boundaryDebug.plan.slotCapacity * 16),
      readBufferBinding(device, { buffer: boundaryDebug.rowVelocities }, boundaryDebug.rowVelocities.size),
      boundaryDebug.topologyTransferAudit
        ? readBufferBinding(device, { buffer: boundaryDebug.topologyTransferAudit },
          boundaryDebug.topologyTransferAudit.size) : undefined,
      readBufferBinding(device, { buffer: boundaryDebug.supportScratch }, 60 * 4),
      readBufferBinding(device, { buffer: boundaryDebug.selectorRows,
        offset: boundaryDebug.ownerDirectoryOffsetWords * 4 },
      dimensions[0] * dimensions[1] * dimensions[2] * 16),
      readBufferBinding(device, { buffer: boundaryDebug.rowGeometry }, boundaryDebug.rowGeometry.size),
      boundaryDebug.advectionSymmetryAudit
        ? readBufferBinding(device, { buffer: boundaryDebug.advectionSymmetryAudit },
          boundaryDebug.advectionSymmetryAudit.size) : undefined,
      readBufferBinding(device, { buffer: boundaryDebug.candidateControl }, 64),
      readBufferBinding(device, { buffer: boundaryDebug.epochState }, 64),
      readBufferBinding(device, { buffer: boundaryDebug.structuredControl }, 64),
      boundaryDebug.readyEpochAudit
        ? readBufferBinding(device, { buffer: boundaryDebug.readyEpochAudit }, 64) : undefined,
      boundaryDebug.readyFrontierAudit
        ? readBufferBinding(device, { buffer: boundaryDebug.readyFrontierAudit }, 64) : undefined,
      boundaryDebug.readyCompactionAudit
        ? readBufferBinding(device, { buffer: boundaryDebug.readyCompactionAudit }, 64) : undefined,
    ]) : undefined;
  if (typeof process !== "undefined" && process.env.FLUID_HEAD_DIFFERENTIAL === "1"
    && boundaryDebug && boundaryDebugBytes) {
    const controlWords = new Uint32Array(boundaryDebugBytes[0].buffer,
      boundaryDebugBytes[0].byteOffset, boundaryDebugBytes[0].byteLength / 4);
    const candidateControlWords = new Uint32Array(boundaryDebugBytes[15].buffer,
      boundaryDebugBytes[15].byteOffset, boundaryDebugBytes[15].byteLength / 4);
    const epochWords = new Uint32Array(boundaryDebugBytes[16].buffer,
      boundaryDebugBytes[16].byteOffset, boundaryDebugBytes[16].byteLength / 4);
    const structuredWords = new Uint32Array(boundaryDebugBytes[17].buffer,
      boundaryDebugBytes[17].byteOffset, boundaryDebugBytes[17].byteLength / 4);
    const readyEpochWords = boundaryDebugBytes[18]
      ? new Uint32Array(boundaryDebugBytes[18].buffer, boundaryDebugBytes[18].byteOffset,
        boundaryDebugBytes[18].byteLength / 4) : undefined;
    const readyFrontierWords = boundaryDebugBytes[19]
      ? new Uint32Array(boundaryDebugBytes[19].buffer, boundaryDebugBytes[19].byteOffset,
        boundaryDebugBytes[19].byteLength / 4) : undefined;
    const readyCompactionWords = boundaryDebugBytes[20]
      ? new Uint32Array(boundaryDebugBytes[20].buffer, boundaryDebugBytes[20].byteOffset,
        boundaryDebugBytes[20].byteLength / 4) : undefined;
    const authorityFloats = new Float32Array(boundaryDebugBytes[1].buffer,
      boundaryDebugBytes[1].byteOffset, boundaryDebugBytes[1].byteLength / 4);
    const candidateFloats = new Float32Array(boundaryDebugBytes[2].buffer,
      boundaryDebugBytes[2].byteOffset, boundaryDebugBytes[2].byteLength / 4);
    const { plan } = boundaryDebug, slotCount = Math.min(controlWords[3] ?? 0, plan.slotCapacity);
    const base = (controlWords[5] ?? 0) * plan.authorityWords;
    console.log(JSON.stringify({ phase: "head-differential-structured-boundary",
      control: Array.from(controlWords), candidateControl: Array.from(candidateControlWords),
      epochState: Array.from(epochWords), structuredControl: Array.from(structuredWords),
      readyEpochState: readyEpochWords ? Array.from(readyEpochWords) : undefined,
      readyFrontierState: readyFrontierWords ? Array.from(readyFrontierWords) : undefined,
      readyCompactionState: readyCompactionWords ? Array.from(readyCompactionWords) : undefined,
      acceptedApertures: exactWordFingerprint(authorityFloats.subarray(
        base + plan.offsets.fractions, base + plan.offsets.fractions + slotCount)),
      acceptedPressureScales: exactWordFingerprint(authorityFloats.subarray(
        base + plan.offsets.pressureScales, base + plan.offsets.pressureScales + slotCount)),
      candidateApertures: exactWordFingerprint(Array.from({ length: slotCount },
        (_unused, handle) => candidateFloats[2 * handle]!)),
      candidatePressureScales: exactWordFingerprint(Array.from({ length: slotCount },
        (_unused, handle) => candidateFloats[2 * handle + 1]!)) }));
  }
  const [nx, ny, nz] = dimensions;
  const cellCount = nx * ny * nz;
  const pressure = new Float32Array(cellCount);
  pressure.fill(Number.NaN);
  const rhs = new Float32Array(cellCount);
  rhs.fill(Number.NaN);
  const diagonal = new Float32Array(cellCount);
  diagonal.fill(Number.NaN);
  const section63Diagonal = section63Rows ? new Float32Array(cellCount) : undefined;
  section63Diagonal?.fill(Number.NaN);
  const section63CaseId = topologyMetricWords ? new Uint32Array(cellCount) : undefined;
  const stageFields = stageRows.map((values) => values ? new Float32Array(cellCount) : undefined);
  for (const field of stageFields) field?.fill(Number.NaN);
  const topology = new Uint32Array(cellCount);
  const owners = new Int32Array(cellCount);
  owners.fill(-1);
  let coveredCells = 0, overlapCells = 0, invalidRows = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const cell = headers[12 * row] >>> 0;
    const size = headers[12 * row + 3] >>> 0;
    const value = rows[row];
    const origin = [cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny))] as const;
    const valid = size > 0 && cell < cellCount && Number.isFinite(value)
      && origin[0] + size <= nx && origin[1] + size <= ny && origin[2] + size <= nz;
    if (!valid) invalidRows += 1;
    if (size === 0 || cell >= cellCount
      || origin[0] + size > nx || origin[1] + size > ny || origin[2] + size > nz) continue;
    for (let z = origin[2]; z < origin[2] + size; z += 1) {
      for (let y = origin[1]; y < origin[1] + size; y += 1) {
        for (let x = origin[0]; x < origin[0] + size; x += 1) {
          const index = x + nx * (y + ny * z);
          if (owners[index] >= 0) {
            overlapCells += 1;
            pressure[index] = Number.NaN;
            topology[index] = 0;
            continue;
          }
          owners[index] = row;
          coveredCells += 1;
          topology[index] = size;
          if (valid) {
            pressure[index] = value;
            // MGPCG consumes the live dynamics RHS. The header slot is a
            // topology-era cache and can remain symmetric after dynamics has
            // already introduced a discrepancy.
            rhs[index] = rhsRows?.[row] ?? headerFloats[12 * row + 5]!;
            diagonal[index] = headerFloats[12 * row + 4]!;
            if (section63Diagonal) section63Diagonal[index] = section63Rows![19 * row]!;
            if (section63CaseId) section63CaseId[index] = topologyMetricWords![4 * row]!;
            for (let stage = 0; stage < stageFields.length; stage += 1) {
              if (stageFields[stage] && stageRows[stage]) stageFields[stage]![index] = stageRows[stage]![row]!;
            }
          }
        }
      }
    }
  }
  if (typeof process !== "undefined" && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1"
    && boundaryDebug && boundaryDebugBytes) {
    const words = new Uint32Array(boundaryDebugBytes[1].buffer,
      boundaryDebugBytes[1].byteOffset, boundaryDebugBytes[1].byteLength / 4);
    const floats = new Float32Array(boundaryDebugBytes[1].buffer,
      boundaryDebugBytes[1].byteOffset, boundaryDebugBytes[1].byteLength / 4);
    const boundaryCandidates = new Float32Array(boundaryDebugBytes[2].buffer,
      boundaryDebugBytes[2].byteOffset, boundaryDebugBytes[2].byteLength / 4);
    const { plan } = boundaryDebug;
    const base = control.activeBank * plan.authorityWords;
    const handles = new Set<number>();
    for (let row = 0; row < rowCount; row += 1) {
      for (let local = 0; local < plan.maximumCaseSlots; local += 1) {
        const handle = words[base + plan.offsets.rowSlotHandles
          + row * plan.maximumCaseSlots + local] ?? 0xffff_ffff;
        if (handle < plan.slotCapacity) handles.add(handle);
      }
    }
    const quantize = (value: number) => Math.round(value * 1e6);
    const key = (point: readonly number[], normal: readonly number[], area: number) =>
      `${point.map(quantize).join(",")}|${normal.map(quantize).join(",")}|${quantize(area)}`;
    const faces = new Map<string, { handle: number; value: number }>();
    for (const handle of handles) {
      const point = Array.from(floats.subarray(base + plan.offsets.centroids + 4 * handle,
        base + plan.offsets.centroids + 4 * handle + 3));
      const normal = Array.from(floats.subarray(base + plan.offsets.normals + 4 * handle,
        base + plan.offsets.normals + 4 * handle + 3));
      const area = floats[base + plan.offsets.areas + handle]!;
      faces.set(key(point, normal, area), { handle, value: floats[base + plan.offsets.values + handle]! });
    }
    const extent = [nx, ny, nz].map((count) => count * solver.info.cellSize_m);
    const transforms = [
      { name: "reflect-x", point: (v: readonly number[]) => [extent[0]! - v[0]!, v[1]!, v[2]!],
        vector: (v: readonly number[]) => [-v[0]!, v[1]!, v[2]!] },
      { name: "reflect-z", point: (v: readonly number[]) => [v[0]!, v[1]!, extent[2]! - v[2]!],
        vector: (v: readonly number[]) => [v[0]!, v[1]!, -v[2]!] },
      { name: "swap-xz", point: (v: readonly number[]) => [v[2]!, v[1]!, v[0]!],
        vector: (v: readonly number[]) => [v[2]!, v[1]!, v[0]!] },
    ] as const;
    let comparedValues = 0, exactMismatchCount = 0, missingMatches = 0, maximumAbsoluteError = 0;
    let first: Record<string, unknown> | undefined, worst: Record<string, unknown> | undefined;
    const projectionInputs = (handle: number) => {
      const owner = words[base + plan.offsets.ownerRows + handle] ?? 0xffff_ffff;
      const neighbor = words[base + plan.offsets.neighborRows + handle] ?? 0xffff_ffff;
      return {
        owner, neighbor,
        ownerPressure: owner < rowCount ? rows[owner] : undefined,
        neighborPressure: neighbor < rowCount ? rows[neighbor] : 0,
        inverseDistance: floats[base + plan.offsets.inverseDistances + handle],
        pressureScale: floats[base + plan.offsets.pressureScales + handle],
        aperture: floats[base + plan.offsets.fractions + handle],
        // The boundary resolver publishes {aperture, pressureScale} here
        // before commit copies the pair into structured authority. Keeping
        // both values in the symmetry failure identifies whether a mismatch
        // was authored by boundary sampling or introduced by commit/project.
        candidateAperture: boundaryCandidates[2 * handle],
        candidatePressureScale: boundaryCandidates[2 * handle + 1],
      };
    };
    for (const handle of handles) {
      const point = Array.from(floats.subarray(base + plan.offsets.centroids + 4 * handle,
        base + plan.offsets.centroids + 4 * handle + 3));
      const normal = Array.from(floats.subarray(base + plan.offsets.normals + 4 * handle,
        base + plan.offsets.normals + 4 * handle + 3));
      const area = floats[base + plan.offsets.areas + handle]!;
      const sourceValue = floats[base + plan.offsets.values + handle]!;
      for (const transform of transforms) {
        const targetPoint = transform.point(point), targetNormal = transform.vector(normal);
        let target = faces.get(key(targetPoint, targetNormal, area));
        let expectedValue = sourceValue;
        if (!target) {
          target = faces.get(key(targetPoint, targetNormal.map((value) => -value), area));
          expectedValue = -sourceValue;
        }
        if (!target) { missingMatches += 1; continue; }
        comparedValues += 1;
        // Signed zero is the same physical face-normal velocity. Preserve the
        // stricter bit-exact comparison for every non-zero value.
        if (expectedValue === target.value) continue;
        exactMismatchCount += 1;
        const absoluteError = Math.abs(expectedValue - target.value);
        const detail = { transform: transform.name, sourceHandle: handle, targetHandle: target.handle,
          point, targetPoint, normal, targetNormal, sourceValue, expectedValue,
          targetValue: target.value, absoluteError,
          sourceProjectionInputs: projectionInputs(handle),
          targetProjectionInputs: projectionInputs(target.handle) };
        first ??= detail;
        if (!worst || absoluteError > Number(worst.absoluteError)) worst = detail;
        maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
      }
    }
    const sortedHandles = [...handles].sort((left, right) => left - right);
    console.log(JSON.stringify({ phase: "fluid-symmetry-face-values",
      metrics: { activeFaces: handles.size, comparedValues, exactMismatchCount,
        missingMatches, maximumAbsoluteError, first, worst,
        ...(process.env.FLUID_HEAD_DIFFERENTIAL === "1" ? {
          valueFingerprint: exactWordFingerprint(sortedHandles.map((handle) =>
            floats[base + plan.offsets.values + handle]!)),
        } : {}) } }));
    const advectionAuditBytes = boundaryDebugBytes[14];
    if (advectionAuditBytes) {
      const auditWords = new Uint32Array(advectionAuditBytes.buffer,
        advectionAuditBytes.byteOffset, advectionAuditBytes.byteLength / 4);
      const auditFloats = new Float32Array(advectionAuditBytes.buffer,
        advectionAuditBytes.byteOffset, advectionAuditBytes.byteLength / 4);
      const auditBank = auditWords[4]! & 1, debugBank = 1 - auditBank;
      const auditHandleCount = Math.min(auditWords[5] ?? 0, plan.slotCapacity);
      const record = (bank: number) => 32 + bank * 15 * plan.slotCapacity;
      const activeRecord = record(auditBank), debugRecord = record(debugBank);
      const auditFaces = new Map<string, { handle: number; value: number }>();
      for (let handle = 0; handle < auditHandleCount; handle += 1) {
        const normalAt = activeRecord + 7 * plan.slotCapacity + 4 * handle;
        const centroidAt = activeRecord + 11 * plan.slotCapacity + 4 * handle;
        const point = Array.from(auditFloats.subarray(centroidAt, centroidAt + 3));
        const normal = Array.from(auditFloats.subarray(normalAt, normalAt + 3));
        const area = auditFloats[activeRecord + 4 * plan.slotCapacity + handle]!;
        if (!Number.isFinite(area) || !(area > 0) || point.some((value) => !Number.isFinite(value))
          || normal.some((value) => !Number.isFinite(value))) continue;
        auditFaces.set(key(point, normal, area), {
          handle, value: auditFloats[activeRecord + handle]!,
        });
      }
      const vectorAt = (offset: number, handle: number) =>
        Array.from(auditFloats.subarray(debugRecord + offset + 4 * handle,
          debugRecord + offset + 4 * handle + 3));
      const scalarVectorAt = (offset: number, handle: number) => [
        auditFloats[debugRecord + (offset + 0) * plan.slotCapacity + handle]!,
        auditFloats[debugRecord + (offset + 1) * plan.slotCapacity + handle]!,
        auditFloats[debugRecord + (offset + 2) * plan.slotCapacity + handle]!,
      ];
      const compareTriple = (source: readonly number[], target: readonly number[],
        expected: readonly number[]) => ({ source, target, expected,
        exact: source.every((value, component) => expected[component] === target[component]),
        errors: source.map((_value, component) =>
          Math.abs(expected[component]! - target[component]!)) });
      const rowGeometryWords = new Uint32Array(boundaryDebugBytes[13].buffer,
        boundaryDebugBytes[13].byteOffset, boundaryDebugBytes[13].byteLength / 4);
      const rowAtPoint = (sample: readonly number[]) => {
        const q = sample.map((value, axis) => Math.max(0, Math.min(
          dimensions[axis]! - 1, Math.floor(value / solver.info.cellSize_m))));
        return owners[q[0]! + nx * (q[1]! + ny * q[2]!)] ?? -1;
      };
      const rowDetail = (row: number) => row < 0 ? undefined : ({ row,
        geometry: Array.from(rowGeometryWords.subarray(
          4 * (control.activeBank * plan.rowCapacity + row),
          4 * (control.activeBank * plan.rowCapacity + row) + 4)),
        metric: topologyMetricWords
          ? Array.from(topologyMetricWords.subarray(4 * row, 4 * row + 4)) : undefined,
      });
      let auditCompared = 0, auditMismatches = 0, auditMissing = 0, auditMaximum = 0;
      let auditFirst: Record<string, unknown> | undefined;
      let auditWorst: Record<string, unknown> | undefined;
      for (let handle = 0; handle < auditHandleCount; handle += 1) {
        const normalAt = activeRecord + 7 * plan.slotCapacity + 4 * handle;
        const centroidAt = activeRecord + 11 * plan.slotCapacity + 4 * handle;
        const point = Array.from(auditFloats.subarray(centroidAt, centroidAt + 3));
        const normal = Array.from(auditFloats.subarray(normalAt, normalAt + 3));
        const area = auditFloats[activeRecord + 4 * plan.slotCapacity + handle]!;
        const sourceValue = auditFloats[activeRecord + handle]!;
        if (!Number.isFinite(area) || !(area > 0) || point.some((value) => !Number.isFinite(value))
          || normal.some((value) => !Number.isFinite(value))) continue;
        for (const transform of transforms) {
          const targetPoint = transform.point(point), targetNormal = transform.vector(normal);
          let target = auditFaces.get(key(targetPoint, targetNormal, area));
          let expectedValue = sourceValue;
          if (!target) {
            target = auditFaces.get(key(targetPoint, targetNormal.map((value) => -value), area));
            expectedValue = -sourceValue;
          }
          if (!target) { auditMissing += 1; continue; }
          auditCompared += 1;
          if (expectedValue === target.value) continue;
          auditMismatches += 1;
          const absoluteError = Math.abs(expectedValue - target.value);
          const adv = scalarVectorAt(1, handle), targetAdv = scalarVectorAt(1, target.handle);
          const midpoint = scalarVectorAt(4, handle), targetMidpoint = scalarVectorAt(4, target.handle);
          const middle = vectorAt(7 * plan.slotCapacity, handle);
          const targetMiddle = vectorAt(7 * plan.slotCapacity, target.handle);
          const departure = vectorAt(11 * plan.slotCapacity, handle);
          const targetDeparture = vectorAt(11 * plan.slotCapacity, target.handle);
          const sourceMidpointRow = rowAtPoint(midpoint);
          const targetMidpointRow = rowAtPoint(targetMidpoint);
          const sourceDepartureRow = rowAtPoint(departure);
          const targetDepartureRow = rowAtPoint(targetDeparture);
          const detail = { transform: transform.name, sourceHandle: handle,
            targetHandle: target.handle, point, targetPoint, normal, targetNormal,
            sourceValue, expectedValue, targetValue: target.value, absoluteError,
            sourceOwner: auditWords[activeRecord + plan.slotCapacity + handle],
            targetOwner: auditWords[activeRecord + plan.slotCapacity + target.handle],
            sourceNeighbor: auditWords[activeRecord + 2 * plan.slotCapacity + handle],
            targetNeighbor: auditWords[activeRecord + 2 * plan.slotCapacity + target.handle],
            sourceOwnerMetric: topologyMetricWords
              ? Array.from(topologyMetricWords.subarray(4 * auditWords[
                activeRecord + plan.slotCapacity + handle]!, 4 * auditWords[
                activeRecord + plan.slotCapacity + handle]! + 4)) : undefined,
            targetOwnerMetric: topologyMetricWords
              ? Array.from(topologyMetricWords.subarray(4 * auditWords[
                activeRecord + plan.slotCapacity + target.handle]!, 4 * auditWords[
                activeRecord + plan.slotCapacity + target.handle]! + 4)) : undefined,
            faceSample: compareTriple(adv, targetAdv, transform.vector(adv)),
            midpoint: compareTriple(midpoint, targetMidpoint, transform.point(midpoint)),
            midpointSample: compareTriple(middle, targetMiddle, transform.vector(middle)),
            departure: compareTriple(departure, targetDeparture, transform.point(departure)),
            sourceMidpointRow: rowDetail(sourceMidpointRow),
            targetMidpointRow: rowDetail(targetMidpointRow),
            sourceDepartureRow: rowDetail(sourceDepartureRow),
            targetDepartureRow: rowDetail(targetDepartureRow) };
          auditFirst ??= detail;
          if (!auditWorst || absoluteError > Number(auditWorst.absoluteError)) auditWorst = detail;
          auditMaximum = Math.max(auditMaximum, absoluteError);
        }
      }
      console.log(JSON.stringify({ phase: "fluid-symmetry-advection-face-values",
        generation: auditWords[3], activeBank: auditBank,
        metrics: { activeFaces: auditFaces.size, comparedValues: auditCompared,
          exactMismatchCount: auditMismatches, missingMatches: auditMissing,
          maximumAbsoluteError: auditMaximum, first: auditFirst, worst: auditWorst,
          ...(process.env.FLUID_HEAD_DIFFERENTIAL === "1" ? {
            valueFingerprint: exactWordFingerprint(Array.from({ length: auditHandleCount },
              (_unused, handle) => auditFloats[activeRecord + handle]!)),
          } : {}) } }));
    }
    const supportScratchBytes = boundaryDebugBytes[11];
    if (supportScratchBytes) {
      const supportScratch = new Uint32Array(supportScratchBytes.buffer,
        supportScratchBytes.byteOffset, supportScratchBytes.byteLength / 4);
      const supportFaces = new Uint32Array(boundaryDebugBytes[6].buffer,
        boundaryDebugBytes[6].byteOffset, boundaryDebugBytes[6].byteLength / 4);
      const supportFaceValues = new Float32Array(boundaryDebugBytes[6].buffer,
        boundaryDebugBytes[6].byteOffset, boundaryDebugBytes[6].byteLength / 4);
      const supportAdjacency = new Uint32Array(boundaryDebugBytes[7].buffer,
        boundaryDebugBytes[7].byteOffset, boundaryDebugBytes[7].byteLength / 4);
      const faceRows = Math.min((supportScratch[2] ?? 0) + (supportScratch[8] ?? 0),
        Math.floor(supportFaces.length / 48));
      type OrdinaryFace = { item: number; axis: number; center: number[]; extent: number;
        value: number; distanceSquared: number; seed: number };
      const ordinaryFaces: OrdinaryFace[] = [];
      const geometry = (row: number) => {
        const at = (row + 1) * boundaryDebug.supportFaceAdjacencyStride - 4;
        return { origin: Array.from(supportAdjacency.subarray(at, at + 3)),
          extent: supportAdjacency[at + 3] ?? 0 };
      };
      for (let row = 0; row < faceRows; row += 1) {
        const { origin, extent } = geometry(row);
        for (let local = 0; local < 12; local += 1) {
          const item = 12 * row + local, axis = Math.floor(local / 4), quadrant = local % 4;
          if ((supportFaces[4 * item + 3] ?? 0) === 0) continue;
          const center = origin.map((value) => 4 * value + 2 * extent);
          center[axis] = 4 * origin[axis]! + 4 * extent;
          let transverse = 0;
          for (let a = 0; a < 3; a += 1) if (a !== axis) {
            center[a] = 4 * origin[a]!
              + ((quadrant & (1 << transverse)) !== 0 ? 3 : 1) * extent;
            transverse += 1;
          }
          ordinaryFaces.push({ item, axis, center, extent,
            value: supportFaceValues[4 * item]!,
            distanceSquared: supportFaceValues[4 * item + 1]!,
            seed: supportFaces[4 * item + 2]! });
        }
      }
      const ordinaryKey = (axis: number, center: readonly number[], extent: number) =>
        `${axis}|${center.join(",")}|${extent}`;
      const ordinaryByGeometry = new Map(ordinaryFaces.map((face) =>
        [ordinaryKey(face.axis, face.center, face.extent), face] as const));
      const quarterDimensions = [4 * nx, 4 * ny, 4 * nz];
      const ordinaryTransforms = [
        { name: "reflect-x", axis: (axis: number) => axis,
          point: (v: readonly number[]) => [quarterDimensions[0]! - v[0]!, v[1]!, v[2]!],
          sign: (axis: number) => axis === 0 ? -1 : 1 },
        { name: "reflect-z", axis: (axis: number) => axis,
          point: (v: readonly number[]) => [v[0]!, v[1]!, quarterDimensions[2]! - v[2]!],
          sign: (axis: number) => axis === 2 ? -1 : 1 },
        { name: "swap-xz", axis: (axis: number) => axis === 0 ? 2 : axis === 2 ? 0 : 1,
          point: (v: readonly number[]) => [v[2]!, v[1]!, v[0]!], sign: () => 1 },
      ];
      const auditOrdinary = (seedsOnly: boolean) => {
        let compared = 0, mismatches = 0, missing = 0, maximumAbsoluteError = 0;
        let firstMismatch: Record<string, unknown> | undefined;
        for (const source of ordinaryFaces) {
          if (seedsOnly && source.distanceSquared !== 0) continue;
          for (const transform of ordinaryTransforms) {
            const targetAxis = transform.axis(source.axis);
            const targetCenter = transform.point(source.center);
            const target = ordinaryByGeometry.get(ordinaryKey(targetAxis, targetCenter, source.extent));
            if (!target || (seedsOnly && target.distanceSquared !== 0)) { missing += 1; continue; }
            compared += 1;
            const expected = transform.sign(source.axis) * source.value;
            const distanceMatches = source.distanceSquared === target.distanceSquared;
            if (expected === target.value && distanceMatches) continue;
            mismatches += 1;
            const absoluteError = Math.abs(expected - target.value);
            maximumAbsoluteError = Math.max(maximumAbsoluteError, absoluteError);
            firstMismatch ??= { transform: transform.name, source, target,
              expected, targetCenter, distanceMatches, absoluteError };
          }
        }
        return { faces: ordinaryFaces.length, compared, mismatches, missing,
          maximumAbsoluteError, firstMismatch };
      };
      console.log(JSON.stringify({ phase: "fluid-symmetry-air-support-faces",
        control: { directRows: supportScratch[2], supportRows: supportScratch[8],
          faceCount: supportScratch[29], retained: supportScratch[50] },
        seeds: auditOrdinary(true), marched: auditOrdinary(false),
        ...(process.env.FLUID_HEAD_DIFFERENTIAL === "1" ? { fingerprints: {
          values: exactWordFingerprint(ordinaryFaces.map((face) => face.value)),
          distances: exactWordFingerprint(ordinaryFaces.map((face) => face.distanceSquared)),
          seeds: exactWordFingerprint(ordinaryFaces.map((face) => face.seed), true),
        } } : {}) }));
      const supportRecords = new Uint32Array(boundaryDebugBytes[5].buffer,
        boundaryDebugBytes[5].byteOffset, boundaryDebugBytes[5].byteLength / 4);
      const supportVectors = new Float32Array(boundaryDebugBytes[4].buffer,
        boundaryDebugBytes[4].byteOffset, boundaryDebugBytes[4].byteLength / 4);
      const supportCount = Math.min(supportScratch[8] ?? 0, boundaryDebug.supportCapacity);
      const supportByCell = new Map<string, number>();
      for (let support = 0; support < supportCount; support += 1) {
        const at = 8 * support;
        const cell = supportRecords[at]! + nx * (supportRecords[at + 1]!
          + ny * supportRecords[at + 2]!);
        supportByCell.set(`${cell}|${supportRecords[at + 3]}`, support);
      }
      let supportCompared = 0, supportMismatches = 0, supportMissing = 0;
      let supportMaximumAbsoluteError = 0;
      let supportFirstMismatch: Record<string, unknown> | undefined;
      for (let support = 0; support < supportCount; support += 1) {
        const at = 8 * support;
        const origin = Array.from(supportRecords.subarray(at, at + 3));
        const size = supportRecords[at + 3]!;
        const source = Array.from(supportVectors.subarray(4 * support, 4 * support + 4));
        for (const transform of ordinaryTransforms) {
          const transformedMinimum = transform.point(origin.map((value) => 4 * value)
            .map((value, axis) => transform.sign(axis) < 0 ? value + 4 * size : value));
          const targetOrigin = transformedMinimum.map((value) => Math.round(value / 4));
          const targetCell = targetOrigin[0]! + nx * (targetOrigin[1]! + ny * targetOrigin[2]!);
          const targetSupport = supportByCell.get(`${targetCell}|${size}`);
          if (targetSupport === undefined) { supportMissing += 1; continue; }
          supportCompared += 1;
          const target = Array.from(supportVectors.subarray(4 * targetSupport, 4 * targetSupport + 4));
          const expected = transform.name === "reflect-x"
            ? [-source[0]!, source[1]!, source[2]!, source[3]!]
            : transform.name === "reflect-z"
              ? [source[0]!, source[1]!, -source[2]!, source[3]!]
              : [source[2]!, source[1]!, source[0]!, source[3]!];
          const componentErrors = expected.map((value, component) => Math.abs(value - target[component]!));
          if (expected.every((value, component) => value === target[component])) continue;
          supportMismatches += 1;
          const absoluteError = Math.max(...componentErrors);
          supportMaximumAbsoluteError = Math.max(supportMaximumAbsoluteError, absoluteError);
          supportFirstMismatch ??= { transform: transform.name, support, targetSupport,
            origin, targetOrigin, size, source, expected, target, componentErrors };
        }
      }
      console.log(JSON.stringify({ phase: "fluid-symmetry-air-support-vectors",
        metrics: { supportCount, supportCompared, supportMismatches, supportMissing,
          supportMaximumAbsoluteError, supportFirstMismatch } }));
      const ownerDirectoryBytes = boundaryDebugBytes[12];
      if (ownerDirectoryBytes) {
        const directory = new Uint32Array(ownerDirectoryBytes.buffer,
          ownerDirectoryBytes.byteOffset, ownerDirectoryBytes.byteLength / 4);
        const rowVelocities = new Float32Array(boundaryDebugBytes[9].buffer,
          boundaryDebugBytes[9].byteOffset, boundaryDebugBytes[9].byteLength / 4);
        const activeRowBase = control.activeBank * plan.rowCapacity;
        const resolvedDirectoryVector = (cell: number) => {
          const tag = directory[4 * cell] ?? 0xffff_ffff;
          if (tag === 0xffff_ffff) return undefined;
          if ((tag & 0x8000_0000) !== 0) {
            const support = tag & 0x7fff_ffff;
            return support < supportCount
              ? Array.from(supportVectors.subarray(4 * support, 4 * support + 4)) : undefined;
          }
          return tag < rowCount ? Array.from(rowVelocities.subarray(
            4 * (activeRowBase + tag), 4 * (activeRowBase + tag + 1))) : undefined;
        };
        let directoryCompared = 0, directoryMismatches = 0;
        let directoryMaximumAbsoluteError = 0;
        let directoryFirstMismatch: Record<string, unknown> | undefined;
        const discreteTransforms = [
          { name: "reflect-x", point: (q: readonly number[]) => [nx - 1 - q[0]!, q[1]!, q[2]!],
            vector: (v: readonly number[]) => [-v[0]!, v[1]!, v[2]!, v[3]!] },
          { name: "reflect-z", point: (q: readonly number[]) => [q[0]!, q[1]!, nz - 1 - q[2]!],
            vector: (v: readonly number[]) => [v[0]!, v[1]!, -v[2]!, v[3]!] },
          { name: "swap-xz", point: (q: readonly number[]) => [q[2]!, q[1]!, q[0]!],
            vector: (v: readonly number[]) => [v[2]!, v[1]!, v[0]!, v[3]!] },
        ];
        for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
          for (let x = 0; x < nx; x += 1) {
            const cell = x + nx * (y + ny * z), source = resolvedDirectoryVector(cell);
            if (!source) continue;
            for (const transform of discreteTransforms) {
              const targetQ = transform.point([x, y, z]);
              const targetCell = targetQ[0]! + nx * (targetQ[1]! + ny * targetQ[2]!);
              const target = resolvedDirectoryVector(targetCell);
              if (!target) continue;
              directoryCompared += 1;
              const expected = transform.vector(source);
              if (expected.every((value, component) => value === target[component])) continue;
              directoryMismatches += 1;
              const componentErrors = expected.map((value, component) => Math.abs(value - target[component]!));
              const absoluteError = Math.max(...componentErrors);
              directoryMaximumAbsoluteError = Math.max(directoryMaximumAbsoluteError, absoluteError);
              directoryFirstMismatch ??= { transform: transform.name, sourceCell: [x, y, z],
                targetCell: targetQ, sourceDirectory: Array.from(directory.subarray(4 * cell, 4 * cell + 4)),
                targetDirectory: Array.from(directory.subarray(4 * targetCell, 4 * targetCell + 4)),
                source, expected, target, componentErrors };
            }
          }
        }
        console.log(JSON.stringify({ phase: "fluid-symmetry-air-owner-directory",
          metrics: { directoryCompared, directoryMismatches,
            directoryMaximumAbsoluteError, directoryFirstMismatch } }));
      }
    }
    const transferBytes = boundaryDebugBytes[10];
    if (transferBytes) {
      const transferWords = new Uint32Array(transferBytes.buffer,
        transferBytes.byteOffset, transferBytes.byteLength / 4);
      const transferFloats = new Float32Array(transferBytes.buffer,
        transferBytes.byteOffset, transferBytes.byteLength / 4);
      const rowGeometryWords = new Uint32Array(boundaryDebugBytes[13].buffer,
        boundaryDebugBytes[13].byteOffset, boundaryDebugBytes[13].byteLength / 4);
      const acceptedControlOffset = 2 * plan.slotCapacity;
      const candidateControlOffset = acceptedControlOffset + 32;
      const transferDebugOffset = candidateControlOffset + 32;
      const candidateControl = transferWords.subarray(candidateControlOffset,
        candidateControlOffset + 32);
      const candidateBank = (candidateControl[5] ?? 0) & 1;
      const candidateRowCount = Math.min(candidateControl[2] ?? 0, plan.rowCapacity);
      const candidateBase = candidateBank * plan.authorityWords;
      const acceptedGeometryBank = control.activeBank * plan.rowCapacity;
      const acceptedRowForGeometry = (candidateRow: number) => {
        const candidateAt = 4 * (candidateBank * plan.rowCapacity + candidateRow);
        const cell = rowGeometryWords[candidateAt], size = rowGeometryWords[candidateAt + 1];
        for (let row = 0; row < rowCount; row += 1) {
          const acceptedAt = 4 * (acceptedGeometryBank + row);
          if (rowGeometryWords[acceptedAt] === cell && rowGeometryWords[acceptedAt + 1] === size) return row;
        }
        return 0xffff_ffff;
      };
      const candidateHandles = new Set<number>();
      for (let row = 0; row < candidateRowCount; row += 1) {
        for (let local = 0; local < plan.maximumCaseSlots; local += 1) {
          const handle = words[candidateBase + plan.offsets.rowSlotHandles
            + row * plan.maximumCaseSlots + local] ?? 0xffff_ffff;
          if (handle < plan.slotCapacity) candidateHandles.add(handle);
        }
      }
      const candidateFaces = new Map<string, { handle: number; value: number }>();
      for (const handle of candidateHandles) {
        const point = Array.from(floats.subarray(candidateBase + plan.offsets.centroids + 4 * handle,
          candidateBase + plan.offsets.centroids + 4 * handle + 3));
        const normal = Array.from(floats.subarray(candidateBase + plan.offsets.normals + 4 * handle,
          candidateBase + plan.offsets.normals + 4 * handle + 3));
        const area = floats[candidateBase + plan.offsets.areas + handle]!;
        candidateFaces.set(key(point, normal, area), { handle,
          value: transferFloats[candidateBank * plan.slotCapacity + handle]! });
      }
      let candidateComparedValues = 0, candidateExactMismatchCount = 0;
      let candidateMissingMatches = 0, candidateMaximumAbsoluteError = 0;
      let candidateFirst: Record<string, unknown> | undefined;
      let candidateWorst: Record<string, unknown> | undefined;
      for (const handle of candidateHandles) {
        const point = Array.from(floats.subarray(candidateBase + plan.offsets.centroids + 4 * handle,
          candidateBase + plan.offsets.centroids + 4 * handle + 3));
        const normal = Array.from(floats.subarray(candidateBase + plan.offsets.normals + 4 * handle,
          candidateBase + plan.offsets.normals + 4 * handle + 3));
        const area = floats[candidateBase + plan.offsets.areas + handle]!;
        const sourceValue = transferFloats[candidateBank * plan.slotCapacity + handle]!;
        for (const transform of transforms) {
          const targetPoint = transform.point(point), targetNormal = transform.vector(normal);
          let target = candidateFaces.get(key(targetPoint, targetNormal, area));
          let expectedValue = sourceValue;
          if (!target) {
            target = candidateFaces.get(key(targetPoint, targetNormal.map((value) => -value), area));
            expectedValue = -sourceValue;
          }
          if (!target) { candidateMissingMatches += 1; continue; }
          candidateComparedValues += 1;
          if (expectedValue === target.value) continue;
          candidateExactMismatchCount += 1;
          const absoluteError = Math.abs(expectedValue - target.value);
          const detail = { transform: transform.name, sourceHandle: handle, targetHandle: target.handle,
            point, targetPoint, normal, targetNormal, sourceValue, expectedValue,
            targetValue: target.value, absoluteError,
            sourceExtendedVector: Array.from(transferFloats.subarray(
              transferDebugOffset + 4 * handle, transferDebugOffset + 4 * handle + 3)),
            targetExtendedVector: Array.from(transferFloats.subarray(
              transferDebugOffset + 4 * target.handle, transferDebugOffset + 4 * target.handle + 3)),
            sourceDebugFlags: transferWords[transferDebugOffset + 4 * handle + 3],
            targetDebugFlags: transferWords[transferDebugOffset + 4 * target.handle + 3],
            sourceOwner: words[candidateBase + plan.offsets.ownerRows + handle],
            sourceNeighbor: words[candidateBase + plan.offsets.neighborRows + handle],
            targetOwner: words[candidateBase + plan.offsets.ownerRows + target.handle],
            targetNeighbor: words[candidateBase + plan.offsets.neighborRows + target.handle],
            sourceOwnerGeometry: Array.from(rowGeometryWords.subarray(
              4 * (candidateBank * plan.rowCapacity
                + words[candidateBase + plan.offsets.ownerRows + handle]!),
              4 * (candidateBank * plan.rowCapacity
                + words[candidateBase + plan.offsets.ownerRows + handle]!) + 4)),
            targetOwnerGeometry: Array.from(rowGeometryWords.subarray(
              4 * (candidateBank * plan.rowCapacity
                + words[candidateBase + plan.offsets.ownerRows + target.handle]!),
              4 * (candidateBank * plan.rowCapacity
                + words[candidateBase + plan.offsets.ownerRows + target.handle]!) + 4)),
            sourceCarryMarker: floats[candidateBase + plan.offsets.centroids + 4 * handle + 3],
            targetCarryMarker: floats[candidateBase + plan.offsets.centroids + 4 * target.handle + 3],
            sourceMetadata: words[candidateBase + plan.offsets.metadata + handle],
            targetMetadata: words[candidateBase + plan.offsets.metadata + target.handle] };
          Object.assign(detail, {
            sourceOwnerAcceptedIdentity: acceptedRowForGeometry(Number(detail.sourceOwner)),
            targetOwnerAcceptedIdentity: acceptedRowForGeometry(Number(detail.targetOwner)),
          });
          candidateFirst ??= detail;
          if (!candidateWorst || absoluteError > Number(candidateWorst.absoluteError)) candidateWorst = detail;
          candidateMaximumAbsoluteError = Math.max(candidateMaximumAbsoluteError, absoluteError);
        }
      }
      console.log(JSON.stringify({ phase: "fluid-symmetry-topology-transfer-face-values",
        acceptedControl: Array.from(transferWords.subarray(acceptedControlOffset,
          acceptedControlOffset + 16)),
        candidateControl: Array.from(candidateControl.subarray(0, 16)),
        metrics: { candidateBank, candidateRowCount, activeFaces: candidateHandles.size,
          comparedValues: candidateComparedValues, exactMismatchCount: candidateExactMismatchCount,
          missingMatches: candidateMissingMatches, maximumAbsoluteError: candidateMaximumAbsoluteError,
          first: candidateFirst, worst: candidateWorst } }));
    }
  }
  if (typeof process !== "undefined" && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1"
    && section63Rows && topologyMetricWords) {
    outer: for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const sourceIndex = x + nx * (y + ny * z);
      const targetIndex = z + nx * (y + ny * x);
      const sourceRow = owners[sourceIndex]!, targetRow = owners[targetIndex]!;
      if (sourceRow >= 0 && targetRow >= 0
        && !Object.is(section63Diagonal![sourceIndex], section63Diagonal![targetIndex])) {
        const debugSlots = (row: number) => {
          if (!boundaryDebug || !boundaryDebugBytes) return undefined;
          const controlWords = new Uint32Array(boundaryDebugBytes[0].buffer,
            boundaryDebugBytes[0].byteOffset, boundaryDebugBytes[0].byteLength / 4);
          const words = new Uint32Array(boundaryDebugBytes[1].buffer,
            boundaryDebugBytes[1].byteOffset, boundaryDebugBytes[1].byteLength / 4);
          const floats = new Float32Array(boundaryDebugBytes[1].buffer,
            boundaryDebugBytes[1].byteOffset, boundaryDebugBytes[1].byteLength / 4);
          const candidates = new Float32Array(boundaryDebugBytes[2].buffer,
            boundaryDebugBytes[2].byteOffset, boundaryDebugBytes[2].byteLength / 4);
          const { plan } = boundaryDebug, base = (controlWords[5] ?? 0) * plan.authorityWords;
          const result = [];
          for (let local = 0; local < plan.maximumCaseSlots; local += 1) {
            const handle = words[base + plan.offsets.rowSlotHandles
              + row * plan.maximumCaseSlots + local] ?? 0xffff_ffff;
            if (handle === 0xffff_ffff) continue;
            result.push({ local, handle,
              owner: words[base + plan.offsets.ownerRows + handle],
              neighbor: words[base + plan.offsets.neighborRows + handle],
              area: floats[base + plan.offsets.areas + handle],
              inverseDistance: floats[base + plan.offsets.inverseDistances + handle],
              normal: Array.from(floats.subarray(base + plan.offsets.normals + 4 * handle,
                base + plan.offsets.normals + 4 * handle + 3)),
              centroid: Array.from(floats.subarray(base + plan.offsets.centroids + 4 * handle,
                base + plan.offsets.centroids + 4 * handle + 3)),
              aperture: floats[base + plan.offsets.fractions + handle],
              scale: floats[base + plan.offsets.pressureScales + handle] });
          }
          return result;
        };
        console.log(JSON.stringify({ phase: "fluid-symmetry-section63-row",
          transform: "swap-xz", source: [x, y, z], target: [z, y, x],
          structuredBank: control.activeBank,
          boundaryBank: boundaryDebugBytes
            ? new Uint32Array(boundaryDebugBytes[0].buffer, boundaryDebugBytes[0].byteOffset, 6)[5]
            : undefined,
          sourceRow, targetRow,
          sourceMetric: Array.from(topologyMetricWords.subarray(4 * sourceRow, 4 * sourceRow + 4)),
          targetMetric: Array.from(topologyMetricWords.subarray(4 * targetRow, 4 * targetRow + 4)),
          sourceCoefficients: Array.from(section63Rows.subarray(19 * sourceRow, 19 * sourceRow + 19)),
          targetCoefficients: Array.from(section63Rows.subarray(19 * targetRow, 19 * targetRow + 19)),
          sourceSlots: debugSlots(sourceRow), targetSlots: debugSlots(targetRow) }));
        break outer;
      }
    }
  }
  if (typeof process !== "undefined" && process.env.FLUID_SYMMETRY_STAGE_AUDIT === "1"
    && boundaryDebug && boundaryDebugBytes) {
    outerRhs: for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
      const sourceIndex = x + nx * (y + ny * z);
      const targetX = nx - 1 - x;
      const targetIndex = targetX + nx * (y + ny * z);
      if (Object.is(rhs[sourceIndex], rhs[targetIndex])) continue;
      const words = new Uint32Array(boundaryDebugBytes[1].buffer,
        boundaryDebugBytes[1].byteOffset, boundaryDebugBytes[1].byteLength / 4);
      const floats = new Float32Array(boundaryDebugBytes[1].buffer,
        boundaryDebugBytes[1].byteOffset, boundaryDebugBytes[1].byteLength / 4);
      const controlWords = new Uint32Array(boundaryDebugBytes[0].buffer,
        boundaryDebugBytes[0].byteOffset, 6);
      const selectorWords = new Uint32Array(boundaryDebugBytes[3].buffer,
        boundaryDebugBytes[3].byteOffset, boundaryDebugBytes[3].byteLength / 4);
      const selectorFloats = new Float32Array(boundaryDebugBytes[8].buffer,
        boundaryDebugBytes[8].byteOffset, boundaryDebugBytes[8].byteLength / 4);
      const supportVectors = new Float32Array(boundaryDebugBytes[4].buffer,
        boundaryDebugBytes[4].byteOffset, boundaryDebugBytes[4].byteLength / 4);
      const supportRecords = new Uint32Array(boundaryDebugBytes[5].buffer,
        boundaryDebugBytes[5].byteOffset, boundaryDebugBytes[5].byteLength / 4);
      const supportFaces = new Uint32Array(boundaryDebugBytes[6].buffer,
        boundaryDebugBytes[6].byteOffset, boundaryDebugBytes[6].byteLength / 4);
      const supportFaceValues = new Float32Array(boundaryDebugBytes[6].buffer,
        boundaryDebugBytes[6].byteOffset, boundaryDebugBytes[6].byteLength / 4);
      const supportAdjacency = new Uint32Array(boundaryDebugBytes[7].buffer,
        boundaryDebugBytes[7].byteOffset, boundaryDebugBytes[7].byteLength / 4);
      const rowVelocities = new Float32Array(boundaryDebugBytes[9].buffer,
        boundaryDebugBytes[9].byteOffset, boundaryDebugBytes[9].byteLength / 4);
      const { plan } = boundaryDebug;
      const base = control.activeBank * plan.authorityWords;
      const auditBase = (1 - control.activeBank) * plan.authorityWords;
      const slots = (row: number) => {
        const result = [];
        for (let local = 0; local < plan.maximumCaseSlots; local += 1) {
          const at = row * plan.maximumCaseSlots + local;
          const handle = words[base + plan.offsets.rowSlotHandles + at] ?? 0xffff_ffff;
          if (handle === 0xffff_ffff) continue;
          const sign = new Int32Array(words.buffer,
            words.byteOffset + 4 * (base + plan.offsets.rowSlotSigns + at), 1)[0]!;
          const area = floats[base + plan.offsets.areas + handle]!;
          const aperture = floats[base + plan.offsets.fractions + handle]!;
          const value = floats[base + plan.offsets.values + handle]!;
          result.push({ local, handle,
            owner: words[base + plan.offsets.ownerRows + handle],
            neighbor: words[base + plan.offsets.neighborRows + handle],
            metadata: words[base + plan.offsets.metadata + handle],
            sign, area, aperture, value,
            transportAdv: Array.from(selectorFloats.subarray(4 * handle, 4 * handle + 4)),
            normal: Array.from(floats.subarray(base + plan.offsets.normals + 4 * handle,
              base + plan.offsets.normals + 4 * handle + 3)),
            centroid: Array.from(floats.subarray(base + plan.offsets.centroids + 4 * handle,
              base + plan.offsets.centroids + 4 * handle + 3)),
            auditAdv: Array.from(floats.subarray(auditBase + plan.offsets.normals + 4 * handle,
              auditBase + plan.offsets.normals + 4 * handle + 3)),
            auditMiddle: Array.from(floats.subarray(auditBase + plan.offsets.centroids + 4 * handle,
              auditBase + plan.offsets.centroids + 4 * handle + 3)),
            auditTransported: [floats[auditBase + plan.offsets.areas + handle],
              floats[auditBase + plan.offsets.inverseDistances + handle],
              floats[auditBase + plan.offsets.fractions + handle]],
            term: sign * area * aperture * value });
        }
        return result;
      };
      const sourceRow = owners[sourceIndex]!, targetRow = owners[targetIndex]!;
      const selectors = (row: number) => Array.from({ length: boundaryDebug.selectorStride },
        (_, selector) => ({ selector, tag: selectorWords[row * boundaryDebug.selectorStride + selector]! }))
        .filter(({ tag }) => tag !== 0xffff_ffff)
        .map(({ selector, tag }) => ({ selector, tag,
          ...((tag & 0x8000_0000) !== 0
            ? { support: Array.from(supportVectors.subarray(
              4 * (tag & 0x7fff_ffff), 4 * (tag & 0x7fff_ffff) + 4)),
              supportCell: Array.from(supportRecords.subarray(
                8 * (tag & 0x7fff_ffff), 8 * (tag & 0x7fff_ffff) + 6)),
              supportFaceCell: Array.from(supportAdjacency.subarray(
                (rowCount + (tag & 0x7fff_ffff) + 1) * boundaryDebug.supportFaceAdjacencyStride - 4,
                (rowCount + (tag & 0x7fff_ffff) + 1) * boundaryDebug.supportFaceAdjacencyStride)),
              supportNegativeRows: Array.from(supportAdjacency.subarray(
                (rowCount + (tag & 0x7fff_ffff)) * boundaryDebug.supportFaceAdjacencyStride + 31,
                (rowCount + (tag & 0x7fff_ffff)) * boundaryDebug.supportFaceAdjacencyStride + 43)),
              supportNegativeCarriers: Array.from({ length: 12 }, (_, local) => {
                const faceRow = rowCount + (tag & 0x7fff_ffff);
                const negativeRow = supportAdjacency[
                  faceRow * boundaryDebug.supportFaceAdjacencyStride + 31 + local]!;
                const axis = Math.floor(local / 4);
                return { local, negativeRow,
                  candidates: negativeRow === 0xffff_ffff ? [] : Array.from({ length: 4 }, (_, quadrant) => {
                    const item = 12 * negativeRow + 4 * axis + quadrant;
                    return { quadrant, value: supportFaceValues[4 * item],
                      distanceSquared: supportFaceValues[4 * item + 1],
                      seed: supportFaces[4 * item + 2], valid: supportFaces[4 * item + 3] };
                  }) };
              }),
              supportCarriers: Array.from({ length: 12 }, (_, local) => {
                const item = 12 * (rowCount + (tag & 0x7fff_ffff)) + local;
                const seed = supportFaces[4 * item + 2]!;
                const seedRow = Math.floor(seed / 12), seedLocal = seed % 12;
                const geometryAt = (seedRow + 1) * boundaryDebug.supportFaceAdjacencyStride - 4;
                const origin = Array.from(supportAdjacency.subarray(geometryAt, geometryAt + 3));
                const extent = supportAdjacency[geometryAt + 3]!;
                const axis = Math.floor(seedLocal / 4), quadrant = seedLocal % 4;
                const centerQuarter = origin.map((value) => 4 * value + 2 * extent);
                if (Number.isFinite(axis) && axis < 3) centerQuarter[axis] = 4 * origin[axis]! + 4 * extent;
                let transverse = 0;
                for (let a = 0; a < 3; a += 1) if (a !== axis) {
                  centerQuarter[a] = 4 * origin[a]!
                    + ((quadrant & (1 << transverse)) !== 0 ? 3 : 1) * extent;
                  transverse += 1;
                }
                return { local, value: supportFaceValues[4 * item],
                  distanceSquared: supportFaceValues[4 * item + 1], seed,
                  valid: supportFaces[4 * item + 3], seedCenterQuarter: centerQuarter };
              }) }
            : { rowCell: headers[12 * tag], rowSize: headers[12 * tag + 3] }) }));
      console.log(JSON.stringify({ phase: "fluid-symmetry-rhs-row",
        transform: "reflect-x", source: [x, y, z], target: [targetX, y, z],
        sourceRow, targetRow, sourceRhs: rhs[sourceIndex], targetRhs: rhs[targetIndex],
        sourceMetric: topologyMetricWords
          ? Array.from(topologyMetricWords.subarray(4 * sourceRow, 4 * sourceRow + 4)) : undefined,
        targetMetric: topologyMetricWords
          ? Array.from(topologyMetricWords.subarray(4 * targetRow, 4 * targetRow + 4)) : undefined,
        sourceRowVelocity: Array.from(rowVelocities.subarray(
          4 * (control.activeBank * plan.rowCapacity + sourceRow),
          4 * (control.activeBank * plan.rowCapacity + sourceRow + 1))),
        targetRowVelocity: Array.from(rowVelocities.subarray(
          4 * (control.activeBank * plan.rowCapacity + targetRow),
          4 * (control.activeBank * plan.rowCapacity + targetRow + 1))),
        sourceSelectorDetails: selectors(sourceRow).filter(({ selector }) => selector === 36),
        targetSelectorDetails: selectors(targetRow).filter(({ selector }) => selector === 36),
        sourceSlots: slots(sourceRow), targetSlots: slots(targetRow) }));
      break outerRhs;
    }
  }
  return {
    pressure, rhs, diagonal,
    ...(section63Diagonal ? { section63Diagonal } : {}),
    ...(section63CaseId ? { section63CaseId } : {}),
    ...(section63Rows ? { section63CoefficientRows: Float32Array.from(section63Rows) } : {}),
    ...(stageFields[0] ? { initialResidual: stageFields[0] } : {}),
    ...(stageFields[1] ? { initialPreconditioned: stageFields[1] } : {}),
    ...(stageFields[2] ? { initialPreconditionedImage: stageFields[2] } : {}),
    ...(stageFields[3] ? { preconditionerPreSmoothed: stageFields[3] } : {}),
    ...(stageFields[4] ? { preconditionerZeroSmoothed: stageFields[4] } : {}),
    ...(stageFields[5] ? { preconditionerFirstOperatorImage: stageFields[5] } : {}),
    ...(stageFields[6] ? { preconditionerFirstSmoothed: stageFields[6] } : {}),
    ...(stageFields[7] ? { preconditionerInnerResidual: stageFields[7] } : {}),
    ...(stageFields[8] ? { preconditionerInnerCorrection: stageFields[8] } : {}),
    ...(stageFields[9] ? { preconditionerPostCorrected: stageFields[9] } : {}),
    topology, coveredCells, overlapCells, invalidRows, rowCount,
    publicationValid: control.flags === 0 && control.firstError === 0xffff_ffff
      && control.epoch > 0 && coveredCells === cellCount && overlapCells === 0 && invalidRows === 0,
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
    if (typeof process !== "undefined" && process.env.FLUID_COARSE_TRACKER_RECEIPT === "1") {
      const receiptReader = (solver as GPUSolverInstance & {
        readCoarseSurfaceTrackerReceipt?(): Promise<Record<string, unknown> | undefined>;
      }).readCoarseSurfaceTrackerReceipt;
      const receipt = await receiptReader?.call(solver);
      console.log(JSON.stringify({ phase: "coarse-tracker-receipt",
        reader: typeof receiptReader, metrics: receipt ?? null }));
    }
    const [directoryBytes, controlBytes, gradientBytes] = await Promise.all([
      readBufferBinding(device, coarseOnly.directory, coarseOnly.directory.size
        ?? coarseOnly.directory.buffer.size - (coarseOnly.directory.offset ?? 0)),
      readBufferBinding(device, coarseOnly.control, 64),
      coarseOnly.gradients
        ? readBufferBinding(device, coarseOnly.gradients, coarseOnly.rowCapacity * 16)
        : Promise.resolve(undefined),
    ]);
    const directoryWords = new Uint32Array(directoryBytes.buffer,
      directoryBytes.byteOffset, directoryBytes.byteLength / 4);
    if (typeof process !== "undefined" && process.env.FLUID_COARSE_TRACKER_RECEIPT === "1") {
      const volume = nx * ny * nz;
      const capacity = (directoryWords.length - 8) / 8;
      const start = capacity - volume;
      let minimumPhi = Number.POSITIVE_INFINITY, maximumPhi = Number.NEGATIVE_INFINITY;
      let negative = 0, zero = 0, positive = 0, valid = 0;
      if ((directoryWords[1]! & 0x4000_0000) !== 0 && start >= coarseOnly.rowCapacity) {
        for (let cell = 0; cell < volume; cell += 1) {
          const base = 8 + 8 * (start + cell);
          const phi = new Float32Array(directoryWords.buffer,
            directoryWords.byteOffset + 4 * (base + 2), 1)[0]!;
          if (directoryWords[base] !== cell + 1 || directoryWords[base + 1] !== 1
            || (directoryWords[base + 5]! & 9) !== 9 || !Number.isFinite(phi)) continue;
          valid += 1; minimumPhi = Math.min(minimumPhi, phi); maximumPhi = Math.max(maximumPhi, phi);
          if (phi < 0) negative += 1; else if (phi > 0) positive += 1; else zero += 1;
        }
      }
      console.log(JSON.stringify({ phase: "coarse-tracker-lattice", metrics: {
        valid, volume, negative, zero, positive,
        minimumPhi: valid ? minimumPhi : null, maximumPhi: valid ? maximumPhi : null,
      } }));
    }
    if (typeof process !== "undefined" && process.env.FLUID_HEAD_DIFFERENTIAL === "1") {
      const rows = Math.min(directoryWords[2] ?? 0, coarseOnly.rowCapacity);
      const entries = directoryWords.subarray(8, 8 + 8 * rows);
      const identityWords = Array.from({ length: rows }, (_unused, row) => [
        entries[8 * row]!, entries[8 * row + 1]!, entries[8 * row + 5]!,
        entries[8 * row + 6]!,
      ]).flat();
      const phiWords = Array.from({ length: rows }, (_unused, row) => [
        entries[8 * row + 2]!, entries[8 * row + 3]!, entries[8 * row + 4]!,
        entries[8 * row + 7]!,
      ]).flat();
      console.log(JSON.stringify({ phase: "head-differential-power-coarse-phi",
        header: Array.from(directoryWords.slice(0, 8)), rows,
        identities: exactWordFingerprint(identityWords, true),
        phi: exactWordFingerprint(phiWords, true),
        records: exactWordFingerprint(entries, true) }));
    }
    const controlWords = new Uint32Array(controlBytes.buffer,
      controlBytes.byteOffset, controlBytes.byteLength / 4);
    if (directoryWords[0] !== 0x8000_0000) {
      throw new Error(`Coarse-only octree QA publication rejected: directory=${JSON.stringify(
        Array.from(directoryWords.slice(0, 8)))}, control=${JSON.stringify(Array.from(controlWords))}`);
    }
    // Adaptive phi advances its surface generation on the GPU before the
    // asynchronous host receipt is adopted. The renderer directory and phi
    // control are copied after that same submission, so compare their paired
    // GPU clocks instead of a legally one-step-old host observation. Legacy
    // coarse publications continue to use their host-published generation.
    const adaptiveRenderer = directoryWords.length >= 14
      && (directoryWords[13]! & 0x1000_0000) !== 0;
    const expectedGeneration = adaptiveRenderer && controlWords[0] === 0x4150_4849
      ? controlWords[2]! : coarseOnly.generation;
    const reconstructed = reconstructCoarseOnlyOctreeOccupancyField(
      directoryWords,
      expectedGeneration,
      [nx, ny, nz],
      gradientBytes ? new Float32Array(gradientBytes.buffer,
        gradientBytes.byteOffset, gradientBytes.byteLength / 4) : undefined,
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
        minimum: 0, maximum: 1, maximumCell: null, cellSum, wetCells: occupied, mixedCells: solver.info.phiInterfaceCellCount ?? 0,
        excessCells: 0, meanColumnAmount: cellSum / Math.max(1, nx * nz), columnAmountStdDev: 0,
        componentCount: occupied > 0 ? 1 : 0, largestComponent: occupied, interfaceFaceCount: 0,
        enclosedAirComponentCount: 0, enclosedAirCells: 0, centroidCells: null,
      } };
    }
    const sampleWords = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    const [metadataBytes, sampleBytes, worklistBytes, coarseBytes, coarseControlBytes,
      fineRestrictionBytes,
      topologyBytes, transportBytes, redistanceBytes, volumeBytes, mgpcgBytes] = await Promise.all([
      readBufferBinding(device, { buffer: source.metadata }, source.plan.maximumResidentBricks * 16),
      readBufferBinding(device, { buffer: source.samples }, sampleWords * 4),
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
    const decodedSamples = decodePackedFineSamples(new Uint32Array(sampleBytes.buffer,
      sampleBytes.byteOffset, sampleBytes.byteLength / 4));
    const compactSnapshot = {
      plan: source.plan,
      generation: source.generation,
      metadata: new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset, metadataBytes.byteLength / 4),
      flags: decodedSamples.flags,
      phi: decodedSamples.phi,
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
      generation: snapshot.metadata[errorPage * 4 + 2],
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
      const key = snapshot.metadata[page * 4 + 1];
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
    if (id < debug.pageCapacity) activeKeys.push(snapshot.metadata[id * 4 + 1]);
  }
  const activeCountByChangedChebyshevRadius = Array.from({ length: 17 }, (_, radius) =>
    activeKeys.reduce((count, key) => count + (key < logicalCount && distance[key] <= radius ? 1 : 0), 0));
  const pageDistanceHistogram = (pages: Uint32Array) => {
    const histogram = new Array<number>(18).fill(0);
    for (const id of pages) {
      const key = id < debug.pageCapacity ? snapshot.metadata[id * 4 + 1] : 0xffff_ffff;
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
