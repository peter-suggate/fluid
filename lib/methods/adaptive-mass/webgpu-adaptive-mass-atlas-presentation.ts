/**
 * Browser-safe presentation bridge for adaptive-mass sparse brick state.
 *
 * The simulation remains sparse. This owner materializes only the presentation
 * lattice expected by the existing raster-water and adaptive-grid overlays:
 *
 * - raw density in `densityTexture`;
 * - signed level set, or a density-derived signed proxy, in `levelSetTexture`;
 * - represented-cell ownership in `gridCellTexture`.
 *
 * There is deliberately no GPU trace construction, physics, pipeline compile,
 * readback, or atlas policy here. A CPU milestone solver may build a dense
 * materialization from its sparse 4/8 brick atlas and upload it; a later GPU
 * materializer can publish the exact same shape without changing consumers.
 */

export type AdaptiveMassAtlasDimensions = readonly [number, number, number];
export type AdaptiveMassAtlasCoordinate = readonly [number, number, number];
export type AdaptiveMassPresentationBrickResolution = 1 | 2 | 4 | 8;

/** Dense, x-major presentation data accepted by the WebGPU texture owner. */
export interface AdaptiveMassAtlasMaterialization {
  readonly dimensions: AdaptiveMassAtlasDimensions;
  /** `x + nx * (y + ny * z)`, one f32 density per finest presentation cell. */
  readonly density: ArrayLike<number>;
  /** Same indexing; signed metres for the shipping water level-set path. */
  readonly levelSetOrProxy: ArrayLike<number>;
  /** Two packed u32 words per cell, matching `adaptiveCellKey` in the grid overlay. */
  readonly ownerKeys: ArrayLike<number>;
  /** Optional collocated xyz velocity plus padding, four f32 values per cell. */
  readonly velocity?: ArrayLike<number>;
  /** Optional materialized composite pressure, one f32 value per cell. */
  readonly pressure?: ArrayLike<number>;
  /** Optional post-projection divergence, one f32 value per cell. */
  readonly divergence?: ArrayLike<number>;
}

/** One sparse source brick before presentation-only finest-lattice expansion. */
export interface AdaptiveMassPresentationBrick {
  /** Origin in the materialization's finest presentation cells. */
  readonly originFine: AdaptiveMassAtlasCoordinate;
  readonly resolution: AdaptiveMassPresentationBrickResolution;
  /**
   * World extent expressed in finest presentation cells. Defaults to eight,
   * so an 8^3 brick has scale one and an equal-world-size 4^3 brick scale two.
   */
  readonly fineSpan?: number;
  /** X-major leaf values, exactly `resolution^3` entries. */
  readonly density: ArrayLike<number>;
  /** Optional signed value per leaf cell. Missing values derive a density proxy. */
  readonly levelSet?: ArrayLike<number>;
}

export interface AdaptiveMassAtlasMaterializationOptions {
  readonly dimensions: AdaptiveMassAtlasDimensions;
  readonly bricks: readonly AdaptiveMassPresentationBrick[];
  /** Positive signed value used outside every resident brick. */
  readonly emptyLevelSet: number;
  /**
   * Width used for `phi = (0.5 - density) * densityProxyBand` when a brick has
   * no level set. Passing four finest-cell widths makes the water extractor's
   * occupancy conversion reproduce density through the 0.5 contour.
   */
  readonly densityProxyBand: number;
}

interface PresentationLeafSample {
  lower: [number, number, number];
  center: [number, number, number];
  scale: number;
  phi: number;
}

interface DirectionalSample {
  value: number;
  distance: number;
}

export interface AdaptiveMassAtlasMaterializationWorkspace {
  density: Float32Array;
  levelSetOrProxy: Float32Array;
  ownerKeys: Uint32Array;
  occupied: Uint8Array;
  sourceLeafAtFineCell: Int32Array;
  readonly leaves: PresentationLeafSample[];
  readonly leafPool: PresentationLeafSample[];
  readonly gradient: [number, number, number];
  readonly negativeSample: DirectionalSample;
  readonly positiveSample: DirectionalSample;
  readonly owner: [number, number];
  result?: AdaptiveMassAtlasMaterialization;
}

export function createAdaptiveMassAtlasMaterializationWorkspace():
AdaptiveMassAtlasMaterializationWorkspace {
  return {
    density: new Float32Array(0), levelSetOrProxy: new Float32Array(0),
    ownerKeys: new Uint32Array(0), occupied: new Uint8Array(0),
    sourceLeafAtFineCell: new Int32Array(0), leaves: [], leafPool: [],
    gradient: [0, 0, 0], negativeSample: { value: 0, distance: 0 },
    positiveSample: { value: 0, distance: 0 }, owner: [0, 0],
  };
}

export interface AdaptiveMassAtlasUploadReceipt {
  readonly generation: number;
  readonly cellCount: number;
  readonly uploadedBytes: number;
}

const OWNER_COMPONENT_MASK = 0x3ff;
const OWNER_MAXIMUM_COORDINATE = OWNER_COMPONENT_MASK;
const WIDE_OWNER_COORDINATE_MASK = 0x7ff;
const WIDE_OWNER_TAG = 0x8000_0000;
const WIDE_OWNER_MAXIMUM_DIMENSION = WIDE_OWNER_COORDINATE_MASK + 1;

function assertDimensions(
  dimensions: AdaptiveMassAtlasDimensions,
  maximumTextureDimension3D = WIDE_OWNER_MAXIMUM_DIMENSION,
): void {
  for (let axis = 0; axis < 3; axis += 1) {
    const value = dimensions[axis];
    if (!Number.isSafeInteger(value) || value < 1
      || value > Math.min(maximumTextureDimension3D, WIDE_OWNER_MAXIMUM_DIMENSION)) {
      throw new RangeError(
        `adaptive-mass presentation dimension ${axis} must be in [1, `
          + `${Math.min(maximumTextureDimension3D, WIDE_OWNER_MAXIMUM_DIMENSION)}]; received ${value}`,
      );
    }
  }
}

function cellCount(dimensions: AdaptiveMassAtlasDimensions): number {
  return dimensions[0] * dimensions[1] * dimensions[2];
}

function linearIndex(
  dimensions: AdaptiveMassAtlasDimensions,
  x: number,
  y: number,
  z: number,
): number {
  return x + dimensions[0] * (y + dimensions[1] * z);
}

/**
 * Area-average the composite neighbours touching one leaf face. Fine neighbours
 * contribute one sample per fine subface, while a coarse neighbour contributes
 * the same value over its covered subfaces. This is the presentation analogue
 * of the conservative 2:1 face quadrature used by the solver.
 */
function faceNeighborSample(
  dimensions: AdaptiveMassAtlasDimensions,
  sourceLeafAtFineCell: Int32Array,
  leaves: readonly PresentationLeafSample[],
  leaf: PresentationLeafSample,
  axis: 0 | 1 | 2,
  side: -1 | 1,
  emptyLevelSet: number,
  output: DirectionalSample,
): boolean {
  const faceCoordinate = side < 0 ? leaf.lower[axis] - 1 : leaf.lower[axis] + leaf.scale;
  if (faceCoordinate < 0 || faceCoordinate >= dimensions[axis]) return false;
  const tangentA = axis === 0 ? 1 : 0;
  const tangentB = axis === 2 ? 1 : 2;
  let value = 0;
  let distance = 0;
  let samples = 0;
  for (let b = 0; b < leaf.scale; b += 1) {
    for (let a = 0; a < leaf.scale; a += 1) {
      let x = leaf.lower[0], y = leaf.lower[1], z = leaf.lower[2];
      if (axis === 0) x = faceCoordinate;
      else if (axis === 1) y = faceCoordinate;
      else z = faceCoordinate;
      if (tangentA === 0) x += a;
      else if (tangentA === 1) y += a;
      else z += a;
      if (tangentB === 1) y += b;
      else z += b;
      const neighborIndex = sourceLeafAtFineCell[linearIndex(dimensions, x, y, z)];
      if (neighborIndex < 0) {
        value += emptyLevelSet;
        distance += Math.abs(faceCoordinate + 0.5 - leaf.center[axis]);
      } else {
        const neighbor = leaves[neighborIndex];
        value += neighbor.phi;
        distance += Math.abs(neighbor.center[axis] - leaf.center[axis]);
      }
      samples += 1;
    }
  }
  output.value = value / samples;
  output.distance = distance / samples;
  return true;
}

/**
 * Prolong a coarse cell-centred scalar to the shared finest presentation
 * lattice. The centred linear reconstruction has exactly the source value as
 * the mean of its children, so density-derived phi retains every coarse cell's
 * integrated CM12 mass. All marching-cubes cubes subsequently load the same
 * fine-lattice samples, including cubes straddling a 4^3 <-> 8^3 brick seam.
 */
function reconstructCoarseLeafPhi(
  dimensions: AdaptiveMassAtlasDimensions,
  sourceLeafAtFineCell: Int32Array,
  leaves: readonly PresentationLeafSample[],
  leafIndex: number,
  emptyLevelSet: number,
  output: [number, number, number],
  negativeSample: DirectionalSample,
  positiveSample: DirectionalSample,
): readonly [number, number, number] {
  const leaf = leaves[leafIndex];
  output[0] = 0;
  output[1] = 0;
  output[2] = 0;
  if (leaf.scale === 1) return output;
  const gradient = output;
  let maximumNeighborDelta = 0;
  for (let axisIndex = 0; axisIndex < 3; axisIndex += 1) {
    const axis = axisIndex as 0 | 1 | 2;
    const hasNegative = faceNeighborSample(
      dimensions, sourceLeafAtFineCell, leaves, leaf, axis, -1, emptyLevelSet,
      negativeSample,
    );
    const hasPositive = faceNeighborSample(
      dimensions, sourceLeafAtFineCell, leaves, leaf, axis, 1, emptyLevelSet,
      positiveSample,
    );
    if (hasNegative) maximumNeighborDelta = Math.max(
      maximumNeighborDelta, Math.abs(negativeSample.value - leaf.phi),
    );
    if (hasPositive) maximumNeighborDelta = Math.max(
      maximumNeighborDelta, Math.abs(positiveSample.value - leaf.phi),
    );
    if (hasNegative && hasPositive) {
      gradient[axis] = (positiveSample.value - negativeSample.value)
        / Math.max(Number.EPSILON, positiveSample.distance + negativeSample.distance);
    } else if (hasPositive) {
      gradient[axis] = (positiveSample.value - leaf.phi)
        / Math.max(Number.EPSILON, positiveSample.distance);
    } else if (hasNegative) {
      gradient[axis] = (leaf.phi - negativeSample.value)
        / Math.max(Number.EPSILON, negativeSample.distance);
    }
  }

  // A noisy adaptive neighbourhood must not extrapolate further than the
  // largest value change already present on a touching face. One scalar theta
  // preserves the direction and the zero-mean child offsets exactly.
  const maximumOffset = 0.5 * (leaf.scale - 1);
  const predictedMaximumDelta = maximumOffset
    * (Math.abs(gradient[0]) + Math.abs(gradient[1]) + Math.abs(gradient[2]));
  if (predictedMaximumDelta > maximumNeighborDelta && predictedMaximumDelta > 0) {
    const theta = maximumNeighborDelta / predictedMaximumDelta;
    gradient[0] *= theta;
    gradient[1] *= theta;
    gradient[2] *= theta;
  }
  return gradient;
}

/**
 * Pack the two-word adaptive ownership key consumed by `webgpu-grid-overlay`.
 * Coordinates are in the dense finest presentation lattice. The existing ABI
 * gives x/z one cubic `cellScale` and y an explicit upper bound.
 */
export function packAdaptiveMassPresentationOwnerKey(
  lower: AdaptiveMassAtlasCoordinate,
  cellScale: number,
  output: [number, number] = [0, 0],
): readonly [number, number] {
  if (!Number.isSafeInteger(cellScale) || cellScale < 1
    || cellScale > WIDE_OWNER_MAXIMUM_DIMENSION) {
    throw new RangeError(`adaptive owner cell scale must be in [1, 2048]; received ${cellScale}`);
  }
  for (let axis = 0; axis < 3; axis += 1) {
    if (!Number.isSafeInteger(lower[axis]) || lower[axis] < 0
      || lower[axis] > WIDE_OWNER_COORDINATE_MASK) {
      throw new RangeError(`adaptive owner lower coordinate ${axis} is invalid: ${lower[axis]}`);
    }
  }
  const upperY = lower[1] + cellScale;
  if (upperY > WIDE_OWNER_MAXIMUM_DIMENSION) {
    throw new RangeError(`adaptive owner upper-y coordinate exceeds 2048: ${upperY}`);
  }
  if (lower.every((value) => value <= OWNER_MAXIMUM_COORDINATE)
    && upperY <= OWNER_MAXIMUM_COORDINATE && cellScale <= OWNER_COMPONENT_MASK) {
    const first = (lower[0] | (lower[2] << 10) | (cellScale << 20)) >>> 0;
    const second = (lower[1] | (upperY << 10)) >>> 0;
    output[0] = first;
    output[1] = second;
    return output;
  }
  if ((cellScale & (cellScale - 1)) !== 0) {
    throw new RangeError(`wide adaptive owner cell scale must be a power of two; received ${cellScale}`);
  }
  const exponent = Math.log2(cellScale);
  const first = (lower[0] | (lower[2] << 11) | (exponent << 22)) >>> 0;
  const second = (WIDE_OWNER_TAG | lower[1]) >>> 0;
  output[0] = first;
  output[1] = second;
  return output;
}

/**
 * Expand sparse 4/8 bricks into the consumer-neutral dense upload shape.
 *
 * Empty space receives one shared background owner key. This keeps the
 * adaptive structure overlay empty outside resident bricks instead of drawing
 * the hidden finest backing lattice; transitions at resident brick boundaries
 * remain visible because their owner key differs from the background.
 */
export function materializeAdaptiveMassPresentationAtlas(
  options: AdaptiveMassAtlasMaterializationOptions,
  workspace: AdaptiveMassAtlasMaterializationWorkspace =
    createAdaptiveMassAtlasMaterializationWorkspace(),
): AdaptiveMassAtlasMaterialization {
  assertDimensions(options.dimensions);
  if (!Number.isFinite(options.emptyLevelSet) || options.emptyLevelSet <= 0) {
    throw new RangeError(`emptyLevelSet must be finite and positive; received ${options.emptyLevelSet}`);
  }
  if (!Number.isFinite(options.densityProxyBand) || options.densityProxyBand <= 0) {
    throw new RangeError(`densityProxyBand must be finite and positive; received ${options.densityProxyBand}`);
  }
  const count = cellCount(options.dimensions);
  const density = workspace.density = workspace.density.length === count
    ? workspace.density : new Float32Array(count);
  density.fill(0);
  const levelSetOrProxy = workspace.levelSetOrProxy =
    workspace.levelSetOrProxy.length === count
      ? workspace.levelSetOrProxy : new Float32Array(count);
  levelSetOrProxy.fill(Math.fround(options.emptyLevelSet));
  const ownerKeys = workspace.ownerKeys = workspace.ownerKeys.length === count * 2
    ? workspace.ownerKeys : new Uint32Array(count * 2);
  const maximumHorizontalSpan = Math.max(options.dimensions[0], options.dimensions[2]);
  const backgroundScale = Math.min(
    WIDE_OWNER_MAXIMUM_DIMENSION,
    2 ** Math.ceil(Math.log2(Math.max(1, maximumHorizontalSpan))),
  );
  const background = packAdaptiveMassPresentationOwnerKey([0, 0, 0],
    backgroundScale, workspace.owner);
  for (let index = 0; index < count; index += 1) {
    ownerKeys[2 * index] = background[0];
    ownerKeys[2 * index + 1] = background[1];
  }
  const occupied = workspace.occupied = workspace.occupied.length === count
    ? workspace.occupied : new Uint8Array(count);
  occupied.fill(0);
  const sourceLeafAtFineCell = workspace.sourceLeafAtFineCell =
    workspace.sourceLeafAtFineCell.length === count
      ? workspace.sourceLeafAtFineCell : new Int32Array(count);
  sourceLeafAtFineCell.fill(-1);
  const leaves = workspace.leaves;
  let leafCount = 0;

  for (let brickIndex = 0; brickIndex < options.bricks.length; brickIndex += 1) {
    const brick = options.bricks[brickIndex];
    const fineSpan = brick.fineSpan ?? 8;
    if (!Number.isSafeInteger(fineSpan) || fineSpan < brick.resolution
      || fineSpan % brick.resolution !== 0) {
      throw new RangeError(
        `brick ${brickIndex} fineSpan must be a positive multiple of resolution ${brick.resolution}`,
      );
    }
    const cellScale = fineSpan / brick.resolution;
    const valueCount = brick.resolution ** 3;
    if (brick.density.length !== valueCount
      || (brick.levelSet !== undefined && brick.levelSet.length !== valueCount)) {
      throw new RangeError(
        `brick ${brickIndex} requires ${valueCount} density/level-set values`,
      );
    }
    for (let axis = 0; axis < 3; axis += 1) {
      const origin = brick.originFine[axis];
      if (!Number.isSafeInteger(origin) || origin < 0
        || origin + fineSpan > options.dimensions[axis]) {
        throw new RangeError(
          `brick ${brickIndex} axis ${axis} extent [${origin}, ${origin + fineSpan}) `
            + `is outside dimension ${options.dimensions[axis]}`,
        );
      }
    }

    for (let localZ = 0; localZ < brick.resolution; localZ += 1) {
      for (let localY = 0; localY < brick.resolution; localY += 1) {
        for (let localX = 0; localX < brick.resolution; localX += 1) {
          const source = localX + brick.resolution
            * (localY + brick.resolution * localZ);
          const rho = Number(brick.density[source]);
          const phi = brick.levelSet === undefined
            ? (0.5 - rho) * options.densityProxyBand
            : Number(brick.levelSet[source]);
          if (!Number.isFinite(rho) || !Number.isFinite(phi)) {
            throw new RangeError(`brick ${brickIndex} cell ${source} has non-finite presentation data`);
          }
          const leafIndex = leafCount++;
          let leaf = workspace.leafPool[leafIndex];
          if (!leaf) {
            leaf = workspace.leafPool[leafIndex] = {
              lower: [0, 0, 0], center: [0, 0, 0], scale: 1, phi: 0,
            };
          }
          const lower = leaf.lower;
          lower[0] = brick.originFine[0] + localX * cellScale;
          lower[1] = brick.originFine[1] + localY * cellScale;
          lower[2] = brick.originFine[2] + localZ * cellScale;
          leaf.center[0] = lower[0] + 0.5 * cellScale;
          leaf.center[1] = lower[1] + 0.5 * cellScale;
          leaf.center[2] = lower[2] + 0.5 * cellScale;
          leaf.scale = cellScale;
          leaf.phi = phi;
          leaves[leafIndex] = leaf;
          const owner = packAdaptiveMassPresentationOwnerKey(
            lower, cellScale, workspace.owner,
          );
          for (let childZ = 0; childZ < cellScale; childZ += 1) {
            for (let childY = 0; childY < cellScale; childY += 1) {
              for (let childX = 0; childX < cellScale; childX += 1) {
                const destination = linearIndex(options.dimensions,
                  lower[0] + childX, lower[1] + childY, lower[2] + childZ);
                if (occupied[destination] !== 0) {
                  throw new Error(`adaptive presentation bricks overlap at dense cell ${destination}`);
                }
                occupied[destination] = 1;
                sourceLeafAtFineCell[destination] = leafIndex;
                density[destination] = Math.fround(rho);
                ownerKeys[2 * destination] = owner[0];
                ownerKeys[2 * destination + 1] = owner[1];
              }
            }
          }
        }
      }
    }
  }
  leaves.length = leafCount;

  // The physics texture remains the exact piecewise-constant leaf density.
  // Only the renderer's phi proxy is reconstructed: it cannot feed transport,
  // pressure, topology, or any mass receipt owned by the sparse solver.
  for (let leafIndex = 0; leafIndex < leaves.length; leafIndex += 1) {
    const leaf = leaves[leafIndex];
    const gradient = reconstructCoarseLeafPhi(
      options.dimensions, sourceLeafAtFineCell, leaves, leafIndex, options.emptyLevelSet,
      workspace.gradient, workspace.negativeSample, workspace.positiveSample,
    );
    for (let childZ = 0; childZ < leaf.scale; childZ += 1) {
      for (let childY = 0; childY < leaf.scale; childY += 1) {
        for (let childX = 0; childX < leaf.scale; childX += 1) {
          const destination = linearIndex(options.dimensions,
            leaf.lower[0] + childX, leaf.lower[1] + childY, leaf.lower[2] + childZ);
          const offsetX = childX + 0.5 - 0.5 * leaf.scale;
          const offsetY = childY + 0.5 - 0.5 * leaf.scale;
          const offsetZ = childZ + 0.5 - 0.5 * leaf.scale;
          levelSetOrProxy[destination] = Math.fround(leaf.phi
            + gradient[0] * offsetX + gradient[1] * offsetY + gradient[2] * offsetZ);
        }
      }
    }
  }
  if (!workspace.result) {
    workspace.result = { dimensions: options.dimensions, density, levelSetOrProxy, ownerKeys };
  } else {
    const result = workspace.result as {
      dimensions: AdaptiveMassAtlasDimensions;
      density: ArrayLike<number>;
      levelSetOrProxy: ArrayLike<number>;
      ownerKeys: ArrayLike<number>;
    };
    result.dimensions = options.dimensions;
    result.density = density;
    result.levelSetOrProxy = levelSetOrProxy;
    result.ownerKeys = ownerKeys;
  }
  return workspace.result;
}

function paddedTextureUploadInto(
  source: Uint8Array,
  rowBytes: number,
  rowsPerImage: number,
  imageCount: number,
  bytes: Uint8Array,
): number {
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  for (let image = 0; image < imageCount; image += 1) {
    for (let row = 0; row < rowsPerImage; row += 1) {
      const sourceRow = row + rowsPerImage * image;
      bytes.set(source.subarray(sourceRow * rowBytes, (sourceRow + 1) * rowBytes),
        (row + rowsPerImage * image) * bytesPerRow);
    }
  }
  return bytesPerRow;
}

/** Texture owner consumed directly by `GPUSolverInstance` publications. */
export class WebGPUAdaptiveMassAtlasPresentation {
  readonly densityTexture: GPUTexture;
  readonly levelSetTexture: GPUTexture;
  readonly gridCellTexture: GPUTexture;
  readonly velocityTexture: GPUTexture;
  readonly pressureTexture: GPUTexture;
  readonly divergenceTexture: GPUTexture;
  readonly allocatedBytes: number;
  generation = 0;
  private destroyed = false;
  private readonly paddedUpload: Uint8Array;
  private readonly byteViews = new WeakMap<ArrayBuffer, Uint8Array>();
  private convertedDensity: Float32Array;
  private convertedLevelSet: Float32Array;
  private convertedOwners: Uint32Array;
  private convertedVelocity: Float32Array;
  private convertedPressure: Float32Array;
  private convertedDivergence: Float32Array;
  private readonly uploadDestination: { texture: GPUTexture };
  private readonly uploadLayout: {
    offset: number;
    bytesPerRow: number;
    rowsPerImage: number;
  };
  private readonly uploadSize: {
    width: number;
    height: number;
    depthOrArrayLayers: number;
  };
  private readonly uploadReceipt: AdaptiveMassAtlasUploadReceipt;

  constructor(
    private readonly device: GPUDevice,
    readonly dimensions: AdaptiveMassAtlasDimensions,
  ) {
    assertDimensions(dimensions, device.limits.maxTextureDimension3D);
    const scalarUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING;
    this.densityTexture = device.createTexture({
      label: "Adaptive mass presentation density",
      size: dimensions,
      dimension: "3d",
      format: "r32float",
      usage: scalarUsage,
    });
    this.levelSetTexture = device.createTexture({
      label: "Adaptive mass presentation level-set proxy",
      size: dimensions,
      dimension: "3d",
      format: "r32float",
      usage: scalarUsage,
    });
    this.gridCellTexture = device.createTexture({
      label: "Adaptive mass presentation cell ownership",
      size: dimensions,
      dimension: "3d",
      format: "rg32uint",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
        | GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
    });
    this.velocityTexture = device.createTexture({
      label: "Adaptive mass presentation velocity",
      size: dimensions,
      dimension: "3d",
      format: "rgba32float",
      usage: scalarUsage,
    });
    this.pressureTexture = device.createTexture({
      label: "Adaptive mass presentation pressure",
      size: dimensions,
      dimension: "3d",
      format: "r32float",
      usage: scalarUsage,
    });
    this.divergenceTexture = device.createTexture({
      label: "Adaptive mass presentation divergence",
      size: dimensions,
      dimension: "3d",
      format: "r32float",
      usage: scalarUsage,
    });
    this.allocatedBytes = cellCount(dimensions) * 40;
    const count = cellCount(dimensions);
    const maximumBytesPerRow = Math.ceil(dimensions[0] * 16 / 256) * 256;
    this.paddedUpload = new Uint8Array(maximumBytesPerRow * dimensions[1] * dimensions[2]);
    this.convertedDensity = new Float32Array(count);
    this.convertedLevelSet = new Float32Array(count);
    this.convertedOwners = new Uint32Array(2 * count);
    this.convertedVelocity = new Float32Array(4 * count);
    this.convertedPressure = new Float32Array(count);
    this.convertedDivergence = new Float32Array(count);
    this.uploadDestination = { texture: this.densityTexture };
    this.uploadLayout = { offset: 0, bytesPerRow: 0, rowsPerImage: dimensions[1] };
    this.uploadSize = {
      width: dimensions[0], height: dimensions[1], depthOrArrayLayers: dimensions[2],
    };
    this.uploadReceipt = { generation: 0, cellCount: count, uploadedBytes: 0 };
  }

  /** Alias matching the optional `GPUSolverInstance.surfaceFieldTexture`. */
  get surfaceFieldTexture(): GPUTexture { return this.levelSetTexture; }

  /** Upload one complete CPU-authored presentation generation. */
  upload(materialization: AdaptiveMassAtlasMaterialization): AdaptiveMassAtlasUploadReceipt {
    this.assertLive();
    if (materialization.dimensions[0] !== this.dimensions[0]
      || materialization.dimensions[1] !== this.dimensions[1]
      || materialization.dimensions[2] !== this.dimensions[2]) {
      throw new RangeError(
        `adaptive presentation upload dimensions ${materialization.dimensions.join("x")} `
          + `do not match ${this.dimensions.join("x")}`,
      );
    }
    const count = cellCount(this.dimensions);
    if (materialization.density.length !== count
      || materialization.levelSetOrProxy.length !== count
      || materialization.ownerKeys.length !== 2 * count
      || (materialization.velocity !== undefined && materialization.velocity.length !== 4 * count)
      || (materialization.pressure !== undefined && materialization.pressure.length !== count)
      || (materialization.divergence !== undefined && materialization.divergence.length !== count)) {
      throw new RangeError(
        `adaptive presentation upload dimensions do not match ${count} cells`,
      );
    }
    const density = materialization.density instanceof Float32Array
      ? materialization.density : (this.convertedDensity.set(materialization.density),
        this.convertedDensity);
    const levelSet = materialization.levelSetOrProxy instanceof Float32Array
      ? materialization.levelSetOrProxy
      : (this.convertedLevelSet.set(materialization.levelSetOrProxy), this.convertedLevelSet);
    const owners = materialization.ownerKeys instanceof Uint32Array
      ? materialization.ownerKeys
      : (this.convertedOwners.set(materialization.ownerKeys), this.convertedOwners);
    let velocity = this.convertedVelocity;
    if (materialization.velocity === undefined) velocity.fill(0);
    else if (materialization.velocity instanceof Float32Array) velocity = materialization.velocity;
    else velocity.set(materialization.velocity);
    let pressure = this.convertedPressure;
    if (materialization.pressure === undefined) pressure.fill(0);
    else if (materialization.pressure instanceof Float32Array) pressure = materialization.pressure;
    else pressure.set(materialization.pressure);
    let divergence = this.convertedDivergence;
    if (materialization.divergence === undefined) divergence.fill(0);
    else if (materialization.divergence instanceof Float32Array) divergence = materialization.divergence;
    else divergence.set(materialization.divergence);
    for (let index = 0; index < count; index += 1) {
      if (!Number.isFinite(density[index]) || !Number.isFinite(levelSet[index])
        || !Number.isFinite(pressure[index]) || !Number.isFinite(divergence[index])
        || !Number.isFinite(velocity[4 * index]) || !Number.isFinite(velocity[4 * index + 1])
        || !Number.isFinite(velocity[4 * index + 2])) {
        throw new RangeError(`adaptive presentation upload cell ${index} is non-finite`);
      }
    }
    let uploadedBytes = this.uploadTexture(this.densityTexture, density, 1);
    uploadedBytes += this.uploadTexture(this.levelSetTexture, levelSet, 1);
    uploadedBytes += this.uploadTexture(this.gridCellTexture, owners, 2);
    uploadedBytes += this.uploadTexture(this.velocityTexture, velocity, 4);
    uploadedBytes += this.uploadTexture(this.pressureTexture, pressure, 1);
    uploadedBytes += this.uploadTexture(this.divergenceTexture, divergence, 1);
    this.generation += 1;
    (this.uploadReceipt as { generation: number }).generation = this.generation;
    (this.uploadReceipt as { uploadedBytes: number }).uploadedBytes = uploadedBytes;
    return this.uploadReceipt;
  }

  private uploadTexture(
    texture: GPUTexture,
    values: Float32Array | Uint32Array,
    channels: 1 | 2 | 4,
  ): number {
    const [nx, ny, nz] = this.dimensions;
    const rowBytes = nx * channels * 4;
    const buffer = values.buffer as ArrayBuffer;
    let fullView = this.byteViews.get(buffer);
    if (!fullView) {
      fullView = new Uint8Array(buffer);
      this.byteViews.set(buffer, fullView);
    }
    const sourceStart = values.byteOffset;
    const sourceLength = values.byteLength;
    // Workspace arrays cover their complete buffers on the repeated path.
    const source = sourceStart === 0 && sourceLength === fullView.length
      ? fullView : fullView.subarray(sourceStart, sourceStart + sourceLength);
    const bytesPerRow = paddedTextureUploadInto(
      source, rowBytes, ny, nz, this.paddedUpload,
    );
    this.uploadDestination.texture = texture;
    this.uploadLayout.bytesPerRow = bytesPerRow;
    this.device.queue.writeTexture(
      this.uploadDestination,
      this.paddedUpload.buffer as ArrayBuffer,
      this.uploadLayout,
      this.uploadSize,
    );
    return bytesPerRow * ny * nz;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.densityTexture.destroy();
    this.levelSetTexture.destroy();
    this.gridCellTexture.destroy();
    this.velocityTexture.destroy();
    this.pressureTexture.destroy();
    this.divergenceTexture.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("adaptive mass atlas presentation is destroyed");
  }
}
