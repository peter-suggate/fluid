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
export type AdaptiveMassPresentationBrickResolution = 4 | 8;

/** Dense, x-major presentation data accepted by the WebGPU texture owner. */
export interface AdaptiveMassAtlasMaterialization {
  readonly dimensions: AdaptiveMassAtlasDimensions;
  /** `x + nx * (y + ny * z)`, one f32 density per finest presentation cell. */
  readonly density: ArrayLike<number>;
  /** Same indexing; signed metres for the shipping water level-set path. */
  readonly levelSetOrProxy: ArrayLike<number>;
  /** Two packed u32 words per cell, matching `adaptiveCellKey` in the grid overlay. */
  readonly ownerKeys: ArrayLike<number>;
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

export interface AdaptiveMassAtlasUploadReceipt {
  readonly generation: number;
  readonly cellCount: number;
  readonly uploadedBytes: number;
}

const OWNER_COMPONENT_MASK = 0x3ff;
const OWNER_MAXIMUM_COORDINATE = OWNER_COMPONENT_MASK;

function assertDimensions(
  dimensions: AdaptiveMassAtlasDimensions,
  maximumTextureDimension3D = OWNER_MAXIMUM_COORDINATE,
): void {
  for (let axis = 0; axis < 3; axis += 1) {
    const value = dimensions[axis];
    if (!Number.isSafeInteger(value) || value < 1
      || value > Math.min(maximumTextureDimension3D, OWNER_MAXIMUM_COORDINATE)) {
      throw new RangeError(
        `adaptive-mass presentation dimension ${axis} must be in [1, `
          + `${Math.min(maximumTextureDimension3D, OWNER_MAXIMUM_COORDINATE)}]; received ${value}`,
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
 * Pack the two-word adaptive ownership key consumed by `webgpu-grid-overlay`.
 * Coordinates are in the dense finest presentation lattice. The existing ABI
 * gives x/z one cubic `cellScale` and y an explicit upper bound.
 */
export function packAdaptiveMassPresentationOwnerKey(
  lower: AdaptiveMassAtlasCoordinate,
  cellScale: number,
): readonly [number, number] {
  if (!Number.isSafeInteger(cellScale) || cellScale < 1 || cellScale > OWNER_COMPONENT_MASK) {
    throw new RangeError(`adaptive owner cell scale must be in [1, 1023]; received ${cellScale}`);
  }
  for (let axis = 0; axis < 3; axis += 1) {
    if (!Number.isSafeInteger(lower[axis]) || lower[axis] < 0
      || lower[axis] > OWNER_MAXIMUM_COORDINATE) {
      throw new RangeError(`adaptive owner lower coordinate ${axis} is invalid: ${lower[axis]}`);
    }
  }
  const upperY = lower[1] + cellScale;
  if (upperY > OWNER_MAXIMUM_COORDINATE) {
    throw new RangeError(`adaptive owner upper-y coordinate exceeds 1023: ${upperY}`);
  }
  const first = (lower[0] | (lower[2] << 10) | (cellScale << 20)) >>> 0;
  const second = (lower[1] | (upperY << 10)) >>> 0;
  return [first, second];
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
): AdaptiveMassAtlasMaterialization {
  assertDimensions(options.dimensions);
  if (!Number.isFinite(options.emptyLevelSet) || options.emptyLevelSet <= 0) {
    throw new RangeError(`emptyLevelSet must be finite and positive; received ${options.emptyLevelSet}`);
  }
  if (!Number.isFinite(options.densityProxyBand) || options.densityProxyBand <= 0) {
    throw new RangeError(`densityProxyBand must be finite and positive; received ${options.densityProxyBand}`);
  }
  const count = cellCount(options.dimensions);
  const density = new Float32Array(count);
  const levelSetOrProxy = new Float32Array(count).fill(Math.fround(options.emptyLevelSet));
  const ownerKeys = new Uint32Array(count * 2);
  const maximumHorizontalSpan = Math.max(options.dimensions[0], options.dimensions[2]);
  const backgroundScale = Math.min(OWNER_COMPONENT_MASK, maximumHorizontalSpan);
  const background = packAdaptiveMassPresentationOwnerKey([0, 0, 0],
    Math.min(backgroundScale, OWNER_COMPONENT_MASK - 1));
  for (let index = 0; index < count; index += 1) {
    ownerKeys[2 * index] = background[0];
    ownerKeys[2 * index + 1] = background[1];
  }
  const occupied = new Uint8Array(count);

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
          const lower: [number, number, number] = [
            brick.originFine[0] + localX * cellScale,
            brick.originFine[1] + localY * cellScale,
            brick.originFine[2] + localZ * cellScale,
          ];
          const owner = packAdaptiveMassPresentationOwnerKey(lower, cellScale);
          for (let childZ = 0; childZ < cellScale; childZ += 1) {
            for (let childY = 0; childY < cellScale; childY += 1) {
              for (let childX = 0; childX < cellScale; childX += 1) {
                const destination = linearIndex(options.dimensions,
                  lower[0] + childX, lower[1] + childY, lower[2] + childZ);
                if (occupied[destination] !== 0) {
                  throw new Error(`adaptive presentation bricks overlap at dense cell ${destination}`);
                }
                occupied[destination] = 1;
                density[destination] = Math.fround(rho);
                levelSetOrProxy[destination] = Math.fround(phi);
                ownerKeys[2 * destination] = owner[0];
                ownerKeys[2 * destination + 1] = owner[1];
              }
            }
          }
        }
      }
    }
  }
  return { dimensions: options.dimensions, density, levelSetOrProxy, ownerKeys };
}

function paddedTextureUpload(
  source: Uint8Array,
  rowBytes: number,
  rowsPerImage: number,
  imageCount: number,
): { readonly bytes: Uint8Array; readonly bytesPerRow: number } {
  const bytesPerRow = Math.ceil(rowBytes / 256) * 256;
  const bytes = new Uint8Array(bytesPerRow * rowsPerImage * imageCount);
  for (let image = 0; image < imageCount; image += 1) {
    for (let row = 0; row < rowsPerImage; row += 1) {
      const sourceRow = row + rowsPerImage * image;
      bytes.set(source.subarray(sourceRow * rowBytes, (sourceRow + 1) * rowBytes),
        (row + rowsPerImage * image) * bytesPerRow);
    }
  }
  return { bytes, bytesPerRow };
}

/** Texture owner consumed directly by `GPUSolverInstance` publications. */
export class WebGPUAdaptiveMassAtlasPresentation {
  readonly densityTexture: GPUTexture;
  readonly levelSetTexture: GPUTexture;
  readonly gridCellTexture: GPUTexture;
  readonly allocatedBytes: number;
  generation = 0;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly dimensions: AdaptiveMassAtlasDimensions,
  ) {
    assertDimensions(dimensions, device.limits.maxTextureDimension3D);
    const scalarUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
      | GPUTextureUsage.COPY_SRC;
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
        | GPUTextureUsage.COPY_SRC,
    });
    this.allocatedBytes = cellCount(dimensions) * 16;
  }

  /** Alias matching the optional `GPUSolverInstance.surfaceFieldTexture`. */
  get surfaceFieldTexture(): GPUTexture { return this.levelSetTexture; }

  /** Upload one complete CPU-authored presentation generation. */
  upload(materialization: AdaptiveMassAtlasMaterialization): AdaptiveMassAtlasUploadReceipt {
    this.assertLive();
    if (materialization.dimensions.some((value, axis) => value !== this.dimensions[axis])) {
      throw new RangeError(
        `adaptive presentation upload dimensions ${materialization.dimensions.join("x")} `
          + `do not match ${this.dimensions.join("x")}`,
      );
    }
    const count = cellCount(this.dimensions);
    if (materialization.density.length !== count
      || materialization.levelSetOrProxy.length !== count
      || materialization.ownerKeys.length !== 2 * count) {
      throw new RangeError(
        `adaptive presentation upload requires ${count} scalar values and ${2 * count} owner words`,
      );
    }
    const density = Float32Array.from(materialization.density);
    const levelSet = Float32Array.from(materialization.levelSetOrProxy);
    const owners = Uint32Array.from(materialization.ownerKeys);
    for (let index = 0; index < count; index += 1) {
      if (!Number.isFinite(density[index]) || !Number.isFinite(levelSet[index])) {
        throw new RangeError(`adaptive presentation upload cell ${index} is non-finite`);
      }
    }
    const [nx, ny, nz] = this.dimensions;
    const upload = (
      texture: GPUTexture,
      values: Float32Array | Uint32Array,
      channels: 1 | 2,
    ): number => {
      const rowBytes = nx * channels * 4;
      const source = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
      const packed = paddedTextureUpload(source, rowBytes, ny, nz);
      this.device.queue.writeTexture(
        { texture },
        packed.bytes.buffer as ArrayBuffer,
        { offset: packed.bytes.byteOffset, bytesPerRow: packed.bytesPerRow, rowsPerImage: ny },
        { width: nx, height: ny, depthOrArrayLayers: nz },
      );
      return packed.bytes.byteLength;
    };
    let uploadedBytes = upload(this.densityTexture, density, 1);
    uploadedBytes += upload(this.levelSetTexture, levelSet, 1);
    uploadedBytes += upload(this.gridCellTexture, owners, 2);
    this.generation += 1;
    return { generation: this.generation, cellCount: count, uploadedBytes };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.densityTexture.destroy();
    this.levelSetTexture.destroy();
    this.gridCellTexture.destroy();
  }

  private assertLive(): void {
    if (this.destroyed) throw new Error("adaptive mass atlas presentation is destroyed");
  }
}
