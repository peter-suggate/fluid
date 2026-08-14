/**
 * Mipped fluid-coverage volume consumed by the dry renderer's shadow cones.
 *
 * The static node-mip pyramid reserves fluid lanes it never fills: it is a
 * sparse, page-directoried, publish-once view of authored scenery, and its page
 * set only covers cells a static proxy or the terrain surface reaches. Evolving
 * water is neither — it moves every frame and it is routinely airborne, exactly
 * where no static page exists.
 *
 * The source is the solver's own coarse level set: the two-resolution surface
 * of Aanjaneya et al. Section 5 keeps fine phi on the interface and lets the
 * coarse field own sign and distance everywhere else, so the coarse field alone
 * already answers "how far to water" over the whole domain. A shadow only needs
 * that resolution, so this volume resamples the coarse field rather than
 * chasing the narrow band.
 *
 * All this structure adds is a mip chain, which the solver's field does not
 * carry: a widening cone needs progressively blurrier levels, and real GPU mips
 * with linear filtering give that in one fetch with no page lookup.
 *
 * Coverage is a fraction in [0, 1], never a shape. Shadow cones convert it to
 * an optical path length rather than an opacity, because water attenuates
 * light by colour over distance instead of blocking it.
 */

export const SVO_FLUID_COVERAGE_LAYOUT = Object.freeze({
  /**
   * Core WebGPU guarantees write-only storage for rgba8unorm but not for r8unorm,
   * which lives behind an optional feature. Coverage occupies the red channel and
   * the remaining lanes stay zero rather than gating the feature on an extension.
   */
  format: "rgba8unorm" as GPUTextureFormat,
  dimension: "3d" as GPUTextureDimension,
  bytesPerTexel: 4,
  /** Coverage lane within the stored texel. */
  coverageChannel: 0,
  fillWorkgroupSize: 4,
  reduceWorkgroupSize: 4,
  /** Frame uniform words: origin.xyz + levelCount, texelSize.xyz + valid, dimensions.xyz + generation. */
  frameWords: 12,
} as const);

export const SVO_FLUID_COVERAGE_LIMITS = Object.freeze({
  /** Beyond this the volume is refused rather than silently truncated. */
  maximumTexelsPerAxis: 2_048,
  maximumLevels: 12,
} as const);

export type SvoFluidCoverageTriple = readonly [number, number, number];

export interface SvoFluidCoveragePlanOptions {
  /** Cell dimensions of the solver's coarse level-set field. */
  fieldDimensions: SvoFluidCoverageTriple;
  /** World corner of that field. It spans the container box exactly. */
  worldOrigin_m: SvoFluidCoverageTriple;
  cellSize_m: SvoFluidCoverageTriple;
  /**
   * Coarse cells per coverage texel on each axis. Two halves the resolution the
   * cone can resolve while cutting allocation eightfold; shadow cones already
   * read a blurred level by the time they leave the receiver.
   */
  cellsPerTexel?: number;
}

export interface SvoFluidCoveragePlan {
  dimensions: SvoFluidCoverageTriple;
  levelCount: number;
  cellsPerTexel: number;
  /** Source field dimensions the fill pass reads. */
  fieldDimensions: SvoFluidCoverageTriple;
  /** World position of the volume's (0,0,0) texel corner. */
  origin_m: SvoFluidCoverageTriple;
  /** World size of one level-zero texel. */
  texelSize_m: SvoFluidCoverageTriple;
  /** World size of the whole volume. */
  extent_m: SvoFluidCoverageTriple;
  /** Level-zero allocation plus its complete mip chain. */
  allocatedBytes: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function finiteTriple(value: SvoFluidCoverageTriple, label: string): SvoFluidCoverageTriple {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
  return value;
}

function positiveTriple(value: SvoFluidCoverageTriple, label: string): SvoFluidCoverageTriple {
  finiteTriple(value, label);
  if (value.some((component) => component <= 0)) throw new RangeError(`${label} must be positive on every axis`);
  return value;
}

function integerTriple(value: SvoFluidCoverageTriple, label: string): SvoFluidCoverageTriple {
  if (value.length !== 3 || value.some((component) => !Number.isSafeInteger(component) || component < 0)) {
    throw new RangeError(`${label} must contain three non-negative integers`);
  }
  return value;
}

/** Size the volume and its mip chain over the solver's coarse level-set field. */
export function planSvoFluidCoverageVolume(options: SvoFluidCoveragePlanOptions): SvoFluidCoveragePlan {
  const fieldDimensions = integerTriple(options.fieldDimensions, "Fluid coverage field dimensions");
  const worldOrigin = finiteTriple(options.worldOrigin_m, "Fluid coverage world origin");
  const cellSize = positiveTriple(options.cellSize_m, "Fluid coverage cell size");
  const cellsPerTexel = positiveInteger(options.cellsPerTexel ?? 2, "Fluid coverage cells per texel");
  if (fieldDimensions.some((component) => component < 1)) throw new RangeError("Fluid coverage field dimensions must be positive");

  const dimensions = fieldDimensions.map((component) => Math.ceil(component / cellsPerTexel)) as unknown as SvoFluidCoverageTriple;
  if (dimensions.some((component) => component > SVO_FLUID_COVERAGE_LIMITS.maximumTexelsPerAxis)) {
    throw new RangeError("Fluid coverage volume exceeds the per-axis texel limit");
  }
  const levelCount = Math.min(
    SVO_FLUID_COVERAGE_LIMITS.maximumLevels,
    Math.floor(Math.log2(Math.max(...dimensions))) + 1,
  );
  const texelSize = cellSize.map((component) => component * cellsPerTexel) as unknown as SvoFluidCoverageTriple;
  const origin = [...worldOrigin] as unknown as SvoFluidCoverageTriple;
  const extent = dimensions.map((component, axis) => component * texelSize[axis]) as unknown as SvoFluidCoverageTriple;

  let allocatedBytes = 0;
  for (let level = 0; level < levelCount; level += 1) {
    const levelDimensions = dimensions.map((component) => Math.max(1, component >> level));
    allocatedBytes += levelDimensions[0] * levelDimensions[1] * levelDimensions[2] * SVO_FLUID_COVERAGE_LAYOUT.bytesPerTexel;
  }
  return {
    dimensions, levelCount, cellsPerTexel, fieldDimensions,
    origin_m: origin, texelSize_m: texelSize, extent_m: extent, allocatedBytes,
  };
}

/**
 * Convert a fluid signed distance to a coverage fraction.
 *
 * This is deliberately the identical band the environment proxy voxelizer uses
 * for solid occupancy: half coverage exactly on the interface, saturating one
 * half-diagonal to either side. Matching it keeps a water surface and a solid
 * surface that pass through the same cell equally represented, so a shadow does
 * not visibly change character where water meets stone.
 */
export function svoFluidCoverageFromSignedDistance(signedDistance_m: number, cellDiagonal_m: number): number {
  if (!Number.isFinite(cellDiagonal_m) || cellDiagonal_m <= 0) {
    throw new RangeError("Fluid coverage texel diagonal must be positive and finite");
  }
  if (!Number.isFinite(signedDistance_m)) return 0;
  return Math.max(0, Math.min(1, 0.5 - signedDistance_m / cellDiagonal_m));
}

/** Mean of eight children, matching the node-mip pyramid's mean-lane reduction. */
export function reduceSvoFluidCoverage(children: readonly number[]): number {
  if (children.length !== 8) throw new RangeError("Fluid coverage reduction requires eight children");
  let total = 0;
  for (const child of children) {
    if (!Number.isFinite(child) || child < 0 || child > 1) throw new RangeError("Fluid coverage child must lie in [0, 1]");
    total += child;
  }
  return total / 8;
}

/** Quantize to the stored byte lane, so CPU expectations match what the GPU reads back. */
export function quantizeSvoFluidCoverage(coverage: number): number {
  if (!Number.isFinite(coverage)) throw new RangeError("Fluid coverage must be finite");
  return Math.max(0, Math.min(255, Math.round(Math.max(0, Math.min(1, coverage)) * 255))) / 255;
}

/**
 * Mip level whose texels match a cone's current diameter.
 *
 * Identical in form to `svoNodeMipLod`, but it resolves against this volume's
 * own texel size rather than the finest simulation cell, so a downsampled
 * volume still reports the level that actually covers the cone footprint.
 */
export function svoFluidCoverageLod(diameter_m: number, texelSize_m: number): number {
  if (!Number.isFinite(texelSize_m) || texelSize_m <= 0) throw new RangeError("Fluid coverage texel size must be positive and finite");
  if (!Number.isFinite(diameter_m)) throw new RangeError("Fluid coverage cone diameter must be finite");
  return Math.max(0, Math.log2(Math.max(diameter_m, texelSize_m) / texelSize_m));
}

export interface SvoFluidCoverageFrame {
  origin_m: SvoFluidCoverageTriple;
  texelSize_m: SvoFluidCoverageTriple;
  dimensions: SvoFluidCoverageTriple;
  levelCount: number;
  /** Zero disables every sample site; the renderer then behaves exactly as it did before fluid shadows. */
  valid: boolean;
  generation: number;
}

/** Stable CPU/GPU frame ABI mirrored by `SvoFluidCoverageFrame` in WGSL. */
export function packSvoFluidCoverageFrame(frame: SvoFluidCoverageFrame): ArrayBuffer {
  finiteTriple(frame.origin_m, "Fluid coverage frame origin");
  positiveTriple(frame.texelSize_m, "Fluid coverage frame texel size");
  integerTriple(frame.dimensions, "Fluid coverage frame dimensions");
  positiveInteger(frame.levelCount, "Fluid coverage frame level count");
  if (!Number.isSafeInteger(frame.generation) || frame.generation < 0) {
    throw new RangeError("Fluid coverage frame generation must be a non-negative integer");
  }
  const buffer = new ArrayBuffer(SVO_FLUID_COVERAGE_LAYOUT.frameWords * 4);
  const floats = new Float32Array(buffer), words = new Uint32Array(buffer);
  floats.set(frame.origin_m, 0); words[3] = frame.levelCount;
  floats.set(frame.texelSize_m, 4); words[7] = frame.valid ? 1 : 0;
  words.set(frame.dimensions, 8); words[11] = frame.generation;
  return buffer;
}

/**
 * Binding-free WGSL. The including shader supplies the volume texture and a
 * filtering sampler; `mipmapFilter: "linear"` on that sampler is what makes the
 * level transition continuous, which the paged node-mip atlas has to hand-blend.
 */
export const svoFluidCoverageWGSL = /* wgsl */ `
struct SvoFluidCoverageFrame{origin_m:vec3f,levelCount:u32,texelSize_m:vec3f,valid:u32,dimensions:vec3u,generation:u32}
fn svoFluidCoverageReady(frame:SvoFluidCoverageFrame)->bool{return frame.valid!=0u&&frame.levelCount>0u;}
fn svoFluidCoverageLod(diameter_m:f32,texelSize_m:f32)->f32{return max(0.0,log2(max(diameter_m,texelSize_m)/texelSize_m));}
// Explicitly zero outside the solver box. The sampler clamps to edge, which
// would otherwise smear the boundary texel across the whole scene and put a
// shadow everywhere downwind of a full edge cell.
fn svoFluidCoverageAt(volume:texture_3d<f32>,volumeSampler:sampler,frame:SvoFluidCoverageFrame,position_m:vec3f,lod:f32)->f32{
  if(!svoFluidCoverageReady(frame)){return 0.0;}
  let uvw=(position_m-frame.origin_m)/(vec3f(frame.dimensions)*frame.texelSize_m);
  if(any(uvw<vec3f(0.0))||any(uvw>vec3f(1.0))){return 0.0;}
  let level=clamp(lod,0.0,f32(frame.levelCount-1u));
  return clamp(textureSampleLevel(volume,volumeSampler,uvw,level).x,0.0,1.0);
}
`;
