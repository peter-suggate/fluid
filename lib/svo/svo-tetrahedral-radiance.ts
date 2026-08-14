import { SVO_NODE_MIP_LAYOUT, type SvoNodeMipPyramidPlan } from "./svo-node-mip-pyramid";

export type SvoRadianceRgb = readonly [number, number, number];
export type SvoTetrahedralRadiance = readonly [
  SvoRadianceRgb,
  SvoRadianceRgb,
  SvoRadianceRgb,
  SvoRadianceRgb,
];

const INV_SQRT_THREE = 1 / Math.sqrt(3);
const RGB9E5_MANTISSA_BITS = 9;
const RGB9E5_EXPONENT_BIAS = 15;
const RGB9E5_MAX = 65_408;

/**
 * Four maximally separated directions. Their sum is zero and every distinct
 * pair has dot product -1/3, so four samples are an invertible representation
 * of an RGB constant-plus-linear directional field (the L0+L1 SH subspace).
 */
export const SVO_TETRAHEDRAL_RADIANCE_DIRECTIONS = Object.freeze([
  [INV_SQRT_THREE, INV_SQRT_THREE, INV_SQRT_THREE],
  [INV_SQRT_THREE, -INV_SQRT_THREE, -INV_SQRT_THREE],
  [-INV_SQRT_THREE, INV_SQRT_THREE, -INV_SQRT_THREE],
  [-INV_SQRT_THREE, -INV_SQRT_THREE, INV_SQRT_THREE],
] as const);

/** Four filterable HDR texels per spatial texel, one RGB9E5 texture per direction. */
export const SVO_TETRAHEDRAL_RADIANCE_LAYOUT = Object.freeze({
  directionCount: 4,
  textureFormat: "rgb9e5ufloat" as GPUTextureFormat,
  bytesPerDirectionTexel: 4,
  bytesPerTexel: 16,
  physicalPageSize: SVO_NODE_MIP_LAYOUT.physicalSize,
  bytesPerPhysicalPage: SVO_NODE_MIP_LAYOUT.physicalSize ** 3 * 16,
} as const);

export interface SvoTetrahedralFirstOrderCoefficients {
  /** Direction-independent radiance. */
  mean: SvoRadianceRgb;
  /** Coefficients multiplying world-space x, y, and z direction. */
  vectorX: SvoRadianceRgb;
  vectorY: SvoRadianceRgb;
  vectorZ: SvoRadianceRgb;
}

function finiteRgb(value: SvoRadianceRgb, label: string, nonNegative = true): SvoRadianceRgb {
  if (value.length !== 3 || value.some((channel) => !Number.isFinite(channel) || (nonNegative && channel < 0))) {
    throw new RangeError(`${label} must contain three ${nonNegative ? "non-negative " : ""}finite channels`);
  }
  return [value[0], value[1], value[2]];
}

function normalized(value: readonly [number, number, number], label: string): readonly [number, number, number] {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function dot(left: readonly number[], right: readonly number[]): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function mapRgb(value: SvoRadianceRgb, fn: (channel: number, index: number) => number): SvoRadianceRgb {
  return [fn(value[0], 0), fn(value[1], 1), fn(value[2], 2)];
}

export function canonicalSvoTetrahedralRadiance(value: SvoTetrahedralRadiance): SvoTetrahedralRadiance {
  if (value.length !== 4) throw new RangeError("Tetrahedral radiance requires four directional samples");
  return value.map((sample, index) => finiteRgb(sample, `Tetrahedral radiance sample ${index}`)) as unknown as SvoTetrahedralRadiance;
}

/** Convert four directional values into the equivalent constant-plus-linear field. */
export function svoTetrahedralFirstOrderCoefficients(
  input: SvoTetrahedralRadiance,
): SvoTetrahedralFirstOrderCoefficients {
  const samples = canonicalSvoTetrahedralRadiance(input);
  const mean = [0, 0, 0], vectorX = [0, 0, 0], vectorY = [0, 0, 0], vectorZ = [0, 0, 0];
  samples.forEach((sample, directionIndex) => {
    const direction = SVO_TETRAHEDRAL_RADIANCE_DIRECTIONS[directionIndex];
    for (let channel = 0; channel < 3; channel += 1) {
      mean[channel] += sample[channel] * .25;
      vectorX[channel] += sample[channel] * direction[0] * .75;
      vectorY[channel] += sample[channel] * direction[1] * .75;
      vectorZ[channel] += sample[channel] * direction[2] * .75;
    }
  });
  return {
    mean: mean as unknown as SvoRadianceRgb,
    vectorX: vectorX as unknown as SvoRadianceRgb,
    vectorY: vectorY as unknown as SvoRadianceRgb,
    vectorZ: vectorZ as unknown as SvoRadianceRgb,
  };
}

/** Raw L0+L1 reconstruction. Exposed for numerical validation; it can ring below zero. */
export function evaluateSvoTetrahedralRadianceRaw(
  input: SvoTetrahedralRadiance,
  directionIn: readonly [number, number, number],
): SvoRadianceRgb {
  const direction = normalized(directionIn, "Tetrahedral radiance direction");
  const coefficients = svoTetrahedralFirstOrderCoefficients(input);
  return [0, 1, 2].map((channel) => coefficients.mean[channel]
    + coefficients.vectorX[channel] * direction[0]
    + coefficients.vectorY[channel] * direction[1]
    + coefficients.vectorZ[channel] * direction[2]) as unknown as SvoRadianceRgb;
}

/** Non-negative outgoing radiance reconstructed in one world-space direction. */
export function evaluateSvoTetrahedralRadiance(
  input: SvoTetrahedralRadiance,
  direction: readonly [number, number, number],
): SvoRadianceRgb {
  return mapRgb(evaluateSvoTetrahedralRadianceRaw(input, direction), (channel) => Math.max(0, channel));
}

/** Analytic cosine convolution of the represented L0+L1 field. */
export function evaluateSvoTetrahedralDiffuseIrradiance(
  input: SvoTetrahedralRadiance,
  normalIn: readonly [number, number, number],
): SvoRadianceRgb {
  const normal = normalized(normalIn, "Tetrahedral irradiance normal");
  const coefficients = svoTetrahedralFirstOrderCoefficients(input);
  return [0, 1, 2].map((channel) => Math.max(0, Math.PI * coefficients.mean[channel]
    + 2 * Math.PI / 3 * (coefficients.vectorX[channel] * normal[0]
      + coefficients.vectorY[channel] * normal[1]
      + coefficients.vectorZ[channel] * normal[2]))) as unknown as SvoRadianceRgb;
}

/**
 * Premultiplied Lambertian surface emission at the four basis directions.
 * Premultiplication makes an eight-child arithmetic mean the correct mip rule.
 */
export function svoTetrahedralLambertianEmission(
  radianceIn: SvoRadianceRgb,
  normalIn: readonly [number, number, number],
  coverage = 1,
): SvoTetrahedralRadiance {
  const radiance = finiteRgb(radianceIn, "Lambertian emission radiance");
  const normal = normalized(normalIn, "Lambertian emission normal");
  if (!Number.isFinite(coverage) || coverage < 0 || coverage > 1) {
    throw new RangeError("Lambertian emission coverage must be from zero to one");
  }
  return SVO_TETRAHEDRAL_RADIANCE_DIRECTIONS.map((direction) => {
    const visible = dot(direction, normal) > 0 ? coverage : 0;
    return mapRgb(radiance, (channel) => channel * visible);
  }) as unknown as SvoTetrahedralRadiance;
}

/** Mean reduction for eight premultiplied directional-radiance children. */
export function reduceSvoTetrahedralRadianceChildren(
  children: readonly SvoTetrahedralRadiance[],
): SvoTetrahedralRadiance {
  if (children.length !== 8) throw new RangeError("Tetrahedral radiance reduction requires eight children");
  const result = Array.from({ length: 4 }, () => [0, 0, 0]);
  children.forEach((child) => canonicalSvoTetrahedralRadiance(child).forEach((sample, direction) => {
    for (let channel = 0; channel < 3; channel += 1) result[direction][channel] += sample[channel] / 8;
  }));
  return result as unknown as SvoTetrahedralRadiance;
}

/** Pack one non-negative HDR RGB value into the WebGPU RGB9E5 bit layout. */
export function packSvoRadianceRgb9e5(input: SvoRadianceRgb): number {
  const rgb = finiteRgb(input, "RGB9E5 radiance").map((channel) => Math.min(channel, RGB9E5_MAX));
  const maximum = Math.max(...rgb);
  if (!(maximum > 0)) return 0;
  let exponent = Math.max(0, Math.floor(Math.log2(maximum)) + 1 + RGB9E5_EXPONENT_BIAS);
  exponent = Math.min(31, exponent);
  let scale = 2 ** (exponent - RGB9E5_EXPONENT_BIAS - RGB9E5_MANTISSA_BITS);
  if (Math.round(maximum / scale) === 1 << RGB9E5_MANTISSA_BITS && exponent < 31) {
    exponent += 1;
    scale *= 2;
  }
  const mantissa = rgb.map((channel) => Math.min(0x1ff, Math.round(channel / scale)));
  return (mantissa[0] | mantissa[1] << 9 | mantissa[2] << 18 | exponent << 27) >>> 0;
}

export function unpackSvoRadianceRgb9e5(packed: number): SvoRadianceRgb {
  if (!Number.isInteger(packed) || packed < 0 || packed > 0xffff_ffff) throw new RangeError("Packed RGB9E5 radiance must fit uint32");
  const exponent = packed >>> 27;
  const scale = 2 ** (exponent - RGB9E5_EXPONENT_BIAS - RGB9E5_MANTISSA_BITS);
  return [(packed & 0x1ff) * scale, ((packed >>> 9) & 0x1ff) * scale, ((packed >>> 18) & 0x1ff) * scale];
}

/** Four uint32 words map directly to four rgb9e5ufloat texture uploads. */
export function packSvoTetrahedralRadianceTexel(input: SvoTetrahedralRadiance): Uint32Array<ArrayBuffer> {
  return Uint32Array.from(canonicalSvoTetrahedralRadiance(input).map(packSvoRadianceRgb9e5));
}

export function unpackSvoTetrahedralRadianceTexel(input: Uint32Array, wordOffset = 0): SvoTetrahedralRadiance {
  if (!Number.isSafeInteger(wordOffset) || wordOffset < 0 || wordOffset + 4 > input.length) {
    throw new RangeError("Packed tetrahedral radiance texel exceeds its word array");
  }
  return [0, 1, 2, 3].map((index) => unpackSvoRadianceRgb9e5(input[wordOffset + index])) as unknown as SvoTetrahedralRadiance;
}

export function svoTetrahedralRadianceAtlasBytes(plan: SvoNodeMipPyramidPlan): number {
  return plan.atlas.capacity * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.bytesPerPhysicalPage;
}

/** Binding-free shader library; the renderer can place the four textures in its existing group. */
export const svoTetrahedralRadianceWGSL = /* wgsl */ `
const SVO_TETRA_INV_SQRT_THREE:f32=0.5773502691896258;
struct SvoTetraRadiance{lobe0:vec3f,lobe1:vec3f,lobe2:vec3f,lobe3:vec3f}
fn svoTetraDirection0()->vec3f{return vec3f(SVO_TETRA_INV_SQRT_THREE);}
fn svoTetraDirection1()->vec3f{return vec3f(SVO_TETRA_INV_SQRT_THREE,-SVO_TETRA_INV_SQRT_THREE,-SVO_TETRA_INV_SQRT_THREE);}
fn svoTetraDirection2()->vec3f{return vec3f(-SVO_TETRA_INV_SQRT_THREE,SVO_TETRA_INV_SQRT_THREE,-SVO_TETRA_INV_SQRT_THREE);}
fn svoTetraDirection3()->vec3f{return vec3f(-SVO_TETRA_INV_SQRT_THREE,-SVO_TETRA_INV_SQRT_THREE,SVO_TETRA_INV_SQRT_THREE);}
fn svoTetraWeights(directionIn:vec3f)->vec4f{
  let direction=normalize(directionIn);return .25*(vec4f(1.0)+3.0*vec4f(dot(svoTetraDirection0(),direction),dot(svoTetraDirection1(),direction),dot(svoTetraDirection2(),direction),dot(svoTetraDirection3(),direction)));
}
fn svoTetraDiffuseWeights(normalIn:vec3f)->vec4f{
  let normal=normalize(normalIn);return .7853981633974483*(vec4f(1.0)+2.0*vec4f(dot(svoTetraDirection0(),normal),dot(svoTetraDirection1(),normal),dot(svoTetraDirection2(),normal),dot(svoTetraDirection3(),normal)));
}
fn svoTetraRadianceAlong(value:SvoTetraRadiance,direction:vec3f)->vec3f{
  let weights=svoTetraWeights(direction);return max(value.lobe0*weights.x+value.lobe1*weights.y+value.lobe2*weights.z+value.lobe3*weights.w,vec3f(0.0));
}
fn svoTetraDiffuseIrradiance(value:SvoTetraRadiance,normal:vec3f)->vec3f{
  let weights=svoTetraDiffuseWeights(normal);return max(value.lobe0*weights.x+value.lobe1*weights.y+value.lobe2*weights.z+value.lobe3*weights.w,vec3f(0.0));
}
fn svoTetraSample(lobe0:texture_3d<f32>,lobe1:texture_3d<f32>,lobe2:texture_3d<f32>,lobe3:texture_3d<f32>,atlasSampler:sampler,uv:vec3f)->SvoTetraRadiance{
  return SvoTetraRadiance(textureSampleLevel(lobe0,atlasSampler,uv,0.0).xyz,textureSampleLevel(lobe1,atlasSampler,uv,0.0).xyz,textureSampleLevel(lobe2,atlasSampler,uv,0.0).xyz,textureSampleLevel(lobe3,atlasSampler,uv,0.0).xyz);
}
`;
