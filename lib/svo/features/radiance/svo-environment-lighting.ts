import { environmentIds, environmentIndex, type EnvironmentId } from "../../../core/environments";
import type { LinearRgb } from "../../../core/webgpu-lighting";
import type { SvoVec3 } from "../primary-visibility/webgpu-svo-traversal";

/** Six host-shareable vec4 lanes; no image or sampler dependency. */
export const SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES = 96;
export const SVO_ENVIRONMENT_LIGHTING_RECORD_WORDS = SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES / 4;
export const SVO_ENVIRONMENT_LIGHTING_VERSION = 1;

export interface SvoEnvironmentLightingRecord {
  environmentId: EnvironmentId;
  revision: number;
  lowerRadianceLinear: LinearRgb;
  diffuseScale: number;
  upperRadianceLinear: LinearRgb;
  specularScale: number;
  accentRadianceLinear: LinearRgb;
  accentPower: number;
  keyLightColorLinear: LinearRgb;
  sunIntensity: number;
  keyLightDirection: SvoVec3;
  sunSharpness: number;
  /** Optional directional cloud coverage, zero preserves the original gradient. */
  cloudCoverage?: number;
}

export interface SvoEnvironmentLightingBuild {
  record: SvoEnvironmentLightingRecord;
  packedRecord: Uint32Array<ArrayBuffer>;
  contentRevision: string;
  cacheKey: string;
}

export interface SvoEnvironmentLightingOptions {
  cloudCoverage?: number;
  keyLightColorLinear?: LinearRgb;
  keyLightDirection?: SvoVec3;
  sunIntensity?: number;
  diffuseScale?: number;
  specularScale?: number;
  lowerRadianceLinear?: LinearRgb;
  upperRadianceLinear?: LinearRgb;
  accentRadianceLinear?: LinearRgb;
}

interface EnvironmentPalette {
  lower: LinearRgb;
  upper: LinearRgb;
  accent: LinearRgb;
  keyColor: LinearRgb;
  keyDirection: SvoVec3;
}

const DEFAULT_DIRECTION: SvoVec3 = [-0.45, 0.86, 0.28];

/** Exact linear values from `environmentLight`, `environmentAccent`, and key-light helpers. */
const PALETTES: Readonly<Record<EnvironmentId, EnvironmentPalette>> = Object.freeze({
  conservatory: { lower: [0.035, 0.055, 0.044], upper: [0.46, 0.58, 0.43], accent: [0.24, 0.55, 0.39], keyColor: [1, 0.86, 0.62], keyDirection: DEFAULT_DIRECTION },
  courtyard: { lower: [0.16, 0.16, 0.13], upper: [0.68, 0.69, 0.59], accent: [0.10, 0.34, 0.44], keyColor: [1, 0.77, 0.52], keyDirection: DEFAULT_DIRECTION },
  "night-lab": { lower: [0.016, 0.017, 0.020], upper: [0.065, 0.068, 0.078], accent: [0.08, 0.11, 0.15], keyColor: [1, 0.94, 0.80], keyDirection: [-0.35, 0.90, 0.20] },
  "concrete-gallery": { lower: [0.045, 0.050, 0.048], upper: [0.30, 0.32, 0.30], accent: [0.72, 0.42, 0.22], keyColor: [1, 0.67, 0.40], keyDirection: [0.32, 0.82, 0.34] },
  bathhouse: { lower: [0.045, 0.038, 0.032], upper: [0.34, 0.31, 0.25], accent: [0.54, 0.39, 0.25], keyColor: [1, 0.83, 0.57], keyDirection: [-0.55, 0.75, -0.12] },
  "research-station": { lower: [0.002, 0.012, 0.022], upper: [0.028, 0.12, 0.17], accent: [0.14, 0.56, 0.68], keyColor: [0.42, 0.83, 1], keyDirection: [0.15, 0.42, 0.90] },
  default: { lower: [0.012, 0.025, 0.028], upper: [0.19, 0.30, 0.29], accent: [0.18, 0.34, 0.31], keyColor: [1, 0.86, 0.66], keyDirection: DEFAULT_DIRECTION },
  // The stage rig's exact values: see STUDIO_STAGE_DRY_SCENE_LIGHTING in
  // `svo-dry-scene-lighting.ts` — a cold, very low fill under one warm key.
  stage: { lower: [0.008, 0.010, 0.014], upper: [0.016, 0.020, 0.028], accent: [0.012, 0.015, 0.022], keyColor: [1, 0.96, 0.9], keyDirection: [-0.22, 0.94, 0.26] },
  // The lower value is the existing misty pale ground line after the garden
  // horizon mix; the upper value is its existing cloud-free pale zenith. The
  // key matches PORCELAIN_GARDEN_DRY_SCENE_LIGHTING's sun, so the specular
  // sun lobe and the shaded key come from the same place.
  garden: { lower: [0.60, 0.61, 0.59], upper: [0.52, 0.60, 0.72], accent: [0.48, 0.50, 0.53], keyColor: [1, 0.93, 0.82], keyDirection: [-0.12507, 0.64279, -0.75575] },
});

function finite(value: number, label: string): number {
  if (!Number.isFinite(value)) throw new RangeError(`${label} must be finite`);
  return value;
}

function nonNegative(value: number, label: string): number {
  const result = finite(value, label);
  if (result < 0) throw new RangeError(`${label} must be non-negative`);
  return result;
}

function color(value: LinearRgb, label: string): [number, number, number] {
  if (value.length !== 3) throw new RangeError(`${label} must contain three channels`);
  return value.map((channel) => nonNegative(channel, label)) as [number, number, number];
}

function normalized(value: SvoVec3, label: string): SvoVec3 {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / length, value[1] / length, value[2] / length];
}

export function canonicalSvoEnvironmentLightingRecord(input: SvoEnvironmentLightingRecord): SvoEnvironmentLightingRecord {
  if (!environmentIds.includes(input.environmentId)) throw new RangeError(`Unknown SVO environment ${String(input.environmentId)}`);
  if (!Number.isSafeInteger(input.revision) || input.revision < 1 || input.revision > 0xffff_ffff) {
    throw new RangeError("SVO environment-lighting revision must be a positive uint32");
  }
  const accentPower = nonNegative(input.accentPower, "SVO environment accent power");
  const sunSharpness = nonNegative(input.sunSharpness, "SVO environment sun sharpness");
  if (!(accentPower > 0) || !(sunSharpness > 0)) throw new RangeError("SVO environment lobe powers must be positive");
  const cloudCoverage = finite(input.cloudCoverage ?? 0, "SVO environment cloud coverage");
  if (cloudCoverage < 0 || cloudCoverage > 1) throw new RangeError("SVO environment cloud coverage must be from zero to one");
  return Object.freeze({
    ...input,
    cloudCoverage,
    lowerRadianceLinear: color(input.lowerRadianceLinear, "SVO environment lower radiance"),
    diffuseScale: nonNegative(input.diffuseScale, "SVO environment diffuse scale"),
    upperRadianceLinear: color(input.upperRadianceLinear, "SVO environment upper radiance"),
    specularScale: nonNegative(input.specularScale, "SVO environment specular scale"),
    accentRadianceLinear: color(input.accentRadianceLinear, "SVO environment accent radiance"),
    accentPower,
    keyLightColorLinear: color(input.keyLightColorLinear, "SVO environment key-light color"),
    sunIntensity: nonNegative(input.sunIntensity, "SVO environment sun intensity"),
    keyLightDirection: normalized(input.keyLightDirection, "SVO environment key-light direction"),
    sunSharpness,
  });
}

/** Build one selected image-free fallback from the existing authored palette. */
export function svoEnvironmentLightingRecord(
  environmentId: EnvironmentId,
  revision = 1,
  options: SvoEnvironmentLightingOptions = {},
): SvoEnvironmentLightingRecord {
  const palette = PALETTES[environmentId];
  return canonicalSvoEnvironmentLightingRecord({
    environmentId,
    revision,
    lowerRadianceLinear: options.lowerRadianceLinear ?? palette.lower,
    diffuseScale: options.diffuseScale ?? 1,
    upperRadianceLinear: options.upperRadianceLinear ?? palette.upper,
    specularScale: options.specularScale ?? 1,
    accentRadianceLinear: options.accentRadianceLinear ?? palette.accent,
    accentPower: 3,
    keyLightColorLinear: options.keyLightColorLinear ?? palette.keyColor,
    sunIntensity: options.sunIntensity ?? 2.5,
    keyLightDirection: options.keyLightDirection ?? palette.keyDirection,
    cloudCoverage: options.cloudCoverage ?? 0,
    sunSharpness: 360,
  });
}

/** Canonical ordered coverage for capture and cross-environment validation. */
export function buildAllSvoEnvironmentLightingRecords(revision = 1): readonly SvoEnvironmentLightingRecord[] {
  return environmentIds.map((environmentId) => svoEnvironmentLightingRecord(environmentId, revision));
}

export function packSvoEnvironmentLightingRecords(records: readonly SvoEnvironmentLightingRecord[]): Uint32Array<ArrayBuffer> {
  const canonical = records.map(canonicalSvoEnvironmentLightingRecord);
  const seen = new Set<EnvironmentId>();
  const buffer = new ArrayBuffer(canonical.length * SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  canonical.forEach((record, index) => {
    if (seen.has(record.environmentId)) throw new RangeError(`Duplicate SVO environment-lighting record ${record.environmentId}`);
    seen.add(record.environmentId);
    const offset = index * SVO_ENVIRONMENT_LIGHTING_RECORD_WORDS;
    floats.set([...record.lowerRadianceLinear, record.diffuseScale], offset);
    floats.set([...record.upperRadianceLinear, record.specularScale], offset + 4);
    floats.set([...record.accentRadianceLinear, record.accentPower], offset + 8);
    floats.set([...record.keyLightColorLinear, record.sunIntensity], offset + 12);
    floats.set([...record.keyLightDirection, record.sunSharpness], offset + 16);
    words.set([environmentIndex(record.environmentId), record.revision, SVO_ENVIRONMENT_LIGHTING_VERSION, 0], offset + 20);
    floats[offset + 23] = record.cloudCoverage ?? 0;
  });
  return words;
}

export function unpackSvoEnvironmentLightingRecord(packed: Uint32Array, recordIndex = 0): SvoEnvironmentLightingRecord {
  if (!Number.isSafeInteger(recordIndex) || recordIndex < 0) throw new RangeError("SVO environment-lighting record index must be non-negative");
  const offset = recordIndex * SVO_ENVIRONMENT_LIGHTING_RECORD_WORDS;
  if (offset + SVO_ENVIRONMENT_LIGHTING_RECORD_WORDS > packed.length) throw new RangeError("SVO environment-lighting record index exceeds the table");
  const environmentId = environmentIds[packed[offset + 20]];
  if (!environmentId || packed[offset + 22] !== SVO_ENVIRONMENT_LIGHTING_VERSION) throw new RangeError("Packed SVO environment-lighting identity is invalid");
  const floats = new Float32Array(packed.buffer, packed.byteOffset, packed.length);
  return canonicalSvoEnvironmentLightingRecord({
    environmentId,
    revision: packed[offset + 21],
    cloudCoverage: floats[offset + 23],
    lowerRadianceLinear: [floats[offset], floats[offset + 1], floats[offset + 2]], diffuseScale: floats[offset + 3],
    upperRadianceLinear: [floats[offset + 4], floats[offset + 5], floats[offset + 6]], specularScale: floats[offset + 7],
    accentRadianceLinear: [floats[offset + 8], floats[offset + 9], floats[offset + 10]], accentPower: floats[offset + 11],
    keyLightColorLinear: [floats[offset + 12], floats[offset + 13], floats[offset + 14]], sunIntensity: floats[offset + 15],
    keyLightDirection: [floats[offset + 16], floats[offset + 17], floats[offset + 18]], sunSharpness: floats[offset + 19],
  });
}

function add(left: LinearRgb, right: LinearRgb): LinearRgb {
  return [left[0] + right[0], left[1] + right[1], left[2] + right[2]];
}

function scale(value: LinearRgb, factor: number): LinearRgb {
  return [value[0] * factor, value[1] * factor, value[2] * factor];
}

function mix(left: LinearRgb, right: LinearRgb, factor: number): LinearRgb {
  return add(scale(left, 1 - factor), scale(right, factor));
}

function dot(left: SvoVec3, right: SvoVec3): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

/** Irradiance (radiance × solid-angle integral), not normalized diffuse
 * radiance. A constant sky L produces πL; Lambertian consumers divide by π. */
export function evaluateSvoEnvironmentDiffuseIrradiance(
  input: SvoEnvironmentLightingRecord,
  worldNormal: SvoVec3,
): LinearRgb {
  const record = canonicalSvoEnvironmentLightingRecord(input);
  const normal = normalized(worldNormal, "SVO environment diffuse normal");
  const average = mix(record.lowerRadianceLinear, record.upperRadianceLinear, 0.5);
  const gradient = scale([
    record.upperRadianceLinear[0] - record.lowerRadianceLinear[0],
    record.upperRadianceLinear[1] - record.lowerRadianceLinear[1],
    record.upperRadianceLinear[2] - record.lowerRadianceLinear[2],
  ], normal[1] / 3);
  const horizon = scale(record.accentRadianceLinear, 0.13 * (1 - Math.abs(normal[1])));
  return scale(add(add(average, gradient), horizon), Math.PI * record.diffuseScale);
}

/** Image-free roughness-prefiltered sky and authored key-light fallback. */
export function evaluateSvoEnvironmentPrefilteredSpecular(
  input: SvoEnvironmentLightingRecord,
  worldDirection: SvoVec3,
  roughness: number,
): LinearRgb {
  const record = canonicalSvoEnvironmentLightingRecord(input);
  const direction = normalized(worldDirection, "SVO environment specular direction");
  if (!Number.isFinite(roughness) || roughness < 0 || roughness > 1) throw new RangeError("SVO environment roughness must be from zero to one");
  const roughnessSquared = roughness * roughness;
  const vertical = Math.max(0, Math.min(1, direction[1] * 0.5 + 0.5));
  const sky = mix(mix(record.lowerRadianceLinear, record.upperRadianceLinear, vertical), mix(record.lowerRadianceLinear, record.upperRadianceLinear, 0.5), roughnessSquared);
  const cosine = Math.max(0, dot(direction, record.keyLightDirection));
  const sharpExponent = record.sunSharpness * (1 - roughnessSquared) + roughnessSquared;
  const broadExponent = 15 * (1 - roughnessSquared) + roughnessSquared;
  const sharpEnergy = (sharpExponent + 2) / (record.sunSharpness + 2);
  const broadEnergy = (broadExponent + 2) / 17;
  const sun = scale(record.keyLightColorLinear, record.sunIntensity * Math.pow(cosine, sharpExponent) * sharpEnergy + 0.22 * (record.sunIntensity / 2.5) * Math.pow(cosine, broadExponent) * broadEnergy);
  const accent = scale(record.accentRadianceLinear, 0.13 * Math.pow(1 - Math.abs(direction[1]), record.accentPower * (1 - roughnessSquared) + roughnessSquared));
  return scale(evaluateSvoEnvironmentCloudRadiance(add(add(sky, sun), accent), direction, roughness, record.cloudCoverage ?? 0, record.lowerRadianceLinear, record.upperRadianceLinear, scale(record.keyLightColorLinear, record.sunIntensity), record.keyLightDirection), record.specularScale);
}

function fnvStep(hash: number, value: number): number {
  return Math.imul((hash ^ value) >>> 0, 0x01000193) >>> 0;
}

export function buildSvoEnvironmentLighting(
  environmentId: EnvironmentId,
  revision = 1,
  options: SvoEnvironmentLightingOptions = {},
): SvoEnvironmentLightingBuild {
  const record = svoEnvironmentLightingRecord(environmentId, revision, options);
  const packedRecord = packSvoEnvironmentLightingRecords([record]);
  let hash = 0x811c9dc5;
  for (const word of packedRecord) for (const shift of [0, 8, 16, 24]) hash = fnvStep(hash, (word >>> shift) & 0xff);
  const contentRevision = hash.toString(16).padStart(8, "0");
  return { record, packedRecord, contentRevision, cacheKey: `svo-environment-lighting-v${SVO_ENVIRONMENT_LIGHTING_VERSION}:${environmentId}:${contentRevision}` };
}

/**
 * The environment's own ambient radiance, with no geometry in it. This is what
 * a frame should show while the sparse world is still being published: the room
 * has not resolved yet, but the light in it is already the authored light, so
 * the first published frame arrives as a reveal rather than a cut from a
 * different scene.
 */
export function svoEnvironmentAmbientBackgroundLinear(
  environmentId: EnvironmentId,
  options: SvoEnvironmentLightingOptions = {},
): readonly [number, number, number] {
  const record = svoEnvironmentLightingRecord(environmentId, 1, options);
  const { lowerRadianceLinear: lower, upperRadianceLinear: upper, diffuseScale } = record;
  const blend = (channel: 0 | 1 | 2) => Math.max(0, 0.5 * (lower[channel] + upper[channel]) * diffuseScale);
  return Object.freeze([blend(0), blend(1), blend(2)] as const);
}

function cloudHash(x: number, y: number): number {
  let h = (Math.imul(x, 374761393) + Math.imul(y, 668265263)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 4294967295;
}
function cloudNoise(x: number, y: number): number {
  const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy;
  const u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy);
  const a = cloudHash(ix, iy), b = cloudHash(ix + 1, iy), c = cloudHash(ix, iy + 1), d = cloudHash(ix + 1, iy + 1);
  return (a + (b - a) * u) * (1 - v) + (c + (d - c) * u) * v;
}
function smoothCloud(a: number, b: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t);
}
/** Deterministic distant cloud layer shared by primary sky and reflections.
 * Roughness suppresses directional detail; this is an analytic approximation,
 * not a sampled physical atmosphere or an exact spherical convolution. */
export function evaluateSvoEnvironmentCloudRadiance(base: LinearRgb, direction: SvoVec3, roughness: number,
  coverage: number, lower: LinearRgb, upper: LinearRgb, keyColor: LinearRgb, keyDirection: SvoVec3): LinearRgb {
  if (coverage <= 0) return base;
  let x = direction[0] * 1.2 / (Math.max(direction[1], 0) + .18) + 17.3;
  let y = direction[2] * 1.2 / (Math.max(direction[1], 0) + .18) - 9.2;
  let noise = 0, amplitude = .5079365079;
  for (let octave = 0; octave < 6; octave++) { noise += cloudNoise(x, y) * amplitude; x = x * 2.03 + 11.7; y = y * 2.03 - 5.2; amplitude *= .5; }
  const edge = 1 - coverage * .85;
  const density = smoothCloud(edge - .18, edge + .12, noise);
  const detail = density * (1 - roughness * roughness) + coverage * .65 * roughness * roughness;
  const amount = detail * smoothCloud(-.03, .14, direction[1]);
  // Dense cloud undersides attenuate the sky fill; thinner edges scatter the
  // authored sun. A constant cloud color collapsed overcast skies to a flat
  // gradient when that color was close to the underlying hemisphere.
  // Keep optical thickness continuous after silhouette opacity saturates.
  // Otherwise broad, fully cloudy regions become uniform flat patches.
  const thickness = Math.max(0, noise - edge + .12) * (1 - roughness * roughness)
    + coverage * .3 * roughness * roughness;
  const bodyShade = .18 + 1.05 * Math.exp(-3.5 * thickness);
  const silver = Math.pow(Math.max(0, dot(direction, keyDirection)), 8)
    * (.03 + .22 * (1 - detail) * (1 - roughness * roughness));
  // Broad sunlight scattered by the cloud body remains visible away from the
  // solar disc. Without it, sunny clouds inherit only the blue sky fill and
  // disappear against it. Zero-key overcast lighting remains unchanged.
  const scatteredSun = .45 * Math.max(0, keyDirection[1]) * bodyShade;
  const cloud = add(scale(mix(scale(lower, .6), scale(upper, 1.15), .65), bodyShade), scale(keyColor, silver + scatteredSun));
  return mix(base, cloud, amount * .94);
}

/** Binding-free cloud evaluator, reused by the liquid sky fallback. */
export const svoEnvironmentCloudsWGSL = /* wgsl */ `
fn svoCloudHash(p:vec2i)->f32{var h=bitcast<u32>(p.x)*374761393u+bitcast<u32>(p.y)*668265263u;h=(h^(h>>13u))*1274126177u;return f32(h^(h>>16u))/4294967295.0;}
fn svoCloudNoise(p:vec2f)->f32{let i=vec2i(floor(p));let f=fract(p);let u=f*f*(vec2f(3.0)-2.0*f);return mix(mix(svoCloudHash(i),svoCloudHash(i+vec2i(1,0)),u.x),mix(svoCloudHash(i+vec2i(0,1)),svoCloudHash(i+vec2i(1,1)),u.x),u.y);}
fn svoEnvironmentCloudRadiance(baseRadiance:vec3f,direction:vec3f,roughness:f32,coverage:f32,lower:vec3f,upper:vec3f,keyColor:vec3f,keyDirection:vec3f)->vec3f{
  if(coverage<=0.0){return baseRadiance;}
  var p=direction.xz*(1.2/(max(direction.y,0.0)+.18))+vec2f(17.3,-9.2);var noise=0.0;var amplitude=.5079365079;
  for(var octave=0u;octave<6u;octave+=1u){noise+=svoCloudNoise(p)*amplitude;p=p*2.03+vec2f(11.7,-5.2);amplitude*=.5;}
  let edge=1.0-coverage*.85;let density=smoothstep(edge-.18,edge+.12,noise);let detail=mix(density,coverage*.65,roughness*roughness);let amount=detail*smoothstep(-.03,.14,direction.y);
  let thickness=mix(max(0.0,noise-edge+.12),coverage*.3,roughness*roughness);let bodyShade=.18+1.05*exp(-3.5*thickness);let silver=pow(max(0.0,dot(direction,keyDirection)),8.0)*(.03+.22*(1.0-detail)*(1.0-roughness*roughness));let scatteredSun=.45*max(0.0,keyDirection.y)*bodyShade;let cloud=mix(lower*.6,upper*1.15,.65)*bodyShade+keyColor*(silver+scatteredSun);return mix(baseRadiance,cloud,amount*.94);
}
`;

/** Binding-free mirror for one selected environment record. */
export const svoEnvironmentLightingWGSL = /* wgsl */ `
${svoEnvironmentCloudsWGSL}
struct SvoEnvironmentLightingRecord{lowerDiffuse:vec4f,upperSpecular:vec4f,accentPower:vec4f,keyColorIntensity:vec4f,keyDirectionSharpness:vec4f,identity:vec4u}
fn svoEnvironmentDiffuseIrradiance(lighting:SvoEnvironmentLightingRecord,worldNormalIn:vec3f)->vec3f{let worldNormal=normalize(worldNormalIn);let average=mix(lighting.lowerDiffuse.xyz,lighting.upperSpecular.xyz,.5);let gradient=(lighting.upperSpecular.xyz-lighting.lowerDiffuse.xyz)*(worldNormal.y/3.0);let horizon=lighting.accentPower.xyz*(.13*(1.0-abs(worldNormal.y)));return max(vec3f(0.0),(average+gradient+horizon)*(3.141592653589793*lighting.lowerDiffuse.w));}
fn svoEnvironmentPrefilteredSpecular(lighting:SvoEnvironmentLightingRecord,worldDirectionIn:vec3f,roughnessIn:f32)->vec3f{let worldDirection=normalize(worldDirectionIn);let roughness=clamp(roughnessIn,0.0,1.0);let r2=roughness*roughness;let vertical=clamp(worldDirection.y*.5+.5,0.0,1.0);let average=mix(lighting.lowerDiffuse.xyz,lighting.upperSpecular.xyz,.5);let sky=mix(mix(lighting.lowerDiffuse.xyz,lighting.upperSpecular.xyz,vertical),average,r2);let cosine=max(0.0,dot(worldDirection,normalize(lighting.keyDirectionSharpness.xyz)));let sharpExponent=mix(lighting.keyDirectionSharpness.w,1.0,r2);let broadExponent=mix(15.0,1.0,r2);let sharpEnergy=(sharpExponent+2.0)/(lighting.keyDirectionSharpness.w+2.0);let broadEnergy=(broadExponent+2.0)/17.0;let sun=lighting.keyColorIntensity.xyz*(lighting.keyColorIntensity.w*pow(cosine,sharpExponent)*sharpEnergy+.22*(lighting.keyColorIntensity.w/2.5)*pow(cosine,broadExponent)*broadEnergy);let accentExponent=mix(lighting.accentPower.w,1.0,r2);let accent=lighting.accentPower.xyz*(.13*pow(1.0-abs(worldDirection.y),accentExponent));return max(vec3f(0.0),svoEnvironmentCloudRadiance(sky+sun+accent,worldDirection,roughness,bitcast<f32>(lighting.identity.w),lighting.lowerDiffuse.xyz,lighting.upperSpecular.xyz,lighting.keyColorIntensity.xyz*lighting.keyColorIntensity.w,normalize(lighting.keyDirectionSharpness.xyz))*lighting.upperSpecular.w);}
`;
