/** Linear RGB tuple used by the CPU side of the shared GPU material contract. */
export type LinearRgb = readonly [red: number, green: number, blue: number];

/**
 * Renderer-independent material inputs for the unified WebGPU lighting model.
 *
 * The current scene stores these values in several different GPU records. The
 * WGSL contract deliberately operates on a local value instead of prescribing
 * a bind-group layout, so raster bodies, optical interfaces, and sparse voxel
 * materials can all construct the same closure from their existing buffers.
 */
export interface UnifiedLightingMaterial {
  baseColorLinear: LinearRgb;
  emissiveLinear: LinearRgb;
  roughness: number;
  /** Zero is a dielectric; one uses base color as the conductor F0. */
  metallic?: number;
  specularF0Linear: LinearRgb;
  specularWeight: number;
  ambientDiffuse: number;
  rimColorLinear: LinearRgb;
  rimWeight: number;
}

export interface UnifiedLightingSample {
  shadingNormal: LinearRgb;
  /** Defaults to shadingNormal for legacy analytic and raster surfaces. */
  geometricNormal?: LinearRgb;
  towardViewer: LinearRgb;
  towardLight: LinearRgb;
  lightColorLinear: LinearRgb;
}

export const WATER_OPTICS = Object.freeze({
  indexOfRefraction: 1.333,
  fresnelF0: 0.02037,
  absorption: [0.45, 0.09, 0.06] as LinearRgb,
  scatter: [0.012, 0.055, 0.049] as LinearRgb
});

export const GLASS_OPTICS = Object.freeze({
  indexOfRefraction: 1.5,
  fresnelF0: 0.04,
  tint: [0.30, 0.58, 0.54] as LinearRgb
});

/** CPU mirror used in tests and tooling that author optical materials. */
export function dielectricFresnel(cosine: number, f0: number) {
  const c = Math.min(1, Math.max(0, cosine));
  const base = Math.min(1, Math.max(0, f0));
  return base + (1 - base) * (1 - c) ** 5;
}

/** CPU mirror of the shared absorption closure. */
export function beerLambert(absorption: LinearRgb, distance: number): [number, number, number] {
  const d = Math.max(0, distance);
  return [Math.exp(-Math.max(0, absorption[0]) * d), Math.exp(-Math.max(0, absorption[1]) * d), Math.exp(-Math.max(0, absorption[2]) * d)];
}

/**
 * The single final-output transform shared by raster and sparse-voxel frames.
 * Lighting and media code must keep values scene-linear and call this only
 * when writing the presentation target.
 */
export function sceneLinearToDisplay(sceneLinear: LinearRgb): [number, number, number] {
  return sceneLinear.map((channel) => {
    const nonNegative = Math.max(0, Number.isFinite(channel) ? channel : 0);
    return (nonNegative / (nonNegative + 1)) ** (1 / 2.2);
  }) as [number, number, number];
}

/** Binding-free WGSL mirror of `sceneLinearToDisplay`. */
export const unifiedDisplayTransferShaderLibrary = /* wgsl */ `
fn unifiedDisplayTransfer(sceneLinear: vec3f) -> vec3f {
  let nonNegative = max(sceneLinear, vec3f(0.0));
  let toneMapped = nonNegative / (nonNegative + vec3f(1.0));
  return pow(toneMapped, vec3f(1.0 / 2.2));
}
`;

const PBR_PI = Math.PI;
const PBR_EPSILON = 1e-8;

function saturate(value: number): number {
  return Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
}

function canonicalColor(value: LinearRgb, maximum = Number.POSITIVE_INFINITY): [number, number, number] {
  return value.map((channel) => Math.min(maximum, Math.max(0, Number.isFinite(channel) ? channel : 0))) as [number, number, number];
}

function dot3(left: LinearRgb, right: LinearRgb): number {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

function normalize3(value: LinearRgb, fallback: LinearRgb): [number, number, number] {
  const magnitude = Math.hypot(value[0], value[1], value[2]);
  if (Number.isFinite(magnitude) && magnitude > PBR_EPSILON) return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
  const fallbackMagnitude = Math.hypot(fallback[0], fallback[1], fallback[2]);
  if (Number.isFinite(fallbackMagnitude) && fallbackMagnitude > PBR_EPSILON) return [fallback[0] / fallbackMagnitude, fallback[1] / fallbackMagnitude, fallback[2] / fallbackMagnitude];
  return [0, 1, 0];
}

/** RGB Schlick Fresnel mirror used by the quantitative lighting tests. */
export function schlickFresnel(cosine: number, f0: LinearRgb): [number, number, number] {
  const grazing = (1 - saturate(cosine)) ** 5;
  const base = canonicalColor(f0, 1);
  return base.map((channel) => channel + (1 - channel) * grazing) as [number, number, number];
}

/** Trowbridge-Reitz/GGX normal-distribution function. */
export function ggxNormalDistribution(nDotH: number, roughness: number): number {
  const perceptualRoughness = Math.min(1, Math.max(0.04, Number.isFinite(roughness) ? roughness : 1));
  const alpha = perceptualRoughness * perceptualRoughness;
  const alphaSquared = alpha * alpha;
  const cosine = saturate(nDotH);
  const denominator = cosine * cosine * (alphaSquared - 1) + 1;
  return alphaSquared / Math.max(PBR_PI * denominator * denominator, PBR_EPSILON);
}

/** Height-correlated Smith GGX visibility, equal to G/(4 NdotL NdotV). */
export function smithGgxVisibility(nDotV: number, nDotL: number, roughness: number): number {
  const view = saturate(nDotV);
  const light = saturate(nDotL);
  if (view <= 0 || light <= 0) return 0;
  const perceptualRoughness = Math.min(1, Math.max(0.04, Number.isFinite(roughness) ? roughness : 1));
  const alpha = perceptualRoughness * perceptualRoughness;
  const alphaSquared = alpha * alpha;
  const viewTerm = light * Math.sqrt(view * view * (1 - alphaSquared) + alphaSquared);
  const lightTerm = view * Math.sqrt(light * light * (1 - alphaSquared) + alphaSquared);
  return 0.5 / Math.max(viewTerm + lightTerm, PBR_EPSILON);
}

/**
 * CPU mirror of `shadeUnifiedSurface`. The direct-light result is a BRDF times
 * incident radiance and the geometric projected cosine. Ambient, rim, and
 * emissive are retained for compatibility with the existing authored scenes.
 */
export function evaluateUnifiedLighting(
  materialInput: UnifiedLightingMaterial,
  input: UnifiedLightingSample,
): [number, number, number] {
  const baseColor = canonicalColor(materialInput.baseColorLinear, 1);
  const emissive = canonicalColor(materialInput.emissiveLinear);
  const f0Dielectric = canonicalColor(materialInput.specularF0Linear, 1);
  const lightColor = canonicalColor(input.lightColorLinear);
  const roughness = Math.min(1, Math.max(0.04, Number.isFinite(materialInput.roughness) ? materialInput.roughness : 1));
  const metallic = saturate(materialInput.metallic ?? 0);
  const specularWeight = saturate(materialInput.specularWeight);
  const ambientDiffuse = saturate(materialInput.ambientDiffuse);
  const rimColor = canonicalColor(materialInput.rimColorLinear);
  const rimWeight = saturate(materialInput.rimWeight);

  const towardViewer = normalize3(input.towardViewer, [0, 0, 1]);
  let geometricNormal = normalize3(input.geometricNormal ?? input.shadingNormal, [0, 1, 0]);
  if (dot3(geometricNormal, towardViewer) < 0) geometricNormal = geometricNormal.map((value) => -value) as [number, number, number];
  let shadingNormal = normalize3(input.shadingNormal, geometricNormal);
  if (dot3(shadingNormal, geometricNormal) < 0) shadingNormal = shadingNormal.map((value) => -value) as [number, number, number];
  const towardLight = normalize3(input.towardLight, geometricNormal);
  const halfVector = normalize3([
    towardViewer[0] + towardLight[0],
    towardViewer[1] + towardLight[1],
    towardViewer[2] + towardLight[2],
  ], shadingNormal);

  const gDotV = Math.max(0, dot3(geometricNormal, towardViewer));
  const gDotL = Math.max(0, dot3(geometricNormal, towardLight));
  const nDotV = Math.max(0, dot3(shadingNormal, towardViewer));
  const nDotL = Math.max(0, dot3(shadingNormal, towardLight));
  const nDotH = Math.max(0, dot3(shadingNormal, halfVector));
  const lDotH = Math.max(0, dot3(towardLight, halfVector));
  const f0 = baseColor.map((channel, index) =>
    f0Dielectric[index] * specularWeight * (1 - metallic) + channel * metallic) as [number, number, number];
  const fresnel = schlickFresnel(lDotH, f0);
  const distribution = ggxNormalDistribution(nDotH, roughness);
  const visibility = smithGgxVisibility(nDotV, nDotL, roughness);
  const shadingCorrection = nDotV > PBR_EPSILON && nDotL > PBR_EPSILON && gDotV > PBR_EPSILON
    ? Math.min(1, Math.max(0, (gDotL * nDotV) / Math.max(nDotL * gDotV, PBR_EPSILON)))
    : 0;
  const directEnabled = gDotL > 0 && gDotV > 0 ? 1 : 0;
  const rimFresnel = schlickFresnel(nDotV, f0);

  return baseColor.map((channel, index) => {
    const diffuse = channel * (1 - metallic) * (1 - fresnel[index]) / PBR_PI;
    const specular = distribution * visibility * fresnel[index];
    const direct = (diffuse + specular) * lightColor[index] * gDotL * shadingCorrection * directEnabled;
    const ambient = channel * (1 - metallic) * ambientDiffuse;
    const rim = rimColor[index] * rimWeight * (1 - rimFresnel[index]) * (1 - nDotV) ** 3;
    return Math.max(0, direct + ambient + rim + emissive[index]);
  }) as [number, number, number];
}

/**
 * Canonical WGSL lighting/material library.
 *
 * Required convention:
 * - all colors are scene-linear RGB;
 * - normal, towardViewer, and towardLight point away from the surface;
 * - lightColor is un-tonemapped radiance;
 * - display transfer remains the responsibility of the final output pass.
 *
 * The library has no resource bindings and no dependency on the environment
 * shader. Consumers supply the active environment's light direction/color.
 */
export const unifiedLightingShaderLibrary = /* wgsl */ `
struct UnifiedLightingMaterial {
  baseColor: vec3f,
  roughness: f32,
  emissive: vec3f,
  ambientDiffuse: f32,
  specularF0: vec3f,
  specularWeight: f32,
  metallic: f32,
  rimColor: vec3f,
  rimWeight: f32,
}

struct UnifiedLightingInput {
  normal: vec3f,
  geometricNormal: vec3f,
  towardViewer: vec3f,
  towardLight: vec3f,
  lightColor: vec3f,
}

const UNIFIED_PI: f32 = 3.141592653589793;
const UNIFIED_EPSILON: f32 = 1e-6;

fn unifiedSafeNormal(value: vec3f, fallback: vec3f) -> vec3f {
  let magnitudeSquared = dot(value, value);
  if (magnitudeSquared > UNIFIED_EPSILON * UNIFIED_EPSILON) { return value * inverseSqrt(magnitudeSquared); }
  let fallbackSquared = dot(fallback, fallback);
  if (fallbackSquared > UNIFIED_EPSILON * UNIFIED_EPSILON) { return fallback * inverseSqrt(fallbackSquared); }
  return vec3f(0.0, 1.0, 0.0);
}

fn unifiedPbrMaterial(
  baseColor: vec3f,
  metallic: f32,
  roughness: f32,
  emissive: vec3f,
  ambientDiffuse: f32,
  specularF0: vec3f,
  specularWeight: f32,
  rimColor: vec3f,
  rimWeight: f32
) -> UnifiedLightingMaterial {
  return UnifiedLightingMaterial(
    clamp(baseColor, vec3f(0.0), vec3f(1.0)),
    clamp(roughness, 0.04, 1.0),
    max(emissive, vec3f(0.0)),
    clamp(ambientDiffuse, 0.0, 1.0),
    clamp(specularF0, vec3f(0.0), vec3f(1.0)),
    clamp(specularWeight, 0.0, 1.0),
    clamp(metallic, 0.0, 1.0),
    max(rimColor, vec3f(0.0)),
    clamp(rimWeight, 0.0, 1.0)
  );
}

fn unifiedMaterial(
  baseColor: vec3f,
  roughness: f32,
  emissive: vec3f,
  ambientDiffuse: f32,
  specularF0: vec3f,
  specularWeight: f32,
  rimColor: vec3f,
  rimWeight: f32
) -> UnifiedLightingMaterial {
  return unifiedPbrMaterial(baseColor, 0.0, roughness, emissive, ambientDiffuse, specularF0, specularWeight, rimColor, rimWeight);
}

fn unifiedLightingInput(normal: vec3f, towardViewer: vec3f, towardLight: vec3f, lightColor: vec3f) -> UnifiedLightingInput {
  return unifiedLightingInputWithGeometry(normal, normal, towardViewer, towardLight, lightColor);
}

fn unifiedLightingInputWithGeometry(shadingNormal: vec3f, geometricNormal: vec3f, towardViewer: vec3f, towardLight: vec3f, lightColor: vec3f) -> UnifiedLightingInput {
  let view = unifiedSafeNormal(towardViewer, vec3f(0.0, 0.0, 1.0));
  var geometry = unifiedSafeNormal(geometricNormal, vec3f(0.0, 1.0, 0.0));
  if (dot(geometry, view) < 0.0) { geometry = -geometry; }
  var shading = unifiedSafeNormal(shadingNormal, geometry);
  if (dot(shading, geometry) < 0.0) { shading = -shading; }
  return UnifiedLightingInput(shading, geometry, view, unifiedSafeNormal(towardLight, geometry), max(lightColor, vec3f(0.0)));
}

fn unifiedSchlick(cosine: f32, f0: vec3f) -> vec3f {
  let grazing = pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
  return f0 + (vec3f(1.0) - f0) * grazing;
}

fn unifiedDielectricFresnel(cosine: f32, f0: f32) -> f32 {
  return unifiedSchlick(cosine, vec3f(clamp(f0, 0.0, 1.0))).x;
}

fn unifiedGgxDistribution(nDotH: f32, roughness: f32) -> f32 {
  let alpha = roughness * roughness;
  let alphaSquared = alpha * alpha;
  let cosine = clamp(nDotH, 0.0, 1.0);
  let denominator = cosine * cosine * (alphaSquared - 1.0) + 1.0;
  return alphaSquared / max(UNIFIED_PI * denominator * denominator, UNIFIED_EPSILON);
}

fn unifiedSmithGgxVisibility(nDotV: f32, nDotL: f32, roughness: f32) -> f32 {
  let view = clamp(nDotV, 0.0, 1.0);
  let light = clamp(nDotL, 0.0, 1.0);
  if (view <= 0.0 || light <= 0.0) { return 0.0; }
  let alpha = roughness * roughness;
  let alphaSquared = alpha * alpha;
  let viewTerm = light * sqrt(view * view * (1.0 - alphaSquared) + alphaSquared);
  let lightTerm = view * sqrt(light * light * (1.0 - alphaSquared) + alphaSquared);
  return 0.5 / max(viewTerm + lightTerm, UNIFIED_EPSILON);
}

fn unifiedSpecularLobe(normal: vec3f, towardViewer: vec3f, towardLight: vec3f, exponent: f32) -> f32 {
  let reflected = reflect(-normalize(towardViewer), normalize(normal));
  return pow(max(dot(reflected, normalize(towardLight)), 0.0), max(exponent, 1.0));
}

fn shadeUnifiedSurface(material: UnifiedLightingMaterial, input: UnifiedLightingInput) -> vec3f {
  let geometricDotV = max(dot(input.geometricNormal, input.towardViewer), 0.0);
  let geometricDotL = max(dot(input.geometricNormal, input.towardLight), 0.0);
  let nDotV = max(dot(input.normal, input.towardViewer), 0.0);
  let nDotL = max(dot(input.normal, input.towardLight), 0.0);
  let halfVector = unifiedSafeNormal(input.towardViewer + input.towardLight, input.normal);
  let nDotH = max(dot(input.normal, halfVector), 0.0);
  let lDotH = max(dot(input.towardLight, halfVector), 0.0);
  let dielectricF0 = material.specularF0 * material.specularWeight;
  let f0 = mix(dielectricF0, material.baseColor, material.metallic);
  let fresnel = unifiedSchlick(lDotH, f0);
  let diffuseColor = material.baseColor * (1.0 - material.metallic);
  let diffuse = diffuseColor * (vec3f(1.0) - fresnel) / UNIFIED_PI;
  let distribution = unifiedGgxDistribution(nDotH, material.roughness);
  let visibility = unifiedSmithGgxVisibility(nDotV, nDotL, material.roughness);
  let specular = distribution * visibility * fresnel;
  var correction = 0.0;
  if (nDotL > UNIFIED_EPSILON && nDotV > UNIFIED_EPSILON && geometricDotV > UNIFIED_EPSILON) {
    correction = clamp((geometricDotL * nDotV) / max(nDotL * geometricDotV, UNIFIED_EPSILON), 0.0, 1.0);
  }
  let direct = (diffuse + specular) * input.lightColor * geometricDotL * correction;
  let ambient = diffuseColor * material.ambientDiffuse;
  let rimRemaining = vec3f(1.0) - unifiedSchlick(nDotV, f0);
  let rim = material.rimColor * material.rimWeight * rimRemaining * pow(1.0 - nDotV, 3.0);
  return max(direct + ambient + rim + material.emissive, vec3f(0.0));
}

fn unifiedBeerLambert(absorption: vec3f, distance: f32) -> vec3f {
  return exp(-max(absorption, vec3f(0.0)) * max(distance, 0.0));
}

fn unifiedAbsorbingTransmission(
  incident: vec3f,
  absorption: vec3f,
  scatterColor: vec3f,
  distance: f32
) -> vec3f {
  let transmission = unifiedBeerLambert(absorption, distance);
  return incident * transmission + max(scatterColor, vec3f(0.0)) * (vec3f(1.0) - transmission);
}
`;
