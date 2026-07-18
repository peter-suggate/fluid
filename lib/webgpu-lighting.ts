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
  specularF0Linear: LinearRgb;
  specularWeight: number;
  ambientDiffuse: number;
  rimColorLinear: LinearRgb;
  rimWeight: number;
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
  rimColor: vec3f,
  rimWeight: f32,
}

struct UnifiedLightingInput {
  normal: vec3f,
  towardViewer: vec3f,
  towardLight: vec3f,
  lightColor: vec3f,
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
  return UnifiedLightingMaterial(
    max(baseColor, vec3f(0.0)),
    clamp(roughness, 0.04, 1.0),
    max(emissive, vec3f(0.0)),
    clamp(ambientDiffuse, 0.0, 1.0),
    clamp(specularF0, vec3f(0.0), vec3f(1.0)),
    max(specularWeight, 0.0),
    max(rimColor, vec3f(0.0)),
    max(rimWeight, 0.0)
  );
}

fn unifiedLightingInput(normal: vec3f, towardViewer: vec3f, towardLight: vec3f, lightColor: vec3f) -> UnifiedLightingInput {
  return UnifiedLightingInput(normalize(normal), normalize(towardViewer), normalize(towardLight), max(lightColor, vec3f(0.0)));
}

fn unifiedSchlick(cosine: f32, f0: vec3f) -> vec3f {
  let grazing = pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
  return f0 + (vec3f(1.0) - f0) * grazing;
}

fn unifiedDielectricFresnel(cosine: f32, f0: f32) -> f32 {
  return unifiedSchlick(cosine, vec3f(clamp(f0, 0.0, 1.0))).x;
}

fn unifiedSpecularLobe(normal: vec3f, towardViewer: vec3f, towardLight: vec3f, exponent: f32) -> f32 {
  let reflected = reflect(-normalize(towardViewer), normalize(normal));
  return pow(max(dot(reflected, normalize(towardLight)), 0.0), max(exponent, 1.0));
}

fn shadeUnifiedSurface(material: UnifiedLightingMaterial, input: UnifiedLightingInput) -> vec3f {
  let nDotL = max(dot(input.normal, input.towardLight), 0.0);
  let nDotV = max(dot(input.normal, input.towardViewer), 0.0);
  let diffuse = material.baseColor * mix(material.ambientDiffuse, 1.0, nDotL);
  let exponent = mix(128.0, 4.0, material.roughness);
  let fresnel = unifiedSchlick(nDotV, material.specularF0);
  let specular = input.lightColor * fresnel * unifiedSpecularLobe(input.normal, input.towardViewer, input.towardLight, exponent) * material.specularWeight;
  let rim = material.rimColor * material.rimWeight * pow(1.0 - nDotV, 3.0);
  return diffuse + specular + rim + material.emissive;
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
