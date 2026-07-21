import { SVO_NODE_MIP_LAYOUT, type SvoNodeMipRgba8 } from "./svo-node-mip-pyramid";

export interface SvoNodeMipCone {
  origin_m: readonly [number, number, number];
  direction: readonly [number, number, number];
  aperture_radians: number;
  minimumVoxelWidth_m: number;
  maximumDistance_m: number;
  maximumSteps?: number;
  opacityCutoff?: number;
}

export interface SvoNodeMipConeQuery {
  position_m: readonly [number, number, number];
  lod: number;
  diameter_m: number;
  distance_m: number;
}

export type SvoNodeMipConeSampler = (query: SvoNodeMipConeQuery) => SvoNodeMipRgba8 | undefined;

export interface SvoNodeMipConeResult {
  opacity: number;
  transmittance: number;
  steps: number;
  missingSamples: number;
  terminated: "distance" | "opacity" | "step-limit";
}

function validateCone(cone: SvoNodeMipCone): void {
  if (cone.origin_m.length !== 3 || cone.direction.length !== 3 || [...cone.origin_m, ...cone.direction].some((v) => !Number.isFinite(v))) {
    throw new RangeError("SVO node-mip cone vectors must contain three finite components");
  }
  const directionLength = Math.hypot(...cone.direction);
  if (Math.abs(directionLength - 1) > 1e-4) throw new RangeError("SVO node-mip cone direction must be normalized");
  if (!Number.isFinite(cone.aperture_radians) || cone.aperture_radians < 0 || cone.aperture_radians >= Math.PI) throw new RangeError("SVO node-mip cone aperture is invalid");
  if (!Number.isFinite(cone.minimumVoxelWidth_m) || cone.minimumVoxelWidth_m <= 0) throw new RangeError("SVO node-mip minimum voxel width must be positive");
  if (!Number.isFinite(cone.maximumDistance_m) || cone.maximumDistance_m < 0) throw new RangeError("SVO node-mip maximum distance must be non-negative");
}

/** Converts a byte mean-coverage sample into opacity over a step measured in voxels. */
export function svoNodeMipCoverageOpacity(coverage: number, stepInVoxels: number): number {
  const alpha = Math.max(0, Math.min(1, coverage / 255));
  if (!Number.isFinite(stepInVoxels) || stepInVoxels < 0) throw new RangeError("SVO node-mip step width must be non-negative and finite");
  return 1 - (1 - alpha) ** stepInVoxels;
}

/** CPU reference for front-to-back solid-opacity cone integration and shader parity tests. */
export function integrateSvoNodeMipCone(cone: SvoNodeMipCone, sample: SvoNodeMipConeSampler): SvoNodeMipConeResult {
  validateCone(cone);
  const maximumSteps = cone.maximumSteps ?? 64;
  if (!Number.isInteger(maximumSteps) || maximumSteps < 1) throw new RangeError("SVO node-mip maximum steps must be positive");
  const opacityCutoff = cone.opacityCutoff ?? 0.995;
  if (!Number.isFinite(opacityCutoff) || opacityCutoff <= 0 || opacityCutoff > 1) throw new RangeError("SVO node-mip opacity cutoff must be in (0, 1]");
  const tangent = Math.tan(cone.aperture_radians * 0.5);
  let distance = cone.minimumVoxelWidth_m * 0.5;
  let transmittance = 1;
  let steps = 0, missingSamples = 0;
  while (distance < cone.maximumDistance_m && steps < maximumSteps && 1 - transmittance < opacityCutoff) {
    const diameter = Math.max(cone.minimumVoxelWidth_m, 2 * distance * tangent);
    const lod = Math.max(0, Math.log2(diameter / cone.minimumVoxelWidth_m));
    const voxelWidth = cone.minimumVoxelWidth_m * 2 ** Math.floor(lod);
    const step = Math.min(Math.max(voxelWidth, diameter * 0.5), cone.maximumDistance_m - distance);
    const position = cone.origin_m.map((value, axis) => value + cone.direction[axis] * distance) as [number, number, number];
    const value = sample({ position_m: position, lod, diameter_m: diameter, distance_m: distance });
    if (value) {
      const alpha = svoNodeMipCoverageOpacity(value[0], step / voxelWidth);
      transmittance *= 1 - alpha;
    } else missingSamples += 1;
    distance += Math.max(step, cone.minimumVoxelWidth_m * 0.25);
    steps += 1;
  }
  const opacity = 1 - transmittance;
  return {
    opacity, transmittance, steps, missingSamples,
    terminated: opacity >= opacityCutoff ? "opacity" : distance >= cone.maximumDistance_m ? "distance" : "step-limit",
  };
}

/** Binding-free WGSL. The caller supplies atlas texture/sampler and a directory-resolved page origin. */
export const svoNodeMipSamplingWGSL = /* wgsl */ `
const SVO_NODE_MIP_INTERIOR_SIZE:u32=${SVO_NODE_MIP_LAYOUT.interiorSize}u;
const SVO_NODE_MIP_PHYSICAL_SIZE:u32=${SVO_NODE_MIP_LAYOUT.physicalSize}u;
const SVO_NODE_MIP_APRON:u32=${SVO_NODE_MIP_LAYOUT.apron}u;
struct SvoNodeMipSample{solidMean:f32,solidMaximum:f32,fluidMean:f32,fluidMaximum:f32}
struct SvoNodeMipDirectoryEntry{generation:u32,level:u32,mortonLow:u32,mortonHigh:u32,pageOrigin:vec3u,slot:u32}
fn svoNodeMipDirectoryEntry(directory:texture_2d<u32>,pageIndex:u32)->SvoNodeMipDirectoryEntry{
  let key=textureLoad(directory,vec2u(0u,pageIndex),0);let location=textureLoad(directory,vec2u(1u,pageIndex),0);
  return SvoNodeMipDirectoryEntry(key.x,key.y,key.z,key.w,location.xyz,location.w);
}
fn svoNodeMipAtlasUv(pageOrigin:vec3u,interiorTexel:vec3f,atlasDimensions:vec3u)->vec3f{
  let physical=vec3f(pageOrigin)+vec3f(f32(SVO_NODE_MIP_APRON))+clamp(interiorTexel,vec3f(-.5),vec3f(f32(SVO_NODE_MIP_INTERIOR_SIZE)-.5));
  return (physical+vec3f(.5))/vec3f(atlasDimensions);
}
fn svoNodeMipSamplePage(atlas:texture_3d<f32>,atlasSampler:sampler,pageOrigin:vec3u,interiorTexel:vec3f)->SvoNodeMipSample{
  let lanes=textureSampleLevel(atlas,atlasSampler,svoNodeMipAtlasUv(pageOrigin,interiorTexel,textureDimensions(atlas)),0.0);
  return SvoNodeMipSample(lanes.x,lanes.y,lanes.z,lanes.w);
}
fn svoNodeMipCoverageOpacity(coverage:f32,stepInVoxels:f32)->f32{return 1.0-pow(max(1.0-clamp(coverage,0.0,1.0),0.0),max(stepInVoxels,0.0));}
fn svoNodeMipCompositeOpacity(accumulated:f32,sampleOpacity:f32)->f32{return accumulated+(1.0-accumulated)*clamp(sampleOpacity,0.0,1.0);}
fn svoNodeMipLod(coneDiameter_m:f32,minimumVoxelWidth_m:f32)->f32{return max(0.0,log2(max(coneDiameter_m,minimumVoxelWidth_m)/minimumVoxelWidth_m));}
`;
