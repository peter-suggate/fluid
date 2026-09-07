import { svoNodeMipCoverageOpacity, svoNodeMipSamplingWGSL } from "./svo-node-mip-sampling";
import {
  evaluateSvoTetrahedralRadiance,
  svoTetrahedralRadianceWGSL,
  type SvoRadianceRgb,
  type SvoTetrahedralRadiance,
} from "./svo-tetrahedral-radiance";

export const SVO_TETRAHEDRAL_RADIANCE_CONE_MAXIMUM_STEPS = 64;
export const SVO_TETRAHEDRAL_RADIANCE_UNPREMULTIPLY_EPSILON = 1 / 255;

export interface SvoTetrahedralRadianceCone {
  origin_m: readonly [number, number, number];
  direction: readonly [number, number, number];
  aperture_radians: number;
  minimumVoxelWidth_m: number;
  maximumDistance_m: number;
  maximumSteps?: number;
  opacityCutoff?: number;
  /** Normalized mean coverage below this value does not get unpremultiplied. */
  unpremultiplyCoverageEpsilon?: number;
  /** Missing opacity blocks and terminates the cone. Defaults to true. */
  failClosedOpacity?: boolean;
}

export interface SvoTetrahedralRadianceConeQuery {
  position_m: readonly [number, number, number];
  /** Fractional node-mip LOD selected from the cone diameter. */
  lod: number;
  diameter_m: number;
  distance_m: number;
  /** Direction from the sampled voxel back toward the cone receiver. */
  outgoingDirection: readonly [number, number, number];
}

export interface SvoTetrahedralRadianceConeSample {
  /** Mean coverage byte from the authoritative opacity atlas; absent means invalid. */
  coverage?: number;
  /** Four coverage-premultiplied directional samples; absent is valid black radiance. */
  radiance?: SvoTetrahedralRadiance;
}

export type SvoTetrahedralRadianceConeSampler = (
  query: SvoTetrahedralRadianceConeQuery,
) => SvoTetrahedralRadianceConeSample | undefined;

export type SvoTetrahedralRadianceConeTermination = "distance" | "opacity" | "step-limit" | "invalid-opacity";

export interface SvoTetrahedralRadianceConeResult {
  radiance: SvoRadianceRgb;
  opacity: number;
  transmittance: number;
  valid: boolean;
  termination: SvoTetrahedralRadianceConeTermination;
  /** Spatial cone samples requested from the sparse source. */
  coneTaps: number;
  opacityTaps: number;
  /** Spatial samples whose four radiance lobes were resident. */
  radianceSamples: number;
  /** Actual filterable radiance texture taps implied by radianceSamples. */
  radianceTextureTaps: number;
  missingOpacitySamples: number;
  missingRadianceSamples: number;
  unpremultiplyRejectedSamples: number;
}

export interface SvoDiffuseHemisphereConeSample {
  direction: readonly [number, number, number];
  /** Normalized cosine-weighted quadrature weight. The samples sum to one. */
  weight: number;
}

function normalized(value: readonly [number, number, number], label: string): readonly [number, number, number] {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new RangeError(`${label} must be non-zero`);
  return [value[0] / length, value[1] / length, value[2] / length];
}

function cross(left: readonly number[], right: readonly number[]): [number, number, number] {
  return [
    left[1] * right[2] - left[2] * right[1],
    left[2] * right[0] - left[0] * right[2],
    left[0] * right[1] - left[1] * right[0],
  ];
}

function validateCone(cone: SvoTetrahedralRadianceCone): Required<Pick<SvoTetrahedralRadianceCone,
  "maximumSteps" | "opacityCutoff" | "unpremultiplyCoverageEpsilon" | "failClosedOpacity">> {
  if (cone.origin_m.length !== 3 || cone.origin_m.some((component) => !Number.isFinite(component))) {
    throw new RangeError("SVO radiance cone origin must contain three finite components");
  }
  normalized(cone.direction, "SVO radiance cone direction");
  if (!Number.isFinite(cone.aperture_radians) || cone.aperture_radians < 0 || cone.aperture_radians >= Math.PI) {
    throw new RangeError("SVO radiance cone aperture is invalid");
  }
  if (!Number.isFinite(cone.minimumVoxelWidth_m) || cone.minimumVoxelWidth_m <= 0) {
    throw new RangeError("SVO radiance cone minimum voxel width must be positive");
  }
  if (!Number.isFinite(cone.maximumDistance_m) || cone.maximumDistance_m < 0) {
    throw new RangeError("SVO radiance cone maximum distance must be non-negative");
  }
  const maximumSteps = cone.maximumSteps ?? SVO_TETRAHEDRAL_RADIANCE_CONE_MAXIMUM_STEPS;
  if (!Number.isInteger(maximumSteps) || maximumSteps < 1 || maximumSteps > SVO_TETRAHEDRAL_RADIANCE_CONE_MAXIMUM_STEPS) {
    throw new RangeError(`SVO radiance cone maximum steps must be from 1 to ${SVO_TETRAHEDRAL_RADIANCE_CONE_MAXIMUM_STEPS}`);
  }
  const opacityCutoff = cone.opacityCutoff ?? 0.995;
  if (!Number.isFinite(opacityCutoff) || opacityCutoff <= 0 || opacityCutoff > 1) {
    throw new RangeError("SVO radiance cone opacity cutoff must be in (0, 1]");
  }
  const unpremultiplyCoverageEpsilon = cone.unpremultiplyCoverageEpsilon
    ?? SVO_TETRAHEDRAL_RADIANCE_UNPREMULTIPLY_EPSILON;
  if (!Number.isFinite(unpremultiplyCoverageEpsilon)
    || unpremultiplyCoverageEpsilon < 0 || unpremultiplyCoverageEpsilon > 1) {
    throw new RangeError("SVO radiance cone unpremultiply coverage epsilon must be from zero to one");
  }
  return { maximumSteps, opacityCutoff, unpremultiplyCoverageEpsilon, failClosedOpacity: cone.failClosedOpacity ?? true };
}

/**
 * Three- or four-cone cosine-hemisphere quadrature. Both patterns match the
 * cosine hemisphere's first axial moment (2/3); four cones add a normal cone.
 */
export function svoDiffuseHemisphereCones(
  normalIn: readonly [number, number, number],
  count: 3 | 4,
  azimuthRotation_radians = 0,
): readonly SvoDiffuseHemisphereConeSample[] {
  const normal = normalized(normalIn, "SVO diffuse hemisphere normal");
  if (count !== 3 && count !== 4) throw new RangeError("SVO diffuse hemisphere requires three or four cones");
  if (!Number.isFinite(azimuthRotation_radians)) throw new RangeError("SVO diffuse hemisphere rotation must be finite");
  const helper = Math.abs(normal[1]) < .999 ? [0, 1, 0] as const : [1, 0, 0] as const;
  const tangent = normalized(cross(helper, normal), "SVO diffuse hemisphere tangent");
  const bitangent = cross(normal, tangent);
  const result: SvoDiffuseHemisphereConeSample[] = [];
  if (count === 4) result.push({ direction: normal, weight: .25 });
  const ringCount = 3;
  const axial = count === 3 ? 2 / 3 : 5 / 9;
  const radial = Math.sqrt(1 - axial * axial);
  const weight = count === 3 ? 1 / 3 : .25;
  for (let index = 0; index < ringCount; index += 1) {
    const angle = azimuthRotation_radians + index * 2 * Math.PI / ringCount;
    const x = radial * Math.cos(angle), y = radial * Math.sin(angle);
    result.push({
      direction: [
        normal[0] * axial + tangent[0] * x + bitangent[0] * y,
        normal[1] * axial + tangent[1] * x + bitangent[1] * y,
        normal[2] * axial + tangent[2] * x + bitangent[2] * y,
      ],
      weight,
    });
  }
  return result;
}

/** CPU oracle for the binding-free WGSL gather. */
export function integrateSvoTetrahedralRadianceCone(
  cone: SvoTetrahedralRadianceCone,
  sample: SvoTetrahedralRadianceConeSampler,
): SvoTetrahedralRadianceConeResult {
  const { maximumSteps, opacityCutoff, unpremultiplyCoverageEpsilon, failClosedOpacity } = validateCone(cone);
  const direction = normalized(cone.direction, "SVO radiance cone direction");
  const tangent = Math.tan(cone.aperture_radians * .5);
  let distance = cone.minimumVoxelWidth_m * .5;
  let transmittance = 1;
  const radiance = [0, 0, 0];
  let coneTaps = 0, opacityTaps = 0, radianceSamples = 0;
  let missingOpacitySamples = 0, missingRadianceSamples = 0, unpremultiplyRejectedSamples = 0;
  let invalidOpacity = false;
  while (distance < cone.maximumDistance_m && coneTaps < maximumSteps && 1 - transmittance < opacityCutoff) {
    const diameter = Math.max(cone.minimumVoxelWidth_m, 2 * distance * tangent);
    const lod = Math.max(0, Math.log2(diameter / cone.minimumVoxelWidth_m));
    const voxelWidth = cone.minimumVoxelWidth_m * 2 ** Math.floor(lod);
    const step = Math.min(Math.max(voxelWidth, diameter * .5), cone.maximumDistance_m - distance);
    const position = cone.origin_m.map((value, axis) => value + direction[axis] * distance) as [number, number, number];
    const outgoingDirection = direction.map((value) => -value) as [number, number, number];
    const value = sample({ position_m: position, lod, diameter_m: diameter, distance_m: distance, outgoingDirection });
    coneTaps += 1;
    if (value?.coverage === undefined) {
      missingOpacitySamples += 1;
      invalidOpacity = true;
      if (failClosedOpacity) {
        transmittance = 0;
        break;
      }
    } else {
      if (!Number.isFinite(value.coverage) || value.coverage < 0 || value.coverage > 255) {
        throw new RangeError("SVO radiance cone coverage must be a byte from zero to 255");
      }
      opacityTaps += 1;
      const coverage = value.coverage / 255;
      const alpha = svoNodeMipCoverageOpacity(value.coverage, step / voxelWidth);
      if (value.radiance === undefined) {
        missingRadianceSamples += 1;
      } else {
        radianceSamples += 1;
        if (coverage > unpremultiplyCoverageEpsilon) {
          const premultiplied = evaluateSvoTetrahedralRadiance(value.radiance, outgoingDirection);
          for (let channel = 0; channel < 3; channel += 1) {
            radiance[channel] += transmittance * alpha * premultiplied[channel] / coverage;
          }
        } else {
          unpremultiplyRejectedSamples += 1;
        }
      }
      transmittance *= 1 - alpha;
    }
    distance += Math.max(step, cone.minimumVoxelWidth_m * .25);
  }
  const opacity = 1 - transmittance;
  const termination: SvoTetrahedralRadianceConeTermination = invalidOpacity && failClosedOpacity
    ? "invalid-opacity"
    : opacity >= opacityCutoff ? "opacity"
      : distance >= cone.maximumDistance_m ? "distance" : "step-limit";
  return {
    radiance: radiance as unknown as SvoRadianceRgb,
    opacity,
    transmittance,
    valid: !invalidOpacity,
    termination,
    coneTaps,
    opacityTaps,
    radianceSamples,
    radianceTextureTaps: radianceSamples * 4,
    missingOpacitySamples,
    missingRadianceSamples,
    unpremultiplyRejectedSamples,
  };
}

/**
 * Binding-free WGSL gather. The embedding shader supplies
 * `svoTetraRadianceConeLoad(query)`, resolving its own sparse directory and
 * returning normalized mean coverage plus premultiplied tetrahedral radiance.
 */
export const svoTetrahedralRadianceConeCoreWGSL = /* wgsl */ `
const SVO_TETRA_RADIANCE_CONE_MAX_STEPS:u32=${SVO_TETRAHEDRAL_RADIANCE_CONE_MAXIMUM_STEPS}u;
const SVO_TETRA_RADIANCE_CONE_TERMINATED_DISTANCE:u32=0u;
const SVO_TETRA_RADIANCE_CONE_TERMINATED_OPACITY:u32=1u;
const SVO_TETRA_RADIANCE_CONE_TERMINATED_STEP_LIMIT:u32=2u;
const SVO_TETRA_RADIANCE_CONE_TERMINATED_INVALID_OPACITY:u32=3u;
struct SvoTetraRadianceConeConfig{
  origin_m:vec3f,direction:vec3f,apertureRadians:f32,minimumVoxelWidth_m:f32,maximumDistance_m:f32,
  maximumSteps:u32,opacityCutoff:f32,unpremultiplyCoverageEpsilon:f32,failClosedOpacity:u32
}
struct SvoTetraRadianceConeQuery{position_m:vec3f,lod:f32,diameter_m:f32,distance_m:f32,outgoingDirection:vec3f}
struct SvoTetraRadianceConeSourceSample{coverage:f32,radiance:SvoTetraRadiance,opacityValid:u32,radianceValid:u32}
struct SvoTetraRadianceConeResult{
  radiance:vec3f,opacity:f32,transmittance:f32,valid:u32,termination:u32,coneTaps:u32,opacityTaps:u32,
  radianceSamples:u32,radianceTextureTaps:u32,missingOpacitySamples:u32,missingRadianceSamples:u32,unpremultiplyRejectedSamples:u32
}
fn svoTetraRadianceConeTrace(config:SvoTetraRadianceConeConfig)->SvoTetraRadianceConeResult{
  let direction=normalize(config.direction);let coneTangent=tan(config.apertureRadians*.5);
  let maximumSteps=min(config.maximumSteps,SVO_TETRA_RADIANCE_CONE_MAX_STEPS);
  let cutoff=clamp(config.opacityCutoff,0.000001,1.0);
  var distance=config.minimumVoxelWidth_m*.5;var transmittance=1.0;var radiance=vec3f(0.0);
  var coneTaps=0u;var opacityTaps=0u;var radianceSamples=0u;var missingOpacitySamples=0u;
  var missingRadianceSamples=0u;var unpremultiplyRejectedSamples=0u;var invalidOpacity=false;
  for(var boundedStep=0u;boundedStep<SVO_TETRA_RADIANCE_CONE_MAX_STEPS;boundedStep+=1u){
    if(distance>=config.maximumDistance_m||coneTaps>=maximumSteps||1.0-transmittance>=cutoff){break;}
    let diameter=max(config.minimumVoxelWidth_m,2.0*distance*coneTangent);
    let lod=svoNodeMipLod(diameter,config.minimumVoxelWidth_m);
    let voxelWidth=config.minimumVoxelWidth_m*exp2(floor(lod));
    let step=min(max(voxelWidth,diameter*.5),config.maximumDistance_m-distance);
    let outgoingDirection=-direction;
    let query=SvoTetraRadianceConeQuery(config.origin_m+direction*distance,lod,diameter,distance,outgoingDirection);
    let value=svoTetraRadianceConeLoad(query);coneTaps+=1u;
    if(value.opacityValid==0u){
      missingOpacitySamples+=1u;invalidOpacity=true;
      if(config.failClosedOpacity!=0u){transmittance=0.0;break;}
    }else{
      opacityTaps+=1u;let coverage=clamp(value.coverage,0.0,1.0);
      let alpha=svoNodeMipCoverageOpacity(coverage,step/voxelWidth);
      if(value.radianceValid==0u){missingRadianceSamples+=1u;}
      else{
        radianceSamples+=1u;
        if(coverage>config.unpremultiplyCoverageEpsilon){
          let premultiplied=svoTetraRadianceAlong(value.radiance,outgoingDirection);
          radiance+=transmittance*alpha*premultiplied/coverage;
        }else{unpremultiplyRejectedSamples+=1u;}
      }
      transmittance*=1.0-alpha;
    }
    distance+=max(step,config.minimumVoxelWidth_m*.25);
  }
  let opacity=1.0-transmittance;var termination=SVO_TETRA_RADIANCE_CONE_TERMINATED_STEP_LIMIT;
  if(invalidOpacity&&config.failClosedOpacity!=0u){termination=SVO_TETRA_RADIANCE_CONE_TERMINATED_INVALID_OPACITY;}
  else if(opacity>=cutoff){termination=SVO_TETRA_RADIANCE_CONE_TERMINATED_OPACITY;}
  else if(distance>=config.maximumDistance_m){termination=SVO_TETRA_RADIANCE_CONE_TERMINATED_DISTANCE;}
  return SvoTetraRadianceConeResult(radiance,opacity,transmittance,select(1u,0u,invalidOpacity),termination,coneTaps,
    opacityTaps,radianceSamples,radianceSamples*4u,missingOpacitySamples,missingRadianceSamples,unpremultiplyRejectedSamples);
}
fn svoTetraRadianceHemisphereWeight(sampleIndex:u32,coneCount:u32)->f32{
  if(coneCount==3u&&sampleIndex<3u){return 0.3333333333333333;}
  if(coneCount==4u&&sampleIndex<4u){return .25;}return 0.0;
}
fn svoTetraRadianceHemisphereDirection(normalIn:vec3f,sampleIndex:u32,coneCount:u32,azimuthRotation:f32)->vec3f{
  let normal=normalize(normalIn);if(coneCount==4u&&sampleIndex==0u){return normal;}
  let helper=select(vec3f(1.0,0.0,0.0),vec3f(0.0,1.0,0.0),abs(normal.y)<.999);
  let tangent=normalize(cross(helper,normal));let bitangent=cross(normal,tangent);
  let ringIndex=select(sampleIndex,sampleIndex-1u,coneCount==4u);
  let axial=select(0.5555555555555556,0.6666666666666666,coneCount==3u);
  let radial=sqrt(max(0.0,1.0-axial*axial));let angle=azimuthRotation+f32(ringIndex)*2.0943951023931953;
  return normal*axial+tangent*(radial*cos(angle))+bitangent*(radial*sin(angle));
}
`;

/** Standalone form used by focused kernels that do not already embed the dependencies. */
export const svoTetrahedralRadianceConeWGSL = /* wgsl */ `
${svoNodeMipSamplingWGSL}
${svoTetrahedralRadianceWGSL}
${svoTetrahedralRadianceConeCoreWGSL}
`;
