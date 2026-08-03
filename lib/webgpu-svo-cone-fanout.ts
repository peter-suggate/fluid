import { SVO_LIGHT_MAXIMUM_RECORDS, svoLightWGSL } from "./svo-light-abi";
import { cameraApertureShaderLibrary } from "./webgpu-camera";
import { SVO_CONTACT_VISIBILITY_CONTRACT } from "./svo-contact-visibility";
import { svoFluidCoverageWGSL } from "./svo-fluid-coverage";
import { svoNodeMipSamplingWGSL } from "./svo-node-mip-sampling";

/**
 * Current-frame cone work is fanned out by deterministic sample, never by
 * time. Four AO samples and two samples for each of eight light slots occupy a
 * fixed 20-layer r32float texture. No history, jitter, or stochastic state is
 * part of this ABI.
 */
export const SVO_CONE_FANOUT_CONTRACT = Object.freeze({
  maximumLights: 8,
  maximumAoSamples: 4,
  maximumLightSamples: 2,
  aoLayerBase: 0,
  lightLayerBase: 4,
  secondaryLightLayerBase: 12,
  layerCount: 4 + 8 * 2,
  receiverFormat: "rgba32float" as GPUTextureFormat,
  temporaryFormat: "r32float" as GPUTextureFormat,
  visibilityFormat: "rg32uint" as GPUTextureFormat,
  workgroupSize: Object.freeze([8, 8, 1] as const),
  frameWords: 4,
  frameBytes: 4 * Uint32Array.BYTES_PER_ELEMENT,
} as const);

export const SVO_CONE_FANOUT_SENTINELS = Object.freeze({
  /** The reduced primary ray missed; reducer writes the established all-ones key. */
  geometryMiss: -3,
  /** This deterministic lane is inactive in the current GPU-owned quality tier. */
  inactive: -2,
  /** A dirty derived page requires full-resolution exact traversal. */
  invalid: -1,
} as const);

export interface SvoConeFanoutFrame {
  width: number;
  height: number;
  lightCount?: number;
  secondaryLightSamples?: boolean;
}

function boundedInteger(value: number, maximum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0 || value > maximum) {
    throw new RangeError(`${label} must be an integer in [0, ${maximum}]`);
  }
  return value;
}

/** Layer mapping shared by dispatch construction, shader tests, and telemetry. */
export function svoConeFanoutAoLayer(sampleIndex: number): number {
  return SVO_CONE_FANOUT_CONTRACT.aoLayerBase
    + boundedInteger(sampleIndex, SVO_CONE_FANOUT_CONTRACT.maximumAoSamples - 1, "AO sample index");
}

export function svoConeFanoutLightLayer(lightIndex: number, sampleIndex: number): number {
  const light = boundedInteger(lightIndex, SVO_CONE_FANOUT_CONTRACT.maximumLights - 1, "Light index");
  const sample = boundedInteger(sampleIndex, SVO_CONE_FANOUT_CONTRACT.maximumLightSamples - 1, "Light sample index");
  return SVO_CONE_FANOUT_CONTRACT.lightLayerBase
    + sample * SVO_CONE_FANOUT_CONTRACT.maximumLights + light;
}

/**
 * Four uint32 words; safe to upload directly to the fan-out frame uniform.
 * The authored light count and one-vs-two-sample mode bound the dispatch and
 * reducer reads. Per-light activity, camera stability, visibility flags, and
 * tuning counts remain live GPU state.
 */
export function packSvoConeFanoutFrame(frame: SvoConeFanoutFrame): Uint32Array<ArrayBuffer> {
  const width = boundedInteger(frame.width, 0xffff_ffff, "Cone fan-out width");
  const height = boundedInteger(frame.height, 0xffff_ffff, "Cone fan-out height");
  if (width === 0 || height === 0) throw new RangeError("Cone fan-out dimensions must be positive");
  const lightCount = boundedInteger(frame.lightCount ?? SVO_CONE_FANOUT_CONTRACT.maximumLights,
    SVO_CONE_FANOUT_CONTRACT.maximumLights, "Cone fan-out light count");
  return new Uint32Array([width, height, lightCount, frame.secondaryLightSamples === false ? 0 : 1]) as Uint32Array<ArrayBuffer>;
}

/** Existing dry-scene buffers/textures rebound through a narrow compute layout. */
export function svoConeFanoutSceneBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "2d" } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "3d" } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "3d" } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "2d" } },
  ];
}

export function svoConeFanoutWorkerBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: SVO_CONE_FANOUT_CONTRACT.temporaryFormat, viewDimension: "2d-array" },
    },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
  ];
}

export function svoConeFanoutReducerBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" } },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: SVO_CONE_FANOUT_CONTRACT.visibilityFormat },
    },
  ];
}

export interface SvoConeFanoutWorkerShaderOptions {
  /**
   * The exact production marcher returned by createSvoDryConeMarcherWGSL.
   * Injection keeps this module independent of webgpu-svo-dry-scene and lets
   * integration share one authoritative marcher implementation.
   */
  coneMarcherWGSL: string;
  visibilityFlags: {
    ambientOcclusion: number;
    exactShadow: number;
    globalIllumination: number;
  };
}

/**
 * Dedicated worker source. The scene layout mirrors existing buffers but
 * deliberately omits traversal, material, glass, rigid-body, and GI resources.
 * Each invocation calls dryConeVisibility at most once.
 */
export function createSvoConeFanoutWorkerWGSL(options: SvoConeFanoutWorkerShaderOptions): string {
  if (!options.coneMarcherWGSL.includes("fn dryConeVisibility(")) {
    throw new Error("Cone fan-out worker requires the production dryConeVisibility marcher");
  }
  const visibilityFlag = (key: keyof SvoConeFanoutWorkerShaderOptions["visibilityFlags"]): number =>
    boundedInteger(options.visibilityFlags[key], 0xffff_ffff, `Cone fan-out ${key} visibility flag`) >>> 0;
  return /* wgsl */ `
${svoLightWGSL}
${svoNodeMipSamplingWGSL}
${svoFluidCoverageWGSL}
const DRY_MISS:f32=1e30;
const UNIFIED_PI:f32=3.141592653589793;
const FANOUT_AO_FLAG:u32=${visibilityFlag("ambientOcclusion")}u;
const FANOUT_SHADOW_FLAG:u32=${visibilityFlag("exactShadow")}u;
const FANOUT_GI_FLAG:u32=${visibilityFlag("globalIllumination")}u;
const FANOUT_GEOMETRY_MISS:f32=${SVO_CONE_FANOUT_SENTINELS.geometryMiss}.0;
const FANOUT_INACTIVE:f32=${SVO_CONE_FANOUT_SENTINELS.inactive}.0;
const FANOUT_INVALID:f32=${SVO_CONE_FANOUT_SENTINELS.invalid}.0;
const FANOUT_AO_LAYERS:u32=${SVO_CONE_FANOUT_CONTRACT.maximumAoSamples}u;
const FANOUT_LIGHT_BASE:u32=${SVO_CONE_FANOUT_CONTRACT.lightLayerBase}u;
const FANOUT_LIGHT_SAMPLES:u32=${SVO_CONE_FANOUT_CONTRACT.maximumLightSamples}u;
const FANOUT_LAYER_COUNT:u32=${SVO_CONE_FANOUT_CONTRACT.layerCount}u;
struct Uniforms {viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f,environment:vec4f,terrainMeta:vec4f,terrainFeatures:array<vec4f,16>}
struct SvoMapping {
  worldOrigin:vec3f,brickSize:u32,
  cellSize:vec3f,maximumDepth:u32,
  nodeCount:u32,leafCount:u32,maxVisits:u32,_padding:u32,
}
struct SvoTerrainMaterialMetadata{baseHeight_m:f32,waterline_m:f32,materialId:u32,policyVersion:u32}
struct DryParams {
  mapping:SvoMapping,
  metadata:vec4u,
  lightDirection:vec4f,
  lightColor:vec4f,
  terrain:vec4u,
  terrainMaterial:SvoTerrainMaterialMetadata,
  materialPublication:vec4u,
  nodeMip:vec4u,
  nodeMipAtlas:vec4u,
  wideFanout:vec4u,
  nodeMipLevelStart:array<vec4u,3>,
  nodeMipOrigin:vec4f,
  fluidCoverage:SvoFluidCoverageFrame,
  tuningCounts0:vec4u,
  tuningCounts1:vec4u,
  tuningCounts2:vec4u,
  tuningRays0:vec4f,
  tuningRays1:vec4f,
  nodeMipDirect:vec4u,
  nodeMipDirectLevelZ:array<vec4u,3>,
  tetrahedralRadiance:vec4u,
  nodeMipExtent:vec4f,
  giLighting:vec4f,
  giCones:vec4f,
}
struct SvoEnvironmentLightingRecord{lowerDiffuse:vec4f,upperSpecular:vec4f,accentPower:vec4f,keyColorIntensity:vec4f,keyDirectionSharpness:vec4f,identity:vec4u}
struct DryLightingArena {
  metadata:vec4u,
  lights:array<SvoLightRecord,${SVO_LIGHT_MAXIMUM_RECORDS}>,
  environment:SvoEnvironmentLightingRecord,
}
struct FanoutFrame {
  dimensions:vec2u,
  activity:vec2u,
}
@group(0) @binding(0) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(1) var<uniform> dry:DryParams;
@group(0) @binding(2) var<uniform> dryLighting:DryLightingArena;
@group(0) @binding(3) var<storage,read> publicationState:array<u32>;
@group(0) @binding(4) var nodeMipAtlas:texture_3d<f32>;
@group(0) @binding(5) var nodeMipSampler:sampler;
@group(0) @binding(6) var nodeMipDirectory:texture_2d<u32>;
@group(0) @binding(7) var fluidCoverageVolume:texture_3d<f32>;
@group(0) @binding(8) var nodeMipPageTable:texture_3d<u32>;
@group(0) @binding(9) var nodeMipPageValidity:texture_2d<u32>;
@group(1) @binding(0) var<uniform> fanout:FanoutFrame;
@group(1) @binding(1) var fanoutReceiver:texture_2d<f32>;
@group(1) @binding(2) var fanoutTemporary:texture_storage_2d_array<r32float,write>;
@group(1) @binding(3) var fanoutGeometry:texture_2d<f32>;
var<private> dryMipSteps:u32;
// The dedicated fan-out layout binds the publication-state slice directly,
// unlike the primary shader which reads it through the structural arena.
// Keep the marcher's publication helper identical at the call site while
// mapping its indices onto that direct binding here.
fn dryPublicationWord(index:u32)->u32{return publicationState[index];}
${options.coneMarcherWGSL}
fn fanoutDecodeNormal(octIn:vec2f)->vec3f{
  var normal=vec3f(octIn,1.0-abs(octIn.x)-abs(octIn.y));
  if(normal.z<0.0){let folded=(vec2f(1.0)-abs(normal.yx))*select(vec2f(-1.0),vec2f(1.0),normal.xy>=vec2f(0.0));normal=vec3f(folded,normal.z);}
  return normalize(normal);
}
fn fanoutRay(coordinate:vec2u)->mat2x3f{
  let uv=vec2f((f32(coordinate.x)+.5)/f32(fanout.dimensions.x),1.0-(f32(coordinate.y)+.5)/f32(fanout.dimensions.y));
  let ndc=uv*2.0-1.0;let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0,1,0)));let up=normalize(cross(right,forward));
  let rd=normalize(forward+right*ndc.x*uniforms.viewport.x/max(uniforms.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
  return mat2x3f(ro,rd);
}
fn fanoutContactRadius()->f32{
  let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let sceneScale=max(uniforms.container.x,max(uniforms.container.y,uniforms.container.z));
  return dry.tuningRays0.z*min(
    sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.maximumSceneRadiusFraction},
    max(cellScale*${SVO_CONTACT_VISIBILITY_CONTRACT.radiusCells}.0,sceneScale*${SVO_CONTACT_VISIBILITY_CONTRACT.minimumSceneRadiusFraction})
  );
}
fn fanoutContactDirection(normalIn:vec3f,featureId:u32,sampleIndex:u32)->vec3f{
  let normal=normalize(normalIn);let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(normal.y)>.9);
  var tangent=normalize(cross(helper,normal));var bitangent=cross(normal,tangent);
  if((featureId&1u)!=0u){let previous=tangent;tangent=bitangent;bitangent=-previous;}
  let signValue=select(1.0,-1.0,sampleIndex!=0u);
  return normalize(normal+signValue*(.55*tangent+.2*bitangent));
}
struct FanoutLightSample{towardLight:vec3f,finiteDistance_m:f32,valid:u32}
fn fanoutInvalidLightSample()->FanoutLightSample{return FanoutLightSample(vec3f(0,1,0),0.0,0u);}
fn fanoutLightSample(light:SvoLightRecord,sampleIndex:u32,position:vec3f)->FanoutLightSample{
  let baseRadiance=svoLightRadiance(light);if(max(max(baseRadiance.x,baseRadiance.y),baseRadiance.z)<=0.0){return fanoutInvalidLightSample();}
  if(light.identity.x==SVO_LIGHT_DIRECTIONAL){
    let lengthSquared=dot(light.directionCone.xyz,light.directionCone.xyz);
    if(lengthSquared<=1e-12){return fanoutInvalidLightSample();}
    return FanoutLightSample(light.directionCone.xyz*inverseSqrt(lengthSquared),0.0,1u);
  }
  var samplePosition=light.positionRange.xyz;
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA){
    let towardCenter=normalize(light.positionRange.xyz-position);let helper=select(vec3f(0,1,0),vec3f(1,0,0),abs(towardCenter.y)>.9);
    let tangent=normalize(cross(towardCenter,helper));let signValue=select(-1.0,1.0,sampleIndex!=0u);
    samplePosition+=tangent*(signValue*.45*light.shape.x);
  }else if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){
    let signValue=select(-1.0,1.0,sampleIndex!=0u);
    samplePosition+=light.axisUWidth.xyz*(signValue*.45*light.axisUWidth.w)+light.axisVHeight.xyz*(signValue*.2*light.axisVHeight.w);
  }
  let offset=samplePosition-position;let distanceSquared=dot(offset,offset);
  if(distanceSquared<=1e-10||(light.positionRange.w>0.0&&distanceSquared>=light.positionRange.w*light.positionRange.w)){return fanoutInvalidLightSample();}
  let distance=sqrt(distanceSquared);let towardLight=offset/distance;
  let rangeFade=select(1.0,pow(clamp(1.0-distance/max(light.positionRange.w,1e-6),0.0,1.0),2.0),light.positionRange.w>0.0);
  var shapeScale=1.0/max(1.0,distanceSquared);
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA){let area=4.0*UNIFIED_PI*light.shape.x*light.shape.x;shapeScale=area/max(area,distanceSquared);}
  if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){
    let area=4.0*light.axisUWidth.w*light.axisVHeight.w;let emitterFacing=max(dot(normalize(light.directionCone.xyz),-towardLight),0.0);
    shapeScale=emitterFacing*area/max(area,distanceSquared);
  }
  if(max(max(baseRadiance.x,baseRadiance.y),baseRadiance.z)*rangeFade*shapeScale<=0.0){return fanoutInvalidLightSample();}
  let visibilityDistance=select(distance,max(0.0,distance-light.shape.x),light.identity.x==SVO_LIGHT_POINT);
  return FanoutLightSample(towardLight,visibilityDistance,1u);
}
fn fanoutDirectionalExit(position:vec3f,direction:vec3f)->f32{
  let minimum=vec3f(-.5*uniforms.container.x,0.0,-.5*uniforms.container.z);
  let maximum=vec3f(.5*uniforms.container.x,uniforms.container.y,.5*uniforms.container.z);
  var enter=0.0;var exit=DRY_MISS;
  for(var axis=0u;axis<3u;axis+=1u){
    if(abs(direction[axis])<=1e-9){if(position[axis]<minimum[axis]||position[axis]>maximum[axis]){return 0.0;}}
    else{let first=(minimum[axis]-position[axis])/direction[axis];let second=(maximum[axis]-position[axis])/direction[axis];
      enter=max(enter,min(first,second));exit=min(exit,max(first,second));if(exit<enter){return 0.0;}}
  }
  return max(exit,0.0);
}
fn fanoutBiasedOrigin(position:vec3f,normal:vec3f,towardLight:vec3f)->vec4f{
  let projectedCellWidth=dot(abs(normal),dry.mapping.cellSize);let originBias=max(dry.tuningRays0.x,0.0)*projectedCellWidth;
  let side=select(-1.0,1.0,dot(normal,towardLight)>=0.0);return vec4f(position+side*normal*originBias,originBias);
}
fn fanoutStore(coordinate:vec2i,layer:i32,value:f32){textureStore(fanoutTemporary,coordinate,layer,vec4f(value,0,0,0));}
@compute @workgroup_size(${SVO_CONE_FANOUT_CONTRACT.workgroupSize.join(",")})
fn svoConeFanoutWorker(@builtin(global_invocation_id) gid:vec3u){
  if(any(gid.xy>=fanout.dimensions)||gid.z>=FANOUT_LAYER_COUNT){return;}
  let coordinate=vec2i(gid.xy);let receiver=textureLoad(fanoutReceiver,coordinate,0);
  if(receiver.x<=0.0){fanoutStore(coordinate,i32(gid.z),FANOUT_GEOMETRY_MISS);return;}
  let geometry=textureLoad(fanoutGeometry,coordinate,0);
  let ray=fanoutRay(gid.xy);let position=ray[0]+ray[1]*receiver.x;let normal=receiver.yzw;
  if(gid.z<FANOUT_AO_LAYERS){
    let sampleIndex=gid.z;
    let sampleCount=select(dry.tuningCounts1.z,dry.tuningCounts1.y,uniforms.viewport.w>=-1.0);
    if((dry.materialPublication.w&FANOUT_AO_FLAG)==0u||sampleIndex>=sampleCount){
      fanoutStore(coordinate,i32(gid.z),FANOUT_INACTIVE);return;
    }
    let radius=fanoutContactRadius();if(radius<=0.0){fanoutStore(coordinate,i32(gid.z),1.0);return;}
    let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
    let origin=position+normal*cellScale*.2;let featureId=u32(round(geometry.w))&15u;
    let direction=fanoutContactDirection(normal,featureId,sampleIndex&1u);
    let rotated=select(direction,normalize(direction+cross(normal,direction)*.7),sampleIndex>=2u);
    let cone=dryConeVisibility(origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);
    fanoutStore(coordinate,i32(gid.z),select(FANOUT_INVALID,cone.transmittance,cone.valid!=0u));return;
  }
  let secondary=gid.z>=${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u;
  let lightIndex=gid.z-select(FANOUT_LIGHT_BASE,${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u,secondary);let sampleIndex=select(0u,1u,secondary);
  let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_CONE_FANOUT_CONTRACT.maximumLights}u));
  if((dry.materialPublication.w&FANOUT_SHADOW_FLAG)==0u||lightIndex>=lightCount){
    fanoutStore(coordinate,i32(gid.z),FANOUT_INACTIVE);return;
  }
  let light=dryLighting.lights[lightIndex];
  let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;
  let settled=uniforms.viewport.w>=-1.0;
  let globalIllumination=(dry.materialPublication.w&FANOUT_GI_FLAG)!=0u;
  let sampleCount=select(select(1u,select(dry.tuningCounts1.x,dry.tuningCounts0.w,settled),area),1u,globalIllumination);
  if(sampleIndex>=sampleCount){fanoutStore(coordinate,i32(gid.z),FANOUT_INACTIVE);return;}
  if(light.identity.w!=dryLighting.metadata.y){fanoutStore(coordinate,i32(gid.z),0.0);return;}
  let sample=fanoutLightSample(light,sampleIndex,position);
  if(sample.valid==0u||dot(normal,sample.towardLight)<=0.0){fanoutStore(coordinate,i32(gid.z),0.0);return;}
  let maximumDistance=select(fanoutDirectionalExit(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
  if(maximumDistance<=0.0){fanoutStore(coordinate,i32(gid.z),0.0);return;}
  let biased=fanoutBiasedOrigin(position,normal,sample.towardLight);
  let rayMaximum=max(0.0,maximumDistance-dot(biased.xyz-position,sample.towardLight));
  let coneCell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let coneEscape=coneCell*dry.tuningRays1.z;
  let coneMaxRaw=max(0.0,rayMaximum-coneEscape*dot(normal,sample.towardLight));
  let coneMax=coneMaxRaw-select(0.0,dry.tuningRays1.w*coneCell,sample.finiteDistance_m>0.0);
  let cone=dryConeVisibility(biased.xyz+normal*coneEscape,sample.towardLight,dry.tuningRays1.y,coneMax,normal,sample.finiteDistance_m>0.0);
  fanoutStore(coordinate,i32(gid.z),select(FANOUT_INVALID,mix(1.0,cone.transmittance,dry.tuningRays0.y),cone.valid!=0u));
}
`;
}

/** Minimal second pass; it preserves the original sample addition and packing order. */
export const svoConeFanoutReducerWGSL = /* wgsl */ `
const FANOUT_GEOMETRY_MISS:f32=${SVO_CONE_FANOUT_SENTINELS.geometryMiss}.0;
const FANOUT_INACTIVE:f32=${SVO_CONE_FANOUT_SENTINELS.inactive}.0;
const FANOUT_INVALID:f32=${SVO_CONE_FANOUT_SENTINELS.invalid}.0;
const FANOUT_INVALID_PACKED:vec2u=vec2u(0xffffffffu,0xfffffffeu);
const FANOUT_AO_LAYERS:u32=${SVO_CONE_FANOUT_CONTRACT.maximumAoSamples}u;
const FANOUT_LIGHT_BASE:u32=${SVO_CONE_FANOUT_CONTRACT.lightLayerBase}u;
struct FanoutFrame {
  dimensions:vec2u,
  activity:vec2u,
}
@group(0) @binding(0) var<uniform> fanout:FanoutFrame;
@group(0) @binding(1) var fanoutTemporary:texture_2d_array<f32>;
@group(0) @binding(2) var fanoutVisibility:texture_storage_2d<rg32uint,write>;
fn fanoutLoad(coordinate:vec2i,layer:u32)->f32{return textureLoad(fanoutTemporary,coordinate,i32(layer),0).x;}
fn fanoutQuantize7(value:f32)->u32{return u32(round(clamp(value,0.0,1.0)*127.0));}
fn fanoutPack(data0:vec4f,data1:vec4f,data2:vec4f)->vec2u{
  let light3=fanoutQuantize7(data1.x);
  let word0=u32(round(clamp(data0.x,0.0,1.0)*255.0))|(fanoutQuantize7(data0.y)<<8u)|(fanoutQuantize7(data0.z)<<15u)
    |(fanoutQuantize7(data0.w)<<22u)|((light3&7u)<<29u);
  let word1=(light3>>3u)|(fanoutQuantize7(data1.y)<<4u)|(fanoutQuantize7(data1.z)<<11u)
    |(fanoutQuantize7(data1.w)<<18u)|(fanoutQuantize7(data2.x)<<25u);
  return vec2u(word0,word1);
}
@compute @workgroup_size(${SVO_CONE_FANOUT_CONTRACT.workgroupSize.join(",")})
fn svoConeFanoutReduce(@builtin(global_invocation_id) gid:vec3u){
  if(any(gid.xy>=fanout.dimensions)){return;}let coordinate=vec2i(gid.xy);
  if(fanoutLoad(coordinate,0u)==FANOUT_GEOMETRY_MISS){textureStore(fanoutVisibility,coordinate,vec4u(0xffffffffu));return;}
  var visibility0=vec4f(1.0);var visibility1=vec4f(1.0);var visibility2=vec4f(1.0);
  if(fanoutLoad(coordinate,0u)!=FANOUT_INACTIVE){
    var ao=0.0;var aoSampleCount=0u;
    for(var sample=0u;sample<FANOUT_AO_LAYERS;sample+=1u){
      let value=fanoutLoad(coordinate,sample);if(value==FANOUT_INACTIVE){break;}if(value==FANOUT_INVALID){textureStore(fanoutVisibility,coordinate,vec4u(FANOUT_INVALID_PACKED,0u,0u));return;}ao+=value;aoSampleCount+=1u;
    }
    if(aoSampleCount>0u){visibility0.x=clamp(ao/f32(aoSampleCount),0.0,1.0);}
  }
  for(var light=0u;light<${SVO_CONE_FANOUT_CONTRACT.maximumLights}u;light+=1u){
    if(light>=fanout.activity.x){break;}let base=FANOUT_LIGHT_BASE+light;var value=fanoutLoad(coordinate,base);
    if(value==FANOUT_INACTIVE){continue;}if(value==FANOUT_INVALID){textureStore(fanoutVisibility,coordinate,vec4u(FANOUT_INVALID_PACKED,0u,0u));return;}var sampleCount=1u;
    if(fanout.activity.y!=0u){let second=fanoutLoad(coordinate,${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u+light);if(second==FANOUT_INVALID){textureStore(fanoutVisibility,coordinate,vec4u(FANOUT_INVALID_PACKED,0u,0u));return;}if(second!=FANOUT_INACTIVE){value+=second;sampleCount=2u;}}
    let packed=clamp(value/f32(sampleCount),0.0,1.0);
    if(light<3u){visibility0[1u+light]=packed;}else if(light<7u){visibility1[light-3u]=packed;}else{visibility2.x=packed;}
  }
  let packed=fanoutPack(visibility0,visibility1,visibility2);
  textureStore(fanoutVisibility,coordinate,vec4u(packed,0u,0u));
}
`;
