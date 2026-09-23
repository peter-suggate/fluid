import { SVO_LIGHT_MAXIMUM_RECORDS, svoLightWGSL } from "../../contracts/svo-light-abi";
import { cameraApertureShaderLibrary } from "../../../core/webgpu-camera";
import { SVO_CONTACT_VISIBILITY_CONTRACT } from "./svo-contact-visibility";
import { svoFluidCoverageWGSL } from "../scene-publication/svo-fluid-coverage";
import { svoNodeMipSamplingWGSL } from "../radiance/svo-node-mip-sampling";

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

/**
 * Lattice visibility evaluates the same cone lanes at half-voxel points of
 * each receiver's face plane instead of at reduced screen texels. A visible
 * surface carries far fewer lattice points than prepass texels, and the
 * cones are wide and trilinear, so the function sampled is smooth there.
 *
 * Every point is a pure function of its 128-bit key. A key pass inserts the
 * four corners around each distinct pixel cell into an open-addressed table
 * rebuilt every frame. The worker marches the compacted entries through the
 * fan-out lanes, and the lighting pass interpolates its four corners. The
 * key is (octahedral normal, feature id, level, plane offset, cell i, cell j).
 * The plane offset is quantised to 1/16 cell, so voxel faces land exactly.
 */
export const SVO_LATTICE_VISIBILITY_CONTRACT = Object.freeze({
  /** Linear-probe bound. The table holds twice the capacity, so misses stay short. */
  maximumProbes: 64,
  maximumLevel: 7,
  /**
   * Smallest projected half-cell spacing, in full-resolution pixels. The
   * shader raises it to the reduced prepass pitch (4 px at x0.25), so the
   * lattice never samples more densely than the screen path it replaces.
   */
  minimumSpacingPixels: 2,
  planeSubdivisions: 16,
  /** Floor on |n·rd| before the area-preserving square root: grazing faces coarsen. */
  grazingCosineFloor: 1 / 64,
  /** Control words: count, overflow, exhausted, then two indirect argument triples. */
  headerWords: 16,
  countWord: 0,
  overflowWord: 1,
  exhaustedWord: 2,
  workerArgumentsWord: 4,
  reduceArgumentsWord: 8,
  /** Two vec4u per slot: the key, then (compact index + 1, packed rg32 visibility, 0). */
  recordBytes: 32,
  keyWorkgroupSize: Object.freeze([16, 16, 1] as const),
  entryWorkgroupSize: 64,
  maximumWorkgroupsPerDimension: 65535,
  /** Main-module group 1, used by the key entry alone. */
  keyBindings: Object.freeze({ tags: 14, records: 15, control: 16 } as const),
  /** Split lighting group, read by the deferred lighting fragment. */
  lookupBindings: Object.freeze({ tags: 20, records: 21 } as const),
} as const);

export interface SvoLatticeVisibilitySizing {
  /** Power-of-two table slots; tags are one u32 each, records 32 bytes each. */
  slots: number;
  /** Compact entries the worker can evaluate; never more than the prepass texels. */
  capacity: number;
}

/**
 * The table keeps at most half its slots occupied, so probes stay short. The
 * compact capacity is the reduced prepass texel count, which the shared
 * fan-out temporary array addresses one texel per entry. Where the device's
 * storage-binding ceiling cannot hold twice that, the capacity gives way,
 * not the occupancy: further keys are counted as overflow and their pixels
 * take the exact-edge tier.
 */
export function svoLatticeVisibilitySizing(prepassTexels: number, bindingLimitBytes: number): SvoLatticeVisibilitySizing {
  const texels = boundedInteger(prepassTexels, 0x4000_0000, "Lattice visibility prepass texels");
  if (texels === 0) throw new RangeError("Lattice visibility needs a non-empty prepass");
  let slots = 1;
  while (slots < 2 * texels) slots *= 2;
  while (slots > 2 && slots * SVO_LATTICE_VISIBILITY_CONTRACT.recordBytes > bindingLimitBytes) slots /= 2;
  if (slots * SVO_LATTICE_VISIBILITY_CONTRACT.recordBytes > bindingLimitBytes) {
    throw new RangeError("Lattice visibility table exceeds the storage-binding limit");
  }
  return { slots, capacity: Math.min(texels, slots / 2) };
}

/**
 * Key codec shared verbatim by the key pass, the lighting lookup, and the
 * cone worker. The worker rebuilds the sample from the key alone, so two
 * pixels inserting one key always evaluate one sample. Eight-bit octahedral
 * levels 0..254 put zero at 127, so the six axis normals round-trip exactly.
 */
export const svoLatticeKeyWGSL = /* wgsl */ `
const SVO_LATTICE_VALID:u32=0x80000000u;
const SVO_LATTICE_PLANE_SUBDIVISIONS:f32=${SVO_LATTICE_VISIBILITY_CONTRACT.planeSubdivisions}.0;
struct SvoLatticeCell{key:vec4u,fraction:vec2f}
struct SvoLatticeReceiver{position:vec3f,normal:vec3f,featureId:u32}
fn svoLatticeEncodeNormal(normalIn:vec3f)->u32{
  var oct=normalIn.xy/max(abs(normalIn.x)+abs(normalIn.y)+abs(normalIn.z),1e-20);
  if(normalIn.z<0.0){oct=(vec2f(1.0)-abs(oct.yx))*select(vec2f(-1.0),vec2f(1.0),oct>=vec2f(0.0));}
  let quantized=vec2u(clamp(round(oct*127.0)+vec2f(127.0),vec2f(0.0),vec2f(254.0)));return quantized.x|(quantized.y<<8u);
}
fn svoLatticeDecodeNormal(bits:u32)->vec3f{
  let oct=(vec2f(f32(bits&255u),f32((bits>>8u)&255u))-vec2f(127.0))/127.0;var normal=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));
  if(normal.z<0.0){normal=vec3f((vec2f(1.0)-abs(normal.yx))*select(vec2f(-1.0),vec2f(1.0),normal.xy>=vec2f(0.0)),normal.z);}
  return normalize(normal);
}
// Dominant axis with ties resolved x, then y, then z; u and v follow cyclically.
fn svoLatticeAxis(normal:vec3f)->u32{let magnitude=abs(normal);var axis=0u;if(magnitude.y>magnitude.x){axis=1u;}if(magnitude.z>magnitude[axis]){axis=2u;}return axis;}
fn svoLatticeCell(position:vec3f,normal:vec3f,featureId:u32,level:u32,origin:vec3f,cellSize:vec3f)->SvoLatticeCell{
  let normalBits=svoLatticeEncodeNormal(normal);let planeNormal=svoLatticeDecodeNormal(normalBits);
  let axis=svoLatticeAxis(planeNormal);let u=(axis+1u)%3u;let v=(axis+2u)%3u;let relative=position-origin;
  let spacing=vec2f(cellSize[u],cellSize[v])*(.5*f32(1u<<level));
  let offset=i32(round(dot(planeNormal,relative)/(cellSize[axis]/SVO_LATTICE_PLANE_SUBDIVISIONS)));
  let scaled=vec2f(relative[u],relative[v])/spacing;let base=floor(scaled);
  return SvoLatticeCell(vec4u(normalBits|((featureId&15u)<<16u)|((level&7u)<<20u)|SVO_LATTICE_VALID,bitcast<u32>(offset),bitcast<vec2u>(vec2i(base))),scaled-base);
}
fn svoLatticeCorner(key:vec4u,corner:u32)->vec4u{return vec4u(key.xy,bitcast<vec2u>(bitcast<vec2i>(key.zw)+vec2i(i32(corner&1u),i32(corner>>1u))));}
// The in-plane lattice point, lifted onto the key's plane along the dominant
// axis. Axis faces have zero in-plane normal terms and land exactly on the face.
fn svoLatticeReceiver(key:vec4u,origin:vec3f,cellSize:vec3f)->SvoLatticeReceiver{
  let normal=svoLatticeDecodeNormal(key.x&0xffffu);let axis=svoLatticeAxis(normal);let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let planar=vec2f(bitcast<vec2i>(key.zw))*vec2f(cellSize[u],cellSize[v])*(.5*f32(1u<<((key.x>>20u)&7u)));
  var relative=vec3f(0.0);relative[u]=planar.x;relative[v]=planar.y;
  relative[axis]=(f32(bitcast<i32>(key.y))*(cellSize[axis]/SVO_LATTICE_PLANE_SUBDIVISIONS)-normal[u]*planar.x-normal[v]*planar.y)/normal[axis];
  return SvoLatticeReceiver(origin+relative,normal,(key.x>>16u)&15u);
}
fn svoLatticeMix(value:u32)->u32{var x=value;x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
// The tag is never zero, the cleared-slot value. The slot uses a second mix so
// its low bits are not the tag's own.
fn svoLatticeHash(key:vec4u)->u32{return svoLatticeMix(key.x^svoLatticeMix(key.y^svoLatticeMix(key.z^svoLatticeMix(key.w))));}
fn svoLatticeTag(hash:u32)->u32{return hash|1u;}
fn svoLatticeSlot(hash:u32,mask:u32)->u32{return svoLatticeMix(hash^0x68e31da4u)&mask;}
`;

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

/** Key-pass table: claimed tags, key/payload records, then the control header and compact list. */
export function svoLatticeKeyBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { tags, records, control } = SVO_LATTICE_VISIBILITY_CONTRACT.keyBindings;
  return [tags, records, control].map((binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const },
  }));
}

/** Read-only table view appended to the split lighting group. */
export function svoLatticeLookupBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { tags, records } = SVO_LATTICE_VISIBILITY_CONTRACT.lookupBindings;
  return [tags, records].map((binding) => ({
    binding, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" as const },
  }));
}

/**
 * The lattice worker shares the fan-out temporary array as [layer][entry];
 * the compact capacity is the prepass texel count, so every entry has a texel.
 * Control is read-only here because the same buffer carries this dispatch's
 * indirect arguments.
 */
export function svoLatticeWorkerBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    {
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      storageTexture: { access: "write-only", format: SVO_CONE_FANOUT_CONTRACT.temporaryFormat, viewDimension: "2d-array" },
    },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ];
}

export function svoLatticeReducerBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d-array" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ];
}

export function svoLatticeArgumentsBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_SPOT){
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
  if(light.identity.x==SVO_LIGHT_SPOT){shapeScale*=svoLightConeFalloff(light,towardLight);}
  if(max(max(baseRadiance.x,baseRadiance.y),baseRadiance.z)*rangeFade*shapeScale<=0.0){return fanoutInvalidLightSample();}
  let visibilityDistance=select(distance,max(0.0,distance-light.shape.x),light.identity.x==SVO_LIGHT_POINT||light.identity.x==SVO_LIGHT_SPOT);
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
// One deterministic lane for one receiver. The screen worker and the lattice
// worker call this same body, so both sources evaluate identical cones and
// differ only in where the receiver comes from.
fn fanoutConeSample(position:vec3f,normal:vec3f,featureId:u32,layer:u32)->f32{
  if(layer<FANOUT_AO_LAYERS){
    let sampleIndex=layer;
    let sampleCount=max(dry.tuningCounts1.z,dry.tuningCounts1.y);
    if((dry.materialPublication.w&FANOUT_AO_FLAG)==0u||sampleIndex>=sampleCount){return FANOUT_INACTIVE;}
    let radius=fanoutContactRadius();if(radius<=0.0){return 1.0;}
    let cellScale=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
    let origin=position+normal*cellScale*.2;
    let direction=fanoutContactDirection(normal,featureId,sampleIndex&1u);
    let rotated=select(direction,normalize(direction+cross(normal,direction)*.7),sampleIndex>=2u);
    let cone=dryConeVisibility(origin,rotated,dry.tuningRays1.x,radius,vec3f(0.0),false);
    return select(FANOUT_INVALID,cone.transmittance,cone.valid!=0u);
  }
  let secondary=layer>=${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u;
  let lightIndex=layer-select(FANOUT_LIGHT_BASE,${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u,secondary);let sampleIndex=select(0u,1u,secondary);
  let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,${SVO_CONE_FANOUT_CONTRACT.maximumLights}u));
  if((dry.materialPublication.w&FANOUT_SHADOW_FLAG)==0u||lightIndex>=lightCount){return FANOUT_INACTIVE;}
  let light=dryLighting.lights[lightIndex];
  let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA;
  let globalIllumination=(dry.materialPublication.w&FANOUT_GI_FLAG)!=0u;
  let sampleCount=select(select(1u,max(dry.tuningCounts1.x,dry.tuningCounts0.w),area),1u,globalIllumination);
  if(sampleIndex>=sampleCount){return FANOUT_INACTIVE;}
  if(light.identity.w!=dryLighting.metadata.y){return 0.0;}
  let sample=fanoutLightSample(light,sampleIndex,position);
  if(sample.valid==0u||dot(normal,sample.towardLight)<=0.0){return 0.0;}
  let maximumDistance=select(fanoutDirectionalExit(position,sample.towardLight),sample.finiteDistance_m,sample.finiteDistance_m>0.0);
  if(maximumDistance<=0.0){return 0.0;}
  let biased=fanoutBiasedOrigin(position,normal,sample.towardLight);
  let rayMaximum=max(0.0,maximumDistance-dot(biased.xyz-position,sample.towardLight));
  let coneCell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let coneEscape=coneCell*dry.tuningRays1.z;
  let coneMaxRaw=max(0.0,rayMaximum-coneEscape*dot(normal,sample.towardLight));
  let coneMax=coneMaxRaw-select(0.0,dry.tuningRays1.w*coneCell,sample.finiteDistance_m>0.0);
  let cone=dryConeVisibility(biased.xyz+normal*coneEscape,sample.towardLight,dry.tuningRays1.y,coneMax,normal,sample.finiteDistance_m>0.0);
  return select(FANOUT_INVALID,mix(1.0,cone.transmittance,dry.tuningRays0.y),cone.valid!=0u);
}
@compute @workgroup_size(${SVO_CONE_FANOUT_CONTRACT.workgroupSize.join(",")})
fn svoConeFanoutWorker(@builtin(global_invocation_id) gid:vec3u){
  if(any(gid.xy>=fanout.dimensions)||gid.z>=FANOUT_LAYER_COUNT){return;}
  let coordinate=vec2i(gid.xy);let receiver=textureLoad(fanoutReceiver,coordinate,0);
  if(receiver.x<=0.0){fanoutStore(coordinate,i32(gid.z),FANOUT_GEOMETRY_MISS);return;}
  let geometry=textureLoad(fanoutGeometry,coordinate,0);
  let ray=fanoutRay(gid.xy);let position=ray[0]+ray[1]*receiver.x;
  fanoutStore(coordinate,i32(gid.z),fanoutConeSample(position,receiver.yzw,u32(round(geometry.w))&15u,gid.z));
}
${svoLatticeKeyWGSL}
@group(1) @binding(4) var<storage,read> latticeRecords:array<vec4u>;
@group(1) @binding(5) var<storage,read> latticeControl:array<u32>;
// One lane per (compact entry, layer). The sample is rebuilt from the key, so
// it does not depend on which pixel inserted the entry.
@compute @workgroup_size(${SVO_LATTICE_VISIBILITY_CONTRACT.entryWorkgroupSize})
fn svoLatticeConeWorker(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let entry=gid.x+gid.y*groups.x*${SVO_LATTICE_VISIBILITY_CONTRACT.entryWorkgroupSize}u;
  let entries=min(latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.countWord}],arrayLength(&latticeControl)-${SVO_LATTICE_VISIBILITY_CONTRACT.headerWords}u);
  if(entry>=entries||gid.z>=FANOUT_LAYER_COUNT){return;}
  let receiver=svoLatticeReceiver(latticeRecords[2u*latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.headerWords}u+entry]],dry.mapping.worldOrigin,dry.mapping.cellSize);
  let texel=vec2i(vec2u(entry%fanout.dimensions.x,entry/fanout.dimensions.x));
  fanoutStore(texel,i32(gid.z),fanoutConeSample(receiver.position,receiver.normal,receiver.featureId,gid.z));
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
// Shared by the screen and lattice reductions: the geometry-miss key, the
// invalid key, or the packed visibility, in the original summation order.
fn fanoutReduce(coordinate:vec2i)->vec2u{
  if(fanoutLoad(coordinate,0u)==FANOUT_GEOMETRY_MISS){return vec2u(0xffffffffu);}
  var visibility0=vec4f(1.0);var visibility1=vec4f(1.0);var visibility2=vec4f(1.0);
  if(fanoutLoad(coordinate,0u)!=FANOUT_INACTIVE){
    var ao=0.0;var aoSampleCount=0u;
    for(var sample=0u;sample<FANOUT_AO_LAYERS;sample+=1u){
      let value=fanoutLoad(coordinate,sample);if(value==FANOUT_INACTIVE){break;}if(value==FANOUT_INVALID){return FANOUT_INVALID_PACKED;}ao+=value;aoSampleCount+=1u;
    }
    if(aoSampleCount>0u){visibility0.x=clamp(ao/f32(aoSampleCount),0.0,1.0);}
  }
  for(var light=0u;light<${SVO_CONE_FANOUT_CONTRACT.maximumLights}u;light+=1u){
    if(light>=fanout.activity.x){break;}let base=FANOUT_LIGHT_BASE+light;var value=fanoutLoad(coordinate,base);
    if(value==FANOUT_INACTIVE){continue;}if(value==FANOUT_INVALID){return FANOUT_INVALID_PACKED;}var sampleCount=1u;
    if(fanout.activity.y!=0u){let second=fanoutLoad(coordinate,${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u+light);if(second==FANOUT_INVALID){return FANOUT_INVALID_PACKED;}if(second!=FANOUT_INACTIVE){value+=second;sampleCount=2u;}}
    let packed=clamp(value/f32(sampleCount),0.0,1.0);
    if(light<3u){visibility0[1u+light]=packed;}else if(light<7u){visibility1[light-3u]=packed;}else{visibility2.x=packed;}
  }
  return fanoutPack(visibility0,visibility1,visibility2);
}
@compute @workgroup_size(${SVO_CONE_FANOUT_CONTRACT.workgroupSize.join(",")})
fn svoConeFanoutReduce(@builtin(global_invocation_id) gid:vec3u){
  if(any(gid.xy>=fanout.dimensions)){return;}let coordinate=vec2i(gid.xy);
  textureStore(fanoutVisibility,coordinate,vec4u(fanoutReduce(coordinate),0u,0u));
}
@group(0) @binding(3) var<storage,read_write> latticeRecords:array<vec4u>;
@group(0) @binding(4) var<storage,read> latticeControl:array<u32>;
@group(0) @binding(5) var<storage,read_write> latticeArguments:array<u32>;
// The packed word lands beside the key; a nonzero compact index marks a
// computed record, which is what the lighting lookup requires.
@compute @workgroup_size(${SVO_LATTICE_VISIBILITY_CONTRACT.entryWorkgroupSize})
fn svoLatticeConeReduce(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
  let entry=gid.x+gid.y*groups.x*${SVO_LATTICE_VISIBILITY_CONTRACT.entryWorkgroupSize}u;
  let entries=min(latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.countWord}],arrayLength(&latticeControl)-${SVO_LATTICE_VISIBILITY_CONTRACT.headerWords}u);
  if(entry>=entries){return;}
  let packed=fanoutReduce(vec2i(vec2u(entry%fanout.dimensions.x,entry/fanout.dimensions.x)));
  latticeRecords[2u*latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.headerWords}u+entry]+1u]=vec4u(entry+1u,packed,0u);
}
// Indirect arguments for the worker (entries x active layers) and the reduce.
// The compact capacity is the control buffer's tail; overflowed inserts were
// counted by the key pass and only bound the entry count here.
@compute @workgroup_size(1)
fn svoLatticeConeArguments(){
  let entries=min(latticeArguments[${SVO_LATTICE_VISIBILITY_CONTRACT.countWord}],arrayLength(&latticeArguments)-${SVO_LATTICE_VISIBILITY_CONTRACT.headerWords}u);
  let groups=(entries+${SVO_LATTICE_VISIBILITY_CONTRACT.entryWorkgroupSize - 1}u)/${SVO_LATTICE_VISIBILITY_CONTRACT.entryWorkgroupSize}u;
  let x=min(groups,${SVO_LATTICE_VISIBILITY_CONTRACT.maximumWorkgroupsPerDimension}u);let y=select(0u,(groups+x-1u)/max(x,1u),x>0u);
  let layers=select(FANOUT_LIGHT_BASE+fanout.activity.x,${SVO_CONE_FANOUT_CONTRACT.layerCount}u,fanout.activity.y!=0u);
  latticeArguments[${SVO_LATTICE_VISIBILITY_CONTRACT.workerArgumentsWord}]=x;latticeArguments[${SVO_LATTICE_VISIBILITY_CONTRACT.workerArgumentsWord + 1}]=y;latticeArguments[${SVO_LATTICE_VISIBILITY_CONTRACT.workerArgumentsWord + 2}]=layers;
  latticeArguments[${SVO_LATTICE_VISIBILITY_CONTRACT.reduceArgumentsWord}]=x;latticeArguments[${SVO_LATTICE_VISIBILITY_CONTRACT.reduceArgumentsWord + 1}]=y;latticeArguments[${SVO_LATTICE_VISIBILITY_CONTRACT.reduceArgumentsWord + 2}]=1u;
}
`;
