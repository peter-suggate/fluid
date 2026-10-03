import { SVO_LIGHT_MAXIMUM_RECORDS, svoLightWGSL } from "../../contracts/svo-light-abi";
import { cameraApertureShaderLibrary } from "../../../core/webgpu-camera";
import { SVO_CONTACT_VISIBILITY_CONTRACT } from "./svo-contact-visibility";
import { svoFluidCoverageWGSL } from "../scene-publication/svo-fluid-coverage";
import { svoNodeMipSamplingWGSL } from "../radiance/svo-node-mip-sampling";
import { backdropTerrainWGSL } from "../backdrop/backdrop-terrain-tiles";

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
 * Every point is a pure function of its 128-bit key, and its visibility is a
 * pure function of the key and the scene, never of the camera. The store is
 * therefore persistent and world-keyed: a key pass refreshes the four corners
 * around each distinct pixel cell and appends only the misses, a grid-stride
 * worker marches exactly those, and the lighting pass interpolates the four
 * corners. The key is (octahedral normal, feature id, level, plane offset,
 * cell i, cell j); the plane offset is quantised to 1/16 cell, so voxel faces
 * land exactly.
 *
 * The store is a two-choice hash of eight-way buckets. A bucket is sixteen
 * words: eight tags, one 32-byte line the lighting lookup reads as two vec4u,
 * then eight use words (frame << 8 | fresh << 7 | tag bits) the key pass
 * claims by compare-exchange. A record is the key, then (packed rg32
 * visibility, stamp, 0). The stamp folds the host's input generation with the
 * container; zero means "not valid", so an invalid march and a stale record
 * both read as absent and are marched again when next requested. The record
 * array ends in one vec4u whose x is the stamp this frame's key pass requested
 * under; the lighting lookup accepts only a record carrying exactly that stamp,
 * so a record left from an earlier input generation is never interpolated.
 */
export const SVO_LATTICE_VISIBILITY_CONTRACT = Object.freeze({
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
  bucketWays: 8,
  /** Words per bucket: eight tags, then eight use words. */
  bucketWords: 16,
  /**
   * Store slots per reduced prepass texel. One frame's cells are about the
   * texel count, but a surface broken into many small planes or features
   * requests four corners per fragment, so keys can exceed it. With the key
   * pass placing each key in the less-loaded bucket of its pair, 2.5x holds up
   * to about 1.75 keys per texel before any pair fills with current keys.
   */
  slotsPerTexel: 2.5,
  /** Scan-claim rounds a contended request may take before it counts as overflow. */
  claimAttempts: 16,
  /**
   * Control words: misses this frame (cleared per frame), sticky overflow,
   * the GPU frame counter the worker advances, and the host input generation;
   * then the miss list, one word per slot.
   */
  headerWords: 4,
  missCountWord: 0,
  overflowWord: 1,
  frameWord: 2,
  generationWord: 3,
  /** A miss is slot | lanes << 23: bit 0 the AO group, bit 1 + l light l's samples. */
  laneShift: 23,
  allLanes: 0x1ff,
  maximumSlots: 1 << 23,
  /** Two vec4u per slot: the key, then (packed rg32 visibility, stamp, 0). */
  recordBytes: 32,
  /** One trailing vec4u after the records: (current stamp, 0, 0, 0), written by the key pass. */
  recordHeaderBytes: 16,
  /**
   * Per-pixel corners, one vec4u per full-resolution pixel, written by the key
   * pass for this frame: the four corner slots as 24-bit fields packed across
   * x, y and z, then the bilinear fraction as two 16-bit unorms. The lighting
   * lookup loads the four records directly and never hashes.
   */
  cornerBytes: 16,
  /** A corner the key pass could not place; never a slot, since slots stay below 1 << 23. */
  cornerAbsent: 0xffffff,
  /** Tag, use, record and miss-list words. */
  slotBytes: 4 + 4 + 32 + 4,
  keyWorkgroupSize: Object.freeze([16, 16, 1] as const),
  marchWorkgroupSize: 64,
  /** Fixed direct grid: the worker strides over however many misses the key pass appended. */
  marchWorkgroups: 1024,
  /** Main-module group 1, used by the key entry alone. */
  keyBindings: Object.freeze({ buckets: 14, records: 15, control: 16, corners: 17 } as const),
  /** Split lighting group, read by the deferred lighting fragment. */
  lookupBindings: Object.freeze({ corners: 20, records: 21 } as const),
} as const);

export interface SvoLatticeVisibilitySizing {
  /** Eight-way buckets; any count, addressed by modulo. */
  buckets: number;
  slots: number;
  /** Buckets, records and control together. */
  bytes: number;
}

/**
 * The reduced prepass is an initial estimate, not a key-count bound: separate
 * voxel faces can request four distinct corners per full-resolution pixel.
 * After overflow, minimumSlots grows the store without changing sampling.
 * The record binding and packed slot field bound all allocations.
 */
export function svoLatticeVisibilitySizing(prepassTexels: number, bindingLimitBytes: number, minimumSlots = 0): SvoLatticeVisibilitySizing {
  const contract = SVO_LATTICE_VISIBILITY_CONTRACT;
  const texels = boundedInteger(prepassTexels, 0x4000_0000, "Lattice visibility prepass texels");
  if (texels === 0) throw new RangeError("Lattice visibility needs a non-empty prepass");
  const minimum = boundedInteger(minimumSlots, contract.maximumSlots * 2, "Lattice visibility minimum slots");
  const buckets = Math.min(Math.ceil(Math.max(contract.slotsPerTexel * texels, minimum) / contract.bucketWays),
    contract.maximumSlots / contract.bucketWays,
    Math.floor((bindingLimitBytes - contract.recordHeaderBytes) / (contract.bucketWays * contract.recordBytes)));
  const slots = buckets * contract.bucketWays;
  if (slots < texels) throw new RangeError("Lattice visibility store cannot hold one frame's keys within the storage-binding limit");
  if (slots > contract.maximumSlots) throw new RangeError("Lattice visibility store exceeds the miss-list slot field");
  return { buckets, slots, bytes: slots * contract.slotBytes + 4 * contract.headerWords + contract.recordHeaderBytes };
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
// The tag is never zero, the empty-slot value. Each bucket choice uses its own
// mix so neither shares the tag's bits; a coinciding second choice moves on.
fn svoLatticeHash(key:vec4u)->u32{return svoLatticeMix(key.x^svoLatticeMix(key.y^svoLatticeMix(key.z^svoLatticeMix(key.w))));}
fn svoLatticeTag(hash:u32)->u32{return hash|1u;}
fn svoLatticeBucket(hash:u32,buckets:u32,choice:u32)->u32{
  let first=svoLatticeMix(hash^0x68e31da4u)%buckets;if(choice==0u){return first;}
  let second=svoLatticeMix(hash^0x2c1b3c6du)%buckets;return select(second,(first+1u)%buckets,second==first);
}
// Never zero, the not-valid value. The container enters here rather than in
// the host generation because the AO radius reads it from the frame uniforms.
fn svoLatticeStamp(generation:u32,container:vec3f)->u32{
  let bits=bitcast<vec3u>(container);return svoLatticeMix(generation^svoLatticeMix(bits.x^svoLatticeMix(bits.y^svoLatticeMix(bits.z))))|1u;
}
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
    { binding: 8, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "2d" } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "uint", viewDimension: "2d" } },
    { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
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

/** Key-pass store: buckets, key/visibility records, the control header and miss list, then the per-pixel corners. */
export function svoLatticeKeyBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { buckets, records, control, corners } = SVO_LATTICE_VISIBILITY_CONTRACT.keyBindings;
  return [buckets, records, control, corners].map((binding) => ({
    binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const },
  }));
}

/** Read-only corners and records appended to the split lighting group. */
export function svoLatticeLookupBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { corners, records } = SVO_LATTICE_VISIBILITY_CONTRACT.lookupBindings;
  return [corners, records].map((binding) => ({
    binding, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" as const },
  }));
}

/**
 * The lattice worker reads its misses from control and writes each record's
 * visibility word in place. It advances the frame counter in control too, so
 * no dispatch reads its arguments from a buffer the key pass wrote.
 */
export function svoLatticeWorkerBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ];
}

/** Packing shared by the screen reducer and the lattice worker. */
const svoConeFanoutPackWGSL = /* wgsl */ `
const FANOUT_INVALID_PACKED:vec2u=vec2u(0xffffffffu,0xfffffffeu);
fn fanoutQuantize7(value:f32)->u32{return u32(round(clamp(value,0.0,1.0)*127.0));}
fn fanoutPack(data0:vec4f,data1:vec4f,data2:vec4f)->vec2u{
  let light3=fanoutQuantize7(data1.x);
  let word0=u32(round(clamp(data0.x,0.0,1.0)*255.0))|(fanoutQuantize7(data0.y)<<8u)|(fanoutQuantize7(data0.z)<<15u)
    |(fanoutQuantize7(data0.w)<<22u)|((light3&7u)<<29u);
  let word1=(light3>>3u)|(fanoutQuantize7(data1.y)<<4u)|(fanoutQuantize7(data1.z)<<11u)
    |(fanoutQuantize7(data1.w)<<18u)|(fanoutQuantize7(data2.x)<<25u);
  return vec2u(word0,word1);
}
`;

/**
 * One reduction body for the screen and lattice sources: the geometry-miss
 * key, the invalid key, or the packed visibility, in the original summation
 * order. Only where a lane is read from differs.
 */
function svoConeFanoutReduceWGSL(signature: string, load: (layer: string) => string): string {
  // The body's locals must not shadow a name the load expression uses: the
  // lattice load reads its entry's lane base, so a local called base would
  // silently index another entry's lanes.
  return /* wgsl */ `
fn ${signature}->vec2u{
  if(${load("0u")}==FANOUT_GEOMETRY_MISS){return vec2u(0xffffffffu);}
  var visibility0=vec4f(1.0);var visibility1=vec4f(1.0);var visibility2=vec4f(1.0);
  if(${load("0u")}!=FANOUT_INACTIVE){
    var ao=0.0;var aoSampleCount=0u;
    for(var sample=0u;sample<FANOUT_AO_LAYERS;sample+=1u){
      let value=${load("sample")};if(value==FANOUT_INACTIVE){break;}if(value==FANOUT_INVALID){return FANOUT_INVALID_PACKED;}ao+=value;aoSampleCount+=1u;
    }
    if(aoSampleCount>0u){visibility0.x=clamp(ao/f32(aoSampleCount),0.0,1.0);}
  }
  for(var light=0u;light<${SVO_CONE_FANOUT_CONTRACT.maximumLights}u;light+=1u){
    if(light>=fanout.activity.x){break;}let lightLayer=FANOUT_LIGHT_BASE+light;var value=${load("lightLayer")};
    if(value==FANOUT_INACTIVE){continue;}if(value==FANOUT_INVALID){return FANOUT_INVALID_PACKED;}var sampleCount=1u;
    if(fanout.activity.y!=0u){let second=${load(`${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u+light`)};if(second==FANOUT_INVALID){return FANOUT_INVALID_PACKED;}if(second!=FANOUT_INACTIVE){value+=second;sampleCount=2u;}}
    let packed=clamp(value/f32(sampleCount),0.0,1.0);
    if(light<3u){visibility0[1u+light]=packed;}else if(light<7u){visibility1[light-3u]=packed;}else{visibility2.x=packed;}
  }
  return fanoutPack(visibility0,visibility1,visibility2);
}
`;
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
 * deliberately omits traversal, material, rigid-body, and GI resources.
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
  fluidClipMinimum:vec4u,
  fluidClipMaximum:vec4u,
  nodeMipReserved:vec4u,
  tetrahedralRadiance:vec4u,
  nodeMipExtent:vec4f,
  giLighting:vec4f,
  giCones:vec4f,
  rigidBounds:vec4f,
  primitiveCandidates:vec4u,
  structureOffsets:vec4u,
  derivedTraversal:vec4u,
  lod:vec4f,
  payloadLanes:vec4u,
  payloadLanes1:vec4u,
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
@group(0) @binding(8) var nodeMipPageTable:texture_2d<u32>;
@group(0) @binding(9) var nodeMipPageValidity:texture_2d<u32>;
@group(0) @binding(10) var<storage,read> scenePayload:array<u32>;
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
// Only dense payloads carry the backdrop table tail. Other layouts and older
// publications without a tail describe an empty backdrop to this worker.
fn fanoutBackdropWord(index:u32)->u32{
  if((dry.payloadLanes1.w&255u)!=0u||index>=arrayLength(&scenePayload)){return 0u;}
  return scenePayload[index];
}
${backdropTerrainWGSL({ load: (index) => `fanoutBackdropWord(${index})`, tableBase: "dry.payloadLanes1.y+dry.payloadLanes1.z" })}
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
  let minimum=dry.nodeMipOrigin.xyz;
  let maximum=minimum+dry.nodeMipExtent.xyz;
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
  if(maximumDistance<=0.0){return 1.0;}
  let biased=fanoutBiasedOrigin(position,normal,sample.towardLight);
  let rayMaximum=max(0.0,maximumDistance-dot(biased.xyz-position,sample.towardLight));
  let coneCell=max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z));
  let coneEscape=max(coneCell,backdropStoredVoxelWidth(backdropStoredLattice(),position))*dry.tuningRays1.z;
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
${svoConeFanoutPackWGSL}
@group(1) @binding(4) var<storage,read_write> latticeRecords:array<vec4u>;
@group(1) @binding(5) var<storage,read_write> latticeControl:array<u32>;
const LATTICE_HEADER_WORDS:u32=${SVO_LATTICE_VISIBILITY_CONTRACT.headerWords}u;
const LATTICE_LANE_SHIFT:u32=${SVO_LATTICE_VISIBILITY_CONTRACT.laneShift}u;
const LATTICE_SLOT_MASK:u32=${(1 << SVO_LATTICE_VISIBILITY_CONTRACT.laneShift) - 1}u;
const FANOUT_SECONDARY_BASE:u32=${SVO_CONE_FANOUT_CONTRACT.secondaryLightLayerBase}u;
var<workgroup> latticeLanes:array<f32,${SVO_LATTICE_VISIBILITY_CONTRACT.marchWorkgroupSize}>;
var<workgroup> latticeMissCount:u32;
// Lanes are the AO samples, the primary light samples, then the secondary
// ones when the frame has them: the fan-out layers minus the inactive tail.
fn latticeLaneLayer(lane:u32)->u32{let primary=FANOUT_AO_LAYERS+fanout.activity.x;return select(FANOUT_SECONDARY_BASE+lane-primary,lane,lane<primary);}
fn latticeLoad(base:u32,layer:u32)->f32{return latticeLanes[base+select(FANOUT_AO_LAYERS+fanout.activity.x+layer-FANOUT_SECONDARY_BASE,layer,layer<FANOUT_SECONDARY_BASE)];}
${svoConeFanoutReduceWGSL("latticeReduce(base:u32)", (layer) => `latticeLoad(base,${layer})`)}
// Bit group 0 is AO, bits 0..7 of the packed pair; group 1 + l is light l, bits 8+7l..14+7l.
fn latticeLaneGroup(layer:u32)->u32{return select(1u+layer-select(FANOUT_LIGHT_BASE,FANOUT_SECONDARY_BASE,layer>=FANOUT_SECONDARY_BASE),0u,layer<FANOUT_AO_LAYERS);}
fn latticeWordBits(first:i32,end:i32)->u32{
  let low=u32(clamp(first,0,32));let high=u32(clamp(end,0,32));if(high<=low){return 0u;}
  return select((1u<<high)-1u,0xffffffffu,high>=32u)&~((1u<<low)-1u);
}
fn latticeGroupBits(group:u32)->vec2u{
  let first=select(8+7*(i32(group)-1),0,group==0u);let end=first+select(7,8,group==0u);
  return vec2u(latticeWordBits(first,end),latticeWordBits(first-32,end-32));
}
// Every miss the key pass appended is marched this frame, over a fixed direct
// grid that strides through the list. A workgroup takes floor(64 / lanes)
// misses at a time, one thread per (miss, lane), and the first lane of each
// reduces. The sample is rebuilt from the key, so it does not depend on which
// pixel requested it. Groups outside the miss's mask keep the record's bits.
// An invalid march stores stamp zero: the lookup reads it as absent and the
// next frame's request marches it again.
@compute @workgroup_size(${SVO_LATTICE_VISIBILITY_CONTRACT.marchWorkgroupSize})
fn svoLatticeConeMarch(@builtin(local_invocation_index) thread:u32,@builtin(workgroup_id) groupId:vec3u){
  if(thread==0u){
    latticeMissCount=min(latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.missCountWord}],arrayLength(&latticeControl)-LATTICE_HEADER_WORDS);
    // The key pass is done with this frame's number. The key pass adds one,
    // so zero stays the empty-slot frame and 24 bits hold every value.
    if(groupId.x==0u){latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.frameWord}]=(latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.frameWord}]+1u)%0xfffffeu;}
  }
  let misses=workgroupUniformLoad(&latticeMissCount);
  let lanes=FANOUT_AO_LAYERS+fanout.activity.x*select(1u,2u,fanout.activity.y!=0u);
  let perBatch=${SVO_LATTICE_VISIBILITY_CONTRACT.marchWorkgroupSize}u/lanes;let entry=thread/lanes;let lane=thread%lanes;
  let stamp=svoLatticeStamp(latticeControl[${SVO_LATTICE_VISIBILITY_CONTRACT.generationWord}],uniforms.container.xyz);
  let batches=(misses+perBatch-1u)/perBatch;
  for(var batch=groupId.x;batch<batches;batch+=${SVO_LATTICE_VISIBILITY_CONTRACT.marchWorkgroups}u){
    let index=batch*perBatch+entry;let pending=entry<perBatch&&index<misses;var miss=0u;
    if(pending){
      miss=latticeControl[LATTICE_HEADER_WORDS+index];let layer=latticeLaneLayer(lane);var value=FANOUT_INACTIVE;
      if(((miss>>LATTICE_LANE_SHIFT)&(1u<<latticeLaneGroup(layer)))!=0u){
        let receiver=svoLatticeReceiver(latticeRecords[2u*(miss&LATTICE_SLOT_MASK)],dry.mapping.worldOrigin,dry.mapping.cellSize);
        value=fanoutConeSample(receiver.position,receiver.normal,receiver.featureId,layer);
      }
      latticeLanes[thread]=value;
    }
    workgroupBarrier();
    if(pending&&lane==0u){
      let slot=miss&LATTICE_SLOT_MASK;var packed=latticeReduce(entry*lanes);let valid=!all(packed==FANOUT_INVALID_PACKED);
      var keep=vec2u(0u);
      for(var bits=0u;bits<=${SVO_CONE_FANOUT_CONTRACT.maximumLights}u;bits+=1u){if(((miss>>LATTICE_LANE_SHIFT)&(1u<<bits))==0u){keep|=latticeGroupBits(bits);}}
      if(valid){packed=(packed&~keep)|(latticeRecords[2u*slot+1u].xy&keep);}
      latticeRecords[2u*slot+1u]=vec4u(packed,select(0u,stamp,valid),0u);
    }
    workgroupBarrier();
  }
}
`;
}

/** Minimal second pass; it preserves the original sample addition and packing order. */
export const svoConeFanoutReducerWGSL = /* wgsl */ `
const FANOUT_GEOMETRY_MISS:f32=${SVO_CONE_FANOUT_SENTINELS.geometryMiss}.0;
const FANOUT_INACTIVE:f32=${SVO_CONE_FANOUT_SENTINELS.inactive}.0;
const FANOUT_INVALID:f32=${SVO_CONE_FANOUT_SENTINELS.invalid}.0;
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
${svoConeFanoutPackWGSL}
${svoConeFanoutReduceWGSL("fanoutReduce(coordinate:vec2i)", (layer) => `fanoutLoad(coordinate,${layer})`)}
@compute @workgroup_size(${SVO_CONE_FANOUT_CONTRACT.workgroupSize.join(",")})
fn svoConeFanoutReduce(@builtin(global_invocation_id) gid:vec3u){
  if(any(gid.xy>=fanout.dimensions)){return;}let coordinate=vec2i(gid.xy);
  textureStore(fanoutVisibility,coordinate,vec4u(fanoutReduce(coordinate),0u,0u));
}
`;
