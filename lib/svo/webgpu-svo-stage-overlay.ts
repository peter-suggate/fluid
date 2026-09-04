import {
  SVO_GBUFFER_PRODUCERS,
  SVO_GBUFFER_PRODUCER_MASK,
  SVO_GBUFFER_PRODUCER_SHIFT,
} from "./svo-gbuffer";
import {
  SVO_RENDER_STAGE_CLAIMANT_LEGEND,
  SVO_PRIMARY_WORK_REFERENCE,
  SVO_RENDER_STAGE_SEQUENTIAL_LEGEND,
  svoRenderStageCode,
  type SvoRenderStageLegendStop,
  type SvoRenderStageView,
} from "./svo-render-diagnostics";
import { unifiedDisplayTransferShaderLibrary } from "../core/webgpu-lighting";

/**
 * Render-stage overlay.
 *
 * One full-screen pass encoded after the optical composite that replaces the
 * presented image with a decode of a plane some earlier pass wrote. Its own
 * bindings stay read-only. Ordinary stage views consume production planes;
 * the explicitly labelled primary-work family requests a counter publication
 * from traced visibility and therefore describes that diagnostic arm, including
 * its write overhead, rather than claiming to be a free view of shipping work.
 *
 * Planes that a given configuration never allocates (no glass discovery, no
 * rigid bodies, cone lighting at full rate) bind 1x1 fallbacks of the matching
 * format, so the bind group is complete whatever the frame contains and the
 * views that need a missing plane simply report it as absent.
 */
export const SVO_RENDER_STAGE_OVERLAY_CONTRACT = Object.freeze({
  entryPoints: Object.freeze({ vertex: "svoStageOverlayVertex" as const, fragment: "svoStageOverlayFragment" as const }),
  /** control(vec4u) + extent(vec4f). */
  paramsBytes: 32,
  bindings: Object.freeze({
    params: 0,
    packedSurface: 1,
    identityMedia: 2,
    hardwareDepth: 3,
    splitGeometry: 4,
    splitOpaqueIdentity: 5,
    splitGlassKey: 6,
    rigidPrimaryGeometry: 7,
    conePrepassVisibility: 8,
    conePrepassGeometry: 9,
    conePrepassIdentity: 10,
    conePrepassRadiance: 11,
    sceneRadiance: 12,
    primaryWork: 13,
  }),
  formats: Object.freeze({
    packedSurface: "rgba32uint" as GPUTextureFormat,
    identityMedia: "rgba16uint" as GPUTextureFormat,
    hardwareDepth: "depth32float" as GPUTextureFormat,
    splitGeometry: "rgba32float" as GPUTextureFormat,
    splitOpaqueIdentity: "rg32uint" as GPUTextureFormat,
    splitGlassKey: "r32uint" as GPUTextureFormat,
    rigidPrimaryGeometry: "rg32uint" as GPUTextureFormat,
    conePrepassVisibility: "rg32uint" as GPUTextureFormat,
    conePrepassGeometry: "rgba16float" as GPUTextureFormat,
    conePrepassIdentity: "r32uint" as GPUTextureFormat,
    conePrepassRadiance: "rgba16float" as GPUTextureFormat,
    sceneRadiance: "rgba16float" as GPUTextureFormat,
    primaryWork: "rgba32uint" as GPUTextureFormat,
  }),
});

/** The planes a frame can offer. Absent entries fall back to a 1x1 texture. */
export interface SvoRenderStagePlanes {
  readonly packedSurface?: GPUTextureView;
  readonly identityMedia?: GPUTextureView;
  readonly hardwareDepth?: GPUTextureView;
  readonly splitGeometry?: GPUTextureView;
  readonly splitOpaqueIdentity?: GPUTextureView;
  readonly splitGlassKey?: GPUTextureView;
  readonly rigidPrimaryGeometry?: GPUTextureView;
  readonly conePrepassVisibility?: GPUTextureView;
  readonly conePrepassGeometry?: GPUTextureView;
  readonly conePrepassIdentity?: GPUTextureView;
  readonly conePrepassRadiance?: GPUTextureView;
  readonly sceneRadiance?: GPUTextureView;
  /** Exact invocation-private primary traversal counters, when explicitly requested. */
  readonly primaryWork?: GPUTextureView;
  /** Cone planes are allocated at a reduced rate and drawn at it, not upsampled. */
  readonly conePrepassWidth?: number;
  readonly conePrepassHeight?: number;
}

export function svoRenderStageOverlayBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_RENDER_STAGE_OVERLAY_CONTRACT;
  const visibility = GPUShaderStage.FRAGMENT;
  const uint = { sampleType: "uint" as const };
  const float = { sampleType: "unfilterable-float" as const };
  return [
    { binding: bindings.params, visibility, buffer: { type: "uniform" } },
    { binding: bindings.packedSurface, visibility, texture: uint },
    { binding: bindings.identityMedia, visibility, texture: uint },
    { binding: bindings.hardwareDepth, visibility, texture: { sampleType: "depth" } },
    { binding: bindings.splitGeometry, visibility, texture: float },
    { binding: bindings.splitOpaqueIdentity, visibility, texture: uint },
    { binding: bindings.splitGlassKey, visibility, texture: uint },
    { binding: bindings.rigidPrimaryGeometry, visibility, texture: uint },
    { binding: bindings.conePrepassVisibility, visibility, texture: uint },
    { binding: bindings.conePrepassGeometry, visibility, texture: float },
    { binding: bindings.conePrepassIdentity, visibility, texture: uint },
    { binding: bindings.conePrepassRadiance, visibility, texture: float },
    { binding: bindings.sceneRadiance, visibility, texture: float },
    { binding: bindings.primaryWork, visibility, texture: uint },
  ];
}

function wgslColor(stop: SvoRenderStageLegendStop): string {
  const hex = stop.color.slice(1);
  const channel = (index: number) => (Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16) / 255).toFixed(4);
  return `vec3f(${channel(0)},${channel(1)},${channel(2)})`;
}

/** Both palettes are generated from the published legends, never re-authored. */
function rampWGSL(): string {
  const stops = SVO_RENDER_STAGE_SEQUENTIAL_LEGEND;
  const segments = stops.slice(1).map((stop, index) => {
    const previous = stops[index];
    const span = (stop.at - previous.at).toFixed(4);
    return `if(value<${stop.at.toFixed(4)}){return mix(${wgslColor(previous)},${wgslColor(stop)},(value-${previous.at.toFixed(4)})/${span});}`;
  });
  return `fn svoStageRamp(valueIn:f32)->vec3f{
  let value=clamp(valueIn,0.0,1.0);
  ${segments.join("\n  ")}
  return ${wgslColor(stops[stops.length - 1])};
}`;
}

function claimantWGSL(): string {
  const colors = SVO_RENDER_STAGE_CLAIMANT_LEGEND.map(wgslColor).join(",");
  return `fn svoStageClaimantColor(index:u32)->vec3f{
  var palette=array<vec3f,${SVO_RENDER_STAGE_CLAIMANT_LEGEND.length}>(${colors});
  return palette[min(index,${SVO_RENDER_STAGE_CLAIMANT_LEGEND.length - 1}u)];
}`;
}

const view = (name: SvoRenderStageView) => `${svoRenderStageCode(name)}u`;

export function createSvoRenderStageOverlayWGSL(): string {
  const { bindings, entryPoints } = SVO_RENDER_STAGE_OVERLAY_CONTRACT;
  return /* wgsl */ `
struct SvoStageParams{
  // view, light slot, cone prepass width, cone prepass height
  control:vec4u,
  // target width, target height, far distance in metres, unused
  extent:vec4f,
}
@group(0) @binding(${bindings.params}) var<uniform> stage:SvoStageParams;
@group(0) @binding(${bindings.packedSurface}) var stagePackedSurface:texture_2d<u32>;
@group(0) @binding(${bindings.identityMedia}) var stageIdentityMedia:texture_2d<u32>;
@group(0) @binding(${bindings.hardwareDepth}) var stageHardwareDepth:texture_depth_2d;
@group(0) @binding(${bindings.splitGeometry}) var stageSplitGeometry:texture_2d<f32>;
@group(0) @binding(${bindings.splitOpaqueIdentity}) var stageSplitIdentity:texture_2d<u32>;
@group(0) @binding(${bindings.splitGlassKey}) var stageGlassKey:texture_2d<u32>;
@group(0) @binding(${bindings.rigidPrimaryGeometry}) var stageRigidGeometry:texture_2d<u32>;
@group(0) @binding(${bindings.conePrepassVisibility}) var stageConeVisibility:texture_2d<u32>;
@group(0) @binding(${bindings.conePrepassGeometry}) var stageConeGeometry:texture_2d<f32>;
@group(0) @binding(${bindings.conePrepassIdentity}) var stageConeIdentity:texture_2d<u32>;
@group(0) @binding(${bindings.conePrepassRadiance}) var stageConeRadiance:texture_2d<f32>;
@group(0) @binding(${bindings.sceneRadiance}) var stageSceneRadiance:texture_2d<f32>;
@group(0) @binding(${bindings.primaryWork}) var stagePrimaryWork:texture_2d<u32>;

${unifiedDisplayTransferShaderLibrary}
${rampWGSL()}
${claimantWGSL()}

const SVO_STAGE_MISS_DISTANCE:f32=3.402823e38;
const SVO_STAGE_ABSENT:vec3f=vec3f(.055,.055,.075);

struct SvoStageOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn ${entryPoints.vertex}(@builtin(vertex_index) index:u32)->SvoStageOut{
  var positions=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  var output:SvoStageOut;
  output.position=vec4f(positions[index],0.0,1.0);
  output.uv=positions[index]*.5+vec2f(.5);
  return output;
}

/** Golden-ratio hash so neighbouring ids never land on neighbouring hues. */
fn svoStageHueColor(hue:f32,saturation:f32,value:f32)->vec3f{
  let k=(vec3f(5.0,3.0,1.0)+hue*6.0)%vec3f(6.0);
  return value-value*saturation*max(vec3f(0.0),min(min(k,4.0-k),vec3f(1.0)));
}
fn svoStageIdentityColor(id:u32)->vec3f{
  var hash=id*2654435761u;
  hash^=hash>>16u; hash*=2246822519u; hash^=hash>>13u;
  return svoStageHueColor(f32(hash&1023u)/1024.0,
    .55+.35*f32((hash>>10u)&7u)/7.0, .62+.38*f32((hash>>13u)&7u)/7.0);
}
fn svoStageNormalColor(normal:vec3f)->vec3f{return normalize(normal)*.5+vec3f(.5);}

fn svoStageMetadata(coordinate:vec2i)->vec4u{
  let packed=textureLoad(stagePackedSurface,coordinate,0);
  let word=packed.w;
  // field source, flags, failure, feature
  return vec4u(word&15u,(word>>4u)&0xffffu,(word>>20u)&255u,word>>28u);
}
fn svoStageDecodeOct8(packed:u32)->vec3f{
  var result=vec3f(f32(i32(packed<<24u)>>24)/127.0,f32(i32(packed<<16u)>>24)/127.0,0.0);
  result.z=1.0-abs(result.x)-abs(result.y);
  if(result.z<0.0){
    let adjustment=-result.z;
    result.x+=select(-adjustment,adjustment,result.x<0.0);
    result.y+=select(-adjustment,adjustment,result.y<0.0);
  }
  return normalize(result);
}
/** Mirrors the rigid pass's own 12-bit octahedral unpack, bit for bit. */
fn svoStageRigidNormal(packed:u32)->vec3f{
  let encoded=vec2f(f32(packed&0xfffu),f32((packed>>12u)&0xfffu))/4095.0*2.0-1.0;
  var normal=vec3f(encoded,1.0-abs(encoded.x)-abs(encoded.y));
  if(normal.z<0.0){
    let original=normal.xy;
    normal=vec3f((vec2f(1.0)-abs(original.yx))*select(vec2f(-1.0),vec2f(1.0),original>=vec2f(0.0)),normal.z);
  }
  return normalize(normal);
}
fn svoStageDecodeConeNormal(oct:vec2f)->vec3f{
  var normal=vec3f(oct,1.0-abs(oct.x)-abs(oct.y));
  if(normal.z<0.0){
    let folded=(vec2f(1.0)-abs(normal.yx))*select(vec2f(-1.0),vec2f(1.0),normal.xy>=vec2f(0.0));
    normal=vec3f(folded,normal.z);
  }
  return normalize(normal);
}
/** Reduced-rate planes are read at their own rate, so their blocks stay visible. */
fn svoStageConeCoordinate(coordinate:vec2i)->vec2i{
  let reduced=vec2f(f32(stage.control.z),f32(stage.control.w));
  let scaled=vec2f(coordinate)*reduced/max(stage.extent.xy,vec2f(1.0));
  return vec2i(clamp(scaled,vec2f(0.0),reduced-vec2f(1.0)));
}
fn svoStageConeAvailable()->bool{return stage.control.z>1u&&stage.control.w>1u;}
/** Channel 0 is ambient visibility; channels 1..8 are the cached light slots. */
fn svoStageConeChannel(packed:vec2u,channel:u32)->f32{
  if(channel==0u){return f32(packed.x&255u)/255.0;}
  if(channel==1u){return f32((packed.x>>8u)&127u)/127.0;}
  if(channel==2u){return f32((packed.x>>15u)&127u)/127.0;}
  if(channel==3u){return f32((packed.x>>22u)&127u)/127.0;}
  if(channel==4u){return f32(((packed.x>>29u)&7u)|((packed.y&15u)<<3u))/127.0;}
  if(channel==5u){return f32((packed.y>>4u)&127u)/127.0;}
  if(channel==6u){return f32((packed.y>>11u)&127u)/127.0;}
  if(channel==7u){return f32((packed.y>>18u)&127u)/127.0;}
  return f32((packed.y>>25u)&127u)/127.0;
}
fn svoStageDistanceRamp(distance_m:f32)->vec3f{
  if(!(distance_m<SVO_STAGE_MISS_DISTANCE)||distance_m<=0.0){return ${wgslColor(SVO_RENDER_STAGE_CLAIMANT_LEGEND[4])};}
  return svoStageRamp(clamp(distance_m/max(stage.extent.z,1e-3),0.0,1.0));
}

fn svoStagePrimaryWorkRecord(coordinate:vec2i)->vec4u{
  return textureLoad(stagePrimaryWork,coordinate,0);
}
fn svoStagePrimaryNodeVisits(record:vec4u)->u32{return record.x&0xffffu;}
fn svoStagePrimaryLeafVisits(record:vec4u)->u32{return record.y&0xffffu;}
fn svoStagePrimaryVoxelCells(record:vec4u)->u32{return record.z;}
fn svoStagePrimaryWorkColor(value:u32,referenceP99:u32)->vec3f{
  return svoStageRamp(f32(value)/max(1.0,f32(referenceP99)));
}
fn svoStagePrimaryEntryTailColor(record:vec4u)->vec3f{
  let work=svoStagePrimaryNodeVisits(record)+svoStagePrimaryVoxelCells(record);
  if(work>=${SVO_PRIMARY_WORK_REFERENCE.total}u){return vec3f(1.0);}
  let seed=(record.x>>16u)&3u;
  if(seed==1u){return vec3f(.0784,.1647,.2902);}
  if(seed==2u){return vec3f(0.0,.7373,.8314);}
  return vec3f(.3608,.2824,.5098);
}

fn svoStagePrimaryClaimant(coordinate:vec2i)->vec3f{
  let metadata=svoStageMetadata(coordinate);
  let flags=metadata.y;
  if((flags&${1 << 0}u)==0u){return svoStageClaimantColor(0u);}
  let producer=(flags>>${SVO_GBUFFER_PRODUCER_SHIFT}u)&${SVO_GBUFFER_PRODUCER_MASK}u;
  if(producer==${SVO_GBUFFER_PRODUCERS.rasterBackground}u){return svoStageClaimantColor(1u);}
  if(producer==${SVO_GBUFFER_PRODUCERS.brickRaster}u){return svoStageClaimantColor(2u);}
  if(producer==${SVO_GBUFFER_PRODUCERS.scenePrimitiveRaster}u){return svoStageClaimantColor(3u);}
  if(producer==${SVO_GBUFFER_PRODUCERS.rigidImpostorRaster}u){return svoStageClaimantColor(4u);}
  if(producer==${SVO_GBUFFER_PRODUCERS.glassDiscovery}u){return svoStageClaimantColor(5u);}
  if(producer==${SVO_GBUFFER_PRODUCERS.tracedPrimary}u){return svoStageClaimantColor(6u);}
  return svoStageClaimantColor(7u);
}

fn svoStageFailure(coordinate:vec2i)->vec3f{
  let metadata=svoStageMetadata(coordinate);
  let flags=metadata.y;
  let failure=metadata.z;
  if((flags&${1 << 12 | 1 << 13}u)!=0u||failure==2u||failure==3u){
    return select(${wgslColor(SVO_RENDER_STAGE_CLAIMANT_LEGEND[3])},vec3f(1.0,0.0,.72),failure==3u||(flags&${1 << 13}u)!=0u);
  }
  if((flags&${1 << 14 | 1 << 15}u)!=0u||failure>=4u){return vec3f(1.0);}
  if((flags&${1 << 0}u)!=0u){return vec3f(.024,.235,.204);}
  return vec3f(.043,.063,.125);
}

fn svoStageMotion(coordinate:vec2i)->vec3f{
  let packed=textureLoad(stagePackedSurface,coordinate,0);
  let metadata=svoStageMetadata(coordinate);
  if((metadata.y&${1 << 0}u)==0u){return SVO_STAGE_ABSENT;}
  if((metadata.y&${1 << 5}u)==0u){return vec3f(.227,.227,.267);}
  let word=packed.z;
  var velocity=vec3f(0.0);
  for(var axis=0u;axis<3u;axis+=1u){
    let bits=(word>>(axis*10u))&1023u;
    velocity[axis]=f32(select(i32(bits),i32(bits)-1024,bits>=512u))/511.0*64.0;
  }
  let speed=length(velocity);
  if(speed<1e-4){return vec3f(.227,.227,.267);}
  return svoStageNormalColor(velocity/speed)*clamp(.25+speed/8.0,0.0,1.0);
}

fn svoStageMedia(coordinate:vec2i)->vec3f{
  let metadata=svoStageMetadata(coordinate);
  if((metadata.y&${1 << 0}u)==0u){return SVO_STAGE_ABSENT;}
  let identity=textureLoad(stageIdentityMedia,coordinate,0);
  let mediumColor=array<vec3f,4>(vec3f(.043,.063,.125),vec3f(.043,.063,.125),vec3f(0.0,.898,1.0),vec3f(1.0,.522,0.0));
  return mix(mediumColor[min(identity.z,3u)],mediumColor[min(identity.w,3u)],.5);
}

@fragment fn ${entryPoints.fragment}(input:SvoStageOut)->@location(0) vec4f{
  let coordinate=vec2i(i32(input.position.x),i32(input.position.y));
  let mode=stage.control.x;
  var color=SVO_STAGE_ABSENT;
  if(mode==${view("primary-work")}){
    let record=svoStagePrimaryWorkRecord(coordinate);
    color=svoStagePrimaryWorkColor(svoStagePrimaryNodeVisits(record)+svoStagePrimaryVoxelCells(record),${SVO_PRIMARY_WORK_REFERENCE.total}u);
  }else if(mode==${view("primary-voxel-cells")}){
    color=svoStagePrimaryWorkColor(svoStagePrimaryVoxelCells(svoStagePrimaryWorkRecord(coordinate)),${SVO_PRIMARY_WORK_REFERENCE.voxelCells}u);
  }else if(mode==${view("primary-node-visits")}){
    color=svoStagePrimaryWorkColor(svoStagePrimaryNodeVisits(svoStagePrimaryWorkRecord(coordinate)),${SVO_PRIMARY_WORK_REFERENCE.nodeVisits}u);
  }else if(mode==${view("primary-leaf-visits")}){
    color=svoStagePrimaryWorkColor(svoStagePrimaryLeafVisits(svoStagePrimaryWorkRecord(coordinate)),${SVO_PRIMARY_WORK_REFERENCE.leafVisits}u);
  }else if(mode==${view("primary-entry-tail")}){
    color=svoStagePrimaryEntryTailColor(svoStagePrimaryWorkRecord(coordinate));
  }else if(mode==${view("pass-claimant")}){
    color=svoStagePrimaryClaimant(coordinate);
  }else if(mode==${view("primary-depth")}){
    color=svoStageDistanceRamp(textureLoad(stageSplitGeometry,coordinate,0).w);
  }else if(mode==${view("surface-normal")}){
    let metadata=svoStageMetadata(coordinate);
    color=select(SVO_STAGE_ABSENT,
      svoStageNormalColor(svoStageDecodeOct8(textureLoad(stagePackedSurface,coordinate,0).x&0xffffu)),
      (metadata.y&${1 << 0}u)!=0u);
  }else if(mode==${view("primary-failure")}){
    color=svoStageFailure(coordinate);
  }else if(mode==${view("publication-generation")}){
    color=svoStageIdentityColor(textureLoad(stagePackedSurface,coordinate,0).y);
  }else if(mode==${view("owner-identity")}){
    let metadata=svoStageMetadata(coordinate);
    let owner=textureLoad(stageIdentityMedia,coordinate,0).y;
    color=select(SVO_STAGE_ABSENT,svoStageIdentityColor(owner),(metadata.y&${1 << 0}u)!=0u);
  }else if(mode==${view("material-identity")}){
    let metadata=svoStageMetadata(coordinate);
    let material=textureLoad(stageIdentityMedia,coordinate,0).x;
    color=select(SVO_STAGE_ABSENT,svoStageIdentityColor(material),(metadata.y&${1 << 0}u)!=0u);
  }else if(mode==${view("media-stack")}){
    color=svoStageMedia(coordinate);
  }else if(mode==${view("surface-motion")}){
    color=svoStageMotion(coordinate);
  }else if(mode==${view("glass-discovery")}){
    let key=textureLoad(stageGlassKey,coordinate,0).x;
    color=select(vec3f(.031,.035,.055),svoStageIdentityColor(key),key>0u);
  }else if(mode==${view("rigid-impostor")}){
    let record=textureLoad(stageRigidGeometry,coordinate,0);
    // The impostor pass leaves an all-zero record wherever it claimed nothing,
    // which the certificate bridge reads as "no rigid winner here".
    color=select(vec3f(.031,.035,.055),svoStageNormalColor(svoStageRigidNormal(record.y)),record.x!=0u);
  }else if(mode==${view("cone-ambient-visibility")}){
    if(!svoStageConeAvailable()){
      color=SVO_STAGE_ABSENT;
    }else{
      let packed=textureLoad(stageConeVisibility,svoStageConeCoordinate(coordinate),0).xy;
      color=select(svoStageRamp(svoStageConeChannel(packed,0u)),SVO_STAGE_ABSENT,packed.x==0xffffffffu);
    }
  }else if(mode==${view("cone-light-visibility")}){
    if(!svoStageConeAvailable()){
      color=SVO_STAGE_ABSENT;
    }else{
      let packed=textureLoad(stageConeVisibility,svoStageConeCoordinate(coordinate),0).xy;
      color=select(svoStageRamp(svoStageConeChannel(packed,1u+stage.control.y)),SVO_STAGE_ABSENT,packed.x==0xffffffffu);
    }
  }else if(mode==${view("cone-radiance")}){
    color=select(SVO_STAGE_ABSENT,
      unifiedDisplayTransfer(textureLoad(stageConeRadiance,svoStageConeCoordinate(coordinate),0).rgb),
      svoStageConeAvailable());
  }else if(mode==${view("cone-geometry")}){
    if(!svoStageConeAvailable()){
      color=SVO_STAGE_ABSENT;
    }else{
      let geometry=textureLoad(stageConeGeometry,svoStageConeCoordinate(coordinate),0);
      let identity=textureLoad(stageConeIdentity,svoStageConeCoordinate(coordinate),0).x;
      // The prepass stores distance zero and an all-ones identity on a miss.
      color=select(svoStageNormalColor(svoStageDecodeConeNormal(geometry.yz))*.35
        +svoStageDistanceRamp(geometry.x)*.65,
        vec3f(.031,.035,.055),geometry.x<=0.0||identity==0xffffffffu);
    }
  }else if(mode==${view("dry-radiance")}){
    // Fidelity PNGs are written from this same HDR plane with clamp + gamma,
    // before the optical compositor applies its scene-owned ACES grade. Keep a
    // UI view of that exact contract so a progress capture is reproducible
    // rather than a look that exists only in a headless tool.
    color=pow(clamp(textureLoad(stageSceneRadiance,coordinate,0).rgb,vec3f(0.0),vec3f(1.0)),vec3f(1.0/2.2));
  }else if(mode==${view("water-depth")}){
    color=svoStageDistanceRamp(textureLoad(stageSceneRadiance,coordinate,0).a);
  }else if(mode==${view("lighting-partition")}){
    // Reversed-Z clears to zero, so a nonzero device depth is exactly the set
    // the deferred draw claimed and zero is exactly the sky draw's.
    let device=textureLoad(stageHardwareDepth,coordinate,0);
    let metadata=svoStageMetadata(coordinate);
    color=select(vec3f(.106,.165,.420),vec3f(0.0,1.0,.522),device>0.0);
    if(device>0.0&&(metadata.y&${1 << 0}u)==0u){color=vec3f(1.0,.09,.302);}
    if(device<=0.0&&(metadata.y&${1 << 0}u)!=0u){color=vec3f(1.0,.09,.302);}
  }
  return vec4f(color,1.0);
}
`;
}

/**
 * Owns the overlay pipeline and its fallback planes. The renderer hands it
 * whichever views exist this frame; everything else resolves to a 1x1 texture
 * of the right format so the bind group never becomes invalid.
 */
export class SparseVoxelRenderStageOverlay {
  private readonly params: GPUBuffer;
  private readonly layout: GPUBindGroupLayout;
  private readonly fallbacks = new Map<string, GPUTextureView>();
  private pipeline?: GPURenderPipeline;
  private bindGroup?: GPUBindGroup;
  private bindGroupKey = "";
  private readonly scratch = new ArrayBuffer(SVO_RENDER_STAGE_OVERLAY_CONTRACT.paramsBytes);

  constructor(private readonly device: GPUDevice, private readonly targetFormat: GPUTextureFormat) {
    this.params = device.createBuffer({
      label: "Sparse voxel render-stage overlay parameters",
      size: SVO_RENDER_STAGE_OVERLAY_CONTRACT.paramsBytes,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.layout = device.createBindGroupLayout({
      label: "Sparse voxel render-stage overlay bindings",
      entries: svoRenderStageOverlayBindGroupLayoutEntries(),
    });
  }

  async initialize(): Promise<void> {
    if (this.pipeline) return;
    const shaderModule = this.device.createShaderModule({
      label: "Sparse voxel render-stage overlay",
      code: createSvoRenderStageOverlayWGSL(),
    });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(`Sparse voxel render-stage overlay:\n${errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n")}`);
    }
    const { entryPoints } = SVO_RENDER_STAGE_OVERLAY_CONTRACT;
    this.pipeline = await this.device.createRenderPipelineAsync({
      label: "Sparse voxel render-stage overlay",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module: shaderModule, entryPoint: entryPoints.vertex },
      fragment: { module: shaderModule, entryPoint: entryPoints.fragment, targets: [{ format: this.targetFormat }] },
      primitive: { topology: "triangle-list" },
    });
  }

  get ready(): boolean {
    return Boolean(this.pipeline);
  }

  /**
   * Draws the selected view over `target`. Returns false when the overlay is
   * off or unavailable, which leaves the composited frame untouched.
   */
  encode(
    encoder: GPUCommandEncoder,
    target: GPUTextureView,
    stageView: SvoRenderStageView,
    lightSlot: number,
    width: number,
    height: number,
    farDistance_m: number,
    planes: SvoRenderStagePlanes,
  ): boolean {
    if (!this.pipeline || stageView === "off") return false;
    const bindGroup = this.ensureBindGroup(planes);
    const control = new Uint32Array(this.scratch, 0, 4);
    const extent = new Float32Array(this.scratch, 16, 4);
    control.set([
      svoRenderStageCode(stageView), Math.max(0, Math.round(lightSlot)),
      Math.max(1, Math.round(planes.conePrepassWidth ?? 1)), Math.max(1, Math.round(planes.conePrepassHeight ?? 1)),
    ]);
    extent.set([width, height, Math.max(1e-3, farDistance_m), 0]);
    this.device.queue.writeBuffer(this.params, 0, this.scratch);
    const pass = encoder.beginRenderPass({
      label: "Sparse voxel render-stage overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(3);
    pass.end();
    return true;
  }

  destroy(): void {
    this.params.destroy();
    this.fallbacks.clear();
    this.pipeline = undefined;
    this.bindGroup = undefined;
    this.bindGroupKey = "";
  }

  private fallback(format: GPUTextureFormat, depth = false): GPUTextureView {
    const cached = this.fallbacks.get(format);
    if (cached) return cached;
    const texture = this.device.createTexture({
      label: `Sparse voxel render-stage overlay ${format} fallback`,
      size: [1, 1],
      format,
      usage: GPUTextureUsage.TEXTURE_BINDING | (depth ? GPUTextureUsage.RENDER_ATTACHMENT : GPUTextureUsage.COPY_DST),
    });
    const created = texture.createView();
    this.fallbacks.set(format, created);
    return created;
  }

  private ensureBindGroup(planes: SvoRenderStagePlanes): GPUBindGroup {
    const { bindings, formats } = SVO_RENDER_STAGE_OVERLAY_CONTRACT;
    const resolved: [number, GPUTextureView][] = [
      [bindings.packedSurface, planes.packedSurface ?? this.fallback(formats.packedSurface)],
      [bindings.identityMedia, planes.identityMedia ?? this.fallback(formats.identityMedia)],
      [bindings.hardwareDepth, planes.hardwareDepth ?? this.fallback(formats.hardwareDepth, true)],
      [bindings.splitGeometry, planes.splitGeometry ?? this.fallback(formats.splitGeometry)],
      [bindings.splitOpaqueIdentity, planes.splitOpaqueIdentity ?? this.fallback(formats.splitOpaqueIdentity)],
      [bindings.splitGlassKey, planes.splitGlassKey ?? this.fallback(formats.splitGlassKey)],
      [bindings.rigidPrimaryGeometry, planes.rigidPrimaryGeometry ?? this.fallback(formats.rigidPrimaryGeometry)],
      [bindings.conePrepassVisibility, planes.conePrepassVisibility ?? this.fallback(formats.conePrepassVisibility)],
      [bindings.conePrepassGeometry, planes.conePrepassGeometry ?? this.fallback(formats.conePrepassGeometry)],
      [bindings.conePrepassIdentity, planes.conePrepassIdentity ?? this.fallback(formats.conePrepassIdentity)],
      [bindings.conePrepassRadiance, planes.conePrepassRadiance ?? this.fallback(formats.conePrepassRadiance)],
      [bindings.sceneRadiance, planes.sceneRadiance ?? this.fallback(formats.sceneRadiance)],
      [bindings.primaryWork, planes.primaryWork ?? this.fallback(formats.primaryWork)],
    ];
    // Views are stable per allocation, so identity is a sufficient cache key and
    // a resize or a republished plane rebuilds the group exactly once.
    const key = resolved.map(([binding, resource]) => `${binding}:${this.viewKey(resource)}`).join("|");
    if (this.bindGroup && key === this.bindGroupKey) return this.bindGroup;
    this.bindGroup = this.device.createBindGroup({
      label: "Sparse voxel render-stage overlay binding",
      layout: this.layout,
      entries: [
        { binding: bindings.params, resource: { buffer: this.params } },
        ...resolved.map(([binding, resource]) => ({ binding, resource })),
      ],
    });
    this.bindGroupKey = key;
    return this.bindGroup;
  }

  private readonly viewIds = new WeakMap<GPUTextureView, number>();
  private nextViewId = 0;

  private viewKey(resource: GPUTextureView): number {
    const existing = this.viewIds.get(resource);
    if (existing !== undefined) return existing;
    this.nextViewId += 1;
    this.viewIds.set(resource, this.nextViewId);
    return this.nextViewId;
  }
}
