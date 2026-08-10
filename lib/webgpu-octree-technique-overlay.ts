/** Slice and full-volume paper-technique views for the unified power/octree method. */

import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import {
  OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM,
  OCTREE_TECHNIQUE_OVERLAY_CODES,
  OCTREE_TECHNIQUE_PROGRAMS,
  octreeTechniqueProgramForCode,
  type OctreeTechniqueDebugSource,
  type OctreeTechniqueProgramId,
} from "./octree-technique-debug";
import {
  visualizationBindGroupEntries,
  visualizationBindingPreambleWGSL,
  type VisualizationProgram,
} from "./visualization-bindings";
import { octreeTechniqueSharedWGSL } from "./webgpu-octree-technique-shared";
import { describeFineFloodLadder } from "./fine-flood-provenance";
import { WebGPUFluidBlastRadius } from "./webgpu-fluid-blast-radius";
import { makeFineLevelSetSortedWorklistLookupWGSL } from "./webgpu-octree-fine-levelset-bricks";
import { PassBroker } from "./webgpu-pass-broker";
import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";

/** Four band widths, four ladder scalars, then four vec4u of prefix reach. */
export const OCTREE_TECHNIQUE_BAND_PREFIX_WORD_OFFSET = 8;
export const OCTREE_TECHNIQUE_BAND_CONFIG_BYTES =
  (OCTREE_TECHNIQUE_BAND_PREFIX_WORD_OFFSET + 16) * 4;


export const octreeTechniqueTopologyShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
struct Metric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct TetraHeader { first:u32, count:u32, flags:u32 }
struct TetraVertex { value:vec4f }
${visualizationBindingPreambleWGSL(OCTREE_TECHNIQUE_PROGRAMS.topology)}
const INVALID:u32=0xffffffffu; const VALID:u32=0x80000000u;
fn edgeInk(point:vec2f,a:vec3f,b:vec3f,width:f32)->f32 { return 1.0-smoothstep(width,2.2*width,segmentDistance(point,slice2(fineToWorld(a)),slice2(fineToWorld(b)))); }
fn topologyFault(row:u32,pointFine:vec3f)->vec4f {
  if(row>=arrayLength(&leaves)||row>=arrayLength(&metrics)||!leafContains(leaves[row],pointFine)){return vec4f(vec3f(1.0,0.01,0.18),0.88);}
  let metric=metrics[row];if((metric.transformAndFlags&VALID)==0u||metric.topologyCode>=arrayLength(&tetraHeaders)){return vec4f(vec3f(1.0,0.01,0.06),0.92);}
  return vec4f(0.0);
}
fn volumeTopology(uv:vec2f,mode:i32)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;
  let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}
  let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);let baseWidth=max(min(min(u.container.x/u.gridInfo.x,u.container.y/u.gridInfo.y),u.container.z/u.gridInfo.z)*0.10,1e-5);
  var accum=vec4f(0.0);var previous=INVALID;
  for(var sample=0u;sample<512u;sample+=1u){if(sample>=steps||accum.a>0.985){break;}let t=interval.x+(f32(sample)+0.5)*dt;let point=ray.origin+ray.direction*t;let pointFine=worldToFine(point);let cell=vec3i(clamp(floor(pointFine),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));let row=textureLoad(ownerRows,cell,0).x;
    if(row==previous){continue;}previous=row;if(row==INVALID){continue;}
    let fault=topologyFault(row,pointFine);if(fault.a>0.0){accum=compositeDisplay(accum,fault.rgb,0.32*volumeOpacity());continue;}
    let leaf=leaves[row];let metric=metrics[row];let header=tetraHeaders[metric.topologyCode];let transition=(header.flags&1u)==0u;let boundary=((metric.transformAndFlags>>8u)&63u)!=0u;
    let centerFine=vec3f(leafOrigin(leaf))+vec3f(0.5*f32(leaf.size));let centerWorld=fineToWorld(centerFine);let site=1.0-smoothstep(baseWidth,2.8*baseWidth,raySegmentDistance(ray,centerWorld,centerWorld+vec3f(0.0,1e-8,0.0)).x);
    if(mode==12){let base=select(vec3f(0.08,0.52,0.42),vec3f(0.42,0.14,0.70),transition);accum=compositeDisplay(accum,mix(base,vec3f(1.0,0.73,0.12),site),volumeOpacity()*(0.24+0.56*site));continue;}
    if(mode==15){if(transition||boundary){let color=select(vec3f(0.94,0.45,0.06),vec3f(0.96,0.08,0.40),boundary);accum=compositeDisplay(accum,color,volumeOpacity()*0.42);}continue;}
    if(mode!=14||!transition||header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){continue;}
    var ink=site;let count=min(header.count,${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u);let transform=metric.transformAndFlags&63u;let width=max(baseWidth,t*1.44/max(u.viewport.y,1.0));
    for(var local=0u;local<count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){ink=1.0;continue;}let a=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.x].value.xyz,transform);let b=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.y].value.xyz,transform);let c=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.z].value.xyz,transform);let aw=fineToWorld(a);let bw=fineToWorld(b);let cw=fineToWorld(c);
      ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centerWorld,aw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centerWorld,bw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centerWorld,cw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,aw,bw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,bw,cw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,cw,aw).x));}
    if(ink>0.01){accum=compositeDisplay(accum,mix(vec3f(0.12,0.72,0.86),vec3f(1.0,0.72,0.12),site),volumeOpacity()*0.92*ink);}
  }
  return finishDisplayVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  let mode=i32(round(u.debug.w));if(i32(round(u.debug.x))==4){return volumeTopology(input.uv,mode);}
  let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;
  if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}let pointFine=worldToFine(hit.xyz);let cell=vec3i(clamp(floor(pointFine),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));
  let row=textureLoad(ownerRows,cell,0).x;let footprint=max(hit.w*1.44/max(u.viewport.y,1.0),1e-5);
  if(row==INVALID){discard;}let fault=topologyFault(row,pointFine);if(fault.a>0.0){return vec4f(displayColor(fault.rgb),0.94);}
  let leaf=leaves[row];let metric=metrics[row];let valid=(metric.transformAndFlags&VALID)!=0u&&metric.topologyCode<arrayLength(&tetraHeaders);
  if(!valid){return vec4f(displayColor(vec3f(1.0,0.01,0.06)),0.96);}let header=tetraHeaders[metric.topologyCode];let transition=(header.flags&1u)==0u;
  let boundary=((metric.transformAndFlags>>8u)&63u)!=0u;let centerFine=vec3f(leafOrigin(leaf))+vec3f(0.5*f32(leaf.size));let centerWorld=fineToWorld(centerFine);
  let siteInk=1.0-smoothstep(1.6*footprint,3.2*footprint,length(slice2(hit.xyz)-slice2(centerWorld)));
  if(mode==12){let base=select(vec3f(0.08,0.52,0.42),vec3f(0.42,0.14,0.70),transition);let color=mix(base,vec3f(1.0,0.73,0.12),siteInk);return vec4f(displayColor(color),max(0.38,0.92*siteInk));}
  if(mode==15){let selected=transition||boundary;let color=select(vec3f(0.03,0.09,0.18),select(vec3f(0.94,0.45,0.06),vec3f(0.96,0.08,0.40),boundary),selected);return vec4f(displayColor(color),select(0.18,0.88,selected));}
  if(mode!=14||!transition||header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){discard;}
  var ink=siteInk;let count=min(header.count,${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u);let transform=metric.transformAndFlags&63u;
  for(var local=0u;local<count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
    if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){return vec4f(displayColor(vec3f(1.0,0.01,0.06)),0.96);}let a=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.x].value.xyz,transform);let b=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.y].value.xyz,transform);let c=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.z].value.xyz,transform);
    ink=max(ink,edgeInk(slice2(hit.xyz),centerFine,a,footprint));ink=max(ink,edgeInk(slice2(hit.xyz),centerFine,b,footprint));ink=max(ink,edgeInk(slice2(hit.xyz),centerFine,c,footprint));ink=max(ink,edgeInk(slice2(hit.xyz),a,b,footprint));ink=max(ink,edgeInk(slice2(hit.xyz),b,c,footprint));ink=max(ink,edgeInk(slice2(hit.xyz),c,a,footprint));}
  if(ink<0.02){discard;}return vec4f(displayColor(mix(vec3f(0.12,0.72,0.86),vec3f(1.0,0.72,0.12),siteInk)),0.94*ink);
}`;

/**
 * Direct views of the accepted generalized-face and structured-velocity
 * publications. Separate pipelines keep each bind group below WebGPU's
 * portable fragment-storage-buffer limit while giving every observatory field
 * a real compact source instead of the octree renderer's zero-texture fallback.
 */
export const octreeTechniqueFaceShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct Metric { topologyCode:u32,transformAndFlags:u32,volume:f32,reserved:u32 }
struct CatalogSlotGeometry { neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f }
${visualizationBindingPreambleWGSL(OCTREE_TECHNIQUE_PROGRAMS.face)}
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;
fn cellCoord(cell:u32)->vec3u{let d=vec3u(max(u.gridInfo.xyz,vec3f(1.0)));return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}
fn rowAt(point:vec3f)->u32{return textureLoad(ownerRows,vec3i(clamp(floor(worldToFine(point)),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0))),0).x;}
fn rowValid(row:u32,point:vec3f)->bool{if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)){return false;}let h=headers[row];let origin=vec3f(cellCoord(h.cell));return h.size>0u&&(metrics[row].transformAndFlags&VALID)!=0u&&all(point>=origin)&&all(point<origin+vec3f(f32(h.size)));}
fn catalogRange(row:u32)->vec2u{let code=metrics[row].topologyCode;if(code>=arrayLength(&entryHeaders)){return vec2u(INVALID,0u);}let range=entryHeaders[code];if(range.x>arrayLength(&catalogFaces)||range.y>arrayLength(&catalogFaces)-range.x){return vec2u(INVALID,0u);}return range;}
fn heat(value:f32)->vec3f{let t=clamp(value,0.0,1.0);return select(mix(vec3f(0.04,0.20,0.70),vec3f(0.06,0.78,0.55),t*2.0),mix(vec3f(0.06,0.78,0.55),vec3f(1.0,0.08,0.025),(t-0.5)*2.0),t>=0.5);}
fn faceSample(row:u32,ray:CameraRay,interval:vec2f,point:vec3f,footprint:f32,volume:bool,operatorView:bool)->vec4f{let range=catalogRange(row);if(range.x==INVALID){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let h=headers[row];let centerFine=vec3f(cellCoord(h.cell))+vec3f(0.5*f32(h.size));let center=fineToWorld(centerFine);let scale=min(min(u.container.x/u.gridInfo.x,u.container.y/u.gridInfo.y),u.container.z/u.gridInfo.z);var planeInk=0.0;var dualInk=0.0;var normalInk=0.0;var coefficient=0.0;
  for(var local=0u;local<${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u;local+=1u){if(local>=range.y){break;}let face=catalogFaces[range.x+local];coefficient=max(coefficient,face.areaCentroid.x*face.normalInverseDistance.w*f32(h.size));if(operatorView){continue;}let transform=metrics[row].transformAndFlags&63u;let centroidFine=centerFine+f32(h.size)*inversePowerTransform(face.areaCentroid.yzw,transform);let centroid=fineToWorld(centroidFine);let normal=normalize(inversePowerTransform(face.normalInverseDistance.xyz,transform));let radius=sqrt(max(face.areaCentroid.x,1e-8)/3.14159265)*f32(h.size)*scale;let neighbor=fineToWorld(centerFine+f32(h.size)*inversePowerTransform(face.neighborOffsetSize.xyz,transform));if(volume){let denominator=dot(ray.direction,normal);if(abs(denominator)>1e-6){let distance=dot(centroid-ray.origin,normal)/denominator;if(distance>=interval.x&&distance<=interval.y){planeInk=max(planeInk,1.0-smoothstep(radius,max(radius+2.0*footprint,1.08*radius),length(ray.origin+ray.direction*distance-centroid)));}}dualInk=max(dualInk,1.0-smoothstep(footprint,2.2*footprint,raySegmentDistance(ray,center,neighbor).x));normalInk=max(normalInk,1.0-smoothstep(footprint,2.2*footprint,raySegmentDistance(ray,centroid,centroid+normal*max(0.75*radius,footprint)).x));}else{let signedDistance=dot(point-centroid,normal);let radial=length((point-centroid)-normal*signedDistance);planeInk=max(planeInk,(1.0-smoothstep(footprint,2.2*footprint,abs(signedDistance)))*(1.0-smoothstep(radius,max(radius+2.0*footprint,1.08*radius),radial)));dualInk=max(dualInk,1.0-smoothstep(footprint,2.2*footprint,segmentDistance3(point,center,neighbor)));normalInk=max(normalInk,1.0-smoothstep(footprint,2.2*footprint,segmentDistance3(point,centroid,centroid+normal*max(0.75*radius,footprint))));}}
  if(operatorView){return vec4f(heat(1.0-exp(-coefficient)),select(0.86,0.22,volume));}let alpha=max(planeInk,max(dualInk,normalInk));var color=vec3f(0.02);if(planeInk>=dualInk&&planeInk>=normalInk){color=mix(color,vec3f(0.04,0.84,0.96),planeInk);}else if(dualInk>=normalInk){color=mix(color,vec3f(0.48,0.10,0.72),dualInk);}else{color=mix(color,vec3f(0.98,0.72,0.08),normalInk);}return vec4f(color,alpha*select(0.96,1.0,volume));}
fn faceVolume(uv:vec2f,operatorView:bool)->vec4f{let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let interval=boxInterval(ray,minimum,minimum+u.container.xyz);if(interval.y<=interval.x){discard;}let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let row=rowAt(point);if(row==previous){continue;}previous=row;if(!rowValid(row,worldToFine(point))){continue;}let sample=faceSample(row,ray,interval,point,max(dt*0.2,1e-5),true,operatorView);accum=compositeDisplay(accum,sample.rgb,sample.a*volumeOpacity());}return finishDisplayVolume(accum);}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f{let operatorView=i32(round(u.debug.w))==16;if(i32(round(u.debug.x))==4){return faceVolume(input.uv,operatorView);}let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);if(any(hit.xyz<minimum)||any(hit.xyz>minimum+u.container.xyz)){discard;}let row=rowAt(hit.xyz);if(row==INVALID){discard;}if(!rowValid(row,worldToFine(hit.xyz))){return vec4f(displayColor(vec3f(1.0,0.01,0.10)),0.94);}let sample=faceSample(row,cameraRay(input.uv),vec2f(0.0),hit.xyz,max(hit.w*1.44/max(u.viewport.y,1.0),1e-5),false,operatorView);if(sample.a<=0.001){discard;}return vec4f(displayColor(sample.rgb),sample.a);}
`;

export const octreeTechniqueStructuredShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }struct Metric { topologyCode:u32,transformAndFlags:u32,volume:f32,reserved:u32 }struct StructuredParams { words:array<vec4u,16> }
${visualizationBindingPreambleWGSL(OCTREE_TECHNIQUE_PROGRAMS.structured)}
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;fn word(i:u32)->u32{return structured.words[i/4u][i%4u];}fn finiteValue(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn cellCoord(cell:u32)->vec3u{let d=vec3u(max(u.gridInfo.xyz,vec3f(1.0)));return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}fn rowAt(point:vec3f)->u32{return textureLoad(ownerRows,vec3i(clamp(floor(worldToFine(point)),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0))),0).x;}fn rowValid(row:u32,point:vec3f)->bool{if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)){return false;}let h=headers[row];let origin=vec3f(cellCoord(h.cell));return h.size>0u&&(metrics[row].transformAndFlags&VALID)!=0u&&all(point>=origin)&&all(point<origin+vec3f(f32(h.size)));}fn publicationValid()->bool{return arrayLength(&accepted)>=6u&&accepted[0]==0u&&accepted[3]!=0u;}fn rowCapacity()->u32{return max(1u,word(0u));}fn authorityBase()->u32{return (accepted[4]&1u)*word(15u);}fn heat(value:f32)->vec3f{let t=clamp(value,0.0,1.0);if(t<0.5){return mix(vec3f(0.04,0.20,0.70),vec3f(0.06,0.78,0.55),t*2.0);}return mix(vec3f(0.06,0.78,0.55),vec3f(1.0,0.08,0.025),(t-0.5)*2.0);}
/**
 * The eighteen power-stencil directions, in the operator's own order.
 *
 * Six faces then twelve edges, matching \`FLUID_CELL_TRACE_DIRECTIONS\` so the
 * picker's arrows and this field describe the same couplings.
 */
fn stencilDirection(index:u32)->vec3i{let d=array<vec3i,18>(vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[index];}

/**
 * Per-row operator cost: the coefficients one apply reads for this unknown.
 *
 * \`entryCount\` is the assembled row's stencil width, so it is exactly what the
 * smoother touches on every sweep, of which there are many per frame. A regular
 * interior row sits near six; a row straddling a refinement transition climbs
 * toward the full eighteen the power diagram needs to stay consistent across a
 * T-junction. So this paints where the adaptivity is charging for itself, which
 * is a different question from where the fluid is.
 */
fn costSample(row:u32,volume:bool)->vec4f{let ceiling=f32(max(word(1u),1u));let entries=f32(headers[row].entryCount);return vec4f(heat(clamp(entries/ceiling,0.0,1.0)),select(0.90,0.13,volume));}

/**
 * Gather locality: how far apart in the compact array this row's neighbours sit.
 *
 * The apply reads \`pressure[neighbour]\` once per coupling, so the spread of
 * neighbour indices is the access pattern the hardware actually sees — and it is
 * invisible to any count of arithmetic. Rows whose eighteen neighbours land
 * within a few hundred indices gather from one region and ride the cache; rows
 * whose neighbours are tens of thousands apart scatter, and pay bandwidth for it
 * every sweep. Normalised by the live row count so the same colour means the
 * same thing at any resolution, and on a log scale because the interesting
 * difference is between "tens away" and "tens of thousands away", which a linear
 * ramp flattens into one colour.
 */
fn localitySample(row:u32,volume:bool)->vec4f{let header=headers[row];let size=i32(max(header.size,1u));let origin=vec3i(cellCoord(header.cell));let centre=origin+vec3i(size/2);var spread=0u;var live=0u;for(var index=0u;index<18u;index+=1u){let probe=centre+stencilDirection(index)*(size/2+1);if(any(probe<vec3i(0))||any(probe>=vec3i(u.gridInfo.xyz))){continue;}let neighbour=textureLoad(ownerRows,probe,0).x;if(neighbour>=accepted[2]||neighbour==row){continue;}if(!rowValid(neighbour,vec3f(probe)+vec3f(0.5))){continue;}live+=1u;spread=max(spread,u32(abs(i32(neighbour)-i32(row))));}
// A row with no live coupling is a boundary or isolated unknown, not a locality
// result; drawing it as "perfectly local" would be a lie the colour cannot undo.
if(live==0u){return vec4f(vec3f(0.10,0.12,0.18),select(0.35,0.06,volume));}
let normalized=log2(f32(spread)+1.0)/log2(f32(max(accepted[2],2u)));return vec4f(heat(clamp(normalized,0.0,1.0)),select(0.90,0.13,volume));}

fn rowSample(row:u32,mode:i32,volume:bool)->vec4f{if(!publicationValid()||row>=accepted[2]||row>=rowCapacity()){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let metric=metrics[row];if(!finiteValue(metric.volume)||metric.volume<=0.0){return vec4f(vec3f(1.0,0.01,0.10),0.94);}if(mode==33){return costSample(row,volume);}if(mode==34){return localitySample(row,volume);}if(mode==27||mode==30){let at=(accepted[4]&1u)*rowCapacity()+row;if(at>=arrayLength(&rowVelocities)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let velocity=rowVelocities[at];if(velocity.w<=0.0||any(velocity.xyz!=velocity.xyz)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}if(mode==30){let direction=vec3f(0.5)+0.5*velocity.xyz/max(length(velocity.xyz),1e-6);return vec4f(direction,select(0.88,0.13,volume));}return vec4f(heat(length(velocity.xyz)/max(u.environment.z,1e-4)),select(0.88,0.13,volume));}let base=authorityBase();let maxSlots=word(1u);let slotBase=row*maxSlots;var ownPressure=0.0;if(row<arrayLength(&pressure)){ownPressure=pressure[row];}var largest=0.0;var flux=0.0;for(var local=0u;local<${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u;local+=1u){if(local>=maxSlots){break;}let at=slotBase+local;let handleAt=base+word(29u)+at;let signAt=base+word(30u)+at;if(handleAt>=arrayLength(&authority)||signAt>=arrayLength(&authority)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let handle=authority[handleAt];if(handle==INVALID||handle>=accepted[5]){continue;}if(mode==29){let valueAt=base+word(16u)+handle;let areaAt=base+word(20u)+handle;let fractionAt=base+word(22u)+handle;if(max(valueAt,max(areaAt,fractionAt))>=arrayLength(&authority)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}flux+=f32(bitcast<i32>(authority[signAt]))*bitcast<f32>(authority[valueAt])*bitcast<f32>(authority[areaAt])*bitcast<f32>(authority[fractionAt]);}else{let neighborAt=base+word(18u)+handle;let inverseAt=base+word(21u)+handle;if(max(neighborAt,inverseAt)>=arrayLength(&authority)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let neighbor=authority[neighborAt];var neighborPressure=0.0;if(neighbor<arrayLength(&pressure)){neighborPressure=pressure[neighbor];}largest=max(largest,abs(neighborPressure-ownPressure)*bitcast<f32>(authority[inverseAt]));}}if(mode==28){return vec4f(heat(largest/max(u.environment.z,0.05)),select(0.90,0.13,volume));}let width=f32(headers[row].size)*bitcast<f32>(word(40u));let divergence=flux/max(metric.volume*width*width*width,1e-9);let scaled=clamp(divergence*max(u.environment.y,1e-6),-1.0,1.0);let color=select(mix(vec3f(0.96),vec3f(0.88,0.10,0.08),scaled),mix(vec3f(0.96),vec3f(0.08,0.28,0.88),-scaled),scaled<0.0);return vec4f(color,select(0.92,0.14,volume));}
fn structuredVolume(uv:vec2f,mode:i32)->vec4f{let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let interval=boxInterval(ray,minimum,minimum+u.container.xyz);if(interval.y<=interval.x){discard;}let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let row=rowAt(point);if(row==previous){continue;}previous=row;if(!rowValid(row,worldToFine(point))){continue;}let sample=rowSample(row,mode,true);accum=composite(accum,sample.rgb,sample.a*volumeOpacity());}return finishVolume(accum);}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f{let mode=i32(round(u.debug.w));if(i32(round(u.debug.x))==4){return structuredVolume(input.uv,mode);}let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);if(any(hit.xyz<minimum)||any(hit.xyz>minimum+u.container.xyz)){discard;}let row=rowAt(hit.xyz);if(row==INVALID){discard;}if(!rowValid(row,worldToFine(hit.xyz))){return vec4f(displayColor(vec3f(1.0,0.01,0.10)),0.94);}let sample=rowSample(row,mode,false);return vec4f(displayColor(sample.rgb),sample.a);}
`;

/**
 * One three-dimensional velocity glyph per accepted Losasso adaptive leaf.
 *
 * The leaf record is the authority for both the glyph extent and the eight
 * corner-node slots. Averaging those accepted nodal records at the cell centre
 * is the same trilinear reconstruction evaluated at (1/2, 1/2, 1/2), so the
 * overlay never substitutes a dense velocity texture for the adaptive field it
 * is meant to explain.
 */
export const octreeTechniqueAdaptiveVelocityShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
struct AdaptiveLeaf { originSpan:vec4u, metadata:vec4u, lowerSlots:vec4u, upperSlots:vec4u }
struct AdaptiveCellVelocity { value:vec3f, valid:u32 }
struct AdaptivePhiClass { air:u32, visible:u32, valid:u32, pad0:u32 }
struct AdaptiveVelocityViewConfig { extensionReach_m:f32, pad0:f32, pad1:f32, pad2:f32 }
${visualizationBindingPreambleWGSL(OCTREE_TECHNIQUE_PROGRAMS.adaptiveVelocity)}
const INVALID:u32=0xffffffffu;
fn heat(value:f32)->vec3f{let t=clamp(value,0.0,1.0);return select(mix(vec3f(0.04,0.20,0.70),vec3f(0.06,0.78,0.55),t*2.0),mix(vec3f(0.06,0.78,0.55),vec3f(1.0,0.08,0.025),(t-0.5)*2.0),t>=0.5);}
fn finite1(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn finite3(value:vec3f)->bool{return all(value==value)&&all(abs(value)<=vec3f(3.402823e38));}
fn publicationValid()->bool{return arrayLength(&adaptiveControl)>=16u&&adaptiveControl[0]!=0u&&adaptiveControl[3]==adaptiveControl[0]&&adaptiveControl[4]==0u&&adaptiveControl[6]==adaptiveControl[0];}
fn rowAt(point:vec3f)->u32{return textureLoad(ownerRows,vec3i(clamp(floor(worldToFine(point)),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0))),0).x;}
fn leafAt(row:u32,pointFine:vec3f)->bool{if(row>=arrayLength(&adaptiveLeaves)||row>=adaptiveControl[1]){return false;}let leaf=adaptiveLeaves[row];return leaf.metadata.x!=0u&&leaf.originSpan.w>0u&&all(pointFine>=vec3f(leaf.originSpan.xyz))&&all(pointFine<vec3f(leaf.originSpan.xyz+vec3u(leaf.originSpan.w)));}
fn classifyPhi(row:u32)->AdaptivePhiClass{
  if(arrayLength(&adaptivePhiControl)<20u||adaptivePhiControl[7u]!=1u||adaptivePhiControl[1u]!=adaptiveControl[0u]||adaptivePhiControl[2u]!=adaptiveControl[5u]||adaptivePhiControl[3u]!=adaptiveControl[6u]||row>=adaptivePhiControl[5u]||row>=arrayLength(&adaptiveRowPhi)){return AdaptivePhiClass(0u,1u,0u,0u);}let packed=adaptiveRowPhi[row];let centre=bitcast<f32>(packed.x);let minimum=bitcast<f32>(packed.y);let maximum=bitcast<f32>(packed.z);
  if((packed.w&3u)!=3u||!finite1(centre)||!finite1(minimum)||!finite1(maximum)){return AdaptivePhiClass(0u,1u,0u,0u);}
  // A straddling leaf is styled as interface even when its centre lies on the
  // air side. Only a wholly air-side leaf is translucent, and its centre must
  // lie inside the same physical reach used by velocity extension.
  let interfaceLeaf=(packed.w&4u)!=0u||(minimum<=0.0&&maximum>=0.0);let air=select(0u,1u,!interfaceLeaf&&minimum>0.0);let visible=select(1u,select(0u,1u,centre<=max(adaptiveVelocityView.extensionReach_m,0.0)),air!=0u);return AdaptivePhiClass(air,visible,1u,0u);
}
fn cellVelocity(row:u32)->AdaptiveCellVelocity{
  if(!publicationValid()||row>=arrayLength(&adaptiveLeaves)||row>=adaptiveControl[1]){return AdaptiveCellVelocity(vec3f(0.0),0u);}let leaf=adaptiveLeaves[row];let nodeCapacity=adaptiveControl[15];var value=vec3f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){let slots=select(leaf.lowerSlots,leaf.upperSlots,corner>=4u);let slot=slots[corner&3u];let record=2u*slot;if(slot>=nodeCapacity||record>=arrayLength(&adaptiveVelocity)){return AdaptiveCellVelocity(vec3f(0.0),0u);}let packed=adaptiveVelocity[record];if((packed.w&7u)!=7u){return AdaptiveCellVelocity(vec3f(0.0),0u);}let node=vec3f(bitcast<f32>(packed.x),bitcast<f32>(packed.y),bitcast<f32>(packed.z));if(!finite3(node)){return AdaptiveCellVelocity(vec3f(0.0),0u);}value+=0.125*node;}
  return AdaptiveCellVelocity(value,1u);
}
fn pointInk(ray:CameraRay,point:vec3f,width:f32)->f32{return 1.0-smoothstep(width,2.25*width,raySegmentDistance(ray,point,point+vec3f(0.0,1e-8,0.0)).x);}
fn arrowSample(row:u32,ray:CameraRay,depth:f32,volume:bool)->vec4f{
  if(row==INVALID||row>=arrayLength(&adaptiveLeaves)||arrayLength(&adaptiveControl)<16u){return vec4f(0.0);}let leaf=adaptiveLeaves[row];let origin=vec3f(leaf.originSpan.xyz);let span=f32(leaf.originSpan.w);let center=fineToWorld(origin+vec3f(0.5*span));let low=fineToWorld(origin);let high=fineToWorld(origin+vec3f(span));let cellWidth=max(min(min(high.x-low.x,high.y-low.y),high.z-low.z),1e-6);let width=max(depth*1.44/max(u.viewport.y,1.0),0.018*cellWidth);let phase=classifyPhi(row);
  if(phase.valid==0u){let ink=pointInk(ray,center,2.2*width);return vec4f(vec3f(1.0,0.01,0.10),select(0.96,0.72,volume)*ink);}if(phase.visible==0u){return vec4f(0.0);}let sampled=cellVelocity(row);let phaseAlpha=select(1.0,0.36,phase.air!=0u);
  if(sampled.valid==0u){let ink=pointInk(ray,center,2.2*width);return vec4f(vec3f(1.0,0.01,0.10),select(0.96,0.72,volume)*ink);}
  let speed=length(sampled.value);let maximum=max(u.environment.z,1e-4);let magnitude=clamp(speed/maximum,0.0,1.0);let color=heat(magnitude);
  if(speed<=1e-7){let ink=pointInk(ray,center,1.8*width);return vec4f(color,select(0.88,0.58,volume)*phaseAlpha*ink);}
  // The colour remains linear in speed while only the glyph length is
  // logarithmic. Sixteen bins make slow motion legible without letting a fast
  // leaf's arrow escape its own cell.
  let logMagnitude=log2(1.0+15.0*magnitude)/4.0;let arrowLength=cellWidth*(0.10+0.72*logMagnitude);let direction=sampled.value/speed;let tail=center-0.5*arrowLength*direction;let head=center+0.5*arrowLength*direction;
  let towardCamera=normalize(u.cameraPosition.xyz-head);var side=cross(direction,towardCamera);if(length(side)<1e-5){side=cross(direction,select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(direction.y)>0.8));}side=normalize(side);
  let headLength=min(0.30*arrowLength,0.22*cellWidth);let wingWidth=min(0.22*arrowLength,0.16*cellWidth);let headBase=head-direction*headLength;let wingA=headBase+side*wingWidth;let wingB=headBase-side*wingWidth;
  var distance=raySegmentDistance(ray,tail,head).x;distance=min(distance,raySegmentDistance(ray,head,wingA).x);distance=min(distance,raySegmentDistance(ray,head,wingB).x);let ink=1.0-smoothstep(width,2.25*width,distance);return vec4f(color,select(0.96,0.72,volume)*phaseAlpha*ink);
}
fn adaptiveVelocityVolume(uv:vec2f)->vec4f{
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let interval=boxInterval(ray,minimum,minimum+u.container.xyz);if(interval.y<=interval.x){discard;}let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;
  for(var sample=0u;sample<512u;sample+=1u){if(sample>=steps||accum.a>0.985){break;}let depth=interval.x+(f32(sample)+0.5)*dt;let point=ray.origin+ray.direction*depth;let row=rowAt(point);if(row==previous){continue;}previous=row;if(row==INVALID||!leafAt(row,worldToFine(point))){continue;}let arrow=arrowSample(row,ray,depth,true);if(arrow.a>0.001){accum=compositeDisplay(accum,arrow.rgb,arrow.a*volumeOpacity());}}
  return finishDisplayVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f{
  if(i32(round(u.debug.x))==4){return adaptiveVelocityVolume(input.uv);}let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);if(any(hit.xyz<minimum)||any(hit.xyz>minimum+u.container.xyz)){discard;}let row=rowAt(hit.xyz);if(row==INVALID||!leafAt(row,worldToFine(hit.xyz))){discard;}let arrow=arrowSample(row,cameraRay(input.uv),hit.w,false);if(arrow.a<=0.001){discard;}return vec4f(displayColor(arrow.rgb),arrow.a);
}`;

const octreeLifecycleMembershipShader = /* wgsl */ `
struct Config { dimensions:vec3u,tileSize:u32,capacity:u32,pad0:u32,pad1:u32,pad2:u32 }
${visualizationBindingPreambleWGSL(OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM)}
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  let slot=id.x;if(slot>=config.capacity){return;}let header=16u;
  let activeCount=min(worklist[0],config.capacity);if(slot<activeCount){let tile=worklist[header+slot];if(tile<arrayLength(&membership)){membership[tile]=1u;}}
  let retired=min(worklist[4],config.capacity);if(slot<retired){let index=header+config.capacity+slot;if(index<arrayLength(&worklist)){let tile=worklist[index];if(tile<arrayLength(&membership)){membership[tile]=2u;}}}
}`;

export const octreeTechniqueLifecycleShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
struct Config { dimensions:vec3u,tileSize:u32,capacity:u32,pad0:u32,pad1:u32,pad2:u32 }
${visualizationBindingPreambleWGSL(OCTREE_TECHNIQUE_PROGRAMS.lifecycle)}
fn lifecycleAt(point:vec3f)->vec4f {
  let fine=worldToFine(point);let tile=vec3u(clamp(floor(fine/f32(max(config.tileSize,1u))),vec3f(0.0),vec3f(config.dimensions)-vec3f(1.0)));
  let index=tile.x+config.dimensions.x*(tile.y+config.dimensions.y*tile.z);if(index>=config.capacity||index>=arrayLength(&membership)){return vec4f(1.0,0.01,0.06,0.92);}
  let state=membership[index];if(state==1u){return vec4f(0.04,0.78,0.86,0.38);}if(state==2u){return vec4f(1.0,0.34,0.04,0.72);}return vec4f(0.02,0.06,0.14,0.04);
}
fn lifecycleVolume(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=0xffffffffu;
  for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let fine=worldToFine(point);let tile=vec3u(clamp(floor(fine/f32(max(config.tileSize,1u))),vec3f(0.0),vec3f(config.dimensions)-vec3f(1.0)));let index=tile.x+config.dimensions.x*(tile.y+config.dimensions.y*tile.z);if(index==previous){continue;}previous=index;let sample=lifecycleAt(point);accum=composite(accum,sample.rgb,sample.a*volumeOpacity());}
  return finishVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  if(i32(round(u.debug.x))==4){return lifecycleVolume(input.uv);}let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}let sample=lifecycleAt(hit.xyz);if(sample.a<=0.001){discard;}return vec4f(displayColor(sample.rgb),sample.a);
}`;

export const octreeTechniqueFineLifecycleShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
struct FineParams { brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32 }
struct FineState { color:vec3f,alpha:f32,address:u32 }
struct BandConfig {
  pressureBandCells:u32,surfaceBandCells:u32,transportBandFineCells:u32,redistanceBandFineCells:u32,
  /** Passes in the ladder the last redistance encode emitted. */
  ladderPasses:u32,pad0:u32,pad1:u32,pad2:u32,
  /** Reach of the first k + 1 encoded passes, so colours name real schedule slots. */
  prefixReach:array<vec4u,4>,
}
${visualizationBindingPreambleWGSL(OCTREE_TECHNIQUE_PROGRAMS.fine)}
${fineLevelSetPackedSampleWGSL("sampleFlags", false, "debugFine")}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;
${makeFineLevelSetSortedWorklistLookupWGSL("fine", "metadata", "worklist", "pageOf")}
fn fineAddress(q:vec3i)->u32 {
  if(any(q<vec3i(0))||any(q>=vec3i(fine.sampleDimensions))){return INVALID;}
  let uq=vec3u(q);let brick=uq/max(fine.brickResolution,1u);let key=brick.x+fine.brickDimensions.x*(brick.y+fine.brickDimensions.y*brick.z);let page=pageOf(key);
  if(page==INVALID||page>=fine.pageCapacity||page*4u+3u>=arrayLength(&metadata)||metadata[page*4u+2u]!=fine.generation){return INVALID;}
  let local=uq-brick*fine.brickResolution;let localIndex=local.x+fine.brickResolution*(local.y+fine.brickResolution*local.z);let address=page*fine.samplesPerBrick+localIndex;
  return select(INVALID,address,address<arrayLength(&sampleFlags)&&(debugFinePackedFlags(address)&VALID)!=0u);
}
fn renderWorldToFine(point:vec3f)->vec3f {
  // The solver's sparse fine lattice uses its local [0, extent] frame while
  // rendering centres x/z around the tank.  domainOrigin belongs to the
  // solver frame and must not be subtracted from centred render coordinates.
  let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);
  return (point-minimum)/max(fine.fineCellWidth,1e-9);
}
fn fineState(point:vec3f)->FineState {
  let relative=renderWorldToFine(point);if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){return FineState(vec3f(0.0),0.0,INVALID);}let q=vec3u(floor(relative));let brick=q/max(fine.brickResolution,1u);let key=brick.x+fine.brickDimensions.x*(brick.y+fine.brickDimensions.y*brick.z);
  if(arrayLength(&topologyControl)>0u&&topologyControl[0]!=0u){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}if(arrayLength(&redistanceControl)>4u&&redistanceControl[4]!=0u){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}
  let page=pageOf(key);if(page==INVALID){let desired=arrayLength(&topologyControl)>4u&&topologyControl[4]==0u;return FineState(select(vec3f(0.03,0.10,0.34),vec3f(1.0,0.34,0.04),desired),select(0.045,0.34,desired),key);}
  let local=q-brick*fine.brickResolution;let localIndex=local.x+fine.brickResolution*(local.y+fine.brickResolution*local.z);let address=page*fine.samplesPerBrick+localIndex;if(address>=arrayLength(&sampleFlags)){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}let flags=debugFinePackedFlags(address);
  if((flags&VALID)==0u){return FineState(vec3f(0.26,0.08,0.48),0.20,address);}if((flags&INTERFACE)!=0u){return FineState(vec3f(1.0,0.02,0.44),0.90,address);}return FineState(vec3f(0.04,0.72,0.82),0.34,address);
}
fn globalFinePhi(point:vec3f)->vec4f {
  if(arrayLength(&topologyControl)==0u||topologyControl[0]!=0u||arrayLength(&redistanceControl)<=4u||redistanceControl[3]==0u||redistanceControl[4]!=0u){return vec4f(1.0,0.01,0.06,0.96);}
  let relative=renderWorldToFine(point);if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){return vec4f(0.0);}
  let q=vec3i(floor(relative));let centerAddress=fineAddress(q);if(centerAddress==INVALID){return vec4f(0.02,0.06,0.18,0.08);}
  let center=debugFinePackedPhi(centerAddress);if(center!=center||abs(center)>=3.402823e38){return vec4f(1.0,0.01,0.06,0.96);}
  var gradient=vec3f(0.0);var complete=true;let h=max(fine.fineCellWidth,1e-9);
  for(var axis=0u;axis<3u;axis+=1u){var delta=vec3i(0);delta[axis]=1;let lo=fineAddress(q-delta);let hi=fineAddress(q+delta);var derivative=0.0;
    if(lo!=INVALID&&hi!=INVALID){derivative=(debugFinePackedPhi(hi)-debugFinePackedPhi(lo))/(2.0*h);}else if(hi!=INVALID){derivative=(debugFinePackedPhi(hi)-center)/h;}else if(lo!=INVALID){derivative=(center-debugFinePackedPhi(lo))/h;}else{complete=false;}
    if(derivative!=derivative||abs(derivative)>=3.402823e38){complete=false;}gradient[axis]=derivative;
  }
  if(!complete){return vec4f(1.0,0.01,0.06,0.96);}
  let signed=clamp(center/(4.0*h),-1.0,1.0);let phiColor=select(mix(vec3f(0.96,0.96,0.90),vec3f(0.93,0.47,0.16),signed),mix(vec3f(0.96,0.96,0.90),vec3f(0.10,0.45,0.92),-signed),signed<0.0);
  let residual=abs(length(gradient)-1.0);let residualInk=clamp(residual/0.25,0.0,1.0);let zeroInk=1.0-smoothstep(0.04*h,0.22*h,abs(center));
  var color=mix(phiColor,vec3f(0.96,0.02,0.72),residualInk);color=mix(color,vec3f(1.0),zeroInk);
  return vec4f(color,0.86);
}
/**
 * Pressure refinement band against surface-band residency.
 *
 * Both widths are authored in finest octree cells, so both are measured here in
 * the same unit: the finest cell is fineFactor fine cells wide. A sample is in
 * the surface band when its fine phi is actually resident and valid -- that is
 * the band as published, not as requested -- and in the pressure band when its
 * distance to the interface is inside the authored pressure reach.
 *
 * The state worth seeing is the fourth one. A resident sample that is inside
 * the pressure reach but has a non-resident face neighbour marks where the
 * pressure discretization wants phi that the surface band does not carry. That
 * is the precondition for the fine-to-coarse restriction finding no valid phi
 * at a pressure cell centre, which rejects the whole publication rather than
 * degrading it -- so the view shows the cause, not just the aftermath.
 */
fn bandResidency(point:vec3f)->vec4f {
  if(arrayLength(&topologyControl)==0u||topologyControl[0]!=0u
    ||arrayLength(&redistanceControl)<=4u||redistanceControl[3]==0u||redistanceControl[4]!=0u){
    return vec4f(1.0,0.01,0.06,0.96);
  }
  let relative=renderWorldToFine(point);
  if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){return vec4f(0.0);}
  let q=vec3i(floor(relative));
  let address=fineAddress(q);
  let finest=max(fine.fineCellWidth,1e-9)*f32(max(fine.fineFactor,1u));
  if(address==INVALID){
    // No resident support here. Drawn faintly so the surface band reads as a
    // shell against the rest of the domain rather than a solid block.
    return vec4f(0.03,0.09,0.28,0.06);
  }
  let phi=debugFinePackedPhi(address);
  if(phi!=phi||abs(phi)>=3.402823e38){return vec4f(1.0,0.01,0.06,0.96);}
  let h=max(fine.fineCellWidth,1e-9);
  let distance=abs(phi);
  // Every authored and derived width, nested outward from the interface. The
  // two authored bands are in finest cells; the planner's transport and
  // redistance widths are in fine cells, so they are scaled differently on
  // purpose rather than sharing one unit.
  if(distance<=0.5*h){return vec4f(1.0,1.0,1.0,0.95);}
  var truncated=false;
  for(var axis=0u;axis<3u;axis+=1u){
    var delta=vec3i(0);delta[axis]=1;
    if(fineAddress(q+delta)==INVALID||fineAddress(q-delta)==INVALID){truncated=true;}
  }
  if(distance<=f32(bands.pressureBandCells)*finest){
    // The pressure solve reads phi here. Residency truncating this reach is the
    // precondition for the restriction finding none, so it outranks the fill.
    if(truncated){return vec4f(0.96,0.73,0.10,0.95);}
    return vec4f(1.0,0.09,0.56,0.72);
  }
  if(distance<=f32(bands.transportBandFineCells)*h){
    // Authored surface band beyond the pressure reach: transported every step.
    return vec4f(0.04,0.72,0.82,0.34);
  }
  if(distance<=f32(bands.redistanceBandFineCells)*h){
    // Kept valid by redistance but not transported -- the support margin that
    // covers the backtrace and the trilinear stencil.
    return vec4f(0.42,0.33,0.92,0.26);
  }
  // Resident past the redistance cutoff: whole-brick dilation and safety rings.
  return vec4f(0.10,0.40,0.30,0.16);
}
/**
 * Where each sample's distance came from.
 *
 * The redistance commit leaves the resolved seed index in the work channel, so
 * the flood's dependency graph is resident after the frame. Colouring a sample
 * by the leading encoded passes whose combined reach covers its hop turns that
 * graph into the question worth asking of the schedule: which passes are load
 * bearing, and where.
 *
 * The band views above answer "which cells are resident"; this answers "how far
 * information had to travel to reach them", which is the cost of residency
 * rather than its extent.
 */
fn floodSampleCell(index:u32)->vec3u {
  let perBrick=max(fine.samplesPerBrick,1u);
  let id=index/perBrick;let local=index-id*perBrick;
  let key=metadata[id*4u+1u];
  let xy=max(fine.brickDimensions.x*fine.brickDimensions.y,1u);
  let bz=key/xy;let brickRemainder=key-bz*xy;let by=brickRemainder/max(fine.brickDimensions.x,1u);
  let brick=vec3u(brickRemainder-by*fine.brickDimensions.x,by,bz);
  let r=max(fine.brickResolution,1u);
  let lz=local/(r*r);let localRemainder=local-lz*r*r;let ly=localRemainder/r;
  return brick*r+vec3u(localRemainder-ly*r,ly,lz);
}
/**
 * Returns 0 for a self-seeded sample, k for a hop the first k encoded passes
 * cover, and ladderPasses + 1 for a hop deeper than the whole encoded reach.
 * The last case is not a fault: a warm publication carries and remaps the
 * previous closest-point field, so a link can be older than this frame's flood.
 */
fn floodPass(reach:u32)->u32 {
  if(reach==0u){return 0u;}
  for(var k=0u;k<bands.ladderPasses;k+=1u){
    if(bands.prefixReach[k/4u][k%4u]>=reach){return k+1u;}
  }
  return bands.ladderPasses+1u;
}
fn floodColor(passIndex:u32)->vec3f {
  if(passIndex==0u){return vec3f(1.0,1.0,1.0);}
  if(passIndex>bands.ladderPasses){return vec3f(1.0,0.05,0.72);}
  let t=f32(passIndex-1u)/max(f32(bands.ladderPasses-1u),1.0);
  if(t<0.5){return mix(vec3f(0.10,0.28,0.92),vec3f(0.06,0.78,0.80),t*2.0);}
  return mix(vec3f(0.06,0.78,0.80),vec3f(0.98,0.62,0.08),(t-0.5)*2.0);
}
fn floodProvenance(point:vec3f)->vec4f {
  if(arrayLength(&topologyControl)==0u||topologyControl[0]!=0u
    ||arrayLength(&redistanceControl)<=4u||redistanceControl[3]==0u||redistanceControl[4]!=0u){
    return vec4f(1.0,0.01,0.06,0.96);
  }
  if(bands.ladderPasses==0u){return vec4f(0.0);}
  let relative=renderWorldToFine(point);
  if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){return vec4f(0.0);}
  let address=fineAddress(vec3i(floor(relative)));
  if(address==INVALID){return vec4f(0.03,0.09,0.28,0.05);}
  if(address>=arrayLength(&floodSeeds)){return vec4f(1.0,0.01,0.06,0.96);}
  let seed=floodSeeds[address];
  // A resident, valid sample with no seed is a correctness fact, not a cost
  // one, so it gets its own ink rather than the deepest pass colour.
  if(seed==INVALID||seed>=arrayLength(&sampleFlags)){return vec4f(1.0,0.02,0.14,0.92);}
  let delta=abs(vec3i(floodSampleCell(seed))-vec3i(floodSampleCell(address)));
  let passIndex=floodPass(u32(max(max(delta.x,delta.y),delta.z)));
  if(passIndex==0u){return vec4f(1.0,1.0,1.0,0.85);}
  if(passIndex>bands.ladderPasses){return vec4f(floodColor(passIndex),0.80);}
  // Later passes carry less of the field, so they are drawn more strongly:
  // the view is meant to reveal the tail, not the bulk the first pass closes.
  let t=f32(passIndex-1u)/max(f32(bands.ladderPasses-1u),1.0);
  return vec4f(floodColor(passIndex),0.10+0.45*t);
}
fn floodProvenanceVolume(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;
  let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}
  let travel=abs(ray.direction*(interval.y-interval.x))/max(fine.fineCellWidth,1e-9);
  let steps=u32(clamp(ceil(travel.x+travel.y+travel.z+1.0),1.0,512.0));
  let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;
  for(var i=0u;i<512u;i+=1u){
    if(i>=steps||accum.a>0.985){break;}
    let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);
    let relative=renderWorldToFine(point);
    if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){continue;}
    let address=fineAddress(vec3i(floor(relative)));
    if(address==previous){continue;}
    previous=address;
    let sample=floodProvenance(point);
    if(sample.a<=0.001){continue;}
    accum=composite(accum,sample.rgb,sample.a*volumeOpacity());
  }
  return finishVolume(accum);
}
fn bandResidencyVolume(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;
  let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}
  let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);
  for(var i=0u;i<512u;i+=1u){
    if(i>=steps||accum.a>0.985){break;}
    let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);
    let sample=bandResidency(point);
    if(sample.a<=0.001){continue;}
    accum=composite(accum,sample.rgb,sample.a*volumeOpacity());
  }
  return finishVolume(accum);
}
fn fineVolume(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}let travel=abs(ray.direction*(interval.y-interval.x))/max(fine.fineCellWidth,1e-9);let steps=u32(clamp(ceil(travel.x+travel.y+travel.z+1.0),1.0,512.0));let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;
  for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let sample=fineState(point);if(sample.address==previous){continue;}previous=sample.address;let alpha=select(sample.alpha*0.20,sample.alpha,sample.alpha>0.30);accum=composite(accum,sample.color,alpha*volumeOpacity());}
  return finishVolume(accum);
}
fn globalFinePhiVolume(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}let travel=abs(ray.direction*(interval.y-interval.x))/max(fine.fineCellWidth,1e-9);let steps=u32(clamp(ceil(travel.x+travel.y+travel.z+1.0),1.0,512.0));let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;
  for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let relative=renderWorldToFine(point);if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){continue;}let address=fineAddress(vec3i(floor(relative)));if(address==previous){continue;}previous=address;let sample=globalFinePhi(point);if(sample.a<=0.001){continue;}accum=composite(accum,sample.rgb,sample.a*0.12*volumeOpacity());}
  return finishVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  let mode=i32(round(u.debug.w));
  if(i32(round(u.debug.x))==4){if(mode==25){return globalFinePhiVolume(input.uv);}if(mode==26){return bandResidencyVolume(input.uv);}if(mode==31){return floodProvenanceVolume(input.uv);}return fineVolume(input.uv);}
  let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}
  if(mode==31){let sample=floodProvenance(hit.xyz);if(sample.a<=0.001){discard;}return vec4f(displayColor(sample.rgb),sample.a);}
  if(mode==26){let sample=bandResidency(hit.xyz);if(sample.a<=0.001){discard;}return vec4f(displayColor(sample.rgb),sample.a);}
  if(mode==25){let sample=globalFinePhi(hit.xyz);if(sample.a<=0.001){discard;}return vec4f(displayColor(sample.rgb),sample.a);}let sample=fineState(hit.xyz);if(sample.alpha<=0.001){discard;}return vec4f(displayColor(sample.color),sample.alpha);
}`;

export class OctreeTechniqueOverlayPipeline {
  /** One render pipeline per declared program, keyed by program id. */
  private readonly pipelines = new Map<OctreeTechniqueProgramId, GPURenderPipeline>();
  /** Bind groups for those programs plus the lifecycle membership compute pass. */
  private readonly groups = new Map<string, GPUBindGroup>();
  private bandConfig?: GPUBuffer;
  private adaptiveVelocityViewConfig?: GPUBuffer;
  private lifecycleMembershipPipeline?: GPUComputePipeline;
  private source?: OctreeTechniqueDebugSource;
  private ownerRows?: GPUTexture;
  private lifecycleMembership?: GPUBuffer;
  private lifecycleConfig?: GPUBuffer;
  private lifecycleWorklist?: GPUBuffer;
  private lifecycleCapacity=0;
  /** Owns its own pipelines and hop field; see `webgpu-fluid-blast-radius.ts`. */
  private blastRadius?: WebGPUFluidBlastRadius;

  constructor(private readonly device: GPUDevice, private readonly targetFormat: GPUTextureFormat,
    private readonly uniformBuffer: GPUBuffer) {}

  private async pipeline(label: string, code: string): Promise<GPURenderPipeline> {
    const shaderModule=this.device.createShaderModule({label,code});const compilation=await shaderModule.getCompilationInfo();
    const errors=compilation.messages.filter((message)=>message.type==="error");
    if(errors.length)throw new Error(errors.map((error)=>`${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    return this.device.createRenderPipelineAsync({label,layout:"auto",vertex:{module:shaderModule,entryPoint:"vertexMain"},fragment:{module:shaderModule,entryPoint:"fragmentMain",targets:[{format:this.targetFormat,blend:{color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha"},alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha"}}}]},primitive:{topology:"triangle-list"}});
  }

  private async computePipeline(label: string, code: string): Promise<GPUComputePipeline> {
    const shaderModule=this.device.createShaderModule({label,code});const compilation=await shaderModule.getCompilationInfo();
    const errors=compilation.messages.filter((message)=>message.type==="error");
    if(errors.length)throw new Error(errors.map((error)=>`${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    return this.device.createComputePipelineAsync({label,layout:"auto",compute:{module:shaderModule,entryPoint:"main"}});
  }

  async initialize(): Promise<void> {
    // Each program's shader is paired with the declaration that generated its
    // preamble, so a program cannot be built against a different binding set
    // than the one its bind group is resolved from.
    const programs: readonly [OctreeTechniqueProgramId, string][] = [
      ["topology", octreeTechniqueTopologyShader],
      ["face", octreeTechniqueFaceShader],
      ["structured", octreeTechniqueStructuredShader],
      ["adaptiveVelocity", octreeTechniqueAdaptiveVelocityShader],
      ["lifecycle", octreeTechniqueLifecycleShader],
      ["fine", octreeTechniqueFineLifecycleShader],
    ];
    const [built, membership] = await Promise.all([
      Promise.all(programs.map(([id, code]) =>
        this.pipeline(OCTREE_TECHNIQUE_PROGRAMS[id].label, code))),
      this.computePipeline(OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM.label, octreeLifecycleMembershipShader),
    ]);
    for (const [index, [id]] of programs.entries()) this.pipelines.set(id, built[index]);
    this.lifecycleMembershipPipeline = membership;
    const blastRadius=new WebGPUFluidBlastRadius(this.device,this.targetFormat,this.uniformBuffer);
    await blastRadius.initialize();
    this.blastRadius=blastRadius;
    this.blastRadius.setSource(this.ownerRows,this.source?.pressureRows);
    this.rebuildGroups();
  }

  setSource(source: OctreeTechniqueDebugSource | undefined): void {
    if(this.source===source)return;this.source=source;this.rebuildGroups();
    this.blastRadius?.setSource(this.ownerRows,source?.pressureRows);
  }

  setOwnerRows(ownerRows: GPUTexture | undefined): void {
    if(this.ownerRows===ownerRows)return;this.ownerRows=ownerRows;this.rebuildGroups();
    this.blastRadius?.setSource(ownerRows,this.source?.pressureRows);
  }

  /**
   * What a declared resource key names this frame.
   *
   * The programs name their bindings; this is the only place a name becomes a
   * buffer. A key with nothing behind it returns undefined and refuses the whole
   * group, because a partially bound program would read whatever the previous
   * publication happened to leave in that slot.
   */
  private resolveResource(key: string): GPUBindingResource | undefined {
    const source = this.source;
    if (key === "uniforms") return { buffer: this.uniformBuffer };
    if (key === "ownerRows") return this.ownerRows?.createView({ dimension: "3d" });
    if (key === "lifecycleMembership") return this.lifecycleMembership && { buffer: this.lifecycleMembership };
    if (key === "lifecycleConfig") return this.lifecycleConfig && { buffer: this.lifecycleConfig };
    if (key === "lifecycleWorklist") return source?.topologyLifecycle?.tileWorklist;
    if (key === "bandConfig") return this.bandConfig && { buffer: this.bandConfig };
    if (key === "adaptiveVelocityViewConfig") return this.adaptiveVelocityViewConfig
      && { buffer: this.adaptiveVelocityViewConfig };
    if (!source) return undefined;
    const fine = source.fineBandLifecycle;
    const fineResources: Readonly<Record<string, GPUBindingResource | undefined>> = {
      fineParams: fine?.params, fineWorklist: fine?.worklist, fineMetadata: fine?.metadata,
      fineSampleFlags: fine?.samples, fineTopologyControl: fine?.topologyControl,
      fineRedistanceControl: fine?.redistanceControl, finePhi: fine?.samples, fineSeeds: fine?.seeds,
    };
    if (key in fineResources) return fineResources[key];
    const adaptive = source.losassoAdaptiveVelocity;
    const adaptiveResources: Readonly<Record<string, GPUBindingResource | undefined>> = {
      losassoAdaptiveVelocityControl: adaptive?.control,
      losassoAdaptiveVelocityLeaves: adaptive?.leaves,
      losassoAdaptiveNodalVelocity: adaptive?.nodalVelocity,
      losassoAdaptivePhiControl: adaptive?.phiControl,
      losassoAdaptiveRowPhi: adaptive?.rowPhi,
    };
    if (key in adaptiveResources) return adaptiveResources[key];
    const bundle = source as unknown as Record<string, GPUBindingResource | undefined>;
    return typeof bundle[key] === "object" ? bundle[key] : undefined;
  }

  private buildGroup(
    program: VisualizationProgram, pipeline: GPURenderPipeline | GPUComputePipeline | undefined,
  ): GPUBindGroup | undefined {
    if (!pipeline) return undefined;
    const entries = visualizationBindGroupEntries(program, (key) => this.resolveResource(key));
    if (!entries) return undefined;
    return this.device.createBindGroup({
      label: program.label, layout: pipeline.getBindGroupLayout(0), entries: [...entries],
    });
  }

  private rebuildGroups(): void {
    this.groups.clear();
    const source = this.source;
    if (!source) return;
    // The lifecycle views own two buffers the publication does not supply, so
    // those are sized before anything is resolved against them.
    const lifecycle = source.topologyLifecycle;
    if (lifecycle) {
      const capacity = Math.max(1, lifecycle.tileCapacity);
      const worklist = lifecycle.tileWorklist.buffer;
      if (this.lifecycleWorklist !== worklist || this.lifecycleCapacity !== capacity
        || !this.lifecycleMembership || !this.lifecycleConfig) {
        this.lifecycleMembership?.destroy(); this.lifecycleConfig?.destroy();
        this.lifecycleWorklist = worklist; this.lifecycleCapacity = capacity;
        this.lifecycleMembership = this.device.createBuffer({ label: "Octree topology lifecycle membership", size: Math.max(4, capacity * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        this.lifecycleConfig = this.device.createBuffer({ label: "Octree topology lifecycle overlay config", size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      }
      this.device.queue.writeBuffer(this.lifecycleConfig, 0, new Uint32Array([
        lifecycle.tileDimensions[0], lifecycle.tileDimensions[1], lifecycle.tileDimensions[2],
        lifecycle.tileSizeCells, capacity, 0, 0, 0,
      ]));
    }
    const fine = source.fineBandLifecycle;
    if (fine) {
      if (!this.bandConfig) this.bandConfig = this.device.createBuffer({
        label: "Octree band-residency overlay config", size: OCTREE_TECHNIQUE_BAND_CONFIG_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      const bandWords = new Uint32Array(OCTREE_TECHNIQUE_BAND_CONFIG_BYTES / 4);
      bandWords.set([
        Math.max(0, Math.round(fine.bands.pressureBandCells)),
        Math.max(0, Math.round(fine.bands.surfaceBandCells)),
        Math.max(0, Math.round(fine.bands.transportBandFineCells)),
        Math.max(0, Math.round(fine.bands.redistanceBandFineCells)),
      ], 0);
      // An empty ladder means no encode has run yet; the provenance view reads
      // the pass count and draws nothing rather than binning against no schedule.
      const ladder = fine.bands.ladderStrides;
      if (ladder.length > 0) {
        const { prefixReach } = describeFineFloodLadder(ladder);
        bandWords[4] = prefixReach.length;
        bandWords.set(prefixReach, OCTREE_TECHNIQUE_BAND_PREFIX_WORD_OFFSET);
      }
      this.device.queue.writeBuffer(this.bandConfig, 0, bandWords);
    }
    const adaptive = source.losassoAdaptiveVelocity;
    if (adaptive) {
      if (!this.adaptiveVelocityViewConfig) {
        this.adaptiveVelocityViewConfig = this.device.createBuffer({
          label: "Losasso adaptive velocity overlay config", size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
      }
      this.device.queue.writeBuffer(this.adaptiveVelocityViewConfig, 0,
        new Float32Array([adaptive.extensionReach_m, 0, 0, 0]));
    }
    for (const [id, program] of Object.entries(OCTREE_TECHNIQUE_PROGRAMS)) {
      const group = this.buildGroup(program, this.pipelines.get(id as OctreeTechniqueProgramId));
      if (group) this.groups.set(id, group);
    }
    const membership = this.buildGroup(OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM, this.lifecycleMembershipPipeline);
    if (membership) this.groups.set(OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM.id, membership);
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView, modeCode: number): boolean {
    // The cone owns its own flood and draw, because it needs a compute pass over
    // the owner map before anything can be shaded. It is the one view that is
    // not a program of this pipeline, and the catalog says so by declaring no
    // program for it.
    if (modeCode === OCTREE_TECHNIQUE_OVERLAY_CODES["blast-radius"]) {
      return this.blastRadius?.encode(encoder, target) === true;
    }
    const programId = octreeTechniqueProgramForCode(modeCode);
    if (!programId) return false;
    const pipeline = this.pipelines.get(programId);
    const group = this.groups.get(programId);
    if (!pipeline || !group) return false;
    // The lifecycle view reads a membership mask that only exists once the
    // worklist has been expanded, so its compute pass runs ahead of the draw.
    if (programId === "lifecycle") {
      const membership = this.groups.get(OCTREE_LIFECYCLE_MEMBERSHIP_PROGRAM.id);
      if (!this.lifecycleMembership || !this.lifecycleMembershipPipeline || !membership) return false;
      const broker = new PassBroker(encoder);
      broker.clearBuffer(this.lifecycleMembership);
      const compute = broker.compute({ label: "Expand octree topology lifecycle membership" });
      compute.setPipeline(this.lifecycleMembershipPipeline);
      compute.setBindGroup(0, membership);
      compute.dispatchWorkgroups(Math.ceil(this.lifecycleCapacity / 64));
      broker.fence("octree topology lifecycle membership complete");
    }
    const pass = encoder.beginRenderPass({
      label: "Octree paper-technique overlay",
      colorAttachments: [{ view: target, loadOp: "load", storeOp: "store" }],
    });
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.draw(3); pass.end();
    return true;
  }
}
