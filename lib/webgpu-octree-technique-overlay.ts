/** Slice and full-volume paper-technique views for the unified power/octree method. */

import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import type { OctreeTechniqueDebugSource } from "./octree-technique-debug";
import { makeFineLevelSetSortedWorklistLookupWGSL } from "./webgpu-octree-fine-levelset-bricks";
import { PassBroker } from "./webgpu-pass-broker";

const sharedWGSL = /* wgsl */ `
struct Uniforms {
  viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f,
  options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f,
}
struct Leaf { originX:u32, originY:u32, originZ:u32, size:u32, flags:u32, pad0:u32, pad1:u32, pad2:u32, phiGradient:vec4f, motion:vec4f }
struct VertexOutput { @builtin(position) position:vec4f, @location(0) uv:vec2f }
struct CameraRay { origin:vec3f, direction:vec3f }
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->VertexOutput {
  var positions=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(3.0,-1.0),vec2f(-1.0,3.0));
  var result:VertexOutput; result.position=vec4f(positions[index],0.0,1.0); result.uv=positions[index]*0.5+0.5; return result;
}
fn unpackOrigin(word:u32)->vec3u{return vec3u(word&1023u,(word>>10u)&1023u,(word>>20u)&1023u);}
fn powerSigns(code:u32)->vec3f { let bits=code&7u; return vec3f(select(1.0,-1.0,(bits&1u)!=0u),select(1.0,-1.0,(bits&2u)!=0u),select(1.0,-1.0,(bits&4u)!=0u)); }
fn inversePowerTransform(value:vec3f,code:u32)->vec3f {
  let q=value*powerSigns(code); let permutation=(code/8u)%6u;
  if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}if(permutation==2u){return q.yxz;}
  if(permutation==3u){return q.zxy;}if(permutation==4u){return q.yzx;}return q.zyx;
}
fn cameraRay(uv:vec2f)->CameraRay {
  let ndc=uv*2.0-1.0;let origin=u.cameraPosition.xyz;let forward=normalize(u.cameraTarget.xyz-origin);
  var right=cross(forward,vec3f(0.0,1.0,0.0));if(length(right)<1e-5){right=vec3f(1.0,0.0,0.0);}right=normalize(right);let up=normalize(cross(right,forward));
  let direction=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*${CAMERA_TAN_HALF_FOV}+up*ndc.y*${CAMERA_TAN_HALF_FOV});return CameraRay(origin,direction);
}
fn boxInterval(ray:CameraRay,minimum:vec3f,maximum:vec3f)->vec2f {
  let inverse=1.0/select(vec3f(1e-20),ray.direction,abs(ray.direction)>vec3f(1e-20));let a=(minimum-ray.origin)*inverse;let b=(maximum-ray.origin)*inverse;
  let lo=min(a,b);let hi=max(a,b);return vec2f(max(max(lo.x,lo.y),max(lo.z,0.0)),min(min(hi.x,hi.y),hi.z));
}
fn traversalSteps(ray:CameraRay,interval:vec2f)->u32 {
  let scale=u.gridInfo.xyz/max(u.container.xyz,vec3f(1e-9));
  let fineTravel=abs(ray.direction*(interval.y-interval.x))*scale;
  return u32(clamp(ceil(fineTravel.x+fineTravel.y+fineTravel.z+1.0),1.0,512.0));
}
fn volumeOpacity()->f32{return clamp(u.debug.y,0.05,1.0);}
fn composite(accum:vec4f,color:vec3f,alpha:f32)->vec4f {
  let contribution=(1.0-accum.a)*clamp(alpha,0.0,1.0);
  return vec4f(accum.rgb+contribution*color,accum.a+contribution);
}
fn finishVolume(accum:vec4f)->vec4f {
  if(accum.a<=0.001){discard;}
  return vec4f(displayColor(accum.rgb/max(accum.a,1e-6)),accum.a);
}
fn raySegmentDistance(ray:CameraRay,a:vec3f,b:vec3f)->vec2f {
  let edge=b-a;let w=ray.origin-a;let aa=dot(ray.direction,ray.direction);let bb=dot(ray.direction,edge);let cc=max(dot(edge,edge),1e-12);let dd=dot(ray.direction,w);let ee=dot(edge,w);let denominator=aa*cc-bb*bb;
  var t=select(0.0,(bb*ee-cc*dd)/denominator,abs(denominator)>1e-10);var s=clamp((aa*ee-bb*dd)/max(denominator,1e-10),0.0,1.0);t=max(0.0,(bb*s-dd)/aa);s=clamp((bb*t+ee)/cc,0.0,1.0);t=max(0.0,(bb*s-dd)/aa);
  return vec2f(length(ray.origin+ray.direction*t-(a+edge*s)),t);
}
fn sliceRay(uv:vec2f)->vec4f {
  let axis=i32(round(u.debug.x)); if(axis<=0||axis>=4){return vec4f(0.0,0.0,0.0,-1.0);}
  let ray=cameraRay(uv);let origin=ray.origin;let direction=ray.direction;
  let boundsMin=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z); let dims=max(u.gridInfo.xyz,vec3f(1.0));
  var denominator=direction.z; var rayOrigin=origin.z; var coordinate=boundsMin.z+(floor(clamp(u.debug.y,0.0,0.999999)*dims.z)+0.5)*u.container.z/dims.z;
  if(axis==2){denominator=direction.x;rayOrigin=origin.x;coordinate=boundsMin.x+(floor(clamp(u.debug.y,0.0,0.999999)*dims.x)+0.5)*u.container.x/dims.x;}
  if(axis==3){denominator=direction.y;rayOrigin=origin.y;coordinate=(floor(clamp(u.debug.y,0.0,0.999999)*dims.y)+0.5)*u.container.y/dims.y;}
  if(abs(denominator)<=1e-6){return vec4f(0.0,0.0,0.0,-1.0);} let distance=(coordinate-rayOrigin)/denominator;
  return vec4f(origin+direction*distance,distance);
}
fn worldToFine(point:vec3f)->vec3f { let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z); return (point-minimum)/u.container.xyz*u.gridInfo.xyz; }
fn fineToWorld(point:vec3f)->vec3f { let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z); return minimum+point/u.gridInfo.xyz*u.container.xyz; }
fn slice2(point:vec3f)->vec2f { let axis=i32(round(u.debug.x)); if(axis==1){return point.xy;}if(axis==2){return point.zy;}return point.xz; }
fn segmentDistance(point:vec2f,a:vec2f,b:vec2f)->f32 { let edge=b-a;let t=clamp(dot(point-a,edge)/max(dot(edge,edge),1e-10),0.0,1.0);return length(point-(a+t*edge)); }
fn segmentDistance3(point:vec3f,a:vec3f,b:vec3f)->f32 { let edge=b-a;let t=clamp(dot(point-a,edge)/max(dot(edge,edge),1e-10),0.0,1.0);return length(point-(a+t*edge)); }
fn leafOrigin(leaf:Leaf)->vec3u{return vec3u(leaf.originX,leaf.originY,leaf.originZ);}
fn leafContains(leaf:Leaf,pointFine:vec3f)->bool { let origin=vec3f(leafOrigin(leaf)); return leaf.size>0u&&all(pointFine>=origin)&&all(pointFine<origin+vec3f(f32(leaf.size))); }
fn displayColor(linear:vec3f)->vec3f { let mapped=linear/(linear+vec3f(1.0));return pow(max(mapped,vec3f(0.0)),vec3f(1.0/2.2)); }
fn compositeDisplay(accum:vec4f,linearColor:vec3f,alpha:f32)->vec4f{return composite(accum,displayColor(linearColor),alpha);}
fn finishDisplayVolume(accum:vec4f)->vec4f{if(accum.a<=0.001){discard;}return vec4f(accum.rgb/max(accum.a,1e-6),accum.a);}
`;

export const octreeTechniqueTopologyShader = /* wgsl */ `
${sharedWGSL}
struct Metric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct TetraHeader { first:u32, count:u32, flags:u32 }
struct TetraVertex { value:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var ownerRows:texture_3d<u32>;
@group(0) @binding(2) var<storage,read> leaves:array<Leaf>;
@group(0) @binding(3) var<storage,read> metrics:array<Metric>;
@group(0) @binding(4) var<storage,read> tetraHeaders:array<TetraHeader>;
@group(0) @binding(5) var<storage,read> tetrahedra:array<u32>;
@group(0) @binding(6) var<storage,read> tetraVertices:array<TetraVertex>;
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
${sharedWGSL}
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct Metric { topologyCode:u32,transformAndFlags:u32,volume:f32,reserved:u32 }
struct CatalogSlotGeometry { neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f }
@group(0) @binding(0) var<uniform> u:Uniforms;@group(0) @binding(1) var ownerRows:texture_3d<u32>;@group(0) @binding(2) var<storage,read> headers:array<LeafHeader>;@group(0) @binding(3) var<storage,read> metrics:array<Metric>;@group(0) @binding(4) var<storage,read> entryHeaders:array<vec2u>;@group(0) @binding(5) var<storage,read> catalogFaces:array<CatalogSlotGeometry>;
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
${sharedWGSL}
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }struct Metric { topologyCode:u32,transformAndFlags:u32,volume:f32,reserved:u32 }struct StructuredParams { words:array<vec4u,16> }
@group(0) @binding(0) var<uniform> u:Uniforms;@group(0) @binding(1) var ownerRows:texture_3d<u32>;@group(0) @binding(2) var<storage,read> headers:array<LeafHeader>;@group(0) @binding(3) var<storage,read> metrics:array<Metric>;@group(0) @binding(4) var<storage,read> accepted:array<u32>;@group(0) @binding(5) var<storage,read> rowVelocities:array<vec4f>;@group(0) @binding(6) var<uniform> structured:StructuredParams;@group(0) @binding(7) var<storage,read> authority:array<u32>;@group(0) @binding(8) var<storage,read> pressure:array<f32>;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;fn word(i:u32)->u32{return structured.words[i/4u][i%4u];}fn finiteValue(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}fn cellCoord(cell:u32)->vec3u{let d=vec3u(max(u.gridInfo.xyz,vec3f(1.0)));return vec3u(cell%d.x,(cell/d.x)%d.y,cell/(d.x*d.y));}fn rowAt(point:vec3f)->u32{return textureLoad(ownerRows,vec3i(clamp(floor(worldToFine(point)),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0))),0).x;}fn rowValid(row:u32,point:vec3f)->bool{if(row>=arrayLength(&headers)||row>=arrayLength(&metrics)){return false;}let h=headers[row];let origin=vec3f(cellCoord(h.cell));return h.size>0u&&(metrics[row].transformAndFlags&VALID)!=0u&&all(point>=origin)&&all(point<origin+vec3f(f32(h.size)));}fn publicationValid()->bool{return arrayLength(&accepted)>=6u&&accepted[0]==0u&&accepted[3]!=0u;}fn rowCapacity()->u32{return max(1u,word(0u));}fn authorityBase()->u32{return (accepted[4]&1u)*word(15u);}fn heat(value:f32)->vec3f{let t=clamp(value,0.0,1.0);if(t<0.5){return mix(vec3f(0.04,0.20,0.70),vec3f(0.06,0.78,0.55),t*2.0);}return mix(vec3f(0.06,0.78,0.55),vec3f(1.0,0.08,0.025),(t-0.5)*2.0);}
fn rowSample(row:u32,mode:i32,volume:bool)->vec4f{if(!publicationValid()||row>=accepted[2]||row>=rowCapacity()){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let metric=metrics[row];if(!finiteValue(metric.volume)||metric.volume<=0.0){return vec4f(vec3f(1.0,0.01,0.10),0.94);}if(mode==27||mode==30){let at=(accepted[4]&1u)*rowCapacity()+row;if(at>=arrayLength(&rowVelocities)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let velocity=rowVelocities[at];if(velocity.w<=0.0||any(velocity.xyz!=velocity.xyz)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}if(mode==30){let direction=vec3f(0.5)+0.5*velocity.xyz/max(length(velocity.xyz),1e-6);return vec4f(direction,select(0.88,0.13,volume));}return vec4f(heat(length(velocity.xyz)/max(u.environment.z,1e-4)),select(0.88,0.13,volume));}let base=authorityBase();let maxSlots=word(1u);let slotBase=row*maxSlots;var ownPressure=0.0;if(row<arrayLength(&pressure)){ownPressure=pressure[row];}var largest=0.0;var flux=0.0;for(var local=0u;local<${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumFaceIncidence}u;local+=1u){if(local>=maxSlots){break;}let at=slotBase+local;let handleAt=base+word(29u)+at;let signAt=base+word(30u)+at;if(handleAt>=arrayLength(&authority)||signAt>=arrayLength(&authority)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let handle=authority[handleAt];if(handle==INVALID||handle>=accepted[5]){continue;}if(mode==29){let valueAt=base+word(16u)+handle;let areaAt=base+word(20u)+handle;let fractionAt=base+word(22u)+handle;if(max(valueAt,max(areaAt,fractionAt))>=arrayLength(&authority)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}flux+=f32(bitcast<i32>(authority[signAt]))*bitcast<f32>(authority[valueAt])*bitcast<f32>(authority[areaAt])*bitcast<f32>(authority[fractionAt]);}else{let neighborAt=base+word(18u)+handle;let inverseAt=base+word(21u)+handle;if(max(neighborAt,inverseAt)>=arrayLength(&authority)){return vec4f(vec3f(1.0,0.01,0.10),0.94);}let neighbor=authority[neighborAt];var neighborPressure=0.0;if(neighbor<arrayLength(&pressure)){neighborPressure=pressure[neighbor];}largest=max(largest,abs(neighborPressure-ownPressure)*bitcast<f32>(authority[inverseAt]));}}if(mode==28){return vec4f(heat(largest/max(u.environment.z,0.05)),select(0.90,0.13,volume));}let width=f32(headers[row].size)*bitcast<f32>(word(40u));let divergence=flux/max(metric.volume*width*width*width,1e-9);let scaled=clamp(divergence*max(u.environment.y,1e-6),-1.0,1.0);let color=select(mix(vec3f(0.96),vec3f(0.88,0.10,0.08),scaled),mix(vec3f(0.96),vec3f(0.08,0.28,0.88),-scaled),scaled<0.0);return vec4f(color,select(0.92,0.14,volume));}
fn structuredVolume(uv:vec2f,mode:i32)->vec4f{let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let interval=boxInterval(ray,minimum,minimum+u.container.xyz);if(interval.y<=interval.x){discard;}let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let row=rowAt(point);if(row==previous){continue;}previous=row;if(!rowValid(row,worldToFine(point))){continue;}let sample=rowSample(row,mode,true);accum=composite(accum,sample.rgb,sample.a*volumeOpacity());}return finishVolume(accum);}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f{let mode=i32(round(u.debug.w));if(i32(round(u.debug.x))==4){return structuredVolume(input.uv,mode);}let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);if(any(hit.xyz<minimum)||any(hit.xyz>minimum+u.container.xyz)){discard;}let row=rowAt(hit.xyz);if(row==INVALID){discard;}if(!rowValid(row,worldToFine(hit.xyz))){return vec4f(displayColor(vec3f(1.0,0.01,0.10)),0.94);}let sample=rowSample(row,mode,false);return vec4f(displayColor(sample.rgb),sample.a);}
`;

const octreeLifecycleMembershipShader = /* wgsl */ `
struct Config { dimensions:vec3u,tileSize:u32,capacity:u32,pad0:u32,pad1:u32,pad2:u32 }
@group(0) @binding(0) var<storage,read> worklist:array<u32>;
@group(0) @binding(1) var<storage,read_write> membership:array<u32>;
@group(0) @binding(2) var<uniform> config:Config;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u) {
  let slot=id.x;if(slot>=config.capacity){return;}let header=16u;
  let activeCount=min(worklist[0],config.capacity);if(slot<activeCount){let tile=worklist[header+slot];if(tile<arrayLength(&membership)){membership[tile]=1u;}}
  let retired=min(worklist[4],config.capacity);if(slot<retired){let index=header+config.capacity+slot;if(index<arrayLength(&worklist)){let tile=worklist[index];if(tile<arrayLength(&membership)){membership[tile]=2u;}}}
}`;

export const octreeTechniqueLifecycleShader = /* wgsl */ `
${sharedWGSL}
struct Config { dimensions:vec3u,tileSize:u32,capacity:u32,pad0:u32,pad1:u32,pad2:u32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> membership:array<u32>;
@group(0) @binding(2) var<uniform> config:Config;
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
${sharedWGSL}
struct FineParams { brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32 }
struct FineState { color:vec3f,alpha:f32,address:u32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<uniform> fine:FineParams;
@group(0) @binding(2) var<storage,read> worklist:array<u32>;
@group(0) @binding(3) var<storage,read> metadata:array<u32>;
struct BandConfig { pressureBandCells:u32,surfaceBandCells:u32,transportBandFineCells:u32,redistanceBandFineCells:u32 }
@group(0) @binding(4) var<uniform> bands:BandConfig;
@group(0) @binding(5) var<storage,read> sampleFlags:array<u32>;
@group(0) @binding(6) var<storage,read> topologyControl:array<u32>;
@group(0) @binding(7) var<storage,read> redistanceControl:array<u32>;
@group(0) @binding(8) var<storage,read> finePhi:array<f32>;
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;
${makeFineLevelSetSortedWorklistLookupWGSL("fine", "metadata", "worklist", "pageOf")}
fn fineAddress(q:vec3i)->u32 {
  if(any(q<vec3i(0))||any(q>=vec3i(fine.sampleDimensions))){return INVALID;}
  let uq=vec3u(q);let brick=uq/max(fine.brickResolution,1u);let key=brick.x+fine.brickDimensions.x*(brick.y+fine.brickDimensions.y*brick.z);let page=pageOf(key);
  if(page==INVALID||page>=fine.pageCapacity||page*10u+3u>=arrayLength(&metadata)||metadata[page*10u+2u]!=fine.generation){return INVALID;}
  let local=uq-brick*fine.brickResolution;let localIndex=local.x+fine.brickResolution*(local.y+fine.brickResolution*local.z);let address=page*fine.samplesPerBrick+localIndex;
  return select(INVALID,address,address<arrayLength(&sampleFlags)&&address<arrayLength(&finePhi)&&(sampleFlags[address]&VALID)!=0u);
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
  let local=q-brick*fine.brickResolution;let localIndex=local.x+fine.brickResolution*(local.y+fine.brickResolution*local.z);let address=page*fine.samplesPerBrick+localIndex;if(address>=arrayLength(&sampleFlags)){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}let flags=sampleFlags[address];
  if((flags&VALID)==0u){return FineState(vec3f(0.26,0.08,0.48),0.20,address);}if((flags&INTERFACE)!=0u){return FineState(vec3f(1.0,0.02,0.44),0.90,address);}return FineState(vec3f(0.04,0.72,0.82),0.34,address);
}
fn globalFinePhi(point:vec3f)->vec4f {
  if(arrayLength(&topologyControl)==0u||topologyControl[0]!=0u||arrayLength(&redistanceControl)<=4u||redistanceControl[3]==0u||redistanceControl[4]!=0u){return vec4f(1.0,0.01,0.06,0.96);}
  let relative=renderWorldToFine(point);if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){return vec4f(0.0);}
  let q=vec3i(floor(relative));let centerAddress=fineAddress(q);if(centerAddress==INVALID){return vec4f(0.02,0.06,0.18,0.08);}
  let center=finePhi[centerAddress];if(center!=center||abs(center)>=3.402823e38){return vec4f(1.0,0.01,0.06,0.96);}
  var gradient=vec3f(0.0);var complete=true;let h=max(fine.fineCellWidth,1e-9);
  for(var axis=0u;axis<3u;axis+=1u){var delta=vec3i(0);delta[axis]=1;let lo=fineAddress(q-delta);let hi=fineAddress(q+delta);var derivative=0.0;
    if(lo!=INVALID&&hi!=INVALID){derivative=(finePhi[hi]-finePhi[lo])/(2.0*h);}else if(hi!=INVALID){derivative=(finePhi[hi]-center)/h;}else if(lo!=INVALID){derivative=(center-finePhi[lo])/h;}else{complete=false;}
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
  let phi=finePhi[address];
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
  if(i32(round(u.debug.x))==4){if(mode==25){return globalFinePhiVolume(input.uv);}if(mode==26){return bandResidencyVolume(input.uv);}return fineVolume(input.uv);}
  let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}
  if(mode==26){let sample=bandResidency(hit.xyz);if(sample.a<=0.001){discard;}return vec4f(displayColor(sample.rgb),sample.a);}
  if(mode==25){let sample=globalFinePhi(hit.xyz);if(sample.a<=0.001){discard;}return vec4f(displayColor(sample.rgb),sample.a);}let sample=fineState(hit.xyz);if(sample.alpha<=0.001){discard;}return vec4f(displayColor(sample.color),sample.alpha);
}`;

export class OctreeTechniqueOverlayPipeline {
  private topologyPipeline?: GPURenderPipeline;
  private facePipeline?: GPURenderPipeline;
  private structuredPipeline?: GPURenderPipeline;
  private lifecyclePipeline?: GPURenderPipeline;
  private fineLifecyclePipeline?: GPURenderPipeline;
  private bandConfig?: GPUBuffer;
  private lifecycleMembershipPipeline?: GPUComputePipeline;
  private source?: OctreeTechniqueDebugSource;
  private ownerRows?: GPUTexture;
  private topologyGroup?: GPUBindGroup;
  private faceGroup?: GPUBindGroup;
  private structuredGroup?: GPUBindGroup;
  private lifecycleGroup?: GPUBindGroup;
  private fineLifecycleGroup?: GPUBindGroup;
  private lifecycleMembershipGroup?: GPUBindGroup;
  private lifecycleMembership?: GPUBuffer;
  private lifecycleConfig?: GPUBuffer;
  private lifecycleWorklist?: GPUBuffer;
  private lifecycleCapacity=0;

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
    [this.topologyPipeline,this.facePipeline,this.structuredPipeline,this.lifecyclePipeline,this.fineLifecyclePipeline,this.lifecycleMembershipPipeline]=await Promise.all([
      this.pipeline("Octree topology technique overlay",octreeTechniqueTopologyShader),
      this.pipeline("Octree generalized-face technique overlay",octreeTechniqueFaceShader),
      this.pipeline("Octree structured-field technique overlay",octreeTechniqueStructuredShader),
      this.pipeline("Octree topology-lifecycle overlay",octreeTechniqueLifecycleShader),
      this.pipeline("Octree fine-band lifecycle overlay",octreeTechniqueFineLifecycleShader),
      this.computePipeline("Octree topology-lifecycle membership",octreeLifecycleMembershipShader),
    ]);this.rebuildGroups();
  }

  setSource(source: OctreeTechniqueDebugSource | undefined): void {
    if(this.source===source)return;this.source=source;this.rebuildGroups();
  }

  setOwnerRows(ownerRows: GPUTexture | undefined): void {
    if(this.ownerRows===ownerRows)return;this.ownerRows=ownerRows;this.rebuildGroups();
  }

  private rebuildGroups(): void {
    this.topologyGroup=undefined;this.faceGroup=undefined;this.structuredGroup=undefined;this.lifecycleGroup=undefined;this.fineLifecycleGroup=undefined;this.lifecycleMembershipGroup=undefined;
    const source=this.source;if(!source)return;const ownerRows=this.ownerRows;
    if(ownerRows&&this.topologyPipeline)this.topologyGroup=this.device.createBindGroup({layout:this.topologyPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:ownerRows.createView({dimension:"3d"})},
      {binding:2,resource:source.leaves},{binding:3,resource:source.topologyMetrics},{binding:4,resource:source.tetrahedronHeaders},
      {binding:5,resource:source.tetrahedra},{binding:6,resource:source.tetrahedronVertices},
    ]});
    if(ownerRows&&this.facePipeline)this.faceGroup=this.device.createBindGroup({layout:this.facePipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:ownerRows.createView({dimension:"3d"})},
      {binding:2,resource:source.leafHeaders},{binding:3,resource:source.topologyMetrics},
      {binding:4,resource:source.catalogEntryHeaders},{binding:5,resource:source.catalogFaces},
    ]});
    if(ownerRows&&this.structuredPipeline)this.structuredGroup=this.device.createBindGroup({layout:this.structuredPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:ownerRows.createView({dimension:"3d"})},
      {binding:2,resource:source.leafHeaders},{binding:3,resource:source.topologyMetrics},
      {binding:4,resource:source.structuredControl},{binding:5,resource:source.structuredRowVelocities},
      {binding:6,resource:source.structuredParams},{binding:7,resource:source.structuredAuthority},
      {binding:8,resource:source.pressure},
    ]});
    const lifecycle=source.topologyLifecycle;
    if(lifecycle){
      const capacity=Math.max(1,lifecycle.tileCapacity);const worklist=lifecycle.tileWorklist.buffer;
      if(this.lifecycleWorklist!==worklist||this.lifecycleCapacity!==capacity||!this.lifecycleMembership||!this.lifecycleConfig){
        this.lifecycleMembership?.destroy();this.lifecycleConfig?.destroy();this.lifecycleWorklist=worklist;this.lifecycleCapacity=capacity;
        this.lifecycleMembership=this.device.createBuffer({label:"Octree topology lifecycle membership",size:Math.max(4,capacity*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
        this.lifecycleConfig=this.device.createBuffer({label:"Octree topology lifecycle overlay config",size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      }
      this.device.queue.writeBuffer(this.lifecycleConfig,0,new Uint32Array([
        lifecycle.tileDimensions[0],lifecycle.tileDimensions[1],lifecycle.tileDimensions[2],lifecycle.tileSizeCells,capacity,0,0,0,
      ]));
      if(this.lifecyclePipeline)this.lifecycleGroup=this.device.createBindGroup({layout:this.lifecyclePipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:{buffer:this.lifecycleMembership}},{binding:2,resource:{buffer:this.lifecycleConfig}},
      ]});
      if(this.lifecycleMembershipPipeline)this.lifecycleMembershipGroup=this.device.createBindGroup({layout:this.lifecycleMembershipPipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:lifecycle.tileWorklist},{binding:1,resource:{buffer:this.lifecycleMembership}},{binding:2,resource:{buffer:this.lifecycleConfig}},
      ]});
    }
    const fine=source.fineBandLifecycle;
    if(fine&&this.fineLifecyclePipeline){
      if(!this.bandConfig)this.bandConfig=this.device.createBuffer({
        label:"Octree band-residency overlay config",size:16,
        usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      this.device.queue.writeBuffer(this.bandConfig,0,new Uint32Array([
        Math.max(0,Math.round(fine.bands.pressureBandCells)),
        Math.max(0,Math.round(fine.bands.surfaceBandCells)),
        Math.max(0,Math.round(fine.bands.transportBandFineCells)),
        Math.max(0,Math.round(fine.bands.redistanceBandFineCells)),
      ]));
      this.fineLifecycleGroup=this.device.createBindGroup({layout:this.fineLifecyclePipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:fine.params},{binding:2,resource:fine.worklist},
        {binding:3,resource:fine.metadata},{binding:4,resource:{buffer:this.bandConfig}},{binding:5,resource:fine.sampleFlags},
        {binding:6,resource:fine.topologyControl},{binding:7,resource:fine.redistanceControl},{binding:8,resource:fine.phi},
      ]});
    }
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView, modeCode: number): boolean {
    let pipeline:GPURenderPipeline|undefined;let group:GPUBindGroup|undefined;
    if(modeCode===12||modeCode===14||modeCode===15){pipeline=this.topologyPipeline;group=this.topologyGroup;}
    else if(modeCode===13||modeCode===16){pipeline=this.facePipeline;group=this.faceGroup;}
    else if(modeCode>=27&&modeCode<=30){pipeline=this.structuredPipeline;group=this.structuredGroup;}
    else if(modeCode===17){pipeline=this.lifecyclePipeline;group=this.lifecycleGroup;if(this.lifecycleMembership&&this.lifecycleMembershipPipeline&&this.lifecycleMembershipGroup){const broker=new PassBroker(encoder);broker.clearBuffer(this.lifecycleMembership);const compute=broker.compute({label:"Expand octree topology lifecycle membership"});compute.setPipeline(this.lifecycleMembershipPipeline);compute.setBindGroup(0,this.lifecycleMembershipGroup);compute.dispatchWorkgroups(Math.ceil(this.lifecycleCapacity/64));broker.fence("octree topology lifecycle membership complete");}else{return false;}}
    else if(modeCode===18||modeCode===25||modeCode===26){pipeline=this.fineLifecyclePipeline;group=this.fineLifecycleGroup;}
    else{return false;}
    if(!pipeline||!group)return false;
    const pass=encoder.beginRenderPass({label:"Octree paper-technique overlay",colorAttachments:[{view:target,loadOp:"load",storeOp:"store"}]});
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();return true;
  }
}
