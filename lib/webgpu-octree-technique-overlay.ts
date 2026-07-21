/** Slice and full-volume paper-technique views for the unified power/octree method. */

import { CAMERA_TAN_HALF_FOV } from "./webgpu-camera";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import type { OctreeTechniqueDebugSource } from "./octree-technique-debug";
import { OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW } from "./octree-face-fast-march";

const sharedWGSL = /* wgsl */ `
struct Uniforms {
  viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f,
  options:vec4f, gridInfo:vec4f, debug:vec4f, environment:vec4f,
}
struct Leaf { packedOrigin:u32, size:u32, flags:u32, pad:u32, phiGradient:vec4f, motion:vec4f }
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
fn leafContains(leaf:Leaf,pointFine:vec3f)->bool { let origin=vec3f(unpackOrigin(leaf.packedOrigin)); return leaf.size>0u&&all(pointFine>=origin)&&all(pointFine<origin+vec3f(f32(leaf.size))); }
fn displayColor(linear:vec3f)->vec3f { let mapped=linear/(linear+vec3f(1.0));return pow(max(mapped,vec3f(0.0)),vec3f(1.0/2.2)); }
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
fn topologyFault(pointFine:vec3f)->vec4f {
  let cell=vec3i(clamp(floor(pointFine),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));
  let row=textureLoad(ownerRows,cell,0).x;
  if(row==INVALID||row>=arrayLength(&leaves)||row>=arrayLength(&metrics)||!leafContains(leaves[row],pointFine)){return vec4f(vec3f(1.0,0.01,0.18),0.88);}
  let metric=metrics[row];if((metric.transformAndFlags&VALID)==0u||metric.topologyCode>=arrayLength(&tetraHeaders)){return vec4f(vec3f(1.0,0.01,0.06),0.92);}
  return vec4f(0.0);
}
fn volumeTopology(uv:vec2f,mode:i32)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;
  let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}
  let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);let baseWidth=max(min(min(u.container.x/u.gridInfo.x,u.container.y/u.gridInfo.y),u.container.z/u.gridInfo.z)*0.10,1e-5);
  var accum=vec4f(0.0);var previous=INVALID;
  for(var sample=0u;sample<512u;sample+=1u){if(sample>=steps||accum.a>0.985){break;}let t=interval.x+(f32(sample)+0.5)*dt;let point=ray.origin+ray.direction*t;let pointFine=worldToFine(point);let cell=vec3i(clamp(floor(pointFine),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));let row=textureLoad(ownerRows,cell,0).x;
    if(row==previous){continue;}previous=row;
    let fault=topologyFault(pointFine);if(fault.a>0.0){accum=composite(accum,fault.rgb,0.32*volumeOpacity());continue;}
    let leaf=leaves[row];let metric=metrics[row];let header=tetraHeaders[metric.topologyCode];let transition=(header.flags&1u)==0u;let boundary=((metric.transformAndFlags>>8u)&63u)!=0u;
    let centerFine=vec3f(unpackOrigin(leaf.packedOrigin))+vec3f(0.5*f32(leaf.size));let centerWorld=fineToWorld(centerFine);let site=1.0-smoothstep(baseWidth,2.8*baseWidth,raySegmentDistance(ray,centerWorld,centerWorld+vec3f(0.0,1e-8,0.0)).x);
    if(mode==12){let base=select(vec3f(0.08,0.52,0.42),vec3f(0.42,0.14,0.70),transition);accum=composite(accum,mix(base,vec3f(1.0,0.73,0.12),site),volumeOpacity()*(0.045+0.34*site));continue;}
    if(mode==15){if(transition||boundary){let color=select(vec3f(0.94,0.45,0.06),vec3f(0.96,0.08,0.40),boundary);accum=composite(accum,color,volumeOpacity()*0.42);}continue;}
    if(mode!=14||!transition||header.first>arrayLength(&tetrahedra)||header.count>arrayLength(&tetrahedra)-header.first){continue;}
    var ink=site;let count=min(header.count,${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumTetrahedra}u);let transform=metric.transformAndFlags&63u;let width=max(baseWidth,t*1.44/max(u.viewport.y,1.0));
    for(var local=0u;local<count;local+=1u){let packed=tetrahedra[header.first+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(arrayLength(&tetraVertices)))){ink=1.0;continue;}let a=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.x].value.xyz,transform);let b=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.y].value.xyz,transform);let c=centerFine+f32(leaf.size)*inversePowerTransform(tetraVertices[selectors.z].value.xyz,transform);let aw=fineToWorld(a);let bw=fineToWorld(b);let cw=fineToWorld(c);
      ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centerWorld,aw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centerWorld,bw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centerWorld,cw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,aw,bw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,bw,cw).x));ink=max(ink,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,cw,aw).x));}
    if(ink>0.01){accum=composite(accum,mix(vec3f(0.12,0.72,0.86),vec3f(1.0,0.72,0.12),site),volumeOpacity()*0.92*ink);}
  }
  return finishVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  let mode=i32(round(u.debug.w));if(i32(round(u.debug.x))==4){return volumeTopology(input.uv,mode);}
  let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;
  if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}let pointFine=worldToFine(hit.xyz);let cell=vec3i(clamp(floor(pointFine),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));
  let row=textureLoad(ownerRows,cell,0).x;let footprint=max(hit.w*1.44/max(u.viewport.y,1.0),1e-5);
  if(row==INVALID||row>=arrayLength(&leaves)||row>=arrayLength(&metrics)||!leafContains(leaves[row],pointFine)){return vec4f(displayColor(vec3f(1.0,0.01,0.18)),0.94);}
  let leaf=leaves[row];let metric=metrics[row];let valid=(metric.transformAndFlags&VALID)!=0u&&metric.topologyCode<arrayLength(&tetraHeaders);
  if(!valid){return vec4f(displayColor(vec3f(1.0,0.01,0.06)),0.96);}let header=tetraHeaders[metric.topologyCode];let transition=(header.flags&1u)==0u;
  let boundary=((metric.transformAndFlags>>8u)&63u)!=0u;let centerFine=vec3f(unpackOrigin(leaf.packedOrigin))+vec3f(0.5*f32(leaf.size));let centerWorld=fineToWorld(centerFine);
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

export const octreeTechniqueFaceShader = /* wgsl */ `
${sharedWGSL}
struct PowerFace { negativeRow:u32, positiveRow:u32, geometryCode:u32, flags:u32, normalVelocity:f32, area:f32, inverseDistance:f32, openFraction:f32 }
struct RowWork { faceCount:u32, incidenceCount:u32, faceOffset:u32, incidenceOffset:u32 }
struct Incidence { face:u32, sign:i32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var ownerRows:texture_3d<u32>;
@group(0) @binding(2) var<storage,read> leaves:array<Leaf>;
@group(0) @binding(3) var<storage,read> faces:array<PowerFace>;
@group(0) @binding(4) var<storage,read> normals:array<vec4f>;
@group(0) @binding(5) var<storage,read> centroids:array<vec4f>;
@group(0) @binding(6) var<storage,read> rows:array<RowWork>;
@group(0) @binding(7) var<storage,read> incidences:array<Incidence>;
@group(0) @binding(8) var<storage,read> control:array<u32>;
const INVALID:u32=0xffffffffu;const FACE_VALID:u32=0x80000000u;const BOUNDARY:u32=2u;const OPEN_BOUNDARY:u32=4u;
fn heat(value:f32)->vec3f { let t=clamp(value,0.0,1.0);return select(mix(vec3f(0.05,0.25,0.60),vec3f(0.10,0.75,0.55),t*2.0),mix(vec3f(0.10,0.75,0.55),vec3f(0.98,0.18,0.04),(t-0.5)*2.0),t<0.5); }
fn volumeFaces(uv:vec2f,mode:i32)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}
  if(arrayLength(&control)<=8u||control[8]!=FACE_VALID){return vec4f(displayColor(vec3f(1.0,0.01,0.06)),0.96);}
  let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);let fineDirection=ray.direction/u.container.xyz*u.gridInfo.xyz;let rayFineOrigin=worldToFine(ray.origin);var accum=vec4f(0.0);var previous=INVALID;
  for(var sample=0u;sample<512u;sample+=1u){if(sample>=steps||accum.a>0.985){break;}let t=interval.x+(f32(sample)+0.5)*dt;let point=ray.origin+ray.direction*t;let pointFine=worldToFine(point);let cell=vec3i(clamp(floor(pointFine),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));let row=textureLoad(ownerRows,cell,0).x;if(row==previous){continue;}previous=row;
    if(row==INVALID||row>=arrayLength(&leaves)||row>=arrayLength(&rows)||!leafContains(leaves[row],pointFine)){accum=composite(accum,vec3f(1.0,0.01,0.18),0.32*volumeOpacity());continue;}
    let work=rows[row];if(work.incidenceOffset>arrayLength(&incidences)||work.incidenceCount>arrayLength(&incidences)-work.incidenceOffset){accum=composite(accum,vec3f(1.0,0.01,0.06),0.44*volumeOpacity());continue;}
    var faceInk=0.0;var dualInk=0.0;var normalInk=0.0;var boundaryInk=0.0;var coefficient=0.0;let width=max(t*1.44/max(u.viewport.y,1.0),min(min(u.container.x/u.gridInfo.x,u.container.y/u.gridInfo.y),u.container.z/u.gridInfo.z)*0.08);
    let centreA=fineToWorld(vec3f(unpackOrigin(leaves[row].packedOrigin))+vec3f(0.5*f32(leaves[row].size)));
    for(var local=0u;local<min(work.incidenceCount,48u);local+=1u){let incidence=incidences[work.incidenceOffset+local];if(incidence.face>=arrayLength(&faces)||incidence.face>=arrayLength(&normals)||incidence.face>=arrayLength(&centroids)){faceInk=1.0;boundaryInk=1.0;continue;}let face=faces[incidence.face];if((face.flags&FACE_VALID)==0u){continue;}let centroidFine=centroids[incidence.face].xyz;let centroidWorld=fineToWorld(centroidFine);let normal=normalize(normals[incidence.face].xyz);let denominator=dot(fineDirection,normal);if(abs(denominator)>1e-8){let faceT=dot(centroidFine-rayFineOrigin,normal)/denominator;if(faceT>=interval.x&&faceT<=interval.y){let at=rayFineOrigin+fineDirection*faceT;let delta=at-centroidFine;let radius=sqrt(max(face.area,1e-8)/3.14159265);let radial=length(delta-normal*dot(delta,normal));let fineWidth=width*max(u.gridInfo.x/u.container.x,1.0);faceInk=max(faceInk,1.0-smoothstep(radius,radius+2.0*fineWidth,radial));}}
      if(face.positiveRow!=INVALID&&face.positiveRow<arrayLength(&leaves)){let other=select(face.negativeRow,face.positiveRow,face.negativeRow==row);if(other<arrayLength(&leaves)){let centreB=fineToWorld(vec3f(unpackOrigin(leaves[other].packedOrigin))+vec3f(0.5*f32(leaves[other].size)));dualInk=max(dualInk,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centreA,centreB).x));}}
      let radius=sqrt(max(face.area,1e-8)/3.14159265);let normalEnd=fineToWorld(centroidFine+normal*max(0.35,radius*0.55));normalInk=max(normalInk,1.0-smoothstep(width,2.2*width,raySegmentDistance(ray,centroidWorld,normalEnd).x));coefficient=max(coefficient,face.area*face.inverseDistance*face.openFraction);if((face.flags&(BOUNDARY|OPEN_BOUNDARY))!=0u){boundaryInk=max(boundaryInk,faceInk);}}
    if(mode==16){let scaled=coefficient/max(f32(leaves[row].size),1.0);accum=composite(accum,heat(scaled),0.075*volumeOpacity());continue;}
    if(mode==13){let ink=max(faceInk,max(0.75*dualInk,normalInk));if(ink>0.01){var color=mix(vec3f(0.55,0.18,0.82),vec3f(0.08,0.82,0.92),faceInk);color=mix(color,vec3f(1.0,0.72,0.10),normalInk);color=mix(color,vec3f(1.0,0.10,0.18),boundaryInk);accum=composite(accum,color,0.94*ink*volumeOpacity());}}
  }
  return finishVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  let mode=i32(round(u.debug.w));if(i32(round(u.debug.x))==4){return volumeFaces(input.uv,mode);}
  let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}
  let pointFine=worldToFine(hit.xyz);let cell=vec3i(clamp(floor(pointFine),vec3f(0.0),u.gridInfo.xyz-vec3f(1.0)));let row=textureLoad(ownerRows,cell,0).x;let footprint=max(hit.w*1.44/max(u.viewport.y,1.0),1e-5);
  if(arrayLength(&control)<=8u||control[8]!=FACE_VALID){return vec4f(displayColor(vec3f(1.0,0.01,0.06)),0.96);}if(row==INVALID||row>=arrayLength(&leaves)||row>=arrayLength(&rows)||!leafContains(leaves[row],pointFine)){return vec4f(displayColor(vec3f(1.0,0.01,0.18)),0.94);}
  let work=rows[row];if(work.incidenceOffset>arrayLength(&incidences)||work.incidenceCount>arrayLength(&incidences)-work.incidenceOffset){return vec4f(displayColor(vec3f(1.0,0.01,0.06)),0.96);}
  var faceInk=0.0;var dualInk=0.0;var normalInk=0.0;var coefficient=0.0;var boundaryInk=0.0;let point2=slice2(hit.xyz);
  for(var local=0u;local<min(work.incidenceCount,48u);local+=1u){let incidence=incidences[work.incidenceOffset+local];if(incidence.face>=arrayLength(&faces)||incidence.face>=arrayLength(&normals)||incidence.face>=arrayLength(&centroids)){return vec4f(displayColor(vec3f(1.0,0.01,0.06)),0.96);}let face=faces[incidence.face];if((face.flags&FACE_VALID)==0u){continue;}let centroidFine=centroids[incidence.face].xyz;let centroidWorld=fineToWorld(centroidFine);let normal=normalize(normals[incidence.face].xyz);let planeDistance=abs(dot(pointFine-centroidFine,normal));let radius=sqrt(max(face.area,1e-8)/3.14159265);let radial=length((pointFine-centroidFine)-normal*dot(pointFine-centroidFine,normal));let line=(1.0-smoothstep(0.6,1.8,planeDistance/max(footprint*max(u.gridInfo.x/u.container.x,1.0),1e-4)))*(1.0-smoothstep(radius,1.25*radius,radial));faceInk=max(faceInk,line);
    let centreA=fineToWorld(vec3f(unpackOrigin(leaves[row].packedOrigin))+vec3f(0.5*f32(leaves[row].size)));if(face.positiveRow!=INVALID&&face.positiveRow<arrayLength(&leaves)){let other=select(face.negativeRow,face.positiveRow,face.negativeRow==row);if(other<arrayLength(&leaves)){let centreB=fineToWorld(vec3f(unpackOrigin(leaves[other].packedOrigin))+vec3f(0.5*f32(leaves[other].size)));dualInk=max(dualInk,1.0-smoothstep(footprint,2.2*footprint,segmentDistance(point2,slice2(centreA),slice2(centreB))));}}
    let normalEnd=fineToWorld(centroidFine+normal*max(0.35,radius*0.55));normalInk=max(normalInk,1.0-smoothstep(footprint,2.2*footprint,segmentDistance(point2,slice2(centroidWorld),slice2(normalEnd))));coefficient=max(coefficient,face.area*face.inverseDistance*face.openFraction);if((face.flags&(BOUNDARY|OPEN_BOUNDARY))!=0u){boundaryInk=max(boundaryInk,line);}}
  if(mode==16){let scaled=coefficient/max(f32(leaves[row].size),1.0);return vec4f(displayColor(heat(scaled)),0.78);}
  if(mode!=13){discard;}let ink=max(faceInk,max(0.75*dualInk,normalInk));if(ink<0.02){discard;}var color=mix(vec3f(0.55,0.18,0.82),vec3f(0.08,0.82,0.92),faceInk);color=mix(color,vec3f(1.0,0.72,0.10),normalInk);color=mix(color,vec3f(1.0,0.10,0.18),boundaryInk);return vec4f(displayColor(color),0.94*ink);
}`;

/** Existing Section 5 band rows/faces rendered in-place. Binding count is
 * seven storage buffers (plus the shared uniform), below the portable limit. */
export const octreeTechniqueSection5FaceBandShader = /* wgsl */ `
${sharedWGSL}
struct Row { cell:u32,globalRow:u32,flags:u32,size:u32,representativePhi:f32,minimumPhi:f32,maximumPhi:f32,padf:f32 }
struct Face { negativeRow:u32,positiveRow:u32,axisSpan:u32,globalFace:u32,velocity:vec4f,centroid:vec4f,phi:f32,area:f32,flags:u32,pad:u32 }
struct State { velocity:vec4f,parent:u32,depth:u32,status:u32,pad:u32 }
struct BandSample { color:vec3f,alpha:f32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> rowHash:array<u32>;
@group(0) @binding(2) var<storage,read> rows:array<Row>;
@group(0) @binding(3) var<storage,read> faces:array<Face>;
@group(0) @binding(4) var<storage,read> incidence:array<u32>;
@group(0) @binding(5) var<storage,read> states:array<State>;
@group(0) @binding(6) var<storage,read> control:array<u32>;
@group(0) @binding(7) var<storage,read> transitionControl:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=0x80000000u;const LIVE:u32=1u;const PHI_VALID:u32=4u;
const UNKNOWN:u32=0u;const TRIAL:u32=1u;const ACCEPTED:u32=2u;const STRIDE:u32=${OCTREE_REGULAR_BAND_INCIDENCE_PER_ROW}u;
fn finite(value:f32)->bool { return value==value&&abs(value)<3.402823e38; }
fn hashKey(k:u32)->u32 { var v=k*0x9e3779b1u;v=(v^(v>>16u))*0x7feb352du;return v^(v>>15u); }
fn cellKey(q:vec3u)->u32 { let dims=vec3u(max(u.gridInfo.xyz,vec3f(1.0)));return q.x+dims.x*(q.y+dims.y*q.z); }
fn findRow(key:u32)->u32 {
  let capacity=arrayLength(&rowHash)/2u;if(capacity==0u||(capacity&(capacity-1u))!=0u){return INVALID;}
  let wanted=key+1u;let start=hashKey(wanted)&(capacity-1u);
  for(var probe=0u;probe<32u;probe+=1u){let slot=(start+probe)&(capacity-1u);let observed=rowHash[slot*2u];if(observed==0u){return INVALID;}if(observed==wanted){let encoded=rowHash[slot*2u+1u];return select(INVALID,encoded-1u,encoded!=0u&&encoded!=INVALID);}}
  return INVALID;
}
fn containingRow(pointFine:vec3f)->u32 {
  let dims=vec3u(max(u.gridInfo.xyz,vec3f(1.0)));let q=vec3u(clamp(floor(pointFine),vec3f(0.0),vec3f(dims)-vec3f(1.0)));
  var size=1u;loop{let origin=(q/vec3u(size))*vec3u(size);let row=findRow(cellKey(origin));if(row<arrayLength(&rows)){let r=rows[row];if(r.size==size&&all(q>=origin)&&all(q<origin+vec3u(size))){return row;}}if(size>=32u||size>=max(max(dims.x,dims.y),dims.z)){break;}size*=2u;}return INVALID;
}
fn rowSample(row:u32,pointFine:vec3f,volume:bool)->BandSample {
  if(arrayLength(&control)<16u||arrayLength(&transitionControl)<16u||row>=arrayLength(&rows)){return BandSample(vec3f(1.0,0.01,0.08),0.94);}
  let rowCapacity=arrayLength(&incidence)/(STRIDE+1u);if(row>=rowCapacity||row>=control[2]){return BandSample(vec3f(1.0,0.01,0.08),0.94);}
  let r=rows[row];let first=control[1];var color=vec3f(0.12,0.26,0.52);var alpha=select(0.38,0.075,volume);
  if((r.flags&4u)!=0u){color=vec3f(0.96,0.08,0.50);}else if((r.flags&8u)!=0u){color=vec3f(0.96,0.38,0.05);}else if((r.flags&16u)!=0u){color=vec3f(0.55,0.20,0.76);}else if((r.flags&32u)!=0u){color=vec3f(0.08,0.68,0.82);}else if((r.flags&64u)!=0u){color=vec3f(0.16,0.48,0.88);}
  if((r.flags&1u)==0u||!finite(r.representativePhi)||r.minimumPhi>r.maximumPhi){return BandSample(vec3f(1.0,0.02,0.08),select(0.96,0.55,volume));}
  let count=min(incidence[row],STRIDE);var accepted=0u;var trial=0u;var unresolved=0u;var acceptedInk=0.0;var trialInk=0.0;var unresolvedInk=0.0;var firstMatch=false;
  let footprint=max(0.08,0.9*length(u.gridInfo.xyz/u.container.xyz)*1.44/max(u.viewport.y,1.0));
  for(var local=0u;local<count;local+=1u){let index=incidence[rowCapacity+row*STRIDE+local];if(index>=arrayLength(&faces)||index>=arrayLength(&states)){unresolved+=1u;continue;}let f=faces[index];let status=states[index].status;let faceValid=(f.flags&(LIVE|PHI_VALID))==(LIVE|PHI_VALID)&&finite(f.phi);if(!faceValid){unresolved+=1u;}else if(status==ACCEPTED){accepted+=1u;}else if(status==TRIAL){trial+=1u;}else{unresolved+=1u;}
    let axis=f.axisSpan&3u;if(axis<3u){var delta=pointFine-f.centroid.xyz;let plane=abs(delta[axis]);delta[axis]=0.0;let radius=sqrt(max(f.area,1e-6)/3.14159265);let ink=(1.0-smoothstep(0.05,0.22,plane))*(1.0-smoothstep(radius,radius+footprint,length(delta)));if(faceValid&&status==ACCEPTED){acceptedInk=max(acceptedInk,ink);}else if(faceValid&&status==TRIAL){trialInk=max(trialInk,ink);}else{unresolvedInk=max(unresolvedInk,ink);}}
    firstMatch=firstMatch||first==f.globalFace||first==index;
  }
  firstMatch=firstMatch||first==r.cell||first==row;
  if(unresolved>0u){color=vec3f(0.88,0.12,0.70);alpha=max(alpha,select(0.88,0.34,volume));}else if(trial>0u){color=vec3f(1.0,0.82,0.03);alpha=max(alpha,select(0.84,0.28,volume));}else if(accepted==count&&count>0u){color=mix(color,vec3f(0.03,0.80,0.68),0.58);}
  if(acceptedInk>0.01){color=mix(color,vec3f(0.10,0.82,0.72),acceptedInk);alpha=max(alpha,0.92*acceptedInk);}
  if(trialInk>0.01){color=mix(color,vec3f(1.0,0.80,0.03),trialInk);alpha=max(alpha,0.94*trialInk);}
  if(unresolvedInk>0.01){color=mix(color,vec3f(0.88,0.12,0.70),unresolvedInk);alpha=max(alpha,0.96*unresolvedInk);}
  if(control[0]!=0u&&first==INVALID){color=mix(color,vec3f(1.0,0.02,0.06),0.35);}
  if(firstMatch){color=vec3f(1.0,0.01,0.05);alpha=0.98;}
  return BandSample(color,alpha);
}
fn volumeBand(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}
  let steps=traversalSteps(ray,interval);let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;
  for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let fine=worldToFine(point);let row=containingRow(fine);if(row==previous){continue;}previous=row;if(row==INVALID){continue;}let sample=rowSample(row,fine,true);accum=composite(accum,sample.color,sample.alpha*volumeOpacity());}
  return finishVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  if(i32(round(u.debug.x))==4){return volumeBand(input.uv);}let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}let fine=worldToFine(hit.xyz);let row=containingRow(fine);if(row==INVALID){discard;}let sample=rowSample(row,fine,false);return vec4f(displayColor(sample.color),sample.alpha);
}`;

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
struct FineParams { brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,hashCapacity:u32,maximumHashProbes:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32 }
struct FineState { color:vec3f,alpha:f32,address:u32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<uniform> fine:FineParams;
@group(0) @binding(2) var<storage,read> pageHash:array<u32>;
@group(0) @binding(3) var<storage,read> metadata:array<u32>;
@group(0) @binding(4) var<storage,read> worklist:array<u32>;
@group(0) @binding(5) var<storage,read> sampleFlags:array<u32>;
@group(0) @binding(6) var<storage,read> topologyControl:array<u32>;
@group(0) @binding(7) var<storage,read> redistanceControl:array<u32>;
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;const KNOWN:u32=4u;const TRIAL:u32=8u;const FRONTIER:u32=32u;
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(fine.hashCapacity-1u);}
fn pageOf(key:u32)->u32 {let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=fine.maximumHashProbes){break;}let slot=(start+probe)&(fine.hashCapacity-1u);let stored=pageHash[slot*2u];if(stored==key){return pageHash[slot*2u+1u];}if(stored==INVALID){return INVALID;}}return INVALID;}
fn fineState(point:vec3f)->FineState {
  let relative=(point-fine.domainOrigin)/max(fine.fineCellWidth,1e-9);if(any(relative<vec3f(0.0))||any(relative>=vec3f(fine.sampleDimensions))){return FineState(vec3f(0.0),0.0,INVALID);}let q=vec3u(floor(relative));let brick=q/max(fine.brickResolution,1u);let key=brick.x+fine.brickDimensions.x*(brick.y+fine.brickDimensions.y*brick.z);
  if(arrayLength(&topologyControl)>0u&&topologyControl[0]!=0u){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}if(arrayLength(&redistanceControl)>4u&&redistanceControl[4]!=0u){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}
  let page=pageOf(key);if(page==INVALID){let desired=arrayLength(&topologyControl)>4u&&topologyControl[4]==0u;return FineState(select(vec3f(0.03,0.10,0.34),vec3f(1.0,0.34,0.04),desired),select(0.045,0.34,desired),key);}
  if(page>=fine.pageCapacity||page*10u+3u>=arrayLength(&metadata)||metadata[page*10u+2u]!=fine.generation||arrayLength(&worklist)<=1u||worklist[1]!=fine.generation){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}
  let local=q-brick*fine.brickResolution;let localIndex=local.x+fine.brickResolution*(local.y+fine.brickResolution*local.z);let address=page*fine.samplesPerBrick+localIndex;if(address>=arrayLength(&sampleFlags)){return FineState(vec3f(1.0,0.01,0.06),0.94,key);}let flags=sampleFlags[address];
  if((flags&VALID)==0u){return FineState(vec3f(0.26,0.08,0.48),0.20,address);}if((flags&(FRONTIER|TRIAL))!=0u){return FineState(vec3f(1.0,0.84,0.04),0.88,address);}if((flags&INTERFACE)!=0u){return FineState(vec3f(1.0,0.02,0.44),0.90,address);}
  let activated=arrayLength(&redistanceControl)>6u&&redistanceControl[6]>0u&&(flags&KNOWN)==0u;if(activated){return FineState(vec3f(1.0,0.48,0.04),0.66,address);}if((flags&KNOWN)!=0u){return FineState(vec3f(0.04,0.72,0.82),0.34,address);}return FineState(vec3f(0.34,0.12,0.62),0.24,address);
}
fn fineVolume(uv:vec2f)->vec4f {
  let ray=cameraRay(uv);let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;let interval=boxInterval(ray,minimum,maximum);if(interval.y<=interval.x){discard;}let travel=abs(ray.direction*(interval.y-interval.x))/max(fine.fineCellWidth,1e-9);let steps=u32(clamp(ceil(travel.x+travel.y+travel.z+1.0),1.0,512.0));let dt=(interval.y-interval.x)/f32(steps);var accum=vec4f(0.0);var previous=INVALID;
  for(var i=0u;i<512u;i+=1u){if(i>=steps||accum.a>0.985){break;}let point=ray.origin+ray.direction*(interval.x+(f32(i)+0.5)*dt);let sample=fineState(point);if(sample.address==previous){continue;}previous=sample.address;let alpha=select(sample.alpha*0.20,sample.alpha,sample.alpha>0.30);accum=composite(accum,sample.color,alpha*volumeOpacity());}
  return finishVolume(accum);
}
@fragment fn fragmentMain(input:VertexOutput)->@location(0) vec4f {
  if(i32(round(u.debug.x))==4){return fineVolume(input.uv);}let hit=sliceRay(input.uv);if(hit.w<=0.0){discard;}let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);let maximum=minimum+u.container.xyz;if(any(hit.xyz<minimum)||any(hit.xyz>maximum)){discard;}let sample=fineState(hit.xyz);if(sample.alpha<=0.001){discard;}return vec4f(displayColor(sample.color),sample.alpha);
}`;

export class OctreeTechniqueOverlayPipeline {
  private topologyPipeline?: GPURenderPipeline;
  private facePipeline?: GPURenderPipeline;
  private lifecyclePipeline?: GPURenderPipeline;
  private fineLifecyclePipeline?: GPURenderPipeline;
  private section5FaceBandPipeline?: GPURenderPipeline;
  private lifecycleMembershipPipeline?: GPUComputePipeline;
  private source?: OctreeTechniqueDebugSource;
  private ownerRows?: GPUTexture;
  private topologyGroup?: GPUBindGroup;
  private faceGroup?: GPUBindGroup;
  private lifecycleGroup?: GPUBindGroup;
  private fineLifecycleGroup?: GPUBindGroup;
  private section5FaceBandGroup?: GPUBindGroup;
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
    [this.topologyPipeline,this.facePipeline,this.lifecyclePipeline,this.fineLifecyclePipeline,this.section5FaceBandPipeline,this.lifecycleMembershipPipeline]=await Promise.all([
      this.pipeline("Octree topology technique overlay",octreeTechniqueTopologyShader),
      this.pipeline("Octree power-face technique overlay",octreeTechniqueFaceShader),
      this.pipeline("Octree topology-lifecycle overlay",octreeTechniqueLifecycleShader),
      this.pipeline("Octree fine-band lifecycle overlay",octreeTechniqueFineLifecycleShader),
      this.pipeline("Octree Section 5 face-band overlay",octreeTechniqueSection5FaceBandShader),
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
    this.topologyGroup=undefined;this.faceGroup=undefined;this.lifecycleGroup=undefined;this.fineLifecycleGroup=undefined;this.section5FaceBandGroup=undefined;this.lifecycleMembershipGroup=undefined;
    const source=this.source;if(!source)return;const ownerRows=this.ownerRows;
    if(ownerRows&&this.topologyPipeline)this.topologyGroup=this.device.createBindGroup({layout:this.topologyPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:ownerRows.createView({dimension:"3d"})},
      {binding:2,resource:source.leaves},{binding:3,resource:source.topologyMetrics},{binding:4,resource:source.tetrahedronHeaders},
      {binding:5,resource:source.tetrahedra},{binding:6,resource:source.tetrahedronVertices},
    ]});
    if(ownerRows&&this.facePipeline)this.faceGroup=this.device.createBindGroup({layout:this.facePipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:ownerRows.createView({dimension:"3d"})},
      {binding:2,resource:source.leaves},{binding:3,resource:source.powerFaces},{binding:4,resource:source.faceNormals},
      {binding:5,resource:source.faceCentroids},{binding:6,resource:source.incidenceRows},{binding:7,resource:source.incidence},
      {binding:8,resource:source.faceControl},
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
    if(fine&&this.fineLifecyclePipeline)this.fineLifecycleGroup=this.device.createBindGroup({layout:this.fineLifecyclePipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:fine.params},{binding:2,resource:fine.hash},
      {binding:3,resource:fine.metadata},{binding:4,resource:fine.worklist},{binding:5,resource:fine.sampleFlags},
      {binding:6,resource:fine.topologyControl},{binding:7,resource:fine.redistanceControl},
    ]});
    const section5=source.section5FaceBand;
    if(section5&&this.section5FaceBandPipeline)this.section5FaceBandGroup=this.device.createBindGroup({layout:this.section5FaceBandPipeline.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:this.uniformBuffer}},{binding:1,resource:section5.rowHash},{binding:2,resource:section5.rows},
      {binding:3,resource:section5.faces},{binding:4,resource:section5.incidence},{binding:5,resource:section5.states},
      {binding:6,resource:section5.control},{binding:7,resource:section5.transitionControl},
    ]});
  }

  encode(encoder: GPUCommandEncoder, target: GPUTextureView, modeCode: number): boolean {
    let pipeline:GPURenderPipeline|undefined;let group:GPUBindGroup|undefined;
    if(modeCode===12||modeCode===14||modeCode===15){pipeline=this.topologyPipeline;group=this.topologyGroup;}
    else if(modeCode===13||modeCode===16){pipeline=this.facePipeline;group=this.faceGroup;}
    else if(modeCode===17){pipeline=this.lifecyclePipeline;group=this.lifecycleGroup;if(this.lifecycleMembership&&this.lifecycleMembershipPipeline&&this.lifecycleMembershipGroup){encoder.clearBuffer(this.lifecycleMembership);const compute=encoder.beginComputePass({label:"Expand octree topology lifecycle membership"});compute.setPipeline(this.lifecycleMembershipPipeline);compute.setBindGroup(0,this.lifecycleMembershipGroup);compute.dispatchWorkgroups(Math.ceil(this.lifecycleCapacity/64));compute.end();}else{return false;}}
    else if(modeCode===18){pipeline=this.fineLifecyclePipeline;group=this.fineLifecycleGroup;}
    else if(modeCode===24){pipeline=this.section5FaceBandPipeline;group=this.section5FaceBandGroup;}
    else{return false;}
    if(!pipeline||!group)return false;
    const pass=encoder.beginRenderPass({label:"Octree paper-technique overlay",colorAttachments:[{view:target,loadOp:"load",storeOp:"store"}]});
    pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.draw(3);pass.end();return true;
  }
}
