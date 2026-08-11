/**
 * Camera, slice-plane and volume helpers every technique view shares.
 *
 * These live apart from the overlay pipeline so a view that needs its own
 * pipeline can project and slice identically without importing the pipeline
 * module back, which would close an import cycle.
 */
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "./generated/octree-power-catalog";
import { cameraApertureShaderLibrary } from "./webgpu-camera";

export const octreeTechniqueSharedWGSL = /* wgsl */ `
${cameraApertureShaderLibrary("u")}
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
  let direction=normalize(forward+right*ndc.x*u.viewport.x/max(u.viewport.y,1.0)*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());return CameraRay(origin,direction);
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
// Slice planes sit between the camera and the water they describe, so they are
// uniformly thinned; volume views keep full authored alpha because their
// opacity is already the user's slider.
const SLICE_OPACITY:f32=0.6;
fn sliceDisplay(linearColor:vec3f,alpha:f32)->vec4f{return vec4f(displayColor(linearColor),alpha*SLICE_OPACITY);}
fn compositeDisplay(accum:vec4f,linearColor:vec3f,alpha:f32)->vec4f{return composite(accum,displayColor(linearColor),alpha);}
fn finishDisplayVolume(accum:vec4f)->vec4f{if(accum.a<=0.001){discard;}return vec4f(accum.rgb/max(accum.a,1e-6),accum.a);}
`;
