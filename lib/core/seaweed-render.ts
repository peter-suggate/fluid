import asset from "./imported-assets/beach-seaweed.generated.json";
import type { RigidBodyDescription } from "./model";

export const SEAWEED_RENDER_SHAPE = 9;
export function seaweedSkin(body: RigidBodyDescription) {
  if (body.appearance !== "seaweed") return undefined;
  const half = asset.extent as [number, number, number];
  return { shape: SEAWEED_RENDER_SHAPE, half, radius: Math.hypot(...half) };
}
const vec = (values: number[]) => `vec${values.length}f(${values.map(v => v.toFixed(8)).join(",")})`;
/** The Blender-authored blades and gas bladders, in body-local metres. A
 * conservative early bound keeps unrelated primary/shadow rays inexpensive. */
export const seaweedShaderLibrary = /* wgsl */ `
struct SeaweedPart { center:vec3f, half:vec3f, rotation:vec4f, color:vec3f }
struct SeaweedHit { t:f32, normal:vec3f }
const SEAWEED_ROUGHNESS:f32=${asset.roughness};
const SEAWEED_PARTS=array<SeaweedPart,${asset.parts.length}>(${asset.parts.map(p => `SeaweedPart(${vec(p.p)},${vec(p.d)},${vec(p.q)},${vec(p.color)})`).join(",")});
fn seaweedRotate(q:vec4f,p:vec3f)->vec3f {return p+2.0*cross(q.yzw,cross(q.yzw,p)+q.x*p);}
fn seaweedHit(origin:vec3f,direction:vec3f)->SeaweedHit {
 var best=SeaweedHit(1e20,vec3f(0,1,0));
 let projected=dot(-origin,direction);let closest=origin+direction*max(0.0,projected);
 if(dot(closest,closest)>${Math.hypot(...asset.extent) ** 2}){return best;}
 for(var i=0u;i<${asset.parts.length}u;i++){
  let p=SEAWEED_PARTS[i];let inverse=vec4f(p.rotation.x,-p.rotation.yzw);
  let ro=seaweedRotate(inverse,origin-p.center);let rd=seaweedRotate(inverse,direction);
  let a=ro/p.half;let b=rd/p.half;let aa=dot(b,b);let ab=dot(a,b);let discriminant=ab*ab-aa*(dot(a,a)-1.0);
  if(discriminant<0.0){continue;}let root=sqrt(discriminant);var t=(-ab-root)/aa;if(t<=1e-5){t=(-ab+root)/aa;}
  if(t>1e-5&&t<best.t){best=SeaweedHit(t,seaweedRotate(p.rotation,normalize((ro+rd*t)/(p.half*p.half))));}
 }return best;
}
fn seaweedColor(position:vec3f)->vec3f {
 var best=1e20;var color=vec3f(.09,.105,.018);
 for(var i=0u;i<${asset.parts.length}u;i++){
  let p=SEAWEED_PARTS[i];let local=seaweedRotate(vec4f(p.rotation.x,-p.rotation.yzw),position-p.center);
  let distance=abs(length(local/p.half)-1.0)*min(p.half.x,min(p.half.y,p.half.z));
  if(distance<best){best=distance;color=p.color;}
 }return color;
}
`;
