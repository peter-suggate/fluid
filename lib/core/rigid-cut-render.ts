import type {RigidBodyDescription} from './model';
import {GPU_RIGID_BODY_CAPACITY} from './rigid-capacity';
import {packRigidBodyCutAtlas,RIGID_BODY_CUT_ATLAS_WIDTH,solidBoxCutWGSL} from './solid-cuts';

/** Stable renderer derivative of the same body-local cuts used by native
 * collision and shared picking. Updating a roster never changes the texture. */
export class RigidCutRenderAtlas {
 readonly texture:GPUTexture;
 readonly view:GPUTextureView;
 private key='';
 constructor(private readonly device:GPUDevice){
  this.texture=device.createTexture({label:'Shared rigid surface cuts',size:[RIGID_BODY_CUT_ATLAS_WIDTH,1],format:'rgba32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  this.view=this.texture.createView();this.setBodies([]);
 }
 setBodies(bodies:readonly Pick<RigidBodyDescription,'cuts'>[]):boolean{
  const key=JSON.stringify(bodies.map(b=>b.cuts??[]));if(key===this.key)return false;
  const data=packRigidBodyCutAtlas(bodies,GPU_RIGID_BODY_CAPACITY);
  this.device.queue.writeTexture({texture:this.texture},data,{bytesPerRow:RIGID_BODY_CUT_ATLAS_WIDTH*16},[RIGID_BODY_CUT_ATLAS_WIDTH,1]);this.key=key;return true;
 }
 destroy():void{this.texture.destroy();}
}

/** All display paths use tag10 for rounded plates; physical tag5 remains
 * independent of the renderer's pre-existing robot tag5. */
export const ROUNDED_PLATE_RENDER_TAG=10;
export function rigidCutRenderWGSL(group:number,binding:number):string{return /* wgsl */`
@group(${group}) @binding(${binding}) var rigidSurfaceCuts:texture_2d<f32>;
${solidBoxCutWGSL.replaceAll('solidBoxCutDistance','rigidSurfaceBoxCutDistance')}
fn roundedPlateSurfaceDistance(p:vec3f,half:vec3f,bodyIndex:u32)->f32{
 let radius=half.y;let core=vec3f(max(0.0,half.x-radius),0.0,max(0.0,half.z-radius));let q=abs(p)-core;
 var distance=length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0)-radius;
 let header=textureLoad(rigidSurfaceCuts,vec2i(i32(bodyIndex),0),0);
 for(var j=0u;j<u32(header.y);j++){let at=i32(u32(header.x)+3u*j);
  let center=textureLoad(rigidSurfaceCuts,vec2i(at,0),0).xyz;let extent=textureLoad(rigidSurfaceCuts,vec2i(at+1,0),0).xyz;let rotation=textureLoad(rigidSurfaceCuts,vec2i(at+2,0),0);
  distance=max(distance,-rigidSurfaceBoxCutDistance(p,center,extent,rotation));
 }return distance;
}
struct RoundedPlateSurfaceHit{t:f32,normal:vec3f}
fn roundedPlateSurfaceCandidate(origin:vec3f,direction:vec3f,half:vec3f,bodyIndex:u32,t:f32,normal:vec3f,best:RoundedPlateSurfaceHit)->RoundedPlateSurfaceHit{
 let epsilon=max(2e-8,max(half.x,half.z)*2e-6);
 if(t<1e-6||t>=best.t){return best;}
 // The primitive patch contributes only where it survives every local cut.
 if(abs(roundedPlateSurfaceDistance(origin+t*direction,half,bodyIndex))>epsilon){return best;}
 return RoundedPlateSurfaceHit(t,normal);
}
fn roundedPlateSurfaceRotate(q:vec4f,p:vec3f)->vec3f{return p+2.0*cross(q.xyz,cross(q.xyz,p)+q.w*p);}
fn roundedPlateSurfaceHit(origin:vec3f,direction:vec3f,half:vec3f,bodyIndex:u32)->RoundedPlateSurfaceHit{
 var best=RoundedPlateSurfaceHit(1e20,vec3f(0,1,0));
 let radius=half.y;let core=vec3f(max(0.0,half.x-radius),0.0,max(0.0,half.z-radius));let epsilon=max(2e-8,max(half.x,half.z)*2e-6);
 var low=0.0;var high=1e20;
 for(var a=0u;a<3u;a++){
  if(abs(direction[a])<1e-12){if(abs(origin[a])>half[a]){return best;}}
  else{let x=(-half[a]-origin[a])/direction[a];let y=(half[a]-origin[a])/direction[a];low=max(low,min(x,y));high=min(high,max(x,y));}
 }if(high<low){return best;}
 // The Minkowski sum of a rectangle and sphere has two planar patches,
 // four half cylinders and four quarter spheres. Enumerate both roots of
 // every curved patch, since a cut may remove its nearer intersection.
 if(abs(direction.y)>1e-12){for(var side=-1.0;side<=1.0;side+=2.0){
  let t=(side*radius-origin.y)/direction.y;let p=origin+t*direction;
  if(abs(p.x)<=core.x+epsilon&&abs(p.z)<=core.z+epsilon){best=roundedPlateSurfaceCandidate(origin,direction,half,bodyIndex,t,vec3f(0,side,0),best);}
 }}
 for(var axis=0u;axis<=2u;axis+=2u){let radial=2u-axis;
  let a=direction.y*direction.y+direction[radial]*direction[radial];if(a<1e-20){continue;}
  for(var side=-1.0;side<=1.0;side+=2.0){
   let offset=vec2f(origin.y,origin[radial]-side*core[radial]);let d=vec2f(direction.y,direction[radial]);
   let center=-dot(offset,d)/a;let closest=offset+center*d;let radiusTerm=(radius*radius-dot(closest,closest))/a;if(radiusTerm<0.0){continue;}
   let root=sqrt(max(0.0,radiusTerm));
   for(var signRoot=-1.0;signRoot<=1.0;signRoot+=2.0){let t=center+signRoot*root;let p=origin+t*direction;
    if(abs(p[axis])<=core[axis]+epsilon&&side*p[radial]>=core[radial]-epsilon){var n=vec3f(0);n.y=p.y;n[radial]=p[radial]-side*core[radial];best=roundedPlateSurfaceCandidate(origin,direction,half,bodyIndex,t,normalize(n),best);}
   }
  }
 }
 let directionSquared=dot(direction,direction);
 for(var sx=-1.0;sx<=1.0;sx+=2.0){for(var sz=-1.0;sz<=1.0;sz+=2.0){
  let sphereCenter=vec3f(sx*core.x,0,sz*core.z);let offset=origin-sphereCenter;let center=-dot(offset,direction)/directionSquared;let closest=offset+center*direction;let radiusTerm=(radius*radius-dot(closest,closest))/directionSquared;if(radiusTerm<0.0){continue;}
  let root=sqrt(max(0.0,radiusTerm));for(var signRoot=-1.0;signRoot<=1.0;signRoot+=2.0){let t=center+signRoot*root;let p=origin+t*direction;
   if(sx*p.x>=core.x-epsilon&&sz*p.z>=core.z-epsilon){best=roundedPlateSurfaceCandidate(origin,direction,half,bodyIndex,t,normalize(p-sphereCenter),best);}
  }
 }}
 // Subtraction exposes inward-facing box planes. The same finite candidates
 // and CSG classification preserve overlaps between any number of cuts.
 let header=textureLoad(rigidSurfaceCuts,vec2i(i32(bodyIndex),0),0);
 for(var j=0u;j<u32(header.y);j++){let at=i32(u32(header.x)+3u*j);
  let center=textureLoad(rigidSurfaceCuts,vec2i(at,0),0).xyz;let extent=textureLoad(rigidSurfaceCuts,vec2i(at+1,0),0).xyz;let rotation=textureLoad(rigidSurfaceCuts,vec2i(at+2,0),0);let inverse=vec4f(-rotation.xyz,rotation.w);
  let o=roundedPlateSurfaceRotate(inverse,origin-center);let d=roundedPlateSurfaceRotate(inverse,direction);
  for(var axis=0u;axis<3u;axis++){if(abs(d[axis])<1e-12){continue;}
   for(var side=-1.0;side<=1.0;side+=2.0){let t=(side*extent[axis]-o[axis])/d[axis];let p=o+t*d;
    if(all(abs(p)<=extent+vec3f(epsilon))){var n=vec3f(0);n[axis]=-side;best=roundedPlateSurfaceCandidate(origin,direction,half,bodyIndex,t,roundedPlateSurfaceRotate(rotation,n),best);}
   }
  }
 }return best;
}
fn rigidSurfaceCutCount(bodyIndex:u32)->u32{return u32(textureLoad(rigidSurfaceCuts,vec2i(i32(bodyIndex),0),0).y);}
// A sphere is the rounded plate with no core (the rectangle shrinks to its
// centre), so a sphere with local cuts takes the same exact patches and cut
// planes. An uncut sphere keeps its own closed form.
fn cutSphereSurfaceHit(origin:vec3f,direction:vec3f,radius:f32,bodyIndex:u32)->RoundedPlateSurfaceHit{return roundedPlateSurfaceHit(origin,direction,vec3f(radius),bodyIndex);}
`;}
