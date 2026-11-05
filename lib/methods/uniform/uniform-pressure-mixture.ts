/** Sparse face material ABI for the production composite pressure solve.
 * Header vec4u: count, reserved. Records sorted z,y,x:
 * vec4i anchor; vec4f dimensionless mobility; vec4f mixture forced velocity;
 * vec4f fluid forced velocity. Absent faces are pure water (mobility 1).
 * Every nontrivial face must have h pressure rows on BOTH sides. */
export const UNIFORM_MIXTURE_HEADER_BYTES=16;
export const UNIFORM_MIXTURE_FACE_BYTES=64;
export function uniformPressureMixtureWGSL(group:number,binding:number):string{return /* wgsl */`
struct UMMixtureFace {p:vec4i,mobility:vec4f,mixture:vec4f,fluid:vec4f};
struct UMMixtureFaces {header:vec4u,faces:array<UMMixtureFace>};
@group(${group}) @binding(${binding}) var umMixture:texture_2d<f32>;
fn umMixtureWord(i:u32)->vec4f{let width=textureDimensions(umMixture).x;return textureLoad(umMixture,vec2i(i32(i%width),i32(i/width)),0);}
fn umMixtureFace(i:u32)->UMMixtureFace{return UMMixtureFace(bitcast<vec4i>(umMixtureWord(1u+4u*i)),umMixtureWord(2u+4u*i),umMixtureWord(3u+4u*i),umMixtureWord(4u+4u*i));}
fn umMixtureLess(a:vec3i,b:vec3i)->bool{return a.z<b.z||(a.z==b.z&&(a.y<b.y||(a.y==b.y&&a.x<b.x)));}
fn umMixtureFind(p:vec3i)->u32{
 let count=bitcast<u32>(umMixtureWord(0u).x);var low=0u;var high=count;
 loop{if(low>=high){break;}let mid=(low+high)/2u;if(umMixtureLess(bitcast<vec4i>(umMixtureWord(1u+4u*mid)).xyz,p)){low=mid+1u;}else{high=mid;}}
 if(low<count&&all(bitcast<vec4i>(umMixtureWord(1u+4u*low)).xyz==p)){return low;}return 0xffffffffu;
}
fn umMixtureMaterial(p:vec3i,axis:u32,fluid:f32)->vec2f{
 let i=umMixtureFind(p);if(i==0xffffffffu){return vec2f(1.0,fluid);}
 let f=umMixtureFace(i);return vec2f(f.mobility[axis],f.mixture[axis]);
}
fn umMixtureFluid(p:vec3i,axis:u32,projected:f32)->f32{
 let i=umMixtureFind(p);if(i==0xffffffffu){return projected;}
 let f=umMixtureFace(i);return f.fluid[axis]+(projected-f.mixture[axis])/max(f.mobility[axis],1e-30);
}
`;}

/** Local eliminated pressure operator, used by CPU oracles. Volumes are
 * fractions of one MAC dual cell; solid masses are split with the SAME
 * interpolation weights. Fixed supports have zero inverse mass. */
export function mixturePressureFace(fluidVelocity:number, density:number, cellVolume:number,
 solids:readonly {fraction:number;mass:number;velocity:number;fixed?:boolean}[]){
 const fraction=solids.reduce((s,x)=>s+x.fraction,0);
 if(!(density>0&&cellVolume>0)||fraction<0||fraction>=1)throw new Error('Invalid mixture face occupancy');
 let velocity=(1-fraction)*fluidVelocity,mobility=1-fraction;
 for(const s of solids){
  if(s.fraction<0||(!s.fixed&&!(s.mass>0)))throw new Error('Invalid mixture constituent');
  velocity+=s.fraction*s.velocity;
  if(!s.fixed)mobility+=density*cellVolume*s.fraction*s.fraction/s.mass;
 }
 return {velocity,mobility,freeFraction:1-fraction};
}
