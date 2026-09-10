import { boundedQefWGSL } from "./bounded-qef";
/** GPU Hermite DC producer. Host only supplies arena offsets and dispatches.
 * dcField is the construction provider's shared signed field, not voxel coverage.
 */
export const svoDualContouringFitWGSL = /* wgsl */ `
struct DcFit { point:vec3f, signs:u32, normal:vec3f, sharp:u32 }
fn dcCorner(i:u32)->vec3f{return vec3f(f32(i&1u),f32((i>>1u)&1u),f32((i>>2u)&1u));}
fn dcGradient(p:vec3f,cell:vec3f,dirty:u32,count:u32)->vec3f{
  let e=max(cell*0.001,vec3f(1e-6));
  let g=vec3f(dcField(p+vec3f(e.x,0,0),dirty,count)-dcField(p-vec3f(e.x,0,0),dirty,count),
    dcField(p+vec3f(0,e.y,0),dirty,count)-dcField(p-vec3f(0,e.y,0),dirty,count),
    dcField(p+vec3f(0,0,e.z),dirty,count)-dcField(p-vec3f(0,0,e.z),dirty,count))/e;
  if(dot(g,g)<1e-20){return vec3f(0,1,0);}return normalize(g);
}
${boundedQefWGSL}
fn dcFit(base:vec3f,cell:vec3f,dirty:u32,count:u32)->DcFit{
  var values:array<f32,8>;var signs=0u;
  for(var i=0u;i<8u;i+=1u){values[i]=dcField(base+dcCorner(i)*cell,dirty,count);if(values[i]<0.){signs|=1u<<i;}}
  if(signs==0u||signs==255u){return DcFit(vec3f(.5),signs,vec3f(0,1,0),0u);}
  var A=mat3x3f(vec3f(0),vec3f(0),vec3f(0));var B=vec3f(0);var mass=vec3f(0);var normal=vec3f(0);var hits=0.;
  var normals:array<vec3f,12>;
  for(var axis=0u;axis<3u;axis+=1u){for(var c=0u;c<8u;c+=1u){
    if((c&(1u<<axis))!=0u){continue;}let other=c|(1u<<axis);
    if((values[c]<0.)==(values[other]<0.)){continue;}
    var lo=dcCorner(c);var hi=dcCorner(other);let inside=values[c]<0.;
    for(var step=0u;step<12u;step+=1u){let mid=.5*(lo+hi);if((dcField(base+mid*cell,dirty,count)<0.)==inside){lo=mid;}else{hi=mid;}}
    let point=.5*(lo+hi);let worldNormal=dcGradient(base+point*cell,cell,dirty,count);
    let n=normalize(worldNormal*cell);let d=dot(n,point);
    A+=mat3x3f(n*n.x,n*n.y,n*n.z);B+=n*d;mass+=point;normal+=worldNormal;normals[u32(hits)]=worldNormal;hits+=1.;
  }}
  var sharp=0u;for(var i=0u;i<u32(hits);i+=1u){for(var j=0u;j<i;j+=1u){if(dot(normals[i],normals[j])<0.75){sharp=1u;}}}
  let point=dcSolve(A,B,mass/max(hits,1.));
  // Global lattice quantization lets separately emitted neighbouring faces
  // agree exactly even when packed relative to different owner cells.
  return DcFit(round(point*256.)/256.,signs,dcGradient(base+point*cell,cell,dirty,count),sharp);
}
`;
