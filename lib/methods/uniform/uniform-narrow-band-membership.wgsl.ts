/** Shared bounded crossing search. Separable squared-distance minimization
 * finds the exact nearest crossing-cell centre within five cells (11 taps
 * per axis). Reconstruction uses its cell-box guard against stale bulk phi.
 * After the search, redistance reuses bank A for nodal metric distances;
 * particle membership samples that field independently of h/4h ownership. */
export const narrowBandMembershipWGSL=/* wgsl */`
const NB_DEPTH_A:u32=NB_SURFACE_TILES+UM_T.x*UM_T.y*UM_T.z;
const NB_DEPTH_B:u32=NB_DEPTH_A+(UM_D.x+1u)*(UM_D.y+1u)*(UM_D.z+1u);
const NB_NO_SURFACE:u32=0xffffffffu;
fn nbCell(i:u32)->vec3u{return vec3u(i%UM_D.x,(i/UM_D.x)%UM_D.y,i/(UM_D.x*UM_D.y));}
@compute @workgroup_size(64) fn depthSeeds(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<UM_D.x*UM_D.y*UM_D.z;i+=65536u){
  let c=nbCell(i);var low=3.0e38;var high=-3.0e38;
  for(var k=0u;k<8u;k++){let value=umSampleVertex(vec3f(c+umCorner(k,2u)));low=min(low,value);high=max(high,value);}
  // Exact zero belongs to either incident cell; a flat zero plateau alone
  // is not an interface. Domain and solid walls are not free surfaces.
  let crossing=low<=0.0&&high>=0.0&&low<high&&umCellOpen(vec3i(c))>=0.5;
  atomicStore(&bins[NB_DEPTH_A+i],select(NB_NO_SURFACE,i,crossing));
 }
}
fn nbDepthSpread(gid:u32,axis:u32,sourceBank:u32,targetBank:u32){
 for(var i=gid;i<UM_D.x*UM_D.y*UM_D.z;i+=65536u){
  let c=vec3i(nbCell(i));var best=NB_NO_SURFACE;var distance=3.0e38;
  for(var offset=-5;offset<=5;offset++){
   var q=c;q[axis]+=offset;
   if(any(q<vec3i(0))||any(q>=vec3i(UM_D))){continue;}
   let candidate=atomicLoad(&bins[sourceBank+cellIndex(q)]);if(candidate==NB_NO_SURFACE){continue;}
   let delta=vec3f(nbCell(candidate))-vec3f(c);let d=dot(delta,delta);
   if(d<distance){best=candidate;distance=d;}
  }
  atomicStore(&bins[targetBank+i],best);
 }
}
@compute @workgroup_size(64) fn depthSpreadX(@builtin(global_invocation_id) gid:vec3u){nbDepthSpread(gid.x,0u,NB_DEPTH_A,NB_DEPTH_B);}
@compute @workgroup_size(64) fn depthSpreadY(@builtin(global_invocation_id) gid:vec3u){nbDepthSpread(gid.x,1u,NB_DEPTH_B,NB_DEPTH_A);}
@compute @workgroup_size(64) fn depthSpreadZ(@builtin(global_invocation_id) gid:vec3u){nbDepthSpread(gid.x,2u,NB_DEPTH_A,NB_DEPTH_B);}
fn bulkDepth(p:vec3f)->f32{
 let phiValue=bandPhi(p);if(phiValue>=0.0){return phiValue;}
 let c=clamp(vec3i(floor(p)),vec3i(0),vec3i(UM_D)-1);
 let nearest=atomicLoad(&bins[NB_DEPTH_B+cellIndex(c)]);
 if(nearest==NB_NO_SURFACE){return min(phiValue,-8.0);}
 let low=vec3f(nbCell(nearest));let delta=max(max(low-p,p-(low+1.0)),vec3f(0));
 return min(phiValue,-length(delta));
}
// The first search bank becomes a nodal metric field after the final sweep.
// Particle membership does not depend on the resolution of simulation owners.
fn nbVertexIndex(p:vec3u)->u32{return p.x+(UM_D.x+1u)*(p.y+(UM_D.y+1u)*p.z);}
fn particleDepth(p:vec3f)->f32{
 let q=clamp(p,vec3f(0),vec3f(UM_D));let c=min(vec3u(floor(q)),UM_D-1u);let f=q-vec3f(c);
 var value=0.0;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let w=select(1.0-f,f,corner!=vec3u(0));
  value+=w.x*w.y*w.z*bitcast<f32>(atomicLoad(&bins[NB_DEPTH_A+nbVertexIndex(c+corner)]));
 }
 return value;
}
`;
