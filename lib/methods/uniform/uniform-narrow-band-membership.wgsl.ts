/** A bounded geometric guard for particle membership. Advected phi is only
 * redistanced near zero; its magnitude cannot certify distance in the bulk.
 * Find the nearest crossing-cell centre in a bounded six-cell box. Separable
 * squared-distance minimization needs 13 taps per axis, rather than three
 * 27-tap jump-flood passes, and finds the exact nearest centre in that box.
 * Distance to its unit-cell box guards stale interior phi, with cell-scale
 * slack at the boundary. The ordinary redistanced phi sets the near band.
 * The ping-pong banks share the particle-bin allocation, outside its counters. */
export const narrowBandMembershipWGSL=/* wgsl */`
const NB_DEPTH_A:u32=NB_SURFACE_TILES+UM_T.x*UM_T.y*UM_T.z;
const NB_DEPTH_B:u32=NB_DEPTH_A+UM_D.x*UM_D.y*UM_D.z;
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
  for(var offset=-6;offset<=6;offset++){
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
fn particleDepth(p:vec3f)->f32{
 let phiValue=bandPhi(p);if(phiValue>=0.0){return phiValue;}
 let c=clamp(vec3i(floor(p)),vec3i(0),vec3i(UM_D)-1);
 let nearest=atomicLoad(&bins[NB_DEPTH_B+cellIndex(c)]);
 if(nearest==NB_NO_SURFACE){return min(phiValue,-8.0);}
 let low=vec3f(nbCell(nearest));let delta=max(max(low-p,p-(low+1.0)),vec3f(0));
 return min(phiValue,-length(delta));
}
`;
