/** Tiled particle deposit (NB-FLIP particleDeposit "tiled"): each occupied
 * tile's samples scatter their quadratic MAC momentum and weight once, into
 * workgroup fixed-point accumulators. One flush adds each touched node to its
 * fine owner's six accumulator words (UM_CELL_KEYS + 6 owner: momentum, then
 * weight, an axis). Fixed-point sums are order independent, so the result
 * does not depend on scheduling. transferProperties then finishes each fine
 * cell from its own words. Classify zeroes them first.
 *
 * Window: a node j along axis a is the positive a face of cell j. A sample in
 * a tile's cells [0,4) reaches nodes -2..4 along a and -1..4 across it. */
export const PARTICLE_DEPOSIT_WORDS_PER_CELL = 6;
export function particleBrickDepositWGSL(apic: boolean): string {
  return /* wgsl */`
const PB_ALONG:i32=7;const PB_ACROSS:i32=6;const PB_WINDOW:u32=252u;const PB_NODES:u32=756u;const PB_LANES:u32=128u;
// Momentum in 1/8192 of the velocity unit, weight in 1/65536: an i32 node sum
// holds weight 32768 and momentum 262144 (velocity times weight).
const PB_MOMENTUM:f32=8192.0;const PB_WEIGHT:f32=65536.0;
var<workgroup> pbMomentum:array<atomic<i32>,756>;
var<workgroup> pbWeight:array<atomic<u32>,756>;
var<workgroup> pbJobs:u32;
var<workgroup> pbTile:u32;
fn pbExtent(axis:u32)->vec3i{var d=vec3i(PB_ACROSS);d[axis]=PB_ALONG;return d;}
fn pbLow(axis:u32)->vec3i{var d=vec3i(-1);d[axis]=-2;return d;}
fn pbSlot(axis:u32,node:vec3i)->u32{let d=pbExtent(axis);let r=node-pbLow(axis);return axis*PB_WINDOW+u32((r.z*d.y+r.y)*d.x+r.x);}
fn pbNode(slot:u32)->vec3i{let axis=slot/PB_WINDOW;let r=i32(slot%PB_WINDOW);let d=pbExtent(axis);return vec3i(r%d.x,(r/d.x)%d.y,r/(d.x*d.y))+pbLow(axis);}
fn pbQuadratic(f:f32)->vec3f{return vec3f(0.5*(1.5-f)*(1.5-f),0.75-(f-1.0)*(f-1.0),0.5*(f-0.5)*(f-0.5));}
// A sample's three MAC components. p is relative to the tile origin.
fn pbScatter(p:vec3f,u:vec3f${apic ? ",c:array<vec3f,3>" : ""}){
 for(var axis=0u;axis<3u;axis++){
  var offset=vec3f(0.5);offset[axis]=1.0;
  let base=vec3i(floor(p-offset-0.5));let f=p-offset-vec3f(base);
  let wx=pbQuadratic(f.x);let wy=pbQuadratic(f.y);let wz=pbQuadratic(f.z);
  for(var z=0;z<3;z++){for(var y=0;y<3;y++){let wyz=wy[y]*wz[z];for(var x=0;x<3;x++){
   let node=base+vec3i(x,y,z);let w=wx[x]*wyz;
   ${apic ? "let v=u[axis]+dot(c[axis],vec3f(node)+offset-p);" : "let v=u[axis];"}
   let slot=pbSlot(axis,node);
   atomicAdd(&pbMomentum[slot],i32(round(w*v*PB_MOMENTUM)));atomicAdd(&pbWeight[slot],u32(round(w*PB_WEIGHT)));
  }}}
 }
}
// Fixed direct groups stride the occupied tile list's current GPU count.
@compute @workgroup_size(128) fn transferDeposit(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 // umCounts is a uniform: the fit is uniform control flow.
 if(!nbTransferFits()||!nbParticlesLive()){return;}
 if(lane==0u){pbJobs=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);}
 let count=workgroupUniformLoad(&pbJobs);
 for(var job=group.x;job<count;job+=groups.x){
  workgroupBarrier();
  for(var slot=lane;slot<PB_NODES;slot+=PB_LANES){atomicStore(&pbMomentum[slot],0);atomicStore(&pbWeight[slot],0u);}
  if(lane==0u){pbTile=atomicLoad(&bins[NB_OCCUPIED+4u+job]);}
  let tile=workgroupUniformLoad(&pbTile);
  if(!umTileLive(tile)){continue;}
  let origin=vec3i(4u*umTileCoord(tile));
  // The sort keeps a tile's cells contiguous, in local cell order.
  let start=links[64u*tile];let end=links[64u*tile+63u]+atomicLoad(&bins[64u*tile+63u]);
  for(var i=start+lane;i<end;i+=PB_LANES){
   let motion=nbMotion(i);if(motion.w==1.0){continue;}
   // Sorted samples lie in their tile; the clamp only bounds the window.
   let p=clamp(nbPosition(i)-vec3f(origin),vec3f(0.0),vec3f(3.9999998));
   ${apic ? "let s=particles[i];pbScatter(p,motion.xyz,array<vec3f,3>(s.cx.xyz,s.cy.xyz,s.cz.xyz));" : "pbScatter(p,motion.xyz);"}
  }
  workgroupBarrier();
  for(var slot=lane;slot<PB_NODES;slot+=PB_LANES){
   let w=atomicLoad(&pbWeight[slot]);if(w==0u){continue;}
   let cell=origin+pbNode(slot);let key=umTileKey(cell>>vec3u(2u));if(!umKeyStored(key)){continue;}
   let word=umTopology[key];if((word&0x80000000u)==0u){continue;}
   let local=vec3u(cell&vec3i(3));let owner=(word&0x3fffffffu)+local.x+4u*local.y+16u*local.z;
   let at=UM_CELL_KEYS+6u*owner+2u*(slot/PB_WINDOW);
   atomicAdd(&bins[at],bitcast<u32>(atomicLoad(&pbMomentum[slot])));atomicAdd(&bins[at+1u],w);
  }
 }
}
`;
}
