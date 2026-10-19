/** DEPB, the particle deposit (NB-FLIP transfer): a workgroup an occupied
 * tile. It classifies each sample once (grid support at its new position:
 * the mode bit the move and snapshot read, its motion words for the seam
 * gather, the receipt's census), then deposits the tile's quadratic MAC
 * momentum and weight into workgroup fixed-point accumulators, flushed once
 * to the fine owners' six words (UM_CELL_KEYS + 6 owner: momentum, then
 * weight, an axis). Fixed-point sums are order independent.
 *
 * The flush also flags each fine tile it reached (NB_DEPOSITED, one word a
 * fine tile after the accumulators). transferProperties (FIN) finishes only
 * flagged tiles from their words, copies the rest, and zeroes what it read,
 * so the words are zero again for the next deposit.
 *
 * Window: a node j along axis a is the positive a face of cell j. A sample in
 * a tile's cells [0,4) reaches nodes -2..4 along a and -1..4 across it. */
export const PARTICLE_DEPOSIT_WORDS_PER_CELL = 6;
export function particleBrickDepositWGSL(apic: boolean): string {
  return /* wgsl */`
const PB_ALONG:i32=7;const PB_ACROSS:i32=6;const PB_WINDOW:u32=252u;const PB_NODES:u32=756u;const PB_LANES:u32=192u;
// Momentum in 1/8192 of the velocity unit, weight in 1/65536: an i32 node sum
// holds weight 32768 and momentum 262144 (velocity times weight).
const PB_MOMENTUM:f32=8192.0;const PB_WEIGHT:f32=65536.0;
var<workgroup> pbMomentum:array<atomic<i32>,756>;
var<workgroup> pbWeight:array<atomic<u32>,756>;
var<workgroup> pbJobs:u32;
var<workgroup> pbTile:u32;
// The fine tiles the flush reached (the 3³ around the tile): their index + 1.
var<workgroup> pbTouched:array<atomic<u32>,27>;
// Brick context, staged once a tile: the 3³ tiles' topology words (0 where
// not a stored fine tile) and the liquid owners of the tile's cells -1..4
// (gridSupported's eight corners of every sample in it).
var<workgroup> pbWords:array<u32,27>;
var<workgroup> pbLiquid:array<u32,216>;
fn nbDeposited()->u32{return UM_CELL_KEYS+384u*umCounts.x;}
fn pbExtent(axis:u32)->vec3i{var d=vec3i(PB_ACROSS);d[axis]=PB_ALONG;return d;}
fn pbLow(axis:u32)->vec3i{var d=vec3i(-1);d[axis]=-2;return d;}
fn pbSlot(axis:u32,node:vec3i)->u32{let d=pbExtent(axis);let r=node-pbLow(axis);return axis*PB_WINDOW+u32((r.z*d.y+r.y)*d.x+r.x);}
fn pbNode(slot:u32)->vec3i{let axis=slot/PB_WINDOW;let r=i32(slot%PB_WINDOW);let d=pbExtent(axis);return vec3i(r%d.x,(r/d.x)%d.y,r/(d.x*d.y))+pbLow(axis);}
fn pbQuadratic(f:f32)->vec3f{return vec3f(0.5*(1.5-f)*(1.5-f),0.75-(f-1.0)*(f-1.0),0.5*(f-0.5)*(f-0.5));}
fn pbAdd(slot:u32,w:f32,m:f32){atomicAdd(&pbMomentum[slot],i32(round(m*PB_MOMENTUM)));atomicAdd(&pbWeight[slot],u32(round(w*PB_WEIGHT)));}
// A sample's three MAC components. p is relative to the tile origin.
fn pbScatter(p:vec3f,u:vec3f${apic?",c:array<vec3f,3>":""}){
 for(var axis=0u;axis<3u;axis++){
  var offset=vec3f(0.5);offset[axis]=1.0;
  let base=vec3i(floor(p-offset-0.5));let f=p-offset-vec3f(base);
  let wx=pbQuadratic(f.x);let wy=pbQuadratic(f.y);let wz=pbQuadratic(f.z);
  for(var z=0;z<3;z++){for(var y=0;y<3;y++){let wyz=wy[y]*wz[z];for(var x=0;x<3;x++){
   let node=base+vec3i(x,y,z);let w=wx[x]*wyz;
   pbAdd(pbSlot(axis,node),w,w*${apic?"(u[axis]+dot(c[axis],vec3f(node)+offset-p))":"u[axis]"});
  }}}
 }
}
// Fixed direct groups stride the occupied tile list's current GPU count.
@compute @workgroup_size(192) fn transferDeposit(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 // Uniforms: both are uniform control flow.
 if(!nbParticlesLive()){return;}
 let fits=nbTransferFits();
 var outside=0.0;var counts=vec3u(0);let census=params.settings.w!=0.0;
 if(lane==0u){pbJobs=min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES);}
 let count=workgroupUniformLoad(&pbJobs);
 for(var job=group.x;job<count;job+=groups.x){
  workgroupBarrier();
  for(var slot=lane;slot<PB_NODES;slot+=PB_LANES){atomicStore(&pbMomentum[slot],0);atomicStore(&pbWeight[slot],0u);}
  if(lane<27u){atomicStore(&pbTouched[lane],0u);}
  if(lane==0u){pbTile=atomicLoad(&bins[NB_OCCUPIED+4u+job]);}
  let tile=workgroupUniformLoad(&pbTile);
  if(!umTileLive(tile)){continue;}
  let origin=vec3i(4u*umTileCoord(tile));
  if(lane<27u){
   let t=(origin>>vec3u(2u))+vec3i(i32(lane%3u),i32((lane/3u)%3u),i32(lane/9u))-1;let key=umTileKey(t);
   var word=0u;if(umKeyStored(key)){word=umTopology[key];if((word&0x80000000u)==0u){word=0u;}}pbWords[lane]=word;
  }
  // A tile lies in one page: its samples' page, the support stencil's.
  let page=origin>>vec3u(5u);
  for(var slot=lane;slot<216u;slot+=PB_LANES){
   let c=origin+vec3i(i32(slot%6u),i32((slot/6u)%6u),i32(slot/36u))-1;
   pbLiquid[slot]=u32(gridLiquidCell(nbHomeBlock(page,c>>vec3u(5u),c>>vec3u(5u)),page,c));
  }
  workgroupBarrier();
  // The sort keeps a tile's cells contiguous, in local cell order.
  let start=links[64u*tile];let end=links[64u*tile+63u]+atomicLoad(&bins[64u*tile+63u]);
  // Classify: support at the new position. The mode bit rides the
  // record's before.w and the compact flag word; motion feeds the seams.
  for(var i=start+lane;i<end;i+=PB_LANES){
   let s=particles[i];
   // gridSupported from the staged cells: a corner of zero weight is skipped.
   let q=s.position.xyz;let fraction=fract(q-0.5);let base=clamp(vec3i(floor(q-0.5))-origin+1,vec3i(0),vec3i(4));
   var supported=false;
   for(var k=0u;k<8u;k++){
    let bit=umCorner(k,2u);let w=select(vec3f(1)-fraction,fraction,bit!=vec3u(0));let r=base+vec3i(bit);
    supported=supported||(w.x*w.y*w.z>1e-6&&pbLiquid[u32(r.x+6*r.y+36*r.z)]!=0u);
   }
   counts.y+=u32(!supported);
   if(census){let d=bandPhi(q);outside=max(outside,d);counts+=vec3u(u32(d>0.5),0u,u32(particleDepth(q)<-4.0));}
   particles[i].before.w=nbParticleState(nbParticleAge(s),select(1.0,0.0,supported));
   let a=UM_CELL_KEYS+4u*arrayLength(&particles)+3u*i;let v=bitcast<vec3u>(s.velocity.xyz);links[a]=v.x;links[a+1u]=v.y;links[a+2u]=v.z;
   links[UM_CELL_KEYS+4u*i+3u]=(bitcast<u32>(s.position.w)&0x7fffffffu)|select(0x80000000u,0u,supported);
   // Sorted samples lie in their tile; the clamp only bounds the window.
   if(fits&&supported){pbScatter(clamp(q-vec3f(origin),vec3f(0.0),vec3f(3.9999998)),s.velocity.xyz${apic?",array<vec3f,3>(s.cx.xyz,s.cy.xyz,s.cz.xyz)":""});}
  }
  if(!fits){continue;}
  workgroupBarrier();
  for(var slot=lane;slot<PB_NODES;slot+=PB_LANES){
   let w=atomicLoad(&pbWeight[slot]);if(w==0u){continue;}
   let cell=origin+pbNode(slot);let t=vec3u((cell>>vec3u(2u))-(origin>>vec3u(2u))+1);let n=t.x+3u*t.y+9u*t.z;
   let word=pbWords[n];if(word==0u){continue;}
   let local=vec3u(cell&vec3i(3));let base=word&0x3fffffffu;let owner=base+local.x+4u*local.y+16u*local.z;
   let at=UM_CELL_KEYS+6u*owner+2u*(slot/PB_WINDOW);
   atomicAdd(&bins[at],bitcast<u32>(atomicLoad(&pbMomentum[slot])));atomicAdd(&bins[at+1u],w);
   atomicStore(&pbTouched[n],1u);
  }
  workgroupBarrier();
  if(lane<27u&&atomicLoad(&pbTouched[lane])!=0u){atomicStore(&bins[nbDeposited()+(pbWords[lane]&0x3fffffffu)/64u],1u);}
 }
 atomicMax(&state[5],bitcast<u32>(outside));
 for(var k=0u;k<3u;k++){if(counts[k]!=0u){atomicAdd(&state[6u+k],counts[k]);}}
}
`;
}
