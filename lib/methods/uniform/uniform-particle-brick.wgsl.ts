import { particleResidencyTravelWGSL } from "./uniform-local-residency";
import { NARROW_BAND_SURFACE_RADIUS } from "./uniform-narrow-band-surface.wgsl";
import { UNIFORM_PAGE_WINDOW } from "./uniform-lattice-address";
/** DEPB, the particle deposit (NB-FLIP transfer): a workgroup an occupied
 * tile. It classifies each sample once (grid support at its new position:
 * the mode bit the move and snapshot read, its motion words for the seam
 * gather, the receipt's census), then deposits the tile's quadratic MAC
 * momentum and weight into workgroup fixed-point accumulators, flushed once
 * to the fine owners' six words (NB_DEPOSIT + 6 owner: momentum, then
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
/** The refinement prediction transferRefine writes after DEPB, for the next census (refine):
 * a join bit a tile key (joins words), then PARTICLE_RESIDENCY_WORDS a page
 * (allocation support masks and travel bounds), at these bins words. */
export interface ParticleRefineWords {readonly joins:number;readonly allocation:number}
export function particleBrickDepositWGSL(apic: boolean, refine: ParticleRefineWords): string {
  return /* wgsl */`
${particleResidencyTravelWGSL}
// Refinement prediction (the census's particle inputs), from the tile's
// sorted samples after classify: what the census reads is exactly these
// records (the move's before, DEPB's mode, the band's nearest crossing).
// A tile's samples join 5³ tiles around it and support the 2³ pages
// (residency regions) about it: workgroup words, flushed once a tile.
const NB_REFINE_JOINS:u32=${refine.joins}u;const NB_REFINE_ALLOCATION:u32=${refine.allocation}u;
const NB_REFINE_WINDOW:i32=${8 * UNIFORM_PAGE_WINDOW - 1};
// A tile's samples reach 5³ join tiles about it (a 25-bit x+5y slice a z)
// and, by the half of the tile they sit in along each axis, one of 8 support
// blocks of 2³ tiles (pbHalves bits, their travel maxima in pbTravel).
var<workgroup> pbJoin:array<atomic<u32>,5>;
var<workgroup> pbHalves:atomic<u32>;
var<workgroup> pbTravel:array<atomic<u32>,48>;
// The tiles a sample's surface reconstruction can request, swept over its
// travel: interior samples ask their crossing cell, not their collar. reach:
// the home tile's reach bit (the nearest-crossing bank holds a tile only
// within two tiles of a crossing).
struct NbRefineBox{low:vec3i,high:vec3i}
fn nbRefineBox(s:Particle,ballistic:bool,home:u32,reach:bool)->NbRefineBox{
 let start=s.position.xyz;let motion=select(s.before.xyz,s.velocity.xyz,ballistic);
 let travel=params.hDt.w*motion/params.hDt.xyz;let c=vec3u(start);
 var low=start-${NARROW_BAND_SURFACE_RADIUS};var high=start+${NARROW_BAND_SURFACE_RADIUS};
 if(reach&&s.velocity.w<=0.0){
  let nearest=atomicLoad(&bins[NB_DEPTH_B+64u*home+((c.x&3u)|((c.y&3u)<<2u)|((c.z&3u)<<4u))]);
  if(nearest!=0xffffffffu){low=vec3f(umCellCoord(nearest));high=low+1.0;}
 }
 // No wall clamps the box: a tile that holds no storage has no key and takes no mark.
 let padding=params.activity.z;
 let lo=clamp(vec3i(floor((low+min(travel,vec3f(0))-padding)/4.0)),vec3i(0),vec3i(NB_REFINE_WINDOW));
 let hi=clamp(vec3i(floor((high+max(travel,vec3f(0))+padding-1e-5)/4.0)),vec3i(0),vec3i(NB_REFINE_WINDOW));
 return NbRefineBox(lo,max(lo,hi));
}
fn nbRefineJoin(t:vec3i){
 let key=umTileKey(t);if(!umKeyStored(key)){return;}
 let bit=1u<<(key%32u);let at=NB_REFINE_JOINS+key/32u;
 if((atomicLoad(&bins[at])&bit)==0u){atomicOr(&bins[at],bit);}
}
// A residency page's word: masks OR, travel bounds (positive) take the max.
// The words only grow until the census copies them: a stale read costs one
// redundant atomic at most.
fn nbRefinePage(region:vec3i,k:u32,value:u32){
 let tile=umTileKey(4*region);if(!umKeyStored(tile)){return;}
 let octant=vec3u(region)&vec3u(1u);let at=NB_REFINE_ALLOCATION+8u*(8u*(tile>>9u)+octant.x+2u*octant.y+4u*octant.z)+k;
 if(k<2u){if((atomicLoad(&bins[at])&value)!=value){atomicOr(&bins[at],value);}}
 else if(atomicLoad(&bins[at])<value){atomicMax(&bins[at],value);}
}
// Support block low tiles .. low+1 on the page at region: its mask word k
// (k<2: 4x4x4 tile bits x+4y+16z), or 0 when the block misses the page.
fn nbRefineMask(low:vec3i,region:vec3i,k:u32)->u32{
 let origin=4*region;let a=max(low-origin,vec3i(0));let b=min(low+1-origin,vec3i(3));
 if(any(a>b)){return 0u;}
 let row=((2u<<u32(b.x-a.x))-1u)<<u32(a.x);var mask=0u;
 for(var z=max(a.z,2*i32(k));z<=min(b.z,2*i32(k)+1);z++){for(var y=a.y;y<=b.y;y++){mask|=row<<u32(4*y+16*(z&1));}}
 return mask;
}
fn nbRefineTravel(travel:array<vec3f,2>,k:u32)->u32{
 let v=select(travel[1][(k-5u)%3u],travel[0][(k-2u)%3u],k<5u);
 return select(0u,bitcast<u32>(v),v>0.0);
}
// Samples outside the tiles' runs (the step's seeds, the spill reserve):
// a support block's pages (low tiles .. low+1, its travel maxima) and a
// swept box's joins straight to the bins.
fn nbRefinePages(low:vec3i,travel:array<vec3f,2>){
 let first=low>>vec3u(2u);let last=(low+1)>>vec3u(2u);
 for(var z=first.z;z<=last.z;z++){for(var y=first.y;y<=last.y;y++){for(var x=first.x;x<=last.x;x++){
  let region=vec3i(x,y,z);
  for(var k=0u;k<8u;k++){
   let value=select(nbRefineTravel(travel,k),nbRefineMask(low,region,k),k<2u);
   if(value!=0u){nbRefinePage(region,k,value);}
  }
 }}}
}
fn nbRefineJoins(box:NbRefineBox){
 for(var z=box.low.z;z<=box.high.z;z++){for(var y=box.low.y;y<=box.high.y;y++){for(var x=box.low.x;x<=box.high.x;x++){nbRefineJoin(vec3i(x,y,z));}}}
}
// A seed or a spilled sample (in no tile's run) moves next step too: its
// two-cell support, travel bounds and swept surface tiles.
fn nbRefineDirect(s:Particle){
 let ballistic=nbParticleMode(s)==1.0;
 nbRefinePages(vec3i(floor((s.position.xyz-2.0)/4.0)),lrParticleTravel(s.velocity.xyz,s.before.xyz,ballistic,params.hDt.w/params.hDt.xyz));
 if(s.velocity.w < -1.0||ballistic){return;}
 let home=umTileAt(vec3u(s.position.xyz)/4u);
 nbRefineJoins(nbRefineBox(s,ballistic,home,(atomicLoad(&bins[NB_BAND_REACH+home])&1u)!=0u));
}
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
fn nbDeposited()->u32{return NB_DEPOSIT+384u*umCounts.x;}
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
   let at=NB_DEPOSIT+6u*owner+2u*(slot/PB_WINDOW);
   atomicAdd(&bins[at],bitcast<u32>(atomicLoad(&pbMomentum[slot])));atomicAdd(&bins[at+1u],w);
   atomicStore(&pbTouched[n],1u);
  }
  workgroupBarrier();
  if(lane<27u&&atomicLoad(&pbTouched[lane])!=0u){atomicStore(&bins[nbDeposited()+(pbWords[lane]&0x3fffffffu)/64u],1u);}
 }
 atomicMax(&state[5],bitcast<u32>(outside));
 for(var k=0u;k<3u;k++){if(counts[k]!=0u){atomicAdd(&state[6u+k],counts[k]);}}
}
// The census's prediction, a lean pass after DEPB (whose classify wrote the
// mode it reads): a workgroup an occupied tile, its sorted samples' support
// and swept surface tiles into workgroup words, flushed once a tile; then the
// samples no tile lists, directly: each current slab's seeds (past the
// records SORT filed, slab word 11), then the spill reserve's (state[21]).
@compute @workgroup_size(64) fn transferRefine(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 // No live sample: no tile job, but the step's first seeds still predict.
 if(lane==0u){pbJobs=select(0u,min(atomicLoad(&bins[NB_OCCUPIED]),UM_TILES),nbParticlesLive());}
 let count=workgroupUniformLoad(&pbJobs);
 for(var job=group.x;job<count;job+=groups.x){
  workgroupBarrier();
  if(lane<5u){atomicStore(&pbJoin[lane],0u);}
  if(lane==5u){atomicStore(&pbHalves,0u);}
  if(lane<48u){atomicStore(&pbTravel[lane],0u);}
  if(lane==0u){pbTile=atomicLoad(&bins[NB_OCCUPIED+4u+job]);}
  let tile=workgroupUniformLoad(&pbTile);
  if(!umTileLive(tile)){continue;}
  let at=vec3i(umTileCoord(tile));let reach=(atomicLoad(&bins[NB_BAND_REACH+tile])&1u)!=0u;
  let start=links[64u*tile];let end=links[64u*tile+63u]+atomicLoad(&bins[64u*tile+63u]);
  for(var i=start+lane;i<end;i+=64u){
   let s=particles[i];let ballistic=nbParticleMode(s)==1.0;
   // Support: the 2³ tiles from floor((p-2)/4), i.e. which half of the tile p sits in.
   let half=vec3u(clamp(vec3i(floor((s.position.xyz-2.0)/4.0))-at+1,vec3i(0),vec3i(1)));
   let o=half.x+2u*half.y+4u*half.z;
   atomicOr(&pbHalves,1u<<o);
   let travel=lrParticleTravel(s.velocity.xyz,s.before.xyz,ballistic,params.hDt.w/params.hDt.xyz);
   for(var axis=0u;axis<3u;axis++){
    if(travel[0][axis]>0.0){atomicMax(&pbTravel[6u*o+axis],bitcast<u32>(travel[0][axis]));}
    if(travel[1][axis]>0.0){atomicMax(&pbTravel[6u*o+3u+axis],bitcast<u32>(travel[1][axis]));}
   }
   if(ballistic||s.velocity.w < -1.0){continue;}
   let box=nbRefineBox(s,false,tile,reach);
   if(all(box.low>=at-2)&&all(box.high<=at+2)){
    let lo=vec3u(box.low-at+2);let hi=vec3u(box.high-at+2);
    let row=((2u<<(hi.x-lo.x))-1u)<<lo.x;var slice=0u;
    for(var y=lo.y;y<=hi.y;y++){slice|=row<<(5u*y);}
    for(var z=lo.z;z<=hi.z;z++){atomicOr(&pbJoin[z],slice);}
   }else{
    for(var z=box.low.z;z<=box.high.z;z++){for(var y=box.low.y;y<=box.high.y;y++){for(var x=box.low.x;x<=box.high.x;x++){nbRefineJoin(vec3i(x,y,z));}}}
   }
  }
  workgroupBarrier();
  for(var n=lane;n<125u;n+=64u){if((atomicLoad(&pbJoin[n/25u])&(1u<<(n%25u)))!=0u){nbRefineJoin(at-2+vec3i(i32(n%5u),i32((n/5u)%5u),i32(n/25u)));}}
  // A lane a page word: the 2³ pages about the tile, word lane%8.
  let halves=atomicLoad(&pbHalves);let r=lane/8u;let k=lane%8u;
  let region=((at-1)>>vec3u(2u))+vec3i(i32(r&1u),i32((r>>1u)&1u),i32(r>>2u));
  var value=0u;
  for(var o=0u;o<8u;o++){
   if((halves&(1u<<o))==0u){continue;}
   let low=at-1+vec3i(i32(o&1u),i32((o>>1u)&1u),i32(o>>2u));
   let mask=nbRefineMask(low,region,min(k,1u));
   if(k<2u){value|=mask;}
   else if((mask|nbRefineMask(low,region,0u))!=0u){value=max(value,atomicLoad(&pbTravel[6u*o+k-2u]));}
  }
  if(value!=0u){nbRefinePage(region,k,value);}
 }
 let slabs=min(atomicLoad(&state[20u]),UB_BRICKS);let spilled=nbSpillCount(false);
 for(var job=group.x;job<slabs+(spilled+63u)/64u;job+=groups.x){
  var first=nbSpillBase(false)+64u*(job-slabs);var n=min(64u,spilled-min(64u*(job-slabs),spilled));
  if(job<slabs){
   let id=atomicLoad(&state[NB_SLAB_OUT+job]);let filed=atomicLoad(&state[nbSlab(id,11u)]);
   first=atomicLoad(&state[nbSlab(id,5u)])+filed;n=nbSlabCount(id)-min(filed,nbSlabCount(id));
  }
  for(var k=lane;k<n;k+=64u){let s=particles[first+k];if(s.position.w>0.0){nbRefineDirect(s);}}
 }
}
`;
}
