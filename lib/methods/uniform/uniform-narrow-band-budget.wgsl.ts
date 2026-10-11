import { UNIFORM_STAGE_IMPORTANCE as I } from "./uniform-stage-grids";

export const NARROW_BAND_BUDGET_ENTRIES=["activityRank","activityCutoff","activityTies","activityTiePrefix","activityAdmit"] as const;
/** Rank keys: a tile's top criterion score. */
const KEYS=256;
/** Then a rank word and a flight word per tile, and a tie count per 64 tiles. */
export const narrowBandBudgetWords=(tiles:number)=>KEYS+3+tiles+Math.ceil(tiles/64)+tiles;

/** Same-census selection: histogram threshold plus a stable tile-order tie
 * quota. Each chunk scans its ties in parallel; the small chunk prefix is
 * serial. Exactly floor(candidate count * percent / 100) seeds survive,
 * including when every score saturates. Physical support is added later.
 *
 * Source liquid in flight asks for a new set of tiles every census, all for
 * its shape. Letter B of the NB-FLIP letters (60 Hz, 50%) asked for 140
 * tiles a census on its way down and was admitted 56, 25, 10 and then 4 of them; its samples lost the surface a
 * quarter second after the drop and it landed 2.6% short of its volume with
 * its lower edge on 4h tiles. So that liquid is outside the budget, as the
 * source that makes it is. A tile is in flight while its liquid came from a
 * source's tiles and no more of the liquid beside it is resident than is in
 * flight. So a body stays in flight through the spray of an earlier splash,
 * as letter D does to its landing, and each part of it lands as it enters
 * the pool. Liquid on a solid or a wall of the domain is resident, so that
 * what a source pours into an empty vessel is not in flight for good.
 * Admitted every census, B lands with its volume and no surface on a 4h
 * tile. */
export const narrowBandBudgetWGSL=/* wgsl */`
const NB_BUDGET_KEYS=${KEYS}u;
const NB_BUDGET_NONE=0xffffffffu;
// Within a census: a requesting tile in flight, which is not ranked.
const NB_BUDGET_FLIGHT=0xfffffffdu;
// A tile's flight word: the last census's pair in bits 0..1, this census's
// in bits 8..9 from activityRank until activityAdmit moves it down. A
// neighbour reads the low pair while the tile's own lane sets the high.
const NB_FLIGHT=1u;
const NB_LIQUID=2u;
fn nbBudgetOn()->bool{return nbBudgetSupported&&policy.activity.x<100u;}
fn nbBudgetHist(b:u32)->u32{return holdIndex(UM_TILES)+b;}
fn nbBudgetState(k:u32)->u32{return nbBudgetHist(NB_BUDGET_KEYS)+k;}
fn nbBudgetRank(t:u32)->u32{return nbBudgetState(3u)+t;}
fn nbBudgetChunk(c:u32)->u32{return nbBudgetRank(UM_TILES)+c;}
fn nbBudgetFlight(t:u32)->u32{return nbBudgetChunk((UM_TILES+63u)/64u)+t;}
fn nbBudgetLiquid(w1:u32)->bool{return (w1&${I.crossing|I.wet}u)!=0u;}
// Liquid the last census found here that was not in flight.
fn nbBudgetResident(flight:u32)->bool{return (flight&3u)==NB_LIQUID;}
// Tiles a step's travel can cross: its own, and one more per 4h of its fastest face (unset bounds have none).
fn nbBudgetReach(t:u32)->i32{
 var travel=0.0;
 for(var a=0u;a<3u;a++){
  let low=atomicLoad(&census[boundIndex(t,a)]);let high=atomicLoad(&census[boundIndex(t,3u+a)]);
  if(low!=0xffffffffu){travel=max(travel,abs(orderValue(low))*policy.step.x/H[a]);}
  if(high!=0u){travel=max(travel,abs(orderValue(high))*policy.step.x/H[a]);}
 }
 return 1+i32(min(travel,12.0)/4.0);
}
// Liquid on a solid or a wall of the domain has landed.
fn nbBudgetGrounded(p:vec3i)->bool{return umSolidTile(vec3u(p))||any(p==vec3i(0))||any(p+vec3i(1)==vec3i(UM_T));}
fn nbBudgetInFlight(t:u32,p:vec3i,w1:u32)->bool{
 if(umSourceTile(p)){return true;}
 if(!nbBudgetLiquid(w1)||nbBudgetGrounded(p)){return false;}
 let own=atomicLoad(&census[nbBudgetFlight(t)]);
 // Liquid that was not here came from within a step's travel: a tile and what it crosses at its speed.
 let reach=select(nbBudgetReach(t),1,(own&3u)!=0u);
 var flying=0u;var resident=0u;
 for(var z=-reach;z<=reach;z++){for(var y=-reach;y<=reach;y++){for(var x=-reach;x<=reach;x++){
  let q=p+vec3i(x,y,z);if(all(q==p)||any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
  let flight=atomicLoad(&census[nbBudgetFlight(umTileAt(vec3u(q)))]);
  flying+=flight&NB_FLIGHT;resident+=u32(nbBudgetResident(flight));
 }}}
 // Its own count is left out: a layer one tile deep on resident liquid is resident.
 return ((own&NB_FLIGHT)!=0u||flying!=0u)&&resident<=flying;
}
fn nbBudgetScore(w0:u32,w1:u32)->u32{
 var top=0u;
 for(var k=0u;k<6u;k++){
  if(!criterionOn(k)){continue;}let s=umScoreByte(w0,w1,k);
  top=max(top,s);
 }
 return top;
}
// After importance: every tile's words are final, and this pass writes none.
@compute @workgroup_size(64) fn activityRank(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x+umDispatchX*64u*gid.y;if(t>=UM_TILES){return;}
 var rank=NB_BUDGET_NONE;
 let own=atomicLoad(&census[importanceIndex(t,1u)]);let p=vec3i(umTileCoord(t));
 let flight=nbBudgetInFlight(t,p,own);
 atomicOr(&census[nbBudgetFlight(t)],(select(0u,NB_FLIGHT,flight)|select(0u,NB_LIQUID,nbBudgetLiquid(own)))<<8u);
 if((own&${I.required}u)!=0u&&flight){rank=NB_BUDGET_FLIGHT;}
 else if((own&${I.required}u)!=0u){
  let key=nbBudgetScore(atomicLoad(&census[importanceIndex(t,0u)]),own);
  rank=key<<8u;atomicAdd(&census[nbBudgetHist(key)],1u);
 }
 atomicStore(&census[nbBudgetRank(t)],rank);
}
@compute @workgroup_size(1) fn activityCutoff(){
 var total=0u;for(var s=0u;s<NB_BUDGET_KEYS;s++){total+=atomicLoad(&census[nbBudgetHist(s)]);}
 let allowed=(total/100u)*policy.activity.x+((total%100u)*policy.activity.x)/100u;
 var above=0u;var cutoff=NB_BUDGET_KEYS;var ties=0u;
 for(var k=0u;k<NB_BUDGET_KEYS;k++){
  let score=NB_BUDGET_KEYS-1u-k;let count=atomicExchange(&census[nbBudgetHist(score)],0u);
  if(cutoff==NB_BUDGET_KEYS&&above+count>=allowed&&count>0u){cutoff=score;ties=allowed-above;}
  if(cutoff==NB_BUDGET_KEYS){above+=count;}
 }
 atomicStore(&census[nbBudgetState(0u)],cutoff);
 atomicStore(&census[nbBudgetState(1u)],ties);
 atomicStore(&census[nbBudgetState(2u)],allowed);
}
var<workgroup> nbTies:array<u32,64>;
@compute @workgroup_size(64) fn activityTies(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let chunk=group.x+umDispatchX*group.y;let t=64u*chunk+lane;
 if(64u*chunk>=UM_TILES){return;}
 var rank=0xffffffffu;if(t<UM_TILES){rank=atomicLoad(&census[nbBudgetRank(t)]);}
 let tie=u32(rank!=0xffffffffu&&(rank>>8u)==atomicLoad(&census[nbBudgetState(0u)]));
 nbTies[lane]=tie;workgroupBarrier();
 for(var stride=1u;stride<64u;stride*=2u){
  var add=0u;if(lane>=stride){add=nbTies[lane-stride];}workgroupBarrier();
  nbTies[lane]+=add;workgroupBarrier();
 }
 if(t<UM_TILES&&tie!=0u){atomicStore(&census[nbBudgetRank(t)],rank|(nbTies[lane]-1u));}
 if(lane==63u){atomicStore(&census[nbBudgetChunk(chunk)],nbTies[lane]);}
}
// A lane per block of chunks: each sums its own, then starts it at the blocks
// before. One lane over every chunk was 0.57 ms of each figure-9 census.
var<workgroup> nbTieBlocks:array<u32,64>;
@compute @workgroup_size(64) fn activityTiePrefix(@builtin(local_invocation_index) lane:u32){
 let chunks=(UM_TILES+63u)/64u;let block=(chunks+63u)/64u;
 let first=min(lane*block,chunks);let last=min(first+block,chunks);
 var own=0u;for(var c=first;c<last;c++){own+=atomicLoad(&census[nbBudgetChunk(c)]);}
 nbTieBlocks[lane]=own;workgroupBarrier();
 var sum=0u;for(var k=0u;k<lane;k++){sum+=nbTieBlocks[k];}
 for(var c=first;c<last;c++){
  let n=atomicLoad(&census[nbBudgetChunk(c)]);atomicStore(&census[nbBudgetChunk(c)],sum);sum+=n;
 }
}
@compute @workgroup_size(64) fn activityAdmit(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x+umDispatchX*64u*gid.y;if(t>=UM_TILES){return;}
 var w1=atomicLoad(&census[importanceIndex(t,1u)]);
 let rank=atomicLoad(&census[nbBudgetRank(t)]);
 atomicStore(&census[nbBudgetFlight(t)],(atomicLoad(&census[nbBudgetFlight(t)])>>8u)&3u);
 // In flight: admitted outside the budget.
 if(rank!=NB_BUDGET_FLIGHT&&rank!=NB_BUDGET_NONE){
  let score=rank>>8u;let cutoff=atomicLoad(&census[nbBudgetState(0u)]);
  let tie=atomicLoad(&census[nbBudgetChunk(t/64u)])+(rank&255u);
  let keep=score>cutoff||(score==cutoff&&tie<atomicLoad(&census[nbBudgetState(1u)]));
  if(!keep){w1=(w1&~${I.required|I.held}u)|${I.dropped}u;atomicStore(&census[importanceIndex(t,1u)],w1);atomicStore(&census[holdIndex(t)],0u);}
 }
 publishImportance(t,w1);
}
`;
