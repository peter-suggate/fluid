import { UNIFORM_STAGE_IMPORTANCE as I } from "./uniform-stage-grids";

export const NARROW_BAND_BUDGET_ENTRIES=["activityRank","activityCutoff","activityAdmit"] as const;
/** Rank keys: a tile's top criterion score. */
const KEYS=256;
/** One cutoff word, then a rank word and a flight word per tile. */
export const narrowBandBudgetWords=(tiles:number)=>KEYS+1+2*tiles;

/** Same-census selection by score, with an indivisible cutoff tie group.
 * The percentage is a soft target: equally important tiles must receive the
 * same decision regardless of storage order, page allocation or reflection.
 * A zero target admits none; otherwise include the entire score group that
 * reaches the target. Physical support is added later.
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
fn nbBudgetOn()->bool{return policy.activity.x<100u;}
fn nbBudgetHist(b:u32)->u32{return holdIndex(UM_TILES)+b;}
fn nbBudgetState(k:u32)->u32{return nbBudgetHist(NB_BUDGET_KEYS)+k;}
fn nbBudgetRank(t:u32)->u32{return nbBudgetState(1u)+t;}
fn nbBudgetFlight(t:u32)->u32{return nbBudgetRank(UM_TILES)+t;}
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
// Liquid on a solid, or on a tile with no stored tile below or above it on an axis, has landed.
fn nbBudgetGrounded(p:vec3i)->bool{
 if(umSolidTile(vec3u(p))||any(umTileLowOpen(p))){return true;}
 return umTileKind(p+vec3i(1,0,0))!=UM_RESIDENT||umTileKind(p+vec3i(0,1,0))!=UM_RESIDENT||umTileKind(p+vec3i(0,0,1))!=UM_RESIDENT;
}
fn nbBudgetInFlight(t:u32,p:vec3i,w1:u32)->bool{
 if(umSourceTile(p)){return true;}
 if(!nbBudgetLiquid(w1)||nbBudgetGrounded(p)){return false;}
 let own=atomicLoad(&census[nbBudgetFlight(t)]);
 // Liquid that was not here came from within a step's travel: a tile and what it crosses at its speed.
 let reach=select(nbBudgetReach(t),1,(own&3u)!=0u);
 var flying=0u;var resident=0u;
 for(var z=-reach;z<=reach;z++){for(var y=-reach;y<=reach;y++){for(var x=-reach;x<=reach;x++){
  let q=p+vec3i(x,y,z);if(all(q==p)||!umTilePresent(q)){continue;}
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
 let t=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(t)){return;}
 var rank=NB_BUDGET_NONE;
 let own=atomicLoad(&census[importanceIndex(t,1u)]);let p=vec3i(umTileCoord(t));
 let flight=nbBudgetInFlight(t,p,own);
 atomicOr(&census[nbBudgetFlight(t)],(select(0u,NB_FLIGHT,flight)|select(0u,NB_LIQUID,nbBudgetLiquid(own)))<<8u);
 // In flight, or deep COLD (its width is frozen: ccDeep, the census's): outside the budget.
 if((own&${I.required}u)!=0u&&(flight||ccDeep(t))){rank=NB_BUDGET_FLIGHT;}
 else if((own&${I.required}u)!=0u){
  let key=nbBudgetScore(atomicLoad(&census[importanceIndex(t,0u)]),own);
  rank=key<<8u;atomicAdd(&census[nbBudgetHist(key)],1u);
 }
 atomicStore(&census[nbBudgetRank(t)],rank);
}
@compute @workgroup_size(1) fn activityCutoff(){
 var total=0u;for(var s=0u;s<NB_BUDGET_KEYS;s++){total+=atomicLoad(&census[nbBudgetHist(s)]);}
 let allowed=(total/100u)*policy.activity.x+((total%100u)*policy.activity.x)/100u;
 var above=0u;var cutoff=NB_BUDGET_KEYS;
 for(var k=0u;k<NB_BUDGET_KEYS;k++){
  let score=NB_BUDGET_KEYS-1u-k;let count=atomicExchange(&census[nbBudgetHist(score)],0u);
  if(allowed>0u&&cutoff==NB_BUDGET_KEYS&&above+count>=allowed&&count>0u){cutoff=score;}
  if(cutoff==NB_BUDGET_KEYS){above+=count;}
 }
 atomicStore(&census[nbBudgetState(0u)],cutoff);
}
@compute @workgroup_size(64) fn activityAdmit(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x+umDispatchX*64u*gid.y;if(!umTileLive(t)){return;}
 var w1=atomicLoad(&census[importanceIndex(t,1u)]);
 let rank=atomicLoad(&census[nbBudgetRank(t)]);
 atomicStore(&census[nbBudgetFlight(t)],(atomicLoad(&census[nbBudgetFlight(t)])>>8u)&3u);
 // In flight: admitted outside the budget.
 if(rank!=NB_BUDGET_FLIGHT&&rank!=NB_BUDGET_NONE){
  let score=rank>>8u;let cutoff=atomicLoad(&census[nbBudgetState(0u)]);
  let keep=score>=cutoff;
  if(!keep){w1=(w1&~${I.required|I.held}u)|${I.dropped}u;atomicStore(&census[importanceIndex(t,1u)],w1);atomicStore(&census[holdIndex(t)],0u);}
 }
 publishImportance(t,w1);
}
`;
