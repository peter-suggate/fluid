import { UNIFORM_STAGE_IMPORTANCE as I } from "./uniform-stage-grids";

export const NARROW_BAND_BUDGET_ENTRIES=["activityCutoff","activityTies","activityTiePrefix","activityAdmit"] as const;
export const narrowBandBudgetWords=(tiles:number)=>256+3+tiles+Math.ceil(tiles/64);

/** Same-census selection: histogram threshold plus a stable tile-order tie
 * quota. Each chunk scans its ties in parallel; the small chunk prefix is
 * serial. Exactly floor(candidate count * percent / 100) seeds survive,
 * including when every score saturates. Physical support is added later. */
export const narrowBandBudgetWGSL=/* wgsl */`
fn nbBudgetOn()->bool{return nbBudgetSupported&&policy.activity.x<100u;}
fn nbBudgetHist(b:u32)->u32{return binIndex(64u)+b;}
fn nbBudgetState(k:u32)->u32{return nbBudgetHist(256u)+k;}
fn nbBudgetRank(t:u32)->u32{return nbBudgetState(3u)+t;}
fn nbBudgetChunk(c:u32)->u32{return nbBudgetRank(UM_TILES)+c;}
fn nbBudgetScore(w0:u32,w1:u32)->u32{
 var top=0u;
 for(var k=0u;k<6u;k++){
  if(!criterionOn(k)){continue;}let s=umScoreByte(w0,w1,k);
  top=max(top,s);
 }
 return top;
}
fn nbBudgetRecord(t:u32,w0:u32,w1:u32,required:bool){
 var rank=0xffffffffu;
 if(required){let score=nbBudgetScore(w0,w1);rank=score<<8u;atomicAdd(&census[nbBudgetHist(score)],1u);}
 atomicStore(&census[nbBudgetRank(t)],rank);
}
@compute @workgroup_size(1) fn activityCutoff(){
 var total=0u;for(var s=0u;s<256u;s++){total+=atomicLoad(&census[nbBudgetHist(s)]);}
 let allowed=(total/100u)*policy.activity.x+((total%100u)*policy.activity.x)/100u;
 var above=0u;var cutoff=256u;var ties=0u;
 for(var k=0u;k<256u;k++){
  let score=255u-k;let count=atomicExchange(&census[nbBudgetHist(score)],0u);
  if(cutoff==256u&&above+count>=allowed&&count>0u){cutoff=score;ties=allowed-above;}
  if(cutoff==256u){above+=count;}
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
@compute @workgroup_size(1) fn activityTiePrefix(){
 var sum=0u;
 for(var c=0u;c<(UM_TILES+63u)/64u;c++){
  let n=atomicLoad(&census[nbBudgetChunk(c)]);atomicStore(&census[nbBudgetChunk(c)],sum);sum+=n;
 }
}
@compute @workgroup_size(64) fn activityAdmit(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x+umDispatchX*64u*gid.y;if(t>=UM_TILES){return;}
 var w1=atomicLoad(&census[importanceIndex(t,1u)]);
 let rank=atomicLoad(&census[nbBudgetRank(t)]);
 if(rank!=0xffffffffu){
  let score=rank>>8u;let cutoff=atomicLoad(&census[nbBudgetState(0u)]);
  let tie=atomicLoad(&census[nbBudgetChunk(t/64u)])+(rank&255u);
  let keep=score>cutoff||(score==cutoff&&tie<atomicLoad(&census[nbBudgetState(1u)]));
  if(!keep){w1=(w1&~${I.required|I.held}u)|${I.dropped}u;atomicStore(&census[importanceIndex(t,1u)],w1);atomicStore(&census[holdIndex(t)],0u);}
 }
 publishImportance(t,w1);
}
`;
