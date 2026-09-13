/** Persistent external hose reservoir, consumed by the production edit encoder.
 * Amounts use finest-cell cubed units. The host preserves all ledger floats
 * across resident replacement with a GPU copy; per-brick scratch is transient.
 *
 * Requested = emitted + pending (subject to recorded f32 arithmetic residual).
 * Requested dose is pi*r^2*speed*dt, independent of solid overlap/residency.
 * Explicit jet edits use coverage-weighted local headroom. Continuous inflow
 * instead freezes a pressure-coupled source rate on anchored liquid components,
 * then adds/debits that rate only through validated volume microsteps.
 * Continuous debit is committed S*dt in f32, with accepted-volume balance QA;
 * it is not a claim of bit-exact equality to rounded scalar storage changes.
 * Pending belongs outside the fluid and persists while the hose is stopped.
 */
export interface GeometricSourceLayout {
  readonly ledgerBaseFloats: number;
  readonly brickScratchBaseFloats: number;
  readonly brickCapacity: number;
  readonly sourceRateBaseFloats?: number;
  readonly componentBaseWords?: number;
}
// Lanes 13/14 and 16..18 are private staging operands. Metal contracts an
// in-invocation Kahan residual to zero; retaining the independently rounded
// operands until an existing later dispatch makes the correction observable.
export const GEOMETRIC_SOURCE_LEDGER_FLOATS = 20;
export const GEOMETRIC_SOURCE_LEDGER = Object.freeze({
  pending: 0, requested: 1, emitted: 2, available: 3,
  factor: 4, eventRequested: 5, eventEmitted: 6, fault: 7,
  requestedCompensation: 8, emittedCompensation: 9,
  eventBalanceResidual: 10, eventPendingBefore: 11, pendingCompensation: 12,
  continuousPlannedRate: 15,
});
export function createGeometricSourceWGSL(layout: GeometricSourceLayout): string {
  for (const [name,value] of Object.entries(layout)) {
    if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`Invalid geometric source ${name}`);
  }
  return /* wgsl */ `
const GEOMETRIC_SOURCE_LEDGER:u32=${layout.ledgerBaseFloats}u;
const GEOMETRIC_SOURCE_SCRATCH:u32=${layout.brickScratchBaseFloats}u;
const GEOMETRIC_SOURCE_BRICKS:u32=${layout.brickCapacity}u;
const GS_STAGE_KIND:u32=13u;
const GS_STAGE_FIRST_BEFORE:u32=14u;
const GS_STAGE_FIRST_INCREMENT:u32=16u;
const GS_STAGE_SECOND_BEFORE:u32=17u;
const GS_STAGE_SECOND_INCREMENT:u32=18u;

// The rounded totals and their operands are written by one dispatch and the
// correction is reconstructed by a later dispatch. This storage boundary is
// intentional: Metal otherwise reassociates the ordinary in-invocation Kahan
// residual to zero. FastTwoSum's magnitude branch yields the exact f32 addition
// error from independently loaded operands, including signed increments.
fn geometricSourceStagedCompensation(total:f32,before:f32,increment:f32)->f32{
  var error=0.0;
  if(abs(before)>=abs(increment)){
    error=(before-total)+increment;
  }else{
    error=(increment-total)+before;
  }
  return -error;
}
fn geometricSourceFinishStagedCompensation(){
  let kind=u32(state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_KIND]);
  if(kind==0u){return;}
  state[GEOMETRIC_SOURCE_LEDGER+12u]=geometricSourceStagedCompensation(
    state[GEOMETRIC_SOURCE_LEDGER],state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_FIRST_BEFORE],
    state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_FIRST_INCREMENT]);
  let secondTotal=select(state[GEOMETRIC_SOURCE_LEDGER+2u],
    state[GEOMETRIC_SOURCE_LEDGER+1u],kind==1u);
  let secondCompensation=geometricSourceStagedCompensation(secondTotal,
    state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_SECOND_BEFORE],
    state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_SECOND_INCREMENT]);
  state[GEOMETRIC_SOURCE_LEDGER+select(9u,8u,kind==1u)]=secondCompensation;
  state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_KIND]=0.0;
}
fn geometricSourceStagePair(kind:u32,firstBefore:f32,firstIncrement:f32,
  secondBefore:f32,secondIncrement:f32){
  state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_FIRST_BEFORE]=firstBefore;
  state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_FIRST_INCREMENT]=firstIncrement;
  state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_SECOND_BEFORE]=secondBefore;
  state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_SECOND_INCREMENT]=secondIncrement;
  state[GEOMETRIC_SOURCE_LEDGER+GS_STAGE_KIND]=f32(kind);
}

fn geometricSourceAvailable(cell:u32)->f32{
  return injectionCoverage(cell)*max(0.0,
    cellOpenVolume(cell)-state[sourceDensity()+cell]*cellVolume(cell));
}

@compute @workgroup_size(64)
fn gatherGeometricSourceCapacity(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=GEOMETRIC_SOURCE_BRICKS){return;}
  let scratch=GEOMETRIC_SOURCE_SCRATCH+2u*brick;
  state[scratch]=0.0;state[scratch+1u]=0.0;
  if(!sparseCM12TopologyLifecycleAccepted()||!brickActive(brick)){return;}
  let range=templateBrickCellRange(brick,scheduledBrickResolution(brick));
  var available=0.0;var correction=0.0;
  for(var local=0u;local<range.y;local+=1u){
    let value=geometricSourceAvailable(range.x+local)-correction;
    let next=available+value;correction=(next-available)-value;available=next;
  }
  state[scratch]=available;
}

@compute @workgroup_size(1)
fn prepareGeometricSourceBudget(){
  geometricSourceFinishStagedCompensation();
  let accepted=sparseCM12TopologyLifecycleAccepted();
  atomicStore(&activity[REGION_EDIT_BACKING_RECEIPT_WORD],select(2u,1u,accepted));
  state[GEOMETRIC_SOURCE_LEDGER+4u]=0.0;
  state[GEOMETRIC_SOURCE_LEDGER+5u]=0.0;
  state[GEOMETRIC_SOURCE_LEDGER+6u]=0.0;
  state[GEOMETRIC_SOURCE_LEDGER+7u]=select(1.0,0.0,accepted);
  if(!accepted){return;}
  let requested=6.283185307179586*p.injectionRadius.w*p.injectionRadius.w
    *length(p.injectionRadius.xyz);
  let previousPending=state[GEOMETRIC_SOURCE_LEDGER];
  let pendingIncrement=requested-state[GEOMETRIC_SOURCE_LEDGER+12u];
  let pending=previousPending+pendingIncrement;
  state[GEOMETRIC_SOURCE_LEDGER+11u]=previousPending;
  state[GEOMETRIC_SOURCE_LEDGER]=pending;
  state[GEOMETRIC_SOURCE_LEDGER+5u]=requested;
  let previousRequested=state[GEOMETRIC_SOURCE_LEDGER+1u];
  let requestedIncrement=requested-state[GEOMETRIC_SOURCE_LEDGER+8u];
  state[GEOMETRIC_SOURCE_LEDGER+1u]=previousRequested+requestedIncrement;
  geometricSourceStagePair(1u,previousPending,pendingIncrement,
    previousRequested,requestedIncrement);
  var available=0.0;var correction=0.0;
  for(var brick=0u;brick<GEOMETRIC_SOURCE_BRICKS;brick+=1u){
    let value=state[GEOMETRIC_SOURCE_SCRATCH+2u*brick]-correction;
    let next=available+value;correction=(next-available)-value;available=next;
  }
  state[GEOMETRIC_SOURCE_LEDGER+3u]=available;
  if(available>0.0&&pending>0.0){
    state[GEOMETRIC_SOURCE_LEDGER+4u]=min(1.0,pending/available);
  }
}

@compute @workgroup_size(64)
fn emitGeometricSourceVolume(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=GEOMETRIC_SOURCE_BRICKS||!brickActive(brick)
    ||!sparseCM12TopologyLifecycleAccepted()){return;}
  let factor=state[GEOMETRIC_SOURCE_LEDGER+4u];if(factor<=0.0){return;}
  let range=templateBrickCellRange(brick,scheduledBrickResolution(brick));
  var emitted=0.0;var correction=0.0;
  for(var local=0u;local<range.y;local+=1u){
    let cell=range.x+local;let volume=cellVolume(cell);if(volume<=0.0){continue;}
    let addition=factor*geometricSourceAvailable(cell);if(addition<=0.0){continue;}
    let previous=state[sourceDensity()+cell];let next=previous+addition/volume;
    // Debit exactly the representable volume admitted to the fluid. Tiny
    // requests rounded away remain in the reservoir for a later event.
    let actual=(next-previous)*volume;
    let value=actual-correction;let nextEmitted=emitted+value;
    correction=(nextEmitted-emitted)-value;emitted=nextEmitted;
    let velocityAt=sourceCellVelocity()+4u*cell;
    let previousVelocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u]);
    var velocity=previousVelocity;
    if(next*volume>0.0){velocity=(previous*volume*previousVelocity
      +actual*injectedJetVelocity())/(next*volume);}
    for(var bank=0u;bank<2u;bank+=1u){
      state[select(p.stateOffsets0.x,p.stateOffsets0.y,bank!=0u)+cell]=next;
      state[select(p.stateOffsets0.z,p.stateOffsets0.w,bank!=0u)+cell]=1.0;
      let at=select(p.stateOffsets1.x,p.stateOffsets1.y,bank!=0u)+4u*cell;
      state[at]=velocity.x;state[at+1u]=velocity.y;state[at+2u]=velocity.z;state[at+3u]=0.0;
    }
    if(actual>0.0){
      cm12PublishCollocatedWetEffectiveVelocity(cell,velocity,true);
      incrementalActivityMarkCellClosure(cell);
    }
  }
  state[GEOMETRIC_SOURCE_SCRATCH+2u*brick+1u]=emitted;
}

@compute @workgroup_size(1)
fn finalizeGeometricSourceLedger(){
  geometricSourceFinishStagedCompensation();
  if(!sparseCM12TopologyLifecycleAccepted()){return;}
  var emitted=0.0;var correction=0.0;
  for(var brick=0u;brick<GEOMETRIC_SOURCE_BRICKS;brick+=1u){
    let value=state[GEOMETRIC_SOURCE_SCRATCH+2u*brick+1u]-correction;
    let next=emitted+value;correction=(next-emitted)-value;emitted=next;
  }
  let before=state[GEOMETRIC_SOURCE_LEDGER];
  let pendingIncrement=-emitted-state[GEOMETRIC_SOURCE_LEDGER+12u];
  let pending=before+pendingIncrement;
  state[GEOMETRIC_SOURCE_LEDGER]=pending;
  state[GEOMETRIC_SOURCE_LEDGER+6u]=emitted;
  let prior=state[GEOMETRIC_SOURCE_LEDGER+2u];
  let emittedIncrement=emitted-state[GEOMETRIC_SOURCE_LEDGER+9u];
  state[GEOMETRIC_SOURCE_LEDGER+2u]=prior+emittedIncrement;
  geometricSourceStagePair(2u,before,pendingIncrement,prior,emittedIncrement);
  state[GEOMETRIC_SOURCE_LEDGER+10u]=(pending+emitted)-before;
  // Preserve the signed rounding residual explicitly, never erase it by a
  // hidden reservoir clamp. Significant overdraft is a source-ledger fault.
  if(!(pending>=-8.0*1.1920928955078125e-7*max(abs(before),abs(emitted)))){
    state[GEOMETRIC_SOURCE_LEDGER+7u]=2.0;
  }
}
@compute @workgroup_size(1)
fn completeGeometricSourceLedger(){geometricSourceFinishStagedCompensation();}
${createContinuousGeometricSourceWGSL(layout)}
`;
}

export type SparseGeometricSourceLayout = GeometricSourceLayout;
export function createGeometricSourceResidentWGSL(layout?: SparseGeometricSourceLayout): string {
  return layout ? createGeometricSourceWGSL(layout) : "";
}

function createContinuousGeometricSourceWGSL(layout: GeometricSourceLayout): string {
  if (layout.sourceRateBaseFloats === undefined || layout.componentBaseWords === undefined) {
    return `fn geometricSourceRate(cell:u32)->f32{return 0.0;}
fn geometricSourceCommitMicrostep(dt:f32){}
fn geometricSourceFinishStagedCompensation(){}`;
  }
  return /* wgsl */ `
const GS_RATE:u32=${layout.sourceRateBaseFloats}u;
const GS_COMPONENT:u32=${layout.componentBaseWords}u;
fn gsEnabled()->bool{return p.inflowVelocity.w>0.5&&p.inflowOutlet.w>0.0
 &&length(p.inflowVelocity.xyz)>0.0&&p.frame.x>0.0;}
// Stopped hose rates are logically zero. Keeping that fact at the reader means
// ordinary no-inflow frames do not have to clear every accepted cell merely to
// overwrite a rate array that no consumer may observe.
fn geometricSourceRate(cell:u32)->f32{
 if(!gsEnabled()){return 0.0;}return state[GS_RATE+cell];
}
fn gsWeight(cell:u32)->f32{
 if(!gsEnabled()||!cellActive(cell)){return 0.0;}
 let speed=length(p.inflowVelocity.xyz);let direction=p.inflowVelocity.xyz/speed;
 let lengthFine=max(speed*p.frame.x,cellMinimumWidth(cell));
 let relative=cellCenter(cell)-p.inflowOutlet.xyz;
 let axial=dot(relative,direction);let radial=length(relative-direction*axial);
 let edge=max(0.5*cellMinimumWidth(cell),0.5);
 let coverage=clamp(0.5-(radial-p.inflowOutlet.w)/edge,0.0,1.0)
  *clamp(0.5+axial/edge,0.0,1.0)*clamp(0.5+(lengthFine-axial)/edge,0.0,1.0);
 let weight=coverage*cellOpenVolume(cell);
 return select(0.0,weight,weight>=0.0&&weight<=3.402823466e38);
}
fn gsCellWithWeight(cell:u32,weight:f32)->bool{
 return gsEnabled()&&cell!=INVALID&&cellActive(cell)&&cellOpenVolume(cell)>1e-8
  &&(rawPressureDensity(cell)>=CM12_LIQUID_ISOVALUE||pressureCellSubmerged(cell)||weight>0.0);
}
fn gsRow(row:u32)->bool{return gsEnabled()&&row!=INVALID&&acceptedRowMember(row)
 &&rowAccepted(row)&&rowOpenFraction(row)>0.0;}
fn gsRoot(cell:u32)->u32{
 var root=cell;
 for(var depth=0u;depth<64u;depth+=1u){
  let parent=bitcast<u32>(atomicLoad(&conditioning[GS_COMPONENT+4u*root]));
  if(parent==root||parent==INVALID){return parent;}root=parent;
 }
 return INVALID;
}
fn gsMember(cell:u32)->bool{
 if(cell==INVALID){return false;}
 return bitcast<u32>(atomicLoad(&conditioning[GS_COMPONENT+4u*cell]))!=INVALID;
}
@compute @workgroup_size(1)
fn beginContinuousGeometricSource(){
 geometricSourceFinishStagedCompensation();
 state[GEOMETRIC_SOURCE_LEDGER+7u]=0.0;
 state[GEOMETRIC_SOURCE_LEDGER+15u]=0.0;
 atomicStore(&conditioning[GS_COMPONENT+3u],0);
 let requested=select(0.0,3.141592653589793*p.inflowOutlet.w*p.inflowOutlet.w
  *length(p.inflowVelocity.xyz)*p.frame.x,gsEnabled());
 state[GEOMETRIC_SOURCE_LEDGER+5u]=requested;
 state[GEOMETRIC_SOURCE_LEDGER+6u]=0.0;
 if(!(requested>=0.0&&requested<=3.402823466e38)){
  atomicStore(&conditioning[GS_COMPONENT+3u],1);state[GEOMETRIC_SOURCE_LEDGER+7u]=3.0;return;
 }
 let pending=state[GEOMETRIC_SOURCE_LEDGER];
 state[GEOMETRIC_SOURCE_LEDGER+11u]=pending;
 let pendingIncrement=requested-state[GEOMETRIC_SOURCE_LEDGER+12u];
 state[GEOMETRIC_SOURCE_LEDGER]=pending+pendingIncrement;
 let total=state[GEOMETRIC_SOURCE_LEDGER+1u];
 let requestedIncrement=requested-state[GEOMETRIC_SOURCE_LEDGER+8u];
 state[GEOMETRIC_SOURCE_LEDGER+1u]=total+requestedIncrement;
 geometricSourceStagePair(1u,pending,pendingIncrement,total,requestedIncrement);
}
@compute @workgroup_size(64)
fn initializeContinuousGeometricSource(@builtin(global_invocation_id)gid:vec3u){
 let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
 // The rate plane is construction scratch until publication. Retaining the
 // geometric weight here removes repeated outlet geometry from every union
 // round and from the later gather/publish pair.
 let weight=gsWeight(cell);state[GS_RATE+cell]=weight;
 let at=GS_COMPONENT+4u*cell;
 atomicStore(&conditioning[at],bitcast<i32>(select(INVALID,cell,
   gsCellWithWeight(cell,weight))));
 atomicStore(&conditioning[at+1u],0);
}
@compute @workgroup_size(64)
fn connectContinuousGeometricSource(@builtin(global_invocation_id)gid:vec3u){
 let row=acceptedTemplateRowInvocation(gid.x);if(!gsRow(row)){return;}
 let terms=rowTermRange(row);var root=INVALID;
 for(var at=terms.x;at<terms.y;at+=1u){let cell=termCell(at);if(gsMember(cell)){root=min(root,gsRoot(cell));}}
 if(root==INVALID){return;}
 for(var at=terms.x;at<terms.y;at+=1u){let cell=termCell(at);if(!gsMember(cell)){continue;}
  let previous=gsRoot(cell);if(previous!=INVALID){atomicMin(&conditioning[GS_COMPONENT+4u*previous],i32(root));}
  atomicMin(&conditioning[GS_COMPONENT+4u*cell],i32(root));
 }
}
@compute @workgroup_size(64)
fn compressContinuousGeometricSource(@builtin(global_invocation_id)gid:vec3u){
 let cell=acceptedTemplateCellInvocation(gid.x);if(!gsMember(cell)){return;}
 let root=gsRoot(cell);
 if(root==INVALID){atomicStore(&conditioning[GS_COMPONENT+3u],1);return;}
 atomicMin(&conditioning[GS_COMPONENT+4u*cell],i32(root));
}
@compute @workgroup_size(64)
fn sealContinuousGeometricSource(@builtin(global_invocation_id)gid:vec3u){
 let row=acceptedTemplateRowInvocation(gid.x);if(!gsRow(row)){return;}
 let terms=rowTermRange(row);var root=INVALID;var sum=0.0;var scale=0.0;var headroom=false;
 for(var at=terms.x;at<terms.y;at+=1u){let cell=termCell(at);if(!gsMember(cell)){continue;}
  let other=gsRoot(cell);
  if(other==INVALID||(root!=INVALID&&root!=other)){
   atomicStore(&conditioning[GS_COMPONENT+3u],1);return;
  }
  root=other;let coefficient=termCoefficient(at);sum+=coefficient;scale+=abs(coefficient);
  headroom=headroom||state[sourceDensity()+cell]*cellVolume(cell)<cellOpenVolume(cell);
 }
 if(root==INVALID){return;}
 if(abs(sum)>9.5367431640625e-7*scale){atomicOr(&conditioning[GS_COMPONENT+4u*root+1u],1);}
 if(headroom){atomicOr(&conditioning[GS_COMPONENT+4u*root+1u],2);}
}
@compute @workgroup_size(64)
fn gatherContinuousGeometricSourceWeights(@builtin(global_invocation_id)gid:vec3u){
 let brick=gid.x;if(brick>=GEOMETRIC_SOURCE_BRICKS){return;}
 let scratch=GEOMETRIC_SOURCE_SCRATCH+2u*brick;state[scratch]=0.0;state[scratch+1u]=0.0;
 if(!brickActive(brick)){return;}
 let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));var total=0.0;var correction=0.0;
 for(var local=0u;local<range.y;local+=1u){let cell=range.x+local;let weight=state[GS_RATE+cell];
  if(weight<=0.0){continue;}let root=gsRoot(cell);
  // Headroom in an unanchored all-pressure component cannot make a positive
  // integrated-divergence RHS solvable. Keep its dose in the external hose.
  if(root!=INVALID&&(atomicLoad(&conditioning[GS_COMPONENT+4u*root+1u])&1)!=0){
   let value=weight-correction;let next=total+value;correction=(next-total)-value;total=next;
  }
 }
 state[scratch]=total;
}
@compute @workgroup_size(1)
fn prepareContinuousGeometricSourceBudget(){
 geometricSourceFinishStagedCompensation();
 var total=0.0;var correction=0.0;
 for(var brick=0u;brick<GEOMETRIC_SOURCE_BRICKS;brick+=1u){
  let value=state[GEOMETRIC_SOURCE_SCRATCH+2u*brick]-correction;
  let next=total+value;correction=(next-total)-value;total=next;
 }
 state[GEOMETRIC_SOURCE_LEDGER+3u]=total;state[GEOMETRIC_SOURCE_LEDGER+4u]=0.0;
 if(atomicLoad(&conditioning[GS_COMPONENT+3u])!=0){state[GEOMETRIC_SOURCE_LEDGER+7u]=3.0;return;}
 let pending=state[GEOMETRIC_SOURCE_LEDGER];
 // Bound catch-up to one weighted source-cell capacity per outer step;
 // unserved requested volume stays pending rather than exhausting a CFL cap.
 if(total>0.0&&pending>0.0&&p.frame.x>0.0){
  let factor=min(pending,total)/total/p.frame.x;
  if(!(factor>=0.0&&factor<=3.402823466e38)){
   state[GEOMETRIC_SOURCE_LEDGER+7u]=3.0;return;
  }
  state[GEOMETRIC_SOURCE_LEDGER+4u]=factor;
 }
}
@compute @workgroup_size(64)
fn publishContinuousGeometricSourceRates(@builtin(global_invocation_id)gid:vec3u){
 let brick=gid.x;if(brick>=GEOMETRIC_SOURCE_BRICKS||!brickActive(brick)){return;}
 let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));var sum=0.0;var correction=0.0;
 for(var local=0u;local<range.y;local+=1u){let cell=range.x+local;var rate=0.0;
  let weight=state[GS_RATE+cell];
  if(weight>0.0){let root=gsRoot(cell);
   if(root!=INVALID&&(atomicLoad(&conditioning[GS_COMPONENT+4u*root+1u])&1)!=0){
    rate=weight*state[GEOMETRIC_SOURCE_LEDGER+4u];
   }
  }
  state[GS_RATE+cell]=rate;
  let value=rate-correction;let next=sum+value;correction=(next-sum)-value;sum=next;
 }
 state[GEOMETRIC_SOURCE_SCRATCH+2u*brick+1u]=sum;
}
@compute @workgroup_size(1)
fn finalizeContinuousGeometricSourceRates(){
 var total=0.0;var correction=0.0;
 for(var brick=0u;brick<GEOMETRIC_SOURCE_BRICKS;brick+=1u){
  let value=state[GEOMETRIC_SOURCE_SCRATCH+2u*brick+1u]-correction;
  let next=total+value;correction=(next-total)-value;total=next;
 }
 state[GEOMETRIC_SOURCE_LEDGER+15u]=total;
}
fn gsLower()->vec3f{return min(p.inflowOutlet.xyz,
 p.inflowOutlet.xyz+2.0*p.inflowVelocity.xyz*p.frame.x)-vec3f(p.inflowOutlet.w+1.0);}
fn gsUpper()->vec3f{return max(p.inflowOutlet.xyz,
 p.inflowOutlet.xyz+2.0*p.inflowVelocity.xyz*p.frame.x)+vec3f(p.inflowOutlet.w+1.0);}
@compute @workgroup_size(4,4,4)
fn allocateContinuousGeometricSourcePages(@builtin(global_invocation_id)gid:vec3u){
 if(!gsEnabled()){return;}
 let width=f32(BRICK_FINE_RESOLUTION);let lower=vec3i(floor(gsLower()/width));
 let upper=vec3i(floor(gsUpper()/width));let extent=vec3u(upper-lower+vec3i(1));
 if(any(gid>=extent)){return;}
 let coordinate=lower+vec3i(gid);
 if(cm12WorldOwnerAt(coordinate)!=CM12_WDR_INVALID){return;}
 revokeCM12SourceTopologyLease();
 let leaf=cm12WorldAllocateUniqueExact(coordinate,0u);
 if(leaf==CM12_WDR_INVALID||leaf<CM12_WDR_INITIAL_LEAVES){return;}
 let page=leaf-CM12_WDR_INITIAL_LEAVES;let base=topologyWorklistBase();
 if(page>=atomicLoad(&topologyArena[base+27u])){return;}
 let pageBase=candidateTopologyPageBase(page);
 let claim=atomicCompareExchangeWeak(&topologyArena[pageBase+2u],0u,0xffffffffu);
 if(!claim.exchanged){return;}
 atomicStore(&topologyArena[pageBase],leaf);
 atomicStore(&topologyArena[pageBase+1u],BRICK_FINE_RESOLUTION);
 atomicStore(&topologyArena[pageBase+3u],0u);
 atomicStore(&topologyArena[pageBase+11u],atomicLoad(&topologyArena[CM12_WDR_BASE+10u]));
 atomicStore(&topologyArena[pageBase+2u],BRICK_FINE_RESOLUTION*BRICK_FINE_RESOLUTION*BRICK_FINE_RESOLUTION);
}
@compute @workgroup_size(64)
fn activateContinuousGeometricSourcePages(@builtin(global_invocation_id)gid:vec3u){
 let brick=gid.x;if(!gsEnabled()||brick>=p.dispatch.w||!cm12WorldLeafAllocated(brick)){return;}
 let lower=vec3f(cm12WorldLeafCoordinate(brick)*i32(BRICK_FINE_RESOLUTION));
 let upper=lower+vec3f(f32(brickSpan(brick)*BRICK_FINE_RESOLUTION));
 if(any(gsUpper()<lower)||any(gsLower()>upper)){return;}
 if(brickActive(brick)&&acceptedBrickResolution(brick)==BRICK_FINE_RESOLUTION){return;}
 stageDemandedFrontierPage(brick);
}
fn geometricSourceCommitMicrostep(dt:f32){
 geometricSourceFinishStagedCompensation();
 let emitted=state[GEOMETRIC_SOURCE_LEDGER+15u]*dt;if(emitted<=0.0){return;}
 let pending=state[GEOMETRIC_SOURCE_LEDGER];
 let decrement=-emitted-state[GEOMETRIC_SOURCE_LEDGER+12u];let next=pending+decrement;
 state[GEOMETRIC_SOURCE_LEDGER]=next;
 let previous=state[GEOMETRIC_SOURCE_LEDGER+2u];
 let increment=emitted-state[GEOMETRIC_SOURCE_LEDGER+9u];
 state[GEOMETRIC_SOURCE_LEDGER+2u]=previous+increment;
 geometricSourceStagePair(2u,pending,decrement,previous,increment);
 state[GEOMETRIC_SOURCE_LEDGER+6u]+=emitted;
 state[GEOMETRIC_SOURCE_LEDGER+10u]=(next+emitted)-pending;
 if(!(next>=-8.0*1.1920928955078125e-7*max(abs(pending),abs(emitted)))){
  state[GEOMETRIC_SOURCE_LEDGER+7u]=2.0;
 }
}
`;
}
