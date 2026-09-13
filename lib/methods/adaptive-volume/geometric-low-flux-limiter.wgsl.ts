import type { SparseGeometricVolumeLayout } from "./resident-volume.wgsl";

/** Shared incoming-flux limits with simultaneous outgoing-volume credit. */
export function createGeometricLowFluxLimiterWGSL(layout: SparseGeometricVolumeLayout): string {
  return /* wgsl */ `
// Reuse the retired pressure-extension control allocation. All counters are
// u32 except words 7/8, which store positive f32 diagnostic maxima.
const GL_CONTROL:u32=${layout.airControlBaseWords}u;
const GL_MOVING_MAX_DUAL_PASSES:u32=1024u;
fn glLoad(word:u32)->u32{return bitcast<u32>(atomicLoad(&conditioning[GL_CONTROL+word]));}
fn glStore(word:u32,value:u32){atomicStore(&conditioning[GL_CONTROL+word],bitcast<i32>(value));}
fn glRunning()->bool{return gvMicroActive()&&glLoad(23u)==1u&&glLoad(3u)!=0u;}
fn glPublishIndirect(){
  glStore(0u,select(0u,acceptedTemplateCellWorkgroups(),glRunning()));
  glStore(1u,1u);glStore(2u,1u);
  let commitReady=gvMicroCommitReady();
  glStore(16u,select(0u,(gvLoad(0u)+63u)/64u,commitReady));
  glStore(17u,1u);glStore(18u,1u);
  glStore(19u,select(0u,acceptedTemplateCellWorkgroups(),commitReady));
  glStore(20u,1u);glStore(21u,1u);
}
// Static transport uses dense ping-pong factors without copying the proposal
// over the current bank. Pass zero has the same implicit all-one generation
// that the former initialization dispatch wrote. Each continued pass advances
// to the preceding proposal by parity. After convergence, advance has already
// incremented the counter, so apply reads the generation that was audited,
// never the unused proposal from that final pass.
fn glStaticFactorAtPass(cell:u32,iteration:u32)->f32{
  if(iteration==0u){return 1.0;}
  return select(state[GV_PLUS+cell],state[GV_MINUS+cell],(iteration&1u)!=0u);
}
fn glStaticAuditedPass()->u32{
  let completed=glLoad(4u);
  if(glLoad(23u)==2u){
    if(completed==0u){return 0u;}
    return completed-1u;
  }
  return completed;
}
fn glStoreStaticProposal(cell:u32,value:f32,iteration:u32){
  if((iteration&1u)==0u){state[GV_MINUS+cell]=value;}
  else{state[GV_PLUS+cell]=value;}
}
fn glReceiverFactorAtPass(face:u32,flux:f32,iteration:u32)->f32{
  let cells=gvCells(face);let receiver=select(cells.x,cells.y,flux>=0.0);
  if(receiver==INVALID){return 1.0;}
  return glStaticFactorAtPass(receiver,iteration);
}
fn glReceiverFactor(face:u32,flux:f32)->f32{
  let cells=gvCells(face);let receiver=select(cells.x,cells.y,flux>=0.0);
  if(receiver==INVALID){return 1.0;}
  if(geometricSolidMotionActive()){return state[GV_PLUS+receiver];}
  return glStaticFactorAtPass(receiver,glStaticAuditedPass());
}
// Capacity-constrained provisional face flux. A dual potential per cell
// modifies only a physical directed sweep, never the stored liquid amount.
// Both endpoints call this identical function against one frozen dual bank.
fn glMovingFaceFlux(face:u32)->f32{
  let sweep=state[GV_FLUX+4u*face+3u];let weight=abs(sweep);
  if(weight==0.0){return 0.0;}
  let cells=gvCells(face);
  let donor=select(cells.y,cells.x,sweep>=0.0);
  let receiver=select(cells.x,cells.y,sweep>=0.0);
  // Exterior air is not a liquid reservoir. Only explicit source accounting
  // may introduce liquid; an inward bulk air sweep has zero liquid budget.
  if(donor==INVALID){return 0.0;}
  var donorDual=0.0;var receiverDual=0.0;
  if(donor!=INVALID){donorDual=state[GV_PLUS+donor];}
  if(receiver!=INVALID){receiverDual=state[GV_PLUS+receiver];}
  let original=state[GV_FLUX+4u*face];
  let base=select(-original,original,sweep>=0.0);
  let magnitude=clamp(base+weight*(donorDual-receiverDual),0.0,weight);
  return select(-magnitude,magnitude,sweep>=0.0);
}
fn glMovingVolumeReady(cell:u32,volume:f32,capacity:f32)->bool{
  if(capacity!=0.0){return gvVolumeValid(volume,capacity);}
  // This tolerance authorizes only the following shared-face rounding
  // allocation. Final zero-capacity storage still requires exact zero.
  return abs(volume)<=gvRoundoff(geometricSolidCapacityAt(cell,0.0));
}
@compute @workgroup_size(1)
fn beginGeometricLowFluxLimits(){
  if(!gvMicroActive()||glLoad(23u)!=0u){return;}
  glStore(3u,1u);glStore(4u,0u);glStore(6u,0u);
  glStore(13u,0x7fffffffu);
  glStore(22u,1u);glStore(23u,1u);
  glPublishIndirect();
}
@compute @workgroup_size(64)
fn initializeGeometricLowFluxLimits(@builtin(global_invocation_id)gid:vec3u){
  // Static pass zero observes an implicit all-one factor generation and gathers
  // its invariant incoming amount in update. Only moving-solid FISTA needs an
  // explicit initialized vector before its first global update.
  if(!geometricSolidMotionActive()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==INVALID||!glRunning()||glLoad(22u)==0u){return;}
  // Diagonal dual step tau=1/(2*sum incident |Q|) is conservative for
  // the weighted incidence Laplacian. LOW retains the preceding proximal
  // point while PLUS is the common extrapolated FISTA point.
  state[GV_LOW+cell]=0.0;state[GV_PLUS+cell]=0.0;state[GV_MINUS+cell]=0.0;
}
@compute @workgroup_size(64)
fn updateGeometricLowFluxLimits(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||!glRunning()){return;}
  if(geometricSolidMotionActive()){
    var delta=0.0;var sweepWeight=0.0;
    let faces=gvCellFaceRange(cell);
    for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
      let entry=gvCellFace(adjacency);let face=entry>>1u;let isNegative=(entry&1u)!=0u;
      delta+=geometricFctCellDelta(glMovingFaceFlux(face),isNegative);
      sweepWeight+=abs(state[GV_FLUX+4u*face+3u]);
    }
    let volume=gvMicroStartingVolume(cell)+delta;let capacity=gvReceiverCapacity(cell);
    if(!(abs(volume)<=3.402823466e38)){
      gvFault(5u,cell,volume,capacity,delta);return;
    }
    let ready=glMovingVolumeReady(cell,volume,capacity);
    let denominator=2.0*sweepWeight;let previous=state[GV_PLUS+cell];
    var next=previous;
    if(denominator>0.0){
      let tau=1.0/denominator;
      // Prox of the support function of [0,C]. The interval restriction acts
      // on a dual proposal, NOT on the scalar authority or its actual gather.
      // This form avoids subtracting two large almost-equal tau*C terms.
      next=max(0.0,previous+tau*(volume-capacity))
        +min(0.0,previous+tau*volume);
    }else if(!ready){gvFault(5u,cell,volume,capacity,denominator);return;}
    if(!(abs(next)<=3.402823466e38)){
      gvFault(5u,cell,next,capacity,denominator);return;
    }
    state[GV_MINUS+cell]=next;
    atomicMax(&conditioning[GL_CONTROL+8u],bitcast<i32>(
      max(0.0,max(-volume,volume-capacity))));
    // Global readiness refers to the CURRENT dual generation. If all cells
    // pass, commit leaves that generation unchanged and final face publication
    // evaluates exactly the fluxes just audited. A finite iteration ceiling
    // is only a failure boundary, not proof that a feasible solution exists.
    if(!ready){
      atomicAdd(&conditioning[GL_CONTROL+6u],1);
      atomicMin(&conditioning[GL_CONTROL+13u],i32(cell));
      if(glLoad(4u)+1u>=GL_MOVING_MAX_DUAL_PASSES){
        gvFault(5u,cell,volume,capacity,select(0.0,1.0,next==previous));return;
      }
    }
    return;
  }
  let limiterPass=glLoad(4u);let firstPass=limiterPass==0u;
  var incoming=select(state[GV_LOW+cell],0.0,firstPass);
  var outgoing=0.0;var signedDelta=0.0;
  let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;let isNegative=(entry&1u)!=0u;
    let flux=state[GV_FLUX+4u*face];
    let delta=geometricFctCellDelta(flux,isNegative);
    var factor=1.0;
    if(!firstPass){factor=glReceiverFactorAtPass(face,flux,limiterPass);}
    if(firstPass){incoming+=max(0.0,delta);}
    // Match the later low-state audit's per-face multiply, signed incidence
    // and accumulation order; regrouping incoming-minus-outgoing can differ
    // by an ulp exactly where physical capacity is exhausted.
    signedDelta+=geometricFctCellDelta(flux*factor,isNegative);
    if(delta<0.0){outgoing-=delta*factor;}
  }
  if(firstPass){state[GV_LOW+cell]=incoming;}
  let previous=glStaticFactorAtPass(cell,limiterPass);
  let volume=gvMicroStartingVolume(cell);let capacity=gvReceiverCapacity(cell);
  // Preserve, but never grow, an already accepted upper roundoff excursion.
  // Requiring every full cell to remove its existing roundoff simultaneously
  // can be infeasible on a closed through-flow cycle. Source excess outside
  // the accepted interval receives no such allowance.
  let allowedVolume=select(capacity,max(capacity,volume),gvVolumeValid(volume,capacity));
  let candidateVolume=volume+signedDelta;
  let defect=max(0.0,max(-candidateVolume,candidateVolume-capacity));
  atomicMax(&conditioning[GL_CONTROL+8u],bitcast<i32>(defect));
  var next=previous;
  // The donor CFL bounds gross upwind outflow by half the donor amount.
  // Only receiver capacity can be violated. Outgoing credit uses the SAME
  // previous receiver factors as every other cell in this synchronized pass.
  // A valid incompressible full-cell through-flow therefore keeps factor one.
  if(candidateVolume>allowedVolume&&incoming>0.0){
    next=min(previous,max(0.0,allowedVolume-volume+outgoing)/incoming);
    // Use the actual signed face gather to correct regrouping roundoff in
    // allowedVolume-V+outgoing. That expression can round to previous while
    // the actual low state exceeds its starting bound. Only this shared
    // receiver factor changes; conserved cell amounts remain untouched.
    next=min(next,max(0.0,previous-(candidateVolume-allowedVolume)/incoming));
    if(next==previous&&previous>0.0){
      next=bitcast<f32>(bitcast<u32>(previous)-1u);
    }
  }
  glStoreStaticProposal(cell,next,limiterPass);
  // Readiness certifies the CURRENT shared factor generation. When all cells
  // pass, commit preserves it, so face publication uses exactly this audit.
  // Proposed changes within the existing roundoff interval are unnecessary;
  // an unchanged factor with an invalid gather never counts as converged.
  if(!gvVolumeValid(candidateVolume,capacity)){
    atomicAdd(&conditioning[GL_CONTROL+6u],1);
    atomicMin(&conditioning[GL_CONTROL+13u],i32(cell));
  }
}
@compute @workgroup_size(64)
fn commitGeometricLowFluxLimits(@builtin(global_invocation_id)gid:vec3u){
  // Static factors advance by dense bank parity in the singleton control pass;
  // no accepted-cell copy is required. Moving-solid FISTA retains its explicit
  // proximal/extrapolated generation commit.
  if(!geometricSolidMotionActive()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||!glRunning()){return;}
  if(glLoad(6u)==0u){return;}
  let x=state[GV_MINUS+cell];let prior=state[GV_LOW+cell];let k=f32(glLoad(4u));
  state[GV_LOW+cell]=x;state[GV_PLUS+cell]=x+(k/(k+3.0))*(x-prior);
}
@compute @workgroup_size(1)
fn advanceGeometricLowFluxLimits(){
  if(glRunning()){
    let passes=glLoad(4u)+1u;glStore(4u,passes);glStore(5u,glLoad(5u)+1u);
    atomicMax(&conditioning[GL_CONTROL+14u],i32(passes));
    let maximumPasses=select(1024u,GL_MOVING_MAX_DUAL_PASSES,geometricSolidMotionActive());
    let invalid=glLoad(6u);
    let continueIteration=invalid!=0u&&passes<maximumPasses;
    if(invalid!=0u&&!continueIteration){
      gvFault(5u,glLoad(13u),f32(passes),f32(invalid),-4.0);
    }
    if(continueIteration){glStore(13u,0x7fffffffu);}
    glStore(3u,select(0u,1u,continueIteration));
    glStore(23u,select(2u,1u,continueIteration));
  }
  glStore(22u,0u);glStore(6u,0u);glPublishIndirect();
}
// Closing cells store a parent-face identifier in the otherwise dead dual
// proposal word. FCT returns zero endpoint limits for them and preserves it.
const GL_NO_CLOSING_PARENT:u32=0x7f7fffffu; // finite f32 sentinel, never NaN
fn glClosingParent(cell:u32)->u32{
  let encoded=bitcast<u32>(state[GV_MINUS+cell]);
  return select(encoded,INVALID,encoded==GL_NO_CLOSING_PARENT);
}
fn glClosingOtherAmount(cell:u32,parent:u32,finalFlux:bool)->f32{
  var mainDelta=0.0;var roundingDelta=0.0;
  let faces=gvCellFaceRange(cell);
  for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
    let entry=gvCellFace(adjacency);let face=entry>>1u;let negative=(entry&1u)!=0u;
    if(face==parent){continue;}
    mainDelta+=geometricFctCellDelta(state[GV_FLUX+4u*face+select(0u,2u,finalFlux)],negative);
    roundingDelta+=geometricFctCellDelta(state[GV_FLUX_ROUNDOFF+face],negative);
  }
  return (gvMicroStartingVolume(cell)+mainDelta)+roundingDelta;
}
fn glClosingOrderedAmount(cell:u32,finalFlux:bool)->f32{
  let parent=glClosingParent(cell);
  let base=glClosingOtherAmount(cell,parent,finalFlux);
  if(parent==INVALID){return base;}
  let negative=gvCells(parent).x==cell;
  return (base+geometricFctCellDelta(
    state[GV_FLUX+4u*parent+select(0u,2u,finalFlux)],negative))
    +geometricFctCellDelta(state[GV_FLUX_ROUNDOFF+parent],negative);
}
fn glClosingCellBefore(a:u32,b:u32)->bool{
  if(b==INVALID){return true;}
  let x=cellCenter(a);let y=cellCenter(b);
  if(x.x!=y.x){return x.x<y.x;}if(x.y!=y.y){return x.y<y.y;}
  return x.z<y.z;
}
fn glClosingFaceBefore(a:u32,b:u32)->bool{
  if(b==INVALID){return true;}
  let ar=gvRow(a);let br=gvRow(b);
  if(rowAxis(ar)!=rowAxis(br)){return rowAxis(ar)<rowAxis(br);}
  let x=rowCenter(ar);let y=rowCenter(br);
  if(x.x!=y.x){return x.x<y.x;}if(x.y!=y.y){return x.y<y.y;}
  if(x.z!=y.z){return x.z<y.z;}
  if(gvArea(a)!=gvArea(b)){return gvArea(a)<gvArea(b);}
  let ac=gvCells(a);let bc=gvCells(b);
  if(ac.x!=bc.x){
    if(ac.x==INVALID){return true;}if(bc.x==INVALID){return false;}
    return glClosingCellBefore(ac.x,bc.x);
  }
  if(ac.y!=bc.y){
    if(ac.y==INVALID){return true;}if(bc.y==INVALID){return false;}
    return glClosingCellBefore(ac.y,bc.y);
  }
  return false;
}
fn glClosingFaceBudget(face:u32)->vec3f{
  let sweep=state[GV_FLUX+4u*face+3u];let cells=gvCells(face);
  let donor=select(cells.y,cells.x,sweep>=0.0);
  let maximum=select(abs(sweep),0.0,donor==INVALID);
  // Slot2 is frozen until FCT. Component discovery must not read the MAIN
  // word concurrently changed by the one elected leader of that component.
  let main=state[GV_FLUX+4u*face+2u];
  let magnitude=select(-main,main,sweep>=0.0);
  return vec3f(magnitude,maximum-magnitude,maximum);
}
@compute @workgroup_size(64)
fn initializeGeometricClosingComponents(@builtin(global_invocation_id)gid:vec3u){
  if(!gvMicroCommitReady()||!geometricSolidMotionActive()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell!=INVALID&&gvReceiverCapacity(cell)==0.0){
    state[GV_MINUS+cell]=bitcast<f32>(GL_NO_CLOSING_PARENT);
  }
}
@compute @workgroup_size(64)
fn allocateGeometricClosingResidual(@builtin(global_invocation_id)gid:vec3u){
  if(!gvMicroCommitReady()||!geometricSolidMotionActive()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==INVALID||gvReceiverCapacity(cell)!=0.0||glLoad(9u)==0u){return;}
  // Explicit bounded component workspace. Oversized closing components fault
  // before any face write; no queue entry or liquid amount is truncated.
  var members:array<u32,128>;var parents:array<u32,128>;
  var depths:array<u32,128>;var processed:array<u32,128>;
  members[0]=cell;var count=1u;var leader=cell;
  for(var cursor=0u;cursor<count;cursor+=1u){
    let current=members[cursor];let faces=gvCellFaceRange(current);
    for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
      let entry=gvCellFace(adjacency);let face=entry>>1u;
      let other=gvOtherCell(face,(entry&1u)!=0u);
      if(other==INVALID||gvReceiverCapacity(other)!=0.0){continue;}
      let budget=glClosingFaceBudget(face);
      // A flexible edge supports either sign of accumulated rounding. Directed
      // saturated chains require a residual-network extension, not permission
      // to exceed their swept-volume budget.
      if(min(budget.x,budget.y)<=0.0){continue;}
      var found=false;
      for(var index=0u;index<count;index+=1u){found=found||members[index]==other;}
      if(found){continue;}
      if(count==128u){gvFault(5u,cell,128.0,0.0,-128.0);return;}
      members[count]=other;count+=1u;
      if(glClosingCellBefore(other,leader)){leader=other;}
    }
  }
  if(cell!=leader){return;}
  var componentBudget=0.0;var totalResidual=0.0;var anyResidual=false;
  for(var index=0u;index<count;index+=1u){
    let member=members[index];
    let residual=glClosingOtherAmount(member,INVALID,true);
    let tolerance=gvRoundoff(geometricSolidCapacityAt(member,0.0));
    if(!(abs(residual)<=tolerance)){gvFault(5u,member,residual,0.0,tolerance);return;}
    componentBudget+=tolerance;totalResidual+=residual;anyResidual=anyResidual||residual!=0.0;
    parents[index]=INVALID;depths[index]=INVALID;processed[index]=0u;
  }
  if(!anyResidual){return;}
  var rootIndex=INVALID;var rootFace=INVALID;var rootBudget=-1.0;
  for(var index=0u;index<count;index+=1u){
    let member=members[index];let faces=gvCellFaceRange(member);
    for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
      let entry=gvCellFace(adjacency);let face=entry>>1u;let negative=(entry&1u)!=0u;
      let other=gvOtherCell(face,negative);
      if(other!=INVALID&&(gvReceiverCapacity(other)<=0.0||state[GV_COVERAGE+other]!=0.0)){continue;}
      let budget=glClosingFaceBudget(face);
      let signedCorrection=select(-totalResidual,totalResidual,negative);
      let correction=select(-signedCorrection,signedCorrection,state[GV_FLUX+4u*face+3u]>=0.0);
      var available=select(budget.x,budget.y,correction>=0.0);
      if(correction==0.0){available=min(budget.x,budget.y);}
      if(available<componentBudget){continue;}
      var preferred=available>rootBudget;
      if(available==rootBudget){
        if(rootIndex==INVALID){preferred=true;}
        else{preferred=glClosingCellBefore(member,members[rootIndex])
          ||(member==members[rootIndex]&&glClosingFaceBefore(face,rootFace));}
      }
      if(preferred){rootIndex=index;rootFace=face;rootBudget=available;}
    }
  }
  if(rootIndex==INVALID){gvFault(5u,cell,totalResidual,0.0,-1.0);return;}
  parents[rootIndex]=rootFace;depths[rootIndex]=0u;
  // Grow a deterministic tree using physical cell/face order for ties, never
  // atomically assigned subface ids. Every edge reserves the component's
  // audited rounding budget in both directions.
  for(var attached=1u;attached<count;attached+=1u){
    var chosen=INVALID;var chosenFace=INVALID;var chosenDepth=INVALID;var best=-1.0;
    for(var index=0u;index<count;index+=1u){
      if(depths[index]!=INVALID){continue;}
      let member=members[index];let faces=gvCellFaceRange(member);
      for(var adjacency=faces.x;adjacency<faces.y;adjacency+=1u){
        let entry=gvCellFace(adjacency);let face=entry>>1u;
        let other=gvOtherCell(face,(entry&1u)!=0u);
        var parentIndex=INVALID;
        for(var candidate=0u;candidate<count;candidate+=1u){
          if(members[candidate]==other&&depths[candidate]!=INVALID){parentIndex=candidate;break;}
        }
        if(parentIndex==INVALID){continue;}
        let budget=glClosingFaceBudget(face);let available=min(budget.x,budget.y);
        if(available<componentBudget){continue;}
        var preferred=available>best;
        if(available==best){
          if(chosen==INVALID){preferred=true;}
          else{preferred=glClosingCellBefore(member,members[chosen])
            ||(member==members[chosen]&&glClosingFaceBefore(face,chosenFace));}
        }
        if(preferred){chosen=index;chosenFace=face;chosenDepth=depths[parentIndex]+1u;best=available;}
      }
    }
    if(chosen==INVALID){gvFault(5u,cell,totalResidual,0.0,-2.0);return;}
    parents[chosen]=chosenFace;depths[chosen]=chosenDepth;
  }
  for(var index=0u;index<count;index+=1u){state[GV_MINUS+members[index]]=bitcast<f32>(parents[index]);}
  for(var completed=0u;completed<count;completed+=1u){
    var chosen=INVALID;
    for(var index=0u;index<count;index+=1u){
      if(processed[index]!=0u){continue;}
      if(chosen==INVALID){chosen=index;}
      else if(depths[index]>depths[chosen]
        ||(depths[index]==depths[chosen]&&glClosingCellBefore(members[index],members[chosen]))){chosen=index;}
    }
    let member=members[chosen];let face=parents[chosen];
    let negative=gvCells(face).x==member;
    let base=glClosingOtherAmount(member,face,false);
    let replacement=select(-base,base,negative);
    let original=state[GV_FLUX+4u*face+2u];let sweep=state[GV_FLUX+4u*face+3u];
    let magnitude=select(-replacement,replacement,sweep>=0.0);
    let maximum=glClosingFaceBudget(face).z;
    if(!(magnitude>=0.0&&magnitude<=maximum&&abs(replacement-original)<=componentBudget)){
      gvFault(5u,member,replacement,maximum,componentBudget);return;
    }
    state[GV_FLUX+4u*face]=replacement;state[GV_FLUX_ROUNDOFF+face]=0.0;
    let remaining=glClosingOrderedAmount(member,false);
    if(remaining!=0.0){gvFault(5u,member,remaining,0.0,-3.0);return;}
    atomicAdd(&conditioning[GL_CONTROL+11u],1);
    atomicMax(&conditioning[GL_CONTROL+12u],bitcast<i32>(abs(replacement-original)));
    processed[chosen]=1u;
  }
  // Positive-capacity receivers still pass the unchanged ordinary low/final
  // bounds checks. No cell's volume authority is written by this pass.
}
@compute @workgroup_size(64)
fn applyGeometricLowFluxFactors(@builtin(global_invocation_id)gid:vec3u){
  let face=gid.x;if(!gvMicroCommitReady()||face>=gvLoad(0u)){return;}
  let original=state[GV_FLUX+4u*face];let factor=glReceiverFactor(face,original);
  var limited=original*factor;
  if(geometricSolidMotionActive()){
    limited=glMovingFaceFlux(face);
  }
  state[GV_FLUX+4u*face]=limited;state[GV_FLUX+4u*face+2u]=limited;
  if(limited!=original&&!geometricSolidMotionActive()){
    atomicMax(&conditioning[GL_CONTROL+7u],bitcast<i32>(1.0-factor));
    atomicAdd(&conditioning[GL_CONTROL+15u],1);
  }
  // High geometric flux remains frozen. The existing FCT cell pass audits
  // the actual limited low state, then bounds high-minus-low corrections.
  // Failure to reach an admissible fixed point never writes volume authority.
}
`;
}
