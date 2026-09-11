import type { SparseGeometricVolumeLayout } from "./resident-volume.wgsl";

/** Secondary velocity extension preserves every primary liquid-adjacent row. */
export function createGeometricAirProjectionWGSL(layout: SparseGeometricVolumeLayout): string {
  return /* wgsl */ `
const GA_DIAGONAL:u32=${layout.airDiagonal}u;
const GA_CONTROL:u32=${layout.airControlBaseWords}u;
const GA_COMPONENTS:u32=${layout.airComponentBaseWords}u;
fn gaGet(word:u32)->f32{return bitcast<f32>(atomicLoad(&conditioning[GA_CONTROL+word]));}
fn gaSet(word:u32,value:f32){atomicStore(&conditioning[GA_CONTROL+word],bitcast<i32>(value));}
fn gaRunning()->bool{return gaGet(0u)>0.5&&gaGet(6u)==0.0;}
fn gaPrimary()->bool{return gaGet(11u)>0.5;}
fn gaCell(cell:u32)->bool{
  return cell!=INVALID&&cellActive(cell)&&cellOpenVolume(cell)>0.0
    &&select(!pcmCellContains(cell),pcmCellContains(cell),gaPrimary());
}
fn gaComponentRoot(cell:u32)->u32{
  var root=cell;
  for(var depth=0u;depth<64u;depth+=1u){
    let parent=bitcast<u32>(atomicLoad(&conditioning[GA_COMPONENTS+4u*root]));
    if(parent==root||parent==INVALID){return parent;}root=parent;
  }
  return INVALID;
}
fn gaClosedComponent(root:u32)->bool{
  return root!=INVALID&&atomicLoad(&conditioning[GA_COMPONENTS+4u*root+1u])==0;
}
fn gaUnknown(cell:u32)->bool{
  if(!gaCell(cell)){return false;}
  let root=gaComponentRoot(cell);
  return root!=INVALID&&!(root==cell&&gaClosedComponent(root));
}
fn gaFreeRow(row:u32)->bool{
  if(!gvAcceptedPhysicalRow(row)||rowOpenFraction(row)<=0.0){return false;}
  if(gaPrimary()){return pcmRowContains(row)&&state[p.stateOffsets3.x+row]>0.0;}
  let terms=rowTermRange(row);
  for(var at=terms.x;at<terms.y;at+=1u){if(pcmCellContains(termCell(at))){return false;}}
  return terms.y>terms.x;
}
fn gaCorrectionScale(row:u32)->f32{
  let theta=select(1.0,state[p.stateOffsets3.x+row],gaPrimary());
  return rowOpenFraction(row)/theta;
}
fn gaWeight(row:u32)->f32{return rowStaticDualWeight(row)*gaCorrectionScale(row);}
fn gaGradient(row:u32,offset:u32)->f32{
  let terms=rowTermRange(row);var value=0.0;
  if(terms.y-terms.x==2u&&termCoefficient(terms.x)==-termCoefficient(terms.x+1u)){
    let a=termCell(terms.x);let b=termCell(terms.x+1u);var pa=0.0;var pb=0.0;
    if(gaUnknown(a)){pa=state[offset+a];}if(gaUnknown(b)){pb=state[offset+b];}
    return termCoefficient(terms.x+1u)*(pb-pa);
  }
  for(var at=terms.x;at<terms.y;at+=1u){let cell=termCell(at);
    if(gaUnknown(cell)){value+=termCoefficient(at)*state[offset+cell];}}
  return value;
}
fn gaOperator(cell:u32,offset:u32)->f32{
  var sides=vec2f(0.0);
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){let row=incidenceRow(at);
    if(!gaFreeRow(row)){continue;}
    let coefficient=termCoefficient(incidenceTerm(at));
    let value=coefficient*gaWeight(row)*gaGradient(row,offset);
    if(coefficient<0.0){sides.x+=value;}else{sides.y+=value;}}
  return sides.x+sides.y;
}
fn gaDivergence(cell:u32)->f32{
  var sides=vec2f(0.0);
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){let row=incidenceRow(at);
    if(!gvAcceptedPhysicalRow(row)){continue;}
    var velocity=state[destinationFaceVelocity()+row];
    if(hasSolidBoundaries()){velocity-=(1.0-rowOpenFraction(row))*rowSolidVelocity(row);}
    let coefficient=termCoefficient(incidenceTerm(at));
    let value=coefficient*rowStaticDualWeight(row)*velocity;
    if(coefficient<0.0){sides.x+=value;}else{sides.y+=value;}}
  return sides.x+sides.y;
}
fn gaFault(cell:u32,residual:f32,limit:f32){
  gaSet(6u,1.0);gaSet(0u,0.0);
  cm12RecordFailure(6u,cell,bitcast<vec4u>(vec4f(8.0,residual,limit,gaGet(7u))));
}
fn gaComponentFault(cell:u32,value:f32,expected:f32){
  gaSet(6u,1.0);gaSet(0u,0.0);
  cm12RecordFailure(6u,cell,bitcast<vec4u>(vec4f(10.0,value,expected,gaGet(11u))));
}

@compute @workgroup_size(64)
fn initializeGeometricProjectionComponents(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  let at=GA_COMPONENTS+4u*cell;
  atomicStore(&conditioning[at],bitcast<i32>(select(INVALID,cell,gaCell(cell))));
  atomicStore(&conditioning[at+1u],0);atomicStore(&conditioning[at+2u],0);
  atomicStore(&conditioning[at+3u],0);
}
@compute @workgroup_size(64)
fn connectGeometricProjectionComponents(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID||!gaFreeRow(row)){return;}
  let range=rowTermRange(row);var root=INVALID;
  for(var at=range.x;at<range.y;at+=1u){let cell=termCell(at);
    if(gaCell(cell)){root=min(root,gaComponentRoot(cell));}}
  if(root==INVALID){return;}
  for(var at=range.x;at<range.y;at+=1u){let cell=termCell(at);
    if(gaCell(cell)){let previousRoot=gaComponentRoot(cell);
      if(previousRoot!=INVALID){atomicMin(&conditioning[GA_COMPONENTS+4u*previousRoot],i32(root));}
      atomicMin(&conditioning[GA_COMPONENTS+4u*cell],i32(root));}}
}
@compute @workgroup_size(64)
fn compressGeometricProjectionComponents(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(!gaCell(cell)){return;}
  let root=gaComponentRoot(cell);
  if(root==INVALID){gaComponentFault(cell,-1.0,64.0);return;}
  atomicMin(&conditioning[GA_COMPONENTS+4u*cell],i32(root));
}
@compute @workgroup_size(64)
fn finalizeGeometricProjectionComponents(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID||!gaFreeRow(row)){return;}
  let range=rowTermRange(row);var root=INVALID;var coefficientSum=0.0;var coefficientScale=0.0;
  for(var at=range.x;at<range.y;at+=1u){let cell=termCell(at);
    if(!gaCell(cell)){continue;}let other=gaComponentRoot(cell);
    if(other==INVALID||(root!=INVALID&&root!=other)){
      gaComponentFault(row,f32(other),f32(root));return;}
    root=other;let coefficient=termCoefficient(at);
    coefficientSum+=coefficient;coefficientScale+=abs(coefficient);
  }
  if(root!=INVALID&&abs(coefficientSum)>9.5367431640625e-7*coefficientScale){
    // A selected component touching an excluded pressure/air endpoint or an
    // open exterior has a Dirichlet anchor; its RHS must not be projected.
    atomicOr(&conditioning[GA_COMPONENTS+4u*root+1u],1);
  }
}
fn gaAtomicAddFloat(word:u32,addition:f32){
  var previous=atomicLoad(&conditioning[word]);
  for(var attempt=0u;attempt<2048u;attempt+=1u){
    let next=bitcast<i32>(bitcast<f32>(previous)+addition);
    let result=atomicCompareExchangeWeak(&conditioning[word],previous,next);
    if(result.exchanged){return;}previous=result.old_value;
  }
  gaComponentFault(word,f32(addition),2048.0);
}
var<workgroup>gaComponentLabels:array<u32,64>;
var<workgroup>gaComponentRhs:array<f32,64>;
var<workgroup>gaComponentCapacity:array<f32,64>;
@compute @workgroup_size(64)
fn gatherGeometricProjectionCompatibility(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);let lane=lid.x;
  var root=INVALID;var rhs=0.0;var capacity=0.0;
  if(gaCell(cell)){root=gaComponentRoot(cell);rhs=state[GV_LOW+cell];capacity=cellOpenVolume(cell);}
  gaComponentLabels[lane]=root;gaComponentRhs[lane]=rhs;gaComponentCapacity[lane]=capacity;
  workgroupBarrier();var first=root!=INVALID;
  for(var prior=0u;prior<lane;prior+=1u){if(gaComponentLabels[prior]==root){first=false;}}
  if(first){
    var rhsSum=0.0;var rhsCorrection=0.0;var capacitySum=0.0;var capacityCorrection=0.0;
    for(var other=lane;other<64u;other+=1u){if(gaComponentLabels[other]!=root){continue;}
      let rhsValue=gaComponentRhs[other]-rhsCorrection;let rhsNext=rhsSum+rhsValue;
      rhsCorrection=(rhsNext-rhsSum)-rhsValue;rhsSum=rhsNext;
      let capacityValue=gaComponentCapacity[other]-capacityCorrection;
      let capacityNext=capacitySum+capacityValue;
      capacityCorrection=(capacityNext-capacitySum)-capacityValue;capacitySum=capacityNext;
    }
    gaAtomicAddFloat(GA_COMPONENTS+4u*root+2u,rhsSum);
    gaAtomicAddFloat(GA_COMPONENTS+4u*root+3u,capacitySum);
  }
}
@compute @workgroup_size(64)
fn applyGeometricProjectionCompatibility(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);var gamma=0.0;var rhs2=0.0;var normalized=0.0;
  if(gaCell(cell)){
    let root=gaComponentRoot(cell);var rhs=state[GV_LOW+cell];
    if(gaClosedComponent(root)){
      let at=GA_COMPONENTS+4u*root;
      let rhsSum=bitcast<f32>(atomicLoad(&conditioning[at+2u]));
      let capacitySum=bitcast<f32>(atomicLoad(&conditioning[at+3u]));
      let defect=abs(rhsSum)*p.frame.x;let limit=GA_NORMALIZED_TARGET*capacitySum;
      if(!(capacitySum>0.0)||!(defect<=limit)){gaComponentFault(root,defect,limit);}
      else{rhs-=rhsSum/capacitySum*cellOpenVolume(cell);}
    }
    // One fixed zero potential removes each closed component's constant
    // nullspace. Compatibility projection changes only the solve RHS; the
    // original physical divergence is still audited after publication.
    let diagonal=state[GA_DIAGONAL+cell];var z=0.0;
    if(gaUnknown(cell)&&diagonal>0.0){z=rhs/diagonal;}
    state[GV_LOW+cell]=rhs;state[GV_PLUS+cell]=z;
    gamma=rhs*z;rhs2=rhs*rhs;normalized=gaNormalizedResidual(cell,rhs);
  }
  gaReduceSumMax(lid.x,wid.x,gamma,normalized,rhs2);
}
// Residual is an integrated face-volume rate. Scaling by dt/open volume
// matches the final per-cell divergence audit rather than a global L2 norm.
const GA_NORMALIZED_TARGET:f32=2.0*1.1920928955078125e-7;
fn gaNormalizedResidual(cell:u32,residual:f32)->f32{
  let value=abs(residual)*p.frame.x/cellOpenVolume(cell);
  return select(3.402823466e38,value,value>=0.0&&value<=3.402823466e38);
}
var<workgroup> gaResidualSquared:array<f32,64>;
fn gaReduceSumMax(lane:u32,group:u32,sum:f32,maximum:f32,squared:f32){
  reduceA[lane]=sum;reduceB[lane]=maximum;gaResidualSquared[lane]=squared;
  workgroupBarrier();var width=32u;
  loop{if(lane<width){reduceA[lane]+=reduceA[lane+width];
    reduceB[lane]=max(reduceB[lane],reduceB[lane+width]);
    gaResidualSquared[lane]+=gaResidualSquared[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u){partials[group]=vec4f(reduceA[0],reduceB[0],gaResidualSquared[0],0.0);}
}
fn gaReduce()->vec3f{
  var result=vec3f(0.0);
  for(var group=0u;group<acceptedTemplateCellWorkgroups();group+=1u){
    let partial=partials[group];result.x+=partial.x;
    result.y=max(result.y,partial.y);result.z+=partial.z;
  }
  return result;
}
@compute @workgroup_size(1)
fn beginGeometricPrimaryRefinement(){gaSet(11u,1.0);}
@compute @workgroup_size(64)
fn initializeGeometricAirProjection(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);var gamma=0.0;var rhs2=0.0;var normalized=0.0;
  if(cell!=INVALID){
    atomicStore(&conditioning[GA_COMPONENTS+4u*cell+2u],0);
    atomicStore(&conditioning[GA_COMPONENTS+4u*cell+3u],0);
    state[GV_CURRENT+cell]=0.0;state[GV_LOW+cell]=0.0;
    state[GV_PLUS+cell]=0.0;state[GV_MINUS+cell]=0.0;state[GA_DIAGONAL+cell]=0.0;
    if(gaCell(cell)){
      var diagonal=0.0;
      for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){let row=incidenceRow(at);
        if(gaFreeRow(row)){let coefficient=termCoefficient(incidenceTerm(at));
          diagonal+=gaWeight(row)*coefficient*coefficient;}}
      let rhs=gaDivergence(cell);var z=0.0;
      if(diagonal>0.0){z=rhs/diagonal;}
      state[GA_DIAGONAL+cell]=diagonal;state[GV_LOW+cell]=rhs;state[GV_PLUS+cell]=z;
      gamma=rhs*z;rhs2=rhs*rhs;normalized=gaNormalizedResidual(cell,rhs);
    }
  }
  gaReduceSumMax(lid.x,wid.x,gamma,normalized,rhs2);
}
@compute @workgroup_size(1)
fn beginGeometricAirSolve(){
  // Host clears the complete air control once per frame. Refinement preserves
  // a prior fault and cumulative work while resetting this correction solve.
  if(gaGet(6u)!=0.0){return;}
  for(var word=0u;word<9u;word+=1u){gaSet(word,0.0);}
  gaSet(10u,gaGet(10u)+1.0);
  let pair=gaReduce();gaSet(1u,pair.x);gaSet(4u,GA_NORMALIZED_TARGET);
  gaSet(5u,pair.z);gaSet(8u,pair.z);
  gaSet(12u,pair.y);gaSet(13u,pair.y);gaSet(14u,pair.y);
  let skip=pair.y<=GA_NORMALIZED_TARGET;
  gaSet(15u,select(0.0,1.0,skip));gaSet(0u,select(1.0,0.0,skip));
}
@compute @workgroup_size(64)
fn applyGeometricAirOperator(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);var curvature=0.0;
  if(gaRunning()&&gaCell(cell)){let value=gaOperator(cell,GV_PLUS);
    state[GV_MINUS+cell]=value;
    if(gaUnknown(cell)){curvature=state[GV_PLUS+cell]*value;}}
  reducePair(lid.x,wid.x,curvature,0.0);
}
@compute @workgroup_size(1)
fn advanceGeometricAirAlpha(){
  if(!gaRunning()){return;}let curvature=gaReduce().x;
  if(!(curvature>0.0)||!(curvature<=3.402823466e38)){gaFault(INVALID,curvature,0.0);return;}
  gaSet(2u,gaGet(1u)/curvature);
}
@compute @workgroup_size(64)
fn updateGeometricAirResidual(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);var gamma=0.0;var residual2=0.0;var normalized=0.0;
  if(gaRunning()&&gaCell(cell)){
    let alpha=gaGet(2u);
    if(gaUnknown(cell)){state[GV_CURRENT+cell]+=alpha*state[GV_PLUS+cell];}
    let residual=state[GV_LOW+cell]-alpha*state[GV_MINUS+cell];state[GV_LOW+cell]=residual;
    let diagonal=state[GA_DIAGONAL+cell];
    if(gaUnknown(cell)&&diagonal>0.0){gamma=residual*residual/diagonal;}residual2=residual*residual;
    normalized=gaNormalizedResidual(cell,residual);
  }
  gaReduceSumMax(lid.x,wid.x,gamma,normalized,residual2);
}
@compute @workgroup_size(1)
fn advanceGeometricAirBeta(){
  if(!gaRunning()){return;}let pair=gaReduce();
  gaSet(3u,pair.x/gaGet(1u));gaSet(1u,pair.x);gaSet(5u,pair.z);
  gaSet(13u,pair.y);gaSet(7u,gaGet(7u)+1.0);
  gaSet(9u,gaGet(9u)+1.0);
  if(pair.y<=gaGet(4u)){gaSet(0u,0.0);}
}
@compute @workgroup_size(64)
fn updateGeometricAirDirection(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(!gaRunning()||!gaUnknown(cell)){return;}
  let diagonal=state[GA_DIAGONAL+cell];var z=0.0;
  if(diagonal>0.0){z=state[GV_LOW+cell]/diagonal;}
  state[GV_PLUS+cell]=z+gaGet(3u)*state[GV_PLUS+cell];
}
@compute @workgroup_size(64)
fn measureGeometricAirTrueResidual(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);var residual2=0.0;var normalized=0.0;
  if(gaCell(cell)){let residual=gaDivergence(cell)-gaOperator(cell,GV_CURRENT);
    residual2=residual*residual;normalized=gaNormalizedResidual(cell,residual);}
  gaReduceSumMax(lid.x,wid.x,residual2,normalized,0.0);
}
@compute @workgroup_size(1)
fn finishGeometricAirSolve(){
  let pair=gaReduce();gaSet(5u,pair.x);gaSet(14u,pair.y);gaSet(0u,0.0);
  // Final authority is the per-cell, volume-compatible divergence audit after
  // f32 velocity publication, rather than recursive CG residual alone.
}
@compute @workgroup_size(64)
fn projectGeometricAirRows(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row==INVALID||gaGet(6u)!=0.0||gaGet(15u)>0.5||!gaFreeRow(row)){return;}
  state[destinationFaceVelocity()+row]-=gaCorrectionScale(row)*gaGradient(row,GV_CURRENT);
}
@compute @workgroup_size(64)
fn auditGeometricAirDivergence(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(!gaCell(cell)){return;}
  let residual=abs(gaDivergence(cell))*p.frame.x;
  let limit=gvRoundoff(cellOpenVolume(cell));
  if(!(residual<=limit)){gaFault(cell,residual,limit);}
}
`;
}
