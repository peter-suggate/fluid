import { airExtensionLayout } from "./sparse-cm12-air-extension";

/** Uses the accepted CNX adjoint pair, static dual weights, and physical
 * apertures. `partials` is the transport plane, never the pressure reduction
 * binding. The dead conditioning arena supplies atomic component scratch. */
export function createAirExtensionWGSL(cells: number, rows: number): string {
  const l = airExtensionLayout(cells, rows);
  return /* wgsl */ `
const AIR_N:u32=${cells}u;
const AIR_C:u32=${l.cellBase}u;
const AIR_F:u32=${l.rowBase}u;
const AIR_H:u32=${l.header}u;
const AIR_R:u32=${l.reduction}u;
var<workgroup> airReduction:array<vec4f,64>;
var<workgroup> airRunning:u32;
fn airSolveRunning(lane:u32)->bool{
  if(lane==0u){airRunning=u32(partials[AIR_H].z==0.0&&partials[AIR_H].w==0.0);}
  return workgroupUniformLoad(&airRunning)!=0u;
}
fn airFinite(v:f32)->bool{return abs(v)<=3.402823466e38;}
fn airCell(c:u32)->u32{return AIR_C+3u*c;}
fn airActive(c:u32)->bool{return partials[airCell(c)].x>0.0;}
fn airRoot(cell:u32)->u32{
  var root=cell;loop{let next=u32(atomicLoad(&conditioning[root]));
    if(next==root){return root;}root=next;}
}
fn airUnion(a:u32,b:u32){
  var left=a;var right=b;
  loop{left=airRoot(left);right=airRoot(right);if(left==right){return;}
    let lo=min(left,right);let hi=max(left,right);
    let result=atomicCompareExchangeWeak(&conditioning[hi],i32(hi),i32(lo));
    if(result.exchanged){return;}}
}
fn airAtomicAdd(at:u32,value:f32){
  var old=atomicLoad(&conditioning[at]);
  loop{let next=bitcast<i32>(bitcast<f32>(old)+value);
    let result=atomicCompareExchangeWeak(&conditioning[at],old,next);
    if(result.exchanged){return;}old=result.old_value;}
}
fn airReduce(value:vec4f,lane:u32,group:u32){
  airReduction[lane]=value;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){
    if(lane<stride){let a=airReduction[lane];let b=airReduction[lane+stride];
      airReduction[lane]=vec4f(a.xyz+b.xyz,max(a.w,b.w));}workgroupBarrier();}
  if(lane==0u){partials[AIR_R+group]=airReduction[0];}
}
fn airGradient(rowOrdinal:u32,component:u32)->f32{
  let range=cnxRowTermRangeByOrdinalUnchecked(rowOrdinal);var result=0.0;
  for(var t=range.x;t<range.y;t+=1u){let c=cnxRowTermCellUnchecked(t);
    if(airActive(c)){result+=cnxRowTermCoefficientUnchecked(t)*partials[airCell(c)+1u][component];}}
  return result;
}
fn airOperator(cell:u32,component:u32)->f32{
  let range=cnxCellIncidenceRangeUnchecked(cell);var sides:array<vec3f,2>;
  for(var at=range.x;at<range.y;at+=1u){let ordinal=cnxIncidenceRowOrdinalUnchecked(at);
    let row=cnxStableRowUnchecked(ordinal);let face=partials[AIR_F+row];
    if(face.y<=0.0){continue;}let own=cnxIncidenceOwnCoefficientUnchecked(at);
    let axis=cnxRowPackedMetadataByOrdinal(ordinal)&3u;
    sides[select(0u,1u,own<0.0)][axis]+=own*face.z*face.y*airGradient(ordinal,component);}
  let sum=sides[0]+sides[1];return (sum.x+sum.y)+sum.z;
}
@compute @workgroup_size(1) fn airBegin(){
  for(var i=0u;i<4u;i+=1u){partials[AIR_H+i]=vec4f(0.0);}
  atomicStore(&conditioning[4u*AIR_N],0);
}
@compute @workgroup_size(64) fn airClassifyCells(@builtin(global_invocation_id)gid:vec3u){
  let c=cnxAcceptedCellInvocation(gid.x);if(c==INVALID){return;}
  let sample=lsvCellSample(c);let volume=cellOpenVolume(c);
  let width=cellWidths(c);let air=sample.valid&&sample.phi>0.0&&volume>1e-8
    &&cm12ExtendedCellSelected(c);
  let selected=air&&sample.phi<=2.0*max(width.x,max(width.y,width.z));
  partials[airCell(c)]=vec4f(select(0.0,volume,selected),0.0,0.0,0.0);
  partials[airCell(c)+1u]=vec4f(0.0);
  partials[airCell(c)+2u]=vec4f(select(0.0,1.0,air),0.0,0.0,0.0);
  atomicStore(&conditioning[c],i32(c));
  for(var k=1u;k<4u;k+=1u){atomicStore(&conditioning[k*AIR_N+c],0);}
}
// Staggered values remain authoritative during extension. The two value and
// validity lanes ping-pong synchronously; an accepted seed is never averaged.
@compute @workgroup_size(64) fn airSeedFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=cnxAcceptedRowInvocation(gid.x);if(row==INVALID){return;}
  let range=cnxRowTermRangeByOrdinalUnchecked(gid.x);
  var seed=range.y-range.x==1u||rowOpenFraction(row)<=1e-8||sparseCM12InflowFaceCoverage(row)>0.0;
  for(var t=range.x;t<range.y;t+=1u){let c=cnxRowTermCellUnchecked(t);
    let sample=lsvCellSample(c);seed=seed||(sample.valid&&sample.phi<=0.0&&cellOpenVolume(c)>1e-8);}
  let value=state[destinationFaceVelocity()+row];
  partials[AIR_F+row]=vec4f(select(0.0,value,seed),select(0.0,value,seed),vec2f(select(0.0,1.0,seed)));
}
fn airExtendFromCell(cell:u32,row:u32,axis:u32,input:u32,widths:vec3f)->vec2f{
  var sum=vec2f(0.0);if(cellOpenVolume(cell)<=1e-8){return sum;}
  let incidence=cnxCellIncidenceRangeUnchecked(cell);
  for(var at=incidence.x;at<incidence.y;at+=1u){let ordinal=cnxIncidenceRowOrdinalUnchecked(at);
    let other=cnxStableRowUnchecked(ordinal);
    if(other==row||(cnxRowPackedMetadataByOrdinal(ordinal)&3u)!=axis){continue;}
    let data=partials[AIR_F+other];if(data[2u+input]==0.0){continue;}
    let delta=(rowCenter(other)-rowCenter(row))/widths;let distance=dot(delta,delta);
    if(distance<=1e-10||distance>2.26){continue;}
    // Extend physical normal velocity, then reapply the target aperture once.
    let open=rowOpenFraction(other);var velocity=data[input];
    if(!rowSeparatingFromClosedWorld(other)){
      velocity=select(rowSolidVelocity(other),(velocity-(1.0-open)*rowSolidVelocity(other))/max(open,1e-8),open>1e-8);}
    let weight=1.0/max(distance,0.0625);sum+=weight*vec2f(velocity,1.0);
  }
  return sum;
}
fn airExtendFace(row:u32,ordinal:u32,input:u32){
  let output=1u-input;let old=partials[AIR_F+row];
  var value=old[input];var valid=old[2u+input];
  if(valid==0.0){
    let terms=cnxRowTermRangeByOrdinalUnchecked(ordinal);let axis=cnxRowPackedMetadataByOrdinal(ordinal)&3u;
    var widths=vec3f(1.0);for(var t=terms.x;t<terms.y;t+=1u){widths=max(widths,cellWidths(cnxRowTermCellUnchecked(t)));}
    var sum=vec2f(0.0);
    for(var t=terms.x;t<terms.y;t+=1u){let cell=cnxRowTermCellUnchecked(t);
      if(cellOpenVolume(cell)<=1e-8){continue;}
      sum+=airExtendFromCell(cell,row,axis,input,widths);
      // Same-component faces in tangential neighbours are not incidences of
      // this face's cells. Follow open cell adjacencies to reach them; never
      // propagate through a closed aperture or a zero-capacity cell.
      let incidence=cnxCellIncidenceRangeUnchecked(cell);
      for(var at=incidence.x;at<incidence.y;at+=1u){let edge=cnxIncidenceRowOrdinalUnchecked(at);
        let crossing=cnxStableRowUnchecked(edge);if(rowOpenFraction(crossing)<=1e-8){continue;}
        let neighbors=cnxRowTermRangeByOrdinalUnchecked(edge);
        for(var n=neighbors.x;n<neighbors.y;n+=1u){let next=cnxRowTermCellUnchecked(n);
          if(next!=cell){sum+=airExtendFromCell(next,row,axis,input,widths);}}}
    }
    if(sum.y>0.0){value=sum.x/sum.y;valid=1.0;
      if(!rowSeparatingFromClosedWorld(row)){
        let open=rowOpenFraction(row);value=open*value+(1.0-open)*rowSolidVelocity(row);}}
  }
  partials[AIR_F+row][output]=value;partials[AIR_F+row][2u+output]=valid;
}
@compute @workgroup_size(64) fn airExtendFacesA(@builtin(global_invocation_id)gid:vec3u){
  let row=cnxAcceptedRowInvocation(gid.x);if(row!=INVALID){airExtendFace(row,gid.x,0u);}}
@compute @workgroup_size(64) fn airExtendFacesB(@builtin(global_invocation_id)gid:vec3u){
  let row=cnxAcceptedRowInvocation(gid.x);if(row!=INVALID){airExtendFace(row,gid.x,1u);}}
// Only a one-sided row on a physical domain plane is a certified exterior
// air anchor. Missing sparse backing elsewhere remains a fixed flux.
fn airExteriorAnchor(row:u32,ordinal:u32)->bool{
  let range=cnxRowTermRangeByOrdinalUnchecked(ordinal);
  if(range.y-range.x!=1u||rowKind(row)!=3u||rowOpenFraction(row)<=1e-8){return false;}
  let axis=cnxRowPackedMetadataByOrdinal(ordinal)&3u;let position=rowCenter(row)[axis];
  let inward=cnxRowTermCoefficientUnchecked(range.x);
  return (abs(position)<1e-5&&inward>0.0)
    ||(abs(position-f32(p.dimensions[axis]))<1e-5&&inward<0.0);
}
@compute @workgroup_size(64) fn airPrepareRows(@builtin(global_invocation_id)gid:vec3u){
  let row=cnxAcceptedRowInvocation(gid.x);if(row==INVALID){return;}
  let range=cnxRowTermRangeByOrdinalUnchecked(gid.x);let open=rowOpenFraction(row);
  // Exterior open planes can anchor air pressure; unknown sparse edges,
  // prescribed inflows, closed walls and moving solids remain fixed.
  let extended=partials[AIR_F+row];
  var free=(range.y-range.x>1u||airExteriorAnchor(row,gid.x))&&open>1e-8&&sparseCM12InflowFaceCoverage(row)==0.0&&extended.z>0.0;
  var selected=false;
  for(var t=range.x;t<range.y;t+=1u){let c=cnxRowTermCellUnchecked(t);
    free=free&&partials[airCell(c)+2u].x>0.0;selected=selected||airActive(c);}
  let value=extended.x;
  let aperture=select(0.0,open,free&&selected);
  partials[AIR_F+row]=vec4f(value,aperture,rowStaticDualWeight(row),extended.z);
  if(aperture>0.0){atomicAdd(&conditioning[4u*AIR_N],1);}
}
@compute @workgroup_size(64) fn airConnect(@builtin(global_invocation_id)gid:vec3u){
  let row=cnxAcceptedRowInvocation(gid.x);if(row==INVALID||partials[AIR_F+row].y<=0.0){return;}
  let range=cnxRowTermRangeByOrdinalUnchecked(gid.x);var first=INVALID;
  for(var t=range.x;t<range.y;t+=1u){let c=cnxRowTermCellUnchecked(t);if(!airActive(c)){continue;}
    if(first==INVALID){first=c;}else{airUnion(first,c);}}
}
@compute @workgroup_size(64) fn airAssemble(@builtin(global_invocation_id)gid:vec3u){
  let c=cnxAcceptedCellInvocation(gid.x);if(c==INVALID||!airActive(c)){return;}
  let range=cnxCellIncidenceRangeUnchecked(c);var rhs=0.0;var diagonal=0.0;var anchored=false;
  for(var at=range.x;at<range.y;at+=1u){let ordinal=cnxIncidenceRowOrdinalUnchecked(at);
    let row=cnxStableRowUnchecked(ordinal);let face=partials[AIR_F+row];
    let own=cnxIncidenceOwnCoefficientUnchecked(at);rhs+=own*face.z*face.x;
    diagonal+=own*own*face.z*face.y;
    if(face.y>0.0){anchored=anchored||airExteriorAnchor(row,ordinal);let terms=cnxRowTermRangeByOrdinalUnchecked(ordinal);
      for(var t=terms.x;t<terms.y;t+=1u){anchored=anchored||!airActive(cnxRowTermCellUnchecked(t));}}}
  partials[airCell(c)].y=rhs;partials[airCell(c)].z=diagonal;
  partials[airCell(c)+2u].y=rhs;
  let root=airRoot(c);airAtomicAdd(AIR_N+root,rhs);
  airAtomicAdd(2u*AIR_N+root,partials[airCell(c)].x);
  if(anchored){atomicStore(&conditioning[3u*AIR_N+root],1);}
}
@compute @workgroup_size(64) fn airInitialize(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  let c=cnxAcceptedCellInvocation(gid.x);var receipt=vec4f(0.0);
  if(c!=INVALID&&airActive(c)){let at=airCell(c);var data=partials[at];let root=airRoot(c);
    var compatible=0.0;
    if(atomicLoad(&conditioning[3u*AIR_N+root])==0){
      compatible=bitcast<f32>(atomicLoad(&conditioning[AIR_N+root]))
        /max(bitcast<f32>(atomicLoad(&conditioning[2u*AIR_N+root])),1e-20);}
    let rhs=data.y-compatible*data.x;data.w=compatible;data.y=rhs;partials[at]=data;
    let z=select(0.0,rhs/max(data.z,1e-20),data.z>0.0);
    partials[at+1u]=vec4f(0.0,rhs,z,0.0);
    receipt=vec4f(rhs*z,1.0,select(0.0,1.0,data.z<=0.0),abs(partials[at+2u].y/data.x));}
  airReduce(receipt,lane,wid.x);
}
@compute @workgroup_size(1) fn airReduceInitial(){
  var sum=vec4f(0.0);for(var g=0u;g<(cnxAcceptedCellCount()+63u)/64u;g+=1u){
    let v=partials[AIR_R+g];sum=vec4f(sum.xyz+v.xyz,max(sum.w,v.w));}
  partials[AIR_H+1u]=vec4f(sum.x,sum.x,sum.y,sum.z);
  partials[AIR_H+2u]=vec4f(sum.w,0.0,0.0,f32(atomicLoad(&conditioning[4u*AIR_N])));
  partials[AIR_H+3u]=vec4f(sum.x,0.0,0.0,max(sum.x*1e-8,1e-14));
  partials[AIR_H].z=select(0.0,1.0,sum.x<=1e-14);
}
@compute @workgroup_size(64) fn airApply(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  if(!airSolveRunning(lane)){return;}
  let c=cnxAcceptedCellInvocation(gid.x);var dot=0.0;
  if(c!=INVALID&&airActive(c)){let image=airOperator(c,2u);let at=airCell(c)+1u;
    partials[at].w=image;dot=partials[at].z*image;}
  airReduce(vec4f(dot,0.0,0.0,0.0),lane,wid.x);
}
@compute @workgroup_size(1) fn airReduceAlpha(){
  if(partials[AIR_H].z!=0.0||partials[AIR_H].w!=0.0){return;}
  var dot=0.0;for(var g=0u;g<(cnxAcceptedCellCount()+63u)/64u;g+=1u){dot+=partials[AIR_R+g].x;}
  if(!(dot>0.0)||!airFinite(dot)){partials[AIR_H].w=1.0;return;}
  partials[AIR_H+3u].y=partials[AIR_H+3u].x/dot;
}
@compute @workgroup_size(64) fn airUpdate(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  if(!airSolveRunning(lane)){return;}
  let c=cnxAcceptedCellInvocation(gid.x);var rho=0.0;
  if(c!=INVALID&&airActive(c)){let at=airCell(c);var v=partials[at+1u];let alpha=partials[AIR_H+3u].y;
    v.x+=alpha*v.z;v.y-=alpha*v.w;partials[at+1u]=v;
    rho=v.y*v.y/max(partials[at].z,1e-20);}
  airReduce(vec4f(rho,0.0,0.0,0.0),lane,wid.x);
}
@compute @workgroup_size(1) fn airReduceBeta(){
  if(partials[AIR_H].z!=0.0||partials[AIR_H].w!=0.0){return;}
  var rho=0.0;for(var g=0u;g<(cnxAcceptedCellCount()+63u)/64u;g+=1u){rho+=partials[AIR_R+g].x;}
  if(!airFinite(rho)){partials[AIR_H].w=1.0;return;}
  partials[AIR_H].y+=1.0;partials[AIR_H].z=select(0.0,1.0,rho<=partials[AIR_H+3u].w);
  partials[AIR_H+3u].z=rho/max(partials[AIR_H+3u].x,1e-30);partials[AIR_H+3u].x=rho;
}
@compute @workgroup_size(64) fn airDirection(@builtin(global_invocation_id)gid:vec3u){
  if(partials[AIR_H].z!=0.0||partials[AIR_H].w!=0.0){return;}
  let c=cnxAcceptedCellInvocation(gid.x);if(c==INVALID||!airActive(c)){return;}
  let at=airCell(c);partials[at+1u].z=partials[at+1u].y/max(partials[at].z,1e-20)
    +partials[AIR_H+3u].z*partials[at+1u].z;
}
@compute @workgroup_size(64) fn airMeasure(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  let c=cnxAcceptedCellInvocation(gid.x);var receipt=vec4f(0.0);
  if(c!=INVALID&&airActive(c)){let data=partials[airCell(c)];let r=data.y-airOperator(c,0u);
    // Maxima use the last two lanes; this entry owns its reduction explicitly.
    receipt=vec4f(select(0.0,r*r/max(data.z,1e-20),data.z>0.0),0.0,abs(data.w),abs(r/data.x));}
  airReduction[lane]=receipt;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){if(lane<stride){let a=airReduction[lane];let b=airReduction[lane+stride];
    airReduction[lane]=vec4f(a.x+b.x,0.0,max(a.z,b.z),max(a.w,b.w));}workgroupBarrier();}
  if(lane==0u){partials[AIR_R+wid.x]=airReduction[0];}
}
@compute @workgroup_size(1) fn airReduceFinal(){
  var sum=vec4f(0.0);for(var g=0u;g<(cnxAcceptedCellCount()+63u)/64u;g+=1u){let v=partials[AIR_R+g];
    sum=vec4f(sum.x+v.x,0.0,max(sum.z,v.z),max(sum.w,v.w));}
  partials[AIR_H+1u].y=sum.x;
  partials[AIR_H+2u].y=sum.w;partials[AIR_H+2u].z=sum.z;
  partials[AIR_H].z=select(0.0,1.0,sum.x<=partials[AIR_H+3u].w);
  let valid=airFinite(sum.x)&&airFinite(sum.w)&&sum.x<=max(1e-14,1.001*partials[AIR_H+1u].x);
  partials[AIR_H].x=select(0.0,1.0,valid&&partials[AIR_H].w==0.0);
}
@compute @workgroup_size(64) fn airCorrect(@builtin(global_invocation_id)gid:vec3u){
  if(partials[AIR_H].x==0.0){return;}
  let row=cnxAcceptedRowInvocation(gid.x);if(row==INVALID){return;}
  if(partials[AIR_F+row].y>0.0){partials[AIR_F+row].x-=partials[AIR_F+row].y*airGradient(gid.x,0u);}
}

fn airTransportReady()->bool{return p.frame.w>0.5&&partials[AIR_H].x==1.0;}
fn airBoundaryFaceVelocity(row:u32)->f32{
  if(airTransportReady()&&partials[AIR_F+row].w>0.0){return partials[AIR_F+row].x;}
  return state[destinationFaceVelocity()+row];
}
// Independent face-component MLS. A regular stencil has zero first moment,
// reducing exactly to staggered trilinear interpolation. Adaptive and wall
// stencils fit an affine field using each face's physical support width.
fn airSampleVelocity(position:vec3f,spansInput:vec3f,direct:bool)->vec3f{
  let spans=max(spansInput,vec3f(1.0));
  let point=cm12ClampToResidentWorld(position,vec3f(0.0));
  let stencilPoint=cm12ClampToResidentWorld(point,spans);
  let boundaryAxes=abs(stencilPoint-point)>vec3f(1e-6);
  let lower=vec3i(floor(stencilPoint/spans-vec3f(0.5)));
  var owners:array<u32,8>;
  for(var k=0u;k<8u;k+=1u){
    let offset=vec3f(f32(k&1u),f32((k>>1u)&1u),f32((k>>2u)&1u));
    let q=cm12ClampToResidentWorld(spans*(vec3f(lower)+offset+vec3f(0.5)),vec3f(1e-4));
    owners[k]=cm12TransportOwnerAtFine(vec3i(floor(q)),direct).cell;
  }
  // A coarse/fine seam can supply just one normal-component sample on
  // the interface plane. Include the next plane to retain tangential rank.
  var adaptive=false;
  for(var k=0u;k<8u;k+=1u){if(owners[k]!=INVALID){adaptive=adaptive||any(cellWidths(owners[k])!=spans);}}
  var moments:array<mat4x4f,3>;var rhs:array<vec4f,3>;
  var wallWeights=vec3f(0.0);var wallValues=vec3f(0.0);
  for(var k=0u;k<8u;k+=1u){let cell=owners[k];if(cell==INVALID){continue;}
    var duplicate=false;for(var previous=0u;previous<k;previous+=1u){duplicate=duplicate||owners[previous]==cell;}
    if(duplicate){continue;}
    let incidence=cnxCellIncidenceRangeUnchecked(cell);
    for(var at=incidence.x;at<incidence.y;at+=1u){
      let ordinal=cnxIncidenceRowOrdinalUnchecked(at);let row=cnxStableRowUnchecked(ordinal);
      let terms=cnxRowTermRangeByOrdinalUnchecked(ordinal);var seen=false;var widths=vec3f(1.0);
      for(var t=terms.x;t<terms.y;t+=1u){let other=cnxRowTermCellUnchecked(t);
        widths=max(widths,cellWidths(other));
        for(var previous=0u;previous<k;previous+=1u){seen=seen||owners[previous]==other;}}
      if(seen){continue;}
      if(partials[AIR_F+row].w==0.0){continue;}
      let axis=cnxRowPackedMetadataByOrdinal(ordinal)&3u;
      let delta=(rowCenter(row)-point)/spans;
      let support=select(max(widths,spans),max(widths,2.0*spans),boundaryAxes|vec3<bool>(adaptive));
      let tent=max(vec3f(0.0),vec3f(1.0)-abs(rowCenter(row)-point)/support);
      let weight=tent.x*tent.y*tent.z;if(weight<=0.0){continue;}
      let open=rowOpenFraction(row);var value=partials[AIR_F+row].x;
      let separating=rowSeparatingFromClosedWorld(row);
      if(open<=1e-8&&!separating&&abs(rowCenter(row)[axis]-point[axis])<=1e-6){
        wallWeights[axis]+=weight;wallValues[axis]+=weight*rowSolidVelocity(row);}
      if(!separating){
        value=select(rowSolidVelocity(row),(value-(1.0-open)*rowSolidVelocity(row))/max(open,1e-8),open>1e-8);}
      let basis=vec4f(1.0,delta);rhs[axis]+=weight*value*basis;
      for(var column=0u;column<4u;column+=1u){moments[axis][column]+=weight*basis[column]*basis;}
    }
  }
  var result=airFitVelocity(moments,rhs);
  for(var axis=0u;axis<3u;axis+=1u){if(wallWeights[axis]>0.0){result[axis]=wallValues[axis]/wallWeights[axis];}}
  return result;
}
fn airFitVelocity(moments:array<mat4x4f,3>,rhs:array<vec4f,3>)->vec3f{
  var result=vec3f(0.0);
  for(var axis=0u;axis<3u;axis+=1u){let mass=moments[axis][0].x;if(mass<=1e-8){continue;}
    let mean=moments[axis][0].yzw/mass;let average=rhs[axis].x/mass;
    if(dot(mean,mean)<1e-12){result[axis]=average;continue;}
    // Solve the centred 3x3 covariance. A tiny diagonal floor handles a
    // one-cell-thick axis without discarding affine tangential information.
    var matrix:array<vec4f,3>;
    let fitRhs=rhs[axis].yzw/mass-mean*average;
    for(var i=0u;i<3u;i+=1u){let covariance=moments[axis][i+1u].yzw/mass-mean[i]*mean;
      matrix[i]=vec4f(covariance,fitRhs[i]);matrix[i][i]+=1e-7;}
    for(var i=0u;i<3u;i+=1u){var pivot=i;
      for(var j=i+1u;j<3u;j+=1u){if(abs(matrix[j][i])>abs(matrix[pivot][i])){pivot=j;}}
      let temporary=matrix[i];matrix[i]=matrix[pivot];matrix[pivot]=temporary;
      matrix[i]/=select(-1.0,1.0,matrix[i][i]>=0.0)*max(abs(matrix[i][i]),1e-12);
      for(var j=0u;j<3u;j+=1u){if(i!=j){matrix[j]-=matrix[j][i]*matrix[i];}}}
    result[axis]=average-dot(mean,vec3f(matrix[0].w,matrix[1].w,matrix[2].w));
  }
  return result;
}
`;
}
