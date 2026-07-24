/**
 * Three-level persistent Galerkin numeric V-cycle.
 *
 * A single 64-lane workgroup owns the complete hierarchy, so storage barriers
 * are hierarchy-wide. Rows use grid-stride loops and the <=64-row bottom solve
 * uses one lane per row. Import and deterministic RAP refresh stay outside
 * this module because only the recurring numeric cycle belongs here.
 */
export const octreePowerGalerkinPersistent3Shader = /* wgsl */ `
struct SolveParams { numerics:vec4f }
@group(0) @binding(0) var<storage,read_write> matrix0:array<u32>;
@group(0) @binding(1) var<storage,read_write> matrix1:array<u32>;
@group(0) @binding(2) var<storage,read_write> matrix2:array<u32>;
@group(0) @binding(3) var<storage,read> topology0:array<u32>;
@group(0) @binding(4) var<storage,read> topology1:array<u32>;
@group(0) @binding(5) var<storage,read_write> vectors0:array<f32>;
@group(0) @binding(6) var<storage,read_write> vectors1:array<f32>;
@group(0) @binding(7) var<storage,read_write> vectors2:array<f32>;
@group(0) @binding(8) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(9) var<uniform> params:SolveParams;
override SMOOTH_PAIRS:u32=1u;
const LANES=64u;
const RHS=0u;const A=1u;const B=2u;
const INVALID_ROW_ERROR=1u;const NONFINITE=4u;const NONPOSITIVE=8u;const NONCONVERGED=16u;
var<workgroup> partial:array<f32,64>;
var<workgroup> directionRows:array<f32,64>;
var<workgroup> bottomActive:atomic<u32>;

fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn report(flag:u32){atomicOr(&control[0],flag);}
fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn count(level:u32)->u32{
 if(level==0u){return topology0[0];}
 if(level==1u){return topology1[0];}
 return topology1[1];
}
fn entries(level:u32)->u32{
 if(level==0u){return topology0[2];}
 if(level==1u){return topology1[2];}
 return topology1[3];
}
fn rowStart(level:u32,row:u32)->u32{
 if(level==0u){return matrix0[row];}
 if(level==1u){return matrix1[row];}
 return matrix2[row];
}
fn entryWord(level:u32,entry:u32)->u32{return count(level)+1u+2u*entry;}
fn column(level:u32,entry:u32)->u32{
 let base=entryWord(level,entry);
 if(level==0u){return matrix0[base];}
 if(level==1u){return matrix1[base];}
 return matrix2[base];
}
fn coefficient(level:u32,entry:u32)->f32{
 let base=entryWord(level,entry)+1u;
 if(level==0u){return bitcast<f32>(matrix0[base]);}
 if(level==1u){return bitcast<f32>(matrix1[base]);}
 return bitcast<f32>(matrix2[base]);
}
fn value(level:u32,channel:u32,row:u32)->f32{
 let index=channel*count(level)+row;
 if(level==0u){return vectors0[index];}
 if(level==1u){return vectors1[index];}
 return vectors2[index];
}
fn setValue(level:u32,channel:u32,row:u32,next:f32){
 let index=channel*count(level)+row;
 if(level==0u){vectors0[index]=next;}
 else if(level==1u){vectors1[index]=next;}
 else{vectors2[index]=next;}
}
fn applied(level:u32,row:u32,channel:u32)->f32{
 let first=rowStart(level,row);let end=rowStart(level,row+1u);let entryCount=entries(level);
 if(first>end||end>entryCount){report(INVALID_ROW_ERROR);return 0.0;}
 var result=0.0;
 for(var entry=first;entry<end;entry+=1u){
  let other=column(level,entry);let weight=coefficient(level,entry);
  if(other>=count(level)||!finite(weight)){report(INVALID_ROW_ERROR);}
  else{result+=weight*value(level,channel,other);}
 }
 return result;
}
fn diagonal(level:u32,row:u32)->f32{
 var diagonalEntry=0u;
 if(level==0u){diagonalEntry=topology0[topology0[13]+row];}
 else if(level==1u){diagonalEntry=topology1[topology1[13]+row];}
 else{
  let first=rowStart(level,row);let end=rowStart(level,row+1u);
  for(var entry=first;entry<end;entry+=1u){
   if(column(level,entry)==row){return coefficient(level,entry);}
  }
  report(INVALID_ROW_ERROR);return 0.0;
 }
 if(diagonalEntry<entries(level)&&column(level,diagonalEntry)==row){
  return coefficient(level,diagonalEntry);
 }
 report(INVALID_ROW_ERROR);return 0.0;
}
fn smoothValue(level:u32,row:u32,source:u32)->f32{
 let d=diagonal(level,row);let current=value(level,source,row);
 if(d==0.0){
  if(value(level,RHS,row)!=0.0||current!=0.0){report(INVALID_ROW_ERROR);}
  return 0.0;
 }
 if(d<0.0){report(NONPOSITIVE);return current;}
 let next=current+params.numerics.x*(value(level,RHS,row)-applied(level,row,source))/d;
 if(!finite(next)){report(NONFINITE);return current;}
 return next;
}
fn smoothPair(level:u32,lane:u32){
 if(!stopped()){
  for(var row=lane;row<count(level);row+=LANES){setValue(level,B,row,smoothValue(level,row,A));}
 }
 storageBarrier();
 if(!stopped()){
  for(var row=lane;row<count(level);row+=LANES){setValue(level,A,row,smoothValue(level,row,B));}
 }
 storageBarrier();
}
fn smoothLevel(level:u32,lane:u32){
 for(var pair=0u;pair<SMOOTH_PAIRS;pair+=1u){smoothPair(level,lane);}
}
fn pBase(pair:u32)->u32{return select(topology0[4],topology1[4],pair==1u);}
fn pCount(pair:u32)->u32{return select(topology0[5],topology1[5],pair==1u);}
fn restrictionOffsetBase(pair:u32)->u32{return select(topology0[6],topology1[6],pair==1u);}
fn restrictionRecordBase(pair:u32)->u32{return select(topology0[7],topology1[7],pair==1u);}
fn pOffsetBase(pair:u32)->u32{return select(topology0[12],topology1[12],pair==1u);}
fn topologyWord(pair:u32,index:u32)->u32{
 if(pair==0u){return topology0[index];}
 return topology1[index];
}
fn pFine(pair:u32,record:u32)->u32{return topologyWord(pair,pBase(pair)+3u*record);}
fn pCoarse(pair:u32,record:u32)->u32{return topologyWord(pair,pBase(pair)+3u*record+1u);}
fn pWeight(pair:u32,record:u32)->f32{
 return bitcast<f32>(topologyWord(pair,pBase(pair)+3u*record+2u));
}
fn restrictDefect(pair:u32,lane:u32){
 let sourceLevel=pair;let targetLevel=pair+1u;
 if(!stopped()){
  for(var coarse=lane;coarse<count(targetLevel);coarse+=LANES){
   let first=topologyWord(pair,restrictionOffsetBase(pair)+coarse);
   let end=topologyWord(pair,restrictionOffsetBase(pair)+coarse+1u);
   var sum=0.0;
   if(first>end){report(INVALID_ROW_ERROR);}
   else{
    for(var cursor=first;cursor<end;cursor+=1u){
     let record=topologyWord(pair,restrictionRecordBase(pair)+cursor);
     if(record>=pCount(pair)||pCoarse(pair,record)!=coarse){report(INVALID_ROW_ERROR);}
     else{
      let fine=pFine(pair,record);
      if(fine>=count(sourceLevel)){report(INVALID_ROW_ERROR);}
      else{sum+=pWeight(pair,record)
        *(value(sourceLevel,RHS,fine)-applied(sourceLevel,fine,A));}
     }
    }
   }
   if(!finite(sum)){report(NONFINITE);}
   else{
    setValue(targetLevel,RHS,coarse,sum);
    setValue(targetLevel,A,coarse,0.0);
    setValue(targetLevel,B,coarse,0.0);
   }
  }
 }
 storageBarrier();
}
fn prolongate(pair:u32,lane:u32){
 let fineLevel=pair;let coarseLevel=pair+1u;
 if(!stopped()){
  for(var fine=lane;fine<count(fineLevel);fine+=LANES){
   if(diagonal(fineLevel,fine)==0.0){setValue(fineLevel,A,fine,0.0);}
   else{
    let first=topologyWord(pair,pOffsetBase(pair)+fine);
    let end=topologyWord(pair,pOffsetBase(pair)+fine+1u);
    var correction=value(fineLevel,A,fine);
    if(first>end||end>pCount(pair)){report(INVALID_ROW_ERROR);}
    else{
     for(var record=first;record<end;record+=1u){
      let coarse=pCoarse(pair,record);
      if(coarse>=count(coarseLevel)){report(INVALID_ROW_ERROR);}
      else{correction+=pWeight(pair,record)*value(coarseLevel,A,coarse);}
     }
    }
    if(!finite(correction)){report(NONFINITE);}
    else{setValue(fineLevel,A,fine,correction);}
   }
  }
 }
 storageBarrier();
}
fn reduce(lane:u32)->f32{
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride>>=1u){
  if(lane<stride){partial[lane]+=partial[lane+stride];}
  workgroupBarrier();
 }
 let total=partial[0];
 workgroupBarrier();
 return total;
}
fn solveBottom(lane:u32){
 let bottomCount=count(2u);
 let valid=bottomCount>0u&&bottomCount<=LANES&&count(1u)>=bottomCount;
 if(!valid&&lane==0u){report(INVALID_ROW_ERROR);}
 let owner=valid&&!stopped()&&lane<bottomCount;
 var x=0.0;var r=0.0;var direction=0.0;
 if(owner){r=value(2u,RHS,lane);direction=r;}
 partial[lane]=select(0.0,r*r,owner);
 var rr=reduce(lane);let initialRR=rr;var live=select(0.0,1.0,valid&&!stopped());
 if(!finite(rr)){if(lane==0u){report(NONFINITE);}live=0.0;}
 for(var iteration=0u;iteration<LANES;iteration+=1u){
  if(lane==0u){
   atomicStore(&bottomActive,select(0u,1u,
     live>0.0&&iteration<bottomCount&&rr>max(initialRR*1e-12,1e-30)));
  }
  let stepActive=workgroupUniformLoad(&bottomActive)!=0u;
  if(!stepActive){break;}
  directionRows[lane]=direction;
  workgroupBarrier();
  var product=0.0;
  if(owner){
   let first=rowStart(2u,lane);let end=rowStart(2u,lane+1u);
   if(first>end||end>entries(2u)){report(INVALID_ROW_ERROR);}
   else{
    for(var entry=first;entry<end;entry+=1u){
     let other=column(2u,entry);
     if(other>=bottomCount){report(INVALID_ROW_ERROR);}
     else{product+=coefficient(2u,entry)*directionRows[other];}
    }
   }
  }
  partial[lane]=select(0.0,direction*product,owner);
  let pAp=reduce(lane);
  var alpha=0.0;
  if(stepActive){
   if(!(pAp>0.0)||!finite(pAp)){if(lane==0u){report(NONPOSITIVE);}live=0.0;}
   else{alpha=rr/pAp;}
  }
  if(owner&&alpha!=0.0){x+=alpha*direction;r-=alpha*product;}
  partial[lane]=select(0.0,r*r,owner);
  let nextRR=reduce(lane);
  if(alpha!=0.0){
   if(!finite(nextRR)){if(lane==0u){report(NONFINITE);}live=0.0;}
   else{
    let beta=nextRR/max(rr,1e-30);
    if(owner){direction=r+beta*direction;}
    rr=nextRR;
   }
  }
 }
 if(owner){setValue(2u,A,lane,x);setValue(2u,B,lane,r);}
 storageBarrier();
}
fn measureAndPublish(lane:u32){
 var residualSquared=0.0;var rhsSquared=0.0;
 if(!stopped()){
  for(var row=lane;row<count(0u);row+=LANES){
   let rhs=value(0u,RHS,row);let residual=rhs-applied(0u,row,A);
   if(!finite(rhs)||!finite(residual)){report(NONFINITE);}
   else{residualSquared+=residual*residual;rhsSquared+=rhs*rhs;}
  }
 }
 partial[lane]=residualSquared;let rr=reduce(lane);
 partial[lane]=rhsSquared;let bb=reduce(lane);
 if(lane==0u&&!stopped()){
  atomicStore(&control[4],bitcast<u32>(rr));
  atomicStore(&control[5],bitcast<u32>(bb));
  let cycle=atomicLoad(&control[2])+1u;atomicStore(&control[2],cycle);
  let relative=sqrt(rr/max(bb,1e-30));
  atomicStore(&control[6],bitcast<u32>(relative));
  let relativeBudget=params.numerics.y*params.numerics.y*bb;
  let absoluteBudget=f32(atomicLoad(&control[3]))*params.numerics.w*params.numerics.w;
  let residualBudget=max(relativeBudget,absoluteBudget);
  if(!finite(relative)||!finite(residualBudget)){report(NONFINITE);}
  else if(rr<=residualBudget){atomicStore(&control[1],1u);}
  else if(cycle>=u32(params.numerics.z)){report(NONCONVERGED);}
 }
 workgroupBarrier();
}
@compute @workgroup_size(64) fn solvePersistent3(
 @builtin(local_invocation_index) lane:u32,
){
 for(var cycle=0u;cycle<u32(params.numerics.z);cycle+=1u){
  smoothLevel(0u,lane);
  restrictDefect(0u,lane);
  smoothLevel(1u,lane);
  restrictDefect(1u,lane);
  solveBottom(lane);
  prolongate(1u,lane);
  smoothLevel(1u,lane);
  prolongate(0u,lane);
  smoothLevel(0u,lane);
  measureAndPublish(lane);
 }
}
`;
