/**
 * GPU kernels for the explicitly selected native-L2 Galerkin hierarchy packed
 * by octree-power-galerkin.ts. All recurring products are parent-owned
 * gathers: no directory probes, float atomics, or cross-workgroup
 * synchronization.
 *
 * This is the default UI pressure mode, but it is not Aanjaneya et al.
 * Section 4.3. The retained PCG(L2) + symmetric boundary-Jacobi / first-order
 * M1 / boundary-Jacobi implementation is a separate immutable selection and
 * never a fallback.
 */
import {
  packOctreePowerGalerkinLevel,
  planOctreePowerGalerkinExecution,
  type OctreePowerGalerkinExecutionPlan,
  type OctreePowerGalerkinHierarchy,
} from "./octree-power-galerkin";
import { PassBroker } from "./webgpu-pass-broker";
import { octreePowerGalerkinPersistent3Shader } from "./webgpu-octree-power-galerkin-persistent3";

export { octreePowerGalerkinPersistent3Shader } from "./webgpu-octree-power-galerkin-persistent3";

export const octreePowerGalerkinShader = /* wgsl */ `
struct SolveParams { numerics:vec4f }
struct LiveLeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct LiveLeafEntry { row:u32,coefficient:f32 }
@group(0) @binding(0) var<storage,read_write> sourceMatrix:array<u32>;
@group(0) @binding(1) var<storage,read_write> targetMatrix:array<u32>;
@group(0) @binding(2) var<storage,read> topology:array<u32>;
@group(0) @binding(3) var<storage,read_write> sourceVectors:array<f32>;
@group(0) @binding(4) var<storage,read_write> targetVectors:array<f32>;
@group(0) @binding(5) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params:SolveParams;
@group(0) @binding(7) var<storage,read> liveHeaders:array<LiveLeafHeader>;
@group(0) @binding(8) var<storage,read> liveEntries:array<LiveLeafEntry>;
@group(0) @binding(9) var<storage,read> authorityControl:array<u32>;
@group(0) @binding(10) var<storage,read_write> compactCorrection:array<f32>;
@group(0) @binding(11) var<storage,read_write> residualScratch:array<vec2f>;
var<workgroup> refreshPartial:array<f32,256>;
var<workgroup> cgDirection:array<f32,64>;
var<workgroup> cgActive:atomic<u32>;
const MEASURE_PARTIAL_WORKGROUPS=32u;
const RHS=0u;const A=1u;const B=2u;
const INVALID_ROW_ERROR=1u;const NONFINITE=4u;const NONPOSITIVE=8u;const NONCONVERGED=16u;
const ASSEMBLED=0x80000000u;const PROJECTED=0x40000000u;
fn sourceCount()->u32{return topology[0];}fn targetCount()->u32{return topology[1];}
fn sourceEntries()->u32{return topology[2];}fn targetEntries()->u32{return topology[3];}
fn pBase()->u32{return topology[4];}fn pCount()->u32{return topology[5];}
fn pOffsetBase()->u32{return topology[12];}
fn restrictionOffsetBase()->u32{return topology[6];}fn restrictionRecordBase()->u32{return topology[7];}
fn refreshOffsetBase()->u32{return topology[8];}fn refreshRecordBase()->u32{return topology[9];}
fn refreshRecordCount()->u32{return topology[10];}fn refreshRecordStride()->u32{return topology[11];}
fn sourceDiagonalBase()->u32{return topology[13];}
fn fineCellToRowBase()->u32{return topology[14];}
fn fineGeometryBase()->u32{return topology[15];}
fn fineCellCount()->u32{return topology[19];}
fn fineLookupLevelCount()->u32{return topology[20];}
fn sourceEntryBase(entry:u32)->u32{return sourceCount()+1u+2u*entry;}
fn targetEntryBase(entry:u32)->u32{return targetCount()+1u+2u*entry;}
fn sourceColumn(entry:u32)->u32{return sourceMatrix[sourceEntryBase(entry)];}
fn sourceCoefficient(entry:u32)->f32{return bitcast<f32>(sourceMatrix[sourceEntryBase(entry)+1u]);}
fn targetColumn(entry:u32)->u32{return targetMatrix[targetEntryBase(entry)];}
fn targetCoefficient(entry:u32)->f32{return bitcast<f32>(targetMatrix[targetEntryBase(entry)+1u]);}
fn pFine(record:u32)->u32{return topology[pBase()+3u*record];}
fn pCoarse(record:u32)->u32{return topology[pBase()+3u*record+1u];}
fn pWeight(record:u32)->f32{return bitcast<f32>(topology[pBase()+3u*record+2u]);}
fn sourceF(channel:u32,row:u32)->f32{return sourceVectors[channel*sourceCount()+row];}
fn setSourceF(channel:u32,row:u32,value:f32){sourceVectors[channel*sourceCount()+row]=value;}
fn targetF(channel:u32,row:u32)->f32{return targetVectors[channel*targetCount()+row];}
fn setTargetF(channel:u32,row:u32,value:f32){targetVectors[channel*targetCount()+row]=value;}
fn finite(value:f32)->bool{return value==value&&abs(value)<=3.402823e38;}
fn report(flag:u32){atomicOr(&control[0],flag);}
fn stopped()->bool{return atomicLoad(&control[0])!=0u||atomicLoad(&control[1])!=0u;}
fn liveCount()->u32{return select(0u,authorityControl[2],arrayLength(&authorityControl)>=3u);}
fn fixedRow(cell:u32,size:u32)->u32{
 if(cell>=fineCellCount()||size==0u||(size&(size-1u))!=0u){return 0xffffffffu;}
 var level=0u;var span=size;
 loop{
  if(span==1u){break;}
  span>>=1u;level+=1u;
 }
 if(level>=fineLookupLevelCount()){return 0xffffffffu;}
 let encoded=topology[fineCellToRowBase()+level*fineCellCount()+cell];
 if(encoded==0u){return 0xffffffffu;}
 let row=encoded-1u;
 if(row>=sourceCount()||topology[fineGeometryBase()+2u*row]!=cell
   ||topology[fineGeometryBase()+2u*row+1u]!=size){return 0xffffffffu;}
 return row;
}
fn sourceApplied(row:u32,channel:u32)->f32{
 var value=0.0;let first=sourceMatrix[row];let end=sourceMatrix[row+1u];
 if(first>end||end>sourceEntries()){report(INVALID_ROW_ERROR);return 0.0;}
 for(var entry=first;entry<end;entry+=1u){let column=sourceColumn(entry);let coefficient=sourceCoefficient(entry);
  if(column>=sourceCount()||!finite(coefficient)){report(INVALID_ROW_ERROR);continue;}
  value+=coefficient*sourceF(channel,column);}
 return value;
}
fn sourceDiagonal(row:u32)->f32{
 let entry=topology[sourceDiagonalBase()+row];
 if(entry<sourceEntries()&&sourceColumn(entry)==row){return sourceCoefficient(entry);}
 report(INVALID_ROW_ERROR);return 0.0;
}
fn smoothSource(row:u32,src:u32)->f32{
 let diagonal=sourceDiagonal(row);let current=sourceF(src,row);
 // Fixed hierarchy rows outside the current liquid operator are exact zero
 // equations. They must not become identity mass in P^T*A*P, and smoothing
 // must leave their zero correction alone without poisoning the solve.
 if(diagonal==0.0){
  if(sourceF(RHS,row)!=0.0||current!=0.0){report(INVALID_ROW_ERROR);}
  return 0.0;
 }
 if(diagonal<0.0){report(NONPOSITIVE);return current;}
 let next=current+params.numerics.x*(sourceF(RHS,row)-sourceApplied(row,src))/diagonal;
 if(!finite(next)){report(NONFINITE);return current;}return next;
}

// Reset the fixed adaptive-superset system to the zero extension of the live
// operator. Live compact rows overwrite their fixed identities in the next
// dispatch. Zero, rather than identity, inactive rows are essential: adding
// identity air equations would pollute P^T*A*P and destroy coarse correction.
@compute @workgroup_size(64) fn initializeFineOperator(@builtin(global_invocation_id) gid:vec3u){
 let row=gid.x;if(row>=sourceCount()){return;}
 let first=sourceMatrix[row];let end=sourceMatrix[row+1u];
 if(first>end||end>sourceEntries()){report(INVALID_ROW_ERROR);return;}
 for(var entry=first;entry<end;entry+=1u){
  sourceMatrix[sourceEntryBase(entry)+1u]=bitcast<u32>(0.0);
 }
 setSourceF(RHS,row,0.0);
 setSourceF(A,row,0.0);setSourceF(B,row,0.0);
}

// Import changing coefficients/RHS from compact live rows into the immutable
// fixed hierarchy. Live row order and entry order may change; identity is
// resolved through LeafHeader.cell and the neighbour header rather than by
// positional coincidence.
@compute @workgroup_size(64) fn importFineOperator(@builtin(global_invocation_id) gid:vec3u){
 let liveRow=gid.x;let count=liveCount();
 if(liveRow==0u){atomicStore(&control[3],count);}
 if(liveRow>=count){return;}
 if(arrayLength(&authorityControl)<3u||liveRow>=arrayLength(&liveHeaders)
   ||sourceDiagonalBase()==0u){report(INVALID_ROW_ERROR);return;}
 let flags=authorityControl[0];
 if((flags&ASSEMBLED)==0u||(flags&~(ASSEMBLED|PROJECTED))!=0u
   ||count>arrayLength(&liveHeaders)){report(INVALID_ROW_ERROR);return;}
 let header=liveHeaders[liveRow];let row=fixedRow(header.cell,header.size);
 if(row==0xffffffffu){report(INVALID_ROW_ERROR);return;}
 if(!(header.diagonal>0.0)||!finite(header.diagonal)
   ||header.entryStart>arrayLength(&liveEntries)
   ||header.entryCount>arrayLength(&liveEntries)-header.entryStart){
  report(INVALID_ROW_ERROR);return;
 }
 let diagonalEntry=topology[sourceDiagonalBase()+row];
 if(diagonalEntry>=sourceEntries()||sourceColumn(diagonalEntry)!=row){
  report(INVALID_ROW_ERROR);return;
 }
 sourceMatrix[sourceEntryBase(diagonalEntry)+1u]=bitcast<u32>(header.diagonal);
 sourceVectors[row]=-header.rhs;
 var initial=compactCorrection[liveRow];
 if(!finite(initial)){report(NONFINITE);initial=0.0;}
 setSourceF(A,row,initial);
 for(var local=0u;local<header.entryCount;local+=1u){
  let liveEntryIndex=header.entryStart+local;
  let entry=liveEntries[liveEntryIndex];
  if(entry.row>=count||entry.row>=arrayLength(&liveHeaders)
    ||entry.coefficient<0.0||!finite(entry.coefficient)){
   report(INVALID_ROW_ERROR);continue;
  }
  let other=liveHeaders[entry.row];let column=fixedRow(other.cell,other.size);
  if(column==0xffffffffu){report(INVALID_ROW_ERROR);continue;}
  var matrixEntry=0xffffffffu;
  for(var candidate=sourceMatrix[row];candidate<sourceMatrix[row+1u];candidate+=1u){
   if(sourceColumn(candidate)==column){matrixEntry=candidate;break;}
  }
  if(matrixEntry==0xffffffffu){report(INVALID_ROW_ERROR);continue;}
  sourceMatrix[sourceEntryBase(matrixEntry)+1u]=bitcast<u32>(-entry.coefficient);
 }
}

// P is immutable, so the packed refresh matrix bakes P(left)*P(right) into
// target-major [fineEntry, weight] records. Four isolated 64-lane tiles share
// each workgroup: every target retains the exact lane assignment and reduction
// tree of the former one-target workgroup while paying one quarter as many
// workgroup launches.
@compute @workgroup_size(256) fn refreshGalerkinTarget(
 @builtin(workgroup_id) group:vec3u,
 @builtin(num_workgroups) groups:vec3u,
 @builtin(local_invocation_index) lane:u32,
){
 let tile=lane>>6u;let tileLane=lane&63u;
 let targetEntry=4u*(group.x+groups.x*group.y)+tile;
 var valid=targetEntry<targetEntries();
 var first=0u;var end=0u;
 if(valid){
  first=topology[refreshOffsetBase()+targetEntry];
  end=topology[refreshOffsetBase()+targetEntry+1u];
  if(first>end||end>refreshRecordCount()||refreshRecordStride()!=2u){
   if(tileLane==0u){report(INVALID_ROW_ERROR);}valid=false;
  }
 }
 var sum=0.0;
 for(var cursor=first+tileLane;valid&&cursor<end;cursor+=64u){
  let base=refreshRecordBase()+2u*cursor;
  let fineEntry=topology[base];let weight=bitcast<f32>(topology[base+1u]);
  if(fineEntry>=sourceEntries()||!finite(weight)){report(INVALID_ROW_ERROR);continue;}
  sum+=sourceCoefficient(fineEntry)*weight;
 }
 refreshPartial[lane]=sum;
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  if(tileLane<stride){refreshPartial[lane]+=refreshPartial[lane+stride];}
  workgroupBarrier();
 }
 if(tileLane==0u&&valid){
  let total=refreshPartial[64u*tile];
  if(!finite(total)){report(NONFINITE);}
  else{targetMatrix[targetEntryBase(targetEntry)+1u]=bitcast<u32>(total);}
 }
}

@compute @workgroup_size(64) fn clearSourceCorrection(@builtin(global_invocation_id) gid:vec3u){
 let count=arrayLength(&sourceVectors)/3u;let row=gid.x;
 if(row<count){sourceVectors[count+row]=0.0;sourceVectors[2u*count+row]=0.0;}
}
@compute @workgroup_size(64) fn smoothSourceAtoB(@builtin(global_invocation_id) gid:vec3u){
 let row=gid.x;if(row<sourceCount()&&!stopped()){setSourceF(B,row,smoothSource(row,A));}
}
@compute @workgroup_size(64) fn smoothSourceBtoA(@builtin(global_invocation_id) gid:vec3u){
 let row=gid.x;if(row<sourceCount()&&!stopped()){setSourceF(A,row,smoothSource(row,B));}
}

// R=P^T gathers the final even pre-sweep defect and initializes the coarse
// correction vectors. Four isolated 64-lane tiles share each workgroup:
// shallow coarse levels retain their row-local reduction order without paying
// one whole workgroup launch per row.
@compute @workgroup_size(256) fn restrictSourceDefect(
 @builtin(workgroup_id) group:vec3u,
 @builtin(num_workgroups) groups:vec3u,
 @builtin(local_invocation_index) lane:u32,
){
 let tile=lane>>6u;let tileLane=lane&63u;
 let coarse=4u*(group.x+groups.x*group.y)+tile;
 var valid=coarse<targetCount();
 var first=0u;var end=0u;
 if(valid){
  first=topology[restrictionOffsetBase()+coarse];
  end=topology[restrictionOffsetBase()+coarse+1u];
  if(first>end){if(tileLane==0u){report(INVALID_ROW_ERROR);}valid=false;}
 }
 var sum=0.0;
 if(valid&&!stopped()){
  for(var cursor=first+tileLane;cursor<end;cursor+=64u){
   let record=topology[restrictionRecordBase()+cursor];
   if(record>=pCount()||pCoarse(record)!=coarse){report(INVALID_ROW_ERROR);continue;}
   let fine=pFine(record);if(fine>=sourceCount()){report(INVALID_ROW_ERROR);continue;}
   sum+=pWeight(record)*(sourceF(RHS,fine)-sourceApplied(fine,A));
  }
 }
 refreshPartial[lane]=sum;
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride>>=1u){
  if(tileLane<stride){refreshPartial[lane]+=refreshPartial[lane+stride];}
  workgroupBarrier();
 }
 if(tileLane==0u&&valid&&!stopped()){
  let total=refreshPartial[64u*tile];
  if(!finite(total)){report(NONFINITE);}
  else{setTargetF(RHS,coarse,total);setTargetF(A,coarse,0.0);setTargetF(B,coarse,0.0);}
 }
}

@compute @workgroup_size(64) fn prolongateTargetCorrection(@builtin(global_invocation_id) gid:vec3u){
 let fine=gid.x;if(fine>=sourceCount()||stopped()){return;}
 // A zero diagonal identifies a fixed row outside the current operator.
 // Prolongating into it would manufacture a nonzero value for a zero equation
 // and then make the following smoother reject an otherwise valid solve.
 if(sourceDiagonal(fine)==0.0){setSourceF(A,fine,0.0);return;}
 var correction=sourceF(A,fine);
 let first=topology[pOffsetBase()+fine];let end=topology[pOffsetBase()+fine+1u];
 if(first>end||end>pCount()){report(INVALID_ROW_ERROR);return;}
 for(var record=first;record<end;record+=1u){let coarse=pCoarse(record);
  if(coarse>=targetCount()){report(INVALID_ROW_ERROR);continue;}
  correction+=pWeight(record)*targetF(A,coarse);
 }
 if(!finite(correction)){report(NONFINITE);}else{setSourceF(A,fine,correction);}
}

// Workgroup-wide reduction over refreshPartial. Callers store one addend per
// lane before calling; every lane receives the total.
fn cgReduce(lane:u32)->f32{
 workgroupBarrier();
 for(var stride=32u;stride>0u;stride>>=1u){
  if(lane<stride){refreshPartial[lane]+=refreshPartial[lane+stride];}
  workgroupBarrier();
 }
 let total=refreshPartial[0];
 workgroupBarrier();
 return total;
}
// The hierarchy contract bounds the last level to 64 rows. Lanes cooperate:
// one row per lane and workgroup-reduced dot products. A workgroup-uniform
// residual test exits the bounded 64-step loop without crossing a barrier.
// The former single-lane serial CG walked every matrix entry sequentially and
// was a measured latency elephant in each cycle.
@compute @workgroup_size(64) fn solveTargetCoarsest(@builtin(local_invocation_index) lane:u32){
 let count=targetCount();
 if(count==0u||count>64u||sourceCount()<count){
  if(lane==0u){report(INVALID_ROW_ERROR);}return;
 }
 let owner=!stopped()&&lane<count;
 var x=0.0;var r=0.0;var direction=0.0;
 if(owner){r=targetF(RHS,lane);direction=r;}
 refreshPartial[lane]=select(0.0,r*r,owner);
 var rr=cgReduce(lane);
 let initialRR=rr;
 var live=1.0;
 if(!finite(rr)){if(lane==0u){report(NONFINITE);}live=0.0;}
 for(var iteration=0u;iteration<64u;iteration+=1u){
  if(lane==0u){
   atomicStore(&cgActive,select(0u,1u,
     live>0.0&&iteration<count&&rr>max(initialRR*1e-12,1e-30)));
  }
  let stepActive=workgroupUniformLoad(&cgActive)!=0u;
  if(!stepActive){break;}
  cgDirection[lane]=direction;
  workgroupBarrier();
  var applied=0.0;
  if(owner){
   for(var entry=targetMatrix[lane];entry<targetMatrix[lane+1u];entry+=1u){
    let column=targetColumn(entry);
    if(column>=count){report(INVALID_ROW_ERROR);}
    else{applied+=targetCoefficient(entry)*cgDirection[column];}
   }
  }
  refreshPartial[lane]=select(0.0,direction*applied,owner);
  let pAp=cgReduce(lane);
  var alpha=0.0;
  if(stepActive){
   if(!(pAp>0.0)||!finite(pAp)){if(lane==0u){report(NONPOSITIVE);}live=0.0;}
   else{alpha=rr/pAp;}
  }
  if(owner&&alpha!=0.0){x+=alpha*direction;r-=alpha*applied;}
  refreshPartial[lane]=select(0.0,r*r,owner);
  let nextRR=cgReduce(lane);
  if(alpha!=0.0){
   if(!finite(nextRR)){if(lane==0u){report(NONFINITE);}live=0.0;}
   else{
    let beta=nextRR/max(rr,1e-30);
    if(owner){direction=r+beta*direction;}
    rr=nextRR;
   }
  }
 }
 if(owner){setTargetF(A,lane,x);setTargetF(B,lane,r);}
}

var<workgroup> residualSums:array<vec2f,256>;
// The former single-workgroup scan serialized the whole fine domain onto one
// GPU core every cycle. Partials spread the scan across the device; the
// one-workgroup reduce folds the bounded partial array.
@compute @workgroup_size(256) fn measureSourceResidualPartials(
 @builtin(workgroup_id) wg:vec3u,
 @builtin(local_invocation_index) lane:u32,
){
 var sum=vec2f(0.0);
 if(!stopped()){
  for(var row=wg.x*256u+lane;row<sourceCount();row+=256u*MEASURE_PARTIAL_WORKGROUPS){
   let rhs=sourceF(RHS,row);let residual=rhs-sourceApplied(row,A);
   if(!finite(rhs)||!finite(residual)){report(NONFINITE);}
   else{sum+=vec2f(residual*residual,rhs*rhs);}
  }
 }
 residualSums[lane]=sum;
 for(var width=128u;width>0u;width>>=1u){
  workgroupBarrier();if(lane<width){residualSums[lane]+=residualSums[lane+width];}
 }
 workgroupBarrier();
 if(lane==0u&&wg.x<arrayLength(&residualScratch)){residualScratch[wg.x]=residualSums[0];}
}
@compute @workgroup_size(32) fn measureSourceResidualReduce(@builtin(local_invocation_index) lane:u32){
 var sum=vec2f(0.0);
 if(lane<MEASURE_PARTIAL_WORKGROUPS&&lane<arrayLength(&residualScratch)){sum=residualScratch[lane];}
 residualSums[lane]=sum;
 for(var width=16u;width>0u;width>>=1u){
  workgroupBarrier();if(lane<width){residualSums[lane]+=residualSums[lane+width];}
 }
 workgroupBarrier();
 if(lane==0u&&!stopped()){
  atomicStore(&control[4],bitcast<u32>(residualSums[0].x));
  atomicStore(&control[5],bitcast<u32>(residualSums[0].y));
  publishSourceResidualValues(residualSums[0].x,residualSums[0].y);
 }
}
fn publishSourceResidualValues(rr:f32,bb:f32){
 if(stopped()){return;}
 let cycle=atomicLoad(&control[2])+1u;atomicStore(&control[2],cycle);
 let relative=sqrt(rr/max(bb,1e-30));atomicStore(&control[6],bitcast<u32>(relative));
 let relativeBudget=params.numerics.y*params.numerics.y*bb;
 // importFineOperator publishes the authoritative compact liquid-row count
 // into control[3]. Normalize the absolute floor per live row rather than by
 // the immutable adaptive-superset source arena.
 let absoluteBudget=f32(atomicLoad(&control[3]))*params.numerics.w*params.numerics.w;
 let residualBudget=max(relativeBudget,absoluteBudget);
 if(!finite(relative)||!finite(residualBudget)){report(NONFINITE);}
 else if(rr<=residualBudget){atomicStore(&control[1],1u);}
 else if(cycle>=u32(params.numerics.z)){report(NONCONVERGED);}
}
// Retained as a standalone diagnostic/test entry point. The recurring solve
// publishes directly from measureSourceResidualReduce and does not pay this
// extra dispatch.
@compute @workgroup_size(1) fn publishSourceResidual(){
 publishSourceResidualValues(bitcast<f32>(atomicLoad(&control[4])),bitcast<f32>(atomicLoad(&control[5])));
}
@compute @workgroup_size(64) fn exportLiveCorrection(@builtin(global_invocation_id) gid:vec3u){
 let liveRow=gid.x;if(liveRow>=liveCount()||atomicLoad(&control[0])!=0u
   ||atomicLoad(&control[1])==0u){return;}
 if(liveRow>=arrayLength(&liveHeaders)||liveRow>=arrayLength(&compactCorrection)){
  report(INVALID_ROW_ERROR);return;
 }
 let header=liveHeaders[liveRow];let row=fixedRow(header.cell,header.size);
 if(row==0xffffffffu){report(INVALID_ROW_ERROR);return;}
 let value=sourceF(A,row);
 if(!finite(value)){report(NONFINITE);}else{compactCorrection[liveRow]=value;}
}
`;

export const OCTREE_POWER_GALERKIN_PIPELINES = Object.freeze([
  "initializeFineOperator",
  "importFineOperator",
  "refreshGalerkinTarget",
  "clearSourceCorrection",
  "smoothSourceAtoB",
  "smoothSourceBtoA",
  "restrictSourceDefect",
  "prolongateTargetCorrection",
  "solveTargetCoarsest",
  "measureSourceResidualPartials",
  "measureSourceResidualReduce",
  "publishSourceResidual",
  "exportLiveCorrection",
] as const);

/** Fixed fan-out of the two-stage residual measurement. */
export const OCTREE_POWER_GALERKIN_MEASURE_PARTIAL_WORKGROUPS = 32;

type OctreePowerGalerkinPipelineName = typeof OCTREE_POWER_GALERKIN_PIPELINES[number];

export const OCTREE_POWER_GALERKIN_CONTROL_BYTES = 64;

export interface WebGPUOctreePowerGalerkinOptions {
  readonly cycles?: number;
  /** Rounded to a symmetric A→B/B→A pair count (2 or 4). */
  readonly smoothingIterations?: number;
  readonly damping?: number;
  readonly relativeTolerance?: number;
  readonly absoluteRmsTolerance?: number;
  /** Test/diagnostic escape hatch. Production defaults to the persistent
   * single-dispatch numeric path for exactly three hierarchy levels. */
  readonly persistentThreeLevel?: boolean;
}

export interface WebGPUOctreePowerGalerkinSolve {
  /** Current production compact rows. Supplying these performs the one-dispatch
   * fine coefficient/RHS import before numeric RAP refresh. */
  readonly liveOperator: Readonly<{
    leafHeaders: GPUBuffer;
    leafEntries: GPUBuffer;
    /** Native L2 authority control (`ASSEMBLED`, rowCount). */
    authorityControl: GPUBuffer;
  }>;
  /** Optional device-local f32 compact correction in current live-row order.
   * This is read only as the warm start; it need not alias `correction`. */
  readonly initialCorrection?: GPUBuffer;
  /** Optional device-local f32 destination receiving the finest correction. */
  readonly correction?: GPUBuffer;
}

/**
 * Experimental executable GPU V-cycle for a prebuilt symbolic hierarchy.
 *
 * Matrix zero owns the externally refreshed fine coefficients. Every coarser
 * matrix is recomputed on device by deterministic RAP gathers before solving.
 * The symbolic hierarchy is immutable and therefore reusable until topology
 * generation changes.
 */
export class WebGPUOctreePowerGalerkin {
  readonly plan: OctreePowerGalerkinExecutionPlan;
  readonly control: GPUBuffer;
  readonly fineMatrix: GPUBuffer;
  readonly finestVectors: GPUBuffer;
  readonly allocatedBytes: number;

  private readonly matrices: readonly GPUBuffer[];
  private readonly vectors: readonly GPUBuffer[];
  private readonly topologies: readonly GPUBuffer[];
  private readonly nodeCounts: readonly number[];
  private readonly entryCounts: readonly number[];
  private readonly pipelines: Readonly<Record<OctreePowerGalerkinPipelineName, GPUComputePipeline>>;
  private readonly bindGroups: readonly Readonly<Partial<Record<OctreePowerGalerkinPipelineName, GPUBindGroup>>>[];
  private readonly params: GPUBuffer;
  private readonly zeroCompactCorrection: GPUBuffer;
  private readonly residualScratch: GPUBuffer;
  private readonly persistent3Pipeline?: GPUComputePipeline;
  private readonly persistent3Group?: GPUBindGroup;
  private importedSource?: Readonly<{
    headers: GPUBuffer;
    entries: GPUBuffer;
    authority: GPUBuffer;
    initialCorrection: GPUBuffer;
    bindGroup: GPUBindGroup;
  }>;
  private exportedCorrection?: Readonly<{
    headers: GPUBuffer;
    authority: GPUBuffer;
    correction: GPUBuffer;
    bindGroup: GPUBindGroup;
  }>;

  constructor(
    private readonly device: GPUDevice,
    hierarchy: OctreePowerGalerkinHierarchy,
    operators: readonly ArrayLike<number>[],
    options: WebGPUOctreePowerGalerkinOptions = {},
  ) {
    this.plan = planOctreePowerGalerkinExecution(hierarchy, options.cycles, options.smoothingIterations);
    if (hierarchy.levels.length < 2) throw new RangeError("GPU power Galerkin solve requires at least two levels");
    if (hierarchy.levels.at(-1)!.nodeCount > 64) {
      throw new RangeError("GPU power Galerkin bottom level exceeds the bounded 64-row solve");
    }
    const packed = hierarchy.levels.slice(0, -1)
      .map((_, level) => packOctreePowerGalerkinLevel(hierarchy, operators, level));
    this.nodeCounts = hierarchy.levels.map((level) => level.nodeCount);
    this.entryCounts = hierarchy.levels.map((level) => level.columns.length);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const upload = (label: string, words: Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer => {
      const buffer = device.createBuffer({ label, size: Math.max(4, words.byteLength),
        usage: usage | GPUBufferUsage.COPY_DST });
      if (words.byteLength > 0) {
        const bytes = new Uint8Array(words.byteLength);
        bytes.set(new Uint8Array(words.buffer, words.byteOffset, words.byteLength));
        device.queue.writeBuffer(buffer, 0, bytes);
      }
      return buffer;
    };
    const matrixWords = [packed[0].sourceMatrixWords, ...packed.map((level) => level.targetMatrixWords)];
    this.matrices = matrixWords.map((words, level) =>
      upload(`Power Galerkin L${level} matrix`, words, storage));
    this.vectors = hierarchy.levels.map((level, index) => device.createBuffer({
      label: `Power Galerkin L${index} vectors`,
      size: Math.max(4, 3 * level.nodeCount * Float32Array.BYTES_PER_ELEMENT),
      usage: storage,
    }));
    this.topologies = packed.map((level, index) =>
      upload(`Power Galerkin L${index} topology`, level.topologyWords, GPUBufferUsage.STORAGE));
    this.control = device.createBuffer({
      label: "Power Galerkin solve control",
      size: OCTREE_POWER_GALERKIN_CONTROL_BYTES,
      usage: storage,
    });
    this.params = device.createBuffer({
      label: "Power Galerkin solve parameters",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.zeroCompactCorrection = device.createBuffer({
      label: "Power Galerkin zero compact correction",
      size: Math.max(4, hierarchy.levels[0].nodeCount * Float32Array.BYTES_PER_ELEMENT),
      usage: storage,
    });
    this.residualScratch = device.createBuffer({
      label: "Power Galerkin residual partials",
      size: OCTREE_POWER_GALERKIN_MEASURE_PARTIAL_WORKGROUPS * 8,
      usage: GPUBufferUsage.STORAGE,
    });
    const damping = Math.max(0.05, Math.min(0.95, options.damping ?? 2 / 3));
    const relativeTolerance = Math.max(1e-4, Math.min(0.5, options.relativeTolerance ?? 0.02));
    const absoluteRmsTolerance = Math.max(1e-12, Math.min(1, options.absoluteRmsTolerance ?? 1e-7));
    device.queue.writeBuffer(this.params, 0, new Float32Array([
      damping, relativeTolerance, this.plan.cycles, absoluteRmsTolerance,
    ]));

    const module = device.createShaderModule({ label: "Power Galerkin solver", code: octreePowerGalerkinShader });
    this.pipelines = Object.fromEntries(OCTREE_POWER_GALERKIN_PIPELINES.map((entryPoint) => [
      entryPoint,
      device.createComputePipeline({
        label: `Power Galerkin ${entryPoint}`,
        layout: "auto",
        compute: { module, entryPoint },
      }),
    ])) as unknown as Readonly<Record<OctreePowerGalerkinPipelineName, GPUComputePipeline>>;
    const bindings: Readonly<Record<OctreePowerGalerkinPipelineName, readonly number[]>> = {
      initializeFineOperator: [0, 2, 3, 5],
      importFineOperator: [0, 2, 3, 5, 7, 8, 9, 10],
      refreshGalerkinTarget: [0, 1, 2, 5],
      clearSourceCorrection: [3],
      smoothSourceAtoB: [0, 2, 3, 5, 6],
      smoothSourceBtoA: [0, 2, 3, 5, 6],
      restrictSourceDefect: [0, 2, 3, 4, 5],
      prolongateTargetCorrection: [0, 2, 3, 4, 5],
      solveTargetCoarsest: [1, 2, 4, 5],
      measureSourceResidualPartials: [0, 2, 3, 5, 11],
      measureSourceResidualReduce: [5, 6, 11],
      publishSourceResidual: [5, 6],
      exportLiveCorrection: [2, 3, 5, 7, 9, 10],
    };
    this.bindGroups = packed.map((_, level) => {
      const resources: Readonly<Record<number, GPUBuffer>> = {
        0: this.matrices[level], 1: this.matrices[level + 1], 2: this.topologies[level],
        3: this.vectors[level], 4: this.vectors[level + 1], 5: this.control, 6: this.params,
        11: this.residualScratch,
      };
      return Object.fromEntries(OCTREE_POWER_GALERKIN_PIPELINES
        .filter((name) => name !== "importFineOperator" && name !== "exportLiveCorrection").map((name) => [name,
        device.createBindGroup({
          label: `Power Galerkin ${name} L${level}→L${level + 1}`,
          layout: this.pipelines[name].getBindGroupLayout(0),
          entries: bindings[name].map((binding) => ({ binding, resource: { buffer: resources[binding] } })),
        }),
      ])) as Readonly<Partial<Record<OctreePowerGalerkinPipelineName, GPUBindGroup>>>;
    });
    // A single resident workgroup only wins when it can keep approximately one
    // fine row per lane. The fixed adaptive mini-dam arena has 4,608 fine rows;
    // serializing that arena onto 64 lanes is substantially slower than the
    // parallel staged solve even though it removes launches. Keep an explicit
    // diagnostic override for parity/performance experiments.
    const persistentThreeLevel = hierarchy.levels.length === 3
      && (options.persistentThreeLevel === true
        || (options.persistentThreeLevel !== false && hierarchy.levels[0].nodeCount <= 64));
    if (persistentThreeLevel) {
      const persistent3Layout = device.createBindGroupLayout({
        label: "Power Galerkin persistent three-level layout",
        entries: [
          ...[0, 1, 2].map((binding) => ({
            binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const },
          })),
          ...[3, 4].map((binding) => ({
            binding, visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" as const },
          })),
          ...[5, 6, 7, 8].map((binding) => ({
            binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const },
          })),
          { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" as const } },
        ],
      });
      const persistent3Module = device.createShaderModule({
        label: "Power Galerkin persistent three-level solver",
        code: octreePowerGalerkinPersistent3Shader,
      });
      this.persistent3Pipeline = device.createComputePipeline({
        label: "Power Galerkin persistent three-level V-cycle",
        layout: device.createPipelineLayout({
          label: "Power Galerkin persistent three-level pipeline layout",
          bindGroupLayouts: [persistent3Layout],
        }),
        compute: {
          module: persistent3Module,
          entryPoint: "solvePersistent3",
          constants: { SMOOTH_PAIRS: this.plan.smoothingIterations / 2 },
        },
      });
      this.persistent3Group = device.createBindGroup({
        label: "Power Galerkin persistent three-level bindings",
        layout: persistent3Layout,
        entries: [
          { binding: 0, resource: { buffer: this.matrices[0] } },
          { binding: 1, resource: { buffer: this.matrices[1] } },
          { binding: 2, resource: { buffer: this.matrices[2] } },
          { binding: 3, resource: { buffer: this.topologies[0] } },
          { binding: 4, resource: { buffer: this.topologies[1] } },
          { binding: 5, resource: { buffer: this.vectors[0] } },
          { binding: 6, resource: { buffer: this.vectors[1] } },
          { binding: 7, resource: { buffer: this.vectors[2] } },
          { binding: 8, resource: { buffer: this.control } },
          { binding: 9, resource: { buffer: this.params } },
        ],
      });
    }
    this.fineMatrix = this.matrices[0];
    this.finestVectors = this.vectors[0];
    this.allocatedBytes = matrixWords.reduce((sum, words) => sum + words.byteLength, 0)
      + packed.reduce((sum, level) => sum + level.topologyWords.byteLength, 0)
      + hierarchy.levels.reduce((sum, level) => sum + 3 * level.nodeCount * 4, 0)
      + hierarchy.levels[0].nodeCount * 4
      + OCTREE_POWER_GALERKIN_CONTROL_BYTES + 16;
  }

  /** Refresh the fine matrix/RHS directly from the production LeafHeader /
   * LeafEntry ABI. The bind group is retained while buffer identity is stable. */
  encodeFineOperatorImport(
    broker: PassBroker,
    leafHeaders: GPUBuffer,
    leafEntries: GPUBuffer,
    authorityControl: GPUBuffer,
    initialCorrection: GPUBuffer,
  ): void {
    if (leafHeaders.size < 48 || leafEntries.size < 8 || authorityControl.size < 12) {
      throw new RangeError("power Galerkin live operator buffers are too small");
    }
    if (this.importedSource?.headers !== leafHeaders || this.importedSource.entries !== leafEntries
      || this.importedSource.authority !== authorityControl
      || this.importedSource.initialCorrection !== initialCorrection) {
      const resources: Readonly<Record<number, GPUBuffer>> = {
        0: this.matrices[0], 2: this.topologies[0], 3: this.vectors[0],
        5: this.control, 7: leafHeaders, 8: leafEntries, 9: authorityControl,
        10: initialCorrection,
      };
      this.importedSource = Object.freeze({
        headers: leafHeaders, entries: leafEntries, authority: authorityControl,
        initialCorrection,
        bindGroup: this.device.createBindGroup({
          label: "Power Galerkin live fine operator import",
          layout: this.pipelines.importFineOperator.getBindGroupLayout(0),
          entries: [0, 2, 3, 5, 7, 8, 9, 10].map((binding) =>
            ({ binding, resource: { buffer: resources[binding] } })),
        }),
      });
    }
    const pass = broker.compute();
    pass.setPipeline(this.pipelines.importFineOperator);
    pass.setBindGroup(0, this.importedSource.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.nodeCounts[0] / 64));
  }

  encode(broker: PassBroker, solve: WebGPUOctreePowerGalerkinSolve): void {
    const finestCount = this.nodeCounts[0];
    if (solve.initialCorrection && solve.initialCorrection.size < 4) {
      throw new RangeError("power Galerkin compact GPU initial correction is too small");
    }
    if (solve.correction && solve.correction.size < 4) {
      throw new RangeError("power Galerkin compact GPU correction is too small");
    }
    broker.clearBuffer(this.control);
    const dispatch = (
      level: number,
      name: OctreePowerGalerkinPipelineName,
      count: number,
      oneWorkgroupPerItem = false,
    ) => {
      const pass = broker.compute();
      pass.setPipeline(this.pipelines[name]);
      const bindGroup = this.bindGroups[level][name];
      if (!bindGroup) throw new Error(`power Galerkin ${name} requires a dynamic binding`);
      pass.setBindGroup(0, bindGroup);
      if (!oneWorkgroupPerItem) {
        pass.dispatchWorkgroups(Math.max(1, Math.ceil(count / 64)));
        return;
      }
      const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
      const width = Math.max(1, Math.min(count, limit));
      const height = Math.max(1, Math.ceil(count / width));
      if (height > limit) throw new RangeError(`power Galerkin ${name} dispatch exceeds device limits`);
      pass.dispatchWorkgroups(width, height);
    };
    dispatch(0, "initializeFineOperator", finestCount);
    this.encodeFineOperatorImport(broker, solve.liveOperator.leafHeaders, solve.liveOperator.leafEntries,
      solve.liveOperator.authorityControl, solve.initialCorrection ?? this.zeroCompactCorrection);
    for (let level = 0; level < this.topologies.length; level += 1) {
      dispatch(level, "refreshGalerkinTarget", Math.ceil(this.entryCounts[level + 1] / 4), true);
    }
    if (this.persistent3Pipeline && this.persistent3Group) {
      const persistent = broker.compute({ label: "Persistent three-level Power Galerkin V-cycles" });
      persistent.setPipeline(this.persistent3Pipeline);
      persistent.setBindGroup(0, this.persistent3Group);
      persistent.dispatchWorkgroups(1);
    } else for (let cycle = 0; cycle < this.plan.cycles; cycle += 1) {
      for (let level = 0; level < this.topologies.length; level += 1) {
        for (let sweep = 0; sweep < this.plan.smoothingIterations; sweep += 2) {
          dispatch(level, "smoothSourceAtoB", this.nodeCounts[level]);
          dispatch(level, "smoothSourceBtoA", this.nodeCounts[level]);
        }
        dispatch(level, "restrictSourceDefect", Math.ceil(this.nodeCounts[level + 1] / 4), true);
      }
      const bottomPair = this.topologies.length - 1;
      dispatch(bottomPair, "solveTargetCoarsest", 1);
      for (let level = this.topologies.length - 1; level >= 0; level -= 1) {
        dispatch(level, "prolongateTargetCorrection", this.nodeCounts[level]);
        for (let sweep = 0; sweep < this.plan.smoothingIterations; sweep += 2) {
          dispatch(level, "smoothSourceAtoB", this.nodeCounts[level]);
          dispatch(level, "smoothSourceBtoA", this.nodeCounts[level]);
        }
      }
      dispatch(0, "measureSourceResidualPartials",
        OCTREE_POWER_GALERKIN_MEASURE_PARTIAL_WORKGROUPS, true);
      dispatch(0, "measureSourceResidualReduce", 1);
    }
    if (solve.correction) {
      const source = solve.liveOperator;
      if (this.exportedCorrection?.headers !== source.leafHeaders
        || this.exportedCorrection.authority !== source.authorityControl
        || this.exportedCorrection.correction !== solve.correction) {
        const resources: Readonly<Record<number, GPUBuffer>> = {
          2: this.topologies[0], 3: this.vectors[0], 5: this.control,
          7: source.leafHeaders, 9: source.authorityControl, 10: solve.correction,
        };
        this.exportedCorrection = Object.freeze({
          headers: source.leafHeaders,
          authority: source.authorityControl,
          correction: solve.correction,
          bindGroup: this.device.createBindGroup({
            label: "Power Galerkin compact correction export",
            layout: this.pipelines.exportLiveCorrection.getBindGroupLayout(0),
            entries: [2, 3, 5, 7, 9, 10].map((binding) =>
              ({ binding, resource: { buffer: resources[binding] } })),
          }),
        });
      }
      const pass = broker.compute();
      pass.setPipeline(this.pipelines.exportLiveCorrection);
      pass.setBindGroup(0, this.exportedCorrection.bindGroup);
      pass.dispatchWorkgroups(Math.max(1, Math.ceil(finestCount / 64)));
    }
  }

  destroy(): void {
    this.importedSource = undefined;
    this.exportedCorrection = undefined;
    for (const buffer of this.matrices) buffer.destroy();
    for (const buffer of this.vectors) buffer.destroy();
    for (const buffer of this.topologies) buffer.destroy();
    this.control.destroy();
    this.params.destroy();
    this.zeroCompactCorrection.destroy();
    this.residualScratch.destroy();
  }
}
