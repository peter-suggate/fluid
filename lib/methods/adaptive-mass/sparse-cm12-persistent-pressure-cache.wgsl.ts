import {
  SPARSE_CM12_PRESSURE_CACHE_CAUSE,
  SPARSE_CM12_PRESSURE_CACHE_FAULT,
  SPARSE_CM12_PRESSURE_CACHE_FLAG,
  SPARSE_CM12_PRESSURE_CACHE_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS,
  SPARSE_CM12_PRESSURE_CACHE_MAGIC,
  SPARSE_CM12_PRESSURE_CACHE_PHASE,
  SPARSE_CM12_PRESSURE_CACHE_VERSION,
  type SparseCM12PersistentPressureCacheLayout,
} from "./sparse-cm12-persistent-pressure-cache";
import { createSparseCM12PersistentPressureCacheAggregateWGSL } from
  "./sparse-cm12-persistent-pressure-cache-aggregate.wgsl";

export interface SparseCM12PersistentPressureCacheWGSLOptions {
  readonly layout: SparseCM12PersistentPressureCacheLayout;
  /** Existing `array<atomic<u32>>` bound to cm12.pressure-cache. */
  readonly arenaName?: string;
  /** Existing PCM1 membership functions. */
  readonly cellContainsFunction?: string;
  readonly rowContainsFunction?: string;
  /** Existing row-owned rigid coefficient scale. Omit for literal 1.0. */
  readonly solidRowScaleFunction?: string;
  readonly workgroupSize?: number;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/**
 * Binding-free PCF1 helpers. HTP1 accessors and PCM1 membership helpers must
 * be emitted in the same module. All arithmetic mirrors the resident HEAD
 * edge/diagonal expressions and never changes PCM stable-rank invocation.
 */
export function createSparseCM12PersistentPressureCacheWGSL(
  options: SparseCM12PersistentPressureCacheWGSLOptions,
): string {
  const { layout: l } = options;
  const arena = identifier(options.arenaName ?? "pressureCache", "arenaName");
  const cellContains = identifier(options.cellContainsFunction ?? "pcmCellContains",
    "cellContainsFunction");
  const rowContains = identifier(options.rowContainsFunction ?? "pcmRowContains",
    "rowContainsFunction");
  const solidScale = options.solidRowScaleFunction
    ? identifier(options.solidRowScaleFunction, "solidRowScaleFunction") : undefined;
  const workgroupSize = options.workgroupSize ?? 64;
  if (workgroupSize !== 64) throw new RangeError("PCF1 workgroup size is fixed at 64");
  const h = SPARSE_CM12_PRESSURE_CACHE_HEADER;
  const solidScaleHelper = solidScale
    ? `fn pcfSolidRowScale(row:u32)->f32{return ${solidScale}(row);}`
    : "fn pcfSolidRowScale(row:u32)->f32{_=row;return 1.0;}";
  return /* wgsl */ `
const PCF_INVALID:u32=0xffffffffu;
const PCF_MAGIC:u32=0x${SPARSE_CM12_PRESSURE_CACHE_MAGIC.toString(16)}u;
const PCF_VERSION:u32=${SPARSE_CM12_PRESSURE_CACHE_VERSION}u;
const PCF_HEADER_WORDS:u32=${SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS}u;
const PCF_BASE:u32=${l.headerBaseWords}u;
const PCF_CELL_COUNT:u32=${l.cellCount}u;
const PCF_ROW_COUNT:u32=${l.rowCount}u;
const PCF_EDGE_COUNT:u32=${l.directedEdgeCount}u;
const PCF_LEAF_BITS:u32=${SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS}u;
const PCF_LEAF_COUNT:u32=${l.dirtyLeafCount}u;
const PCF_THETA:u32=${l.thetaBaseWords}u;
const PCF_EDGES:u32=${l.effectiveEdgeBaseWords}u;
const PCF_DIAGONAL:u32=${l.diagonalBaseWords}u;
const PCF_DIRTY_TOKENS:u32=${l.dirtyTokenBaseWords}u;
const PCF_DIRTY_LEAF_STAMPS:u32=${l.dirtyLeafStampBaseWords}u;
const PCF_DIRTY_LEAF_LIST:u32=${l.dirtyLeafListBaseWords}u;
const PCF_FLAG_REQUIRED:u32=${SPARSE_CM12_PRESSURE_CACHE_FLAG.complete
  | SPARSE_CM12_PRESSURE_CACHE_FLAG.validated}u;
const PCF_PHASE_UNINITIALIZED:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.uninitialized}u;
const PCF_PHASE_ACCEPTED:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.accepted}u;
const PCF_PHASE_COLLECTING:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.collecting}u;
const PCF_PHASE_REPAIRING:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.repairing}u;
const PCF_PHASE_FAULT:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.fault}u;
const PCF_FAULT_HEADER:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidHeader}u;
const PCF_FAULT_GENERATION:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.generationExhausted}u;
const PCF_FAULT_PHASE:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidPhase}u;
const PCF_FAULT_CELL:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidCell}u;
const PCF_FAULT_ROW:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidRow}u;
const PCF_FAULT_TOPOLOGY:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidTopology}u;
const PCF_FAULT_CAPACITY:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.dirtyCapacity}u;
const PCF_FAULT_COVERAGE:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.coverageGap}u;
const PCF_FAULT_REPAIR:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.repairGap}u;
const PCF_FAULT_NONFINITE:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.nonFiniteCoefficient}u;
const PCF_CAUSE_CELL_MEMBERSHIP:u32=${SPARSE_CM12_PRESSURE_CACHE_CAUSE.cellMembership}u;
const PCF_CAUSE_ROW_MEMBERSHIP:u32=${SPARSE_CM12_PRESSURE_CACHE_CAUSE.rowMembership}u;
const PCF_CAUSE_THETA:u32=${SPARSE_CM12_PRESSURE_CACHE_CAUSE.theta}u;
const PCF_CAUSE_TOPOLOGY:u32=${SPARSE_CM12_PRESSURE_CACHE_CAUSE.topology}u;
const PCF_CAUSE_SOLID:u32=${SPARSE_CM12_PRESSURE_CACHE_CAUSE.solidCoefficient}u;
const PCF_CAUSE_BOOTSTRAP:u32=${SPARSE_CM12_PRESSURE_CACHE_CAUSE.bootstrap}u;
const PCF_H_MAGIC:u32=${h.magic}u;const PCF_H_VERSION:u32=${h.version}u;
const PCF_H_HEADER_WORDS:u32=${h.headerWords}u;const PCF_H_FLAGS:u32=${h.flags}u;
const PCF_H_CELL_COUNT:u32=${h.cellCount}u;const PCF_H_ROW_COUNT:u32=${h.rowCount}u;
const PCF_H_EDGE_COUNT:u32=${h.directedEdgeCount}u;const PCF_H_LEAF_BITS:u32=${h.leafBits}u;
const PCF_H_PHASE:u32=${h.phase}u;const PCF_H_CANDIDATE_GEN:u32=${h.candidateGeneration}u;
const PCF_H_ACCEPTED_GEN:u32=${h.acceptedGeneration}u;const PCF_H_FAULT:u32=${h.fault}u;
const PCF_H_FIRST_FAULT:u32=${h.firstFaultId}u;
const PCF_H_DIRTY_LEAVES:u32=${h.dirtyLeafCount}u;
const PCF_H_INDIRECT_X:u32=${h.repairIndirectX}u;
const PCF_H_DIRECT_EVENTS:u32=${h.directEventCount}u;
const PCF_H_CLOSURE_WRITES:u32=${h.closureWriteCount}u;
const PCF_H_CAUSES:u32=${h.causeMask}u;
const PCF_H_EXPECTED_EVENTS:u32=${h.expectedEventCount}u;
const PCF_H_COVERED_EVENTS:u32=${h.coveredEventCount}u;
const PCF_H_REPAIRED_LEAVES:u32=${h.repairedLeafCount}u;
const PCF_H_CHANGED_EDGES:u32=${h.changedEdgeCount}u;
const PCF_H_CHANGED_DIAGONALS:u32=${h.changedDiagonalCount}u;

${solidScaleHelper}

fn pcfFinite(value:f32)->bool{return value==value&&abs(value)<=3.402823466e38;}
fn pcfHeaderValid()->bool{
  return arrayLength(&${arena})>=${l.bufferSizeWords}u&&cm12HotHeaderValid()
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_MAGIC])==PCF_MAGIC
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_VERSION])==PCF_VERSION
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_HEADER_WORDS])==PCF_HEADER_WORDS
    &&(atomicLoad(&${arena}[PCF_BASE+PCF_H_FLAGS])&PCF_FLAG_REQUIRED)==PCF_FLAG_REQUIRED
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_CELL_COUNT])==PCF_CELL_COUNT
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_ROW_COUNT])==PCF_ROW_COUNT
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_EDGE_COUNT])==PCF_EDGE_COUNT
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_LEAF_BITS])==PCF_LEAF_BITS;
}
fn pcfFault(code:u32,id:u32){
  let claim=atomicCompareExchangeWeak(&${arena}[PCF_BASE+PCF_H_FAULT],0u,code);
  if(claim.exchanged){atomicStore(&${arena}[PCF_BASE+PCF_H_FIRST_FAULT],id);}
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCF_PHASE_FAULT);
  atomicStore(&${arena}[PCF_BASE+PCF_H_INDIRECT_X],0u);
  pcfaZeroIndirects();
}
fn pcfBegin(expectedEvents:u32)->bool{
  if(!pcfHeaderValid()){pcfFault(PCF_FAULT_HEADER,PCF_INVALID);return false;}
  let phase=atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE]);
  if(phase!=PCF_PHASE_UNINITIALIZED&&phase!=PCF_PHASE_ACCEPTED){
    pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return false;}
  let accepted=atomicLoad(&${arena}[PCF_BASE+PCF_H_ACCEPTED_GEN]);
  if(accepted>=0x7ffffffeu){pcfFault(PCF_FAULT_GENERATION,PCF_INVALID);return false;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN],accepted+1u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_FAULT],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_FIRST_FAULT],PCF_INVALID);
  atomicStore(&${arena}[PCF_BASE+PCF_H_DIRTY_LEAVES],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_INDIRECT_X],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_DIRECT_EVENTS],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_CLOSURE_WRITES],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_CAUSES],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_EXPECTED_EVENTS],expectedEvents);
  atomicStore(&${arena}[PCF_BASE+PCF_H_COVERED_EVENTS],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_REPAIRED_LEAVES],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_CHANGED_EDGES],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_CHANGED_DIAGONALS],0u);
  if(!pcfaBegin()){return false;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCF_PHASE_COLLECTING);
  return true;
}
fn pcfMarkCell(cell:u32,cause:u32,closure:bool)->bool{
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCF_PHASE_COLLECTING){
    pcfFault(PCF_FAULT_PHASE,cell);return false;}
  if(cell>=PCF_CELL_COUNT){pcfFault(PCF_FAULT_CELL,cell);return false;}
  let generation=atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]);
  if(atomicExchange(&${arena}[PCF_DIRTY_TOKENS+cell],generation)==generation){return true;}
  if(closure){atomicAdd(&${arena}[PCF_BASE+PCF_H_CLOSURE_WRITES],1u);}
  atomicOr(&${arena}[PCF_BASE+PCF_H_CAUSES],cause);
  let leaf=cell/PCF_LEAF_BITS;
  if(atomicExchange(&${arena}[PCF_DIRTY_LEAF_STAMPS+leaf],generation)!=generation){
    let slot=atomicAdd(&${arena}[PCF_BASE+PCF_H_DIRTY_LEAVES],1u);
    if(slot>=PCF_LEAF_COUNT){pcfFault(PCF_FAULT_CAPACITY,cell);return false;}
    atomicStore(&${arena}[PCF_DIRTY_LEAF_LIST+slot],leaf);
  }
  return true;
}
fn pcfMarkRowCells(row:u32,cause:u32)->bool{
  if(row>=PCF_ROW_COUNT||!cm12HotRowValid(row)){pcfFault(PCF_FAULT_ROW,row);return false;}
  let count=cm12HotRowTermCount(row);
  for(var ordinal=0u;ordinal<count;ordinal+=1u){
    let cell=cm12HotRowTermCell(row,ordinal);
    if(cell==PCF_INVALID||!pcfMarkCell(cell,cause,true)){return false;}
  }
  return true;
}
fn pcfRecordCellMembershipEvent(cell:u32)->bool{
  atomicAdd(&${arena}[PCF_BASE+PCF_H_EXPECTED_EVENTS],1u);
  atomicAdd(&${arena}[PCF_BASE+PCF_H_DIRECT_EVENTS],1u);
  if(!pcfMarkCell(cell,PCF_CAUSE_CELL_MEMBERSHIP,false)){return false;}
  let range=cm12HotIncidenceRange(cell);
  if(range.x==PCF_INVALID){pcfFault(PCF_FAULT_TOPOLOGY,cell);return false;}
  for(var local=0u;local<range.y;local+=1u){
    let incidence=cm12HotIncidence(range.x+local);
    if(incidence.x==PCF_INVALID
      ||!pcfMarkRowCells(incidence.x,PCF_CAUSE_CELL_MEMBERSHIP)){return false;}
  }
  atomicAdd(&${arena}[PCF_BASE+PCF_H_COVERED_EVENTS],1u);return true;
}
fn pcfRecordTopologyCellEvent(cell:u32)->bool{
  atomicAdd(&${arena}[PCF_BASE+PCF_H_EXPECTED_EVENTS],1u);
  atomicAdd(&${arena}[PCF_BASE+PCF_H_DIRECT_EVENTS],1u);
  if(!pcfMarkCell(cell,PCF_CAUSE_TOPOLOGY,false)){return false;}
  let range=cm12HotIncidenceRange(cell);
  if(range.x==PCF_INVALID){pcfFault(PCF_FAULT_TOPOLOGY,cell);return false;}
  for(var local=0u;local<range.y;local+=1u){
    let incidence=cm12HotIncidence(range.x+local);
    if(incidence.x==PCF_INVALID||!pcfMarkRowCells(incidence.x,PCF_CAUSE_TOPOLOGY)){
      return false;}
  }
  atomicAdd(&${arena}[PCF_BASE+PCF_H_COVERED_EVENTS],1u);return true;
}
fn pcfRecordRowEvent(row:u32,cause:u32)->bool{
  atomicAdd(&${arena}[PCF_BASE+PCF_H_EXPECTED_EVENTS],1u);
  atomicAdd(&${arena}[PCF_BASE+PCF_H_DIRECT_EVENTS],1u);
  if(!pcfMarkRowCells(row,cause)){return false;}
  atomicAdd(&${arena}[PCF_BASE+PCF_H_COVERED_EVENTS],1u);return true;
}
fn pcfRecordRowMembershipEvent(row:u32)->bool{
  return pcfRecordRowEvent(row,PCF_CAUSE_ROW_MEMBERSHIP);
}
fn pcfRecordSolidRowEvent(row:u32)->bool{return pcfRecordRowEvent(row,PCF_CAUSE_SOLID);}
fn pcfStoreThetaAndRecord(row:u32,theta:f32)->bool{
  if(row>=PCF_ROW_COUNT||!pcfFinite(theta)||theta<0.0){pcfFault(PCF_FAULT_ROW,row);return false;}
  atomicAdd(&${arena}[PCF_BASE+PCF_H_EXPECTED_EVENTS],1u);
  let changed=atomicExchange(&${arena}[PCF_THETA+row],bitcast<u32>(theta))!=bitcast<u32>(theta);
  atomicAdd(&${arena}[PCF_BASE+PCF_H_DIRECT_EVENTS],1u);
  if(changed&&!pcfMarkRowCells(row,PCF_CAUSE_THETA)){return false;}
  atomicAdd(&${arena}[PCF_BASE+PCF_H_COVERED_EVENTS],1u);return true;
}
fn pcfFinalizeFrontier()->bool{
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCF_PHASE_COLLECTING){return false;}
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])!=0u){return false;}
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_EXPECTED_EVENTS])
    !=atomicLoad(&${arena}[PCF_BASE+PCF_H_COVERED_EVENTS])){
    pcfFault(PCF_FAULT_COVERAGE,PCF_INVALID);return false;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_INDIRECT_X],
    atomicLoad(&${arena}[PCF_BASE+PCF_H_DIRTY_LEAVES]));
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCF_PHASE_REPAIRING);return true;
}
fn pcfEdgeWeight(edge:u32)->f32{
  return bitcast<f32>(atomicLoad(&${arena}[PCF_EDGES+edge]));
}
fn pcfStoreEdgeWeight(edge:u32,value:f32){
  atomicStore(&${arena}[PCF_EDGES+edge],bitcast<u32>(value));
}
fn pcfDiagonal(cell:u32)->f32{
  return bitcast<f32>(atomicLoad(&${arena}[PCF_DIAGONAL+cell]));
}
fn pcfCandidateGeneration()->u32{
  return atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]);
}
fn pcfAcceptedGeneration()->u32{
  return atomicLoad(&${arena}[PCF_BASE+PCF_H_ACCEPTED_GEN]);
}
fn pcfRepairCell(cell:u32){
  let isActive=${cellContains}(cell);
  let edgeRange=cm12HotDirectedEdgeRange(cell);
  if(edgeRange.x==PCF_INVALID){pcfFault(PCF_FAULT_TOPOLOGY,cell);return;}
  for(var local=0u;local<edgeRange.y;local+=1u){
    let edgeId=edgeRange.x+local;let edge=cm12HotDirectedEdge(edgeId);
    if(edge.x==PCF_INVALID){pcfFault(PCF_FAULT_TOPOLOGY,edgeId);return;}
    let theta=bitcast<f32>(atomicLoad(&${arena}[PCF_THETA+edge.y]));var weight=0.0;
    if(isActive&&${cellContains}(edge.x)&&${rowContains}(edge.y)&&theta>0.0){
      weight=bitcast<f32>(edge.z)/theta;
      weight*=pcfSolidRowScale(edge.y);
    }
    if(!pcfFinite(weight)){pcfFault(PCF_FAULT_NONFINITE,edgeId);return;}
    let old=atomicExchange(&${arena}[PCF_EDGES+edgeId],bitcast<u32>(weight));
    let changed=old!=bitcast<u32>(weight);
    if(changed){atomicAdd(&${arena}[PCF_BASE+PCF_H_CHANGED_EDGES],1u);}
    if(changed||PCFA_QA){pcfAggregateFineEdgeChanged(cell,edgeId);}
  }
  var diagonal=0.0;
  if(isActive){
    let range=cm12HotIncidenceRange(cell);
    if(range.x==PCF_INVALID){pcfFault(PCF_FAULT_TOPOLOGY,cell);return;}
    for(var local=0u;local<range.y;local+=1u){
      let incidence=cm12HotIncidence(range.x+local);let row=incidence.x;
      if(row==PCF_INVALID){pcfFault(PCF_FAULT_TOPOLOGY,cell);return;}
      let theta=bitcast<f32>(atomicLoad(&${arena}[PCF_THETA+row]));
      if(!${rowContains}(row)||theta<=0.0){continue;}
      let coefficient=cm12HotRowTermCoefficient(row,incidence.y);
      diagonal+=cm12HotRowDualWeight(row)*coefficient*coefficient/theta;
    }
  }
  if(!pcfFinite(diagonal)){pcfFault(PCF_FAULT_NONFINITE,cell);return;}
  let old=atomicExchange(&${arena}[PCF_DIAGONAL+cell],bitcast<u32>(diagonal));
  let changed=old!=bitcast<u32>(diagonal);
  if(changed){atomicAdd(&${arena}[PCF_BASE+PCF_H_CHANGED_DIAGONALS],1u);}
  if(changed||PCFA_QA){pcfAggregateFineDiagonalChanged(cell);}
}

@compute @workgroup_size(64)
fn repairPersistentPressureCache(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  let validPhase=atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])==PCF_PHASE_REPAIRING
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])==0u;
  let dirtyCount=atomicLoad(&${arena}[PCF_BASE+PCF_H_DIRTY_LEAVES]);
  let invocationValid=wid.x<dirtyCount;var leaf=0u;
  if(invocationValid){leaf=atomicLoad(&${arena}[PCF_DIRTY_LEAF_LIST+wid.x]);}
  let valid=validPhase&&invocationValid&&leaf<PCF_LEAF_COUNT;
  if(valid){
    let generation=atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]);
    for(var laneCell=lid.x;laneCell<PCF_LEAF_BITS;laneCell+=64u){
      let cell=leaf*PCF_LEAF_BITS+laneCell;
      if(cell<PCF_CELL_COUNT
        &&atomicLoad(&${arena}[PCF_DIRTY_TOKENS+cell])==generation){pcfRepairCell(cell);}
    }
  }
  workgroupBarrier();
  if(lid.x==0u&&valid){atomicAdd(&${arena}[PCF_BASE+PCF_H_REPAIRED_LEAVES],1u);}
}

${createSparseCM12PersistentPressureCacheAggregateWGSL(l, arena)}
`;
}
