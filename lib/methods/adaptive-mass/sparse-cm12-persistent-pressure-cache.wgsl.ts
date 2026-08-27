import {
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
  /** Unique-owner ordinary f32 edge image. */
  readonly ordinaryEdgeStorage: Readonly<{
    readonly arrayName: string;
    readonly baseWords: number;
  }>;
  /** Construction-time bounds keep cooperative edge-fetch loops uniform. */
  readonly aggregateEdgeMaximumContributionCount: number;
  readonly hierarchyEdgeMaximumContributionCount: number;
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
  const workgroupSize = options.workgroupSize ?? 64;
  if (workgroupSize !== 64) throw new RangeError("PCF1 workgroup size is fixed at 64");
  const h = SPARSE_CM12_PRESSURE_CACHE_HEADER;
  const edgeStorage = options.ordinaryEdgeStorage;
  const edgeArray = identifier(edgeStorage.arrayName, "ordinaryEdgeStorage.arrayName");
  const edgeHelpers = /* wgsl */ `
fn pcfEdgeWeight(edge:u32)->f32{return ${edgeArray}[${edgeStorage.baseWords}u+edge];}
fn pcfExchangeEdgeWeight(edge:u32,value:f32)->u32{
  let address=${edgeStorage.baseWords}u+edge;
  let old=bitcast<u32>(${edgeArray}[address]);${edgeArray}[address]=value;return old;
}`;
  return /* wgsl */ `
const PCF_INVALID:u32=0xffffffffu;
const PCF_MAGIC:u32=0x${SPARSE_CM12_PRESSURE_CACHE_MAGIC.toString(16)}u;
const PCF_VERSION:u32=${SPARSE_CM12_PRESSURE_CACHE_VERSION}u;
const PCF_HEADER_WORDS:u32=${SPARSE_CM12_PRESSURE_CACHE_HEADER_WORDS}u;
const PCF_BASE:u32=${l.headerBaseWords}u;
const PCF_CELL_COUNT:u32=${l.cellCount}u;
const PCF_EDGE_COUNT:u32=${l.directedEdgeCount}u;
const PCF_FLAG_REQUIRED:u32=${SPARSE_CM12_PRESSURE_CACHE_FLAG.complete
  | SPARSE_CM12_PRESSURE_CACHE_FLAG.validated}u;
const PCF_PHASE_UNINITIALIZED:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.uninitialized}u;
const PCF_PHASE_ACCEPTED:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.accepted}u;
const PCF_PHASE_COLLECTING:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.collecting}u;
const PCF_PHASE_FAULT:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.fault}u;
const PCF_FAULT_HEADER:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidHeader}u;
const PCF_FAULT_GENERATION:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.generationExhausted}u;
const PCF_FAULT_PHASE:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidPhase}u;
const PCF_FAULT_TOPOLOGY:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.invalidTopology}u;
const PCF_FAULT_CAPACITY:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.dirtyCapacity}u;
const PCF_FAULT_REPAIR:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.repairGap}u;
const PCF_FAULT_NONFINITE:u32=${SPARSE_CM12_PRESSURE_CACHE_FAULT.nonFiniteCoefficient}u;
const PCF_H_MAGIC:u32=${h.magic}u;const PCF_H_VERSION:u32=${h.version}u;
const PCF_H_HEADER_WORDS:u32=${h.headerWords}u;const PCF_H_FLAGS:u32=${h.flags}u;
const PCF_H_CELL_COUNT:u32=${h.cellCount}u;const PCF_H_ROW_COUNT:u32=${h.rowCount}u;
const PCF_H_EDGE_COUNT:u32=${h.directedEdgeCount}u;const PCF_H_LEAF_BITS:u32=${h.leafBits}u;
const PCF_H_PHASE:u32=${h.phase}u;const PCF_H_CANDIDATE_GEN:u32=${h.candidateGeneration}u;
const PCF_H_ACCEPTED_GEN:u32=${h.acceptedGeneration}u;const PCF_H_FAULT:u32=${h.fault}u;
const PCF_H_FIRST_FAULT:u32=${h.firstFaultId}u;
const PCF_H_CHANGED_EDGES:u32=${h.changedEdgeCount}u;
const PCF_H_CHANGED_DIAGONALS:u32=${h.changedDiagonalCount}u;

fn pcfFinite(value:f32)->bool{return value==value&&abs(value)<=3.402823466e38;}
fn pcfHeaderValid()->bool{
  return arrayLength(&${arena})>=${l.bufferSizeWords}u&&cm12HotHeaderValid()
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_MAGIC])==PCF_MAGIC
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_VERSION])==PCF_VERSION
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_HEADER_WORDS])==PCF_HEADER_WORDS
    &&(atomicLoad(&${arena}[PCF_BASE+PCF_H_FLAGS])&PCF_FLAG_REQUIRED)==PCF_FLAG_REQUIRED
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_CELL_COUNT])==PCF_CELL_COUNT
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_ROW_COUNT])==${l.rowCount}u
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_EDGE_COUNT])==PCF_EDGE_COUNT
    &&atomicLoad(&${arena}[PCF_BASE+PCF_H_LEAF_BITS])==${SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS}u;
}
fn pcfFault(code:u32,id:u32){
  let claim=atomicCompareExchangeWeak(&${arena}[PCF_BASE+PCF_H_FAULT],0u,code);
  if(claim.exchanged){atomicStore(&${arena}[PCF_BASE+PCF_H_FIRST_FAULT],id);}
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCF_PHASE_FAULT);
  pcfaZeroIndirects();
}
fn pcfBegin()->bool{
  if(!pcfHeaderValid()){pcfFault(PCF_FAULT_HEADER,PCF_INVALID);return false;}
  let phase=atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE]);
  if(phase!=PCF_PHASE_UNINITIALIZED&&phase!=PCF_PHASE_ACCEPTED){
    pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return false;}
  let accepted=atomicLoad(&${arena}[PCF_BASE+PCF_H_ACCEPTED_GEN]);
  if(accepted>=0x7ffffffeu){pcfFault(PCF_FAULT_GENERATION,PCF_INVALID);return false;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN],accepted+1u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_FAULT],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_FIRST_FAULT],PCF_INVALID);
  atomicStore(&${arena}[PCF_BASE+PCF_H_CHANGED_EDGES],0u);
  atomicStore(&${arena}[PCF_BASE+PCF_H_CHANGED_DIAGONALS],0u);
  if(!pcfaBeginReceiptOnly()){return false;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCF_PHASE_COLLECTING);
  return true;
}
fn pcfFinalizeFine()->bool{
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCF_PHASE_COLLECTING
    ||atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])!=0u){
    pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return false;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_ACCEPTED_GEN],
    atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]));
  pcfaFinalizeReceiptOnly();
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCF_PHASE_ACCEPTED);
  return true;
}
fn pcfFinePublicationOpen()->bool{return atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])
  ==PCF_PHASE_COLLECTING&&atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])==0u;}
${edgeHelpers}
fn pcfCandidateGeneration()->u32{
  return atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]);
}
fn pcfAcceptedGeneration()->u32{
  return atomicLoad(&${arena}[PCF_BASE+PCF_H_ACCEPTED_GEN]);
}
${createSparseCM12PersistentPressureCacheAggregateWGSL(l, arena, {
    aggregateEdgeMaximumContributionCount:
      options.aggregateEdgeMaximumContributionCount,
    hierarchyEdgeMaximumContributionCount:
      options.hierarchyEdgeMaximumContributionCount,
  })}
`;
}
