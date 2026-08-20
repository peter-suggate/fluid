import {
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_MAGIC,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_MODE,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE,
  SPARSE_CM12_PRESSURE_ADDRESSING_AB_VERSION,
  type SparseCM12PressureAddressingABLayout,
  type SparseCM12PressureAddressingABModeName,
} from "./sparse-cm12-pressure-addressing-ab";

export interface SparseCM12PressureAddressingABWGSLOptions {
  readonly layout: SparseCM12PressureAddressingABLayout;
  /** Existing `array<atomic<u32>>`; QA construction appends PAB1 here. */
  readonly arenaName?: string;
  readonly workgroupSize?: 64;
  /** Production fixes list addressing in WGSL; only paired QA uses an override. */
  readonly fixedMode?: SparseCM12PressureAddressingABModeName;
}

const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`Invalid PAB1 WGSL identifier ${value}`);
  }
  return value;
};

/**
 * Requires the resident source to provide PCM's accepted count/generation,
 * rank-select and contains helpers plus `cellActive`.  The solve kernels call
 * only `pabPressureCellAddress`; their arithmetic source is otherwise shared.
 */
export function createSparseCM12PressureAddressingABWGSL(
  options: SparseCM12PressureAddressingABWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "activity");
  const h = SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER;
  const at = (word: number) => `${layout.baseWords + word}u`;
  const modeDeclaration = options.fixedMode === undefined
    ? "override CM12_PRESSURE_ADDRESS_MODE:u32=PAB_MODE_RANK_SELECT;"
    : `const CM12_PRESSURE_ADDRESS_MODE:u32=${
      SPARSE_CM12_PRESSURE_ADDRESSING_AB_MODE[options.fixedMode]}u;`;
  // The rank-select verifier is a construction oracle. Production already
  // consumes the materialized list fail-closed by generation/count receipt;
  // repeating rank-select over the whole pressure domain would restore the
  // very addressing cost this cutover removes.
  const verifyCountClause = options.fixedMode === undefined
    ? `||pabLoad(${at(h.verifiedExecutions)})!=count`
    : "";
  const verifyReceiptClause = options.fixedMode === undefined
    ? `
  if(pabLoad(${at(h.mismatchCount)})!=0u
    ||pabLoad(${at(h.materializedHash)})!=pabLoad(${at(h.verifiedHash)})){
    pabFail(PAB_FAULT_LIST,pabLoad(${at(h.firstMismatchRank)}));return;
  }`
    : "";
  return /* wgsl */ `
const PAB_MAGIC:u32=0x${SPARSE_CM12_PRESSURE_ADDRESSING_AB_MAGIC.toString(16)}u;
const PAB_VERSION:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_VERSION}u;
const PAB_HEADER_WORDS:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_HEADER_WORDS}u;
const PAB_INVALID:u32=0x${SPARSE_CM12_PRESSURE_ADDRESSING_AB_INVALID.toString(16)}u;
const PAB_LIST_BASE:u32=${layout.listBaseWords}u;
const PAB_LIST_CAPACITY:u32=${layout.cellCapacity}u;
const PAB_PHASE_UNINITIALIZED:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.uninitialized}u;
const PAB_PHASE_COLLECTING:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.collecting}u;
const PAB_PHASE_ACCEPTED:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.accepted}u;
const PAB_PHASE_FAULT:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_PHASE.fault}u;
const PAB_FAULT_INVALID_HEADER:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.invalidHeader}u;
const PAB_FAULT_INVALID_PHASE:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.invalidPhase}u;
const PAB_FAULT_GENERATION:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.generationMismatch}u;
const PAB_FAULT_COUNT:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.countMismatch}u;
const PAB_FAULT_CAPACITY:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.capacity}u;
const PAB_FAULT_RANK:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.invalidRankSelect}u;
const PAB_FAULT_LIST:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.listMismatch}u;
const PAB_FAULT_ORDER:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_FAULT.nonCanonicalOrder}u;
const PAB_MODE_RANK_SELECT:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_MODE.canonicalRankSelect}u;
const PAB_MODE_LIST:u32=${SPARSE_CM12_PRESSURE_ADDRESSING_AB_MODE.materializedList}u;
${modeDeclaration}

fn pabLoad(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn pabStore(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn pabHeaderValid()->bool{
  return arrayLength(&${arena})>=${layout.totalWords}u
    &&pabLoad(${at(h.magic)})==PAB_MAGIC&&pabLoad(${at(h.version)})==PAB_VERSION
    &&pabLoad(${at(h.headerWords)})==PAB_HEADER_WORDS
    &&pabLoad(${at(h.listCapacity)})==PAB_LIST_CAPACITY
    &&pabLoad(${at(h.listBaseWords)})==PAB_LIST_BASE;
}
fn pabFail(code:u32,rank:u32){
  let claimed=atomicCompareExchangeWeak(&${arena}[${at(h.fault)}],0u,code);
  if(claimed.exchanged){pabStore(${at(h.firstFaultRank)},rank);}
  pabStore(${at(h.materializeIndirectX)},0u);
  pabStore(${at(h.phase)},PAB_PHASE_FAULT);
}
fn pabHash(rank:u32,cell:u32)->u32{
  var value=cell^(rank*0x9e3779b9u);value^=value>>16u;value*=0x85ebca6bu;
  value^=value>>13u;value*=0xc2b2ae35u;return value^(value>>16u);
}

@compute @workgroup_size(1)
fn beginSparseCM12PressureAddressMaterialization(){
  if(!pabHeaderValid()){pabFail(PAB_FAULT_INVALID_HEADER,PAB_INVALID);return;}
  let phase=pabLoad(${at(h.phase)});
  if(phase!=PAB_PHASE_UNINITIALIZED&&phase!=PAB_PHASE_ACCEPTED){
    pabFail(PAB_FAULT_INVALID_PHASE,PAB_INVALID);return;
  }
  let generation=pcmCellAcceptedGeneration();let count=pcmCellAcceptedCount();
  if(generation==0u){pabFail(PAB_FAULT_GENERATION,PAB_INVALID);return;}
  if(count>PAB_LIST_CAPACITY){pabFail(PAB_FAULT_CAPACITY,count);return;}
  pabStore(${at(h.fault)},0u);pabStore(${at(h.firstFaultRank)},PAB_INVALID);
  pabStore(${at(h.expectedPCMGeneration)},generation);
  pabStore(${at(h.materializedPCMGeneration)},0u);
  pabStore(${at(h.expectedCount)},count);pabStore(${at(h.materializedCount)},0u);
  pabStore(${at(h.materializedExecutions)},0u);pabStore(${at(h.verifiedExecutions)},0u);
  pabStore(${at(h.mismatchCount)},0u);pabStore(${at(h.firstMismatchRank)},PAB_INVALID);
  pabStore(${at(h.firstExpectedCell)},PAB_INVALID);pabStore(${at(h.firstActualCell)},PAB_INVALID);
  pabStore(${at(h.materializedHash)},0u);pabStore(${at(h.verifiedHash)},0u);
  pabStore(${at(h.materializeIndirectX)},(count+63u)/64u);
  pabStore(${at(h.materializeIndirectY)},1u);pabStore(${at(h.materializeIndirectZ)},1u);
  pabStore(${at(h.phase)},PAB_PHASE_COLLECTING);
}

@compute @workgroup_size(64)
fn materializeSparseCM12PressureCellAddresses(@builtin(global_invocation_id)gid:vec3u){
  let rank=gid.x;if(rank>=pabLoad(${at(h.expectedCount)})){return;}
  if(pabLoad(${at(h.phase)})!=PAB_PHASE_COLLECTING){return;}
  let cell=pcmCellRankSelect(rank);
  if(cell==PAB_INVALID||cell>=PAB_LIST_CAPACITY||!pcmCellContains(cell)||!cellActive(cell)){
    pabFail(PAB_FAULT_RANK,rank);return;
  }
  pabStore(PAB_LIST_BASE+rank,cell);
  atomicXor(&${arena}[${at(h.materializedHash)}],pabHash(rank,cell));
  atomicAdd(&${arena}[${at(h.materializedExecutions)}],1u);
}

fn pabRecordMismatch(rank:u32,expected:u32,actual:u32,code:u32){
  atomicAdd(&${arena}[${at(h.mismatchCount)}],1u);
  let first=atomicMin(&${arena}[${at(h.firstMismatchRank)}],rank);
  if(rank<first){pabStore(${at(h.firstExpectedCell)},expected);
    pabStore(${at(h.firstActualCell)},actual);}
  pabFail(code,rank);
}

@compute @workgroup_size(64)
fn verifySparseCM12PressureCellAddresses(@builtin(global_invocation_id)gid:vec3u){
  let rank=gid.x;if(rank>=pabLoad(${at(h.expectedCount)})){return;}
  if(pabLoad(${at(h.phase)})!=PAB_PHASE_COLLECTING){return;}
  let expected=pcmCellRankSelect(rank);let actual=pabLoad(PAB_LIST_BASE+rank);
  if(expected==PAB_INVALID||actual!=expected){pabRecordMismatch(rank,expected,actual,PAB_FAULT_LIST);return;}
  if(rank>0u&&pabLoad(PAB_LIST_BASE+rank-1u)>=actual){
    pabRecordMismatch(rank,expected,actual,PAB_FAULT_ORDER);return;
  }
  atomicXor(&${arena}[${at(h.verifiedHash)}],pabHash(rank,actual));
  atomicAdd(&${arena}[${at(h.verifiedExecutions)}],1u);
}

@compute @workgroup_size(1)
fn finalizeSparseCM12PressureAddressMaterialization(){
  if(pabLoad(${at(h.phase)})!=PAB_PHASE_COLLECTING){return;}
  let generation=pcmCellAcceptedGeneration();let count=pcmCellAcceptedCount();
  if(generation!=pabLoad(${at(h.expectedPCMGeneration)})){
    pabFail(PAB_FAULT_GENERATION,PAB_INVALID);return;
  }
  if(count!=pabLoad(${at(h.expectedCount)})
    ||pabLoad(${at(h.materializedExecutions)})!=count${verifyCountClause}){
    pabFail(PAB_FAULT_COUNT,count);return;
  }
${verifyReceiptClause}
  pabStore(${at(h.materializedPCMGeneration)},generation);
  pabStore(${at(h.materializedCount)},count);
  atomicAdd(&${arena}[${at(h.acceptedReceipts)}],1u);
  pabStore(${at(h.phase)},PAB_PHASE_ACCEPTED);
}

// Compile-time only. The list arm returns INVALID on any receipt/generation
// gap; it never falls back to canonical rank select.
fn pabPressureCellAddress(rank:u32)->u32{
  if(CM12_PRESSURE_ADDRESS_MODE==PAB_MODE_RANK_SELECT){
    return pcmCellRankSelect(rank);
  }
  if(CM12_PRESSURE_ADDRESS_MODE!=PAB_MODE_LIST
    ||pabLoad(${at(h.phase)})!=PAB_PHASE_ACCEPTED
    ||pabLoad(${at(h.fault)})!=0u
    ||pabLoad(${at(h.materializedPCMGeneration)})!=pcmCellAcceptedGeneration()
    ||rank>=pabLoad(${at(h.materializedCount)})){return PAB_INVALID;}
  return pabLoad(PAB_LIST_BASE+rank);
}
@compute @workgroup_size(64)
fn exerciseSparseCM12PressureAddressing(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x<pcmCellAcceptedCount()){_=pabPressureCellAddress(gid.x);}
}
`;
}
