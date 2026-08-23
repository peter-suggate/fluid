import {
  SPARSE_CM12_CANONICAL_MEMBERSHIP_BRANCH,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER_WORDS,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_MAGIC,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE,
  SPARSE_CM12_CANONICAL_MEMBERSHIP_VERSION,
  type SparseCM12CanonicalMembershipDomainLayout,
  type SparseCM12CanonicalMembershipLayout,
  type SparseCM12CanonicalRowImageLayout,
} from "./sparse-cm12-canonical-membership";

export interface SparseCM12CanonicalMembershipWGSLOptions {
  readonly layout: SparseCM12CanonicalMembershipLayout;
  /** Name of an `array<atomic<u32>>` storage variable. */
  readonly arenaName?: string;
  readonly workgroupSize?: number;
}

function domainConstants(
  prefix: string,
  domain: SparseCM12CanonicalMembershipDomainLayout,
): string {
  return /* wgsl */ `
const PCM_${prefix}_HEADER:u32=${domain.headerBaseWords}u;
const PCM_${prefix}_CAPACITY:u32=${domain.capacity}u;
const PCM_${prefix}_ACTIVE_BITS:u32=${domain.activeBitsBaseWords}u;
const PCM_${prefix}_CANDIDATE_TOKENS:u32=${domain.candidateTokenBaseWords}u;
const PCM_${prefix}_DIRTY_STAMPS:u32=${domain.dirtyStampBaseWords}u;
const PCM_${prefix}_DIRTY_LIST:u32=${domain.dirtyListBaseWords}u;
const PCM_${prefix}_LEAF_COUNT:u32=${domain.leafCount}u;
${domain.treeLevelBaseWords.map((base, level) =>
    `const PCM_${prefix}_TREE_${level}:u32=${base}u;`).join("\n")}
`;
}

function directRowConstants(
  domain: SparseCM12CanonicalRowImageLayout,
): string {
  return /* wgsl */ `
const PCM_ROW_HEADER:u32=${domain.headerBaseWords}u;
const PCM_ROW_CAPACITY:u32=${domain.capacity}u;
const PCM_ROW_ACTIVE_BITS:u32=${domain.activeBitsBaseWords}u;
const PCM_ROW_ACTIVE_WORDS:u32=${domain.activeBitWordCount}u;
`;
}

function rankSelectBody(prefix: string, label: string,
  domain: SparseCM12CanonicalMembershipDomainLayout, arena: string): string {
  const descend = Array.from({ length: domain.treeLevelCounts.length - 1 }, (_, at) => {
    const level = domain.treeLevelCounts.length - 2 - at;
    return /* wgsl */ `
  {
    let childBegin=node*PCM_BRANCH;
    var selected=PCM_INVALID;
    for(var child=0u;child<PCM_BRANCH;child+=1u){
      let candidate=childBegin+child;
      if(candidate>=${domain.treeLevelCounts[level]}u){break;}
      let count=atomicLoad(&${arena}[PCM_${prefix}_TREE_${level}+candidate]);
      if(selected==PCM_INVALID&&remaining<count){selected=candidate;}
      else if(selected==PCM_INVALID){remaining-=count;}
    }
    if(selected==PCM_INVALID){return PCM_INVALID;}
    node=selected;
  }`;
  }).join("\n");
  return /* wgsl */ `
fn pcm${label}RankSelect(rank:u32)->u32{
  let phase=atomicLoad(&${arena}[PCM_${prefix}_HEADER+PCM_D_PHASE]);
  if(phase!=PCM_PHASE_ACCEPTED&&phase!=PCM_PHASE_COLLECTING){
    return PCM_INVALID;
  }
  let total=atomicLoad(&${arena}[PCM_${prefix}_HEADER+PCM_D_TOTAL]);
  if(rank>=total){return PCM_INVALID;}
  var remaining=rank;var node=0u;
${descend}
  let firstWord=node*PCM_LEAF_WORDS;
  for(var localWord=0u;localWord<PCM_LEAF_WORDS;localWord+=1u){
    let word=atomicLoad(&${arena}[PCM_${prefix}_ACTIVE_BITS+firstWord+localWord]);
    let count=countOneBits(word);
    if(remaining>=count){remaining-=count;continue;}
    for(var bit=0u;bit<32u;bit+=1u){
      if((word&(1u<<bit))==0u){continue;}
      if(remaining==0u){
        let id=node*PCM_LEAF_BITS+localWord*32u+bit;
        return select(PCM_INVALID,id,id<PCM_${prefix}_CAPACITY);
      }
      remaining-=1u;
    }
  }
  return PCM_INVALID;
}`;
}

function repairBody(prefix: string, label: string,
  domain: SparseCM12CanonicalMembershipDomainLayout, arena: string,
  workgroupSize: number): string {
  const ancestorUpdates = domain.treeLevelBaseWords.slice(1).map((base, level) => /* wgsl */ `
    node/=PCM_BRANCH;
    let previous${level}=atomicAdd(&${arena}[${base}u+node],bitcast<u32>(delta));
    if((delta<0&&previous${level}<u32(-delta))
      ||(delta>0&&previous${level}>0xffffffffu-u32(delta))){
      pcmFault(PCM_${prefix}_HEADER,PCM_FAULT_COUNT_RANGE,leaf);
    }`).join("");
  return /* wgsl */ `
@compute @workgroup_size(${workgroupSize})
fn repairCanonicalPressure${label}Leaves(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  let header=PCM_${prefix}_HEADER;
  let epochValid=atomicLoad(&${arena}[header+PCM_D_PHASE])==PCM_PHASE_REPAIRING
    &&atomicLoad(&${arena}[header+PCM_D_FAULT])==0u;
  let dirtyCount=atomicLoad(&${arena}[header+PCM_D_DIRTY_COUNT]);
  let invocationValid=wid.x<dirtyCount;var leaf=0u;
  if(invocationValid){leaf=atomicLoad(&${arena}[PCM_${prefix}_DIRTY_LIST+wid.x]);}
  let valid=epochValid&&invocationValid&&leaf<PCM_${prefix}_LEAF_COUNT;
  if(lid.x==0u&&epochValid&&invocationValid&&leaf>=PCM_${prefix}_LEAF_COUNT){
    pcmFault(header,PCM_FAULT_INVALID_ID,leaf);
  }
  let generation=atomicLoad(&${arena}[header+PCM_D_CANDIDATE_GENERATION]);
  if(lid.x<PCM_LEAF_WORDS&&valid){
    let wordIndex=leaf*PCM_LEAF_WORDS+lid.x;
    var word=atomicLoad(&${arena}[PCM_${prefix}_ACTIVE_BITS+wordIndex]);
    for(var bit=0u;bit<32u;bit+=1u){
      let id=leaf*PCM_LEAF_BITS+lid.x*32u+bit;
      if(id>=PCM_${prefix}_CAPACITY){word&=~(1u<<bit);continue;}
      let token=atomicLoad(&${arena}[PCM_${prefix}_CANDIDATE_TOKENS+id]);
      if((token>>1u)==generation){
        word=select(word&~(1u<<bit),word|(1u<<bit),(token&1u)!=0u);
      }
    }
    atomicStore(&${arena}[PCM_${prefix}_ACTIVE_BITS+wordIndex],word);
    pcmLeafCounts[lid.x]=countOneBits(word);
  }else if(lid.x<PCM_LEAF_WORDS){
    pcmLeafCounts[lid.x]=0u;
  }
  workgroupBarrier();
  if(lid.x==0u&&valid){
    var newCount=0u;for(var at=0u;at<PCM_LEAF_WORDS;at+=1u){newCount+=pcmLeafCounts[at];}
    let oldCount=atomicExchange(&${arena}[PCM_${prefix}_TREE_0+leaf],newCount);
    let delta=i32(newCount)-i32(oldCount);var node=leaf;
${ancestorUpdates}
  }
}`;
}

export function createSparseCM12CanonicalMembershipWGSL(
  options: SparseCM12CanonicalMembershipWGSLOptions,
): string {
  const arena = options.arenaName ?? "canonicalMembershipArena";
  const workgroupSize = options.workgroupSize ?? 64;
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(arena)) throw new Error(`Invalid WGSL arena name ${arena}`);
  if (!Number.isSafeInteger(workgroupSize) || workgroupSize < 8 || workgroupSize > 256) {
    throw new RangeError("canonical membership workgroupSize must be in [8, 256]");
  }
  const d = SPARSE_CM12_CANONICAL_MEMBERSHIP_DOMAIN_HEADER;
  const h = SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER;
  return /* wgsl */ `
const PCM_INVALID:u32=0xffffffffu;
const PCM_BASE:u32=${options.layout.baseWords}u;
const PCM_MAGIC:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_MAGIC}u;
const PCM_VERSION:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_VERSION}u;
const PCM_HEADER_WORDS:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_HEADER_WORDS}u;
const PCM_LEAF_BITS:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS}u;
const PCM_LEAF_WORDS:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS / 32}u;
const PCM_BRANCH:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_BRANCH}u;
const PCM_H_MAGIC:u32=${h.magic}u;const PCM_H_VERSION:u32=${h.version}u;
const PCM_H_HEADER_WORDS:u32=${h.headerWords}u;
const PCM_PHASE_UNINITIALIZED:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.uninitialized}u;
const PCM_PHASE_ACCEPTED:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.accepted}u;
const PCM_PHASE_COLLECTING:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.collecting}u;
const PCM_PHASE_REPAIRING:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.repairing}u;
const PCM_PHASE_FAULT:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.fault}u;
const PCM_FAULT_INVALID_HEADER:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.invalidHeader}u;
const PCM_FAULT_GENERATION:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.generationExhausted}u;
const PCM_FAULT_INVALID_ID:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.invalidStableId}u;
const PCM_FAULT_CONFLICT:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.conflictingCandidate}u;
const PCM_FAULT_DIRTY_CAPACITY:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.dirtyCapacity}u;
const PCM_FAULT_INVALID_PHASE:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.invalidPhase}u;
const PCM_FAULT_COUNT_RANGE:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.countUnderflow}u;
const PCM_FAULT_PUBLICATION_GAP:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.publicationGap}u;
const PCM_FAULT_ATOMIC_CONTENTION:u32=${SPARSE_CM12_CANONICAL_MEMBERSHIP_FAULT.atomicContention}u;
const PCM_D_PHASE:u32=${d.phase}u;
const PCM_D_CANDIDATE_GENERATION:u32=${d.candidateGeneration}u;
const PCM_D_ACCEPTED_GENERATION:u32=${d.acceptedGeneration}u;
const PCM_D_FAULT:u32=${d.fault}u;const PCM_D_FIRST_FAULT:u32=${d.firstFaultId}u;
const PCM_D_DIRTY_COUNT:u32=${d.dirtyCount}u;
const PCM_D_DIRECT_WRITES:u32=${d.directWriteCount}u;
const PCM_D_CLOSURE_WRITES:u32=${d.closureWriteCount}u;
const PCM_D_DIRECT_CAUSES:u32=${d.directCauseMask}u;
const PCM_D_CLOSURE_CAUSES:u32=${d.closureCauseMask}u;
const PCM_D_TOTAL:u32=${d.totalCount}u;const PCM_D_INDIRECT_X:u32=${d.repairIndirectX}u;
const PCM_D_EXPECTED_CLOSURE:u32=${d.expectedClosureCount}u;
const PCM_D_COVERED_CLOSURE:u32=${d.coveredClosureCount}u;
const PCM_D_FLAGS:u32=${d.flags}u;
// The direct row image reuses three retired incremental-header words. Cell PCM
// retains their original closure semantics.
const PCM_ROW_ACCEPTED_TOPOLOGY:u32=PCM_D_EXPECTED_CLOSURE;
const PCM_ROW_CANDIDATE_TOPOLOGY:u32=PCM_D_COVERED_CLOSURE;
const PCM_ROW_PUBLISHED_WORDS:u32=PCM_D_DIRECT_WRITES;
${domainConstants("CELL", options.layout.cell)}
${directRowConstants(options.layout.row)}
var<workgroup>pcmLeafCounts:array<u32,${SPARSE_CM12_CANONICAL_MEMBERSHIP_LEAF_BITS / 32}>;

fn pcmHeaderValid()->bool{return atomicLoad(&${arena}[PCM_BASE+PCM_H_MAGIC])==PCM_MAGIC
  &&atomicLoad(&${arena}[PCM_BASE+PCM_H_VERSION])==PCM_VERSION
  &&atomicLoad(&${arena}[PCM_BASE+PCM_H_HEADER_WORDS])==PCM_HEADER_WORDS;}
fn pcmFault(header:u32,code:u32,id:u32){
  let claimed=atomicCompareExchangeWeak(&${arena}[header+PCM_D_FAULT],0u,code);
  if(claimed.exchanged){atomicStore(&${arena}[header+PCM_D_FIRST_FAULT],id);}
  atomicStore(&${arena}[header+PCM_D_PHASE],PCM_PHASE_FAULT);
  atomicStore(&${arena}[header+PCM_D_INDIRECT_X],0u);
}
fn pcmBegin(header:u32,expectedClosureCount:u32)->bool{
  if(!pcmHeaderValid()){pcmFault(header,PCM_FAULT_INVALID_HEADER,PCM_INVALID);return false;}
  let phase=atomicLoad(&${arena}[header+PCM_D_PHASE]);
  if(phase!=PCM_PHASE_UNINITIALIZED&&phase!=PCM_PHASE_ACCEPTED){
    pcmFault(header,PCM_FAULT_INVALID_PHASE,PCM_INVALID);return false;
  }
  let accepted=atomicLoad(&${arena}[header+PCM_D_ACCEPTED_GENERATION]);
  if(accepted>=0x7ffffffeu){pcmFault(header,PCM_FAULT_GENERATION,PCM_INVALID);return false;}
  atomicStore(&${arena}[header+PCM_D_CANDIDATE_GENERATION],accepted+1u);
  atomicStore(&${arena}[header+PCM_D_FAULT],0u);
  atomicStore(&${arena}[header+PCM_D_FIRST_FAULT],PCM_INVALID);
  atomicStore(&${arena}[header+PCM_D_DIRTY_COUNT],0u);
  atomicStore(&${arena}[header+PCM_D_DIRECT_WRITES],0u);
  atomicStore(&${arena}[header+PCM_D_CLOSURE_WRITES],0u);
  atomicStore(&${arena}[header+PCM_D_DIRECT_CAUSES],0u);
  atomicStore(&${arena}[header+PCM_D_CLOSURE_CAUSES],0u);
  atomicStore(&${arena}[header+PCM_D_EXPECTED_CLOSURE],expectedClosureCount);
  atomicStore(&${arena}[header+PCM_D_COVERED_CLOSURE],0u);
  atomicStore(&${arena}[header+PCM_D_FLAGS],0u);
  atomicStore(&${arena}[header+PCM_D_INDIRECT_X],0u);
  atomicStore(&${arena}[header+PCM_D_PHASE],PCM_PHASE_COLLECTING);return true;
}
fn pcmSetCandidate(header:u32,capacity:u32,tokens:u32,stamps:u32,list:u32,
 dirtyCapacity:u32,id:u32,enabled:bool,causeMask:u32,closure:bool)->bool{
  if(atomicLoad(&${arena}[header+PCM_D_PHASE])!=PCM_PHASE_COLLECTING){
    pcmFault(header,PCM_FAULT_INVALID_PHASE,id);return false;
  }
  if(id>=capacity){pcmFault(header,PCM_FAULT_INVALID_ID,id);return false;}
  let generation=atomicLoad(&${arena}[header+PCM_D_CANDIDATE_GENERATION]);
  let token=(generation<<1u)|select(0u,1u,enabled);var observed=atomicLoad(&${arena}[tokens+id]);
  var published=false;
  for(var attempt=0u;attempt<64u;attempt+=1u){
    if((observed>>1u)==generation){
      if((observed&1u)!=(token&1u)){
        atomicStore(&${arena}[header+PCM_D_FLAGS],
          (observed&0xffffu)|((token&1u)<<16u)|((causeMask&0x7fffu)<<17u));
        pcmFault(header,PCM_FAULT_CONFLICT,id);return false;}
      return true;
    }
    let exchanged=atomicCompareExchangeWeak(&${arena}[tokens+id],observed,token);
    if(exchanged.exchanged){published=true;break;}observed=exchanged.old_value;
  }
  if(!published){
    pcmFault(header,PCM_FAULT_ATOMIC_CONTENTION,id);return false;
  }
  if(closure){atomicAdd(&${arena}[header+PCM_D_CLOSURE_WRITES],1u);
    atomicOr(&${arena}[header+PCM_D_CLOSURE_CAUSES],causeMask);
    atomicAdd(&${arena}[header+PCM_D_COVERED_CLOSURE],1u);
  }else{atomicAdd(&${arena}[header+PCM_D_DIRECT_WRITES],1u);
    atomicOr(&${arena}[header+PCM_D_DIRECT_CAUSES],causeMask);}
  let leaf=id/PCM_LEAF_BITS;
  if(atomicExchange(&${arena}[stamps+leaf],generation)!=generation){
    let slot=atomicAdd(&${arena}[header+PCM_D_DIRTY_COUNT],1u);
    if(slot>=dirtyCapacity){pcmFault(header,PCM_FAULT_DIRTY_CAPACITY,id);return false;}
    atomicStore(&${arena}[list+slot],leaf);
  }
  return true;
}
fn pcmFinalizeFrontier(header:u32)->bool{
  if(atomicLoad(&${arena}[header+PCM_D_PHASE])!=PCM_PHASE_COLLECTING){return false;}
  if(atomicLoad(&${arena}[header+PCM_D_FAULT])!=0u){return false;}
  if(atomicLoad(&${arena}[header+PCM_D_EXPECTED_CLOSURE])
    !=atomicLoad(&${arena}[header+PCM_D_COVERED_CLOSURE])){
    pcmFault(header,PCM_FAULT_PUBLICATION_GAP,PCM_INVALID);return false;
  }
  let dirty=atomicLoad(&${arena}[header+PCM_D_DIRTY_COUNT]);
  atomicStore(&${arena}[header+PCM_D_INDIRECT_X],dirty);
  atomicStore(&${arena}[header+PCM_D_PHASE],PCM_PHASE_REPAIRING);return true;
}
fn pcmFinalize(header:u32,root:u32)->bool{
  if(atomicLoad(&${arena}[header+PCM_D_PHASE])!=PCM_PHASE_REPAIRING
    ||atomicLoad(&${arena}[header+PCM_D_FAULT])!=0u){return false;}
  let generation=atomicLoad(&${arena}[header+PCM_D_CANDIDATE_GENERATION]);
  let total=atomicLoad(&${arena}[root]);
  atomicStore(&${arena}[header+PCM_D_TOTAL],total);
  // The same copy-isolated indirect slot first schedules dirty leaves, then
  // becomes the canonical member invocation after publication.
  atomicStore(&${arena}[header+PCM_D_INDIRECT_X],(total+${workgroupSize - 1}u)/${workgroupSize}u);
  atomicStore(&${arena}[header+PCM_D_ACCEPTED_GENERATION],generation);
  atomicStore(&${arena}[header+PCM_D_PHASE],PCM_PHASE_ACCEPTED);return true;
}
fn pcmCellBegin(expectedClosureCount:u32)->bool{return pcmBegin(PCM_CELL_HEADER,expectedClosureCount);}
fn pcmCellSetCandidate(id:u32,enabled:bool,cause:u32,closure:bool)->bool{
  return pcmSetCandidate(PCM_CELL_HEADER,PCM_CELL_CAPACITY,PCM_CELL_CANDIDATE_TOKENS,
    PCM_CELL_DIRTY_STAMPS,PCM_CELL_DIRTY_LIST,PCM_CELL_LEAF_COUNT,id,enabled,cause,closure);
}
fn pcmRowBegin(topologyGeneration:u32)->bool{
  if(!pcmHeaderValid()){pcmFault(PCM_ROW_HEADER,PCM_FAULT_INVALID_HEADER,PCM_INVALID);
    return false;}
  let phase=atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_D_PHASE]);
  if(phase!=PCM_PHASE_UNINITIALIZED&&phase!=PCM_PHASE_ACCEPTED){
    pcmFault(PCM_ROW_HEADER,PCM_FAULT_INVALID_PHASE,phase);return false;}
  let accepted=atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_D_ACCEPTED_GENERATION]);
  if(accepted>=0x7ffffffeu){pcmFault(PCM_ROW_HEADER,PCM_FAULT_GENERATION,PCM_INVALID);
    return false;}
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_D_CANDIDATE_GENERATION],accepted+1u);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_D_FAULT],0u);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_D_FIRST_FAULT],PCM_INVALID);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_D_TOTAL],0u);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_ROW_PUBLISHED_WORDS],0u);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_ROW_CANDIDATE_TOPOLOGY],topologyGeneration);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_D_PHASE],PCM_PHASE_COLLECTING);
  return true;
}
fn pcmRowPublicationOpen()->bool{return atomicLoad(&${arena}[
  PCM_ROW_HEADER+PCM_D_PHASE])==PCM_PHASE_COLLECTING
  &&atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_D_FAULT])==0u;}
fn pcmRowPriorTopologyGeneration()->u32{
  let accepted=atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_D_ACCEPTED_GENERATION]);
  return select(PCM_INVALID,atomicLoad(&${arena}[
    PCM_ROW_HEADER+PCM_ROW_ACCEPTED_TOPOLOGY]),accepted!=0u);
}
fn pcmRowPublishWord(word:u32,bits:u32)->bool{
  if(!pcmRowPublicationOpen()){return false;}
  if(word>=PCM_ROW_ACTIVE_WORDS){
    pcmFault(PCM_ROW_HEADER,PCM_FAULT_INVALID_ID,word);return false;}
  atomicStore(&${arena}[PCM_ROW_ACTIVE_BITS+word],bits);
  atomicAdd(&${arena}[PCM_ROW_HEADER+PCM_D_TOTAL],countOneBits(bits));
  atomicAdd(&${arena}[PCM_ROW_HEADER+PCM_ROW_PUBLISHED_WORDS],1u);
  return true;
}
fn pcmRowFinalize(topologyGeneration:u32)->bool{
  if(!pcmRowPublicationOpen()){return false;}
  if(atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_ROW_CANDIDATE_TOPOLOGY])
      !=topologyGeneration
    ||atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_ROW_PUBLISHED_WORDS])
      !=PCM_ROW_ACTIVE_WORDS){
    pcmFault(PCM_ROW_HEADER,PCM_FAULT_PUBLICATION_GAP,PCM_INVALID);return false;}
  let generation=atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_D_CANDIDATE_GENERATION]);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_ROW_ACCEPTED_TOPOLOGY],topologyGeneration);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_D_ACCEPTED_GENERATION],generation);
  atomicStore(&${arena}[PCM_ROW_HEADER+PCM_D_PHASE],PCM_PHASE_ACCEPTED);
  return true;
}
fn pcmContains(header:u32,bits:u32,capacity:u32,id:u32)->bool{
  let phase=atomicLoad(&${arena}[header+PCM_D_PHASE]);
  if(id>=capacity||(phase!=PCM_PHASE_ACCEPTED&&phase!=PCM_PHASE_COLLECTING)){
    return false;
  }
  return (atomicLoad(&${arena}[bits+id/32u])&(1u<<(id&31u)))!=0u;
}
fn pcmCellContains(id:u32)->bool{return pcmContains(PCM_CELL_HEADER,
  PCM_CELL_ACTIVE_BITS,PCM_CELL_CAPACITY,id);}
fn pcmRowContains(id:u32)->bool{return pcmContains(PCM_ROW_HEADER,
  PCM_ROW_ACTIVE_BITS,PCM_ROW_CAPACITY,id);}
fn pcmAcceptedCount(header:u32)->u32{
  let phase=atomicLoad(&${arena}[header+PCM_D_PHASE]);
  return select(0u,atomicLoad(&${arena}[header+PCM_D_TOTAL]),
    phase==PCM_PHASE_ACCEPTED||phase==PCM_PHASE_COLLECTING);
}
fn pcmCellAcceptedCount()->u32{return pcmAcceptedCount(PCM_CELL_HEADER);}
fn pcmRowAcceptedCount()->u32{return pcmAcceptedCount(PCM_ROW_HEADER);}
fn pcmCellBootstrapEpoch()->bool{return atomicLoad(&${arena}[
  PCM_CELL_HEADER+PCM_D_ACCEPTED_GENERATION])==0u;}
fn pcmCellCandidateGeneration()->u32{return atomicLoad(&${arena}[
  PCM_CELL_HEADER+PCM_D_CANDIDATE_GENERATION]);}
fn pcmRowCandidateGeneration()->u32{return atomicLoad(&${arena}[
  PCM_ROW_HEADER+PCM_D_CANDIDATE_GENERATION]);}
fn pcmCellFault()->u32{return atomicLoad(&${arena}[PCM_CELL_HEADER+PCM_D_FAULT]);}
fn pcmRowFault()->u32{return atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_D_FAULT]);}
fn pcmCellFirstFault()->u32{return atomicLoad(&${arena}[PCM_CELL_HEADER+PCM_D_FIRST_FAULT]);}
fn pcmRowFirstFault()->u32{return atomicLoad(&${arena}[PCM_ROW_HEADER+PCM_D_FIRST_FAULT]);}
fn pcmCellAcceptedGeneration()->u32{return atomicLoad(&${arena}[
  PCM_CELL_HEADER+PCM_D_ACCEPTED_GENERATION]);}
fn pcmRowAcceptedGeneration()->u32{return atomicLoad(&${arena}[
  PCM_ROW_HEADER+PCM_D_ACCEPTED_GENERATION]);}
fn pcmCellFinalizeFrontier()->bool{return pcmFinalizeFrontier(PCM_CELL_HEADER);}
fn pcmCellFinalize()->bool{return pcmFinalize(PCM_CELL_HEADER,
  PCM_CELL_TREE_${options.layout.cell.treeLevelCounts.length - 1});}
${repairBody("CELL", "Cell", options.layout.cell, arena, workgroupSize)}
${rankSelectBody("CELL", "Cell", options.layout.cell, arena)}
`;
}
