import {
  SPARSE_CM12_FACE_PROJECTION_BRANCH,
  SPARSE_CM12_FACE_PROJECTION_CAUSE,
  SPARSE_CM12_FACE_PROJECTION_FAULT,
  SPARSE_CM12_FACE_PROJECTION_HEADER,
  SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS,
  SPARSE_CM12_FACE_PROJECTION_INVALID,
  SPARSE_CM12_FACE_PROJECTION_LEAF_BITS,
  SPARSE_CM12_FACE_PROJECTION_MAGIC,
  SPARSE_CM12_FACE_PROJECTION_PHASE,
  SPARSE_CM12_FACE_PROJECTION_STAGE,
  SPARSE_CM12_FACE_PROJECTION_STAGE_COUNT,
  SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER,
  SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS,
  SPARSE_CM12_FACE_PROJECTION_VERSION,
  type SparseCM12FaceProjectionAuthorityLayout,
  type SparseCM12FaceProjectionStageLayout,
} from "./sparse-cm12-face-projection-authority";

export interface SparseCM12FaceProjectionAuthorityWGSLOptions {
  readonly layout: SparseCM12FaceProjectionAuthorityLayout;
  /** Existing `array<atomic<u32>>` storage arena. */
  readonly arenaName?: string;
  readonly prefix?: string;
  readonly workgroupSize?: number;
  /** Construction/bootstrap stable-row mapper. Omit for dense stable ids. */
  readonly preparationBootstrapInvocationFunction?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError(`${label} is not WGSL`);
  return value;
};

function stageConstants(prefix: string, label: string,
  stage: SparseCM12FaceProjectionStageLayout): string {
  return /* wgsl */ `
const ${prefix}${label}Header:u32=${stage.headerBaseWords}u;
const ${prefix}${label}Bits:u32=${stage.activeBitsBaseWords}u;
const ${prefix}${label}CandidateGeneration:u32=${stage.candidateGenerationBaseWords}u;
const ${prefix}${label}CandidateCause:u32=${stage.candidateCauseBaseWords}u;
const ${prefix}${label}CandidateDepth:u32=${stage.candidateDepthBaseWords}u;
const ${prefix}${label}CandidateDependency:u32=${stage.candidateDependencyGenerationBaseWords}u;
const ${prefix}${label}AcceptedDependency:u32=${stage.acceptedDependencyGenerationBaseWords}u;
const ${prefix}${label}ExecutionGeneration:u32=${stage.executionGenerationBaseWords}u;
const ${prefix}${label}DirtyStamp:u32=${stage.dirtyLeafStampBaseWords}u;
const ${prefix}${label}DirtyList:u32=${stage.dirtyLeafListBaseWords}u;
const ${prefix}${label}ActiveLeafList:u32=${stage.activeLeafListBaseWords}u;
const ${prefix}${label}LeafCount:u32=${stage.leafCount}u;
${stage.treeLevelBaseWords.map((base, level) =>
    `const ${prefix}${label}Tree${level}:u32=${base}u;`).join("\n")}
`;
}

function rankSelect(prefix: string, label: string,
  stage: SparseCM12FaceProjectionStageLayout, arena: string): string {
  const descend = Array.from({ length: stage.treeLevelCounts.length - 1 }, (_, index) => {
    const level = stage.treeLevelCounts.length - 2 - index;
    return /* wgsl */ `
  {
    let childBegin=node*${prefix}Branch;var selected=${prefix}Invalid;
    for(var child=0u;child<${prefix}Branch;child+=1u){
      let candidate=childBegin+child;if(candidate>=${stage.treeLevelCounts[level]}u){break;}
      let count=atomicLoad(&${arena}[${prefix}${label}Tree${level}+candidate]);
      if(selected==${prefix}Invalid&&remaining<count){selected=candidate;}
      else if(selected==${prefix}Invalid){remaining-=count;}
    }
    if(selected==${prefix}Invalid){return ${prefix}Invalid;}node=selected;
  }`;
  }).join("\n");
  return /* wgsl */ `
fn ${prefix}${label}RankSelect(rank:u32)->u32{
  let phase=atomicLoad(&${arena}[${prefix}${label}Header+${prefix}DPhase]);
  if(phase!=${prefix}PhaseExecuting&&phase!=${prefix}PhaseAccepted){return ${prefix}Invalid;}
  let total=atomicLoad(&${arena}[${prefix}${label}Header+${prefix}DWorkCount]);
  if(rank>=total){return ${prefix}Invalid;}var remaining=rank;var node=0u;
${descend}
  let firstWord=node*${prefix}LeafWords;
  for(var wordAt=0u;wordAt<${prefix}LeafWords;wordAt+=1u){
    let word=atomicLoad(&${arena}[${prefix}${label}Bits+firstWord+wordAt]);
    let count=countOneBits(word);if(remaining>=count){remaining-=count;continue;}
    for(var bit=0u;bit<32u;bit+=1u){if((word&(1u<<bit))==0u){continue;}
      if(remaining==0u){let row=node*${prefix}LeafBits+32u*wordAt+bit;
        return select(${prefix}Invalid,row,row<${prefix}RowCapacity);}remaining-=1u;}
  }
  return ${prefix}Invalid;
}`;
}

function repairKernel(prefix: string, label: string, stageIndex: number,
  stage: SparseCM12FaceProjectionStageLayout, arena: string, workgroupSize: number): string {
  const updates = stage.treeLevelBaseWords.slice(1).map((base, index) => /* wgsl */ `
    node/=${prefix}Branch;
    let prior${index}=atomicAdd(&${arena}[${base}u+node],bitcast<u32>(delta));
    if((delta<0&&prior${index}<u32(-delta))
      ||(delta>0&&prior${index}>0xffffffffu-u32(delta))){
      ${prefix}Fail(${stageIndex}u,${SPARSE_CM12_FACE_PROJECTION_FAULT.executionCoverageGap}u,
        leaf*${prefix}LeafBits);}
  `).join("");
  return /* wgsl */ `
@compute @workgroup_size(${workgroupSize})
fn repairSparseCM12Face${label}Leaves(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let header=${prefix}${label}Header;
  let phase=atomicLoad(&${arena}[header+${prefix}DPhase]);
  let dirty=atomicLoad(&${arena}[header+${prefix}DDirtyLeafCount]);
  let invocationValid=wid.x<dirty;var leaf=0u;
  if(invocationValid){leaf=atomicLoad(&${arena}[${prefix}${label}DirtyList+wid.x]);}
  let valid=phase==${prefix}PhaseRepairing&&invocationValid&&leaf<${prefix}${label}LeafCount
    &&atomicLoad(&${arena}[header+${prefix}DFault])==0u;
  let generation=atomicLoad(&${arena}[header+${prefix}DCandidateGeneration]);
  if(lane<${prefix}LeafWords){
    var word=0u;if(valid){let wordIndex=leaf*${prefix}LeafWords+lane;
      for(var bit=0u;bit<32u;bit+=1u){let row=leaf*${prefix}LeafBits+32u*lane+bit;
        if(row<${prefix}RowCapacity
          &&atomicLoad(&${arena}[${prefix}${label}CandidateGeneration+row])==generation){
          word|=1u<<bit;
        }}
      atomicStore(&${arena}[${prefix}${label}Bits+wordIndex],word);
    }
    ${prefix}LeafCounts[lane]=countOneBits(word);
  }
  workgroupBarrier();
  if(lane==0u&&valid){var count=0u;
    for(var at=0u;at<${prefix}LeafWords;at+=1u){count+=${prefix}LeafCounts[at];}
    let previous=atomicExchange(&${arena}[${prefix}${label}Tree0+leaf],count);
    let delta=i32(count)-i32(previous);var node=leaf;${updates}
    if(count>0u){let slot=atomicAdd(&${arena}[header+${prefix}DActiveLeafCount],1u);
      if(slot>=${prefix}${label}LeafCount){
        ${prefix}Fail(${stageIndex}u,${SPARSE_CM12_FACE_PROJECTION_FAULT.activeLeafCapacity}u,
          leaf*${prefix}LeafBits);
      }else{atomicStore(&${arena}[${prefix}${label}ActiveLeafList+slot],leaf);}}
  }
}`;
}

function verifyKernel(prefix: string, label: string, stageIndex: number,
  arena: string, workgroupSize: number): string {
  return /* wgsl */ `
@compute @workgroup_size(${workgroupSize})
fn verifySparseCM12Face${label}Leaves(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let header=${prefix}${label}Header;
  let count=atomicLoad(&${arena}[header+${prefix}DActiveLeafCount]);
  let valid=wid.x<count&&atomicLoad(&${arena}[header+${prefix}DPhase])==${prefix}PhaseExecuting;
  var leaf=0u;if(valid){leaf=atomicLoad(&${arena}[${prefix}${label}ActiveLeafList+wid.x]);}
  if(lane==0u){atomicStore(&${prefix}VerifyCount,0u);atomicStore(&${prefix}VerifyFault,${prefix}Invalid);}
  workgroupBarrier();
  if(valid&&lane<${prefix}LeafWords){let word=atomicLoad(&${arena}[
      ${prefix}${label}Bits+leaf*${prefix}LeafWords+lane]);
    let generation=atomicLoad(&${arena}[header+${prefix}DCandidateGeneration]);
    for(var bit=0u;bit<32u;bit+=1u){if((word&(1u<<bit))==0u){continue;}
      let row=leaf*${prefix}LeafBits+32u*lane+bit;
      if(atomicLoad(&${arena}[${prefix}${label}ExecutionGeneration+row])!=generation){
        atomicMin(&${prefix}VerifyFault,row);
      }else{atomicAdd(&${prefix}VerifyCount,1u);}}
  }
  workgroupBarrier();
  if(lane==0u&&valid){let fault=atomicLoad(&${prefix}VerifyFault);
    if(fault!=${prefix}Invalid){${prefix}Fail(${stageIndex}u,
      ${SPARSE_CM12_FACE_PROJECTION_FAULT.executionCoverageGap}u,fault);
    }else{atomicAdd(&${arena}[header+${prefix}DExecutedCount],atomicLoad(&${prefix}VerifyCount));
      atomicAdd(&${arena}[header+${prefix}DVerifiedLeafCount],1u);}}
}`;
}

/**
 * Binding-free authority. Integration provides HTP1/PCM helpers and five
 * producer hooks named in the generated preamble. Numerical kernels retain
 * their existing bodies and call the completion helpers only after the final
 * face word (including mirror writes) is authoritative.
 */
export function createSparseCM12FaceProjectionAuthorityWGSL(
  options: SparseCM12FaceProjectionAuthorityWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "faceProjectionArena", "arenaName");
  const p = identifier(options.prefix ?? "fpa", "prefix");
  const preparationBootstrapInvocation = options.preparationBootstrapInvocationFunction
    ? identifier(options.preparationBootstrapInvocationFunction,
      "preparationBootstrapInvocationFunction") : undefined;
  const workgroupSize = options.workgroupSize ?? 64;
  if (!Number.isSafeInteger(workgroupSize) || workgroupSize < 8 || workgroupSize > 256) {
    throw new RangeError("FPA1 workgroupSize must be in [8, 256]");
  }
  const h = SPARSE_CM12_FACE_PROJECTION_HEADER;
  const d = SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER;
  const hb = (word: number) => `${layout.baseWords + word}u`;
  const prepRoot = layout.preparation.treeLevelBaseWords.at(-1)!;
  const projectionRoot = layout.projection.treeLevelBaseWords.at(-1)!;
  return /* wgsl */ `
// Required integration hooks:
// fpaFrameGeneration/TopologyGeneration/PCMGeneration/SourceParity/PolicyBits()->u32
// fpaExpectedPreparationReceipts/ExpectedProjectionReceipts()->u32
// fpaPreparationRowLive/fpaProjectionRowLive(row)->bool
// fpaPreparationDependencyGeneration/fpaProjectionDependencyGeneration(row)->u32
// Plus HTP1 cm12HotHeaderValid/RowValid/IncidenceRange/Incidence/RowTermCount/RowTermCell.
const ${p}Magic:u32=0x${SPARSE_CM12_FACE_PROJECTION_MAGIC.toString(16)}u;
const ${p}Version:u32=${SPARSE_CM12_FACE_PROJECTION_VERSION}u;
const ${p}Invalid:u32=0x${SPARSE_CM12_FACE_PROJECTION_INVALID.toString(16)}u;
const ${p}RowCapacity:u32=${layout.rowCapacity}u;
const ${p}CellCapacity:u32=${layout.cellCapacity}u;
const ${p}PreparedAuthority:u32=${layout.preparedAuthorityBaseWords}u;
const ${p}AcceptedPressureBits:u32=${layout.acceptedPressureBitsBaseWords}u;
const ${p}LeafBits:u32=${SPARSE_CM12_FACE_PROJECTION_LEAF_BITS}u;
const ${p}LeafWords:u32=${SPARSE_CM12_FACE_PROJECTION_LEAF_BITS / 32}u;
const ${p}Branch:u32=${SPARSE_CM12_FACE_PROJECTION_BRANCH}u;
const ${p}Preparation:u32=${SPARSE_CM12_FACE_PROJECTION_STAGE.preparation}u;
const ${p}Projection:u32=${SPARSE_CM12_FACE_PROJECTION_STAGE.projection}u;
const ${p}PhaseUninitialized:u32=${SPARSE_CM12_FACE_PROJECTION_PHASE.uninitialized}u;
const ${p}PhaseAccepted:u32=${SPARSE_CM12_FACE_PROJECTION_PHASE.accepted}u;
const ${p}PhaseCollecting:u32=${SPARSE_CM12_FACE_PROJECTION_PHASE.collecting}u;
const ${p}PhaseRepairing:u32=${SPARSE_CM12_FACE_PROJECTION_PHASE.repairing}u;
const ${p}PhaseExecuting:u32=${SPARSE_CM12_FACE_PROJECTION_PHASE.executing}u;
const ${p}PhaseFault:u32=${SPARSE_CM12_FACE_PROJECTION_PHASE.fault}u;
const ${p}DPhase:u32=${d.phase}u;const ${p}DAcceptedGeneration:u32=${d.acceptedGeneration}u;
const ${p}DCandidateGeneration:u32=${d.candidateGeneration}u;
const ${p}DFrameGeneration:u32=${d.frameGeneration}u;
const ${p}DTopologyGeneration:u32=${d.topologyGeneration}u;
const ${p}DPCMGeneration:u32=${d.pcmGeneration}u;
const ${p}DSourceParity:u32=${d.sourceParity}u;const ${p}DPolicyBits:u32=${d.policyBits}u;
const ${p}DFault:u32=${d.fault}u;const ${p}DFirstFaultRow:u32=${d.firstFaultRow}u;
const ${p}DExpectedReceipts:u32=${d.expectedProducerReceipts}u;
const ${p}DCoveredReceipts:u32=${d.coveredProducerReceipts}u;
const ${p}DDirectWrites:u32=${d.directWriteCount}u;
const ${p}DClosureWrites:u32=${d.closureWriteCount}u;
const ${p}DCauseMask:u32=${d.causeMask}u;const ${p}DDirtyLeafCount:u32=${d.dirtyLeafCount}u;
const ${p}DPreviousActiveLeafCount:u32=${d.previousActiveLeafCount}u;
const ${p}DActiveLeafCount:u32=${d.activeLeafCount}u;
const ${p}DWorkCount:u32=${d.workCount}u;const ${p}DReusedCount:u32=${d.reusedCount}u;
const ${p}DExecutedCount:u32=${d.executedCount}u;
const ${p}DRepairIndirectX:u32=${d.repairIndirectX}u;
const ${p}DWorkIndirectX:u32=${d.workIndirectX}u;
const ${p}DVerifiedLeafCount:u32=${d.verifiedLeafCount}u;
const ${p}QAOracle:bool=${layout.qaFullOracle ? "true" : "false"};
${stageConstants(p, "Preparation", layout.preparation)}
${stageConstants(p, "Projection", layout.projection)}
var<workgroup>${p}LeafCounts:array<u32,${SPARSE_CM12_FACE_PROJECTION_LEAF_BITS / 32}>;
var<workgroup>${p}VerifyCount:atomic<u32>;
var<workgroup>${p}VerifyFault:atomic<u32>;

fn ${p}Header(stage:u32)->u32{return select(${p}PreparationHeader,${p}ProjectionHeader,
  stage==${p}Projection);}
fn ${p}CandidateGenerationBase(stage:u32)->u32{return select(
  ${p}PreparationCandidateGeneration,${p}ProjectionCandidateGeneration,stage==${p}Projection);}
fn ${p}CandidateCauseBase(stage:u32)->u32{return select(
  ${p}PreparationCandidateCause,${p}ProjectionCandidateCause,stage==${p}Projection);}
fn ${p}CandidateDepthBase(stage:u32)->u32{return select(
  ${p}PreparationCandidateDepth,${p}ProjectionCandidateDepth,stage==${p}Projection);}
fn ${p}CandidateDependencyBase(stage:u32)->u32{return select(
  ${p}PreparationCandidateDependency,${p}ProjectionCandidateDependency,stage==${p}Projection);}
fn ${p}AcceptedDependencyBase(stage:u32)->u32{return select(
  ${p}PreparationAcceptedDependency,${p}ProjectionAcceptedDependency,stage==${p}Projection);}
fn ${p}ExecutionGenerationBase(stage:u32)->u32{return select(
  ${p}PreparationExecutionGeneration,${p}ProjectionExecutionGeneration,stage==${p}Projection);}
fn ${p}DirtyStampBase(stage:u32)->u32{return select(
  ${p}PreparationDirtyStamp,${p}ProjectionDirtyStamp,stage==${p}Projection);}
fn ${p}DirtyListBase(stage:u32)->u32{return select(
  ${p}PreparationDirtyList,${p}ProjectionDirtyList,stage==${p}Projection);}
fn ${p}ActiveLeafListBase(stage:u32)->u32{return select(
  ${p}PreparationActiveLeafList,${p}ProjectionActiveLeafList,stage==${p}Projection);}
fn ${p}LeafCount(stage:u32)->u32{return select(
  ${p}PreparationLeafCount,${p}ProjectionLeafCount,stage==${p}Projection);}
fn ${p}GlobalHeaderValid()->bool{return cm12HotHeaderValid()
  &&arrayLength(&${arena})>=${layout.totalWords}u
  &&atomicLoad(&${arena}[${hb(h.magic)}])==${p}Magic
  &&atomicLoad(&${arena}[${hb(h.version)}])==${p}Version
  &&atomicLoad(&${arena}[${hb(h.headerWords)}])==${SPARSE_CM12_FACE_PROJECTION_HEADER_WORDS}u
  &&atomicLoad(&${arena}[${hb(h.stageHeaderWords)}])==${SPARSE_CM12_FACE_PROJECTION_STAGE_HEADER_WORDS}u
  &&atomicLoad(&${arena}[${hb(h.stageCount)}])==${SPARSE_CM12_FACE_PROJECTION_STAGE_COUNT}u
  &&atomicLoad(&${arena}[${hb(h.leafBits)}])==${p}LeafBits
  &&atomicLoad(&${arena}[${hb(h.branch)}])==${p}Branch
  &&atomicLoad(&${arena}[${hb(h.rowCapacity)}])==${p}RowCapacity
  &&atomicLoad(&${arena}[${hb(h.cellCapacity)}])==${p}CellCapacity
  &&atomicLoad(&${arena}[${hb(h.preparationHeaderBase)}])==${p}PreparationHeader
  &&atomicLoad(&${arena}[${hb(h.projectionHeaderBase)}])==${p}ProjectionHeader
  &&atomicLoad(&${arena}[${hb(h.totalWords)}])==${layout.totalWords}u
  &&atomicLoad(&${arena}[${hb(h.preparedAuthorityBase)}])==${p}PreparedAuthority
  &&atomicLoad(&${arena}[${hb(h.reserved0)}])==${p}AcceptedPressureBits
  &&atomicLoad(&${arena}[${hb(h.brickFineResolution)}])==${layout.brickFineResolution}u
  &&atomicLoad(&${arena}[${hb(h.presentationPageResolution)}])
    ==${layout.presentationPageResolution}u;}
fn ${p}Fail(stage:u32,code:u32,row:u32){let header=${p}Header(stage);
  let won=atomicCompareExchangeWeak(&${arena}[header+${p}DFault],0u,code).exchanged;
  if(won){atomicStore(&${arena}[header+${p}DFirstFaultRow],row);
    atomicStore(&${arena}[${hb(h.firstFaultStage)}],stage);}
  atomicStore(&${arena}[header+${p}DPhase],${p}PhaseFault);
  atomicStore(&${arena}[header+${p}DRepairIndirectX],0u);
  atomicStore(&${arena}[header+${p}DWorkIndirectX],0u);
}
fn ${p}Begin(stage:u32,expectedReceipts:u32)->bool{let header=${p}Header(stage);
  if(!${p}GlobalHeaderValid()){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.invalidHeader}u,${p}Invalid);return false;}
  let phase=atomicLoad(&${arena}[header+${p}DPhase]);
  if(phase!=${p}PhaseUninitialized&&phase!=${p}PhaseAccepted){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.invalidPhase}u,${p}Invalid);return false;}
  let accepted=atomicLoad(&${arena}[header+${p}DAcceptedGeneration]);
  if(accepted>=0x7ffffffeu){${p}Fail(stage,
    ${SPARSE_CM12_FACE_PROJECTION_FAULT.generationExhausted}u,${p}Invalid);return false;}
  let bootstrap=phase==${p}PhaseUninitialized;
  atomicStore(&${arena}[header+${p}DCandidateGeneration],accepted+1u);
  atomicStore(&${arena}[header+${p}DFrameGeneration],fpaFrameGeneration());
  atomicStore(&${arena}[header+${p}DTopologyGeneration],fpaTopologyGeneration());
  atomicStore(&${arena}[header+${p}DPCMGeneration],fpaPCMGeneration());
  atomicStore(&${arena}[header+${p}DSourceParity],fpaSourceParity());
  atomicStore(&${arena}[header+${p}DPolicyBits],fpaPolicyBits());
  atomicStore(&${arena}[header+${p}DFault],0u);
  atomicStore(&${arena}[header+${p}DFirstFaultRow],${p}Invalid);
  atomicStore(&${arena}[header+${p}DExpectedReceipts],expectedReceipts);
  atomicStore(&${arena}[header+${p}DCoveredReceipts],0u);
  atomicStore(&${arena}[header+${p}DDirectWrites],0u);
  atomicStore(&${arena}[header+${p}DClosureWrites],0u);
  atomicStore(&${arena}[header+${p}DCauseMask],0u);
  atomicStore(&${arena}[header+${p}DDirtyLeafCount],0u);
  let previous=atomicLoad(&${arena}[header+${p}DActiveLeafCount]);
  atomicStore(&${arena}[header+${p}DPreviousActiveLeafCount],previous);
  atomicStore(&${arena}[header+${p}DActiveLeafCount],0u);
  atomicStore(&${arena}[header+${p}DWorkCount],0u);
  atomicStore(&${arena}[header+${p}DReusedCount],0u);
  atomicStore(&${arena}[header+${p}DExecutedCount],0u);
  atomicStore(&${arena}[header+${p}DVerifiedLeafCount],0u);
  atomicStore(&${arena}[header+${p}DRepairIndirectX],0u);
  atomicStore(&${arena}[header+${p}DRepairIndirectX+1u],1u);
  atomicStore(&${arena}[header+${p}DRepairIndirectX+2u],1u);
  atomicStore(&${arena}[header+${p}DWorkIndirectX],0u);
  atomicStore(&${arena}[header+${p}DWorkIndirectX+1u],1u);
  atomicStore(&${arena}[header+${p}DWorkIndirectX+2u],1u);
  // reserved0..2 form the copy-isolated bootstrap/oracle indirect triplet.
  atomicStore(&${arena}[header+${d.reserved0}u],select(0u,
    (${p}RowCapacity+${workgroupSize - 1}u)/${workgroupSize}u,bootstrap||${p}QAOracle));
  atomicStore(&${arena}[header+${d.reserved1}u],1u);
  atomicStore(&${arena}[header+${d.reserved2}u],1u);
  atomicStore(&${arena}[header+${p}DPhase],${p}PhaseCollecting);return true;
}
@compute @workgroup_size(1)
fn beginSparseCM12FacePreparationAuthority(){
  _=${p}Begin(${p}Preparation,fpaExpectedPreparationReceipts());
}
@compute @workgroup_size(1)
fn beginSparseCM12FaceProjectionAuthority(){
  _=${p}Begin(${p}Projection,fpaExpectedProjectionReceipts());
}
fn ${p}QueueLeaf(stage:u32,leaf:u32)->bool{let header=${p}Header(stage);
  let generation=atomicLoad(&${arena}[header+${p}DCandidateGeneration]);
  if(atomicExchange(&${arena}[${p}DirtyStampBase(stage)+leaf],generation)==generation){return true;}
  let slot=atomicAdd(&${arena}[header+${p}DDirtyLeafCount],1u);
  if(slot>=${p}LeafCount(stage)){${p}Fail(stage,
    ${SPARSE_CM12_FACE_PROJECTION_FAULT.dirtyCapacity}u,leaf*${p}LeafBits);return false;}
  atomicStore(&${arena}[${p}DirtyListBase(stage)+slot],leaf);return true;
}
fn ${p}Mark(stage:u32,row:u32,cause:u32,depth:u32,dependencyGeneration:u32,
 receipt:bool)->bool{let header=${p}Header(stage);
  if(atomicLoad(&${arena}[header+${p}DPhase])!=${p}PhaseCollecting){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.invalidPhase}u,row);return false;}
  if(row>=${p}RowCapacity||!cm12HotRowValid(row)){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.invalidRow}u,row);return false;}
  if(dependencyGeneration==0u){${p}Fail(stage,
    ${SPARSE_CM12_FACE_PROJECTION_FAULT.dependencyGenerationGap}u,row);return false;}
  let generation=atomicLoad(&${arena}[header+${p}DCandidateGeneration]);
  let generationAt=${p}CandidateGenerationBase(stage)+row;
  var observed=atomicLoad(&${arena}[generationAt]);var first=false;
  var published=false;
  for(var attempt=0u;attempt<64u;attempt+=1u){if(observed==generation){published=true;break;}
    let changed=atomicCompareExchangeWeak(&${arena}[generationAt],observed,generation);
    if(changed.exchanged){first=true;published=true;break;}observed=changed.old_value;}
  if(!published){${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.atomicContention}u,row);
    return false;}
  let dependencyAt=${p}CandidateDependencyBase(stage)+row;
  if(first){atomicStore(&${arena}[dependencyAt],dependencyGeneration);
    atomicStore(&${arena}[${p}CandidateCauseBase(stage)+row],cause);
    atomicStore(&${arena}[${p}CandidateDepthBase(stage)+row],depth);
    atomicAdd(&${arena}[header+select(${p}DDirectWrites,${p}DClosureWrites,depth>0u)],1u);
    _=${p}QueueLeaf(stage,row/${p}LeafBits);
  }else{atomicOr(&${arena}[${p}CandidateCauseBase(stage)+row],cause);
    atomicMax(&${arena}[${p}CandidateDepthBase(stage)+row],depth);}
  atomicOr(&${arena}[header+${p}DCauseMask],cause);
  if(receipt){atomicAdd(&${arena}[header+${p}DCoveredReceipts],1u);}return true;
}
fn fpaMarkPreparationRow(row:u32,cause:u32,depth:u32,receipt:bool)->bool{
  if(!fpaPreparationRowLive(row)){if(receipt){atomicAdd(&${arena}[
    ${p}PreparationHeader+${p}DCoveredReceipts],1u);}return true;}
  return ${p}Mark(${p}Preparation,row,cause,depth,
    fpaPreparationDependencyGeneration(row),receipt);
}
fn fpaMarkProjectionRow(row:u32,cause:u32,depth:u32,receipt:bool)->bool{
  if(!fpaProjectionRowLive(row)){if(receipt){atomicAdd(&${arena}[
    ${p}ProjectionHeader+${p}DCoveredReceipts],1u);}return true;}
  return ${p}Mark(${p}Projection,row,cause,depth,
    fpaProjectionDependencyGeneration(row),receipt);
}
fn fpaCoverPreparationReceipt(){atomicAdd(&${arena}[
  ${p}PreparationHeader+${p}DCoveredReceipts],1u);}
fn fpaMarkProjectionPressureCell(cell:u32,bits:u32,cause:u32)->bool{
  if(cell>=${p}CellCapacity){${p}Fail(${p}Projection,
    ${SPARSE_CM12_FACE_PROJECTION_FAULT.invalidCell}u,cell);return false;}
  if(atomicExchange(&${arena}[${p}AcceptedPressureBits+cell],bits)==bits){return true;}
  atomicAdd(&${arena}[${p}ProjectionHeader+${p}DExpectedReceipts],1u);
  var ok=true;let incidences=cm12HotIncidenceRange(cell);
  for(var at=incidences.x;at<incidences.x+incidences.y;at+=1u){
    ok=fpaMarkProjectionRow(cm12HotIncidence(at).x,cause,0u,false)&&ok;}
  if(ok){atomicAdd(&${arena}[${p}ProjectionHeader+${p}DCoveredReceipts],1u);}
  return ok;
}
// One producer receipt covers the complete bounded HTP incidence/one-ring
// closure. All row writes below are internally owned and therefore receipt=false.
fn fpaMarkTopologyCellBlast(cell:u32,cause:u32)->bool{
  if(cell>=${p}CellCapacity){${p}Fail(${p}Preparation,
    ${SPARSE_CM12_FACE_PROJECTION_FAULT.invalidCell}u,cell);return false;}
  var ok=true;let incidences=cm12HotIncidenceRange(cell);
  for(var at=incidences.x;at<incidences.x+incidences.y;at+=1u){let incidence=cm12HotIncidence(at);
    let row=incidence.x;ok=fpaMarkPreparationRow(row,cause,0u,false)&&ok;
    let terms=cm12HotRowTermCount(row);
    for(var ordinal=0u;ordinal<terms;ordinal+=1u){let neighbor=cm12HotRowTermCell(row,ordinal);
      if(neighbor==${p}Invalid||neighbor>=${p}CellCapacity){continue;}
      let closure=cm12HotIncidenceRange(neighbor);
      for(var nested=closure.x;nested<closure.x+closure.y;nested+=1u){
        let closureRow=cm12HotIncidence(nested).x;
        ok=fpaMarkPreparationRow(closureRow,
          cause|${SPARSE_CM12_FACE_PROJECTION_CAUSE.closure}u,1u,false)&&ok;
      }}
  }
  if(ok){atomicAdd(&${arena}[${p}PreparationHeader+${p}DCoveredReceipts],1u);}return ok;
}
@compute @workgroup_size(${workgroupSize})
fn seedSparseCM12FacePreparationBootstrap(@builtin(global_invocation_id)gid:vec3u){
  let row=${preparationBootstrapInvocation
    ? `${preparationBootstrapInvocation}(gid.x)` : "gid.x"};
  if(row==${p}Invalid||row>=${p}RowCapacity){return;}
  let cause=select(${SPARSE_CM12_FACE_PROJECTION_CAUSE.bootstrap}u,
    ${SPARSE_CM12_FACE_PROJECTION_CAUSE.qaOracle}u,${p}QAOracle);
  _=fpaMarkPreparationRow(row,cause,0u,false);
}
@compute @workgroup_size(${workgroupSize})
fn seedSparseCM12FaceProjectionBootstrap(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;if(row>=${p}RowCapacity||!fpaProjectionRowLive(row)){return;}
  let cause=select(${SPARSE_CM12_FACE_PROJECTION_CAUSE.bootstrap}u,
    ${SPARSE_CM12_FACE_PROJECTION_CAUSE.qaOracle}u,${p}QAOracle);
  _=fpaMarkProjectionRow(row,cause,0u,false);
}
fn ${p}SeedPrevious(stage:u32,invocation:u32){let header=${p}Header(stage);
  let count=atomicLoad(&${arena}[header+${p}DPreviousActiveLeafCount]);
  if(invocation>=count){return;}let leaf=atomicLoad(&${arena}[
    ${p}ActiveLeafListBase(stage)+invocation]);if(leaf>=${p}LeafCount(stage)){
      ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.activeLeafCapacity}u,
        leaf*${p}LeafBits);return;}_=${p}QueueLeaf(stage,leaf);
}
@compute @workgroup_size(${workgroupSize})
fn seedSparseCM12PreviousFacePreparationLeaves(@builtin(global_invocation_id)gid:vec3u){
  ${p}SeedPrevious(${p}Preparation,gid.x);
}
@compute @workgroup_size(${workgroupSize})
fn seedSparseCM12PreviousFaceProjectionLeaves(@builtin(global_invocation_id)gid:vec3u){
  ${p}SeedPrevious(${p}Projection,gid.x);
}
@compute @workgroup_size(${workgroupSize})
fn seedSparseCM12ProjectionFromPreparation(
 @builtin(global_invocation_id)gid:vec3u){
  let row=${p}PreparationRankSelect(gid.x);if(row==${p}Invalid){return;}
  if(fpaProjectionRowLive(row)){_=fpaMarkProjectionRow(row,
    ${SPARSE_CM12_FACE_PROJECTION_CAUSE.preparationDependency}u,1u,false);}
}
fn ${p}FinalizeFrontier(stage:u32)->bool{let header=${p}Header(stage);
  if(atomicLoad(&${arena}[header+${p}DPhase])!=${p}PhaseCollecting
    ||atomicLoad(&${arena}[header+${p}DFault])!=0u){return false;}
  if(atomicLoad(&${arena}[header+${p}DExpectedReceipts])
    !=atomicLoad(&${arena}[header+${p}DCoveredReceipts])){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.producerCoverageGap}u,
      ${p}Invalid);return false;}
  atomicStore(&${arena}[header+${p}DRepairIndirectX],atomicLoad(&${arena}[
    header+${p}DDirtyLeafCount]));
  atomicStore(&${arena}[header+${p}DPhase],${p}PhaseRepairing);return true;
}
@compute @workgroup_size(1)
fn finalizeSparseCM12FacePreparationFrontier(){_=${p}FinalizeFrontier(${p}Preparation);}
@compute @workgroup_size(1)
fn finalizeSparseCM12FaceProjectionFrontier(){_=${p}FinalizeFrontier(${p}Projection);}
${repairKernel(p, "Preparation", SPARSE_CM12_FACE_PROJECTION_STAGE.preparation,
    layout.preparation, arena, workgroupSize)}
${repairKernel(p, "Projection", SPARSE_CM12_FACE_PROJECTION_STAGE.projection,
    layout.projection, arena, workgroupSize)}
fn ${p}FinalizePlan(stage:u32,root:u32)->bool{let header=${p}Header(stage);
  if(atomicLoad(&${arena}[header+${p}DPhase])!=${p}PhaseRepairing
    ||atomicLoad(&${arena}[header+${p}DFault])!=0u){return false;}
  let work=atomicLoad(&${arena}[root]);atomicStore(&${arena}[header+${p}DWorkCount],work);
  atomicStore(&${arena}[header+${p}DReusedCount],${p}RowCapacity-work);
  atomicStore(&${arena}[header+${p}DWorkIndirectX],
    (work+${workgroupSize - 1}u)/${workgroupSize}u);
  // Reuse the bootstrap triplet after bootstrap has executed: verification is
  // one workgroup per active leaf and remains copy-isolated from storage use.
  atomicStore(&${arena}[header+${d.reserved0}u],atomicLoad(&${arena}[
    header+${p}DActiveLeafCount]));
  atomicStore(&${arena}[header+${p}DPhase],${p}PhaseExecuting);return true;
}
@compute @workgroup_size(1)
fn finalizeSparseCM12FacePreparationPlan(){
  _=${p}FinalizePlan(${p}Preparation,${prepRoot}u);}
@compute @workgroup_size(1)
fn finalizeSparseCM12FaceProjectionPlan(){
  _=${p}FinalizePlan(${p}Projection,${projectionRoot}u);}
${rankSelect(p, "Preparation", layout.preparation, arena)}
${rankSelect(p, "Projection", layout.projection, arena)}
fn fpaPreparationRowInvocation(invocation:u32)->u32{
  return ${p}PreparationRankSelect(invocation);}
fn fpaPreparationRowCause(row:u32)->u32{
  if(row>=${p}RowCapacity){return 0u;}
  return atomicLoad(&${arena}[${p}PreparationCandidateCause+row]);
}
// Packet execution addresses one already-materialized 256-row authority leaf.
// Each of 64 lanes owns four stable row bits, avoiding one tree rank-select per
// row while preserving the exact candidate bitset and completion protocol.
fn fpaPreparationPacketRow(packet:u32,lane:u32,slot:u32)->u32{
  let header=${p}PreparationHeader;
  if(slot>=4u||packet>=atomicLoad(&${arena}[header+${p}DActiveLeafCount])
    ||atomicLoad(&${arena}[header+${p}DPhase])!=${p}PhaseExecuting){return ${p}Invalid;}
  let leaf=atomicLoad(&${arena}[${p}PreparationActiveLeafList+packet]);
  let local=lane+64u*slot;let word=atomicLoad(&${arena}[
    ${p}PreparationBits+leaf*${p}LeafWords+local/32u]);
  let row=leaf*${p}LeafBits+local;
  return select(${p}Invalid,row,row<${p}RowCapacity&&(word&(1u<<(local&31u)))!=0u);
}
fn fpaProjectionRowInvocation(invocation:u32)->u32{
  return ${p}ProjectionRankSelect(invocation);}
fn ${p}Complete(stage:u32,row:u32)->bool{let header=${p}Header(stage);
  if(row>=${p}RowCapacity||atomicLoad(&${arena}[header+${p}DPhase])!=${p}PhaseExecuting){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.invalidPhase}u,row);return false;}
  let generation=atomicLoad(&${arena}[header+${p}DCandidateGeneration]);
  if(atomicLoad(&${arena}[${p}CandidateGenerationBase(stage)+row])!=generation){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.executionCoverageGap}u,row);return false;}
  atomicStore(&${arena}[${p}AcceptedDependencyBase(stage)+row],atomicLoad(&${arena}[
    ${p}CandidateDependencyBase(stage)+row]));
  atomicStore(&${arena}[${p}ExecutionGenerationBase(stage)+row],generation);return true;
}
fn fpaPreparationComplete(row:u32)->bool{return ${p}Complete(${p}Preparation,row);}
fn fpaProjectionComplete(row:u32)->bool{return ${p}Complete(${p}Projection,row);}
fn fpaStorePreparedAuthority(row:u32,bits:u32){
  if(row<${p}RowCapacity){atomicStore(&${arena}[${p}PreparedAuthority+row],bits);}}
fn fpaPreparedAuthorityBits(row:u32)->u32{
  if(row>=${p}RowCapacity){return 0u;}
  return atomicLoad(&${arena}[${p}PreparedAuthority+row]);}
fn fpaPreparationMustMirrorUnprojected(row:u32)->bool{return !fpaProjectionRowLive(row);}
fn fpaProjectionMustMirror(row:u32)->bool{_=row;return true;}
${verifyKernel(p, "Preparation", SPARSE_CM12_FACE_PROJECTION_STAGE.preparation,
    arena, workgroupSize)}
${verifyKernel(p, "Projection", SPARSE_CM12_FACE_PROJECTION_STAGE.projection,
    arena, workgroupSize)}
fn ${p}Accept(stage:u32)->bool{let header=${p}Header(stage);
  if(atomicLoad(&${arena}[header+${p}DPhase])!=${p}PhaseExecuting
    ||atomicLoad(&${arena}[header+${p}DFault])!=0u){return false;}
  if(atomicLoad(&${arena}[header+${p}DFrameGeneration])!=fpaFrameGeneration()
    ||atomicLoad(&${arena}[header+${p}DTopologyGeneration])!=fpaTopologyGeneration()
    ||atomicLoad(&${arena}[header+${p}DSourceParity])!=fpaSourceParity()
    ||atomicLoad(&${arena}[header+${p}DPolicyBits])!=fpaPolicyBits()){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.topologyGap}u,${p}Invalid);return false;}
  if(stage==${p}Projection
    &&atomicLoad(&${arena}[header+${p}DPCMGeneration])!=fpaPCMGeneration()){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.pcmGenerationGap}u,
      ${p}Invalid);return false;}
  if(atomicLoad(&${arena}[header+${p}DExecutedCount])
      !=atomicLoad(&${arena}[header+${p}DWorkCount])
    ||atomicLoad(&${arena}[header+${p}DVerifiedLeafCount])
      !=atomicLoad(&${arena}[header+${p}DActiveLeafCount])){
    ${p}Fail(stage,${SPARSE_CM12_FACE_PROJECTION_FAULT.executionCoverageGap}u,
      ${p}Invalid);return false;}
  atomicStore(&${arena}[header+${p}DAcceptedGeneration],atomicLoad(&${arena}[
    header+${p}DCandidateGeneration]));
  atomicStore(&${arena}[header+${p}DPhase],${p}PhaseAccepted);return true;
}
@compute @workgroup_size(1)
fn finalizeSparseCM12FacePreparationExecution(){_=${p}Accept(${p}Preparation);}
@compute @workgroup_size(1)
fn finalizeSparseCM12FaceProjectionExecution(){_=${p}Accept(${p}Projection);}
`;
}
