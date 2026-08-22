import {
  SPARSE_CM12_VELOCITY_EXTENSION_FLAG,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  type SparseCM12VelocityExtensionLayout,
} from "./sparse-cm12-velocity-extension";
import {
  SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT as F,
  SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER as H,
  SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER_WORDS,
  SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID,
  SPARSE_CM12_VEX_DELTA_AUTHORITY_MAGIC,
  SPARSE_CM12_VEX_DELTA_AUTHORITY_PHASE as P,
  SPARSE_CM12_VEX_DELTA_AUTHORITY_VERSION,
  type SparseCM12VexDeltaAuthorityLayout,
} from "./sparse-cm12-vex-delta-authority";

export interface SparseCM12VexDeltaAuthorityWGSLOptions {
  readonly layout: SparseCM12VexDeltaAuthorityLayout;
  readonly velocityExtensionLayout: SparseCM12VelocityExtensionLayout;
  /** Existing `array<atomic<u32>>` containing VDA1 and VEX1. */
  readonly arenaName?: string;
  /** Frozen candidate-slot predicates. They must not consult the accepted slot. */
  readonly finalCellActiveFunction: string;
  readonly finalCellRetiredFunction: string;
  readonly topologyGenerationExpression: string;
  /** Topology/injection transactions preflight a sealed VEX generation and
   * reopen it only after authorization. Ordinary users keep the stricter
   * collecting-roots requirement. */
  readonly allowSealedPreflight?: boolean;
  /** FCA expressions used only when `d4Required` is set in BeginPreflight. */
  readonly frameGenerationExpression: string;
  readonly framePhaseValidExpression: string;
  /** Optional prefix, permitting focused Dawn composition beside VXP1. */
  readonly prefix?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/**
 * Exact two-phase ABI. All fallible work is in traversal preflight and
 * `SealPreflight`, before any
 * VEX/cache write. The two `PublishAuthorized*` helpers contain no capacity
 * check, retry loop, or fault path and are valid only under the authorization
 * receipt. Input requests may repeat endpoints: the VEX root generation stamp
 * performs the exact unique merge during infallible publication. Coverage is
 * preflighted from producer-authored count plus commutative request receipts.
 */
export function createSparseCM12VexDeltaAuthorityWGSL(
  options: SparseCM12VexDeltaAuthorityWGSLOptions,
): string {
  const l = options.layout, v = options.velocityExtensionLayout;
  const arena = identifier(options.arenaName ?? "activity", "arenaName");
  const active = identifier(options.finalCellActiveFunction, "finalCellActiveFunction");
  const retired = identifier(options.finalCellRetiredFunction, "finalCellRetiredFunction");
  const p = identifier(options.prefix ?? "vda1", "prefix");
  const at = (word: number) => `${l.baseWords + word}u`;
  const vh = (word: number) => `${v.headerBaseWords + word}u`;
  return /* wgsl */ `
const ${p}Invalid:u32=0x${SPARSE_CM12_VEX_DELTA_AUTHORITY_INVALID.toString(16)}u;
const ${p}PhaseCollecting:u32=${P.collecting}u;
const ${p}PhasePreflighted:u32=${P.preflighted}u;
const ${p}PhaseAuthorized:u32=${P.authorized}u;
const ${p}PhasePublished:u32=${P.published}u;
fn ${p}Load(word:u32)->u32{return atomicLoad(&${arena}[word]);}
fn ${p}Store(word:u32,value:u32){atomicStore(&${arena}[word],value);}
fn ${p}HeaderValid()->bool{
  return arrayLength(&${arena})>=${l.totalWords}u
    &&${p}Load(${at(H.magic)})==0x${SPARSE_CM12_VEX_DELTA_AUTHORITY_MAGIC.toString(16)}u
    &&${p}Load(${at(H.version)})==${SPARSE_CM12_VEX_DELTA_AUTHORITY_VERSION}u
    &&${p}Load(${at(H.headerWords)})==${SPARSE_CM12_VEX_DELTA_AUTHORITY_HEADER_WORDS}u
    &&${p}Load(${at(H.rootInputCapacity)})==0u
    &&${p}Load(${at(H.retiredInputCapacity)})==0u
    &&${p}Load(${at(H.cellCapacity)})==${l.cellCapacity}u;
}
fn ${p}Fail(code:u32,rank:u32,cell:u32){
  ${p}Store(${at(H.fault)},code);${p}Store(${at(H.firstFaultRank)},rank);
  ${p}Store(${at(H.firstFaultCell)},cell);${p}Store(${at(H.phase)},0xffffffffu);
}
fn ${p}VexGeneration()->u32{return ${p}Load(${vh(
    SPARSE_CM12_VELOCITY_EXTENSION_HEADER.candidateGeneration)});}
fn ${p}VexCollecting()->bool{return (${p}Load(${vh(
    SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)})&${
      SPARSE_CM12_VELOCITY_EXTENSION_FLAG.collectingRoots}u)!=0u;}

fn ${p}BeginPreflight(transactionGeneration:u32,expectedRootCount:u32,
 expectedRetiredCount:u32,expectedRootHash:u32,expectedRetiredHash:u32,
 expectedFrameGeneration:u32,injectionRequired:bool,d4Required:bool){
  ${p}Store(${at(H.phase)},${p}PhaseCollecting);
  ${p}Store(${at(H.transactionGeneration)},transactionGeneration);
  ${p}Store(${at(H.topologyGeneration)},${options.topologyGenerationExpression});
  ${p}Store(${at(H.vexGeneration)},${p}VexGeneration());
  ${p}Store(${at(H.rootInputCount)},expectedRootCount);
  ${p}Store(${at(H.retiredInputCount)},expectedRetiredCount);
  ${p}Store(${at(H.existingRootCount)},0u);${p}Store(${at(H.newRootCount)},0u);
  ${p}Store(${at(H.expectedFrameGeneration)},expectedFrameGeneration);
  ${p}Store(${at(H.rootPublishedCount)},0u);${p}Store(${at(H.retiredPublishedCount)},0u);
  ${p}Store(${at(H.rootCoverageCount)},0u);${p}Store(${at(H.retiredCoverageCount)},0u);
  ${p}Store(${at(H.authorizationGeneration)},0u);${p}Store(${at(H.successGeneration)},0u);
  ${p}Store(${at(H.injectionRequired)},select(0u,1u,injectionRequired));
  ${p}Store(${at(H.injectionPublished)},0u);
  ${p}Store(${at(H.d4Required)},select(0u,1u,d4Required));${p}Store(${at(H.d4Published)},0u);
  ${p}Store(${at(H.fault)},0u);${p}Store(${at(H.firstFaultRank)},${p}Invalid);
  ${p}Store(${at(H.firstFaultCell)},${p}Invalid);
  ${p}Store(${at(H.expectedRootHash)},expectedRootHash);
  ${p}Store(${at(H.observedRootHash)},0u);
  ${p}Store(${at(H.expectedRetiredHash)},expectedRetiredHash);
  ${p}Store(${at(H.observedRetiredHash)},0u);
}
fn ${p}RootReceipt(cell:u32,cause:u32)->u32{
  return (cell*0x9e3779b9u)^(cause*0x85ebca6bu)^0xc2b2ae35u;
}
fn ${p}RetiredReceipt(cell:u32)->u32{return (cell*0x27d4eb2du)^0x165667b1u;}
// The producer census and the semantic preflight deliberately run as two
// ordered O(delta) traversals.  They share no request storage: the first owns
// only the expected count/hash words, while the second owns the observed
// count/hash words below.  A missing, duplicated, or differently classified
// request therefore rejects before authorization.
fn ${p}BeginCensus(transactionGeneration:u32,injectionRequired:bool,d4Required:bool){
  ${p}BeginPreflight(transactionGeneration,0u,0u,0u,0u,
    ${options.frameGenerationExpression},injectionRequired,d4Required);
}
fn ${p}CensusRoot(cell:u32,cause:u32){
  atomicAdd(&${arena}[${at(H.rootInputCount)}],1u);
  atomicAdd(&${arena}[${at(H.expectedRootHash)}],${p}RootReceipt(cell,cause));
}
fn ${p}CensusRetired(cell:u32){
  atomicAdd(&${arena}[${at(H.retiredInputCount)}],1u);
  atomicAdd(&${arena}[${at(H.expectedRetiredHash)}],${p}RetiredReceipt(cell));
}
fn ${p}BeginSemanticPreflight()->bool{
  if(!${p}HeaderValid()||${p}Load(${at(H.phase)})!=${p}PhaseCollecting){
    ${p}Fail(${F.invalidPhase}u,${p}Invalid,${p}Invalid);return false;}
  ${p}Store(${at(H.rootCoverageCount)},0u);
  ${p}Store(${at(H.retiredCoverageCount)},0u);
  ${p}Store(${at(H.observedRootHash)},0u);
  ${p}Store(${at(H.observedRetiredHash)},0u);
  return true;
}
fn ${p}PreflightRoot(cell:u32,cause:u32){
  let rank=atomicAdd(&${arena}[${at(H.rootCoverageCount)}],1u);
  if(cell>=${l.cellCapacity}u||cause==0u||!${active}(cell)){
    atomicOr(&${arena}[${at(H.fault)}],${F.invalidRoot}u);
    ${p}Store(${at(H.firstFaultRank)},rank);${p}Store(${at(H.firstFaultCell)},cell);
  }
  atomicAdd(&${arena}[${at(H.observedRootHash)}],${p}RootReceipt(cell,cause));
}
fn ${p}PreflightRetired(cell:u32){
  let rank=atomicAdd(&${arena}[${at(H.retiredCoverageCount)}],1u);
  if(cell>=${l.cellCapacity}u||!${retired}(cell)){
    atomicOr(&${arena}[${at(H.fault)}],${F.invalidRetirement}u);
    ${p}Store(${at(H.firstFaultRank)},rank);${p}Store(${at(H.firstFaultCell)},cell);
  }
  atomicAdd(&${arena}[${at(H.observedRetiredHash)}],${p}RetiredReceipt(cell));
}

// Singleton receipt seal. The preceding exact traversal is O(delta); this is
// the last operation allowed to reject.
fn ${p}SealPreflight(expectedTopologyGeneration:u32,expectedVexGeneration:u32)->bool{
  if(!${p}HeaderValid()||${p}Load(${at(H.phase)})!=${p}PhaseCollecting){
    ${p}Fail(${F.invalidPhase}u,${p}Invalid,${p}Invalid);return false;}
  if(${p}Load(${at(H.topologyGeneration)})!=expectedTopologyGeneration){
    ${p}Fail(${F.staleTopologyGeneration}u,${p}Invalid,expectedTopologyGeneration);return false;}
  if(${options.allowSealedPreflight ? "false" : `!${p}VexCollecting()`}
    ||${p}VexGeneration()!=expectedVexGeneration
    ||${p}Load(${at(H.vexGeneration)})!=expectedVexGeneration){
    ${p}Fail(${F.staleVexGeneration}u,${p}Invalid,expectedVexGeneration);return false;}
  if(${p}Load(${at(H.d4Required)})!=0u
    &&(!(${options.framePhaseValidExpression})
      ||(${options.frameGenerationExpression})!=${p}Load(${at(H.expectedFrameGeneration)}))){
    ${p}Fail(${F.invalidPhase}u,${p}Invalid,${p}Load(${at(H.expectedFrameGeneration)}));return false;}
  let rootCount=${p}Load(${at(H.rootInputCount)});
  let retiredCount=${p}Load(${at(H.retiredInputCount)});
  if(${p}Load(${at(H.fault)})!=0u){${p}Store(${at(H.phase)},0xffffffffu);return false;}
  if(${p}Load(${at(H.rootCoverageCount)})!=rootCount
    ||${p}Load(${at(H.retiredCoverageCount)})!=retiredCount
    ||${p}Load(${at(H.observedRootHash)})!=${p}Load(${at(H.expectedRootHash)})
    ||${p}Load(${at(H.observedRetiredHash)})!=${p}Load(${at(H.expectedRetiredHash)})){
    ${p}Fail(${F.missingCoverage}u,${p}Invalid,${p}Invalid);return false;}
  let rootBase=${p}Load(${vh(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)});
  if(rootBase>${v.cellCapacity}u){${p}Fail(${F.rootCapacity}u,${p}Invalid,rootBase);return false;}
  ${p}Store(${at(H.phase)},${p}PhasePreflighted);return true;
}
fn ${p}PreflightReady(transactionGeneration:u32)->bool{
  return ${p}Load(${at(H.phase)})==${p}PhasePreflighted
    &&${p}Load(${at(H.fault)})==0u
    &&${p}Load(${at(H.transactionGeneration)})==transactionGeneration;
}
fn ${p}Authorize(){
  ${p}Store(${at(H.authorizationGeneration)},${p}Load(${at(H.transactionGeneration)}));
  ${p}Store(${at(H.phase)},${p}PhaseAuthorized);
}
fn ${p}TransactionAuthorized()->bool{return ${p}Load(${at(H.phase)})==${p}PhaseAuthorized
  &&${p}Load(${at(H.authorizationGeneration)})==${p}Load(${at(H.transactionGeneration)});}

// Authorized-only helpers: no capacity branch, retry, compare-exchange or
// fault call is permitted below this boundary.
fn ${p}PublishAuthorizedRoot(cell:u32,cause:u32){
  let generation=${p}Load(${at(H.vexGeneration)});
  let previous=atomicExchange(&${arena}[${v.rootStampBaseWords}u+cell],generation);
  if(previous!=generation){
    // All duplicate requests in one authorized producer batch carry the same
    // cause. Thus a racing duplicate OR and this store are value-identical.
    ${p}Store(${v.rootCauseBaseWords}u+cell,cause);
    let slot=atomicAdd(&${arena}[${vh(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)}],1u);
    ${p}Store(${v.rootListBaseWords}u+slot,cell);
    atomicAdd(&${arena}[${at(H.newRootCount)}],1u);
  }else{atomicOr(&${arena}[${v.rootCauseBaseWords}u+cell],cause);
    atomicAdd(&${arena}[${at(H.existingRootCount)}],1u);}
  atomicAdd(&${arena}[${at(H.rootPublishedCount)}],1u);
}
fn ${p}PublishAuthorizedRetirement(cell:u32,cause:u32){
  ${p}Store(${v.acceptedOwnerBaseWords}u+cell,${p}Invalid);
  ${p}Store(${v.acceptedDepthBaseWords}u+cell,${p}Invalid);
  ${p}Store(${v.rootCauseBaseWords}u+cell,cause);
  atomicAdd(&${arena}[${at(H.retiredPublishedCount)}],1u);
}
fn ${p}SealPublicationNoFail(){${p}Store(${at(H.phase)},${p}PhasePublished);}
// Called by the sole topology finalizer immediately BEFORE its unconditional
// selector store. The target generation cannot satisfy TransactionSucceeded
// until that final store makes the same generation accepted. A rejected
// transaction never calls this function, and the selector remains the final
// publication instruction.
fn ${p}CommitSuccessNoFail(targetTopologyGeneration:u32){
  ${p}Store(${at(H.successGeneration)},targetTopologyGeneration);
}
fn ${p}TransactionSucceeded(acceptedTopologyGeneration:u32)->bool{
  return ${p}Load(${at(H.phase)})==${p}PhasePublished
    &&${p}Load(${at(H.successGeneration)})==acceptedTopologyGeneration;
}
fn ${p}MarkInjectionPublishedNoFail(){${p}Store(${at(H.injectionPublished)},1u);}
fn ${p}MarkD4PublishedNoFail(){${p}Store(${at(H.d4Published)},1u);}
`;
}
