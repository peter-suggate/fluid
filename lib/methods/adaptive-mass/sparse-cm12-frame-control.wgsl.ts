import {
  SPARSE_CM12_FRAME_CONTROL_COVERAGE,
  SPARSE_CM12_FRAME_CONTROL_FAULT,
  SPARSE_CM12_FRAME_CONTROL_FAMILY,
  SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT,
  SPARSE_CM12_FRAME_CONTROL_FLAG,
  SPARSE_CM12_FRAME_CONTROL_HEADER,
  SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS,
  SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS,
  SPARSE_CM12_FRAME_CONTROL_INVALID,
  SPARSE_CM12_FRAME_CONTROL_MAGIC,
  SPARSE_CM12_FRAME_CONTROL_PHASE,
  SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS,
  SPARSE_CM12_FRAME_CONTROL_VERSION,
  type SparseCM12FrameControlLayout,
} from "./sparse-cm12-frame-control";

export interface SparseCM12FrameControlWGSLOptions {
  readonly layout: SparseCM12FrameControlLayout;
  /** Existing `array<atomic<u32>>` storage binding. */
  readonly controlName?: string;
  /** Candidate-only post-transaction D4 seam. No FCA bytes are reserved when
   * omitted, and the baseline WGSL/host layout remains unchanged. */
  readonly authorizedD4Invalidation?: boolean;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError(`${label} must be a WGSL identifier`);
  return value;
};

export function createSparseCM12FrameControlWGSL(
  options: SparseCM12FrameControlWGSLOptions,
): string {
  const l = options.layout;
  const control = identifier(options.controlName ?? "frameControl", "controlName");
  const staticFlags = SPARSE_CM12_FRAME_CONTROL_FLAG.complete
    | SPARSE_CM12_FRAME_CONTROL_FLAG.validated
    | (l.d4Capable ? SPARSE_CM12_FRAME_CONTROL_FLAG.d4Capable : 0)
    | (l.rigidCapable ? SPARSE_CM12_FRAME_CONTROL_FLAG.rigidCapable : 0);
  const authorityCoverage = SPARSE_CM12_FRAME_CONTROL_COVERAGE.authority;
  const h = (word: number) => `${l.baseWords + word}u`;
  const f = (family: number) => `${l.baseWords + SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS
    + SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS * family}u`;
  const authorizedD4Invalidation = options.authorizedD4Invalidation ? /* wgsl */ `
// VDA1 calls this only after its topology success latch matches the accepted
// generation. Every fallible phase/generation check happened in preflight.
// This function intentionally has no fault, retry, or capacity path.
fn cm12FCInvalidateD4Authorized(cause:u32,owner:u32,generation:u32,sealed:bool){
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4Generation)},generation);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.scalarD4Authority)},0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.faceD4Authority)},0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationCause)},cause);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationOwner)},owner);
  atomicOr(&${control}[${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)}],
    ${SPARSE_CM12_FRAME_CONTROL_COVERAGE.scalarD4
      | SPARSE_CM12_FRAME_CONTROL_COVERAGE.faceD4}u);
  if(sealed){
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work}u,0u);
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Bypass}u,1u);
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work}u,0u);
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Bypass}u,1u);
  }
}
` : "";
  return /* wgsl */ `
const cm12FCMagic:u32=0x${SPARSE_CM12_FRAME_CONTROL_MAGIC.toString(16)}u;
const cm12FCVersion:u32=${SPARSE_CM12_FRAME_CONTROL_VERSION}u;
const cm12FCInvalid:u32=0x${SPARSE_CM12_FRAME_CONTROL_INVALID.toString(16)}u;
const cm12FCTotalWords:u32=${SPARSE_CM12_FRAME_CONTROL_TOTAL_WORDS}u;
const cm12FCCellGroups:u32=${l.cellWorkgroups}u;
const cm12FCRowGroups:u32=${l.rowWorkgroups}u;
const cm12FCBodyCapacity:u32=${l.bodyCapacity}u;
const cm12FCStaticFlags:u32=${staticFlags}u;
const cm12FCD4Capable:bool=${l.d4Capable ? "true" : "false"};
const cm12FCRigidCapable:bool=${l.rigidCapable ? "true" : "false"};
const cm12FCPhaseAccepted:u32=${SPARSE_CM12_FRAME_CONTROL_PHASE.accepted}u;
const cm12FCPhaseCollecting:u32=${SPARSE_CM12_FRAME_CONTROL_PHASE.collecting}u;
const cm12FCPhaseSealed:u32=${SPARSE_CM12_FRAME_CONTROL_PHASE.sealed}u;
const cm12FCPhaseFault:u32=${SPARSE_CM12_FRAME_CONTROL_PHASE.fault}u;
const cm12FCCoverageAuthority:u32=${authorityCoverage}u;
const cm12FCCoverageOutput:u32=${SPARSE_CM12_FRAME_CONTROL_COVERAGE.output}u;

fn cm12FCLoad(word:u32)->u32{return atomicLoad(&${control}[word]);}
fn cm12FCStore(word:u32,value:u32){atomicStore(&${control}[word],value);}

fn cm12FCHeaderValid()->bool{
  if(arrayLength(&${control})<${l.baseWords}u+cm12FCTotalWords){return false;}
  if(cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.magic)})!=cm12FCMagic
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.version)})!=cm12FCVersion
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.headerWords)})
      !=${SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS}u
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.totalWords)})!=cm12FCTotalWords
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.flags)})!=cm12FCStaticFlags
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.brickFineResolution)})
      !=${l.brickFineResolution}u
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.presentationPageResolution)})
      !=${l.presentationPageResolution}u
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.indirectWords)})
      !=${SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS}u
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.familyCount)})
      !=${SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT}u
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.indirectBase)})
      !=${SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS}u
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.cellWorkgroups)})!=cm12FCCellGroups
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.rowWorkgroups)})!=cm12FCRowGroups
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyCapacity)})!=cm12FCBodyCapacity
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.requiredAuthorityCoverage)})
      !=cm12FCCoverageAuthority
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.requiredOutputCoverage)})
      !=cm12FCCoverageOutput){return false;}
  for(var word=${SPARSE_CM12_FRAME_CONTROL_HEADER.reservedBase}u;
    word<${SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS}u;word+=1u){
    if(cm12FCLoad(${l.baseWords}u+word)!=0u){return false;}
  }
  return true;
}

fn cm12FCFamilyBase(family:u32)->u32{
  return ${l.baseWords + SPARSE_CM12_FRAME_CONTROL_HEADER_WORDS}u
    +${SPARSE_CM12_FRAME_CONTROL_INDIRECT_WORDS}u*family;
}

fn cm12FCSetTriplet(family:u32,x:u32){
  if(family>=${SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT}u){return;}
  let at=cm12FCFamilyBase(family);cm12FCStore(at,x);cm12FCStore(at+1u,1u);cm12FCStore(at+2u,1u);
}

fn cm12FCIndirectX(family:u32)->u32{
  if(family>=${SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT}u){return 0u;}
  return cm12FCLoad(cm12FCFamilyBase(family));
}

fn cm12FCZeroDispatches(blocked:bool){
  for(var family=0u;family<${SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT}u;family+=1u){
    cm12FCSetTriplet(family,0u);
  }
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.frameBlocked}u,select(0u,1u,blocked));
}

fn cm12FCFail(code:u32,owner:u32){
  let claimed=atomicCompareExchangeWeak(&${control}[${h(SPARSE_CM12_FRAME_CONTROL_HEADER.fault)}],
    ${SPARSE_CM12_FRAME_CONTROL_FAULT.none}u,code);
  if(claimed.exchanged){cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.firstFaultOwner)},owner);}
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)},cm12FCPhaseFault);
  cm12FCZeroDispatches(true);
}

fn cm12FCBegin()->bool{
  if(!cm12FCHeaderValid()){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidHeader}u,cm12FCInvalid);return false;
  }
  if(cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)})!=cm12FCPhaseAccepted){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidPhase}u,cm12FCInvalid);return false;
  }
  let accepted=cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.acceptedGeneration)});
  if(accepted>=0x7ffffffeu){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.generationExhausted}u,accepted);return false;
  }
  let candidate=accepted+1u;
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.candidateGeneration)},candidate);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyGeneration)},0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyCount)},0u);
  // D4 authority is a sticky accepted producer receipt. Frame-local solid
  // predicates can suppress its work without destroying it; injection or a
  // topology producer explicitly invalidates it on device.
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4Generation)},candidate);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationCause)},0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationOwner)},cm12FCInvalid);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)},
    ${SPARSE_CM12_FRAME_CONTROL_COVERAGE.scalarD4
      | SPARSE_CM12_FRAME_CONTROL_COVERAGE.faceD4}u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.fault)},0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.firstFaultOwner)},cm12FCInvalid);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.sealedGeneration)},0u);
  cm12FCZeroDispatches(false);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)},cm12FCPhaseCollecting);
  return true;
}

fn cm12FCCandidateGeneration()->u32{
  return cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.candidateGeneration)});
}
fn cm12FCFrameSealed()->bool{
  return cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)})==cm12FCPhaseSealed;
}
// Candidate effects are authorized either at the sealed tail of a physics
// frame or between frames while the last frame is accepted.  Record the exact
// generation belonging to that phase in VDA1; accepting any other phase would
// make a later no-fail D4 publication ambiguous.
fn cm12FCEffectsPhaseValid()->bool{
  if(!cm12FCHeaderValid()){return false;}
  let phase=cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)});
  return phase==cm12FCPhaseAccepted||phase==cm12FCPhaseSealed;
}
fn cm12FCEffectsGeneration()->u32{
  let phase=cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)});
  return select(cm12FCCandidateGeneration(),
    cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.acceptedGeneration)}),
    phase==cm12FCPhaseAccepted);
}

fn cm12FCPublishBody(bodyCount:u32)->bool{
  if(!cm12FCHeaderValid()||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)})
    !=cm12FCPhaseCollecting){cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidPhase}u,bodyCount);return false;}
  if(bodyCount>cm12FCBodyCapacity){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.bodyCapacity}u,bodyCount);return false;
  }
  if(bodyCount>0u&&!cm12FCRigidCapable){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.capability}u,bodyCount);return false;
  }
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyGeneration)},cm12FCCandidateGeneration());
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyCount)},bodyCount);
  atomicOr(&${control}[${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)}],
    ${SPARSE_CM12_FRAME_CONTROL_COVERAGE.body}u);
  return true;
}

fn cm12FCPublishD4(scalarAuthority:bool,faceAuthority:bool)->bool{
  if(!cm12FCHeaderValid()||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)})
    !=cm12FCPhaseCollecting){cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidPhase}u,0u);return false;}
  if((scalarAuthority||faceAuthority)&&!cm12FCD4Capable){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.capability}u,0u);return false;
  }
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4Generation)},cm12FCCandidateGeneration());
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.scalarD4Authority)},
    select(0u,1u,scalarAuthority));
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.faceD4Authority)},
    select(0u,1u,faceAuthority));
  atomicOr(&${control}[${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)}],
    ${SPARSE_CM12_FRAME_CONTROL_COVERAGE.scalarD4
      | SPARSE_CM12_FRAME_CONTROL_COVERAGE.faceD4}u);
  return true;
}

// Any GPU producer may invalidate D4 locally. Invalidation is not a frame
// fault: the fixed D4 work triplet becomes x=0 and its bypass becomes live.
fn cm12FCInvalidateD4(cause:u32,owner:u32)->bool{
  let phase=cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)});
  if(!cm12FCHeaderValid()||(phase!=cm12FCPhaseAccepted
    &&phase!=cm12FCPhaseCollecting&&phase!=cm12FCPhaseSealed)){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidPhase}u,owner);return false;
  }
  let generation=select(cm12FCCandidateGeneration(),
    cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.acceptedGeneration)}),
    phase==cm12FCPhaseAccepted);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4Generation)},generation);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.scalarD4Authority)},0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.faceD4Authority)},0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationCause)},cause);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationOwner)},owner);
  atomicOr(&${control}[${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)}],
    ${SPARSE_CM12_FRAME_CONTROL_COVERAGE.scalarD4
      | SPARSE_CM12_FRAME_CONTROL_COVERAGE.faceD4}u);
  if(phase==cm12FCPhaseSealed){
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work}u,0u);
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Bypass}u,1u);
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work}u,0u);
    cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Bypass}u,1u);
  }
  return true;
}

fn cm12FCSeal()->bool{
  if(!cm12FCHeaderValid()||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)})
    !=cm12FCPhaseCollecting){cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidPhase}u,0u);return false;}
  let candidate=cm12FCCandidateGeneration();
  let coverage=cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)});
  if((coverage&cm12FCCoverageAuthority)!=cm12FCCoverageAuthority){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.missingEvidence}u,
      cm12FCCoverageAuthority&~coverage);return false;
  }
  if(cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyGeneration)})!=candidate
    ||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.d4Generation)})!=candidate){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.staleGeneration}u,candidate);return false;
  }
  let bodyCount=cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.bodyCount)});
  let bodyLive=cm12FCRigidCapable&&bodyCount>0u;
  let solidLive=bodyLive;
  let scalarD4=cm12FCD4Capable&&!solidLive
    &&cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.scalarD4Authority)})!=0u;
  let faceD4=cm12FCD4Capable&&!solidLive
    &&cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.faceD4Authority)})!=0u;
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work}u,
    select(0u,cm12FCCellGroups,scalarD4));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Bypass}u,
    select(1u,0u,scalarD4));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work}u,
    select(0u,cm12FCCellGroups,faceD4));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Bypass}u,
    select(1u,0u,faceD4));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellWork}u,
    select(0u,cm12FCCellGroups,solidLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellBypass}u,
    select(1u,0u,solidLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.solidRowWork}u,
    select(0u,cm12FCRowGroups,solidLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.solidRowBypass}u,
    select(1u,0u,solidLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyWork}u,
    select(0u,cm12FCCellGroups,bodyLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyBypass}u,
    select(1u,0u,bodyLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowWork}u,
    select(0u,cm12FCRowGroups,bodyLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowBypass}u,
    select(1u,0u,bodyLive));
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.frameWork}u,1u);
  cm12FCSetTriplet(${SPARSE_CM12_FRAME_CONTROL_FAMILY.frameBlocked}u,0u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.sealedGeneration)},candidate);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)},cm12FCPhaseSealed);
  return true;
}

fn cm12FCPublishOutput(mask:u32)->bool{
  if(!cm12FCHeaderValid()||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)})
    !=cm12FCPhaseSealed||(mask&~cm12FCCoverageOutput)!=0u){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidPhase}u,mask);return false;
  }
  atomicOr(&${control}[${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)}],mask);
  return true;
}

fn cm12FCCommit()->bool{
  if(!cm12FCHeaderValid()||cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)})
    !=cm12FCPhaseSealed){cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.invalidPhase}u,0u);return false;}
  let candidate=cm12FCCandidateGeneration();
  let coverage=cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.coverage)});
  if((coverage&cm12FCCoverageOutput)!=cm12FCCoverageOutput){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.incompleteOutput}u,
      cm12FCCoverageOutput&~coverage);return false;
  }
  if(cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.sealedGeneration)})!=candidate){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.staleGeneration}u,candidate);return false;
  }
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.acceptedGeneration)},candidate);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity)},
    cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity)})^1u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.faceParity)},
    cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.faceParity)})^1u);
  atomicAdd(&${control}[${h(SPARSE_CM12_FRAME_CONTROL_HEADER.committedFrames)}],1u);
  cm12FCStore(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.phase)},cm12FCPhaseAccepted);
  return true;
}

fn cm12FCAcceptedGeneration()->u32{
  return cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.acceptedGeneration)});
}
fn cm12FCSourceScalarParity()->u32{return cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity)});}
fn cm12FCDestinationScalarParity()->u32{return cm12FCSourceScalarParity()^1u;}
fn cm12FCSourceFaceParity()->u32{return cm12FCLoad(${h(SPARSE_CM12_FRAME_CONTROL_HEADER.faceParity)});}
fn cm12FCDestinationFaceParity()->u32{return cm12FCSourceFaceParity()^1u;}
${authorizedD4Invalidation}
`;
}
