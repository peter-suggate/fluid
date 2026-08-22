import {
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  SPARSE_CM12_VELOCITY_EXTENSION_CAUSE,
  SPARSE_CM12_VELOCITY_EXTENSION_FLAG,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS,
  SPARSE_CM12_VELOCITY_EXTENSION_MAGIC,
  SPARSE_CM12_VELOCITY_EXTENSION_VERSION,
  type SparseCM12VelocityExtensionLayout,
} from "./sparse-cm12-velocity-extension";

export interface SparseCM12VelocityExtensionWGSLOptions {
  readonly layout: SparseCM12VelocityExtensionLayout;
  /** Existing `array<atomic<u32>>` storage arena. */
  readonly arenaName?: string;
  /** Existing `array<f32>` state arena. */
  readonly stateName?: string;
  /** Persistent vec4f cache appended to the state arena. */
  readonly acceptedVelocityFloatBase: number;
  /** Device-authored accepted topology generation expression. */
  readonly topologyGenerationExpression?: string;
  /** FCA1 accepted generation whose scalar/velocity source banks are read. */
  readonly sourceFrameGenerationExpression?: string;
  /** FCA1 candidate generation current-frame producers are authoring. */
  readonly nextFrameGenerationExpression?: string;
  /** True only after FCA1 has sealed the frame authority and dispatches. */
  readonly frameAuthorityReadyExpression?: string;
  /** Candidate transaction phase required before injection may reopen VEX. */
  readonly injectionReopenReadyExpression?: string;
  /**
   * Optional resident provenance bridge. When supplied it must define:
   * `<prefix>VelocityExtensionRoot(cell,cause,generation)->bool`,
   * `<prefix>VelocityExtensionClosure(cell,depth,generation)->bool`,
   * `<prefix>VelocityExtensionScheduled(cell,depth,generation)->bool`, and
   * `<prefix>VelocityExtensionOwner(cell)->u32`, and
   * `<prefix>VelocityExtensionFault(cell,depth)`.
   *
   * The bridge authors CMD/FPL stage-0 direct/closure evidence and faults FCA;
   * VEX1 itself remains independent of the resident's tile-addressing ABI.
   */
  readonly provenanceHookPrefix?: string;
  /**
   * Optional persistent effective-velocity publication hook. When supplied it
   * must define `<prefix>PublishVexAcceptedEffectiveVelocity(cell, value)`.
   * The hook runs only after a fault-free blast cell has been accepted.
   */
  readonly effectiveVelocityHookPrefix?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/** Binding-free VEX1 helpers and kernels for the resident CM12 shader. */
export function createSparseCM12VelocityExtensionWGSL(
  options: SparseCM12VelocityExtensionWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "activity", "arenaName");
  const state = identifier(options.stateName ?? "state", "stateName");
  const velocityBase = options.acceptedVelocityFloatBase;
  if (!Number.isSafeInteger(velocityBase) || velocityBase < 0) {
    throw new RangeError("acceptedVelocityFloatBase must be a non-negative safe integer");
  }
  const topologyGeneration = options.topologyGenerationExpression
    ?? `atomicLoad(&${arena}[12])`;
  const sourceFrameGeneration = options.sourceFrameGenerationExpression
    ?? "cm12ExtensionAcceptedGeneration()+1u";
  const nextFrameGeneration = options.nextFrameGenerationExpression
    ?? `${sourceFrameGeneration}+1u`;
  const frameAuthorityReady = options.frameAuthorityReadyExpression ?? "true";
  const injectionReopenReady = options.injectionReopenReadyExpression ?? "true";
  const hook = options.provenanceHookPrefix
    ? identifier(options.provenanceHookPrefix, "provenanceHookPrefix") : undefined;
  const effectiveVelocityHook = options.effectiveVelocityHookPrefix
    ? identifier(options.effectiveVelocityHookPrefix,
      "effectiveVelocityHookPrefix") : undefined;
  const hookFault = hook ? `${hook}VelocityExtensionFault(cell,depth);` : "";
  const hookRoot = hook
    ? `if(!${hook}VelocityExtensionRoot(cell,cause,generation)){cm12ExtensionFail(cell,0u);return false;}`
    : "";
  const hookClosure = hook
    ? `if(depth>0u&&!${hook}VelocityExtensionClosure(cell,depth,generation)){cm12ExtensionFail(cell,depth);return false;}`
    : "";
  const hookScheduled = hook
    ? `if(!${hook}VelocityExtensionScheduled(cell,cm12ExtensionLoad(cm12ExtensionBlastDepth+cell),cm12ExtensionGeneration())){cm12ExtensionStore(cm12ExtensionCandidateDepth+cell,1u);cm12ExtensionFail(cell,cm12ExtensionLoad(cm12ExtensionBlastDepth+cell));return;}`
    : "";
  const publishEffectiveVelocity = effectiveVelocityHook
    ? `${effectiveVelocityHook}PublishVexAcceptedEffectiveVelocity(cell,vec4f(\n    ${state}[output],${state}[output+1u],${state}[output+2u],${state}[output+3u]));`
    : "";
  const h = (word: number) => `${layout.headerBaseWords + word}u`;
  return /* wgsl */ `
const cm12ExtensionMagic:u32=0x${SPARSE_CM12_VELOCITY_EXTENSION_MAGIC.toString(16)}u;
const cm12ExtensionVersion:u32=${SPARSE_CM12_VELOCITY_EXTENSION_VERSION}u;
const cm12ExtensionInvalid:u32=0xffffffffu;
const cm12ExtensionCapacity:u32=${layout.cellCapacity}u;
const cm12ExtensionRootStamp:u32=${layout.rootStampBaseWords}u;
const cm12ExtensionBlastStamp:u32=${layout.blastStampBaseWords}u;
const cm12ExtensionBlastDepth:u32=${layout.blastDepthBaseWords}u;
const cm12ExtensionCandidateDepth:u32=${layout.candidateDepthBaseWords}u;
const cm12ExtensionRootCause:u32=${layout.rootCauseBaseWords}u;
const cm12ExtensionRootList:u32=${layout.rootListBaseWords}u;
const cm12ExtensionFrontierA:u32=${layout.frontierABaseWords}u;
const cm12ExtensionFrontierB:u32=${layout.frontierBBaseWords}u;
const cm12ExtensionBlastList:u32=${layout.blastListBaseWords}u;
const cm12ExtensionAcceptedDepth:u32=${layout.acceptedDepthBaseWords}u;
const cm12ExtensionAcceptedOwner:u32=${layout.acceptedOwnerBaseWords}u;
const cm12ExtensionReuseStamp:u32=${layout.reuseStampBaseWords}u;
const cm12ExtensionAcceptedVelocity:u32=${velocityBase}u;
const cm12ExtensionFlagCollecting:u32=${SPARSE_CM12_VELOCITY_EXTENSION_FLAG.collectingRoots}u;
const cm12ExtensionFlagSealed:u32=${SPARSE_CM12_VELOCITY_EXTENSION_FLAG.recurrenceSealed}u;
const cm12ExtensionFlagFault:u32=${SPARSE_CM12_VELOCITY_EXTENSION_FLAG.fault}u;
// The reserved header word is a local lifecycle receipt. In particular it
// distinguishes a fault while authoring FPL Next from a fault while consuming
// FPL Current, so planning can never retroactively corrupt the accepted plan.
const cm12ExtensionPhaseCollecting:u32=0u;
const cm12ExtensionPhasePlanning:u32=1u;
const cm12ExtensionPhasePlanned:u32=2u;
const cm12ExtensionPhaseConsuming:u32=3u;
override EXTENSION_FRONTIER_DEPTH:u32=1u;
override EXTENSION_RECURRENCE_DEPTH:u32=1u;

fn cm12ExtensionLoad(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn cm12ExtensionStore(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn cm12ExtensionGeneration()->u32{
  return cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.candidateGeneration)});
}
fn cm12ExtensionAcceptedGeneration()->u32{
  return cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.acceptedGeneration)});
}
fn cm12ExtensionFaulted()->bool{
  return (cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)})
    &cm12ExtensionFlagFault)!=0u;
}
fn cm12ExtensionZeroDispatches(){
  // X is the only indirect dimension that must be zero to reject work. Keep
  // Y/Z intact: after the first fault the resident bridge repurposes those
  // words for immutable FPL provenance, and later fail-closed drains must not
  // erase the winning invocation's receipt.
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBCount)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastCount)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchX)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchX)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchX)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchX)},0u);
}
fn cm12ExtensionReceiptGeneration()->u32{
  let flags=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)});
  return select(cm12ExtensionAcceptedGeneration(),cm12ExtensionGeneration(),
    (flags&cm12ExtensionFlagSealed)!=0u);
}
fn cm12ExtensionHeaderValid()->bool{
  return arrayLength(&${arena})>=${layout.totalWords}u
    &&cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.magic)})==cm12ExtensionMagic
    &&cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.version)})==cm12ExtensionVersion
    &&cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.headerWords)})
      ==${SPARSE_CM12_VELOCITY_EXTENSION_HEADER_WORDS}u
    &&cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.capacity)})
      ==cm12ExtensionCapacity;
}
fn cm12ExtensionFail(cell:u32,depth:u32){
  // A failed wide guard is one transaction fault, not one fault per lane.
  // The first flag transition owns the entire fail-closed publication. Losing
  // lanes do one atomic operation and retire; in particular they must not
  // stampede the shared indirect header or FPL fault receipt.
  let priorFlags=atomicOr(
    &${arena}[${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)}],
    cm12ExtensionFlagFault);
  if((priorFlags&cm12ExtensionFlagFault)!=0u){return;}
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.faultCount)},1u);
  // Compatibility field until the receipt ABI splits coverage from faults.
  // It is deliberately the same binary latch, never a failed-lane counter.
  cm12ExtensionStore(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.uncoveredWriteCount)},1u);
  cm12ExtensionZeroDispatches();
  // The winner clears the overloaded QA words once before publishing its FPL
  // receipt. Later fail-closed drains clear X only and preserve this snapshot.
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchY)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchZ)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchY)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchZ)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchY)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchZ)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchY)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchZ)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultCell)},cell);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultDepth)},depth);
  ${hookFault}
}
fn cm12ExtensionOwner(cell:u32)->u32{
  ${hook ? `return ${hook}VelocityExtensionOwner(cell);`
    : "return cellBrick(cell)|(acceptedBrickResolution(cellBrick(cell))<<24u);"}
}
fn cm12ExtensionAcceptedVelocityChanged(cell:u32,velocity:vec3f)->bool{
  if(cell>=cm12ExtensionCapacity
    ||cm12ExtensionLoad(cm12ExtensionAcceptedOwner+cell)!=cm12ExtensionOwner(cell)){
    return true;
  }
  let at=cm12ExtensionAcceptedVelocity+4u*cell;
  return any(bitcast<vec3u>(vec3f(${state}[at],${state}[at+1u],${state}[at+2u]))
    !=bitcast<vec3u>(velocity));
}

@compute @workgroup_size(1)
fn beginVelocityExtensionCandidate(){
  if(!cm12ExtensionHeaderValid()){cm12ExtensionFail(cm12ExtensionInvalid,0u);return;}
  let flags=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)});
  if((flags&cm12ExtensionFlagFault)!=0u){cm12ExtensionZeroDispatches();return;}
  let accepted=cm12ExtensionAcceptedGeneration();
  let sourceGeneration=${sourceFrameGeneration};
  let candidate=cm12ExtensionGeneration();
  let construction=accepted==0u&&candidate==sourceGeneration;
  let runtime=(${frameAuthorityReady})&&accepted==sourceGeneration
    &&candidate==${nextFrameGeneration};
  let injectionReplan=accepted!=0u&&accepted+1u==sourceGeneration
    &&candidate==sourceGeneration;
  if((flags&cm12ExtensionFlagCollecting)==0u||sourceGeneration==0u
    ||sourceGeneration>=0x7ffffffeu
    ||(!construction&&!runtime&&!injectionReplan)){
    cm12ExtensionFail(cm12ExtensionInvalid,0u);return;
  }
  // Planning is the late-frame half of VEX. Preserve the producer-owned roots
  // and reset only transient closure/recurrence receipts. The sealed blast is
  // consumed at the start of the following frame.
  let executedCurrent=cm12ExtensionLoad(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.executedCellCount)});
  let reusedCurrent=cm12ExtensionLoad(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reusedCellCount)});
  for(var at=${SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount}u;
      at<=${SPARSE_CM12_VELOCITY_EXTENSION_HEADER.uncoveredWriteCount}u;at+=1u){
    cm12ExtensionStore(${layout.headerBaseWords}u+at,0u);
  }
  // Root/blast now describe the planned next generation, while these two
  // counters remain the completed current-generation execution receipt.
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.executedCellCount)},
    executedCurrent);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reusedCellCount)},
    reusedCurrent);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchY)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchZ)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchY)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchZ)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchY)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchZ)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchY)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchZ)},1u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultCell)},cm12ExtensionInvalid);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultDepth)},cm12ExtensionInvalid);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.topologyGeneration)},
    ${topologyGeneration});
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
    cm12ExtensionPhasePlanning);
}

// An out-of-band liquid injection occurs after the preceding frame sealed its
// next blast. Reopen that same generation without dropping any existing root;
// injection/topology producers append, then the ordinary planning half reseals
// an exact replacement before FPL is rebuilt.
@compute @workgroup_size(1)
fn reopenVelocityExtensionPlanForInjection(){
  if(!(${injectionReopenReady})||!cm12ExtensionHeaderValid()
    ||cm12ExtensionFaulted()){return;}
  let flags=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)});
  let accepted=cm12ExtensionAcceptedGeneration();
  let candidate=cm12ExtensionGeneration();
  let source=${sourceFrameGeneration};
  if((flags&cm12ExtensionFlagSealed)==0u||candidate!=source
    ||(accepted!=0u&&accepted+1u!=candidate)){
    cm12ExtensionFail(cm12ExtensionInvalid,0u);return;
  }
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)},
    cm12ExtensionFlagCollecting);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
    cm12ExtensionPhaseCollecting);
}

fn cm12ExtensionRecordRoot(cell:u32,cause:u32)->bool{
  if(cell>=cm12ExtensionCapacity||!cellActive(cell)){
    cm12ExtensionFail(cell,0u);return false;
  }
  let generation=cm12ExtensionGeneration();
  if((cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)})
    &cm12ExtensionFlagCollecting)==0u){
    cm12ExtensionFail(cell,0u);return false;
  }
  for(var spin=0u;spin<64u;spin+=1u){
    let observed=cm12ExtensionLoad(cm12ExtensionRootStamp+cell);
    if(observed==generation){
      atomicOr(&${arena}[cm12ExtensionRootCause+cell],cause);
      ${hookRoot}
      return true;
    }
    // Another invocation from this producer dispatch already owns list
    // publication. Calls within one producer dispatch carry the same constant
    // cause, so waiting for that invocation is neither necessary nor safe:
    // bounded spinning used to turn ordinary incidence contention into a
    // physics-stopping fault. A later producer/cause is dispatch-ordered after
    // publication and takes the ordinary observed==generation path above.
    if(observed==(generation|0x80000000u)){
      ${hookRoot}
      return true;
    }
    if((observed&0x80000000u)==0u){
      let claim=atomicCompareExchangeWeak(&${arena}[cm12ExtensionRootStamp+cell],
        observed,generation|0x80000000u);
      if(claim.exchanged){
        cm12ExtensionStore(cm12ExtensionRootCause+cell,cause);
        let slot=atomicAdd(&${arena}[${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)}],1u);
        if(slot>=cm12ExtensionCapacity){cm12ExtensionFail(cell,0u);return false;}
        cm12ExtensionStore(cm12ExtensionRootList+slot,cell);
        cm12ExtensionStore(cm12ExtensionRootStamp+cell,generation);
        ${hookRoot}
        return true;
      }
    }
  }
  cm12ExtensionFail(cell,0u);return false;
}

// Topology transfer authors entering template IDs before the accepted-slot
// flip makes cellActive() observe them. Only that producer may use this hook;
// the root is consumed after commit, when the ID is active.
fn cm12ExtensionRecordPendingRoot(cell:u32,cause:u32)->bool{
  if(cell>=cm12ExtensionCapacity){cm12ExtensionFail(cell,0u);return false;}
  let generation=cm12ExtensionGeneration();
  if((cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)})
    &cm12ExtensionFlagCollecting)==0u){cm12ExtensionFail(cell,0u);return false;}
  for(var spin=0u;spin<64u;spin+=1u){
    let observed=cm12ExtensionLoad(cm12ExtensionRootStamp+cell);
    if(observed==generation){atomicOr(&${arena}[cm12ExtensionRootCause+cell],cause);
      ${hookRoot} return true;}
    if(observed==(generation|0x80000000u)){${hookRoot} return true;}
    if((observed&0x80000000u)==0u){
      let claim=atomicCompareExchangeWeak(&${arena}[cm12ExtensionRootStamp+cell],
        observed,generation|0x80000000u);
      if(claim.exchanged){cm12ExtensionStore(cm12ExtensionRootCause+cell,cause);
        let slot=atomicAdd(&${arena}[${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)}],1u);
        if(slot>=cm12ExtensionCapacity){cm12ExtensionFail(cell,0u);return false;}
        cm12ExtensionStore(cm12ExtensionRootList+slot,cell);
        cm12ExtensionStore(cm12ExtensionRootStamp+cell,generation);
        ${hookRoot} return true;}
    }
  }
  cm12ExtensionFail(cell,0u);return false;
}

// A retired template cell cannot enter the active recurrence, but its cache
// identity must be revoked immediately. The topology producer separately
// records every still-active incident endpoint as an ordinary root.
fn cm12ExtensionInvalidateRetiredCell(cell:u32,cause:u32)->bool{
  if(cell>=cm12ExtensionCapacity
    ||(cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)})
      &cm12ExtensionFlagCollecting)==0u){
    cm12ExtensionFail(cell,0u);return false;
  }
  let generation=cm12ExtensionGeneration();
  cm12ExtensionStore(cm12ExtensionAcceptedOwner+cell,cm12ExtensionInvalid);
  cm12ExtensionStore(cm12ExtensionAcceptedDepth+cell,cm12ExtensionInvalid);
  cm12ExtensionStore(cm12ExtensionRootCause+cell,cause);
  ${hookRoot}
  return true;
}

@compute @workgroup_size(1)
fn sealVelocityExtensionRoots(){
  if(cm12ExtensionFaulted()){cm12ExtensionZeroDispatches();return;}
  if((cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)})
    &cm12ExtensionFlagCollecting)==0u){
    cm12ExtensionFail(cm12ExtensionInvalid,0u);return;
  }
  let count=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)});
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchX)},
    (count+63u)/64u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)},cm12ExtensionFlagSealed);
}
fn cm12ExtensionAppendBlast(cell:u32,depth:u32,frontierBase:u32,frontierCountAt:u32)->bool{
  let generation=cm12ExtensionGeneration();
  if(atomicExchange(&${arena}[cm12ExtensionBlastStamp+cell],generation)==generation){return false;}
  cm12ExtensionStore(cm12ExtensionBlastDepth+cell,depth);
  let blastSlot=atomicAdd(&${arena}[${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastCount)}],1u);
  let frontierSlot=atomicAdd(&${arena}[frontierCountAt],1u);
  if(blastSlot>=cm12ExtensionCapacity||frontierSlot>=cm12ExtensionCapacity){
    cm12ExtensionFail(cell,depth);return false;
  }
  cm12ExtensionStore(cm12ExtensionBlastList+blastSlot,cell);
  cm12ExtensionStore(frontierBase+frontierSlot,cell);
  atomicMax(&${arena}[${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.maximumDepth)}],depth);
  ${hookClosure}
  return true;
}

@compute @workgroup_size(64)
fn seedVelocityExtensionRoots(@builtin(global_invocation_id)gid:vec3u){
  if(cm12ExtensionFaulted()){return;}
  let count=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)});
  if(gid.x>=count){return;}
  let cell=cm12ExtensionLoad(cm12ExtensionRootList+gid.x);
  // Root producers run throughout the frame. A cell can retire after it was
  // recorded but before this fixed planning boundary; prune it before it can
  // become immutable next-frame recurrence authority.
  if(!cellActive(cell)){return;}
  _=cm12ExtensionAppendBlast(cell,0u,cm12ExtensionFrontierA,
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount)});
}

@compute @workgroup_size(1)
fn sealVelocityExtensionSeedFrontier(){
  if(cm12ExtensionFaulted()){cm12ExtensionZeroDispatches();return;}
  let count=cm12ExtensionLoad(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount)});
  cm12ExtensionStore(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchX)},
    (count+63u)/64u);
}

// Construction-only bootstrap. The resident dispatches this once, after
// beginVelocityExtensionCandidate, over the immutable accepted-cell worklist.
// It is not a runtime fallback: generation zero has no cache to reuse, so the
// first accepted cache is deliberately authored by the same compact recurrence
// with every accepted cell in its blast radius.
@compute @workgroup_size(64)
fn bootstrapVelocityExtensionRoots(@builtin(global_invocation_id)gid:vec3u){
  if(cm12ExtensionAcceptedGeneration()!=0u){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==cm12ExtensionInvalid){return;}
  // The accepted topology worklist includes dormant receiver capacity. Those
  // cells carry no legacy extrapolation work until their brick is activated.
  if(!cellActive(cell)){return;}
  _=cm12ExtensionRecordRoot(cell,
    ${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.bootstrap}u);
}

// Construction-token QA only: snapshot the retired full eight-sweep result
// into the otherwise unused VEX tail of the immutable legacy oracle. Keep the
// passive clock collecting while the unchanged late physics kernels author
// their ordinary roots; a QA-only singleton seals the receipt after the last
// producer. Ordinary production encoding never dispatches either entry point.
@compute @workgroup_size(64)
fn captureLegacyVelocityExtensionForQA(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==cm12ExtensionInvalid){return;}
  let input=destinationCellVelocity()+4u*cell;
  let output=cm12ExtensionAcceptedVelocity+4u*cell;
  ${state}[output]=${state}[input];${state}[output+1u]=${state}[input+1u];
  ${state}[output+2u]=${state}[input+2u];${state}[output+3u]=${state}[input+3u];
  let valid=${state}[input+3u]>0.5;
  cm12ExtensionStore(cm12ExtensionAcceptedDepth+cell,
    select(cm12ExtensionInvalid,0u,valid));
  cm12ExtensionStore(cm12ExtensionAcceptedOwner+cell,cm12ExtensionOwner(cell));
  // The dispatch boundary orders every cache write before later FCA consumers;
  // lane zero can publish the passive oracle clock without another pipeline.
  if(gid.x==0u&&!cm12ExtensionFaulted()){
    // Construction populated the root list for production's generation-one
    // plan. The passive legacy oracle never consumed that queue, so carrying
    // its count into generation two would append behind a full stale list and
    // manufacture a capacity fault. Reset only compact header receipts here;
    // root/frontier/blast stamps remain generation-keyed and need no scan.
    cm12ExtensionZeroDispatches();
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.faultCount)},0u);
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultCell)},
      cm12ExtensionInvalid);
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultDepth)},
      cm12ExtensionInvalid);
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.maximumDepth)},0u);
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.executedCellCount)},0u);
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reusedCellCount)},0u);
    cm12ExtensionStore(
      ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.uncoveredWriteCount)},0u);
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.acceptedGeneration)},
      ${sourceFrameGeneration});
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.candidateGeneration)},
      ${nextFrameGeneration});
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.topologyGeneration)},
      ${topologyGeneration});
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)},
      cm12ExtensionFlagCollecting);
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
      cm12ExtensionPhaseCollecting);
  }
}

// Immutable-oracle clock publication after every physical/root producer. The
// oracle never consumes these roots or indirects; clearing them is lifecycle
// bookkeeping only and cannot become physical authority or a runtime fallback.
@compute @workgroup_size(1)
fn finalizeLegacyVelocityExtensionClockForQA(){
  if(!cm12ExtensionHeaderValid()||cm12ExtensionFaulted()){return;}
  let source=${sourceFrameGeneration};let next=${nextFrameGeneration};
  if(source==0u||next<=source||next>=0x7ffffffeu){
    cm12ExtensionFail(cm12ExtensionInvalid,0u);return;
  }
  cm12ExtensionZeroDispatches();
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.acceptedGeneration)},source);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.candidateGeneration)},next);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)},
    cm12ExtensionFlagSealed);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
    cm12ExtensionPhasePlanned);
}

@compute @workgroup_size(1)
fn prepareVelocityExtensionFrontier(){
  if(cm12ExtensionFaulted()){cm12ExtensionZeroDispatches();return;}
  if(EXTENSION_FRONTIER_DEPTH<1u
    ||EXTENSION_FRONTIER_DEPTH>${SPARSE_CM12_VELOCITY_EXTENSION_DEPTH}u){
    cm12ExtensionFail(cm12ExtensionInvalid,EXTENSION_FRONTIER_DEPTH);return;
  }
  let outputCountAt=select(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount)},
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBCount)},
    (EXTENSION_FRONTIER_DEPTH&1u)==1u);
  cm12ExtensionStore(outputCountAt,0u);cm12ExtensionStore(outputCountAt+1u,0u);
  cm12ExtensionStore(outputCountAt+2u,1u);cm12ExtensionStore(outputCountAt+3u,1u);
}

@compute @workgroup_size(64)
fn expandVelocityExtensionFrontier(@builtin(global_invocation_id)gid:vec3u){
  if(cm12ExtensionFaulted()){return;}
  let odd=(EXTENSION_FRONTIER_DEPTH&1u)==1u;
  let inputCountAt=select(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBCount)},
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount)},odd);
  let outputCountAt=select(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount)},
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBCount)},odd);
  let inputBase=select(cm12ExtensionFrontierB,cm12ExtensionFrontierA,odd);
  let outputBase=select(cm12ExtensionFrontierA,cm12ExtensionFrontierB,odd);
  let count=cm12ExtensionLoad(inputCountAt);if(gid.x>=count){return;}
  let cell=cm12ExtensionLoad(inputBase+gid.x);
  let edgeOffsets=pressureTemplateWord(15u);
  let end=pressureTemplateWord(edgeOffsets+cell+1u);
  for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
    let row=pressureTemplateWord(pressureEdgeRows()+edge);if(!rowAccepted(row)){continue;}
    let neighbor=fineSamples[pressureNeighborOffset()+edge];
    if(neighbor>=cm12ExtensionCapacity||!cellActive(neighbor)){continue;}
    _=cm12ExtensionAppendBlast(neighbor,EXTENSION_FRONTIER_DEPTH,outputBase,outputCountAt);
  }
}

@compute @workgroup_size(1)
fn sealVelocityExtensionFrontier(){
  if(cm12ExtensionFaulted()){cm12ExtensionZeroDispatches();return;}
  let outputCountAt=select(
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierACount)},
    ${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBCount)},
    (EXTENSION_FRONTIER_DEPTH&1u)==1u);
  let count=cm12ExtensionLoad(outputCountAt);
  cm12ExtensionStore(outputCountAt+1u,(count+63u)/64u);
}

@compute @workgroup_size(1)
fn finalizeVelocityExtensionBlast(){
  let flags=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)});
  if((flags&cm12ExtensionFlagFault)!=0u){cm12ExtensionZeroDispatches();return;}
  let count=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastCount)});
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchX)},
    (count+63u)/64u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)},
    cm12ExtensionFlagSealed);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
    cm12ExtensionPhasePlanned);
}
fn cm12ExtensionBlastInvocation(invocation:u32)->u32{
  let count=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastCount)});
  return select(cm12ExtensionInvalid,
    cm12ExtensionLoad(cm12ExtensionBlastList+invocation),invocation<count);
}
fn cm12ExtensionCachedVelocity(cell:u32,depth:u32)->vec4f{
  if(cell>=cm12ExtensionCapacity
    ||cm12ExtensionLoad(cm12ExtensionAcceptedOwner+cell)!=cm12ExtensionOwner(cell)){
    if(cell<cm12ExtensionCapacity){cm12ExtensionStore(cm12ExtensionCandidateDepth+cell,2u);}
    cm12ExtensionFail(cell,depth);return vec4f(0.0);
  }
  let acceptedDepth=cm12ExtensionLoad(cm12ExtensionAcceptedDepth+cell);
  if(acceptedDepth>=depth){return vec4f(0.0);}
  let at=cm12ExtensionAcceptedVelocity+4u*cell;
  let generation=cm12ExtensionReceiptGeneration();
  if(atomicExchange(&${arena}[cm12ExtensionReuseStamp+cell],generation)!=generation){
    atomicAdd(&${arena}[${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reusedCellCount)}],1u);
  }
  return vec4f(${state}[at],${state}[at+1u],${state}[at+2u],1.0);
}

@compute @workgroup_size(1)
fn beginVelocityExtensionExecution(){
  if(!cm12ExtensionHeaderValid()||cm12ExtensionFaulted()){return;}
  let flags=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)});
  if((flags&cm12ExtensionFlagSealed)==0u
    ||cm12ExtensionGeneration()!=${sourceFrameGeneration}){
    cm12ExtensionFail(cm12ExtensionInvalid,0u);return;
  }
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.executedCellCount)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reusedCellCount)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
    cm12ExtensionPhaseConsuming);
}

@compute @workgroup_size(64)
fn initializeVelocityExtensionCandidates(@builtin(global_invocation_id)gid:vec3u){
  if(cm12ExtensionFaulted()){return;}
  let cell=cm12ExtensionBlastInvocation(gid.x);if(cell==cm12ExtensionInvalid){return;}
  ${hookScheduled}
  let output=destinationCellVelocity()+4u*cell;
  let input=sourceCellVelocity()+4u*cell;
  let liquid=state[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE;
  ${state}[output]=select(0.0,${state}[input],liquid);
  ${state}[output+1u]=select(0.0,${state}[input+1u],liquid);
  ${state}[output+2u]=select(0.0,${state}[input+2u],liquid);
  ${state}[output+3u]=select(0.0,1.0,liquid);
  cm12ExtensionStore(cm12ExtensionCandidateDepth+cell,
    select(cm12ExtensionInvalid,0u,liquid));
}

@compute @workgroup_size(64)
fn advanceVelocityExtensionCandidates(@builtin(global_invocation_id)gid:vec3u){
  if(cm12ExtensionFaulted()){return;}
  let cell=cm12ExtensionBlastInvocation(gid.x);if(cell==cm12ExtensionInvalid){return;}
  if(EXTENSION_RECURRENCE_DEPTH<1u
    ||EXTENSION_RECURRENCE_DEPTH>${SPARSE_CM12_VELOCITY_EXTENSION_DEPTH}u){
    cm12ExtensionFail(cell,EXTENSION_RECURRENCE_DEPTH);return;
  }
  let odd=(EXTENSION_RECURRENCE_DEPTH&1u)==1u;
  let inputBase=select(sourceCellVelocity(),destinationCellVelocity(),odd);
  let outputBase=select(destinationCellVelocity(),sourceCellVelocity(),odd);
  let input=inputBase+4u*cell;let output=outputBase+4u*cell;
  if(${state}[input+3u]>0.5){
    ${state}[output]=${state}[input];${state}[output+1u]=${state}[input+1u];
    ${state}[output+2u]=${state}[input+2u];${state}[output+3u]=1.0;return;
  }
  var velocity=vec3f(0.0);var weight=0.0;
  let edgeOffsets=pressureTemplateWord(15u);
  let end=pressureTemplateWord(edgeOffsets+cell+1u);
  let generation=cm12ExtensionGeneration();
  for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
    let row=pressureTemplateWord(pressureEdgeRows()+edge);if(!rowAccepted(row)){continue;}
    let neighbor=fineSamples[pressureNeighborOffset()+edge];if(!cellActive(neighbor)){continue;}
    var candidate=vec4f(0.0);
    if(cm12ExtensionLoad(cm12ExtensionBlastStamp+neighbor)==generation){
      let at=inputBase+4u*neighbor;
      candidate=vec4f(${state}[at],${state}[at+1u],${state}[at+2u],${state}[at+3u]);
    }else if(${state}[sourceDensity()+neighbor]>CM12_LIQUID_ISOVALUE){
      let at=sourceCellVelocity()+4u*neighbor;
      candidate=vec4f(${state}[at],${state}[at+1u],${state}[at+2u],1.0);
    }else{candidate=cm12ExtensionCachedVelocity(neighbor,EXTENSION_RECURRENCE_DEPTH);}
    if(candidate.w<=0.5){continue;}
    let w=bitcast<f32>(fineSamples[pressureExtrapolationWeightOffset()+edge]);
    velocity+=w*candidate.xyz;weight+=w;
  }
  if(weight>0.0){velocity/=weight;}
  ${state}[output]=velocity.x;${state}[output+1u]=velocity.y;${state}[output+2u]=velocity.z;
  ${state}[output+3u]=select(0.0,1.0,weight>0.0);
  if(weight>0.0){cm12ExtensionStore(cm12ExtensionCandidateDepth+cell,
    EXTENSION_RECURRENCE_DEPTH);}
}

@compute @workgroup_size(64)
fn commitVelocityExtensionCandidates(@builtin(global_invocation_id)gid:vec3u){
  // Accepted cache/depth/owner are the transaction boundary. A fault in any
  // prior root, closure, scheduling, or recurrence invocation leaves all three
  // at the preceding accepted generation.
  if(cm12ExtensionFaulted()){return;}
  let cell=cm12ExtensionBlastInvocation(gid.x);if(cell==cm12ExtensionInvalid){return;}
  let input=destinationCellVelocity()+4u*cell;
  let output=cm12ExtensionAcceptedVelocity+4u*cell;
  let valid=${state}[input+3u]>0.5;
  ${state}[output]=select(0.0,${state}[input],valid);
  ${state}[output+1u]=select(0.0,${state}[input+1u],valid);
  ${state}[output+2u]=select(0.0,${state}[input+2u],valid);
  ${state}[output+3u]=select(0.0,1.0,valid);
  cm12ExtensionStore(cm12ExtensionAcceptedDepth+cell,
    select(cm12ExtensionInvalid,cm12ExtensionLoad(cm12ExtensionCandidateDepth+cell),valid));
  cm12ExtensionStore(cm12ExtensionAcceptedOwner+cell,cm12ExtensionOwner(cell));
  ${publishEffectiveVelocity}
  atomicAdd(&${arena}[${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.executedCellCount)}],1u);
}

@compute @workgroup_size(1)
fn finalizeVelocityExtensionCandidate(){
  let flags=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)});
  if((flags&cm12ExtensionFlagFault)!=0u){return;}
  if((flags&cm12ExtensionFlagSealed)==0u){cm12ExtensionFail(cm12ExtensionInvalid,0u);return;}
  let nextGeneration=${nextFrameGeneration};
  if(nextGeneration<=cm12ExtensionGeneration()||nextGeneration>=0x7ffffffeu){
    cm12ExtensionFail(cm12ExtensionInvalid,0u);return;
  }
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.acceptedGeneration)},
    cm12ExtensionGeneration());
  // The recurrence list is dead after commit. Reuse it as the next frame's
  // producer-owned root queue without clearing the accepted blast receipt.
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.candidateGeneration)},
    nextGeneration);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchX)},0u);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)},
    cm12ExtensionFlagCollecting);
  cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
    cm12ExtensionPhaseCollecting);
}

fn cm12ExtensionReadyForFrameCommit()->bool{
  let flags=cm12ExtensionLoad(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)});
  return (flags&cm12ExtensionFlagFault)==0u
    &&(flags&cm12ExtensionFlagSealed)!=0u
    &&cm12ExtensionAcceptedGeneration()==${sourceFrameGeneration}
    &&cm12ExtensionGeneration()==${nextFrameGeneration};
}

fn cm12ExtensionTransportVelocity(cell:u32)->vec4f{
  // Immutable construction QA solvers do not execute VEX1. Generation zero
  // therefore names the already-computed legacy destination bank, not a
  // runtime recovery path from a production fault.
  if(cm12ExtensionAcceptedGeneration()==0u){
    let at=destinationCellVelocity()+4u*cell;
    return vec4f(${state}[at],${state}[at+1u],${state}[at+2u],${state}[at+3u]);
  }
  // A production fault is consumed only by the frame-commit gate. Returning
  // invalid data here prevents a stale accepted cache from becoming hidden
  // physical authority while the remaining fixed dispatch chain drains.
  if(cm12ExtensionFaulted()){return vec4f(0.0);}
  if(${state}[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE){
    let at=sourceCellVelocity()+4u*cell;
    return vec4f(${state}[at],${state}[at+1u],${state}[at+2u],1.0);
  }
  let generation=cm12ExtensionAcceptedGeneration();
  if(cm12ExtensionLoad(cm12ExtensionBlastStamp+cell)==generation){
    let at=destinationCellVelocity()+4u*cell;
    return vec4f(${state}[at],${state}[at+1u],${state}[at+2u],${state}[at+3u]);
  }
  return cm12ExtensionCachedVelocity(cell,${SPARSE_CM12_VELOCITY_EXTENSION_DEPTH + 1}u);
}
`;
}
