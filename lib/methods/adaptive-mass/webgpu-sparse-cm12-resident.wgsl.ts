import { createCm12NumericsWGSL } from "../../core/cm12-numerics";
import {
  SPARSE_CM12_DIRTY_CAUSE_BIT,
} from "../../core/sparse-cm12-dirty-visualizations";
import type { SparseCM12IncrementalActivityLayout } from
  "./sparse-cm12-incremental-activity";
import { createSparseCM12IncrementalActivityWGSL } from
  "./sparse-cm12-incremental-activity.wgsl";
import type { SparseCM12CanonicalMembershipLayout } from
  "./sparse-cm12-canonical-membership";
import { createSparseCM12CanonicalMembershipWGSL } from
  "./sparse-cm12-canonical-membership.wgsl";
import { SPARSE_CM12_PRESSURE_REPAIR_HEADER } from
  "./sparse-cm12-pressure-membership";
import { createSparseCM12FramePlanWGSL } from "../../core/sparse-cm12-frame-plan.wgsl";
import type { SparseCM12FramePlanLayout } from "../../core/sparse-cm12-frame-plan";
import { createSparseCM12FramePlanPresentationWGSL } from
  "./sparse-cm12-frame-plan-presentation.wgsl";
import type { SparseCM12FramePlanPresentationLayout } from
  "./sparse-cm12-frame-plan-presentation";
import {
  SPARSE_CM12_FRAME_CONTROL_COVERAGE,
  type SparseCM12FrameControlLayout,
} from "./sparse-cm12-frame-control";
import { createSparseCM12FrameControlWGSL } from "./sparse-cm12-frame-control.wgsl";
import type { SparseCM12PressureTopologyRepairLayout } from
  "./sparse-cm12-pressure-topology-repair";
import { createSparseCM12PressureTopologyRepairWGSL } from
  "./sparse-cm12-pressure-topology-repair.wgsl";
import { SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE } from
  "./sparse-cm12-pressure-topology-repair";
import type { SparseCM12PersistentPressureCacheLayout } from
  "./sparse-cm12-persistent-pressure-cache";
import { createSparseCM12PersistentPressureCacheWGSL } from
  "./sparse-cm12-persistent-pressure-cache.wgsl";
import type {
  SparseCM12VelocityExtensionLayout,
  SparseCM12VelocityExtensionStateLayout,
} from "./sparse-cm12-velocity-extension";
import { createSparseCM12VelocityExtensionWGSL } from
  "./sparse-cm12-velocity-extension.wgsl";
import type { SparseCM12DirtyFaceRowMaskLayout } from
  "./sparse-cm12-dirty-face-row-masks";
import { createSparseCM12DirtyFaceRowMaskWGSL } from
  "./sparse-cm12-dirty-face-row-masks.wgsl";
import type { SparseCM12PressureAddressingABLayout,
  SparseCM12PressureAddressingABModeName } from
  "./sparse-cm12-pressure-addressing-ab";
import { createSparseCM12PressureAddressingABWGSL } from
  "./sparse-cm12-pressure-addressing-ab.wgsl";
import type { SparseCM12LogicalOwnerDirectoryLayout } from
  "./sparse-cm12-logical-owner-directory";
import { createSparseCM12LogicalOwnerDirectoryWGSL } from
  "./sparse-cm12-logical-owner-directory.wgsl";
import type { SparseCM12TransportExecutionImageLayout } from
  "./sparse-cm12-transport-execution-image";
import { createSparseCM12TransportExecutionImageWGSL } from
  "./sparse-cm12-transport-execution-image.wgsl";
import type { SparseCM12EffectiveTransportVelocityLayout } from
  "./sparse-cm12-effective-transport-velocity";
import { createSparseCM12EffectiveTransportVelocityWGSL } from
  "./sparse-cm12-effective-transport-velocity.wgsl";
import type { SparseCM12TransportPacketAuthorityLayout } from
  "./sparse-cm12-transport-packet-authority";
import { createSparseCM12TransportPacketAuthorityWGSL } from
  "./sparse-cm12-transport-packet-authority.wgsl";
import type { SparseCM12TransportProducerMaskLayout } from
  "./sparse-cm12-transport-producer-masks";
import { createSparseCM12TransportProducerMaskWGSL } from
  "./sparse-cm12-transport-producer-masks.wgsl";
import type { SparseCM12FinalScalarPacketMaskLayout } from
  "./sparse-cm12-final-scalar-packet-masks";
import { createSparseCM12FinalScalarPacketMaskWGSL } from
  "./sparse-cm12-final-scalar-packet-masks.wgsl";
import type { SparseCM12Phase1TransportQALayout } from
  "./sparse-cm12-phase1-transport-receipt";
import { createSparseCM12Phase1TransportQAWGSL } from
  "./sparse-cm12-phase1-transport-receipt.wgsl";
import type { SparseCM12InternedBoundaryLayout } from
  "./sparse-cm12-interned-boundary-operators";
import { createSparseCM12InternedBoundaryImageWGSL } from
  "./sparse-cm12-interned-boundary-image.wgsl";
import type { SparseCM12InternedRefLookupLayout } from
  "./sparse-cm12-interned-ref-lookup";
import { createSparseCM12InternedRefLookupWGSL } from
  "./sparse-cm12-interned-ref-lookup.wgsl";
import type { SparseCM12IboTRASupplementLayout } from
  "./sparse-cm12-ibo-tra-supplement";
import { createSparseCM12IboTRASupplementWGSL } from
  "./sparse-cm12-ibo-tra-supplement.wgsl";
import { createSparseCM12IboTRAResidentHooksWGSL } from
  "./sparse-cm12-ibo-tra-resident-hooks.wgsl";
import { createSparseCM12GeometryFaceNeighborsWGSL,
  createSparseCM12IBOSemanticAuthorityWGSL } from
  "./sparse-cm12-ibo-semantic-authority.wgsl";
import { SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER,
  SPARSE_CM12_TOPOLOGY_EFFECTS_PHASE,
  type SparseCM12TopologyEffectsAuthorityLayout } from
  "./sparse-cm12-topology-effects-authority";
import { createSparseCM12TopologyEffectsAuthorityWGSL } from
  "./sparse-cm12-topology-effects-authority.wgsl";

/**
 * Compact, GPU-resident Sparse CM12 frame kernel.
 *
 * Topology is packed once at construction.  Every per-frame field, reduction,
 * pressure iteration, projection, diagnostic field, and dense presentation
 * publication remains in device storage.  The host only writes the small
 * timestep uniform and submits the pre-sized dispatch sequence.
 */
export type SparseCM12BrickFineResolution = 4 | 8 | 16;

/** Construction-time specialization keeps brick geometry in WGSL constants,
 * so the hot lookup/transfer paths pay no runtime ladder branch. */
import {
  createSparseCM12CellAccessWGSL,
  createSparseCM12RowAccessWGSL,
  SPARSE_CM12_ATOMIC_ARENA_READERS,
} from "./sparse-cm12-row-access.wgsl";

/** Persistent dense pressure membership plus bounded local change journals. */
export interface SparseCM12PressureRepairLayout {
  readonly aggregateEdgeForFineEdgeBaseWords: number;
  readonly aggregateEdgeSourceBaseWords: number;
  readonly hierarchyEdgeForAggregateBaseWords: readonly number[];
  readonly headerBaseWords: number;
  readonly totalWords: number;
}

export function createWebgpuSparseCM12ResidentWGSL(
  brickFineResolution: SparseCM12BrickFineResolution = 8,
  presentationPageResolution: SparseCM12BrickFineResolution = brickFineResolution,
  pressureRepairLayout?: SparseCM12PressureRepairLayout,
  incrementalActivityLayout?: SparseCM12IncrementalActivityLayout,
  canonicalMembershipLayout?: SparseCM12CanonicalMembershipLayout,
  framePlanLayout?: SparseCM12FramePlanLayout,
  framePlanPresentationLayout?: SparseCM12FramePlanPresentationLayout,
  frameControlLayout?: SparseCM12FrameControlLayout,
  pressureTopologyRepairLayout?: SparseCM12PressureTopologyRepairLayout,
  persistentPressureCacheLayout?: SparseCM12PersistentPressureCacheLayout,
  velocityExtensionLayouts?: Readonly<{
    activity: SparseCM12VelocityExtensionLayout;
    state: SparseCM12VelocityExtensionStateLayout;
  }>,
  pressureAddressingABLayout?: SparseCM12PressureAddressingABLayout,
  pressureAddressingFixedMode?: SparseCM12PressureAddressingABModeName,
  pressureExecutionQAOracle = false,
  logicalOwnerDirectory?: {
    readonly layout: SparseCM12LogicalOwnerDirectoryLayout;
    readonly baseWords: number;
    readonly packedOwner16BaseWords?: number;
  },
  faceVelocitySupportBaseFloats = 0,
  effectiveTransportVelocityLayout?: SparseCM12EffectiveTransportVelocityLayout,
  transportExecutionImageLayout?: SparseCM12TransportExecutionImageLayout,
  transportPacketAuthorityLayout?: SparseCM12TransportPacketAuthorityLayout,
  transportProducerMaskLayout?: SparseCM12TransportProducerMaskLayout,
  phase1TransportQALayout?: SparseCM12Phase1TransportQALayout,
  phase1TransportProfileBaseWords?: number,
  internedBoundaryImage?: Readonly<{
    layout: SparseCM12InternedBoundaryLayout;
    refLookupLayout: SparseCM12InternedRefLookupLayout;
    traSupplementLayout: SparseCM12IboTRASupplementLayout;
    baseWords: number;
    semanticAuthority: Readonly<{
      geometryBaseWords: number;
      geometryOffsetBaseWords: number;
      geometryNeighborBaseWords: number;
      authorityBaseWords: number;
      leafCapacity: number;
      immutableContentHash: number;
      immutableCertificateHash: number;
    }>;
  }>,
  topologyEffectsAuthorityLayout?: SparseCM12TopologyEffectsAuthorityLayout,
  finalScalarPacketMaskLayout?: SparseCM12FinalScalarPacketMaskLayout,
  dirtyFaceRowMaskLayout?: SparseCM12DirtyFaceRowMaskLayout,
): string {
  if (presentationPageResolution > brickFineResolution
    || brickFineResolution % presentationPageResolution !== 0) {
    throw new RangeError(`presentation page ${presentationPageResolution} does not divide brick ladder ${brickFineResolution}`);
  }
  if (!internedBoundaryImage || !dirtyFaceRowMaskLayout) {
    throw new Error("Sparse CM12 production WGSL requires compiled ITR and DFRM layouts");
  }
  if (internedBoundaryImage
    && (!transportPacketAuthorityLayout || !transportProducerMaskLayout)) {
    throw new Error("ITR1 production composition requires TPA1 and TPM1");
  }
  const templateLevelCount = Math.log2(brickFineResolution) + 1;
  const candidateCellCount = brickFineResolution ** 3;
  const candidateFaceSampleCount = brickFineResolution ** 2;
  const presentationPagesPerAxis = brickFineResolution / presentationPageResolution;
  const transportCellCapacity = dirtyFaceRowMaskLayout?.cellCapacity ?? 1;
  // Span-one pages need at most (P/2 + 2)^3 samples: scale one addresses the
  // authority directly, so scale two is the finest cached reconstruction.
  // A finer macro can exceed this bounded cache and takes the direct fallback.
  const presentationCacheCapacity = (presentationPageResolution / 2 + 2) ** 3;
  const incrementalActivityEntries = incrementalActivityLayout
    ? createSparseCM12IncrementalActivityWGSL(incrementalActivityLayout,
      brickFineResolution / 4)
    : /* wgsl */ `
fn incrementalActivityGeneration()->u32{return 0u;}
fn incrementalActivityMarkCellClosure(cell:u32,cause:u32){_=cell;_=cause;}
fn incrementalActivityBrickInvocation(invocation:u32)->u32{
  return select(INVALID,invocation,invocation<p.dispatch.w);
}
fn incrementalActivityBrickDirty(brick:u32)->bool{return brick<p.dispatch.w;}
fn incrementalActivityBrickVelocityDirty(brick:u32)->bool{return brick<p.dispatch.w;}
fn incrementalActivityAcceptMeasuredTopology(brick:u32){_=brick;}
fn incrementalActivityRemoveCensus(brick:u32){_=brick;}
fn incrementalActivityAddCensus(brick:u32,score:u32,reasons:u32){
  _=brick;atomicMax(&activity[1],score);
  if((reasons&1u)!=0u){atomicAdd(&activity[2],1u);}
  let value=f32(score)/255.0;
  if(value>=p.activityTiming.y){atomicAdd(&activity[3],1u);}
  if(value<=p.activityTiming.w){atomicAdd(&activity[4],1u);}
  atomicAdd(&activity[6],1u);
}
`;
  const framePlanEntries = framePlanLayout
    ? createSparseCM12FramePlanWGSL({ layout: framePlanLayout, arenaName: "activity" })
    : "";
  const framePlanPresentationEntries = framePlanLayout && framePlanPresentationLayout
    ? createSparseCM12FramePlanPresentationWGSL({
      layout: framePlanPresentationLayout,
      framePlanPrefix: "cm12FramePlan",
      packetArenaName: "activity",
      hookPrefix: "cm12Presentation",
    })
    : "";
  const velocityExtensionEntries = velocityExtensionLayouts
    ? createSparseCM12VelocityExtensionWGSL({
      layout: velocityExtensionLayouts.activity,
      arenaName: "activity",
      stateName: "state",
      topologyGenerationExpression: "atomicLoad(&topologyArena[topologyWorklistBase()])",
      sourceFrameGenerationExpression: "cm12FCAcceptedGeneration()",
      effectiveVelocityHookPrefix: phase1TransportQALayout ? "cm12Phase1QA"
        : effectiveTransportVelocityLayout ? "cm12" : undefined,
    })
    : /* wgsl */ `
fn cm12ExtendedPacketMask(_packet:u32)->vec2u{return vec2u(0u);}
fn cm12ExtendedPacketLaneSelected(_packet:u32,_lane:u32)->bool{return false;}
`;
  const effectiveTransportVelocityEntries = effectiveTransportVelocityLayout
    ? createSparseCM12EffectiveTransportVelocityWGSL({
      layout: effectiveTransportVelocityLayout,
      planeName: "partials",
    })
    : /* wgsl */ `
fn cm12EffectiveTransportVelocity(cell:u32)->vec4f{
  _=cell;return vec4f(0.0);
}
fn cm12PublishCollocatedWetEffectiveVelocity(cell:u32,velocity:vec3f,wet:bool){
  _=cell;_=velocity;_=wet;
}
fn cm12PublishTransferredEffectiveVelocity(cell:u32,velocity:vec3f){
  _=cell;_=velocity;
}`;
  const phase1TransportQAEntries = phase1TransportQALayout
    ? createSparseCM12Phase1TransportQAWGSL({ layout: phase1TransportQALayout,
      publishEffectiveVelocity: effectiveTransportVelocityLayout !== undefined,
      validateExecutionImage: transportExecutionImageLayout !== undefined })
    : "";
  const phase1QATraceCapture = phase1TransportQALayout
    ? "cm12Phase1QACaptureTrace(id,departure,stencil);" : "";
  const phase1QABetaCapture = phase1TransportQALayout
    ? "if(donor!=INVALID){cm12Phase1QACaptureBeta(donor,atomicLoad(&conditioning[donor]));}"
    : "";
  const phase1QADeficitCapture = phase1TransportQALayout
    ? `if(id!=INVALID){cm12Phase1QACaptureDeficit(id,
    atomicLoad(&conditioning[p.counts.x+id]),
    atomicLoad(&conditioning[2u*p.counts.x+id]));}` : "";
  const phase1QAMassCapture = phase1TransportQALayout
    ? `if(id!=INVALID){cm12Phase1QACaptureMass(id,
    state[destinationDensity()+id],state[destinationGamma()+id]);}` : "";
  const phase1QATransferredVelocityPublish = phase1TransportQALayout
    ? "cm12Phase1QAPublishTransferredEffectiveVelocity(cell,"
      + "vec3f(state[at],state[at+1u],state[at+2u]));"
    : "cm12PublishTransferredEffectiveVelocity(cell,"
      + "vec3f(state[at],state[at+1u],state[at+2u]));";
  const canonicalMembershipEntries = canonicalMembershipLayout
    ? createSparseCM12CanonicalMembershipWGSL({
      layout: canonicalMembershipLayout,
      arenaName: "activity",
      workgroupSize: 64,
    })
    : /* wgsl */ `
fn pcmCellBegin(expected:u32)->bool{_=expected;return true;}
fn pcmRowBegin(expected:u32)->bool{_=expected;return true;}
fn pcmCellSetCandidate(id:u32,enabled:bool,cause:u32,closure:bool)->bool{
  _=id;_=enabled;_=cause;_=closure;return true;
}
fn pcmRowSetCandidate(id:u32,enabled:bool,cause:u32,closure:bool)->bool{
  _=id;_=enabled;_=cause;_=closure;return true;
}
fn pcmCellFinalizeFrontier()->bool{return true;}
fn pcmRowFinalizeFrontier()->bool{return true;}
fn pcmCellFinalize()->bool{return true;}
fn pcmRowFinalize()->bool{return true;}
fn pcmCellRankSelect(_rank:u32)->u32{return INVALID;}
fn pcmRowRankSelect(_rank:u32)->u32{return INVALID;}
fn pcmCellContains(_id:u32)->bool{return false;}
fn pcmRowContains(_id:u32)->bool{return false;}
fn pcmCellAcceptedCount()->u32{return 0u;}
fn pcmRowAcceptedCount()->u32{return 0u;}
fn pcmCellBootstrapEpoch()->bool{return true;}
fn pcmRowBootstrapEpoch()->bool{return true;}
fn pcmCellCandidateGeneration()->u32{return 1u;}
fn pcmRowCandidateGeneration()->u32{return 1u;}
fn pcmCellFault()->u32{return 0u;}
fn pcmRowFault()->u32{return 0u;}
fn pcmCellFirstFault()->u32{return INVALID;}
fn pcmRowFirstFault()->u32{return INVALID;}
fn pcmCellPhase()->u32{return 1u;}
fn pcmRowPhase()->u32{return 1u;}
fn pcmCellDirtyCount()->u32{return 0u;}
fn pcmRowDirtyCount()->u32{return 0u;}
fn pcmCellAcceptedGeneration()->u32{return 0u;}
fn pcmRowAcceptedGeneration()->u32{return 0u;}
`;
  const pressureRepairEntries = pressureRepairLayout ? /* wgsl */ `
const PRESSURE_REPAIR_HEADER:u32=${pressureRepairLayout.headerBaseWords}u;
const PRESSURE_REPAIR_CELL_FIRST_FAULT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.cellFirstFault}u;
const PRESSURE_REPAIR_ROW_FIRST_FAULT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.rowFirstFault}u;
const PRESSURE_REPAIR_FAULT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.fault}u;
const PRESSURE_BOOTSTRAP_CELL_INDIRECT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.bootstrapCellIndirect}u;
const PRESSURE_BOOTSTRAP_ROW_INDIRECT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.bootstrapRowIndirect}u;
` : /* wgsl */ `
const PRESSURE_REPAIR_HEADER:u32=0u;
const PRESSURE_REPAIR_CELL_FIRST_FAULT:u32=0u;
const PRESSURE_REPAIR_ROW_FIRST_FAULT:u32=1u;
const PRESSURE_REPAIR_FAULT:u32=2u;
const PRESSURE_BOOTSTRAP_CELL_INDIRECT:u32=3u;
const PRESSURE_BOOTSTRAP_ROW_INDIRECT:u32=6u;
`;
  const frameControlEntries = frameControlLayout
    ? createSparseCM12FrameControlWGSL({
      layout: frameControlLayout, controlName: "topologyArena",
      authorizedD4Invalidation: true,
    })
    : /* wgsl */ `
fn cm12FCSourceScalarParity()->u32{return select(0u,1u,p.frame.w>0.5);}
fn cm12FCDestinationScalarParity()->u32{return cm12FCSourceScalarParity()^1u;}
fn cm12FCSourceFaceParity()->u32{return cm12FCSourceScalarParity();}
fn cm12FCDestinationFaceParity()->u32{return cm12FCSourceFaceParity()^1u;}
fn cm12FCCandidateGeneration()->u32{return 1u;}
fn cm12FCAcceptedGeneration()->u32{return 1u;}
fn cm12FCFrameSealed()->bool{return true;}
fn cm12FCEffectsPhaseValid()->bool{return true;}
fn cm12FCEffectsGeneration()->u32{return cm12FCAcceptedGeneration();}
fn cm12FCBegin()->bool{return true;}
fn cm12FCPublishBody(count:u32)->bool{_=count;return true;}
fn cm12FCPublishBoundary(live:bool)->bool{_=live;return true;}
fn cm12FCSeal()->bool{return true;}
fn cm12FCPublishOutput(mask:u32)->bool{_=mask;return true;}
fn cm12FCCommit()->bool{return true;}
fn cm12FCInvalidateD4(cause:u32,owner:u32)->bool{_=cause;_=owner;return true;}
fn cm12FCInvalidateD4Authorized(cause:u32,owner:u32,
 generation:u32,sealed:bool){_=cause;_=owner;_=generation;_=sealed;}
`;
  const pressureAddressingABEntries = pressureAddressingABLayout
    ? createSparseCM12PressureAddressingABWGSL({
      layout: pressureAddressingABLayout, arenaName: "activity", workgroupSize: 64,
      fixedMode: pressureAddressingFixedMode,
    })
    : /* wgsl */ `
fn pabPressureCellAddress(rank:u32)->u32{return pcmCellRankSelect(rank);}
`;
  const pressureTopologyRepairEntries = pressureTopologyRepairLayout
    ? createSparseCM12PressureTopologyRepairWGSL({
      layout: pressureTopologyRepairLayout, arenaName: "topologyArena",
      prefix: "ptr", workgroupSize: 64,
      preflightedTopologyPublication: topologyEffectsAuthorityLayout !== undefined,
    }) + /* wgsl */ `
fn ptrResidentTopologyDeltaReady()->bool{return ptrHeaderValid()
  &&atomicLoad(&topologyArena[ptrH_PHASE])==ptrPhaseCollecting
  &&atomicLoad(&topologyArena[ptrH_FAULT])==0u;}
${topologyEffectsAuthorityLayout ? "" :
  "fn ptrSealPreflightedTopologyJournalNoFail(generation:u32){_=generation;}"}
`
    : /* wgsl */ `
const ptrH_COVERED_PRODUCER_RECEIPTS:u32=0u;
fn ptrRecordChangedBrick(brick:u32,oldState:u32,newState:u32,cause:u32,
 receipt:bool)->bool{_=brick;_=oldState;_=newState;_=cause;_=receipt;return true;}
fn ptrSealTopologyJournal(expected:u32,topologyGeneration:u32)->bool{
 _=expected;_=topologyGeneration;return true;}
fn ptrRejectTopologyJournal()->bool{return true;}
fn ptrResidentTopologyDeltaReady()->bool{return true;}
fn ptrSealPreflightedTopologyJournalNoFail(generation:u32){_=generation;}
`;
  const topologyEffectsEntries = topologyEffectsAuthorityLayout
    && pressureTopologyRepairLayout
    ? /* wgsl */ `
fn tfxPTRTargetGeneration()->u32{
 return atomicLoad(&topologyArena[ptrH_CANDIDATE_GENERATION]);}
fn tfxPTRReady(generation:u32,newCount:u32,newLeafCount:u32)->bool{
 return ptrPreflightReady(generation,newCount,newLeafCount);}
fn tfxPTRWillAppend(brick:u32,generation:u32)->bool{
 return ptrPreflightWillAppend(brick,generation);}
fn tfxPTRDirtyLeafWillAppend(leaf:u32,generation:u32)->bool{
 return ptrPreflightDirtyLeafWillAppend(leaf,generation);}
fn tfxPTRCompatible(brick:u32,oldState:u32,newState:u32)->bool{
 return ptrPreflightCompatible(brick,oldState,newState);}
fn tfxPTRPublish(brick:u32,oldState:u32,newState:u32,cause:u32,
 ownsLeaf:bool,generation:u32){
 ptrPublishPreflightedChangedBrick(brick,oldState,newState,cause,ownsLeaf,generation);}
${createSparseCM12TopologyEffectsAuthorityWGSL({
    layout: topologyEffectsAuthorityLayout,
    arenaName: "topologyArena",
    authorizationExpression:
      "atomicLoad(&topologyArena[topologyWorklistBase()+3u])==2u",
  })}
fn residentTopologyEffectsPreflightReady()->bool{
 return tfxPreflightReady(
  atomicLoad(&topologyArena[${topologyEffectsAuthorityLayout.baseWords
    + SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.generation}u]),
  atomicLoad(&topologyArena[${topologyEffectsAuthorityLayout.baseWords
    + SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.ptrTargetGeneration}u]),
  atomicLoad(&topologyArena[${topologyEffectsAuthorityLayout.baseWords
    + SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.ptrCount}u]),
  atomicLoad(&topologyArena[${topologyEffectsAuthorityLayout.baseWords
    + SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.ptrHash}u]));}
fn authorizeEmptySparseCM12CandidateEffectsNoFail(acceptedGeneration:u32)->bool{
 let tfxBase=${topologyEffectsAuthorityLayout.baseWords}u;
 let exactEmpty=atomicLoad(&topologyArena[tfxBase+
  ${SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.phase}u])
    ==${SPARSE_CM12_TOPOLOGY_EFFECTS_PHASE.preflighted}u
  &&atomicLoad(&topologyArena[tfxBase+${SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.fault}u])==0u
  &&atomicLoad(&topologyArena[tfxBase+${SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.ptrCount}u])==0u
  &&atomicLoad(&topologyArena[tfxBase+${SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.ptrLeafCount}u])==0u
  &&atomicLoad(&topologyArena[tfxBase+${SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.expectedEffects}u])==0u
  &&atomicLoad(&topologyArena[tfxBase+${SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER.coveredEffects}u])==0u;
 if(!exactEmpty){return false;}
 tfxAuthorize();_=acceptedGeneration;return true;
}
` : /* wgsl */ `
fn tfxRecordPTRBrick(brick:u32,oldState:u32,newState:u32,cause:u32)->bool{
 _=brick;_=oldState;_=newState;_=cause;return true;}
fn residentTopologyEffectsPreflightReady()->bool{return true;}
fn authorizeEmptySparseCM12CandidateEffectsNoFail(acceptedGeneration:u32)->bool{
 _=acceptedGeneration;return true;}
fn tfxAuthorize(){}
fn tfxPublished()->bool{return true;}
`;
  const persistentPressureCacheEntries = persistentPressureCacheLayout
    ? createSparseCM12PersistentPressureCacheWGSL({
      layout: persistentPressureCacheLayout,
      arenaName: "topologyArena",
      cellContainsFunction: "pcmCellContains",
      rowContainsFunction: "pcmRowContains",
      solidRowScaleFunction: "pcfResidentSolidRowScale",
      workgroupSize: 64,
    })
    : /* wgsl */ `
fn pcfBegin(expectedEvents:u32)->bool{_=expectedEvents;return true;}
fn pcfFinalizeFrontier()->bool{return true;}
fn pcfRecordCellMembershipEvent(cell:u32)->bool{_=cell;return true;}
fn pcfRecordTopologyCellEvent(cell:u32)->bool{_=cell;return true;}
fn pcfStoreThetaAndRecord(row:u32,theta:f32)->bool{_=row;_=theta;return true;}
fn pcfEdgeWeight(edge:u32)->f32{_=edge;return 0.0;}
fn pcfStoreEdgeWeight(edge:u32,value:f32){_=edge;_=value;}
fn pcfDiagonal(cell:u32)->f32{_=cell;return 0.0;}
fn pcfCandidateGeneration()->u32{return 0u;}
fn pcfAcceptedGeneration()->u32{return 0u;}
fn pcfCaptureConsumerGenerations(){}
fn pcfBrickWorkCount()->u32{return 0u;}
fn pcfBrickRankSelect(rank:u32)->u32{_=rank;return INVALID;}
`;
  const logicalOwnerEntries = logicalOwnerDirectory
    ? createSparseCM12LogicalOwnerDirectoryWGSL({
      layout: logicalOwnerDirectory.layout,
      directoryName: "topology",
      baseWords: logicalOwnerDirectory.baseWords,
      trustedHeader: true,
    })
    : "";
  const logicalOwnerLookup = logicalOwnerDirectory
    ? "if(EXP_LOGICAL_OWNER_DIRECTORY){return cm12LogicalOwnerBrickAtKey(key);}" : "";
  const packedLogicalOwnerEntries = logicalOwnerDirectory?.packedOwner16BaseWords === undefined
    ? "" : /* wgsl */ `
const CM12_PACKED_LOGICAL_OWNER16_BASE:u32=${logicalOwnerDirectory.packedOwner16BaseWords}u;
fn cm12PackedLogicalOwner16AtKey(key:u32)->u32{
  if(key>=cm12LogicalOwnerCount){return INVALID;}
  let packed=topology[CM12_PACKED_LOGICAL_OWNER16_BASE+(key>>1u)];
  let owner=(packed>>(16u*(key&1u)))&0xffffu;
  return select(owner,INVALID,owner==0xffffu);
}`;
  const packedLogicalOwnerLookup = logicalOwnerDirectory?.packedOwner16BaseWords === undefined
    ? "" : "if(EXP_PACKED_LOGICAL_OWNER16){return cm12PackedLogicalOwner16AtKey(key);}";
  const transportExecutionImageEntries = transportExecutionImageLayout
    && logicalOwnerDirectory?.packedOwner16BaseWords !== undefined
    ? createSparseCM12TransportExecutionImageWGSL({
      layout: transportExecutionImageLayout,
      packedLogicalOwner16BaseWords: logicalOwnerDirectory.packedOwner16BaseWords,
    }) : /* wgsl */ `
struct CM12TransportLeaf{generation:u32,flags:u32,first:u32,count:u32,
  originKey:u32,valid:vec3u,scale:u32,scaleLog2:u32}
struct CM12TransportOwner{cell:u32,widths:vec3u,volume:u32}
struct CM12TransportPacket{first:u32,counts:vec3u,strideY:u32,strideZ:u32}
struct CM12TransportSpatialTile{packetId:u32,laneMask:vec2u}
fn cm12TeiLogicalCoordinate(key:u32)->vec3u{
  let dimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;let xy=dimensions.x*dimensions.y;
  let z=key/xy;let remainder=key-z*xy;let y=remainder/dimensions.x;
  return vec3u(remainder-y*dimensions.x,y,z);
}
fn cm12TeiLogicalKey(q:vec3u)->u32{
  let dimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  return q.x+dimensions.x*(q.y+dimensions.y*q.z);
}
fn cm12TeiLoadLeaf(_slot:u32,brick:u32)->CM12TransportLeaf{
  if(brick>=p.dispatch.w||!brickActive(brick)){return CM12TransportLeaf(
    0u,0u,0xffffffffu,0u,0xffffffffu,vec3u(0u),0u,0u);}
  let resolution=acceptedBrickResolution(brick);let span=brickSpan(brick);
  let scale=BRICK_FINE_RESOLUTION*span/resolution;
  let key=topology[p.topologyOffsets2.z+2u*brick+1u];
  let origin=cm12TeiLogicalCoordinate(key)*BRICK_FINE_RESOLUTION;
  let extent=min(vec3u(BRICK_FINE_RESOLUTION*span),
    p.dimensions.xyz-min(origin,p.dimensions.xyz));
  let range=templateBrickCellRange(brick,resolution);
  return CM12TransportLeaf(1u,resolution|0x80000000u,range.x,range.y,key,
    (extent+vec3u(scale-1u))/scale,scale,u32(firstLeadingBit(scale)));
}
fn cm12TeiStageDirectory(_origin:vec3u,_lane:u32,_slot:u32){}
fn cm12TeiOwnerAtFine(_q:vec3i)->CM12TransportOwner{
  return CM12TransportOwner(0xffffffffu,vec3u(0u),0u);}
fn cm12TeiPacket(_packet:u32,_slot:u32)->CM12TransportPacket{
  return CM12TransportPacket(0xffffffffu,vec3u(0u),0u,0u);}
fn cm12TeiPacketFineOrigin(_packet:u32,_slot:u32)->vec3u{
  return vec3u(0xffffffffu);}
fn cm12TeiPacketCell(_packet:u32,_lane:u32,_slot:u32)->u32{return 0xffffffffu;}
fn cm12TeiLeafLocalPacketAddress(leaf:u32,resolution:u32,local:vec3u)->vec2u{
  if(resolution==0u){return vec2u(0xffffffffu);}
  let packetAxis=max(1u,(resolution+3u)/4u);let coordinate=local/4u;
  let packet=64u*leaf+coordinate.x+packetAxis*(coordinate.y+packetAxis*coordinate.z);
  let q=local&vec3u(3u);return vec2u(packet,q.x+4u*(q.y+4u*q.z));}
fn cm12TeiSpatialTile(_tile:u32,_slot:u32)->CM12TransportSpatialTile{
  return CM12TransportSpatialTile(0xffffffffu,vec2u(0u));}
fn cm12TeiStagedScaleAtFine(_fine:vec3u)->u32{return 1u;}
@compute @workgroup_size(64)
fn compileSparseCM12TransportExecutionImageShadow(
 @builtin(global_invocation_id)_gid:vec3u){}
@compute @workgroup_size(64)
fn replaySparseCM12TransportExecutionImageRetired(
 @builtin(global_invocation_id)_gid:vec3u){}`;
  const transportPacketAuthorityEntries = transportPacketAuthorityLayout
    ? createSparseCM12TransportPacketAuthorityWGSL({
      layout: transportPacketAuthorityLayout,
      arenaName: "activity",
    })
    : /* wgsl */ `
@compute @workgroup_size(1) fn beginSparseCM12TransportPacketAuthority(){}
@compute @workgroup_size(64) fn clearSparseCM12TransportPacketAuthority(){}
@compute @workgroup_size(64) fn compileSparseCM12TransportPacketsFromFinalScalarMasks(){}
@compute @workgroup_size(1) fn finalizeSparseCM12TransportPacketAuthority(){}
fn cm12TransportPacketCount(family:u32)->u32{_=family;return 0u;}
fn cm12TransportPacketId(rank:u32,family:u32)->u32{
  _=rank;_=family;return INVALID;}
var<workgroup>cm12TransportStagedPacketId:u32;
var<workgroup>cm12TransportStagedPacketOriginFine:vec3u;
var<workgroup>cm12TransportStagedTopologySlot:u32;
fn cm12StageTransportPacket(rank:u32,lane:u32,family:u32){
  _=rank;_=lane;_=family;}
fn cm12TransportStagedExecutionCell(lane:u32)->u32{
  _=lane;return INVALID;}
fn cm12TransportExecutionCell(rank:u32,lane:u32,family:u32)->u32{
  _=rank;_=lane;_=family;return INVALID;}
fn cm12TransportSpatialTileCell(tile:u32,lane:u32)->u32{_=tile;_=lane;return INVALID;}
fn cm12TransportSpatialTileId(fine:vec3u)->u32{_=fine;return INVALID;}`;
  // Production AEI resolves through the staged 27-leaf directory and the
  // dedicated vec4 plane directly. The measured per-site value halo reduced
  // Metal occupancy enough to regress all three transport passes, so it is no
  // longer an experiment or a runtime/build-time choice.
  const transportProducerMaskEntries = transportProducerMaskLayout
    ? createSparseCM12TransportProducerMaskWGSL({
      layout: transportProducerMaskLayout,
      arenaName: "topologyArena",
      hookPrefix: "cm12Resident",
    })
    : /* wgsl */ `
@compute @workgroup_size(1) fn beginSparseCM12TransportProducerMasks(){}
@compute @workgroup_size(1) fn sealSparseCM12TransportProducerMasks(){}
fn cm12TransportProducerMaskPublish(packet:u32,lane:u32,cell:u32,
 surfaceFeature:bool){
  _=packet;_=lane;_=cell;_=surfaceFeature;
 }
fn cm12TransportSharpeningMaskPublish(packet:u32,lane:u32,cell:u32,
 sharpeningSource:bool,cellScale:u32){
  _=packet;_=lane;_=cell;_=sharpeningSource;_=cellScale;
}`;
  const finalScalarPacketMaskEntries = finalScalarPacketMaskLayout
    ? createSparseCM12FinalScalarPacketMaskWGSL({
      layout: finalScalarPacketMaskLayout,
      arenaName: "topologyArena",
    })
    : /* wgsl */ `
fn fsm1ChangedCell(cell:u32)->bool{_=cell;return false;}
fn fsm1FlipCell(cell:u32)->bool{_=cell;return false;}
fn fsm1ChangedOrFlipCell(cell:u32)->bool{_=cell;return false;}
fn fsm1Generation()->u32{return 0u;}
fn fsm1TopologyGeneration()->u32{return 0u;}
fn fsm1Published()->bool{return false;}
fn fsm1Changed(packet:u32)->vec2u{_=packet;return vec2u(0u);}
fn fsm1Nonexact(packet:u32)->vec2u{_=packet;return vec2u(0u);}
fn fsm1Bulk(packet:u32)->vec2u{_=packet;return vec2u(0u);}
fn fsm1Flip(packet:u32)->vec2u{_=packet;return vec2u(0u);}
fn fsm1Lane(mask:vec2u,lane:u32)->bool{_=mask;_=lane;return false;}
@compute @workgroup_size(1) fn beginSparseCM12FinalScalarMasks(){}
@compute @workgroup_size(64) fn publishSparseCM12FinalScalarMasks(){}
@compute @workgroup_size(1) fn sealSparseCM12FinalScalarMasks(){}
`;
  const internedBoundaryEntries = internedBoundaryImage
    ? /* wgsl */ `
fn cm12IBOSharedAcceptedSlot()->u32{return acceptedTopologySlot();}
fn cm12IBOSharedAcceptedGeneration()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()]);}
${createSparseCM12InternedBoundaryImageWGSL({
  layout: internedBoundaryImage.layout,
  arenaName: "topologyArena", hookPrefix: "cm12",
  baseWords: internedBoundaryImage.baseWords,
  packetsPerLeaf: 64,
  acceptedSlotHook: "cm12IBOSharedAcceptedSlot",
  acceptedGenerationHook: "cm12IBOSharedAcceptedGeneration",
})}
${createSparseCM12InternedRefLookupWGSL({
  layout: internedBoundaryImage.refLookupLayout,
  arenaName: "topologyArena", iboPrefix: "cm12",
  baseWords: internedBoundaryImage.baseWords,
})}
${createSparseCM12GeometryFaceNeighborsWGSL({
  baseWords: internedBoundaryImage.semanticAuthority.geometryBaseWords,
  leafCapacity: internedBoundaryImage.semanticAuthority.leafCapacity,
  offsetBaseWords: internedBoundaryImage.semanticAuthority.geometryOffsetBaseWords,
  neighborBaseWords: internedBoundaryImage.semanticAuthority.geometryNeighborBaseWords,
  arenaName: "topologyArena", hookPrefix: "cm12",
})}
const ISA1_AUTHORITY_BASE:u32=${internedBoundaryImage.semanticAuthority.authorityBaseWords}u;
const ISA1_AUTHORITY_LEAF_CAPACITY:u32=${
  internedBoundaryImage.semanticAuthority.leafCapacity}u;
const ISA1_IMMUTABLE_CONTENT_HASH:u32=${
  internedBoundaryImage.semanticAuthority.immutableContentHash}u;
const ISA1_IMMUTABLE_CERTIFICATE_HASH:u32=${
  internedBoundaryImage.semanticAuthority.immutableCertificateHash}u;
const ISA1_AUTHORITY_STAMPS:u32=ISA1_AUTHORITY_BASE+16u;
const ISA1_AUTHORITY_LIST:u32=ISA1_AUTHORITY_STAMPS+ISA1_AUTHORITY_LEAF_CAPACITY;
fn cm12ISARecordFault(code:u32,leaf:u32){
  if(code==0u){return;}atomicMax(&topologyArena[ISA1_AUTHORITY_BASE+7u],code);
  atomicMin(&topologyArena[ISA1_AUTHORITY_BASE+11u],leaf);
}
@compute @workgroup_size(1) fn validateSparseCM12InternedBoundaryImmutable(){
  let fault=cm12IBOValidateImmutableContent(
    ISA1_IMMUTABLE_CONTENT_HASH,ISA1_IMMUTABLE_CERTIFICATE_HASH);
  if(fault.x==0u){atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+1u],0x80000001u);
  }else{cm12ISARecordFault(fault.x,fault.y);}
}
fn cm12ISALeafMix(leaf:u32)->u32{
  return ((leaf^0x9e3779b9u)*0x01000193u)^0x85ebca6bu;
}
fn cm12ISAClaimClosureLeaf(leaf:u32,generation:u32)->bool{
  if(leaf>=ISA1_AUTHORITY_LEAF_CAPACITY){cm12ISARecordFault(1u,leaf);return false;}
  return atomicExchange(&topologyArena[ISA1_AUTHORITY_STAMPS+leaf],generation)!=generation;
}
fn cm12ISAAppendClosureLeaf(leaf:u32){
  let slot=atomicAdd(&topologyArena[ISA1_AUTHORITY_BASE+4u],1u);
  if(slot>=ISA1_AUTHORITY_LEAF_CAPACITY){cm12ISARecordFault(2u,leaf);return;}
  atomicStore(&topologyArena[ISA1_AUTHORITY_LIST+slot],leaf);
  let mixed=cm12ISALeafMix(leaf);
  atomicXor(&topologyArena[ISA1_AUTHORITY_BASE+5u],mixed);
  atomicAdd(&topologyArena[ISA1_AUTHORITY_BASE+6u],mixed*0x85ebca6bu);
}
fn cm12ISARowWord(row:u32,plane:u32)->u32{return ta(rowWord(row,plane));}
fn cm12ISARowTermFirst(row:u32)->u32{return rowTermOffset(row);}
fn cm12ISARowTermCount(row:u32)->u32{return rowTermCount(row);}
fn cm12ISATermCell(term:u32)->u32{return termCell(term);}
fn cm12ISATermBits(term:u32)->u32{return bitcast<u32>(termCoefficient(term));}
fn cm12ISACandidateFaceRange(descriptor:u32,side:u32,boundary:u32)->vec2u{
  return candidateFaceBoundaryRowRange(cm12IBOCanonicalWord(descriptor,1u),
    cm12IBOCanonicalWord(descriptor,2u),side,boundary);
}
fn cm12ISACandidateFaceRow(index:u32)->u32{return candidateFaceRow(index);}
fn cm12ISAScheduledRow(row:u32)->bool{return shadowRowScheduled(row);}
${createSparseCM12IBOSemanticAuthorityWGSL({ hookPrefix: "cm12", iboPrefix: "cm12" })}
// The selected IBO refs are the exact delta program for one leaf.  Validation
// executes that program packet-wise: SCMT supplies an independent expected
// semantic receipt while the immutable, upload-certified IBO payload supplies
// the observed receipt.  The receipt fold is commutative across rows, so 64
// lanes preserve its exact count/xor/sum result without serial row discovery.
var<workgroup> cm12ISAPacketActive:u32;
var<workgroup> cm12ISAPacketDescriptor:u32;
var<workgroup> cm12ISAPacketRootLeaf:u32;
var<workgroup> cm12ISAPacketNeighborCount:u32;
var<workgroup> cm12ISAPacketNeighborLeaf:u32;
var<workgroup> cm12ISAPacketExpectedCount:array<u32,64>;
var<workgroup> cm12ISAPacketExpectedXor:array<u32,64>;
var<workgroup> cm12ISAPacketExpectedSum:array<u32,64>;
var<workgroup> cm12ISAPacketObservedCount:array<u32,64>;
var<workgroup> cm12ISAPacketObservedXor:array<u32,64>;
var<workgroup> cm12ISAPacketObservedSum:array<u32,64>;
fn cm12ISAValidateScheduledLeafPacket(leaf:u32,lane:u32){
  let slot=cm12IBOShadowSlot();let generation=cm12IBOCandidateGeneration();
  let validated=generation|0x80000000u;
  if(lane==0u){cm12ISAPacketActive=0u;cm12ISAPacketDescriptor=IBO1_INVALID;
    for(var attempt=0u;leaf<IBO1_LEAF_CAPACITY&&attempt<64u;attempt+=1u){
      let stamp=atomicLoad(&topologyArena[ISA1_AUTHORITY_STAMPS+leaf]);
      if(stamp==validated){break;}
      if(stamp!=generation){cm12ISARecordFault(4u,leaf);break;}
      let claim=atomicCompareExchangeWeak(&topologyArena[ISA1_AUTHORITY_STAMPS+leaf],
        generation,validated);
      if(claim.exchanged){cm12ISAPacketActive=1u;
        cm12ISAPacketDescriptor=cm12IBOScheduledCanonical(leaf);break;}
    }}
  if(workgroupUniformLoad(&cm12ISAPacketActive)==0u){return;}
  let descriptor=workgroupUniformLoad(&cm12ISAPacketDescriptor);
  var expected=vec3u(0u);var observed=vec3u(0u);
  if(descriptor!=IBO1_INVALID){
    let resolution=cm12IBOCanonicalWord(descriptor,2u);
    let faceArea=resolution*resolution;
    for(var packet=lane;packet<6u*faceArea;packet+=64u){
      let side=packet/faceArea;let boundary=packet-side*faceArea;
      let range=cm12ISACandidateFaceRange(descriptor,side,boundary);
      for(var local=0u;local<range.y;local+=1u){
        let row=cm12ISACandidateFaceRow(range.x+local);
        if(cm12ISAScheduledRow(row)){expected=cm12ISAFold(expected,
          cm12ISASCMTStableRowHash(row),row);}}}
    for(var side=0u;side<6u;side+=1u){
      let refCount=cm12IBOFaceRefCount(slot,leaf,side);
      for(var localRef=0u;localRef<refCount;localRef+=1u){
        let reference=cm12IBORef(slot,leaf,side,localRef);
        let rowCount=cm12IBOTemplateHeaderWord(reference.x,6u);
        for(var localRow=lane;localRow<rowCount;localRow+=64u){
          let semantic=cm12ISAIBOTemplateRowHash(slot,leaf,reference,localRow);
          observed=cm12ISAFold(observed,semantic.y,semantic.x);}}}}
  cm12ISAPacketExpectedCount[lane]=expected.x;
  cm12ISAPacketExpectedXor[lane]=expected.y;
  cm12ISAPacketExpectedSum[lane]=expected.z;
  cm12ISAPacketObservedCount[lane]=observed.x;
  cm12ISAPacketObservedXor[lane]=observed.y;
  cm12ISAPacketObservedSum[lane]=observed.z;
  workgroupBarrier();
  var width=32u;loop{if(lane<width){
    cm12ISAPacketExpectedCount[lane]+=cm12ISAPacketExpectedCount[lane+width];
    cm12ISAPacketExpectedXor[lane]^=cm12ISAPacketExpectedXor[lane+width];
    cm12ISAPacketExpectedSum[lane]+=cm12ISAPacketExpectedSum[lane+width];
    cm12ISAPacketObservedCount[lane]+=cm12ISAPacketObservedCount[lane+width];
    cm12ISAPacketObservedXor[lane]^=cm12ISAPacketObservedXor[lane+width];
    cm12ISAPacketObservedSum[lane]+=cm12ISAPacketObservedSum[lane+width];
  }workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u){atomicAdd(&topologyArena[ISA1_AUTHORITY_BASE+8u],1u);
    let mixed=cm12ISALeafMix(leaf);
    atomicXor(&topologyArena[ISA1_AUTHORITY_BASE+9u],mixed);
    atomicAdd(&topologyArena[ISA1_AUTHORITY_BASE+10u],mixed*0x85ebca6bu);
    let expectedReceipt=vec3u(cm12ISAPacketExpectedCount[0],
      cm12ISAPacketExpectedXor[0],cm12ISAPacketExpectedSum[0]);
    let observedReceipt=vec3u(cm12ISAPacketObservedCount[0],
      cm12ISAPacketObservedXor[0],cm12ISAPacketObservedSum[0]);
    if(descriptor==IBO1_INVALID||cm12IBOLeafDescriptorId(slot,leaf)!=descriptor){
      cm12ISARecordFault(3u,leaf);cm12IBORecordFault(
        vec2u(IBO1_FAULT_REFERENCE,leaf));
    }else if(any(expectedReceipt!=observedReceipt)){
      cm12ISARecordFault(5u,leaf);cm12IBORecordFault(
        vec2u(IBO1_FAULT_REFERENCE,leaf));}}
  workgroupBarrier();
}
fn cm12ISABeginAuthority(){let generation=cm12IBOCandidateGeneration();
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+3u],generation);
  for(var at=4u;at<=10u;at+=1u){atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+at],0u);}
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+11u],IBO1_INVALID);
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+15u],0u);
}
fn cm12ISASetExpectedChanged(count:u32,xorHash:u32,sumHash:u32){
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+12u],count);
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+13u],xorHash);
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+14u],sumHash);
}
fn cm12ISARecordChangedLeaf(leaf:u32){
  atomicAdd(&topologyArena[ISA1_AUTHORITY_BASE+15u],1u);
  let mixed=cm12ISALeafMix(leaf);
  atomicXor(&topologyArena[ISA1_AUTHORITY_BASE+8u],mixed);
  atomicAdd(&topologyArena[ISA1_AUTHORITY_BASE+9u],mixed*0x85ebca6bu);
}
fn cm12ISAFinalizeAuthority()->bool{
  return atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE])==0x49534131u
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+1u])==0x80000001u
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+2u])==ISA1_AUTHORITY_LEAF_CAPACITY
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+3u])==cm12IBOCandidateGeneration()
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+7u])==0u
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+4u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+8u])
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+5u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+9u])
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+6u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+10u]);
}
fn cm12ISAAuthorityReady()->bool{
  return atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+3u])==cm12IBOCandidateGeneration()
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+7u])==0u
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+4u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+8u])
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+5u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+9u])
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+6u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+10u])
    ;
}
const CM12_IBO_LEVELS_PER_LEAF:u32=${Math.floor(
  internedBoundaryImage.layout.canonicalCapacity
    / internedBoundaryImage.layout.leafCapacity)}u;
fn cm12IBOShadowSlot()->u32{return 1u-acceptedTopologySlot();}
fn cm12IBOCandidateGeneration()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()])+1u;}
fn cm12IBORecordFault(fault:vec2u){
  if(fault.x==0u){return;}let header=cm12IBOSlotBase(cm12IBOShadowSlot());
  _=atomicMax(&topologyArena[header+5u],fault.x);
  _=atomicMin(&topologyArena[header+6u],fault.y);
}
fn cm12IBOScheduledCanonical(leaf:u32)->u32{
  if(leaf>=IBO1_LEAF_CAPACITY){return IBO1_INVALID;}
  let resolution=scheduledBrickResolution(leaf);
  for(var level=0u;level<CM12_IBO_LEVELS_PER_LEAF;level+=1u){
    let descriptor=leaf*CM12_IBO_LEVELS_PER_LEAF+level;
    if(cm12IBOCanonicalWord(descriptor,0u)!=0u
      &&cm12IBOCanonicalWord(descriptor,1u)==leaf
      &&cm12IBOCanonicalWord(descriptor,2u)==resolution){return descriptor;}}
  return IBO1_INVALID;
}
fn cm12IBOTryClaimLeaf(slot:u32,leaf:u32,generation:u32)->bool{
  let at=cm12IBOSlotLeafBase(slot)+IBO1_SLOT_LEAF_WORDS*leaf;
  for(var attempt=0u;attempt<64u;attempt+=1u){
    let current=cm12IBOLoad(at);if(current==generation){return false;}
    if(current==0u){cm12IBORecordFault(vec2u(IBO1_FAULT_GENERATION,leaf));
      return false;}
    let claim=atomicCompareExchangeWeak(&topologyArena[at],current,generation);
    if(claim.exchanged){return true;}}
  cm12IBORecordFault(vec2u(IBO1_FAULT_GENERATION,leaf));return false;
}
fn cm12IBOCompileScheduledLeaf(leaf:u32){
  if(leaf>=IBO1_LEAF_CAPACITY){return;}let slot=cm12IBOShadowSlot();
  let generation=cm12IBOCandidateGeneration();
  if(!cm12IBOTryClaimLeaf(slot,leaf,generation)){return;}
  let descriptor=cm12IBOScheduledCanonical(leaf);let scheduledActive=scheduledBrickActive(leaf);
  let begun=cm12IBOBeginDeltaLeaf(slot,leaf,generation,scheduledActive,descriptor);
  if(begun.x!=0u){cm12IBORecordFault(begun);return;}
  if(scheduledActive){for(var side=0u;side<6u;side+=1u){var output=0u;
    let count=cm12IBOInstantiationCount(descriptor,side);
    for(var local=0u;local<count;local+=1u){
      let entry=cm12IBOInstantiationEntry(descriptor,side,local);
      var selected=entry.x==IRL1_INVALID;var targetLeaf=IRL1_INVALID;
      if(entry.x!=IRL1_INVALID){targetLeaf=cm12IBOCanonicalWord(entry.x,1u);
        selected=targetLeaf<IBO1_LEAF_CAPACITY&&scheduledBrickActive(targetLeaf)
          &&cm12IBOScheduledCanonical(targetLeaf)==entry.x;}
      if(selected){let fault=cm12IBOWriteDeltaRef(slot,leaf,side,output,
          vec3u(entry.y,targetLeaf,entry.z));cm12IBORecordFault(fault);output+=1u;}
    }}}
  cm12IBORecordFault(cm12IBOSealDeltaLeaf(slot,leaf));
}
fn cm12IBOForEachGeometryCompile(leaf:u32){
  cm12IBOCompileScheduledLeaf(leaf);
  let count=cm12ISAGeometryNeighborCount(leaf);
  for(var local=0u;local<count;local+=1u){
    cm12IBOCompileScheduledLeaf(cm12ISAGeometryNeighbor(leaf,local));}
}
@compute @workgroup_size(1) fn beginSparseCM12InternedBoundaryDelta(){
  let header=cm12IBOSlotBase(cm12IBOShadowSlot());
  cm12IBOStore(header,cm12IBOCandidateGeneration());cm12IBOStore(header+1u,1u);
  cm12IBOStore(header+2u,cm12IBOAcceptedGeneration());
  cm12IBOStore(header+3u,0u);cm12IBOStore(header+4u,0u);
  cm12IBOStore(header+5u,0u);cm12IBOStore(header+6u,IBO1_INVALID);
  cm12ISABeginAuthority();
}
@compute @workgroup_size(1) fn compileSparseCM12InternedBoundaryDelta(
 @builtin(workgroup_id)wid:vec3u){let leaf=topologyDeltaLeafInvocation(wid.x);
  if(leaf!=INVALID){cm12ISARecordChangedLeaf(leaf);
    cm12ISAAppendGeometryClosure(leaf,cm12IBOCandidateGeneration());
    cm12IBOForEachGeometryCompile(leaf);}}
@compute @workgroup_size(1) fn finalizeSparseCM12ISAChangedSetReceipt(){
  let exact=atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+12u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+15u])
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+13u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+8u])
    &&atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+14u])
      ==atomicLoad(&topologyArena[ISA1_AUTHORITY_BASE+9u]);
  if(!exact){cm12ISARecordFault(6u,IBO1_INVALID);}
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+8u],0u);
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+9u],0u);
  atomicStore(&topologyArena[ISA1_AUTHORITY_BASE+10u],0u);
}
@compute @workgroup_size(64) fn validateSparseCM12InternedBoundaryDeltaPackets(
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  if(lid.x==0u){let listed=topologyDeltaLeafInvocation(wid.x);
    cm12ISAPacketRootLeaf=listed;
    cm12ISAPacketNeighborCount=0u;
    if(listed!=INVALID){
      cm12ISAPacketNeighborCount=cm12ISAGeometryNeighborCount(listed);}}
  let leaf=workgroupUniformLoad(&cm12ISAPacketRootLeaf);
  cm12ISAValidateScheduledLeafPacket(leaf,lid.x);
  let count=workgroupUniformLoad(&cm12ISAPacketNeighborCount);
  for(var local=0u;local<count;local+=1u){
    if(lid.x==0u){cm12ISAPacketNeighborLeaf=cm12ISAGeometryNeighbor(leaf,local);}
    cm12ISAValidateScheduledLeafPacket(
      workgroupUniformLoad(&cm12ISAPacketNeighborLeaf),lid.x);}}
@compute @workgroup_size(1) fn finalizeSparseCM12InternedBoundaryDelta(){
  let header=cm12IBOSlotBase(cm12IBOShadowSlot());let valid=cm12IBOLoad(header)
    ==cm12IBOCandidateGeneration()&&cm12IBOLoad(header+2u)==cm12IBOAcceptedGeneration()
    &&cm12IBOLoad(header+5u)==0u&&cm12IBOAcceptedSlot()==acceptedTopologySlot()
    &&cm12ISAFinalizeAuthority();
  cm12IBOStore(header+1u,select(3u,2u,valid));
}
@compute @workgroup_size(1) fn replaySparseCM12InternedBoundaryDelta(
 @builtin(workgroup_id)wid:vec3u){let leaf=topologyDeltaLeafInvocation(wid.x);
  if(leaf==INVALID){return;}let source=acceptedTopologySlot();let retiredSlot=1u-source;
  cm12IBOReplayDeltaLeaf(source,retiredSlot,leaf,cm12IBOAcceptedGeneration());
  let count=cm12ISAGeometryNeighborCount(leaf);
  for(var local=0u;local<count;local+=1u){cm12IBOReplayDeltaLeaf(
    source,retiredSlot,cm12ISAGeometryNeighbor(leaf,local),cm12IBOAcceptedGeneration());}
  if(wid.x==0u){cm12IBOStore(IBO1_BASE+2u,source);
    let generation=cm12IBOAcceptedGeneration();let header=cm12IBOSlotBase(retiredSlot);
    cm12IBOStore(IBO1_BASE+3u,generation);cm12IBOStore(header,generation);
    cm12IBOStore(header+1u,0u);cm12IBOStore(header+2u,generation);
    cm12IBOStore(header+5u,0u);cm12IBOStore(header+6u,IBO1_INVALID);}}
` : /* wgsl */ `
fn cm12ISASetExpectedChanged(count:u32,xorHash:u32,sumHash:u32){
  _=count;_=xorHash;_=sumHash;
}
@compute @workgroup_size(1) fn finalizeSparseCM12ISAChangedSetReceipt(){}
`;
  const implicitPressureInteriorEntries = internedBoundaryImage ? /* wgsl */ `
// Resolve the canonical brick descriptor already shared by transport and
// topology publication.  A non-INVALID result carries the three arithmetic
// cell strides for a certified strict interior; boundary/seam cells keep the
// compiled directed-edge path.
fn pressureImplicitInteriorStrides(cell:u32)->vec3u{
  let leaf=cellBrick(cell);let slot=cm12IBOAcceptedSlot();
  let descriptor=cm12IBOLeafDescriptorId(slot,leaf);
  if(descriptor==IBO1_INVALID||cm12IBOCanonicalWord(descriptor,0u)==0u){
    return vec3u(INVALID);}
  let first=cm12IBOLeafCellFirst(slot,leaf);
  let dimensions=cm12IBOLeafDimensions(slot,leaf);
  let count=dimensions.x*dimensions.y*dimensions.z;
  if(cell<first||cell-first>=count){return vec3u(INVALID);}
  let local=cell-first;let xy=dimensions.x*dimensions.y;
  let z=local/xy;let remainder=local-z*xy;
  let y=remainder/dimensions.x;let x=remainder-y*dimensions.x;
  if(x==0u||x+1u>=dimensions.x||y==0u||y+1u>=dimensions.y
    ||z==0u||z+1u>=dimensions.z){return vec3u(INVALID);}
  return vec3u(1u,dimensions.x,xy);
}
` : /* wgsl */ `
fn pressureImplicitInteriorStrides(cell:u32)->vec3u{
  _=cell;return vec3u(INVALID);
}
`;
  const iboTRAEntries = /* wgsl */ `
${createSparseCM12IboTRAResidentHooksWGSL({
  iboPrefix: "cm12", residentPrefix: "cm12", arenaName: "topologyArena",
})}
${createSparseCM12IboTRASupplementWGSL({
  layout: internedBoundaryImage.traSupplementLayout,
  arenaName: "topologyArena", hookPrefix: "cm12",
  baseWords: internedBoundaryImage.baseWords,
})}
`;
  const dirtyFaceRowMaskEntries = createSparseCM12DirtyFaceRowMaskWGSL({
    layout: dirtyFaceRowMaskLayout, arenaName: "activity",
  });
  const internedBoundaryCommitReceipt = internedBoundaryImage ? /* wgsl */ `
  let iboSlot=cm12IBOShadowSlot();let iboHeader=cm12IBOSlotBase(iboSlot);
  valid=valid&&cm12IBOLoad(iboHeader)==cm12IBOCandidateGeneration()
    &&cm12IBOLoad(iboHeader+1u)==2u
    &&cm12IBOLoad(iboHeader+2u)==cm12IBOAcceptedGeneration()
    &&cm12IBOLoad(iboHeader+5u)==0u
    &&cm12ISAAuthorityReady()
    &&cm12IBOSelectorMirror()==acceptedTopologySlot()
    &&cm12IBOGenerationMirror()==cm12IBOAcceptedGeneration();
` : "";
  return /* wgsl */ `
${createCm12NumericsWGSL()}

const INVALID:u32=0xffffffffu;
const WORKGROUP:u32=64u;
const PRESSURE_EXECUTION_QA_ORACLE:bool=${pressureExecutionQAOracle
  ? "true" : "false"};
// LOD1 is production. The legacy hash/span ladder remains only as a
// construction-specialized equivalence and timing oracle.
const EXP_LOGICAL_OWNER_DIRECTORY:bool=true;
// Production reads the two-per-word owner plane. The unpacked LOD1 arm keeps
// its self-describing record ABI solely as an equivalence oracle.
const EXP_PACKED_LOGICAL_OWNER16:bool=${logicalOwnerDirectory?.packedOwner16BaseWords !== undefined
  ? "true" : "false"};
// Receipt-visible construction specialization: the directory window and
// effective-velocity plane remain, while every per-site halo declaration,
// fill, and sizing reduction is absent from the generated module.
const EXP_TRANSPORT_PROFILE:bool=${phase1TransportProfileBaseWords === undefined
    ? "false" : "true"};
const CM12_TRANSPORT_PROFILE_BASE:u32=${phase1TransportProfileBaseWords ?? 0}u;
// TEI2 packets are published from accepted leaf/rung state and compiled from
// the stable SRR spatial-tile authority without changing receipt identity.
const EXP_ACTIVITY_SCALAR_BRICKS:bool=true;
const ACTIVITY_HEADER_WORDS:u32=28u;
const ACTIVITY_RECORD_WORDS:u32=39u;
const ACTIVITY_RECOVERY_LOCK:u32=0x80000000u;
// Word 9 otherwise contains small planning-reason enums.  The high bit is a
// tail-transaction marker for same-rung activation/retirement, ensuring that
// lifecycle membership enters the same shadow worklist/image flip as a rung
// change instead of mutating accepted membership after publication.
const ACTIVITY_LIFECYCLE_CHANGED:u32=0x80000000u;
const ACTIVITY_TOPOLOGY_PREPARATION_SCHEDULED:u32=1u;
const ACTIVITY_CANDIDATE_ACTIVE:u32=0x80000000u;
const ACCEPTED_COARSE_ROW_COUNT:u32=22u;
const ACCEPTED_MIXED_ROW_COUNT:u32=23u;
const PRESSURE_ACTIVE_ROW_COUNT:u32=24u;
const BRICK_FINE_RESOLUTION:u32=${brickFineResolution}u;
const TEMPLATE_LEVEL_COUNT:u32=${templateLevelCount}u;
const CANDIDATE_CELLS_PER_BRICK:u32=${candidateCellCount}u;
const CANDIDATE_FACE_SAMPLES_PER_SIDE:u32=${candidateFaceSampleCount}u;
const PRESENTATION_PAGES_PER_AXIS:u32=${presentationPagesPerAxis}u;
const PRESENTATION_PAGE_RESOLUTION:u32=${presentationPageResolution}u;
const PRESENTATION_SAMPLES_PER_PAGE:u32=${presentationPageResolution ** 3}u;
const EXP_PRESENTATION_UNIFORM_BULK:bool=true;
const PRESENTATION_CACHE_CAPACITY:u32=${presentationCacheCapacity}u;
const TEMPLATE_CELL_RESOLUTION_BITS:u32=5u;
const TEMPLATE_CELL_RESOLUTION_MASK:u32=31u;
const ACTIVITY_FIXED:f32=65536.0;
const FACE_VELOCITY_SUPPORT:u32=${faceVelocitySupportBaseFloats}u;

// Pressure-journal region, a tail range of state past every physics field.
// It lives there rather than in a buffer of its own because the compute bind
// group is already at the device's ten-storage-buffer ceiling — the same
// reason the tracer lattice sits past the physics fields.
const JOURNAL_HEADER_FLOATS:u32=8u;
const JOURNAL_ITERATION_FLOATS:u32=16u;
const JOURNAL_FIELD_COUNT:u32=4u;

struct Params {
  counts:vec4u,             // cell, row, incidence, dense
  dimensions:vec4u,
  topologyOffsets:vec4u,    // cells, rows, terms, incidenceOffsets
  topologyOffsets2:vec4u,   // incidences, direct logical-brick owners, brick records, background owner
  stateOffsets0:vec4u,      // density A/B, gamma A/B
  stateOffsets1:vec4u,      // cell velocity A/B, face A/B
  stateOffsets2:vec4u,      // pressure, rhs, diagonal, liquid
  stateOffsets3:vec4u,      // theta, residual, preconditioned, direction
  stateOffsets4:vec4u,      // applied, divergence, presentation brick wet, reserved
  stateOffsets5:vec4u,      // sharpening/D4 rho, D4 gamma, tracers, dense face support
  frame:vec4f,              // dt, finest cell metres, pressure scale, parity
  acceleration:vec4f,       // finest cells / second^2
  dispatch:vec4u,           // cell workgroups, row workgroups, pcg iterations, brick count
  injectionCenter:vec4f,
  injectionRadius:vec4f,
  sharpening:vec4f,         // Algorithm 2 distance/substeps, residency density/mass
  activityThresholds:vec4f, // 8/4/2 travel floors, thin-feature width
  activityDensity:vec4f,    // thin floor, surface low/high, detail tolerance
  activityTiming:vec4f,     // front lookahead, promote/emergency/demote scores
  activityEpochs:vec4u,     // cadence, promotion epochs, demotion epochs, activity signals enabled
  boundaryCenter:vec4f,
  boundaryRadii:vec4f,
  topologyScheduling:vec4u, // ordinary shadow bricks/frame, pressure tolerance bits
  solidOffsets:vec4u,       // dynamic cell open and row data
  rigidWorld:vec4f,         // world metres and rigid-body count
  tracerGrid:vec4u,         // tracer lattice dimensions, tracer count
  tracerOrigin:vec4f,       // lattice origin in fine cells, isotropic spacing
  journal:vec4u,            // pressure journal base, iteration/snapshot capacity, cell stride
  refinementRegionControl:vec4u,
  refinementRegions:array<vec4f,16>, // min.xyz/floor, max.xyz/optional ceiling
}

@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>topology:array<u32>;
@group(0)@binding(2)var<storage,read_write>state:array<f32>;
@group(0)@binding(3)var<storage,read_write>partials:array<vec4f>;
@group(0)@binding(4)var<storage,read_write>scalars:array<f32>;
@group(0)@binding(11)var<storage,read_write>conditioning:array<atomic<i32>>;
// Device-owned activity, planning, and logical-residency arena. Physics reads
// the published active bit, while immutable packed topology and accepted CM12
// fields remain separate storage.
@group(0)@binding(12)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(13)var<storage,read_write>candidateState:array<f32>;
@group(0)@binding(14)var<storage,read_write>fineMetadata:array<u32>;
@group(0)@binding(15)var<storage,read_write>fineSamples:array<u32>;
// Physical dyadic cell and row templates are immutable after construction.
// Worklists are device-owned and double-buffered: a commit publishes by
// flipping word 2 only after shadow transfer/projection has completed.
@group(0)@binding(16)var<storage,read_write>topologyArena:array<atomic<u32>>;

${frameControlEntries}
${pressureRepairEntries}
${incrementalActivityEntries}
${canonicalMembershipEntries}
${framePlanEntries}
${framePlanPresentationEntries}
${pressureTopologyRepairEntries}
${persistentPressureCacheEntries}
${effectiveTransportVelocityEntries}
${velocityExtensionEntries}
${pressureAddressingABEntries}

fn topologyWorklistBase()->u32{return atomicLoad(&topologyArena[14u]);}
fn sparseCM12TopologyLifecycleAccepted()->bool{
  let base=topologyWorklistBase();let phase=atomicLoad(&topologyArena[base+3u]);
  let topologyAccepted=phase==2u||(phase==0u
    &&atomicLoad(&topologyArena[base])==atomicLoad(&topologyArena[base+1u])
    &&atomicLoad(&activity[12])==atomicLoad(&topologyArena[base]));
  return topologyAccepted;
}
fn residencyDensityThreshold()->f32{
  return max(CM12_DRY_CELL_THRESHOLD,p.sharpening.z);
}
fn acceptedTopologySlot()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+2u])&1u;
}
fn acceptedTemplateCellCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+4u]);
}
fn acceptedTemplateRowCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+5u]);
}
fn acceptedTemplateCellWorkgroups()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+8u]);
}
fn acceptedTemplateRowWorkgroups()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+11u]);
}
fn acceptedTemplateCellInvocation(invocation:u32)->u32{
  if(invocation>=acceptedTemplateCellCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+14u+acceptedTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
// Binding 15 is the presentation sample arena in ordinary frame passes and a
// compact ordinary-u32 pressure worklist/neighbor arena in the pressure bind
// group. Counters are atomic only while compaction is being built; finalized
// pressure kernels read this immutable snapshot without atomic semantics.
fn pressureCellCount()->u32{return pcmCellAcceptedCount();}
fn pressureCellWorkgroups()->u32{return (pressureCellCount()+63u)/64u;}
fn pressureCellInvocation(invocation:u32)->u32{
  let cell=pabPressureCellAddress(invocation);
  return select(INVALID,cell,cell!=INVALID&&pcmCellContains(cell)
    &&cellActive(cell));
}
fn pressureRowCount()->u32{return pcmRowAcceptedCount();}
fn pressureRowInvocation(invocation:u32)->u32{
  let row=pcmRowRankSelect(invocation);
  return select(INVALID,row,row!=INVALID&&pcmRowContains(row)&&rowAccepted(row)
    &&state[p.stateOffsets3.x+row]>0.0);
}
fn pressureNeighborOffset()->u32{return 0u;}
fn acceptedTemplateRowInvocation(invocation:u32)->u32{
  if(invocation>=acceptedTemplateRowCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+16u+acceptedTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
fn acceptedLeafManifestBase()->u32{
  let base=topologyWorklistBase();
  return base+atomicLoad(&topologyArena[base+30u])
    +atomicLoad(&topologyArena[base+27u])*atomicLoad(&topologyArena[base+31u]);
}
fn acceptedLeafCount()->u32{
  let base=acceptedLeafManifestBase();
  return atomicLoad(&topologyArena[base+acceptedTopologySlot()]);
}
fn acceptedLeafInvocation(invocation:u32)->u32{
  let base=acceptedLeafManifestBase();let slot=acceptedTopologySlot();
  if(invocation>=atomicLoad(&topologyArena[base+slot])){return INVALID;}
  let offset=atomicLoad(&topologyArena[base+2u+slot]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
fn shadowLeafInvocation(invocation:u32)->u32{
  let base=acceptedLeafManifestBase();let slot=shadowTopologySlot();
  if(invocation>=atomicLoad(&topologyArena[base+slot])){return INVALID;}
  let offset=atomicLoad(&topologyArena[base+2u+slot]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
fn topologyDeltaLeafInvocation(invocation:u32)->u32{
  let base=acceptedLeafManifestBase();
  if(invocation>=atomicLoad(&topologyArena[base+10u])){return INVALID;}
  let offset=atomicLoad(&topologyArena[base+11u]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
fn acceptedRowMembershipStampBase()->u32{
  return acceptedLeafManifestBase()+20u+3u*p.dispatch.w;
}
fn acceptedRowMember(row:u32)->bool{
  return row<p.counts.y&&(atomicLoad(&topologyArena[
    acceptedRowMembershipStampBase()+row])&(1u<<acceptedTopologySlot()))!=0u;
}
// Pressure passes bind an immutable copy of the physical template arena at
// binding 14.  Unlike topologyArena this path is ordinary read-only storage,
// so recurring SpMVs do not pay atomic-load semantics for data which never
// changes after construction.
fn pressureTemplateWord(index:u32)->u32{return fineMetadata[index];}
fn pressureEdgeCount()->u32{
  return pressureTemplateWord(pressureTemplateWord(15u)+p.counts.x);
}
fn pressureEdgeRows()->u32{return pressureTemplateWord(15u)+p.counts.x+1u;}
fn pressureEdgeWeights()->u32{return pressureEdgeRows()+pressureEdgeCount();}
fn brickAggregateTopology()->u32{return pressureTemplateWord(14u);}
fn brickAggregateEdgeCount()->u32{
  return pressureTemplateWord(brickAggregateTopology()+1u);
}
fn brickAggregateEdgeWeightOffset()->u32{return pressureEdgeCount();}
fn brickAggregateRhsOffset()->u32{return pressureEdgeCount()+brickAggregateEdgeCount();}
fn brickAggregateDiagonalOffset()->u32{return brickAggregateRhsOffset()+p.dispatch.w;}
fn brickAggregateAOffset()->u32{return brickAggregateRhsOffset()+2u*p.dispatch.w;}
fn brickAggregateBOffset()->u32{return brickAggregateRhsOffset()+3u*p.dispatch.w;}
fn brickAggregateRangeOffset()->u32{return brickAggregateRhsOffset()+4u*p.dispatch.w;}
fn cachedPressureBrickRange(brick:u32)->vec2u{
  if(PRESSURE_EXECUTION_QA_ORACLE){
    let packed=bitcast<u32>(candidateState[brickAggregateRangeOffset()+brick]);
    if(packed==0u){return vec2u(0u);}let level=(packed&7u)-1u;
    return vec2u((packed>>3u)-1u,1u<<(3u*level));
  }
  let packed=atomicLoad(&topologyArena[
    ${persistentPressureCacheLayout?.brickAggregateRangeBaseWords ?? 0}u+brick]);
  if(packed==0u){return vec2u(0u);}
  let level=(packed&7u)-1u;
  return vec2u((packed>>3u)-1u,1u<<(3u*level));
}
fn pressureHierarchyTopology()->u32{return pressureTemplateWord(13u);}
fn pressureHierarchyDescriptor(level:u32)->u32{
  return pressureHierarchyTopology()+1u+10u*level;
}
fn pressureHierarchyGroupCount(level:u32)->u32{
  return pressureTemplateWord(pressureHierarchyDescriptor(level));
}
fn pressureHierarchyLevelCount()->u32{return pressureTemplateWord(pressureHierarchyTopology());}
fn pressureHierarchyEdgeCount(level:u32)->u32{
  let descriptor=pressureHierarchyDescriptor(level);
  let offsets=pressureTemplateWord(descriptor+6u);
  return pressureTemplateWord(offsets+pressureHierarchyGroupCount(level));
}
fn pressureHierarchyDynamicBase(level:u32)->u32{
  return brickAggregateRangeOffset()+p.dispatch.w
    +pressureTemplateWord(pressureHierarchyDescriptor(level)+9u);
}
fn pressureHierarchyEdgeWeightOffset(level:u32)->u32{
  return pressureHierarchyDynamicBase(level);
}
fn pressureHierarchyDiagonalOffset(level:u32)->u32{
  return pressureHierarchyDynamicBase(level)+pressureHierarchyEdgeCount(level);
}
fn pressureHierarchyRhsOffset(level:u32)->u32{
  return pressureHierarchyDiagonalOffset(level)+pressureHierarchyGroupCount(level);
}
fn pressureHierarchyAOffset(level:u32)->u32{
  return pressureHierarchyDiagonalOffset(level)+2u*pressureHierarchyGroupCount(level);
}
fn pressureHierarchyBOffset(level:u32)->u32{
  return pressureHierarchyDiagonalOffset(level)+3u*pressureHierarchyGroupCount(level);
}
fn pressureDensityCacheOffset()->u32{
  let last=pressureHierarchyLevelCount()-1u;
  return pressureHierarchyBOffset(last)+pressureHierarchyGroupCount(last);
}
fn pressureHierarchyGroupAddress(linear:u32)->vec2u{
  var remainder=linear;
  for(var level=0u;level<pressureHierarchyLevelCount();level+=1u){
    let count=pressureHierarchyGroupCount(level);
    if(remainder<count){return vec2u(level,remainder);}
    remainder-=count;
  }
  return vec2u(INVALID,INVALID);
}
fn pressureHierarchyEdgeAddress(linear:u32)->vec2u{
  var remainder=linear;
  for(var level=0u;level<pressureHierarchyLevelCount();level+=1u){
    let count=pressureHierarchyEdgeCount(level);
    if(remainder<count){return vec2u(level,remainder);}
    remainder-=count;
  }
  return vec2u(INVALID,INVALID);
}
fn shadowTopologySlot()->u32{return 1u-acceptedTopologySlot();}
fn shadowTemplateCellCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+18u]);
}
fn shadowTemplateRowCount()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()+19u]);
}
fn shadowTemplateCellInvocation(invocation:u32)->u32{
  if(invocation>=shadowTemplateCellCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+14u+shadowTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}
fn shadowTemplateRowInvocation(invocation:u32)->u32{
  if(invocation>=shadowTemplateRowCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+16u+shadowTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
}

fn ta(index:u32)->u32{return atomicLoad(&topologyArena[index]);}
fn taf(index:u32)->f32{return bitcast<f32>(ta(index));}
fn sourceDensity()->u32{return select(p.stateOffsets0.x,p.stateOffsets0.y,
  cm12FCSourceScalarParity()!=0u);}
fn destinationDensity()->u32{return select(p.stateOffsets0.y,p.stateOffsets0.x,
  cm12FCDestinationScalarParity()==0u);}
fn sourceGamma()->u32{return select(p.stateOffsets0.z,p.stateOffsets0.w,
  cm12FCSourceScalarParity()!=0u);}
fn destinationGamma()->u32{return select(p.stateOffsets0.w,p.stateOffsets0.z,
  cm12FCDestinationScalarParity()==0u);}
fn sourceCellVelocity()->u32{return select(p.stateOffsets1.x,p.stateOffsets1.y,
  cm12FCSourceFaceParity()!=0u);}
fn destinationCellVelocity()->u32{return select(p.stateOffsets1.y,p.stateOffsets1.x,
  cm12FCDestinationFaceParity()==0u);}
fn sourceFaceVelocity()->u32{return select(p.stateOffsets1.z,p.stateOffsets1.w,
  cm12FCSourceFaceParity()!=0u);}
fn destinationFaceVelocity()->u32{return select(p.stateOffsets1.w,p.stateOffsets1.z,
  cm12FCDestinationFaceParity()==0u);}

@compute @workgroup_size(1)
fn beginSparseCM12FrameControl(){_=cm12FCBegin();}
@compute @workgroup_size(1)
fn publishSparseCM12FrameBodyAuthority(){
  _=cm12FCPublishBody(min(12u,u32(max(0.0,round(p.rigidWorld.w)))));
}
@compute @workgroup_size(1)
fn publishSparseCM12FrameBoundaryAuthority(){
  _=cm12FCPublishBoundary((p.dimensions.w&1u)!=0u);
}
@compute @workgroup_size(1)
fn sealSparseCM12FrameControl(){_=cm12FCSeal();}
@compute @workgroup_size(1)
fn publishSparseCM12FrameScalarOutput(){
  _=cm12FCPublishOutput(${SPARSE_CM12_FRAME_CONTROL_COVERAGE.scalarOutput}u);
}
@compute @workgroup_size(1)
fn publishSparseCM12FrameFaceOutput(){
  _=cm12FCPublishOutput(${SPARSE_CM12_FRAME_CONTROL_COVERAGE.faceOutput}u);
}
@compute @workgroup_size(1)
fn commitSparseCM12FrameControl(){
  _=cm12FCCommit();
}
@compute @workgroup_size(1)
fn invalidateSparseCM12FrameD4ForInjection(){
  if(!sparseCM12TopologyLifecycleAccepted()){return;}
}
@compute @workgroup_size(64)
fn sparseCM12FrameControlNoop(){ }
fn hasRigidBodies()->bool{return p.rigidWorld.w>=0.5;}
fn pressureRelativeTolerance()->f32{return bitcast<f32>(p.topologyScheduling.y);}
fn pipelinedPressureActive()->bool{return scalars[5]>0.5&&scalars[14]<0.5;}

// True only on the pipeline variant the host encodes at a snapshot iteration.
// A dispatch cannot be told which iteration it is — the uniform is written once
// per frame and WebGPU has no push constant — so "is this a snapshot" is a
// pipeline property and "which snapshot" is a device-side cursor.
override JOURNAL_SNAPSHOT:bool=false;
fn journalBase()->u32{return p.journal.x;}
fn journalIterationCapacity()->u32{return p.journal.y;}
fn journalSnapshotCapacity()->u32{return p.journal.z;}
fn journalCellStride()->u32{return p.journal.w;}
fn journalArmed()->bool{return p.journal.x!=0u&&p.journal.y!=0u&&p.journal.w!=0u;}
fn journalSnapshotField(slot:u32,field:u32)->u32{
  return journalBase()+JOURNAL_HEADER_FLOATS
    +journalIterationCapacity()*JOURNAL_ITERATION_FLOATS
    +(slot*JOURNAL_FIELD_COUNT+field)*journalCellStride();
}

${createSparseCM12CellAccessWGSL(SPARSE_CM12_ATOMIC_ARENA_READERS)}
${createSparseCM12RowAccessWGSL(SPARSE_CM12_ATOMIC_ARENA_READERS)}
fn cellOpenFraction(id:u32)->f32{
  if(!hasRigidBodies()){return 1.0;}
  return state[p.solidOffsets.x+id];
}
fn cellOpenVolume(id:u32)->f32{
  if(!hasRigidBodies()){return cellVolume(id);}
  return cellVolume(id)*state[p.solidOffsets.x+id];
}
fn cellSeparatingMinimum(id:u32)->bool{return false;}
fn rowOpenFraction(id:u32)->f32{
  if(!hasRigidBodies()){return 1.0;}return state[p.solidOffsets.y+3u*id];
}
fn rowSolidVelocity(id:u32)->f32{
  if(!hasRigidBodies()){return 0.0;}return state[p.solidOffsets.y+3u*id+1u];
}
fn rowPressureOpenFraction(id:u32)->f32{
  if(!hasRigidBodies()){return 1.0;}return state[p.solidOffsets.y+3u*id+2u];
}
fn rowDualWeight(id:u32)->f32{
  if(!hasRigidBodies()){return rowStaticDualWeight(id);}
  return rowStaticDualWeight(id)*rowPressureOpenFraction(id);
}
fn rowArea(id:u32)->f32{
  if(!hasRigidBodies()){return rowStaticArea(id);}
  return rowStaticArea(id)*rowOpenFraction(id);
}
fn candidateFaceBoundaryRowRange(brick:u32,accepted:u32,
 side:u32,boundary:u32)->vec2u{
  let configuration=(brick*TEMPLATE_LEVEL_COUNT+templateLevelIndex(accepted))*6u+side;
  let offsets=ta(ta(24u)+configuration);
  let begin=ta(offsets+boundary);return vec2u(begin,ta(offsets+boundary+1u)-begin);
}
fn candidateFaceRow(index:u32)->u32{return ta(index);}
fn activityRecord(brick:u32)->u32{
  return ACTIVITY_HEADER_WORDS+ACTIVITY_RECORD_WORDS*brick;
}
fn topologyPreparationScheduledAt(record:u32)->bool{
  return (atomicLoad(&activity[record+35u])
    &ACTIVITY_TOPOLOGY_PREPARATION_SCHEDULED)!=0u;
}
fn setTopologyPreparationScheduled(record:u32,scheduled:bool){
  let value=select(0u,ACTIVITY_TOPOLOGY_PREPARATION_SCHEDULED,scheduled);
  atomicStore(&activity[record+35u],
    (atomicLoad(&activity[record+35u])&ACTIVITY_CANDIDATE_ACTIVE)|value);
}
fn setCandidateBrickActiveAt(record:u32,enabled:bool){
  let value=select(0u,ACTIVITY_CANDIDATE_ACTIVE,enabled);
  atomicStore(&activity[record+35u],(atomicLoad(&activity[record+35u])
    &ACTIVITY_TOPOLOGY_PREPARATION_SCHEDULED)|value);
}
fn scheduledBrickResolution(brick:u32)->u32{
  let record=activityRecord(brick);
  return select(atomicLoad(&activity[record+12u]),atomicLoad(&activity[record+13u]),
    topologyPreparationScheduledAt(record));
}
fn templateLevelIndex(resolution:u32)->u32{
  var level=0u;var rung=resolution;
  while(rung>1u){rung/=2u;level+=1u;}
  return level;
}
fn acceptedBrickResolution(brick:u32)->u32{
  return atomicLoad(&activity[activityRecord(brick)+12u]);
}
fn brickSpan(brick:u32)->u32{
  return 1u<<(topology[p.topologyOffsets2.z+2u*brick]&31u);
}
fn brickPackedCandidateSlot(brick:u32)->u32{
  let encoded=topology[p.topologyOffsets2.z+2u*brick]>>5u;
  return select(INVALID,encoded-1u,encoded!=0u);
}
fn brickCandidateSlot(brick:u32)->u32{
  return brickPackedCandidateSlot(brick);
}
fn brickCandidateTopologyComplete(brick:u32)->bool{
  // A packed slot is the compact completeness receipt: unlike a geometry-only
  // page, it owns the candidate cells, shared rows, incidence, and field slot
  // needed by transfer, PTR, PCM, and publication.
  return brickPackedCandidateSlot(brick)!=INVALID;
}
fn brickCandidatePlanningEnabled(brick:u32)->bool{
  // A candidate rung is publishable only when the immutable template catalog
  // gives this brick a packed slot containing cells, rows, and incidence. The
  // dynamic page pool currently synthesizes cell geometry only; treating that
  // partial page as a publishable rung lets PTR retire and re-enable the same
  // accepted stable IDs. Keep those leaves at their accepted rung until the
  // complete dynamic descriptor has one generation/ownership contract.
  return brickCandidateTopologyComplete(brick);
}

// A dormant construction leaf already owns immutable stable cells, rows and
// incidence at its accepted rung. Activating exactly that rung needs no
// mutable candidate slot: the candidate field is the dry construction state
// (rho=0, gamma=1, velocity=pressure=0), and the ordinary shadow image can use
// the construction identities directly. This exception is deliberately
// membership-only; every active-leaf or rerung transition still requires a
// complete packed candidate.
fn constructionActivationIntentWithoutSlot(brick:u32,candidate:u32)->bool{
  if(brick>=p.dispatch.w||brickPackedCandidateSlot(brick)!=INVALID
    ||brickActive(brick)||!candidateBrickActive(brick)){return false;}
  return candidate==acceptedBrickResolution(brick);
}
fn scheduledConstructionActivationWithoutSlot(brick:u32)->bool{
  if(brick>=p.dispatch.w){return false;}let output=activityRecord(brick);
  return topologyPreparationScheduledAt(output)
    &&constructionActivationIntentWithoutSlot(
      brick,atomicLoad(&activity[output+13u]));
}

// Authored cell-size bounds operate at Sparse CM12's topology granularity: a
// whole brick. Full containment prevents a box from changing cells outside its
// boundary. The downstream refine-only closure may still raise this result by
// one or more rungs where strict 2:1 grading requires it.
fn sparseCM12RefinementRegionResolutionBounds(brick:u32)->vec2u{
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;
  let edge=BRICK_FINE_RESOLUTION*brickSpan(brick);
  let low=vec3f(vec3u(x,y,z)*BRICK_FINE_RESOLUTION);
  let high=vec3f(min(vec3u(low)+vec3u(edge),p.dimensions.xyz));
  var floorSize=1u;var ceilingSize=0u;
  let count=min(p.refinementRegionControl.x,8u);
  for(var index=0u;index<count;index+=1u){
    let lo=p.refinementRegions[2u*index];let hi=p.refinementRegions[2u*index+1u];
    if(all(low>=lo.xyz)&&all(high<=hi.xyz)){
      floorSize=max(floorSize,u32(lo.w));let authoredCeiling=u32(hi.w);
      if(authoredCeiling>0u){ceilingSize=select(min(ceilingSize,authoredCeiling),
        authoredCeiling,ceilingSize==0u);}
    }
  }
  let maximumResolution=clamp(edge/max(1u,floorSize),1u,BRICK_FINE_RESOLUTION);
  let minimumResolution=select(1u,
    clamp(edge/max(1u,ceilingSize),1u,BRICK_FINE_RESOLUTION),ceilingSize>0u);
  return vec2u(minimumResolution,maximumResolution);
}

fn applySparseCM12RefinementRegionBounds(brick:u32,requested:u32)->u32{
  let bounds=sparseCM12RefinementRegionResolutionBounds(brick);
  // A finer ceiling wins if overlapping authored boxes conflict, matching the
  // conservative ordering in the legacy octree refinement gate.
  return max(bounds.x,min(bounds.y,requested));
}
fn templateBrickCellRange(brick:u32,resolution:u32)->vec2u{
  let at=ta(11u)+2u*(TEMPLATE_LEVEL_COUNT*brick+templateLevelIndex(resolution));
  return vec2u(ta(at),ta(at+1u));
}
fn templateRowOwnerRange(brick:u32,resolution:u32)->vec2u{
  let index=TEMPLATE_LEVEL_COUNT*brick+templateLevelIndex(resolution);
  let offsets=ta(16u);let first=ta(offsets+index);
  return vec2u(first,ta(offsets+index+1u)-first);
}
fn templateRowOwnerInvocation(index:u32)->u32{return index;}
fn brickActive(brick:u32)->bool{
  return brick<p.dispatch.w&&atomicLoad(&activity[activityRecord(brick)+10u])!=0u;
}
fn candidateBrickActive(brick:u32)->bool{
  return brick<p.dispatch.w&&(atomicLoad(&activity[activityRecord(brick)+35u])
    &ACTIVITY_CANDIDATE_ACTIVE)!=0u;
}
fn candidateTopologyCellActive(cell:u32)->bool{
  if(cell>=p.counts.x){return false;}let brick=cellBrick(cell);
  if(!candidateBrickActive(brick)){return false;}
  let range=templateBrickCellRange(brick,scheduledBrickResolution(brick));
  return cell>=range.x&&cell<range.x+range.y;
}
fn candidateTopologyCellRetired(cell:u32)->bool{
  return cell<p.counts.x&&cellActive(cell)&&!candidateTopologyCellActive(cell);
}
fn scheduledBrickActive(brick:u32)->bool{return candidateBrickActive(brick);}
fn cellActive(cell:u32)->bool{
  let brick=cellBrick(cell);
  return brickActive(brick)&&cellResolution(cell)==acceptedBrickResolution(brick);
}
fn rowAccepted(row:u32)->bool{
  let requirements=rowRequirementOffset(row);let count=ta(requirements);
  for(var at=0u;at<count;at+=1u){let metadata=ta(requirements+1u+at);
    let brick=metadata>>TEMPLATE_CELL_RESOLUTION_BITS;
    let resolution=metadata&TEMPLATE_CELL_RESOLUTION_MASK;
    if(!brickActive(brick)||acceptedBrickResolution(brick)!=resolution){return false;}
  }
  return true;
}
fn cellTransportActive(cell:u32)->bool{return cellActive(cell)&&cellOpenVolume(cell)>1e-8;}
// A body can cover a wet cell in one frame. Keep its conservative mass receipt
// while V_i is exactly zero; partial cells use the Sec. 3.6 excess scatter, and
// an uncovered cell re-enters transport with the same mass instead of making
// water disappear inside a moving solid.
fn dynamicallyCoveredCell(cell:u32)->bool{
  return hasRigidBodies()&&cellActive(cell)&&cellVolume(cell)>1e-8
    &&state[p.solidOffsets.x+cell]<=1e-8;
}
// The host clears the complete template field before preparePressure marks the
// accepted liquid cells. The bit itself is therefore the pressure-epoch
// activity mask; repeating cellActive here would re-walk brick metadata for
// every neighbor of every SpMV.
fn isLiquid(cell:u32)->bool{return state[p.stateOffsets2.w+cell]>0.5;}

fn hasEmbeddedBoundary()->bool{return (p.dimensions.w&1u)!=0u;}
fn boundaryDistance2(q:vec3f)->f32{
  let offset=(q-p.boundaryCenter.xyz)/max(p.boundaryRadii.xyz,vec3f(1e-6));
  return dot(offset,offset);
}
fn insideEmbeddedBoundary(q:vec3f)->bool{
  return !hasEmbeddedBoundary()||boundaryDistance2(q)<=1.0;
}
fn clampInsideEmbeddedBoundary(q:vec3f)->vec3f{
  if(!hasEmbeddedBoundary()){return q;}let offset=q-p.boundaryCenter.xyz;
  let scaled=offset/max(p.boundaryRadii.xyz,vec3f(1e-6));let d2=dot(scaled,scaled);
  return select(q,p.boundaryCenter.xyz+offset*(0.9999/sqrt(d2)),d2>0.9998);
}
fn clipBoundarySegment(startInput:vec3f,candidate:vec3f)->vec3f{
  if(!hasEmbeddedBoundary()||insideEmbeddedBoundary(candidate)){return candidate;}
  let start=clampInsideEmbeddedBoundary(startInput);
  let radii=max(p.boundaryRadii.xyz,vec3f(1e-6));
  let origin=(start-p.boundaryCenter.xyz)/radii;let direction=(candidate-start)/radii;
  let a=dot(direction,direction);let b=2.0*dot(origin,direction);
  let c=dot(origin,origin)-1.0;let discriminant=max(0.0,b*b-4.0*a*c);
  let hit=select(0.0,(-b+sqrt(discriminant))/max(2.0*a,1e-12),a>1e-12);
  return mix(start,candidate,clamp(hit-1e-4,0.0,1.0));
}

${logicalOwnerEntries}
${packedLogicalOwnerEntries}
${transportExecutionImageEntries}
${transportPacketAuthorityEntries}
${transportProducerMaskEntries}
${finalScalarPacketMaskEntries}
${internedBoundaryEntries}
${implicitPressureInteriorEntries}
${iboTRAEntries}
${dirtyFaceRowMaskEntries}

fn brickDirectoryLookupLegacy(key:u32)->u32{
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let count=brickDimensions.x*brickDimensions.y*brickDimensions.z;
  if(key>=count){return INVALID;}
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let coordinate=vec3u(remainder-y*brickDimensions.x,y,z);
  let tableCapacity=p.topologyOffsets2.z-p.topologyOffsets2.y;
  let tableMask=tableCapacity-1u;
  let maximumSpanLog=topology[p.topologyOffsets2.w+1u]&31u;
  for(var spanLog=0u;spanLog<=maximumSpanLog;spanLog+=1u){
    let span=1u<<spanLog;let origin=(coordinate/span)*span;
    let originKey=origin.x+brickDimensions.x*(origin.y+brickDimensions.y*origin.z);
    var slot=(originKey*0x9e3779b1u)&tableMask;
    for(var probe=0u;probe<tableCapacity;probe+=1u){
      let brick=topology[p.topologyOffsets2.y+slot];if(brick==INVALID){break;}
      let candidateKey=topology[p.topologyOffsets2.z+2u*brick+1u];
      if(candidateKey==originKey){
        if(brick<p.dispatch.w&&brickSpan(brick)==span){return brick;}break;}
      slot=(slot+1u)&tableMask;
    }
  }
  return INVALID;
}

fn brickDirectoryLookup(key:u32)->u32{
  ${packedLogicalOwnerLookup}
  ${logicalOwnerLookup}
  return brickDirectoryLookupLegacy(key);
}

fn brickDirectoryLookupAtCoordinate(coordinate:vec3u)->u32{
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  if(any(coordinate>=brickDimensions)){return INVALID;}
  let key=coordinate.x+brickDimensions.x*(coordinate.y+brickDimensions.y*coordinate.z);
  return brickDirectoryLookup(key);
}

fn compactOwnerCellAt(q:vec3i)->vec3u{
  if(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))){return vec3u(INVALID);}
  let uq=vec3u(q);let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let queryCoordinate=uq/BRICK_FINE_RESOLUTION;
  let brick=brickDirectoryLookupAtCoordinate(queryCoordinate);
  if(brick==INVALID){return vec3u(INVALID);}
  let span=brickSpan(brick);
  let brickCoordinate=(queryCoordinate/span)*span;
  let resolution=acceptedBrickResolution(brick);
  let range=templateBrickCellRange(brick,resolution);let first=range.x;let count=range.y;
  let scale=BRICK_FINE_RESOLUTION*span/resolution;
  let local=(uq-brickCoordinate*BRICK_FINE_RESOLUTION)/scale;
  let origin=brickCoordinate*BRICK_FINE_RESOLUTION;
  let valid=(min(p.dimensions.xyz-origin+vec3u(scale-1u),
    vec3u(BRICK_FINE_RESOLUTION*span)))/scale;
  let cell=first+local.x+valid.x*(local.y+valid.y*local.z);
  return select(vec3u(INVALID),vec3u(cell,brick,resolution),cell<first+count);
}

fn ownerCellAt(q:vec3i)->u32{
  let owner=compactOwnerCellAt(q);if(owner.x==INVALID||!brickActive(owner.y)){return INVALID;}
  // compactOwnerCellAt already resolved the accepted brick and rung. Avoid
  // reloading the cell's brick, active bit, and accepted resolution inside
  // cellTransportActive: this helper sits under every characteristic corner.
  let available=cellResolution(owner.x)==owner.z&&cellOpenVolume(owner.x)>1e-8;
  return select(INVALID,owner.x,available);
}

fn presentationOwnerCellAt(q:vec3i)->u32{
  let owner=compactOwnerCellAt(q);if(owner.x==INVALID){return INVALID;}
  return select(INVALID,owner.x,
    (atomicLoad(&activity[activityRecord(owner.y)+1u])&64u)!=0u);
}

struct FaceVelocitySupport {
  velocity:vec3f,
  spans:vec3f,
  owner:bool,
  extended:bool,
  liquid:bool,
}
fn faceVelocitySupportAt(q:vec3i)->FaceVelocitySupport{
  if(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))){
    return FaceVelocitySupport(vec3f(0.0),vec3f(1.0),false,false,false);}
  let uq=vec3u(q);let index=uq.x+p.dimensions.x*(uq.y+p.dimensions.y*uq.z);
  let at=FACE_VELOCITY_SUPPORT+4u*index;
  let packed=state[at+3u];
  let flags=u32(round(8.0*fract(packed)));let span=max(1.0,floor(packed));
  return FaceVelocitySupport(vec3f(state[at],state[at+1u],state[at+2u]),
    vec3f(span),
    (flags&1u)!=0u,(flags&2u)!=0u,(flags&4u)!=0u);
}

// Face tracing deliberately retains a dense read cache: B8/P8 Dawn A/B showed
// that resolving a TEI owner at every RK2 corner costs 2.4-5.4x more than the
// cache's publish plus all traces. The cache is only a derived representation;
// its producer consumes the accepted TEI leaf and effective-velocity plane.
@compute @workgroup_size(256)
fn publishSparseCM12FaceVelocitySupport(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let brick=wid.x;if(brick>=p.dispatch.w){return;}
  let leaf=cm12TeiLoadLeaf(acceptedTopologySlot(),brick);
  if((leaf.flags&0x80000000u)==0u||leaf.scale==0u){return;}
  let origin=cm12TeiLogicalCoordinate(leaf.originKey)*BRICK_FINE_RESOLUTION;
  let extent=min(leaf.scale*leaf.valid,p.dimensions.xyz-min(origin,p.dimensions.xyz));
  let count=extent.x*extent.y*extent.z;
  for(var local=lane;local<count;local+=256u){let z=local/(extent.x*extent.y);
    let localRemainder=local-z*extent.x*extent.y;let y=localRemainder/extent.x;
    let x=localRemainder-y*extent.x;let q=origin+vec3u(x,y,z);
    let cellCoordinate=vec3u(x,y,z)/leaf.scale;
    let cell=leaf.first+cellCoordinate.x+leaf.valid.x
      *(cellCoordinate.y+leaf.valid.y*cellCoordinate.z);
    let value=cm12EffectiveTransportVelocity(cell);
    let extendedAddress=cm12TeiLeafLocalPacketAddress(
      brick,leaf.flags&31u,cellCoordinate);
    let lower=origin+leaf.scale*cellCoordinate;
    let widths=min(vec3u(leaf.scale),p.dimensions.xyz-lower);
    let span=f32(max(1u,min(widths.x,min(widths.y,widths.z))));
    let flags=1u|select(0u,2u,cm12ExtendedPacketLaneSelected(
      extendedAddress.x,extendedAddress.y))
      |select(0u,4u,state[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE);
    let index=q.x+p.dimensions.x*(q.y+p.dimensions.y*q.z);
    let at=FACE_VELOCITY_SUPPORT+4u*index;
    state[at]=value.x;state[at+1u]=value.y;state[at+2u]=value.z;
    state[at+3u]=span+f32(flags)/8.0;
  }
}

@compute @workgroup_size(256)
fn clearSparseCM12RetiredFaceVelocitySupport(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let brick=incrementalActivityBrickInvocation(wid.x);
  if(brick==INVALID||brickActive(brick)){return;}
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let key=topology[p.topologyOffsets2.z+2u*brick+1u];let xy=brickDimensions.x*brickDimensions.y;
  let bz=key/xy;let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;
  let origin=vec3u(bx,by,bz)*BRICK_FINE_RESOLUTION;
  let width=BRICK_FINE_RESOLUTION*brickSpan(brick);
  let extent=min(vec3u(width),p.dimensions.xyz-min(origin,p.dimensions.xyz));
  let count=extent.x*extent.y*extent.z;
  for(var local=lane;local<count;local+=256u){let z=local/(extent.x*extent.y);
    let localRemainder=local-z*extent.x*extent.y;let y=localRemainder/extent.x;
    let x=localRemainder-y*extent.x;let q=origin+vec3u(x,y,z);
    let index=q.x+p.dimensions.x*(q.y+p.dimensions.y*q.z);
    let at=FACE_VELOCITY_SUPPORT+4u*index;
    state[at]=0.0;state[at+1u]=0.0;state[at+2u]=0.0;state[at+3u]=0.0;
  }
}

fn sampleFaceVelocitySupport(position:vec3f)->vec3f{
  let q=vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4))));
  let probe=faceVelocitySupportAt(q);
  let spans=max(vec3f(1.0),probe.spans);
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result=vec3f(0.0);
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let value=faceVelocitySupportAt(vec3i(floor(lattice)));if(!value.owner){continue;}
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);
    result+=wx*wy*wz*value.velocity;
  }}}return result;
}

fn traceFaceDeparture(position:vec3f)->vec3f{
  let initialPosition=clampInsideEmbeddedBoundary(position);
  let initial=sampleFaceVelocitySupport(initialPosition);
  let substeps=clamp(i32(ceil(length(initial)*p.frame.x)),1,16);
  let subDt=p.frame.x/f32(substeps);var traced=initialPosition;
  let lower=vec3f(0.5);let upper=vec3f(p.dimensions.xyz)-vec3f(0.5);
  for(var step=0;step<substeps;step+=1){
    var first=initial;if(step>0){first=sampleFaceVelocitySupport(traced);}
    let midpoint=clipBoundarySegment(traced,
      clamp(traced-0.5*subDt*first,lower,upper));
    let candidate=traced-subDt*sampleFaceVelocitySupport(midpoint);
    traced=clipBoundarySegment(traced,clamp(candidate,lower,upper));
  }
  return traced;
}

fn presentationPhiAt(cell:u32,densityOffset:u32)->f32{
  let effective=state[densityOffset+cell]/max(cellOpenFraction(cell),1e-6);
  return (CM12_LIQUID_ISOVALUE-effective)*4.0*p.frame.y;
}
fn presentationPhi(cell:u32)->f32{return presentationPhiAt(cell,destinationDensity());}

// Restrict authoritative rho by finest-cell volume. A coarse presentation
// stencil can therefore cross a 2:1 seam without choosing an arbitrary fine
// child, and its sample has exactly the mass of that virtual coarse cell.
fn restrictedPresentationDensityAt(lower:vec3i,cellScale:i32,densityOffset:u32)->f32{
  // The common adaptive case asks for a virtual coarse cell that is already
  // represented by one coarse authority cell. Resolve that owner once instead
  // of repeating the same hash-table lookup for every finest child. Only a
  // genuinely finer owner needs the volume restriction below.
  let owner=compactOwnerCellAt(lower);
  if(owner.x!=INVALID
    &&(atomicLoad(&activity[activityRecord(owner.y)+1u])&64u)!=0u){
    let ownerScale=BRICK_FINE_RESOLUTION*brickSpan(owner.y)/owner.z;
    if(ownerScale>=u32(cellScale)){
      return state[densityOffset+owner.x]
        /max(cellOpenFraction(owner.x),1e-6);
    }
  }
  var rho=0.0;
  for(var dz=0;dz<cellScale;dz+=1){for(var dy=0;dy<cellScale;dy+=1){for(var dx=0;dx<cellScale;dx+=1){
    let cell=presentationOwnerCellAt(lower+vec3i(dx,dy,dz));
    if(cell!=INVALID){rho+=state[densityOffset+cell]
      /max(cellOpenFraction(cell),1e-6);}
  }}}
  return rho/f32(cellScale*cellScale*cellScale);
}
fn restrictedPresentationDensity(lower:vec3i,cellScale:i32)->f32{
  return restrictedPresentationDensityAt(lower,cellScale,destinationDensity());
}

// Shared compiled-topology sampler for transport, sharpening, and tracers. The
// expression and dz/dy/dx corner order retain the canonical interpolation.
fn sampleEffectiveTransportVelocity(position:vec3f)->vec3f{
  let probe=cm12TeiOwnerAtFine(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe.cell!=INVALID){spans=vec3f(probe.widths);}
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=vec3f(0.0);
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let owner=cm12TeiOwnerAtFine(vec3i(floor(lattice)));let cell=owner.cell;
    if(cell==INVALID){continue;}
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);
    result+=wx*wy*wz*cm12EffectiveTransportVelocity(cell).xyz;
  }}}return result;
}

fn traceEffectiveTransportCharacteristic(position:vec3f,direction:f32)->vec3f{
  let initialPosition=clampInsideEmbeddedBoundary(position);
  let initial=sampleEffectiveTransportVelocity(initialPosition);
  let substeps=clamp(i32(ceil(length(initial)*p.frame.x)),1,16);
  let subDt=p.frame.x/f32(substeps);var traced=initialPosition;
  let lower=vec3f(0.5);let upper=vec3f(p.dimensions.xyz)-vec3f(0.5);
  for(var step=0;step<substeps;step+=1){
    var first=initial;if(step>0){first=sampleEffectiveTransportVelocity(traced);}
    let midpoint=clipBoundarySegment(traced,
      clamp(traced+direction*0.5*subDt*first,lower,upper));
    let candidate=traced+direction*subDt*sampleEffectiveTransportVelocity(midpoint);
    traced=clipBoundarySegment(traced,clamp(candidate,lower,upper));
  }
  return traced;
}
fn traceEffectiveTransportDeparture(position:vec3f)->vec3f{
  return traceEffectiveTransportCharacteristic(position,-1.0);}
fn traceEffectiveTransportArrival(position:vec3f)->vec3f{
  return traceEffectiveTransportCharacteristic(position,1.0);}

struct TransportStencil{cells:array<u32,8>,weights:array<f32,8>}

${phase1TransportQAEntries}

// Sharpening samples every corner of the same adaptive trilinear stencil.
// Compute its invariant probe and lattice coordinates once while retaining the
// exact corner order and weight expression used by transport.
fn transportStencil(position:vec3f)->TransportStencil{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);
    spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  // Clamp to the local half-span and pull out-of-domain corners back inside,
  // as the CPU sampleWeights does. Without the clamp a cell wider than one
  // fine unit loses stencil weight at the wall (the outside corner resolves to
  // INVALID); gatherConservativeDensity renormalizes that away but
  // traceGammaAndBeta does not, so the sampled gamma collapses at coarse
  // wall cells. Identical to the old behaviour on width-1 cells.
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result:TransportStencil;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=clamp(vec3i(floor(spans*(vec3f(lower+offset)+vec3f(0.5)))),
      vec3i(0),vec3i(p.dimensions.xyz)-vec3i(1));
    let cell=ownerCellAt(lattice);
    let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
      *select(1.0-fraction.y,fraction.y,offset.y==1)
      *select(1.0-fraction.z,fraction.z,offset.z==1);
    result.cells[corner]=cell;
    result.weights[corner]=select(0.0,weight,cell!=INVALID);
  }
  return result;
}

// Same stencil geometry/order, but routed through the Phase-1 name so the
// packet transport kernels have a mechanically isolated access contract.
fn effectiveTransportStencil(position:vec3f)->TransportStencil{
  let probe=cm12TeiOwnerAtFine(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe.cell!=INVALID){spans=vec3f(probe.widths);}
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result:TransportStencil;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=clamp(vec3i(floor(spans*(vec3f(lower+offset)+vec3f(0.5)))),
      vec3i(0),vec3i(p.dimensions.xyz)-vec3i(1));
    let cell=cm12TeiOwnerAtFine(lattice).cell;
    let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
      *select(1.0-fraction.y,fraction.y,offset.y==1)
      *select(1.0-fraction.z,fraction.z,offset.z==1);
    result.cells[corner]=cell;
    result.weights[corner]=select(0.0,weight,cell!=INVALID);
  }
  return result;
}

${topologyEffectsEntries}

const TRANSPORT_CHARACTERISTIC_CLEARANCE:u32=${velocityExtensionLayouts
    ?.state.characteristicSupportFloatBase ?? 0}u;
const CM12_SPATIAL_TILES_PER_AXIS:u32=BRICK_FINE_RESOLUTION/4u;
// The three conservative physics passes use their independently compiled,
// deduplicated rung-packet domain. Lane zero snapshots the sealed packet once;
// hot lanes use mask + packet arithmetic with no owner or cell-open query.
fn cm12MassExecutionCell(rank:u32,lane:u32,family:u32)->u32{
  _=rank;_=family;return cm12TransportStagedExecutionCell(lane);
}
// Packet geometry every lane of a transport packet can read after the stage
// barriers: fine min corner of packet lane q is origin+scale*q.
var<workgroup>cm12TransportPacketOriginFine:vec3u;
fn stageSparseCM12TransportExecutionImage(rank:u32,lane:u32,family:u32){
  cm12StageTransportPacket(rank,lane,family);
    let profileCell=cm12TransportStagedExecutionCell(lane);
    if(EXP_TRANSPORT_PROFILE){
      if(lane==0u){atomicAdd(&activity[CM12_TRANSPORT_PROFILE_BASE+family],1u);}
      if(profileCell!=INVALID){atomicAdd(
        &activity[CM12_TRANSPORT_PROFILE_BASE+3u+family],1u);}
    }
    if(family==2u){return;}
    let candidate=cm12TransportStagedPacketOriginFine;
    let origin=select(vec3u(0u),candidate,candidate.x!=INVALID);
    if(lane==0u){cm12TransportPacketOriginFine=origin;}
  cm12TeiStageDirectory(origin,lane,cm12TransportStagedTopologySlot);
}

fn cm12ResidentTransportProducerMaskGeneration()->u32{
  return max(1u,cm12FCCandidateGeneration());}
fn cm12ResidentTransportProducerMaskPacketCount()->u32{
  return cm12TransportPacketCount(2u);}
fn cm12ResidentTransportProducerMaskPacket(workRank:u32)->u32{
  return cm12TransportPacketId(workRank,2u);}
fn cm12ResidentTransportProducerMaskCell(packet:u32,lane:u32)->u32{
  return cm12TeiPacketCell(packet,lane,acceptedTopologySlot());}
fn cm12ResidentTransportProducerMaskCellValid(cell:u32)->bool{
  return cell!=INVALID&&cell<p.counts.x&&cellActive(cell);}
fn cm12ResidentTransportProducerMaskIncidenceBegin(cell:u32)->u32{
  return incidenceBegin(cell);}
fn cm12ResidentTransportProducerMaskIncidenceEnd(cell:u32)->u32{
  return incidenceEnd(cell);}
fn cm12ResidentTransportProducerMaskIncidenceRow(incidence:u32)->u32{
  return incidenceRow(incidence);}
fn cm12ResidentTransportProducerMaskRowAccepted(row:u32)->bool{
  return rowAccepted(row);}
fn cm12ResidentTransportProducerMaskRowTermBegin(row:u32)->u32{
  return rowTermOffset(row);}
fn cm12ResidentTransportProducerMaskRowTermEnd(row:u32)->u32{
  return rowTermOffset(row)+rowTermCount(row);}
fn cm12ResidentTransportProducerMaskRowTermCell(term:u32)->u32{
  return termCell(term);}
fn cm12ResidentTransportProducerSharpeningMask(packet:u32,mask:vec2u){
  cm12TransportOverwritePacketMask(packet,0u,mask);}
fn sharpeningSourceCell(packetRank:u32,lane:u32)->u32{
  let packet=cm12TransportPacketId(packetRank,0u);
  let mask=cm12TransportPacketMask(packet,0u);
  if(packet==INVALID||!cm12TransportPacketLaneSelected(mask,lane)){return INVALID;}
  return cm12TeiPacketCell(packet,lane,acceptedTopologySlot());
}
fn sharpeningSourceCellCurrent(cell:u32)->bool{
  if(cell==INVALID||cell>=p.counts.x||!cellTransportActive(cell)){return false;}
  let rho=conditionedDensity(cell);
  return rho>0.0&&rho<=CM12_LIQUID_ISOVALUE;
}

fn addSharpeningReceipt(cell:u32,value:i32){
  if(cell>=p.counts.x||value==0){return;}
  atomicAdd(&conditioning[3u*p.counts.x+cell],value);
}
fn sharpeningReceipt(cell:u32)->f32{
  return f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
}

fn recordCandidateTopologyEffectsWork(brick:u32,validBrick:bool){
  if(!validBrick||brick>=p.dispatch.w){return;}let output=activityRecord(brick);
  if(!topologyPreparationScheduledAt(output)){return;}
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+13u]);
  let acceptedActive=brickActive(brick);let candidateActive=candidateBrickActive(brick);
  let oldState=accepted|select(0u,0x80000000u,acceptedActive);
  let newState=candidate|select(0u,0x80000000u,candidateActive);
  let cause=select(${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.resolutionChanged}u,
    select(${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.topologyRetired}u,
      ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.topologyCreated}u,candidateActive),
    acceptedActive!=candidateActive);
  _=tfxRecordPTRBrick(brick,oldState,newState,cause);
}

@compute @workgroup_size(1)
fn recordCandidateTopologyEffectsFromTopologyDelta(
 @builtin(workgroup_id)wid:vec3u){
  let brick=topologyDeltaLeafInvocation(wid.x);
  recordCandidateTopologyEffectsWork(select(0u,brick,brick!=INVALID),brick!=INVALID);
}

fn candidateInjectionAsymmetric()->bool{
  return p.injectionCenter.w!=0.0
    &&(abs(p.injectionCenter.x-0.5*f32(p.dimensions.x))>1e-6
      ||abs(p.injectionCenter.z-0.5*f32(p.dimensions.z))>1e-6
      ||abs(p.injectionRadius.x-p.injectionRadius.z)>1e-6);
}
@compute @workgroup_size(1) fn beginSparseCM12CandidateEffectsCensus(){
  // VEX2 recomputes from current fields after the accepted-slot flip.
}
@compute @workgroup_size(1) fn beginSparseCM12CandidateEffectsSemanticPreflight(){
}
@compute @workgroup_size(64) fn preflightSparseCM12CandidateInjectionEffects(
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  _=lane;_=wid;
}
@compute @workgroup_size(1) fn finalizeSparseCM12CandidateEffectsPreflight(){
}
@compute @workgroup_size(1) fn beginSparseCM12AuthorizedCandidateEffects(){
}
@compute @workgroup_size(1) fn finalizeSparseCM12AuthorizedCandidateEffects(){
  if(atomicLoad(&topologyArena[topologyWorklistBase()+3u])!=2u){return;}
  if(candidateInjectionAsymmetric()){cm12FCInvalidateD4Authorized(
      1u,0u,cm12FCEffectsGeneration(),cm12FCFrameSealed());}
  ptrSealPreflightedTopologyJournalNoFail(
    atomicLoad(&topologyArena[topologyWorklistBase()+1u]));
}

// Exact algebraic phase certificate for the final scalar banks. Returning 0/1
// means both physical rho banks contain exact dry/flooded capacity and both
// gamma banks contain exact one. Two is unresolved and can never certify clean.
fn cm12FinalScalarExactPhase(cell:u32)->u32{
  if(bitcast<u32>(cellOpenFraction(cell))!=0x3f800000u){return 2u;}
  let rho0=bitcast<u32>(state[p.stateOffsets0.x+cell]);
  let rho1=bitcast<u32>(state[p.stateOffsets0.y+cell]);
  let gamma0=bitcast<u32>(state[p.stateOffsets0.z+cell]);
  let gamma1=bitcast<u32>(state[p.stateOffsets0.w+cell]);
  if(gamma0!=0x3f800000u||gamma1!=0x3f800000u||rho0!=rho1){return 2u;}
  // Dormant capacity is canonical dry authority. Activation/topology writers
  // invalidate its tile explicitly, so rejecting it here only manufactures a
  // full dry-apron event frontier every frame.
  if(!cellActive(cell)){return select(2u,0u,rho0==0u);}
  if(rho0==0u){return 0u;}if(rho0==0x3f800000u){return 1u;}return 2u;
}
// Persistent interior-liquid certificate. The shared characteristic producer
// proves that both departure donors and arrival receivers lie wholly inside
// the authored bulk-liquid band and records a spatial clearance, in fine-cell
// units, around that support. Absolute speed is irrelevant: a fast steady
// trace is accepted when its actual swept support is deep. A velocity producer
// revokes it only when the bounded RK2 trajectory shift consumes that
// clearance; topology and scalar producers still revoke their closures.
fn cm12FinalScalarPersistentBulk(cell:u32)->bool{
  let rhoAfter=state[destinationDensity()+cell];
  let gammaAfter=state[destinationGamma()+cell];
  let saturatedTolerance=0.005;
  if(!cellTransportActive(cell)||hasRigidBodies()
    ||cellOpenFraction(cell)<1.0-1e-6
    ||state[TRANSPORT_CHARACTERISTIC_CLEARANCE+cell]<=0.0
    ||abs(rhoAfter-1.0)>saturatedTolerance
    ||abs(gammaAfter-1.0)>saturatedTolerance){return false;}
  return true;
}
fn cm12FinalScalarGeneration()->u32{return max(1u,cm12FCCandidateGeneration());}
fn cm12FinalScalarTopologyGeneration()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()]);}
fn cm12FinalScalarTopologySlot()->u32{return acceptedTopologySlot();}
fn cm12FinalScalarLeafPacketCount(leaf:u32,slot:u32)->u32{
  let descriptor=cm12TeiLoadLeaf(slot,leaf);
  if((descriptor.flags&0x80000000u)==0u||descriptor.scale==0u){return 0u;}
  let resolution=descriptor.flags&31u;let axis=max(1u,(resolution+3u)/4u);
  return axis*axis*axis;
}
fn cm12FinalScalarPacketCell(packet:u32,lane:u32,slot:u32)->u32{
  return cm12TeiPacketCell(packet,lane,slot);}
fn cm12FinalScalarPacketAddress(cell:u32)->vec2u{
  if(cell==INVALID||cell>=p.counts.x||!cellActive(cell)){return vec2u(INVALID);}
  // Consumers run under transport and pressure bind groups; binding 14 is TEI2
  // only in the former. Derive the stable TEI2 address from immutable cell
  // ownership instead of reading the execution image.
  let leafId=cellBrick(cell);let resolution=acceptedBrickResolution(leafId);
  let spanFine=BRICK_FINE_RESOLUTION*brickSpan(leafId);
  let scale=spanFine/resolution;
  let logicalDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let key=topology[p.topologyOffsets2.z+2u*leafId+1u];
  let xy=logicalDimensions.x*logicalDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/logicalDimensions.x;
  let bx=remainder-by*logicalDimensions.x;
  let origin=BRICK_FINE_RESOLUTION*vec3u(bx,by,bz);
  let minimum=cellMinimum(cell);if(any(minimum<origin)){return vec2u(INVALID);}
  let local=(minimum-origin)/scale;
  let packetAxis=max(1u,(resolution+3u)/4u);
  let packetLocal=(local.x>>2u)+packetAxis*((local.y>>2u)+packetAxis*(local.z>>2u));
  let lane=(local.x&3u)+4u*((local.y&3u)+4u*(local.z&3u));
  return vec2u(64u*leafId+packetLocal,lane);
}
fn cm12FinalScalarCellFacts(cell:u32)->vec4u{
  if(cell==INVALID||cell>=p.counts.x||!cellActive(cell)){return vec4u(0u);}
  let changed=bitcast<u32>(state[destinationDensity()+cell])
      !=bitcast<u32>(state[sourceDensity()+cell])
    ||bitcast<u32>(state[destinationGamma()+cell])
      !=bitcast<u32>(state[sourceGamma()+cell]);
  let nonexact=cm12FinalScalarExactPhase(cell)>1u;
  let bulk=cm12FinalScalarPersistentBulk(cell);
  let flip=pressureCellMembershipPredicate(cell)!=pcmCellContains(cell);
  if(bulk){
    // Value-bearing dead-bank mirror. It remains in finalization even though
    // every dirty carrier now consumes packet masks.
    state[sourceDensity()+cell]=state[destinationDensity()+cell];
    state[sourceGamma()+cell]=state[destinationGamma()+cell];
  }
  return vec4u(u32(changed),u32(nonexact),u32(bulk),u32(flip));
}
fn transportBeta(cell:u32)->f32{
  return f32(atomicLoad(&conditioning[cell]))/CM12_TRANSPORT_FIXED;
}
// traceGammaAndBeta and the later gather use the identical backward
// characteristic. Cache its adaptive stencil by stable cell ID: six packed
// u32 words hold eight 24-bit IDs, followed by the exact eight f32 weights.
// The scratch is dead pressure/candidate storage at this point in the frame.
fn massStencilCacheBase(cell:u32)->u32{return 14u*cell;}
fn massStencilPackedCell(cell:u32)->u32{
  return select(cell,0x00ffffffu,cell==INVALID);
}
fn storeMassStencil(base:u32,stencil:TransportStencil){
  var ids:array<u32,8>;
  for(var corner=0u;corner<8u;corner+=1u){
    ids[corner]=massStencilPackedCell(stencil.cells[corner]);
    candidateState[base+6u+corner]=stencil.weights[corner];
  }
  for(var group=0u;group<2u;group+=1u){let source=4u*group;let output=base+3u*group;
    candidateState[output]=bitcast<f32>(ids[source]|(ids[source+1u]<<24u));
    candidateState[output+1u]=bitcast<f32>((ids[source+1u]>>8u)|(ids[source+2u]<<16u));
    candidateState[output+2u]=bitcast<f32>((ids[source+2u]>>16u)|(ids[source+3u]<<8u));
  }
}
fn storeMassDepartureStencil(cell:u32,stencil:TransportStencil){
  storeMassStencil(massStencilCacheBase(cell),stencil);
}
fn massStencilCell(baseInput:u32,corner:u32)->u32{
  let base=baseInput+3u*(corner/4u);let local=corner&3u;
  let a=bitcast<u32>(candidateState[base]);
  let b=bitcast<u32>(candidateState[base+1u]);
  let c=bitcast<u32>(candidateState[base+2u]);var result=0u;
  if(local==0u){result=a&0x00ffffffu;}
  else if(local==1u){result=((a>>24u)|(b<<8u))&0x00ffffffu;}
  else if(local==2u){result=((b>>16u)|(c<<16u))&0x00ffffffu;}
  else{result=(c>>8u)&0x00ffffffu;}
  return select(result,INVALID,result==0x00ffffffu);
}
fn massDepartureStencilCell(cell:u32,corner:u32)->u32{
  return massStencilCell(massStencilCacheBase(cell),corner);
}
fn massDepartureStencilWeight(cell:u32,corner:u32)->f32{
  return candidateState[massStencilCacheBase(cell)+6u+corner];
}
fn transportCharacteristicStencilClearance(stencil:TransportStencil)->f32{
  var visible=0.0;var minimumWidth=1e30;
  let saturatedTolerance=0.005;
  for(var corner=0u;corner<8u;corner+=1u){let weight=stencil.weights[corner];
    visible+=weight;let cell=stencil.cells[corner];
    // Include zero-weight corners. They are the immediate support entered by
    // an arbitrarily small characteristic movement at a lattice plane, so a
    // weighted-corners-only receipt has no spatial validity radius there.
    if(cell==INVALID||!cellTransportActive(cell)
      ||cellOpenFraction(cell)<1.0-1e-6
      ||abs(state[sourceDensity()+cell]-1.0)>saturatedTolerance
      ||abs(state[sourceGamma()+cell]-1.0)>saturatedTolerance){return 0.0;}
    minimumWidth=min(minimumWidth,cellMinimumWidth(cell));
  }
  if(visible<0.999999){return 0.0;}
  // All eight corners are bulk liquid. Retain a small physical movement
  // budget inside that inspected support: two percent of the narrowest
  // adaptive cell admits roundoff-scale and quiescent projection updates but
  // makes an accelerating/deforming characteristic publish fresh work well
  // before it approaches a different interpolation neighbourhood.
  return 0.02*minimumWidth;
}
fn transportDepartureCharacteristicClearance(stencil:TransportStencil)->f32{
  let clearance=transportCharacteristicStencilClearance(stencil);
  if(clearance<=0.0){return 0.0;}
  var reference=vec3f(0.0);var haveReference=false;var maximumDelta=0.0;
  for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];
    if(cell==INVALID){return 0.0;}
    let at=destinationCellVelocity()+4u*cell;
    let velocity=vec3f(state[at],state[at+1u],state[at+2u]);
    if(!haveReference){reference=velocity;haveReference=true;}
    maximumDelta=max(maximumDelta,length(velocity-reference));
  }
  let supportingWidth=clearance/0.02;
  return select(0.0,clearance,
    p.frame.x*maximumDelta<=0.002*supportingWidth);
}
fn accumulateTransportBeta(rank:u32,cell:u32,value:i32){
  _=rank;atomicAdd(&conditioning[cell],value);
}
fn accumulateTransportDeficit(rank:u32,cell:u32,density:i32,gamma:i32){
  _=rank;atomicAdd(&conditioning[p.counts.x+cell],density);
  atomicAdd(&conditioning[2u*p.counts.x+cell],gamma);
}

fn finishTransportFaceRow(row:u32,characteristic:f32,touchesLiquid:bool){
  state[destinationFaceVelocity()+row]=select(characteristic,
    mix(characteristic,state[sourceFaceVelocity()+row],0.4),touchesLiquid);
  if(hasRigidBodies()){
    let open=rowOpenFraction(row);let fluid=state[destinationFaceVelocity()+row];
    state[destinationFaceVelocity()+row]=open*fluid+(1.0-open)*rowSolidVelocity(row);
  }
}
fn prepareTransportFaceRow(row:u32){
  if(rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  let axis=rowAxis(row);var normal=vec3f(0.0);normal[axis]=0.25*rowDistance(row);
  let negative=faceVelocitySupportAt(vec3i(floor(rowCenter(row)-normal)));
  let positive=faceVelocitySupportAt(vec3i(floor(rowCenter(row)+normal)));
  let touchesExtendedVelocity=negative.extended||positive.extended;
  let touchesLiquid=negative.liquid||positive.liquid;
  // Dry receiver rows still carry the extrapolated velocity band. The next
  // characteristic needs that velocity before any mass has reached the row;
  // treating rho==0 as inert pins and then releases a moving sparse front.
  // Beyond that eight-sweep validity band the interpolant is identically zero,
  // so those rows can still avoid an RK2 trace without changing a face value.
  if(!touchesExtendedVelocity){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  let departure=traceFaceDeparture(rowCenter(row));
  let characteristic=sampleFaceVelocitySupport(departure)[axis];
  finishTransportFaceRow(row,characteristic,touchesLiquid);
}
@compute @workgroup_size(64)
fn publishSparseCM12MovingSolidActivity(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||!hasRigidBodies()){return;}
  if(cellOpenFraction(cell)<0.999||dynamicallyCoveredCell(cell)){
    incrementalActivityMarkCellClosure(cell,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.movingSolidSweep}u);
  }
}

// Coverage of one accepted cell by the dropped ball, in the same partial-cell
// units the authored initial volumes seed. injectionCenter.w is the only
// enable: every ordinary frame writes zero there, so the two readers below cost
// one uniform compare outside a drop.
fn injectionCoverage(id:u32)->f32{
  let b=cellBase(id);
  let center=vec3f(taf(b),taf(b+1u),taf(b+2u));
  let q=(center-p.injectionCenter.xyz)/max(p.injectionRadius.xyz,vec3f(1e-6));
  let signed=length(q)-1.0;
  return clamp(0.5-signed*min(p.injectionRadius.x,
    min(p.injectionRadius.y,p.injectionRadius.z))/max(cellMinimumWidth(id),1e-6),0.0,1.0);
}

// A drop reaching a dormant brick is the same evidence as a free surface swept
// into it: the water is about to be there. activateSweptReceivers consumes
// this so a ball landing in the dry apron makes those bricks resident by the
// one existing activation path instead of vanishing into an inactive brick.
fn injectionReachesBrick(brick:u32)->bool{
  if(p.injectionCenter.w==0.0){return false;}
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;
  let lower=vec3f(vec3u(x,y,z)*BRICK_FINE_RESOLUTION);
  let upper=min(lower+vec3f(f32(BRICK_FINE_RESOLUTION*brickSpan(brick))),
    vec3f(p.dimensions.xyz));
  // Residency is conservative: use the ellipsoid's bounding box so a brick
  // sharing only a face/edge with the drop is promoted too. injectLiquid still
  // applies the exact smooth ellipsoid coverage and therefore writes no false
  // liquid into the conservative receiver shell.
  return all(p.injectionCenter.xyz+p.injectionRadius.xyz>=lower)
    &&all(p.injectionCenter.xyz-p.injectionRadius.xyz<=upper);
}

// Wetting walks each brick's accepted template range rather than the accepted
// cell worklist. The bricks this drop just activated are not in that worklist
// until the next commit folds them in, and a ball that has to wait a commit to
// exist is the vanishing drop again.
@compute @workgroup_size(64)
fn injectLiquid(@builtin(global_invocation_id)gid:vec3u){
  if(!sparseCM12TopologyLifecycleAccepted()){return;}
  let brick=gid.x;if(brick>=p.dispatch.w||!brickActive(brick)){return;}
  let range=templateBrickCellRange(brick,scheduledBrickResolution(brick));
  for(var at=0u;at<range.y;at+=1u){
    let id=range.x+at;if(cellOpenVolume(id)<=1e-8){continue;}
    let coverage=injectionCoverage(id);
    let clippedCoverage=coverage*cellOpenFraction(id);
    state[p.stateOffsets0.x+id]=max(state[p.stateOffsets0.x+id],clippedCoverage);
    state[p.stateOffsets0.y+id]=max(state[p.stateOffsets0.y+id],clippedCoverage);
    if(coverage>0.0){state[p.stateOffsets0.z+id]=1.0;state[p.stateOffsets0.w+id]=1.0;
    }
  }
}

// CM12 Sec. 3.4 steps 1-3. Backward-advect cumulative gamma, then scatter
// gamma_i w^-_li into each donor's volume-weighted beta column.
@compute @workgroup_size(64)
fn traceGammaAndBeta(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  stageSparseCM12TransportExecutionImage(wid.x,lane,0u);
  let id=cm12MassExecutionCell(wid.x,lane,0u);
  if(id!=INVALID){if(!cellTransportActive(id)){state[destinationGamma()+id]=1.0;
      state[TRANSPORT_CHARACTERISTIC_CLEARANCE+id]=0.0;}
    else{let b=cellBase(id);let center=vec3f(taf(b),taf(b+1u),taf(b+2u));
      var departure=vec3f(0.0);var stencil:TransportStencil;
      departure=traceEffectiveTransportDeparture(center);
      stencil=effectiveTransportStencil(departure);
      ${phase1QATraceCapture}
      state[TRANSPORT_CHARACTERISTIC_CLEARANCE+id]
        =transportDepartureCharacteristicClearance(stencil);
      storeMassDepartureStencil(id,stencil);
      var visible=0.0;var sampledGamma=0.0;
      for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];let weight=stencil.weights[corner];
        visible+=weight;if(cell!=INVALID){sampledGamma+=weight*state[sourceGamma()+cell];}}
      let advectedGamma=cm12ConditionedGamma(sampledGamma,visible);
      state[destinationGamma()+id]=advectedGamma;
      if(visible>1e-9){for(var corner=0u;corner<8u;corner+=1u){
        let cell=stencil.cells[corner];let weight=stencil.weights[corner];
        if(cell==INVALID||weight<=0.0){continue;}
        let coefficient=advectedGamma*weight/visible;
        let contribution=cm12VolumeWeightedBetaContribution(
          cellVolume(id),cellVolume(cell),coefficient);
        accumulateTransportBeta(wid.x,cell,i32(round(contribution*CM12_TRANSPORT_FIXED)));
      }}
    }
  }
  var sharpeningSource=false;if(id!=INVALID&&cellTransportActive(id)){
    let rho=state[sourceDensity()+id];
    sharpeningSource=rho>0.0&&rho<=CM12_LIQUID_ISOVALUE;
  }
  cm12TransportSharpeningMaskPublish(cm12TransportStagedPacketId,lane,id,
    sharpeningSource,cm12TeiStagedScaleAtFine(cm12TransportPacketOriginFine));
}

// CM12 Sec. 3.4 steps 6-7. Every deficient donor returns its missing column
// weight along the forward characteristic. The fixed-point scatters are
// deterministic and keep rho and gamma transfers paired.
@compute @workgroup_size(64)
fn scatterDensityDeficit(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  stageSparseCM12TransportExecutionImage(wid.x,lane,1u);
  let donor=cm12MassExecutionCell(wid.x,lane,1u);
  ${phase1QABetaCapture}
  if(donor!=INVALID&&cellTransportActive(donor)){
    let deficit=max(0.0,1.0-transportBeta(donor));
    if(deficit>1.0/CM12_TRANSPORT_FIXED){let b=cellBase(donor);
      var visible=0.0;var arrivalStencil:TransportStencil;
      let arrival=traceEffectiveTransportArrival(vec3f(taf(b),taf(b+1u),taf(b+2u)));
      arrivalStencil=effectiveTransportStencil(arrival);
      for(var corner=0u;corner<8u;corner+=1u){visible+=arrivalStencil.weights[corner];}
      if(visible<=1e-9){accumulateTransportDeficit(wid.x,donor,i32(round(
          state[sourceDensity()+donor]*deficit*CM12_TRANSPORT_FIXED)),i32(round(
          state[sourceGamma()+donor]*deficit*CM12_TRANSPORT_FIXED)));
      }else{for(var corner=0u;corner<8u;corner+=1u){
        var cell=INVALID;var weight=0.0;
        cell=arrivalStencil.cells[corner];weight=arrivalStencil.weights[corner];
        if(cell==INVALID||weight<=0.0){continue;}let normalized=weight/visible;
        let densityTransfer=cm12VolumeScaledDeficitTransfer(state[sourceDensity()+donor],
          cellVolume(donor),cellVolume(cell),deficit,normalized);
        let gammaTransfer=cm12VolumeScaledDeficitTransfer(state[sourceGamma()+donor],
          cellVolume(donor),cellVolume(cell),deficit,normalized);
        accumulateTransportDeficit(wid.x,cell,i32(round(densityTransfer*CM12_TRANSPORT_FIXED)),
          i32(round(gammaTransfer*CM12_TRANSPORT_FIXED)));
      }}
    }
  }
}

// CM12 Sec. 3.4 steps 4-5 plus the forward-deficit resolve. Scaling each
// donor by max(1,beta_l) clamps the column without materializing A.
@compute @workgroup_size(64)
fn gatherConservativeDensity(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  stageSparseCM12TransportExecutionImage(wid.x,lane,2u);
  let id=cm12MassExecutionCell(wid.x,lane,2u);
  var surfaceFeature=false;
  ${phase1QADeficitCapture}
  if(id!=INVALID){if(!cellActive(id)){
    state[destinationDensity()+id]=0.0;state[destinationGamma()+id]=1.0;
  }else if(!cellTransportActive(id)){
    if(dynamicallyCoveredCell(id)){
      state[destinationDensity()+id]=state[sourceDensity()+id];
      state[destinationGamma()+id]=state[sourceGamma()+id];
    }else{state[destinationDensity()+id]=0.0;state[destinationGamma()+id]=1.0;}
  }else{
  let advectedGamma=state[destinationGamma()+id];var visible=0.0;
  var rhoNext=0.0;var gammaNext=0.0;
  for(var corner=0u;corner<8u;corner+=1u){visible+=massDepartureStencilWeight(id,corner);}
  if(visible>1e-9){for(var corner=0u;corner<8u;corner+=1u){
    let cell=massDepartureStencilCell(id,corner);
    let weight=massDepartureStencilWeight(id,corner);if(cell==INVALID||weight<=0.0){continue;}
    let coefficient=cm12ConditionedRowCoefficient(
      advectedGamma,weight/visible,transportBeta(cell));
    rhoNext+=coefficient*state[sourceDensity()+cell];gammaNext+=coefficient;
  }}
  rhoNext+=f32(atomicLoad(&conditioning[p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  gammaNext+=f32(atomicLoad(&conditioning[2u*p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  if(rhoNext<CM12_DRY_CELL_THRESHOLD){gammaNext=1.0;}
  let nextDensity=max(0.0,rhoNext);
  state[destinationDensity()+id]=nextDensity;
  state[destinationGamma()+id]=max(0.0,gammaNext);
  surfaceFeature=hasRigidBodies()||cellOpenFraction(id)<1.0-1e-6
    ||state[sourceDensity()+id]<p.activityDensity.z
    ||state[destinationDensity()+id]<p.activityDensity.z;
  }}
  ${phase1QAMassCapture}
  cm12TransportProducerMaskPublish(
    cm12TransportStagedPacketId,lane,id,surfaceFeature);
}

// Massless markers riding the same characteristic the conservative transport
// integrates, so a coloured parcel and the mass it stands for cannot drift
// apart by construction: both call the shared compiled-topology characteristic
// against the same
// extrapolated transport velocity, in the same frame, before the density is
// touched. Where the dots and the mass disagree, the transport is wrong.
//
// A tracer stores only where it is. Its seed position — and therefore the
// colour it is drawn with — is a pure function of its invocation index, so the
// vertex stage recovers the colour from the instance index and no per-tracer
// colour is ever stored or advected. That is what keeps this a fixed
// vec4-per-tracer cost independent of grid resolution, and what keeps the
// spectrum perfectly sharp: nothing about the colour passes through an
// advection scheme, so it cannot be numerically diffused.
fn tracerCount()->u32{return p.tracerGrid.w;}
fn tracerBase(index:u32)->u32{return p.stateOffsets5.z+4u*index;}

fn tracerSeedPosition(index:u32)->vec3f{
  let dimensions=max(p.tracerGrid.xyz,vec3u(1u));
  let lattice=vec3u(index%dimensions.x,(index/dimensions.x)%dimensions.y,
    index/(dimensions.x*dimensions.y));
  return p.tracerOrigin.xyz+(vec3f(lattice)+vec3f(0.5))*p.tracerOrigin.w;
}

fn tracerCellAt(position:vec3f)->u32{
  return cm12TeiOwnerAtFine(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4))))).cell;
}

var<workgroup>cm12TracerDirectoryKey:u32;
var<workgroup>cm12TracerTopologySlot:u32;
fn stageTracerExecutionImage(position:vec3f,lane:u32){
  if(lane==0u){let fine=vec3u(floor(clamp(position,vec3f(0.0),
      vec3f(p.dimensions.xyz)-vec3f(1e-4))));
    cm12TracerDirectoryKey=cm12TeiLogicalKey(fine/BRICK_FINE_RESOLUTION);
    cm12TracerTopologySlot=acceptedTopologySlot();}
  let key=workgroupUniformLoad(&cm12TracerDirectoryKey);
  let slot=workgroupUniformLoad(&cm12TracerTopologySlot);
  cm12TeiStageDirectory(cm12TeiLogicalCoordinate(key)
    *BRICK_FINE_RESOLUTION,lane,slot);
}

// Seeding is re-runnable on purpose. Re-seeding mid-run re-reads the mixing
// from that instant rather than from t=0, which is the question being asked
// most of the time once a scene has already broken up.
@compute @workgroup_size(64)
fn seedTracers(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let index=WORKGROUP*wid.x+lane;
  stageTracerExecutionImage(tracerSeedPosition(WORKGROUP*wid.x),lane);
  if(index>=tracerCount()){return;}
  let position=tracerSeedPosition(index);
  let cell=tracerCellAt(position);
  // Markers seeded into air stay retired rather than being compacted away: a
  // dead lane costs one predicated early-out here and one collapsed quad in the
  // draw, which is cheaper than maintaining a compaction the topology would
  // invalidate every epoch anyway.
  let live=cell!=INVALID&&state[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE;
  let at=tracerBase(index);
  state[at]=position.x;state[at+1u]=position.y;state[at+2u]=position.z;
  state[at+3u]=select(0.0,1.0,live);
}

@compute @workgroup_size(64)
fn advanceTracers(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let index=WORKGROUP*wid.x+lane;var firstPosition=vec3f(0.0);
  if(lane==0u){let first=tracerBase(WORKGROUP*wid.x);
    firstPosition=vec3f(state[first],state[first+1u],state[first+2u]);}
  stageTracerExecutionImage(firstPosition,lane);
  if(index>=tracerCount()){return;}
  let at=tracerBase(index);if(state[at+3u]<0.5){return;}
  let traced=traceEffectiveTransportArrival(
    vec3f(state[at],state[at+1u],state[at+2u]));
  let cell=tracerCellAt(traced);
  // Retire on vacuum, not on the liquid isovalue. A marker inside a thin film
  // or a shedding sheet is still following fluid and is exactly what this view
  // exists to show; one stranded in an emptied cell would freeze in place and
  // read as motion that stopped.
  // Read the accepted density, not the destination: this pass runs before
  // conservative transport writes it, so the destination still holds the
  // previous frame's image and would retire live markers at random.
  let stranded=cell==INVALID||state[sourceDensity()+cell]<CM12_DRY_CELL_THRESHOLD;
  state[at]=traced.x;state[at+1u]=traced.y;state[at+2u]=traced.z;
  if(stranded){state[at+3u]=0.0;}
}

// ITR1 resolves each selected compact gamma-row bit directly. Receipts accumulate in
// two dead conservative-transport planes, so gamma needs no private pair CSR,
// pair generation plane, or reverse cell-pair incidence.
@compute @workgroup_size(64)
fn clearGammaReceipts(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  atomicStore(&conditioning[cell],0);atomicStore(&conditioning[p.counts.x+cell],0);
}
fn scatterGammaRow(row:u32,inputRho:u32,inputGamma:u32){
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var negativeCount=0u;var positiveCount=0u;
  for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
    negativeCount+=select(0u,1u,coefficient<0.0);
    positiveCount+=select(0u,1u,coefficient>0.0);}
  if(negativeCount==0u||positiveCount==0u){return;}
  let rowAreaValue=rowArea(row);
  if(rowAreaValue<=1e-8){return;}
  let pairArea=rowAreaValue/f32(negativeCount*positiveCount);
  // A regular interior cell participates in six rows. The CM12 flux already
  // contains its one-half diffusion coefficient, so /3 gives a convex
  // simultaneous six-neighbour update at the paper's full 1/30 s step.
  let scale=min(1.0,30.0*p.frame.x)/3.0;
  for(var negativeTerm=begin;negativeTerm<end;negativeTerm+=1u){
    if(termCoefficient(negativeTerm)>=0.0){continue;}
    let negative=termCell(negativeTerm);
    for(var positiveTerm=begin;positiveTerm<end;positiveTerm+=1u){
      if(termCoefficient(positiveTerm)<=0.0){continue;}
      let positive=termCell(positiveTerm);
      if(!cellTransportActive(negative)||!cellTransportActive(positive)){
        continue;
      }
      let conductedVolume=scale*min(pairArea*cellMinimumWidth(negative),
        pairArea*cellMinimumWidth(positive));
      let fluxIntoNegative=cm12GammaDiffusionFluxInto(
        state[inputRho+negative],state[inputGamma+negative],
        state[inputRho+positive],state[inputGamma+positive],
        conductedVolume/cellVolume(negative));
      let rhoReceipt=i32(round(fluxIntoNegative.x*cellVolume(negative)
        *CM12_TRANSPORT_FIXED));
      let gammaReceipt=i32(round(fluxIntoNegative.y*cellVolume(negative)
        *CM12_TRANSPORT_FIXED));
      atomicAdd(&conditioning[negative],rhoReceipt);
      atomicAdd(&conditioning[p.counts.x+negative],gammaReceipt);
      atomicAdd(&conditioning[positive],-rhoReceipt);
      atomicAdd(&conditioning[p.counts.x+positive],-gammaReceipt);
    }
  }
}

fn finalizeGammaCell(cell:u32,inputRho:u32,inputGamma:u32,
 outputRho:u32,outputGamma:u32){
  let ownRho=state[inputRho+cell];let ownGamma=state[inputGamma+cell];
  if(!cellTransportActive(cell)){
    if(dynamicallyCoveredCell(cell)){
      state[outputRho+cell]=ownRho;state[outputGamma+cell]=ownGamma;
    }else{state[outputRho+cell]=0.0;state[outputGamma+cell]=1.0;}
    return;
  }
  let inverseVolume=1.0/cellVolume(cell);
  let rhoReceipt=f32(atomicLoad(&conditioning[cell]))/CM12_TRANSPORT_FIXED;
  let gammaReceipt=f32(atomicLoad(&conditioning[p.counts.x+cell]))
    /CM12_TRANSPORT_FIXED;
  state[outputRho+cell]=ownRho+rhoReceipt*inverseVolume;
  state[outputGamma+cell]=ownGamma+gammaReceipt*inverseVolume;
}

@compute @workgroup_size(64)
fn finalizeGammaSnapshot(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID){
    finalizeGammaCell(cell,destinationDensity(),destinationGamma(),
      p.stateOffsets2.x,p.stateOffsets2.y);
  }
}

@compute @workgroup_size(64)
fn finalizeGammaRefinement(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID){
    finalizeGammaCell(cell,p.stateOffsets2.x,p.stateOffsets2.y,
      p.stateOffsets2.z,p.stateOffsets2.w);
  }
}

fn conditionedDensity(cell:u32)->f32{return state[p.stateOffsets2.z+cell];}
fn conditionedGamma(cell:u32)->f32{return state[p.stateOffsets2.w+cell];}

struct SharpeningStats {
  maximumDifference:f32,
  negativeArea:vec3f,
  positiveArea:vec3f,
  negativeDensity:vec3f,
  positiveDensity:vec3f,
  negativeDistance:vec3f,
  positiveDistance:vec3f,
}

// Expand each composite G row into the same physical scalar subfaces used by
// sparse-atlas-surface-conditioning.ts. This retains the paper's adjacent-cell
// stencil at 4^3/8^3 seams instead of treating an aggregate port as one cell.
fn sharpeningStats(cell:u32)->SharpeningStats{
  var result:SharpeningStats;let rho=conditionedDensity(cell);
  result.maximumDifference=0.0;result.negativeArea=vec3f(0.0);
  result.positiveArea=vec3f(0.0);result.negativeDensity=vec3f(0.0);
  result.positiveDensity=vec3f(0.0);result.negativeDistance=vec3f(0.0);
  result.positiveDistance=vec3f(0.0);
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);if(!rowAccepted(row)){continue;}
    let own=termCoefficient(incidenceTerm(at));
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    var negativeCount=0.0;var positiveCount=0.0;
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      negativeCount+=select(0.0,1.0,coefficient<0.0);
      positiveCount+=select(0.0,1.0,coefficient>0.0);}
    if(negativeCount==0.0||positiveCount==0.0){continue;}
    let area=rowArea(row)/(negativeCount*positiveCount);let distance=rowDistance(row);
    let axis=rowAxis(row);
    for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
      if(own*coefficient>=0.0){continue;}let neighbor=termCell(term);
      if(!cellActive(neighbor)){continue;}
      let neighborRho=conditionedDensity(neighbor);
      result.maximumDifference=max(result.maximumDifference,abs(rho-neighborRho));
      if(own<0.0){result.positiveArea[axis]+=area;
        result.positiveDensity[axis]+=area*neighborRho;
        result.positiveDistance[axis]+=area*distance;
      }else{result.negativeArea[axis]+=area;
        result.negativeDensity[axis]+=area*neighborRho;
        result.negativeDistance[axis]+=area*distance;}
    }
  }
  return result;
}

fn sharpeningDelta(cell:u32,stats:SharpeningStats)->f32{
  let rho=conditionedDensity(cell);
  // CM12 Eqs. 6-15 first integrate a unit-speed flux over a cell and then
  // divide the Godunov mass increment by cell volume. The resulting density
  // update is 3 dt |grad rho|, so distances stored in finest-cell units must
  // be converted back to metres. Multiplying by the local cell width here
  // made the old expression dimensionless and weakened sharpening by 1/h:
  // 10x on the all-coarse symmetric-expansion control and 20x when all fine.
  let pseudoTimeFineCells=3.0*p.frame.x/p.frame.y;
  var plusSquared=0.0;var minusSquared=0.0;
  for(var axis=0u;axis<3u;axis+=1u){
    var before=rho;var beforeDistance=1.0;
    if(stats.negativeArea[axis]>0.0){
      before=stats.negativeDensity[axis]/stats.negativeArea[axis];
      beforeDistance=stats.negativeDistance[axis]/stats.negativeArea[axis];
    }
    var after=rho;var afterDistance=1.0;
    if(stats.positiveArea[axis]>0.0){
      after=stats.positiveDensity[axis]/stats.positiveArea[axis];
      afterDistance=stats.positiveDistance[axis]/stats.positiveArea[axis];
    }
    let backward=-(rho-before)*pseudoTimeFineCells/beforeDistance;
    let forward=-(after-rho)*pseudoTimeFineCells/afterDistance;
    plusSquared+=max(max(backward,0.0)*max(backward,0.0),
      min(forward,0.0)*min(forward,0.0));
    minusSquared+=max(min(backward,0.0)*min(backward,0.0),
      max(forward,0.0)*max(forward,0.0));
  }
  let weight=cm12SharpeningWeight(rho,stats.maximumDifference);
  var delta=weight*sqrt(select(minusSquared,plusSquared,weight>=0.0));
  if(rho+delta<0.0||rho<CM12_DRY_CELL_THRESHOLD){delta=-rho;}else if(rho>0.5){delta=0.0;}
  return min(0.0,delta);
}

// Freeze the per-cell sharpening dose once. The trace itself differentiates
// the exact continuous adaptive density interpolant analytically below.
fn prepareSharpeningCell(cell:u32){
  if(!cellTransportActive(cell)){
    state[p.stateOffsets5.x+cell]=0.0;return;
  }
  let stats=sharpeningStats(cell);
  let rho=conditionedDensity(cell);
  state[p.stateOffsets5.x+cell]=select(sharpeningDelta(cell,stats),0.0,
    rho>CM12_LIQUID_ISOVALUE);
}
fn sampleSharpeningField(position:vec3f)->vec4f{
  let probe=cm12TeiOwnerAtFine(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe.cell!=INVALID){spans=vec3f(probe.widths);}
  // Same half-span clamp as transportStencil: an unclamped probe below the
  // first cell-centre plane drops its outside corner without renormalising,
  // biasing the sampled rho low and fabricating an away-from-wall gradient
  // component right where sharpening traces start. Clamping is the Neumann
  // extension: the through-wall gradient reads zero instead.
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result=vec4f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=clamp(vec3i(floor(spans*(vec3f(lower+offset)+vec3f(0.5)))),
      vec3i(0),vec3i(p.dimensions.xyz)-vec3i(1));
    let cell=cm12TeiOwnerAtFine(lattice).cell;if(cell==INVALID){continue;}
    let sx=select(-1.0,1.0,offset.x==1);let sy=select(-1.0,1.0,offset.y==1);
    let sz=select(-1.0,1.0,offset.z==1);
    let wx=select(1.0-fraction.x,fraction.x,offset.x==1);
    let wy=select(1.0-fraction.y,fraction.y,offset.y==1);
    let wz=select(1.0-fraction.z,fraction.z,offset.z==1);
    let rho=conditionedDensity(cell);
    result+=rho*vec4f(wx*wy*wz,sx*wy*wz/spans.x,
      wx*sy*wz/spans.y,wx*wy*sz/spans.z);
  }
  return result;
}

// CM12 Algorithm 2's TraceAlongField in composite-grid coordinates. As in the
// Uniform reference, half-cell forward-Euler substeps follow the frozen density
// gradient until rho=.5 or the configured paper-range D bound is reached. An invalid
// or inactive owner is the sparse equivalent of the paper's solid-stop rule.
fn traceSharpeningMass(source:u32)->vec3f{
  let b=cellBase(source);var position=vec3f(taf(b),taf(b+1u),taf(b+2u));
  let sourceWidth=cellMinimumWidth(source);let maximumDistance=p.sharpening.x*sourceWidth;
  var travelled=0.0;
  for(var step=0u;step<40u;step+=1u){
    if(step>=u32(p.sharpening.y)){break;}
    let field=sampleSharpeningField(position);
    if(field.x>=CM12_LIQUID_ISOVALUE||travelled>=maximumDistance){break;}
    let owner=cm12TeiOwnerAtFine(vec3i(floor(position))).cell;
    if(owner==INVALID){break;}
    let gradient=field.yzw;let magnitude=length(gradient);
    if(magnitude<1e-6){break;}
    let distance=min(0.5*cellMinimumWidth(owner),maximumDistance-travelled);
    let candidate=position+gradient/magnitude*distance;
    if(cm12TeiOwnerAtFine(vec3i(floor(candidate))).cell==INVALID){break;}
    position=candidate;travelled+=distance;
  }
  return position;
}

// CM12 Sec. 3.5, Eqs. 4-17 and Algorithm 2. Remove the air-side correction,
// trace along grad(rho), then scatter its integrated mass trilinearly at the
// traced point. Fixed-point atomics keep simultaneous GPU scatters additive.
fn scatterSharpeningCell(cell:u32){
  if(!cellTransportActive(cell)){state[p.stateOffsets5.x+cell]=0.0;return;}
  // Sec. 3.5 only removes density on the air side of the interface. Avoid
  // expanding the composite incidence stencil when sharpeningDelta's final
  // branches provably override the computed gradient. Preserve -rho for an
  // exact zero so even the scratch field's signed-zero behavior is unchanged.
  let rho=conditionedDensity(cell);
  if(rho==0.0){state[p.stateOffsets5.x+cell]=-rho;return;}
  if(rho>CM12_LIQUID_ISOVALUE){state[p.stateOffsets5.x+cell]=0.0;return;}
  let delta=state[p.stateOffsets5.x+cell];
  state[p.stateOffsets5.x+cell]=delta;if(delta>=0.0){return;}
  let removed=-delta*cellVolume(cell);let removedFixed=i32(round(removed*CM12_TRANSPORT_FIXED));
  let position=traceSharpeningMass(cell);let stencil=effectiveTransportStencil(position);
  var total=0.0;var lastCorner=INVALID;
  for(var corner=0u;corner<8u;corner+=1u){let targetCell=stencil.cells[corner];
    let weight=stencil.weights[corner];
    if(targetCell!=INVALID&&weight>0.0){total+=weight;lastCorner=corner;}}
  if(total<=1e-8){
    addSharpeningReceipt(cell,removedFixed);return;
  }
  var remainingFixed=removedFixed;
  for(var corner=0u;corner<8u;corner+=1u){let targetCell=stencil.cells[corner];
    let weight=stencil.weights[corner];if(targetCell==INVALID||weight<=0.0){continue;}
    var offeredFixed=i32(round(f32(removedFixed)*weight/total));
    if(corner==lastCorner){offeredFixed=remainingFixed;}
    else{offeredFixed=clamp(offeredFixed,0,remainingFixed);}
    addSharpeningReceipt(targetCell,offeredFixed);
    remainingFixed-=offeredFixed;
  }
}
// Preparation writes only this source's delta; scatter reads that same value
// and contributes through the separate fixed-point receipt plane. There is no
// cross-source dependency between the former dispatches.
@compute @workgroup_size(64)
fn sharpenAndScatterSparseCM12(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let packet=cm12TransportPacketId(wid.x,0u);
  let candidate=cm12TeiPacketFineOrigin(packet,acceptedTopologySlot());
  let origin=select(vec3u(0u),candidate,candidate.x!=INVALID);
  cm12TeiStageDirectory(origin,lane,acceptedTopologySlot());
  let cell=sharpeningSourceCell(wid.x,lane);
  if(sharpeningSourceCellCurrent(cell)){
    prepareSharpeningCell(cell);scatterSharpeningCell(cell);
  }
}
fn finalizeSharpeningCell(cell:u32){
  if(!cellTransportActive(cell)){
    if(!dynamicallyCoveredCell(cell)){
      state[destinationDensity()+cell]=0.0;state[destinationGamma()+cell]=1.0;
    }
    return;
  }
  let delta=state[p.stateOffsets5.x+cell];
  let incoming=sharpeningReceipt(cell);
  var density=max(0.0,conditionedDensity(cell)+delta+incoming/cellVolume(cell));
  let capacity=cellOpenFraction(cell);
  // Fixed-point receipts can leave a mathematically full hydrostatic cell one
  // quantisation unit below its exact capacity. Restore that endpoint exactly;
  // this changes no resolved interface value and prevents zero-flow topology
  // transfers from becoming visible density churn on the next frame.
  density=select(density,capacity,
    abs(density-capacity)<=1.0/CM12_TRANSPORT_FIXED);
  state[destinationDensity()+cell]=density;
  state[destinationGamma()+cell]=max(0.0,conditionedGamma(cell));
}
@compute @workgroup_size(64)
fn finalizeSharpening(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID){
    finalizeSharpeningCell(cell);}
}
@compute @workgroup_size(64)
fn clearSolidExcess(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  atomicStore(&conditioning[3u*p.counts.x+cell],0);
  state[p.stateOffsets5.x+cell]=0.0;
}

// Conservatively debit density beyond the cut-cell capacity and move that
// mass through open faces toward cells with a larger open fraction. Atomic
// fixed-point receipts preserve the global debit/credit identity.
@compute @workgroup_size(64)
fn scatterSolidExcess(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==INVALID||!cellTransportActive(cell)){return;}
  let rho=state[destinationDensity()+cell];let excessDensity=max(0.0,rho-cellOpenFraction(cell));
  let excessMass=excessDensity*cellVolume(cell);if(excessMass<=1e-9){return;}
  var totalSpare=0.0;var lastNeighbor=INVALID;
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){let row=incidenceRow(at);
    if(!rowAccepted(row)||rowArea(row)<=1e-8){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor==cell||!cellTransportActive(neighbor)
        ||cellOpenFraction(neighbor)<=cellOpenFraction(cell)){continue;}
      let spare=max(0.0,cellOpenFraction(neighbor)-state[destinationDensity()+neighbor])
        *cellVolume(neighbor);if(spare>0.0){totalSpare+=spare;lastNeighbor=neighbor;}
    }
  }
  if(totalSpare<=1e-9){return;}
  let movedMass=min(excessMass,totalSpare);
  let movedFixed=i32(round(movedMass*CM12_TRANSPORT_FIXED));var remaining=movedFixed;
  state[p.stateOffsets5.x+cell]=-f32(movedFixed)/(CM12_TRANSPORT_FIXED*cellVolume(cell));
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){let row=incidenceRow(at);
    if(!rowAccepted(row)||rowArea(row)<=1e-8){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor==cell||!cellTransportActive(neighbor)
        ||cellOpenFraction(neighbor)<=cellOpenFraction(cell)){continue;}
      let spare=max(0.0,cellOpenFraction(neighbor)-state[destinationDensity()+neighbor])
        *cellVolume(neighbor);if(spare<=0.0){continue;}
      var offered=i32(round(f32(movedFixed)*spare/totalSpare));
      if(neighbor==lastNeighbor){offered=remaining;}else{offered=clamp(offered,0,remaining);}
      atomicAdd(&conditioning[3u*p.counts.x+neighbor],offered);remaining-=offered;
    }
  }
}

@compute @workgroup_size(64)
fn finalizeSolidExcess(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellTransportActive(cell)){
    if(!dynamicallyCoveredCell(cell)){state[destinationDensity()+cell]=0.0;}
    return;
  }
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
  state[destinationDensity()+cell]=max(0.0,state[destinationDensity()+cell]
    +state[p.stateOffsets5.x+cell]+incoming/cellVolume(cell));
}

// The CPU sparse path retains a proven horizontal D4 invariant after surface
// conditioning. Quantizing the orbit sum before division makes that invariant
// bit-exact despite transformed cells visiting the same values in another
// floating-point order. This pass is encoded only while the topology and
// authored material are D4 symmetric.
@compute @workgroup_size(64)
fn preserveHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}let b=cellBase(cell);
  if(!cellActive(cell)){
    state[p.stateOffsets5.x+cell]=0.0;state[p.stateOffsets5.y+cell]=1.0;return;
  }
  let center=vec3f(taf(b),taf(b+1u),taf(b+2u));let extent=f32(p.dimensions.x);
  let xs=array<f32,8>(center.x,extent-center.x,center.x,extent-center.x,
    center.z,extent-center.z,center.z,extent-center.z);
  let zs=array<f32,8>(center.z,center.z,extent-center.z,extent-center.z,
    center.x,center.x,extent-center.x,extent-center.x);
  var rhoSum=0;var gammaSum=0;var count=0;
  for(var transform=0u;transform<8u;transform+=1u){
    let member=ownerCellAt(vec3i(i32(floor(xs[transform])),i32(floor(center.y)),
      i32(floor(zs[transform]))));
    if(member==INVALID){continue;}
    rhoSum+=i32(round(state[destinationDensity()+member]*CM12_TRANSPORT_FIXED));
    gammaSum+=i32(round(state[destinationGamma()+member]*CM12_TRANSPORT_FIXED));
    count+=1;
  }
  state[p.stateOffsets5.x+cell]=f32(rhoSum)/(f32(count)*CM12_TRANSPORT_FIXED);
  state[p.stateOffsets5.y+cell]=f32(gammaSum)/(f32(count)*CM12_TRANSPORT_FIXED);
}

@compute @workgroup_size(64)
fn commitHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellActive(cell)){return;}
  state[destinationDensity()+cell]=state[p.stateOffsets5.x+cell];
  state[destinationGamma()+cell]=state[p.stateOffsets5.y+cell];
}

fn publishForcedFace(row:u32,value:f32){
  let changed=bitcast<u32>(state[destinationFaceVelocity()+row])!=bitcast<u32>(value);
  state[destinationFaceVelocity()+row]=value;
  state[sourceFaceVelocity()+row]=value;
  if(changed){dfrm1MarkRow(row);}
}
@compute @workgroup_size(64)
fn forceFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  if(!rowAccepted(row)){publishForcedFace(row,0.0);return;}
  if(rowArea(row)<=1e-8){
    let boundary=select(0.0,rowSolidVelocity(row),hasRigidBodies());
    publishForcedFace(row,boundary);return;
  }
  let open=select(1.0,rowOpenFraction(row),hasRigidBodies());
  publishForcedFace(row,state[destinationFaceVelocity()+row]
    +open*p.frame.x*p.acceleration[rowAxis(row)]);
}

fn rawPressureDensity(cell:u32)->f32{
  return state[destinationDensity()+cell]/max(cellOpenFraction(cell),1e-6);
}

// As in the uniform cut-cell reference, a pressure sample centred in solid
// continues the nearest open liquid density and carries p >= 0.
fn pressureDensity(cell:u32)->f32{
  if(!cellSeparatingMinimum(cell)){return rawPressureDensity(cell);}
  var continued=0.0;
  for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
    let row=incidenceRow(at);if(!rowAccepted(row)){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let other=termCell(term);
      if(other!=cell&&cellTransportActive(other)){
        continued=max(continued,rawPressureDensity(other));
      }
    }
  }
  return continued;
}

// A cell beneath the surface cannot be air. With V==1 everywhere, the paper's
// Sec. 3.7 guard against false-air classification (rho' = rho/V, Eq. 20,
// "a cell with V < 0.5 will likely have rho < 0.5 causing the solver to treat
// it erroneously as air") is inert, yet conservative transport can still carry
// an interior cell's rho below the isovalue. Excluding such a cell opens a
// p = 0 Dirichlet hole underneath standing water; the repair jets through the
// surviving cut rows are the observed bottom-edge/corner velocity bursts. So
// a previous member whose every accepted-row neighbour was also a member --
// and that faces no sparse-air row (a one-term row is an open air port) --
// is submerged and stays in the solve regardless of its instantaneous rho.
fn pressureCellSubmerged(id:u32)->bool{
  if(!pcmCellContains(id)){return false;}
  var neighbours=0u;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);if(!rowAccepted(row)){continue;}
    let begin=rowTermOffset(row);let count=rowTermCount(row);
    if(count<2u){return false;}
    for(var term=begin;term<begin+count;term+=1u){let other=termCell(term);
      if(other==id){continue;}
      neighbours+=1u;
      if(!pcmCellContains(other)){return false;}
    }
  }
  return neighbours>0u;
}
fn pressureCellMembershipFromDensity(id:u32,rho:f32)->bool{
  return cellActive(id)
    &&(rho>=CM12_LIQUID_ISOVALUE||pressureCellSubmerged(id))
    &&(cellOpenVolume(id)>1e-8||cellSeparatingMinimum(id));
}
fn pressureCellMembershipPredicate(id:u32)->bool{
  return pressureCellMembershipFromDensity(id,pressureDensity(id));
}

fn classifyPressureCell(id:u32)->bool{
  let rho=pressureDensity(id);candidateState[pressureDensityCacheOffset()+id]=rho;
  let liquid=pressureCellMembershipFromDensity(id,rho);
  state[p.stateOffsets2.w+id]=select(0.0,1.0,liquid);
  if(!liquid){
    state[p.stateOffsets2.y+id]=0.0;state[p.stateOffsets2.z+id]=0.0;
    state[p.stateOffsets2.x+id]=0.0;
  }
  return liquid;
}
fn classifyPressureCellAndRecord(id:u32)->bool{
  let previous=pcmCellContains(id);let enabled=classifyPressureCell(id);
  if(previous!=enabled){_=pcfRecordCellMembershipEvent(id);}
  return enabled;
}

@compute @workgroup_size(64)
fn classifyPressureCells(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  _=pcmCellSetCandidate(id,classifyPressureCellAndRecord(id),1u,false);
}

@compute @workgroup_size(1)
fn beginCanonicalPressureCells(){_=pcmCellBegin(0u);}

@compute @workgroup_size(1)
fn beginCanonicalPressureRows(){_=pcmRowBegin(0u);}

@compute @workgroup_size(1)
fn planPressureMembershipEpoch(){
  let cellGroups=select(0u,acceptedTemplateCellWorkgroups(),pcmCellBootstrapEpoch());
  let rowGroups=select(0u,acceptedTemplateRowWorkgroups(),pcmRowBootstrapEpoch());
  let cellAt=PRESSURE_REPAIR_HEADER+PRESSURE_BOOTSTRAP_CELL_INDIRECT;
  let rowAt=PRESSURE_REPAIR_HEADER+PRESSURE_BOOTSTRAP_ROW_INDIRECT;
  fineSamples[cellAt]=cellGroups;fineSamples[cellAt+1u]=1u;fineSamples[cellAt+2u]=1u;
  fineSamples[rowAt]=rowGroups;fineSamples[rowAt+1u]=1u;fineSamples[rowAt+2u]=1u;
}

@compute @workgroup_size(64)
fn classifyDirtyPressureCells(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);
  if(id==INVALID||!fsm1FlipCell(id)){return;}
  _=pcmCellSetCandidate(id,classifyPressureCellAndRecord(id),2u,false);
}

fn pressureBrickStateAtResolution(brick:u32,resolution:u32)->u32{
  return resolution|select(0u,0x80000000u,brickActive(brick));
}

// PTR1 and PCF1 are binding-free and consume the resident's canonical
// PCM/state APIs. PCF1 lives in the non-aliased atomic topology-arena tail.
fn ptrFrameGeneration()->u32{return atomicLoad(&activity[0]);}
fn ptrTopologyGeneration()->u32{return atomicLoad(&topologyArena[topologyWorklistBase()]);}
fn ptrPCMCellCandidateGeneration()->u32{return pcmCellCandidateGeneration();}
fn ptrPCMRowCandidateGeneration()->u32{return pcmRowCandidateGeneration();}
fn ptrPCFCandidateGeneration()->u32{return pcfCandidateGeneration();}
fn ptrPCMCellAcceptedGeneration()->u32{return pcmCellAcceptedGeneration();}
fn ptrPCMRowAcceptedGeneration()->u32{return pcmRowAcceptedGeneration();}
fn ptrPCFAcceptedGeneration()->u32{return pcfAcceptedGeneration();}
fn ptrTopologyJournalCount()->u32{return 0u;}
fn ptrTopologyJournalRecord(at:u32)->vec4u{_=at;return vec4u(INVALID);}
fn ptrCellCapacity()->u32{return p.counts.x;}
fn ptrBrickCellRange(brick:u32,encoded:u32)->vec2u{
  if(encoded==INVALID||(encoded&0x80000000u)==0u){return vec2u(0u);}
  return templateBrickCellRange(brick,encoded&0x7fffffffu);
}
fn ptrCellIncidenceRange(cell:u32)->vec2u{
  return vec2u(incidenceBegin(cell),incidenceEnd(cell));
}
fn ptrIncidenceRow(at:u32)->u32{return incidenceRow(at);}
fn ptrApplyPressureCellClassification(cell:u32,current:bool)->bool{
  if(current){return classifyPressureCell(cell);}
  state[p.stateOffsets2.w+cell]=0.0;return false;
}
fn ptrClassifyPressureRow(row:u32)->bool{return classifyPressureRow(row);}
fn ptrPressureTheta(row:u32)->f32{return state[p.stateOffsets3.x+row];}

// HTP1 compatibility accessors preserve the resident template's exact stable
// IDs and packed arithmetic while PCF1 is migrated into the resident arena.
fn cm12HotHeaderValid()->bool{return true;}
fn cm12HotRowValid(row:u32)->bool{return row<p.counts.y;}
fn cm12HotRowTermCount(row:u32)->u32{return rowTermCount(row);}
fn cm12HotRowTermCell(row:u32,ordinal:u32)->u32{
  return termCell(rowTermOffset(row)+ordinal);
}
fn cm12HotRowTermCoefficient(row:u32,ordinal:u32)->f32{
  return termCoefficient(rowTermOffset(row)+ordinal);
}
fn cm12HotRowDualWeight(row:u32)->f32{return rowDualWeight(row);}
fn cm12HotIncidenceRange(cell:u32)->vec2u{
  let first=incidenceBegin(cell);return vec2u(first,incidenceEnd(cell)-first);
}
fn cm12HotIncidence(at:u32)->vec2u{
  let row=incidenceRow(at);return vec2u(row,incidenceTerm(at)-rowTermOffset(row));
}
fn cm12HotDirectedEdgeRange(cell:u32)->vec2u{
  let offsets=pressureTemplateWord(15u);let first=pressureTemplateWord(offsets+cell);
  return vec2u(first,pressureTemplateWord(offsets+cell+1u)-first);
}
fn cm12HotDirectedEdge(edge:u32)->vec3u{return vec3u(
  fineSamples[pressureNeighborOffset()+edge],
  pressureTemplateWord(pressureEdgeRows()+edge),
  pressureTemplateWord(pressureEdgeWeights()+edge));
}
fn pcfResidentSolidRowScale(row:u32)->f32{
  return select(1.0,state[p.solidOffsets.y+3u*row+2u],hasRigidBodies());
}
fn pcfTopologyGeneration()->u32{return ptrTopologyGeneration();}
fn pcfPCMGeneration()->u32{
  return max(pcmCellAcceptedGeneration(),pcmRowAcceptedGeneration());
}
fn pcfAggregateTopologyGeneration()->u32{return 1u;}
fn pcfCellBrick(cell:u32)->u32{return cellBrick(cell);}
fn pcfBrickCellRange(brick:u32)->vec2u{
  return templateBrickCellRange(brick,acceptedBrickResolution(brick));
}
fn pcfAggregateEdgeForFineEdge(edge:u32)->u32{
  return fineSamples[${pressureRepairLayout?.aggregateEdgeForFineEdgeBaseWords ?? 0}u+edge];
}
fn pcfAggregateEdgeContributionRange(edge:u32)->vec2u{
  let base=brickAggregateTopology();let records=base+4u+p.dispatch.w+1u;
  let record=records+3u*edge;return vec2u(pressureTemplateWord(record+1u),
    pressureTemplateWord(record+2u));
}
fn pcfAggregateEdgeContribution(at:u32)->u32{return pressureTemplateWord(at);}
fn pcfAggregateEdgeSourceBrick(edge:u32)->u32{
  return fineSamples[${pressureRepairLayout?.aggregateEdgeSourceBaseWords ?? 0}u+edge];
}
fn pcfHierarchyParent(level:u32,brick:u32)->u32{
  let descriptor=pressureHierarchyDescriptor(level);
  return pressureTemplateWord(pressureTemplateWord(descriptor+1u)+brick);
}
fn pcfHierarchyEdgeForAggregate(level:u32,aggregateEdge:u32)->u32{
  ${pressureRepairLayout?.hierarchyEdgeForAggregateBaseWords.map((base, level) =>
    `if(level==${level}u){return fineSamples[${base}u+aggregateEdge];}`).join("\n  ") ?? ""}
  return INVALID;
}
fn pcfHierarchyChildRange(level:u32,group:u32)->vec2u{
  let descriptor=pressureHierarchyDescriptor(level);let offsets=pressureTemplateWord(descriptor+2u);
  let first=pressureTemplateWord(offsets+group);
  return vec2u(first,pressureTemplateWord(offsets+group+1u)-first);
}
fn pcfHierarchyChild(level:u32,at:u32)->u32{
  return pressureTemplateWord(pressureTemplateWord(pressureHierarchyDescriptor(level)+3u)+at);
}
fn pcfHierarchyInternalEdgeRange(level:u32,group:u32)->vec2u{
  let descriptor=pressureHierarchyDescriptor(level);let offsets=pressureTemplateWord(descriptor+4u);
  let first=pressureTemplateWord(offsets+group);
  return vec2u(first,pressureTemplateWord(offsets+group+1u)-first);
}
fn pcfHierarchyInternalEdge(level:u32,at:u32)->u32{
  return pressureTemplateWord(pressureTemplateWord(pressureHierarchyDescriptor(level)+5u)+at);
}
fn pcfHierarchyEdgeContributionRange(level:u32,edge:u32)->vec2u{
  let records=pressureTemplateWord(pressureHierarchyDescriptor(level)+7u);
  let record=records+3u*edge;return vec2u(pressureTemplateWord(record+1u),
    pressureTemplateWord(record+2u));
}
fn pcfHierarchyEdgeContribution(level:u32,at:u32)->u32{_=level;return pressureTemplateWord(at);}
fn persistentBrickAggregateEdge(edge:u32)->f32{return select(bitcast<f32>(atomicLoad(
  &topologyArena[${persistentPressureCacheLayout?.brickAggregateEdgeBaseWords ?? 0}u+edge])),
  candidateState[brickAggregateEdgeWeightOffset()+edge],PRESSURE_EXECUTION_QA_ORACLE);}
fn persistentBrickAggregateDiagonal(brick:u32)->f32{return select(bitcast<f32>(atomicLoad(
  &topologyArena[${persistentPressureCacheLayout?.brickAggregateDiagonalBaseWords ?? 0}u+brick])),
  candidateState[brickAggregateDiagonalOffset()+brick],PRESSURE_EXECUTION_QA_ORACLE);}
fn persistentHierarchyEdge(level:u32,edge:u32)->f32{
  if(PRESSURE_EXECUTION_QA_ORACLE){return candidateState[
    pressureHierarchyEdgeWeightOffset(level)+edge];}
  ${persistentPressureCacheLayout?.hierarchyEdgeBaseWords.map((base, level) =>
    `if(level==${level}u){return bitcast<f32>(atomicLoad(&topologyArena[${base}u+edge]));}`)
    .join("\n  ") ?? ""}
  return 0.0;
}
fn persistentHierarchyDiagonal(level:u32,group:u32)->f32{
  if(PRESSURE_EXECUTION_QA_ORACLE){return candidateState[
    pressureHierarchyDiagonalOffset(level)+group];}
  ${persistentPressureCacheLayout?.hierarchyDiagonalBaseWords.map((base, level) =>
    `if(level==${level}u){return bitcast<f32>(atomicLoad(&topologyArena[${base}u+group]));}`)
    .join("\n  ") ?? ""}
  return 1e-12;
}
fn stablePressureBrickInvocation(invocation:u32)->u32{
  return select(INVALID,invocation,invocation<p.dispatch.w);
}
fn stablePressureHierarchyAddress(invocation:u32)->vec2u{
  return pressureHierarchyGroupAddress(invocation);
}

@compute @workgroup_size(1)
fn beginPersistentPressureCache(){_=pcfBegin(0u);}
@compute @workgroup_size(1)
fn finalizePersistentPressureCacheFrontier(){
  pcfCaptureConsumerGenerations();_=pcfFinalizeFrontier();
}

@compute @workgroup_size(1)
fn finalizeCanonicalPressureCellFrontier(){_=pcmCellFinalizeFrontier();}

@compute @workgroup_size(1)
fn finalizeCanonicalPressureCells(){
  _=pcmCellFinalize();
  fineSamples[PRESSURE_REPAIR_HEADER+PRESSURE_REPAIR_CELL_FIRST_FAULT]=pcmCellFirstFault();
  fineSamples[PRESSURE_REPAIR_HEADER+PRESSURE_REPAIR_FAULT]=pcmCellFault();
}

var<workgroup>pressurePrefix:array<u32,64>;
fn stablePressurePrefix(lane:u32,flag:u32)->u32{
  pressurePrefix[lane]=flag;workgroupBarrier();
  var width=1u;loop{
    if(width>=64u){break;}
    var add=0u;if(lane>=width){add=pressurePrefix[lane-width];}
    workgroupBarrier();pressurePrefix[lane]+=add;workgroupBarrier();width*=2u;
  }
  return pressurePrefix[lane]-flag;
}

fn classifyPressureRow(row:u32)->bool{
  // Retirement may deactivate a brick after the accepted list was published;
  // keep this dynamic fence even though physical row topology is immutable.
  if(!rowAccepted(row)){state[p.stateOffsets3.x+row]=0.0;return false;}
  let requirements=rowRequirementOffset(row);let requirementCount=ta(requirements);
  var sameLevel=requirementCount>0u;var allCoarse=requirementCount>0u;
  var firstResolution=0u;
  for(var requirement=0u;requirement<requirementCount;requirement+=1u){
    let resolution=ta(requirements+1u+requirement)&TEMPLATE_CELL_RESOLUTION_MASK;
    if(requirement==0u){firstResolution=resolution;}
    else{sameLevel=sameLevel&&resolution==firstResolution;}
    allCoarse=allCoarse&&resolution<BRICK_FINE_RESOLUTION;
  }
  if(rowKind(row)==2u){atomicAdd(&activity[ACCEPTED_MIXED_ROW_COUNT],1u);}
  else if(sameLevel&&allCoarse){atomicAdd(&activity[ACCEPTED_COARSE_ROW_COUNT],1u);}
  if(rowDualWeight(row)<=1e-8){state[p.stateOffsets3.x+row]=0.0;return false;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var liquidCount=0u;var airCount=0u;var liquidPhiSum=0.0;var liquidWeight=0.0;
  var airPhiSum=0.0;var airWeight=0.0;
  for(var at=begin;at<end;at+=1u){let cell=termCell(at);let w=abs(termCoefficient(at));
    let phi=CM12_LIQUID_ISOVALUE-pressureDensity(cell);
    let liquid=pcmCellContains(cell);
    if(liquid){liquidCount+=1u;liquidPhiSum+=w*phi;liquidWeight+=w;}
    else{airCount+=1u;airPhiSum+=w*phi;airWeight+=w;}}
  if(liquidCount==0u){state[p.stateOffsets3.x+row]=0.0;return false;}
  if(rowKind(row)==3u){let w=liquidWeight;airPhiSum+=w*rowExteriorPhi(row);airWeight+=w;}
  let cut=airCount>0u||rowKind(row)==3u;
  let theta=select(1.0,cm12GhostFluidTheta(liquidPhiSum/max(liquidWeight,1e-9),
    airPhiSum/max(airWeight,1e-9),1e-12),cut);
  state[p.stateOffsets3.x+row]=theta;
  atomicAdd(&activity[PRESSURE_ACTIVE_ROW_COUNT],1u);
  return true;
}

@compute @workgroup_size(64)
fn classifyRows(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  let enabled=classifyPressureRow(row);_=pcfStoreThetaAndRecord(row,
    state[p.stateOffsets3.x+row]);_=pcmRowSetCandidate(row,enabled,1u,false);
}

@compute @workgroup_size(64)
fn classifyDirtyPressureRows(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);var changed=false;
  for(var term=begin;term<end;term+=1u){
    changed=changed||fsm1ChangedOrFlipCell(termCell(term));}
  if(!changed){return;}
  let enabled=classifyPressureRow(row);
  // Scalar-driven membership or ghost-fluid theta changes occur after the
  // dirty-brick row mask was compiled. Publish the row directly into the same
  // production projection mask; pressure bits are allowed to remain identical.
  dfrm1MarkRow(row);_=pcfStoreThetaAndRecord(row,
    state[p.stateOffsets3.x+row]);_=pcmRowSetCandidate(row,enabled,2u,false);
}

@compute @workgroup_size(1)
fn finalizeCanonicalPressureRowFrontier(){_=pcmRowFinalizeFrontier();}

@compute @workgroup_size(1)
fn finalizeCanonicalPressureRows(){
  _=pcmRowFinalize();
  atomicStore(&activity[PRESSURE_ACTIVE_ROW_COUNT],pcmRowAcceptedCount());
  fineSamples[PRESSURE_REPAIR_HEADER+PRESSURE_REPAIR_ROW_FIRST_FAULT]=pcmRowFirstFault();
  let rowFault=pcmRowFault();if(rowFault!=0u){
    fineSamples[PRESSURE_REPAIR_HEADER+PRESSURE_REPAIR_FAULT]=0x10000u|rowFault;
  }
}

// Theta and any live rigid scaling are constant throughout one pressure
// epoch. candidateState is transient pressure scratch and every current
// pressure member refreshes its directed edges before the solve.
fn effectivePressureEdgeWeight(edge:u32)->f32{
  return pcfEdgeWeight(edge);
}

// One coarse scalar per resident brick supplies a sparse additive coarse
// space. The Galerkin diagonal is the sum of fine diagonals plus every directed
// internal off-diagonal, so internal fluxes cancel and only aggregate-boundary
// and Dirichlet stiffness remains. This is topology-linear and uses the same
// path for every scene size.
@compute @workgroup_size(64)
fn bakeBrickAggregateDiagonal(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;var diagonal=0.0;var wet=0.0;
  if(brick<p.dispatch.w&&brickActive(brick)){
    let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    let edgeOffsets=pressureTemplateWord(15u);
    for(var local=lid.x;local<range.y;local+=64u){let cell=range.x+local;
      if(!pcmCellContains(cell)){continue;}
      wet=1.0;
      diagonal+=state[p.stateOffsets2.z+cell];
      let end=pressureTemplateWord(edgeOffsets+cell+1u);
      for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
        let other=fineSamples[pressureNeighborOffset()+edge];
        if(cellBrick(other)==brick){diagonal+=effectivePressureEdgeWeight(edge);}
      }
    }
  }
  reduceA[lid.x]=diagonal;reduceB[lid.x]=wet;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];
      reduceB[lid.x]+=reduceB[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&brick<p.dispatch.w){
    candidateState[brickAggregateDiagonalOffset()+brick]=max(reduceA[0],1e-12);
    var packed=0u;if(reduceB[0]>0.0){
      let resolution=acceptedBrickResolution(brick);
      let range=templateBrickCellRange(brick,resolution);
      packed=((range.x+1u)<<3u)|(templateLevelIndex(resolution)+1u);
    }
    candidateState[brickAggregateRangeOffset()+brick]=bitcast<f32>(packed);
  }
}

fn bakePressureHierarchyDiagonalWork(lid:vec3u,wid:vec3u,level:u32){
  let group=wid.x;let count=pressureHierarchyGroupCount(level);var diagonal=0.0;
  let descriptor=pressureHierarchyDescriptor(level);
  let childOffsets=pressureTemplateWord(descriptor+2u);
  let children=pressureTemplateWord(descriptor+3u);
  let internalOffsets=pressureTemplateWord(descriptor+4u);
  let internalEdges=pressureTemplateWord(descriptor+5u);
  if(group<count){
    let childBegin=pressureTemplateWord(childOffsets+group);
    let childEnd=pressureTemplateWord(childOffsets+group+1u);
    for(var at=childBegin+lid.x;at<childEnd;at+=64u){
      let brick=pressureTemplateWord(children+at);
      diagonal+=persistentBrickAggregateDiagonal(brick);
    }
    let edgeBegin=pressureTemplateWord(internalOffsets+group);
    let edgeEnd=pressureTemplateWord(internalOffsets+group+1u);
    for(var at=edgeBegin+lid.x;at<edgeEnd;at+=64u){
      let edge=pressureTemplateWord(internalEdges+at);
      diagonal+=persistentBrickAggregateEdge(edge);
    }
  }
  reduceA[lid.x]=diagonal;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&group<count){
    candidateState[pressureHierarchyDiagonalOffset(level)+group]=max(reduceA[0],1e-12);
  }
}

fn bakePressureHierarchyEdgesWork(gid:vec3u,level:u32){
  let edge=gid.x;if(edge>=pressureHierarchyEdgeCount(level)){return;}
  let descriptor=pressureHierarchyDescriptor(level);
  let records=pressureTemplateWord(descriptor+7u);
  let record=records+3u*edge;
  let contributionBegin=pressureTemplateWord(record+1u);
  let contributionEnd=contributionBegin+pressureTemplateWord(record+2u);
  var weight=0.0;
  for(var at=contributionBegin;at<contributionEnd;at+=1u){
    let coarseEdge=pressureTemplateWord(at);
    weight+=candidateState[brickAggregateEdgeWeightOffset()+coarseEdge];
  }
  candidateState[pressureHierarchyEdgeWeightOffset(level)+edge]=weight;
}

fn restrictPressureHierarchyResidualWork(lid:vec3u,wid:vec3u,level:u32){
  let group=wid.x;let count=pressureHierarchyGroupCount(level);var residual=0.0;
  let descriptor=pressureHierarchyDescriptor(level);
  let childOffsets=pressureTemplateWord(descriptor+2u);
  let children=pressureTemplateWord(descriptor+3u);
  if(group<count){let begin=pressureTemplateWord(childOffsets+group);
    let end=pressureTemplateWord(childOffsets+group+1u);
    for(var at=begin+lid.x;at<end;at+=64u){
      residual+=candidateState[brickAggregateRhsOffset()+pressureTemplateWord(children+at)];
    }
  }
  reduceA[lid.x]=residual;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&group<count){
    let diagonal=persistentHierarchyDiagonal(level,group);
    candidateState[pressureHierarchyRhsOffset(level)+group]=reduceA[0];
    candidateState[pressureHierarchyAOffset(level)+group]
      =select(0.0,2.8573742*reduceA[0]/diagonal,diagonal>1e-12);
  }
}

fn refinePressureHierarchyCorrectionWork(group:u32,level:u32){
  let count=pressureHierarchyGroupCount(level);var offDiagonal=0.0;
  let descriptor=pressureHierarchyDescriptor(level);
  let offsets=pressureTemplateWord(descriptor+6u);
  let records=pressureTemplateWord(descriptor+7u);
  if(group<count){
    let edgeBegin=pressureTemplateWord(offsets+group);
    let edgeEnd=pressureTemplateWord(offsets+group+1u);
    for(var edge=edgeBegin;edge<edgeEnd;edge+=1u){
      let otherGroup=pressureTemplateWord(records+3u*edge);
      offDiagonal+=persistentHierarchyEdge(level,edge)
        *candidateState[pressureHierarchyAOffset(level)+otherGroup];
    }
  }
  if(group<count){
    let diagonal=persistentHierarchyDiagonal(level,group);
    let current=candidateState[pressureHierarchyAOffset(level)+group];
    let rhs=candidateState[pressureHierarchyRhsOffset(level)+group];
    let residual=rhs-(diagonal*current+offDiagonal);
    candidateState[pressureHierarchyBOffset(level)+group]
      =current+select(0.0,0.5821642*residual/diagonal,diagonal>1e-12);
  }
}

@compute @workgroup_size(64)
fn bakePressureHierarchyEdges(@builtin(global_invocation_id)gid:vec3u){
  let address=pressureHierarchyEdgeAddress(gid.x);if(address.x==INVALID){return;}
  bakePressureHierarchyEdgesWork(vec3u(address.y,0u,0u),address.x);
}
@compute @workgroup_size(64)
fn bakePressureHierarchyDiagonal(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let address=pressureHierarchyGroupAddress(wid.x);
  bakePressureHierarchyDiagonalWork(lid,vec3u(address.y,0u,0u),address.x);
}
@compute @workgroup_size(64)
fn restrictPressureHierarchyResidual(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let address=stablePressureHierarchyAddress(wid.x);
  restrictPressureHierarchyResidualWork(lid,vec3u(address.y,0u,0u),address.x);
}
@compute @workgroup_size(64)
fn refinePressureHierarchyCorrection(@builtin(global_invocation_id)gid:vec3u){
  let address=stablePressureHierarchyAddress(gid.x);
  if(address.x==INVALID){return;}
  refinePressureHierarchyCorrectionWork(address.y,address.x);
}

@compute @workgroup_size(64)
fn combinePressureHierarchyCorrection(@builtin(global_invocation_id)gid:vec3u){
  let brick=stablePressureBrickInvocation(gid.x);
  if(brick==INVALID){return;}var correction=0.0;
  for(var level=0u;level<pressureHierarchyLevelCount();level+=1u){
    let parents=pressureTemplateWord(pressureHierarchyDescriptor(level)+1u);
    let parent=pressureTemplateWord(parents+brick);
    correction+=exp2(-f32(level+1u))
      *candidateState[pressureHierarchyBOffset(level)+parent];
  }
  candidateState[brickAggregateBOffset()+brick]+=correction;
}

@compute @workgroup_size(64)
fn restrictBrickAggregateResidual(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=stablePressureBrickInvocation(wid.x);var residual=0.0;
  if(brick!=INVALID){
    let range=cachedPressureBrickRange(brick);
    for(var local=lid.x;local<range.y;local+=64u){let cell=range.x+local;
      if(pcmCellContains(cell)){residual+=state[p.stateOffsets3.y+cell];}
    }
  }
  reduceA[lid.x]=residual;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&brick!=INVALID){
    let diagonal=persistentBrickAggregateDiagonal(brick);
    candidateState[brickAggregateRhsOffset()+brick]=reduceA[0];
    candidateState[brickAggregateAOffset()+brick]
      =select(0.0,2.8573742*reduceA[0]/diagonal,diagonal>1e-12);
  }
}

@compute @workgroup_size(64)
fn bakeBrickAggregateEdges(@builtin(global_invocation_id)gid:vec3u){
  let coarseEdge=gid.x;if(coarseEdge>=brickAggregateEdgeCount()){return;}
  let topologyBase=brickAggregateTopology();
  let recordBase=topologyBase+4u+p.dispatch.w+1u;
  let record=recordBase+3u*coarseEdge;
  let contributionBegin=pressureTemplateWord(record+1u);
  let contributionCount=pressureTemplateWord(record+2u);var weight=0.0;
  for(var at=0u;at<contributionCount;at+=1u){
    weight+=effectivePressureEdgeWeight(pressureTemplateWord(contributionBegin+at));
  }
  candidateState[brickAggregateEdgeWeightOffset()+coarseEdge]=weight;
}

fn refineBrickAggregateCorrectionWork(brick:u32,sourceOffset:u32,
 destinationOffset:u32,weight:f32){
  var offDiagonal=0.0;
  if(brick!=INVALID&&cachedPressureBrickRange(brick).y!=0u){
    let topologyBase=brickAggregateTopology();let offsets=topologyBase+4u;
    let recordBase=offsets+p.dispatch.w+1u;
    let begin=pressureTemplateWord(offsets+brick);let end=pressureTemplateWord(offsets+brick+1u);
    for(var coarseEdge=begin;coarseEdge<end;coarseEdge+=1u){
      let otherBrick=pressureTemplateWord(recordBase+3u*coarseEdge);
      offDiagonal+=persistentBrickAggregateEdge(coarseEdge)
        *candidateState[sourceOffset+otherBrick];
    }
  }
  if(brick!=INVALID){
    let diagonal=persistentBrickAggregateDiagonal(brick);
    let current=candidateState[sourceOffset+brick];
    let rhs=candidateState[brickAggregateRhsOffset()+brick];
    let residual=rhs-(diagonal*current+offDiagonal);
    candidateState[destinationOffset+brick]
      =current+select(0.0,weight*residual/diagonal,diagonal>1e-12);
  }
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB1(@builtin(global_invocation_id)gid:vec3u){
  refineBrickAggregateCorrectionWork(stablePressureBrickInvocation(gid.x),brickAggregateAOffset(),
    brickAggregateBOffset(),0.5821642);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA2(@builtin(global_invocation_id)gid:vec3u){
  refineBrickAggregateCorrectionWork(stablePressureBrickInvocation(gid.x),brickAggregateBOffset(),
    brickAggregateAOffset(),0.53435177);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB3(@builtin(global_invocation_id)gid:vec3u){
  refineBrickAggregateCorrectionWork(stablePressureBrickInvocation(gid.x),brickAggregateAOffset(),
    brickAggregateBOffset(),1.23564);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA4(@builtin(global_invocation_id)gid:vec3u){
  refineBrickAggregateCorrectionWork(stablePressureBrickInvocation(gid.x),brickAggregateBOffset(),
    brickAggregateAOffset(),0.83447);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB5(@builtin(global_invocation_id)gid:vec3u){
  refineBrickAggregateCorrectionWork(stablePressureBrickInvocation(gid.x),brickAggregateAOffset(),
    brickAggregateBOffset(),0.64192);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA6(@builtin(global_invocation_id)gid:vec3u){
  refineBrickAggregateCorrectionWork(stablePressureBrickInvocation(gid.x),brickAggregateBOffset(),
    brickAggregateAOffset(),0.54557);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB7(@builtin(global_invocation_id)gid:vec3u){
  refineBrickAggregateCorrectionWork(stablePressureBrickInvocation(gid.x),brickAggregateAOffset(),
    brickAggregateBOffset(),0.50458);
}

fn brickAggregatePreconditioned(cell:u32)->f32{
  let diagonal=state[p.stateOffsets2.z+cell];let residual=state[p.stateOffsets3.y+cell];
  let local=select(0.0,residual/diagonal,diagonal>0.0);let brick=cellBrick(cell);
  return local+candidateState[brickAggregateBOffset()+brick];
}

@compute @workgroup_size(64)
fn applyBrickAggregatePreconditioner(@builtin(global_invocation_id)gid:vec3u){
  let cell=pressureCellInvocation(gid.x);if(cell==INVALID){return;}
  state[p.stateOffsets3.z+cell]=brickAggregatePreconditioned(cell);
}

@compute @workgroup_size(64)
fn initializeBrickAggregateDirection(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let cell=pressureCellInvocation(gid.x);var gamma=0.0;var rhs2=0.0;
  if(cell!=INVALID){let z=brickAggregatePreconditioned(cell);
    state[p.stateOffsets3.z+cell]=z;state[p.stateOffsets3.w+cell]=z;
    let residual=state[p.stateOffsets3.y+cell];gamma=residual*z;
    let rhs=state[p.stateOffsets2.y+cell];rhs2=rhs*rhs;
  }
  reducePair(lid.x,wid.x,gamma,rhs2);
}

fn applyPressureEdge(value:f32,edge:u32,inputOffset:u32,other:u32)->f32{
  let weight=effectivePressureEdgeWeight(edge);
  if(weight==0.0){return value;}
  return value+weight*state[inputOffset+other];
}
fn applyOperator(cell:u32,inputOffset:u32)->f32{
  // The pressure epoch has already classified rows and assembled the exact
  // diagonal. Physical topology packing expands every off-diagonal once.
  let edgeOffsets=pressureTemplateWord(15u);
  var result=state[p.stateOffsets2.z+cell]*state[inputOffset+cell];
  let begin=pressureTemplateWord(edgeOffsets+cell);
  let end=pressureTemplateWord(edgeOffsets+cell+1u);
  let strides=pressureImplicitInteriorStrides(cell);
  if(strides.x!=INVALID&&end-begin==6u){
    result=applyPressureEdge(result,begin,inputOffset,cell-strides.x);
    result=applyPressureEdge(result,begin+1u,inputOffset,cell+strides.x);
    result=applyPressureEdge(result,begin+2u,inputOffset,cell-strides.y);
    result=applyPressureEdge(result,begin+3u,inputOffset,cell+strides.y);
    result=applyPressureEdge(result,begin+4u,inputOffset,cell-strides.z);
    result=applyPressureEdge(result,begin+5u,inputOffset,cell+strides.z);
    return result;
  }
  for(var edge=begin;edge<end;edge+=1u){
    let weight=effectivePressureEdgeWeight(edge);if(weight==0.0){continue;}
    let other=fineSamples[pressureNeighborOffset()+edge];
    result+=weight*state[inputOffset+other];
  }
  return result;
}
@compute @workgroup_size(64)
fn preparePressure(@builtin(global_invocation_id)gid:vec3u){
  let id=pressureCellInvocation(gid.x);if(id==INVALID){return;}
  let rho=pressureDensity(id);
  var rhs=0.0;let diagonal=pcfDiagonal(id);
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);let theta=state[p.stateOffsets3.x+row];
    if(theta<=0.0){continue;}
    let coefficient=termCoefficient(incidenceTerm(at));
    let fluxWeight=select(rowDualWeight(row),rowStaticDualWeight(row),hasRigidBodies());
    rhs+=coefficient*fluxWeight*state[destinationFaceVelocity()+row];
  }
  // Sec. 3.7's source only corrects rho' > 1. The dual defect -- a submerged
  // member holding rho' < 1 -- is unreachable by any paper mechanism (the
  // projection conserves member mass; sharpening removes only from rho < 0.5
  // cells and deposits at the iso-contour), so a pool smeared by a violent
  // phase would stay under-dense forever, parked just above the isovalue
  // where membership flicker opens pressure holes. Apply the paper's own
  // rate with the opposite sign, gated to cells with no air neighbour so
  // free-surface and splash cells are never touched; it self-extinguishes
  // at rho' = 1.
  var targetDivergence=select(cm12VolumeCorrectionDivergence(
    rho,p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
  if(!cellSeparatingMinimum(id)&&rho<1.0&&pressureCellSubmerged(id)){
    targetDivergence=-min(CM12_VOLUME_CORRECTION_LAMBDA*(1.0-rho),
      CM12_VOLUME_CORRECTION_ETA)/(p.frame.y*cellMinimumWidth(id));
  }
  let controlVolume=select(cellOpenVolume(id),cellVolume(id),cellSeparatingMinimum(id));
  state[p.stateOffsets2.y+id]=rhs+controlVolume*targetDivergence;
  state[p.stateOffsets2.z+id]=diagonal;
}

// Construction-only full coefficient oracle. It reproduces the legacy stable
// PCM traversal into the QA instance's persistent edge arena without touching
// PCF work headers or producer receipts. Production never dispatches it and
// has no selector or full-work fallback path.
@compute @workgroup_size(64)
fn preparePressureFullOracle(@builtin(global_invocation_id)gid:vec3u){
  let id=pressureCellInvocation(gid.x);if(id==INVALID){return;}
  let edgeOffsets=pressureTemplateWord(15u);
  let edgeEnd=pressureTemplateWord(edgeOffsets+id+1u);
  for(var edge=pressureTemplateWord(edgeOffsets+id);edge<edgeEnd;edge+=1u){
    let row=pressureTemplateWord(pressureEdgeRows()+edge);
    let theta=state[p.stateOffsets3.x+row];let other=fineSamples[pressureNeighborOffset()+edge];
    var weight=0.0;
    if(theta>0.0&&pcmCellContains(other)){
      weight=bitcast<f32>(pressureTemplateWord(pressureEdgeWeights()+edge))/theta;
      if(hasRigidBodies()){weight*=state[p.solidOffsets.y+3u*row+2u];}
    }
    pcfStoreEdgeWeight(edge,weight);
  }
  var diagonal=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);let theta=state[p.stateOffsets3.x+row];
    if(theta<=0.0||!pcmRowContains(row)){continue;}
    let coefficient=termCoefficient(incidenceTerm(at));
    diagonal+=rowDualWeight(row)*coefficient*coefficient/theta;
  }
  let rho=pressureDensity(id);var rhs=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){
    let row=incidenceRow(at);let theta=state[p.stateOffsets3.x+row];
    if(theta<=0.0){continue;}let coefficient=termCoefficient(incidenceTerm(at));
    let fluxWeight=select(rowDualWeight(row),rowStaticDualWeight(row),hasRigidBodies());
    rhs+=coefficient*fluxWeight*state[destinationFaceVelocity()+row];
  }
  // Same submerged under-density dual as preparePressure, so the QA oracle
  // and the production path assemble the same right-hand side.
  var targetDivergence=select(cm12VolumeCorrectionDivergence(
    rho,p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
  if(!cellSeparatingMinimum(id)&&rho<1.0&&pressureCellSubmerged(id)){
    targetDivergence=-min(CM12_VOLUME_CORRECTION_LAMBDA*(1.0-rho),
      CM12_VOLUME_CORRECTION_ETA)/(p.frame.y*cellMinimumWidth(id));
  }
  let controlVolume=select(cellOpenVolume(id),cellVolume(id),cellSeparatingMinimum(id));
  state[p.stateOffsets2.y+id]=rhs+controlVolume*targetDivergence;
  state[p.stateOffsets2.z+id]=diagonal;
}

var<workgroup>reduceA:array<f32,64>;
var<workgroup>reduceB:array<f32,64>;
var<workgroup>reduceC:array<f32,64>;
// Activity is reduced as one record per lane rather than eleven separate
// planes.
var<workgroup>activityMoments:array<vec4i,64>;
var<workgroup>activityMetrics:array<vec4f,64>;
var<workgroup>activityMasks:array<vec2u,64>;
var<workgroup>transferMassBefore:array<f32,64>;
var<workgroup>transferMassAfter:array<f32,64>;
var<workgroup>transferGammaDelta:array<f32,64>;
var<workgroup>transferGammaScale:array<f32,64>;
var<workgroup>transferMomentumXDelta:array<f32,64>;
var<workgroup>transferMomentumYDelta:array<f32,64>;
var<workgroup>transferMomentumZDelta:array<f32,64>;
var<workgroup>transferMomentumXScale:array<f32,64>;
var<workgroup>transferMomentumYScale:array<f32,64>;
var<workgroup>transferMomentumZScale:array<f32,64>;
var<workgroup>candidateCellScheduled:u32;
var<workgroup>candidateCellConstructionActivation:u32;
var<workgroup>candidateFaceScheduled:u32;
var<workgroup>candidateFaceConstructionActivation:u32;
var<workgroup>candidateFaceActive:u32;
var<workgroup>candidateFaceAcceptedResolution:u32;
var<workgroup>candidateFaceResolution:u32;
var<workgroup>candidatePublicationScheduled:u32;
var<workgroup>candidatePublicationConstructionActivation:u32;
var<workgroup>candidatePublicationAccepted:u32;
var<workgroup>candidatePublicationResolution:u32;
var<workgroup>candidatePublicationAcceptedActive:u32;
var<workgroup>candidatePublicationActive:u32;
// A presentation page has at most eight source-cell steps between its first
// and last sample on any supported native lattice. Cache that small stencil
// once per workgroup instead of reconstructing the same eight corners for all
// 64 output samples independently.
var<workgroup>presentationDensityCache:array<f32,${presentationCacheCapacity}>;
fn reducePair(lane:u32,group:u32,a:f32,b:f32){
  reduceA[lane]=a;reduceB[lane]=b;workgroupBarrier();
  var width=32u;loop{if(lane<width){reduceA[lane]+=reduceA[lane+width];reduceB[lane]+=reduceB[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u){partials[group]=vec4f(reduceA[0],reduceB[0],0.0,0.0);}
}

@compute @workgroup_size(64)
fn initializePCG(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,
  @builtin(workgroup_id)wid:vec3u){
  let id=pressureCellInvocation(gid.x);
  var rz=0.0;var rhs2=0.0;
  if(id!=INVALID){
    let image=applyOperator(id,p.stateOffsets2.x);
    let residual=state[p.stateOffsets2.y+id]-image;
    let diagonal=state[p.stateOffsets2.z+id];let z=select(0.0,residual/diagonal,diagonal>0.0);
    state[p.stateOffsets3.y+id]=residual;state[p.stateOffsets3.z+id]=z;
    state[p.stateOffsets3.w+id]=z;rz=residual*z;
    let rhs=state[p.stateOffsets2.y+id];rhs2=rhs*rhs;
  }
  reducePair(lid.x,wid.x,rz,rhs2);
}

@compute @workgroup_size(64)
fn reduceInitialize(@builtin(local_invocation_id)lid:vec3u){
  var a=0.0;var b=0.0;for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){a+=partials[at].x;b+=partials[at].y;}
  reduceA[lid.x]=a;reduceB[lid.x]=b;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];reduceB[lid.x]+=reduceB[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[0]=reduceA[0];scalars[1]=reduceB[0];scalars[2]=0.0;
    scalars[3]=0.0;scalars[4]=0.0;scalars[5]=1.0;
    // 8/9 initial true residual; 10/11 final true residual; 12 executed
    // iterations; 13 first tolerance crossing; 14 curvature breakdown;
    // 16 recursive/true ratio; 17 material residual-drift flag; 18 recovered
    // curvature collapses.
    for(var at=8u;at<19u;at+=1u){scalars[at]=0.0;}scalars[13]=-1.0;}
}

// Chronopoulos-Gear PCG retains the composite operator and applies the sparse
// brick/root hierarchy while reducing the two globally synchronized dot
// products in each ordinary iteration to one packed reduction. stateOffsets5.x is surface
// scratch whose lifetime ended before pressure; it carries w=A z here.
@compute @workgroup_size(64)
fn initializePipelinedImage(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let enabled=scalars[5]>0.5;let cell=pressureCellInvocation(gid.x);
  var delta=0.0;
  if(enabled&&cell!=INVALID){
    let z=state[p.stateOffsets3.z+cell];
    let image=applyOperator(cell,p.stateOffsets3.z);
    state[p.stateOffsets5.x+cell]=image;state[p.stateOffsets4.x+cell]=image;
    delta=z*image;
  }
  reducePair(lid.x,wid.x,delta,0.0);
}

@compute @workgroup_size(64)
fn reducePipelinedInitialize(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5;var delta=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){delta+=partials[at].x;}}
  reduceA[lid.x]=delta;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){if(reduceA[0]>1e-20){scalars[2]=scalars[0]/reduceA[0];}
    else{scalars[2]=0.0;scalars[14]=1.0;scalars[18]+=1.0;}}
}

@compute @workgroup_size(64)
fn updatePipelinedState(@builtin(global_invocation_id)gid:vec3u){
  if(!pipelinedPressureActive()){return;}
  let cell=pressureCellInvocation(gid.x);if(cell==INVALID){return;}
  if(scalars[12]>0.0){let beta=scalars[3];
    state[p.stateOffsets3.w+cell]=state[p.stateOffsets3.z+cell]
      +beta*state[p.stateOffsets3.w+cell];
    state[p.stateOffsets4.x+cell]=state[p.stateOffsets5.x+cell]
      +beta*state[p.stateOffsets4.x+cell];
  }
  let alpha=scalars[2];state[p.stateOffsets2.x+cell]+=alpha*state[p.stateOffsets3.w+cell];
  let residual=state[p.stateOffsets3.y+cell]-alpha*state[p.stateOffsets4.x+cell];
  let diagonal=state[p.stateOffsets2.z+cell];state[p.stateOffsets3.y+cell]=residual;
  state[p.stateOffsets3.z+cell]=select(0.0,residual/diagonal,diagonal>0.0);
}

@compute @workgroup_size(64)
fn applyPipelinedImage(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,
  @builtin(workgroup_id)wid:vec3u){
  let enabled=pipelinedPressureActive();let cell=pressureCellInvocation(gid.x);
  var gamma=0.0;var delta=0.0;var residual2=0.0;
  if(enabled&&cell!=INVALID){
    let residual=state[p.stateOffsets3.y+cell];let z=state[p.stateOffsets3.z+cell];
    let image=applyOperator(cell,p.stateOffsets3.z);
    state[p.stateOffsets5.x+cell]=image;
    gamma=residual*z;delta=image*z;residual2=residual*residual;
  }
  reduceA[lid.x]=gamma;reduceB[lid.x]=delta;reduceC[lid.x]=residual2;
  workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];
      reduceB[lid.x]+=reduceB[lid.x+width];reduceC[lid.x]+=reduceC[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u){partials[wid.x]=vec4f(reduceA[0],reduceB[0],reduceC[0],0.0);}
}

@compute @workgroup_size(64)
fn reducePipelinedIteration(@builtin(local_invocation_id)lid:vec3u){
  let enabled=pipelinedPressureActive();var gamma=0.0;var delta=0.0;var residual2=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){
    gamma+=partials[at].x;delta+=partials[at].y;residual2+=partials[at].z;
  }}
  reduceA[lid.x]=gamma;reduceB[lid.x]=delta;reduceC[lid.x]=residual2;
  workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];
      reduceB[lid.x]+=reduceB[lid.x+width];reduceC[lid.x]+=reduceC[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){
    let previousGamma=scalars[0];let previousAlpha=scalars[2];
    let beta=select(0.0,reduceA[0]/previousGamma,previousGamma>1e-20);
    let denominator=reduceB[0]-beta*reduceA[0]/max(previousAlpha,1e-20);
    scalars[3]=beta;scalars[0]=reduceA[0];scalars[4]=reduceC[0];
    scalars[12]+=1.0;
    if(denominator>1e-20){scalars[2]=reduceA[0]/denominator;}
    else{scalars[2]=0.0;scalars[14]=1.0;scalars[18]+=1.0;}
  }
}

// One journal record per *encoded* iteration.
//
// Deliberately ungated: the solve encodes a fixed ceiling and the residual gate
// zeroes the tail, so a gated kernel would stop recording exactly where the
// interesting thing — the gate closing — happens. This runs every encoded
// iteration and stores the gate rather than obeying it, which is what lets the
// film distinguish "converged at 43" from "ran 128 times".
//
// The cursor is the encoded iteration index because this kernel is dispatched
// exactly once per encoded iteration, in order, on a queue with an implicit
// barrier between dispatches.
@compute @workgroup_size(64)
fn journalIteration(@builtin(local_invocation_id)lid:vec3u){
  if(lid.x!=0u||!journalArmed()){return;}
  let base=journalBase();
  let cursor=u32(max(0.0,state[base]));
  if(cursor>=journalIterationCapacity()){return;}
  var snapshot=-1.0;
  if(JOURNAL_SNAPSHOT){
    let slot=u32(max(0.0,state[base+1u]));
    if(slot<journalSnapshotCapacity()){snapshot=f32(slot);state[base+1u]=f32(slot+1u);}
  }
  let at=base+JOURNAL_HEADER_FLOATS+cursor*JOURNAL_ITERATION_FLOATS;
  state[at]=select(0.0,1.0,pipelinedPressureActive());
  state[at+1u]=scalars[0];
  state[at+2u]=scalars[2];
  state[at+3u]=scalars[3];
  state[at+4u]=scalars[4];
  state[at+5u]=scalars[1];
  state[at+6u]=scalars[12];
  state[at+7u]=scalars[14];
  state[at+8u]=scalars[10];
  state[at+9u]=scalars[11];
  state[at+10u]=scalars[18];
  state[at+11u]=scalars[13];
  state[at+12u]=scalars[8];
  state[at+13u]=scalars[9];
  state[at+14u]=scalars[16];
  state[at+15u]=snapshot;
  state[base]=f32(cursor+1u);
  state[base+2u]=1.0;
  state[base+3u]=1.0;
}

// A whole-field snapshot of the four cell fields the picture is made of.
//
// Runs over accepted cells rather than the compacted pressure worklist, so a
// dry cell inside the topology reads as an explicit zero instead of keeping
// whatever the previous capture left there. The slot is the snapshot cursor the
// paired journalIteration dispatch just advanced.
@compute @workgroup_size(64)
fn journalSnapshot(@builtin(global_invocation_id)gid:vec3u){
  if(!journalArmed()){return;}
  let base=journalBase();
  let cursorValue=state[base+1u];
  if(cursorValue<0.5){return;}
  let slot=u32(cursorValue)-1u;
  if(slot>=journalSnapshotCapacity()){return;}
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==INVALID||cell>=journalCellStride()){return;}
  let live=pcmCellContains(cell);
  state[journalSnapshotField(slot,0u)+cell]=
    select(0.0,state[p.stateOffsets2.x+cell],live);
  state[journalSnapshotField(slot,1u)+cell]=
    select(0.0,state[p.stateOffsets3.y+cell],live);
  state[journalSnapshotField(slot,2u)+cell]=
    select(0.0,state[p.stateOffsets3.z+cell],live);
  state[journalSnapshotField(slot,3u)+cell]=
    select(0.0,state[p.stateOffsets3.w+cell],live);
}

@compute @workgroup_size(64)
fn applyPipelinedRecovery(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,
  @builtin(workgroup_id)wid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;
  let cell=pressureCellInvocation(gid.x);var delta=0.0;
  if(enabled&&cell!=INVALID){
    let z=state[p.stateOffsets3.z+cell];
    let image=applyOperator(cell,p.stateOffsets3.z);
    state[p.stateOffsets5.x+cell]=image;state[p.stateOffsets4.x+cell]=image;
    delta=z*image;
  }
  reducePair(lid.x,wid.x,delta,0.0);
}

@compute @workgroup_size(64)
fn reducePipelinedRecovery(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;var delta=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){delta+=partials[at].x;}}
  reduceA[lid.x]=delta;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){if(reduceA[0]>1e-20){
    scalars[2]=scalars[0]/reduceA[0];scalars[3]=0.0;scalars[14]=0.0;
  }else{scalars[5]=0.0;}}
}

// A fresh b-Ap application is the convergence receipt.  The ordinary PCG
// recurrence remains useful for iteration-local alpha/beta updates, but f32
// recurrence drift is never published as the final residual authority.
fn measureTrueResidualWork(gid:vec3u,lid:vec3u,wid:vec3u,enabled:bool,
 writeResidual:bool,writeGuardScratch:bool){
  let id=pressureCellInvocation(gid.x);
  var residual2=0.0;var maximum=0.0;
  if(enabled&&id!=INVALID){
    let residual=state[p.stateOffsets2.y+id]
      -applyOperator(id,p.stateOffsets2.x);
    if(writeResidual){state[p.stateOffsets3.y+id]=residual;}
    if(writeGuardScratch){state[p.stateOffsets5.y+id]=residual;}
    residual2=residual*residual;maximum=abs(residual);
  }
  reduceA[lid.x]=residual2;reduceB[lid.x]=maximum;workgroupBarrier();
  var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];
      reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u){partials[wid.x]=vec4f(reduceA[0],reduceB[0],0.0,0.0);}
}

@compute @workgroup_size(64)
fn measureTrueResidual(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  measureTrueResidualWork(gid,lid,wid,true,true,false);
}

@compute @workgroup_size(64)
fn measureGuardedTrueResidual(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  // Preserve the fresh vector separately. The reduction decides whether f32
  // recurrence drift warrants a full pipelined-CG replacement.
  measureTrueResidualWork(gid,lid,wid,scalars[5]>0.5,false,true);
}

fn reduceTrueResidualPartials(lane:u32)->vec2f{
  var sum=0.0;var maximum=0.0;
  for(var at=lane;at<pressureCellWorkgroups();at+=64u){
    sum+=partials[at].x;maximum=max(maximum,partials[at].y);
  }
  reduceA[lane]=sum;reduceB[lane]=maximum;workgroupBarrier();
  var width=32u;loop{
    if(lane<width){reduceA[lane]+=reduceA[lane+width];
      reduceB[lane]=max(reduceB[lane],reduceB[lane+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  return vec2f(reduceA[0],reduceB[0]);
}

@compute @workgroup_size(64)
fn reduceInitialTrueResidual(@builtin(local_invocation_id)lid:vec3u){
  let receipt=reduceTrueResidualPartials(lid.x);
  if(lid.x==0u){
    scalars[8]=receipt.x;scalars[9]=receipt.y;
    let tolerance=pressureRelativeTolerance();
    if(tolerance>0.0&&receipt.x<=tolerance*tolerance*scalars[1]){
      scalars[5]=0.0;scalars[13]=0.0;
    }
  }
}

@compute @workgroup_size(64)
fn reduceFinalTrueResidual(@builtin(local_invocation_id)lid:vec3u){
  let receipt=reduceTrueResidualPartials(lid.x);
  if(lid.x==0u){
    scalars[10]=receipt.x;scalars[11]=receipt.y;
    scalars[16]=select(1.0,sqrt(max(0.0,scalars[4])/max(receipt.x,1e-30)),receipt.x>0.0);
    scalars[17]=select(0.0,1.0,16.0*max(0.0,scalars[4])<receipt.x);
    let tolerance=pressureRelativeTolerance();
    if(tolerance>0.0&&receipt.x<=tolerance*tolerance*scalars[1]&&scalars[13]<0.0){
      scalars[13]=scalars[12];
    }
  }
}

@compute @workgroup_size(64)
fn reduceGuardedTrueResidual(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5;
  let receipt=reduceTrueResidualPartials(lid.x);
  if(lid.x==0u&&enabled){
    scalars[10]=receipt.x;scalars[11]=receipt.y;
    let tolerance=pressureRelativeTolerance();
    if(tolerance>0.0&&receipt.x<=tolerance*tolerance*scalars[1]){
      if(scalars[13]<0.0){scalars[13]=scalars[12];}scalars[5]=0.0;
    }else if(receipt.x>16.0*max(scalars[4],1e-30)){
      scalars[14]=1.0;scalars[18]+=1.0;
    }
  }
}

@compute @workgroup_size(64)
fn restartPCGAfterCurvatureLoss(@builtin(global_invocation_id)gid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;
  let id=pressureCellInvocation(gid.x);
  if(enabled&&id!=INVALID){
    state[p.stateOffsets3.y+id]=state[p.stateOffsets5.y+id];
  }
}

@compute @workgroup_size(64)
fn initializeBrickAggregateRecoveryDirection(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;
  let id=pressureCellInvocation(gid.x);var rz=0.0;
  if(enabled&&id!=INVALID){let residual=state[p.stateOffsets3.y+id];
    let z=brickAggregatePreconditioned(id);
    state[p.stateOffsets3.z+id]=z;state[p.stateOffsets3.w+id]=z;rz=residual*z;
  }
  reducePair(lid.x,wid.x,rz,0.0);
}

@compute @workgroup_size(64)
fn reduceCurvatureRecovery(@builtin(local_invocation_id)lid:vec3u){
  let enabled=scalars[5]>0.5&&scalars[14]>0.5;var sum=0.0;
  if(enabled){for(var at=lid.x;at<pressureCellWorkgroups();at+=64u){sum+=partials[at].x;}}
  reduceA[lid.x]=sum;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&enabled){scalars[0]=reduceA[0];scalars[3]=0.0;}
}

fn projectPressureRow(row:u32){
  let theta=state[p.stateOffsets3.x+row];if(theta<=0.0||rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;}
  var jump=0.0;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var at=begin;at<end;at+=1u){let cell=termCell(at);if(pcmCellContains(cell)){
    jump+=termCoefficient(at)*state[p.stateOffsets2.x+cell];}}
  let pressureOpen=select(1.0,rowPressureOpenFraction(row),hasRigidBodies());
  state[destinationFaceVelocity()+row]-=pressureOpen*jump/theta;
}
@compute @workgroup_size(64)
fn projectFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=pressureRowInvocation(gid.x);if(row==INVALID){return;}
  projectPressureRow(row);
}
@compute @workgroup_size(64)
fn collocateAndDiagnose(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  if(!cellTransportActive(id)){
    let output=destinationCellVelocity()+4u*id;
    if(cellActive(id)&&any(bitcast<vec3u>(vec3f(state[output],state[output+1u],
      state[output+2u]))!=vec3u(0u))){
      incrementalActivityMarkCellClosure(id,
        ${SPARSE_CM12_DIRTY_CAUSE_BIT.velocityCharacteristic
          | SPARSE_CM12_DIRTY_CAUSE_BIT.movingSolidSweep}u);
    }
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    state[p.stateOffsets4.y+id]=0.0;return;
  }
  let previousAt=destinationCellVelocity()+4u*id;
  let previousVelocity=vec3f(state[previousAt],state[previousAt+1u],state[previousAt+2u]);
  var velocity=vec3f(0.0);var weight=vec3f(0.0);var equation=0.0;var correction=0.0;
  for(var at=incidenceBegin(id);at<incidenceEnd(id);at+=1u){let row=incidenceRow(at);
    if(!rowAccepted(row)){continue;}
    let term=incidenceTerm(at);let axis=rowAxis(row);
    let fluxWeight=select(rowDualWeight(row),rowStaticDualWeight(row),hasRigidBodies());
    let w=abs(termCoefficient(term))*fluxWeight;var faceVelocity=state[destinationFaceVelocity()+row];
    if(hasRigidBodies()){
      let open=rowOpenFraction(row);
      faceVelocity=select(rowSolidVelocity(row),
        (faceVelocity-(1.0-open)*rowSolidVelocity(row))/max(open,1e-6),open>1e-6);
    }
    velocity[axis]+=w*faceVelocity;weight[axis]+=w;
    if(pcmCellContains(id)){let value=termCoefficient(term)*fluxWeight*state[destinationFaceVelocity()+row];
      let adjusted=value-correction;let next=equation+adjusted;correction=(next-equation)-adjusted;equation=next;}}
  // Domain-wall faces are fixed zero-velocity ports with no row. They still
  // contribute one side of the MAC-to-cell average; without their zero-valued
  // weight a boundary cell reads its sole interior face at full strength --
  // twice the collocated wall velocity, doubled once per touched wall plane
  // (three times over in a bottom corner). Mirrors the CPU reference in
  // sparse-atlas-composite-projection.ts, restated in this loop's area weight
  // (|coefficient|*dualWeight = overlap area) rather than the CPU's
  // area/distance.
  {let base=cellBase(id);
    let center=vec3f(taf(base),taf(base+1u),taf(base+2u));
    let widths=vec3f(taf(base+4u),taf(base+5u),taf(base+6u));
    for(var axis=0u;axis<3u;axis+=1u){
      let t0=(axis+1u)%3u;let t1=(axis+2u)%3u;
      let port=widths[t0]*widths[t1];
      if(center[axis]-0.5*widths[axis]<=1e-4){weight[axis]+=port;}
      if(center[axis]+0.5*widths[axis]>=f32(p.dimensions[axis])-1e-4){weight[axis]+=port;}
    }
  }
  for(var axis=0u;axis<3u;axis+=1u){if(weight[axis]>0.0){velocity[axis]/=weight[axis];}}
  let velocityDelta=velocity-previousVelocity;
  let velocityChanged=length(velocityDelta)>0.0;
  if(velocityChanged){
    incrementalActivityMarkCellClosure(id,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.velocityCharacteristic}u);
  }
  state[destinationCellVelocity()+4u*id]=velocity.x;state[destinationCellVelocity()+4u*id+1u]=velocity.y;
  state[destinationCellVelocity()+4u*id+2u]=velocity.z;state[destinationCellVelocity()+4u*id+3u]=0.0;
  cm12PublishCollocatedWetEffectiveVelocity(id,velocity,
    state[destinationDensity()+id]>CM12_LIQUID_ISOVALUE);
  let rawDensity=rawPressureDensity(id);
  var targetDivergence=select(cm12VolumeCorrectionDivergence(rawDensity,
    p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
  // Sec. 3.7's source only corrects rho' > 1. The dual defect -- a submerged
  // cell holding rho' < 1 -- is unreachable by any paper mechanism: the
  // projection conserves member mass, sharpening removes only from rho < 0.5
  // cells and deposits at the iso-contour, so a pool smeared by a violent
  // phase stays under-dense forever, parked just above the isovalue where
  // membership flicker opens pressure holes. Apply the same rate with the
  // opposite sign, using the paper's own lambda and eta, gated to cells with
  // no air neighbour so free-surface and splash cells are never touched. It
  // self-extinguishes at rho' = 1.
  if(!cellSeparatingMinimum(id)&&rawDensity<1.0&&pressureCellSubmerged(id)){
    targetDivergence=-min(CM12_VOLUME_CORRECTION_LAMBDA*(1.0-rawDensity),
      CM12_VOLUME_CORRECTION_ETA)/(p.frame.y*cellMinimumWidth(id));
  }
  let controlVolume=select(cellOpenVolume(id),cellVolume(id),cellSeparatingMinimum(id));
  state[p.stateOffsets4.y+id]=select(0.0,-equation/max(controlVolume,1e-8)
    -targetDivergence,pcmCellContains(id));
}

// Collocated velocity is transport state, not the conservative projected face
// authority. A newly published receiver reconstructs its first face values
// from this field. Make the existing scene-level D4 authority apply here too,
// before sub-ULP collocation differences are amplified by the next pressure
// RHS on the much more highly decomposed 4-cell brick grid.
@compute @workgroup_size(64)
fn preserveVelocityHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  if(!cellActive(cell)){
    for(var component=0u;component<3u;component+=1u){
      atomicStore(&conditioning[component*p.counts.x+cell],bitcast<i32>(0.0));
    }
    state[p.stateOffsets5.x+cell]=0.0;
    return;
  }
  let b=cellBase(cell);let center=vec3f(taf(b),taf(b+1u),taf(b+2u));
  let extent=f32(p.dimensions.x);
  let xs=array<f32,8>(center.x,extent-center.x,center.x,extent-center.x,
    center.z,extent-center.z,center.z,extent-center.z);
  let zs=array<f32,8>(center.z,center.z,extent-center.z,extent-center.z,
    center.x,center.x,extent-center.x,extent-center.x);
  var sum=vec3f(0.0);var count=0u;var pressureSum=0.0;var pressureCount=0u;
  for(var transform=0u;transform<8u;transform+=1u){
    let member=ownerCellAt(vec3i(i32(floor(xs[transform])),i32(floor(center.y)),
      i32(floor(zs[transform]))));
    if(member==INVALID||!cellActive(member)){continue;}
    let at=destinationCellVelocity()+4u*member;
    var v=vec3f(state[at],state[at+1u],state[at+2u]);
    if(transform==1u){v.x=-v.x;}
    else if(transform==2u){v.z=-v.z;}
    else if(transform==3u){v.x=-v.x;v.z=-v.z;}
    else if(transform==4u){let x=v.x;v.x=v.z;v.z=x;}
    else if(transform==5u){let x=v.x;v.x=v.z;v.z=-x;}
    else if(transform==6u){let x=v.x;v.x=-v.z;v.z=x;}
    else if(transform==7u){let x=v.x;v.x=-v.z;v.z=-x;}
    sum+=v;count+=1u;
    if(pcmCellContains(member)){
      pressureSum+=state[p.stateOffsets2.x+member];pressureCount+=1u;
    }
  }
  let average=select(vec3f(0.0),sum/f32(count),count>0u);
  atomicStore(&conditioning[cell],bitcast<i32>(average.x));
  atomicStore(&conditioning[p.counts.x+cell],bitcast<i32>(average.y));
  atomicStore(&conditioning[2u*p.counts.x+cell],bitcast<i32>(average.z));
  state[p.stateOffsets5.x+cell]=select(0.0,
    pressureSum/f32(pressureCount),pressureCount>0u);
}

@compute @workgroup_size(64)
fn commitVelocityHorizontalD4(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||!cellActive(cell)){return;}
  let at=destinationCellVelocity()+4u*cell;
  let previous=vec3f(state[at],state[at+1u],state[at+2u]);
  let next=vec3f(bitcast<f32>(atomicLoad(&conditioning[cell])),
    bitcast<f32>(atomicLoad(&conditioning[p.counts.x+cell])),
    bitcast<f32>(atomicLoad(&conditioning[2u*p.counts.x+cell])));
  let velocityDelta=next-previous;
  if(length(velocityDelta)>0.0){
    incrementalActivityMarkCellClosure(cell,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.velocityCharacteristic}u);
  }
  state[at]=next.x;state[at+1u]=next.y;state[at+2u]=next.z;
  cm12PublishCollocatedWetEffectiveVelocity(cell,next,
    state[destinationDensity()+cell]>CM12_LIQUID_ISOVALUE);
  if(pcmCellContains(cell)){state[p.stateOffsets2.x+cell]=state[p.stateOffsets5.x+cell];}
}

@compute @workgroup_size(64)
fn measureDivergenceDiagnostics(@builtin(global_invocation_id)gid:vec3u,
 @builtin(local_invocation_id)lid:vec3u,@builtin(workgroup_id)wid:vec3u){
  var globalMaximum=0.0;var mixedMaximum=0.0;
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell!=INVALID){let value=abs(state[p.stateOffsets4.y+cell]);
    globalMaximum=value;var touchesMixed=false;
    for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
      let row=incidenceRow(at);
      touchesMixed=touchesMixed||(rowAccepted(row)&&rowKind(row)==2u);
    }
    mixedMaximum=select(0.0,value,touchesMixed);
  }
  reduceA[lid.x]=globalMaximum;reduceB[lid.x]=mixedMaximum;workgroupBarrier();
  var width=32u;loop{if(lid.x<width){reduceA[lid.x]=max(reduceA[lid.x],reduceA[lid.x+width]);
    reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){partials[wid.x]=vec4f(reduceA[0],reduceB[0],0.0,0.0);}
}

@compute @workgroup_size(64)
fn reduceDivergenceDiagnostics(@builtin(local_invocation_id)lid:vec3u){
  var globalMaximum=0.0;var mixedMaximum=0.0;
  for(var at=lid.x;at<acceptedTemplateCellWorkgroups();at+=64u){
    globalMaximum=max(globalMaximum,partials[at].x);
    mixedMaximum=max(mixedMaximum,partials[at].y);
  }
  reduceA[lid.x]=globalMaximum;reduceB[lid.x]=mixedMaximum;workgroupBarrier();
  var width=32u;loop{if(lid.x<width){reduceA[lid.x]=max(reduceA[lid.x],reduceA[lid.x+width]);
    reduceB[lid.x]=max(reduceB[lid.x],reduceB[lid.x+width]);}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lid.x==0u){scalars[6]=reduceA[0];scalars[7]=reduceB[0];}
}

// Advance and clear the compact receipt entirely on the device. The host
// encodes this fixed singleton but neither supplies nor consumes a policy
// decision while advancing the simulation.
@compute @workgroup_size(1)
fn advanceActivityClock(){
  let step=atomicAdd(&activity[0],1u)+1u;
  // Score/class counters are persistent. Dirty bricks transactionally remove
  // their old census contribution before publishing the new one.
  atomicStore(&activity[5],select(0u,1u,step%p.activityEpochs.x==0u));
  atomicStore(&activity[6],0u); // measured bricks
  atomicStore(&activity[7],0u); // reserved failure flags
  atomicStore(&activity[9],0u); // newly activated bricks
  atomicStore(&activity[14],0u); // urgent queued
  atomicStore(&activity[15],0u); // ordinary queued
  atomicStore(&activity[16],0u); // prepared this frame
  atomicStore(&activity[17],0u); // committed this frame
  atomicStore(&activity[18],0u); // deferred this frame
  atomicStore(&activity[21],0u); // commit failure latch
}

fn activityF32(index:u32)->f32{return bitcast<f32>(atomicLoad(&activity[index]));}

// Finest-cell displacement in one accepted step is the resolution signal the
// user can reason about directly. The live policy uniform supplies the three
// descending 8^3/4^3/2^3 thresholds; slower bulk may use 1^3. Surface evidence
// independently overrides this floor to 8^3 below.
fn activitySignalsEnabled()->bool{return p.activityEpochs.w!=0u;}

fn velocityResolutionFloor(travelFineCells:f32)->u32{
  if(!activitySignalsEnabled()){return 1u;}
  if(travelFineCells>=p.activityThresholds.x){return BRICK_FINE_RESOLUTION;}
  if(travelFineCells>=p.activityThresholds.y){return max(1u,BRICK_FINE_RESOLUTION/2u);}
  if(travelFineCells>=p.activityThresholds.z){return max(1u,BRICK_FINE_RESOLUTION/4u);}
  return 1u;
}

// One workgroup owns one brick. Fixed-point density moments make the compact
// history exactly invariant to x/z lane permutations for a D4-symmetric field;
// only maxima are used for floating activity channels.
@compute @workgroup_size(64)
fn measureBrickActivity(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let requestedBrick=incrementalActivityBrickInvocation(wid.x);let lane=lid.x;
  // Keep invalid-worklist fail-closed handling outside the workgroup-uniform
  // reduction. All lanes reduce an empty brick-zero payload, then lane zero
  // exits without publishing it.
  let brick=select(0u,requestedBrick,requestedBrick!=INVALID);
  let resident=requestedBrick!=INVALID&&brickActive(brick);
  let resolution=acceptedBrickResolution(brick);
  let range=templateBrickCellRange(brick,resolution);
  let first=range.x;let count=range.y;
  var densitySum=0;var momentX=0;var momentY=0;var momentZ=0;
  var deformation=0.0;var predictedMotion=0.0;var detailError=0.0;
  var velocityTravel=0.0;
  var surfaceAxes=0u;var occupiedCell=false;var substantialDensityCell=false;
  var thinFluidCell=false;
  var cutBoundaryCell=false;
  var supportMask=0u;var sweptSupportMask=0u;
  let measuredCount=select(0u,count,resident);
  for(var cell=first+lane;cell<first+measuredCount;cell+=64u){
    let rho=state[destinationDensity()+cell];
    let fill=rho/max(cellOpenFraction(cell),1e-6);
    cutBoundaryCell=cutBoundaryCell||cellOpenFraction(cell)<0.999;
    let local=cell-first;let x=local%resolution;
    let yz=local/resolution;let y=yz%resolution;let z=yz/resolution;
    // Every cell in one accepted brick rung has the same geometric volume.
    // Accumulate its dimensionless density here and apply that common volume
    // after the reduction. Including macro-cell volume in the fixed-point
    // term overflowed i32 for a full 32^3 cell (exactly 2^31 at rho=1), which
    // made calm deep-water bricks look empty and eligible for retirement.
    let ownFixed=i32(round(rho*ACTIVITY_FIXED));
    densitySum+=ownFixed;
    momentX+=i32(round(rho
      *(f32(2u*x+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    momentY+=i32(round(rho
      *(f32(2u*y+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    momentZ+=i32(round(rho
      *(f32(2u*z+1u)-f32(resolution))/f32(resolution)*ACTIVITY_FIXED));
    // A fractional density is not sufficient surface evidence. Wall
    // conditioning and conservative transport can leave submerged cells in
    // the broad 0<rho<1 band indefinitely. Require a represented liquid/air
    // crossing or an interior air-facing side below; the independent
    // thin-feature test preserves sheets that never reach the rho=.5 contour.
    let fractionalCell=fill>p.activityDensity.y&&fill<p.activityDensity.z;
    var interfaceCell=false;
    // The arithmetic dry epsilon is deliberately smaller than the residency
    // floor. Numerically transported mist must not pin an otherwise empty
    // sparse region after a scene settles.
    let residencyDensity=residencyDensityThreshold();
    occupiedCell=occupiedCell||rho>residencyDensity;
    // Residency mass is measured in finest-cell equivalents. At a coarse
    // rung, a broad sub-surface-density haze can therefore exceed one finest
    // cell of total mass even though it contains no represented liquid
    // feature. Record an independent concentration witness; genuinely dilute
    // sheets remain protected by the geometric thin-fluid flag below.
    substantialDensityCell=substantialDensityCell||fill>p.activityDensity.y;
    let ownVelocityAt=destinationCellVelocity()+4u*cell;
    let ownVelocity=vec3f(state[ownVelocityAt],state[ownVelocityAt+1u],
      state[ownVelocityAt+2u]);
    if(rho>residencyDensity){
      velocityTravel=max(velocityTravel,p.frame.x*length(ownVelocity));
    }
    let ownWet=fill>=CM12_LIQUID_ISOVALUE;
    let featureDensity=max(residencyDensity,p.activityDensity.x);
    let b=cellBase(cell);let cellCenter=vec3f(taf(b),taf(b+1u),taf(b+2u));
    var exposedSides=0u;var surfaceAirSides=0u;
    for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
      let row=incidenceRow(incidence);if(!rowAccepted(row)){continue;}
      let own=termCoefficient(incidenceTerm(incidence));
      if(rowArea(row)<=1e-8){continue;}
      let axis=rowAxis(row);let rowPosition=rowCenter(row);
      // A sparse-air row on the domain box is the closed container wall, not
      // a liquid-air interface. Treating the floor and side walls as surface
      // made calm bottom bricks permanently complex/hot and refined them after
      // the tank had settled. Only omitted air strictly inside the domain is
      // free-surface evidence.
      let omittedAirInside=rowKind(row)==3u&&rowPosition[axis]>1e-4
        &&rowPosition[axis]<f32(p.dimensions[axis])-1e-4;
      var crosses=omittedAirInside&&ownWet;var sideHasFluid=false;
      var sideHasSurfaceFluid=false;
      let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
      for(var term=begin;term<end;term+=1u){
        let coefficient=termCoefficient(term);if(own*coefficient>=0.0){continue;}
        let neighbor=termCell(term);
        // Acceptance of the owning row already proves the neighbor's brick and
        // resolution; only the dynamic cut-cell openness remains to test.
        if(cellOpenVolume(neighbor)<=1e-8){continue;}
        let neighborDensity=state[destinationDensity()+neighbor]
          /max(cellOpenFraction(neighbor),1e-6);
        sideHasFluid=sideHasFluid||neighborDensity>featureDensity;
        sideHasSurfaceFluid=sideHasSurfaceFluid
          ||neighborDensity>p.activityDensity.y;
        let crossesIsovalue=(neighborDensity>=CM12_LIQUID_ISOVALUE)!=ownWet;
        // The renderer's represented surface is the rho=.5 isovalue, including
        // a transition broadened by restriction onto a coarse rung. Requiring
        // one endpoint to reach the configured air band made this predicate
        // resolution-dependent: a genuine 40%-60% coarse crossing disappeared
        // from activity and was then allowed to coarsen again. Publish every
        // accepted isovalue crossing here. Activity mode's brickDeeplyEnclosed
        // predicate independently rejects surrounded bulk ripples at planning
        // time; Surface-distance mode deliberately follows the rendered field.
        crosses=crosses||crossesIsovalue;
        let neighborVelocityAt=destinationCellVelocity()+4u*neighbor;
        let neighborVelocity=vec3f(state[neighborVelocityAt],
          state[neighborVelocityAt+1u],state[neighborVelocityAt+2u]);
        deformation=max(deformation,p.frame.x*max(abs(ownVelocity.x-neighborVelocity.x),
          max(abs(ownVelocity.y-neighborVelocity.y),abs(ownVelocity.z-neighborVelocity.z)))
          /max(0.15*rowDistance(row),1e-12));
      }
      if(rho>featureDensity&&!sideHasFluid){
        let side=select(0u,1u,rowPosition[axis]>cellCenter[axis]);
        exposedSides|=1u<<(2u*axis+side);
      }
      if(fractionalCell&&!sideHasSurfaceFluid){
        let side=select(0u,1u,rowPosition[axis]>cellCenter[axis]);
        surfaceAirSides|=1u<<(2u*axis+side);
      }
      if(crosses){
        interfaceCell=true;
        surfaceAxes|=1u<<rowAxis(row);
        predictedMotion=max(predictedMotion,p.frame.x
          *abs(state[destinationFaceVelocity()+row])/max(0.25*rowDistance(row),1e-12));
      }
    }
    // A surface test alone misses dilute sheets whose density never reaches
    // the rho=.5 contour. Preserve any represented liquid slab thinner than
    // the configured finest-cell width: it must have exposed support on both
    // sides of an axis and remain above the feature-density floor.
    let representedThickness=clamp(rho,0.0,1.0)*cellMinimumWidth(cell);
    var cellIsThinFluid=false;
    for(var axis=0u;axis<3u;axis+=1u){
      let oppositeSides=3u<<(2u*axis);
      let airFacing=(surfaceAirSides&oppositeSides)!=0u;
      if(fractionalCell&&airFacing){
        interfaceCell=true;surfaceAxes|=1u<<axis;
      }
      cellIsThinFluid=cellIsThinFluid||(fill>featureDensity
        &&representedThickness<p.activityThresholds.w
        &&(exposedSides&oppositeSides)==oppositeSides);
    }
    thinFluidCell=thinFluidCell||cellIsThinFluid;
    // The immediate static receiver shell is structural capacity around the
    // represented interface, independent of whether activity scoring is on.
    // Swept prediction below extends that shell directionally; it does not
    // replace the one-brick transport support that exists at zero velocity.
    if(interfaceCell||cellIsThinFluid){
      var minimumOffset=vec3i(0);var maximumOffset=vec3i(0);
      if(x==0u){minimumOffset.x=-1;}if(x+1u==resolution){maximumOffset.x=1;}
      if(y==0u){minimumOffset.y=-1;}if(y+1u==resolution){maximumOffset.y=1;}
      if(z==0u){minimumOffset.z=-1;}if(z+1u==resolution){maximumOffset.z=1;}
      for(var dz=minimumOffset.z;dz<=maximumOffset.z;dz+=1){
        for(var dy=minimumOffset.y;dy<=maximumOffset.y;dy+=1){
          for(var dx=minimumOffset.x;dx<=maximumOffset.x;dx+=1){
            if(dx!=0||dy!=0||dz!=0){
              let bit=u32(dx+1)+3u*u32(dy+1)+9u*u32(dz+1);
              supportMask|=1u<<bit;
            }
      }}}
    }
    if(interfaceCell||cellIsThinFluid){
      let brickDimensions=vec3i((p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
        /BRICK_FINE_RESOLUTION);
      let sourceBrick=vec3i(cellMinimum(cell)/BRICK_FINE_RESOLUTION);
      let sweptBrick=clamp(vec3i(floor((cellCenter
        +p.activityTiming.x*p.frame.x*ownVelocity)/f32(BRICK_FINE_RESOLUTION))),
        vec3i(0),brickDimensions-vec3i(1));
      let sweptOffset=clamp(sweptBrick-sourceBrick,vec3i(-1),vec3i(1));
      if(any(sweptOffset!=vec3i(0))){
        let bit=u32(sweptOffset.x+1)+3u*u32(sweptOffset.y+1)
          +9u*u32(sweptOffset.z+1);
        supportMask|=1u<<bit;sweptSupportMask|=1u<<bit;
      }
    }
    if(resolution>1u){
      let group=2u*(vec3u(x,y,z)/2u);var childSum=0;
      for(var dz=0u;dz<2u;dz+=1u){for(var dy=0u;dy<2u;dy+=1u){
        for(var dx=0u;dx<2u;dx+=1u){
          let q=group+vec3u(dx,dy,dz);
          let child=first+q.x+resolution*(q.y+resolution*q.z);
          if(child<first+count){childSum+=i32(round(
            state[destinationDensity()+child]*ACTIVITY_FIXED));}
      }}}
      detailError=max(detailError,
        f32(abs(8*ownFixed-childSum))/(8.0*ACTIVITY_FIXED));
    }
  }
  activityMoments[lane]=vec4i(densitySum,momentX,momentY,momentZ);
  activityMetrics[lane]=vec4f(deformation,predictedMotion,detailError,velocityTravel);
  let activityFlags=surfaceAxes|select(0u,8u,substantialDensityCell)
    |select(0u,16u,occupiedCell)
    |select(0u,32u,thinFluidCell)|select(0u,64u,cutBoundaryCell);
  // support and swept-support each occupy bits 0..26. Split the seven flag
  // bits across their unused high bits: 61 bits become two words exactly.
  activityMasks[lane]=vec2u((supportMask&0x07ffffffu)|((activityFlags&31u)<<27u),
    (sweptSupportMask&0x07ffffffu)|((activityFlags>>5u)<<27u));
  workgroupBarrier();
  var width=32u;loop{
    if(lane<width){
      activityMoments[lane]+=activityMoments[lane+width];
      activityMetrics[lane]=max(activityMetrics[lane],activityMetrics[lane+width]);
      activityMasks[lane]|=activityMasks[lane+width];
    }
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lane!=0u){return;}if(requestedBrick==INVALID){return;}
  let output=activityRecord(brick);let step=atomicLoad(&activity[0]);
  incrementalActivityRemoveCensus(brick);
  if(!resident){
    atomicStore(&activity[output],0u);atomicStore(&activity[output+1u],0u);
    atomicStore(&activity[output+32u],0u);atomicStore(&activity[output+3u],0u);
    incrementalActivityAcceptMeasuredTopology(brick);
    return;
  }
  let reducedMoments=activityMoments[0];let reducedMetrics=activityMetrics[0];
  let reducedPackedMasks=activityMasks[0];
  let reducedSurfaceAxes=(reducedPackedMasks.x>>27u)|(reducedPackedMasks.y>>22u);
  let reducedSupportMask=reducedPackedMasks.x&0x07ffffffu;
  let reducedSweptSupportMask=reducedPackedMasks.y&0x07ffffffu;
  let meanDensity=f32(reducedMoments.x)/(f32(count)*ACTIVITY_FIXED);
  let moments=vec3f(reducedMoments.yzw)/(f32(count)*ACTIVITY_FIXED);
  var temporal=0.0;
  if(step>1u){
    temporal=max(abs(meanDensity-activityF32(output+4u))/0.05,
      max(abs(moments.x-activityF32(output+5u))/0.02,
      max(abs(moments.y-activityF32(output+6u))/0.02,
        abs(moments.z-activityF32(output+7u))/0.02)));
  }
  let densityMassFineCells=f32(reducedMoments.x)/ACTIVITY_FIXED*cellVolume(first);
  let densityPresent=(reducedSurfaceAxes&16u)!=0u;
  // A few concentrated interpolation remnants can carry too little total mass,
  // while broad coarse-grid haze can carry enough mass without containing a
  // liquid feature. Neither is a surface or neighbor-support authority.
  let representedFeature=(reducedSurfaceAxes&(8u|32u))!=0u;
  let occupied=densityPresent&&representedFeature
    &&densityMassFineCells>=p.sharpening.w;
  let axes=select(0u,reducedSurfaceAxes&7u,occupied);
  let surface=occupied&&axes!=0u;
  let thinFluid=occupied&&(reducedSurfaceAxes&32u)!=0u;
  let shape=select(0.0,1.0,countOneBits(axes)>=2u);
  let velocityActivity=reducedMetrics.w;
  // Restriction error is useful in flooded bulk and genuinely complex/thin
  // interface geometry. A calm one-axis free surface is different: CM12 is
  // supposed to keep that interface sharp, so its fine children can retain a
  // large restriction residual forever even though no dynamics require the
  // fine rung. Counting that residual made refinement irreversible after a
  // splash. Shape/thinness already protect non-planar or unresolved surfaces.
  let scoredDetailError=select(reducedMetrics.z,0.0,
    surface&&!thinFluid&&shape==0.0);
  // Deep incompressible liquid has no density interface to resolve. Pressure
  // and velocity gradients there are represented on the coarse composite
  // grid; scoring their hydrostatic start-up residue would erase the very
  // bulk coarsening this method exists to obtain. Dynamic change becomes a
  // resolution signal only at the interface or in a thin feature. Flooded
  // bulk can still veto a merge through genuine restriction detail.
  let dynamicActivity=select(0.0,max(reducedMetrics.x,temporal),
    surface||thinFluid);
  let detailActivity=max(0.0,scoredDetailError/p.activityDensity.w-1.0);
  let featureActivity=max(max(dynamicActivity,reducedMetrics.y),
    max(max(0.0,max(shape,select(0.0,1.0,thinFluid))),
      detailActivity));
  // Velocity thresholds define both the rung floor and the score scale. Using
  // raw cells/step here made a tuned 4-cell finest threshold irrelevant: one
  // cell of calm travel still saturated the emergency score and promoted one
  // rung every frame. A score of one now means the configured 8^3 threshold.
  let normalizedVelocityActivity=velocityActivity
    /max(p.activityThresholds.x,1e-6);
  // Uniform translation of a fully flooded brick carries no missing spatial
  // detail. Score travel only where a liquid-air interface/thin feature needs
  // characteristic lookahead; bulk refinement is driven by deformation,
  // temporal change and restriction error instead.
  let scoredVelocityActivity=select(0.0,normalizedVelocityActivity,surface||thinFluid);
  let scoreValue=clamp(max(scoredVelocityActivity,featureActivity),0.0,1.0);
  let score=u32(round(255.0*scoreValue));var reasons=0u;
  if(surface){reasons|=1u;}if(reducedMetrics.x>0.0){reasons|=2u;}
  if(temporal>0.0){reasons|=4u;}
  if(reducedMetrics.z>p.activityDensity.w){reasons|=8u;}
  if(reducedMetrics.y>0.0){reasons|=16u;}if(step==1u){reasons|=32u;}
  if(occupied){reasons|=64u;}
  if(velocityResolutionFloor(velocityActivity)>1u){reasons|=128u;}
  if(thinFluid){reasons|=256u;}
  if((reducedSurfaceAxes&64u)!=0u){reasons|=512u;}
  let topologyEpoch=atomicLoad(&activity[5])!=0u;
  let history=atomicLoad(&activity[output+2u]);
  var hotEpochs=history&255u;var quietEpochs=(history>>8u)&255u;
  if(topologyEpoch){
    let activitySignals=activitySignalsEnabled();
    let featureHot=activitySignals&&featureActivity>=p.activityTiming.y;
    hotEpochs=select(0u,min(255u,hotEpochs+1u),featureHot);
    let current=atomicLoad(&activity[output+12u]);
    let velocityFloor=select(1u,velocityResolutionFloor(velocityActivity),
      surface||thinFluid);
    let activityQuiet=!featureHot&&scoreValue<=p.activityTiming.w
      &&scoredDetailError<=p.activityDensity.w;
    let quiet=!thinFluid&&velocityFloor<current
      &&select(!surface,activityQuiet,activitySignals);
    quietEpochs=select(0u,min(255u,quietEpochs+1u),quiet);
  }
  atomicStore(&activity[output],score);atomicStore(&activity[output+1u],reasons);
  atomicStore(&activity[output+2u],hotEpochs|(quietEpochs<<8u));
  atomicStore(&activity[output+4u],bitcast<u32>(meanDensity));
  atomicStore(&activity[output+5u],bitcast<u32>(moments.x));
  atomicStore(&activity[output+6u],bitcast<u32>(moments.y));
  atomicStore(&activity[output+7u],bitcast<u32>(moments.z));
  atomicStore(&activity[output+32u],select(0u,reducedSupportMask,occupied));
  atomicStore(&activity[output+3u],select(0u,reducedSweptSupportMask,occupied));
  atomicStore(&activity[output+33u],bitcast<u32>(velocityActivity));
  atomicStore(&activity[output+34u],0u);
  incrementalActivityAddCensus(brick,score,reasons);
  incrementalActivityAcceptMeasuredTopology(brick);
}

// A receiver request is an immutable activity-snapshot predicate, not merely
// an activation predicate. An already-resident empty apron brick must remain
// fine when a neighbour's outward characteristic can reach it before the next
// topology epoch; otherwise it demotes for one paper step and immediately
// promotes after the front has crossed, which is both a resolution ping-pong
// and an under-resolved transport step.
fn brickRequestedAsReceiver(brick:u32)->bool{
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  var requested=injectionReachesBrick(brick);
  for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){
    for(var dx=-1;dx<=1;dx+=1){if(dx==0&&dy==0&&dz==0){continue;}
      let neighborCoordinate=coordinate+vec3i(dx,dy,dz);
      if(any(neighborCoordinate<vec3i(0))
        ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
      let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
        *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
      let neighbor=brickDirectoryLookup(neighborKey);
      if(neighbor==INVALID||!brickActive(neighbor)){continue;}
      let bit=u32(1-dx)+3u*u32(1-dy)+9u*u32(1-dz);
      let neighborOutput=activityRecord(neighbor);
      // Empty immediate support participates in the next transport stencil,
      // even at zero velocity, and is therefore a physical receiver. The
      // planner drops this floor after it becomes flooded unless its own
      // surface/velocity receipts still require it.
      let receiverMask=atomicLoad(&activity[neighborOutput+32u]);
      requested=requested||(receiverMask&(1u<<bit))!=0u;
  }}}
  return requested;
}

// A filled brick with majority-liquid face neighbours in every non-wall direction is
// deep bulk even if a low-amplitude density ripple happens to cross rho=.5 in
// one of its composite rows. Treating that internal crossing as a free surface
// permanently spread 8^3 resolution down through a tank after an impact. The
// mean-density test is deliberately much stronger than the residency bit: a
// trace of liquid in an air receiver must not make the real top surface look
// enclosed. The lookup remains span-aware beside immutable macro leaves.
fn brickDeeplyEnclosed(brick:u32)->bool{
  let output=activityRecord(brick);
  if((atomicLoad(&activity[output+1u])&64u)==0u){return false;}
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    // The container wall is closed support, not missing liquid.
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighbor=brickDirectoryLookupAtCoordinate(vec3u(neighborCoordinate));
    if(neighbor==INVALID||!brickActive(neighbor)
      ||activityF32(activityRecord(neighbor)+4u)<CM12_LIQUID_ISOVALUE){return false;}
  }
  return true;
}

// First candidate-planning rung. The accepted topology remains immutable:
// this pass publishes only the resolution requested by the CM12 surface floor,
// characteristic prediction, and retained activity history. Transfer and
// atomic generation publication consume this record in a later transaction.
@compute @workgroup_size(64)
fn planBrickResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  // Begin a candidate epoch by mirroring accepted membership. Lifecycle
  // planners below edit only this intent; word 10 remains accepted authority.
  setCandidateBrickActiveAt(output,brickActive(brick));
  let current=atomicLoad(&activity[output+12u]);
  // Macro leaves are immutable deep pages. Only span-one surface leaves own
  // mutation candidates; splitting a macro is a later sparse page-pool event.
  if(!brickCandidatePlanningEnabled(brick)){
    atomicStore(&activity[output+8u],current);
    atomicStore(&activity[output+9u],1024u);
    return;
  }
  let injectionReceiver=injectionReachesBrick(brick);
  // Injection is a topology transaction of its own. Preserve every brick the
  // drop does not touch so that a wetting gesture can refine its receivers and
  // 2:1 support without opportunistically coarsening an unrelated calm region.
  if(p.injectionCenter.w!=0.0&&!injectionReceiver){
    atomicStore(&activity[output+8u],current);
    atomicStore(&activity[output+9u],32u);
    return;
  }
  var requested=atomicLoad(&activity[output+8u]);
  var planReasons=atomicLoad(&activity[output+9u]);
  // Dormant capacity contributes no accepted worklist cells, so changing its
  // metadata cannot save simulation work. Asking every dormant brick to
  // coarsen also conflicts conceptually with the swept-activation pass below:
  // one epoch prepares a 1^3 candidate only for a receiver request to replace
  // it with 8^3. Retain the accepted rung until a physical receiver activates
  // the brick directly at 8^3; this removes background topology transactions
  // without weakening the front floor.
  if(!brickActive(brick)){
    atomicStore(&activity[output+8u],current);
    atomicStore(&activity[output+9u],128u);
    return;
  }
  let score=atomicLoad(&activity[output]);
  let reasons=atomicLoad(&activity[output+1u]);
  let history=atomicLoad(&activity[output+2u]);
  let hotEpochs=history&255u;let quietEpochs=(history>>8u)&255u;
  let recoveryState=atomicLoad(&activity[output+38u]);
  let recoveryFloor=recoveryState&31u;
  let recoveryLocked=(recoveryState&ACTIVITY_RECOVERY_LOCK)!=0u;
  let surface=(reasons&1u)!=0u;let predicted=(reasons&16u)!=0u;
  let thinFluid=(reasons&256u)!=0u;
  let cutBoundary=(reasons&512u)!=0u;
  let receiverRequested=brickRequestedAsReceiver(brick);
  // Both selector modes use this one physical planning path. Surface distance
  // differs only by disabling the activity-derived velocity, history, detail,
  // recovery and boundary floors below; surface/thin/receiver classification,
  // 2:1 closure, transfer and publication remain identical.
  let activitySignals=activitySignalsEnabled();
  // Raw detail remains in diagnostics, but a quiet planar surface's permanent
  // sharpening residual is not an additional activity veto; the independent
  // interface floor below already keeps that brick fine.
  let measuredVelocityFloor=velocityResolutionFloor(activityF32(output+33u));
  let velocityFloor=select(1u,measuredVelocityFloor,activitySignals);
  // Neighbour means can all exceed rho=.5 while a fast dam front still cuts
  // this brick internally. That is a moving interface, not settled submerged
  // restriction residue, and it must veto the aggressive deep-water request.
  // Once the internal crossing is slow again, the recovery lock/quiet history
  // below remains free to restore its exact calm coarse level.
  let movingInternalSurface=activitySignals&&surface&&velocityFloor>1u;
  let enclosed=activitySignals&&brickDeeplyEnclosed(brick)&&!movingInternalSurface;
  // The first accepted promotion closes the calm-baseline record for this
  // brick. Once its motion is quiet and it is again overwhelmingly liquid, a
  // new internal rho crossing may not overwrite that known-safe deep level.
  // Genuine surface bricks have an 8^3 recovery floor and remain fine.
  let settledRecoveredBulk=activitySignals&&recoveryLocked&&surface
    &&activityF32(output+4u)>=1.0-2.0*p.activityDensity.y
    &&quietEpochs>=p.activityEpochs.z;
  // Only an exposed liquid-air interface owns the hard surface floor. An
  // enclosed rho crossing is bulk restriction residue and must be allowed to
  // return through the same coarse ladder it occupied before an impact.
  let adaptiveSurface=surface&&!enclosed&&!settledRecoveredBulk;
  let slowSurface=adaptiveSurface&&!thinFluid&&velocityFloor==1u;
  let detail=activitySignals&&(reasons&8u)!=0u
    &&(!adaptiveSurface||thinFluid||(score>=u32(round(255.0))))
    &&!enclosed&&!slowSurface&&!settledRecoveredBulk;
  let receiver=injectionReceiver
    ||(receiverRequested&&((reasons&64u)==0u
      ||(activitySignals&&velocityFloor>1u)));
  // Every genuine free surface retains the conservative B^3 interface
  // invariant. Activity mode coarsens only enclosed bulk; uniform bulk
  // translation is absent from emergency scoring because it does not imply
  // missing spatial detail. A swept receiver is the predicted destination of
  // a moving interface, so it retains the same B^3 safety floor while 2:1
  // closure grades its neighbours.
  let requiredSurface=select(surface,adaptiveSurface,activitySignals);
  let interfaceVelocityFloor=select(1u,velocityFloor,
    activitySignals&&(adaptiveSurface||thinFluid||enclosed));
  let recoveryRequired=activitySignals&&recoveryLocked;
  let boundaryRequired=activitySignals&&cutBoundary;
  let dynamicRequired=max(select(1u,recoveryFloor,recoveryRequired),max(max(interfaceVelocityFloor,
    select(1u,BRICK_FINE_RESOLUTION,requiredSurface||thinFluid||receiver)),
    select(1u,max(1u,BRICK_FINE_RESOLUTION/2u),boundaryRequired)));
  // Fully surrounded liquid has no liquid-air feature to resolve. Ignore its
  // history and bulk-translation floors and retain only an embedded-boundary
  // floor; accepted/candidate 2:1 closure supplies all remaining resolution.
  let required=select(dynamicRequired,
    select(1u,max(1u,BRICK_FINE_RESOLUTION/2u),boundaryRequired),enclosed);
  let emergencyScore=u32(round(255.0*p.activityTiming.z));
  if(required>current
    ||(activitySignals&&!enclosed&&!slowSurface&&score>=emergencyScore)){
    // Surfaces, thin sheets, and receivers are safety floors and may
    // jump directly. Ordinary measured activity advances one rung, preventing
    // a low-speed emergency score from erasing the hierarchy in two frames.
    let urgent=requiredSurface||thinFluid||receiver;
    requested=select(min(BRICK_FINE_RESOLUTION,max(required,2u*current)),required,urgent);
    if(requiredSurface){planReasons=1u;}
    else if(receiver||predicted){planReasons=2u;}
    else if(thinFluid){planReasons=256u;}
    else if(velocityFloor>current){planReasons=64u;}else{planReasons=4u;}
  }else if(!activitySignals||atomicLoad(&activity[5])!=0u){
    requested=current;planReasons=32u;
    if(activitySignals&&!enclosed&&!slowSurface&&hotEpochs>=p.activityEpochs.y){
      requested=min(BRICK_FINE_RESOLUTION,2u*current);planReasons=8u;
    }else if(current>required
      &&(!activitySignals||enclosed||slowSurface||quietEpochs>=p.activityEpochs.z)&&!detail){
      // Deep liquid has no interface detail to preserve. Ask for its
      // coarsest physical level immediately; the refine-only closure below
      // raises only the cells needed to retain the 2:1 invariant. Exposed
      // quiet surfaces continue to descend one rung at a time.
      let directBulk=!activitySignals||enclosed;
      requested=select(max(required,current/2u),required,directBulk);
      planReasons=select(16u,2048u,directBulk);
    }
  }
  requested=applySparseCM12RefinementRegionBounds(brick,requested);
  atomicStore(&activity[output+8u],requested);
  atomicStore(&activity[output+9u],planReasons);
}

// Refine-only closure of GPU-authored candidate levels. Each dispatch moves a
// violation one rung toward a valid 2:1 plan; log2(B) ordered dispatches cover
// the complete dyadic ladder without ever coarsening a requested surface.
// The accepted-neighbour floor additionally keeps any bounded subset of the
// candidate transaction 2:1-valid when the coarsening scheduler publishes it.
@compute @workgroup_size(64)
fn closePlannedResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  if(!brickCandidatePlanningEnabled(brick)){return;}
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  var required=atomicLoad(&activity[activityRecord(brick)+8u]);
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
      *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
    let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
    let neighborOutput=activityRecord(neighbor);
    let neighborResolution=atomicLoad(&activity[neighborOutput+8u]);
    let neighborAccepted=atomicLoad(&activity[neighborOutput+12u]);
    required=max(required,max(neighborResolution,neighborAccepted)/2u);
  }
  atomicMax(&activity[activityRecord(brick)+8u],required);
}

fn validBrickResolution(resolution:u32)->bool{
  return resolution>=1u&&resolution<=BRICK_FINE_RESOLUTION
    &&(resolution&(resolution-1u))==0u;
}

// Candidate ABI boundary. This pass validates the closed device-authored plan
// and records transfer-pending state beside the still-immutable accepted
// level. It cannot publish topology or make candidate fields authoritative.
@compute @workgroup_size(64)
fn validateCandidateResolution(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+8u]);
  let constructionActivation=
    constructionActivationIntentWithoutSlot(brick,candidate);
  if(!brickCandidatePlanningEnabled(brick)&&!constructionActivation){
    let accepted=atomicLoad(&activity[output+12u]);
    atomicStore(&activity[output+13u],accepted);atomicStore(&activity[output+14u],0u);
    return;
  }
  var invalid=!validBrickResolution(accepted)||!validBrickResolution(candidate);
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),
    vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));
  for(var side=0u;side<6u;side+=1u){let neighborCoordinate=coordinate+directions[side];
    if(any(neighborCoordinate<vec3i(0))
      ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
    let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
      *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
    let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
    let neighborCandidate=atomicLoad(&activity[activityRecord(neighbor)+8u]);
    let larger=max(candidate,neighborCandidate);let smaller=min(candidate,neighborCandidate);
    invalid=invalid||larger>2u*smaller;
  }
  atomicStore(&activity[output+13u],candidate);
  let transition=candidate!=accepted
    ||candidateBrickActive(brick)!=brickActive(brick)
    ||injectionReachesBrick(brick);
  atomicStore(&activity[output+14u],select(select(0u,1u,transition),2u,invalid));
  atomicStore(&activity[output+15u],atomicLoad(&activity[0]));
  if(invalid){atomicOr(&activity[7],1u);}
}

// All refinement is urgent because the refine-only 2:1 closure may have
// introduced support rungs around a surface brick. Coarsening is bounded by a
// rotating brick-ID window, so no atomic ticket race can starve a quiet brick.
// Both walks are 64 lanes striding the same ranges the single lane used to.
// Classification is a pure per-brick predicate, so it parallelizes exactly. The
// coarsening window is order-dependent — it takes the first
// topologyScheduling.x demotions at or after the rotating cursor — so its chunk
// walks the *rotated* order and selects by exclusive prefix, which is the same
// set the cursor selected. The chunk loop deliberately never breaks early:
// selected is read from workgroup memory and is therefore non-uniform to
// WGSL's analysis, so making it a loop condition would put the scan's barriers
// in non-uniform control flow. Eighteen unconditional 64-wide chunks still cost
// a fraction of 1,152 serial iterations.
var<workgroup>topologyScheduleTotals:array<vec2u,64>;
@compute @workgroup_size(64)
fn scheduleTopologyPreparation(@builtin(local_invocation_id)lid:vec3u){
  let lane=lid.x;let count=p.dispatch.w;
  let generation=atomicLoad(&activity[12])+1u;
  var urgent=0u;var ordinary=0u;
  for(var brick=lane;brick<count;brick+=64u){
    let output=activityRecord(brick);
    setTopologyPreparationScheduled(output,false);
    if(atomicLoad(&activity[output+14u])!=1u){continue;}
    let accepted=atomicLoad(&activity[output+12u]);
    let candidate=atomicLoad(&activity[output+13u]);
    let lifecycle=candidateBrickActive(brick)!=brickActive(brick)
      ||injectionReachesBrick(brick);
    if(candidate>accepted||lifecycle){setTopologyPreparationScheduled(output,true);
      atomicStore(&activity[output+36u],generation);urgent+=1u;
    }else{ordinary+=1u;}
  }
  // The cursor rotation means brick b is cleared above by lane b%64 and may be
  // scheduled below by a different lane. That is a write-write hazard on word
  // 35 across lanes, which the single-lane scheduler could not have; order it
  // explicitly. workgroupBarrier alone would not — activity is storage.
  storageBarrier();
  topologyScheduleTotals[lane]=vec2u(urgent,ordinary);workgroupBarrier();
  var width=32u;loop{
    if(lane<width){topologyScheduleTotals[lane]+=topologyScheduleTotals[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  let urgentTotal=topologyScheduleTotals[0].x;
  let ordinaryTotal=topologyScheduleTotals[0].y;
  let budget=p.topologyScheduling.x;
  let cursor=atomicLoad(&activity[13])%max(1u,count);
  var scanned=0u;
  for(var chunk=0u;chunk<count;chunk+=64u){
    let distance=chunk+lane;var pending=0u;var brick=0u;
    if(distance<count){
      brick=(cursor+distance)%count;let output=activityRecord(brick);
      pending=select(0u,1u,atomicLoad(&activity[output+14u])==1u
        &&atomicLoad(&activity[output+13u])<atomicLoad(&activity[output+12u]));
    }
    let prefix=stablePressurePrefix(lane,pending);
    if(pending==1u&&scanned+prefix<budget){
      let output=activityRecord(brick);
      setTopologyPreparationScheduled(output,true);
      atomicStore(&activity[output+36u],generation);
      let accepted=atomicLoad(&activity[output+12u]);
      let candidate=atomicLoad(&activity[output+13u]);
    }
    // pressurePrefix now holds the inclusive scan; lane 63 is the chunk total.
    let chunkTotal=pressurePrefix[63];
    workgroupBarrier();
    scanned+=chunkTotal;
  }
  if(lane!=0u){return;}
  // The cursor walk visits every brick, so it always retires exactly
  // min(budget, |ordinary|) demotions. Deriving that here keeps the header
  // words independent of the non-uniform running count above.
  let selected=min(budget,ordinaryTotal);
  atomicStore(&activity[14],urgentTotal);atomicStore(&activity[15],ordinaryTotal);
  atomicStore(&activity[16],urgentTotal+selected);
  atomicStore(&activity[18],ordinaryTotal-selected);
}

fn acquireTopologyPage()->u32{
  let base=topologyWorklistBase();let capacity=atomicLoad(&topologyArena[base+27u]);
  // A bounded retry keeps validators from treating the terminal return as
  // unreachable while remaining far above the maximum 512-page contention.
  for(var attempt=0u;attempt<4096u;attempt+=1u){
    let count=atomicLoad(&topologyArena[base+26u]);
    if(count==0u){atomicAdd(&topologyArena[base+29u],1u);return INVALID;}
    let claimed=atomicCompareExchangeWeak(&topologyArena[base+26u],count,count-1u);
    if(claimed.exchanged){
      let freeList=atomicLoad(&topologyArena[base+28u]);
      let page=atomicLoad(&topologyArena[base+freeList+count-1u]);
      return select(INVALID,page,page<capacity);
    }
  }
  atomicAdd(&topologyArena[base+29u],1u);
  return INVALID;
}

fn releaseTopologyPage(page:u32){
  if(page==INVALID){return;}
  let base=topologyWorklistBase();let capacity=atomicLoad(&topologyArena[base+27u]);
  let index=atomicAdd(&topologyArena[base+26u],1u);
  if(index<capacity){
    let freeList=atomicLoad(&topologyArena[base+28u]);
    atomicStore(&topologyArena[base+freeList+index],page);
  }else{
    atomicSub(&topologyArena[base+26u],1u);atomicAdd(&topologyArena[base+29u],1u);
  }
}

// Allocation is a separate device transaction from topology synthesis. Large
// scenes keep planning locked until synthesized cell/row descriptors can make
// the claimed page authoritative; small template-backed scenes already carry
// a packed slot and therefore take the no-op branch.
@compute @workgroup_size(64)
fn allocateCandidateTopologyPages(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||brickPackedCandidateSlot(brick)!=INVALID){return;}
  let output=activityRecord(brick);
  if(scheduledConstructionActivationWithoutSlot(brick)){return;}
  if(!topologyPreparationScheduledAt(output)
    ||atomicLoad(&activity[output+37u])!=INVALID){return;}
  let page=acquireTopologyPage();
  if(page==INVALID){
    atomicStore(&activity[output+14u],2u);setTopologyPreparationScheduled(output,false);
    atomicOr(&activity[7],16u);return;
  }
  atomicStore(&activity[output+37u],page);
}

fn candidateTopologyPageBase(page:u32)->u32{
  let base=topologyWorklistBase();
  return base+atomicLoad(&topologyArena[base+30u])
    +page*atomicLoad(&topologyArena[base+31u]);
}

// Generate cell geometry directly into the claimed page. This is independent
// of field transfer: no accepted state or worklist can observe the descriptor
// until row synthesis, incidence construction, and publication all validate.
@compute @workgroup_size(64)
fn synthesizeCandidateCellPages(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);let page=atomicLoad(&activity[output+37u]);
  if(page==INVALID||!topologyPreparationScheduledAt(output)){return;}
  let resolution=atomicLoad(&activity[output+13u]);let count=resolution*resolution*resolution;
  let pageBase=candidateTopologyPageBase(page);
  if(lane==0u){
    atomicStore(&topologyArena[pageBase],brick);
    atomicStore(&topologyArena[pageBase+1u],resolution);
    atomicStore(&topologyArena[pageBase+2u],count);
    atomicStore(&topologyArena[pageBase+3u],atomicLoad(&activity[output+36u]));
  }
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;
  let brickOrigin=vec3u(bx,by,bz)*BRICK_FINE_RESOLUTION;
  let scale=BRICK_FINE_RESOLUTION/resolution;
  for(var local=lane;local<count;local+=64u){
    let z=local/(resolution*resolution);let yz=local-z*resolution*resolution;
    let y=yz/resolution;let x=yz-y*resolution;
    let lower=brickOrigin+vec3u(x,y,z)*scale;
    let upper=min(lower+vec3u(scale),p.dimensions.xyz);let widths=upper-lower;
    let center=vec3f(lower)+0.5*vec3f(widths);let volume=f32(widths.x*widths.y*widths.z);
    let cell=pageBase+4u+8u*local;
    atomicStore(&topologyArena[cell],bitcast<u32>(center.x));
    atomicStore(&topologyArena[cell+1u],bitcast<u32>(center.y));
    atomicStore(&topologyArena[cell+2u],bitcast<u32>(center.z));
    atomicStore(&topologyArena[cell+3u],bitcast<u32>(volume));
    atomicStore(&topologyArena[cell+4u],bitcast<u32>(f32(widths.x)));
    atomicStore(&topologyArena[cell+5u],bitcast<u32>(f32(widths.y)));
    atomicStore(&topologyArena[cell+6u],bitcast<u32>(f32(widths.z)));
    atomicStore(&topologyArena[cell+7u],
      (brick<<TEMPLATE_CELL_RESOLUTION_BITS)|resolution);
  }
}

// Dynamic pages are not template slots. Keep field-bearing transfer/publication
// asleep until a page also carries rows and incidence; those transitions index
// candidateState by packed template slot. The exact dormant, same-rung dry
// activation is retained because it is membership-only and has zero field/face
// payload by construction.
@compute @workgroup_size(1)
fn deferDynamicTopologyPublication(){
  let base=topologyWorklistBase();
  if(atomicLoad(&topologyArena[base+27u])==0u){return;}
  var retained=0u;
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){
    let record=activityRecord(brick);
    if(topologyPreparationScheduledAt(record)
      &&brickPackedCandidateSlot(brick)==INVALID
      &&!scheduledConstructionActivationWithoutSlot(brick)){
      setTopologyPreparationScheduled(record,false);
    }
    retained+=select(0u,1u,topologyPreparationScheduledAt(record));
  }
  atomicStore(&activity[16],retained);
}

@compute @workgroup_size(1)
fn beginShadowTopology(){
  let base=topologyWorklistBase();
  atomicStore(&topologyArena[base+3u],1u);
  atomicStore(&topologyArena[base+18u],0u);
  atomicStore(&topologyArena[base+19u],0u);
  atomicStore(&topologyArena[base+1u],atomicLoad(&topologyArena[base])+1u);
  let leaves=acceptedLeafManifestBase();
  atomicStore(&topologyArena[leaves+10u],0u);
  if(atomicLoad(&activity[16])!=0u){
    let slot=shadowTopologySlot();
    atomicStore(&topologyArena[leaves+slot],0u);
  }
}

// Remove only the old shadow slot's compact row membership before that slot is
// rebuilt. This is O(previous shadow rows), never O(template rows), and permits
// one two-bit membership word to represent rows shared by both topology slots.
@compute @workgroup_size(64)
fn clearShadowRowMembership(@builtin(global_invocation_id)gid:vec3u){
  let row=shadowTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  atomicAnd(&topologyArena[acceptedRowMembershipStampBase()+row],
    ~(1u<<shadowTopologySlot()));
}

var<workgroup>mainChangedXor:atomic<u32>;
var<workgroup>mainChangedSum:atomic<u32>;
@compute @workgroup_size(64)
fn buildShadowLeafWorklist(@builtin(local_invocation_index)lane:u32){
  if(lane==0u){atomicStore(&mainChangedXor,0u);atomicStore(&mainChangedSum,0u);}
  workgroupBarrier();
  let enabled=atomicLoad(&activity[16])!=0u;
  let base=acceptedLeafManifestBase();let slot=shadowTopologySlot();
  let offset=atomicLoad(&topologyArena[base+2u+slot]);
  let deltaOffset=atomicLoad(&topologyArena[base+11u]);
  var compacted=0u;var deltaCompacted=0u;
  for(var chunk=0u;chunk<p.dispatch.w;chunk+=64u){
    let brick=chunk+lane;
    let selected=select(0u,1u,enabled&&brick<p.dispatch.w&&scheduledBrickActive(brick));
    let prefix=stablePressurePrefix(lane,selected);
    if(selected!=0u){atomicStore(&topologyArena[base+offset+compacted+prefix],brick);}
    let chunkCount=pressurePrefix[63];workgroupBarrier();
    compacted+=chunkCount;
    var deltaSelected=0u;
    if(enabled&&brick<p.dispatch.w
      &&topologyPreparationScheduledAt(activityRecord(brick))){deltaSelected=1u;}
    let deltaPrefix=stablePressurePrefix(lane,deltaSelected);
    if(deltaSelected!=0u){
      atomicStore(&topologyArena[base+deltaOffset+deltaCompacted+deltaPrefix],brick);
      let mixed=((brick^0x9e3779b9u)*0x01000193u)^0x85ebca6bu;
      atomicXor(&mainChangedXor,mixed);atomicAdd(&mainChangedSum,mixed*0x85ebca6bu);
    }
    let deltaChunkCount=pressurePrefix[63];workgroupBarrier();
    deltaCompacted+=deltaChunkCount;
  }
  if(lane==0u){
    atomicStore(&topologyArena[base+slot],compacted);
    atomicStore(&topologyArena[base+10u],deltaCompacted);
    atomicStore(&topologyArena[base+15u],compacted);
    atomicStore(&topologyArena[base+16u],1u);
    atomicStore(&topologyArena[base+17u],1u);
    cm12ISASetExpectedChanged(deltaCompacted,
      atomicLoad(&mainChangedXor),atomicLoad(&mainChangedSum));
  }
}

var<workgroup>shadowLeafOutputBase:u32;
var<workgroup>shadowLeafOwnerFirst:u32;
var<workgroup>shadowLeafOwnerCount:u32;
var<workgroup>shadowLeafRowOutputBase:u32;
fn shadowRowScheduled(row:u32)->bool{
  let requirements=rowRequirementOffset(row);let count=ta(requirements);var enabled=true;
  for(var at=0u;at<count;at+=1u){let metadata=ta(requirements+1u+at);
    let brick=metadata>>TEMPLATE_CELL_RESOLUTION_BITS;
    let resolution=metadata&TEMPLATE_CELL_RESOLUTION_MASK;
    enabled=enabled&&scheduledBrickActive(brick)
      &&scheduledBrickResolution(brick)==resolution;
  }
  return enabled;
}

// One compact leaf traversal publishes all structural views needed by the
// shadow transaction: contiguous cells, owner rows, and slot membership.
// Each leaf reserves each output once; row validation and writes are spread
// across the workgroup without rediscovering global template topology.
@compute @workgroup_size(64)
fn buildShadowStructureWorklist(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let brick=shadowLeafInvocation(wid.x);var cellRange=vec2u(0u);var resolution=1u;
  if(brick!=INVALID){resolution=scheduledBrickResolution(brick);
    cellRange=templateBrickCellRange(brick,resolution);}
  let base=topologyWorklistBase();
  if(lane==0u){
    shadowLeafOutputBase=atomicAdd(&topologyArena[base+18u],cellRange.y);
    var ownerRange=vec2u(0u);
    if(brick!=INVALID){ownerRange=templateRowOwnerRange(brick,resolution);}
    shadowLeafOwnerFirst=ownerRange.x;shadowLeafOwnerCount=ownerRange.y;
  }
  let cellOutput=workgroupUniformLoad(&shadowLeafOutputBase);
  let ownerFirst=workgroupUniformLoad(&shadowLeafOwnerFirst);
  let ownerCount=workgroupUniformLoad(&shadowLeafOwnerCount);
  if(brick!=INVALID){let cellOffset=atomicLoad(
      &topologyArena[base+14u+shadowTopologySlot()]);
    for(var local=lane;local<cellRange.y;local+=64u){atomicStore(
      &topologyArena[base+cellOffset+cellOutput+local],cellRange.x+local);}
  }
  var laneCount=0u;
  for(var chunk=0u;chunk<ownerCount;chunk+=64u){let local=chunk+lane;
    if(local<ownerCount){let row=templateRowOwnerInvocation(ownerFirst+local);
      laneCount+=select(0u,1u,shadowRowScheduled(row));}
  }
  let lanePrefix=stablePressurePrefix(lane,laneCount);
  let rowCount=pressurePrefix[63];
  if(lane==0u){shadowLeafRowOutputBase=atomicAdd(&topologyArena[base+19u],rowCount);}
  workgroupBarrier();
  let rowOffset=atomicLoad(&topologyArena[base+16u+shadowTopologySlot()]);
  var written=0u;
  for(var chunk=0u;chunk<ownerCount;chunk+=64u){let local=chunk+lane;
    if(local<ownerCount){let row=templateRowOwnerInvocation(ownerFirst+local);
      if(shadowRowScheduled(row)){atomicStore(&topologyArena[
          base+rowOffset+shadowLeafRowOutputBase+lanePrefix+written],row);
        atomicOr(&topologyArena[acceptedRowMembershipStampBase()+row],
          1u<<shadowTopologySlot());written+=1u;}}
  }
}

@compute @workgroup_size(64)
fn buildShadowCellWorklist(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let brick=shadowLeafInvocation(wid.x);var range=vec2u(0u);
  if(brick!=INVALID){range=templateBrickCellRange(brick,scheduledBrickResolution(brick));}
  let base=topologyWorklistBase();
  if(lane==0u){shadowLeafOutputBase=atomicAdd(&topologyArena[base+18u],range.y);}
  workgroupBarrier();
  if(brick==INVALID){return;}
  let offset=atomicLoad(&topologyArena[base+14u+shadowTopologySlot()]);
  for(var local=lane;local<range.y;local+=64u){
    atomicStore(&topologyArena[base+offset+shadowLeafOutputBase+local],range.x+local);
  }
}

@compute @workgroup_size(64)
fn buildShadowRowWorklist(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  if(lane!=0u||atomicLoad(&activity[16])==0u){return;}
  let owner=shadowLeafInvocation(wid.x);if(owner==INVALID){return;}
  let range=templateRowOwnerRange(owner,scheduledBrickResolution(owner));
  var selected=0u;
  for(var local=0u;local<range.y;local+=1u){
    let row=templateRowOwnerInvocation(range.x+local);
    let requirements=rowRequirementOffset(row);let count=ta(requirements);var enabled=true;
    for(var at=0u;at<count;at+=1u){let metadata=ta(requirements+1u+at);
      let brick=metadata>>TEMPLATE_CELL_RESOLUTION_BITS;
      let resolution=metadata&TEMPLATE_CELL_RESOLUTION_MASK;
      enabled=enabled&&scheduledBrickActive(brick)
        &&scheduledBrickResolution(brick)==resolution;
    }
    selected+=select(0u,1u,enabled);
  }
  let base=topologyWorklistBase();let output=atomicAdd(&topologyArena[base+19u],selected);
  let offset=atomicLoad(&topologyArena[base+16u+shadowTopologySlot()]);var compacted=0u;
  for(var local=0u;local<range.y;local+=1u){
    let row=templateRowOwnerInvocation(range.x+local);
    let requirements=rowRequirementOffset(row);let count=ta(requirements);var enabled=true;
      for(var at=0u;at<count;at+=1u){let metadata=ta(requirements+1u+at);
        let brick=metadata>>TEMPLATE_CELL_RESOLUTION_BITS;
        let resolution=metadata&TEMPLATE_CELL_RESOLUTION_MASK;
        enabled=enabled&&scheduledBrickActive(brick)
          &&scheduledBrickResolution(brick)==resolution;
      }
    if(enabled){atomicStore(&topologyArena[base+offset+output+compacted],row);
      atomicOr(&topologyArena[acceptedRowMembershipStampBase()+row],
        1u<<shadowTopologySlot());
      compacted+=1u;}
  }
}

@compute @workgroup_size(1)
fn finalizeShadowWorklists(){
  let base=topologyWorklistBase();let cells=atomicLoad(&topologyArena[base+18u]);
  let rows=atomicLoad(&topologyArena[base+19u]);
  atomicStore(&topologyArena[base+20u],(cells+63u)/64u);
  atomicStore(&topologyArena[base+21u],1u);atomicStore(&topologyArena[base+22u],1u);
  atomicStore(&topologyArena[base+23u],(rows+63u)/64u);
  atomicStore(&topologyArena[base+24u],1u);atomicStore(&topologyArena[base+25u],1u);
  let leaves=acceptedLeafManifestBase();let slot=shadowTopologySlot();
  atomicStore(&topologyArena[leaves+18u+slot],rows);
  atomicStore(&topologyArena[leaves+7u],
    (atomicLoad(&topologyArena[leaves+slot])+63u)/64u);
  atomicStore(&topologyArena[leaves+8u],1u);
  atomicStore(&topologyArena[leaves+9u],1u);
  // Candidate transfer assigns one workgroup to each changed leaf, unlike
  // the linear 64-invocation accepted-leaf triplets above.
  atomicStore(&topologyArena[leaves+12u],atomicLoad(&topologyArena[leaves+10u]));
  atomicStore(&topologyArena[leaves+13u],1u);
  atomicStore(&topologyArena[leaves+14u],1u);
}

fn candidateFieldIndex(channel:u32,brick:u32,local:u32)->u32{
  let slots=topology[p.topologyOffsets2.w];let slot=brickCandidateSlot(brick);
  let cellCapacity=slots*CANDIDATE_CELLS_PER_BRICK;
  if(channel<6u){
    return channel*cellCapacity+slot*CANDIDATE_CELLS_PER_BRICK+local;
  }
  let faceCapacity=slots*CANDIDATE_FACE_SAMPLES_PER_SIDE;
  return 6u*cellCapacity+(channel-6u)*faceCapacity
    +slot*CANDIDATE_FACE_SAMPLES_PER_SIDE+local;
}

fn candidateCellVolume(brick:u32,resolution:u32,local:u32)->f32{
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;
  let z=local/(resolution*resolution);let yz=local-z*resolution*resolution;
  let y=yz/resolution;let x=yz-y*resolution;
  let scale=BRICK_FINE_RESOLUTION/resolution;
  let lower=vec3u(bx,by,bz)*BRICK_FINE_RESOLUTION+vec3u(x,y,z)*scale;
  let upper=min(lower+vec3u(scale),p.dimensions.xyz);
  let widths=upper-lower;return f32(widths.x*widths.y*widths.z);
}

fn finiteTransferValue(value:f32)->bool{
  return value==value&&abs(value)<3.4e38;
}

// Exact-overlap scalar and cell-momentum transfer into an isolated max-B^3
// candidate slot. The accepted topology and state remain untouched. Face-flux
// transfer, row patching, reprojection, and publication are later gates.
fn transferCandidateCellsWork(lid:vec3u,brick:u32,validBrick:bool){
  let lane=lid.x;
  let output=activityRecord(brick);
  if(lane==0u){candidateCellScheduled=select(0u,
    select(0u,1u,validBrick&&topologyPreparationScheduledAt(output)),validBrick);
    candidateCellConstructionActivation=select(0u,1u,
      validBrick&&scheduledConstructionActivationWithoutSlot(brick));}
  let scheduled=workgroupUniformLoad(&candidateCellScheduled);
  if(scheduled==0u){return;}
  let candidateStatus=atomicLoad(&activity[output+14u]);
  if(lane==0u){
    for(var side=0u;side<6u;side+=1u){atomicStore(&activity[output+24u+side],0u);}
    atomicStore(&activity[output+30u],0u);
    atomicStore(&activity[output+31u],select(select(0u,1u,candidateStatus==1u),2u,
      candidateStatus==2u));
  }
  // Membership-only construction activation has no candidate-state slot by
  // design. Its independently known dry field makes every conservative
  // transfer receipt exactly zero, so close the receipt without indexing the
  // INVALID packed slot.
  if(workgroupUniformLoad(&candidateCellConstructionActivation)!=0u){
    if(lane==0u){
      for(var word=16u;word<=22u;word+=1u){atomicStore(&activity[output+word],0u);}
      atomicStore(&activity[output+23u],select(2u,1u,candidateStatus==1u));
    }
    return;
  }
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+13u]);
  let acceptedActive=brickActive(brick);let candidateActive=candidateBrickActive(brick);
  let acceptedRange=templateBrickCellRange(brick,accepted);let first=acceptedRange.x;
  let candidateRange=templateBrickCellRange(brick,candidate);
  let sourceCount=select(0u,acceptedRange.y,acceptedActive);
  let candidateCount=select(0u,candidateRange.y,candidateActive);
  var beforeMass=0.0;var beforeGamma=0.0;var beforeMomentum=vec3f(0.0);
  var beforeGammaScale=0.0;var beforeMomentumScale=vec3f(0.0);
  for(var local=lane;local<sourceCount;local+=64u){let cell=first+local;
    let volume=cellVolume(cell);let rho=state[destinationDensity()+cell];
    let gammaContribution=state[destinationGamma()+cell]*volume;
    let velocityAt=destinationCellVelocity()+4u*cell;
    let velocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u]);
    let momentumContribution=rho*volume*velocity;
    beforeMass+=rho*volume;beforeGamma+=gammaContribution;
    beforeMomentum+=momentumContribution;beforeGammaScale+=abs(gammaContribution);
    beforeMomentumScale+=abs(momentumContribution);
  }
  var afterMass=0.0;var afterGamma=0.0;var afterMomentum=vec3f(0.0);
  var afterGammaScale=0.0;var afterMomentumScale=vec3f(0.0);
  for(var local=lane;local<candidateCount;local+=64u){
    let cz=local/(candidate*candidate);let yz=local-cz*candidate*candidate;
    let cy=yz/candidate;let cx=yz-cy*candidate;
    var rho=0.0;var gamma=0.0;var velocity=vec3f(0.0);var pressure=0.0;
    if(!acceptedActive){gamma=1.0;
    }else if(candidate<accepted){let factor=accepted/candidate;
      var volumeSum=0.0;var massSum=0.0;var momentumSum=vec3f(0.0);
      for(var dz=0u;dz<factor;dz+=1u){for(var dy=0u;dy<factor;dy+=1u){
        for(var dx=0u;dx<factor;dx+=1u){
          let sx=factor*cx+dx;let sy=factor*cy+dy;let sz=factor*cz+dz;
          let sourceLocal=sx+accepted*(sy+accepted*sz);let cell=first+sourceLocal;
          let volume=cellVolume(cell);let sourceRho=state[destinationDensity()+cell];
          let velocityAt=destinationCellVelocity()+4u*cell;
          let sourceVelocity=vec3f(state[velocityAt],state[velocityAt+1u],
            state[velocityAt+2u]);
          volumeSum+=volume;massSum+=sourceRho*volume;
          rho+=sourceRho*volume;gamma+=state[destinationGamma()+cell]*volume;
          pressure+=state[p.stateOffsets2.x+cell]*volume;
          velocity+=sourceVelocity*volume;momentumSum+=sourceRho*volume*sourceVelocity;
      }}}
      if(volumeSum>0.0){rho/=volumeSum;gamma/=volumeSum;pressure/=volumeSum;
        velocity=select(velocity/volumeSum,momentumSum/massSum,abs(massSum)>1e-12);}
    }else{
      let factor=candidate/accepted;let sx=cx/factor;let sy=cy/factor;let sz=cz/factor;
      let cell=first+sx+accepted*(sy+accepted*sz);rho=state[destinationDensity()+cell];
      gamma=state[destinationGamma()+cell];pressure=state[p.stateOffsets2.x+cell];
      let velocityAt=destinationCellVelocity()+4u*cell;
      velocity=vec3f(state[velocityAt],state[velocityAt+1u],state[velocityAt+2u]);
    }
    candidateState[candidateFieldIndex(0u,brick,local)]=rho;
    candidateState[candidateFieldIndex(1u,brick,local)]=gamma;
    candidateState[candidateFieldIndex(2u,brick,local)]=velocity.x;
    candidateState[candidateFieldIndex(3u,brick,local)]=velocity.y;
    candidateState[candidateFieldIndex(4u,brick,local)]=velocity.z;
    candidateState[candidateFieldIndex(5u,brick,local)]=pressure;
    let volume=candidateCellVolume(brick,candidate,local);
    let gammaContribution=gamma*volume;let momentumContribution=rho*volume*velocity;
    afterMass+=rho*volume;afterGamma+=gammaContribution;
    afterMomentum+=momentumContribution;afterGammaScale+=abs(gammaContribution);
    afterMomentumScale+=abs(momentumContribution);
  }
  transferMassBefore[lane]=beforeMass;transferMassAfter[lane]=afterMass;
  transferGammaDelta[lane]=afterGamma-beforeGamma;
  transferGammaScale[lane]=beforeGammaScale+afterGammaScale;
  transferMomentumXDelta[lane]=afterMomentum.x-beforeMomentum.x;
  transferMomentumYDelta[lane]=afterMomentum.y-beforeMomentum.y;
  transferMomentumZDelta[lane]=afterMomentum.z-beforeMomentum.z;
  transferMomentumXScale[lane]=beforeMomentumScale.x+afterMomentumScale.x;
  transferMomentumYScale[lane]=beforeMomentumScale.y+afterMomentumScale.y;
  transferMomentumZScale[lane]=beforeMomentumScale.z+afterMomentumScale.z;
  workgroupBarrier();var width=32u;loop{if(lane<width){
    transferMassBefore[lane]+=transferMassBefore[lane+width];
    transferMassAfter[lane]+=transferMassAfter[lane+width];
    transferGammaDelta[lane]+=transferGammaDelta[lane+width];
    transferGammaScale[lane]+=transferGammaScale[lane+width];
    transferMomentumXDelta[lane]+=transferMomentumXDelta[lane+width];
    transferMomentumYDelta[lane]+=transferMomentumYDelta[lane+width];
    transferMomentumZDelta[lane]+=transferMomentumZDelta[lane+width];
    transferMomentumXScale[lane]+=transferMomentumXScale[lane+width];
    transferMomentumYScale[lane]+=transferMomentumYScale[lane+width];
    transferMomentumZScale[lane]+=transferMomentumZScale[lane+width];
  }workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane!=0u){return;}
  if(candidateStatus!=1u){
    atomicStore(&activity[output+23u],select(0u,2u,candidateStatus==2u));return;
  }
  // Retirement deliberately discards only the separately receipted residue;
  // it has no active target field. Do not misclassify that lifecycle intent
  // as a failed conservative rerung transfer.
  let conservativeTransfer=acceptedActive&&candidateActive;
  let massError=select(0.0,transferMassAfter[0]-transferMassBefore[0],
    conservativeTransfer);
  let gammaError=select(0.0,transferGammaDelta[0],conservativeTransfer);
  let momentumError=select(vec3f(0.0),vec3f(
    transferMomentumXDelta[0],transferMomentumYDelta[0],transferMomentumZDelta[0]),
    conservativeTransfer);
  let massTolerance=max(1e-4,1e-6*abs(transferMassBefore[0]));
  let gammaTolerance=max(1e-3,1e-6*transferGammaScale[0]);
  let momentumTolerance=max(vec3f(1e-3),1e-6*vec3f(
    transferMomentumXScale[0],transferMomentumYScale[0],transferMomentumZScale[0]));
  let valid=finiteTransferValue(transferMassBefore[0])
    &&finiteTransferValue(transferMassAfter[0])&&finiteTransferValue(gammaError)
    &&all(momentumError==momentumError)&&abs(massError)<=massTolerance
    &&abs(gammaError)<=gammaTolerance
    &&all(abs(momentumError)<=momentumTolerance);
  atomicStore(&activity[output+16u],bitcast<u32>(transferMassBefore[0]));
  atomicStore(&activity[output+17u],bitcast<u32>(transferMassAfter[0]));
  atomicStore(&activity[output+18u],bitcast<u32>(massError));
  atomicStore(&activity[output+19u],bitcast<u32>(gammaError));
  atomicStore(&activity[output+20u],bitcast<u32>(momentumError.x));
  atomicStore(&activity[output+21u],bitcast<u32>(momentumError.y));
  atomicStore(&activity[output+22u],bitcast<u32>(momentumError.z));
  atomicStore(&activity[output+23u],select(2u,1u,valid));
  if(!valid){atomicOr(&activity[7],2u);}
}

@compute @workgroup_size(64)
fn transferCandidateCells(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let valid=wid.x<p.dispatch.w;let brick=select(0u,wid.x,valid);
  transferCandidateCellsWork(lid,brick,valid);
}

@compute @workgroup_size(64)
fn transferCandidateCellsFromTopologyDelta(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let listed=topologyDeltaLeafInvocation(wid.x);let valid=listed!=INVALID;
  let brick=select(0u,listed,valid);
  transferCandidateCellsWork(lid,brick,valid);
}

@compute @workgroup_size(64)
fn prepareCandidateFaceReceipts(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}let output=activityRecord(brick);
  for(var side=0u;side<6u;side+=1u){atomicStore(&activity[output+24u+side],0u);}
  atomicStore(&activity[output+30u],0u);
  let candidateStatus=atomicLoad(&activity[output+14u]);
  let scheduled=topologyPreparationScheduledAt(output);
  atomicStore(&activity[output+31u],select(select(0u,1u,candidateStatus==1u&&scheduled),2u,
    candidateStatus==2u&&scheduled));
}

fn boundaryCellLocal(axis:u32,positive:bool,resolution:u32,u:u32,v:u32)->u32{
  let normal=select(0u,resolution-1u,positive);
  let x=select(u,normal,axis==0u);
  let y=select(select(v,u,axis==0u),normal,axis==1u);
  let z=select(v,normal,axis==2u);
  return x+resolution*(y+resolution*z);
}

// Area-average every authoritative accepted normal flux into the candidate's
// exterior face patches. Each row is claimed once per incident brick, so the
// six integrated exterior fluxes are exact transfer receipts rather than a
// cell-velocity reconstruction.
fn transferCandidateFaceSide(lid:vec3u,brick:u32,side:u32,validBrick:bool){
  let lane=lid.x;
  let output=activityRecord(brick);
  // The old path area-averaged every face of every brick even when the
  // scheduler had prepared nothing. Candidate storage is isolated and a
  // scheduled brick recomputes all six receipts before it can commit, so this
  // uniform workgroup return removes only provably dead writes.
  if(lane==0u){candidateFaceScheduled=select(0u,
    select(0u,1u,validBrick&&topologyPreparationScheduledAt(output)),validBrick);
    candidateFaceConstructionActivation=select(0u,1u,
      validBrick&&scheduledConstructionActivationWithoutSlot(brick));
    candidateFaceActive=select(0u,1u,validBrick&&candidateBrickActive(brick));
    candidateFaceAcceptedResolution=atomicLoad(&activity[output+12u]);
    candidateFaceResolution=atomicLoad(&activity[output+13u]);}
  let scheduled=workgroupUniformLoad(&candidateFaceScheduled);
  if(scheduled==0u){return;}
  // Construction activation publishes a zero effective-velocity boundary.
  // No candidate face slot exists or is needed for that exact authority.
  if(workgroupUniformLoad(&candidateFaceConstructionActivation)!=0u){
    if(lane==0u){atomicStore(&activity[output+24u+side],0u);
      atomicStore(&activity[output+31u],1u);}
    return;
  }
  let candidate=workgroupUniformLoad(&candidateFaceResolution);
  if(workgroupUniformLoad(&candidateFaceActive)==0u){
    for(var faceSample=lane;faceSample<CANDIDATE_FACE_SAMPLES_PER_SIDE;faceSample+=64u){
      candidateState[candidateFieldIndex(6u+side,brick,faceSample)]=0.0;}
    if(lane==0u){atomicStore(&activity[output+24u+side],0u);
      atomicStore(&activity[output+31u],1u);}return;
  }
  let accepted=workgroupUniformLoad(&candidateFaceAcceptedResolution);
  let positive=(side&1u)!=0u;
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;
  let origin=vec3f(vec3u(bx,by,bz)*BRICK_FINE_RESOLUTION);
  let axis=side/2u;
  let tangent0=select(select(0u,0u,axis==2u),1u,axis==0u);
  let tangent1=select(2u,1u,axis==2u);
  let scale=f32(BRICK_FINE_RESOLUTION)/f32(candidate);
  var flux=0.0;var reconstructedFlux=0.0;
  for(var patchIndex=lane;patchIndex<candidate*candidate;patchIndex+=64u){
    var patchFlux=0.0;var patchArea=0.0;
    let patchU=patchIndex%candidate;let patchV=patchIndex/candidate;
    let sourceSpan=max(1u,accepted/candidate);
    let sourceU=(patchU*accepted)/candidate;
    let sourceV=(patchV*accepted)/candidate;
    for(var dv=0u;dv<sourceSpan;dv+=1u){for(var du=0u;du<sourceSpan;du+=1u){
      let boundary=sourceU+du+accepted*(sourceV+dv);
      let range=candidateFaceBoundaryRowRange(brick,accepted,side,boundary);
      for(var at=0u;at<range.y;at+=1u){
        let row=candidateFaceRow(range.x+at);if(!acceptedRowMember(row)){continue;}
        let center=rowCenter(row);
        let p0=min(candidate-1u,u32(max(0.0,floor(
          (center[tangent0]-origin[tangent0])/scale))));
        let p1=min(candidate-1u,u32(max(0.0,floor(
          (center[tangent1]-origin[tangent1])/scale))));
        if(patchIndex!=p0+candidate*p1){continue;}
        let rowAreaValue=rowArea(row);patchArea+=rowAreaValue;
        patchFlux+=state[destinationFaceVelocity()+row]*rowAreaValue
          *select(-1.0,1.0,positive);
      }
    }}
    let patchVelocity=select(0.0,patchFlux/patchArea,patchArea>0.0);
    candidateState[candidateFieldIndex(6u+side,brick,patchIndex)]=patchVelocity;
    flux+=patchFlux;reconstructedFlux+=patchVelocity*patchArea;
  }
  reduceA[lane]=flux;reduceB[lane]=reconstructedFlux;
  workgroupBarrier();var width=32u;loop{if(lane<width){
    reduceA[lane]+=reduceA[lane+width];reduceB[lane]+=reduceB[lane+width];
  }workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane!=0u){return;}let error=reduceB[0]-reduceA[0];
  atomicStore(&activity[output+24u+side],bitcast<u32>(error));
  atomicMax(&activity[output+30u],bitcast<u32>(abs(error)));
  if(atomicLoad(&activity[output+14u])==1u){
    let valid=finiteTransferValue(error)&&abs(error)<=max(1e-4,1e-6*abs(reduceA[0]));
    if(!valid){atomicStore(&activity[output+31u],2u);atomicOr(&activity[7],4u);}
  }
}

@compute @workgroup_size(64)
fn transferCandidateFaces(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  transferCandidateFaceSide(lid,wid.x,wid.y,wid.x<p.dispatch.w&&wid.y<6u);
}

@compute @workgroup_size(64)
fn transferCandidateFacesFromTopologyDelta(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=topologyDeltaLeafInvocation(wid.x);let valid=brick!=INVALID;
  for(var side=0u;side<6u;side+=1u){
    transferCandidateFaceSide(lid,select(0u,brick,valid),side,valid);
  }
}

fn publishCandidateTopologyDeltaWork(lid:vec3u,brick:u32,validBrick:bool){
  let lane=lid.x;let output=activityRecord(brick);
  if(lane==0u){candidatePublicationScheduled=select(0u,1u,validBrick
    &&topologyPreparationScheduledAt(output)
    &&atomicLoad(&activity[output+23u])==1u
    &&atomicLoad(&activity[output+31u])==1u
    &&atomicLoad(&topologyArena[topologyWorklistBase()+3u])==2u
    &&atomicLoad(&topologyArena[topologyWorklistBase()+1u])
      ==atomicLoad(&activity[output+36u]));
    candidatePublicationAccepted=atomicLoad(&activity[output+12u]);
    candidatePublicationResolution=atomicLoad(&activity[output+13u]);
    candidatePublicationAcceptedActive=select(0u,1u,brickActive(brick));
    candidatePublicationActive=select(0u,1u,candidateBrickActive(brick));
    candidatePublicationConstructionActivation=select(0u,1u,
      validBrick&&scheduledConstructionActivationWithoutSlot(brick));}
  if(workgroupUniformLoad(&candidatePublicationScheduled)==0u){return;}
  let accepted=workgroupUniformLoad(&candidatePublicationAccepted);
  let candidate=workgroupUniformLoad(&candidatePublicationResolution);
  let acceptedActive=workgroupUniformLoad(&candidatePublicationAcceptedActive)!=0u;
  let candidateActive=workgroupUniformLoad(&candidatePublicationActive)!=0u;
  let acceptedRange=templateBrickCellRange(brick,accepted);
  let candidateRange=templateBrickCellRange(brick,candidate);
  let constructionActivation=
    workgroupUniformLoad(&candidatePublicationConstructionActivation)!=0u;
  let publishCount=select(acceptedRange.y,candidateRange.y,candidateActive);
  for(var local=lane;local<publishCount;local+=64u){
    let cell=select(acceptedRange.x+local,candidateRange.x+local,candidateActive);
    var rho=0.0;var gamma=1.0;var vx=0.0;var vy=0.0;var vz=0.0;var pressure=0.0;
    if(candidateActive&&!constructionActivation){
      rho=candidateState[candidateFieldIndex(0u,brick,local)];
      gamma=candidateState[candidateFieldIndex(1u,brick,local)];
      vx=candidateState[candidateFieldIndex(2u,brick,local)];
      vy=candidateState[candidateFieldIndex(3u,brick,local)];
      vz=candidateState[candidateFieldIndex(4u,brick,local)];
      pressure=candidateState[candidateFieldIndex(5u,brick,local)];
    }
    state[p.stateOffsets0.x+cell]=rho;state[p.stateOffsets0.y+cell]=rho;
    state[p.stateOffsets0.z+cell]=gamma;state[p.stateOffsets0.w+cell]=gamma;
    for(var slot=0u;slot<2u;slot+=1u){let velocity=select(
      p.stateOffsets1.x,p.stateOffsets1.y,slot==1u)+4u*cell;
      state[velocity]=vx;state[velocity+1u]=vy;state[velocity+2u]=vz;state[velocity+3u]=0.0;
    }
    state[p.stateOffsets2.x+cell]=pressure;
    state[p.stateOffsets2.y+cell]=0.0;state[p.stateOffsets2.z+cell]=0.0;
    state[p.stateOffsets2.w+cell]=select(0.0,1.0,
      rho/max(cellOpenFraction(cell),1e-6)>=CM12_LIQUID_ISOVALUE);
    state[p.stateOffsets3.y+cell]=0.0;state[p.stateOffsets3.z+cell]=0.0;
    state[p.stateOffsets3.w+cell]=0.0;state[p.stateOffsets4.x+cell]=0.0;
    state[p.stateOffsets4.y+cell]=0.0;
    let at=destinationCellVelocity()+4u*cell;
    ${phase1QATransferredVelocityPublish}
  }
  workgroupBarrier();if(lane!=0u){return;}
  atomicStore(&activity[output+10u],select(0u,1u,candidateActive));
  atomicStore(&activity[output+12u],candidate);
  if(acceptedActive!=candidateActive){
    if(candidateActive){atomicAdd(&activity[8],1u);atomicAdd(&activity[9],1u);
      atomicAdd(&activity[11],candidateRange.y);atomicStore(&activity[output+34u],0u);
    }else{atomicSub(&activity[8],1u);atomicSub(&activity[11],acceptedRange.y);
      atomicStore(&activity[output+34u],atomicLoad(&activity[output+16u]));}
    atomicAdd(&activity[10],1u);
  }else if(candidateActive&&acceptedRange.y!=candidateRange.y){
    if(candidateRange.y>acceptedRange.y){atomicAdd(&activity[11],candidateRange.y-acceptedRange.y);
    }else{atomicSub(&activity[11],acceptedRange.y-candidateRange.y);}
  }
  if(acceptedActive){if(accepted==BRICK_FINE_RESOLUTION){atomicSub(&activity[19],1u);}
    else{atomicSub(&activity[20],1u);}}
  if(candidateActive){if(candidate==BRICK_FINE_RESOLUTION){atomicAdd(&activity[19],1u);}
    else{atomicAdd(&activity[20],1u);}}
  let recoveryState=atomicLoad(&activity[output+38u]);
  let recoveryFloor=recoveryState&31u;let recoveryLocked=
    (recoveryState&ACTIVITY_RECOVERY_LOCK)!=0u;
  if(candidate>accepted){atomicStore(&activity[output+38u],
    recoveryFloor|ACTIVITY_RECOVERY_LOCK);
  }else if(!recoveryLocked&&candidate<recoveryFloor){
    atomicStore(&activity[output+38u],candidate);}
  atomicStore(&activity[output+2u],0u);atomicStore(&activity[output+11u],
    atomicLoad(&activity[0]));atomicAdd(&activity[17],1u);
}

@compute @workgroup_size(64)
fn publishCandidateTopologyDelta(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  publishCandidateTopologyDeltaWork(lid,wid.x,wid.x<p.dispatch.w);
}

@compute @workgroup_size(64)
fn publishCandidateTopologyDeltaFromWorklist(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=topologyDeltaLeafInvocation(wid.x);
  publishCandidateTopologyDeltaWork(
    lid,select(0u,brick,brick!=INVALID),brick!=INVALID);
}

fn candidateTransferredVelocity(cell:u32)->vec3f{
  let brick=cellBrick(cell);let resolution=scheduledBrickResolution(brick);
  let range=templateBrickCellRange(brick,resolution);
  if(topologyPreparationScheduledAt(activityRecord(brick))
    &&candidateBrickActive(brick)&&cell>=range.x&&cell<range.x+range.y){
    if(scheduledConstructionActivationWithoutSlot(brick)){return vec3f(0.0);}
    let local=cell-range.x;return vec3f(
      candidateState[candidateFieldIndex(2u,brick,local)],
      candidateState[candidateFieldIndex(3u,brick,local)],
      candidateState[candidateFieldIndex(4u,brick,local)]);}
  let at=destinationCellVelocity()+4u*cell;
  return vec3f(state[at],state[at+1u],state[at+2u]);
}

// Independent preflip receipt: reconstruct candidate row values from isolated
// candidateState plus unchanged accepted cells, but author no stable face.
@compute @workgroup_size(64)
fn validateCandidateShadowFaces(@builtin(global_invocation_id)gid:vec3u){
  if(atomicLoad(&activity[16])==0u){return;}
  let invocation=gid.x;if(invocation>=shadowTemplateRowCount()){return;}
  let row=shadowTemplateRowInvocation(invocation);if(row==INVALID){return;}
  let requirements=rowRequirementOffset(row);let requirementCount=ta(requirements);
  var changed=false;for(var at=0u;at<requirementCount;at+=1u){
    let brick=ta(requirements+1u+at)>>TEMPLATE_CELL_RESOLUTION_BITS;
    changed=changed||topologyPreparationScheduledAt(activityRecord(brick));
  }
  if(!changed){return;}let axis=rowAxis(row);var velocity=0.0;var weight=0.0;
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){let cell=termCell(term);
    let candidateVelocity=candidateTransferredVelocity(cell);
    let w=abs(termCoefficient(term));velocity+=w*candidateVelocity[axis];weight+=w;
  }
  velocity=select(0.0,velocity/weight,weight>0.0);
  if(velocity!=velocity||abs(velocity)>=3.4e38){
    for(var at=0u;at<requirementCount;at+=1u){let brick=
      ta(requirements+1u+at)>>TEMPLATE_CELL_RESOLUTION_BITS;
      if(topologyPreparationScheduledAt(activityRecord(brick))){
        atomicStore(&activity[activityRecord(brick)+31u],2u);}}
  }
}

// Postflip stable publication of the exact row value validated above.
@compute @workgroup_size(64)
fn publishCandidateShadowFaces(@builtin(global_invocation_id)gid:vec3u){
  let invocation=gid.x;if(invocation>=shadowTemplateRowCount()){return;}
  let row=shadowTemplateRowInvocation(invocation);if(row==INVALID){return;}
  let requirements=rowRequirementOffset(row);let requirementCount=ta(requirements);
  var changed=false;var committed=false;
  for(var at=0u;at<requirementCount;at+=1u){let brick=
    ta(requirements+1u+at)>>TEMPLATE_CELL_RESOLUTION_BITS;
    let scheduled=topologyPreparationScheduledAt(activityRecord(brick));
    changed=changed||scheduled;committed=committed||(scheduled
      &&atomicLoad(&activity[activityRecord(brick)+36u])
        ==atomicLoad(&topologyArena[topologyWorklistBase()+1u])
      &&atomicLoad(&topologyArena[topologyWorklistBase()+3u])==2u);}
  if(!changed||!committed){return;}let axis=rowAxis(row);var velocity=0.0;var weight=0.0;
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){let cell=termCell(term);
    let at=destinationCellVelocity()+4u*cell;let w=abs(termCoefficient(term));
    velocity+=w*state[at+axis];weight+=w;}
  velocity=select(0.0,velocity/weight,weight>0.0);
  state[p.stateOffsets1.z+row]=velocity;state[p.stateOffsets1.w+row]=velocity;
}

@compute @workgroup_size(1)
fn validateAndAuthorizeShadowTopology(){
  let prepared=atomicLoad(&activity[16]);let base=topologyWorklistBase();
  if(prepared==0u){
    // A genuinely empty candidate still closes the effects transaction. This
    // is O(1): every fallible producer receipt must prove an exact empty set
    // before the no-fail success latch is published for the unchanged
    // accepted generation. Any non-empty effect without a compact topology
    // delta is a missing-authority fault, never a reason to scan the world.
    if(!authorizeEmptySparseCM12CandidateEffectsNoFail(
        atomicLoad(&topologyArena[base]))){
      atomicStore(&activity[21],1u);atomicStore(&topologyArena[base+3u],3u);return;
    }
    _=ptrSealTopologyJournal(atomicLoad(&topologyArena[ptrH_COVERED_PRODUCER_RECEIPTS]),
      atomicLoad(&topologyArena[base]));
    atomicStore(&activity[13],(atomicLoad(&activity[13])+p.topologyScheduling.x)
      %max(1u,p.dispatch.w));atomicStore(&topologyArena[base+3u],0u);return;
  }
  var valid=shadowTemplateCellCount()<=atomicLoad(&topologyArena[base+6u])
    &&shadowTemplateRowCount()<=atomicLoad(&topologyArena[base+7u])
    &&ptrResidentTopologyDeltaReady()
    &&residentTopologyEffectsPreflightReady();
  ${internedBoundaryCommitReceipt}
  let leaves=acceptedLeafManifestBase();let deltaCount=atomicLoad(&topologyArena[leaves+10u]);
  for(var index=0u;index<deltaCount;index+=1u){let brick=topologyDeltaLeafInvocation(index);
    if(brick==INVALID){valid=false;continue;}let output=activityRecord(brick);
    valid=valid&&atomicLoad(&activity[output+23u])==1u
      &&atomicLoad(&activity[output+31u])==1u
      &&atomicLoad(&activity[output+36u])==atomicLoad(&topologyArena[base+1u]);
  }
  if(!valid){atomicStore(&activity[21],1u);
    atomicStore(&topologyArena[base+3u],3u);return;}
  // Authorization is candidate-only. The selector and every accepted
  // worklist/generation header remain unchanged until bounded publication
  // completes in later dispatches.
  atomicStore(&activity[17],0u);
  tfxAuthorize();
  atomicStore(&topologyArena[base+3u],2u);
}

@compute @workgroup_size(1)
fn finalizeAuthorizedShadowTopology(){
  let base=topologyWorklistBase();let leaves=acceptedLeafManifestBase();
  if(atomicLoad(&topologyArena[base+3u])!=2u){return;}
  let slot=shadowTopologySlot();
  atomicStore(&topologyArena[base+4u],shadowTemplateCellCount());
  atomicStore(&topologyArena[base+5u],shadowTemplateRowCount());
  atomicStore(&topologyArena[leaves+4u],atomicLoad(&topologyArena[leaves+7u]));
  atomicStore(&topologyArena[leaves+5u],1u);
  atomicStore(&topologyArena[leaves+6u],1u);
  for(var at=0u;at<3u;at+=1u){
    atomicStore(&topologyArena[base+8u+at],atomicLoad(&topologyArena[base+20u+at]));
    atomicStore(&topologyArena[base+11u+at],atomicLoad(&topologyArena[base+23u+at]));
  }
  atomicStore(&topologyArena[base],atomicLoad(&topologyArena[base+1u]));
  atomicStore(&topologyArena[base+3u],0u);
  atomicStore(&activity[12],atomicLoad(&topologyArena[base]));
  atomicStore(&activity[13],(atomicLoad(&activity[13])+p.topologyScheduling.x)
    %max(1u,p.dispatch.w));
  // Sole publication point: all membership, fields, faces, cache/journal
  // effects and accepted headers are complete before this final selector flip.
  atomicStore(&topologyArena[base+2u],slot);
}

// Publish only the directional free-surface stencil and swept receivers from
// the immutable activity snapshot. Compare-exchange makes publication
// single-writer; all following frame dispatches observe the active bit.
@compute @workgroup_size(64)
fn activateSweptReceivers(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||brickActive(brick)){return;}
  if(!brickRequestedAsReceiver(brick)){return;}
  let output=activityRecord(brick);
  // Intent only. Accepted membership, stable fields, cache ownership,
  // receipts, counters and journals remain untouched until the shared flip.
  atomicStore(&activity[output+8u],
    applySparseCM12RefinementRegionBounds(brick,BRICK_FINE_RESOLUTION));
  atomicStore(&activity[output+9u],1u|ACTIVITY_LIFECYCLE_CHANGED);
  setCandidateBrickActiveAt(output,true);
}

// Retire every non-feature brick outside the directional interface stencil and
// swept receiver mask. The discarded mass is recorded per brick and is bounded
// by the surface-density floor times one 8^3 brick volume; thin dilute sheets
// remain explicit represented features and do not take this path.
@compute @workgroup_size(64)
fn retireUnsupportedEmptyBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||!brickActive(brick)){return;}
  let output=activityRecord(brick);
  if((atomicLoad(&activity[output+1u])&64u)!=0u){return;}
  let record=p.topologyOffsets2.z+2u*brick;let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let z=key/xy;
  let remainder=key-z*xy;let y=remainder/brickDimensions.x;
  let x=remainder-y*brickDimensions.x;let coordinate=vec3i(i32(x),i32(y),i32(z));
  for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){
    for(var dx=-1;dx<=1;dx+=1){if(dx==0&&dy==0&&dz==0){continue;}
      let neighborCoordinate=coordinate+vec3i(dx,dy,dz);
      if(any(neighborCoordinate<vec3i(0))
        ||any(neighborCoordinate>=vec3i(brickDimensions))){continue;}
      let neighborKey=u32(neighborCoordinate.x)+brickDimensions.x
        *(u32(neighborCoordinate.y)+brickDimensions.y*u32(neighborCoordinate.z));
      let neighbor=brickDirectoryLookup(neighborKey);if(neighbor==INVALID){continue;}
      let bit=u32(1-dx)+3u*u32(1-dy)+9u*u32(1-dz);
      if((atomicLoad(&activity[activityRecord(neighbor)+32u])&(1u<<bit))!=0u){return;}
    }
  }}
  // Intent only. Publication records residue, invalidates caches/journals and
  // updates counters after the shared selector flip.
  let resolution=acceptedBrickResolution(brick);
  atomicStore(&activity[output+8u],resolution);
  atomicOr(&activity[output+9u],ACTIVITY_LIFECYCLE_CHANGED);
  setCandidateBrickActiveAt(output,false);
}

// Retirement now participates in the shadow transaction before candidate
// validation. Seal the exact final producer count after that coupled commit so
// the next pressure epoch can fail closed on a real coverage mismatch.
@compute @workgroup_size(1)
fn sealSparseCM12PressureTopologyJournal(){
  if(atomicLoad(&topologyArena[topologyWorklistBase()+3u])!=0u){return;}
  _=ptrSealTopologyJournal(
    atomicLoad(&topologyArena[ptrH_COVERED_PRODUCER_RECEIPTS]),
    ptrTopologyGeneration());
}

${framePlanLayout && framePlanPresentationLayout ? /* wgsl */ `
const PRESENTATION_FRAME_PLAN_STAGE:u32=5u;
const PRESENTATION_FRAME_PLAN_STAGE_BIT:u32=1u<<PRESENTATION_FRAME_PLAN_STAGE;
const PRESENTATION_FRAME_PLAN_CAUSES:u32=${(
  SPARSE_CM12_DIRTY_CAUSE_BIT.topologyCreated
  | SPARSE_CM12_DIRTY_CAUSE_BIT.topologyRetired
  | SPARSE_CM12_DIRTY_CAUSE_BIT.phaseCrossing
  | SPARSE_CM12_DIRTY_CAUSE_BIT.densityChanged
  | SPARSE_CM12_DIRTY_CAUSE_BIT.boundarySource
  | SPARSE_CM12_DIRTY_CAUSE_BIT.dependencyClosure
  | SPARSE_CM12_DIRTY_CAUSE_BIT.pageActivated
  | SPARSE_CM12_DIRTY_CAUSE_BIT.pageRetired
) >>> 0}u;
const PRESENTATION_FRAME_PLAN_DIRECT_CAUSES:u32=${(
  SPARSE_CM12_DIRTY_CAUSE_BIT.topologyCreated
  | SPARSE_CM12_DIRTY_CAUSE_BIT.topologyRetired
  | SPARSE_CM12_DIRTY_CAUSE_BIT.phaseCrossing
  | SPARSE_CM12_DIRTY_CAUSE_BIT.densityChanged
  | SPARSE_CM12_DIRTY_CAUSE_BIT.boundarySource
  | SPARSE_CM12_DIRTY_CAUSE_BIT.pageActivated
  | SPARSE_CM12_DIRTY_CAUSE_BIT.pageRetired
) >>> 0}u;

@compute @workgroup_size(64)
fn populateSparseCM12PresentationFramePlan(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let brick=wid.x;if(brick>=p.dispatch.w){return;}
  let key=topology[p.topologyOffsets2.z+2u*brick+1u];
  if(lane==0u){
    cm12FramePlanSetNextBrickLogicalKey(brick,key);
    if(brick==0u){
      let frameGeneration=atomicLoad(&activity[0]);
      cm12FramePlanSetNextTopologyGeneration(
        atomicLoad(&topologyArena[topologyWorklistBase()]));
      cm12FramePlanSetNextFrameAuthority(frameGeneration,frameGeneration&1u);
    }
  }
  if(lane>=ACTIVITY_TILES_PER_BRICK){return;}
  let bootstrap=cm12FramePlanCandidateGeneration()==0u;
  let injected=p.injectionCenter.w>0.5&&injectionReachesBrick(brick);
  // Stages 2..4 have not yet cut over to FPL scheduling. Publish their actual
  // global accepted-cell/row blast radius. This is conservative work, not an
  // FPL generation/provenance fault.
  // Stage 1 reads FSM1 directly; ACT1 carries only its brick activity census.
  // Stage 0 is VEX-owned and may only come from its construction bootstrap or
  // its dynamic root/closure publication.
  let previousTopology=atomicLoad(&activity[ACTIVITY_BRICK_TOPOLOGY+brick]);
  let previouslyActive=previousTopology!=INVALID&&(previousTopology&0x80000000u)!=0u;
  if(previouslyActive&&p.injectionCenter.w<=0.5){
    cm12FramePlanMarkOwnedNextTile(brick,lane,0x1cu,0u,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.coefficientChanged}u,0u,0u);
  }
  let stable=key*ACTIVITY_TILES_PER_BRICK+lane;
  let tile=cm12TeiSpatialTile(stable,acceptedTopologySlot());
  let scalarChanged=tile.packetId!=INVALID&&((fsm1Changed(tile.packetId)
    &tile.laneMask).x|(fsm1Changed(tile.packetId)&tile.laneMask).y)!=0u;
  let topologyChanged=previousTopology!=incrementalActivityTopologyState(brick);
  let dynamicBrick=incrementalActivityBrickDirty(brick);
  let page=cm12FppLoad(cm12FppBrickPages+brick);
  let pageNeedsActivation=page<arrayLength(&fineMetadata)/4u
    &&fineMetadata[4u*page+2u]!=1u;
  let scheduled=bootstrap||injected||pageNeedsActivation||scalarChanged
    ||topologyChanged||dynamicBrick;
  if(!scheduled){return;}
  var origin=select(0u,${SPARSE_CM12_DIRTY_CAUSE_BIT.densityChanged}u,scalarChanged);
  var inherited=select(0u,${SPARSE_CM12_DIRTY_CAUSE_BIT.dependencyClosure}u,
    dynamicBrick&&!scalarChanged);
  if(topologyChanged){origin|=${(SPARSE_CM12_DIRTY_CAUSE_BIT.topologyCreated
    | SPARSE_CM12_DIRTY_CAUSE_BIT.topologyRetired) >>> 0}u;}
  if(bootstrap){origin=${(SPARSE_CM12_DIRTY_CAUSE_BIT.topologyCreated
    | SPARSE_CM12_DIRTY_CAUSE_BIT.densityChanged) >>> 0}u;inherited=0u;}
  if(injected){origin|=${(SPARSE_CM12_DIRTY_CAUSE_BIT.phaseCrossing
    | SPARSE_CM12_DIRTY_CAUSE_BIT.densityChanged) >>> 0}u;}
  if(pageNeedsActivation){origin|=${SPARSE_CM12_DIRTY_CAUSE_BIT.pageActivated}u;}
  let direct=(origin&PRESENTATION_FRAME_PLAN_DIRECT_CAUSES)!=0u;
  if(!direct){inherited|=${SPARSE_CM12_DIRTY_CAUSE_BIT.dependencyClosure}u;}
  let depth=select(2u,0u,direct);
  cm12FramePlanMarkOwnedNextTile(brick,lane,
    select(0u,PRESENTATION_FRAME_PLAN_STAGE_BIT,direct),
    select(PRESENTATION_FRAME_PLAN_STAGE_BIT,0u,direct),origin,inherited,
    depth<<(4u*PRESENTATION_FRAME_PLAN_STAGE));
}

@compute @workgroup_size(64)
fn markSparseCM12GlobalFramePlanReceipts(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let brick=wid.x;if(brick>=p.dispatch.w||lane>=ACTIVITY_TILES_PER_BRICK){return;}
  let previousTopology=atomicLoad(&activity[ACTIVITY_BRICK_TOPOLOGY+brick]);
  if(previousTopology==INVALID||(previousTopology&0x80000000u)==0u
    ||p.injectionCenter.w>0.5){return;}
  // Stage 0 was retired with VEX planning and has no scheduled work.
  for(var stage=1u;stage<PRESENTATION_FRAME_PLAN_STAGE;stage+=1u){
    // The compact mass plan intentionally omits clean tiles. The former global
    // receipt called MarkCurrentTileExecuted on those omissions,
    // converting successful sparsity into a missing-coverage brick fault and
    // making every overlay render magenta. Certify only work the accepted FPL
    // actually scheduled; pressure still covers its complete compatibility
    // domain because every one of those tiles is present in stage 4.
    if(cm12FramePlanCurrentTileScheduled(brick,lane,stage)){
      cm12FramePlanMarkCurrentTileExecuted(brick,lane,stage);
    }
  }
}

var<workgroup>cm12PresentationBrick:u32;
var<workgroup>cm12PresentationPage:u32;
var<workgroup>cm12PresentationPageOrigin:vec3u;
var<workgroup>cm12PresentationBrickOrigin:vec3u;
var<workgroup>cm12PresentationSampleScale:u32;
var<workgroup>cm12PresentationResolution:u32;
var<workgroup>cm12PresentationScale:u32;
var<workgroup>cm12PresentationCacheFirst:vec3i;
var<workgroup>cm12PresentationCacheDimensions:vec3u;
var<workgroup>cm12PresentationCacheFits:u32;
var<workgroup>cm12PresentationWet:u32;
var<workgroup>cm12PresentationResolvedFeature:u32;
var<workgroup>cm12PresentationUniformPhi:f32;
var<workgroup>cm12PresentationDensityOffset:u32;

fn cm12PresentationSourceBrick(source:u32)->u32{
  if(PRESENTATION_PAGES_PER_AXIS==4u){
    let macroPage=(source&0x80000000u)!=0u;
    return select((source>>6u)&0xffffffu,source&0xffffffu,macroPage);
  }
  return (source>>3u)&0xffffffu;
}
fn cm12PresentationSourceSpan(source:u32)->u32{
  if(PRESENTATION_PAGES_PER_AXIS==4u){
    return select(1u,1u<<((source>>24u)&31u),(source&0x80000000u)!=0u);
  }
  return 1u<<(source>>27u);
}
fn cm12PresentationPageMatches(page:u32,brick:u32,logicalKey:u32,
 topologyGeneration:u32)->bool{
  let pageCount=arrayLength(&fineMetadata)/4u;if(page>=pageCount||brick>=p.dispatch.w){return false;}
  let source=fineMetadata[4u*page+3u];
  return cm12PresentationSourceBrick(source)==brick
    &&cm12PresentationSourceSpan(source)==brickSpan(brick)
    &&fineMetadata[4u*page]==page&&fineMetadata[4u*page+1u]==logicalKey
    &&fineMetadata[4u*page+2u]<=1u
    &&topology[p.topologyOffsets2.z+2u*brick+1u]==logicalKey
    &&topologyGeneration==atomicLoad(&topologyArena[topologyWorklistBase()]);
}
fn cm12PresentationPreparePage(brick:u32,page:u32,lane:u32,
 generation:u32,causeMask:u32)->u32{
  _=generation;_=causeMask;
  if(lane==0u){
    let source=fineMetadata[4u*page+3u];var octant=0u;var span=1u;
    if(PRESENTATION_PAGES_PER_AXIS==4u){
      let macroPage=(source&0x80000000u)!=0u;
      octant=select(source&63u,0u,macroPage);
      span=select(1u,1u<<((source>>24u)&31u),macroPage);
    }else{octant=source&7u;span=1u<<(source>>27u);}
    let brickKey=topology[p.topologyOffsets2.z+2u*brick+1u];
    let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
      /BRICK_FINE_RESOLUTION;
    let brickXY=brickDimensions.x*brickDimensions.y;let brickZ=brickKey/brickXY;
    let brickRemainder=brickKey-brickZ*brickXY;
    let brickY=brickRemainder/brickDimensions.x;
    let brickX=brickRemainder-brickY*brickDimensions.x;
    let pageZ=octant/(PRESENTATION_PAGES_PER_AXIS*PRESENTATION_PAGES_PER_AXIS);
    let pageRemainder=octant-pageZ*PRESENTATION_PAGES_PER_AXIS*PRESENTATION_PAGES_PER_AXIS;
    let pageY=pageRemainder/PRESENTATION_PAGES_PER_AXIS;
    let pageX=pageRemainder-pageY*PRESENTATION_PAGES_PER_AXIS;
    let pageOffset=vec3u(pageX,pageY,pageZ)*PRESENTATION_PAGE_RESOLUTION;
    let sampleScale=select(1u,
      BRICK_FINE_RESOLUTION*span/PRESENTATION_PAGE_RESOLUTION,span>1u);
    let brickOrigin=vec3u(brickX,brickY,brickZ)*BRICK_FINE_RESOLUTION;
    let pageOrigin=brickOrigin+pageOffset;let resolution=acceptedBrickResolution(brick);
    let scale=BRICK_FINE_RESOLUTION*span/resolution;let scaleF=f32(scale);
    let firstShifted=(vec3f(pageOrigin)+vec3f(0.5))/scaleF-vec3f(0.5);
    let lastQ=pageOrigin+vec3u(PRESENTATION_PAGE_RESOLUTION-1u)*sampleScale;
    let lastShifted=(vec3f(lastQ)+vec3f(0.5))/scaleF-vec3f(0.5);
    let cacheFirst=vec3i(floor(firstShifted));
    let cacheDimensions=vec3u(vec3i(floor(lastShifted))-cacheFirst)+vec3u(2u);
    let cacheCount=cacheDimensions.x*cacheDimensions.y*cacheDimensions.z;
    cm12PresentationBrick=brick;cm12PresentationPage=page;
    cm12PresentationPageOrigin=pageOrigin;cm12PresentationBrickOrigin=brickOrigin;
    cm12PresentationSampleScale=sampleScale;cm12PresentationResolution=resolution;
    cm12PresentationScale=scale;cm12PresentationCacheFirst=cacheFirst;
    cm12PresentationCacheDimensions=cacheDimensions;
    cm12PresentationCacheFits=select(0u,1u,cacheCount<=PRESENTATION_CACHE_CAPACITY);
    cm12PresentationWet=select(0u,1u,brick<p.dispatch.w
      &&(atomicLoad(&activity[activityRecord(brick)+1u])&64u)!=0u);
    let activityOutput=activityRecord(brick);
    let activityReasons=atomicLoad(&activity[activityOutput+1u]);
    // Surface, thin-fluid and cut-boundary bricks retain exact samples. A
    // resolved feature-free brick has one represented phase, so its interior
    // sign is completely described by the already-reduced brick mean.
    // Generation zero has only the cheap wet classification below; its mean
    // and feature bits do not exist until the first activity census. Injection
    // similarly reclassifies wetness without recomputing those reductions.
    // Keep both paths exact instead of mistaking their zeroed/stale mean for a
    // feature-free air brick. Ordinary frames may use the uniform bulk value.
    let uniformBulkReady=atomicLoad(&activity[0])!=0u&&p.injectionCenter.w<=0.5;
    cm12PresentationResolvedFeature=select(1u,0u,EXP_PRESENTATION_UNIFORM_BULK
      &&uniformBulkReady&&(activityReasons&(1u|256u|512u))==0u);
    let bulkLiquid=cm12PresentationWet!=0u
      &&activityF32(activityOutput+4u)>=CM12_LIQUID_ISOVALUE;
    cm12PresentationUniformPhi=select(4.0*p.frame.y,-4.0*p.frame.y,bulkLiquid);
    cm12PresentationDensityOffset=select(p.stateOffsets0.x,p.stateOffsets0.y,
      cm12FramePlanAcceptedParity()!=0u);
  }
  workgroupBarrier();
  let cacheCount=cm12PresentationCacheDimensions.x*cm12PresentationCacheDimensions.y
    *cm12PresentationCacheDimensions.z;
  if(cm12PresentationResolvedFeature!=0u&&cm12PresentationScale>1u
    &&cm12PresentationWet!=0u
    &&cm12PresentationCacheFits!=0u){
    let coarseDimensions=max(vec3i(1),vec3i(p.dimensions.xyz)/i32(cm12PresentationScale));
    for(var cacheIndex=lane;cacheIndex<cacheCount;cacheIndex+=64u){
      let cacheZ=cacheIndex/(cm12PresentationCacheDimensions.x
        *cm12PresentationCacheDimensions.y);
      let cacheRemainder=cacheIndex-cacheZ*cm12PresentationCacheDimensions.x
        *cm12PresentationCacheDimensions.y;
      let cacheY=cacheRemainder/cm12PresentationCacheDimensions.x;
      let cacheX=cacheRemainder-cacheY*cm12PresentationCacheDimensions.x;
      let coarse=clamp(cm12PresentationCacheFirst
        +vec3i(i32(cacheX),i32(cacheY),i32(cacheZ)),vec3i(0),coarseDimensions-vec3i(1));
      presentationDensityCache[cacheIndex]=restrictedPresentationDensityAt(
        coarse*i32(cm12PresentationScale),i32(cm12PresentationScale),
        cm12PresentationDensityOffset);
    }
  }
  return 0u;
}
fn cm12PresentationExactSample(brick:u32,page:u32,tile:u32,sample:u32,
 generation:u32,causeMask:u32)->vec2u{
  _=generation;_=causeMask;_=brick;
  let localIndex=cm12FppLocalIndex(tile,sample);
  let localZ=localIndex/(PRESENTATION_PAGE_RESOLUTION*PRESENTATION_PAGE_RESOLUTION);
  let localRemainder=localIndex-localZ*PRESENTATION_PAGE_RESOLUTION*PRESENTATION_PAGE_RESOLUTION;
  let localY=localRemainder/PRESENTATION_PAGE_RESOLUTION;
  let localX=localRemainder-localY*PRESENTATION_PAGE_RESOLUTION;
  let local=vec3u(localX,localY,localZ);
  let q=cm12PresentationPageOrigin+local*cm12PresentationSampleScale;
  var phi=cm12PresentationUniformPhi;
  if(!all(q<p.dimensions.xyz)){phi=4.0*p.frame.y;}
  else if(cm12PresentationResolvedFeature!=0u&&cm12PresentationWet!=0u
    &&insideEmbeddedBoundary(vec3f(q)+vec3f(0.5))){
    if(cm12PresentationScale==1u){
      let range=templateBrickCellRange(cm12PresentationBrick,cm12PresentationResolution);
      let valid=min(p.dimensions.xyz-cm12PresentationBrickOrigin,vec3u(BRICK_FINE_RESOLUTION));
      let localCell=q-cm12PresentationBrickOrigin;
      let cell=range.x+localCell.x+valid.x*(localCell.y+valid.y*localCell.z);
      if(cell<range.x+range.y){
        phi=presentationPhiAt(cell,cm12PresentationDensityOffset);
      }
    }else{
      let scaleF=f32(cm12PresentationScale);
      let shifted=(vec3f(q)+vec3f(0.5))/scaleF-vec3f(0.5);
      let lower=vec3i(floor(shifted));let fraction=fract(shifted);var rho=0.0;
      let coarseDimensions=max(vec3i(1),vec3i(p.dimensions.xyz)/i32(cm12PresentationScale));
      for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
        let offset=vec3i(dx,dy,dz);var density=0.0;
        if(cm12PresentationCacheFits!=0u){
          let cache=vec3u(lower+offset-cm12PresentationCacheFirst);
          let cacheIndex=cache.x+cm12PresentationCacheDimensions.x
            *(cache.y+cm12PresentationCacheDimensions.y*cache.z);
          density=presentationDensityCache[cacheIndex];
        }else{
          let coarse=clamp(lower+offset,vec3i(0),coarseDimensions-vec3i(1));
          density=restrictedPresentationDensityAt(coarse*i32(cm12PresentationScale),
            i32(cm12PresentationScale),cm12PresentationDensityOffset);
        }
        let wx=select(1.0-fraction.x,fraction.x,dx==1);
        let wy=select(1.0-fraction.y,fraction.y,dy==1);
        let wz=select(1.0-fraction.z,fraction.z,dz==1);
        rho+=wx*wy*wz*density;
      }}}
      phi=(CM12_LIQUID_ISOVALUE-rho)*4.0*p.frame.y;
    }
  }
  let flags=1u|select(0u,16u,phi<0.0);
  return vec2u((pack2x16float(vec2f(phi,0.0))&0xffffu)|(flags<<16u),0u);
}
fn cm12PresentationCandidateBase()->u32{
  return (arrayLength(&fineMetadata)/4u)*PRESENTATION_SAMPLES_PER_PAGE;
}
fn cm12PresentationStoreCandidate(page:u32,localIndex:u32,generation:u32,payload:u32){
  _=generation;fineSamples[cm12PresentationCandidateBase()
    +page*PRESENTATION_SAMPLES_PER_PAGE+localIndex]=payload;
}
fn cm12PresentationLoadCandidate(page:u32,localIndex:u32,generation:u32)->u32{
  _=generation;return fineSamples[cm12PresentationCandidateBase()
    +page*PRESENTATION_SAMPLES_PER_PAGE+localIndex];
}
fn cm12PresentationStoreAccepted(page:u32,localIndex:u32,payload:u32){
  fineSamples[page*PRESENTATION_SAMPLES_PER_PAGE+localIndex]=payload;
}
fn cm12PresentationCommitCandidate(page:u32,generation:u32){
  _=generation;fineMetadata[4u*page+2u]=1u;
}
fn cm12PresentationRejectAccepted(page:u32){
  if(page<arrayLength(&fineMetadata)/4u){fineMetadata[4u*page+2u]=0u;}
}
` : ""}

fn classifyPresentationBrick(brick:u32){
  if(brick>=p.dispatch.w){return;}
  let reasons=activityRecord(brick)+1u;
  if(!brickActive(brick)){atomicAnd(&activity[reasons],0xffffffbfu);return;}
  let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
  let first=range.x;let count=range.y;
  var wet=false;var massFineCells=0.0;
  for(var at=first;at<first+count;at+=1u){
    let rho=state[destinationDensity()+at];
    let fill=rho/max(cellOpenFraction(at),1e-6);
    wet=wet||fill>p.activityDensity.y;
    massFineCells+=max(0.0,rho)*cellVolume(at);
  }
  wet=wet&&massFineCells>=p.sharpening.w;
  if(wet){atomicOr(&activity[reasons],64u);}
  else{atomicAnd(&activity[reasons],0xffffffbfu);}
}

@compute @workgroup_size(64)
fn classifyPresentationBricks(@builtin(global_invocation_id)gid:vec3u){
  classifyPresentationBrick(gid.x);
}

// Publish one compact renderer page per workgroup. A page may hold more than
// 64 samples, so the 64 lanes walk it cooperatively. Span-one leaves use their
// ordinary P^3 subpages. A macro leaf uses one P^3 page sampled across its
// complete physical extent, so deep liquid remains authoritative without an
// O(span^3) page expansion. Metadata remains sorted by page-origin key.
@compute @workgroup_size(64)
fn publishSparseLevelSet(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let page=wid.x;let pageCount=arrayLength(&fineMetadata)/4u;
  // local_invocation_index is guaranteed to be 0..63 by workgroup_size(64).
  // Keeping that redundant lane predicate makes the return look non-uniform
  // to WGSL validation and invalidates the cache-fill barrier below.
  if(page>=pageCount){return;}
  let source=fineMetadata[4u*page+3u];
  var brick=0u;var octant=0u;var span=1u;
  if(PRESENTATION_PAGES_PER_AXIS==4u){
    let macroPage=(source&0x80000000u)!=0u;
    brick=select((source>>6u)&0xffffffu,source&0xffffffu,macroPage);
    octant=select(source&63u,0u,macroPage);
    span=select(1u,1u<<((source>>24u)&31u),macroPage);
  }else{
    brick=(source>>3u)&0xffffffu;octant=source&7u;span=1u<<(source>>27u);
  }
  let brickRecord=p.topologyOffsets2.z+2u*brick;
  let brickKey=topology[brickRecord+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let brickXY=brickDimensions.x*brickDimensions.y;let brickZ=brickKey/brickXY;
  let brickRemainder=brickKey-brickZ*brickXY;let brickY=brickRemainder/brickDimensions.x;
  let brickX=brickRemainder-brickY*brickDimensions.x;
  let pageZ=octant/(PRESENTATION_PAGES_PER_AXIS*PRESENTATION_PAGES_PER_AXIS);
  let pageRemainder=octant-pageZ*PRESENTATION_PAGES_PER_AXIS*PRESENTATION_PAGES_PER_AXIS;
  let pageY=pageRemainder/PRESENTATION_PAGES_PER_AXIS;
  let pageX=pageRemainder-pageY*PRESENTATION_PAGES_PER_AXIS;
  let pageOffset=vec3u(pageX,pageY,pageZ)*PRESENTATION_PAGE_RESOLUTION;
  let sampleScale=select(1u,
    BRICK_FINE_RESOLUTION*span/PRESENTATION_PAGE_RESOLUTION,span>1u);
  let brickOrigin=vec3u(brickX,brickY,brickZ)*BRICK_FINE_RESOLUTION;
  let pageOrigin=brickOrigin+pageOffset;
  let resolution=acceptedBrickResolution(brick);
  let scale=BRICK_FINE_RESOLUTION*span/resolution;
  let scaleF=f32(scale);let firstShifted=(vec3f(pageOrigin)+vec3f(0.5))/scaleF-vec3f(0.5);
  let lastQ=pageOrigin+vec3u(PRESENTATION_PAGE_RESOLUTION-1u)*sampleScale;
  let lastShifted=(vec3f(lastQ)+vec3f(0.5))/scaleF-vec3f(0.5);
  let cacheFirst=vec3i(floor(firstShifted));
  let cacheDimensions=vec3u(vec3i(floor(lastShifted))-cacheFirst)+vec3u(2u);
  let cacheCount=cacheDimensions.x*cacheDimensions.y*cacheDimensions.z;
  let cacheFits=cacheCount<=PRESENTATION_CACHE_CAPACITY;
  let wet=brick<p.dispatch.w
    &&(atomicLoad(&activity[activityRecord(brick)+1u])&64u)!=0u;
  if(scale>1u&&wet&&cacheFits){
    let coarseDimensions=max(vec3i(1),vec3i(p.dimensions.xyz)/i32(scale));
    for(var cacheIndex=lane;cacheIndex<cacheCount;cacheIndex+=64u){
      let cacheZ=cacheIndex/(cacheDimensions.x*cacheDimensions.y);
      let cacheRemainder=cacheIndex-cacheZ*cacheDimensions.x*cacheDimensions.y;
      let cacheY=cacheRemainder/cacheDimensions.x;
      let cacheX=cacheRemainder-cacheY*cacheDimensions.x;
      let coarse=clamp(cacheFirst+vec3i(i32(cacheX),i32(cacheY),i32(cacheZ)),
        vec3i(0),coarseDimensions-vec3i(1));
      presentationDensityCache[cacheIndex]=restrictedPresentationDensity(
        coarse*i32(scale),i32(scale));
    }
  }
  workgroupBarrier();
  for(var localIndex=lane;localIndex<PRESENTATION_SAMPLES_PER_PAGE;localIndex+=64u){
    let localZ=localIndex/(PRESENTATION_PAGE_RESOLUTION*PRESENTATION_PAGE_RESOLUTION);
    let localRemainder=localIndex-localZ*PRESENTATION_PAGE_RESOLUTION*PRESENTATION_PAGE_RESOLUTION;
    let localY=localRemainder/PRESENTATION_PAGE_RESOLUTION;
    let localX=localRemainder-localY*PRESENTATION_PAGE_RESOLUTION;
    let local=vec3u(localX,localY,localZ);
    let q=pageOrigin+local*sampleScale;
    var phi=4.0*p.frame.y;
    if(wet&&all(q<p.dimensions.xyz)
      &&insideEmbeddedBoundary(vec3f(q)+vec3f(0.5))){
      if(scale==1u){
        // Metadata already names the source brick and subpage. Fine pages can
        // address their packed source cell directly; resolving q through the
        // sparse directory here would redo a hash lookup for every output sample.
        let range=templateBrickCellRange(brick,resolution);
        let valid=min(p.dimensions.xyz-brickOrigin,vec3u(BRICK_FINE_RESOLUTION));
        let localCell=q-brickOrigin;
        let cell=range.x+localCell.x+valid.x*(localCell.y+valid.y*localCell.z);
        if(cell<range.x+range.y){
          phi=presentationPhi(cell);
        }
      }else{
        let shifted=(vec3f(q)+vec3f(0.5))/scaleF-vec3f(0.5);
        let lower=vec3i(floor(shifted));let fraction=fract(shifted);var rho=0.0;
        let coarseDimensions=max(vec3i(1),vec3i(p.dimensions.xyz)/i32(scale));
        for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
          let offset=vec3i(dx,dy,dz);var density=0.0;
          if(cacheFits){
            let cache=vec3u(lower+offset-cacheFirst);
            let cacheIndex=cache.x+cacheDimensions.x*(cache.y+cacheDimensions.y*cache.z);
            density=presentationDensityCache[cacheIndex];
          }else{
            let coarse=clamp(lower+offset,vec3i(0),coarseDimensions-vec3i(1));
            density=restrictedPresentationDensity(coarse*i32(scale),i32(scale));
          }
          let wx=select(1.0-fraction.x,fraction.x,dx==1);
          let wy=select(1.0-fraction.y,fraction.y,dy==1);
          let wz=select(1.0-fraction.z,fraction.z,dz==1);
          rho+=wx*wy*wz*density;
        }}}
        phi=(CM12_LIQUID_ISOVALUE-rho)*4.0*p.frame.y;
      }
    }
    let flags=1u|select(0u,16u,phi<0.0);
    fineSamples[page*PRESENTATION_SAMPLES_PER_PAGE+localIndex]
      =(pack2x16float(vec2f(phi,0.0))&0xffffu)|(flags<<16u);
  }
}

`;
}
