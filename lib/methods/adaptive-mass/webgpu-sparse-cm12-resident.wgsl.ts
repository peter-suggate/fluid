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
import {
  SPARSE_CM12_PRESSURE_MEMBERSHIP_MODE,
  SPARSE_CM12_PRESSURE_REPAIR_HEADER,
} from "./sparse-cm12-pressure-membership";
import { createSparseCM12FramePlanWGSL } from "../../core/sparse-cm12-frame-plan.wgsl";
import type { SparseCM12FramePlanLayout } from "../../core/sparse-cm12-frame-plan";
import { createSparseCM12FramePlanPresentationWGSL } from
  "./sparse-cm12-frame-plan-presentation.wgsl";
import type { SparseCM12FramePlanPresentationLayout } from
  "./sparse-cm12-frame-plan-presentation";
import {
  SPARSE_CM12_FRAME_CONTROL_COVERAGE,
  SPARSE_CM12_FRAME_CONTROL_FAULT,
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
import type { SparseCM12PressureSolveAuthorityLayout } from
  "./sparse-cm12-pressure-solve-authority";
import { createSparseCM12PressureSolveAuthorityWGSL } from
  "./sparse-cm12-pressure-solve-authority.wgsl";
import type { SparseCM12FaceProjectionAuthorityLayout } from
  "./sparse-cm12-face-projection-authority";
import { createSparseCM12FaceProjectionAuthorityWGSL } from
  "./sparse-cm12-face-projection-authority.wgsl";
import type { SparseCM12FacePreparationTileCensusLayout } from
  "./sparse-cm12-face-preparation-tile-census";
import { createSparseCM12FacePreparationTileCensusWGSL } from
  "./sparse-cm12-face-preparation-tile-census.wgsl";
import type { SparseCM12FpaVexReadCensusLayout } from
  "./sparse-cm12-fpa-vex-read-census";
import { createSparseCM12FpaVexReadCensusWGSL } from
  "./sparse-cm12-fpa-vex-read-census.wgsl";
import type { SparseCM12SRR1IngressLayout } from
  "./sparse-cm12-srr1-runtime-adapter";
import { createSparseCM12SRR1ResidentIngressWGSL } from
  "./sparse-cm12-srr1-resident-ingress.wgsl";
import { SPARSE_CM12_SCALAR_RESULT_CAUSE } from
  "./sparse-cm12-scalar-result-receipts";
import type { SparseCM12VexActivityBatchLayout } from
  "./sparse-cm12-vex-activity-batch";
import { createSparseCM12VexActivityBatchWGSL } from
  "./sparse-cm12-vex-activity-batch.wgsl";
import type { SparseCM12PressureAddressingABLayout,
  SparseCM12PressureAddressingABModeName } from
  "./sparse-cm12-pressure-addressing-ab";
import { createSparseCM12PressureAddressingABWGSL } from
  "./sparse-cm12-pressure-addressing-ab.wgsl";
import {
  SPARSE_CM12_VELOCITY_EXTENSION_CAUSE,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
} from "./sparse-cm12-velocity-extension";

/**
 * Compact, GPU-resident Sparse CM12 frame kernel.
 *
 * Topology is packed once at construction.  Every per-frame field, reduction,
 * pressure iteration, projection, diagnostic field, and dense presentation
 * publication remains in device storage.  The host only writes the small
 * timestep uniform and submits the pre-sized dispatch sequence.
 */
export type SparseCM12BrickFineResolution = 4 | 8 | 16;

export interface SparseCM12TemporalWorklistLayout {
  readonly headerBaseWords: number;
  readonly cellListBaseWords: number;
  readonly rowListBaseWords: number;
  readonly cellFlagABaseWords: number;
  readonly cellFlagBBaseWords: number;
  readonly totalWords: number;
}

/** Persistent dense pressure membership plus bounded local change journals. */
export interface SparseCM12PressureRepairLayout {
  readonly cellSlotBaseWords: number;
  readonly rowSlotBaseWords: number;
  readonly cellChangeBaseWords: number;
  readonly rowChangeBaseWords: number;
  readonly brickStateBaseWords: number;
  readonly rowTopologyStampBaseWords: number;
  readonly aggregateEdgeForFineEdgeBaseWords: number;
  readonly aggregateEdgeSourceBaseWords: number;
  readonly hierarchyEdgeForAggregateBaseWords: readonly number[];
  readonly headerBaseWords: number;
  readonly totalWords: number;
}

function createTemporalWorklistWGSL(
  layout: SparseCM12TemporalWorklistLayout,
): string {
  const h = layout.headerBaseWords;
  return /* wgsl */ `
const TEMPORAL_CELL_COUNT_AT:u32=${h}u;
const TEMPORAL_ROW_COUNT_AT:u32=${h + 4}u;
const TEMPORAL_CELL_LIST:u32=${layout.cellListBaseWords}u;
const TEMPORAL_ROW_LIST:u32=${layout.rowListBaseWords}u;
const TEMPORAL_CELL_FLAG_A:u32=${layout.cellFlagABaseWords}u;
const TEMPORAL_CELL_FLAG_B:u32=${layout.cellFlagBBaseWords}u;
const TEMPORAL_F32_POSITIVE_ZERO:u32=0x00000000u;
const TEMPORAL_F32_ONE:u32=0x3f800000u;
// The exact bit-change seed is the production path. The immutable "current"
// QA constructor overrides this to false so the former canonical-value seed
// remains available as a byte-exact comparison oracle.
override TEMPORAL_CHANGE_SEED_QA:bool=true;
override TEMPORAL_SEED_COUNT_QA:bool=false;
const TEMPORAL_SEED_COUNT_AT:u32=${h + 8}u;
const TEMPORAL_DILATION_A_COUNT_AT:u32=${h + 9}u;

// 0 = exact dry authority, 1 = exact flooded authority, 2 = unresolved.
fn temporalScalarEndpointPhaseAt(cell:u32,densityOffset:u32,gammaOffset:u32)->u32{
  if(!cellActive(cell)||bitcast<u32>(cellOpenFraction(cell))!=TEMPORAL_F32_ONE
    ||bitcast<u32>(state[gammaOffset+cell])!=TEMPORAL_F32_ONE){return 2u;}
  let rho=bitcast<u32>(state[densityOffset+cell]);
  if(rho==TEMPORAL_F32_POSITIVE_ZERO){return 0u;}
  if(rho==TEMPORAL_F32_ONE){return 1u;}
  return 2u;
}

// Exact constant-field authority. This is a value/topology certificate, not a
// scene heuristic: any cut face, non-two-term row, phase value, moving solid,
// high local CFL, or noncanonical neighbor stays on the ordinary path.
fn temporalScalarCellCanonicalAt(cell:u32,densityOffset:u32,gammaOffset:u32)->bool{
  let phase=temporalScalarEndpointPhaseAt(cell,densityOffset,gammaOffset);
  if(phase==2u){atomicOr(&activity[27],1u);return false;}
  let velocityAt=destinationCellVelocity()+4u*cell;
  // A dry constant field has no characteristic payload to move. Flooded cells
  // require a bounded trace. The collocation producer deliberately clears w
  // after projection because w is the *next extrapolation* marker, not a
  // validity bit for the projected xyz values observed here. The next-frame
  // mass receipt checks w again after all eight extrapolation sweeps.
  if(phase==1u&&(length(vec3f(state[velocityAt],state[velocityAt+1u],
      state[velocityAt+2u]))*p.frame.x
      >0.75*cellMinimumWidth(cell))){atomicOr(&activity[27],4u);return false;}
  var sideMask=0u;
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);
    if(!rowAccepted(row)||rowArea(row)<=1e-8){
      atomicOr(&activity[27],8u);return false;}
    let coefficient=termCoefficient(incidenceTerm(incidence));
    let side=2u*rowAxis(row)+select(0u,1u,coefficient>0.0);
    sideMask|=1u<<side;
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);var hasNeighbor=false;
    for(var term=begin;term<end;term+=1u){
      let candidate=termCell(term);if(candidate==cell){continue;}
      hasNeighbor=true;
      if(temporalScalarEndpointPhaseAt(candidate,densityOffset,gammaOffset)!=phase){
        atomicOr(&activity[27],16u);return false;
      }
    }
    if(!hasNeighbor){atomicOr(&activity[27],16u);return false;}
  }
  if(sideMask!=63u){atomicOr(&activity[27],2u);return false;}
  return true;
}

@compute @workgroup_size(1)
fn resetTemporalScalarWorklists(){
  atomicStore(&activity[TEMPORAL_CELL_COUNT_AT],0u);
  atomicStore(&activity[TEMPORAL_CELL_COUNT_AT+1u],0u);
  atomicStore(&activity[TEMPORAL_CELL_COUNT_AT+2u],1u);
  atomicStore(&activity[TEMPORAL_CELL_COUNT_AT+3u],1u);
  atomicStore(&activity[TEMPORAL_ROW_COUNT_AT],0u);
  atomicStore(&activity[TEMPORAL_ROW_COUNT_AT+1u],0u);
  atomicStore(&activity[TEMPORAL_ROW_COUNT_AT+2u],1u);
  atomicStore(&activity[TEMPORAL_ROW_COUNT_AT+3u],1u);
  if(TEMPORAL_SEED_COUNT_QA){
    atomicStore(&activity[TEMPORAL_SEED_COUNT_AT],0u);
    atomicStore(&activity[TEMPORAL_DILATION_A_COUNT_AT],0u);
  }
  atomicStore(&activity[27],0x80000000u);
}

// Pressure membership consumes the fully conditioned destination fields, not
// the pre-transport source banks. This observation-only pass builds the same
// conservative two-ring frontier after sharpening has published those values.
@compute @workgroup_size(64)
fn classifyTemporalPressureCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  // Exact endpoint transitions can look canonical on the current frame while
  // the persistent PCM bit still carries the previous phase. The global QA
  // census is presently our correctness authority, so explicitly root that
  // transition without mutating pressure state; the ordinary incidence
  // dilation below supplies the dependent row closure.
  let membershipChanged=pressureCellMembershipPredicate(cell)!=pcmCellContains(cell);
  var slow=false;
  if(TEMPORAL_CHANGE_SEED_QA){
    // Bitwise-identical scalar endpoints do not require
    // pressure/presentation/activity repair merely because their value is
    // noncanonical. Rigid/cut cells retain the conservative certificate.
    let scalarChanged=bitcast<u32>(state[sourceDensity()+cell])
        !=bitcast<u32>(state[destinationDensity()+cell])
      ||bitcast<u32>(state[sourceGamma()+cell])
        !=bitcast<u32>(state[destinationGamma()+cell]);
    let conservative=hasRigidBodies()
      ||bitcast<u32>(cellOpenFraction(cell))!=TEMPORAL_F32_ONE;
    slow=membershipChanged||scalarChanged||conservative;
  }else{
    slow=membershipChanged
      ||!temporalScalarCellCanonicalAt(cell,destinationDensity(),destinationGamma());
  }
  atomicStore(&activity[TEMPORAL_CELL_FLAG_A+cell],select(0u,1u,slow));
  if(TEMPORAL_SEED_COUNT_QA&&slow){
    atomicAdd(&activity[TEMPORAL_SEED_COUNT_AT],1u);
  }
}

fn dilateTemporalScalarCell(cell:u32,inputBase:u32)->u32{
  if(atomicLoad(&activity[inputBase+cell])!=0u){return 1u;}
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);if(!rowAccepted(row)){return 1u;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){
      let neighbor=termCell(term);
      if(atomicLoad(&activity[inputBase+neighbor])!=0u){return 1u;}
    }
  }
  return 0u;
}

@compute @workgroup_size(64)
fn dilateTemporalScalarAtoB(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  let slow=dilateTemporalScalarCell(cell,TEMPORAL_CELL_FLAG_A);
  atomicStore(&activity[TEMPORAL_CELL_FLAG_B+cell],slow);
  if(TEMPORAL_SEED_COUNT_QA&&slow!=0u){
    atomicAdd(&activity[TEMPORAL_DILATION_A_COUNT_AT],1u);
  }
}

@compute @workgroup_size(64)
fn dilateTemporalScalarBtoA(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID){return;}
  atomicStore(&activity[TEMPORAL_CELL_FLAG_A+cell],
    dilateTemporalScalarCell(cell,TEMPORAL_CELL_FLAG_B));
}

@compute @workgroup_size(64)
fn compactTemporalScalarCells(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID
    ||atomicLoad(&activity[TEMPORAL_CELL_FLAG_A+cell])==0u){return;}
  let slot=atomicAdd(&activity[TEMPORAL_CELL_COUNT_AT],1u);
  atomicStore(&activity[TEMPORAL_CELL_LIST+slot],cell);
}

@compute @workgroup_size(64)
fn compactTemporalScalarRows(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);var slow=false;
  for(var term=begin;term<end;term+=1u){
    slow=slow||atomicLoad(&activity[TEMPORAL_CELL_FLAG_A+termCell(term)])!=0u;
  }
  if(!slow){return;}
  let slot=atomicAdd(&activity[TEMPORAL_ROW_COUNT_AT],1u);
  atomicStore(&activity[TEMPORAL_ROW_LIST+slot],row);
}

@compute @workgroup_size(1)
fn finalizeTemporalScalarWorklists(){
  let cells=atomicLoad(&activity[TEMPORAL_CELL_COUNT_AT]);
  let rows=atomicLoad(&activity[TEMPORAL_ROW_COUNT_AT]);
  atomicStore(&activity[TEMPORAL_CELL_COUNT_AT+1u],(cells+63u)/64u);
  atomicStore(&activity[TEMPORAL_ROW_COUNT_AT+1u],(rows+63u)/64u);
  atomicStore(&activity[25],cells);atomicStore(&activity[26],rows);
}

fn temporalScalarCellInvocation(invocation:u32)->u32{
  let count=atomicLoad(&activity[TEMPORAL_CELL_COUNT_AT]);
  return select(INVALID,atomicLoad(&activity[TEMPORAL_CELL_LIST+invocation]),invocation<count);
}
fn temporalScalarRowInvocation(invocation:u32)->u32{
  let count=atomicLoad(&activity[TEMPORAL_ROW_COUNT_AT]);
  return select(INVALID,atomicLoad(&activity[TEMPORAL_ROW_LIST+invocation]),invocation<count);
}
`;
}

/** Construction-time specialization keeps brick geometry in WGSL constants,
 * so the hot lookup/transfer paths pay no runtime ladder branch. */
export function createWebgpuSparseCM12ResidentWGSL(
  brickFineResolution: SparseCM12BrickFineResolution = 16,
  presentationPageResolution: SparseCM12BrickFineResolution = 16,
  temporalWorklistLayout?: SparseCM12TemporalWorklistLayout,
  pressureRepairLayout?: SparseCM12PressureRepairLayout,
  incrementalActivityLayout?: SparseCM12IncrementalActivityLayout,
  canonicalMembershipLayout?: SparseCM12CanonicalMembershipLayout,
  framePlanLayout?: SparseCM12FramePlanLayout,
  framePlanPresentationLayout?: SparseCM12FramePlanPresentationLayout,
  frameControlLayout?: SparseCM12FrameControlLayout,
  scalarResultIngressLayout?: SparseCM12SRR1IngressLayout,
  pressureTopologyRepairLayout?: SparseCM12PressureTopologyRepairLayout,
  persistentPressureCacheLayout?: SparseCM12PersistentPressureCacheLayout,
  pressureSolveAuthorityLayout?: SparseCM12PressureSolveAuthorityLayout,
  faceProjectionAuthorityLayout?: SparseCM12FaceProjectionAuthorityLayout,
  vexActivityBatchLayout?: SparseCM12VexActivityBatchLayout,
  pressureAddressingABLayout?: SparseCM12PressureAddressingABLayout,
  pressureAddressingFixedMode?: SparseCM12PressureAddressingABModeName,
  facePreparationTileCensusLayout?: SparseCM12FacePreparationTileCensusLayout,
  fpaVexReadCensusLayout?: SparseCM12FpaVexReadCensusLayout,
): string {
  if (presentationPageResolution > brickFineResolution
    || brickFineResolution % presentationPageResolution !== 0) {
    throw new RangeError(`presentation page ${presentationPageResolution} does not divide brick ladder ${brickFineResolution}`);
  }
  const templateLevelCount = Math.log2(brickFineResolution) + 1;
  const candidateCellCount = brickFineResolution ** 3;
  const candidateFaceSampleCount = brickFineResolution ** 2;
  const presentationPagesPerAxis = brickFineResolution / presentationPageResolution;
  // Span-one pages need at most (P/2 + 2)^3 samples: scale one addresses the
  // authority directly, so scale two is the finest cached reconstruction.
  // A finer macro can exceed this bounded cache and takes the direct fallback.
  const presentationCacheCapacity = (presentationPageResolution / 2 + 2) ** 3;
  const temporalWorklistEntries = temporalWorklistLayout
    ? createTemporalWorklistWGSL(temporalWorklistLayout)
    : /* wgsl */ `
const TEMPORAL_CELL_FLAG_A:u32=0u;
fn temporalScalarCellInvocation(invocation:u32)->u32{
  return acceptedTemplateCellInvocation(invocation);
}
fn temporalScalarRowInvocation(invocation:u32)->u32{
  return acceptedTemplateRowInvocation(invocation);
}
`;
  const incrementalActivityEntries = incrementalActivityLayout
    ? createSparseCM12IncrementalActivityWGSL(incrementalActivityLayout,
      brickFineResolution / 4)
    : /* wgsl */ `
fn incrementalActivityGeneration()->u32{return 0u;}
fn incrementalActivityMarkCellClosure(cell:u32,cause:u32){_=cell;_=cause;}
fn incrementalActivityBrickInvocation(invocation:u32)->u32{
  return select(INVALID,invocation,invocation<p.dispatch.w);
}
fn incrementalActivityBrickTileMask(brick:u32)->vec2u{_=brick;return vec2u(0u);}
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
  const vexActivityBatchEntries = vexActivityBatchLayout
    ? createSparseCM12VexActivityBatchWGSL({
      layout: vexActivityBatchLayout,
      arenaName: "activity",
      stateName: "state",
      topologyGenerationExpression: "atomicLoad(&topologyArena[topologyWorklistBase()])",
      sourceFrameGenerationExpression: "cm12FCAcceptedGeneration()",
      nextFrameGenerationExpression: "cm12FCCandidateGeneration()",
      frameAuthorityReadyExpression: "cm12FCFrameSealed()",
      provenanceHookPrefix: "cm12Resident",
    })
    : "";
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
fn pcmCellRankSelect(rank:u32)->u32{return pressureCellListInvocation(rank);}
fn pcmRowRankSelect(rank:u32)->u32{return pressureRowListInvocation(rank);}
fn pcmCellContains(id:u32)->bool{return isLiquid(id);}
fn pcmRowContains(id:u32)->bool{return state[p.stateOffsets3.x+id]>0.0;}
fn pcmCellAcceptedCount()->u32{return fineSamples[0];}
fn pcmRowAcceptedCount()->u32{return fineSamples[pressureRowHeader()];}
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
const PRESSURE_CELL_SLOT_BASE:u32=${pressureRepairLayout.cellSlotBaseWords}u;
const PRESSURE_ROW_SLOT_BASE:u32=${pressureRepairLayout.rowSlotBaseWords}u;
const PRESSURE_CELL_CHANGE_BASE:u32=${pressureRepairLayout.cellChangeBaseWords}u;
const PRESSURE_ROW_CHANGE_BASE:u32=${pressureRepairLayout.rowChangeBaseWords}u;
const PRESSURE_BRICK_STATE_BASE:u32=${pressureRepairLayout.brickStateBaseWords}u;
const PRESSURE_ROW_TOPOLOGY_STAMP_BASE:u32=${pressureRepairLayout.rowTopologyStampBaseWords}u;
const PRESSURE_REPAIR_HEADER:u32=${pressureRepairLayout.headerBaseWords}u;
const PRESSURE_REPAIR_INITIALIZED:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.initialized}u;
const PRESSURE_REPAIR_CELL_CHANGES:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.cellChanges}u;
const PRESSURE_REPAIR_ROW_CHANGES:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.rowChanges}u;
const PRESSURE_REPAIR_FAULT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.fault}u;
const PRESSURE_REPAIR_TOPOLOGY_GENERATION:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.topologyGeneration}u;
const PRESSURE_REPAIR_EPOCH:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.epoch}u;
const PRESSURE_REPAIR_MODE:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.mode}u;
const PRESSURE_BOOTSTRAP_CELL_INDIRECT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.bootstrapCellIndirect}u;
const PRESSURE_BOOTSTRAP_ROW_INDIRECT:u32=${SPARSE_CM12_PRESSURE_REPAIR_HEADER.bootstrapRowIndirect}u;
const PRESSURE_MODE_FAIL_CLOSED:u32=${SPARSE_CM12_PRESSURE_MEMBERSHIP_MODE.failClosed}u;
const PRESSURE_MODE_BOOTSTRAP:u32=${SPARSE_CM12_PRESSURE_MEMBERSHIP_MODE.bootstrap}u;
const PRESSURE_MODE_INCREMENTAL:u32=${SPARSE_CM12_PRESSURE_MEMBERSHIP_MODE.incremental}u;
` : /* wgsl */ `
const PRESSURE_CELL_SLOT_BASE:u32=0u;
const PRESSURE_ROW_SLOT_BASE:u32=0u;
const PRESSURE_CELL_CHANGE_BASE:u32=0u;
const PRESSURE_ROW_CHANGE_BASE:u32=0u;
const PRESSURE_BRICK_STATE_BASE:u32=0u;
const PRESSURE_ROW_TOPOLOGY_STAMP_BASE:u32=0u;
const PRESSURE_REPAIR_HEADER:u32=0u;
const PRESSURE_REPAIR_INITIALIZED:u32=0u;
const PRESSURE_REPAIR_CELL_CHANGES:u32=1u;
const PRESSURE_REPAIR_ROW_CHANGES:u32=2u;
const PRESSURE_REPAIR_FAULT:u32=3u;
const PRESSURE_REPAIR_TOPOLOGY_GENERATION:u32=4u;
const PRESSURE_REPAIR_EPOCH:u32=5u;
const PRESSURE_REPAIR_MODE:u32=6u;
const PRESSURE_BOOTSTRAP_CELL_INDIRECT:u32=7u;
const PRESSURE_BOOTSTRAP_ROW_INDIRECT:u32=10u;
const PRESSURE_MODE_FAIL_CLOSED:u32=0u;
const PRESSURE_MODE_BOOTSTRAP:u32=1u;
const PRESSURE_MODE_INCREMENTAL:u32=2u;
`;
  const presentationTemporalReuse = temporalWorklistLayout ? /* wgsl */ `
  // Fine samples sourced from exact-authority scalar cells retain their prior
  // payload. Keep one workgroup per page: workgroup-per-tile publication is
  // prohibitively expensive on Metal even when most tiles return early.
  let temporalReady=(atomicLoad(&activity[27])&0x80000000u)!=0u;
  let reusePresentationSamples=temporalReady&&atomicLoad(&activity[17])==0u
    &&brickActive(brick)&&scale==1u;
` : "  let reusePresentationSamples=false;";
  const frameControlEntries = frameControlLayout
    ? createSparseCM12FrameControlWGSL({
      layout: frameControlLayout, controlName: "topologyArena",
    })
    : /* wgsl */ `
fn cm12FCSourceScalarParity()->u32{return select(0u,1u,p.frame.w>0.5);}
fn cm12FCDestinationScalarParity()->u32{return cm12FCSourceScalarParity()^1u;}
fn cm12FCSourceFaceParity()->u32{return cm12FCSourceScalarParity();}
fn cm12FCDestinationFaceParity()->u32{return cm12FCSourceFaceParity()^1u;}
fn cm12FCCandidateGeneration()->u32{return 1u;}
fn cm12FCAcceptedGeneration()->u32{return 1u;}
fn cm12FCFrameSealed()->bool{return true;}
fn cm12FCBegin()->bool{return true;}
fn cm12FCPublishBody(count:u32)->bool{_=count;return true;}
fn cm12FCPublishBoundary(live:bool)->bool{_=live;return true;}
fn cm12FCSeal()->bool{return true;}
fn cm12FCPublishOutput(mask:u32)->bool{_=mask;return true;}
fn cm12FCCommit()->bool{return true;}
fn cm12FCInvalidateD4(cause:u32,owner:u32)->bool{_=cause;_=owner;return true;}
`;
  const scalarAuthorityEntries = scalarResultIngressLayout
    ? createSparseCM12SRR1ResidentIngressWGSL({
      layout: scalarResultIngressLayout, arenaName: "activity",
    })
    : /* wgsl */ `
fn sirResidentBeginFrame(generation:u32,topologyGeneration:u32,sourceParity:u32)->bool{
  _=generation;_=topologyGeneration;_=sourceParity;return true;}
fn sirResidentAppendEvent(tile:u32,generation:u32,cause:u32)->bool{
  _=tile;_=generation;_=cause;return true;}
fn sirResidentEventStamped(tile:u32,generation:u32)->bool{
  _=tile;_=generation;return false;}
fn sirResidentEventCause(tile:u32,generation:u32)->u32{
  _=tile;_=generation;return 0u;}
fn sirResidentResetCopiedEvents(){}
fn sirResidentBeginReceiptBatch(){}
fn sirResidentCandidateGeneration()->u32{return 1u;}
fn sirResidentTopologyGeneration()->u32{return 1u;}
fn sirResidentSourceParity()->u32{return 0u;}
fn sirResidentWorkTile(rank:u32)->u32{return rank;}
fn sirResidentPublishReceipt(rank:u32,tile:u32,topologyGeneration:u32,
  bank0Generation:u32,bank1Generation:u32,headResultGeneration:u32,
  coveredWords:u32,expectedWords:u32,sourceMismatchCount:u32,
  destinationMismatchCount:u32,dependencyGeneration:u32,
  dependencyCertifiedGeneration:u32,resultKind:u32,generation:u32)->bool{
  _=rank;_=tile;_=topologyGeneration;_=bank0Generation;_=bank1Generation;
  _=headResultGeneration;_=coveredWords;_=expectedWords;_=sourceMismatchCount;
  _=destinationMismatchCount;_=dependencyGeneration;_=dependencyCertifiedGeneration;
  _=resultKind;_=generation;return true;}
`;
  const pressureAddressingABEntries = pressureAddressingABLayout
    ? createSparseCM12PressureAddressingABWGSL({
      layout: pressureAddressingABLayout, arenaName: "activity", workgroupSize: 64,
      fixedMode: pressureAddressingFixedMode,
    })
    : /* wgsl */ `
fn pabPressureCellAddress(rank:u32)->u32{return pcmCellRankSelect(rank);}
fn pabPressureAddressingReady()->bool{return true;}
`;
  const pressureTopologyRepairEntries = pressureTopologyRepairLayout
    ? createSparseCM12PressureTopologyRepairWGSL({
      layout: pressureTopologyRepairLayout, arenaName: "topologyArena",
      prefix: "ptr", workgroupSize: 64,
    })
    : /* wgsl */ `
const ptrH_COVERED_PRODUCER_RECEIPTS:u32=0u;
fn ptrRecordChangedBrick(brick:u32,oldState:u32,newState:u32,cause:u32,
 receipt:bool)->bool{_=brick;_=oldState;_=newState;_=cause;_=receipt;return true;}
fn ptrSealTopologyJournal(expected:u32,topologyGeneration:u32)->bool{
 _=expected;_=topologyGeneration;return true;}
fn ptrRejectTopologyJournal()->bool{return true;}
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
  const pressureSolveAuthorityEntries = pressureSolveAuthorityLayout
    ? createSparseCM12PressureSolveAuthorityWGSL({
      layout: pressureSolveAuthorityLayout, arenaName: "topologyArena",
      prefix: "psa", workgroupSize: 64,
    })
    : /* wgsl */ `
fn psaWetBrickInvocation(invocation:u32)->u32{return invocation;}
fn psaActiveHierarchyNodeAddress(invocation:u32)->vec2u{
  return pressureHierarchyGroupAddress(invocation);}
fn psaWetBrickCount()->u32{return p.dispatch.w;}
fn psaActiveHierarchyNodeCount()->u32{return 0u;}
fn psaAcceptedGeneration()->u32{return 0u;}
fn psaFaultCode()->u32{return 0u;}
fn psaWetBrickContains(brick:u32)->bool{return brick<p.dispatch.w;}
fn psaMarkTopologyBrickBlast(brick:u32,receipt:bool)->bool{
  _=brick;_=receipt;return true;}
`;
  const faceProjectionAuthorityEntries = faceProjectionAuthorityLayout
    ? createSparseCM12FaceProjectionAuthorityWGSL({
      layout: faceProjectionAuthorityLayout, arenaName: "topologyArena",
      prefix: "fpa", workgroupSize: 64,
      preparationBootstrapInvocationFunction: "acceptedTemplateRowInvocation",
    })
    : /* wgsl */ `
fn fpaPreparationRowInvocation(invocation:u32)->u32{
  return acceptedTemplateRowInvocation(invocation);}
fn fpaProjectionRowInvocation(invocation:u32)->u32{
  return pressureRowInvocation(invocation);}
fn fpaPreparationComplete(row:u32)->bool{_=row;return true;}
fn fpaProjectionComplete(row:u32)->bool{_=row;return true;}
fn fpaStorePreparedAuthority(row:u32,bits:u32){_=row;_=bits;}
fn fpaPreparedAuthorityBits(row:u32)->u32{_=row;return 0u;}
fn fpaPreparationMustMirrorUnprojected(row:u32)->bool{_=row;return false;}
fn fpaProjectionMustMirror(row:u32)->bool{_=row;return false;}
fn fpaMarkPreparationRow(row:u32,cause:u32,depth:u32,receipt:bool)->bool{
  _=row;_=cause;_=depth;_=receipt;return true;}
fn fpaMarkProjectionRow(row:u32,cause:u32,depth:u32,receipt:bool)->bool{
  _=row;_=cause;_=depth;_=receipt;return true;}
fn fpaCoverPreparationReceipt(){}
fn fpaMarkProjectionPressureCell(cell:u32,bits:u32,cause:u32)->bool{
  _=cell;_=bits;_=cause;return true;}
`;
  const facePreparationTileCensusEntries = facePreparationTileCensusLayout
    && incrementalActivityLayout
    ? createSparseCM12FacePreparationTileCensusWGSL({
      layout: facePreparationTileCensusLayout, activity: incrementalActivityLayout,
    }) : "";
  const fpaVexReadCensusEntries = fpaVexReadCensusLayout
    ? createSparseCM12FpaVexReadCensusWGSL(fpaVexReadCensusLayout) : "";
  // The census is a construction-only QA specialization. Keep the ordinary
  // production interpolation source byte-clean: no census parameter, call,
  // or no-op seam is emitted for the production resident.
  const fvrParameter = fpaVexReadCensusLayout ? ",censusRow:u32" : "";
  const fvrArgument = fpaVexReadCensusLayout ? ",censusRow" : "";
  const fvrInvalidArgument = fpaVexReadCensusLayout ? ",INVALID" : "";
  const fvrRowArgument = fpaVexReadCensusLayout ? ",row" : "";
  const fvrSampleRead = fpaVexReadCensusLayout
    ? `let effective=cm12ExtensionTransportVelocity(cell);
    fvrRecordXyzRead(censusRow,cell,effective.xyz);
    result+=wx*wy*wz*effective.xyz;`
    : "result+=wx*wy*wz*cm12ExtensionTransportVelocity(cell).xyz;";
  return /* wgsl */ `
${createCm12NumericsWGSL()}

const INVALID:u32=0xffffffffu;
const WORKGROUP:u32=64u;
const PRESSURE_EXECUTION_QA_ORACLE:bool=${pressureSolveAuthorityLayout?.qaFullOracle
  ? "true" : "false"};
const ACTIVITY_HEADER_WORDS:u32=28u;
const ACTIVITY_RECORD_WORDS:u32=39u;
const ACTIVITY_RECOVERY_LOCK:u32=0x80000000u;
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
const PRESENTATION_CACHE_CAPACITY:u32=${presentationCacheCapacity}u;
const TEMPLATE_CELL_RESOLUTION_BITS:u32=5u;
const TEMPLATE_CELL_RESOLUTION_MASK:u32=31u;
const ACTIVITY_FIXED:f32=65536.0;

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
  stateOffsets5:vec4u,      // sharpening delta / D4 rho scratch, D4 gamma scratch, reserved
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
${temporalWorklistEntries}
${pressureRepairEntries}
${incrementalActivityEntries}
${canonicalMembershipEntries}
${framePlanEntries}
${framePlanPresentationEntries}
${pressureTopologyRepairEntries}
${persistentPressureCacheEntries}
${pressureSolveAuthorityEntries}
${faceProjectionAuthorityEntries}
${facePreparationTileCensusEntries}
${fpaVexReadCensusEntries}
${vexActivityBatchEntries}
${pressureAddressingABEntries}

fn topologyWorklistBase()->u32{return atomicLoad(&topologyArena[14u]);}
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
fn pressureCellWorkgroups()->u32{return fineSamples[1];}
fn pressureCellListInvocation(invocation:u32)->u32{
  if(invocation>=pressureCellCount()){return INVALID;}
  return fineSamples[4u+invocation];
}
fn pressureCellInvocation(invocation:u32)->u32{
  let cell=pabPressureCellAddress(invocation);
  return select(INVALID,cell,cell!=INVALID&&pcmCellContains(cell)
    &&cellActive(cell));
}
fn pressureRowHeader()->u32{return 4u+p.counts.x;}
fn pressureRowCount()->u32{return pcmRowAcceptedCount();}
fn pressureRowListInvocation(invocation:u32)->u32{
  if(invocation>=pressureRowCount()){return INVALID;}
  return fineSamples[pressureRowHeader()+4u+invocation];
}
fn pressureRowInvocation(invocation:u32)->u32{
  let row=pcmRowRankSelect(invocation);
  return select(INVALID,row,row!=INVALID&&pcmRowContains(row)&&rowAccepted(row)
    &&state[p.stateOffsets3.x+row]>0.0);
}
fn pressureNeighborOffset()->u32{return pressureRowHeader()+4u+p.counts.y;}
fn pressureExtrapolationWeightOffset()->u32{
  return pressureNeighborOffset()+pressureEdgeCount();
}
fn acceptedTemplateRowInvocation(invocation:u32)->u32{
  if(invocation>=acceptedTemplateRowCount()){return INVALID;}
  let base=topologyWorklistBase();
  let offset=atomicLoad(&topologyArena[base+16u+acceptedTopologySlot()]);
  return atomicLoad(&topologyArena[base+offset+invocation]);
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
  ${vexActivityBatchLayout ? `if(!cm12ExtensionReadyForFrameCommit()){
    cm12FCFail(${SPARSE_CM12_FRAME_CONTROL_FAULT.velocityExtensionReceipt}u,
      cm12ExtensionAcceptedGeneration());return;
  }` : ""}
  _=cm12FCCommit();
}
@compute @workgroup_size(1)
fn invalidateSparseCM12FrameD4ForInjection(){
  let asymmetric=abs(p.injectionCenter.x-0.5*f32(p.dimensions.x))>1e-6
    ||abs(p.injectionCenter.z-0.5*f32(p.dimensions.z))>1e-6
    ||abs(p.injectionRadius.x-p.injectionRadius.z)>1e-6;
  if(asymmetric){_=cm12FCInvalidateD4(1u,0u);}
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

fn cellBase(id:u32)->u32{return ta(6u)+id*8u;}
fn cellMetadata(id:u32)->u32{return ta(cellBase(id)+7u);}
fn cellBrick(id:u32)->u32{return cellMetadata(id)>>TEMPLATE_CELL_RESOLUTION_BITS;}
fn cellResolution(id:u32)->u32{return cellMetadata(id)&TEMPLATE_CELL_RESOLUTION_MASK;}
fn cellVolume(id:u32)->f32{return taf(cellBase(id)+3u);}
fn cellOpenFraction(id:u32)->f32{
  if(!hasRigidBodies()){return 1.0;}
  return state[p.solidOffsets.x+id];
}
fn cellOpenVolume(id:u32)->f32{
  if(!hasRigidBodies()){return cellVolume(id);}
  return cellVolume(id)*state[p.solidOffsets.x+id];
}
fn cellSeparatingMinimum(id:u32)->bool{return false;}
fn cellMinimumWidth(id:u32)->f32{
  let b=cellBase(id);return min(taf(b+4u),min(taf(b+5u),taf(b+6u)));
}
fn cellMinimum(id:u32)->vec3u{
  let b=cellBase(id);let center=vec3f(taf(b),taf(b+1u),taf(b+2u));
  let widths=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));
  return vec3u(center-0.5*widths);
}
fn rowWord(id:u32,plane:u32)->u32{return ta(7u)+plane*ta(3u)+id;}
fn rowPackedTerms(id:u32)->u32{return ta(rowWord(id,0u));}
fn rowPackedMetadata(id:u32)->u32{return ta(rowWord(id,1u));}
fn rowTermOffset(id:u32)->u32{return rowPackedTerms(id)&0x0fffffffu;}
fn rowTermCount(id:u32)->u32{return rowPackedTerms(id)>>28u;}
fn rowAxis(id:u32)->u32{return rowPackedMetadata(id)>>30u;}
fn rowKind(id:u32)->u32{return (rowPackedMetadata(id)>>28u)&3u;}
fn rowRequirementOffset(id:u32)->u32{return rowPackedMetadata(id)&0x0fffffffu;}
fn rowStaticDualWeight(id:u32)->f32{return taf(rowWord(id,2u));}
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
fn rowStaticArea(id:u32)->f32{return taf(rowWord(id,3u));}
fn rowArea(id:u32)->f32{
  if(!hasRigidBodies()){return rowStaticArea(id);}
  return rowStaticArea(id)*rowOpenFraction(id);
}
fn rowDistance(id:u32)->f32{return taf(rowWord(id,4u));}
fn rowExteriorPhi(id:u32)->f32{return taf(rowWord(id,5u));}
fn rowCenter(id:u32)->vec3f{return vec3f(taf(rowWord(id,6u)),taf(rowWord(id,7u)),taf(rowWord(id,8u)));}
fn termCell(index:u32)->u32{return ta(ta(8u)+2u*index);}
fn termCoefficient(index:u32)->f32{return taf(ta(8u)+2u*index+1u);}
fn incidenceBegin(cell:u32)->u32{return ta(ta(9u)+cell);}
fn incidenceEnd(cell:u32)->u32{return ta(ta(9u)+cell+1u);}
fn incidenceRow(index:u32)->u32{return ta(ta(10u)+2u*index);}
fn incidenceTerm(index:u32)->u32{return ta(ta(10u)+2u*index+1u);}
fn activityRecord(brick:u32)->u32{
  return ACTIVITY_HEADER_WORDS+ACTIVITY_RECORD_WORDS*brick;
}
fn scheduledBrickResolution(brick:u32)->u32{
  let record=activityRecord(brick);
  return select(atomicLoad(&activity[record+12u]),atomicLoad(&activity[record+13u]),
    atomicLoad(&activity[record+35u])!=0u);
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
fn brickCandidatePlanningEnabled(brick:u32)->bool{
  let pageCapacity=atomicLoad(&topologyArena[topologyWorklistBase()+27u]);
  return brickPackedCandidateSlot(brick)!=INVALID||(brickSpan(brick)==1u&&pageCapacity!=0u);
}
fn templateBrickCellRange(brick:u32,resolution:u32)->vec2u{
  let at=ta(11u)+2u*(TEMPLATE_LEVEL_COUNT*brick+templateLevelIndex(resolution));
  return vec2u(ta(at),ta(at+1u));
}
fn brickActive(brick:u32)->bool{
  return brick<p.dispatch.w&&atomicLoad(&activity[activityRecord(brick)+10u])!=0u;
}
fn cellActive(cell:u32)->bool{
  let brick=cellBrick(cell);
  return brickActive(brick)&&cellResolution(cell)==acceptedBrickResolution(brick);
}
fn rowAccepted(row:u32)->bool{
  let requirements=rowRequirementOffset(row);let count=ta(requirements);
  for(var at=0u;at<count;at+=1u){let metadata=ta(requirements+1u+at);
    let brick=metadata>>TEMPLATE_CELL_RESOLUTION_BITS;
    let resolution=metadata&TEMPLATE_CELL_RESOLUTION_MASK;
    if(!brickActive(brick)||acceptedBrickResolution(brick)!=resolution){
      return false;
    }
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

fn brickDirectoryLookup(key:u32)->u32{
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

fn sampleVelocity(position:vec3f${fvrParameter})->vec3f{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  let clamped=clamp(position,0.5*spans,vec3f(p.dimensions.xyz)-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=vec3f(0.0);
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let cell=ownerCellAt(vec3i(floor(lattice)));if(cell==INVALID){continue;}
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);
    ${fvrSampleRead}
  }}}return result;
}

fn traceCharacteristic(position:vec3f,direction:f32${fvrParameter})->vec3f{
  let initialPosition=clampInsideEmbeddedBoundary(position);
  let initial=sampleVelocity(initialPosition${fvrArgument});
  let substeps=clamp(i32(ceil(length(initial)*p.frame.x)),1,16);
  let subDt=p.frame.x/f32(substeps);var traced=initialPosition;
  let lower=vec3f(0.5);let upper=vec3f(p.dimensions.xyz)-vec3f(0.5);
  for(var step=0;step<substeps;step+=1){
    // On the first substep traced is still initialPosition, whose velocity was
    // just sampled to choose the substep count. Reuse it rather than repeating
    // nine sparse owner resolutions.
    var first=initial;if(step>0){first=sampleVelocity(traced${fvrArgument});}
    let midpoint=clipBoundarySegment(traced,
      clamp(traced+direction*0.5*subDt*first,lower,upper));
    let candidate=traced+direction*subDt*sampleVelocity(midpoint${fvrArgument});
    traced=clipBoundarySegment(traced,clamp(candidate,lower,upper));
  }
  return traced;
}
fn traceDeparture(position:vec3f${fvrParameter})->vec3f{
  return traceCharacteristic(position,-1.0${fvrArgument});}
fn traceArrival(position:vec3f)->vec3f{
  return traceCharacteristic(position,1.0${fvrInvalidArgument});}

struct TransportStencil{cells:array<u32,8>,weights:array<f32,8>}

// Sharpening samples every corner of the same adaptive trilinear stencil.
// Compute its invariant probe and lattice coordinates once while retaining the
// exact corner order and weight expression used by transport.
fn transportStencil(position:vec3f)->TransportStencil{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);
    spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  let shifted=position/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result:TransportStencil;
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=spans*(vec3f(lower+offset)+vec3f(0.5));
    let cell=ownerCellAt(vec3i(floor(lattice)));
    let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
      *select(1.0-fraction.y,fraction.y,offset.y==1)
      *select(1.0-fraction.z,fraction.z,offset.z==1);
    result.cells[corner]=cell;
    result.weights[corner]=select(0.0,weight,cell!=INVALID);
  }
  return result;
}

${scalarAuthorityEntries}

const SIR1_CAUSE_SCALAR:u32=${SPARSE_CM12_SCALAR_RESULT_CAUSE.scalarWrite}u;
const SIR1_CAUSE_TOPOLOGY:u32=${SPARSE_CM12_SCALAR_RESULT_CAUSE.topologyWrite}u;
const SIR1_CAUSE_DEPENDENCY_CLOSURE:u32=${SPARSE_CM12_SCALAR_RESULT_CAUSE.dependencyClosure}u;
fn scaLogicalBrickDimensions()->vec3u{
  return (p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))/BRICK_FINE_RESOLUTION;
}
fn scaTileAtFine(qInput:vec3i)->u32{
  let q=vec3u(clamp(qInput,vec3i(0),vec3i(p.dimensions.xyz)-vec3i(1)));
  let brick=q/BRICK_FINE_RESOLUTION;let dims=scaLogicalBrickDimensions();
  let key=brick.x+dims.x*(brick.y+dims.y*brick.z);
  let local=(q%BRICK_FINE_RESOLUTION)/4u;
  return key*64u+local.x+4u*(local.y+4u*local.z);
}
fn scaCellBounds(cell:u32)->array<vec3i,2>{
  let b=cellBase(cell);let center=vec3f(taf(b),taf(b+1u),taf(b+2u));
  let width=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));var result:array<vec3i,2>;
  result[0]=vec3i(round(center-0.5*width));
  result[1]=vec3i(round(center+0.5*width))-vec3i(1);return result;
}
fn sirMassTileCell(rank:u32,lane:u32)->u32{
  let tile=sirResidentWorkTile(rank);if(tile==INVALID){return INVALID;}
  let logical=tile/64u;let localTile=tile%64u;let dims=scaLogicalBrickDimensions();
  let xy=dims.x*dims.y;if(xy==0u||logical>=xy*dims.z){return INVALID;}
  let bz=logical/xy;let rem=logical-bz*xy;let by=rem/dims.x;let bx=rem-by*dims.x;
  let tx=localTile&3u;let ty=(localTile>>2u)&3u;let tz=(localTile>>4u)&3u;
  let q=vec3u(bx,by,bz)*BRICK_FINE_RESOLUTION+4u*vec3u(tx,ty,tz)
    +vec3u(lane&3u,(lane>>2u)&3u,(lane>>4u)&3u);
  if(any(q>=p.dimensions.xyz)){return INVALID;}
  let cell=ownerCellAt(vec3i(q));if(cell==INVALID){return INVALID;}
  return select(INVALID,cell,all(scaCellBounds(cell)[0]==vec3i(q)));
}

@compute @workgroup_size(1)
fn beginSparseCM12SRR1Ingress(){
  _=sirResidentBeginFrame(cm12FCCandidateGeneration(),
    atomicLoad(&topologyArena[topologyWorklistBase()]),cm12FCSourceScalarParity());
}
@compute @workgroup_size(64)
fn resetSparseCM12SRR1CopiedEvents(@builtin(global_invocation_id)gid:vec3u){
  sirResidentResetCopiedEvents(gid.x);
}
@compute @workgroup_size(1)
fn beginSparseCM12SRR1ReceiptBatch(){sirResidentBeginReceiptBatch();}

fn scaInvalidateBrickTopologyClosure(brick:u32){
  if(brick>=p.dispatch.w){return;}let dims=scaLogicalBrickDimensions();
  let key=topology[p.topologyOffsets2.z+2u*brick+1u];let xy=dims.x*dims.y;
  if(xy==0u||key>=xy*dims.z){return;}let z=key/xy;let rem=key-z*xy;
  let y=rem/dims.x;let x=rem-y*dims.x;let origin=vec3i(i32(x),i32(y),i32(z));
  let span=i32(brickSpan(brick));let lower=max(vec3i(0),origin-vec3i(1));
  let upper=min(vec3i(dims)-vec3i(1),origin+vec3i(span));
  let generation=cm12FCCandidateGeneration()+1u;
  for(var bz=lower.z;bz<=upper.z;bz+=1){for(var by=lower.y;by<=upper.y;by+=1){
    for(var bx=lower.x;bx<=upper.x;bx+=1){let logical=u32(bx)+dims.x*(u32(by)+dims.y*u32(bz));
      for(var local=0u;local<64u;local+=1u){
        _=sirResidentAppendEvent(64u*logical+local,generation,SIR1_CAUSE_TOPOLOGY);
      }
  }}}
}

fn scaInvalidateCellClosure(cell:u32,generation:u32,cause:u32){
  let bounds=scaCellBounds(cell);let dims=scaLogicalBrickDimensions();
  let tileDims=4u*dims;let owner=scaTileAtFine(bounds[0]);
  // Never suppress extent publication because the owner is already stamped:
  // a fine-cell producer can pre-stamp it before a later coarse producer needs
  // to publish the rest of this cell's extent. Append deduplicates each
  // destination tile independently.
  _=sirResidentAppendEvent(owner,generation,cause);
  let lower=max(vec3i(0),bounds[0]/vec3i(4)-vec3i(1));
  let upper=min(vec3i(tileDims)-vec3i(1),bounds[1]/vec3i(4)+vec3i(1));
  for(var z=lower.z;z<=upper.z;z+=1){for(var y=lower.y;y<=upper.y;y+=1){
    for(var x=lower.x;x<=upper.x;x+=1){let q=vec3u(u32(x),u32(y),u32(z));
      let brick=q/4u;let sub=q%4u;let key=brick.x+dims.x*(brick.y+dims.y*brick.z);
      let tile=64u*key+sub.x+4u*(sub.y+4u*sub.z);
      _=sirResidentAppendEvent(tile,generation,cause|SIR1_CAUSE_DEPENDENCY_CLOSURE);
  }}}
}

var<workgroup>sirCovered:array<u32,64>;
var<workgroup>sirBank0Mismatch:array<u32,64>;
var<workgroup>sirBank1Mismatch:array<u32,64>;
var<workgroup>sirDependencyMismatch:array<u32,64>;

// Exact algebraic phase certificate for the final scalar banks. Returning 0/1
// means both physical rho banks contain exact dry/flooded capacity and both
// gamma banks contain exact one. Two is unresolved and can never certify clean.
fn sirExactScalarPhase(cell:u32)->u32{
  if(!cellActive(cell)||bitcast<u32>(cellOpenFraction(cell))!=0x3f800000u){return 2u;}
  let rho0=bitcast<u32>(state[p.stateOffsets0.x+cell]);
  let rho1=bitcast<u32>(state[p.stateOffsets0.y+cell]);
  let gamma0=bitcast<u32>(state[p.stateOffsets0.z+cell]);
  let gamma1=bitcast<u32>(state[p.stateOffsets0.w+cell]);
  if(gamma0!=0x3f800000u||gamma1!=0x3f800000u||rho0!=rho1){return 2u;}
  if(rho0==0u){return 0u;}if(rho0==0x3f800000u){return 1u;}return 2u;
}
fn sirExactScalarDependency(cell:u32,phase:u32)->bool{
  if(phase>1u){return false;}var sideMask=0u;
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);
    if(!rowAccepted(row)||bitcast<u32>(rowOpenFraction(row))!=0x3f800000u
      ||rowArea(row)<=1e-8){return false;}
    let coefficient=termCoefficient(incidenceTerm(incidence));
    let side=2u*rowAxis(row)+select(0u,1u,coefficient>0.0);sideMask|=1u<<side;
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);var neighbor=false;
    for(var term=begin;term<end;term+=1u){let other=termCell(term);
      if(other==cell){continue;}neighbor=true;
      if(sirExactScalarPhase(other)!=phase){return false;}
    }
    if(!neighbor){return false;}
  }
  return sideMask==63u;
}
fn sirRecordFinalScalarProducer(cell:u32){
  if(sirExactScalarPhase(cell)<=1u){return;}
  scaInvalidateCellClosure(cell,cm12FCCandidateGeneration()+1u,SIR1_CAUSE_SCALAR);
}

fn transportBeta(cell:u32)->f32{
  return f32(atomicLoad(&conditioning[cell]))/CM12_TRANSPORT_FIXED;
}

fn extrapolateTransportVelocity(id:u32,inputOffset:u32,outputOffset:u32){
  let input=inputOffset+4u*id;let output=outputOffset+4u*id;
  if(!cellActive(id)){
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    return;
  }
  if(state[input+3u]>0.5){
    state[output]=state[input];state[output+1u]=state[input+1u];
    state[output+2u]=state[input+2u];state[output+3u]=1.0;return;
  }
  var velocity=vec3f(0.0);var weight=0.0;
  let edgeOffsets=pressureTemplateWord(15u);
  let end=pressureTemplateWord(edgeOffsets+id+1u);
  for(var edge=pressureTemplateWord(edgeOffsets+id);edge<end;edge+=1u){
    let row=pressureTemplateWord(pressureEdgeRows()+edge);if(!rowAccepted(row)){continue;}
    let neighbor=fineSamples[pressureNeighborOffset()+edge];
    let source=inputOffset+4u*neighbor;if(state[source+3u]<=0.5){continue;}
    let w=bitcast<f32>(fineSamples[pressureExtrapolationWeightOffset()+edge]);
    velocity+=w*vec3f(state[source],state[source+1u],state[source+2u]);weight+=w;
  }
  if(weight>0.0){velocity/=weight;}
  state[output]=velocity.x;state[output+1u]=velocity.y;state[output+2u]=velocity.z;
  state[output+3u]=select(0.0,1.0,weight>0.0);
}

@compute @workgroup_size(64)
fn initializeTransportVelocity(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}let input=sourceCellVelocity()+4u*id;
  let output=destinationCellVelocity()+4u*id;
  if(!cellActive(id)){
    state[output]=0.0;state[output+1u]=0.0;state[output+2u]=0.0;state[output+3u]=0.0;
    return;
  }
  state[output]=state[input];state[output+1u]=state[input+1u];state[output+2u]=state[input+2u];
  state[output+3u]=select(0.0,1.0,state[sourceDensity()+id]>CM12_LIQUID_ISOVALUE);
}

@compute @workgroup_size(64)
fn extrapolateTransportVelocityToSource(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  extrapolateTransportVelocity(id,destinationCellVelocity(),sourceCellVelocity());
}

@compute @workgroup_size(64)
fn extrapolateTransportVelocityToDestination(@builtin(global_invocation_id)gid:vec3u){
  let id=acceptedTemplateCellInvocation(gid.x);if(id==INVALID){return;}
  extrapolateTransportVelocity(id,sourceCellVelocity(),destinationCellVelocity());
}

fn prepareTransportFaceRow(row:u32){
  if(rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  var touchesExtendedVelocity=false;var touchesLiquid=false;
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){
    let cell=termCell(term);let rho=state[sourceDensity()+cell];
    touchesExtendedVelocity=touchesExtendedVelocity
      ||cm12ExtensionTransportVelocity(cell).w>0.5;
    touchesLiquid=touchesLiquid||rho>CM12_LIQUID_ISOVALUE;
    if(touchesExtendedVelocity&&touchesLiquid){break;}
  }
  // Dry receiver rows still carry the extrapolated velocity band. The next
  // characteristic needs that velocity before any mass has reached the row;
  // treating rho==0 as inert pins and then releases a moving sparse front.
  // Beyond that eight-sweep validity band the interpolant is identically zero,
  // so those rows can still avoid an RK2 trace without changing a face value.
  if(!touchesExtendedVelocity){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  let departure=traceDeparture(rowCenter(row)${fvrRowArgument});
  let characteristic=sampleVelocity(departure${fvrRowArgument})[rowAxis(row)];
  state[destinationFaceVelocity()+row]=select(characteristic,
    mix(characteristic,state[sourceFaceVelocity()+row],0.4),touchesLiquid);
  if(hasRigidBodies()){
    let open=rowOpenFraction(row);let fluid=state[destinationFaceVelocity()+row];
    state[destinationFaceVelocity()+row]=open*fluid+(1.0-open)*rowSolidVelocity(row);
  }
}
@compute @workgroup_size(64)
fn publishSparseCM12MovingSolidVelocityRoots(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell==INVALID||!hasRigidBodies()){return;}
  if(cellOpenFraction(cell)<0.999||dynamicallyCoveredCell(cell)){
    cm12ResidentRecordExtensionIncidence(cell,32u);
  }
}
@compute @workgroup_size(64)
fn prepareTransportFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);
  if(row!=INVALID&&fpaPreparationRowLive(row)){prepareTransportFaceRow(row);}
}
// Construction-only FPA oracle capture. The host's immutable QA specialization
// first executes the original accepted-row preparation traversal, then snapshots
// the exact destination bank into the same stable authority arena production
// FPA consumes. This kernel is observational: it does not publish an FPA epoch
// or alter either face bank.
@compute @workgroup_size(64)
fn captureLegacyFacePreparationAuthority(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  fpaStorePreparedAuthority(row,bitcast<u32>(state[destinationFaceVelocity()+row]));
}
@compute @workgroup_size(64)
fn executeSparseCM12FacePreparation(@builtin(global_invocation_id)gid:vec3u){
  let row=fpaPreparationRowInvocation(gid.x);if(row==INVALID){return;}
  prepareTransportFaceRow(row);let bits=bitcast<u32>(state[destinationFaceVelocity()+row]);
  fpaStorePreparedAuthority(row,bits);
  _=fpaPreparationComplete(row);
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
  atomicAnd(&activity[27],0x7fffffffu);
  let brick=gid.x;if(brick>=p.dispatch.w||!brickActive(brick)){return;}
  let range=templateBrickCellRange(brick,scheduledBrickResolution(brick));
  for(var at=0u;at<range.y;at+=1u){
    let id=range.x+at;if(cellOpenVolume(id)<=1e-8){continue;}
    let coverage=injectionCoverage(id);
    let clippedCoverage=coverage*cellOpenFraction(id);
    state[p.stateOffsets0.x+id]=max(state[p.stateOffsets0.x+id],clippedCoverage);
    state[p.stateOffsets0.y+id]=max(state[p.stateOffsets0.y+id],clippedCoverage);
    if(coverage>0.0){state[p.stateOffsets0.z+id]=1.0;state[p.stateOffsets0.w+id]=1.0;
      cm12ResidentRecordExtensionIncidence(id,16u);
      scaInvalidateCellClosure(id,cm12FCCandidateGeneration()+1u,SIR1_CAUSE_SCALAR);}
  }
}

// CM12 Sec. 3.4 steps 1-3. Backward-advect cumulative gamma, then scatter
// gamma_i w^-_li into each donor's volume-weighted beta column.
@compute @workgroup_size(64)
fn traceGammaAndBeta(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let id=sirMassTileCell(wid.x,lane);if(id==INVALID){return;}
  if(!cellTransportActive(id)){state[destinationGamma()+id]=1.0;return;}
  let b=cellBase(id);
  let departure=traceDeparture(vec3f(taf(b),taf(b+1u),taf(b+2u))${fvrInvalidArgument});
  let stencil=transportStencil(departure);
  var visible=0.0;var sampledGamma=0.0;
  for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];let weight=stencil.weights[corner];
    visible+=weight;if(cell!=INVALID){sampledGamma+=weight*state[sourceGamma()+cell];}}
  let advectedGamma=cm12ConditionedGamma(sampledGamma,visible);
  state[destinationGamma()+id]=advectedGamma;if(visible<=1e-9){return;}
  for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];let weight=stencil.weights[corner];
    if(cell==INVALID||weight<=0.0){continue;}
    let coefficient=advectedGamma*weight/visible;
    let contribution=cm12VolumeWeightedBetaContribution(
      cellVolume(id),cellVolume(cell),coefficient);
    atomicAdd(&conditioning[cell],i32(round(contribution*CM12_TRANSPORT_FIXED)));
  }
}

// CM12 Sec. 3.4 steps 6-7. Every deficient donor returns its missing column
// weight along the forward characteristic. The fixed-point scatters are
// deterministic and keep rho and gamma transfers paired.
@compute @workgroup_size(64)
fn scatterDensityDeficit(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let donor=sirMassTileCell(wid.x,lane);
  if(donor==INVALID||!cellTransportActive(donor)){return;}
  let deficit=max(0.0,1.0-transportBeta(donor));
  if(deficit<=1.0/CM12_TRANSPORT_FIXED){return;}let b=cellBase(donor);
  let arrival=traceArrival(vec3f(taf(b),taf(b+1u),taf(b+2u)));
  let stencil=transportStencil(arrival);
  var visible=0.0;for(var corner=0u;corner<8u;corner+=1u){visible+=stencil.weights[corner];}
  if(visible<=1e-9){
    atomicAdd(&conditioning[p.counts.x+donor],i32(round(
      state[sourceDensity()+donor]*deficit*CM12_TRANSPORT_FIXED)));
    atomicAdd(&conditioning[2u*p.counts.x+donor],i32(round(
      state[sourceGamma()+donor]*deficit*CM12_TRANSPORT_FIXED)));return;
  }
  for(var corner=0u;corner<8u;corner+=1u){let cell=stencil.cells[corner];let weight=stencil.weights[corner];
    if(cell==INVALID||weight<=0.0){continue;}let normalized=weight/visible;
    let densityTransfer=cm12VolumeScaledDeficitTransfer(state[sourceDensity()+donor],
      cellVolume(donor),cellVolume(cell),deficit,normalized);
    let gammaTransfer=cm12VolumeScaledDeficitTransfer(state[sourceGamma()+donor],
      cellVolume(donor),cellVolume(cell),deficit,normalized);
    atomicAdd(&conditioning[p.counts.x+cell],i32(round(densityTransfer*CM12_TRANSPORT_FIXED)));
    atomicAdd(&conditioning[2u*p.counts.x+cell],i32(round(gammaTransfer*CM12_TRANSPORT_FIXED)));
  }
}

// CM12 Sec. 3.4 steps 4-5 plus the forward-deficit resolve. Scaling each
// donor by max(1,beta_l) clamps the column without materializing A.
@compute @workgroup_size(64)
fn gatherConservativeDensity(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let id=sirMassTileCell(wid.x,lane);if(id==INVALID){return;}
  if(!cellActive(id)){
    state[destinationDensity()+id]=0.0;state[destinationGamma()+id]=1.0;return;
  }
  if(!cellTransportActive(id)){
    if(dynamicallyCoveredCell(id)){
      state[destinationDensity()+id]=state[sourceDensity()+id];
      state[destinationGamma()+id]=state[sourceGamma()+id];
    }else{state[destinationDensity()+id]=0.0;state[destinationGamma()+id]=1.0;}
    return;
  }
  let b=cellBase(id);
  let departure=traceDeparture(vec3f(taf(b),taf(b+1u),taf(b+2u))${fvrInvalidArgument});
  let stencil=transportStencil(departure);
  let advectedGamma=state[destinationGamma()+id];var visible=0.0;
  for(var corner=0u;corner<8u;corner+=1u){visible+=stencil.weights[corner];}
  var rhoNext=0.0;var gammaNext=0.0;
  if(visible>1e-9){for(var corner=0u;corner<8u;corner+=1u){
    let cell=stencil.cells[corner];let weight=stencil.weights[corner];if(cell==INVALID||weight<=0.0){continue;}
    let coefficient=cm12ConditionedRowCoefficient(
      advectedGamma,weight/visible,transportBeta(cell));
    rhoNext+=coefficient*state[sourceDensity()+cell];gammaNext+=coefficient;
  }}
  rhoNext+=f32(atomicLoad(&conditioning[p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  gammaNext+=f32(atomicLoad(&conditioning[2u*p.counts.x+id]))/CM12_TRANSPORT_FIXED;
  if(rhoNext<CM12_DRY_CELL_THRESHOLD){gammaNext=1.0;}
  let nextDensity=max(0.0,rhoNext);
  let densityChanged=bitcast<u32>(nextDensity)!=bitcast<u32>(state[sourceDensity()+id]);
  state[destinationDensity()+id]=nextDensity;
  state[destinationGamma()+id]=max(0.0,gammaNext);
  if(densityChanged){cm12ResidentRecordExtensionIncidence(id,4u);}
}

@compute @workgroup_size(64)
fn compareSparseCM12MassResult(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let cell=sirMassTileCell(wid.x,lane);var covered=0u;var mismatch0=0u;
  var mismatch1=0u;var dependencyMismatch=0u;
  if(cell!=INVALID){let phase=sirExactScalarPhase(cell);covered=4u;
    mismatch0=select(0u,1u,phase>1u);mismatch1=mismatch0;
    dependencyMismatch=select(1u,0u,sirExactScalarDependency(cell,phase));
  }
  sirCovered[lane]=covered;sirBank0Mismatch[lane]=mismatch0;
  sirBank1Mismatch[lane]=mismatch1;sirDependencyMismatch[lane]=dependencyMismatch;
  workgroupBarrier();var width=32u;loop{if(lane<width){sirCovered[lane]+=sirCovered[lane+width];
    sirBank0Mismatch[lane]+=sirBank0Mismatch[lane+width];
    sirBank1Mismatch[lane]+=sirBank1Mismatch[lane+width];
    sirDependencyMismatch[lane]+=sirDependencyMismatch[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u){let tile=sirResidentWorkTile(wid.x);let generation=sirResidentCandidateGeneration();
    _=sirResidentPublishReceipt(wid.x,tile,sirResidentTopologyGeneration(),generation,
      generation,generation,sirCovered[0],sirCovered[0],sirBank0Mismatch[0],
      sirBank1Mismatch[0],generation,select(0u,generation,sirDependencyMismatch[0]==0u),
      1u,generation);}
}

// Massless markers riding the same characteristic the conservative transport
// integrates, so a coloured parcel and the mass it stands for cannot drift
// apart by construction: both call traceCharacteristic against the same
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
  return ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
}

// Seeding is re-runnable on purpose. Re-seeding mid-run re-reads the mixing
// from that instant rather than from t=0, which is the question being asked
// most of the time once a scene has already broken up.
@compute @workgroup_size(64)
fn seedTracers(@builtin(global_invocation_id)gid:vec3u){
  let index=gid.x;if(index>=tracerCount()){return;}
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
fn advanceTracers(@builtin(global_invocation_id)gid:vec3u){
  let index=gid.x;if(index>=tracerCount()){return;}
  let at=tracerBase(index);if(state[at+3u]<0.5){return;}
  let traced=traceArrival(vec3f(state[at],state[at+1u],state[at+2u]));
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

// Two immutable, row-owned iterations replace the mirrored xyz/zyx chain.
// An accepted composite row owns each physical negative/positive subface pair
// exactly once. It quantizes integrated rho and gamma receipts and adds each
// integer to one endpoint and its exact opposite to the other, making both
// fields conservative independent of dispatch order and unequal cell volumes.
fn scatterGammaRow(row:u32,inputRho:u32,inputGamma:u32){
  let rowAreaValue=rowArea(row);if(rowAreaValue<=1e-8){return;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var negativeCount=0.0;var positiveCount=0.0;
  for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
    negativeCount+=select(0.0,1.0,coefficient<0.0);
    positiveCount+=select(0.0,1.0,coefficient>0.0);}
  if(negativeCount==0.0||positiveCount==0.0){return;}
  let pairArea=rowAreaValue/(negativeCount*positiveCount);
  // A regular interior cell participates in six rows. The CM12 flux already
  // contains its one-half diffusion coefficient, so /3 gives a convex
  // simultaneous six-neighbour update at the paper's full 1/30 s step.
  let scale=min(1.0,30.0*p.frame.x)/3.0;
  for(var negativeTerm=begin;negativeTerm<end;negativeTerm+=1u){
    if(termCoefficient(negativeTerm)>=0.0){continue;}
    let negative=termCell(negativeTerm);if(!cellTransportActive(negative)){continue;}
    for(var positiveTerm=begin;positiveTerm<end;positiveTerm+=1u){
      if(termCoefficient(positiveTerm)<=0.0){continue;}
      let positive=termCell(positiveTerm);if(!cellTransportActive(positive)){continue;}
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
      atomicAdd(&conditioning[positive],-rhoReceipt);
      atomicAdd(&conditioning[p.counts.x+negative],gammaReceipt);
      atomicAdd(&conditioning[p.counts.x+positive],-gammaReceipt);
    }
  }
}

@compute @workgroup_size(64)
fn scatterGammaSnapshot(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row!=INVALID){
    scatterGammaRow(row,destinationDensity(),destinationGamma());
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
fn scatterGammaRefinement(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row!=INVALID){
    scatterGammaRow(row,p.stateOffsets2.x,p.stateOffsets2.y);
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
@compute @workgroup_size(64)
fn prepareSharpeningField(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID){
    prepareSharpeningCell(cell);}
}
fn sampleSharpeningField(position:vec3f)->vec4f{
  let probe=ownerCellAt(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(p.dimensions.xyz)-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe!=INVALID){let b=cellBase(probe);
    spans=vec3f(taf(b+4u),taf(b+5u),taf(b+6u));}
  let shifted=position/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result=vec4f(0.0);
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=spans*(vec3f(lower+offset)+vec3f(0.5));
    let cell=ownerCellAt(vec3i(floor(lattice)));if(cell==INVALID){continue;}
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
    let owner=ownerCellAt(vec3i(floor(position)));if(owner==INVALID){break;}
    let gradient=field.yzw;let magnitude=length(gradient);
    if(magnitude<1e-6){break;}
    let distance=min(0.5*cellMinimumWidth(owner),maximumDistance-travelled);
    let candidate=position+gradient/magnitude*distance;
    if(ownerCellAt(vec3i(floor(candidate)))==INVALID){break;}
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
  let position=traceSharpeningMass(cell);let stencil=transportStencil(position);
  var total=0.0;var lastCorner=INVALID;
  for(var corner=0u;corner<8u;corner+=1u){let targetCell=stencil.cells[corner];
    let weight=stencil.weights[corner];
    if(targetCell!=INVALID&&weight>0.0){total+=weight;lastCorner=corner;}}
  if(total<=1e-8){
    atomicAdd(&conditioning[3u*p.counts.x+cell],removedFixed);return;
  }
  var remainingFixed=removedFixed;
  for(var corner=0u;corner<8u;corner+=1u){let targetCell=stencil.cells[corner];
    let weight=stencil.weights[corner];if(targetCell==INVALID||weight<=0.0){continue;}
    var offeredFixed=i32(round(f32(removedFixed)*weight/total));
    if(corner==lastCorner){offeredFixed=remainingFixed;}
    else{offeredFixed=clamp(offeredFixed,0,remainingFixed);}
    atomicAdd(&conditioning[3u*p.counts.x+targetCell],offeredFixed);
    remainingFixed-=offeredFixed;
  }
}
@compute @workgroup_size(64)
fn scatterSharpeningMass(@builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);if(cell!=INVALID){
    scatterSharpeningCell(cell);}
}
fn finalizeSharpeningCell(cell:u32){
  if(!cellTransportActive(cell)){
    if(!dynamicallyCoveredCell(cell)){
      state[destinationDensity()+cell]=0.0;state[destinationGamma()+cell]=1.0;
    }
    if(bitcast<u32>(state[destinationDensity()+cell])
      !=bitcast<u32>(state[sourceDensity()+cell])){
      cm12ResidentRecordExtensionIncidence(cell,4u);}
    sirRecordFinalScalarProducer(cell);
    return;
  }
  let delta=state[p.stateOffsets5.x+cell];
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
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
  if(bitcast<u32>(density)!=bitcast<u32>(state[sourceDensity()+cell])){
    cm12ResidentRecordExtensionIncidence(cell,4u);}
  sirRecordFinalScalarProducer(cell);
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
    if(bitcast<u32>(state[destinationDensity()+cell])
      !=bitcast<u32>(state[sourceDensity()+cell])){
      cm12ResidentRecordExtensionIncidence(cell,4u);}
    sirRecordFinalScalarProducer(cell);return;
  }
  let incoming=f32(atomicLoad(&conditioning[3u*p.counts.x+cell]))/CM12_TRANSPORT_FIXED;
  state[destinationDensity()+cell]=max(0.0,state[destinationDensity()+cell]
    +state[p.stateOffsets5.x+cell]+incoming/cellVolume(cell));
  if(bitcast<u32>(state[destinationDensity()+cell])
    !=bitcast<u32>(state[sourceDensity()+cell])){
    cm12ResidentRecordExtensionIncidence(cell,4u);}
  sirRecordFinalScalarProducer(cell);
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

@compute @workgroup_size(64)
fn forceFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=acceptedTemplateRowInvocation(gid.x);if(row==INVALID){return;}
  if(!rowAccepted(row)){state[destinationFaceVelocity()+row]=0.0;return;}
  if(rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;
  }
  let open=select(1.0,rowOpenFraction(row),hasRigidBodies());
  state[destinationFaceVelocity()+row]+=open*p.frame.x*p.acceleration[rowAxis(row)];
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

fn pressureCellMembershipFromDensity(id:u32,rho:f32)->bool{
  return cellActive(id)&&rho>=CM12_LIQUID_ISOVALUE
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
  let id=temporalScalarCellInvocation(gid.x);if(id==INVALID){return;}
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
fn psaFrameGeneration()->u32{return atomicLoad(&activity[0]);}
fn psaTopologyGeneration()->u32{return ptrTopologyGeneration();}
fn psaPCMGeneration()->u32{return pcfPCMGeneration();}
fn psaPCFGeneration()->u32{return pcfAcceptedGeneration();}
fn psaExpectedProducerReceipts()->u32{return pcfBrickWorkCount();}
fn psaPCMCellWorkgroupCount()->u32{return (pcmCellAcceptedCount()+63u)/64u;}
fn psaCellBrick(cell:u32)->u32{return cellBrick(cell);}
fn psaBrickWetExact(brick:u32)->bool{return cachedPressureBrickRange(brick).y!=0u;}
fn psaHierarchyParent(level:u32,brick:u32)->u32{return pcfHierarchyParent(level,brick);}
fn psaHierarchyNodeActiveExact(linear:u32)->bool{
  let address=pressureHierarchyGroupAddress(linear);if(address.x==INVALID){return false;}
  let range=pcfHierarchyChildRange(address.x,address.y);
  for(var at=0u;at<range.y;at+=1u){
    if(psaWetBrickContains(pcfHierarchyChild(address.x,range.x+at))){return true;}}
  return false;
}
fn psaPressureArithmeticActive()->bool{return pipelinedPressureActive();}
fn psaPressureAddressingReady()->bool{return pabPressureAddressingReady();}
@compute @workgroup_size(64)
fn markSparseCM12PressureSolveFromPersistentBricks(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  if(lane!=0u){return;}let brick=pcfBrickRankSelect(wid.x);
  if(brick!=INVALID){_=psaMarkTopologyBrickBlast(brick,psaBrickWetExact(brick));}
}
fn fpaFrameGeneration()->u32{return max(1u,atomicLoad(&activity[0]));}
fn fpaTopologyGeneration()->u32{return ptrTopologyGeneration();}
fn fpaPCMGeneration()->u32{return max(1u,pcmRowAcceptedGeneration());}
fn fpaSourceParity()->u32{return cm12FCSourceFaceParity();}
fn fpaPolicyBits()->u32{return bitcast<u32>(p.frame.x)
  ^bitcast<u32>(p.frame.y)^select(0u,0x80000000u,hasRigidBodies());}
fn fpaExpectedPreparationReceipts()->u32{return atomicLoad(&activity[
  ${(incrementalActivityLayout?.headerBaseWords ?? 0) + 6}u]);}
fn fpaExpectedProjectionReceipts()->u32{return 0u;}
fn fpaPreparationRowLive(row:u32)->bool{return rowAccepted(row);}
fn fpaProjectionRowLive(row:u32)->bool{return pcmRowContains(row);}
fn fpaPreparationDependencyGeneration(row:u32)->u32{
  _=row;return max(1u,incrementalActivityGeneration());}
fn fpaProjectionDependencyGeneration(row:u32)->u32{
  _=row;return max(1u,pcfAcceptedGeneration());}
@compute @workgroup_size(64)
fn markSparseCM12FacePreparationFromActivity(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let brick=incrementalActivityBrickInvocation(wid.x);
  if(brick!=INVALID){let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    for(var local=lane;local<range.y;local+=64u){let cell=range.x+local;
      for(var at=incidenceBegin(cell);at<incidenceEnd(cell);at+=1u){
        _=fpaMarkPreparationRow(incidenceRow(at),7u,0u,false);}}}
  workgroupBarrier();if(lane==0u&&brick!=INVALID){fpaCoverPreparationReceipt();}
}
@compute @workgroup_size(64)
fn markSparseCM12FaceProjectionFromPressure(
 @builtin(global_invocation_id)gid:vec3u){
  let cell=pressureCellInvocation(gid.x);if(cell==INVALID){return;}
  _=fpaMarkProjectionPressureCell(cell,
    bitcast<u32>(state[p.stateOffsets2.x+cell]),512u);
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
  let count=pcmCellAcceptedCount();let groups=(count+63u)/64u;
  fineSamples[0]=count;fineSamples[1]=groups;fineSamples[2]=1u;fineSamples[3]=1u;
  atomicStore(&conditioning[0],i32(count));atomicStore(&conditioning[1],i32(groups));
  atomicStore(&conditioning[2],1);atomicStore(&conditioning[3],1);
  fineSamples[PRESSURE_REPAIR_HEADER+PRESSURE_REPAIR_CELL_CHANGES]=pcmCellFirstFault();
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
    if(pcmCellContains(cell)){liquidCount+=1u;liquidPhiSum+=w*phi;liquidWeight+=w;}
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
  let row=temporalScalarRowInvocation(gid.x);if(row==INVALID){return;}
  let enabled=classifyPressureRow(row);_=pcfStoreThetaAndRecord(row,
    state[p.stateOffsets3.x+row]);_=pcmRowSetCandidate(row,enabled,2u,false);
}

@compute @workgroup_size(1)
fn finalizeCanonicalPressureRowFrontier(){_=pcmRowFinalizeFrontier();}

@compute @workgroup_size(1)
fn finalizeCanonicalPressureRows(){
  _=pcmRowFinalize();
  let count=pcmRowAcceptedCount();let groups=(count+63u)/64u;
  let header=pressureRowHeader();fineSamples[header]=count;fineSamples[header+1u]=groups;
  fineSamples[header+2u]=1u;fineSamples[header+3u]=1u;
  atomicStore(&conditioning[header],i32(count));
  atomicStore(&conditioning[header+1u],i32(groups));
  atomicStore(&conditioning[header+2u],1);atomicStore(&conditioning[header+3u],1);
  atomicStore(&activity[PRESSURE_ACTIVE_ROW_COUNT],count);
  fineSamples[PRESSURE_REPAIR_HEADER+PRESSURE_REPAIR_ROW_CHANGES]=pcmRowFirstFault();
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

fn refinePressureHierarchyCorrectionWork(lid:vec3u,wid:vec3u,level:u32){
  let group=wid.x;let count=pressureHierarchyGroupCount(level);var offDiagonal=0.0;
  let descriptor=pressureHierarchyDescriptor(level);
  let offsets=pressureTemplateWord(descriptor+6u);
  let records=pressureTemplateWord(descriptor+7u);
  if(group<count){
    let edgeBegin=pressureTemplateWord(offsets+group);
    let edgeEnd=pressureTemplateWord(offsets+group+1u);
    for(var edge=edgeBegin+lid.x;edge<edgeEnd;edge+=64u){
      let otherGroup=pressureTemplateWord(records+3u*edge);
      offDiagonal+=persistentHierarchyEdge(level,edge)
        *candidateState[pressureHierarchyAOffset(level)+otherGroup];
    }
  }
  reduceA[lid.x]=offDiagonal;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&group<count){
    let diagonal=persistentHierarchyDiagonal(level,group);
    let current=candidateState[pressureHierarchyAOffset(level)+group];
    let rhs=candidateState[pressureHierarchyRhsOffset(level)+group];
    let residual=rhs-(diagonal*current+reduceA[0]);
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
  let address=psaActiveHierarchyNodeAddress(wid.x);
  restrictPressureHierarchyResidualWork(lid,vec3u(address.y,0u,0u),address.x);
}
@compute @workgroup_size(64)
fn refinePressureHierarchyCorrection(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let address=psaActiveHierarchyNodeAddress(wid.x);
  refinePressureHierarchyCorrectionWork(lid,vec3u(address.y,0u,0u),address.x);
}

@compute @workgroup_size(64)
fn combinePressureHierarchyCorrection(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){
  if(lane!=0u){return;}let brick=psaWetBrickInvocation(wid.x);
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
  let brick=psaWetBrickInvocation(wid.x);var residual=0.0;
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

fn refineBrickAggregateCorrectionWork(lid:vec3u,wid:vec3u,sourceOffset:u32,
 destinationOffset:u32,weight:f32){
  let brick=psaWetBrickInvocation(wid.x);var offDiagonal=0.0;
  if(brick!=INVALID&&cachedPressureBrickRange(brick).y!=0u){
    let topologyBase=brickAggregateTopology();let offsets=topologyBase+4u;
    let recordBase=offsets+p.dispatch.w+1u;
    let begin=pressureTemplateWord(offsets+brick);let end=pressureTemplateWord(offsets+brick+1u);
    for(var coarseEdge=begin+lid.x;coarseEdge<end;coarseEdge+=64u){
      let otherBrick=pressureTemplateWord(recordBase+3u*coarseEdge);
      offDiagonal+=persistentBrickAggregateEdge(coarseEdge)
        *candidateState[sourceOffset+otherBrick];
    }
  }
  reduceA[lid.x]=offDiagonal;workgroupBarrier();var width=32u;loop{
    if(lid.x<width){reduceA[lid.x]+=reduceA[lid.x+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;
  }
  if(lid.x==0u&&brick!=INVALID){
    let diagonal=persistentBrickAggregateDiagonal(brick);
    let current=candidateState[sourceOffset+brick];
    let rhs=candidateState[brickAggregateRhsOffset()+brick];
    let residual=rhs-(diagonal*current+reduceA[0]);
    candidateState[destinationOffset+brick]
      =current+select(0.0,weight*residual/diagonal,diagonal>1e-12);
  }
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB1(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
    brickAggregateBOffset(),0.5821642);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA2(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateBOffset(),
    brickAggregateAOffset(),0.53435177);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB3(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
    brickAggregateBOffset(),1.23564);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA4(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateBOffset(),
    brickAggregateAOffset(),0.83447);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB5(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
    brickAggregateBOffset(),0.64192);
}

@compute @workgroup_size(64)
fn refineBrickAggregateBtoA6(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateBOffset(),
    brickAggregateAOffset(),0.54557);
}

@compute @workgroup_size(64)
fn refineBrickAggregateAtoB7(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  refineBrickAggregateCorrectionWork(lid,wid,brickAggregateAOffset(),
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

fn applyOperator(cell:u32,inputOffset:u32)->f32{
  // The pressure epoch has already classified rows and assembled the exact
  // diagonal. Physical topology packing expands every off-diagonal once.
  let edgeOffsets=pressureTemplateWord(15u);
  var result=state[p.stateOffsets2.z+cell]*state[inputOffset+cell];
  let end=pressureTemplateWord(edgeOffsets+cell+1u);
  for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
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
  let targetDivergence=select(cm12VolumeCorrectionDivergence(
    rho,p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
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
  let targetDivergence=select(cm12VolumeCorrectionDivergence(
    rho,p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
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
var<workgroup>candidateFaceScheduled:u32;
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
 @builtin(local_invocation_id)lid:vec3u,
  @builtin(workgroup_id)wid:vec3u){
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

@compute @workgroup_size(64)
fn projectFaces(@builtin(global_invocation_id)gid:vec3u){
  let row=pressureRowInvocation(gid.x);if(row==INVALID){return;}
  let theta=state[p.stateOffsets3.x+row];if(theta<=0.0||rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());return;}
  var jump=0.0;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var at=begin;at<end;at+=1u){let cell=termCell(at);if(pcmCellContains(cell)){
    jump+=termCoefficient(at)*state[p.stateOffsets2.x+cell];}}
  let pressureOpen=select(1.0,rowPressureOpenFraction(row),hasRigidBodies());
  state[destinationFaceVelocity()+row]-=pressureOpen*jump/theta;
}
@compute @workgroup_size(64)
fn executeSparseCM12FaceProjection(@builtin(global_invocation_id)gid:vec3u){
  let row=fpaProjectionRowInvocation(gid.x);if(row==INVALID){return;}
  // Project the live destination authority. Body forces are applied after
  // preparation and before the pressure solve; restoring the pre-force FPA
  // snapshot here silently erased them and changed activity/topology. The
  // stored prepared bits remain a preparation reuse receipt, not a substitute
  // for the force-conditioned face bank.
  let theta=state[p.stateOffsets3.x+row];
  if(theta<=0.0||rowArea(row)<=1e-8){
    state[destinationFaceVelocity()+row]=select(0.0,rowSolidVelocity(row),hasRigidBodies());
  }else{var jump=0.0;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var at=begin;at<end;at+=1u){let cell=termCell(at);if(pcmCellContains(cell)){
      jump+=termCoefficient(at)*state[p.stateOffsets2.x+cell];}}
    let pressureOpen=select(1.0,rowPressureOpenFraction(row),hasRigidBodies());
    state[destinationFaceVelocity()+row]-=pressureOpen*jump/theta;}
  _=fpaProjectionComplete(row);
}

// The extrapolation recurrence uses the accepted collocated liquid velocities
// only at the liquid/air interface. Both collocated banks are recurrence
// scratch by the time projection publishes the next accepted source, so a
// comparison with either bank cannot prove that a boundary seed is unchanged.
// VEX1 retains the prior accepted seed bits alongside its air cache. Publish
// only interface seeds whose newly projected value differs from that receipt;
// VEX1 expands the compact root set by the fixed eight-layer dependency closure.
fn cm12VelocityExtensionChangedInterfaceSeed(cell:u32,velocity:vec3f)->bool{
  if(state[destinationDensity()+cell]<=CM12_LIQUID_ISOVALUE){return false;}
  let edgeOffsets=pressureTemplateWord(15u);
  let end=pressureTemplateWord(edgeOffsets+cell+1u);
  for(var edge=pressureTemplateWord(edgeOffsets+cell);edge<end;edge+=1u){
    let row=pressureTemplateWord(pressureEdgeRows()+edge);if(!rowAccepted(row)){continue;}
    let neighbor=fineSamples[pressureNeighborOffset()+edge];
    if(neighbor<p.counts.x&&cellActive(neighbor)
      &&state[destinationDensity()+neighbor]<=CM12_LIQUID_ISOVALUE){
      return cm12ExtensionAcceptedVelocityChanged(cell,velocity);
    }
  }
  return false;
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
    cm12ResidentRecordExtensionIncidence(id,2u);
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
  for(var axis=0u;axis<3u;axis+=1u){if(weight[axis]>0.0){velocity[axis]/=weight[axis];}}
  let velocityChanged=any(bitcast<vec3u>(velocity)!=bitcast<vec3u>(previousVelocity));
  if(velocityChanged){
    incrementalActivityMarkCellClosure(id,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.velocityCharacteristic}u);
  }
  if(velocityChanged||cm12VelocityExtensionChangedInterfaceSeed(id,velocity)){
    cm12ResidentRecordExtensionIncidence(id,2u);
  }
  state[destinationCellVelocity()+4u*id]=velocity.x;state[destinationCellVelocity()+4u*id+1u]=velocity.y;
  state[destinationCellVelocity()+4u*id+2u]=velocity.z;state[destinationCellVelocity()+4u*id+3u]=0.0;
  let targetDivergence=select(cm12VolumeCorrectionDivergence(rawPressureDensity(id),
    p.frame.y*cellMinimumWidth(id),p.frame.x),0.0,cellSeparatingMinimum(id));
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
  if(any(bitcast<vec3u>(previous)!=bitcast<vec3u>(next))){
    incrementalActivityMarkCellClosure(cell,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.velocityCharacteristic}u);
    cm12ResidentRecordExtensionIncidence(cell,2u);
  }
  state[at]=next.x;state[at+1u]=next.y;state[at+2u]=next.z;
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

${vexActivityBatchLayout ? /* wgsl */ `
// VEX1/A4D2 resident bridge. These functions are the sole semantic seam
// between the generic compact authorities and the frozen CM12 field/layout
// arithmetic below.
fn cm12ResidentCellTile(cell:u32)->vec2u{
  let brick=cellBrick(cell);let minimum=cellMinimum(cell);
  let local=(minimum%vec3u(BRICK_FINE_RESOLUTION))/4u;
  return vec2u(brick,local.x+4u*(local.y+4u*local.z));
}
fn cm12ResidentRecordExtensionIncidence(cell:u32,cause:u32){
  if(cell>=p.counts.x||!cellActive(cell)){return;}
  _=cm12ExtensionRecordRoot(cell,cause);
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    let row=incidenceRow(incidence);if(!rowAccepted(row)){continue;}
    let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
    for(var term=begin;term<end;term+=1u){let neighbor=termCell(term);
      if(neighbor!=cell&&cellActive(neighbor)){_=cm12ExtensionRecordRoot(neighbor,cause);}
    }
  }
}
fn cm12ResidentVelocityExtensionRoot(cell:u32,cause:u32,generation:u32)->bool{
  _=cause;return cell<p.counts.x&&generation==cm12ExtensionGeneration();
}
fn cm12ResidentVelocityExtensionClosure(cell:u32,depth:u32,generation:u32)->bool{
  _=depth;return cell<p.counts.x&&generation==cm12ExtensionGeneration();
}
fn cm12ResidentVelocityExtensionScheduled(cell:u32,depth:u32,generation:u32)->bool{
  if(cell>=p.counts.x||generation!=cm12ExtensionGeneration()){return false;}
  let owner=cm12ResidentCellTile(cell);
  if(!cm12FramePlanCurrentTileScheduled(owner.x,owner.y,0u)){
    cm12FramePlanMarkCurrentTileFault(owner.x,owner.y,0u);return false;
  }
  cm12FramePlanMarkCurrentTileExecuted(owner.x,owner.y,0u);_=depth;return true;
}
fn cm12ResidentVelocityExtensionOwner(cell:u32)->u32{
  if(cell>=p.counts.x){return INVALID;}
  let brick=cellBrick(cell);let rung=cellResolution(cell);
  // The cell ID already names the local template/rung identity. The owner word is
  // therefore the exact immutable logical brick key, accepted only while that
  // rung remains current. Do not include the live active bit: receiver planning
  // toggles it before accepted topology publication and used to invalidate an
  // otherwise unchanged cache identity in the all-fine companion lane.
  if(brick>=p.dispatch.w||rung!=acceptedBrickResolution(brick)){return INVALID;}
  return topology[p.topologyOffsets2.z+2u*brick+1u];
}
fn cm12ResidentVelocityExtensionFault(cell:u32,depth:u32){
  // Expansion authors FPL Next, so it must never mutate accepted FPL Current.
  // Only the recurrence consumer owns a current-plan fault disposition.
  if(cm12ExtensionLoad(${vexActivityBatchLayout.velocityExtension.headerBaseWords
      + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved}u)
      ==cm12ExtensionPhaseConsuming&&cell<p.counts.x){
    let owner=cm12ResidentCellTile(cell);
    cm12FramePlanMarkCurrentTileFault(owner.x,owner.y,0u);
    // First-fault QA receipt in dispatch words already zeroed by fail-closed.
    // No production decision consumes these words after a fault.
    if(cm12ExtensionLoad(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultCell}u)==cell
      &&cm12ExtensionLoad(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.firstFaultDepth}u)==depth){
      let slot=cm12FramePlanCurrentSlot();
      let tile=cm12FramePlanTileAt(slot,owner.x,owner.y);
      cm12ExtensionStore(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchY}u,owner.x);
      cm12ExtensionStore(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootDispatchZ}u,owner.y);
      cm12ExtensionStore(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchY}u,slot);
      cm12ExtensionStore(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierADispatchZ}u,
        cm12FramePlanLoad(tile));
      cm12ExtensionStore(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchY}u,
        cm12FramePlanLoad(tile+1u));
      cm12ExtensionStore(${vexActivityBatchLayout.velocityExtension.headerBaseWords
        + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.frontierBDispatchZ}u,
        cm12FramePlanLoad(cm12FramePlanBrickAt(slot,owner.x)+4u));
    }
  }
  _=depth;
}
fn cm12ResidentVelocityExtensionDirtyCause(cause:u32)->u32{
  var result=0u;
  if((cause&${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.bootstrap}u)!=0u){
    result|=${SPARSE_CM12_DIRTY_CAUSE_BIT.generationMismatch}u;}
  if((cause&${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.projectedVelocity}u)!=0u){
    result|=${SPARSE_CM12_DIRTY_CAUSE_BIT.velocityCharacteristic}u;}
  if((cause&${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.liquidClassification}u)!=0u){
    result|=${SPARSE_CM12_DIRTY_CAUSE_BIT.densityChanged
      | SPARSE_CM12_DIRTY_CAUSE_BIT.phaseCrossing}u;}
  if((cause&${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.topologyOrEdgeOwnership}u)!=0u){
    result|=${SPARSE_CM12_DIRTY_CAUSE_BIT.topologyCreated
      | SPARSE_CM12_DIRTY_CAUSE_BIT.coefficientChanged}u;}
  if((cause&${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.injection}u)!=0u){
    result|=${SPARSE_CM12_DIRTY_CAUSE_BIT.boundarySource}u;}
  if((cause&${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.movingSolid}u)!=0u){
    result|=${SPARSE_CM12_DIRTY_CAUSE_BIT.movingSolidSweep}u;}
  return result;
}
// Late-frame VEX planning owns the exact blast list. Import that list only
// after FPL Next is open: depth zero is direct root evidence, positive depth
// is dependency closure with its exact graph depth. Atomic FPL masks dedupe
// multiple cells that share one stable B16 tile.
@compute @workgroup_size(64)
fn importVelocityExtensionBlastToFramePlanNext(
 @builtin(global_invocation_id)gid:vec3u){
  let invocation=gid.x;
  let count=cm12ExtensionLoad(
    ${vexActivityBatchLayout.velocityExtension.headerBaseWords
      + SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastCount}u);
  if(invocation>=count){return;}
  let cell=cm12ExtensionLoad(cm12ExtensionBlastList+invocation);
  // The sealed blast is the exact recurrence authority. Seed-time pruning is
  // its activity boundary; independently filtering here could drop FPL
  // coverage after a same-frame retirement while the immutable blast still
  // reaches next frame's consumer.
  if(cell>=p.counts.x){cm12ExtensionFail(cell,0u);return;}
  let depth=cm12ExtensionLoad(cm12ExtensionBlastDepth+cell);
  let owner=cm12ResidentCellTile(cell);
  if(depth==0u){
    let cause=cm12ResidentVelocityExtensionDirtyCause(
      cm12ExtensionLoad(cm12ExtensionRootCause+cell));
    cm12FramePlanMarkOwnedNextTile(owner.x,owner.y,1u,0u,cause,0u,0u);
  }else{
    cm12FramePlanMarkOwnedNextTile(owner.x,owner.y,0u,1u,0u,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.dependencyClosure}u,depth);
  }
}

fn cm12ActivityCandidateGeneration()->u32{return incrementalActivityGeneration();}
fn cm12ActivityCandidateBrickCount()->u32{return incrementalActivityBrickCount();}
fn cm12ActivityCandidateListGeneration()->u32{return incrementalActivityListGeneration();}
fn cm12ActivityCandidateBrickInvocation(candidate:u32)->u32{
  return incrementalActivityBrickInvocation(candidate);
}
fn cm12ActivityTopologyGeneration()->u32{
  return atomicLoad(&topologyArena[topologyWorklistBase()]);
}
fn cm12ActivityFramePlanGeneration()->u32{return cm12FramePlanCandidateGeneration();}
fn cm12ActivityBrickTopologySignature(brick:u32)->u32{
  return incrementalActivityTopologyState(brick);
}
fn cm12ActivityBrickTopologyChanged(brick:u32)->bool{
  let mask=incrementalActivityBrickTileMask(brick);return (mask.x|mask.y)!=0u;
}
fn cm12ActivityBuildTileTrigger(brick:u32,tile:u32)->vec4u{
  let generation=cm12ActivityCandidateGeneration();
  let mask=incrementalActivityBrickTileMask(brick);
  let dirty=select(0u,1u,(mask.x|mask.y)!=0u);
  let certificates=0x807f0000u;
  return vec4u(generation,${SPARSE_CM12_DIRTY_CAUSE_BIT.dependencyClosure}u,
    certificates|select(0u,0x03ffu,dirty!=0u),cm12ActivityBrickTopologySignature(brick));
}
fn cm12ActivityExpectedTileContributionCount(brick:u32,tile:u32)->u32{
  if(!brickActive(brick)){return 0u;}
  let resolution=acceptedBrickResolution(brick);let scale=BRICK_FINE_RESOLUTION/resolution;
  let tx=tile%4u;let yz=tile/4u;let ty=yz%4u;let tz=yz/4u;
  let lo=vec3u(tx,ty,tz)*4u;let hi=lo+vec3u(4u);
  let begin=(lo+vec3u(scale-1u))/scale;let end=(hi+vec3u(scale-1u))/scale;
  return (end.x-begin.x)*(end.y-begin.y)*(end.z-begin.z);
}
fn cm12ActivityExpectedTileCheck(brick:u32,tile:u32)->u32{
  return cm12ActivityBrickTopologySignature(brick)^(tile*0x9e3779b9u)
    ^cm12ActivityExpectedTileContributionCount(brick,tile);
}
fn cm12ActivityPublishFramePlanRoot(brick:u32,tile:u32,directStages:u32,
 closureStages:u32,cause:u32,depth:u32,generation:u32)->bool{
  _=brick;_=tile;_=directStages;_=closureStages;_=cause;_=depth;_=generation;
  return true;
}
fn cm12ActivityRebuildExactTile(brick:u32,tile:u32)->A4D2TileSummary{
  if(!brickActive(brick)){return A4D2TileSummary(vec4i(0),vec4f(0.0),
    0u,0u,0u,0u,cm12ActivityExpectedTileCheck(brick,tile));}
  let resolution=acceptedBrickResolution(brick);let range=templateBrickCellRange(brick,resolution);
  let first=range.x;let scale=BRICK_FINE_RESOLUTION/resolution;
  let tx=tile%4u;let yz=tile/4u;let ty=yz%4u;let tz=yz/4u;
  let lo=(vec3u(tx,ty,tz)*4u+vec3u(scale-1u))/scale;
  let hi=(vec3u(tx,ty,tz)*4u+vec3u(4u)+vec3u(scale-1u))/scale;
  var sums=vec4i(0);var metrics=vec4f(0.0);var surfaceAxes=0u;
  var occupiedCell=false;var thinFluidCell=false;var cutBoundaryCell=false;
  var supportMask=0u;var sweptSupportMask=0u;var contributionCount=0u;
  for(var z=lo.z;z<hi.z;z+=1u){for(var y=lo.y;y<hi.y;y+=1u){
    for(var x=lo.x;x<hi.x;x+=1u){
      let cell=first+x+resolution*(y+resolution*z);contributionCount+=1u;
      let rho=state[destinationDensity()+cell];let fill=rho/max(cellOpenFraction(cell),1e-6);
      cutBoundaryCell=cutBoundaryCell||cellOpenFraction(cell)<0.999;
      let ownFixed=i32(round(rho*ACTIVITY_FIXED));sums.x+=ownFixed;
      sums.y+=i32(round(rho*(f32(2u*x+1u)-f32(resolution))
        /f32(resolution)*ACTIVITY_FIXED));
      sums.z+=i32(round(rho*(f32(2u*y+1u)-f32(resolution))
        /f32(resolution)*ACTIVITY_FIXED));
      sums.w+=i32(round(rho*(f32(2u*z+1u)-f32(resolution))
        /f32(resolution)*ACTIVITY_FIXED));
      let fractionalCell=fill>p.activityDensity.y&&fill<p.activityDensity.z;
      var interfaceCell=false;let residencyDensity=residencyDensityThreshold();
      occupiedCell=occupiedCell||rho>residencyDensity;
      let ownVelocityAt=destinationCellVelocity()+4u*cell;
      let ownVelocity=vec3f(state[ownVelocityAt],state[ownVelocityAt+1u],
        state[ownVelocityAt+2u]);
      if(rho>residencyDensity){metrics.w=max(metrics.w,p.frame.x*length(ownVelocity));}
      let ownWet=fill>=CM12_LIQUID_ISOVALUE;
      let featureDensity=max(residencyDensity,p.activityDensity.x);
      let b=cellBase(cell);let cellCenter=vec3f(taf(b),taf(b+1u),taf(b+2u));
      var exposedSides=0u;var surfaceAirSides=0u;
      for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
        let row=incidenceRow(incidence);if(!rowAccepted(row)){continue;}
        let own=termCoefficient(incidenceTerm(incidence));if(rowArea(row)<=1e-8){continue;}
        let axis=rowAxis(row);let rowPosition=rowCenter(row);
        let omittedAirInside=rowKind(row)==3u&&rowPosition[axis]>1e-4
          &&rowPosition[axis]<f32(p.dimensions[axis])-1e-4;
        var crosses=omittedAirInside&&ownWet;var sideHasFluid=false;
        var sideHasSurfaceFluid=false;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
        for(var term=begin;term<end;term+=1u){let coefficient=termCoefficient(term);
          if(own*coefficient>=0.0){continue;}let neighbor=termCell(term);
          if(cellOpenVolume(neighbor)<=1e-8){continue;}
          let neighborDensity=state[destinationDensity()+neighbor]
            /max(cellOpenFraction(neighbor),1e-6);
          sideHasFluid=sideHasFluid||neighborDensity>featureDensity;
          sideHasSurfaceFluid=sideHasSurfaceFluid||neighborDensity>p.activityDensity.y;
          crosses=crosses||((neighborDensity>=CM12_LIQUID_ISOVALUE)!=ownWet
            &&min(fill,neighborDensity)<=p.activityDensity.y);
          let neighborVelocityAt=destinationCellVelocity()+4u*neighbor;
          let neighborVelocity=vec3f(state[neighborVelocityAt],state[neighborVelocityAt+1u],
            state[neighborVelocityAt+2u]);
          metrics.x=max(metrics.x,p.frame.x*max(abs(ownVelocity.x-neighborVelocity.x),
            max(abs(ownVelocity.y-neighborVelocity.y),abs(ownVelocity.z-neighborVelocity.z)))
            /max(0.15*rowDistance(row),1e-12));
        }
        if(rho>featureDensity&&!sideHasFluid){let side=select(0u,1u,rowPosition[axis]>cellCenter[axis]);
          exposedSides|=1u<<(2u*axis+side);}
        if(fractionalCell&&!sideHasSurfaceFluid){
          let side=select(0u,1u,rowPosition[axis]>cellCenter[axis]);
          surfaceAirSides|=1u<<(2u*axis+side);}
        if(crosses){interfaceCell=true;surfaceAxes|=1u<<axis;
          metrics.y=max(metrics.y,p.frame.x*abs(state[destinationFaceVelocity()+row])
            /max(0.25*rowDistance(row),1e-12));}
      }
      let representedThickness=clamp(rho,0.0,1.0)*cellMinimumWidth(cell);
      var cellIsThinFluid=false;
      for(var axis=0u;axis<3u;axis+=1u){let oppositeSides=3u<<(2u*axis);
        if(fractionalCell&&(surfaceAirSides&oppositeSides)!=0u){
          interfaceCell=true;surfaceAxes|=1u<<axis;}
        cellIsThinFluid=cellIsThinFluid||(fill>featureDensity
          &&representedThickness<p.activityThresholds.w
          &&(exposedSides&oppositeSides)==oppositeSides);
      }
      thinFluidCell=thinFluidCell||cellIsThinFluid;
      if(interfaceCell||cellIsThinFluid){var minimumOffset=vec3i(0);var maximumOffset=vec3i(0);
        if(x==0u){minimumOffset.x=-1;}if(x+1u==resolution){maximumOffset.x=1;}
        if(y==0u){minimumOffset.y=-1;}if(y+1u==resolution){maximumOffset.y=1;}
        if(z==0u){minimumOffset.z=-1;}if(z+1u==resolution){maximumOffset.z=1;}
        for(var dz=minimumOffset.z;dz<=maximumOffset.z;dz+=1){
          for(var dy=minimumOffset.y;dy<=maximumOffset.y;dy+=1){
            for(var dx=minimumOffset.x;dx<=maximumOffset.x;dx+=1){
              if(dx!=0||dy!=0||dz!=0){let bit=u32(dx+1)+3u*u32(dy+1)+9u*u32(dz+1);
                supportMask|=1u<<bit;}
        }}}
        let brickDimensions=vec3i((p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
          /BRICK_FINE_RESOLUTION);let sourceBrick=vec3i(cellMinimum(cell)/BRICK_FINE_RESOLUTION);
        let sweptBrick=clamp(vec3i(floor((cellCenter+p.activityTiming.x*p.frame.x*ownVelocity)
          /f32(BRICK_FINE_RESOLUTION))),vec3i(0),brickDimensions-vec3i(1));
        let sweptOffset=clamp(sweptBrick-sourceBrick,vec3i(-1),vec3i(1));
        if(any(sweptOffset!=vec3i(0))){let bit=u32(sweptOffset.x+1)+3u*u32(sweptOffset.y+1)
          +9u*u32(sweptOffset.z+1);supportMask|=1u<<bit;sweptSupportMask|=1u<<bit;}
      }
      if(resolution>1u){let group=2u*(vec3u(x,y,z)/2u);var childSum=0;
        for(var dz=0u;dz<2u;dz+=1u){for(var dy=0u;dy<2u;dy+=1u){
          for(var dx=0u;dx<2u;dx+=1u){let q=group+vec3u(dx,dy,dz);
            let child=first+q.x+resolution*(q.y+resolution*q.z);
            if(child<first+range.y){childSum+=i32(round(
              state[destinationDensity()+child]*ACTIVITY_FIXED));}
        }}}
        metrics.z=max(metrics.z,f32(abs(8*ownFixed-childSum))/(8.0*ACTIVITY_FIXED));
      }
  }}}
  let flags=surfaceAxes|select(0u,16u,occupiedCell)|select(0u,32u,thinFluidCell)
    |select(0u,64u,cutBoundaryCell);
  return A4D2TileSummary(sums,metrics,flags,supportMask,sweptSupportMask,
    contributionCount,cm12ActivityExpectedTileCheck(brick,tile));
}
fn cm12ActivityPublishExactBrick(brick:u32,moments:vec4i,metrics:vec4f,
 masks:vec4u,prior:vec4u)->vec4u{
  _=prior;let output=activityRecord(brick);let step=atomicLoad(&activity[0]);
  incrementalActivityRemoveCensus(brick);
  if(!brickActive(brick)){
    atomicStore(&activity[output],0u);atomicStore(&activity[output+1u],0u);
    atomicStore(&activity[output+32u],0u);atomicStore(&activity[output+3u],0u);
    incrementalActivityAcceptMeasuredTopology(brick);
    return vec4u(0u,0u,0u,0u);
  }
  let count=max(1u,masks.w);let first=templateBrickCellRange(
    brick,acceptedBrickResolution(brick)).x;
  let meanDensity=f32(moments.x)/(f32(count)*ACTIVITY_FIXED);
  let normalizedMoments=vec3f(moments.yzw)/(f32(count)*ACTIVITY_FIXED);
  var temporal=0.0;if(step>1u){temporal=max(abs(meanDensity-activityF32(output+4u))/0.05,
    max(abs(normalizedMoments.x-activityF32(output+5u))/0.02,
    max(abs(normalizedMoments.y-activityF32(output+6u))/0.02,
      abs(normalizedMoments.z-activityF32(output+7u))/0.02)));}
  let densityMassFineCells=f32(moments.x)/ACTIVITY_FIXED*cellVolume(first);
  let densityPresent=(masks.x&16u)!=0u;
  let occupied=densityPresent&&densityMassFineCells>=p.sharpening.w;
  let axes=select(0u,masks.x&7u,occupied);let surface=occupied&&axes!=0u;
  let thinFluid=occupied&&(masks.x&32u)!=0u;
  let shape=select(0.0,1.0,countOneBits(axes)>=2u);
  let velocityActivity=metrics.w;
  let scoredDetailError=select(metrics.z,0.0,surface&&!thinFluid&&shape==0.0);
  let dynamicActivity=select(0.0,max(metrics.x,temporal),surface||thinFluid);
  let detailActivity=max(0.0,scoredDetailError/p.activityDensity.w-1.0);
  let featureActivity=max(max(dynamicActivity,metrics.y),
    max(max(0.0,max(shape,select(0.0,1.0,thinFluid))),detailActivity));
  let normalizedVelocityActivity=velocityActivity/max(p.activityThresholds.x,1e-6);
  let scoredVelocityActivity=select(0.0,normalizedVelocityActivity,surface||thinFluid);
  let scoreValue=clamp(max(scoredVelocityActivity,featureActivity),0.0,1.0);
  let score=u32(round(255.0*scoreValue));var reasons=0u;
  if(surface){reasons|=1u;}if(metrics.x>0.0){reasons|=2u;}if(temporal>0.0){reasons|=4u;}
  if(metrics.z>p.activityDensity.w){reasons|=8u;}if(metrics.y>0.0){reasons|=16u;}
  if(step==1u){reasons|=32u;}if(occupied){reasons|=64u;}
  if(velocityResolutionFloor(velocityActivity)>1u){reasons|=128u;}
  if(thinFluid){reasons|=256u;}if((masks.x&64u)!=0u){reasons|=512u;}
  let topologyEpoch=atomicLoad(&activity[5])!=0u;
  let oldHistory=atomicLoad(&activity[output+2u]);
  var hotEpochs=oldHistory&255u;var quietEpochs=(oldHistory>>8u)&255u;
  if(topologyEpoch){let activitySignals=activitySignalsEnabled();
    let featureHot=activitySignals&&featureActivity>=p.activityTiming.y;
    hotEpochs=select(0u,min(255u,hotEpochs+1u),featureHot);
    let current=atomicLoad(&activity[output+12u]);
    let velocityFloor=select(1u,velocityResolutionFloor(velocityActivity),surface||thinFluid);
    let activityQuiet=!featureHot&&scoreValue<=p.activityTiming.w
      &&scoredDetailError<=p.activityDensity.w;
    let quiet=!thinFluid&&velocityFloor<current&&select(!surface,activityQuiet,activitySignals);
    quietEpochs=select(0u,min(255u,quietEpochs+1u),quiet);
  }
  let history=hotEpochs|(quietEpochs<<8u);
  atomicStore(&activity[output],score);atomicStore(&activity[output+1u],reasons);
  atomicStore(&activity[output+2u],history);
  atomicStore(&activity[output+4u],bitcast<u32>(meanDensity));
  atomicStore(&activity[output+5u],bitcast<u32>(normalizedMoments.x));
  atomicStore(&activity[output+6u],bitcast<u32>(normalizedMoments.y));
  atomicStore(&activity[output+7u],bitcast<u32>(normalizedMoments.z));
  atomicStore(&activity[output+32u],select(0u,masks.y,occupied));
  atomicStore(&activity[output+3u],select(0u,masks.z,occupied));
  atomicStore(&activity[output+33u],bitcast<u32>(velocityActivity));
  atomicStore(&activity[output+34u],0u);
  incrementalActivityAddCensus(brick,score,reasons);
  incrementalActivityAcceptMeasuredTopology(brick);
  let a4d2Flags=1u|select(0u,2u,surface)
    |select(0u,4u,featureActivity>=p.activityTiming.y)
    |select(0u,8u,scoreValue<=p.activityTiming.w);
  return vec4u(score,reasons,history,a4d2Flags);
}
` : /* wgsl */ `
fn cm12ExtensionTransportVelocity(cell:u32)->vec4f{
  let at=destinationCellVelocity()+4u*cell;
  return vec4f(state[at],state[at+1u],state[at+2u],state[at+3u]);
}
fn cm12ResidentRecordExtensionIncidence(cell:u32,cause:u32){_=cell;_=cause;}
`}

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
  var surfaceAxes=0u;var occupiedCell=false;var thinFluidCell=false;
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
        // A small submerged oscillation around rho=.5 is not an interface.
        // Require the crossing to reach the configured air band; diffuse
        // interfaces that do not cross .5 are caught by surfaceAirSides.
        crosses=crosses||(crossesIsovalue
          &&min(fill,neighborDensity)<=p.activityDensity.y);
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
  let activityFlags=surfaceAxes|select(0u,16u,occupiedCell)
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
  // A few concentrated interpolation remnants can exceed the per-cell density
  // floor while carrying less than one finest cell of liquid in the whole
  // brick. They are residue, not a surface or neighbor-support authority.
  let occupied=densityPresent&&densityMassFineCells>=p.sharpening.w;
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
  if(!brickCandidatePlanningEnabled(brick)){
    let accepted=atomicLoad(&activity[output+12u]);
    atomicStore(&activity[output+13u],accepted);atomicStore(&activity[output+14u],0u);
    return;
  }
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+8u]);
  // Receiver activation uses a bounded weak-CAS. If its unique writer still
  // observes 64 spurious failures, preserve that local rejection through this
  // validation pass instead of silently turning the requested receiver into
  // ordinary topology work.
  let activationFault=(atomicLoad(&activity[7])&32u)!=0u
    &&brickRequestedAsReceiver(brick)&&!brickActive(brick);
  var invalid=!validBrickResolution(accepted)||!validBrickResolution(candidate)
    ||activationFault;
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
  let transition=candidate!=accepted;
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
    atomicStore(&activity[output+35u],0u);
    if(atomicLoad(&activity[output+14u])!=1u){continue;}
    let accepted=atomicLoad(&activity[output+12u]);
    let candidate=atomicLoad(&activity[output+13u]);
    if(candidate>accepted){atomicStore(&activity[output+35u],1u);
      atomicStore(&activity[output+36u],generation);urgent+=1u;
      _=ptrRecordChangedBrick(brick,pressureBrickStateAtResolution(brick,accepted),
        pressureBrickStateAtResolution(brick,candidate),
        ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.resolutionChanged}u,true);
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
      atomicStore(&activity[output+35u],1u);
      atomicStore(&activity[output+36u],generation);
      let accepted=atomicLoad(&activity[output+12u]);
      let candidate=atomicLoad(&activity[output+13u]);
      _=ptrRecordChangedBrick(brick,pressureBrickStateAtResolution(brick,accepted),
        pressureBrickStateAtResolution(brick,candidate),
        ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.resolutionChanged}u,true);
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
  if(atomicLoad(&activity[output+35u])==0u
    ||atomicLoad(&activity[output+37u])!=INVALID){return;}
  let page=acquireTopologyPage();
  if(page==INVALID){
    atomicStore(&activity[output+14u],2u);atomicStore(&activity[output+35u],0u);
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
  if(page==INVALID||atomicLoad(&activity[output+35u])==0u){return;}
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

// Dynamic pages are not template slots. Keep the old transfer/publication
// transaction asleep until this page also carries rows and incidence; that
// transaction indexes candidateState by packed template slot. Planning and
// synthesis remain live, so large scenes perform useful GPU topology work
// without exposing a partial generation to accepted kernels.
@compute @workgroup_size(1)
fn deferDynamicTopologyPublication(){
  let base=topologyWorklistBase();
  if(atomicLoad(&topologyArena[base+27u])==0u){return;}
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){
    if(brickPackedCandidateSlot(brick)==INVALID){
      atomicStore(&activity[activityRecord(brick)+35u],0u);
    }
  }
  atomicStore(&activity[16],0u);
}

@compute @workgroup_size(1)
fn beginShadowTopology(){
  let base=topologyWorklistBase();
  atomicStore(&topologyArena[base+3u],1u);
  atomicStore(&topologyArena[base+18u],0u);
  atomicStore(&topologyArena[base+19u],0u);
  atomicStore(&topologyArena[base+1u],atomicLoad(&topologyArena[base])+1u);
}

@compute @workgroup_size(64)
fn buildShadowCellWorklist(@builtin(global_invocation_id)gid:vec3u){
  // No accepted field can observe the inactive list. Avoid walking the full
  // dyadic template library on the overwhelmingly common no-change frame;
  // a later prepared transaction clears and rebuilds the list before publish.
  if(atomicLoad(&activity[16])==0u){return;}
  let cell=gid.x;if(cell>=ta(2u)){return;}
  let brick=cellBrick(cell);let resolution=cellResolution(cell);
  if(!brickActive(brick)||resolution!=scheduledBrickResolution(brick)){return;}
  let base=topologyWorklistBase();let index=atomicAdd(&topologyArena[base+18u],1u);
  let offset=atomicLoad(&topologyArena[base+14u+shadowTopologySlot()]);
  atomicStore(&topologyArena[base+offset+index],cell);
}

@compute @workgroup_size(64)
fn buildShadowRowWorklist(@builtin(global_invocation_id)gid:vec3u){
  if(atomicLoad(&activity[16])==0u){return;}
  let row=gid.x;if(row>=ta(3u)){return;}
  let requirements=rowRequirementOffset(row);let count=ta(requirements);var enabled=true;
  for(var at=0u;at<count;at+=1u){let metadata=ta(requirements+1u+at);
    let brick=metadata>>TEMPLATE_CELL_RESOLUTION_BITS;
    let resolution=metadata&TEMPLATE_CELL_RESOLUTION_MASK;
    enabled=enabled&&brickActive(brick)&&scheduledBrickResolution(brick)==resolution;
  }
  if(!enabled){return;}let base=topologyWorklistBase();
  let index=atomicAdd(&topologyArena[base+19u],1u);
  let offset=atomicLoad(&topologyArena[base+16u+shadowTopologySlot()]);
  atomicStore(&topologyArena[base+offset+index],row);
}

@compute @workgroup_size(1)
fn finalizeShadowWorklists(){
  let base=topologyWorklistBase();let cells=atomicLoad(&topologyArena[base+18u]);
  let rows=atomicLoad(&topologyArena[base+19u]);
  atomicStore(&topologyArena[base+20u],(cells+63u)/64u);
  atomicStore(&topologyArena[base+21u],1u);atomicStore(&topologyArena[base+22u],1u);
  atomicStore(&topologyArena[base+23u],(rows+63u)/64u);
  atomicStore(&topologyArena[base+24u],1u);atomicStore(&topologyArena[base+25u],1u);
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
@compute @workgroup_size(64)
fn transferCandidateCells(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  if(lane==0u){candidateCellScheduled=atomicLoad(&activity[output+35u]);}
  let scheduled=workgroupUniformLoad(&candidateCellScheduled);
  if(scheduled==0u){return;}
  let candidateStatus=atomicLoad(&activity[output+14u]);
  let accepted=atomicLoad(&activity[output+12u]);
  let candidate=atomicLoad(&activity[output+13u]);
  let acceptedRange=templateBrickCellRange(brick,accepted);let first=acceptedRange.x;
  let candidateRange=templateBrickCellRange(brick,candidate);
  let sourceCount=acceptedRange.y;let candidateCount=candidateRange.y;
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
    if(candidate<accepted){let factor=accepted/candidate;
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
  let massError=transferMassAfter[0]-transferMassBefore[0];
  let gammaError=transferGammaDelta[0];let momentumError=vec3f(
    transferMomentumXDelta[0],transferMomentumYDelta[0],transferMomentumZDelta[0]);
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
fn prepareCandidateFaceReceipts(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}let output=activityRecord(brick);
  for(var side=0u;side<6u;side+=1u){atomicStore(&activity[output+24u+side],0u);}
  atomicStore(&activity[output+30u],0u);
  let candidateStatus=atomicLoad(&activity[output+14u]);
  let scheduled=atomicLoad(&activity[output+35u])!=0u;
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
@compute @workgroup_size(64)
fn transferCandidateFaces(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let side=wid.y;let lane=lid.x;
  if(brick>=p.dispatch.w||side>=6u){return;}
  let output=activityRecord(brick);
  // The old path area-averaged every face of every brick even when the
  // scheduler had prepared nothing. Candidate storage is isolated and a
  // scheduled brick recomputes all six receipts before it can commit, so this
  // uniform workgroup return removes only provably dead writes.
  if(lane==0u){candidateFaceScheduled=atomicLoad(&activity[output+35u]);}
  let scheduled=workgroupUniformLoad(&candidateFaceScheduled);
  if(scheduled==0u){return;}
  let candidate=atomicLoad(&activity[output+13u]);
  let accepted=atomicLoad(&activity[output+12u]);let acceptedRange=templateBrickCellRange(brick,accepted);
  let record=p.topologyOffsets2.z+2u*brick;let first=acceptedRange.x;
  let key=topology[record+1u];
  let brickDimensions=(p.dimensions.xyz+vec3u(BRICK_FINE_RESOLUTION-1u))
    /BRICK_FINE_RESOLUTION;
  let xy=brickDimensions.x*brickDimensions.y;let bz=key/xy;
  let remainder=key-bz*xy;let by=remainder/brickDimensions.x;
  let bx=remainder-by*brickDimensions.x;
  let origin=vec3f(vec3u(bx,by,bz)*BRICK_FINE_RESOLUTION);
  let axis=side/2u;let positive=(side&1u)!=0u;
  let tangent0=select(select(0u,0u,axis==2u),1u,axis==0u);
  let tangent1=select(2u,1u,axis==2u);
  let brickWidth=f32(BRICK_FINE_RESOLUTION);
  let plane=origin[axis]+select(0.0,brickWidth,positive);
  let scale=brickWidth/f32(candidate);
  var flux=0.0;var reconstructedFlux=0.0;
  for(var patchIndex=lane;patchIndex<candidate*candidate;patchIndex+=64u){
    var patchFlux=0.0;var patchArea=0.0;
    // Only accepted cells on this exterior patch can contribute. The old
    // receipt walked all accepted^3 cells for every candidate patch, although
    // its later plane/patch tests rejected all but accepted^2 cells across the
    // entire workgroup. Dyadic candidate levels let each lane address its
    // exact source footprint directly while retaining the same row acceptance,
    // ownership, geometric patch, and conservation gates below.
    let patchU=patchIndex%candidate;let patchV=patchIndex/candidate;
    let sourceSpan=max(1u,accepted/candidate);
    let sourceU=(patchU*accepted)/candidate;
    let sourceV=(patchV*accepted)/candidate;
    for(var dv=0u;dv<sourceSpan;dv+=1u){for(var du=0u;du<sourceSpan;du+=1u){
      let local=boundaryCellLocal(axis,positive,accepted,sourceU+du,sourceV+dv);
      let cell=first+local;
      for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
        let row=incidenceRow(incidence);
        if(!rowAccepted(row)||rowAxis(row)!=axis){continue;}
        let center=rowCenter(row);if(abs(center[axis]-plane)>1e-4){continue;}
        var claimant=INVALID;let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
        for(var term=begin;term<end;term+=1u){let termOwner=termCell(term);
          if(cellBrick(termOwner)==brick){claimant=min(claimant,termOwner);}}
        if(cell!=claimant){continue;}
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
fn writeCandidateCellsToShadow(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;let lane=lid.x;if(brick>=p.dispatch.w){return;}
  let output=activityRecord(brick);
  if(atomicLoad(&activity[output+35u])==0u
    ||atomicLoad(&activity[output+23u])!=1u
    ||atomicLoad(&activity[output+31u])!=1u){return;}
  let candidate=atomicLoad(&activity[output+13u]);
  let range=templateBrickCellRange(brick,candidate);
  for(var local=lane;local<range.y;local+=64u){let cell=range.x+local;
    let rho=candidateState[candidateFieldIndex(0u,brick,local)];
    let gamma=candidateState[candidateFieldIndex(1u,brick,local)];
    let vx=candidateState[candidateFieldIndex(2u,brick,local)];
    let vy=candidateState[candidateFieldIndex(3u,brick,local)];
    let vz=candidateState[candidateFieldIndex(4u,brick,local)];
    state[p.stateOffsets0.x+cell]=rho;state[p.stateOffsets0.y+cell]=rho;
    state[p.stateOffsets0.z+cell]=gamma;state[p.stateOffsets0.w+cell]=gamma;
    for(var slot=0u;slot<2u;slot+=1u){let velocity=select(
      p.stateOffsets1.x,p.stateOffsets1.y,slot==1u)+4u*cell;
      state[velocity]=vx;state[velocity+1u]=vy;state[velocity+2u]=vz;state[velocity+3u]=0.0;
    }
    state[p.stateOffsets2.x+cell]=candidateState[candidateFieldIndex(5u,brick,local)];
    state[p.stateOffsets2.y+cell]=0.0;state[p.stateOffsets2.z+cell]=0.0;
    state[p.stateOffsets2.w+cell]=select(0.0,1.0,
      rho/max(cellOpenFraction(cell),1e-6)>=CM12_LIQUID_ISOVALUE);
    state[p.stateOffsets3.y+cell]=0.0;state[p.stateOffsets3.z+cell]=0.0;
    state[p.stateOffsets3.w+cell]=0.0;state[p.stateOffsets4.x+cell]=0.0;
    state[p.stateOffsets4.y+cell]=0.0;
  }
}

// New internal and seam rows have no accepted face slot. Reconstruct only rows
// incident to a scheduled brick; unchanged accepted rows remain untouched.
@compute @workgroup_size(64)
fn reconstructShadowFaces(@builtin(global_invocation_id)gid:vec3u){
  if(atomicLoad(&activity[16])==0u){return;}
  let invocation=gid.x;if(invocation>=shadowTemplateRowCount()){return;}
  let row=shadowTemplateRowInvocation(invocation);if(row==INVALID){return;}
  let requirements=rowRequirementOffset(row);let requirementCount=ta(requirements);
  var changed=false;for(var at=0u;at<requirementCount;at+=1u){
    let brick=ta(requirements+1u+at)>>TEMPLATE_CELL_RESOLUTION_BITS;
    changed=changed||atomicLoad(&activity[activityRecord(brick)+35u])!=0u;
  }
  if(!changed){return;}let axis=rowAxis(row);var velocity=0.0;var weight=0.0;
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  for(var term=begin;term<end;term+=1u){let cell=termCell(term);
    let at=destinationCellVelocity()+4u*cell;let w=abs(termCoefficient(term));
    velocity+=w*state[at+axis];weight+=w;
  }
  velocity=select(0.0,velocity/weight,weight>0.0);
  state[p.stateOffsets1.z+row]=velocity;state[p.stateOffsets1.w+row]=velocity;
}

@compute @workgroup_size(1)
fn validateAndCommitShadowTopology(){
  let prepared=atomicLoad(&activity[16]);let base=topologyWorklistBase();
  if(prepared==0u){
    _=ptrSealTopologyJournal(atomicLoad(&topologyArena[ptrH_COVERED_PRODUCER_RECEIPTS]),
      atomicLoad(&topologyArena[base]));
    atomicStore(&activity[13],(atomicLoad(&activity[13])+p.topologyScheduling.x)
      %max(1u,p.dispatch.w));return;
  }
  var valid=shadowTemplateCellCount()<=atomicLoad(&topologyArena[base+6u])
    &&shadowTemplateRowCount()<=atomicLoad(&topologyArena[base+7u]);
  var fine=0u;var coarse=0u;var activeCells=0u;
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){let output=activityRecord(brick);
    if(atomicLoad(&activity[output+35u])!=0u){
      valid=valid&&atomicLoad(&activity[output+23u])==1u
        &&atomicLoad(&activity[output+31u])==1u;
    }
    let resolution=scheduledBrickResolution(brick);
    if(brickActive(brick)){activeCells+=templateBrickCellRange(brick,resolution).y;
      if(resolution==BRICK_FINE_RESOLUTION){fine+=1u;}else{coarse+=1u;}}
  }
  if(!valid){_=ptrRejectTopologyJournal();atomicStore(&activity[21],1u);
    atomicStore(&topologyArena[base+3u],3u);return;}
  for(var brick=0u;brick<p.dispatch.w;brick+=1u){let output=activityRecord(brick);
    if(atomicLoad(&activity[output+35u])!=0u){
      let accepted=atomicLoad(&activity[output+12u]);
      let next=atomicLoad(&activity[output+13u]);
      // The next-frame scalar candidate was already published. Poison only
      // this changed topology closure before publishing the replacement rung.
      scaInvalidateBrickTopologyClosure(brick);
      atomicStore(&activity[output+12u],next);
      // The replacement rung may cover a larger logical span. Stamp its full
      // page identity and receiver halo as well; generation dedupe unions this
      // with the old-span closure without growing the event journal twice.
      scaInvalidateBrickTopologyClosure(brick);
      let recoveryState=atomicLoad(&activity[output+38u]);
      let recoveryFloor=recoveryState&31u;
      let recoveryLocked=(recoveryState&ACTIVITY_RECOVERY_LOCK)!=0u;
      if(next>accepted){
        atomicStore(&activity[output+38u],recoveryFloor|ACTIVITY_RECOVERY_LOCK);
      }else if(!recoveryLocked&&next<recoveryFloor){
        atomicStore(&activity[output+38u],next);
      }
      atomicStore(&activity[output+2u],0u);
      // Keep the local scheduled bit through the post-commit VEX root pass;
      // the next planner clears it before authoring another transaction.
      atomicAdd(&activity[17],1u);
    }
  }
  let slot=shadowTopologySlot();
  atomicStore(&topologyArena[base+4u],shadowTemplateCellCount());
  atomicStore(&topologyArena[base+5u],shadowTemplateRowCount());
  for(var at=0u;at<3u;at+=1u){
    atomicStore(&topologyArena[base+8u+at],atomicLoad(&topologyArena[base+20u+at]));
    atomicStore(&topologyArena[base+11u+at],atomicLoad(&topologyArena[base+23u+at]));
  }
  atomicStore(&topologyArena[base],atomicLoad(&topologyArena[base+1u]));
  _=ptrSealTopologyJournal(atomicLoad(&topologyArena[ptrH_COVERED_PRODUCER_RECEIPTS]),
    atomicLoad(&topologyArena[base]));
  atomicStore(&activity[12],atomicLoad(&topologyArena[base]));
  atomicStore(&activity[19],fine);atomicStore(&activity[20],coarse);
  atomicStore(&activity[11],activeCells);atomicStore(&activity[13],
    (atomicLoad(&activity[13])+p.topologyScheduling.x)%max(1u,p.dispatch.w));
  atomicStore(&topologyArena[base+3u],0u);
  // Release-like publication point: every state/list write is ordered before
  // the next dispatch observes the new accepted slot.
  atomicStore(&topologyArena[base+2u],slot);
}

@compute @workgroup_size(64)
fn publishSparseCM12TopologyVelocityRoots(@builtin(local_invocation_id)lid:vec3u,
 @builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;if(brick>=p.dispatch.w){return;}let output=activityRecord(brick);
  if(atomicLoad(&activity[output+35u])==0u){return;}
  let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
  for(var local=lid.x;local<range.y;local+=64u){
    cm12ResidentRecordExtensionIncidence(range.x+local,8u);
  }
}

// Publish only the directional free-surface stencil and swept receivers from
// the immutable activity snapshot. Compare-exchange makes publication
// single-writer; all following frame dispatches observe the active bit.
@compute @workgroup_size(64)
fn activateSweptReceivers(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w||brickActive(brick)){return;}
  if(!brickRequestedAsReceiver(brick)){return;}
  let output=activityRecord(brick);
  // A swept receiver contains the moving free surface by construction. It is
  // requested fine before publication; closePlannedResolution then grows the
  // dyadic support skirt around it without weakening this hard floor.
  atomicStore(&activity[output+8u],BRICK_FINE_RESOLUTION);
  atomicStore(&activity[output+9u],1u);
  var claimed=atomicCompareExchangeWeak(&activity[output+10u],0u,1u);
  for(var attempt=1u;attempt<64u&&!claimed.exchanged
      &&claimed.old_value==0u;attempt+=1u){
    claimed=atomicCompareExchangeWeak(&activity[output+10u],0u,1u);
  }
  if(!claimed.exchanged&&claimed.old_value==0u){
    // One invocation owns each brick, so this is weak-CAS exhaustion rather
    // than ordinary contention. Poison only this topology/scalar closure and
    // let validateCandidateResolution retain status=invalid; the scheduler
    // then publishes no transfer work for the affected receiver.
    scaInvalidateBrickTopologyClosure(brick);
    atomicStore(&activity[output+14u],2u);
    atomicStore(&activity[output+35u],0u);
    atomicOr(&activity[7],32u);
    return;
  }
  if(claimed.exchanged){
    // Activation changes the physical page identity even when its accepted
    // rung is unchanged. Reject this local scalar closure for the next frame.
    scaInvalidateBrickTopologyClosure(brick);
    let resolution=acceptedBrickResolution(brick);
    _=ptrRecordChangedBrick(brick,resolution,0x80000000u|resolution,
      ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.topologyCreated}u,true);
    // A retired residue brick may later become a legitimate receiver. Never
    // resurrect its stale sub-threshold fields as new liquid.
    let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    let first=range.x;let count=range.y;
    for(var cell=first;cell<first+count;cell+=1u){
      state[p.stateOffsets0.x+cell]=0.0;state[p.stateOffsets0.y+cell]=0.0;
      state[p.stateOffsets0.z+cell]=1.0;state[p.stateOffsets0.w+cell]=1.0;
      for(var component=0u;component<4u;component+=1u){
        state[p.stateOffsets1.x+4u*cell+component]=0.0;
        state[p.stateOffsets1.y+4u*cell+component]=0.0;
      }
      state[p.stateOffsets2.x+cell]=0.0;state[p.stateOffsets2.y+cell]=0.0;
      state[p.stateOffsets2.z+cell]=0.0;state[p.stateOffsets2.w+cell]=0.0;
      state[p.stateOffsets4.y+cell]=0.0;
      // Construction deliberately leaves dormant receiver cache ownership
      // invalid. A same-rung activation is not a resolution transaction, so
      // output+35 and the later topology-root pass do not cover it. Publish
      // every newly accepted support cell directly into the open next-frame
      // VEX plan: relying on the old active boundary's eight-hop closure
      // leaves the ninth layer as an active cell with no accepted cache.
      _=cm12ExtensionRecordRoot(cell,
        ${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.topologyOrEdgeOwnership}u);
    }
    atomicStore(&activity[output+34u],0u);
    atomicStore(&activity[output+11u],atomicLoad(&activity[0]));
    atomicAdd(&activity[8],1u);atomicAdd(&activity[9],1u);
    atomicAdd(&activity[10],1u);atomicAdd(&activity[11],count);
  }
}

// Retire every sub-residency brick outside the directional interface stencil
// and swept receiver mask. The discarded mass is recorded per brick and is
// bounded by the residency threshold times one 8^3 brick volume.
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
  let retired=atomicCompareExchangeWeak(&activity[output+10u],1u,0u);
  if(retired.exchanged){
    // Retirement removes scatter/incidence receivers. Preserve the old page's
    // full logical span and receiver halo as explicit next-frame work.
    scaInvalidateBrickTopologyClosure(brick);
    let resolution=acceptedBrickResolution(brick);
    _=ptrRecordChangedBrick(brick,0x80000000u|resolution,resolution,
      ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.topologyRetired}u,true);
    let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
    let first=range.x;let count=range.y;var residueMass=0.0;
    for(var cell=first;cell<first+count;cell+=1u){
      residueMass+=max(0.0,state[destinationDensity()+cell])*cellVolume(cell);
      state[p.stateOffsets0.x+cell]=0.0;state[p.stateOffsets0.y+cell]=0.0;
      state[p.stateOffsets0.z+cell]=1.0;state[p.stateOffsets0.w+cell]=1.0;
      for(var component=0u;component<4u;component+=1u){
        state[p.stateOffsets1.x+4u*cell+component]=0.0;
        state[p.stateOffsets1.y+4u*cell+component]=0.0;
      }
      state[p.stateOffsets2.x+cell]=0.0;state[p.stateOffsets2.y+cell]=0.0;
      state[p.stateOffsets2.z+cell]=0.0;state[p.stateOffsets2.w+cell]=0.0;
      state[p.stateOffsets4.y+cell]=0.0;
      // Retirement revokes the accepted VEX identity locally. A later
      // same-rung receiver activation republishes every cell as an ordinary
      // root, so stale cache ownership can never survive deactivate/reactivate.
      _=cm12ExtensionInvalidateRetiredCell(cell,
        ${SPARSE_CM12_VELOCITY_EXTENSION_CAUSE.topologyOrEdgeOwnership}u);
    }
    atomicStore(&activity[output+34u],bitcast<u32>(residueMass));
    atomicSub(&activity[8],1u);atomicSub(&activity[11],count);
    atomicAdd(&activity[10],1u);
  }
}

// Retirement is the last topology producer in the frame. Candidate commit
// seals the journal earlier, so republish the exact final producer count after
// retirement has appended its local records. The next pressure epoch can then
// fail closed on a real coverage mismatch without rejecting legitimate late
// retirement work.
@compute @workgroup_size(1)
fn sealSparseCM12PressureTopologyJournal(){
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
  // Stages 1..4 have not yet cut over to FPL scheduling. Publish their actual
  // global blast radius as four distinct physical packets and label the lack
  // of accepted local authority explicitly; never paint them green/reused.
  // Stage 0 is VEX-owned and may only come from its construction bootstrap or
  // its dynamic root/closure publication.
  let previousTopology=atomicLoad(&activity[ACTIVITY_BRICK_TOPOLOGY+brick]);
  let previouslyActive=previousTopology!=INVALID&&(previousTopology&0x80000000u)!=0u;
  if(previouslyActive&&p.injectionCenter.w<=0.5){
    cm12FramePlanMarkOwnedNextTile(brick,lane,0x1eu,0u,
      ${SPARSE_CM12_DIRTY_CAUSE_BIT.generationMismatch}u,0u,0u);
  }
  var mask=incrementalActivityBrickStageTileMask(brick,
    PRESENTATION_FRAME_PLAN_STAGE_BIT,PRESENTATION_FRAME_PLAN_CAUSES);
  let page=cm12FppLoad(cm12FppBrickPages+brick);
  let pageNeedsActivation=page<arrayLength(&fineMetadata)/4u
    &&fineMetadata[4u*page+2u]!=1u;
  if(bootstrap||injected||pageNeedsActivation){
    mask=vec2u(ACTIVITY_ALL_TILE_LOW,ACTIVITY_ALL_TILE_HIGH);
  }
  let scheduled=(select(mask.x,mask.y,lane>=32u)&(1u<<(lane&31u)))!=0u;
  if(!scheduled){return;}
  let stable=key*ACTIVITY_TILES_PER_BRICK+lane;
  let activityCauses=atomicLoad(&activity[ACTIVITY_TILE_CAUSE+stable])
    &PRESENTATION_FRAME_PLAN_CAUSES;
  var origin=activityCauses&PRESENTATION_FRAME_PLAN_DIRECT_CAUSES;
  var inherited=activityCauses&${SPARSE_CM12_DIRTY_CAUSE_BIT.dependencyClosure}u;
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
  // Stage 0 is executed only by the VEX consumer hook. Marking it here would
  // fabricate coverage for a recurrence which has not run yet.
  for(var stage=1u;stage<PRESENTATION_FRAME_PLAN_STAGE;stage+=1u){
    cm12FramePlanMarkCurrentTileExecuted(brick,lane,stage);
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
var<workgroup>cm12PresentationTemporalReuse:u32;
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
    let temporalReady=(atomicLoad(&activity[27])&0x80000000u)!=0u;
    cm12PresentationBrick=brick;cm12PresentationPage=page;
    cm12PresentationPageOrigin=pageOrigin;cm12PresentationBrickOrigin=brickOrigin;
    cm12PresentationSampleScale=sampleScale;cm12PresentationResolution=resolution;
    cm12PresentationScale=scale;cm12PresentationCacheFirst=cacheFirst;
    cm12PresentationCacheDimensions=cacheDimensions;
    cm12PresentationCacheFits=select(0u,1u,cacheCount<=PRESENTATION_CACHE_CAPACITY);
    cm12PresentationWet=select(0u,1u,brick<p.dispatch.w
      &&(atomicLoad(&activity[activityRecord(brick)+1u])&64u)!=0u);
    cm12PresentationTemporalReuse=select(0u,1u,temporalReady
      &&atomicLoad(&activity[17])==0u&&brickActive(brick)&&scale==1u);
    cm12PresentationDensityOffset=select(p.stateOffsets0.x,p.stateOffsets0.y,
      cm12FramePlanAcceptedParity()!=0u);
  }
  workgroupBarrier();
  let cacheCount=cm12PresentationCacheDimensions.x*cm12PresentationCacheDimensions.y
    *cm12PresentationCacheDimensions.z;
  if(cm12PresentationScale>1u&&cm12PresentationWet!=0u
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
  var phi=4.0*p.frame.y;
  if(cm12PresentationWet!=0u&&all(q<p.dimensions.xyz)
    &&insideEmbeddedBoundary(vec3f(q)+vec3f(0.5))){
    if(cm12PresentationScale==1u){
      let range=templateBrickCellRange(cm12PresentationBrick,cm12PresentationResolution);
      let valid=min(p.dimensions.xyz-cm12PresentationBrickOrigin,vec3u(BRICK_FINE_RESOLUTION));
      let localCell=q-cm12PresentationBrickOrigin;
      let cell=range.x+localCell.x+valid.x*(localCell.y+valid.y*localCell.z);
      if(cell<range.x+range.y){
        if(cm12PresentationTemporalReuse!=0u
          &&atomicLoad(&activity[TEMPORAL_CELL_FLAG_A+cell])==0u){
          return vec2u(fineSamples[page*PRESENTATION_SAMPLES_PER_PAGE+localIndex],0u);
        }
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

@compute @workgroup_size(64)
fn classifyPresentationBricks(@builtin(global_invocation_id)gid:vec3u){
  let brick=gid.x;if(brick>=p.dispatch.w){return;}
  let reasons=activityRecord(brick)+1u;
  if(!brickActive(brick)){atomicAnd(&activity[reasons],0xffffffbfu);return;}
  let range=templateBrickCellRange(brick,acceptedBrickResolution(brick));
  let first=range.x;let count=range.y;
  var wet=false;var massFineCells=0.0;
  for(var at=first;at<first+count;at+=1u){
    let rho=state[destinationDensity()+at];
    wet=wet||rho>residencyDensityThreshold();
    massFineCells+=max(0.0,rho)*cellVolume(at);
  }
  wet=wet&&massFineCells>=p.sharpening.w;
  if(wet){atomicOr(&activity[reasons],64u);}
  else{atomicAnd(&activity[reasons],0xffffffbfu);}
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
${presentationTemporalReuse}
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
          if(reusePresentationSamples
            &&atomicLoad(&activity[TEMPORAL_CELL_FLAG_A+cell])==0u){continue;}
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

/** Legacy/default specialization retained for direct shader-source consumers. */
export const webgpuSparseCM12ResidentWGSL = createWebgpuSparseCM12ResidentWGSL(16, 16);
