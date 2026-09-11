import {
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION,
  type SparseCM12PressureTopologyRepairLayout,
} from "./sparse-cm12-pressure-topology-repair";

export interface SparseCM12PressureTopologyRepairWGSLOptions {
  readonly layout: SparseCM12PressureTopologyRepairLayout;
  readonly arenaName?: string;
  readonly prefix?: string;
  readonly workgroupSize?: 64;
}

const upper = (value: string) => value.replace(/([a-z])([A-Z])/g, "$1_$2").toUpperCase();

/** Binding-free PTR1 source. The enclosing resident declares the atomic arena and hooks. */
export function createSparseCM12PressureTopologyRepairWGSL(
  options: SparseCM12PressureTopologyRepairWGSLOptions,
): string {
  const { layout } = options;
  const arena = options.arenaName ?? "pressureTopologyRepair";
  const p = options.prefix ?? "ptr";
  const workgroupSize = options.workgroupSize ?? 64;
  if (workgroupSize !== 64) throw new Error("PTR1 resident ABI requires 64 lanes");
  const h = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER;
  const constants = Object.entries(h).map(([name, value]) =>
    `const ${p}H_${upper(name)}=${layout.baseWords + value}u;`).join("\n");

  return /* wgsl */ `
const ${p}Magic=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC}u;
const ${p}Version=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION}u;
const ${p}HeaderWords=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS}u;
const ${p}Invalid=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID}u;
const ${p}BrickFamily=0u;
const ${p}BrickCapacity=${layout.brickCapacity}u;
const ${p}BrickLeafCount=${layout.brick.leafCount}u;
const ${p}BrickCandidate=${layout.brick.candidateGenerationBaseWords}u;
const ${p}BrickChangedList=${layout.brick.changedBrickListBaseWords}u;
const ${p}BrickDirtyStamp=${layout.brick.dirtyLeafStampBaseWords}u;
const ${p}BrickOldState=${layout.brickOldStateBaseWords}u;
const ${p}BrickNewState=${layout.brickNewStateBaseWords}u;
const ${p}PhaseUninitialized=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.uninitialized}u;
const ${p}PhaseAccepted=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.accepted}u;
const ${p}PhaseCollecting=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.collecting}u;
const ${p}PhaseExecutingCells=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.executingCells}u;
const ${p}PhaseFault=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.fault}u;
${constants}
fn ${p}HeaderValid()->bool{return atomicLoad(&${arena}[${p}H_MAGIC])==${p}Magic
 &&atomicLoad(&${arena}[${p}H_VERSION])==${p}Version
 &&atomicLoad(&${arena}[${p}H_HEADER_WORDS])==${p}HeaderWords
 &&atomicLoad(&${arena}[${p}H_TOTAL_WORDS])==${layout.totalWords}u;}
fn ${p}Fail(family:u32,code:u32,id:u32){let prior=atomicCompareExchangeWeak(&${arena}[
 ${p}H_FAULT],0u,code);if(prior.exchanged){atomicStore(&${arena}[${p}H_FIRST_FAULT_FAMILY],family);
 atomicStore(&${arena}[${p}H_FIRST_FAULT_ID],id);}
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseFault);}
@compute @workgroup_size(1) fn beginSparseCM12PressureTopologyRepair(){
 if(!${p}HeaderValid()){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidHeader}u,${p}Invalid);return;}
 let phase=atomicLoad(&${arena}[${p}H_PHASE]);
 if(phase==${p}PhaseCollecting){return;}
 if(phase!=${p}PhaseUninitialized&&phase!=${p}PhaseAccepted){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidPhase}u,
   ${p}Invalid);return;}
 let accepted=atomicLoad(&${arena}[${p}H_ACCEPTED_GENERATION]);if(accepted>=0x7ffffffeu){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.generationExhausted}u,
   ${p}Invalid);return;}
 atomicStore(&${arena}[${p}H_CANDIDATE_GENERATION],accepted+1u);
 atomicStore(&${arena}[${p}H_TOPOLOGY_GENERATION],ptrTopologyGeneration());
 atomicStore(&${arena}[${p}H_FAULT],0u);
 atomicStore(&${arena}[${p}H_FIRST_FAULT_FAMILY],${p}Invalid);
 atomicStore(&${arena}[${p}H_FIRST_FAULT_ID],${p}Invalid);
 atomicStore(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS],0u);
 atomicStore(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS],0u);
 atomicStore(&${arena}[${p}H_CELL_EXECUTION_COUNT],0u);
 atomicStore(&${arena}[${p}H_CHANGED_BRICK_COUNT],0u);
 atomicStore(&${arena}[${p}H_CAUSE_MASK],0u);
 atomicStore(&${arena}[${p}H_CANDIDATE_WRITE_COUNT],0u);
 atomicStore(&${arena}[${p}H_DIRTY_LEAF_COUNT],0u);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseCollecting);}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyBrickFrontier(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting
  ||atomicLoad(&${arena}[${p}H_FAULT])!=0u){return;}
 if(atomicLoad(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS])!=atomicLoad(&${arena}[
  ${p}H_COVERED_PRODUCER_RECEIPTS])){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.producerCoverageGap}u,
   ${p}Invalid);return;}
 let count=atomicLoad(&${arena}[${p}H_CANDIDATE_WRITE_COUNT]);
 let cause=atomicLoad(&${arena}[${p}H_CAUSE_MASK]);
 for(var rank=0u;rank<count;rank+=1u){
  let brick=atomicLoad(&${arena}[${p}BrickChangedList+rank]);
  if(brick>=${p}BrickCapacity){${p}Fail(${p}BrickFamily,
    ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidCellRange}u,brick);return;}
  let oldState=atomicLoad(&${arena}[${p}BrickOldState+brick]);
  let newState=atomicLoad(&${arena}[${p}BrickNewState+brick]);
  if(oldState!=newState){let oldRange=ptrBrickCellRange(brick,oldState);
   if(!pcmCellQueueTopologyRetiredRange(oldRange.x,oldRange.y,cause)){
    ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidCellRange}u,
     brick);return;}}
 }
 atomicStore(&${arena}[${p}H_CHANGED_BRICK_COUNT],count);
 atomicStore(&${arena}[${p}H_CELL_EXECUTION_COUNT],count);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseExecutingCells);}
@compute @workgroup_size(1) fn finalizeSparseCM12BoundedPressureTopologyRepair(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseExecutingCells){return;}
 if(atomicLoad(&${arena}[${p}H_CELL_EXECUTION_COUNT])!=atomicLoad(&${arena}[
  ${p}H_CHANGED_BRICK_COUNT])){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.cellExecutionGap}u,
   ${p}Invalid);return;}
 if(ptrPressureCoefficientCandidateGeneration()!=ptrPressureCoefficientAcceptedGeneration()){
  ${p}Fail(${p}BrickFamily,
   ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.coefficientGenerationGap}u,
   ${p}Invalid);return;}
 atomicStore(&${arena}[${p}H_ACCEPTED_CHANGED_BRICK_COUNT],atomicLoad(&${arena}[
  ${p}H_CHANGED_BRICK_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_CELL_EXECUTION_COUNT],atomicLoad(&${arena}[
  ${p}H_CELL_EXECUTION_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_BRICK_DIRTY_LEAF_COUNT],atomicLoad(&${arena}[
  ${p}H_DIRTY_LEAF_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_GENERATION],atomicLoad(&${arena}[
  ${p}H_CANDIDATE_GENERATION]));
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseAccepted);}
fn ${p}PreflightWillAppend(brick:u32,generation:u32)->bool{
 return atomicLoad(&${arena}[${p}BrickCandidate+brick])!=generation;}
fn ${p}PreflightDirtyLeafWillAppend(leaf:u32,generation:u32)->bool{
 return atomicLoad(&${arena}[${p}BrickDirtyStamp+leaf])!=generation;}
fn ${p}PreflightCompatible(brick:u32,oldState:u32,newState:u32)->bool{
 let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
 let stamp=atomicLoad(&${arena}[${p}BrickCandidate+brick]);if(stamp!=generation){return true;}
 let acceptedOld=atomicLoad(&${arena}[${p}BrickOldState+brick]);
 let pendingNew=atomicLoad(&${arena}[${p}BrickNewState+brick]);
 return (acceptedOld==oldState&&pendingNew==newState)||pendingNew==oldState;}
fn ${p}PreflightReady(generation:u32,newCount:u32,newLeafCount:u32)->bool{
 return ${p}HeaderValid()&&atomicLoad(&${arena}[${p}H_PHASE])==${p}PhaseCollecting
  &&atomicLoad(&${arena}[${p}H_FAULT])==0u
  &&atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION])==generation
  &&atomicLoad(&${arena}[${p}H_CANDIDATE_WRITE_COUNT])+newCount<=${p}BrickCapacity
  &&atomicLoad(&${arena}[${p}H_DIRTY_LEAF_COUNT])+newLeafCount<=${p}BrickLeafCount;}
// TFX1 proved every bound. This helper has no failure return, retry loop,
// capacity branch, or fault mutation.
fn ${p}PublishPreflightedChangedBrick(brick:u32,oldState:u32,newState:u32,cause:u32,
 ownsLeaf:bool,generation:u32){
 let append=atomicLoad(&${arena}[${p}BrickCandidate+brick])!=generation;
 if(append){atomicStore(&${arena}[${p}BrickOldState+brick],oldState);
  atomicStore(&${arena}[${p}BrickNewState+brick],newState);
  atomicStore(&${arena}[${p}BrickCandidate+brick],generation);
  let slot=atomicAdd(&${arena}[${p}H_CANDIDATE_WRITE_COUNT],1u);
  atomicStore(&${arena}[${p}BrickChangedList+slot],brick);
  atomicAdd(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS],1u);
 }else{let pendingNew=atomicLoad(&${arena}[${p}BrickNewState+brick]);
  if(pendingNew==oldState){atomicStore(&${arena}[${p}BrickNewState+brick],newState);}}
 if(ownsLeaf){let leaf=brick/256u;
  let first=atomicExchange(&${arena}[${p}BrickDirtyStamp+leaf],generation)!=generation;
  if(first){atomicAdd(&${arena}[${p}H_DIRTY_LEAF_COUNT],1u);}}
 atomicOr(&${arena}[${p}H_CAUSE_MASK],cause);}
fn ${p}SealPreflightedTopologyJournalNoFail(topologyGeneration:u32){
 let covered=atomicLoad(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS]);
 atomicStore(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS],covered);
 atomicStore(&${arena}[${p}H_TOPOLOGY_GENERATION],topologyGeneration);}
`;
}
