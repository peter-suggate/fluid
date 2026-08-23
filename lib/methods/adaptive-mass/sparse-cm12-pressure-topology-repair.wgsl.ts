import {
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION,
  type SparseCM12PressureTopologyRepairFamilyLayout,
  type SparseCM12PressureTopologyRepairLayout,
} from "./sparse-cm12-pressure-topology-repair";

export interface SparseCM12PressureTopologyRepairWGSLOptions {
  readonly layout: SparseCM12PressureTopologyRepairLayout;
  readonly arenaName?: string;
  readonly prefix?: string;
  readonly workgroupSize?: 64;
}

const upper = (value: string) => value.replace(/([a-z])([A-Z])/g, "$1_$2").toUpperCase();

/** Binding-free PTR1 source. The enclosing resident declares the atomic arena and hook functions. */
export function createSparseCM12PressureTopologyRepairWGSL(
  options: SparseCM12PressureTopologyRepairWGSLOptions,
): string {
  const { layout } = options;
  const arena = options.arenaName ?? "pressureTopologyRepair";
  const p = options.prefix ?? "ptr";
  const workgroupSize = options.workgroupSize ?? 64;
  if (workgroupSize !== 64) throw new Error("PTR1 numerical brick repair requires 64 lanes");
  const h = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER;
  const f = SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAMILY_HEADER;
  const unusedHeaderFields = new Set([
    "familyHeaderWords", "familyCount", "leafBits", "branch", "brickCapacity",
      "brickHeaderBase", "brickOldStateBase", "brickNewStateBase",
      "brickCauseBase", "brickSeedIndirectY",
    "brickSeedIndirectZ",
  ]);
  const constants = Object.entries(h).filter(([name]) => !unusedHeaderFields.has(name))
    .map(([name, value]) =>
    `const ${p}H_${upper(name)}=${layout.baseWords + value}u;`).join("\n");
  const familyConstants = (family: SparseCM12PressureTopologyRepairFamilyLayout) => {
    const unusedFamilyFields = new Set([
      "capacity", "repairIndirectY", "repairIndirectZ", "workIndirectY",
      "workIndirectZ", "candidateGenerationBase", "bitsBase",
      "dirtyLeafStampBase", "dirtyLeafListBase", "activeLeafListBase",
      "treeLevelCount",
    ]);
    const entries = Object.entries(f).filter(([field]) => !unusedFamilyFields.has(field))
      .map(([field, value]) =>
      `const ${p}BrickF_${upper(field)}=${family.headerBaseWords + value}u;`).join("\n");
    return `${entries}
const ${p}BrickCapacity=${family.capacity}u;
const ${p}BrickCandidate=${family.candidateGenerationBaseWords}u;
const ${p}BrickBits=${family.activeBitsBaseWords}u;
const ${p}BrickDirtyStamp=${family.dirtyLeafStampBaseWords}u;
const ${p}BrickDirtyList=${family.dirtyLeafListBaseWords}u;
const ${p}BrickActiveLeaves=${family.activeLeafListBaseWords}u;
const ${p}BrickLeafCount=${family.leafCount}u;
const ${p}BrickLevelCount=${family.treeLevelCounts.length}u;`;
  };
  const treeSwitch = (family: SparseCM12PressureTopologyRepairFamilyLayout) =>
    `fn ${p}BrickTreeBase(level:u32)->u32{
  switch level{${family.treeLevelBaseWords.map((base, level) =>
    `case ${level}u:{return ${base}u;}`).join("")}default:{return ${p}Invalid;}}}`;
  const treeCountSwitch = (family: SparseCM12PressureTopologyRepairFamilyLayout) =>
    `fn ${p}BrickTreeCount(level:u32)->u32{
  switch level{${family.treeLevelCounts.map((count, level) =>
    `case ${level}u:{return ${count}u;}`).join("")}default:{return 0u;}}}`;
  const reduceEntries = (family: SparseCM12PressureTopologyRepairFamilyLayout) =>
    family.treeLevelCounts
    .slice(1).map((_count, index) => {
      const level = index + 1;
      return `@compute @workgroup_size(64) fn reduceSparseCM12PressureTopologyBrickLevel${level}(
 @builtin(global_invocation_id)gid:vec3u){let dirtyCount=atomicLoad(&${arena}[
 ${p}BrickF_DIRTY_LEAF_COUNT]);if(gid.x>=dirtyCount){return;}
 let leaf=atomicLoad(&${arena}[${p}BrickDirtyList+gid.x]);var parent=leaf;
 for(var depth=0u;depth<${level}u;depth+=1u){parent/=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;}
 let source=${p}BrickTreeBase(${level - 1}u);let first=parent*${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;
 var total=0u;for(var child=0u;child<${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;child+=1u){
  let at=first+child;if(at>=${family.treeLevelCounts[level - 1]}u){break;}
  total+=atomicLoad(&${arena}[source+at]);}
 atomicStore(&${arena}[${p}BrickTreeBase(${level}u)+parent],total);}`;
    }).join("\n");
  const preflightedTopologyPublication = /* wgsl */ `
fn ${p}PreflightWillAppend(brick:u32,generation:u32)->bool{
 return atomicLoad(&${arena}[${p}BrickCandidate+brick])!=generation;}
fn ${p}PreflightDirtyLeafWillAppend(leaf:u32,generation:u32)->bool{
 return atomicLoad(&${arena}[${p}BrickDirtyStamp+leaf])!=generation;}
fn ${p}PreflightCompatible(brick:u32,oldState:u32,newState:u32)->bool{
 let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
 let stamp=atomicLoad(&${arena}[${p}BrickCandidate+brick]);if(stamp!=generation){return true;}
 let acceptedOld=atomicLoad(&${arena}[${p}BrickOldState+brick]);
 let pendingNew=atomicLoad(&${arena}[${p}BrickNewState+brick]);
 return acceptedOld==oldState&&(pendingNew==newState||pendingNew==oldState);}
fn ${p}PreflightReady(generation:u32,newCount:u32,newLeafCount:u32)->bool{
 return ${p}HeaderValid()&&atomicLoad(&${arena}[${p}H_PHASE])==${p}PhaseCollecting
  &&atomicLoad(&${arena}[${p}H_FAULT])==0u
  &&atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION])==generation
  &&atomicLoad(&${arena}[${p}BrickF_CANDIDATE_WRITE_COUNT])+newCount<=${p}BrickCapacity
  &&atomicLoad(&${arena}[${p}BrickF_DIRTY_LEAF_COUNT])+newLeafCount<=${p}BrickLeafCount;}
// TFX1 proved every bound. This helper has no failure return, retry loop,
// capacity branch, or fault mutation.
fn ${p}PublishPreflightedChangedBrick(brick:u32,oldState:u32,newState:u32,cause:u32,
 ownsLeaf:bool,generation:u32){
 let append=atomicLoad(&${arena}[${p}BrickCandidate+brick])!=generation;
 if(append){atomicStore(&${arena}[${p}BrickOldState+brick],oldState);
  atomicStore(&${arena}[${p}BrickNewState+brick],newState);
  atomicStore(&${arena}[${p}BrickCause+brick],cause);
  atomicStore(&${arena}[${p}BrickCandidate+brick],generation);
  atomicAdd(&${arena}[${p}BrickF_CANDIDATE_WRITE_COUNT],1u);
  atomicAdd(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS],1u);
 }else{let pendingNew=atomicLoad(&${arena}[${p}BrickNewState+brick]);
  if(pendingNew==oldState){atomicStore(&${arena}[${p}BrickNewState+brick],newState);}
  atomicOr(&${arena}[${p}BrickCause+brick],cause);}
 if(ownsLeaf){let leaf=brick/${p}LeafBits;
  let first=atomicExchange(&${arena}[${p}BrickDirtyStamp+leaf],generation)!=generation;
  if(first){let slot=atomicAdd(&${arena}[${p}BrickF_DIRTY_LEAF_COUNT],1u);
   atomicStore(&${arena}[${p}BrickDirtyList+slot],leaf);}}
 atomicOr(&${arena}[${p}H_CAUSE_MASK],cause);}
fn ${p}SealPreflightedTopologyJournalNoFail(topologyGeneration:u32){
 let covered=atomicLoad(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS]);
 atomicStore(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS],covered);
 atomicStore(&${arena}[${p}H_TOPOLOGY_GENERATION],topologyGeneration);
}
`;

  return /* wgsl */ `
const ${p}Magic=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC}u;
const ${p}Version=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION}u;
const ${p}HeaderWords=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS}u;
const ${p}Invalid=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID}u;
const ${p}LeafBits=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS}u;
const ${p}Branch=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;
const ${p}BrickFamily=0u;
const ${p}PhaseUninitialized=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.uninitialized}u;
const ${p}PhaseAccepted=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.accepted}u;
const ${p}PhaseCollecting=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.collecting}u;
const ${p}PhaseRepairingBricks=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.repairingBricks}u;
const ${p}PhaseExecutingCells=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.executingCells}u;
const ${p}PhaseAwaitingAcceptance=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.awaitingAcceptance}u;
const ${p}PhaseFault=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.fault}u;
${constants}
${familyConstants(layout.brick)}
const ${p}BrickOldState=${layout.brickOldStateBaseWords}u;
const ${p}BrickNewState=${layout.brickNewStateBaseWords}u;
const ${p}BrickCause=${layout.brickCauseBaseWords}u;
${treeSwitch(layout.brick)}
${treeCountSwitch(layout.brick)}
fn ${p}HeaderValid()->bool{return atomicLoad(&${arena}[${p}H_MAGIC])==${p}Magic
 &&atomicLoad(&${arena}[${p}H_VERSION])==${p}Version
 &&atomicLoad(&${arena}[${p}H_HEADER_WORDS])==${p}HeaderWords
 &&atomicLoad(&${arena}[${p}H_TOTAL_WORDS])==${layout.totalWords}u;}
fn ${p}ZeroIndirects(){
 atomicStore(&${arena}[${p}BrickF_REPAIR_INDIRECT_X],0u);
 atomicStore(&${arena}[${p}BrickF_WORK_INDIRECT_X],0u);
 atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_X],0u);
 atomicStore(&${arena}[${p}H_BRICK_SEED_INDIRECT_X],0u);}
fn ${p}Fail(family:u32,code:u32,id:u32){let prior=atomicCompareExchangeWeak(&${arena}[
 ${p}H_FAULT],0u,code);if(prior.exchanged){atomicStore(&${arena}[${p}H_FIRST_FAULT_FAMILY],family);
 atomicStore(&${arena}[${p}H_FIRST_FAULT_ID],id);}${p}ZeroIndirects();
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseFault);}
fn ${p}QueueBrickLeaf(leaf:u32)->bool{if(leaf>=${p}BrickLeafCount){
 ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);return false;}
 let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
 if(atomicExchange(&${arena}[${p}BrickDirtyStamp+leaf],generation)==generation){return true;}
 let slot=atomicAdd(&${arena}[${p}BrickF_DIRTY_LEAF_COUNT],1u);if(slot>=${p}BrickLeafCount){
 ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);return false;}
 atomicStore(&${arena}[${p}BrickDirtyList+slot],leaf);return true;}
@compute @workgroup_size(1) fn beginSparseCM12PressureTopologyRepair(){
 if(!${p}HeaderValid()){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidHeader}u,${p}Invalid);return;}
 let phase=atomicLoad(&${arena}[${p}H_PHASE]);if(phase!=${p}PhaseUninitialized
  &&phase!=${p}PhaseAccepted){if(phase==${p}PhaseCollecting){return;}${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidPhase}u,${p}Invalid);return;}
 let accepted=atomicLoad(&${arena}[${p}H_ACCEPTED_GENERATION]);if(accepted>=0x7ffffffeu){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.generationExhausted}u,
  ${p}Invalid);return;}atomicStore(&${arena}[${p}H_CANDIDATE_GENERATION],accepted+1u);
 atomicStore(&${arena}[${p}H_FRAME_GENERATION],ptrFrameGeneration());
 atomicStore(&${arena}[${p}H_TOPOLOGY_GENERATION],ptrTopologyGeneration());
 atomicStore(&${arena}[${p}H_PCM_CELL_GENERATION],${p}Invalid);
 atomicStore(&${arena}[${p}H_PCM_ROW_GENERATION],${p}Invalid);
 atomicStore(&${arena}[${p}H_COEFFICIENT_GENERATION],${p}Invalid);
 atomicStore(&${arena}[${p}H_FAULT],0u);atomicStore(&${arena}[${p}H_FIRST_FAULT_FAMILY],${p}Invalid);
 atomicStore(&${arena}[${p}H_FIRST_FAULT_ID],${p}Invalid);
 atomicStore(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS],0u);
 atomicStore(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS],0u);
 atomicStore(&${arena}[${p}H_CELL_EXECUTION_COUNT],0u);
 atomicStore(&${arena}[${p}H_BRICK_STATE_COMMIT_COUNT],0u);atomicStore(&${arena}[${p}H_CAUSE_MASK],0u);
 atomicStore(&${arena}[${p}BrickF_PREVIOUS_ACTIVE_LEAF_COUNT],atomicLoad(&${arena}[
  ${p}BrickF_ACTIVE_LEAF_COUNT]));atomicStore(&${arena}[${p}BrickF_ACTIVE_LEAF_COUNT],0u);
 atomicStore(&${arena}[${p}BrickF_CANDIDATE_WRITE_COUNT],0u);
 atomicStore(&${arena}[${p}BrickF_DIRTY_LEAF_COUNT],0u);
 atomicStore(&${arena}[${p}BrickF_REPAIRED_LEAF_COUNT],0u);
 atomicStore(&${arena}[${p}BrickF_REPAIR_INDIRECT_X],0u);
 atomicStore(&${arena}[${p}BrickF_WORK_INDIRECT_X],0u);
 atomicStore(&${arena}[${p}H_BRICK_SEED_INDIRECT_X],(atomicLoad(&${arena}[
  ${p}BrickF_PREVIOUS_ACTIVE_LEAF_COUNT])+63u)/64u);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseCollecting);}
@compute @workgroup_size(1) fn captureSparseCM12PressureTopologyConsumerGenerations(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting){return;}
 atomicStore(&${arena}[${p}H_PCM_CELL_GENERATION],ptrPCMCellCandidateGeneration());
 atomicStore(&${arena}[${p}H_PCM_ROW_GENERATION],ptrPCMRowCandidateGeneration());
 atomicStore(&${arena}[${p}H_COEFFICIENT_GENERATION],ptrPressureCoefficientCandidateGeneration());}
@compute @workgroup_size(64) fn seedPreviousSparseCM12PressureTopologyBrickLeaves(
 @builtin(global_invocation_id)gid:vec3u){let count=atomicLoad(&${arena}[
 ${p}BrickF_PREVIOUS_ACTIVE_LEAF_COUNT]);if(gid.x>=count){return;}
 let leaf=atomicLoad(&${arena}[${p}BrickActiveLeaves+gid.x]);_=${p}QueueBrickLeaf(leaf);}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyBrickFrontier(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting
  ||atomicLoad(&${arena}[${p}H_FAULT])!=0u){return;}
 if(atomicLoad(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS])!=atomicLoad(&${arena}[
  ${p}H_COVERED_PRODUCER_RECEIPTS])){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.producerCoverageGap}u,${p}Invalid);return;}
 atomicStore(&${arena}[${p}BrickF_REPAIR_INDIRECT_X],atomicLoad(&${arena}[
  ${p}BrickF_DIRTY_LEAF_COUNT]));atomicStore(&${arena}[${p}H_PHASE],${p}PhaseRepairingBricks);}
var<workgroup>${p}LeafCounts:array<u32,8>;
fn ${p}RepairBrickLeaf(lane:u32,work:u32){
 let dirtyCount=atomicLoad(&${arena}[${p}BrickF_DIRTY_LEAF_COUNT]);let invocationOk=work<dirtyCount;
 let safeWork=select(0u,work,invocationOk);
 let leaf=atomicLoad(&${arena}[${p}BrickDirtyList+safeWork]);
 let leafOk=leaf<${p}BrickLeafCount;let workOk=invocationOk&&leafOk;
 if(lane==0u&&invocationOk&&!leafOk){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);}
 var count=0u;if(lane<8u&&workOk){let word=leaf*8u+lane;let first=word*32u;var bits=0u;
  let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
  for(var bit=0u;bit<32u;bit+=1u){let id=first+bit;if(id<${p}BrickCapacity
   &&atomicLoad(&${arena}[${p}BrickCandidate+id])==generation){bits|=1u<<bit;}}
  atomicStore(&${arena}[${p}BrickBits+word],bits);count=countOneBits(bits);
  }if(lane<8u){${p}LeafCounts[lane]=count;}workgroupBarrier();if(lane==0u&&workOk){var total=0u;
  for(var word=0u;word<8u;word+=1u){total+=${p}LeafCounts[word];}
  atomicStore(&${arena}[${p}BrickTreeBase(0u)+leaf],total);
  if(total>0u){let slot=atomicAdd(&${arena}[${p}BrickF_ACTIVE_LEAF_COUNT],1u);
   if(slot<${p}BrickLeafCount){atomicStore(&${arena}[
    ${p}BrickActiveLeaves+slot],leaf);}else{${p}Fail(${p}BrickFamily,
    ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);}}
  atomicAdd(&${arena}[${p}BrickF_REPAIRED_LEAF_COUNT],1u);}}
@compute @workgroup_size(64) fn repairSparseCM12PressureTopologyBrickLeaves(
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
 ${p}RepairBrickLeaf(lane,wid.x);}
${reduceEntries(layout.brick)}
fn ptrChangedBrickInvocation(rank:u32)->u32{let root=atomicLoad(&${arena}[
 ${p}BrickTreeBase(${p}BrickLevelCount-1u)]);if(rank>=root){return ${p}Invalid;}
 var node=0u;var remaining=rank;var level=${p}BrickLevelCount-1u;loop{if(level==0u){break;}
  let childBase=node*${p}Branch;var selected=${p}Invalid;
  for(var child=0u;child<${p}Branch;child+=1u){let at=childBase+child;
   if(at>=${p}BrickTreeCount(level-1u)){break;}
   let count=atomicLoad(&${arena}[${p}BrickTreeBase(level-1u)+at]);
   if(remaining<count){selected=at;break;}remaining-=count;}
  if(selected==${p}Invalid){return ${p}Invalid;}node=selected;level-=1u;}
 let first=node*${p}LeafBits;for(var word=0u;word<8u;word+=1u){let bits=atomicLoad(&${arena}[
  ${p}BrickBits+node*8u+word]);let count=countOneBits(bits);if(remaining>=count){
  remaining-=count;continue;}for(var bit=0u;bit<32u;bit+=1u){if((bits&(1u<<bit))==0u){continue;}
  if(remaining==0u){let id=first+word*32u+bit;return select(${p}Invalid,id,
   id<${p}BrickCapacity);}remaining-=1u;}}return ${p}Invalid;}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyBrickPlan(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseRepairingBricks){return;}
 if(atomicLoad(&${arena}[${p}BrickF_REPAIRED_LEAF_COUNT])!=atomicLoad(&${arena}[
  ${p}BrickF_DIRTY_LEAF_COUNT])){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.brickRepairGap}u,${p}Invalid);return;}
 let root=atomicLoad(&${arena}[${p}BrickTreeBase(${layout.brick.treeLevelCounts.length - 1}u)]);
 atomicStore(&${arena}[${p}H_CHANGED_BRICK_COUNT],root);
 atomicStore(&${arena}[${p}BrickF_WORK_COUNT],root);
 atomicStore(&${arena}[${p}BrickF_WORK_INDIRECT_X],root);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseExecutingCells);}
fn ${p}RepairCell(cell:u32,current:bool,cause:u32){
 let enabled=ptrApplyPressureCellClassification(cell,current);
 _=pcmCellSetCandidate(cell,enabled,cause,false);}
@compute @workgroup_size(64) fn repairSparseCM12PressureTopologyChangedBricks(
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
 let phaseOk=atomicLoad(&${arena}[${p}H_PHASE])==${p}PhaseExecutingCells;
 let selected=ptrChangedBrickInvocation(wid.x);let brick=select(0u,selected,
  phaseOk&&selected!=${p}Invalid);let selectedOk=phaseOk&&selected!=${p}Invalid;
 let oldState=atomicLoad(&${arena}[${p}BrickOldState+brick]);
 let newState=atomicLoad(&${arena}[${p}BrickNewState+brick]);
 let cause=atomicLoad(&${arena}[${p}BrickCause+brick]);
 let oldRange=ptrBrickCellRange(brick,oldState);let newRange=ptrBrickCellRange(brick,newState);
 let oldRangeOk=oldRange.x+oldRange.y>=oldRange.x
  &&oldRange.x+oldRange.y<=ptrCellCapacity();
 let newRangeOk=newRange.x+newRange.y>=newRange.x
  &&newRange.x+newRange.y<=ptrCellCapacity();
 let workOk=selectedOk&&oldRangeOk&&newRangeOk;
 if(lane==0u&&selectedOk&&!oldRangeOk){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidCellRange}u,brick);}
 if(lane==0u&&selectedOk&&!newRangeOk){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidCellRange}u,brick);}
 if(workOk&&oldState!=newState){for(var local=lane;local<oldRange.y;local+=64u){
  ${p}RepairCell(oldRange.x+local,false,cause);}}
 if(workOk){for(var local=lane;local<newRange.y;local+=64u){let cell=newRange.x+local;
  ${p}RepairCell(cell,true,cause);}}
 workgroupBarrier();if(lane==0u&&workOk){atomicAdd(&${arena}[${p}H_CELL_EXECUTION_COUNT],1u);}}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyCellExecution(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseExecutingCells){return;}
 if(atomicLoad(&${arena}[${p}H_CELL_EXECUTION_COUNT])!=atomicLoad(&${arena}[
  ${p}H_CHANGED_BRICK_COUNT])){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.cellExecutionGap}u,${p}Invalid);return;}
}
@compute @workgroup_size(1) fn sealSparseCM12PressureTopologyRowImage(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseExecutingCells){return;}
 if(atomicLoad(&${arena}[${p}H_TOPOLOGY_GENERATION])!=ptrTopologyGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.topologyGenerationGap}u,
  ${p}Invalid);return;}
 if(atomicLoad(&${arena}[${p}H_PCM_CELL_GENERATION])!=ptrPCMCellAcceptedGeneration()
  ||atomicLoad(&${arena}[${p}H_PCM_ROW_GENERATION])!=ptrPCMRowAcceptedGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.pcmGenerationGap}u,
  ${p}Invalid);return;}
 // The direct coefficient image consumes this topology closure after the row
 // frontier is complete. The captured PCA generation must remain its live
 // candidate until PEI accepts the complete fine/coarse image.
 if(atomicLoad(&${arena}[${p}H_COEFFICIENT_GENERATION])
   !=ptrPressureCoefficientCandidateGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.coefficientGenerationGap}u,
  ${p}Invalid);return;}
 let count=atomicLoad(&${arena}[${p}H_CHANGED_BRICK_COUNT]);
 atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_X],(count+63u)/64u);
 atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_Y],1u);atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_Z],1u);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseAwaitingAcceptance);}
@compute @workgroup_size(64) fn commitSparseCM12PressureTopologyBrickStates(
 @builtin(global_invocation_id)gid:vec3u){if(atomicLoad(&${arena}[${p}H_PHASE])
 !=${p}PhaseAwaitingAcceptance){return;}let brick=ptrChangedBrickInvocation(gid.x);
 // Never publish PTR's persistent brick state unless the coefficient image
 // which consumed the same PCM/topology epoch has accepted.
 if(atomicLoad(&${arena}[${p}H_COEFFICIENT_GENERATION])
   !=ptrPressureCoefficientAcceptedGeneration()){return;}
 if(brick==${p}Invalid){return;}atomicAdd(&${arena}[${p}H_BRICK_STATE_COMMIT_COUNT],1u);}
@compute @workgroup_size(1) fn finalizeSparseCM12BoundedPressureTopologyRepair(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseAwaitingAcceptance){return;}
 if(atomicLoad(&${arena}[${p}H_COEFFICIENT_GENERATION])
   !=ptrPressureCoefficientAcceptedGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.coefficientGenerationGap}u,
  ${p}Invalid);return;}
 if(atomicLoad(&${arena}[${p}H_BRICK_STATE_COMMIT_COUNT])!=atomicLoad(&${arena}[
 ${p}H_CHANGED_BRICK_COUNT])){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.cellExecutionGap}u,${p}Invalid);return;}
 atomicStore(&${arena}[${p}H_ACCEPTED_CHANGED_BRICK_COUNT],atomicLoad(&${arena}[
  ${p}H_CHANGED_BRICK_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_CELL_EXECUTION_COUNT],atomicLoad(&${arena}[
  ${p}H_CELL_EXECUTION_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_BRICK_DIRTY_LEAF_COUNT],atomicLoad(&${arena}[
  ${p}BrickF_DIRTY_LEAF_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_GENERATION],atomicLoad(&${arena}[
  ${p}H_CANDIDATE_GENERATION]));${p}ZeroIndirects();atomicStore(&${arena}[
  ${p}H_PHASE],${p}PhaseAccepted);}
${preflightedTopologyPublication}
`;
}
