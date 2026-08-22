import {
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH,
  SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE,
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
  /** Opt-in TFX1 seam. Omitted output remains byte-for-byte unchanged. */
  readonly preflightedTopologyPublication?: boolean;
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
  const constants = Object.entries(h).map(([name, value]) =>
    `const ${p}H_${upper(name)}=${layout.baseWords + value}u;`).join("\n");
  const familyConstants = (name: "Brick" | "Row",
    family: SparseCM12PressureTopologyRepairFamilyLayout) => {
    const entries = Object.entries(f).map(([field, value]) =>
      `const ${p}${name}F_${upper(field)}=${family.headerBaseWords + value}u;`).join("\n");
    return `${entries}
const ${p}${name}Capacity=${family.capacity}u;
const ${p}${name}Candidate=${family.candidateGenerationBaseWords}u;
const ${p}${name}Bits=${family.activeBitsBaseWords}u;
const ${p}${name}DirtyStamp=${family.dirtyLeafStampBaseWords}u;
const ${p}${name}DirtyList=${family.dirtyLeafListBaseWords}u;
const ${p}${name}ActiveLeaves=${family.activeLeafListBaseWords}u;
const ${p}${name}Execution=${family.executionGenerationBaseWords}u;
const ${p}${name}LeafCount=${family.leafCount}u;
const ${p}${name}LevelCount=${family.treeLevelCounts.length}u;`;
  };
  const treeSwitch = (name: "Brick" | "Row",
    family: SparseCM12PressureTopologyRepairFamilyLayout) => `fn ${p}${name}TreeBase(level:u32)->u32{
  switch level{${family.treeLevelBaseWords.map((base, level) =>
    `case ${level}u:{return ${base}u;}`).join("")}default:{return ${p}Invalid;}}}`;
  const treeCountSwitch = (name: "Brick" | "Row",
    family: SparseCM12PressureTopologyRepairFamilyLayout) => `fn ${p}${name}TreeCount(level:u32)->u32{
  switch level{${family.treeLevelCounts.map((count, level) =>
    `case ${level}u:{return ${count}u;}`).join("")}default:{return 0u;}}}`;
  const reduceEntries = (name: "Brick" | "Row",
    family: SparseCM12PressureTopologyRepairFamilyLayout) => family.treeLevelCounts
    .slice(1).map((_count, index) => {
      const level = index + 1;
      return `@compute @workgroup_size(64) fn reduceSparseCM12PressureTopology${name}Level${level}(
 @builtin(global_invocation_id)gid:vec3u){let dirtyCount=atomicLoad(&${arena}[
 ${p}${name}F_DIRTY_LEAF_COUNT]);if(gid.x>=dirtyCount){return;}
 let leaf=atomicLoad(&${arena}[${p}${name}DirtyList+gid.x]);var parent=leaf;
 for(var depth=0u;depth<${level}u;depth+=1u){parent/=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;}
 let source=${p}${name}TreeBase(${level - 1}u);let first=parent*${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;
 var total=0u;for(var child=0u;child<${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;child+=1u){
  let at=first+child;if(at>=${family.treeLevelCounts[level - 1]}u){break;}
  total+=atomicLoad(&${arena}[source+at]);}
 atomicStore(&${arena}[${p}${name}TreeBase(${level}u)+parent],total);}`;
    }).join("\n");
  const preflightedTopologyPublication = options.preflightedTopologyPublication ? /* wgsl */ `
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
` : "";

  return /* wgsl */ `
const ${p}Magic=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_MAGIC}u;
const ${p}Version=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_VERSION}u;
const ${p}HeaderWords=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_HEADER_WORDS}u;
const ${p}Invalid=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_INVALID}u;
const ${p}LeafBits=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_LEAF_BITS}u;
const ${p}Branch=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_BRANCH}u;
const ${p}BrickFamily=0u;const ${p}RowFamily=1u;
const ${p}PhaseUninitialized=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.uninitialized}u;
const ${p}PhaseAccepted=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.accepted}u;
const ${p}PhaseCollecting=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.collecting}u;
const ${p}PhaseRepairingBricks=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.repairingBricks}u;
const ${p}PhaseExecutingCells=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.executingCells}u;
const ${p}PhaseRepairingRows=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.repairingRows}u;
const ${p}PhaseExecutingRows=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.executingRows}u;
const ${p}PhaseAwaitingAcceptance=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.awaitingAcceptance}u;
const ${p}PhaseFault=${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_PHASE.fault}u;
${constants}
${familyConstants("Brick", layout.brick)}
${familyConstants("Row", layout.row)}
const ${p}BrickOldState=${layout.brickOldStateBaseWords}u;
const ${p}BrickNewState=${layout.brickNewStateBaseWords}u;
const ${p}BrickCause=${layout.brickCauseBaseWords}u;
const ${p}BrickAcceptedState=${layout.brickAcceptedStateBaseWords}u;
${treeSwitch("Brick", layout.brick)}
${treeSwitch("Row", layout.row)}
${treeCountSwitch("Brick", layout.brick)}
${treeCountSwitch("Row", layout.row)}
fn ${p}HeaderValid()->bool{return atomicLoad(&${arena}[${p}H_MAGIC])==${p}Magic
 &&atomicLoad(&${arena}[${p}H_VERSION])==${p}Version
 &&atomicLoad(&${arena}[${p}H_HEADER_WORDS])==${p}HeaderWords
 &&atomicLoad(&${arena}[${p}H_TOTAL_WORDS])==${layout.totalWords}u;}
fn ${p}FamilyHeader(family:u32)->u32{return select(${layout.row.headerBaseWords}u,
 ${layout.brick.headerBaseWords}u,family==${p}BrickFamily);}
fn ${p}FamilyCapacity(family:u32)->u32{return select(${p}RowCapacity,
 ${p}BrickCapacity,family==${p}BrickFamily);}
fn ${p}FamilyCandidate(family:u32)->u32{return select(${p}RowCandidate,
 ${p}BrickCandidate,family==${p}BrickFamily);}
fn ${p}FamilyBits(family:u32)->u32{return select(${p}RowBits,
 ${p}BrickBits,family==${p}BrickFamily);}
fn ${p}FamilyDirtyStamp(family:u32)->u32{return select(${p}RowDirtyStamp,
 ${p}BrickDirtyStamp,family==${p}BrickFamily);}
fn ${p}FamilyDirtyList(family:u32)->u32{return select(${p}RowDirtyList,
 ${p}BrickDirtyList,family==${p}BrickFamily);}
fn ${p}FamilyActiveLeaves(family:u32)->u32{return select(${p}RowActiveLeaves,
 ${p}BrickActiveLeaves,family==${p}BrickFamily);}
fn ${p}FamilyLeafCount(family:u32)->u32{return select(${p}RowLeafCount,
 ${p}BrickLeafCount,family==${p}BrickFamily);}
fn ${p}FamilyTreeBase(family:u32,level:u32)->u32{return select(${p}RowTreeBase(level),
 ${p}BrickTreeBase(level),family==${p}BrickFamily);}
fn ${p}FamilyTreeCount(family:u32,level:u32)->u32{return select(${p}RowTreeCount(level),
 ${p}BrickTreeCount(level),family==${p}BrickFamily);}
fn ${p}ZeroIndirects(){for(var family=0u;family<2u;family+=1u){let header=${p}FamilyHeader(family);
 atomicStore(&${arena}[header+${f.repairIndirectX}u],0u);
 atomicStore(&${arena}[header+${f.workIndirectX}u],0u);}
 atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_X],0u);
 atomicStore(&${arena}[${p}H_BRICK_SEED_INDIRECT_X],0u);
 atomicStore(&${arena}[${p}H_ROW_SEED_INDIRECT_X],0u);}
fn ${p}Fail(family:u32,code:u32,id:u32){let prior=atomicCompareExchangeWeak(&${arena}[
 ${p}H_FAULT],0u,code);if(prior.exchanged){atomicStore(&${arena}[${p}H_FIRST_FAULT_FAMILY],family);
 atomicStore(&${arena}[${p}H_FIRST_FAULT_ID],id);}${p}ZeroIndirects();
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseFault);}
fn ${p}QueueLeaf(family:u32,leaf:u32)->bool{if(leaf>=${p}FamilyLeafCount(family)){
 ${p}Fail(family,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);return false;}
 let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
 if(atomicExchange(&${arena}[${p}FamilyDirtyStamp(family)+leaf],generation)==generation){return true;}
 let header=${p}FamilyHeader(family);let slot=atomicAdd(&${arena}[
 header+${f.dirtyLeafCount}u],1u);if(slot>=${p}FamilyLeafCount(family)){
 ${p}Fail(family,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);return false;}
 atomicStore(&${arena}[${p}FamilyDirtyList(family)+slot],leaf);return true;}
fn ${p}MarkRow(row:u32,cause:u32)->bool{if(row>=${p}RowCapacity){${p}Fail(${p}RowFamily,
 ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidRow}u,row);return false;}
 let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
 let first=atomicExchange(&${arena}[${p}RowCandidate+row],generation)!=generation;
 if(first){atomicAdd(&${arena}[${p}RowF_CANDIDATE_WRITE_COUNT],1u);
  atomicAdd(&${arena}[${p}RowF_CLOSURE_WRITE_COUNT],1u);_=${p}QueueLeaf(${p}RowFamily,row/${p}LeafBits);}
 atomicOr(&${arena}[${p}RowF_CAUSE_MASK],cause);return true;}
fn ptrRecordChangedBrick(brick:u32,oldState:u32,newState:u32,cause:u32,receipt:bool)->bool{
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting){${p}Fail(${p}BrickFamily,
 ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidPhase}u,brick);return false;}
 if(brick>=${p}BrickCapacity){${p}Fail(${p}BrickFamily,
 ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidBrick}u,brick);return false;}
 let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
 let publishing=generation|0x80000000u;var first=false;var spins=0u;
 loop{let observed=atomicLoad(&${arena}[${p}BrickCandidate+brick]);
  if(observed==generation){break;}if(observed==publishing){spins+=1u;if(spins<1024u){continue;}
   ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.producerCoverageGap}u,
    brick);return false;}if((observed&0x80000000u)!=0u){${p}Fail(${p}BrickFamily,
   ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.conflictingBrickRecord}u,brick);return false;}
  let claim=atomicCompareExchangeWeak(&${arena}[${p}BrickCandidate+brick],observed,publishing);
  if(claim.exchanged){first=true;break;}}
 if(first){atomicStore(&${arena}[${p}BrickOldState+brick],oldState);
  atomicStore(&${arena}[${p}BrickNewState+brick],newState);
  atomicStore(&${arena}[${p}BrickCause+brick],cause);
  atomicStore(&${arena}[${p}BrickCandidate+brick],generation);
  atomicAdd(&${arena}[${p}BrickF_CANDIDATE_WRITE_COUNT],1u);
  if(receipt){atomicAdd(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS],1u);}
  _=${p}QueueLeaf(${p}BrickFamily,brick/${p}LeafBits);
 }else{let acceptedOld=atomicLoad(&${arena}[${p}BrickOldState+brick]);
  let pendingNew=atomicLoad(&${arena}[${p}BrickNewState+brick]);
  if(acceptedOld==oldState&&pendingNew==newState){
   atomicOr(&${arena}[${p}BrickCause+brick],cause);
  }else if(pendingNew==oldState){
   atomicStore(&${arena}[${p}BrickNewState+brick],newState);
   atomicOr(&${arena}[${p}BrickCause+brick],cause);
  }else{${p}Fail(${p}BrickFamily,
   ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.conflictingBrickRecord}u,brick);return false;}}
 atomicOr(&${arena}[${p}H_CAUSE_MASK],cause);return true;}
fn ptrSealTopologyJournal(expected:u32,topologyGeneration:u32)->bool{
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidPhase}u,${p}Invalid);return false;}
 atomicStore(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS],expected);
 atomicStore(&${arena}[${p}H_TOPOLOGY_GENERATION],topologyGeneration);
 if(atomicLoad(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS])!=expected){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.producerCoverageGap}u,${p}Invalid);return false;}
 return true;}
fn ptrRejectTopologyJournal()->bool{
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting){return false;}
 ${p}ZeroIndirects();atomicStore(&${arena}[${p}H_PHASE],${p}PhaseAccepted);return true;}
@compute @workgroup_size(64) fn importSparseCM12PressureTopologyChangedBricks(
 @builtin(global_invocation_id)gid:vec3u){let count=ptrTopologyJournalCount();if(gid.x>=count){return;}
 let record=ptrTopologyJournalRecord(gid.x);_=ptrRecordChangedBrick(record.x,record.y,record.z,
  record.w,true);}
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
 atomicStore(&${arena}[${p}H_PCF_GENERATION],${p}Invalid);
 atomicStore(&${arena}[${p}H_FAULT],0u);atomicStore(&${arena}[${p}H_FIRST_FAULT_FAMILY],${p}Invalid);
 atomicStore(&${arena}[${p}H_FIRST_FAULT_ID],${p}Invalid);
 atomicStore(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS],0u);
 atomicStore(&${arena}[${p}H_COVERED_PRODUCER_RECEIPTS],0u);
 atomicStore(&${arena}[${p}H_CELL_EXECUTION_COUNT],0u);atomicStore(&${arena}[${p}H_ROW_EXECUTION_COUNT],0u);
 atomicStore(&${arena}[${p}H_BRICK_STATE_COMMIT_COUNT],0u);atomicStore(&${arena}[${p}H_CAUSE_MASK],0u);
 for(var family=0u;family<2u;family+=1u){let header=${p}FamilyHeader(family);
  atomicStore(&${arena}[header+${f.previousActiveLeafCount}u],atomicLoad(&${arena}[
   header+${f.activeLeafCount}u]));atomicStore(&${arena}[header+${f.activeLeafCount}u],0u);
  atomicStore(&${arena}[header+${f.candidateWriteCount}u],0u);
  atomicStore(&${arena}[header+${f.closureWriteCount}u],0u);
  atomicStore(&${arena}[header+${f.dirtyLeafCount}u],0u);
  atomicStore(&${arena}[header+${f.repairedLeafCount}u],0u);
  atomicStore(&${arena}[header+${f.executedCount}u],0u);
  atomicStore(&${arena}[header+${f.repairIndirectX}u],0u);
  atomicStore(&${arena}[header+${f.workIndirectX}u],0u);}
 atomicStore(&${arena}[${p}H_BRICK_SEED_INDIRECT_X],(atomicLoad(&${arena}[
  ${p}BrickF_PREVIOUS_ACTIVE_LEAF_COUNT])+63u)/64u);
 atomicStore(&${arena}[${p}H_ROW_SEED_INDIRECT_X],(atomicLoad(&${arena}[
  ${p}RowF_PREVIOUS_ACTIVE_LEAF_COUNT])+63u)/64u);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseCollecting);}
@compute @workgroup_size(1) fn captureSparseCM12PressureTopologyConsumerGenerations(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting){return;}
 atomicStore(&${arena}[${p}H_PCM_CELL_GENERATION],ptrPCMCellCandidateGeneration());
 atomicStore(&${arena}[${p}H_PCM_ROW_GENERATION],ptrPCMRowCandidateGeneration());
 atomicStore(&${arena}[${p}H_PCF_GENERATION],ptrPCFCandidateGeneration());}
fn ${p}SeedPrevious(family:u32,invocation:u32){let header=${p}FamilyHeader(family);
 let count=atomicLoad(&${arena}[header+${f.previousActiveLeafCount}u]);if(invocation>=count){return;}
 let leaf=atomicLoad(&${arena}[${p}FamilyActiveLeaves(family)+invocation]);
 _=${p}QueueLeaf(family,leaf);}
@compute @workgroup_size(64) fn seedPreviousSparseCM12PressureTopologyBrickLeaves(
 @builtin(global_invocation_id)gid:vec3u){${p}SeedPrevious(${p}BrickFamily,gid.x);}
@compute @workgroup_size(64) fn seedPreviousSparseCM12PressureTopologyRowLeaves(
 @builtin(global_invocation_id)gid:vec3u){${p}SeedPrevious(${p}RowFamily,gid.x);}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyBrickFrontier(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseCollecting
  ||atomicLoad(&${arena}[${p}H_FAULT])!=0u){return;}
 if(atomicLoad(&${arena}[${p}H_EXPECTED_PRODUCER_RECEIPTS])!=atomicLoad(&${arena}[
  ${p}H_COVERED_PRODUCER_RECEIPTS])){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.producerCoverageGap}u,${p}Invalid);return;}
 atomicStore(&${arena}[${p}BrickF_REPAIR_INDIRECT_X],atomicLoad(&${arena}[
  ${p}BrickF_DIRTY_LEAF_COUNT]));atomicStore(&${arena}[${p}H_PHASE],${p}PhaseRepairingBricks);}
var<workgroup>${p}LeafCounts:array<u32,8>;
fn ${p}RepairLeaf(family:u32,lane:u32,work:u32){let header=${p}FamilyHeader(family);
 let dirtyCount=atomicLoad(&${arena}[header+${f.dirtyLeafCount}u]);let invocationOk=work<dirtyCount;
 let safeWork=select(0u,work,invocationOk);
 let leaf=atomicLoad(&${arena}[${p}FamilyDirtyList(family)+safeWork]);
 let leafOk=leaf<${p}FamilyLeafCount(family);let workOk=invocationOk&&leafOk;
 if(lane==0u&&invocationOk&&!leafOk){${p}Fail(family,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);}
 var count=0u;if(lane<8u&&workOk){let word=leaf*8u+lane;let first=word*32u;var bits=0u;
  let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
  for(var bit=0u;bit<32u;bit+=1u){let id=first+bit;if(id<${p}FamilyCapacity(family)
   &&atomicLoad(&${arena}[${p}FamilyCandidate(family)+id])==generation){bits|=1u<<bit;}}
  atomicStore(&${arena}[${p}FamilyBits(family)+word],bits);count=countOneBits(bits);
  }if(lane<8u){${p}LeafCounts[lane]=count;}workgroupBarrier();if(lane==0u&&workOk){var total=0u;
  for(var word=0u;word<8u;word+=1u){total+=${p}LeafCounts[word];}
  atomicStore(&${arena}[${p}FamilyTreeBase(family,0u)+leaf],total);
  if(total>0u){let slot=atomicAdd(&${arena}[header+${f.activeLeafCount}u],1u);
   if(slot<${p}FamilyLeafCount(family)){atomicStore(&${arena}[
    ${p}FamilyActiveLeaves(family)+slot],leaf);}else{${p}Fail(family,
    ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.dirtyLeafCapacity}u,leaf);}}
  atomicAdd(&${arena}[header+${f.repairedLeafCount}u],1u);}}
@compute @workgroup_size(64) fn repairSparseCM12PressureTopologyBrickLeaves(
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
 ${p}RepairLeaf(${p}BrickFamily,lane,wid.x);}
@compute @workgroup_size(64) fn repairSparseCM12PressureTopologyRowLeaves(
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
 ${p}RepairLeaf(${p}RowFamily,lane,wid.x);}
${reduceEntries("Brick", layout.brick)}
${reduceEntries("Row", layout.row)}
fn ${p}RankSelect(family:u32,rank:u32)->u32{let levels=select(${p}RowLevelCount,
 ${p}BrickLevelCount,family==${p}BrickFamily);let root=atomicLoad(&${arena}[
 ${p}FamilyTreeBase(family,levels-1u)]);if(rank>=root){return ${p}Invalid;}
 var node=0u;var remaining=rank;var level=levels-1u;loop{if(level==0u){break;}
  let childBase=node*${p}Branch;var selected=${p}Invalid;
  for(var child=0u;child<${p}Branch;child+=1u){let at=childBase+child;
   if(at>=${p}FamilyTreeCount(family,level-1u)){break;}
   let count=atomicLoad(&${arena}[${p}FamilyTreeBase(family,level-1u)+at]);
   if(remaining<count){selected=at;break;}remaining-=count;}
  if(selected==${p}Invalid){return ${p}Invalid;}node=selected;level-=1u;}
 let first=node*${p}LeafBits;for(var word=0u;word<8u;word+=1u){let bits=atomicLoad(&${arena}[
  ${p}FamilyBits(family)+node*8u+word]);let count=countOneBits(bits);if(remaining>=count){
  remaining-=count;continue;}for(var bit=0u;bit<32u;bit+=1u){if((bits&(1u<<bit))==0u){continue;}
  if(remaining==0u){let id=first+word*32u+bit;return select(${p}Invalid,id,
   id<${p}FamilyCapacity(family));}remaining-=1u;}}return ${p}Invalid;}
fn ptrChangedBrickInvocation(rank:u32)->u32{return ${p}RankSelect(${p}BrickFamily,rank);}
fn ptrChangedRowInvocation(rank:u32)->u32{return ${p}RankSelect(${p}RowFamily,rank);}
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
fn ${p}MarkCellRows(cell:u32,cause:u32)->bool{if(cell>=ptrCellCapacity()){
 ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.invalidCell}u,cell);
 return false;}let range=ptrCellIncidenceRange(cell);var ok=true;
 for(var at=range.x;at<range.y;at+=1u){ok=${p}MarkRow(ptrIncidenceRow(at),cause)&&ok;}return ok;}
fn ${p}RepairCell(cell:u32,current:bool,cause:u32){let previous=pcmCellContains(cell);
 let enabled=ptrApplyPressureCellClassification(cell,current);
 _=pcmCellSetCandidate(cell,enabled,cause,false);if(previous!=enabled){
  _=pcfRecordCellMembershipEvent(cell);}_=pcfRecordTopologyCellEvent(cell);
 _=${p}MarkCellRows(cell,cause|${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_CAUSE.incidenceClosure}u);}
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
 workgroupBarrier();if(lane==0u&&workOk){atomicStore(&${arena}[${p}BrickExecution+brick],atomicLoad(&${arena}[
  ${p}H_CANDIDATE_GENERATION]));atomicAdd(&${arena}[${p}H_CELL_EXECUTION_COUNT],1u);
  atomicAdd(&${arena}[${p}BrickF_EXECUTED_COUNT],1u);}}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyCellExecution(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseExecutingCells){return;}
 if(atomicLoad(&${arena}[${p}H_CELL_EXECUTION_COUNT])!=atomicLoad(&${arena}[
  ${p}H_CHANGED_BRICK_COUNT])){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.cellExecutionGap}u,${p}Invalid);return;}
 atomicStore(&${arena}[${p}RowF_REPAIR_INDIRECT_X],atomicLoad(&${arena}[
  ${p}RowF_DIRTY_LEAF_COUNT]));atomicStore(&${arena}[${p}H_PHASE],${p}PhaseRepairingRows);}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyRowPlan(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseRepairingRows){return;}
 if(atomicLoad(&${arena}[${p}RowF_REPAIRED_LEAF_COUNT])!=atomicLoad(&${arena}[
  ${p}RowF_DIRTY_LEAF_COUNT])){${p}Fail(${p}RowFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.rowRepairGap}u,${p}Invalid);return;}
 let root=atomicLoad(&${arena}[${p}RowTreeBase(${layout.row.treeLevelCounts.length - 1}u)]);
 atomicStore(&${arena}[${p}H_CHANGED_ROW_COUNT],root);atomicStore(&${arena}[${p}RowF_WORK_COUNT],root);
 atomicStore(&${arena}[${p}RowF_WORK_INDIRECT_X],(root+63u)/64u);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseExecutingRows);}
@compute @workgroup_size(64) fn repairSparseCM12PressureTopologyRows(
 @builtin(global_invocation_id)gid:vec3u){if(atomicLoad(&${arena}[${p}H_PHASE])
 !=${p}PhaseExecutingRows){return;}let row=ptrChangedRowInvocation(gid.x);
 if(row==${p}Invalid){return;}let enabled=ptrClassifyPressureRow(row);
 _=pcmRowSetCandidate(row,enabled,1u,false);_=pcfStoreThetaAndRecord(row,ptrPressureTheta(row));
 let generation=atomicLoad(&${arena}[${p}H_CANDIDATE_GENERATION]);
 atomicStore(&${arena}[${p}RowExecution+row],generation);
 atomicAdd(&${arena}[${p}H_ROW_EXECUTION_COUNT],1u);atomicAdd(&${arena}[
 ${p}RowF_EXECUTED_COUNT],1u);}
@compute @workgroup_size(1) fn finalizeSparseCM12PressureTopologyRowExecution(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseExecutingRows){return;}
 if(atomicLoad(&${arena}[${p}H_ROW_EXECUTION_COUNT])!=atomicLoad(&${arena}[
  ${p}H_CHANGED_ROW_COUNT])){${p}Fail(${p}RowFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.rowExecutionGap}u,${p}Invalid);return;}
 if(atomicLoad(&${arena}[${p}H_TOPOLOGY_GENERATION])!=ptrTopologyGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.topologyGenerationGap}u,
  ${p}Invalid);return;}
 if(atomicLoad(&${arena}[${p}H_PCM_CELL_GENERATION])!=ptrPCMCellAcceptedGeneration()
  ||atomicLoad(&${arena}[${p}H_PCM_ROW_GENERATION])!=ptrPCMRowAcceptedGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.pcmGenerationGap}u,
  ${p}Invalid);return;}
 // Fine/aggregate PCF repair consumes this topology closure after the row
 // frontier is complete.  At this seam the captured generation must still be
 // the live candidate; acceptance is checked again by the brick-state commit
 // after PCF has transactionally published that candidate.
 if(atomicLoad(&${arena}[${p}H_PCF_GENERATION])!=ptrPCFCandidateGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.pcfGenerationGap}u,
  ${p}Invalid);return;}
 let count=atomicLoad(&${arena}[${p}H_CHANGED_BRICK_COUNT]);
 atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_X],(count+63u)/64u);
 atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_Y],1u);atomicStore(&${arena}[${p}H_COMMIT_INDIRECT_Z],1u);
 atomicStore(&${arena}[${p}H_PHASE],${p}PhaseAwaitingAcceptance);}
@compute @workgroup_size(64) fn commitSparseCM12PressureTopologyBrickStates(
 @builtin(global_invocation_id)gid:vec3u){if(atomicLoad(&${arena}[${p}H_PHASE])
 !=${p}PhaseAwaitingAcceptance){return;}let brick=ptrChangedBrickInvocation(gid.x);
 // Never publish PTR's persistent brick state unless the PCF generation that
 // consumed the same PCM/topology events has accepted.  This is a GPU-only
 // transaction boundary: a PCF failure leaves every PTR brick state intact.
 if(atomicLoad(&${arena}[${p}H_PCF_GENERATION])!=ptrPCFAcceptedGeneration()){return;}
 if(brick==${p}Invalid){return;}atomicStore(&${arena}[${p}BrickAcceptedState+brick],
 atomicLoad(&${arena}[${p}BrickNewState+brick]));atomicAdd(&${arena}[${p}H_BRICK_STATE_COMMIT_COUNT],1u);}
@compute @workgroup_size(1) fn finalizeSparseCM12BoundedPressureTopologyRepair(){
 if(atomicLoad(&${arena}[${p}H_PHASE])!=${p}PhaseAwaitingAcceptance){return;}
 if(atomicLoad(&${arena}[${p}H_PCF_GENERATION])!=ptrPCFAcceptedGeneration()){
  ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.pcfGenerationGap}u,
  ${p}Invalid);return;}
 if(atomicLoad(&${arena}[${p}H_BRICK_STATE_COMMIT_COUNT])!=atomicLoad(&${arena}[
 ${p}H_CHANGED_BRICK_COUNT])){${p}Fail(${p}BrickFamily,
  ${SPARSE_CM12_PRESSURE_TOPOLOGY_REPAIR_FAULT.cellExecutionGap}u,${p}Invalid);return;}
 atomicStore(&${arena}[${p}H_ACCEPTED_CHANGED_BRICK_COUNT],atomicLoad(&${arena}[
  ${p}H_CHANGED_BRICK_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_CHANGED_ROW_COUNT],atomicLoad(&${arena}[
  ${p}H_CHANGED_ROW_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_CELL_EXECUTION_COUNT],atomicLoad(&${arena}[
  ${p}H_CELL_EXECUTION_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_ROW_EXECUTION_COUNT],atomicLoad(&${arena}[
  ${p}H_ROW_EXECUTION_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_BRICK_DIRTY_LEAF_COUNT],atomicLoad(&${arena}[
  ${p}BrickF_DIRTY_LEAF_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_ROW_DIRTY_LEAF_COUNT],atomicLoad(&${arena}[
  ${p}RowF_DIRTY_LEAF_COUNT]));
 atomicStore(&${arena}[${p}H_ACCEPTED_GENERATION],atomicLoad(&${arena}[
  ${p}H_CANDIDATE_GENERATION]));${p}ZeroIndirects();atomicStore(&${arena}[
  ${p}H_PHASE],${p}PhaseAccepted);}
${preflightedTopologyPublication}
`;
}
