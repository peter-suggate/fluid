import {
  SPARSE_CM12_TWA_CHANGE,
  SPARSE_CM12_TWA_CHANGE_WORDS,
  SPARSE_CM12_TWA_FAMILY,
  SPARSE_CM12_TWA_FAMILY_COUNT,
  SPARSE_CM12_TWA_FAMILY_WORDS,
  SPARSE_CM12_TWA_FAULT,
  SPARSE_CM12_TWA_HEADER,
  SPARSE_CM12_TWA_HEADER_WORDS,
  SPARSE_CM12_TWA_INDIRECT_FAMILY_COUNT,
  SPARSE_CM12_TWA_MAGIC,
  SPARSE_CM12_TWA_PLANNER_FAMILY,
  SPARSE_CM12_TWA_PHASE,
  SPARSE_CM12_TWA_VERSION,
  type SparseCM12TWALayout,
} from "./sparse-cm12-topology-work-authority";

/**
 * Standalone TWA1 module. `twaCellIncidenceRange` and
 * `twaCellIncidenceRow` are supplied by the compiled immutable HTP topology.
 */
export function createSparseCM12TopologyWorkAuthorityWGSL(options: {
  readonly layout: SparseCM12TWALayout;
  readonly constructionMode?: "temporal" | "immutable-full-oracle";
}): string {
  const l = options.layout; const h = SPARSE_CM12_TWA_HEADER;
  const c = SPARSE_CM12_TWA_CHANGE; const f = SPARSE_CM12_TWA_FAMILY;
  const pf = SPARSE_CM12_TWA_PLANNER_FAMILY;
  const oracle = options.constructionMode === "immutable-full-oracle" ? "true" : "false";
  return /* wgsl */`
const twaInvalid:u32=0xffffffffu;
const twaHeader:u32=${l.headerBaseWords}u;
const twaBrickCapacity:u32=${l.brickCapacity}u;
const twaCellCapacity:u32=${l.cellCapacity}u;
const twaRowCapacity:u32=${l.rowCapacity}u;
const twaFamilyCount:u32=${SPARSE_CM12_TWA_FAMILY_COUNT}u;
const twaIndirectFamilyCount:u32=${SPARSE_CM12_TWA_INDIRECT_FAMILY_COUNT}u;
const twaBrickTreeLeaves:u32=${l.brickTreeLeafCapacity}u;
const twaCellTreeLeaves:u32=${l.cellTreeLeafCapacity}u;
const twaRowTreeLeaves:u32=${l.rowTreeLeafCapacity}u;
fn twaLoad(at:u32)->u32{return atomicLoad(&twaArena[at]);}
fn twaStore(at:u32,value:u32){atomicStore(&twaArena[at],value);}
fn twaValid()->bool{return arrayLength(&twaArena)>=${l.totalWords}u
  &&twaLoad(twaHeader+${h.magic}u)==0x${SPARSE_CM12_TWA_MAGIC.toString(16)}u
  &&twaLoad(twaHeader+${h.version}u)==${SPARSE_CM12_TWA_VERSION}u
  &&twaLoad(twaHeader+${h.headerWords}u)==${SPARSE_CM12_TWA_HEADER_WORDS}u
  &&twaLoad(twaHeader+${h.brickCapacity}u)==twaBrickCapacity
  &&twaLoad(twaHeader+${h.cellCapacity}u)==twaCellCapacity
  &&twaLoad(twaHeader+${h.rowCapacity}u)==twaRowCapacity;}
fn twaIndirect(family:u32,items:u32){let at=3u*family;
  atomicStore(&twaIndirectArgs[at],(items+63u)/64u);
  atomicStore(&twaIndirectArgs[at+1u],1u);atomicStore(&twaIndirectArgs[at+2u],1u);}
fn twaZeroIndirect(){for(var family=0u;family<twaIndirectFamilyCount;family+=1u){
  twaIndirect(family,0u);}}
// Failure seals the arena immediately. Worker entry points can therefore use the
// authority without binding the indirect buffer as storage; every physical
// kernel must call twaRecordFamilyPacket before its first topology write.
fn twaFail(code:u32,owner:u32)->bool{
  twaStore(twaHeader+${h.phase}u,${SPARSE_CM12_TWA_PHASE.fault}u);
  let claimed=atomicCompareExchangeWeak(&twaArena[twaHeader+${h.fault}u],0u,code);
  if(claimed.exchanged){twaStore(twaHeader+${h.firstFaultId}u,owner);}return false;}
fn twaTreeSet(base:u32,leaves:u32,owner:u32,value:u32){
  var node=leaves-1u+owner;let prior=atomicExchange(&twaArena[base+node],value);
  if(prior==value){return;}let add=value>prior;loop{if(node==0u){break;}
    node=(node-1u)/2u;if(add){atomicAdd(&twaArena[base+node],1u);}
    else{atomicSub(&twaArena[base+node],1u);}}
}
fn twaRankSelect(base:u32,leaves:u32,capacity:u32,rankInput:u32)->u32{
  if(rankInput>=twaLoad(base)){return twaInvalid;}var rank=rankInput;var node=0u;
  loop{if(node>=leaves-1u){break;}let left=2u*node+1u;let leftCount=twaLoad(base+left);
    if(rank<leftCount){node=left;}else{rank-=leftCount;node=left+1u;}}
  let owner=node-(leaves-1u);return select(owner,twaInvalid,owner>=capacity);
}
fn twaMarkDomain(stampBase:u32,treeBase:u32,leaves:u32,capacity:u32,
 owner:u32,generation:u32)->bool{if(owner>=capacity){return twaFail(
  ${SPARSE_CM12_TWA_FAULT.capacity}u,owner);}
  if(atomicExchange(&twaArena[stampBase+owner],generation)!=generation){
    twaTreeSet(treeBase,leaves,owner,1u);}return true;}
fn twaMarkBrick(owner:u32,generation:u32)->bool{if(owner>=twaBrickCapacity){
  return twaFail(${SPARSE_CM12_TWA_FAULT.capacity}u,owner);}
  if(atomicExchange(&twaArena[${l.brickStampBaseWords}u+2u*owner],generation)!=generation){
    twaTreeSet(${l.brickTreeBaseWords}u,twaBrickTreeLeaves,owner,1u);}return true;}
fn twaMarkCell(owner:u32,generation:u32)->bool{return twaMarkDomain(
  ${l.cellStampBaseWords}u,${l.cellTreeBaseWords}u,twaCellTreeLeaves,
  twaCellCapacity,owner,generation);}
fn twaMarkRow(owner:u32,generation:u32)->bool{return twaMarkDomain(
  ${l.rowStampBaseWords}u,${l.rowTreeBaseWords}u,twaRowTreeLeaves,
  twaRowCapacity,owner,generation);}
fn twaChangeAt(brick:u32)->u32{return ${l.changeBaseWords}u
  +${SPARSE_CM12_TWA_CHANGE_WORDS}u*brick;}
fn twaAppendChangedBrick(brick:u32,generation:u32,cause:u32,
 oldCells:vec2u,newCells:vec2u,oldRows:vec2u,newRows:vec2u)->bool{
  if(twaLoad(twaHeader+${h.phase}u)!=${SPARSE_CM12_TWA_PHASE.collecting}u
    ||brick>=twaBrickCapacity||generation!=twaLoad(twaHeader+${h.candidateGeneration}u)){
    return twaFail(${SPARSE_CM12_TWA_FAULT.brick}u,brick);}
  let at=twaChangeAt(brick);let prior=atomicExchange(&twaArena[
    ${l.brickStampBaseWords}u+2u*brick],generation);
  if(prior==generation){atomicOr(&twaArena[${l.brickStampBaseWords}u+2u*brick+1u],cause);
    return true;}
  twaStore(at+${c.brick}u,brick);twaStore(at+${c.generation}u,generation);
  twaStore(at+${c.causeMask}u,cause);twaStore(at+${c.oldCellFirst}u,oldCells.x);
  twaStore(at+${c.oldCellCount}u,oldCells.y);twaStore(at+${c.newCellFirst}u,newCells.x);
  twaStore(at+${c.newCellCount}u,newCells.y);twaStore(at+${c.oldRowFirst}u,oldRows.x);
  twaStore(at+${c.oldRowCount}u,oldRows.y);twaStore(at+${c.newRowFirst}u,newRows.x);
  twaStore(at+${c.newRowCount}u,newRows.y);
  twaTreeSet(${l.brickTreeBaseWords}u,twaBrickTreeLeaves,brick,1u);
  atomicAdd(&twaArena[twaHeader+${h.changeCount}u],1u);
  atomicOr(&twaArena[twaHeader+${h.causeMask}u],cause);return true;
}
@compute @workgroup_size(1) fn prepareTWABegin(){
  if(!twaValid()){_=twaFail(${SPARSE_CM12_TWA_FAULT.header}u,twaInvalid);return;}
  let phase=twaLoad(twaHeader+${h.phase}u);if(phase!=${SPARSE_CM12_TWA_PHASE.accepted}u
    &&phase!=${SPARSE_CM12_TWA_PHASE.rejected}u){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.phase}u,twaInvalid);return;}
  let generation=twaLoad(twaHeader+${h.candidateGeneration}u);
  let accepted=twaLoad(twaHeader+${h.acceptedGeneration}u);
  if(${oracle}&&accepted!=0u){_=twaFail(${SPARSE_CM12_TWA_FAULT.runtimeOracle}u,
    twaInvalid);return;}
  if(generation==0u||(accepted!=0u&&generation!=accepted+1u)
    ||twaLoad(twaHeader+${h.topologyGeneration}u)==0u){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.generation}u,twaInvalid);return;}
  twaIndirect(${pf.clearPriorBricks}u,twaLoad(twaHeader+${h.brickCount}u));
  twaIndirect(${pf.clearPriorCells}u,twaLoad(twaHeader+${h.cellCount}u));
  twaIndirect(${pf.clearPriorRows}u,twaLoad(twaHeader+${h.rowCount}u));
}
@compute @workgroup_size(64) fn clearTWAPriorBricks(
 @builtin(global_invocation_id)gid:vec3u){let rank=gid.x;
  if(rank>=twaLoad(twaHeader+${h.brickCount}u)){return;}
  let owner=twaLoad(${l.brickListBaseWords}u+rank);if(owner>=twaBrickCapacity){return;}
  twaTreeSet(${l.brickTreeBaseWords}u,twaBrickTreeLeaves,owner,0u);
  twaStore(${l.brickStampBaseWords}u+2u*owner,0u);
  twaStore(${l.brickStampBaseWords}u+2u*owner+1u,0u);
}
@compute @workgroup_size(64) fn clearTWAPriorCells(
 @builtin(global_invocation_id)gid:vec3u){let rank=gid.x;
  if(rank>=twaLoad(twaHeader+${h.cellCount}u)){return;}
  let owner=twaLoad(${l.cellListBaseWords}u+rank);if(owner>=twaCellCapacity){return;}
  twaTreeSet(${l.cellTreeBaseWords}u,twaCellTreeLeaves,owner,0u);
  twaStore(${l.cellStampBaseWords}u+owner,0u);
}
@compute @workgroup_size(64) fn clearTWAPriorRows(
 @builtin(global_invocation_id)gid:vec3u){let rank=gid.x;
  if(rank>=twaLoad(twaHeader+${h.rowCount}u)){return;}
  let owner=twaLoad(${l.rowListBaseWords}u+rank);if(owner>=twaRowCapacity){return;}
  twaTreeSet(${l.rowTreeBaseWords}u,twaRowTreeLeaves,owner,0u);
  twaStore(${l.rowStampBaseWords}u+owner,0u);
}
@compute @workgroup_size(1) fn finishTWABegin(){
  if(twaLoad(${l.brickTreeBaseWords}u)!=0u||twaLoad(${l.cellTreeBaseWords}u)!=0u
    ||twaLoad(${l.rowTreeBaseWords}u)!=0u){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.treeRoot}u,twaInvalid);return;}
  twaStore(twaHeader+${h.phase}u,${SPARSE_CM12_TWA_PHASE.collecting}u);
  twaStore(twaHeader+${h.fault}u,0u);twaStore(twaHeader+${h.firstFaultId}u,twaInvalid);
  twaStore(twaHeader+${h.changeCount}u,0u);twaStore(twaHeader+${h.causeMask}u,0u);
  twaStore(twaHeader+${h.expectedReceipts}u,0u);twaStore(twaHeader+${h.coveredReceipts}u,0u);
  twaStore(twaHeader+${h.uncoveredWriteCount}u,0u);
  twaStore(twaHeader+${h.firstUncoveredOwner}u,twaInvalid);twaZeroIndirect();
}
@compute @workgroup_size(1) fn prepareTWAExpansion(){
  if(twaLoad(twaHeader+${h.phase}u)!=${SPARSE_CM12_TWA_PHASE.collecting}u){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.phase}u,twaInvalid);return;}
  twaIndirect(${pf.expandChangedBricks}u,twaLoad(${l.brickTreeBaseWords}u));
}
@compute @workgroup_size(64) fn expandTWAChangedBricks(
 @builtin(global_invocation_id)gid:vec3u){let rank=gid.x;
  let brick=twaRankSelect(${l.brickTreeBaseWords}u,twaBrickTreeLeaves,
    twaBrickCapacity,rank);if(brick==twaInvalid){return;}let at=twaChangeAt(brick);
  let generation=twaLoad(twaHeader+${h.candidateGeneration}u);
  let ranges=array<vec2u,2>(vec2u(twaLoad(at+${c.oldCellFirst}u),
    twaLoad(at+${c.oldCellCount}u)),vec2u(twaLoad(at+${c.newCellFirst}u),
    twaLoad(at+${c.newCellCount}u)));
  for(var side=0u;side<2u;side+=1u){for(var offset=0u;offset<ranges[side].y;offset+=1u){
    let cell=ranges[side].x+offset;if(cell>=twaCellCapacity){
      _=twaFail(${SPARSE_CM12_TWA_FAULT.capacity}u,cell);return;}_=twaMarkCell(cell,generation);
    let incidence=twaCellIncidenceRange(cell);for(var i=0u;i<incidence.y;i+=1u){
      _=twaMarkRow(twaCellIncidenceRow(incidence.x+i),generation);}
  }}
  let rowRanges=array<vec2u,2>(vec2u(twaLoad(at+${c.oldRowFirst}u),
    twaLoad(at+${c.oldRowCount}u)),vec2u(twaLoad(at+${c.newRowFirst}u),
    twaLoad(at+${c.newRowCount}u)));
  for(var side=0u;side<2u;side+=1u){for(var offset=0u;offset<rowRanges[side].y;offset+=1u){
    _=twaMarkRow(rowRanges[side].x+offset,generation);}}
}
@compute @workgroup_size(1) fn prepareTWAFinalize(){
  if(twaLoad(twaHeader+${h.phase}u)!=${SPARSE_CM12_TWA_PHASE.collecting}u){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.phase}u,twaInvalid);return;}
  let bricks=twaLoad(${l.brickTreeBaseWords}u);let cells=twaLoad(${l.cellTreeBaseWords}u);
  let rows=twaLoad(${l.rowTreeBaseWords}u);twaStore(twaHeader+${h.brickCount}u,bricks);
  twaStore(twaHeader+${h.cellCount}u,cells);twaStore(twaHeader+${h.rowCount}u,rows);
  twaIndirect(${pf.writeBrickList}u,bricks);twaIndirect(${pf.writeCellList}u,cells);
  twaIndirect(${pf.writeRowList}u,rows);
}
@compute @workgroup_size(64) fn writeTWABrickList(
 @builtin(global_invocation_id)gid:vec3u){let rank=gid.x;
  let owner=twaRankSelect(${l.brickTreeBaseWords}u,twaBrickTreeLeaves,
    twaBrickCapacity,rank);if(owner!=twaInvalid){twaStore(${l.brickListBaseWords}u+rank,owner);}}
@compute @workgroup_size(64) fn writeTWACellList(
 @builtin(global_invocation_id)gid:vec3u){let rank=gid.x;
  let owner=twaRankSelect(${l.cellTreeBaseWords}u,twaCellTreeLeaves,
    twaCellCapacity,rank);if(owner!=twaInvalid){twaStore(${l.cellListBaseWords}u+rank,owner);}}
@compute @workgroup_size(64) fn writeTWARowList(
 @builtin(global_invocation_id)gid:vec3u){let rank=gid.x;
  let owner=twaRankSelect(${l.rowTreeBaseWords}u,twaRowTreeLeaves,
    twaRowCapacity,rank);if(owner!=twaInvalid){twaStore(${l.rowListBaseWords}u+rank,owner);}}
@compute @workgroup_size(1) fn finishTWAFinalize(){
  let bricks=twaLoad(twaHeader+${h.brickCount}u);
  let cells=twaLoad(twaHeader+${h.cellCount}u);let rows=twaLoad(twaHeader+${h.rowCount}u);
  let counts=array<u32,${SPARSE_CM12_TWA_FAMILY_COUNT}>(bricks,bricks,cells,rows,
    bricks,bricks,6u*bricks,bricks,rows,bricks,bricks);
  var receipts=0u;for(var family=0u;family<twaFamilyCount;family+=1u){
    let descriptor=${l.familyBaseWords}u+${SPARSE_CM12_TWA_FAMILY_WORDS}u*family;
    twaStore(descriptor,counts[family]);twaStore(descriptor+1u,0u);
    twaStore(descriptor+2u,0u);
    twaIndirect(family,counts[family]);receipts+=select(0u,1u,counts[family]>0u);}
  twaStore(twaHeader+${h.expectedReceipts}u,receipts);
  twaStore(twaHeader+${h.maximumClosureDepth}u,select(0u,1u,rows>0u));
  twaStore(twaHeader+${h.phase}u,${SPARSE_CM12_TWA_PHASE.sealed}u);
}
fn twaFamilyOwner(family:u32,rank:u32)->u32{
  if(family==${f.buildShadowCells}u){return twaRankSelect(${l.cellTreeBaseWords}u,
    twaCellTreeLeaves,twaCellCapacity,rank);}
  if(family==${f.buildShadowRows}u||family==${f.reconstructShadowFacesRows}u){
    return twaRankSelect(${l.rowTreeBaseWords}u,twaRowTreeLeaves,twaRowCapacity,rank);}
  if(family==${f.transferFaces}u){let brick=twaRankSelect(${l.brickTreeBaseWords}u,
    twaBrickTreeLeaves,twaBrickCapacity,rank/6u);return select(twaInvalid,
      6u*brick+rank%6u,brick!=twaInvalid);}
  return twaRankSelect(${l.brickTreeBaseWords}u,twaBrickTreeLeaves,
    twaBrickCapacity,rank);
}
fn twaRecordFamilyPacket(family:u32,rank:u32,owner:u32)->bool{
  if(family>=twaFamilyCount||twaLoad(twaHeader+${h.phase}u)
    !=${SPARSE_CM12_TWA_PHASE.sealed}u||owner!=twaFamilyOwner(family,rank)){
    return twaFail(${SPARSE_CM12_TWA_FAULT.missingExecution}u,owner);}
  let descriptor=${l.familyBaseWords}u+${SPARSE_CM12_TWA_FAMILY_WORDS}u*family;
  atomicAdd(&twaArena[descriptor+1u],1u);return true;
}
fn twaPublishFamilyExecution(family:u32)->bool{
  if(family>=twaFamilyCount||twaLoad(twaHeader+${h.phase}u)
    !=${SPARSE_CM12_TWA_PHASE.sealed}u){return false;}
  let descriptor=${l.familyBaseWords}u+${SPARSE_CM12_TWA_FAMILY_WORDS}u*family;
  let executed=twaLoad(descriptor+1u);
  if(executed!=twaLoad(descriptor)||atomicExchange(&twaArena[descriptor+2u],1u)!=0u){
    return twaFail(${SPARSE_CM12_TWA_FAULT.missingExecution}u,family);}
  if(executed>0u){atomicAdd(&twaArena[twaHeader+${h.coveredReceipts}u],1u);}return true;
}
fn twaRecordUncoveredWrite(owner:u32){let prior=atomicAdd(&twaArena[
  twaHeader+${h.uncoveredWriteCount}u],1u);if(prior==0u){
  twaStore(twaHeader+${h.firstUncoveredOwner}u,owner);}}
@compute @workgroup_size(1) fn commitTWA(){
  if(twaLoad(twaHeader+${h.phase}u)!=${SPARSE_CM12_TWA_PHASE.sealed}u){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.phase}u,twaInvalid);return;}
  for(var family=0u;family<twaFamilyCount;family+=1u){let descriptor=
    ${l.familyBaseWords}u+${SPARSE_CM12_TWA_FAMILY_WORDS}u*family;
    if(twaLoad(descriptor)!=twaLoad(descriptor+1u)){
      _=twaFail(${SPARSE_CM12_TWA_FAULT.missingExecution}u,family);return;}}
  if(twaLoad(twaHeader+${h.expectedReceipts}u)!=twaLoad(twaHeader+${h.coveredReceipts}u)){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.missingExecution}u,twaInvalid);return;}
  if(twaLoad(twaHeader+${h.uncoveredWriteCount}u)!=0u){_=twaFail(
    ${SPARSE_CM12_TWA_FAULT.uncoveredWrite}u,
    twaLoad(twaHeader+${h.firstUncoveredOwner}u));return;}
  twaStore(twaHeader+${h.acceptedGeneration}u,
    twaLoad(twaHeader+${h.candidateGeneration}u));
  atomicAdd(&twaArena[twaHeader+${h.acceptedFrames}u],1u);
  twaStore(twaHeader+${h.phase}u,${SPARSE_CM12_TWA_PHASE.accepted}u);
}
@compute @workgroup_size(1) fn seedTWAConstructionOracle(){
  if(!${oracle}){return;}if(twaLoad(twaHeader+${h.acceptedGeneration}u)!=0u){
    _=twaFail(${SPARSE_CM12_TWA_FAULT.runtimeOracle}u,twaInvalid);return;}
  let generation=twaLoad(twaHeader+${h.candidateGeneration}u);
  for(var brick=0u;brick<twaBrickCapacity;brick+=1u){_=twaMarkBrick(brick,generation);}
  for(var cell=0u;cell<twaCellCapacity;cell+=1u){_=twaMarkCell(cell,generation);}
  for(var row=0u;row<twaRowCapacity;row+=1u){_=twaMarkRow(row,generation);}
  twaStore(twaHeader+${h.changeCount}u,twaBrickCapacity);
  twaStore(twaHeader+${h.causeMask}u,32u);
}
`;
}

export const SPARSE_CM12_TWA_ENTRY_POINTS = Object.freeze([
  "prepareTWABegin", "clearTWAPriorBricks", "clearTWAPriorCells",
  "clearTWAPriorRows", "finishTWABegin", "prepareTWAExpansion",
  "expandTWAChangedBricks", "prepareTWAFinalize", "writeTWABrickList",
  "writeTWACellList", "writeTWARowList", "finishTWAFinalize", "commitTWA",
  "seedTWAConstructionOracle",
] as const);
