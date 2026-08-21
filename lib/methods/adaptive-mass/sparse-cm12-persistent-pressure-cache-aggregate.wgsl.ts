import {
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_BRANCH,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_MAGIC,
  SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS,
  SPARSE_CM12_PRESSURE_CACHE_PHASE,
  type SparseCM12PersistentPressureCacheLayout,
  type SparseCM12PressureCacheAggregateFamilyLayout,
  type SparseCM12PressureCacheAggregateFamilyName,
} from "./sparse-cm12-persistent-pressure-cache";

const FAMILY_NAMES = ["brick", "aggregateEdge", "hierarchyNode", "hierarchyEdge"] as const;
const label = (name: SparseCM12PressureCacheAggregateFamilyName) => ({
  brick: "Brick", aggregateEdge: "AggregateEdge",
  hierarchyNode: "HierarchyNode", hierarchyEdge: "HierarchyEdge",
})[name];

function constants(name: SparseCM12PressureCacheAggregateFamilyName,
  family: SparseCM12PressureCacheAggregateFamilyLayout): string {
  const p = `PCFA_${label(name).toUpperCase()}`;
  return `const ${p}_HEADER:u32=${family.headerBaseWords}u;
const ${p}_CAPACITY:u32=${family.capacity}u;const ${p}_CANDIDATE:u32=${family.candidateGenerationBaseWords}u;
const ${p}_BITS:u32=${family.bitsBaseWords}u;const ${p}_STAMP:u32=${family.dirtyLeafStampBaseWords}u;
const ${p}_LIST:u32=${family.dirtyLeafListBaseWords}u;const ${p}_ACTIVE_LEAVES:u32=${family.activeLeafListBaseWords}u;
const ${p}_LEAF_COUNT:u32=${family.leafCount}u;
${family.treeLevelBaseWords.map((base, level) =>
    `const ${p}_TREE_${level}:u32=${base}u;`).join("\n")}`;
}

function rankSelect(name: SparseCM12PressureCacheAggregateFamilyName,
  family: SparseCM12PressureCacheAggregateFamilyLayout, arena: string): string {
  const n = label(name), p = `PCFA_${n.toUpperCase()}`;
  const descend = Array.from({ length: family.treeLevelCounts.length - 1 }, (_, index) => {
    const level = family.treeLevelCounts.length - 2 - index;
    return `{let begin=node*PCFA_BRANCH;var selected=PCF_INVALID;
    for(var child=0u;child<PCFA_BRANCH;child+=1u){let candidate=begin+child;
      if(candidate>=${family.treeLevelCounts[level]}u){break;}
      let count=atomicLoad(&${arena}[${p}_TREE_${level}+candidate]);
      if(selected==PCF_INVALID&&remaining<count){selected=candidate;}
      else if(selected==PCF_INVALID){remaining-=count;}}
    if(selected==PCF_INVALID){return PCF_INVALID;}node=selected;}`;
  }).join("\n  ");
  return `fn pcf${n}RankSelect(rank:u32)->u32{
  let total=atomicLoad(&${arena}[${p}_HEADER+PCFA_F_WORK_COUNT]);
  if(rank>=total){return PCF_INVALID;}var remaining=rank;var node=0u;
  ${descend}
  let firstWord=node*PCFA_LEAF_WORDS;
  for(var wordAt=0u;wordAt<PCFA_LEAF_WORDS;wordAt+=1u){
    let word=atomicLoad(&${arena}[${p}_BITS+firstWord+wordAt]);let count=countOneBits(word);
    if(remaining>=count){remaining-=count;continue;}
    for(var bit=0u;bit<32u;bit+=1u){if((word&(1u<<bit))==0u){continue;}
      if(remaining==0u){let id=node*PCFA_LEAF_BITS+32u*wordAt+bit;
        return select(PCF_INVALID,id,id<${p}_CAPACITY);}remaining-=1u;}}
  return PCF_INVALID;}`;
}

function repairWorkset(name: SparseCM12PressureCacheAggregateFamilyName,
  family: SparseCM12PressureCacheAggregateFamilyLayout, arena: string): string {
  const n = label(name), p = `PCFA_${n.toUpperCase()}`;
  const familyIndex = FAMILY_NAMES.indexOf(name);
  const updates = family.treeLevelBaseWords.slice(1).map((base, index) =>
    `node/=PCFA_BRANCH;let prior${index}=atomicAdd(&${arena}[${base}u+node],bitcast<u32>(delta));
    if((delta<0&&prior${index}<u32(-delta))||(delta>0&&prior${index}>0xffffffffu-u32(delta))){
      pcfaFault(${familyIndex}u,PCF_FAULT_REPAIR,leaf*PCFA_LEAF_BITS);}`)
    .join("");
  return `@compute @workgroup_size(64)
fn repairPersistentPressure${n}Workset(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let dirty=atomicLoad(&${arena}[${p}_HEADER+PCFA_F_DIRTY_LEAVES]);
  let valid=wid.x<dirty&&atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])==0u;
  var leaf=0u;if(valid){leaf=atomicLoad(&${arena}[${p}_LIST+wid.x]);}
  if(lane<PCFA_LEAF_WORDS){var word=0u;if(valid&&leaf<${p}_LEAF_COUNT){
    let generation=atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]);
    for(var bit=0u;bit<32u;bit+=1u){let id=leaf*PCFA_LEAF_BITS+32u*lane+bit;
      if(id<${p}_CAPACITY&&atomicLoad(&${arena}[${p}_CANDIDATE+id])==generation){word|=1u<<bit;}}
    atomicStore(&${arena}[${p}_BITS+leaf*PCFA_LEAF_WORDS+lane],word);}
    pcfaLeafCounts[lane]=countOneBits(word);}workgroupBarrier();
  if(lane==0u&&valid&&leaf<${p}_LEAF_COUNT){var count=0u;
    for(var at=0u;at<PCFA_LEAF_WORDS;at+=1u){count+=pcfaLeafCounts[at];}
    let previous=atomicExchange(&${arena}[${p}_TREE_0+leaf],count);
    let delta=i32(count)-i32(previous);var node=leaf;${updates}
    if(count>0u){let slot=atomicAdd(&${arena}[${p}_HEADER+PCFA_F_ACTIVE_LEAVES],1u);
      if(slot>=${p}_LEAF_COUNT){pcfaFault(${familyIndex}u,PCF_FAULT_CAPACITY,leaf*PCFA_LEAF_BITS);}
      else{atomicStore(&${arena}[${p}_ACTIVE_LEAVES+slot],leaf);}}
    atomicAdd(&${arena}[${p}_HEADER+PCFA_F_REPAIRED_LEAVES],1u);}}
`;
}

export function createSparseCM12PersistentPressureCacheAggregateWGSL(
  layout: SparseCM12PersistentPressureCacheLayout,
  arena: string,
): string {
  const a = layout.aggregateHeaderBaseWords;
  const ah = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER;
  const fh = SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER;
  const familySelect = (field: keyof SparseCM12PressureCacheAggregateFamilyLayout) => {
    const values = FAMILY_NAMES.map((name) => layout.aggregateFamilies[name][field]);
    return `if(family==0u){return ${values[0]}u;}if(family==1u){return ${values[1]}u;}`
      + `if(family==2u){return ${values[2]}u;}return ${values[3]}u;`;
  };
  const levelAddress = layout.hierarchyLevelCounts.map((count, level) => {
    const offset = layout.hierarchyLevelOffsets[level]!;
    return `if(id<${offset + count}u){return vec2u(${level}u,id-${offset}u);}`;
  }).join("");
  const edgeAddress = layout.hierarchyEdgeLevelCounts.map((count, level) => {
    const offset = layout.hierarchyEdgeLevelOffsets[level]!;
    return `if(id<${offset + count}u){return vec2u(${level}u,id-${offset}u);}`;
  }).join("");
  const nodeLinear = layout.hierarchyLevelOffsets.map((offset, level) =>
    `if(level==${level}u){return ${offset}u+group;}`).join("");
  const edgeLinear = layout.hierarchyEdgeLevelOffsets.map((offset, level) =>
    `if(level==${level}u){return ${offset}u+edge;}`).join("");
  const hierarchyEdgeOffset = layout.hierarchyEdgeBaseWords.map((offset, level) =>
    `if(level==${level}u){return ${offset}u+edge;}`).join("");
  const hierarchyDiagonalOffset = layout.hierarchyDiagonalBaseWords.map((offset, level) =>
    `if(level==${level}u){return ${offset}u+group;}`).join("");
  const roots = Object.fromEntries(FAMILY_NAMES.map((name) => [name,
    layout.aggregateFamilies[name].treeLevelBaseWords.at(-1)!])) as
    Record<SparseCM12PressureCacheAggregateFamilyName, number>;
  return /* wgsl */ `
// Aggregate integration hooks preserve the existing packed-list order:
// pcfCellBrick, pcfBrickCellRange
// pcfAggregateEdgeForFineEdge, pcfAggregateEdgeContributionRange/Contribution
// pcfHierarchyParent, pcfHierarchyEdgeForAggregate
// pcfHierarchyChildRange/Child, pcfHierarchyInternalEdgeRange/InternalEdge
// pcfHierarchyEdgeContributionRange/Contribution.
// pcfTopologyGeneration/pcfPCMGeneration/pcfAggregateTopologyGeneration()->u32.
const PCFA_MAGIC:u32=0x${SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_MAGIC.toString(16)}u;
const PCFA_BASE:u32=${a}u;const PCFA_LEAF_BITS:u32=${SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS}u;
const PCFA_LEAF_WORDS:u32=${SPARSE_CM12_PRESSURE_CACHE_LEAF_BITS / 32}u;
const PCFA_BRANCH:u32=${SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_BRANCH}u;
const PCFA_QA:bool=${layout.qaFullOracle ? "true" : "false"};
const PCFA_PHASE_AGGREGATE_REPAIR:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.aggregateRepairing}u;
const PCFA_PHASE_AGGREGATE_EXECUTE:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.aggregateExecuting}u;
const PCFA_PHASE_HIERARCHY_REPAIR:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.hierarchyRepairing}u;
const PCFA_PHASE_HIERARCHY_EXECUTE:u32=${SPARSE_CM12_PRESSURE_CACHE_PHASE.hierarchyExecuting}u;
const PCFA_F_DIRTY_LEAVES:u32=${fh.dirtyLeafCount}u;const PCFA_F_REPAIRED_LEAVES:u32=${fh.repairedLeafCount}u;
const PCFA_F_WORK_COUNT:u32=${fh.workCount}u;const PCFA_F_EXECUTED:u32=${fh.executedCount}u;
const PCFA_F_REPAIR_X:u32=${fh.repairIndirectX}u;const PCFA_F_WORK_X:u32=${fh.workIndirectX}u;
const PCFA_F_CAUSE:u32=${fh.causeMask}u;const PCFA_F_PREVIOUS_LEAVES:u32=${fh.previousActiveLeafCount}u;
const PCFA_F_ACTIVE_LEAVES:u32=${fh.activeLeafCount}u;const PCFA_F_SEED_X:u32=${fh.seedIndirectX}u;
${FAMILY_NAMES.map((name) => constants(name, layout.aggregateFamilies[name])).join("\n")}
var<workgroup>pcfaLeafCounts:array<u32,8>;
fn pcfaFamilyHeader(family:u32)->u32{${familySelect("headerBaseWords")}}
fn pcfaFamilyCapacity(family:u32)->u32{${familySelect("capacity")}}
fn pcfaFamilyCandidate(family:u32)->u32{${familySelect("candidateGenerationBaseWords")}}
fn pcfaFamilyStamp(family:u32)->u32{${familySelect("dirtyLeafStampBaseWords")}}
fn pcfaFamilyList(family:u32)->u32{${familySelect("dirtyLeafListBaseWords")}}
fn pcfaFamilyActiveLeaves(family:u32)->u32{${familySelect("activeLeafListBaseWords")}}
fn pcfaFamilyLeafCount(family:u32)->u32{${familySelect("leafCount")}}
fn pcfaNodeAddress(id:u32)->vec2u{${levelAddress}return vec2u(PCF_INVALID);}
fn pcfaEdgeAddress(id:u32)->vec2u{${edgeAddress}return vec2u(PCF_INVALID);}
fn pcfaNodeLinear(level:u32,group:u32)->u32{${nodeLinear}return PCF_INVALID;}
fn pcfaEdgeLinear(level:u32,edge:u32)->u32{${edgeLinear}return PCF_INVALID;}
fn pcfaHierarchyEdgeWord(level:u32,edge:u32)->u32{${hierarchyEdgeOffset}return PCF_INVALID;}
fn pcfaHierarchyDiagonalWord(level:u32,group:u32)->u32{${hierarchyDiagonalOffset}return PCF_INVALID;}
fn pcfaAggregateHeaderValid()->bool{return atomicLoad(&${arena}[PCFA_BASE+${ah.magic}u])==PCFA_MAGIC
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.headerWords}u])==${SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_HEADER_WORDS}u
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.familyHeaderWords}u])==${SPARSE_CM12_PRESSURE_CACHE_AGGREGATE_FAMILY_HEADER_WORDS}u
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.familyCount}u])==4u
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.brickCount}u])==${layout.brickCount}u
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.aggregateEdgeCount}u])==${layout.aggregateEdgeCount}u
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.hierarchyNodeCount}u])==${layout.hierarchyNodeCount}u
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.hierarchyEdgeCount}u])==${layout.hierarchyEdgeCount}u
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.brickHeaderBase}u])==PCFA_BRICK_HEADER
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.aggregateEdgeHeaderBase}u])==PCFA_AGGREGATEEDGE_HEADER
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.hierarchyNodeHeaderBase}u])==PCFA_HIERARCHYNODE_HEADER
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.hierarchyEdgeHeaderBase}u])==PCFA_HIERARCHYEDGE_HEADER
  &&atomicLoad(&${arena}[PCFA_BASE+${ah.qaFullOracle}u])==select(0u,1u,PCFA_QA);}
fn pcfaZeroIndirects(){for(var family=0u;family<4u;family+=1u){let header=pcfaFamilyHeader(family);
  atomicStore(&${arena}[header+PCFA_F_REPAIR_X],0u);atomicStore(&${arena}[header+PCFA_F_WORK_X],0u);
  atomicStore(&${arena}[header+PCFA_F_SEED_X],0u);}}
fn pcfaFault(family:u32,code:u32,id:u32){let claim=atomicCompareExchangeWeak(&${arena}[
  PCFA_BASE+${ah.firstFaultFamily}u],PCF_INVALID,family);if(claim.exchanged){
    atomicStore(&${arena}[PCFA_BASE+${ah.firstFaultId}u],id);}pcfFault(code,id);}
fn pcfaBegin()->bool{if(!pcfaAggregateHeaderValid()){pcfFault(PCF_FAULT_HEADER,PCF_INVALID);return false;}
  let aggregateTopology=pcfAggregateTopologyGeneration();
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])==PCF_PHASE_ACCEPTED
    &&atomicLoad(&${arena}[PCFA_BASE+${ah.acceptedAggregateTopologyGeneration}u])
      !=aggregateTopology){pcfFault(PCF_FAULT_TOPOLOGY,PCF_INVALID);return false;}
  atomicStore(&${arena}[PCFA_BASE+${ah.topologyGeneration}u],pcfTopologyGeneration());
  atomicStore(&${arena}[PCFA_BASE+${ah.pcmGeneration}u],pcfPCMGeneration());
  atomicStore(&${arena}[PCFA_BASE+${ah.aggregateTopologyGeneration}u],aggregateTopology);
  atomicStore(&${arena}[PCFA_BASE+${ah.firstFaultFamily}u],PCF_INVALID);
  atomicStore(&${arena}[PCFA_BASE+${ah.firstFaultId}u],PCF_INVALID);
  for(var family=0u;family<4u;family+=1u){let header=pcfaFamilyHeader(family);
    let previous=atomicLoad(&${arena}[header+PCFA_F_ACTIVE_LEAVES]);
    atomicStore(&${arena}[header+PCFA_F_PREVIOUS_LEAVES],previous);
    atomicStore(&${arena}[header+PCFA_F_ACTIVE_LEAVES],0u);
    atomicStore(&${arena}[header+PCFA_F_DIRTY_LEAVES],0u);
    atomicStore(&${arena}[header+PCFA_F_REPAIRED_LEAVES],0u);
    atomicStore(&${arena}[header+PCFA_F_WORK_COUNT],0u);atomicStore(&${arena}[header+PCFA_F_EXECUTED],0u);
    atomicStore(&${arena}[header+PCFA_F_CAUSE],0u);atomicStore(&${arena}[header+PCFA_F_REPAIR_X],0u);
    atomicStore(&${arena}[header+PCFA_F_WORK_X],0u);
    atomicStore(&${arena}[header+PCFA_F_SEED_X],(previous+63u)/64u);}return true;}
fn pcfCaptureConsumerGenerations(){
  atomicStore(&${arena}[PCFA_BASE+${ah.topologyGeneration}u],pcfTopologyGeneration());
  atomicStore(&${arena}[PCFA_BASE+${ah.pcmGeneration}u],pcfPCMGeneration());
  atomicStore(&${arena}[PCFA_BASE+${ah.aggregateTopologyGeneration}u],
    pcfAggregateTopologyGeneration());
}
fn pcfaQueueLeaf(family:u32,leaf:u32)->bool{if(leaf>=pcfaFamilyLeafCount(family)){
  pcfaFault(family,PCF_FAULT_CAPACITY,leaf*PCFA_LEAF_BITS);return false;}
  let generation=atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]);
  if(atomicExchange(&${arena}[pcfaFamilyStamp(family)+leaf],generation)==generation){return true;}
  let header=pcfaFamilyHeader(family);let slot=atomicAdd(&${arena}[header+PCFA_F_DIRTY_LEAVES],1u);
  if(slot>=pcfaFamilyLeafCount(family)){pcfaFault(family,PCF_FAULT_CAPACITY,leaf*PCFA_LEAF_BITS);return false;}
  atomicStore(&${arena}[pcfaFamilyList(family)+slot],leaf);return true;}
fn pcfaMark(family:u32,id:u32,cause:u32)->bool{if(id>=pcfaFamilyCapacity(family)){
  pcfaFault(family,PCF_FAULT_TOPOLOGY,id);return false;}
  let generation=atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]);
  atomicStore(&${arena}[pcfaFamilyCandidate(family)+id],generation);
  atomicOr(&${arena}[pcfaFamilyHeader(family)+PCFA_F_CAUSE],cause);
  return pcfaQueueLeaf(family,id/PCFA_LEAF_BITS);}
fn pcfaSeedPrevious(family:u32,invocation:u32){let header=pcfaFamilyHeader(family);
  let count=atomicLoad(&${arena}[header+PCFA_F_PREVIOUS_LEAVES]);if(invocation>=count){return;}
  let leaf=atomicLoad(&${arena}[pcfaFamilyActiveLeaves(family)+invocation]);_=pcfaQueueLeaf(family,leaf);}
@compute @workgroup_size(64) fn seedPreviousPCFBrickLeaves(@builtin(global_invocation_id)gid:vec3u){pcfaSeedPrevious(0u,gid.x);}
@compute @workgroup_size(64) fn seedPreviousPCFAggregateEdgeLeaves(@builtin(global_invocation_id)gid:vec3u){pcfaSeedPrevious(1u,gid.x);}
@compute @workgroup_size(64) fn seedPreviousPCFHierarchyNodeLeaves(@builtin(global_invocation_id)gid:vec3u){pcfaSeedPrevious(2u,gid.x);}
@compute @workgroup_size(64) fn seedPreviousPCFHierarchyEdgeLeaves(@builtin(global_invocation_id)gid:vec3u){pcfaSeedPrevious(3u,gid.x);}
fn pcfAggregateFineEdgeChanged(cell:u32,edge:u32){_=pcfaMark(0u,pcfCellBrick(cell),1u);
  let aggregateEdge=pcfAggregateEdgeForFineEdge(edge);
  if(aggregateEdge!=PCF_INVALID){_=pcfaMark(1u,aggregateEdge,1u);}}
fn pcfAggregateFineDiagonalChanged(cell:u32){_=pcfaMark(0u,pcfCellBrick(cell),2u);}
fn pcfaMarkBrickAncestors(brick:u32){for(var level=0u;level<${layout.hierarchyLevelCounts.length}u;level+=1u){
  _=pcfaMark(2u,pcfaNodeLinear(level,pcfHierarchyParent(level,brick)),4u);}}
fn pcfaMarkAggregateEdgeAncestors(edge:u32){for(var level=0u;level<${layout.hierarchyLevelCounts.length}u;level+=1u){
  let hierarchyEdge=pcfHierarchyEdgeForAggregate(level,edge);
  if(hierarchyEdge!=PCF_INVALID){_=pcfaMark(3u,pcfaEdgeLinear(level,hierarchyEdge),8u);}
  else{_=pcfaMark(2u,pcfaNodeLinear(level,pcfHierarchyParent(level,
    pcfAggregateEdgeSourceBrick(edge))),8u);}}}
${FAMILY_NAMES.map((name) => repairWorkset(name, layout.aggregateFamilies[name], arena)).join("\n")}
${FAMILY_NAMES.map((name) => rankSelect(name, layout.aggregateFamilies[name], arena)).join("\n")}
fn pcfaFinalizeFamily(family:u32,root:u32,edgeFamily:bool){let header=pcfaFamilyHeader(family);
  if(atomicLoad(&${arena}[header+PCFA_F_REPAIRED_LEAVES])
      !=atomicLoad(&${arena}[header+PCFA_F_DIRTY_LEAVES])){pcfaFault(family,PCF_FAULT_REPAIR,family);return;}
  let count=select(atomicLoad(&${arena}[root]),pcfaFamilyCapacity(family),PCFA_QA);
  atomicStore(&${arena}[header+PCFA_F_WORK_COUNT],count);
  atomicStore(&${arena}[header+PCFA_F_WORK_X],select(count,(count+63u)/64u,edgeFamily));}
fn pcfAggregateFinalizeFineRepair(){if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCF_PHASE_REPAIRING
    ||atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])!=0u){pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return;}
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_REPAIRED_LEAVES])
    !=atomicLoad(&${arena}[PCF_BASE+PCF_H_DIRTY_LEAVES])){pcfFault(PCF_FAULT_REPAIR,PCF_INVALID);return;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCFA_PHASE_AGGREGATE_REPAIR);
  atomicStore(&${arena}[PCFA_BRICK_HEADER+PCFA_F_REPAIR_X],atomicLoad(&${arena}[PCFA_BRICK_HEADER+PCFA_F_DIRTY_LEAVES]));
  atomicStore(&${arena}[PCFA_AGGREGATEEDGE_HEADER+PCFA_F_REPAIR_X],atomicLoad(&${arena}[PCFA_AGGREGATEEDGE_HEADER+PCFA_F_DIRTY_LEAVES]));}
@compute @workgroup_size(1) fn finalizePersistentPressureFineCache(){pcfAggregateFinalizeFineRepair();}
@compute @workgroup_size(1) fn finalizePersistentPressureAggregatePlan(){
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCFA_PHASE_AGGREGATE_REPAIR){pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return;}
  pcfaFinalizeFamily(0u,${roots.brick}u,false);pcfaFinalizeFamily(1u,${roots.aggregateEdge}u,true);
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])==0u){atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCFA_PHASE_AGGREGATE_EXECUTE);}}
@compute @workgroup_size(64) fn repairPersistentPressureAggregateEdges(@builtin(global_invocation_id)gid:vec3u){
  let edge=select(pcfAggregateEdgeRankSelect(gid.x),gid.x,PCFA_QA);
  if(edge>=${layout.aggregateEdgeCount}u){return;}let range=pcfAggregateEdgeContributionRange(edge);var weight=0.0;
  for(var at=0u;at<range.y;at+=1u){weight+=pcfEdgeWeight(pcfAggregateEdgeContribution(range.x+at));}
  let address=${layout.brickAggregateEdgeBaseWords}u+edge;
  let changed=atomicExchange(&${arena}[address],bitcast<u32>(weight))!=bitcast<u32>(weight);
  if(changed||PCFA_QA){pcfaMarkAggregateEdgeAncestors(edge);}atomicAdd(&${arena}[PCFA_AGGREGATEEDGE_HEADER+PCFA_F_EXECUTED],1u);}
@compute @workgroup_size(64) fn repairPersistentPressureBrickDiagonals(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){let brick=select(pcfBrickRankSelect(wid.x),wid.x,PCFA_QA);
  var diagonal=0.0;var wet=0u;if(brick<${layout.brickCount}u){let range=pcfBrickCellRange(brick);
    for(var local=lane;local<range.y;local+=64u){let cell=range.x+local;if(!pcmCellContains(cell)){continue;}
      wet=1u;diagonal+=pcfDiagonal(cell);let edges=cm12HotDirectedEdgeRange(cell);
      for(var at=0u;at<edges.y;at+=1u){let edgeId=edges.x+at;let edge=cm12HotDirectedEdge(edgeId);
        if(pcfCellBrick(edge.x)==brick){diagonal+=pcfEdgeWeight(edgeId);}}}}
  pcfaReduce[lane]=diagonal;pcfaWet[lane]=wet;workgroupBarrier();var width=32u;loop{if(lane<width){pcfaReduce[lane]+=pcfaReduce[lane+width];pcfaWet[lane]|=pcfaWet[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u&&brick<${layout.brickCount}u){let value=max(pcfaReduce[0],1e-12);
    let address=${layout.brickAggregateDiagonalBaseWords}u+brick;
    let changed=atomicExchange(&${arena}[address],bitcast<u32>(value))!=bitcast<u32>(value);
    let range=pcfBrickCellRange(brick);var level=0u;var cells=range.y;
    while(cells>1u){cells/=8u;level+=1u;}let packed=select(0u,
      ((range.x+1u)<<3u)|(level+1u),pcfaWet[0]!=0u);
    atomicStore(&${arena}[${layout.brickAggregateRangeBaseWords}u+brick],packed);
    if(changed||PCFA_QA){pcfaMarkBrickAncestors(brick);}atomicAdd(&${arena}[PCFA_BRICK_HEADER+PCFA_F_EXECUTED],1u);}}
var<workgroup>pcfaReduce:array<f32,64>;
var<workgroup>pcfaWet:array<u32,64>;
@compute @workgroup_size(1) fn finalizePersistentPressureAggregateExecution(){
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCFA_PHASE_AGGREGATE_EXECUTE){pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return;}
  if((!PCFA_QA&&atomicLoad(&${arena}[PCFA_BRICK_HEADER+PCFA_F_EXECUTED])!=atomicLoad(&${arena}[PCFA_BRICK_HEADER+PCFA_F_WORK_COUNT]))
    ||(!PCFA_QA&&atomicLoad(&${arena}[PCFA_AGGREGATEEDGE_HEADER+PCFA_F_EXECUTED])!=atomicLoad(&${arena}[PCFA_AGGREGATEEDGE_HEADER+PCFA_F_WORK_COUNT]))){pcfFault(PCF_FAULT_REPAIR,PCF_INVALID);return;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCFA_PHASE_HIERARCHY_REPAIR);
  atomicStore(&${arena}[PCFA_HIERARCHYNODE_HEADER+PCFA_F_REPAIR_X],atomicLoad(&${arena}[PCFA_HIERARCHYNODE_HEADER+PCFA_F_DIRTY_LEAVES]));
  atomicStore(&${arena}[PCFA_HIERARCHYEDGE_HEADER+PCFA_F_REPAIR_X],atomicLoad(&${arena}[PCFA_HIERARCHYEDGE_HEADER+PCFA_F_DIRTY_LEAVES]));}
@compute @workgroup_size(1) fn finalizePersistentPressureHierarchyPlan(){
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCFA_PHASE_HIERARCHY_REPAIR){pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return;}
  pcfaFinalizeFamily(2u,${roots.hierarchyNode}u,false);pcfaFinalizeFamily(3u,${roots.hierarchyEdge}u,true);
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_FAULT])==0u){atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCFA_PHASE_HIERARCHY_EXECUTE);}}
@compute @workgroup_size(64) fn repairPersistentPressureHierarchyEdges(@builtin(global_invocation_id)gid:vec3u){
  let linear=select(pcfHierarchyEdgeRankSelect(gid.x),gid.x,PCFA_QA);let address=pcfaEdgeAddress(linear);
  if(address.x==PCF_INVALID){return;}let range=pcfHierarchyEdgeContributionRange(address.x,address.y);var weight=0.0;
  for(var at=0u;at<range.y;at+=1u){let edge=pcfHierarchyEdgeContribution(address.x,range.x+at);
    weight+=bitcast<f32>(atomicLoad(&${arena}[${layout.brickAggregateEdgeBaseWords}u+edge]));}
  atomicStore(&${arena}[pcfaHierarchyEdgeWord(address.x,address.y)],bitcast<u32>(weight));
  atomicAdd(&${arena}[PCFA_HIERARCHYEDGE_HEADER+PCFA_F_EXECUTED],1u);}
@compute @workgroup_size(64) fn repairPersistentPressureHierarchyDiagonals(@builtin(local_invocation_index)lane:u32,
 @builtin(workgroup_id)wid:vec3u){let linear=select(pcfHierarchyNodeRankSelect(wid.x),wid.x,PCFA_QA);
  let address=pcfaNodeAddress(linear);var diagonal=0.0;if(address.x!=PCF_INVALID){
    let children=pcfHierarchyChildRange(address.x,address.y);
    for(var at=lane;at<children.y;at+=64u){let brick=pcfHierarchyChild(address.x,children.x+at);
      diagonal+=bitcast<f32>(atomicLoad(&${arena}[${layout.brickAggregateDiagonalBaseWords}u+brick]));}
    let internal=pcfHierarchyInternalEdgeRange(address.x,address.y);
    for(var at=lane;at<internal.y;at+=64u){let edge=pcfHierarchyInternalEdge(address.x,internal.x+at);
      diagonal+=bitcast<f32>(atomicLoad(&${arena}[${layout.brickAggregateEdgeBaseWords}u+edge]));}}
  pcfaReduce[lane]=diagonal;workgroupBarrier();var width=32u;loop{if(lane<width){pcfaReduce[lane]+=pcfaReduce[lane+width];}
    workgroupBarrier();if(width==1u){break;}width/=2u;}
  if(lane==0u&&address.x!=PCF_INVALID){atomicStore(&${arena}[pcfaHierarchyDiagonalWord(address.x,address.y)],
    bitcast<u32>(max(pcfaReduce[0],1e-12)));atomicAdd(&${arena}[PCFA_HIERARCHYNODE_HEADER+PCFA_F_EXECUTED],1u);}}
@compute @workgroup_size(1) fn finalizePersistentPressureCache(){
  if(atomicLoad(&${arena}[PCF_BASE+PCF_H_PHASE])!=PCFA_PHASE_HIERARCHY_EXECUTE){pcfFault(PCF_FAULT_PHASE,PCF_INVALID);return;}
  if(atomicLoad(&${arena}[PCFA_BASE+${ah.topologyGeneration}u])!=pcfTopologyGeneration()
    ||atomicLoad(&${arena}[PCFA_BASE+${ah.pcmGeneration}u])!=pcfPCMGeneration()
    ||atomicLoad(&${arena}[PCFA_BASE+${ah.aggregateTopologyGeneration}u])!=pcfAggregateTopologyGeneration()){
    pcfFault(PCF_FAULT_TOPOLOGY,PCF_INVALID);return;}
  if((!PCFA_QA&&atomicLoad(&${arena}[PCFA_HIERARCHYNODE_HEADER+PCFA_F_EXECUTED])!=atomicLoad(&${arena}[PCFA_HIERARCHYNODE_HEADER+PCFA_F_WORK_COUNT]))
    ||(!PCFA_QA&&atomicLoad(&${arena}[PCFA_HIERARCHYEDGE_HEADER+PCFA_F_EXECUTED])!=atomicLoad(&${arena}[PCFA_HIERARCHYEDGE_HEADER+PCFA_F_WORK_COUNT]))){pcfFault(PCF_FAULT_REPAIR,PCF_INVALID);return;}
  atomicStore(&${arena}[PCF_BASE+PCF_H_ACCEPTED_GEN],atomicLoad(&${arena}[PCF_BASE+PCF_H_CANDIDATE_GEN]));
  atomicStore(&${arena}[PCFA_BASE+${ah.acceptedAggregateTopologyGeneration}u],
    atomicLoad(&${arena}[PCFA_BASE+${ah.aggregateTopologyGeneration}u]));
  pcfaZeroIndirects();atomicStore(&${arena}[PCF_BASE+PCF_H_PHASE],PCF_PHASE_ACCEPTED);}
fn pcfBrickWorkCount()->u32{return atomicLoad(&${arena}[
  PCFA_BRICK_HEADER+PCFA_F_WORK_COUNT]);}
`;
}
