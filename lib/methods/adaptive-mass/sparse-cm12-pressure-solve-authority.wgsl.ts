import {
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_BRANCH,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_CAUSE,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_INVALID,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_LEAF_BITS,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_MAGIC,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE,
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_VERSION,
  type SparseCM12PressureSolveAuthorityFamilyLayout,
  type SparseCM12PressureSolveAuthorityLayout,
} from "./sparse-cm12-pressure-solve-authority";

export interface SparseCM12PressureSolveAuthorityWGSLOptions {
  readonly layout: SparseCM12PressureSolveAuthorityLayout;
  /** Existing `array<atomic<u32>>` storage arena. */
  readonly arenaName?: string;
  readonly prefix?: string;
  readonly workgroupSize?: number;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError(`${label} is not WGSL`);
  return value;
};

function familyConstants(prefix: string, label: string,
  family: SparseCM12PressureSolveAuthorityFamilyLayout): string {
  return /* wgsl */ `
const ${prefix}${label}Header:u32=${family.headerBaseWords}u;
const ${prefix}${label}EntityCapacity:u32=${family.capacity}u;
const ${prefix}${label}Bits:u32=${family.activeBitsBaseWords}u;
const ${prefix}${label}DirtyStamp:u32=${family.dirtyLeafStampBaseWords}u;
const ${prefix}${label}DirtyList:u32=${family.dirtyLeafListBaseWords}u;
const ${prefix}${label}LeafCount:u32=${family.leafCount}u;
${family.treeLevelBaseWords.map((base, level) =>
    `const ${prefix}${label}Tree${level}:u32=${base}u;`).join("\n")}
`;
}

function rankSelect(prefix: string, label: string,
  family: SparseCM12PressureSolveAuthorityFamilyLayout, arena: string): string {
  const descend = Array.from({ length: family.treeLevelCounts.length - 1 }, (_, index) => {
    const level = family.treeLevelCounts.length - 2 - index;
    return /* wgsl */ `
  {
    let childBegin=node*${prefix}Branch;var selected=${prefix}Invalid;
    for(var child=0u;child<${prefix}Branch;child+=1u){
      let candidate=childBegin+child;if(candidate>=${family.treeLevelCounts[level]}u){break;}
      let count=atomicLoad(&${arena}[${prefix}${label}Tree${level}+candidate]);
      if(selected==${prefix}Invalid&&remaining<count){selected=candidate;}
      else if(selected==${prefix}Invalid){remaining-=count;}
    }
    if(selected==${prefix}Invalid){return ${prefix}Invalid;}node=selected;
  }`;
  }).join("\n");
  return /* wgsl */ `
fn ${prefix}${label}RankSelect(rank:u32)->u32{
  let total=atomicLoad(&${arena}[${prefix}${label}Header+${prefix}FActiveCount]);
  if(rank>=total){return ${prefix}Invalid;}var remaining=rank;var node=0u;
${descend}
  let firstWord=node*${prefix}LeafWords;
  for(var wordAt=0u;wordAt<${prefix}LeafWords;wordAt+=1u){
    let word=atomicLoad(&${arena}[${prefix}${label}Bits+firstWord+wordAt]);
    let count=countOneBits(word);if(remaining>=count){remaining-=count;continue;}
    for(var bit=0u;bit<32u;bit+=1u){if((word&(1u<<bit))==0u){continue;}
      if(remaining==0u){let id=node*${prefix}LeafBits+32u*wordAt+bit;
        return select(${prefix}Invalid,id,id<${prefix}${label}EntityCapacity);}remaining-=1u;}
  }
  return ${prefix}Invalid;
}`;
}

function repairKernel(prefix: string, label: "Brick" | "Node", familyIndex: number,
  family: SparseCM12PressureSolveAuthorityFamilyLayout, arena: string,
  exactPredicate: string, workgroupSize: number): string {
  const updates = family.treeLevelBaseWords.slice(1).map((base, index) => /* wgsl */ `
    node/=${prefix}Branch;
    let prior${index}=atomicAdd(&${arena}[${base}u+node],bitcast<u32>(delta));
    if(delta<0&&prior${index}<u32(-delta)){
      ${prefix}Fail(${familyIndex}u,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.countTreeUnderflow}u,
        leaf*${prefix}LeafBits);
    }else if(delta>0&&prior${index}>0xffffffffu-u32(delta)){
      ${prefix}Fail(${familyIndex}u,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.countTreeOverflow}u,
        leaf*${prefix}LeafBits);
    }
  `).join("");
  const brickClosure = label === "Brick" ? /* wgsl */ `
    // The construction-only oracle revalidates every parent even when the
    // brick bit is unchanged. Production queues only actual membership deltas.
    let changed=select(previous^word,0xffffffffu,${prefix}QAOracle);
    for(var changedBit=0u;changedBit<32u;changedBit+=1u){
      if((changed&(1u<<changedBit))==0u){continue;}
      let brick=leaf*${prefix}LeafBits+32u*lane+changedBit;
      if(brick<${prefix}BrickCapacity){${prefix}MarkHierarchyClosureForBrick(brick);}
    }` : "";
  return /* wgsl */ `
@compute @workgroup_size(${workgroupSize})
fn repairSparseCM12Pressure${label}Leaves(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let family=${familyIndex}u;let header=${prefix}FamilyHeader(family);
  let dirty=atomicLoad(&${arena}[header+${prefix}FDirtyLeafCount]);
  let invocationValid=wid.x<dirty;var leaf=0u;
  if(invocationValid){leaf=atomicLoad(&${arena}[${prefix}${label}DirtyList+wid.x]);}
  let expectedPhase=select(${prefix}PhaseRepairingNodes,${prefix}PhaseRepairingBricks,
    family==${prefix}BrickFamily);
  let valid=atomicLoad(&${arena}[${prefix}HPhase])==expectedPhase&&invocationValid
    &&leaf<${prefix}${label}LeafCount&&atomicLoad(&${arena}[${prefix}HFault])==0u;
  if(lane<${prefix}LeafWords){
    var word=0u;if(valid){for(var bit=0u;bit<32u;bit+=1u){
      let id=leaf*${prefix}LeafBits+32u*lane+bit;
      if(id<${prefix}${label}EntityCapacity&&${exactPredicate}(id)){word|=1u<<bit;}
    }
    let wordAt=${prefix}${label}Bits+leaf*${prefix}LeafWords+lane;
    let previous=atomicExchange(&${arena}[wordAt],word);${brickClosure}}
    ${prefix}LeafCounts[lane]=countOneBits(word);
  }
  workgroupBarrier();
  if(lane==0u&&valid){var count=0u;
    for(var at=0u;at<${prefix}LeafWords;at+=1u){count+=${prefix}LeafCounts[at];}
    let previous=atomicExchange(&${arena}[${prefix}${label}Tree0+leaf],count);
    let delta=i32(count)-i32(previous);var node=leaf;${updates}
    atomicAdd(&${arena}[header+${prefix}FRepairedLeafCount],1u);
  }
}`;
}

/**
 * Binding-free PSA1 helper source.
 *
 * Integration supplies the exact PCM/PCF/topology hooks listed in the WGSL
 * preamble. There are no floating certificates and no runtime full mode.
 */
export function createSparseCM12PressureSolveAuthorityWGSL(
  options: SparseCM12PressureSolveAuthorityWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "pressureSolveArena", "arenaName");
  const p = identifier(options.prefix ?? "psa", "prefix");
  const workgroupSize = options.workgroupSize ?? 64;
  if (!Number.isSafeInteger(workgroupSize) || workgroupSize < 8 || workgroupSize > 256) {
    throw new RangeError("PSA1 workgroupSize must be in [8, 256]");
  }
  const h = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER;
  const f = SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER;
  const hb = (word: number) => `${layout.baseWords + word}u`;
  const brickRoot = layout.brick.treeLevelBaseWords.at(-1)!;
  const nodeRoot = layout.hierarchyNode.treeLevelBaseWords.at(-1)!;
  const levelAddress = layout.hierarchyLevelCounts.map((count, level) => {
    const offset = layout.hierarchyLevelOffsets[level]!;
    return `if(linear<${offset + count}u){return vec2u(${level}u,linear-${offset}u);}`;
  }).join("\n  ");
  const levelLinear = layout.hierarchyLevelCounts.map((_, level) =>
    `if(level==${level}u){return ${layout.hierarchyLevelOffsets[level]}u+group;}`)
    .join("\n  ");
  const familyBases = (kind: "bits" | "stamp" | "list" | "leafCount") => {
    const key = kind === "bits" ? "activeBitsBaseWords"
      : kind === "stamp" ? "dirtyLeafStampBaseWords"
        : kind === "list" ? "dirtyLeafListBaseWords" : "leafCount";
    return `select(${layout.brick[key]}u,${layout.hierarchyNode[key]}u,`
      + `family==${p}NodeFamily)`;
  };
  return /* wgsl */ `
// Required integration hooks (all GPU-authored/current-epoch):
// psaFrameGeneration/TopologyGeneration/PCMGeneration/PCFGeneration()->u32
// psaExpectedProducerReceipts/PCMCellWorkgroupCount()->u32
// psaCellBrick(cell)->u32; psaBrickWetExact(brick)->bool
// psaHierarchyParent(level,brick)->u32
// psaHierarchyNodeActiveExact(linearNode)->bool
// psaPressureAddressingReady()->bool
const ${p}Magic:u32=0x${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_MAGIC.toString(16)}u;
const ${p}Version:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_VERSION}u;
const ${p}Invalid:u32=0x${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_INVALID.toString(16)}u;
const ${p}BrickFamily:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY.brick}u;
const ${p}NodeFamily:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY.hierarchyNode}u;
const ${p}PhaseUninitialized:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE.uninitialized}u;
const ${p}PhaseAccepted:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE.accepted}u;
const ${p}PhaseCollecting:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE.collecting}u;
const ${p}PhaseRepairingBricks:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE.repairingBricks}u;
const ${p}PhaseRepairingNodes:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE.repairingNodes}u;
const ${p}PhaseFault:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_PHASE.fault}u;
const ${p}LeafBits:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_LEAF_BITS}u;
const ${p}LeafWords:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_LEAF_BITS / 32}u;
const ${p}Branch:u32=${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_BRANCH}u;
const ${p}BrickCapacity:u32=${layout.brickCapacity}u;
const ${p}NodeCapacity:u32=${layout.hierarchyNodeCapacity}u;
const ${p}LevelCount:u32=${layout.hierarchyLevelCounts.length}u;
const ${p}QAOracle:bool=${layout.qaFullOracle ? "true" : "false"};
const ${p}HPhase:u32=${hb(h.phase)};const ${p}HAcceptedGeneration:u32=${hb(h.acceptedGeneration)};
const ${p}HCandidateGeneration:u32=${hb(h.candidateGeneration)};
const ${p}HFrameGeneration:u32=${hb(h.frameGeneration)};
const ${p}HTopologyGeneration:u32=${hb(h.topologyGeneration)};
const ${p}HPCMGeneration:u32=${hb(h.pcmGeneration)};const ${p}HPCFGeneration:u32=${hb(h.pcfGeneration)};
const ${p}HFault:u32=${hb(h.fault)};const ${p}HFirstFaultFamily:u32=${hb(h.firstFaultFamily)};
const ${p}HFirstFaultId:u32=${hb(h.firstFaultId)};
const ${p}HExpectedReceipts:u32=${hb(h.expectedProducerReceipts)};
const ${p}HCoveredReceipts:u32=${hb(h.coveredProducerReceipts)};
const ${p}HCauseMask:u32=${hb(h.causeMask)};
const ${p}HBootstrapIndirectX:u32=${hb(h.bootstrapIndirectX)};
const ${p}FActiveCount:u32=${f.activeCount}u;const ${p}FDirtyLeafCount:u32=${f.dirtyLeafCount}u;
const ${p}FDirectWriteCount:u32=${f.directWriteCount}u;
const ${p}FClosureWriteCount:u32=${f.closureWriteCount}u;
const ${p}FRepairIndirectX:u32=${f.repairIndirectX}u;
const ${p}FWorkIndirectX:u32=${f.workIndirectX}u;
const ${p}FRepairedLeafCount:u32=${f.repairedLeafCount}u;
${familyConstants(p, "Brick", layout.brick)}
${familyConstants(p, "Node", layout.hierarchyNode)}
var<workgroup>${p}LeafCounts:array<u32,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_LEAF_BITS / 32}>;
fn ${p}FamilyHeader(family:u32)->u32{return select(${p}BrickHeader,${p}NodeHeader,
  family==${p}NodeFamily);}
fn ${p}FamilyBits(family:u32)->u32{return ${familyBases("bits")};}
fn ${p}FamilyDirtyStamp(family:u32)->u32{return ${familyBases("stamp")};}
fn ${p}FamilyDirtyList(family:u32)->u32{return ${familyBases("list")};}
fn ${p}FamilyLeafCount(family:u32)->u32{return ${familyBases("leafCount")};}
fn ${p}HierarchyNodeAddress(linear:u32)->vec2u{
  ${levelAddress}
  return vec2u(${p}Invalid);}
fn ${p}HierarchyNodeLinear(level:u32,group:u32)->u32{
  ${levelLinear}
  return ${p}Invalid;}
fn ${p}HeaderValid()->bool{return arrayLength(&${arena})>=${layout.totalWords}u
  &&atomicLoad(&${arena}[${hb(h.magic)}])==${p}Magic
  &&atomicLoad(&${arena}[${hb(h.version)}])==${p}Version
  &&atomicLoad(&${arena}[${hb(h.headerWords)}])==${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER_WORDS}u
  &&atomicLoad(&${arena}[${hb(h.familyHeaderWords)}])==${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY_HEADER_WORDS}u
  &&atomicLoad(&${arena}[${hb(h.familyCount)}])==2u
  &&atomicLoad(&${arena}[${hb(h.leafBits)}])==${p}LeafBits
  &&atomicLoad(&${arena}[${hb(h.branch)}])==${p}Branch
  &&atomicLoad(&${arena}[${hb(h.brickCapacity)}])==${p}BrickCapacity
  &&atomicLoad(&${arena}[${hb(h.hierarchyLevelCount)}])==${p}LevelCount
  &&atomicLoad(&${arena}[${hb(h.hierarchyNodeCapacity)}])==${p}NodeCapacity
  &&atomicLoad(&${arena}[${hb(h.brickHeaderBase)}])==${p}BrickHeader
  &&atomicLoad(&${arena}[${hb(h.nodeHeaderBase)}])==${p}NodeHeader
  &&atomicLoad(&${arena}[${hb(h.totalWords)}])==${layout.totalWords}u;}
fn ${p}ZeroIndirects(){
  let enabled=false;
  atomicStore(&${arena}[${p}BrickHeader+${p}FRepairIndirectX],0u);
  atomicStore(&${arena}[${p}BrickHeader+${p}FWorkIndirectX],0u);
  atomicStore(&${arena}[${p}NodeHeader+${p}FRepairIndirectX],0u);
  atomicStore(&${arena}[${p}NodeHeader+${p}FWorkIndirectX],0u);
  atomicStore(&${arena}[${p}HBootstrapIndirectX],0u);
}
fn ${p}Fail(family:u32,code:u32,id:u32){
  let won=atomicCompareExchangeWeak(&${arena}[${p}HFault],0u,code).exchanged;
  if(won){atomicStore(&${arena}[${p}HFirstFaultFamily],family);
    atomicStore(&${arena}[${p}HFirstFaultId],id);}
  atomicStore(&${arena}[${p}HPhase],${p}PhaseFault);${p}ZeroIndirects();
}
fn ${p}QueueLeaf(family:u32,leaf:u32,closure:bool)->bool{
  if(leaf>=${p}FamilyLeafCount(family)){
    ${p}Fail(family,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.dirtyCapacity}u,
      leaf*${p}LeafBits);return false;}
  let generation=atomicLoad(&${arena}[${p}HCandidateGeneration]);
  if(atomicExchange(&${arena}[${p}FamilyDirtyStamp(family)+leaf],generation)==generation){
    return true;}
  let header=${p}FamilyHeader(family);
  let slot=atomicAdd(&${arena}[header+${p}FDirtyLeafCount],1u);
  if(slot>=${p}FamilyLeafCount(family)){
    ${p}Fail(family,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.dirtyCapacity}u,
      leaf*${p}LeafBits);return false;}
  atomicStore(&${arena}[${p}FamilyDirtyList(family)+slot],leaf);
  atomicAdd(&${arena}[header+select(${p}FDirectWriteCount,${p}FClosureWriteCount,closure)],1u);
  return true;
}
fn ${p}MarkBrick(brick:u32,cause:u32,closure:bool,receipt:bool)->bool{
  if(atomicLoad(&${arena}[${p}HPhase])!=${p}PhaseCollecting){
    ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.invalidPhase}u,brick);
    return false;}
  if(brick>=${p}BrickCapacity){
    ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.invalidBrick}u,brick);
    return false;}
  let ok=${p}QueueLeaf(${p}BrickFamily,brick/${p}LeafBits,closure);
  atomicOr(&${arena}[${p}HCauseMask],cause);
  if(receipt&&ok){atomicAdd(&${arena}[${p}HCoveredReceipts],1u);}return ok;
}
fn psaMarkPCMCellTransition(cell:u32,cause:u32,receipt:bool)->bool{
  return ${p}MarkBrick(psaCellBrick(cell),cause,false,receipt);}
fn psaMarkTopologyBrickBlast(brick:u32,receipt:bool)->bool{
  return ${p}MarkBrick(brick,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_CAUSE.topologyBrick}u,
    false,receipt);}
fn ${p}MarkHierarchyClosureForBrick(brick:u32){
  for(var level=0u;level<${p}LevelCount;level+=1u){
    let node=${p}HierarchyNodeLinear(level,psaHierarchyParent(level,brick));
    if(node==${p}Invalid||node>=${p}NodeCapacity){
      ${p}Fail(${p}NodeFamily,
        ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.invalidHierarchyNode}u,node);return;}
    _=${p}QueueLeaf(${p}NodeFamily,node/${p}LeafBits,true);
  }
  atomicOr(&${arena}[${p}HCauseMask],
    ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_CAUSE.hierarchyClosure}u);
}
@compute @workgroup_size(1)
fn beginSparseCM12PressureSolveAuthority(){
  if(!${p}HeaderValid()){
    ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.invalidHeader}u,
      ${p}Invalid);return;}
  if(!psaPressureAddressingReady()){
    ${p}Fail(${p}BrickFamily,
      ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.pressureAddressingGap}u,
      ${p}Invalid);return;}
  let phase=atomicLoad(&${arena}[${p}HPhase]);
  if(phase!=${p}PhaseUninitialized&&phase!=${p}PhaseAccepted){
    ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.invalidPhase}u,
      ${p}Invalid);return;}
  let accepted=atomicLoad(&${arena}[${p}HAcceptedGeneration]);
  if(accepted>=0x7ffffffeu){${p}Fail(${p}BrickFamily,
    ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.generationExhausted}u,${p}Invalid);return;}
  let bootstrap=phase==${p}PhaseUninitialized;
  atomicStore(&${arena}[${p}HCandidateGeneration],accepted+1u);
  atomicStore(&${arena}[${p}HFrameGeneration],psaFrameGeneration());
  atomicStore(&${arena}[${p}HTopologyGeneration],psaTopologyGeneration());
  atomicStore(&${arena}[${p}HPCMGeneration],psaPCMGeneration());
  atomicStore(&${arena}[${p}HPCFGeneration],psaPCFGeneration());
  atomicStore(&${arena}[${p}HFault],0u);
  atomicStore(&${arena}[${p}HFirstFaultFamily],${p}Invalid);
  atomicStore(&${arena}[${p}HFirstFaultId],${p}Invalid);
  atomicStore(&${arena}[${p}HExpectedReceipts],psaExpectedProducerReceipts());
  atomicStore(&${arena}[${p}HCoveredReceipts],0u);
  atomicStore(&${arena}[${p}HCauseMask],0u);
  for(var family=0u;family<2u;family+=1u){let header=${p}FamilyHeader(family);
    atomicStore(&${arena}[header+${p}FDirtyLeafCount],0u);
    atomicStore(&${arena}[header+${p}FDirectWriteCount],0u);
    atomicStore(&${arena}[header+${p}FClosureWriteCount],0u);
    atomicStore(&${arena}[header+${p}FRepairedLeafCount],0u);
    atomicStore(&${arena}[header+${p}FRepairIndirectX],0u);
    atomicStore(&${arena}[header+${p}FWorkIndirectX],0u);
  }
  atomicStore(&${arena}[${p}HBootstrapIndirectX],select(0u,
    (${p}BrickCapacity+${workgroupSize - 1}u)/${workgroupSize}u,bootstrap||${p}QAOracle));
  atomicStore(&${arena}[${p}HPhase],${p}PhaseCollecting);
}
@compute @workgroup_size(${workgroupSize})
fn seedSparseCM12PressureSolveBootstrap(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=${p}BrickCapacity){return;}let cause=select(
    ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_CAUSE.bootstrap}u,
    ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_CAUSE.qaOracle}u,${p}QAOracle);
  _=${p}MarkBrick(gid.x,cause,false,false);
}
@compute @workgroup_size(1)
fn finalizeSparseCM12PressureBrickFrontier(){
  if(atomicLoad(&${arena}[${p}HPhase])!=${p}PhaseCollecting
    ||atomicLoad(&${arena}[${p}HFault])!=0u){return;}
  if(atomicLoad(&${arena}[${p}HExpectedReceipts])
      !=atomicLoad(&${arena}[${p}HCoveredReceipts])){
    ${p}Fail(${p}BrickFamily,
      ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.producerCoverageGap}u,${p}Invalid);return;}
  atomicStore(&${arena}[${p}BrickHeader+${p}FRepairIndirectX],atomicLoad(&${arena}[
    ${p}BrickHeader+${p}FDirtyLeafCount]));
  atomicStore(&${arena}[${p}HPhase],${p}PhaseRepairingBricks);
}
${repairKernel(p, "Brick", SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY.brick,
    layout.brick, arena, "psaBrickWetExact", workgroupSize)}
@compute @workgroup_size(1)
fn finalizeSparseCM12PressureBrickRepair(){
  if(atomicLoad(&${arena}[${p}HPhase])!=${p}PhaseRepairingBricks
    ||atomicLoad(&${arena}[${p}HFault])!=0u){return;}
  if(atomicLoad(&${arena}[${p}BrickHeader+${p}FRepairedLeafCount])
      !=atomicLoad(&${arena}[${p}BrickHeader+${p}FDirtyLeafCount])){
    ${p}Fail(${p}BrickFamily,
      ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.producerCoverageGap}u,${p}Invalid);return;}
  atomicStore(&${arena}[${p}NodeHeader+${p}FRepairIndirectX],atomicLoad(&${arena}[
    ${p}NodeHeader+${p}FDirtyLeafCount]));
  atomicStore(&${arena}[${p}HPhase],${p}PhaseRepairingNodes);
}
${repairKernel(p, "Node", SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAMILY.hierarchyNode,
    layout.hierarchyNode, arena, "psaHierarchyNodeActiveExact", workgroupSize)}
@compute @workgroup_size(1)
fn finalizeSparseCM12PressureSolveAuthority(){
  if(atomicLoad(&${arena}[${p}HPhase])!=${p}PhaseRepairingNodes
    ||atomicLoad(&${arena}[${p}HFault])!=0u){return;}
  if(atomicLoad(&${arena}[${p}NodeHeader+${p}FRepairedLeafCount])
      !=atomicLoad(&${arena}[${p}NodeHeader+${p}FDirtyLeafCount])){
    ${p}Fail(${p}NodeFamily,
      ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.producerCoverageGap}u,${p}Invalid);return;}
  if(atomicLoad(&${arena}[${p}HFrameGeneration])!=psaFrameGeneration()
    ||atomicLoad(&${arena}[${p}HTopologyGeneration])!=psaTopologyGeneration()){
    ${p}Fail(${p}BrickFamily,
      ${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.topologyGenerationGap}u,
      ${p}Invalid);return;}
  if(atomicLoad(&${arena}[${p}HPCMGeneration])!=psaPCMGeneration()){
    ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.pcmGenerationGap}u,
      ${p}Invalid);return;}
  if(atomicLoad(&${arena}[${p}HPCFGeneration])!=psaPCFGeneration()){
    ${p}Fail(${p}BrickFamily,${SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_FAULT.pcfGenerationGap}u,
      ${p}Invalid);return;}
  let wet=atomicLoad(&${arena}[${brickRoot}u]);let nodes=atomicLoad(&${arena}[${nodeRoot}u]);
  atomicStore(&${arena}[${p}BrickHeader+${p}FActiveCount],wet);
  atomicStore(&${arena}[${p}NodeHeader+${p}FActiveCount],nodes);
  atomicStore(&${arena}[${p}BrickHeader+${p}FWorkIndirectX],
    select(wet,${p}BrickCapacity,${p}QAOracle));
  atomicStore(&${arena}[${p}NodeHeader+${p}FWorkIndirectX],
    select(nodes,${p}NodeCapacity,${p}QAOracle));
  atomicStore(&${arena}[${p}HAcceptedGeneration],atomicLoad(&${arena}[
    ${p}HCandidateGeneration]));
  atomicStore(&${arena}[${p}HPhase],${p}PhaseAccepted);
}
${rankSelect(p, "Brick", layout.brick, arena)}
${rankSelect(p, "Node", layout.hierarchyNode, arena)}
fn ${p}BrickContains(brick:u32)->bool{return brick<${p}BrickCapacity
  &&(atomicLoad(&${arena}[${p}BrickBits+(brick>>5u)])&(1u<<(brick&31u)))!=0u;}
fn ${p}NodeContains(node:u32)->bool{return node<${p}NodeCapacity
  &&(atomicLoad(&${arena}[${p}NodeBits+(node>>5u)])&(1u<<(node&31u)))!=0u;}
fn psaWetBrickContains(brick:u32)->bool{return ${p}BrickContains(brick);}
fn psaActiveHierarchyNodeContains(node:u32)->bool{return ${p}NodeContains(node);}
fn psaAcceptedGeneration()->u32{return atomicLoad(&${arena}[${p}HAcceptedGeneration]);}
fn psaFaultCode()->u32{return atomicLoad(&${arena}[${p}HFault]);}
fn psaWetBrickCount()->u32{return atomicLoad(&${arena}[
  ${p}BrickHeader+${p}FActiveCount]);}
fn psaActiveHierarchyNodeCount()->u32{return atomicLoad(&${arena}[
  ${p}NodeHeader+${p}FActiveCount]);}
fn ${p}BrickExecutionCount()->u32{return select(atomicLoad(&${arena}[
  ${p}BrickHeader+${p}FActiveCount]),${p}BrickCapacity,${p}QAOracle);}
fn ${p}NodeExecutionCount()->u32{return select(atomicLoad(&${arena}[
  ${p}NodeHeader+${p}FActiveCount]),${p}NodeCapacity,${p}QAOracle);}
fn psaWetBrickInvocation(invocation:u32)->u32{
  if(${p}QAOracle){return select(${p}Invalid,invocation,invocation<${p}BrickCapacity);}
  return ${p}BrickRankSelect(invocation);}
fn psaActiveHierarchyNodeInvocation(invocation:u32)->u32{
  if(${p}QAOracle){return select(${p}Invalid,invocation,invocation<${p}NodeCapacity);}
  return ${p}NodeRankSelect(invocation);}
fn psaActiveHierarchyNodeAddress(invocation:u32)->vec2u{
  return ${p}HierarchyNodeAddress(psaActiveHierarchyNodeInvocation(invocation));}
`;
}
