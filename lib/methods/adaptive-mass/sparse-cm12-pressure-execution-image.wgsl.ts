import {
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER_WORDS,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_MAGIC,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE,
  SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_VERSION,
  type SparseCM12PressureExecutionImageLayout,
} from "./sparse-cm12-pressure-execution-image";

export interface SparseCM12PressureExecutionImageWGSLOptions {
  readonly layout: SparseCM12PressureExecutionImageLayout;
  readonly arenaName?: string;
  readonly sourcePrefix?: string;
}

const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`Invalid PEI1 WGSL identifier ${value}`);
  }
  return value;
};

/**
 * Binding-free pressure PEI1 compiler. The resident supplies accepted generation
 * adapters plus BrickCount, BrickLive, BrickDeactivate, HierarchyCount,
 * HierarchyToken, HierarchyLive and HierarchyDeactivate under `sourcePrefix`.
 */
export function createSparseCM12PressureExecutionImageWGSL(
  options: SparseCM12PressureExecutionImageWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "fineSamples");
  const source = identifier(options.sourcePrefix ?? "peiSource");
  const h = SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER;
  const at = (word: number) => `${layout.baseWords + word}u`;
  return /* wgsl */ `
const PEI_MAGIC:u32=0x${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_MAGIC.toString(16)}u;
const PEI_VERSION:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_VERSION}u;
const PEI_HEADER_WORDS:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_HEADER_WORDS}u;
const PEI_INVALID:u32=0x${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_INVALID.toString(16)}u;
const PEI_PHASE_UNINITIALIZED:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE.uninitialized}u;
const PEI_PHASE_COMPILING:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE.compiling}u;
const PEI_PHASE_ACCEPTED:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE.accepted}u;
const PEI_PHASE_FAULT:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_PHASE.fault}u;
const PEI_FAULT_HEADER:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT.invalidHeader}u;
const PEI_FAULT_PHASE:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT.invalidPhase}u;
const PEI_FAULT_BRICK_CAPACITY:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT.brickCapacity}u;
const PEI_FAULT_HIERARCHY_CAPACITY:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT.hierarchyCapacity}u;
const PEI_FAULT_CELL_CAPACITY:u32=${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT.cellCapacity}u;
const PEI_CELL_CAPACITY:u32=${layout.cellCapacity}u;
const PEI_BRICK_CAPACITY:u32=${layout.brickCapacity}u;
const PEI_HIERARCHY_CAPACITY:u32=${layout.hierarchyCapacity}u;
const PEI_PRESSURE_CELLS:u32=${layout.pressureCellBaseWords}u;
const PEI_PRESSURE_MEMBERSHIP:u32=${layout.pressureMembershipBaseWords}u;
const PEI_PRESSURE_MEMBERSHIP_WORDS:u32=${layout.pressureMembershipWordCount}u;
const PEI_WET_BRICKS:u32=${layout.wetBrickBaseWords}u;
const PEI_HIERARCHY_TOKENS:u32=${layout.hierarchyTokenBaseWords}u;

fn peiHeaderValid()->bool{return arrayLength(&${arena})>=${layout.totalWords}u
  &&${arena}[${at(h.magic)}]==PEI_MAGIC&&${arena}[${at(h.version)}]==PEI_VERSION
  &&${arena}[${at(h.headerWords)}]==PEI_HEADER_WORDS
  &&${arena}[${at(h.totalWords)}]==${layout.totalWords}u;}
fn peiFail(code:u32,id:u32){${arena}[${at(h.fault)}]=code;
  ${arena}[${at(h.firstFaultId)}]=id;
  ${arena}[${at(h.cellIndirectX)}]=0u;
  ${arena}[${at(h.brickIndirectX)}]=0u;
  ${arena}[${at(h.brickReductionIndirectX)}]=0u;
  ${arena}[${at(h.hierarchyIndirectX)}]=0u;
  ${arena}[${at(h.hierarchyReductionIndirectX)}]=0u;
  ${arena}[${at(h.phase)}]=PEI_PHASE_FAULT;}

@compute @workgroup_size(1)
fn beginSparseCM12PressureExecutionImage(){
  if(!peiHeaderValid()){peiFail(PEI_FAULT_HEADER,PEI_INVALID);return;}
  let phase=${arena}[${at(h.phase)}];
  if(phase!=PEI_PHASE_UNINITIALIZED&&phase!=PEI_PHASE_ACCEPTED){
    peiFail(PEI_FAULT_PHASE,phase);return;}
  let cells=${source}CellCount();let bricks=${source}BrickCount();
  let hierarchy=${source}HierarchyCount();
  if(cells>PEI_CELL_CAPACITY){peiFail(PEI_FAULT_CELL_CAPACITY,cells);return;}
  if(bricks>PEI_BRICK_CAPACITY){peiFail(PEI_FAULT_BRICK_CAPACITY,bricks);return;}
  if(hierarchy>PEI_HIERARCHY_CAPACITY){
    peiFail(PEI_FAULT_HIERARCHY_CAPACITY,hierarchy);return;}
  let topologyGeneration=${source}TopologyGeneration();
  let pcmCellGeneration=${source}PCMCellGeneration();
  let pcmRowGeneration=${source}PCMRowGeneration();
  let coefficientGeneration=${source}CoefficientCandidateGeneration();
  ${arena}[${at(h.fault)}]=0u;${arena}[${at(h.firstFaultId)}]=PEI_INVALID;
  ${arena}[${at(h.topologyGeneration)}]=topologyGeneration;
  ${arena}[${at(h.pcmGeneration)}]=pcmCellGeneration;
  ${arena}[${at(h.pcmRowGeneration)}]=pcmRowGeneration;
  ${arena}[${at(h.coefficientGeneration)}]=coefficientGeneration;
  ${arena}[${at(h.brickCount)}]=bricks;
  ${arena}[${at(h.pressureCellCount)}]=cells;
  ${arena}[${at(h.wetBrickCount)}]=0u;
  ${arena}[${at(h.hierarchyCount)}]=0u;
  // This triplet schedules the one-time canonical PCM rank-select publisher.
  // Iterative consumers still receive a fresh fail-closed copy after finalize.
  ${arena}[${at(h.cellIndirectX)}]=(cells+63u)/64u;
  ${arena}[${at(h.brickIndirectX)}]=0u;
  ${arena}[${at(h.brickReductionIndirectX)}]=0u;
  ${arena}[${at(h.hierarchyIndirectX)}]=0u;
  ${arena}[${at(h.hierarchyReductionIndirectX)}]=0u;
  ${arena}[${at(h.phase)}]=PEI_PHASE_COMPILING;
}

var<workgroup> peiCompactScan:array<u32,64>;
var<workgroup> peiCompactBase:u32;
var<workgroup> peiFinalizeActive:u32;
var<workgroup> peiFinalizeBrickCount:u32;
var<workgroup> peiFinalizeHierarchyCount:u32;

fn peiCompactPrefix(lane:u32,selected:u32)->u32{
  peiCompactScan[lane]=selected;workgroupBarrier();
  var width=1u;loop{
    if(width>=64u){break;}
    var add=0u;if(lane>=width){add=peiCompactScan[lane-width];}
    workgroupBarrier();peiCompactScan[lane]+=add;workgroupBarrier();width*=2u;
  }
  return peiCompactScan[lane]-selected;
}

// One workgroup retains the exact stable ascending lists authored by PEI1,
// while evaluating each stable 64-item chunk in parallel. The staged phase
// and loop bounds make every barrier reachable through workgroup-uniform flow.
@compute @workgroup_size(64)
fn finalizeSparseCM12PressureExecutionImage(
 @builtin(local_invocation_index)lane:u32){
  if(lane==0u){
    peiFinalizeActive=select(0u,1u,
      ${arena}[${at(h.phase)}]==PEI_PHASE_COMPILING);
    peiFinalizeBrickCount=${arena}[${at(h.brickCount)}];
    peiFinalizeHierarchyCount=${source}HierarchyCount();
    peiCompactBase=0u;
  }
  if(workgroupUniformLoad(&peiFinalizeActive)==0u){return;}
  let brickCount=workgroupUniformLoad(&peiFinalizeBrickCount);
  let hierarchyCount=workgroupUniformLoad(&peiFinalizeHierarchyCount);

  for(var first=0u;first<brickCount;first+=64u){
    let brick=first+lane;let valid=brick<brickCount;
    var selected=0u;
    if(valid){
      let live=${source}BrickLive(brick);selected=select(0u,1u,live);
      if(!live){${source}BrickDeactivate(brick);}
    }
    let prefix=peiCompactPrefix(lane,selected);
    let base=workgroupUniformLoad(&peiCompactBase);
    if(selected!=0u){${arena}[PEI_WET_BRICKS+base+prefix]=brick;}
    workgroupBarrier();
    if(lane==0u){peiCompactBase=base+peiCompactScan[63];}
    workgroupBarrier();
  }
  if(lane==0u){
    ${arena}[${at(h.wetBrickCount)}]=peiCompactBase;
    peiCompactBase=0u;
  }
  workgroupBarrier();

  for(var first=0u;first<hierarchyCount;first+=64u){
    let linear=first+lane;let valid=linear<hierarchyCount;
    var selected=0u;
    if(valid){
      let live=${source}HierarchyLive(linear);selected=select(0u,1u,live);
      if(!live){${source}HierarchyDeactivate(linear);}
    }
    let prefix=peiCompactPrefix(lane,selected);
    let base=workgroupUniformLoad(&peiCompactBase);
    if(selected!=0u){
      ${arena}[PEI_HIERARCHY_TOKENS+base+prefix]=${source}HierarchyToken(linear);
    }
    workgroupBarrier();
    if(lane==0u){peiCompactBase=base+peiCompactScan[63];}
    workgroupBarrier();
  }
  workgroupBarrier();
  if(lane==0u){
    let wetCount=${arena}[${at(h.wetBrickCount)}];
    let liveHierarchyCount=peiCompactBase;
    ${arena}[${at(h.hierarchyCount)}]=liveHierarchyCount;
    let topologyGeneration=${arena}[${at(h.topologyGeneration)}];
    let pcmCellGeneration=${arena}[${at(h.pcmGeneration)}];
    let pcmRowGeneration=${arena}[${at(h.pcmRowGeneration)}];
    let coefficientGeneration=${arena}[${at(h.coefficientGeneration)}];
    if(${source}TopologyGeneration()!=topologyGeneration
      ||${source}PCMCellGeneration()!=pcmCellGeneration
      ||${source}PCMRowGeneration()!=pcmRowGeneration
      ||${source}CoefficientAcceptedGeneration()!=coefficientGeneration){
      peiFail(${SPARSE_CM12_PRESSURE_EXECUTION_IMAGE_FAULT.invalidSource}u,PEI_INVALID);
      return;
    }
    ${arena}[${at(h.cellIndirectX)}]=(${arena}[${at(h.pressureCellCount)}]+63u)/64u;
    ${arena}[${at(h.brickIndirectX)}]=(wetCount+63u)/64u;
    ${arena}[${at(h.brickReductionIndirectX)}]=wetCount;
    ${arena}[${at(h.hierarchyIndirectX)}]=(liveHierarchyCount+63u)/64u;
    ${arena}[${at(h.hierarchyReductionIndirectX)}]=liveHierarchyCount;
    ${arena}[${at(h.generation)}]=${arena}[${at(h.generation)}]+1u;
    ${arena}[${at(h.acceptedReceipts)}]=${arena}[${at(h.acceptedReceipts)}]+1u;
    ${arena}[${at(h.phase)}]=PEI_PHASE_ACCEPTED;
  }
}

// Failure publishes x=0 to every copied indirect triplet. Iterative consumers
// therefore trust publication and need only guard the final partial workgroup.
fn peiWetBrickCount()->u32{return ${arena}[${at(h.wetBrickCount)}];}
fn peiPublicationOpen()->bool{
  return ${arena}[${at(h.phase)}]==PEI_PHASE_COMPILING
    &&${arena}[${at(h.fault)}]==0u;
}
fn peiPressureCellCount()->u32{return ${arena}[${at(h.pressureCellCount)}];}
fn peiPressureCell(linear:u32)->u32{
  if(linear>=peiPressureCellCount()){return PEI_INVALID;}
  return ${arena}[PEI_PRESSURE_CELLS+linear];}
fn peiPressureCellMember(cell:u32)->bool{
  return cell<PEI_CELL_CAPACITY
    &&(${arena}[PEI_PRESSURE_MEMBERSHIP+(cell>>5u)]
      &(1u<<(cell&31u)))!=0u;
}
fn peiWetBrick(linear:u32)->u32{
  if(linear>=peiWetBrickCount()){return PEI_INVALID;}
  return ${arena}[PEI_WET_BRICKS+linear];}
fn peiHierarchyCount()->u32{return ${arena}[${at(h.hierarchyCount)}];}
fn peiHierarchyToken(linear:u32)->u32{
  if(linear>=peiHierarchyCount()){return PEI_INVALID;}
  return ${arena}[PEI_HIERARCHY_TOKENS+linear];}
`;
}
