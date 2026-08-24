import {
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_MAGIC,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_STAGE,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_VERSION,
  type SparseCM12FramePlanPresentationLayout,
} from "./sparse-cm12-frame-plan-presentation";
import { SPARSE_CM12_FRAME_PLAN_BRICK } from "../../core/sparse-cm12-frame-plan";

export interface SparseCM12FramePlanPresentationWGSLOptions {
  readonly layout: SparseCM12FramePlanPresentationLayout;
  /** Existing FPL1 helper prefix emitted by createSparseCM12FramePlanWGSL. */
  readonly framePlanPrefix?: string;
  /** Existing array<atomic<u32>> FPP1 arena. */
  readonly packetArenaName?: string;
  /** Integration hook prefix; see the contract returned in generated WGSL. */
  readonly hookPrefix?: string;
  readonly prefix?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/** Binding-free FPP1 packet builder and transactional page executor. */
export function createSparseCM12FramePlanPresentationWGSL(
  options: SparseCM12FramePlanPresentationWGSLOptions,
): string {
  const layout = options.layout;
  const fpl = identifier(options.framePlanPrefix ?? "cm12FramePlan", "framePlanPrefix");
  const arena = identifier(options.packetArenaName ?? "presentationPacket", "packetArenaName");
  const hook = identifier(options.hookPrefix ?? "cm12Presentation", "hookPrefix");
  const p = identifier(options.prefix ?? "cm12Fpp", "prefix");
  const h = (word: number) => `${layout.baseWords + word}u`;
  const allowedCauses = Object.values(SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE)
    .reduce((mask, bit) => mask | bit, 0) >>> 0;
  const lifecycleCauses = (SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE.topologyCreated
    | SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE.topologyRetired
    | SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE.pageActivated
    | SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE.pageRetired) >>> 0;
  const directCauses = (allowedCauses
    & ~SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE.dependencyClosure) >>> 0;
  return /* wgsl */ `
// Integration hooks are deliberately external to FPP1:
//   ${hook}PageMatches(page,brick,logicalKey,topologyGeneration)->bool
//   ${hook}PreparePage(brick,page,lane,generation,causeMask)->u32 status
//   ${hook}ExactSample(brick,page,tile,sample,generation,causeMask)->vec2u(payload,status)
//   ${hook}StoreCandidate(page,localIndex,generation,payload)
//   ${hook}LoadCandidate(page,localIndex,generation)->u32
//   ${hook}StoreAccepted(page,localIndex,payload)
//   ${hook}CommitCandidate(page,generation)
//   ${hook}RejectAccepted(page)
// Candidate storage must not alias accepted samples. CommitCandidate publishes
// page generation only after the dedicated copy kernel has completed.
const ${p}Magic:u32=0x${SPARSE_CM12_FRAME_PLAN_PRESENTATION_MAGIC.toString(16)}u;
const ${p}Version:u32=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_VERSION}u;
const ${p}Invalid:u32=0xffffffffu;
const ${p}Capacity:u32=${layout.pageCapacity}u;
const ${p}BrickCapacity:u32=${layout.brickCapacity}u;
const ${p}Tiles:u32=${layout.tilesPerPage}u;
const ${p}PageResolution:u32=${layout.pageResolution}u;
const ${p}Packet:u32=${layout.packetIndex}u;
const ${p}Stage:u32=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_STAGE}u;
const ${p}AllowedCauses:u32=${allowedCauses}u;
const ${p}DirectCauses:u32=${directCauses}u;
const ${p}LifecycleCauses:u32=${lifecycleCauses}u;
const ${p}List:u32=${layout.listBaseWords}u;
const ${p}BrickPages:u32=${layout.brickPagesBaseWords}u;
const ${p}Records:u32=${layout.recordsBaseWords}u;
var<workgroup>${p}DirtyLow:atomic<u32>;
var<workgroup>${p}DirtyHigh:atomic<u32>;
var<workgroup>${p}Causes:atomic<u32>;
var<workgroup>${p}Fault:atomic<u32>;
var<workgroup>${p}FaultTile:atomic<u32>;
var<workgroup>${p}ExecutedLow:atomic<u32>;
var<workgroup>${p}ExecutedHigh:atomic<u32>;

fn ${p}Load(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn ${p}Store(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn ${p}Record(brick:u32)->u32{
  return ${p}Records+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS}u*brick;
}
fn ${p}HeaderValid()->bool{
  return arrayLength(&${arena})>=${layout.totalWords}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.magic)})==${p}Magic
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.version)})==${p}Version
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.headerWords)})
      ==${SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.pageWords)})
      ==${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.brickFineResolution)})
      ==${layout.brickFineResolution}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.pageResolution)})
      ==${layout.pageResolution}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.tilesPerPage)})==${p}Tiles
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.pageCapacity)})==${p}Capacity
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.packetIndex)})==${p}Packet
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.listBase)})==${p}List
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.brickPagesBase)})
      ==${p}BrickPages
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.recordsBase)})==${p}Records
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.totalWords)})
      ==${layout.totalWords}u;
}
fn ${p}FirstFault(brick:u32,tile:u32,cause:u32){
  let won=atomicCompareExchangeWeak(&${arena}[
    ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.firstFaultBrick)}],
    ${p}Invalid,brick).exchanged;
  if(won){
    ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.firstFaultTile)},tile);
    ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.firstFaultCause)},cause);
  }
}
fn ${p}Omit(brick:u32,tile:u32,cause:u32,code:u32){
  let record=${p}Record(brick);
  ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.faultCode}u,code);
  ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.firstFaultTile}u,tile);
  atomicAnd(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u],
    ~${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.executing}u);
  atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u],
    ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.omitted
      | SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.localFault}u);
  atomicOr(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)}],
    ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.localFaults}u);
  atomicAdd(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.omittedPageCount)}],1u);
  ${p}FirstFault(brick,tile,cause);
}
fn ${p}GlobalFail(code:u32){
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.faultCode)},code);
  atomicOr(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)}],
    ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.globalFault}u);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectX)},0u);
}
fn ${p}MaskContains(low:u32,high:u32,tile:u32)->bool{
  return (select(low,high,tile>=32u)&(1u<<(tile&31u)))!=0u;
}
fn ${p}LocalIndex(tile:u32,sample:u32)->u32{
  let axis=${p}PageResolution/4u;
  let tz=tile/(axis*axis);let tr=tile-tz*axis*axis;let ty=tr/axis;let tx=tr-ty*axis;
  let sz=sample/16u;let sr=sample-sz*16u;let sy=sr/4u;let sx=sr-sy*4u;
  let q=vec3u(4u*tx+sx,4u*ty+sy,4u*tz+sz);
  return q.x+${p}PageResolution*(q.y+${p}PageResolution*q.z);
}

@compute @workgroup_size(1)
fn beginSparseCM12FramePlanPresentationPacket(){
  if(!${p}HeaderValid()||!${fpl}HeaderValid()){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.invalidHeader}u);return;
  }
  let generation=${fpl}AcceptedFrameGeneration();let slot=${fpl}CurrentSlot();
  let topologyGeneration=${fpl}Load(${fpl}SlotBase(slot)+1u);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.acceptedGeneration)},generation);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.topologyGeneration)},topologyGeneration);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)},
    ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.initialized
      | SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.open}u);
  for(var at=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.faultCode}u;
      at<=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.publishedPageCount}u;at+=1u){
    if(at==${SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.packetIndex}u
      ||at==${SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.listBase}u
      ||at==${SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.recordsBase}u
      ||at==${SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.totalWords}u){continue;}
    ${p}Store(${layout.baseWords}u+at,0u);
  }
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectY)},1u);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectZ)},1u);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.firstFaultBrick)},${p}Invalid);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.firstFaultTile)},${p}Invalid);
}

@compute @workgroup_size(64)
fn buildSparseCM12FramePlanPresentationPacket(
  @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let brick=wid.x;let resident=brick<${p}BrickCapacity;
  if(lane==0u){
    atomicStore(&${p}DirtyLow,0u);atomicStore(&${p}DirtyHigh,0u);
    atomicStore(&${p}Causes,0u);atomicStore(&${p}Fault,${p}Invalid);
    atomicStore(&${p}FaultTile,${p}Invalid);
    if(resident){
      let record=${p}Record(brick);
      for(var word=0u;word<${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_WORDS}u;word+=1u){
        ${p}Store(record+word,0u);
      }
      ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.firstFaultTile}u,${p}Invalid);
      ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.page}u,${p}Invalid);
    }
  }
  workgroupBarrier();
  if(resident&&lane<${p}Tiles){
    let slot=${fpl}CurrentSlot();let brickAt=${fpl}BrickAt(slot,brick);
    let validLow=${fpl}Load(brickAt+${SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskLow}u);
    let validHigh=${fpl}Load(brickAt+${SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskHigh}u);
    if(${p}MaskContains(validLow,validHigh,lane)){
      let tile=${fpl}OverlayTileAt(brick,lane,false);
      let bit=1u<<${p}Stage;let direct=(tile.directStages&bit)!=0u;
      let closure=(tile.closureStages&bit)!=0u;
      let scheduled=${fpl}CurrentTileScheduled(brick,lane,${p}Stage);
      // FPL1 causes are shared by all logical stages in the hot tile. Select
      // the presentation dimension here; causes belonging only to another
      // coalesced stage must neither dirty nor invalidate presentation.
      let causes=(tile.originCauses|tile.inheritedCauses)&${p}AllowedCauses;
      let depth=(tile.packedClosureDepths>>(4u*${p}Stage))&15u;
      if(scheduled){
        if(lane<32u){atomicOr(&${p}DirtyLow,1u<<lane);}
        else{atomicOr(&${p}DirtyHigh,1u<<(lane-32u));}
        atomicOr(&${p}Causes,causes);
      }
      var fault=${p}Invalid;
      if(!tile.valid){fault=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.localFramePlan}u;}
      else if(scheduled!=(direct||closure)||(direct&&closure)){
        fault=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.dirtyMaskMismatch}u;
      }else if(scheduled&&((tile.uncoveredStages&bit)!=0u)){
        fault=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.uncoveredTile}u;
      }else if(scheduled&&(causes==0u
        ||(direct&&(causes&${p}DirectCauses)==0u)
        ||(direct&&depth!=0u)||(closure&&(depth==0u
          ||(causes&${SPARSE_CM12_FRAME_PLAN_PRESENTATION_CAUSE.dependencyClosure}u)==0u)))){
        fault=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.unsupportedCause}u;
      }
      if(fault!=${p}Invalid){
        atomicMin(&${p}Fault,fault);atomicMin(&${p}FaultTile,lane);
      }
    }
  }
  workgroupBarrier();
  if(lane==0u&&resident){
    let record=${p}Record(brick);let generation=${fpl}AcceptedFrameGeneration();
    let topologyGeneration=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.topologyGeneration)});
    let dirtyLow=atomicLoad(&${p}DirtyLow);let dirtyHigh=atomicLoad(&${p}DirtyHigh);
    let causes=atomicLoad(&${p}Causes);var fault=atomicLoad(&${p}Fault);
    let slot=${fpl}CurrentSlot();let brickAt=${fpl}BrickAt(slot,brick);
    let key=${fpl}Load(brickAt+${SPARSE_CM12_FRAME_PLAN_BRICK.logicalBrickKey}u);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u,generation);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.topologyGeneration}u,
      topologyGeneration);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.dirtyMaskLow}u,dirtyLow);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.dirtyMaskHigh}u,dirtyHigh);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.causeMask}u,causes);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.logicalBrickKey}u,key);
    if((dirtyLow|dirtyHigh)!=0u){
      // Count the complete expected set before any local validation. Omitted
      // pages belong to the transaction and can never disappear from its
      // coverage denominator.
      atomicAdd(&${arena}[
        ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.dirtyPageCount)}],1u);
      let validLow=${fpl}Load(brickAt+${SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskLow}u);
      let validHigh=${fpl}Load(brickAt+${SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskHigh}u);
      if((causes&${p}LifecycleCauses)!=0u
        &&(dirtyLow!=validLow||dirtyHigh!=validHigh)){
        fault=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.lifecycleCoverage}u;
      }
      if(!${fpl}CurrentStageInPacket(brick,${p}Stage,${p}Packet)){
        fault=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.dirtyMaskMismatch}u;
      }
      let page=${p}Load(${p}BrickPages+brick);
      if(page==${p}Invalid||!${hook}PageMatches(page,brick,key,topologyGeneration)){
        fault=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.invalidPage}u;
      }
      ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.page}u,page);
      if(fault!=${p}Invalid){
        ${p}Omit(brick,atomicLoad(&${p}FaultTile),causes,fault);
      }else{
        let index=atomicAdd(&${arena}[
          ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectX)}],1u);
        if(index>=${p}Capacity){
          ${p}Omit(brick,${p}Invalid,causes,
            ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.listCapacity}u);
          ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.listCapacity}u);
        }else{
          ${p}Store(${p}List+index,brick);
          let lifecycle=select(0u,
            ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.lifecycle}u,
            (causes&${p}LifecycleCauses)!=0u);
          ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u,
            ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.scheduled}u|lifecycle);
        }
      }
    }
  }
}

@compute @workgroup_size(1)
fn finalizeSparseCM12FramePlanPresentationPacket(){
  if(!${p}HeaderValid()||(${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)})
    &${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.globalFault}u)!=0u){
    ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectX)},0u);return;
  }
  atomicAnd(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)}],
    ~${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.open}u);
  atomicOr(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)}],
    ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.sealed}u);
}

@compute @workgroup_size(64)
fn executeSparseCM12FramePlanPresentationPacket(
  @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let count=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectX)});
  var resident=wid.x<count;var brick=${p}Invalid;var record=${p}Records;var page=${p}Invalid;
  if(resident){brick=${p}Load(${p}List+wid.x);resident=brick<${p}BrickCapacity;}
  if(resident){record=${p}Record(brick);page=${p}Load(record+
    ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.page}u);}
  if(lane==0u){
    atomicStore(&${p}ExecutedLow,0u);atomicStore(&${p}ExecutedHigh,0u);
    atomicStore(&${p}Fault,${p}Invalid);atomicStore(&${p}FaultTile,${p}Invalid);
    if(resident){
      let recordFlags=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u);
      if(${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u)
          !=${fpl}AcceptedFrameGeneration()
        ||${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u)
          !=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.acceptedGeneration)})
        ||${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.topologyGeneration}u)
          !=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.topologyGeneration)})
        ||(recordFlags&${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.scheduled}u)==0u
        ||(recordFlags&${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.localFault}u)!=0u){
        atomicStore(&${p}Fault,
          ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.generationMismatch}u);
      }else{atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u],
        ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.executing}u);}
    }
  }
  workgroupBarrier();
  // PreparePage owns workgroup barriers. Invoke it uniformly with a harmless
  // page-zero tuple for an invalid defensive workgroup; indirect x normally
  // makes that branch unreachable, but validation cannot infer buffer values.
  let prepareBrick=select(0u,brick,resident);
  let preparePage=select(0u,page,resident);
  let prepareRecord=select(${p}Records,record,resident);
  let prepareStatus=${hook}PreparePage(prepareBrick,preparePage,lane,
    ${p}Load(prepareRecord+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u),
    ${p}Load(prepareRecord+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.causeMask}u));
  if(resident&&atomicLoad(&${p}Fault)==${p}Invalid&&prepareStatus!=0u){
      atomicMin(&${p}Fault,prepareStatus);
      atomicMin(&${p}FaultTile,${p}Invalid);}
  workgroupBarrier();
  if(resident&&lane<${p}Tiles){
    let dirtyLow=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.dirtyMaskLow}u);
    let dirtyHigh=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.dirtyMaskHigh}u);
    if(${p}MaskContains(dirtyLow,dirtyHigh,lane)&&atomicLoad(&${p}Fault)==${p}Invalid){
      var tileValid=true;
      for(var sample=0u;sample<${SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE}u;
          sample+=1u){
        let exact=${hook}ExactSample(brick,page,lane,sample,
          ${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u),
          ${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.causeMask}u));
        if(exact.y!=0u){tileValid=false;atomicMin(&${p}Fault,exact.y);
          atomicMin(&${p}FaultTile,lane);}
        if(tileValid){${hook}StoreCandidate(page,${p}LocalIndex(lane,sample),
          ${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u),exact.x);}
      }
      if(tileValid){
        if(lane<32u){atomicOr(&${p}ExecutedLow,1u<<lane);}
        else{atomicOr(&${p}ExecutedHigh,1u<<(lane-32u));}
      }
    }
  }
  storageBarrier();workgroupBarrier();
  if(lane==0u&&resident){
    let dirtyLow=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.dirtyMaskLow}u);
    let dirtyHigh=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.dirtyMaskHigh}u);
    let executedLow=atomicLoad(&${p}ExecutedLow);let executedHigh=atomicLoad(&${p}ExecutedHigh);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.executedMaskLow}u,executedLow);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.executedMaskHigh}u,executedHigh);
    if(atomicLoad(&${p}Fault)!=${p}Invalid||dirtyLow!=executedLow||dirtyHigh!=executedHigh){
      atomicAdd(&${arena}[
        ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.coverageFaultCount)}],1u);
      ${p}Omit(brick,atomicLoad(&${p}FaultTile),
        ${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.causeMask}u),
        ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.candidateWriteCoverage}u);
    }else{
      atomicAnd(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u],
        ~${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.executing}u);
      atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u],
        ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.candidateReady}u);
      atomicAdd(&${arena}[
        ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.executedPageCount)}],1u);
      atomicAdd(&${arena}[
        ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.executedTileCount)}],
        countOneBits(executedLow)+countOneBits(executedHigh));
      let samples=${SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE}u
        *(countOneBits(executedLow)+countOneBits(executedHigh));
      let previous=atomicAdd(&${arena}[
        ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.exactSampleCountLow)}],samples);
      if(previous>0xffffffffu-samples){atomicAdd(&${arena}[
        ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.exactSampleCountHigh)}],1u);}
    }
  }
}

@compute @workgroup_size(64)
fn commitSparseCM12FramePlanPresentationPacket(
  @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let count=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.indirectX)});
  var resident=wid.x<count;var brick=${p}Invalid;var record=${p}Records;var page=${p}Invalid;
  if(resident){brick=${p}Load(${p}List+wid.x);}
  if(resident&&brick<${p}BrickCapacity){
    record=${p}Record(brick);page=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.page}u);
  }else{resident=false;}
  if(resident){
    let flags=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u);
    resident=(flags&${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.candidateReady}u)!=0u
      &&(flags&${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.localFault}u)==0u
      &&${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u)
        ==${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.acceptedGeneration)});
  }
  if(resident&&lane<${p}Tiles){
    let executedLow=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.executedMaskLow}u);
    let executedHigh=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.executedMaskHigh}u);
    if(${p}MaskContains(executedLow,executedHigh,lane)){
      let generation=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u);
      for(var sample=0u;sample<${SPARSE_CM12_FRAME_PLAN_PRESENTATION_SAMPLES_PER_TILE}u;
          sample+=1u){
        let localIndex=${p}LocalIndex(lane,sample);
        ${hook}StoreAccepted(page,localIndex,
          ${hook}LoadCandidate(page,localIndex,generation));
      }
    }
  }
  storageBarrier();workgroupBarrier();
  if(lane==0u&&resident){
    let generation=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.generation}u);
    ${hook}CommitCandidate(page,generation);
    atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u],
      ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.published}u);
    atomicAdd(&${arena}[
      ${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.publishedPageCount)}],1u);
  }
  workgroupBarrier();
  if(resident&&lane<${p}Tiles){
    let executedLow=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.executedMaskLow}u);
    let executedHigh=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.executedMaskHigh}u);
    if(${p}MaskContains(executedLow,executedHigh,lane)){
      ${fpl}MarkCurrentTileExecuted(brick,lane,${p}Stage);
    }
  }
}

@compute @workgroup_size(1)
fn finalizeSparseCM12FramePlanPresentationExecution(){
  let expected=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.dirtyPageCount)});
  let executed=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.executedPageCount)});
  let published=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.publishedPageCount)});
  let omitted=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.omittedPageCount)});
  let flags=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)});
  if(omitted!=0u||executed+omitted!=expected||published!=executed
    ||(flags&(${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.localFaults
      | SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.globalFault}u))!=0u){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FAULT.candidateWriteCoverage}u);return;
  }
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.generationReceipt)},
    ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.acceptedGeneration)}));
  atomicOr(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)}],
    ${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.executionComplete}u);
}

// The renderer-visible metadata is the acceptance gate. On any packet fault,
// invalidate every affected physical page with one fixed brick dispatch. This
// is a bounded metadata pass, not a sample-domain fallback or repair path.
@compute @workgroup_size(1)
fn rejectSparseCM12FramePlanPresentationFaults(@builtin(workgroup_id)wid:vec3u){
  let brick=wid.x;if(brick>=${p}BrickCapacity){return;}
  let flags=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER.flags)});
  let record=${p}Record(brick);
  let recordFlags=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE.flags}u);
  if((flags&${SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG.globalFault}u)!=0u
    ||(recordFlags&(${SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.localFault
      | SPARSE_CM12_FRAME_PLAN_PRESENTATION_PAGE_FLAG.omitted}u))!=0u){
    let page=${p}Load(${p}BrickPages+brick);
    if(page!=${p}Invalid){${hook}RejectAccepted(page);}
  }
}
`;
}
