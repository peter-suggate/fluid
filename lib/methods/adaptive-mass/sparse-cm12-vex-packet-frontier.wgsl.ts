import {
  SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS,
  SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER,
  SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER_WORDS,
  SPARSE_CM12_VEX_PACKET_FRONTIER_MAGIC,
  SPARSE_CM12_VEX_PACKET_FRONTIER_PHASE,
  SPARSE_CM12_VEX_PACKET_FRONTIER_VERSION,
  type SparseCM12VexPacketFrontierLayout,
} from "./sparse-cm12-vex-packet-frontier";
import {
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  type SparseCM12VelocityExtensionLayout,
} from "./sparse-cm12-velocity-extension";

export interface SparseCM12VexPacketFrontierWGSLOptions {
  readonly layout: SparseCM12VexPacketFrontierLayout;
  readonly velocityExtensionLayout: SparseCM12VelocityExtensionLayout;
  readonly arenaName?: string;
  readonly generationExpression: string;
  readonly topologyGenerationExpression: string;
  readonly topologySlotExpression: string;
  /** `(packet,lane,slot)->cell`, INVALID for a nonresident lane. */
  readonly packetCellFunction: string;
  /** `(cell,slot)->vec2u(packet,lane)`, INVALID packet when unmapped. */
  readonly cellPacketLaneFunction: string;
  /** `(cell,slot)->bool` in the frozen target TEI slot. */
  readonly targetCellActiveFunction: string;
  /** Optional `(cell)->bool` guard for ordinary, non-pending producers. */
  readonly currentCellActiveFunction?: string;
  /** Optional `(cell,cause,generation)->bool` exact provenance receipt. */
  readonly rootReceiptFunction?: string;
  /** Optional `(cell,depth,generation)->bool` exact closure receipt. */
  readonly closureReceiptFunction?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError(`${label} must be an identifier`);
  return value;
};
const expression = (value: string, label: string): string => {
  if (value.trim().length === 0) throw new TypeError(`${label} must not be empty`);
  return value;
};

/**
 * Binding-free VXP1 producer, packet-frontier and blast-worklist helpers.
 * The resident supplies only TEI packet/cell mapping and the existing activity
 * arena. Expansion itself is supplied by the compiled local operator and calls
 * `vxp1MergeFrontierTarget`; no incidence walk is embedded here.
 */
export function createSparseCM12VexPacketFrontierWGSL(
  options: SparseCM12VexPacketFrontierWGSLOptions,
): string {
  const layout = options.layout, vex = options.velocityExtensionLayout;
  const arena = identifier(options.arenaName ?? "activity", "arenaName");
  const packetCell = identifier(options.packetCellFunction, "packetCellFunction");
  const cellPacket = identifier(options.cellPacketLaneFunction, "cellPacketLaneFunction");
  const targetCellActive = identifier(
    options.targetCellActiveFunction, "targetCellActiveFunction",
  );
  const currentCellActive = options.currentCellActiveFunction
    ? identifier(options.currentCellActiveFunction, "currentCellActiveFunction") : undefined;
  const rootReceipt = options.rootReceiptFunction
    ? identifier(options.rootReceiptFunction, "rootReceiptFunction") : undefined;
  const closureReceipt = options.closureReceiptFunction
    ? identifier(options.closureReceiptFunction, "closureReceiptFunction") : undefined;
  const generation = expression(options.generationExpression, "generationExpression");
  const topologyGeneration = expression(
    options.topologyGenerationExpression, "topologyGenerationExpression",
  );
  const topologySlot = expression(options.topologySlotExpression, "topologySlotExpression");
  const h = (word: number) => `${layout.headerBaseWords + word}u`;
  const vh = (word: number) => `${vex.headerBaseWords + word}u`;
  const causeLow = layout.rootCauseMaskLowBaseWords.map((value) => `${value}u`).join(",");
  const causeHigh = layout.rootCauseMaskHighBaseWords.map((value) => `${value}u`).join(",");
  const rootHook = rootReceipt
    ? `if(!${rootReceipt}(cell,cause,generation)){vxp1Fault(packet,lane);}` : "";
  const closureHook = closureReceipt
    ? `if(!${closureReceipt}(cell,VXP1_FRONTIER_DEPTH,generation)){vxp1Fault(packet,lane);return;}`
    : "";
  return /* wgsl */ `
const vxp1Magic:u32=0x${SPARSE_CM12_VEX_PACKET_FRONTIER_MAGIC.toString(16)}u;
const vxp1Version:u32=${SPARSE_CM12_VEX_PACKET_FRONTIER_VERSION}u;
const vxp1HeaderWords:u32=${SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER_WORDS}u;
const vxp1Invalid:u32=0xffffffffu;
const vxp1Claim:u32=0x80000000u;
const vxp1PacketCapacity:u32=${layout.packetCapacity}u;
const vxp1PhaseCollecting:u32=${SPARSE_CM12_VEX_PACKET_FRONTIER_PHASE.collecting}u;
const vxp1PhaseRootsSealed:u32=${SPARSE_CM12_VEX_PACKET_FRONTIER_PHASE.rootsSealed}u;
const vxp1PhasePlanning:u32=${SPARSE_CM12_VEX_PACKET_FRONTIER_PHASE.planning}u;
const vxp1PhasePlanned:u32=${SPARSE_CM12_VEX_PACKET_FRONTIER_PHASE.planned}u;
const vxp1PhaseFault:u32=0xffffffffu;
const vxp1RootStamp:u32=${layout.rootStampBaseWords}u;
const vxp1RootPacketList:u32=${layout.rootPacketListBaseWords}u;
const vxp1CauseLow:array<u32,${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}>=array<u32,
  ${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}>(${causeLow});
const vxp1CauseHigh:array<u32,${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}>=array<u32,
  ${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}>(${causeHigh});
const vxp1FrontierAGenerationStamp:u32=${layout.frontierA.generationStampBaseWords}u;
const vxp1FrontierADepthStamp:u32=${layout.frontierA.depthStampBaseWords}u;
const vxp1FrontierACandidateList:u32=${layout.frontierA.candidateListBaseWords}u;
const vxp1FrontierALow:u32=${layout.frontierA.maskLowBaseWords}u;
const vxp1FrontierAHigh:u32=${layout.frontierA.maskHighBaseWords}u;
const vxp1FrontierAList:u32=${layout.frontierA.packetListBaseWords}u;
const vxp1FrontierBGenerationStamp:u32=${layout.frontierB.generationStampBaseWords}u;
const vxp1FrontierBDepthStamp:u32=${layout.frontierB.depthStampBaseWords}u;
const vxp1FrontierBCandidateList:u32=${layout.frontierB.candidateListBaseWords}u;
const vxp1FrontierBLow:u32=${layout.frontierB.maskLowBaseWords}u;
const vxp1FrontierBHigh:u32=${layout.frontierB.maskHighBaseWords}u;
const vxp1FrontierBList:u32=${layout.frontierB.packetListBaseWords}u;
const vxp1BlastStamp:u32=${layout.blast.stampBaseWords}u;
const vxp1BlastList:u32=${layout.blast.packetListBaseWords}u;
const vxp1BlastLow:u32=${layout.blast.maskLowBaseWords}u;
const vxp1BlastHigh:u32=${layout.blast.maskHighBaseWords}u;
const vxp1CellRootStamp:u32=${vex.rootStampBaseWords}u;
const vxp1CellRootCause:u32=${vex.rootCauseBaseWords}u;
const vxp1CellBlastStamp:u32=${vex.blastStampBaseWords}u;
const vxp1CellBlastDepth:u32=${vex.blastDepthBaseWords}u;
override VXP1_FRONTIER_DEPTH:u32=1u;
var<workgroup> vxp1SharedCauseLow:array<u32,${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}>;
var<workgroup> vxp1SharedCauseHigh:array<u32,${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}>;
var<workgroup> vxp1SharedPacket:u32;
var<workgroup> vxp1SharedLow:u32;
var<workgroup> vxp1SharedHigh:u32;

fn vxp1Load(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn vxp1Store(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn vxp1Generation()->u32{return vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.generation)});}
fn vxp1TopologySlot()->u32{return vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.topologySlot)});}
fn vxp1Fault(packet:u32,lane:u32){
  let first=atomicAdd(&${arena}[${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.faultCount)}],1u);
  if(first==0u){vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.firstFaultPacket)},packet);
    vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.firstFaultLane)},lane);}
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)},vxp1PhaseFault);
}
fn vxp1ValidHeader()->bool{return vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.magic)})
    ==vxp1Magic&&vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.version)})==vxp1Version
  &&vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.headerWords)})==vxp1HeaderWords
  &&vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.packetCapacity)})==vxp1PacketCapacity;}

@compute @workgroup_size(1) fn beginSparseCM12VexPacketFrontier(){
  let generation=${generation};let topologyGeneration=${topologyGeneration};
  let slot=${topologySlot};
  if(generation==0u||generation>=0x7ffffffeu||slot>1u){return;}
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.magic)},vxp1Magic);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.version)},vxp1Version);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.headerWords)},vxp1HeaderWords);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)},vxp1PhaseCollecting);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.generation)},generation);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.topologyGeneration)},topologyGeneration);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.topologySlot)},slot);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.packetCapacity)},vxp1PacketCapacity);
  for(var at=${SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.rootPacketCount}u;
      at<=${SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.currentDepth}u;at+=1u){
    vxp1Store(${layout.headerBaseWords}u+at,0u);}
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.firstFaultPacket)},vxp1Invalid);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.firstFaultLane)},vxp1Invalid);
}

// Reconcile only the header's frozen decode slot after the topology commit.
// Pre-flip packet masks are not decoded before this point; the changed-leaf
// producer must invalidate every packet whose lane map was replaced first.
@compute @workgroup_size(1) fn reconcileSparseCM12VexPacketTopologySlot(){
  if(!vxp1ValidHeader()||vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)})
      !=vxp1PhaseCollecting){vxp1Fault(vxp1Invalid,vxp1Invalid);return;}
  let slot=${topologySlot};
  if(slot>1u){vxp1Fault(vxp1Invalid,vxp1Invalid);return;}
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.topologyGeneration)},
    ${topologyGeneration});
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.topologySlot)},slot);
}

fn vxp1ClaimRootPacket(packet:u32)->bool{
  if(packet>=vxp1PacketCapacity||vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)})
      !=vxp1PhaseCollecting){vxp1Fault(packet,vxp1Invalid);return false;}
  let generation=vxp1Generation();
  for(var spin=0u;spin<256u;spin+=1u){
    let observed=vxp1Load(vxp1RootStamp+packet);
    if(observed==generation){return true;}
    if((observed&vxp1Claim)!=0u){continue;}
    let claim=atomicCompareExchangeWeak(&${arena}[vxp1RootStamp+packet],observed,generation|vxp1Claim);
    if(!claim.exchanged){continue;}
    for(var cause=0u;cause<${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}u;cause+=1u){
      vxp1Store(vxp1CauseLow[cause]+packet,0u);vxp1Store(vxp1CauseHigh[cause]+packet,0u);}
    let rank=atomicAdd(&${arena}[${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.rootPacketCount)}],1u);
    if(rank>=vxp1PacketCapacity){vxp1Fault(packet,vxp1Invalid);return false;}
    vxp1Store(vxp1RootPacketList+rank,packet);vxp1Store(vxp1RootStamp+packet,generation);return true;
  }
  vxp1Fault(packet,vxp1Invalid);return false;
}
fn vxp1RecordRootPacketMask(packet:u32,low:u32,high:u32,cause:u32)->bool{
  if((low|high)==0u){return true;}
  if(cause==0u||(cause&~0x3fu)!=0u||!vxp1ClaimRootPacket(packet)){vxp1Fault(packet,vxp1Invalid);return false;}
  for(var bit=0u;bit<${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}u;bit+=1u){
    if((cause&(1u<<bit))==0u){continue;}
    atomicOr(&${arena}[vxp1CauseLow[bit]+packet],low);
    atomicOr(&${arena}[vxp1CauseHigh[bit]+packet],high);
  }
  return true;
}
// Commit-time reconciliation invalidates every pre-flip transport packet in a
// changed leaf. Unchanged leaf packet/lane mappings remain bit-identical. The
// topology lifecycle bridge repopulates candidate-slot roots after the flip.
fn vxp1InvalidateChangedRootPacket(packet:u32){
  if(packet>=vxp1PacketCapacity||vxp1Load(vxp1RootStamp+packet)!=vxp1Generation()){return;}
  for(var bit=0u;bit<${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}u;bit+=1u){
    vxp1Store(vxp1CauseLow[bit]+packet,0u);vxp1Store(vxp1CauseHigh[bit]+packet,0u);}
}
fn vxp1RecordCellRoot(cell:u32,cause:u32)->bool{
  ${currentCellActive ? `if(!${currentCellActive}(cell)){return false;}` : ""}
  let address=${cellPacket}(cell,vxp1TopologySlot());
  if(address.x==vxp1Invalid||address.y>=64u||!${targetCellActive}(cell,vxp1TopologySlot())){
    vxp1Fault(address.x,address.y);return false;}
  return vxp1RecordRootPacketMask(address.x,
    select(0u,1u<<(address.y&31u),address.y<32u),
    select(0u,1u<<(address.y&31u),address.y>=32u),cause);
}
// The post-commit bridge uses this for roots originally recorded while their
// cells were pending. It resolves only through the frozen final slot and does
// not consult the pre-flip active bit. Direct pre-flip calls are forbidden by
// the VXP1 reconciliation schedule.
fn vxp1RecordPendingCellRoot(cell:u32,cause:u32)->bool{
  let address=${cellPacket}(cell,vxp1TopologySlot());
  if(address.x==vxp1Invalid||address.y>=64u||!${targetCellActive}(cell,vxp1TopologySlot())){
    vxp1Fault(address.x,address.y);return false;}
  return vxp1RecordRootPacketMask(address.x,
    select(0u,1u<<(address.y&31u),address.y<32u),
    select(0u,1u<<(address.y&31u),address.y>=32u),cause);
}
// Run only after the accepted selector has flipped to the header's frozen
// target slot. It preserves the legacy root stamp/cause arrays and converts
// every still-active non-transport producer root into the packet union.
@compute @workgroup_size(64) fn bridgeSparseCM12LegacyVexRoots(
 @builtin(global_invocation_id)gid:vec3u){
  let count=vxp1Load(${vh(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.rootCount)});
  if(gid.x>=count){return;}let cell=vxp1Load(${vex.rootListBaseWords}u+gid.x);
  if(cell==vxp1Invalid||!${targetCellActive}(cell,vxp1TopologySlot())){return;}
  let cause=vxp1Load(vxp1CellRootCause+cell);
  if(cause==0u){vxp1Fault(vxp1Invalid,vxp1Invalid);return;}
  _=vxp1RecordPendingCellRoot(cell,cause);
}

@compute @workgroup_size(1) fn sealSparseCM12VexPacketRoots(){
  if(!vxp1ValidHeader()||vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)})
      !=vxp1PhaseCollecting){vxp1Fault(vxp1Invalid,vxp1Invalid);return;}
  let count=vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.rootPacketCount)});
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.rootDispatchX)},count);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierAPacketCount)},0u);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastPacketCount)},0u);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)},vxp1PhaseRootsSealed);
}

@compute @workgroup_size(64) fn materializeSparseCM12VexPacketRoots(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let rank=wid.x;let count=vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.rootPacketCount)});
  // The indirect dispatch is copied from rootPacketCount; no partial terminal
  // workgroup exists because one workgroup owns one packet.
  _=count;let packet=vxp1Load(vxp1RootPacketList+rank);
  if(lane<${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}u){
    vxp1SharedCauseLow[lane]=vxp1Load(vxp1CauseLow[lane]+packet);
    vxp1SharedCauseHigh[lane]=vxp1Load(vxp1CauseHigh[lane]+packet);}
  workgroupBarrier();
  var cause=0u;for(var bit=0u;bit<${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}u;bit+=1u){
    let mask=select(vxp1SharedCauseLow[bit],vxp1SharedCauseHigh[bit],lane>=32u);
    if(((mask>>(lane&31u))&1u)!=0u){cause|=1u<<bit;}}
  if(cause!=0u){let cell=${packetCell}(packet,lane,vxp1TopologySlot());
    if(cell==vxp1Invalid||!${targetCellActive}(cell,vxp1TopologySlot())){vxp1Fault(packet,lane);
    }else{let generation=vxp1Generation();vxp1Store(vxp1CellRootStamp+cell,generation);
      vxp1Store(vxp1CellRootCause+cell,cause);vxp1Store(vxp1CellBlastStamp+cell,generation);
      vxp1Store(vxp1CellBlastDepth+cell,0u);
      atomicAdd(&${arena}[${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.rootCellCount)}],1u);
      atomicAdd(&${arena}[${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastCellCount)}],1u);
      ${rootHook}}}
  workgroupBarrier();
  if(lane==0u){var low=0u;var high=0u;
    for(var bit=0u;bit<${SPARSE_CM12_VEX_PACKET_FRONTIER_CAUSE_BITS}u;bit+=1u){
      low|=vxp1SharedCauseLow[bit];high|=vxp1SharedCauseHigh[bit];}
    if((low|high)!=0u){let generation=vxp1Generation();
      vxp1Store(vxp1FrontierAGenerationStamp+packet,generation);
      vxp1Store(vxp1FrontierADepthStamp+packet,0u);
      vxp1Store(vxp1FrontierALow+packet,low);vxp1Store(vxp1FrontierAHigh+packet,high);
      vxp1Store(vxp1BlastStamp+packet,generation);
      vxp1Store(vxp1BlastLow+packet,low);vxp1Store(vxp1BlastHigh+packet,high);}}
}

// Separate compaction is required: packet masks are indexed by packet id while
// the raw collection list is indexed by rank. A dispatch boundary guarantees
// every cause plane has been consumed before output list writes may alias it.
@compute @workgroup_size(64) fn compactSparseCM12VexPacketRoots(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  if(lane!=0u){return;}let rank=wid.x;let packet=vxp1Load(vxp1RootPacketList+rank);
  if(vxp1Load(vxp1FrontierAGenerationStamp+packet)!=vxp1Generation()){return;}
  let low=vxp1Load(vxp1FrontierALow+packet);let high=vxp1Load(vxp1FrontierAHigh+packet);
  if((low|high)==0u){return;}
  let frontierRank=atomicAdd(&${arena}[
    ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierAPacketCount)}],1u);
  let blastRank=atomicAdd(&${arena}[
    ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastPacketCount)}],1u);
  if(frontierRank>=vxp1PacketCapacity||blastRank>=vxp1PacketCapacity){
    vxp1Fault(packet,vxp1Invalid);return;}
  vxp1Store(vxp1FrontierAList+frontierRank,packet);vxp1Store(vxp1BlastList+blastRank,packet);
}

fn vxp1BankGenerationStamp(depth:u32)->u32{return select(
  vxp1FrontierAGenerationStamp,vxp1FrontierBGenerationStamp,(depth&1u)==1u);}
fn vxp1BankDepthStamp(depth:u32)->u32{return select(
  vxp1FrontierADepthStamp,vxp1FrontierBDepthStamp,(depth&1u)==1u);}
fn vxp1BankCandidateList(depth:u32)->u32{return select(
  vxp1FrontierACandidateList,vxp1FrontierBCandidateList,(depth&1u)==1u);}
fn vxp1BankLow(depth:u32)->u32{return select(vxp1FrontierALow,vxp1FrontierBLow,(depth&1u)==1u);}
fn vxp1BankHigh(depth:u32)->u32{return select(vxp1FrontierAHigh,vxp1FrontierBHigh,(depth&1u)==1u);}
fn vxp1BankList(depth:u32)->u32{return select(vxp1FrontierAList,vxp1FrontierBList,(depth&1u)==1u);}
fn vxp1BankCandidateCountAt(depth:u32)->u32{return select(
  ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierACandidateCount)},
  ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierBCandidateCount)},(depth&1u)==1u);}
fn vxp1BankPacketCountAt(depth:u32)->u32{return select(
  ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierAPacketCount)},
  ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierBPacketCount)},(depth&1u)==1u);}

@compute @workgroup_size(1) fn prepareSparseCM12VexPacketFrontier(){
  if(VXP1_FRONTIER_DEPTH<1u||VXP1_FRONTIER_DEPTH>8u){vxp1Fault(vxp1Invalid,vxp1Invalid);return;}
  let inputDepth=VXP1_FRONTIER_DEPTH-1u;
  let inputCount=vxp1Load(vxp1BankPacketCountAt(inputDepth));
  let inputDispatch=select(
    ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierADispatchX)},
    ${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.frontierBDispatchX)},
    (inputDepth&1u)==1u);
  vxp1Store(inputDispatch,(inputCount+3u)/4u);
  vxp1Store(vxp1BankCandidateCountAt(VXP1_FRONTIER_DEPTH),0u);
  vxp1Store(vxp1BankPacketCountAt(VXP1_FRONTIER_DEPTH),0u);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.currentDepth)},VXP1_FRONTIER_DEPTH);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)},vxp1PhasePlanning);
}
fn vxp1ClaimFrontierTarget(packet:u32,depth:u32)->bool{
  if(packet>=vxp1PacketCapacity){vxp1Fault(packet,vxp1Invalid);return false;}
  let generation=vxp1Generation();let generationAt=vxp1BankGenerationStamp(depth)+packet;
  for(var spin=0u;spin<256u;spin+=1u){let observed=vxp1Load(generationAt);
    if(observed==generation){break;}if((observed&vxp1Claim)!=0u){continue;}
    let claim=atomicCompareExchangeWeak(&${arena}[generationAt],observed,generation|vxp1Claim);
    if(claim.exchanged){vxp1Store(vxp1BankDepthStamp(depth)+packet,vxp1Invalid);
      vxp1Store(generationAt,generation);break;}
    if(spin==255u){vxp1Fault(packet,vxp1Invalid);return false;}}
  if(vxp1Load(generationAt)!=generation){vxp1Fault(packet,vxp1Invalid);return false;}
  let depthAt=vxp1BankDepthStamp(depth)+packet;
  for(var spin=0u;spin<256u;spin+=1u){let observed=vxp1Load(depthAt);
    if(observed==depth){return true;}if((observed&vxp1Claim)!=0u){continue;}
    let claim=atomicCompareExchangeWeak(&${arena}[depthAt],observed,depth|vxp1Claim);
    if(!claim.exchanged){continue;}vxp1Store(vxp1BankLow(depth)+packet,0u);
    vxp1Store(vxp1BankHigh(depth)+packet,0u);
    let rank=atomicAdd(&${arena}[vxp1BankCandidateCountAt(depth)],1u);
    if(rank>=vxp1PacketCapacity){vxp1Fault(packet,vxp1Invalid);return false;}
    vxp1Store(vxp1BankCandidateList(depth)+rank,packet);vxp1Store(depthAt,depth);return true;}
  vxp1Fault(packet,vxp1Invalid);return false;
}
fn vxp1MergeFrontierTarget(packet:u32,low:u32,high:u32)->bool{
  if((low|high)==0u){return true;}if(!vxp1ClaimFrontierTarget(packet,VXP1_FRONTIER_DEPTH)){return false;}
  atomicOr(&${arena}[vxp1BankLow(VXP1_FRONTIER_DEPTH)+packet],low);
  atomicOr(&${arena}[vxp1BankHigh(VXP1_FRONTIER_DEPTH)+packet],high);return true;
}
fn vxp1FrontierPacketCount(depth:u32)->u32{return vxp1Load(vxp1BankPacketCountAt(depth));}
fn vxp1FrontierPacket(depth:u32,rank:u32)->u32{return vxp1Load(vxp1BankList(depth)+rank);}
fn vxp1FrontierMask(depth:u32,packet:u32)->vec2u{return vec2u(
  vxp1Load(vxp1BankLow(depth)+packet),vxp1Load(vxp1BankHigh(depth)+packet));}

@compute @workgroup_size(64) fn finalizeSparseCM12VexPacketFrontier(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let rank=wid.x;let candidateCount=vxp1Load(vxp1BankCandidateCountAt(VXP1_FRONTIER_DEPTH));
  // The candidate indirect dispatch is exact packet count, not ceil(cells/64).
  _=candidateCount;let packet=vxp1Load(
    vxp1BankCandidateList(VXP1_FRONTIER_DEPTH)+rank);
  if(lane==0u){let candidateLow=vxp1Load(vxp1BankLow(VXP1_FRONTIER_DEPTH)+packet);
    let candidateHigh=vxp1Load(vxp1BankHigh(VXP1_FRONTIER_DEPTH)+packet);
    let generation=vxp1Generation();let hasBlast=vxp1Load(vxp1BlastStamp+packet)==generation;
    let oldLow=select(0u,vxp1Load(vxp1BlastLow+packet),hasBlast);
    let oldHigh=select(0u,vxp1Load(vxp1BlastHigh+packet),hasBlast);
    let novelLow=candidateLow&~oldLow;let novelHigh=candidateHigh&~oldHigh;
    vxp1SharedPacket=packet;vxp1SharedLow=novelLow;vxp1SharedHigh=novelHigh;
    vxp1Store(vxp1BankLow(VXP1_FRONTIER_DEPTH)+packet,novelLow);
    vxp1Store(vxp1BankHigh(VXP1_FRONTIER_DEPTH)+packet,novelHigh);
    var publish=(novelLow|novelHigh)!=0u;
    if(publish&&!hasBlast){vxp1Store(vxp1BlastStamp+packet,generation);
        let blastRank=atomicAdd(&${arena}[${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastPacketCount)}],1u);
        if(blastRank>=vxp1PacketCapacity){vxp1Fault(packet,vxp1Invalid);publish=false;
        }else{vxp1Store(vxp1BlastList+blastRank,packet);
          vxp1Store(vxp1BlastLow+packet,novelLow);vxp1Store(vxp1BlastHigh+packet,novelHigh);}
      }else if(publish){atomicOr(&${arena}[vxp1BlastLow+packet],novelLow);
        atomicOr(&${arena}[vxp1BlastHigh+packet],novelHigh);}
    if(publish){
      let frontierRank=atomicAdd(&${arena}[vxp1BankPacketCountAt(VXP1_FRONTIER_DEPTH)],1u);
      if(frontierRank>=vxp1PacketCapacity){vxp1Fault(packet,vxp1Invalid);publish=false;
      }else{vxp1Store(vxp1BankList(VXP1_FRONTIER_DEPTH)+frontierRank,packet);}}
    if(!publish){vxp1SharedLow=0u;vxp1SharedHigh=0u;}}
  workgroupBarrier();let selected=((select(vxp1SharedLow,vxp1SharedHigh,lane>=32u)
    >>(lane&31u))&1u)!=0u;if(!selected){return;}
  let selectedPacket=vxp1SharedPacket;
  let cell=${packetCell}(selectedPacket,lane,vxp1TopologySlot());
  if(cell==vxp1Invalid||!${targetCellActive}(cell,vxp1TopologySlot())){
    vxp1Fault(selectedPacket,lane);return;}
  let generation=vxp1Generation();vxp1Store(vxp1CellBlastStamp+cell,generation);
  vxp1Store(vxp1CellBlastDepth+cell,VXP1_FRONTIER_DEPTH);
  atomicAdd(&${arena}[${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastCellCount)}],1u);
  ${closureHook}
}

@compute @workgroup_size(1) fn finalizeSparseCM12VexPacketBlast(){
  if(vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)})==vxp1PhaseFault){
    cm12ExtensionFail(vxp1Invalid,0u);return;}
  let count=vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastPacketCount)});
  let cells=vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastCellCount)});
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastDispatchX)},count);
  // Preserve the legacy header as the transaction/QA receipt while routing
  // candidate execution through the packet dispatch explicitly. The legacy
  // cell list is overlaid and therefore must never be dispatched here.
  vxp1Store(${vh(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastCount)},cells);
  vxp1Store(${vh(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.blastDispatchX)},
    (cells+63u)/64u);
  vxp1Store(${vh(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.flags)},cm12ExtensionFlagSealed);
  vxp1Store(${vh(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.reserved)},
    cm12ExtensionPhasePlanned);
  vxp1Store(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.phase)},vxp1PhasePlanned);
}
fn vxp1BlastPacketCount()->u32{return vxp1Load(${h(SPARSE_CM12_VEX_PACKET_FRONTIER_HEADER.blastPacketCount)});}
fn vxp1BlastPacket(rank:u32)->u32{return select(vxp1Invalid,vxp1Load(vxp1BlastList+rank),
  rank<vxp1BlastPacketCount());}
fn vxp1BlastMask(packet:u32)->vec2u{return vec2u(vxp1Load(vxp1BlastLow+packet),
  vxp1Load(vxp1BlastHigh+packet));}
fn vxp1BlastLaneSelected(packet:u32,lane:u32)->bool{let mask=vxp1BlastMask(packet);
  return ((select(mask.x,mask.y,lane>=32u)>>(lane&31u))&1u)!=0u;}
fn vxp1CellInBlast(cell:u32)->bool{let address=${cellPacket}(cell,vxp1TopologySlot());
  return address.x!=vxp1Invalid&&address.y<64u&&vxp1Load(vxp1BlastStamp+address.x)==vxp1Generation()
    &&vxp1BlastLaneSelected(address.x,address.y);}
`;
}
