import {
  SPARSE_CM12_FINAL_SCALAR_MASK_HEADER,
  SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS,
  SPARSE_CM12_FINAL_SCALAR_MASK_MAGIC,
  SPARSE_CM12_FINAL_SCALAR_MASK_PHASE,
  SPARSE_CM12_FINAL_SCALAR_MASK_VERSION,
  type SparseCM12FinalScalarPacketMaskLayout,
} from "./sparse-cm12-final-scalar-packet-masks";

/** Binding-free FSM1 storage and portable workgroup-ballot helpers. */
export function createSparseCM12FinalScalarPacketMaskWGSL(options: Readonly<{
  layout: SparseCM12FinalScalarPacketMaskLayout;
  arenaName?: string;
}>): string {
  const layout = options.layout;
  const arena = options.arenaName ?? "topologyArena";
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(arena)) throw new TypeError("FSM1 arena name");
  const h = (word: number) => `${layout.baseWords + word}u`;
  return /* wgsl */ `
const FSM1_MAGIC:u32=0x${SPARSE_CM12_FINAL_SCALAR_MASK_MAGIC.toString(16)}u;
const FSM1_VERSION:u32=${SPARSE_CM12_FINAL_SCALAR_MASK_VERSION}u;
const FSM1_CAPACITY:u32=${layout.packetCapacity}u;
const FSM1_PACKETS_PER_ACTIVE_LEAF:u32=${layout.maximumPacketsPerLeaf}u;
const FSM1_CHANGED_LOW:u32=${layout.changedLowBaseWords}u;
const FSM1_CHANGED_HIGH:u32=${layout.changedHighBaseWords}u;
const FSM1_NONEXACT_LOW:u32=${layout.nonexactLowBaseWords}u;
const FSM1_NONEXACT_HIGH:u32=${layout.nonexactHighBaseWords}u;
const FSM1_BULK_LOW:u32=${layout.bulkLowBaseWords}u;
const FSM1_BULK_HIGH:u32=${layout.bulkHighBaseWords}u;
const FSM1_FLIP_LOW:u32=${layout.flipLowBaseWords}u;
const FSM1_FLIP_HIGH:u32=${layout.flipHighBaseWords}u;
const FSM1_PHASE_COLLECTING:u32=${SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.collecting}u;
const FSM1_PHASE_PUBLISHED:u32=${SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.published}u;

var<workgroup> fsm1ChangedLow:atomic<u32>;
var<workgroup> fsm1ChangedHigh:atomic<u32>;
var<workgroup> fsm1NonexactLow:atomic<u32>;
var<workgroup> fsm1NonexactHigh:atomic<u32>;
var<workgroup> fsm1BulkLow:atomic<u32>;
var<workgroup> fsm1BulkHigh:atomic<u32>;
var<workgroup> fsm1FlipLow:atomic<u32>;
var<workgroup> fsm1FlipHigh:atomic<u32>;

fn fsm1HeaderValid()->bool{return arrayLength(&${arena})>=${layout.totalWords}u
  &&atomicLoad(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.magic)}])==FSM1_MAGIC
  &&atomicLoad(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.version)}])==FSM1_VERSION
  &&atomicLoad(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.headerWords)}])
    ==${SPARSE_CM12_FINAL_SCALAR_MASK_HEADER_WORDS}u
  &&atomicLoad(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.packetCapacity)}])
    ==FSM1_CAPACITY;}
fn fsm1Generation()->u32{return atomicLoad(&${arena}[
  ${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.generation)}]);}
fn fsm1TopologyGeneration()->u32{return atomicLoad(&${arena}[
  ${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.topologyGeneration)}]);}
fn fsm1Published()->bool{return fsm1HeaderValid()&&atomicLoad(&${arena}[
  ${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.phase)}])==FSM1_PHASE_PUBLISHED;}
fn fsm1TopologySlot()->u32{return atomicLoad(&${arena}[
  ${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.topologySlot)}])&1u;}
fn fsm1PacketReady(packet:u32)->bool{return packet<FSM1_CAPACITY
  &&atomicLoad(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.phase)}])
    >=FSM1_PHASE_PUBLISHED;}
fn fsm1Lane(mask:vec2u,lane:u32)->bool{return lane<64u
  &&((mask[lane>>5u]>>(lane&31u))&1u)!=0u;}
fn fsm1Mask(packet:u32,low:u32,high:u32)->vec2u{
  if(!fsm1PacketReady(packet)){return vec2u(0u);}
  return vec2u(atomicLoad(&${arena}[low+packet]),atomicLoad(&${arena}[high+packet]));}
fn fsm1Changed(packet:u32)->vec2u{return fsm1Mask(packet,FSM1_CHANGED_LOW,FSM1_CHANGED_HIGH);}
fn fsm1Nonexact(packet:u32)->vec2u{return fsm1Mask(packet,FSM1_NONEXACT_LOW,FSM1_NONEXACT_HIGH);}
fn fsm1Bulk(packet:u32)->vec2u{return fsm1Mask(packet,FSM1_BULK_LOW,FSM1_BULK_HIGH);}
fn fsm1Flip(packet:u32)->vec2u{return fsm1Mask(packet,FSM1_FLIP_LOW,FSM1_FLIP_HIGH);}
fn fsm1ChangedCell(cell:u32)->bool{let address=cm12FinalScalarPacketAddress(cell);
  return address.x!=0xffffffffu&&fsm1Lane(fsm1Changed(address.x),address.y);}
fn fsm1FlipCell(cell:u32)->bool{let address=cm12FinalScalarPacketAddress(cell);
  return address.x!=0xffffffffu&&fsm1Lane(fsm1Flip(address.x),address.y);}
fn fsm1ChangedOrFlipCell(cell:u32)->bool{let address=cm12FinalScalarPacketAddress(cell);
  return address.x!=0xffffffffu&&fsm1Lane(
    fsm1Changed(address.x)|fsm1Flip(address.x),address.y);}

fn fsm1ResetBallot(lane:u32){if(lane==0u){
  atomicStore(&fsm1ChangedLow,0u);atomicStore(&fsm1ChangedHigh,0u);
  atomicStore(&fsm1NonexactLow,0u);atomicStore(&fsm1NonexactHigh,0u);
  atomicStore(&fsm1BulkLow,0u);atomicStore(&fsm1BulkHigh,0u);
  atomicStore(&fsm1FlipLow,0u);atomicStore(&fsm1FlipHigh,0u);}workgroupBarrier();}
fn fsm1BallotBit(targetLow:ptr<workgroup,atomic<u32>>,
 targetHigh:ptr<workgroup,atomic<u32>>,lane:u32,enabled:bool){
  if(!enabled){return;}if(lane<32u){_=atomicOr(targetLow,1u<<lane);}
  else{_=atomicOr(targetHigh,1u<<(lane-32u));}}
fn fsm1PublishBallot(packet:u32,lane:u32,cell:u32,changed:bool,nonexact:bool,
 bulk:bool,flip:bool){
  fsm1BallotBit(&fsm1ChangedLow,&fsm1ChangedHigh,lane,changed);
  fsm1BallotBit(&fsm1NonexactLow,&fsm1NonexactHigh,lane,nonexact);
  fsm1BallotBit(&fsm1BulkLow,&fsm1BulkHigh,lane,bulk);
  fsm1BallotBit(&fsm1FlipLow,&fsm1FlipHigh,lane,flip);workgroupBarrier();
  if(lane==0u&&packet<FSM1_CAPACITY){
    let c=vec2u(atomicLoad(&fsm1ChangedLow),atomicLoad(&fsm1ChangedHigh));
    let n=vec2u(atomicLoad(&fsm1NonexactLow),atomicLoad(&fsm1NonexactHigh));
    let b=vec2u(atomicLoad(&fsm1BulkLow),atomicLoad(&fsm1BulkHigh));
    let f=vec2u(atomicLoad(&fsm1FlipLow),atomicLoad(&fsm1FlipHigh));
    atomicStore(&${arena}[FSM1_CHANGED_LOW+packet],c.x);
    atomicStore(&${arena}[FSM1_CHANGED_HIGH+packet],c.y);
    atomicStore(&${arena}[FSM1_NONEXACT_LOW+packet],n.x);
    atomicStore(&${arena}[FSM1_NONEXACT_HIGH+packet],n.y);
    atomicStore(&${arena}[FSM1_BULK_LOW+packet],b.x);
    atomicStore(&${arena}[FSM1_BULK_HIGH+packet],b.y);
    atomicStore(&${arena}[FSM1_FLIP_LOW+packet],f.x);
    atomicStore(&${arena}[FSM1_FLIP_HIGH+packet],f.y);
    atomicAdd(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.changedCellCount)}],
      countOneBits(c.x)+countOneBits(c.y));
    atomicAdd(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.nonexactCellCount)}],
      countOneBits(n.x)+countOneBits(n.y));
    atomicAdd(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.bulkCellCount)}],
      countOneBits(b.x)+countOneBits(b.y));
    atomicAdd(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.flipCellCount)}],
      countOneBits(f.x)+countOneBits(f.y));
  }_=cell;}


@compute @workgroup_size(1) fn beginSparseCM12FinalScalarMasks(){
  if(!fsm1HeaderValid()){atomicStore(&${arena}[
    ${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.phase)}],
    ${SPARSE_CM12_FINAL_SCALAR_MASK_PHASE.fault}u);return;}
  let generation=cm12FinalScalarGeneration();
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.generation)}],generation);
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.topologyGeneration)}],
    cm12FinalScalarTopologyGeneration());
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.topologySlot)}],
    cm12FinalScalarTopologySlot());
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.changedCellCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.nonexactCellCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.bulkCellCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.flipCellCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.fault)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.firstFaultPacket)}],
    0xffffffffu);
  atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.phase)}],
    FSM1_PHASE_COLLECTING);
}

@compute @workgroup_size(64) fn publishSparseCM12FinalScalarMasks(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let leaf=wid.x;
  for(var localPacket=0u;localPacket<FSM1_PACKETS_PER_ACTIVE_LEAF;localPacket+=1u){
    let packet=64u*leaf+localPacket;fsm1ResetBallot(lane);
    let cell=cm12FinalScalarPacketCell(packet,lane,cm12FinalScalarTopologySlot());
    let facts=cm12FinalScalarCellFacts(cell);
    fsm1PublishBallot(packet,lane,cell,facts.x!=0u,facts.y!=0u,
      facts.z!=0u,facts.w!=0u);
    workgroupBarrier();
  }
}

@compute @workgroup_size(1) fn sealSparseCM12FinalScalarMasks(){
  if(atomicLoad(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.phase)}])
      ==FSM1_PHASE_COLLECTING){
    atomicStore(&${arena}[${h(SPARSE_CM12_FINAL_SCALAR_MASK_HEADER.phase)}],
      FSM1_PHASE_PUBLISHED);
  }
}
`;
}
