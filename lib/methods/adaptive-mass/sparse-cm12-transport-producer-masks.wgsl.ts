import {
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_FAULT,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER_WORDS,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_MAGIC,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_PHASE,
  SPARSE_CM12_TRANSPORT_PRODUCER_MASK_VERSION,
  type SparseCM12TransportProducerMaskLayout,
} from "./sparse-cm12-transport-producer-masks";

export interface SparseCM12TransportProducerMaskWGSLOptions {
  readonly layout: SparseCM12TransportProducerMaskLayout;
  /** Existing `array<atomic<u32>>` arena. */
  readonly arenaName?: string;
  /** Prefix for the resident integration hooks documented below. */
  readonly hookPrefix?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/**
 * Binding-free TPM1 helpers and compiler entry points.
 *
 * The resident supplies these prefix hooks:
 * - `<p>TransportProducerMaskGeneration() -> u32`
 * - `<p>TransportProducerMaskPacketCount() -> u32`
 * - `<p>TransportProducerMaskPacket(workRank) -> u32`
 * - `<p>TransportProducerMaskCell(packet, lane) -> u32`
 * - `<p>TransportProducerMaskCellValid(cell) -> bool`
 * - `<p>TransportProducerMaskIncidenceBegin/End(cell) -> u32`
 * - `<p>TransportProducerMaskIncidenceRow(incidence) -> u32`
 * - `<p>TransportProducerMaskRowAccepted(row) -> bool`
 * - `<p>TransportProducerMaskRowTermBegin/End(row) -> u32`
 * - `<p>TransportProducerMaskRowTermCell(term) -> u32`
 * - `<p>TransportProducerMaskMarkRow(row)`
 * - `<p>TransportProducerSharpeningMask(packet, mask)`
 */
export function createSparseCM12TransportProducerMaskWGSL(
  options: SparseCM12TransportProducerMaskWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "topologyArena", "arenaName");
  const p = identifier(options.hookPrefix ?? "cm12Resident", "hookPrefix");
  const h = (word: number) => `${layout.baseWords + word}u`;
  return /* wgsl */ `
const TPM1_MAGIC:u32=0x${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_MAGIC.toString(16)}u;
const TPM1_VERSION:u32=${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_VERSION}u;
const TPM1_CAPACITY:u32=${layout.packetCapacity}u;
const TPM1_STAMP:u32=${layout.packetStampBaseWords}u;
const TPM1_SURFACE_LOW:u32=${layout.surfaceLowBaseWords}u;
const TPM1_SURFACE_HIGH:u32=${layout.surfaceHighBaseWords}u;
const TPM1_PHASE_COLLECTING:u32=${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_PHASE.collecting}u;
const TPM1_PHASE_PUBLISHED:u32=${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_PHASE.published}u;
const TPM1_PHASE_FAULT:u32=${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_PHASE.fault}u;
const TPM1_INVALID:u32=0xffffffffu;

var<workgroup>tpm1SurfaceLow:atomic<u32>;
var<workgroup>tpm1SurfaceHigh:atomic<u32>;
var<workgroup>tpm1SharpeningLow:atomic<u32>;
var<workgroup>tpm1SharpeningHigh:atomic<u32>;

fn tpm1HeaderValid()->bool{
  return arrayLength(&${arena})>=${layout.totalWords}u
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.magic)}])==TPM1_MAGIC
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.version)}])==TPM1_VERSION
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.headerWords)}])
      ==${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER_WORDS}u
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.packetCapacity)}])
      ==TPM1_CAPACITY;
}
fn tpm1Generation()->u32{
  return atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.candidateGeneration)}]);
}
fn tpm1Fault(packet:u32,code:u32){
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.phase)}],TPM1_PHASE_FAULT);
  var won=false;
  loop{
    if(atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.fault)}])!=0u){break;}
    let claimed=atomicCompareExchangeWeak(
      &${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.fault)}],0u,code);
    if(claimed.exchanged){won=true;break;}
  }
  if(won){atomicStore(
    &${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.firstFaultPacket)}],packet);}
}

@compute @workgroup_size(1)
fn beginSparseCM12TransportProducerMasks(){
  if(!tpm1HeaderValid()){
    tpm1Fault(TPM1_INVALID,${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_FAULT.invalidHeader}u);return;
  }
  let generation=${p}TransportProducerMaskGeneration();
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.fault)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.firstFaultPacket)}],TPM1_INVALID);
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.candidateGeneration)}],generation);
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.publishedPacketCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.surfaceCellCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.phase)}],TPM1_PHASE_COLLECTING);
}

// Exactly one invocation by every lane of every transport packet.  The two
// atomics are a portable workgroup ballot; no subgroup feature is required.
fn cm12TransportProducerMaskPublish(packet:u32,lane:u32,cell:u32,
 surfaceFeature:bool){
  if(lane==0u){
    atomicStore(&tpm1SurfaceLow,0u);atomicStore(&tpm1SurfaceHigh,0u);
  }
  workgroupBarrier();
  let valid=${p}TransportProducerMaskCellValid(cell);
  if(valid&&surfaceFeature){
    if(lane<32u){_=atomicOr(&tpm1SurfaceLow,1u<<lane);}
    else{_=atomicOr(&tpm1SurfaceHigh,1u<<(lane-32u));}
  }
  workgroupBarrier();
  if(lane==0u){
    if(packet>=TPM1_CAPACITY){
      tpm1Fault(packet,${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_FAULT.packetOutOfRange}u);return;
    }
    let generation=tpm1Generation();
    let lowS=atomicLoad(&tpm1SurfaceLow);let highS=atomicLoad(&tpm1SurfaceHigh);
    atomicStore(&${arena}[TPM1_SURFACE_LOW+packet],lowS);
    atomicStore(&${arena}[TPM1_SURFACE_HIGH+packet],highS);
    atomicAdd(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.surfaceCellCount)}],
      countOneBits(lowS)+countOneBits(highS));
    atomicAdd(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.publishedPacketCount)}],1u);
    // Publication stamp is deliberately last.  Compiler dispatches occur only
    // after this producer dispatch has completed.
    atomicStore(&${arena}[TPM1_STAMP+packet],generation);
  }
}

// Replace family zero's mask with the exact trace-time sharpening closure.
// The trace has already staged the transport mask and uniquely owns this list
// entry, so later sharpening can reuse the list without a second mask plane.
fn cm12TransportSharpeningMaskPublish(packet:u32,lane:u32,cell:u32,
 sharpeningSource:bool,cellScale:u32){
  if(lane==0u){
    atomicStore(&tpm1SharpeningLow,0u);atomicStore(&tpm1SharpeningHigh,0u);
  }
  workgroupBarrier();
  let valid=${p}TransportProducerMaskCellValid(cell);
  if(valid&&sharpeningSource){
    var low=0u;var high=0u;
    if(cellScale==1u){low=0xffffffffu;high=0xffffffffu;
    }else if(cellScale==2u){
      let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
      let base=(q.x&~1u)+4u*(q.y&~1u)+16u*(q.z&~1u);
      for(var dz=0u;dz<2u;dz+=1u){for(var dy=0u;dy<2u;dy+=1u){
        for(var dx=0u;dx<2u;dx+=1u){let member=base+dx+4u*dy+16u*dz;
          if(member<32u){low|=1u<<member;}else{high|=1u<<(member-32u);}
        }
      }}
    }else if(lane<32u){low=1u<<lane;}else{high=1u<<(lane-32u);}
    _=atomicOr(&tpm1SharpeningLow,low);_=atomicOr(&tpm1SharpeningHigh,high);
  }
  workgroupBarrier();
  if(lane==0u&&packet<TPM1_CAPACITY){
    ${p}TransportProducerSharpeningMask(packet,vec2u(
      atomicLoad(&tpm1SharpeningLow),atomicLoad(&tpm1SharpeningHigh)));
  }
}

@compute @workgroup_size(1)
fn sealSparseCM12TransportProducerMasks(){
  if(atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.phase)}])
      !=TPM1_PHASE_COLLECTING){return;}
  let expected=${p}TransportProducerMaskPacketCount();
  let actual=atomicLoad(&${arena}[
    ${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.publishedPacketCount)}]);
  if(expected>TPM1_CAPACITY||actual!=expected){
    tpm1Fault(select(actual,expected,expected>TPM1_CAPACITY),
      ${SPARSE_CM12_TRANSPORT_PRODUCER_MASK_FAULT.incompletePublication}u);return;
  }
  atomicStore(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.phase)}],TPM1_PHASE_PUBLISHED);
}

fn tpm1PacketReady(packet:u32)->bool{
  return packet<TPM1_CAPACITY
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_TRANSPORT_PRODUCER_MASK_HEADER.phase)}])
      ==TPM1_PHASE_PUBLISHED
    &&atomicLoad(&${arena}[TPM1_STAMP+packet])==tpm1Generation();
}
fn tpm1SurfaceMask(packet:u32)->vec2u{
  if(!tpm1PacketReady(packet)){return vec2u(0u);}
  return vec2u(atomicLoad(&${arena}[TPM1_SURFACE_LOW+packet]),
    atomicLoad(&${arena}[TPM1_SURFACE_HIGH+packet]));
}
`;
}
