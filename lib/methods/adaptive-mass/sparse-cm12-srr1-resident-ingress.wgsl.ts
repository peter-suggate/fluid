import {
  SPARSE_CM12_SRR1_EVENT,
  SPARSE_CM12_SRR1_EVENT_STAMP,
  SPARSE_CM12_SRR1_EVENT_WORDS,
  SPARSE_CM12_SRR1_INGRESS_HEADER,
  SPARSE_CM12_SRR1_INGRESS_MAGIC,
  SPARSE_CM12_SRR1_RECEIPT,
  SPARSE_CM12_SRR1_RECEIPT_WORDS,
  type SparseCM12SRR1IngressLayout,
} from "./sparse-cm12-srr1-runtime-adapter";

const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError("WGSL identifier");
  return value;
};

/**
 * Resident producer helpers for the SIR1 activity tail. They use the existing
 * activity binding and therefore add no hot-physics storage binding.
 */
export function createSparseCM12SRR1ResidentIngressWGSL(options: {
  readonly layout: SparseCM12SRR1IngressLayout;
  readonly arenaName?: string;
  /** Opt-in TFX1 seam. Omitted output remains byte-for-byte unchanged. */
  readonly preflightedTopologyPublication?: boolean;
}): string {
  const l = options.layout;
  const arena = identifier(options.arenaName ?? "activity");
  const h = SPARSE_CM12_SRR1_INGRESS_HEADER;
  const s = SPARSE_CM12_SRR1_EVENT_STAMP;
  const e = SPARSE_CM12_SRR1_EVENT;
  const r = SPARSE_CM12_SRR1_RECEIPT;
  const preflightedTopologyPublication = options.preflightedTopologyPublication ? /* wgsl */ `
fn sirResidentPreflightWillAppend(tile:u32,generation:u32)->bool{
  return atomicLoad(&${arena}[sirResidentStampBase+2u*tile+${s.generation}u])!=generation;
}
fn sirResidentPreflightReady(generation:u32,newCount:u32)->bool{
  let current=atomicLoad(&${arena}[sirResidentHeader+${h.candidateGeneration}u]);
  return sirResidentHeaderValid()
    &&atomicLoad(&${arena}[sirResidentHeader+${h.fault}u])==0u
    // Topology effects are the bounded producers for the *next* SRR1 frame;
    // the current generation has already been sealed and planned.
    &&current<0x7ffffffeu&&current+1u==generation
    &&atomicLoad(&${arena}[sirResidentHeader+${h.eventCount}u])+newCount
      <=sirResidentEventCapacity;
}
// TFX1 has already proved phase, generation, uniqueness and capacity. This
// helper has no failure return, retry loop, capacity branch, or fault mutation.
fn sirResidentPublishPreflightedEvent(tile:u32,generation:u32,cause:u32){
  let stamp=sirResidentStampBase+2u*tile;
  let append=atomicLoad(&${arena}[stamp+${s.generation}u])!=generation;
  if(append){
    atomicStore(&${arena}[stamp+${s.generation}u],generation);
    atomicStore(&${arena}[stamp+${s.causeMask}u],cause);
    let slot=atomicAdd(&${arena}[sirResidentHeader+${h.eventCount}u],1u);
    let at=sirResidentEventBase+${SPARSE_CM12_SRR1_EVENT_WORDS}u*slot;
    atomicStore(&${arena}[at+${e.tile}u],tile);
    atomicStore(&${arena}[at+${e.generation}u],generation);
    atomicStore(&${arena}[at+${e.causeMask}u],cause);
  }else{atomicOr(&${arena}[stamp+${s.causeMask}u],cause);}
}
` : "";
  return /* wgsl */ `
const sirResidentTileCapacity:u32=${l.tileCapacity}u;
const sirResidentEventCapacity:u32=${l.eventCapacity}u;
const sirResidentHeader:u32=${l.headerBaseWords}u;
const sirResidentStampBase:u32=${l.eventStampBaseWords}u;
const sirResidentEventBase:u32=${l.eventBaseWords}u;
const sirResidentWorkListBase:u32=${l.workListBaseWords}u;
const sirResidentReceiptBase:u32=${l.receiptBaseWords}u;
fn sirResidentHeaderValid()->bool{
  return atomicLoad(&${arena}[sirResidentHeader+${h.magic}u])
      ==0x${SPARSE_CM12_SRR1_INGRESS_MAGIC.toString(16)}u
    &&atomicLoad(&${arena}[sirResidentHeader+${h.version}u])==1u
    &&atomicLoad(&${arena}[sirResidentHeader+${h.tileCapacity}u])==sirResidentTileCapacity
    &&atomicLoad(&${arena}[sirResidentHeader+${h.eventCapacity}u])==sirResidentEventCapacity;
}
fn sirResidentFail(code:u32,tile:u32){
  let fault=sirResidentHeader+${h.fault}u;
  let claimed=atomicCompareExchangeWeak(&${arena}[fault],0u,code);
  if(claimed.exchanged){atomicStore(&${arena}[sirResidentHeader+${h.firstFaultTile}u],tile);}
}
fn sirResidentBeginFrame(generation:u32,topologyGeneration:u32,sourceParity:u32)->bool{
  if(!sirResidentHeaderValid()||generation==0u||topologyGeneration==0u||sourceParity>1u){
    sirResidentFail(1u,0xffffffffu);return false;
  }
  atomicStore(&${arena}[sirResidentHeader+${h.candidateGeneration}u],generation);
  atomicStore(&${arena}[sirResidentHeader+${h.topologyGeneration}u],topologyGeneration);
  atomicStore(&${arena}[sirResidentHeader+${h.sourceParity}u],sourceParity);
  return true;
}
fn sirResidentAppendEvent(tile:u32,generation:u32,cause:u32)->bool{
  if(tile>=sirResidentTileCapacity||generation==0u||generation>=0x80000000u){return false;}
  let stamp=sirResidentStampBase+2u*tile;
  // The copied generation was cleared before producers began. No busy owner
  // is required: the winner and all same-generation waiters merge into the
  // pre-cleared cause word, so no workgroup waits for another to be scheduled.
  for(var attempt=0u;attempt<64u;attempt+=1u){
    let old=atomicLoad(&${arena}[stamp+${s.generation}u]);
    if(old==generation){atomicOr(&${arena}[stamp+${s.causeMask}u],cause);return true;}
    if(old>generation){return false;}
    let claimed=atomicCompareExchangeWeak(&${arena}[stamp+${s.generation}u],old,generation);
    if(!claimed.exchanged){continue;}
    atomicOr(&${arena}[stamp+${s.causeMask}u],cause);
    let index=atomicAdd(&${arena}[sirResidentHeader+${h.eventCount}u],1u);
    if(index>=sirResidentEventCapacity){sirResidentFail(2u,tile);return false;}
    let at=sirResidentEventBase+${SPARSE_CM12_SRR1_EVENT_WORDS}u*index;
    atomicStore(&${arena}[at+${e.tile}u],tile);
    atomicStore(&${arena}[at+${e.generation}u],generation);
    atomicStore(&${arena}[at+${e.causeMask}u],cause);
    return true;
  }
  sirResidentFail(3u,tile);return false;
}
fn sirResidentEventStamped(tile:u32,generation:u32)->bool{
  return tile<sirResidentTileCapacity
    &&atomicLoad(&${arena}[sirResidentStampBase+2u*tile+${s.generation}u])==generation;
}
fn sirResidentEventCause(tile:u32,generation:u32)->u32{
  if(!sirResidentEventStamped(tile,generation)){return 0u;}
  return atomicLoad(&${arena}[sirResidentStampBase+2u*tile+${s.causeMask}u]);
}
// Called only after the event list and stamp table have been copied into SRR1.
fn sirResidentResetCopiedEvents(tile:u32){
  if(tile<sirResidentTileCapacity){
    let stamp=sirResidentStampBase+2u*tile;
    atomicStore(&${arena}[stamp+${s.generation}u],0u);
    atomicStore(&${arena}[stamp+${s.causeMask}u],0u);
  }
  if(tile==0u){
    atomicStore(&${arena}[sirResidentHeader+${h.eventCount}u],0u);
    atomicStore(&${arena}[sirResidentHeader+${h.fault}u],0u);
    atomicStore(&${arena}[sirResidentHeader+${h.firstFaultTile}u],0xffffffffu);
  }
}
fn sirResidentBeginReceiptBatch(){
  atomicStore(&${arena}[sirResidentHeader+${h.receiptCount}u],0u);
}
fn sirResidentCandidateGeneration()->u32{
  return atomicLoad(&${arena}[sirResidentHeader+${h.candidateGeneration}u]);
}
fn sirResidentTopologyGeneration()->u32{
  return atomicLoad(&${arena}[sirResidentHeader+${h.topologyGeneration}u]);
}
fn sirResidentSourceParity()->u32{
  return atomicLoad(&${arena}[sirResidentHeader+${h.sourceParity}u]);
}
fn sirResidentWorkTile(rank:u32)->u32{
  if(rank>=sirResidentTileCapacity){return 0xffffffffu;}
  return atomicLoad(&${arena}[sirResidentWorkListBase+rank]);
}
fn sirResidentPublishReceipt(rank:u32,tile:u32,topologyGeneration:u32,
  bank0Generation:u32,bank1Generation:u32,headResultGeneration:u32,
  coveredWords:u32,expectedWords:u32,sourceMismatchCount:u32,
  destinationMismatchCount:u32,dependencyGeneration:u32,
  dependencyCertifiedGeneration:u32,resultKind:u32,generation:u32)->bool{
  if(rank>=sirResidentTileCapacity||tile!=sirResidentWorkTile(rank)
    ||generation!=atomicLoad(&${arena}[sirResidentHeader+${h.candidateGeneration}u])){return false;}
  let at=sirResidentReceiptBase+${SPARSE_CM12_SRR1_RECEIPT_WORDS}u*rank;
  atomicStore(&${arena}[at+${r.tile}u],tile);
  atomicStore(&${arena}[at+${r.topologyGeneration}u],topologyGeneration);
  atomicStore(&${arena}[at+${r.bank0Generation}u],bank0Generation);
  atomicStore(&${arena}[at+${r.bank1Generation}u],bank1Generation);
  atomicStore(&${arena}[at+${r.headResultGeneration}u],headResultGeneration);
  atomicStore(&${arena}[at+${r.coveredWords}u],coveredWords);
  atomicStore(&${arena}[at+${r.expectedWords}u],expectedWords);
  atomicStore(&${arena}[at+${r.sourceMismatchCount}u],sourceMismatchCount);
  atomicStore(&${arena}[at+${r.destinationMismatchCount}u],destinationMismatchCount);
  atomicStore(&${arena}[at+${r.dependencyGeneration}u],dependencyGeneration);
  atomicStore(&${arena}[at+${r.dependencyCertifiedGeneration}u],dependencyCertifiedGeneration);
  atomicStore(&${arena}[at+${r.resultKind}u],resultKind);
  atomicStore(&${arena}[at+${r.generation}u],generation);
  atomicMax(&${arena}[sirResidentHeader+${h.receiptCount}u],rank+1u);
  return true;
}
${preflightedTopologyPublication}
`;
}
