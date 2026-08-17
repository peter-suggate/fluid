import {
  SPARSE_CM12_DIRTY_DOMAIN_COUNT,
  SPARSE_CM12_DIRTY_EPOCH,
  SPARSE_CM12_DIRTY_FAULT,
  SPARSE_CM12_DIRTY_INVALID,
  SPARSE_CM12_DIRTY_PACKET_FLAG,
  SPARSE_CM12_DIRTY_SCHEDULER_FLAG,
  SPARSE_CM12_DIRTY_SCHEDULER_HEADER,
  SPARSE_CM12_DIRTY_SCHEDULER_HEADER_WORDS,
  SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL,
  SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL_WORDS,
  SPARSE_CM12_DIRTY_SCHEDULER_MAGIC,
  SPARSE_CM12_DIRTY_SCHEDULER_PACKET,
  SPARSE_CM12_DIRTY_SCHEDULER_PACKET_WORDS,
  SPARSE_CM12_DIRTY_SCHEDULER_QUEUE,
  SPARSE_CM12_DIRTY_SCHEDULER_QUEUE_HEADER_WORDS,
  SPARSE_CM12_DIRTY_SCHEDULER_VERSION,
  type SparseCM12DirtySchedulerLayout,
} from "./sparse-cm12-dirty-scheduler";
import {
  SPARSE_CM12_DIRTY_CAUSE_BIT,
  SPARSE_CM12_DIRTY_GPU_HEADER,
  SPARSE_CM12_DIRTY_GPU_HEADER_WORDS,
  SPARSE_CM12_DIRTY_GPU_RECORD,
  SPARSE_CM12_DIRTY_GPU_RECORD_WORDS,
  SPARSE_CM12_DIRTY_PUBLICATION_FLAG,
  SPARSE_CM12_DIRTY_STAGE_COUNT,
} from "../../core/sparse-cm12-dirty-visualizations";

export interface SparseCM12DirtySchedulerWGSLOptions {
  readonly layout: SparseCM12DirtySchedulerLayout;
  /** Name of an existing `var<storage, read_write> ...: array<atomic<u32>>`. */
  readonly arenaName: string;
  /** Consumer workgroup width used when indirect dispatch counts are sealed. */
  readonly workgroupSize?: number;
  /** Prefix permits more than one independently generated scheduler library. */
  readonly prefix?: string;
}

function identifier(value: string, label: string): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError(`${label} must be a WGSL identifier`);
  return value;
}

/**
 * Emits binding-free WGSL helpers. The caller owns pass boundaries and entry
 * points, while this library owns stable packet initialization, journal/queue
 * bounds, generation checks, coverage receipts, and fail-closed publication.
 */
export function createSparseCM12DirtySchedulerWGSL(
  options: SparseCM12DirtySchedulerWGSLOptions,
): string {
  const arena = identifier(options.arenaName, "arenaName");
  const p = identifier(options.prefix ?? "cm12Dirty", "prefix");
  const workgroupSize = options.workgroupSize ?? 64;
  if (!Number.isSafeInteger(workgroupSize) || workgroupSize < 1 || workgroupSize > 1024) {
    throw new RangeError("Sparse CM12 dirty scheduler workgroupSize must be in 1..1024");
  }
  const layout = options.layout;
  const allStages = (1 << SPARSE_CM12_DIRTY_STAGE_COUNT) - 1;
  const h = (word: number) => `${layout.schedulerBaseWords + word}u`;
  const q = (word: number) => `${layout.queueHeaderBaseWords + word}u`;
  return /* wgsl */ `
const ${p}Magic:u32=0x${SPARSE_CM12_DIRTY_SCHEDULER_MAGIC.toString(16)}u;
const ${p}Version:u32=${SPARSE_CM12_DIRTY_SCHEDULER_VERSION}u;
const ${p}Invalid:u32=0x${SPARSE_CM12_DIRTY_INVALID.toString(16)}u;
const ${p}InitLock:u32=0x80000000u;
const ${p}AllStages:u32=${allStages}u;
const ${p}StageCount:u32=${SPARSE_CM12_DIRTY_STAGE_COUNT}u;
const ${p}DomainCount:u32=${SPARSE_CM12_DIRTY_DOMAIN_COUNT}u;
const ${p}PacketWords:u32=${SPARSE_CM12_DIRTY_SCHEDULER_PACKET_WORDS}u;
const ${p}JournalWords:u32=${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL_WORDS}u;
const ${p}PacketBase:u32=${layout.packetBaseWords}u;
const ${p}JournalBase:u32=${layout.journalBaseWords}u;
const ${p}FrontierA:u32=${layout.frontierABaseWords}u;
const ${p}FrontierB:u32=${layout.frontierBBaseWords}u;
const ${p}PublicationBase:u32=${layout.publicationBaseWords}u;
const ${p}StableTileCount:u32=${layout.stableTileCount}u;

fn ${p}Load(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn ${p}Store(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn ${p}PacketAt(tile:u32)->u32{return ${p}PacketBase+tile*${p}PacketWords;}
fn ${p}RecordAt(tile:u32)->u32{
  return ${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER_WORDS}u
    +tile*${SPARSE_CM12_DIRTY_GPU_RECORD_WORDS}u;
}
fn ${p}HeaderValid()->bool{
  return arrayLength(&${arena})>=${layout.totalWords}u
    &&${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.magic)})==${p}Magic
    &&${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.version)})==${p}Version
    &&${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.headerWords)})==${SPARSE_CM12_DIRTY_SCHEDULER_HEADER_WORDS}u
    &&${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.packetWords)})==${p}PacketWords
    &&${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.journalWords)})==${p}JournalWords
    &&${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.stableTileCount)})==${p}StableTileCount
    &&${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.totalWords)})==${layout.totalWords}u;
}
fn ${p}LowestBitIndex(mask:u32)->u32{
  return select(${p}Invalid,firstTrailingBit(mask),mask!=0u);
}
fn ${p}Fail(code:u32,tile:u32,stageMask:u32,domainMask:u32){
  let won=atomicCompareExchangeWeak(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.faultCode)}],
    ${SPARSE_CM12_DIRTY_FAULT.none}u,code).exchanged;
  atomicOr(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)}],
    ${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.fault | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.rejected}u);
  atomicAnd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)}],
    ~${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.accepted}u);
  if(won){
    ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultTile)},tile);
    ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultStage)},${p}LowestBitIndex(stageMask));
    ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultDomain)},${p}LowestBitIndex(domainMask));
  }
}
fn ${p}BeginCandidate(expectedAccepted:u32,candidate:u32)->bool{
  if(!${p}HeaderValid()){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.invalidHeader}u,${p}Invalid,0u,0u);return false;
  }
  let oldFlags=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)});
  let pendingPost=${p}Load(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.post)});
  if(candidate>=${p}InitLock||expectedAccepted>=${p}InitLock-1u
    ||candidate!=expectedAccepted+1u
    ||${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.acceptedGeneration)})!=expectedAccepted
    ||(oldFlags&${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.accepted}u)==0u){
    ${p}Fail(select(${SPARSE_CM12_DIRTY_FAULT.invalidEpochTransition}u,
      ${SPARSE_CM12_DIRTY_FAULT.generationExhausted}u,expectedAccepted>=${p}InitLock-1u),
      ${p}Invalid,0u,0u);return false;
  }
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)},candidate);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.preEpochGeneration)},candidate);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.postEpochGeneration)},expectedAccepted);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)},
    ${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.initialized | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.preOpen}u);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.faultCode)},0u);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.journalCount)},0u);
  for(var at=${SPARSE_CM12_DIRTY_SCHEDULER_HEADER.expectedPreWrites}u;
      at<=${SPARSE_CM12_DIRTY_SCHEDULER_HEADER.maxClosureDepth}u;at+=1u){
    ${p}Store(${layout.schedulerBaseWords}u+at,0u);
  }
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultTile)},${p}Invalid);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultStage)},${p}Invalid);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultDomain)},${p}Invalid);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.initializedPacketCount)},0u);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.finalizedPacketCount)},0u);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.promotedPostCount)},0u);
  for(var at=0u;at<${SPARSE_CM12_DIRTY_SCHEDULER_QUEUE_HEADER_WORDS}u;at+=1u){
    ${p}Store(${layout.queueHeaderBaseWords}u+at,0u);
  }
  // Frontier B still contains the prior accepted epoch's post list. Preserve
  // its count as the first closure-B dispatch until promotion consumes it.
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureB)},pendingPost);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureB + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchX)},
    (pendingPost+${workgroupSize - 1}u)/${workgroupSize}u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.pre + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchY)},1u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.pre + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchZ)},1u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureA + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchY)},1u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureA + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchZ)},1u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureB + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchY)},1u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureB + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchZ)},1u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.post + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchY)},1u);
  ${p}Store(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.post + SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.dispatchZ)},1u);
  return true;
}
fn ${p}EnsurePacket(tile:u32)->u32{
  if(tile>=${p}StableTileCount){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.invalidStableTile}u,tile,0u,0u);return ${p}Invalid;
  }
  let packet=${p}PacketAt(tile);
  let generation=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)});
  for(var spin=0u;spin<64u;spin+=1u){
    let observed=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u);
    if(observed==generation){return packet;}
    if(observed!=(generation|${p}InitLock)){
      let claim=atomicCompareExchangeWeak(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u],
        observed,generation|${p}InitLock);
      if(claim.exchanged){
        for(var word=1u;word<${p}PacketWords;word+=1u){${p}Store(packet+word,0u);}
        ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.preQueuedGeneration}u,${p}Invalid);
        ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postQueuedGeneration}u,${p}Invalid);
        ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.firstJournal}u,${p}Invalid);
        ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.producerGeneration}u,generation);
        ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u,generation);
        atomicAdd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.initializedPacketCount)}],1u);
        return packet;
      }
    }
  }
  ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.packetInitializationRace}u,tile,0u,0u);
  return ${p}Invalid;
}
// Fuse additional producer causes into an already-covered packet without
// consuming another bounded journal slot. The first event remains the coverage
// authority; this only enriches its per-tile provenance census.
fn ${p}MergeCause(tile:u32,stageMask:u32,causeMask:u32,direct:bool)->bool{
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return false;}
  atomicOr(&${arena}[packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.closureStageMask}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.directStageMask}u,direct)],stageMask);
  atomicOr(&${arena}[packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.inheritedCauseMask}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.originCauseMask}u,direct)],causeMask);
  return true;
}
fn ${p}StageMask(tile:u32)->u32{
  if(tile>=${p}StableTileCount){return 0u;}
  let packet=${p}PacketAt(tile);
  if(${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u)
    !=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)})){return 0u;}
  return ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.directStageMask}u)
    |${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.closureStageMask}u);
}
fn ${p}DirectStageMask(tile:u32)->u32{
  if(tile>=${p}StableTileCount){return 0u;}
  let packet=${p}PacketAt(tile);
  if(${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u)
    !=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)})){return 0u;}
  return ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.directStageMask}u);
}
fn ${p}ClosureStageMask(tile:u32)->u32{
  if(tile>=${p}StableTileCount){return 0u;}
  let packet=${p}PacketAt(tile);
  if(${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u)
    !=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)})){return 0u;}
  return ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.closureStageMask}u);
}
fn ${p}OriginCauseMask(tile:u32)->u32{
  if(tile>=${p}StableTileCount){return 0u;}
  let packet=${p}PacketAt(tile);
  if(${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u)
    !=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)})){return 0u;}
  return ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.originCauseMask}u);
}
fn ${p}InheritedCauseMask(tile:u32)->u32{
  if(tile>=${p}StableTileCount){return 0u;}
  let packet=${p}PacketAt(tile);
  if(${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u)
    !=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)})){return 0u;}
  return ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.inheritedCauseMask}u);
}
fn ${p}ResolvedClosureDepths(tile:u32)->u32{
  if(tile>=${p}StableTileCount){return 0u;}
  let packet=${p}PacketAt(tile);
  if(${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u)
    !=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)})){return 0u;}
  return ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.resolvedClosureDepths}u);
}
fn ${p}SetPackedLaneMaximum(tile:u32,at:u32,shift:u32,value:u32)->bool{
  for(var attempt=0u;attempt<64u;attempt+=1u){
    let old=${p}Load(at);let previous=(old>>shift)&15u;
    if(value<=previous){return false;}
    let desired=(old&~(15u<<shift))|(value<<shift);
    if(atomicCompareExchangeWeak(&${arena}[at],old,desired).exchanged){return true;}
  }
  ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.provenanceGap}u,tile,0u,0u);
  return false;
}
fn ${p}SetOriginStages(tile:u32,packet:u32,stageMask:u32,originStage:u32){
  if(originStage==0u){return;}
  for(var stage=0u;stage<${p}StageCount;stage+=1u){
    if((stageMask&(1u<<stage))!=0u){
      let shift=stage*3u;
      var published=false;
      for(var attempt=0u;attempt<64u;attempt+=1u){
        let old=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.packedOriginStages}u);
        if(((old>>shift)&7u)!=0u){published=true;break;}
        if(atomicCompareExchangeWeak(
          &${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.packedOriginStages}u],
          old,old|(originStage<<shift)).exchanged){published=true;break;}
      }
      if(!published){${p}Fail(${SPARSE_CM12_DIRTY_FAULT.provenanceGap}u,tile,
        stageMask,0u);return;}
    }
  }
}
fn ${p}AppendJournal(tile:u32,epoch:u32,stageMask:u32,domainMask:u32,causeMask:u32,
  closureDepths:u32,originStage:u32,coverageRequired:bool,flags:u32)->u32{
  let slot=atomicAdd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.journalCount)}],1u);
  if(slot>=${layout.journalCapacity}u){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.journalCapacity}u,tile,stageMask,domainMask);return ${p}Invalid;
  }
  let at=${p}JournalBase+slot*${p}JournalWords;
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.stableTile}u,tile);
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.epochAndFlags}u,(epoch&1u)|(flags<<8u));
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.stageMask}u,stageMask);
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.domainMask}u,domainMask);
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.causeMask}u,causeMask);
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.closureDepthsAndOrigin}u,
    closureDepths|((originStage&7u)<<24u));
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.producerGeneration}u,
    ${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)}));
  ${p}Store(at+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.receiptState}u,
    select(0u,1u,coverageRequired));
  let packet=${p}PacketAt(tile);
  atomicMin(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.firstJournal}u],slot);
  return slot;
}
fn ${p}QueuePacket(tile:u32,packet:u32,epoch:u32)->bool{
  let generation=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)});
  let stampAt=packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.preQueuedGeneration}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postQueuedGeneration}u,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u);
  if(atomicExchange(&${arena}[stampAt],generation)==generation){return true;}
  let queueAt=select(${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.pre)},
    ${q(SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.post)},epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u);
  let frontier=select(${p}FrontierA,${p}FrontierB,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u);
  let slot=atomicAdd(&${arena}[queueAt],1u);
  if(slot>=${p}StableTileCount){
    ${p}Fail(select(${SPARSE_CM12_DIRTY_FAULT.preQueueCapacity}u,
      ${SPARSE_CM12_DIRTY_FAULT.postQueueCapacity}u,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u),
      tile,0u,0u);return false;
  }
  ${p}Store(frontier+slot,tile);return true;
}
fn ${p}RecordEvent(tile:u32,epoch:u32,stageMask:u32,domainMask:u32,causeMask:u32,
  originStage:u32,direct:bool,coverageRequired:bool)->u32{
  if(stageMask==0u||(stageMask&~${p}AllStages)!=0u||domainMask==0u
    ||(domainMask>>${p}DomainCount)!=0u||epoch>1u||originStage>${p}StageCount){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.invalidStageOrDomain}u,tile,stageMask,domainMask);return ${p}Invalid;
  }
  let flags=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)});
  let epochOpen=select((flags&${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.preOpen}u)!=0u,
    (flags&${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.postOpen}u)!=0u,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u);
  if(!epochOpen){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.invalidEpochTransition}u,tile,stageMask,domainMask);return ${p}Invalid;
  }
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return ${p}Invalid;}
  let stageAt=packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.preStageMask}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postStageMask}u,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u);
  let domainAt=packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.preDomainMask}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postDomainMask}u,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u);
  atomicOr(&${arena}[stageAt],stageMask);atomicOr(&${arena}[domainAt],domainMask);
  atomicOr(&${arena}[packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.closureStageMask}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.directStageMask}u,direct)],stageMask);
  atomicOr(&${arena}[packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.inheritedCauseMask}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.originCauseMask}u,direct)],causeMask);
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.flags}u],
    select(${SPARSE_CM12_DIRTY_PACKET_FLAG.postSeen}u,${SPARSE_CM12_DIRTY_PACKET_FLAG.preSeen}u,
      epoch==${SPARSE_CM12_DIRTY_EPOCH.pre}u));
  atomicAdd(&${arena}[packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.preEventCount}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postEventCount}u,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u)],1u);
  ${p}SetOriginStages(tile,packet,stageMask,originStage);
  if(coverageRequired){
    atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.expectedCoverageStageMask}u],stageMask);
    atomicAdd(&${arena}[select(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.expectedPreWrites)},
      ${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.expectedPostWrites)},epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u)],1u);
  }
  let receipt=${p}AppendJournal(tile,epoch,stageMask,domainMask,causeMask,0u,
    originStage,coverageRequired,select(0u,1u,coverageRequired));
  if(receipt==${p}Invalid||!${p}QueuePacket(tile,packet,epoch)){return ${p}Invalid;}
  return receipt;
}
fn ${p}RecordCoverage(tile:u32,epoch:u32,stageMask:u32,domainMask:u32,receipt:u32)->bool{
  if(receipt>=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.journalCount)})
    ||receipt>=${layout.journalCapacity}u){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.provenanceGap}u,tile,stageMask,domainMask);return false;
  }
  let journal=${p}JournalBase+receipt*${p}JournalWords;
  let journalEpoch=${p}Load(journal+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.epochAndFlags}u)&1u;
  if(${p}Load(journal+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.stableTile}u)!=tile
    ||journalEpoch!=epoch
    ||${p}Load(journal+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.stageMask}u)!=stageMask
    ||${p}Load(journal+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.domainMask}u)!=domainMask
    ||!atomicCompareExchangeWeak(
      &${arena}[journal+${SPARSE_CM12_DIRTY_SCHEDULER_JOURNAL.receiptState}u],1u,2u).exchanged){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.provenanceGap}u,tile,stageMask,domainMask);return false;
  }
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return false;}
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.coveredStageMask}u],stageMask);
  atomicAdd(&${arena}[packet+select(${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.preCoverageCount}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postCoverageCount}u,epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u)],1u);
  atomicAdd(&${arena}[select(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.coveredPreWrites)},
    ${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.coveredPostWrites)},epoch==${SPARSE_CM12_DIRTY_EPOCH.post}u)],1u);
  return true;
}
fn ${p}RequestClosure(tile:u32,stage:u32,depth:u32,domainMask:u32)->bool{
  if(stage>=${p}StageCount||depth>15u||depth==0u){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.invalidStageOrDomain}u,tile,1u<<stage,domainMask);return false;
  }
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return false;}
  let changed=${p}SetPackedLaneMaximum(tile,
    packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.requestedClosureDepths}u,stage*4u,depth);
  if(changed){
    atomicAdd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.closureRequestedCount)}],1u);
    atomicMax(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.maxClosureDepth)}],depth);
  }
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.closureStageMask}u],1u<<stage);
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.inheritedCauseMask}u],
    ${SPARSE_CM12_DIRTY_CAUSE_BIT.dependencyClosure}u);
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.flags}u],
    ${SPARSE_CM12_DIRTY_PACKET_FLAG.closureRequested}u);
  return changed;
}
fn ${p}ResolveClosure(tile:u32,stage:u32,depth:u32)->bool{
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return false;}
  let changed=${p}SetPackedLaneMaximum(tile,
    packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.resolvedClosureDepths}u,stage*4u,depth);
  if(changed){atomicAdd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.closureResolvedCount)}],1u);}
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.flags}u],
    ${SPARSE_CM12_DIRTY_PACKET_FLAG.closureResolved}u);
  return changed;
}
fn ${p}MarkExecuted(tile:u32,stageMask:u32,consumerGeneration:u32){
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return;}
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.executedStageMask}u],stageMask);
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.flags}u],
    ${SPARSE_CM12_DIRTY_PACKET_FLAG.executed}u);
  ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.consumerGeneration}u,consumerGeneration);
}
fn ${p}MarkSkipped(tile:u32,stageMask:u32,consumerGeneration:u32){
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return;}
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.skippedStageMask}u],stageMask);
  atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.flags}u],
    ${SPARSE_CM12_DIRTY_PACKET_FLAG.skipped}u);
  ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.consumerGeneration}u,consumerGeneration);
}
fn ${p}SealQueue(queueOffset:u32){
  let count=${p}Load(${layout.queueHeaderBaseWords}u+queueOffset);
  ${p}Store(${layout.queueHeaderBaseWords}u+queueOffset+1u,(count+${workgroupSize - 1}u)/${workgroupSize}u);
  ${p}Store(${layout.queueHeaderBaseWords}u+queueOffset+2u,1u);
  ${p}Store(${layout.queueHeaderBaseWords}u+queueOffset+3u,1u);
}
fn ${p}ResetQueue(queueOffset:u32){
  ${p}Store(${layout.queueHeaderBaseWords}u+queueOffset,0u);
  ${p}Store(${layout.queueHeaderBaseWords}u+queueOffset+1u,0u);
  ${p}Store(${layout.queueHeaderBaseWords}u+queueOffset+2u,1u);
  ${p}Store(${layout.queueHeaderBaseWords}u+queueOffset+3u,1u);
}
fn ${p}EnqueueClosure(tile:u32,useB:bool)->bool{
  if(tile>=${p}StableTileCount){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.invalidStableTile}u,tile,0u,0u);return false;
  }
  let queueOffset=select(${SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureA}u,
    ${SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.closureB}u,useB);
  let slot=atomicAdd(&${arena}[${layout.queueHeaderBaseWords}u+queueOffset],1u);
  if(slot>=${p}StableTileCount){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.closureQueueCapacity}u,tile,0u,0u);return false;
  }
  ${p}Store(select(${p}FrontierA,${p}FrontierB,useB)+slot,tile);return true;
}
fn ${p}SealPreOpenPost(){
  atomicAnd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)}],
    ~${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.preOpen}u);
  atomicOr(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)}],
    ${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.preSealed | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.postOpen}u);
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.postEpochGeneration)},
    ${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)}));
  ${p}SealQueue(${SPARSE_CM12_DIRTY_SCHEDULER_QUEUE.pre}u);
}
fn ${p}FinalizePacket(tile:u32)->bool{
  let packet=${p}EnsurePacket(tile);if(packet==${p}Invalid){return false;}
  let expected=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.expectedCoverageStageMask}u);
  let covered=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.coveredStageMask}u);
  let uncovered=expected&~covered;
  let requested=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.requestedClosureDepths}u);
  let resolved=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.resolvedClosureDepths}u);
  var fault=0u;
  if(uncovered!=0u){fault=${SPARSE_CM12_DIRTY_FAULT.missingCoverage}u;}
  if(requested!=resolved){fault=${SPARSE_CM12_DIRTY_FAULT.unresolvedClosure}u;}
  ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.uncoveredStageMask}u,uncovered);
  if(fault!=0u){
    ${p}Store(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.faultCode}u,fault);
    atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.flags}u],
      ${SPARSE_CM12_DIRTY_PACKET_FLAG.fault}u);
    ${p}Fail(fault,tile,uncovered|${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.closureStageMask}u),0u);
  }else{
    atomicOr(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.flags}u],
      ${SPARSE_CM12_DIRTY_PACKET_FLAG.coverageComplete}u);
  }
  let record=${p}RecordAt(tile);
  let direct=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.directStageMask}u);
  let closure=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.closureStageMask}u);
  let executed=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.executedStageMask}u);
  let skipped=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.skippedStageMask}u);
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.directStageMask}u,direct);
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.closureStageMask}u,closure);
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.processedStageMask}u,executed);
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.skippedStageMask}u,skipped);
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.unknownStageMask}u,select(0u,${p}AllStages,fault!=0u));
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.uncoveredWriteStageMask}u,uncovered);
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.originCauseMask}u,
    ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.originCauseMask}u));
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.inheritedCauseMask}u,
    ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.inheritedCauseMask}u));
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.packedClosureDepths}u,requested);
  ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.packedOriginStages}u,
    ${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.packedOriginStages}u));
  let producer=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.producerGeneration}u)&0xffffu;
  let consumer=${p}Load(packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.consumerGeneration}u)&0xffffu;
  for(var stage=0u;stage<${p}StageCount;stage+=1u){
    ${p}Store(record+${SPARSE_CM12_DIRTY_GPU_RECORD.stageGenerationPairs}u+stage,
      producer|(consumer<<16u));
  }
  let generation=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)});
  if(atomicExchange(&${arena}[packet+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.finalizedGeneration}u],
      generation)!=generation){
    atomicAdd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.finalizedPacketCount)}],1u);
  }
  return fault==0u;
}
fn ${p}FinalizeEpoch()->bool{
  atomicAnd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)}],
    ~${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.postOpen}u);
  let expectedPre=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.expectedPreWrites)});
  let coveredPre=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.coveredPreWrites)});
  let expectedPost=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.expectedPostWrites)});
  let coveredPost=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.coveredPostWrites)});
  if(expectedPre!=coveredPre||expectedPost!=coveredPost){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.missingCoverage}u,${p}Invalid,0u,0u);
  }
  if(${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.initializedPacketCount)})
    !=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.finalizedPacketCount)})){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.provenanceGap}u,${p}Invalid,0u,0u);
  }
  let fault=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.faultCode)})!=0u;
  let candidate=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.candidateGeneration)});
  ${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.flags)},
    ${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.initialized | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.complete}u
      |select(${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.accepted}u,
        ${SPARSE_CM12_DIRTY_SCHEDULER_FLAG.rejected | SPARSE_CM12_DIRTY_SCHEDULER_FLAG.fault}u,fault));
  if(!fault){${p}Store(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.acceptedGeneration)},candidate);}
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.acceptedGeneration}u,
    select(${p}Load(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.acceptedGeneration}u),candidate,!fault));
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.candidateGeneration}u,candidate);
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.provenanceGeneration}u,candidate);
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.publicationFlags}u,
    ${SPARSE_CM12_DIRTY_PUBLICATION_FLAG.complete}u
      |select(${SPARSE_CM12_DIRTY_PUBLICATION_FLAG.accepted}u,
        ${SPARSE_CM12_DIRTY_PUBLICATION_FLAG.rejected | SPARSE_CM12_DIRTY_PUBLICATION_FLAG.capacityOrProvenanceFault}u,
        fault));
  let uncoveredPre=select(coveredPre-expectedPre,expectedPre-coveredPre,expectedPre>=coveredPre);
  let uncoveredPost=select(coveredPost-expectedPost,expectedPost-coveredPost,expectedPost>=coveredPost);
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.uncoveredWriteCount}u,
    uncoveredPre+uncoveredPost);
  let firstTile=${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultTile)});
  let tilesPerBrick=${layout.tilesPerLogicalBrickAxis ** 3}u;
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.firstFaultLogicalBrick}u,
    select(${p}Invalid,firstTile/tilesPerBrick,firstTile!=${p}Invalid));
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.firstFaultLocalTile}u,
    select(${p}Invalid,firstTile%tilesPerBrick,firstTile!=${p}Invalid));
  ${p}Store(${p}PublicationBase+${SPARSE_CM12_DIRTY_GPU_HEADER.firstFaultStage}u,
    ${p}Load(${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.firstFaultStage)}));
  return !fault;
}
fn ${p}PromotePostToPre(tile:u32,oldGeneration:u32)->bool{
  if(tile>=${p}StableTileCount){return false;}
  let oldPacket=${p}PacketAt(tile);
  if(${p}Load(oldPacket+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.generation}u)!=oldGeneration){
    ${p}Fail(${SPARSE_CM12_DIRTY_FAULT.provenanceGap}u,tile,0u,0u);return false;
  }
  let stages=${p}Load(oldPacket+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postStageMask}u);
  let domains=${p}Load(oldPacket+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.postDomainMask}u);
  let causes=${p}Load(oldPacket+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.originCauseMask}u)
    |${p}Load(oldPacket+${SPARSE_CM12_DIRTY_SCHEDULER_PACKET.inheritedCauseMask}u);
  if(stages==0u){return true;}
  let promoted=${p}RecordEvent(tile,${SPARSE_CM12_DIRTY_EPOCH.pre}u,stages,domains,causes,
    0u,false,true)!=${p}Invalid;
  if(promoted){atomicAdd(&${arena}[${h(SPARSE_CM12_DIRTY_SCHEDULER_HEADER.promotedPostCount)}],1u);}
  return promoted;
}
`;
}
