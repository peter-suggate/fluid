import {
  SPARSE_CM12_SCALAR_RESULT_CANDIDATE,
  SPARSE_CM12_SCALAR_RESULT_CANDIDATE_WORDS,
  SPARSE_CM12_SCALAR_RESULT_FAULT,
  SPARSE_CM12_SCALAR_RESULT_FLAG,
  SPARSE_CM12_SCALAR_RESULT_HEADER,
  SPARSE_CM12_SCALAR_RESULT_INVALID,
  SPARSE_CM12_SCALAR_RESULT_LEAF,
  SPARSE_CM12_SCALAR_RESULT_LEAF_WORDS,
  SPARSE_CM12_SCALAR_RESULT_MAGIC,
  SPARSE_CM12_SCALAR_RESULT_PHASE,
  SPARSE_CM12_SCALAR_RESULT_TILE,
  SPARSE_CM12_SCALAR_RESULT_TILE_WORDS,
  SPARSE_CM12_SCALAR_RESULT_VERSION,
  type SparseCM12ScalarResultLayout,
} from "./sparse-cm12-scalar-result-receipts";

const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError("invalid WGSL identifier");
  }
  return value;
};

/**
 * Storage helpers only. The caller supplies `array<atomic<u32>>`; no physics
 * binding is visible here, so SRR1 cannot mutate or canonicalize solver state.
 */
export function createSparseCM12ScalarResultReceiptsWGSL(options: {
  readonly layout: SparseCM12ScalarResultLayout;
  readonly arenaName?: string;
}): string {
  const l = options.layout;
  const arena = identifier(options.arenaName ?? "scalarResultReceipts");
  const h = SPARSE_CM12_SCALAR_RESULT_HEADER;
  const t = SPARSE_CM12_SCALAR_RESULT_TILE;
  const c = SPARSE_CM12_SCALAR_RESULT_CANDIDATE;
  const f = SPARSE_CM12_SCALAR_RESULT_FLAG;
  const leaf = SPARSE_CM12_SCALAR_RESULT_LEAF;
  return /* wgsl */ `
const cm12SRRInvalid:u32=0x${SPARSE_CM12_SCALAR_RESULT_INVALID.toString(16)}u;
const cm12SRRTileCapacity:u32=${l.tileCapacity}u;
const cm12SRRLeafCapacity:u32=${l.leafCapacity}u;
const cm12SRRTreeLeafCapacity:u32=${l.treeLeafCapacity}u;
fn cm12SRRLoad(word:u32)->u32{return atomicLoad(&${arena}[word]);}
fn cm12SRRStore(word:u32,value:u32){atomicStore(&${arena}[word],value);}
fn cm12SRRTileAt(tile:u32)->u32{
  return ${l.tileBaseWords}u+${SPARSE_CM12_SCALAR_RESULT_TILE_WORDS}u*tile;
}
fn cm12SRRCandidateAt(tile:u32)->u32{
  return ${l.candidateReceiptBaseWords}u+${SPARSE_CM12_SCALAR_RESULT_CANDIDATE_WORDS}u*tile;
}
fn cm12SRRLeafAt(index:u32)->u32{
  return ${l.leafBaseWords}u+${SPARSE_CM12_SCALAR_RESULT_LEAF_WORDS}u*index;
}
fn cm12SRRHeaderValid()->bool{
  return arrayLength(&${arena})==${l.totalWords}u
    &&cm12SRRLoad(${h.magic}u)==0x${SPARSE_CM12_SCALAR_RESULT_MAGIC.toString(16)}u
    &&cm12SRRLoad(${h.version}u)==${SPARSE_CM12_SCALAR_RESULT_VERSION}u
    &&cm12SRRLoad(${h.brickFineResolution}u)==${l.brickFineResolution}u
    &&cm12SRRLoad(${h.presentationPageResolution}u)==${l.presentationPageResolution}u
    &&cm12SRRLoad(${h.tileCapacity}u)==cm12SRRTileCapacity
    &&cm12SRRLoad(${h.leafCapacity}u)==cm12SRRLeafCapacity
    &&cm12SRRLoad(${h.tileBase}u)==${l.tileBaseWords}u
    &&cm12SRRLoad(${h.candidateReceiptBase}u)==${l.candidateReceiptBaseWords}u
    &&cm12SRRLoad(${h.leafBase}u)==${l.leafBaseWords}u
    &&cm12SRRLoad(${h.workListBase}u)==${l.workListBaseWords}u
    &&cm12SRRLoad(${h.workTreeBase}u)==${l.workTreeBaseWords}u
    &&cm12SRRLoad(${h.cleanTreeBase}u)==${l.cleanTreeBaseWords}u
    &&cm12SRRLoad(${h.treeWords}u)==${l.treeWords}u
    &&cm12SRRLoad(${h.totalWords}u)==${l.totalWords}u;
}
fn cm12SRRFail(code:u32,tile:u32){
  let claimed=atomicCompareExchangeWeak(&${arena}[${h.fault}u],0u,code);
  if(claimed.exchanged){cm12SRRStore(${h.firstFaultTile}u,tile);}
  cm12SRRStore(${h.scheduledWorkCount}u,0u);
  cm12SRRStore(${h.phase}u,${SPARSE_CM12_SCALAR_RESULT_PHASE.fault}u);
}
fn cm12SRRReject(code:u32,tile:u32){
  cm12SRRStore(${h.fault}u,code);cm12SRRStore(${h.firstFaultTile}u,tile);
  cm12SRRStore(${h.scheduledWorkCount}u,0u);
  cm12SRRStore(${h.phase}u,${SPARSE_CM12_SCALAR_RESULT_PHASE.rejected}u);
}
fn cm12SRRBegin(generation:u32,topologyGeneration:u32,sourceParity:u32)->bool{
  if(!cm12SRRHeaderValid()){
    cm12SRRFail(${SPARSE_CM12_SCALAR_RESULT_FAULT.header}u,cm12SRRInvalid);return false;
  }
  let priorPhase=cm12SRRLoad(${h.phase}u);
  if(priorPhase!=${SPARSE_CM12_SCALAR_RESULT_PHASE.accepted}u
    &&priorPhase!=${SPARSE_CM12_SCALAR_RESULT_PHASE.rejected}u){
    cm12SRRFail(${SPARSE_CM12_SCALAR_RESULT_FAULT.phase}u,cm12SRRInvalid);return false;
  }
  let acceptedGeneration=cm12SRRLoad(${h.acceptedGeneration}u);
  let constructionBootstrap=acceptedGeneration==0u
    &&cm12SRRLoad(${h.bootstrapFrames}u)==0u;
  // FCA1 begins at construction generation one, so its first candidate is
  // generation two. A fresh SRR1 adopts that positive construction baseline
  // exactly once; every subsequent frame remains strict accepted+1.
  if(generation==0u||(!constructionBootstrap&&generation!=acceptedGeneration+1u)
    ||topologyGeneration==0u||sourceParity>1u){
    cm12SRRFail(${SPARSE_CM12_SCALAR_RESULT_FAULT.generation}u,cm12SRRInvalid);return false;
  }
  if(priorPhase==${SPARSE_CM12_SCALAR_RESULT_PHASE.rejected}u){
    let workCount=cm12SRRLoad(${l.workTreeBaseWords}u);
    for(var rank=0u;rank<workCount;rank+=1u){
      let tile=cm12SRRRankSelect(${l.workTreeBaseWords}u,rank);
      if(tile<cm12SRRTileCapacity){cm12SRRStore(cm12SRRCandidateAt(tile)+${c.generation}u,0u);}
    }
  }
  cm12SRRStore(${h.candidateGeneration}u,generation);
  cm12SRRStore(${h.topologyGeneration}u,topologyGeneration);
  cm12SRRStore(${h.sourceParity}u,sourceParity);
  cm12SRRStore(${h.leafCount}u,0u);
  cm12SRRStore(${h.scheduledWorkCount}u,0u);
  cm12SRRStore(${h.executedWorkCount}u,0u);
  cm12SRRStore(${h.fault}u,0u);
  cm12SRRStore(${h.firstFaultTile}u,cm12SRRInvalid);
  cm12SRRStore(${h.phase}u,${SPARSE_CM12_SCALAR_RESULT_PHASE.collecting}u);
  return true;
}
fn cm12SRRTreeSet(base:u32,tile:u32,value:u32){
  var node=cm12SRRTreeLeafCapacity-1u+tile;
  let previous=atomicExchange(&${arena}[base+node],value);
  if(previous==value){return;}
  let add=value>previous;
  loop{
    if(node==0u){break;}
    node=(node-1u)/2u;
    if(add){atomicAdd(&${arena}[base+node],1u);}
    else{atomicSub(&${arena}[base+node],1u);}
  }
}
fn cm12SRRSetClassification(tile:u32,clean:bool){
  let at=cm12SRRTileAt(tile);
  let receiptMask=${f.producerExecuted | f.fullWordCoverage
    | f.dualBankExact | f.dependencyExact}u;
  var flags=cm12SRRLoad(at+${t.flags}u)&receiptMask;
  flags|=${f.classified}u|select(${f.work}u,${f.clean}u,clean);
  cm12SRRStore(at+${t.flags}u,flags);
  cm12SRRTreeSet(${l.workTreeBaseWords}u,tile,select(1u,0u,clean));
  cm12SRRTreeSet(${l.cleanTreeBaseWords}u,tile,select(0u,1u,clean));
}
fn cm12SRRAppendCandidate(tile:u32,cause:u32)->bool{
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.collecting}u
    ||tile>=cm12SRRTileCapacity){return false;}
  let at=cm12SRRTileAt(tile);
  let generation=cm12SRRLoad(${h.candidateGeneration}u);
  let claimed=atomicCompareExchangeWeak(&${arena}[at+${t.candidateGeneration}u],
    generation-1u,generation);
  if(!claimed.exchanged){
    if(claimed.old_value==generation){atomicOr(&${arena}[at+${t.causeMask}u],cause);return true;}
    // Producers may legitimately skip generations for this tile. A bounded
    // Bounded CAS retries claim any older generation but never a future one.
    var published=false;
    for(var attempt=0u;attempt<64u;attempt+=1u){
      let old=cm12SRRLoad(at+${t.candidateGeneration}u);
      if(old==generation){atomicOr(&${arena}[at+${t.causeMask}u],cause);return true;}
      if(old>generation){return false;}
      let retry=atomicCompareExchangeWeak(&${arena}[at+${t.candidateGeneration}u],old,generation);
      if(retry.exchanged){published=true;break;}
    }
    if(!published){cm12SRRFail(${SPARSE_CM12_SCALAR_RESULT_FAULT.atomicContention}u,tile);
      return false;}
  }
  atomicOr(&${arena}[at+${t.causeMask}u],cause);
  cm12SRRSetClassification(tile,false);
  let index=atomicAdd(&${arena}[${h.leafCount}u],1u);
  if(index>=cm12SRRLeafCapacity){
    cm12SRRFail(${SPARSE_CM12_SCALAR_RESULT_FAULT.leafOverflow}u,tile);return false;
  }
  let leafAt=cm12SRRLeafAt(index);
  cm12SRRStore(leafAt+${leaf.tile}u,tile);
  cm12SRRStore(leafAt+${leaf.generation}u,generation);
  cm12SRRStore(leafAt+${leaf.causeMask}u,cause);
  return true;
}
fn cm12SRRCanSkip(tile:u32,topologyGeneration:u32,bank0Generation:u32,
  bank1Generation:u32,headResultGeneration:u32,expectedWords:u32,
  dependencyGeneration:u32)->bool{
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.collecting}u
    ||tile>=cm12SRRTileCapacity){return false;}
  let at=cm12SRRTileAt(tile);
  let required=${f.producerExecuted | f.fullWordCoverage
    | f.dualBankExact | f.dependencyExact}u;
  return cm12SRRLoad(at+${t.topologyGeneration}u)==topologyGeneration
    &&cm12SRRLoad(at+${t.bank0Generation}u)==bank0Generation
    &&cm12SRRLoad(at+${t.bank1Generation}u)==bank1Generation
    &&cm12SRRLoad(at+${t.headResultGeneration}u)==headResultGeneration
    &&cm12SRRLoad(at+${t.coveredWords}u)==expectedWords
    &&cm12SRRLoad(at+${t.expectedWords}u)==expectedWords
    &&cm12SRRLoad(at+${t.dependencyGeneration}u)==dependencyGeneration
    &&cm12SRRLoad(at+${t.dependencyCertifiedGeneration}u)==dependencyGeneration
    &&(cm12SRRLoad(at+${t.flags}u)&required)==required;
}
fn cm12SRRClassifyCandidate(index:u32,topologyGeneration:u32,
  bank0Generation:u32,bank1Generation:u32,
  headResultGeneration:u32,expectedWords:u32,dependencyGeneration:u32){
  if(index>=cm12SRRLoad(${h.leafCount}u)){return;}
  let leafAt=cm12SRRLeafAt(index);
  if(cm12SRRLoad(leafAt+${leaf.generation}u)!=cm12SRRLoad(${h.candidateGeneration}u)){return;}
  let tile=cm12SRRLoad(leafAt+${leaf.tile}u);
  // Candidate topology/dependency state is speculative. Keep the tile dirty;
  // only a committed producer-authored candidate receipt may make it clean.
  _=topologyGeneration;_=bank0Generation;_=bank1Generation;
  _=headResultGeneration;_=expectedWords;_=dependencyGeneration;
  cm12SRRSetClassification(tile,false);
}
fn cm12SRRPrepareSeal()->bool{
  if(cm12SRRLoad(${h.acceptedGeneration}u)==0u
    &&cm12SRRLoad(${l.workTreeBaseWords}u)!=cm12SRRTileCapacity){
    cm12SRRFail(${SPARSE_CM12_SCALAR_RESULT_FAULT.bootstrapRequired}u,cm12SRRInvalid);return false;
  }
  if(cm12SRRLoad(${h.acceptedGeneration}u)==0u
    &&cm12SRRLoad(${h.bootstrapFrames}u)==0u){
    cm12SRRStore(${h.bootstrapFrames}u,1u);
  }
  cm12SRRStore(${h.workCount}u,cm12SRRLoad(${l.workTreeBaseWords}u));
  cm12SRRStore(${h.cleanCount}u,cm12SRRLoad(${l.cleanTreeBaseWords}u));
  cm12SRRStore(${h.scheduledWorkCount}u,cm12SRRLoad(${l.workTreeBaseWords}u));
  cm12SRRStore(${h.executedWorkCount}u,0u);
  return true;
}
fn cm12SRRWriteWorkRank(rank:u32){
  if(rank>=cm12SRRLoad(${h.scheduledWorkCount}u)){return;}
  cm12SRRStore(${l.workListBaseWords}u+rank,
    cm12SRRRankSelect(${l.workTreeBaseWords}u,rank));
}
fn cm12SRRFinishSeal(){
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.collecting}u){return;}
  cm12SRRStore(${h.phase}u,${SPARSE_CM12_SCALAR_RESULT_PHASE.sealed}u);
}
fn cm12SRRSeal(){
  if(!cm12SRRPrepareSeal()){return;}
  for(var rank=0u;rank<cm12SRRLoad(${h.scheduledWorkCount}u);rank+=1u){
    cm12SRRWriteWorkRank(rank);
  }
  cm12SRRFinishSeal();
}
fn cm12SRRRankSelect(base:u32,rankInput:u32)->u32{
  if(rankInput>=cm12SRRLoad(base)){return cm12SRRInvalid;}
  var rank=rankInput;var node=0u;
  loop{
    if(node>=cm12SRRTreeLeafCapacity-1u){break;}
    let left=2u*node+1u;let leftCount=cm12SRRLoad(base+left);
    if(rank<leftCount){node=left;}else{rank-=leftCount;node=left+1u;}
  }
  let tile=node-(cm12SRRTreeLeafCapacity-1u);
  return select(tile,cm12SRRInvalid,tile>=cm12SRRTileCapacity);
}
fn cm12SRRWorkTile(rank:u32)->u32{
  return select(cm12SRRLoad(${l.workListBaseWords}u+rank),cm12SRRInvalid,
    rank>=cm12SRRLoad(${h.scheduledWorkCount}u));
}
fn cm12SRRCleanTile(rank:u32)->u32{return cm12SRRRankSelect(${l.cleanTreeBaseWords}u,rank);}
fn cm12SRRPublishExactResult(tile:u32,topologyGeneration:u32,
  bank0Generation:u32,bank1Generation:u32,
  headResultGeneration:u32,coveredWords:u32,expectedWords:u32,
  sourceMismatchCount:u32,destinationMismatchCount:u32,
  dependencyGeneration:u32,dependencyCertifiedGeneration:u32,resultKind:u32)->bool{
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.sealed}u
    ||tile>=cm12SRRTileCapacity){return false;}
  let at=cm12SRRCandidateAt(tile);let generation=cm12SRRLoad(${h.candidateGeneration}u);
  let prior=atomicExchange(&${arena}[at+${c.generation}u],generation);
  if(prior==generation){
    cm12SRRReject(${SPARSE_CM12_SCALAR_RESULT_FAULT.duplicateExecution}u,tile);return false;
  }
  cm12SRRStore(at+${c.topologyGeneration}u,topologyGeneration);
  cm12SRRStore(at+${c.bank0Generation}u,bank0Generation);
  cm12SRRStore(at+${c.bank1Generation}u,bank1Generation);
  cm12SRRStore(at+${c.headResultGeneration}u,headResultGeneration);
  cm12SRRStore(at+${c.coveredWords}u,coveredWords);
  cm12SRRStore(at+${c.expectedWords}u,expectedWords);
  cm12SRRStore(at+${c.sourceMismatchCount}u,sourceMismatchCount);
  cm12SRRStore(at+${c.destinationMismatchCount}u,destinationMismatchCount);
  cm12SRRStore(at+${c.dependencyGeneration}u,dependencyGeneration);
  cm12SRRStore(at+${c.dependencyCertifiedGeneration}u,dependencyCertifiedGeneration);
  cm12SRRStore(at+${c.resultKind}u,resultKind);
  var flags=${f.producerExecuted}u;
  if(coveredWords==expectedWords){flags|=${f.fullWordCoverage}u;}
  if(sourceMismatchCount==0u&&destinationMismatchCount==0u){flags|=${f.dualBankExact}u;}
  if(dependencyGeneration==dependencyCertifiedGeneration){flags|=${f.dependencyExact}u;}
  cm12SRRStore(at+${c.flags}u,flags);
  // A stable 4^3 tile can own zero unique cells on a coarse accepted rung.
  // Zero/zero is exact empty work, not missing execution: topology events
  // invalidate the tile before a later rung can assign cells to it.
  let exact=coveredWords==expectedWords
    &&sourceMismatchCount==0u&&destinationMismatchCount==0u
    &&dependencyGeneration==dependencyCertifiedGeneration
    &&topologyGeneration==cm12SRRLoad(${h.topologyGeneration}u);
  atomicAdd(&${arena}[${h.executedWorkCount}u],1u);
  return exact;
}
fn cm12SRRPrepareCommit()->bool{
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.sealed}u){return false;}
  if(cm12SRRLoad(${h.executedWorkCount}u)!=cm12SRRLoad(${h.scheduledWorkCount}u)){
    cm12SRRReject(${SPARSE_CM12_SCALAR_RESULT_FAULT.missingExecution}u,cm12SRRInvalid);return false;
  }
  return true;
}
fn cm12SRRValidateReceiptRank(rank:u32){
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.sealed}u
    ||rank>=cm12SRRLoad(${h.scheduledWorkCount}u)){return;}
  let tile=cm12SRRLoad(${l.workListBaseWords}u+rank);
  if(tile>=cm12SRRTileCapacity
    ||cm12SRRLoad(cm12SRRCandidateAt(tile)+${c.generation}u)
      !=cm12SRRLoad(${h.candidateGeneration}u)){
    cm12SRRReject(${SPARSE_CM12_SCALAR_RESULT_FAULT.missingExecution}u,tile);
  }
}
fn cm12SRRPromoteReceiptRank(rank:u32){
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.sealed}u
    ||rank>=cm12SRRLoad(${h.scheduledWorkCount}u)){return;}
  let generation=cm12SRRLoad(${h.candidateGeneration}u);
  let tile=cm12SRRLoad(${l.workListBaseWords}u+rank);
  let receiptRequired=${f.producerExecuted | f.fullWordCoverage
    | f.dualBankExact | f.dependencyExact}u;
  let candidateAt=cm12SRRCandidateAt(tile);let at=cm12SRRTileAt(tile);
    cm12SRRStore(at+${t.resultGeneration}u,generation);
    cm12SRRStore(at+${t.topologyGeneration}u,cm12SRRLoad(candidateAt+${c.topologyGeneration}u));
    cm12SRRStore(at+${t.bank0Generation}u,cm12SRRLoad(candidateAt+${c.bank0Generation}u));
    cm12SRRStore(at+${t.bank1Generation}u,cm12SRRLoad(candidateAt+${c.bank1Generation}u));
    cm12SRRStore(at+${t.headResultGeneration}u,cm12SRRLoad(candidateAt+${c.headResultGeneration}u));
    cm12SRRStore(at+${t.coveredWords}u,cm12SRRLoad(candidateAt+${c.coveredWords}u));
    cm12SRRStore(at+${t.expectedWords}u,cm12SRRLoad(candidateAt+${c.expectedWords}u));
    cm12SRRStore(at+${t.sourceMismatchCount}u,cm12SRRLoad(candidateAt+${c.sourceMismatchCount}u));
    cm12SRRStore(at+${t.destinationMismatchCount}u,
      cm12SRRLoad(candidateAt+${c.destinationMismatchCount}u));
    cm12SRRStore(at+${t.dependencyGeneration}u,cm12SRRLoad(candidateAt+${c.dependencyGeneration}u));
    cm12SRRStore(at+${t.dependencyCertifiedGeneration}u,
      cm12SRRLoad(candidateAt+${c.dependencyCertifiedGeneration}u));
    cm12SRRStore(at+${t.resultKind}u,cm12SRRLoad(candidateAt+${c.resultKind}u));
    cm12SRRStore(at+${t.receiptGeneration}u,generation);
    let flags=cm12SRRLoad(candidateAt+${c.flags}u);cm12SRRStore(at+${t.flags}u,flags);
    let exact=(flags&receiptRequired)==receiptRequired
      &&cm12SRRLoad(candidateAt+${c.coveredWords}u)
        ==cm12SRRLoad(candidateAt+${c.expectedWords}u)
      &&cm12SRRLoad(candidateAt+${c.topologyGeneration}u)==cm12SRRLoad(${h.topologyGeneration}u);
  cm12SRRSetClassification(tile,exact);
}
fn cm12SRRFinalizeCommit()->bool{
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.sealed}u){return false;}
  cm12SRRStore(${h.acceptedGeneration}u,cm12SRRLoad(${h.candidateGeneration}u));
  cm12SRRStore(${h.workCount}u,cm12SRRLoad(${l.workTreeBaseWords}u));
  cm12SRRStore(${h.cleanCount}u,cm12SRRLoad(${l.cleanTreeBaseWords}u));
  atomicAdd(&${arena}[${h.committedFrames}u],1u);
  cm12SRRStore(${h.phase}u,${SPARSE_CM12_SCALAR_RESULT_PHASE.accepted}u);
  return true;
}
fn cm12SRRCommit()->bool{
  if(!cm12SRRPrepareCommit()){return false;}
  let scheduled=cm12SRRLoad(${h.scheduledWorkCount}u);
  for(var rank=0u;rank<scheduled;rank+=1u){cm12SRRValidateReceiptRank(rank);}
  if(cm12SRRLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_RESULT_PHASE.sealed}u){return false;}
  for(var rank=0u;rank<scheduled;rank+=1u){cm12SRRPromoteReceiptRank(rank);}
  return cm12SRRFinalizeCommit();
}
`;
}
