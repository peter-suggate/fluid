import {
  SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT,
  SPARSE_CM12_SCALAR_AUTHORITY_CAUSE,
  SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT,
  SPARSE_CM12_SCALAR_AUTHORITY_FAULT,
  SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT,
  SPARSE_CM12_SCALAR_AUTHORITY_HEADER,
  SPARSE_CM12_SCALAR_AUTHORITY_HEADER_WORDS,
  SPARSE_CM12_SCALAR_AUTHORITY_INVALID,
  SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES,
  SPARSE_CM12_SCALAR_AUTHORITY_MAGIC,
  SPARSE_CM12_SCALAR_AUTHORITY_PHASE,
  SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT,
  SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER,
  SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS,
  SPARSE_CM12_SCALAR_AUTHORITY_TILE,
  SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG,
  SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH,
  SPARSE_CM12_SCALAR_AUTHORITY_VERSION,
  SPARSE_CM12_SCALAR_DEPENDENCY,
  SPARSE_CM12_SCALAR_FPL_STAGE,
  SPARSE_CM12_SCALAR_REQUIRED_DEPENDENCIES,
  type SparseCM12ScalarWorkAuthorityLayout,
} from "./sparse-cm12-scalar-work-authority";

const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError("invalid WGSL identifier");
  return value;
};

/**
 * SAW1 storage helpers. They can author only the authority arena supplied by
 * `arenaName`; no density, gamma, surface, or other physics binding exists.
 */
export function createSparseCM12ScalarWorkAuthorityWGSL(options: {
  readonly layout: SparseCM12ScalarWorkAuthorityLayout;
  readonly arenaName?: string;
}): string {
  const l = options.layout; const arena = identifier(options.arenaName ?? "scalarAuthority");
  const h = SPARSE_CM12_SCALAR_AUTHORITY_HEADER;
  const sh = SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER;
  const stageTreeCases = l.treeLevelBaseWords.map((levels, stage) =>
    levels.map((base, level) => `if(stage==${stage}u&&level==${level}u){return ${base}u;}`)
      .join("\n")).join("\n");
  const treeCounts = l.treeLevelCounts.map((count, level) =>
    `if(level==${level}u){return ${count}u;}`).join("\n");
  const required = [0, 1, 2].map((stage) => {
    // The immutable required masks are initialized by TS and also checked in
    // the header path; spelling them here keeps malformed mutable masks from
    // broadening a clean certificate.
    return `if(stage==${stage}u){return ${SPARSE_CM12_SCALAR_REQUIRED_DEPENDENCIES[stage]}u;}`;
  }).join("\n");
  const fplStages = SPARSE_CM12_SCALAR_FPL_STAGE.map((value, stage) =>
    `if(stage==${stage}u){return ${value}u;}`).join("\n");
  return /* wgsl */ `
const cm12SAWInvalid:u32=0x${SPARSE_CM12_SCALAR_AUTHORITY_INVALID.toString(16)}u;
const cm12SAWTileCapacity:u32=${l.tileCapacity}u;
const cm12SAWTreeLevels:u32=${l.treeLevelCounts.length}u;
fn cm12SAWLoad(word:u32)->u32{return atomicLoad(&${arena}[word]);}
fn cm12SAWStore(word:u32,value:u32){atomicStore(&${arena}[word],value);}
fn cm12SAWStageBase(stage:u32)->u32{
  return ${l.stageHeadersBaseWords}u+${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS}u*stage;
}
fn cm12SAWDependencyAt(stage:u32,tile:u32,dependency:u32)->u32{
  return ${l.dependencyBaseWords}u+2u*((stage*cm12SAWTileCapacity+tile)
    *${SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT}u+dependency);
}
fn cm12SAWBankAt(stage:u32,tile:u32,bank:u32)->u32{
  return ${l.bankReceiptBaseWords}u+4u*((stage*cm12SAWTileCapacity+tile)*2u+bank);
}
fn cm12SAWFPLAt(stage:u32,tile:u32)->u32{
  return ${l.fplReceiptBaseWords}u+4u*(stage*cm12SAWTileCapacity+tile);
}
fn cm12SAWTileAt(stage:u32,tile:u32)->u32{
  return ${l.tileBaseWords}u+4u*(stage*cm12SAWTileCapacity+tile);
}
fn cm12SAWTreeBase(stage:u32,level:u32)->u32{
  ${stageTreeCases}
  return cm12SAWInvalid;
}
fn cm12SAWTreeCount(level:u32)->u32{${treeCounts}return 0u;}
fn cm12SAWRequired(stage:u32)->u32{${required}return 0u;}
fn cm12SAWFPLStage(stage:u32)->u32{${fplStages}return cm12SAWInvalid;}
fn cm12SAWHeaderValid()->bool{
  if(arrayLength(&${arena})!=${l.totalWords}u){return false;}
  return cm12SAWLoad(${h.magic}u)==0x${SPARSE_CM12_SCALAR_AUTHORITY_MAGIC.toString(16)}u
    &&cm12SAWLoad(${h.version}u)==${SPARSE_CM12_SCALAR_AUTHORITY_VERSION}u
    &&cm12SAWLoad(${h.headerWords}u)==${SPARSE_CM12_SCALAR_AUTHORITY_HEADER_WORDS}u
    &&cm12SAWLoad(${h.stageHeaderWords}u)==${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_HEADER_WORDS}u
    &&cm12SAWLoad(${h.stageCount}u)==${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u
    &&cm12SAWLoad(${h.dependencyCount}u)==${SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT}u
    &&cm12SAWLoad(${h.brickFineResolution}u)==16u
    &&cm12SAWLoad(${h.presentationPageResolution}u)==16u
    &&cm12SAWLoad(${h.tileCapacity}u)==cm12SAWTileCapacity
    &&cm12SAWLoad(${h.dependencyBase}u)==${l.dependencyBaseWords}u
    &&cm12SAWLoad(${h.bankReceiptBase}u)==${l.bankReceiptBaseWords}u
    &&cm12SAWLoad(${h.fplReceiptBase}u)==${l.fplReceiptBaseWords}u
    &&cm12SAWLoad(${h.tileBase}u)==${l.tileBaseWords}u
    &&cm12SAWLoad(${h.workListBase}u)==${l.workListBaseWords}u
    &&cm12SAWLoad(${h.closureOffsetBase}u)==${l.closureOffsetBaseWords}u
    &&cm12SAWLoad(${h.closureIdBase}u)==${l.closureIdBaseWords}u
    &&cm12SAWLoad(${h.totalWords}u)==${l.totalWords}u
    &&cm12SAWLoad(${h.leafTiles}u)==${SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES}u
    &&cm12SAWLoad(${h.treeBranch}u)==${SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH}u;
}
fn cm12SAWGlobalFail(code:u32,stage:u32,tile:u32){
  let claimed=atomicCompareExchangeWeak(&${arena}[${h.fault}u],0u,code);
  if(claimed.exchanged){cm12SAWStore(${h.firstFaultStage}u,stage);cm12SAWStore(${h.firstFaultTile}u,tile);}
  cm12SAWStore(${h.phase}u,${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.fault}u);
  for(var s=0u;s<${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u;s+=1u){
    let base=cm12SAWStageBase(s);cm12SAWStore(base+${sh.workIndirectX}u,0u);
    cm12SAWStore(base+${sh.cleanIndirectX}u,0u);
  }
}
fn cm12SAWBegin(frameGeneration:u32,topologyGeneration:u32,sourceParity:u32)->bool{
  if(!cm12SAWHeaderValid()){cm12SAWGlobalFail(${SPARSE_CM12_SCALAR_AUTHORITY_FAULT.invalidHeader}u,
    cm12SAWInvalid,cm12SAWInvalid);return false;}
  if(cm12SAWLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.accepted}u
    ||frameGeneration==0u||topologyGeneration==0u||sourceParity>1u){
    cm12SAWGlobalFail(${SPARSE_CM12_SCALAR_AUTHORITY_FAULT.invalidPhase}u,
      cm12SAWInvalid,cm12SAWInvalid);return false;
  }
  cm12SAWStore(${h.candidateGeneration}u,frameGeneration);
  cm12SAWStore(${h.topologyGeneration}u,topologyGeneration);
  cm12SAWStore(${h.sourceParity}u,sourceParity);cm12SAWStore(${h.fault}u,0u);
  cm12SAWStore(${h.firstFaultStage}u,cm12SAWInvalid);cm12SAWStore(${h.firstFaultTile}u,cm12SAWInvalid);
  cm12SAWStore(${h.phase}u,${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u);return true;
}
fn cm12SAWConfigureStage(stage:u32,headGeneration:u32,headWords:u32,
  fplGeneration:u32,fplEpoch:u32)->bool{
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u
    ||cm12SAWLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u
    ||headGeneration==0u||headWords==0u
    ||fplGeneration!=cm12SAWLoad(${h.candidateGeneration}u)||fplEpoch==0u){
    cm12SAWGlobalFail(${SPARSE_CM12_SCALAR_AUTHORITY_FAULT.invalidStage}u,stage,cm12SAWInvalid);return false;
  }
  let base=cm12SAWStageBase(stage);
  cm12SAWStore(base+${sh.phase}u,${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.collecting}u);
  cm12SAWStore(base+${sh.frameGeneration}u,cm12SAWLoad(${h.candidateGeneration}u));
  cm12SAWStore(base+${sh.topologyGeneration}u,cm12SAWLoad(${h.topologyGeneration}u));
  cm12SAWStore(base+${sh.sourceParity}u,cm12SAWLoad(${h.sourceParity}u));
  cm12SAWStore(base+${sh.requiredDependencyMask}u,cm12SAWRequired(stage));
  cm12SAWStore(base+${sh.headResultGeneration}u,headGeneration);
  cm12SAWStore(base+${sh.headWordsPerTile}u,headWords);
  cm12SAWStore(base+${sh.fplGeneration}u,fplGeneration);cm12SAWStore(base+${sh.fplPacketEpoch}u,fplEpoch);
  for(var word=${sh.workCount}u;word<=${sh.treeRootCount}u;word+=1u){
    cm12SAWStore(base+word,select(0u,1u,word==${sh.workIndirectY}u||word==${sh.workIndirectZ}u
      ||word==${sh.cleanIndirectY}u||word==${sh.cleanIndirectZ}u));
  }
  cm12SAWStore(base+${sh.firstFaultTile}u,cm12SAWInvalid);return true;
}
fn cm12SAWPublishDependency(stage:u32,tile:u32,dependency:u32,
  producerGeneration:u32,headCertifiedGeneration:u32)->bool{
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u||tile>=cm12SAWTileCapacity
    ||dependency>=${SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT}u
    ||producerGeneration==0u||headCertifiedGeneration==0u){return false;}
  let at=cm12SAWDependencyAt(stage,tile,dependency);
  cm12SAWStore(at,producerGeneration);cm12SAWStore(at+1u,headCertifiedGeneration);return true;
}
fn cm12SAWPublishBank(stage:u32,tile:u32,bank:u32,resultGeneration:u32,
  topologyGeneration:u32,coveredWords:u32,mismatchCount:u32)->bool{
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u||tile>=cm12SAWTileCapacity||bank>1u){return false;}
  let at=cm12SAWBankAt(stage,tile,bank);cm12SAWStore(at,resultGeneration);
  cm12SAWStore(at+1u,topologyGeneration);cm12SAWStore(at+2u,coveredWords);
  cm12SAWStore(at+3u,mismatchCount);return true;
}
fn cm12SAWPublishFPL(stage:u32,tile:u32,frameGeneration:u32,topologyGeneration:u32,
  packetEpoch:u32,packet:u32,coverageComplete:bool)->bool{
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u||tile>=cm12SAWTileCapacity
    ||packet>=6u){return false;}
  let at=cm12SAWFPLAt(stage,tile);cm12SAWStore(at,frameGeneration);
  cm12SAWStore(at+1u,topologyGeneration);cm12SAWStore(at+2u,packetEpoch);
  cm12SAWStore(at+3u,packet|(cm12SAWFPLStage(stage)<<16u)|select(0u,1u<<24u,coverageComplete));return true;
}
fn cm12SAWExactBank(stage:u32,tile:u32,bank:u32)->bool{
  let base=cm12SAWStageBase(stage);let at=cm12SAWBankAt(stage,tile,bank);
  let topology=cm12SAWDependencyAt(stage,tile,${SPARSE_CM12_SCALAR_DEPENDENCY.topology}u);
  return cm12SAWLoad(at+${SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.resultGeneration}u)
      ==cm12SAWLoad(base+${sh.headResultGeneration}u)
    &&cm12SAWLoad(at+${SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.topologyGeneration}u)==cm12SAWLoad(topology)
    &&cm12SAWLoad(at+${SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.coveredWords}u)
      ==cm12SAWLoad(base+${sh.headWordsPerTile}u)
    &&cm12SAWLoad(at+${SPARSE_CM12_SCALAR_AUTHORITY_BANK_RECEIPT.mismatchCount}u)==0u;
}
fn cm12SAWFPLValid(stage:u32,tile:u32)->bool{
  let base=cm12SAWStageBase(stage);let at=cm12SAWFPLAt(stage,tile);let packed=cm12SAWLoad(at+3u);
  return cm12SAWLoad(at+${SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT.frameGeneration}u)
      ==cm12SAWLoad(base+${sh.fplGeneration}u)
    &&cm12SAWLoad(at+${SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT.topologyGeneration}u)
      ==cm12SAWLoad(base+${sh.topologyGeneration}u)
    &&cm12SAWLoad(at+${SPARSE_CM12_SCALAR_AUTHORITY_FPL_RECEIPT.packetEpoch}u)
      ==cm12SAWLoad(base+${sh.fplPacketEpoch}u)
    &&((packed>>16u)&255u)==cm12SAWFPLStage(stage)&&(packed&(1u<<24u))!=0u;
}
fn cm12SAWDependenciesValid(stage:u32,tile:u32)->bool{
  let required=cm12SAWRequired(stage);
  let offsets=${l.closureOffsetBaseWords}u+stage*(cm12SAWTileCapacity+1u);
  let begin=cm12SAWLoad(offsets+tile);let end=cm12SAWLoad(offsets+tile+1u);
  for(var cursor=begin;cursor<end;cursor+=1u){
    let dependencyTile=cm12SAWLoad(${l.closureIdBaseWords}u+cursor);
    if(dependencyTile>=cm12SAWTileCapacity){return false;}
    for(var dependency=0u;dependency<${SPARSE_CM12_SCALAR_AUTHORITY_DEPENDENCY_COUNT}u;dependency+=1u){
      if((required&(1u<<dependency))==0u){continue;}
      let at=cm12SAWDependencyAt(stage,dependencyTile,dependency);
      let current=cm12SAWLoad(at);
      if(current==0u||current!=cm12SAWLoad(at+1u)){return false;}
    }
  }
  return true;
}
fn cm12SAWClassify(stage:u32,tile:u32)->bool{
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u||tile>=cm12SAWTileCapacity){return true;}
  var causes=0u;
  if(!cm12SAWExactBank(stage,tile,0u)){causes|=${SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.bank0NotExact}u;}
  if(!cm12SAWExactBank(stage,tile,1u)){causes|=${SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.bank1NotExact}u;}
  if(!cm12SAWDependenciesValid(stage,tile)){causes|=${SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.dependency}u;}
  let hasFPLReceipt=cm12SAWFPLValid(stage,tile);
  if(!hasFPLReceipt){causes|=${SPARSE_CM12_SCALAR_AUTHORITY_CAUSE.fplReceipt}u;}
  let clean=causes==0u;let at=cm12SAWTileAt(stage,tile);
  cm12SAWStore(at+${SPARSE_CM12_SCALAR_AUTHORITY_TILE.generation}u,cm12SAWLoad(${h.candidateGeneration}u));
  cm12SAWStore(at+${SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags}u,
    ${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.classified}u
      |select(0u,${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.fplReceipt}u,hasFPLReceipt)
      |select(${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work}u,
        ${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.exactCleanSkip}u,clean));
  cm12SAWStore(at+${SPARSE_CM12_SCALAR_AUTHORITY_TILE.causeMask}u,causes);
  cm12SAWStore(at+${SPARSE_CM12_SCALAR_AUTHORITY_TILE.fplPacket}u,cm12SAWLoad(cm12SAWFPLAt(stage,tile)+3u)&65535u);
  return !clean;
}
fn cm12SAWCountLeaf(stage:u32,leaf:u32){
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u||leaf>=cm12SAWTreeCount(0u)){return;}
  var count=0u;let begin=leaf*${SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES}u;
  let end=min(cm12SAWTileCapacity,begin+${SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES}u);
  for(var tile=begin;tile<end;tile+=1u){
    if((cm12SAWLoad(cm12SAWTileAt(stage,tile)+${SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags}u)
      &${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work}u)!=0u){count+=1u;}
  }
  cm12SAWStore(cm12SAWTreeBase(stage,0u)+leaf,count);
}
fn cm12SAWReduceTree(stage:u32,level:u32,node:u32){
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u||level==0u||level>=cm12SAWTreeLevels
    ||node>=cm12SAWTreeCount(level)){return;}
  let begin=node*${SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH}u;
  let end=min(cm12SAWTreeCount(level-1u),begin+${SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH}u);
  var sum=0u;for(var child=begin;child<end;child+=1u){sum+=cm12SAWLoad(cm12SAWTreeBase(stage,level-1u)+child);}
  cm12SAWStore(cm12SAWTreeBase(stage,level)+node,sum);
}
fn cm12SAWRankSelect(stage:u32,rankInput:u32)->u32{
  var rank=rankInput;var node=0u;var level=cm12SAWTreeLevels-1u;
  loop{
    if(level==0u){break;}
    let begin=node*${SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH}u;
    let end=min(cm12SAWTreeCount(level-1u),begin+${SPARSE_CM12_SCALAR_AUTHORITY_TREE_BRANCH}u);
    for(var child=begin;child<end;child+=1u){let count=cm12SAWLoad(cm12SAWTreeBase(stage,level-1u)+child);
      if(rank<count){node=child;break;}rank-=count;}
    level-=1u;
  }
  let begin=node*${SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES}u;
  let end=min(cm12SAWTileCapacity,begin+${SPARSE_CM12_SCALAR_AUTHORITY_LEAF_TILES}u);
  for(var tile=begin;tile<end;tile+=1u){
    if((cm12SAWLoad(cm12SAWTileAt(stage,tile)+${SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags}u)
      &${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work}u)==0u){continue;}
    if(rank==0u){return tile;}rank-=1u;
  }
  return cm12SAWInvalid;
}
fn cm12SAWSealStage(stage:u32)->bool{
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u){return false;}
  let base=cm12SAWStageBase(stage);let root=cm12SAWLoad(cm12SAWTreeBase(stage,cm12SAWTreeLevels-1u));
  if(root>cm12SAWTileCapacity){cm12SAWGlobalFail(${SPARSE_CM12_SCALAR_AUTHORITY_FAULT.treeMismatch}u,
    stage,cm12SAWInvalid);return false;}
  cm12SAWStore(base+${sh.workCount}u,root);cm12SAWStore(base+${sh.cleanCount}u,cm12SAWTileCapacity-root);
  cm12SAWStore(base+${sh.classifiedCount}u,cm12SAWTileCapacity);cm12SAWStore(base+${sh.treeRootCount}u,root);
  cm12SAWStore(base+${sh.workIndirectX}u,root);cm12SAWStore(base+${sh.cleanIndirectX}u,cm12SAWTileCapacity-root);
  cm12SAWStore(base+${sh.phase}u,${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.classified}u);return true;
}
fn cm12SAWScatterWork(stage:u32,rank:u32)->bool{
  let tile=cm12SAWRankSelect(stage,rank);if(tile==cm12SAWInvalid){return false;}
  cm12SAWStore(${l.workListBaseWords}u+stage*cm12SAWTileCapacity+rank,tile);return true;
}
fn cm12SAWSeal()->bool{
  for(var stage=0u;stage<${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u;stage+=1u){
    if(cm12SAWLoad(cm12SAWStageBase(stage)+${sh.phase}u)
      !=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.classified}u){return false;}
  }
  cm12SAWStore(${h.phase}u,${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.sealed}u);return true;
}
fn cm12SAWPublishExecution(stage:u32,tile:u32)->bool{
  if(stage>=${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u||tile>=cm12SAWTileCapacity
    ||cm12SAWLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.sealed}u){return false;}
  let at=cm12SAWTileAt(stage,tile)+${SPARSE_CM12_SCALAR_AUTHORITY_TILE.flags}u;
  let old=atomicOr(&${arena}[at],${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.executed}u);
  if((old&${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.work}u)==0u){return false;}
  if((old&${SPARSE_CM12_SCALAR_AUTHORITY_TILE_FLAG.executed}u)==0u){
    atomicAdd(&${arena}[cm12SAWStageBase(stage)+${sh.receiptCount}u],1u);
  }
  return true;
}
fn cm12SAWCommit()->bool{
  if(cm12SAWLoad(${h.phase}u)!=${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.sealed}u){return false;}
  for(var stage=0u;stage<${SPARSE_CM12_SCALAR_AUTHORITY_STAGE_COUNT}u;stage+=1u){let base=cm12SAWStageBase(stage);
    if(cm12SAWLoad(base+${sh.receiptCount}u)!=cm12SAWLoad(base+${sh.workCount}u)){
      cm12SAWGlobalFail(${SPARSE_CM12_SCALAR_AUTHORITY_FAULT.missingExecution}u,stage,cm12SAWInvalid);return false;}
  }
  cm12SAWStore(${h.acceptedGeneration}u,cm12SAWLoad(${h.candidateGeneration}u));
  cm12SAWStore(${h.phase}u,${SPARSE_CM12_SCALAR_AUTHORITY_PHASE.accepted}u);return true;
}
`;
}
