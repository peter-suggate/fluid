import {
  SPARSE_CM12_COMPILED_TOPOLOGY_FAULT as F,
  SPARSE_CM12_COMPILED_TOPOLOGY_HEADER as H,
  SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS,
  SPARSE_CM12_COMPILED_TOPOLOGY_MAGIC,
  SPARSE_CM12_COMPILED_TOPOLOGY_PHASE as P,
  SPARSE_CM12_COMPILED_TOPOLOGY_VERSION,
  SPARSE_CM12_COMPILED_TOPOLOGY_VIEW as V,
  type SparseCM12CompiledTopologyLayout,
} from "./sparse-cm12-compiled-topology";

export interface SparseCM12CompiledTopologySourceWGSL {
  readonly generation?: string;
  readonly acceptedSlot?: string;
  readonly lifecycleAccepted?: string;
  readonly cellCount?: string;
  readonly rowCount?: string;
  readonly cellInvocation?: (ordinal: string) => string;
  readonly rowInvocation?: (ordinal: string) => string;
  /** Static geometry certificate only; moving-body openness is never baked. */
  readonly rowStaticOpen?: (stableRow: string) => string;
}

export interface SparseCM12CompiledTopologyExternalViewWGSL {
  readonly generation: string;
  readonly fault?: string;
}

export interface SparseCM12CompiledTopologyTransportViewWGSL
  extends SparseCM12CompiledTopologyExternalViewWGSL {
  readonly faceCount: string;
  readonly cellFaceEntryCount: string;
}

export interface SparseCM12CompiledTopologyWGSLOptions {
  readonly layout: SparseCM12CompiledTopologyLayout;
  readonly source?: SparseCM12CompiledTopologySourceWGSL;
  /**
   * Publish a rejected full build into the enclosing runtime's sticky failure
   * authority. The callback receives WGSL expressions for the CNX fault mask
   * and first offending stable id. Standalone shader oracles may omit it.
   */
  readonly publishFailure?: (fault: string, owner: string) => string;
  /** Existing geometric-volume arrays used as CNX's physical-face view. */
  readonly transportStorage?: SparseCM12CompiledTopologyTransportLayout;
  readonly transport?: SparseCM12CompiledTopologyTransportViewWGSL;
  readonly velocityExtension?: SparseCM12CompiledTopologyExternalViewWGSL;
  readonly projection?: SparseCM12CompiledTopologyExternalViewWGSL;
  readonly presentation?: SparseCM12CompiledTopologyExternalViewWGSL;
}

const boolView = (
  bit: number,
  view: SparseCM12CompiledTopologyExternalViewWGSL | undefined,
  generationTarget = "sourceGeneration",
): string => {
  const name = bit === V.velocityExtension ? "velocityExtensionGeneration"
    : bit === V.projection ? "projectionGeneration" : "presentationGeneration";
  return view
  ? `let ${name}=${view.generation};
  cnxStore(${bit === V.velocityExtension ? H.velocityExtensionGeneration
    : bit === V.projection ? H.projectionGeneration : H.presentationGeneration}u,${name});
  if(${name}==${generationTarget}${view.fault ? `&&(${view.fault})==0u` : ""}){ready|=${bit}u;}`
  : "";
};

/**
 * Emits the only writer and the hot read API for the compiled connectivity
 * image.  Source terms/incidences are read once by the build entry points;
 * consumers traverse the copied, generation-fenced planes below.
 */
export function createSparseCM12CompiledTopologyWGSL(
  options: SparseCM12CompiledTopologyWGSLOptions,
): string {
  const { layout } = options;
  const source = options.source ?? {};
  const generation = source.generation ??
    "atomicLoad(&topologyArena[topologyWorklistBase()])";
  const slot = source.acceptedSlot ?? "acceptedTopologySlot()";
  const lifecycle = source.lifecycleAccepted ?? "sparseCM12TopologyLifecycleAccepted()";
  const cellCount = source.cellCount ?? "acceptedTemplateCellCount()";
  const rowCount = source.rowCount ?? "acceptedTemplateRowCount()";
  const cellAt = source.cellInvocation ?? ((ordinal: string) =>
    `acceptedTemplateCellInvocation(${ordinal})`);
  const rowAt = source.rowInvocation ?? ((ordinal: string) =>
    `acceptedTemplateRowInvocation(${ordinal})`);
  const staticOpen = source.rowStaticOpen;
  const publishFailure = options.publishFailure
    ? options.publishFailure("cnxLoad(CNX_H_FAULT)", "cnxLoad(CNX_H_FIRST_FAULT)")
    : "";
  const transport = options.transport;
  const transportSeal = transport ? `let transportGeneration=${transport.generation};
  let physicalFaces=${transport.faceCount};let physicalEntries=${transport.cellFaceEntryCount};
  cnxStore(${H.transportGeneration}u,transportGeneration);
  cnxStore(${H.physicalFaceCount}u,physicalFaces);
  cnxStore(${H.physicalFaceEntryCount}u,physicalEntries);
  if(physicalFaces>CNX_FACE_CAPACITY||physicalEntries>2u*CNX_FACE_CAPACITY){
    cnxFault(${F.transportCapacity}u,physicalFaces);
  }
  if(transportGeneration==sourceGeneration${transport.fault ? `&&(${transport.fault})==0u` : ""}){
    ready|=${V.transport}u;
  }` : "";
  const compiled = /* wgsl */ `
const CNX_MAGIC:u32=${SPARSE_CM12_COMPILED_TOPOLOGY_MAGIC}u;
const CNX_VERSION:u32=${SPARSE_CM12_COMPILED_TOPOLOGY_VERSION}u;
const CNX_HEADER_WORDS:u32=${SPARSE_CM12_COMPILED_TOPOLOGY_HEADER_WORDS}u;
const CNX_HEADER:u32=${layout.headerBaseWords}u;
const CNX_CELL_IDS:u32=${layout.acceptedCellIdsBaseWords}u;
const CNX_CELL_ORDINALS:u32=${layout.cellOrdinalByStableBaseWords}u;
const CNX_CELL_RANGES:u32=${layout.acceptedCellRangesBaseWords}u;
const CNX_ROW_RECORDS:u32=${layout.acceptedRowRecordsBaseWords}u;
const CNX_ROW_ORDINALS:u32=${layout.rowOrdinalByStableBaseWords}u;
const CNX_TERMS:u32=${layout.orderedTermsBaseWords}u;
const CNX_INCIDENCES:u32=${layout.cellIncidencesBaseWords}u;
const CNX_CELL_CAPACITY:u32=${layout.cellCapacity}u;
const CNX_ROW_CAPACITY:u32=${layout.rowCapacity}u;
const CNX_TERM_CAPACITY:u32=${layout.termCapacity}u;
const CNX_INCIDENCE_CAPACITY:u32=${layout.incidenceCapacity}u;
const CNX_FACE_CAPACITY:u32=${layout.physicalFaceCapacity}u;
const CNX_REQUIRED_VIEWS:u32=${layout.requiredViews}u;
const CNX_H_FAULT:u32=${H.fault}u;
const CNX_H_FIRST_FAULT:u32=${H.firstFaultId}u;

fn cnxLoad(word:u32)->u32{return atomicLoad(&topologyArena[CNX_HEADER+word]);}
fn cnxStore(word:u32,value:u32){atomicStore(&topologyArena[CNX_HEADER+word],value);}
fn cnxArenaLoad(word:u32)->u32{return atomicLoad(&topologyArena[word]);}
fn cnxArenaStore(word:u32,value:u32){atomicStore(&topologyArena[word],value);}
// WebGPU limits one dispatch dimension to 65,535 workgroups. Host dispatches
// a full-width x slab before adding y slabs, so this remains a dense ordinal
// for B16 capacity clears as well as ordinary one-dimensional worklists.
fn cnxLinearInvocation(gid:vec3u)->u32{return gid.x+64u*65535u*gid.y;}
fn cnxMix(hash:u32,value:u32)->u32{let mixed=(hash^value)*0x01000193u;return mixed^(mixed>>16u);}
fn cnxCertificate()->u32{
  var hash=0x811c9dc5u;
  hash=cnxMix(hash,CNX_MAGIC);hash=cnxMix(hash,CNX_VERSION);
  hash=cnxMix(hash,cnxLoad(${H.sourceTopologyGeneration}u));
  hash=cnxMix(hash,cnxLoad(${H.sourceAcceptedSlot}u));
  hash=cnxMix(hash,cnxLoad(${H.acceptedCellCount}u));
  hash=cnxMix(hash,cnxLoad(${H.acceptedRowWorklistCount}u));
  hash=cnxMix(hash,cnxLoad(${H.acceptedRowCount}u));
  hash=cnxMix(hash,cnxLoad(${H.orderedTermCount}u));
  hash=cnxMix(hash,cnxLoad(${H.cellIncidenceCount}u));
  hash=cnxMix(hash,cnxLoad(${H.readyViews}u));
  hash=cnxMix(hash,cnxLoad(${H.physicalFaceCount}u));
  hash=cnxMix(hash,cnxLoad(${H.physicalFaceEntryCount}u));
  hash=cnxMix(hash,CNX_CELL_CAPACITY);hash=cnxMix(hash,CNX_ROW_CAPACITY);
  hash=cnxMix(hash,CNX_TERM_CAPACITY);hash=cnxMix(hash,CNX_INCIDENCE_CAPACITY);
  return cnxMix(hash,CNX_FACE_CAPACITY);
}
fn cnxSourceGeneration()->u32{return ${generation};}
fn cnxSourceAcceptedSlot()->u32{return ${slot};}
fn cnxSourceLifecycleAccepted()->bool{return ${lifecycle};}
fn cnxAccepted()->bool{
  if(cnxLoad(${H.magic}u)!=CNX_MAGIC||cnxLoad(${H.version}u)!=CNX_VERSION
    ||cnxLoad(${H.phase}u)!=${P.accepted}u||cnxLoad(${H.fault}u)!=0u
    ||cnxLoad(${H.generation}u)!=cnxSourceGeneration()
    ||cnxLoad(${H.sourceTopologyGeneration}u)!=cnxSourceGeneration()
    ||cnxLoad(${H.sourceAcceptedSlot}u)!=cnxSourceAcceptedSlot()
    ||!cnxSourceLifecycleAccepted()
    ||(cnxLoad(${H.readyViews}u)&CNX_REQUIRED_VIEWS)!=CNX_REQUIRED_VIEWS){return false;}
  return true;
}
fn cnxTransportViewValidForAcceptedTopology()->bool{return cnxAccepted()
  &&(cnxLoad(${H.readyViews}u)&${V.transport}u)!=0u
  &&cnxLoad(${H.transportGeneration}u)==cnxSourceGeneration();}
fn cnxBuilding()->bool{return cnxLoad(${H.phase}u)==${P.building}u
  &&cnxLoad(${H.rebuildRequired}u)!=0u
  &&cnxLoad(${H.sourceTopologyGeneration}u)==cnxSourceGeneration()
  &&cnxLoad(${H.sourceAcceptedSlot}u)==cnxSourceAcceptedSlot();}
fn cnxBuildRequired()->bool{return cnxLoad(${H.rebuildRequired}u)!=0u;}
fn cnxAcceptedCellCount()->u32{return select(0u,cnxLoad(${H.acceptedCellCount}u),cnxAccepted());}
fn cnxAcceptedRowWorklistCount()->u32{return select(0u,cnxLoad(${H.acceptedRowWorklistCount}u),cnxAccepted());}
fn cnxAcceptedRowCount()->u32{return select(0u,cnxLoad(${H.acceptedRowCount}u),cnxAccepted());}
fn cnxPhysicalFaceCount()->u32{return select(0u,cnxLoad(${H.physicalFaceCount}u),cnxAccepted());}
fn cnxPhysicalFaceCountBuildingUnchecked()->u32{return cnxLoad(${H.physicalFaceCount}u);}
fn cnxPhysicalFaceEntryCountBuildingUnchecked()->u32{return cnxLoad(${H.physicalFaceEntryCount}u);}
fn cnxAllocatePhysicalFaces(count:u32,owner:u32)->u32{
  if(!cnxBuilding()){return INVALID;}
  let first=atomicAdd(&topologyArena[CNX_HEADER+${H.physicalFaceCount}u],count);
  if(first>CNX_FACE_CAPACITY||count>CNX_FACE_CAPACITY-min(first,CNX_FACE_CAPACITY)){
    cnxFault(${F.transportCapacity}u,owner);return INVALID;
  }
  return first;
}
fn cnxAllocatePhysicalFaceEntries(count:u32,owner:u32)->u32{
  if(!cnxBuilding()){return INVALID;}let capacity=2u*CNX_FACE_CAPACITY;
  let first=atomicAdd(&topologyArena[CNX_HEADER+${H.physicalFaceEntryCount}u],count);
  if(first>capacity||count>capacity-min(first,capacity)){
    cnxFault(${F.transportCapacity}u,owner);return INVALID;
  }
  return first;
}
fn cnxPhysicalBuildFailed()->bool{return !cnxBuilding()||cnxLoad(${H.fault}u)!=0u;}
fn cnxPhysicalBuildMalformed(owner:u32){cnxFault(${F.transportMalformed}u,owner);}
fn cnxPublishTransportView(){
  if(cnxPhysicalBuildFailed()){return;}
  cnxStore(${H.transportGeneration}u,cnxSourceGeneration());
  atomicOr(&topologyArena[CNX_HEADER+${H.readyViews}u],${V.transport}u);
}
// Hot accessors below have an explicit precondition: the entry point checked
// cnxAccepted() once. The checked invocation helpers are retained for kernels
// whose first lookup is also their generation fence.
fn cnxAcceptedCellInvocationUnchecked(ordinal:u32)->u32{
  if(ordinal>=cnxLoad(${H.acceptedCellCount}u)){return INVALID;}
  return cnxArenaLoad(CNX_CELL_IDS+ordinal);
}
fn cnxAcceptedCellInvocation(ordinal:u32)->u32{
  if(!cnxAccepted()){return INVALID;}return cnxAcceptedCellInvocationUnchecked(ordinal);
}
fn cnxAcceptedRowInvocationUnchecked(ordinal:u32)->u32{
  if(ordinal>=cnxLoad(${H.acceptedRowWorklistCount}u)){return INVALID;}
  return cnxArenaLoad(CNX_ROW_RECORDS+4u*ordinal);
}
fn cnxAcceptedRowInvocation(ordinal:u32)->u32{
  if(!cnxAccepted()){return INVALID;}return cnxAcceptedRowInvocationUnchecked(ordinal);
}
fn cnxCellOrdinalUnchecked(stableCell:u32)->u32{
  if(stableCell>=CNX_CELL_CAPACITY){return INVALID;}
  let ordinal=cnxArenaLoad(CNX_CELL_ORDINALS+stableCell);
  if(ordinal>=cnxLoad(${H.acceptedCellCount}u)
    ||cnxArenaLoad(CNX_CELL_IDS+ordinal)!=stableCell){return INVALID;}
  return ordinal;
}
fn cnxRowOrdinalUnchecked(stableRow:u32)->u32{
  if(stableRow>=CNX_ROW_CAPACITY){return INVALID;}
  let ordinal=cnxArenaLoad(CNX_ROW_ORDINALS+stableRow);
  if(ordinal>=cnxLoad(${H.acceptedRowWorklistCount}u)
    ||cnxArenaLoad(CNX_ROW_RECORDS+4u*ordinal)!=stableRow){return INVALID;}
  return ordinal;
}
fn cnxRowOrdinal(stableRow:u32)->u32{
  if(!cnxAccepted()){return INVALID;}return cnxRowOrdinalUnchecked(stableRow);
}
fn cnxStableRowUnchecked(rowOrdinal:u32)->u32{
  if(rowOrdinal>=cnxLoad(${H.acceptedRowWorklistCount}u)){return INVALID;}
  return cnxArenaLoad(CNX_ROW_RECORDS+4u*rowOrdinal);
}
fn cnxCellIncidenceRangeUnchecked(stableCell:u32)->vec2u{
  let ordinal=cnxCellOrdinalUnchecked(stableCell);if(ordinal==INVALID){return vec2u(0u);}
  return vec2u(cnxArenaLoad(CNX_CELL_RANGES+2u*ordinal),
    cnxArenaLoad(CNX_CELL_RANGES+2u*ordinal+1u));
}
fn cnxIncidenceRowOrdinalUnchecked(at:u32)->u32{
  if(at>=CNX_INCIDENCE_CAPACITY){return INVALID;}
  return cnxArenaLoad(CNX_INCIDENCES+2u*at);
}
fn cnxIncidenceOwnTermUnchecked(at:u32)->u32{
  if(at>=CNX_INCIDENCE_CAPACITY){return INVALID;}
  return cnxArenaLoad(CNX_INCIDENCES+2u*at+1u);
}
fn cnxRowTermRangeByOrdinalUnchecked(ordinal:u32)->vec2u{
  if(ordinal>=cnxLoad(${H.acceptedRowWorklistCount}u)
    ||cnxArenaLoad(CNX_ROW_RECORDS+4u*ordinal)==INVALID){return vec2u(0u);}
  return vec2u(cnxArenaLoad(CNX_ROW_RECORDS+4u*ordinal+1u),
    cnxArenaLoad(CNX_ROW_RECORDS+4u*ordinal+2u));
}
fn cnxRowPackedMetadataByOrdinal(ordinal:u32)->u32{return cnxArenaLoad(CNX_ROW_RECORDS+4u*ordinal+3u);}
fn cnxRowStaticOpenUnchecked(stableRow:u32)->bool{let ordinal=cnxRowOrdinalUnchecked(stableRow);
  return ordinal!=INVALID&&(cnxRowPackedMetadataByOrdinal(ordinal)&(1u<<8u))!=0u;}
fn cnxRowTermCellUnchecked(term:u32)->u32{
  if(term>=CNX_TERM_CAPACITY){return INVALID;}return cnxArenaLoad(CNX_TERMS+2u*term);}
fn cnxRowTermCoefficientUnchecked(term:u32)->f32{
  if(term>=CNX_TERM_CAPACITY){return 0.0;}
  return bitcast<f32>(cnxArenaLoad(CNX_TERMS+2u*term+1u));
}
fn cnxIncidenceOwnCoefficientUnchecked(at:u32)->f32{
  return cnxRowTermCoefficientUnchecked(cnxIncidenceOwnTermUnchecked(at));
}

fn cnxFault(reason:u32,id:u32){atomicOr(&topologyArena[CNX_HEADER+${H.fault}u],reason);
  let prior=atomicCompareExchangeWeak(&topologyArena[CNX_HEADER+${H.firstFaultId}u],INVALID,id);
  if(!prior.exchanged&&prior.old_value==INVALID){
    _=atomicCompareExchangeWeak(&topologyArena[CNX_HEADER+${H.firstFaultId}u],INVALID,id);
  }
}
fn cnxPublishSealFailure(){
  if(cnxLoad(CNX_H_FAULT)==0u){return;}
  ${publishFailure}
}
fn cnxClaim(map:u32,key:u32,value:u32,fault:u32)->bool{
  loop{let claimed=atomicCompareExchangeWeak(&topologyArena[map+key],INVALID,value);
    if(claimed.exchanged){return true;}if(claimed.old_value!=INVALID){
      if(claimed.old_value!=value){cnxFault(fault,key);}return claimed.old_value==value;}
  }
  return false;
}

@compute @workgroup_size(1)
fn beginCompiledTopologyGeneration(){
  let sourceGeneration=cnxSourceGeneration();let sourceSlot=cnxSourceAcceptedSlot();
  let current=cnxLoad(${H.magic}u)==CNX_MAGIC&&cnxLoad(${H.version}u)==CNX_VERSION
    &&cnxLoad(${H.phase}u)==${P.accepted}u&&cnxLoad(${H.fault}u)==0u
    &&cnxLoad(${H.generation}u)==sourceGeneration
    &&cnxLoad(${H.sourceTopologyGeneration}u)==sourceGeneration
    &&cnxLoad(${H.sourceAcceptedSlot}u)==sourceSlot&&cnxSourceLifecycleAccepted()
    &&(cnxLoad(${H.readyViews}u)&CNX_REQUIRED_VIEWS)==CNX_REQUIRED_VIEWS
    &&cnxLoad(${H.certificate}u)==cnxCertificate();
  cnxStore(${H.rebuildRequired}u,select(1u,0u,current));if(current){return;}
  cnxStore(${H.magic}u,CNX_MAGIC);cnxStore(${H.version}u,CNX_VERSION);
  cnxStore(${H.headerWords}u,CNX_HEADER_WORDS);cnxStore(${H.totalWords}u,${layout.totalWords}u);
  cnxStore(${H.phase}u,${P.building}u);cnxStore(${H.fault}u,0u);cnxStore(${H.firstFaultId}u,INVALID);
  cnxStore(${H.sourceTopologyGeneration}u,sourceGeneration);cnxStore(${H.sourceAcceptedSlot}u,sourceSlot);
  cnxStore(${H.acceptedCellCount}u,${cellCount});
  cnxStore(${H.acceptedRowWorklistCount}u,${rowCount});
  cnxStore(${H.acceptedRowCount}u,0u);cnxStore(${H.orderedTermCount}u,0u);
  cnxStore(${H.cellIncidenceCount}u,0u);cnxStore(${H.requiredViews}u,CNX_REQUIRED_VIEWS);
  cnxStore(${H.readyViews}u,0u);cnxStore(${H.physicalFaceCount}u,0u);
  cnxStore(${H.physicalFaceEntryCount}u,0u);cnxStore(${H.certificate}u,0u);
  cnxStore(${H.cellCapacity}u,CNX_CELL_CAPACITY);cnxStore(${H.rowCapacity}u,CNX_ROW_CAPACITY);
  cnxStore(${H.termCapacity}u,CNX_TERM_CAPACITY);cnxStore(${H.incidenceCapacity}u,CNX_INCIDENCE_CAPACITY);
  cnxStore(${H.physicalFaceCapacity}u,CNX_FACE_CAPACITY);
  cnxStore(${H.acceptedCellIdsBase}u,CNX_CELL_IDS);cnxStore(${H.cellOrdinalByStableBase}u,CNX_CELL_ORDINALS);
  cnxStore(${H.acceptedCellRangesBase}u,CNX_CELL_RANGES);cnxStore(${H.acceptedRowRecordsBase}u,CNX_ROW_RECORDS);
  cnxStore(${H.rowOrdinalByStableBase}u,CNX_ROW_ORDINALS);cnxStore(${H.orderedTermsBase}u,CNX_TERMS);
  cnxStore(${H.cellIncidencesBase}u,CNX_INCIDENCES);cnxStore(${H.clearedGeneration}u,INVALID);
  cnxStore(${H.cellPlaneGeneration}u,INVALID);cnxStore(${H.rowPlaneGeneration}u,INVALID);
  cnxStore(${H.incidencePlaneGeneration}u,INVALID);
  if(!cnxSourceLifecycleAccepted()){cnxFault(${F.sourceNotAccepted}u,sourceGeneration);}
  if(${cellCount}>CNX_CELL_CAPACITY){cnxFault(${F.cellCapacity}u,${cellCount});}
  if(${rowCount}>CNX_ROW_CAPACITY){cnxFault(${F.rowCapacity}u,${rowCount});}
}

@compute @workgroup_size(64)
fn clearCompiledTopologyGeneration(@builtin(global_invocation_id)gid:vec3u){
  if(!cnxBuilding()){return;}let at=cnxLinearInvocation(gid);
  if(at<CNX_CELL_CAPACITY){cnxArenaStore(CNX_CELL_ORDINALS+at,INVALID);}
  if(at<CNX_ROW_CAPACITY){cnxArenaStore(CNX_ROW_ORDINALS+at,INVALID);
    cnxArenaStore(CNX_ROW_RECORDS+4u*at,INVALID);}
  if(at==0u){cnxStore(${H.clearedGeneration}u,cnxSourceGeneration());}
}

@compute @workgroup_size(64)
fn compileCompiledTopologyCells(@builtin(global_invocation_id)gid:vec3u){
  if(!cnxBuilding()||cnxLoad(${H.clearedGeneration}u)!=cnxSourceGeneration()){return;}
  let ordinal=cnxLinearInvocation(gid);if(ordinal==0u){cnxStore(${H.cellPlaneGeneration}u,cnxSourceGeneration());}
  if(ordinal>=cnxLoad(${H.acceptedCellCount}u)){return;}
  let cell=${cellAt("ordinal")};if(cell==INVALID||cell>=CNX_CELL_CAPACITY){cnxFault(${F.cellCapacity}u,cell);return;}
  cnxArenaStore(CNX_CELL_IDS+ordinal,cell);
  _=cnxClaim(CNX_CELL_ORDINALS,cell,ordinal,${F.duplicateCell}u);
}

@compute @workgroup_size(64)
fn compileCompiledTopologyRows(@builtin(global_invocation_id)gid:vec3u){
  if(!cnxBuilding()||cnxLoad(${H.cellPlaneGeneration}u)!=cnxSourceGeneration()){return;}
  let ordinal=cnxLinearInvocation(gid);if(ordinal==0u){cnxStore(${H.rowPlaneGeneration}u,cnxSourceGeneration());}
  if(ordinal>=cnxLoad(${H.acceptedRowWorklistCount}u)){return;}
  let row=${rowAt("ordinal")};if(row==INVALID){return;}
  if(row>=CNX_ROW_CAPACITY){cnxFault(${F.rowCapacity}u,row);return;}
  if(!rowAccepted(row)){cnxArenaStore(CNX_ROW_RECORDS+4u*ordinal,INVALID);return;}
  let range=rowTermRange(row);if(range.y<range.x||range.y>CNX_TERM_CAPACITY){
    cnxFault(${F.termCapacity}u,row);return;
  }
  let count=range.y-range.x;if(count==0u||count>511u){cnxFault(${F.malformedRow}u,row);return;}
  if(!cnxClaim(CNX_ROW_ORDINALS,row,ordinal,${F.duplicateRow}u)){return;}
  var negative=0u;var positive=0u;
  for(var term=range.x;term<range.y;term+=1u){
    let stableCell=termCell(term);let coefficient=termCoefficient(term);
    if(stableCell>=CNX_CELL_CAPACITY){cnxFault(${F.malformedRow}u,row);return;}
    // Every term of an accepted row must address the same accepted cell image.
    // Numerical consumers use the stable id directly after the generation
    // fence, so accepting an unlisted endpoint would turn it into an unchecked
    // state-buffer index. Treat that source inconsistency as a whole-image
    // rejection rather than publishing a partially connected row.
    if(cnxCellOrdinalUnchecked(stableCell)==INVALID){
      cnxFault(${F.malformedRow}u,row);return;
    }
    negative+=select(0u,1u,coefficient<0.0);positive+=select(0u,1u,coefficient>0.0);
    // Fields remain stable-id indexed. Store that id directly so every hot
    // gradient term avoids ordinal -> acceptedCellIds remapping. The optional
    // ordinal accessor maps only when a consumer explicitly needs it.
    cnxArenaStore(CNX_TERMS+2u*term,stableCell);
    cnxArenaStore(CNX_TERMS+2u*term+1u,bitcast<u32>(coefficient));
  }
  var opcode=0u;
  if(count==2u&&negative==1u&&positive==1u){opcode=1u;}
  if(count==3u&&((negative==1u&&positive==2u)||(negative==2u&&positive==1u))){opcode=2u;}
  if(count==5u&&((negative==1u&&positive==4u)||(negative==4u&&positive==1u))){opcode=3u;}
  cnxArenaStore(CNX_ROW_RECORDS+4u*ordinal,row);cnxArenaStore(CNX_ROW_RECORDS+4u*ordinal+1u,range.x);
  cnxArenaStore(CNX_ROW_RECORDS+4u*ordinal+2u,range.y);
  var metadata=rowAxis(row)|(rowKind(row)<<2u)|(opcode<<4u);
  ${staticOpen ? `if(${staticOpen("row")}){metadata|=1u<<8u;}` : ""}
  cnxArenaStore(CNX_ROW_RECORDS+4u*ordinal+3u,metadata);
  atomicAdd(&topologyArena[CNX_HEADER+${H.acceptedRowCount}u],1u);
  atomicAdd(&topologyArena[CNX_HEADER+${H.orderedTermCount}u],count);
}

@compute @workgroup_size(64)
fn compileCompiledTopologyCellIncidences(@builtin(global_invocation_id)gid:vec3u){
  if(!cnxBuilding()||cnxLoad(${H.rowPlaneGeneration}u)!=cnxSourceGeneration()){return;}
  let ordinal=cnxLinearInvocation(gid);if(ordinal==0u){cnxStore(${H.incidencePlaneGeneration}u,cnxSourceGeneration());}
  if(ordinal>=cnxLoad(${H.acceptedCellCount}u)){return;}
  let cell=cnxArenaLoad(CNX_CELL_IDS+ordinal);let sourceRange=incidenceRange(cell);
  if(sourceRange.y<sourceRange.x||sourceRange.y>CNX_INCIDENCE_CAPACITY){
    cnxFault(${F.incidenceCapacity}u,cell);return;
  }
  var out=sourceRange.x;
  for(var sourceAt=sourceRange.x;sourceAt<sourceRange.y;sourceAt+=1u){
    let stableRow=incidenceRow(sourceAt);if(stableRow>=CNX_ROW_CAPACITY){continue;}
    let rowOrdinal=cnxArenaLoad(CNX_ROW_ORDINALS+stableRow);if(rowOrdinal==INVALID){continue;}
    let ownTerm=incidenceTerm(sourceAt);let rowBegin=cnxArenaLoad(CNX_ROW_RECORDS+4u*rowOrdinal+1u);
    let rowEnd=cnxArenaLoad(CNX_ROW_RECORDS+4u*rowOrdinal+2u);
    if(ownTerm<rowBegin||ownTerm>=rowEnd||cnxArenaLoad(CNX_TERMS+2u*ownTerm)!=cell){
      cnxFault(${F.malformedIncidence}u,cell);continue;
    }
    cnxArenaStore(CNX_INCIDENCES+2u*out,rowOrdinal);
    cnxArenaStore(CNX_INCIDENCES+2u*out+1u,ownTerm);out+=1u;
  }
  cnxArenaStore(CNX_CELL_RANGES+2u*ordinal,sourceRange.x);
  cnxArenaStore(CNX_CELL_RANGES+2u*ordinal+1u,out);
  atomicAdd(&topologyArena[CNX_HEADER+${H.cellIncidenceCount}u],out-sourceRange.x);
}

@compute @workgroup_size(1)
fn sealCompiledTopologyGeneration(){
  if(!cnxBuildRequired()){return;}
  let sourceGeneration=cnxSourceGeneration();
  if(!cnxBuilding()){
    cnxFault(${F.staleSource}u,sourceGeneration);cnxPublishSealFailure();
    cnxStore(${H.phase}u,${P.fault}u);return;
  }
  if(cnxLoad(${H.cellPlaneGeneration}u)!=sourceGeneration
    ||cnxLoad(${H.rowPlaneGeneration}u)!=sourceGeneration
    ||cnxLoad(${H.incidencePlaneGeneration}u)!=sourceGeneration){
    cnxFault(${F.staleSource}u,sourceGeneration);cnxPublishSealFailure();
    cnxStore(${H.phase}u,${P.fault}u);return;
  }
  var ready=${V.connectivity}u|(cnxLoad(${H.readyViews}u)&${V.transport}u);
  ${transportSeal}
  ${boolView(V.velocityExtension, options.velocityExtension)}
  ${boolView(V.projection, options.projection)}
  ${boolView(V.presentation, options.presentation)}
  cnxStore(${H.readyViews}u,ready);
  if((ready&CNX_REQUIRED_VIEWS)!=CNX_REQUIRED_VIEWS){
    cnxFault(${F.missingRequiredView}u,ready);cnxPublishSealFailure();
    cnxStore(${H.phase}u,${P.fault}u);return;
  }
  if(cnxLoad(${H.fault}u)!=0u){cnxPublishSealFailure();cnxStore(${H.phase}u,${P.fault}u);return;}
  cnxStore(${H.generation}u,sourceGeneration);cnxStore(${H.certificate}u,cnxCertificate());
  cnxStore(${H.rebuildRequired}u,0u);cnxStore(${H.phase}u,${P.accepted}u);
}
`;
  return compiled + (options.transportStorage
    ? createSparseCM12CompiledTopologyTransportAccessWGSL(options.transportStorage) : "");
}

export interface SparseCM12CompiledTopologyTransportLayout {
  readonly rowSubfaceRanges: number;
  readonly cellSubfaceRanges: number;
  readonly cellSubfaceEntries: number;
  readonly subfaceMetadata: number;
}

/** Hot read-only aliases over the geometric-volume image sealed by CNX.
 * Entry points must fence cnxTransportViewValidForAcceptedTopology() once. */
export function createSparseCM12CompiledTopologyTransportAccessWGSL(
  layout: SparseCM12CompiledTopologyTransportLayout,
): string {
  return /* wgsl */ `
const CNX_GV_ROWS:u32=${layout.rowSubfaceRanges}u;
const CNX_GV_CELL_FACES:u32=${layout.cellSubfaceRanges}u;
const CNX_GV_CELL_FACE_ENTRIES:u32=${layout.cellSubfaceEntries}u;
const CNX_GV_META:u32=${layout.subfaceMetadata}u;
fn cnxPhysicalFaceRangeUnchecked(stableRow:u32)->vec2u{
  let first=bitcast<u32>(state[CNX_GV_ROWS+2u*stableRow]);
  let count=bitcast<u32>(state[CNX_GV_ROWS+2u*stableRow+1u]);return vec2u(first,first+count);
}
fn cnxPhysicalFaceCellsUnchecked(face:u32)->vec2u{
  return vec2u(bitcast<u32>(state[CNX_GV_META+4u*face]),
    bitcast<u32>(state[CNX_GV_META+4u*face+1u]));
}
fn cnxPhysicalFaceRowUnchecked(face:u32)->u32{
  return bitcast<u32>(state[CNX_GV_META+4u*face+2u]);
}
fn cnxPhysicalFaceAreaUnchecked(face:u32)->f32{
  return state[CNX_GV_META+4u*face+3u];
}
fn cnxCellFaceRangeUnchecked(stableCell:u32)->vec2u{
  return vec2u(bitcast<u32>(state[CNX_GV_CELL_FACES+2u*stableCell]),
    bitcast<u32>(state[CNX_GV_CELL_FACES+2u*stableCell+1u]));
}
fn cnxCellFaceEntryUnchecked(at:u32)->u32{
  return bitcast<u32>(state[CNX_GV_CELL_FACE_ENTRIES+at]);
}`;
}
