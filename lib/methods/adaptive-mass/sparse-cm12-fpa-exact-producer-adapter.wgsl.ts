import {
  SPARSE_CM12_FPA_EXACT_PRODUCER_CAUSE as C,
  SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY as F,
  SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_COUNT,
  SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_HEADER as FH,
  SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS,
  SPARSE_CM12_FPA_EXACT_PRODUCER_FAULT as FAULT,
  SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER as H,
  SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER_WORDS,
  SPARSE_CM12_FPA_EXACT_PRODUCER_INVALID,
  SPARSE_CM12_FPA_EXACT_PRODUCER_MAGIC,
  SPARSE_CM12_FPA_EXACT_PRODUCER_PHASE as PHASE,
  SPARSE_CM12_FPA_EXACT_PRODUCER_VERSION,
  type SparseCM12FPAExactProducerAdapterLayout,
} from "./sparse-cm12-fpa-exact-producer-adapter";

/** Standalone FPE1 library. Required hooks are declared by its source manifest. */
export function createSparseCM12FPAExactProducerAdapterWGSL(options: {
  readonly layout: SparseCM12FPAExactProducerAdapterLayout;
  readonly arenaName?: string;
  readonly prefix?: string;
}): string {
  const { layout } = options;
  const arena = options.arenaName ?? "topologyArena";
  const p = options.prefix ?? "fpe";
  const h = (word: number) => `${layout.baseWords + word}u`;
  const familyAt = (family: number, word: number) =>
    `${layout.familyBaseWords + family * SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS
      + word}u`;
  return /* wgsl */ `
const ${p}Invalid:u32=${SPARSE_CM12_FPA_EXACT_PRODUCER_INVALID}u;
const ${p}FamilyCount:u32=${SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_COUNT}u;
const ${p}CellCapacity:u32=${layout.cellCapacity}u;
const ${p}RowCapacity:u32=${layout.rowCapacity}u;
const ${p}Liquid:u32=${F.liquidPhaseCell}u;
const ${p}Vex:u32=${F.vexCell}u;
const ${p}SourceFace:u32=${F.sourceFaceRow}u;
const ${p}Topology:u32=${F.topologyCell}u;
const ${p}MovingCell:u32=${F.movingSolidCell}u;
const ${p}MovingRow:u32=${F.movingSolidRow}u;
const ${p}Policy:u32=${F.policyRow}u;

fn ${p}FamilyAt(family:u32,word:u32)->u32{
  return ${layout.familyBaseWords}u
    +family*${SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS}u+word;
}
fn ${p}HeaderValid()->bool{
  return atomicLoad(&${arena}[${h(H.magic)}])==${SPARSE_CM12_FPA_EXACT_PRODUCER_MAGIC}u
    &&atomicLoad(&${arena}[${h(H.version)}])==${SPARSE_CM12_FPA_EXACT_PRODUCER_VERSION}u
    &&atomicLoad(&${arena}[${h(H.headerWords)}])
      ==${SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER_WORDS}u
    &&atomicLoad(&${arena}[${h(H.familyWords)}])
      ==${SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY_WORDS}u
    &&atomicLoad(&${arena}[${h(H.familyCount)}])==${p}FamilyCount
    &&atomicLoad(&${arena}[${h(H.familyBaseWords)}])==${layout.familyBaseWords}u;
}
fn ${p}Fail(code:u32,family:u32,id:u32){
  let old=atomicCompareExchangeWeak(&${arena}[${h(H.fault)}],0u,code);
  if(old.exchanged){
    atomicStore(&${arena}[${h(H.firstFaultFamily)}],family);
    atomicStore(&${arena}[${h(H.firstFaultId)}],id);
  }
  atomicStore(&${arena}[${h(H.phase)}],${PHASE.fault}u);
}
fn ${p}Begin()->bool{
  if(!${p}HeaderValid()){
    ${p}Fail(${FAULT.invalidHeader}u,${p}Invalid,${p}Invalid);return false;}
  let phase=atomicLoad(&${arena}[${h(H.phase)}]);
  if(phase!=${PHASE.uninitialized}u&&phase!=${PHASE.accepted}u){
    ${p}Fail(${FAULT.invalidPhase}u,${p}Invalid,phase);return false;}
  let accepted=atomicLoad(&${arena}[${h(H.acceptedGeneration)}]);
  if(accepted>=0x7ffffffeu){
    ${p}Fail(${FAULT.generationGap}u,${p}Invalid,accepted);return false;}
  let candidate=accepted+1u;let frame=fpeaFrameGeneration();
  if(frame==0u){${p}Fail(${FAULT.generationGap}u,${p}Invalid,frame);return false;}
  atomicStore(&${arena}[${h(H.candidateGeneration)}],candidate);
  atomicStore(&${arena}[${h(H.frameGeneration)}],frame);
  atomicStore(&${arena}[${h(H.fault)}],0u);
  atomicStore(&${arena}[${h(H.firstFaultFamily)}],${p}Invalid);
  atomicStore(&${arena}[${h(H.firstFaultId)}],${p}Invalid);
  atomicStore(&${arena}[${h(H.coveredReceipts)}],0u);
  atomicStore(&${arena}[${h(H.producerEventCount)}],0u);
  atomicStore(&${arena}[${h(H.preparationRowWriteCount)}],0u);
  atomicStore(&${arena}[${h(H.pendingEventCount)}],0u);
  atomicStore(&${arena}[${h(H.causeMask)}],0u);
  var expectedTotal=0u;
  for(var family=0u;family<${p}FamilyCount;family+=1u){
    let expected=fpeaExpectedProducerReceipts(family);
    if(expected>0xffffffffu-expectedTotal){
      ${p}Fail(${FAULT.receiptOverflow}u,family,expected);return false;}
    expectedTotal+=expected;
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.generation}u)],candidate);
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.expectedReceipts}u)],expected);
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.coveredReceipts}u)],0u);
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.producerEventCount}u)],0u);
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.preparationRowWriteCount}u)],0u);
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.causeMask}u)],0u);
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.firstProducerId}u)],${p}Invalid);
    atomicStore(&${arena}[${p}FamilyAt(family,${FH.lastProducerId}u)],0u);
  }
  atomicStore(&${arena}[${h(H.expectedReceipts)}],expectedTotal);
  atomicStore(&${arena}[${h(H.phase)}],${PHASE.collecting}u);return true;
}
@compute @workgroup_size(1)
fn beginSparseCM12FPAExactProducerAdapter(){_=${p}Begin();}

fn ${p}OpenEvent(family:u32,id:u32,generation:u32,idCapacity:u32)->bool{
  if(atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.collecting}u){
    ${p}Fail(${FAULT.invalidPhase}u,family,id);return false;}
  if(family>=${p}FamilyCount){
    ${p}Fail(${FAULT.invalidFamily}u,family,id);return false;}
  if(id>=idCapacity){${p}Fail(${FAULT.invalidId}u,family,id);return false;}
  let candidate=atomicLoad(&${arena}[${h(H.candidateGeneration)}]);
  if(generation!=candidate
    ||atomicLoad(&${arena}[${p}FamilyAt(family,${FH.generation}u)])!=candidate){
    ${p}Fail(${FAULT.generationGap}u,family,id);return false;}
  atomicAdd(&${arena}[${h(H.pendingEventCount)}],1u);
  atomicMin(&${arena}[${p}FamilyAt(family,${FH.firstProducerId}u)],id);
  atomicMax(&${arena}[${p}FamilyAt(family,${FH.lastProducerId}u)],id);
  return true;
}
fn ${p}MarkRow(family:u32,row:u32,cause:u32)->bool{
  if(row>=${p}RowCapacity){${p}Fail(${FAULT.invalidId}u,family,row);return false;}
  if(!fpaPreparationRowLive(row)){return true;}
  if(!fpaMarkPreparationRow(row,cause,0u,false)){
    ${p}Fail(${FAULT.fpaMarkRejected}u,family,row);return false;}
  atomicAdd(&${arena}[${h(H.preparationRowWriteCount)}],1u);
  atomicAdd(&${arena}[${p}FamilyAt(family,${FH.preparationRowWriteCount}u)],1u);
  atomicOr(&${arena}[${h(H.causeMask)}],cause);
  atomicOr(&${arena}[${p}FamilyAt(family,${FH.causeMask}u)],cause);
  return true;
}
fn ${p}CloseEvent(family:u32,id:u32)->bool{
  let expected=atomicLoad(&${arena}[${p}FamilyAt(family,${FH.expectedReceipts}u)]);
  let old=atomicAdd(&${arena}[${p}FamilyAt(family,${FH.coveredReceipts}u)],1u);
  if(old>=expected){${p}Fail(${FAULT.receiptOverflow}u,family,id);return false;}
  atomicAdd(&${arena}[${h(H.coveredReceipts)}],1u);
  atomicAdd(&${arena}[${h(H.producerEventCount)}],1u);
  atomicAdd(&${arena}[${p}FamilyAt(family,${FH.producerEventCount}u)],1u);
  atomicSub(&${arena}[${h(H.pendingEventCount)}],1u);
  return true;
}
fn ${p}ExpandIncidence(family:u32,cell:u32,cause:u32)->bool{
  for(var at=fpeaIncidenceBegin(cell);at<fpeaIncidenceEnd(cell);at+=1u){
    if(!${p}MarkRow(family,fpeaIncidenceRow(at),cause)){return false;}
  }return true;
}
fn ${p}ExpandVexReverse(cell:u32,cause:u32)->bool{
  if(!fpeaVexReverseProvenanceValid()){
    ${p}Fail(${FAULT.reverseDependency}u,${p}Vex,cell);return false;}
  let begin=fpeaVexReverseBegin(cell);let end=fpeaVexReverseEnd(cell);
  if(begin==${p}Invalid||end==${p}Invalid||begin>end){
    ${p}Fail(${FAULT.reverseDependency}u,${p}Vex,cell);return false;}
  for(var at=begin;at<end;at+=1u){
    let row=fpeaVexReverseRow(at);
    if(row==${p}Invalid||row>=${p}RowCapacity){
      ${p}Fail(${FAULT.reverseDependency}u,${p}Vex,cell);return false;}
    if(!${p}MarkRow(${p}Vex,row,cause)){return false;}
  }return true;
}
fn ${p}Changed4(before:vec4u,after:vec4u)->bool{return any(before!=after);}

fn fpeRecordLiquidPhaseCell(cell:u32,before:u32,after:u32,generation:u32)->bool{
  if(!${p}OpenEvent(${p}Liquid,cell,generation,${p}CellCapacity)){return false;}
  if(before==after){${p}Fail(${FAULT.nonChangeEvent}u,${p}Liquid,cell);return false;}
  if(!${p}ExpandIncidence(${p}Liquid,cell,${C.liquidPhaseCell}u)){return false;}
  return ${p}CloseEvent(${p}Liquid,cell);
}
fn fpeRecordVexCell(cell:u32,before:vec4u,after:vec4u,generation:u32)->bool{
  if(!${p}OpenEvent(${p}Vex,cell,generation,${p}CellCapacity)){return false;}
  if(!${p}Changed4(before,after)){
    ${p}Fail(${FAULT.nonChangeEvent}u,${p}Vex,cell);return false;}
  if(!${p}ExpandVexReverse(cell,${C.vexCell}u)){return false;}
  return ${p}CloseEvent(${p}Vex,cell);
}
fn fpeRecordSourceFaceRow(row:u32,before:u32,after:u32,generation:u32)->bool{
  if(!${p}OpenEvent(${p}SourceFace,row,generation,${p}RowCapacity)){return false;}
  if(before==after){${p}Fail(${FAULT.nonChangeEvent}u,${p}SourceFace,row);return false;}
  if(!${p}MarkRow(${p}SourceFace,row,${C.sourceFaceRow}u)){return false;}
  return ${p}CloseEvent(${p}SourceFace,row);
}
fn fpeRecordTopologyCell(cell:u32,before:vec2u,after:vec2u,generation:u32)->bool{
  if(!${p}OpenEvent(${p}Topology,cell,generation,${p}CellCapacity)){return false;}
  if(all(before==after)){
    ${p}Fail(${FAULT.nonChangeEvent}u,${p}Topology,cell);return false;}
  if(!${p}ExpandIncidence(${p}Topology,cell,${C.topologyCell}u)){return false;}
  return ${p}CloseEvent(${p}Topology,cell);
}
fn fpeRecordMovingSolidCell(cell:u32,before:vec4u,after:vec4u,generation:u32)->bool{
  if(!${p}OpenEvent(${p}MovingCell,cell,generation,${p}CellCapacity)){return false;}
  if(!${p}Changed4(before,after)){
    ${p}Fail(${FAULT.nonChangeEvent}u,${p}MovingCell,cell);return false;}
  if(!${p}ExpandIncidence(${p}MovingCell,cell,${C.movingSolidCell}u)){return false;}
  return ${p}CloseEvent(${p}MovingCell,cell);
}
fn fpeRecordMovingSolidRow(row:u32,before:vec4u,after:vec4u,generation:u32)->bool{
  if(!${p}OpenEvent(${p}MovingRow,row,generation,${p}RowCapacity)){return false;}
  if(!${p}Changed4(before,after)){
    ${p}Fail(${FAULT.nonChangeEvent}u,${p}MovingRow,row);return false;}
  if(!${p}MarkRow(${p}MovingRow,row,${C.movingSolidRow}u)){return false;}
  return ${p}CloseEvent(${p}MovingRow,row);
}
// Policy epochs never imply a scan. The policy owner calls this once for each
// exact affected stable row and declares that count before begin.
fn fpeRecordPolicyRow(row:u32,beforeEpoch:u32,afterEpoch:u32,generation:u32)->bool{
  if(!${p}OpenEvent(${p}Policy,row,generation,${p}RowCapacity)){return false;}
  if(beforeEpoch==afterEpoch){${p}Fail(${FAULT.nonChangeEvent}u,${p}Policy,row);return false;}
  if(!${p}MarkRow(${p}Policy,row,${C.policyRow}u)){return false;}
  return ${p}CloseEvent(${p}Policy,row);
}

fn ${p}PrepareSeal()->bool{
  if(atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.collecting}u){
    ${p}Fail(${FAULT.invalidPhase}u,${p}Invalid,${p}Invalid);return false;}
  if(atomicLoad(&${arena}[${h(H.pendingEventCount)}])!=0u){
    ${p}Fail(${FAULT.uncoveredWrite}u,${p}Invalid,${p}Invalid);return false;}
  for(var family=0u;family<${p}FamilyCount;family+=1u){
    if(atomicLoad(&${arena}[${p}FamilyAt(family,${FH.coveredReceipts}u)])
      !=atomicLoad(&${arena}[${p}FamilyAt(family,${FH.expectedReceipts}u)])){
      ${p}Fail(${FAULT.coverageGap}u,family,${p}Invalid);return false;}
  }
  if(atomicLoad(&${arena}[${h(H.coveredReceipts)}])
    !=atomicLoad(&${arena}[${h(H.expectedReceipts)}])){
    ${p}Fail(${FAULT.coverageGap}u,${p}Invalid,${p}Invalid);return false;}
  // Publish no FPA receipts until every family is exact and fully covered.
  // Thus overflow/uncovered-write/coverage faults cannot leave FPA apparently
  // complete merely because the expected prefix was already covered.
  atomicStore(&${arena}[${h(H.phase)}],${PHASE.covering}u);return true;
}
@compute @workgroup_size(64)
fn sealSparseCM12FPAExactProducerAdapter(
 @builtin(local_invocation_index)lane:u32){
  if(lane==0u){_=${p}PrepareSeal();}
  storageBarrier();workgroupBarrier();
  if(atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.covering}u){return;}
  let expected=atomicLoad(&${arena}[${h(H.expectedReceipts)}]);
  for(var receipt=lane;receipt<expected;receipt+=64u){
    fpaCoverPreparationReceipt();
  }
  storageBarrier();workgroupBarrier();
  if(lane==0u){atomicStore(&${arena}[${h(H.phase)}],${PHASE.sealed}u);}
}
@compute @workgroup_size(1)
fn acceptSparseCM12FPAExactProducerAdapter(){
  if(atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.sealed}u
    ||atomicLoad(&${arena}[${h(H.fault)}])!=0u){
    ${p}Fail(${FAULT.invalidPhase}u,${p}Invalid,${p}Invalid);return;}
  atomicStore(&${arena}[${h(H.acceptedGeneration)}],
    atomicLoad(&${arena}[${h(H.candidateGeneration)}]));
  atomicStore(&${arena}[${h(H.phase)}],${PHASE.accepted}u);
}
`;
}
