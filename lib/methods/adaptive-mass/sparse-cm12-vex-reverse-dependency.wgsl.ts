import {
  SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_FAULT as JFAULT,
  SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_HEADER as JH,
  SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_PHASE as JPHASE,
  SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_ROW_WORDS,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_FLAG as FLAG,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER as H,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER_WORDS,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_INVALID,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_MAGIC,
  SPARSE_CM12_VEX_REVERSE_DEPENDENCY_VERSION,
  type SparseCM12VexReverseDependency,
  type SparseCM12VexDependencyJournalLayout,
} from "./sparse-cm12-vex-reverse-dependency";

/** Immutable VRD1 lookup library. It performs no writes and authors no work. */
export function createSparseCM12VexReverseDependencyLookupWGSL(options: {
  readonly dependency: SparseCM12VexReverseDependency;
  readonly bindingGroup?: number;
  readonly binding?: number;
  readonly bufferName?: string;
  readonly prefix?: string;
}): string {
  const { dependency } = options;
  const { layout } = dependency;
  const group = options.bindingGroup ?? 0;
  const binding = options.binding ?? 0;
  const buffer = options.bufferName ?? "vexReverseDependency";
  const p = options.prefix ?? "vrd";
  const h = (word: number) => `${word}u`;
  const hash = (at: number, words: readonly number[]) => words.map((word, lane) =>
    `${buffer}[${h(at + lane)}]==${word}u`).join("&&");
  const flags = FLAG.complete | FLAG.validated | FLAG.supportedPolicy;
  return /* wgsl */ `
@group(${group})@binding(${binding})var<storage,read>${buffer}:array<u32>;
const ${p}Invalid:u32=${SPARSE_CM12_VEX_REVERSE_DEPENDENCY_INVALID}u;
fn ${p}ProvenanceValid()->bool{
  return arrayLength(&${buffer})>=${layout.totalWords - layout.baseWords}u
    &&${buffer}[${h(H.magic)}]==${SPARSE_CM12_VEX_REVERSE_DEPENDENCY_MAGIC}u
    &&${buffer}[${h(H.version)}]==${SPARSE_CM12_VEX_REVERSE_DEPENDENCY_VERSION}u
    &&${buffer}[${h(H.headerWords)}]==${SPARSE_CM12_VEX_REVERSE_DEPENDENCY_HEADER_WORDS}u
    &&(${buffer}[${h(H.flags)}]&${flags}u)==${flags}u
    &&${buffer}[${h(H.cellCount)}]==${layout.cellCount}u
    &&${buffer}[${h(H.rowCount)}]==${layout.rowCount}u
    &&${buffer}[${h(H.edgeCount)}]==${layout.edgeCount}u
    &&${buffer}[${h(H.edgeCapacity)}]==${layout.edgeCapacity}u
    &&${buffer}[${h(H.offsetBaseWords)}]==${layout.offsetBaseWords}u
    &&${buffer}[${h(H.rowBaseWords)}]==${layout.rowBaseWords}u
    &&${buffer}[${h(H.totalWords)}]==${layout.totalWords}u
    &&${buffer}[${h(H.topologyGeneration)}]
      ==${dependency.words[H.topologyGeneration]}u
    &&${hash(H.topologyHash0, dependency.topologyHash)}
    &&${hash(H.policyHash0, dependency.policyHash)}
    &&${hash(H.contentHash0, dependency.contentHash)};
}

fn ${p}Range(cell:u32)->vec2u{
  if(!${p}ProvenanceValid()||cell>=${layout.cellCount}u){
    return vec2u(${p}Invalid);}
  let begin=${buffer}[${layout.offsetBaseWords - layout.baseWords}u+cell];
  let end=${buffer}[${layout.offsetBaseWords - layout.baseWords}u+cell+1u];
  if(begin>end||end>${layout.edgeCount}u){return vec2u(${p}Invalid);}
  return vec2u(begin,end);
}
fn ${p}Row(at:u32)->u32{
  if(!${p}ProvenanceValid()||at>=${layout.edgeCount}u){return ${p}Invalid;}
  let row=${buffer}[${layout.rowBaseWords - layout.baseWords}u+at];
  return select(${p}Invalid,row,row<${layout.rowCount}u);
}
fn fpeaVexReverseProvenanceValid()->bool{return ${p}ProvenanceValid();}
fn fpeaVexReverseBegin(cell:u32)->u32{return ${p}Range(cell).x;}
fn fpeaVexReverseEnd(cell:u32)->u32{return ${p}Range(cell).y;}
fn fpeaVexReverseRow(at:u32)->u32{return ${p}Row(at);}
`;
}

/**
 * Candidate actual-read logger. One FPA invocation exclusively owns a row;
 * only 32-read chunk allocation and aggregate receipts require atomics.
 */
export function createSparseCM12VexDependencyJournalWGSL(options: {
  readonly layout: SparseCM12VexDependencyJournalLayout;
  readonly bindingGroup?: number; readonly binding?: number;
  readonly bufferName?: string; readonly prefix?: string;
}): string {
  const { layout } = options; const group = options.bindingGroup ?? 0;
  const binding = options.binding ?? 1; const arena = options.bufferName ?? "vexReadJournal";
  const p = options.prefix ?? "vrdj"; const h = (word: number) =>
    `${layout.baseWords + word}u`;
  return /* wgsl */ `
@group(${group})@binding(${binding})var<storage,read_write>${arena}:array<atomic<u32>>;
const ${p}Invalid:u32=0xffffffffu;
struct ${p}Cursor{row:u32,generation:u32,chunk:u32,used:u32,valid:u32}
fn ${p}RowAt(row:u32,word:u32)->u32{return ${layout.rowBaseWords}u
  +row*${SPARSE_CM12_VEX_DEPENDENCY_JOURNAL_ROW_WORDS}u+word;}
fn ${p}ChunkAt(chunk:u32,word:u32)->u32{return ${layout.chunkBaseWords}u
  +chunk*${layout.chunkWords}u+word;}
fn ${p}Fail(code:u32,row:u32,cell:u32){
  let first=atomicCompareExchangeWeak(&${arena}[${h(JH.fault)}],0u,code);
  if(first.exchanged){atomicStore(&${arena}[${h(JH.firstFaultRow)}],row);
    atomicStore(&${arena}[${h(JH.firstFaultCell)}],cell);}
  atomicStore(&${arena}[${h(JH.phase)}],${JPHASE.fault}u);
}
@compute @workgroup_size(1)
fn beginSparseCM12VexDependencyJournal(){
  let phase=atomicLoad(&${arena}[${h(JH.phase)}]);
  if(phase!=${JPHASE.accepted}u){${p}Fail(${JFAULT.invalidPhase}u,${p}Invalid,${p}Invalid);return;}
  let accepted=atomicLoad(&${arena}[${h(JH.acceptedGeneration)}]);
  atomicStore(&${arena}[${h(JH.candidateGeneration)}],accepted+1u);
  atomicStore(&${arena}[${h(JH.expectedRows)}],${p}ExpectedExecutedRows());
  atomicStore(&${arena}[${h(JH.coveredRows)}],0u);
  atomicStore(&${arena}[${h(JH.allocatedChunks)}],0u);
  atomicStore(&${arena}[${h(JH.rawReadCount)}],0u);
  atomicStore(&${arena}[${h(JH.fault)}],0u);
  atomicStore(&${arena}[${h(JH.firstFaultRow)}],${p}Invalid);
  atomicStore(&${arena}[${h(JH.firstFaultCell)}],${p}Invalid);
  atomicStore(&${arena}[${h(JH.phase)}],${JPHASE.collecting}u);
}
fn ${p}AllocateChunk(row:u32,generation:u32)->u32{
  let chunk=atomicAdd(&${arena}[${h(JH.allocatedChunks)}],1u);
  if(chunk>=${layout.chunkCapacity}u){
    ${p}Fail(${JFAULT.chunkOverflow}u,row,${p}Invalid);return ${p}Invalid;}
  atomicStore(&${arena}[${p}ChunkAt(chunk,0u)],${p}Invalid);
  atomicStore(&${arena}[${p}ChunkAt(chunk,1u)],0u);
  atomicStore(&${arena}[${p}ChunkAt(chunk,2u)],row);
  atomicStore(&${arena}[${p}ChunkAt(chunk,3u)],generation);return chunk;
}
fn ${p}BeginRow(row:u32,generation:u32)->${p}Cursor{
  var cursor:${p}Cursor;cursor.row=row;cursor.generation=generation;
  cursor.chunk=${p}Invalid;cursor.used=0u;cursor.valid=0u;
  if(atomicLoad(&${arena}[${h(JH.phase)}])!=${JPHASE.collecting}u){
    ${p}Fail(${JFAULT.invalidPhase}u,row,${p}Invalid);return cursor;}
  if(row>=${layout.rowCapacity}u){
    ${p}Fail(${JFAULT.invalidRow}u,row,${p}Invalid);return cursor;}
  if(generation!=atomicLoad(&${arena}[${h(JH.candidateGeneration)}])){
    ${p}Fail(${JFAULT.generation}u,row,${p}Invalid);return cursor;}
  atomicStore(&${arena}[${p}RowAt(row,0u)],${p}Invalid);
  atomicStore(&${arena}[${p}RowAt(row,1u)],${p}Invalid);
  atomicStore(&${arena}[${p}RowAt(row,2u)],0u);
  atomicStore(&${arena}[${p}RowAt(row,3u)],generation);
  atomicStore(&${arena}[${p}RowAt(row,4u)],0u);cursor.valid=1u;return cursor;
}
fn ${p}Record(cursor:ptr<function,${p}Cursor>,cell:u32,kind:u32)->bool{
  if((*cursor).valid==0u){return false;}
  if(cell>=${layout.cellCapacity}u||kind>3u){
    ${p}Fail(${JFAULT.invalidCell}u,(*cursor).row,cell);(*cursor).valid=0u;return false;}
  if((*cursor).chunk==${p}Invalid||(*cursor).used==${layout.readsPerChunk}u){
    let next=${p}AllocateChunk((*cursor).row,(*cursor).generation);
    if(next==${p}Invalid){(*cursor).valid=0u;return false;}
    if((*cursor).chunk==${p}Invalid){
      atomicStore(&${arena}[${p}RowAt((*cursor).row,0u)],next);
    }else{atomicStore(&${arena}[${p}ChunkAt((*cursor).chunk,0u)],next);}
    atomicStore(&${arena}[${p}RowAt((*cursor).row,1u)],next);
    (*cursor).chunk=next;(*cursor).used=0u;
  }
  let at=${p}ChunkAt((*cursor).chunk,4u+2u*(*cursor).used);
  atomicStore(&${arena}[at],cell);atomicStore(&${arena}[at+1u],kind);
  (*cursor).used+=1u;
  atomicStore(&${arena}[${p}ChunkAt((*cursor).chunk,1u)],(*cursor).used);
  atomicAdd(&${arena}[${p}RowAt((*cursor).row,2u)],1u);
  atomicAdd(&${arena}[${h(JH.rawReadCount)}],1u);return true;
}
fn ${p}SealRow(cursor:ptr<function,${p}Cursor>,resultGeneration:u32)->bool{
  if((*cursor).valid==0u||resultGeneration!=(*cursor).generation){
    ${p}Fail(${JFAULT.incompleteRow}u,(*cursor).row,${p}Invalid);return false;}
  atomicStore(&${arena}[${p}RowAt((*cursor).row,4u)],resultGeneration);
  atomicAdd(&${arena}[${h(JH.coveredRows)}],1u);(*cursor).valid=0u;return true;
}
@compute @workgroup_size(1)
fn sealSparseCM12VexDependencyJournal(){
  if(atomicLoad(&${arena}[${h(JH.phase)}])!=${JPHASE.collecting}u){
    ${p}Fail(${JFAULT.invalidPhase}u,${p}Invalid,${p}Invalid);return;}
  if(atomicLoad(&${arena}[${h(JH.coveredRows)}])
    !=atomicLoad(&${arena}[${h(JH.expectedRows)}])){
    ${p}Fail(${JFAULT.coverage}u,${p}Invalid,${p}Invalid);return;}
  atomicStore(&${arena}[${h(JH.phase)}],${JPHASE.sealed}u);
}
`;
}
