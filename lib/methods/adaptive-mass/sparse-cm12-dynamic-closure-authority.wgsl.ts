import {
  SPARSE_CM12_DYNAMIC_CLOSURE_FAULT,
  SPARSE_CM12_DYNAMIC_CLOSURE_HEADER,
  SPARSE_CM12_DYNAMIC_CLOSURE_HEADER_WORDS,
  SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT,
  SPARSE_CM12_DYNAMIC_CLOSURE_INVALID,
  SPARSE_CM12_DYNAMIC_CLOSURE_MAGIC,
  SPARSE_CM12_DYNAMIC_CLOSURE_VERSION,
  type SparseCM12DynamicClosureLayout,
} from "./sparse-cm12-dynamic-closure-authority";

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError(`${label} is not an identifier`);
  return value;
};

/**
 * Binding-free DCA1 source. Resident hook contract:
 * Every `packet` below is the stable TEI `leaf*64+localPacket` id. Compact
 * gather work ranks are never stored in DCA1.
 * - `<p>DynamicClosureGeneration() -> u32`
 * - `<p>DynamicClosureSurfaceMask(packet) -> vec2u`
 * - `<p>DynamicClosureDensityMask(packet) -> vec2u`
 * - `<p>DynamicClosureApplyTRA(packet, mask, worker16)`; calls
 *   `dca1MarkRowMask(cellPacket, axis, low, high)` for exact spatial row masks.
 * - `<p>DynamicClosureApplyVEX(packet, mask, worker16)`; calls
 *   `dca1MarkCellMask(cellPacket, low, high)`.
 * - `<p>DynamicClosureScatterGammaSnapshotRow(cellPacket, axis, lane)` resolves
 *   the stable row and scatters from destination density/gamma.
 * - `<p>DynamicClosureScatterGammaRefinementRow(cellPacket, axis, lane)` resolves
 *   the same stable row and scatters from the refinement input banks.
 *   Both hooks are direct consumers and must not decode through TRA1 stamps.
 * - `<p>DynamicClosureSeedVEXFrontier(cellPacket, mask, worker64)` seeds the
 *   packetized depth-zero VEX frontier. It must not call per-cell root records.
 */
export function createSparseCM12DynamicClosureAuthorityWGSL(options: Readonly<{
  layout: SparseCM12DynamicClosureLayout;
  arenaName?: string;
  hookPrefix?: string;
}>): string {
  const l = options.layout;
  const arena = identifier(options.arenaName ?? "topologyArena", "DCA1 arenaName");
  const p = identifier(options.hookPrefix ?? "cm12Resident", "DCA1 hookPrefix");
  const h = (word: number) => `${l.baseWords + word}u`;
  const indirect = (record: number, word: number) => `${l.indirectBaseWords + record + word}u`;
  return /* wgsl */ `
const DCA1_MAGIC:u32=0x${SPARSE_CM12_DYNAMIC_CLOSURE_MAGIC.toString(16)}u;
const DCA1_VERSION:u32=${SPARSE_CM12_DYNAMIC_CLOSURE_VERSION}u;
const DCA1_SOURCE_CAP:u32=${l.sourcePacketCapacity}u;
const DCA1_TARGET_CAP:u32=${l.targetPacketCapacity}u;
const DCA1_SURFACE_LIST:u32=${l.surfaceListBaseWords}u;
const DCA1_DENSITY_LIST:u32=${l.densityListBaseWords}u;
const DCA1_ROW_STAMP:u32=${l.rowStampBaseWords}u;
const DCA1_ROW_MASK:u32=${l.rowMaskBaseWords}u;
const DCA1_ROW_TOUCHED:u32=${l.rowTouchedBaseWords}u;
const DCA1_CELL_STAMP:u32=${l.cellStampBaseWords}u;
const DCA1_CELL_MASK:u32=${l.cellMaskBaseWords}u;
const DCA1_CELL_TOUCHED:u32=${l.cellTouchedBaseWords}u;
const DCA1_INVALID:u32=0xffffffffu;

fn dca1Fault(packet:u32,code:u32){
  let won=atomicCompareExchangeWeak(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.fault)}],0u,code);
  if(won.exchanged){atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.firstFaultPacket)}],packet);}
}
fn dca1HeaderValid()->bool{
  return arrayLength(&${arena})>=${l.totalWords}u
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.magic)}])==DCA1_MAGIC
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.version)}])==DCA1_VERSION
    &&atomicLoad(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.headerWords)}])
      ==${SPARSE_CM12_DYNAMIC_CLOSURE_HEADER_WORDS}u;
}
fn dca1Generation()->u32{return atomicLoad(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.generation)}]);}

@compute @workgroup_size(1)
fn beginSparseCM12DynamicClosure(){
  if(!dca1HeaderValid()){dca1Fault(DCA1_INVALID,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.invalidHeader}u);return;}
  let generation=${p}DynamicClosureGeneration();
  if(generation==0u||generation==DCA1_INVALID){
    dca1Fault(generation,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.invalidGeneration}u);return;
  }
  atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.generation)}],generation);
  atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.fault)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.firstFaultPacket)}],DCA1_INVALID);
  atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.surfaceSourceCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.densitySourceCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.rowTouchedCount)}],0u);
  atomicStore(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.cellTouchedCount)}],0u);
  for(var word=0u;word<${SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT.words}u;word+=1u){
    atomicStore(&${arena}[${l.indirectBaseWords}u+word],select(1u,0u,(word&3u)==0u));
  }
}

// These sparse cleanup passes run from the prior frame's replay indirects,
// before begin resets the touched counts. This avoids cross-workgroup spin
// locks: every future claim sees either zero or its own generation.
@compute @workgroup_size(256)
fn clearSparseCM12DynamicRows(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let rank=4u*wid.x+lane/64u;let worker=lane&63u;
  let count=min(DCA1_TARGET_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.rowTouchedCount)}]));
  if(rank>=count){return;}let packet=atomicLoad(&${arena}[DCA1_ROW_TOUCHED+rank]);
  if(worker<6u){atomicStore(&${arena}[DCA1_ROW_MASK+worker*DCA1_TARGET_CAP+packet],0u);}
  if(worker==0u){atomicStore(&${arena}[DCA1_ROW_STAMP+packet],0u);}
}
@compute @workgroup_size(256)
fn clearSparseCM12DynamicCells(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let rank=4u*wid.x+lane/64u;let worker=lane&63u;
  let count=min(DCA1_TARGET_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.cellTouchedCount)}]));
  if(rank>=count){return;}let packet=atomicLoad(&${arena}[DCA1_CELL_TOUCHED+rank]);
  if(worker<2u){atomicStore(&${arena}[DCA1_CELL_MASK+worker*DCA1_TARGET_CAP+packet],0u);}
  if(worker==0u){atomicStore(&${arena}[DCA1_CELL_STAMP+packet],0u);}
}

// Called once by lane zero after the gather workgroup has formed both ballots.
fn cm12DynamicClosurePublishSourcePacket(packet:u32,surface:vec2u,density:vec2u){
  if((surface.x|surface.y)!=0u){
    let rank=atomicAdd(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.surfaceSourceCount)}],1u);
    if(rank<DCA1_SOURCE_CAP){atomicStore(&${arena}[DCA1_SURFACE_LIST+rank],packet);}
    else{dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.sourceOverflow}u);}
  }
  if((density.x|density.y)!=0u){
    let rank=atomicAdd(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.densitySourceCount)}],1u);
    if(rank<DCA1_SOURCE_CAP){atomicStore(&${arena}[DCA1_DENSITY_LIST+rank],packet);}
    else{dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.sourceOverflow}u);}
  }
}

@compute @workgroup_size(1)
fn sealSparseCM12DynamicClosureSources(){
  let surface=min(DCA1_SOURCE_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.surfaceSourceCount)}]));
  let density=min(DCA1_SOURCE_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.densitySourceCount)}]));
  atomicStore(&${arena}[${indirect(SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT.traCompile, 0)}],(surface+3u)/4u);
  atomicStore(&${arena}[${indirect(SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT.vexCompile, 0)}],(density+3u)/4u);
}

fn dca1ClaimRow(packet:u32)->bool{
  if(packet>=DCA1_TARGET_CAP){dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.rowTargetOverflow}u);return false;}
  let generation=dca1Generation();
  loop{
    let stamp=atomicLoad(&${arena}[DCA1_ROW_STAMP+packet]);
    if(stamp==generation){return true;}
    if(stamp!=0u){dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.targetNotCleared}u);return false;}
    let claim=atomicCompareExchangeWeak(&${arena}[DCA1_ROW_STAMP+packet],0u,generation);
    if(!claim.exchanged){continue;}
    let rank=atomicAdd(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.rowTouchedCount)}],1u);
    if(rank<DCA1_TARGET_CAP){atomicStore(&${arena}[DCA1_ROW_TOUCHED+rank],packet);}
    else{dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.rowTargetOverflow}u);}
    return rank<DCA1_TARGET_CAP;
  }
}
fn dca1MarkRowMask(packet:u32,axis:u32,low:u32,high:u32){
  if((low|high)==0u||axis>=3u||!dca1ClaimRow(packet)){return;}
  _=atomicOr(&${arena}[DCA1_ROW_MASK+(2u*axis)*DCA1_TARGET_CAP+packet],low);
  _=atomicOr(&${arena}[DCA1_ROW_MASK+(2u*axis+1u)*DCA1_TARGET_CAP+packet],high);
}
fn dca1ClaimCell(packet:u32)->bool{
  if(packet>=DCA1_TARGET_CAP){dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.cellTargetOverflow}u);return false;}
  let generation=dca1Generation();
  loop{
    let stamp=atomicLoad(&${arena}[DCA1_CELL_STAMP+packet]);
    if(stamp==generation){return true;}
    if(stamp!=0u){dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.targetNotCleared}u);return false;}
    let claim=atomicCompareExchangeWeak(&${arena}[DCA1_CELL_STAMP+packet],0u,generation);
    if(!claim.exchanged){continue;}
    let rank=atomicAdd(&${arena}[${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.cellTouchedCount)}],1u);
    if(rank<DCA1_TARGET_CAP){atomicStore(&${arena}[DCA1_CELL_TOUCHED+rank],packet);}
    else{dca1Fault(packet,${SPARSE_CM12_DYNAMIC_CLOSURE_FAULT.cellTargetOverflow}u);}
    return rank<DCA1_TARGET_CAP;
  }
}
fn dca1MarkCellMask(packet:u32,low:u32,high:u32){
  if((low|high)==0u||!dca1ClaimCell(packet)){return;}
  _=atomicOr(&${arena}[DCA1_CELL_MASK+packet],low);
  _=atomicOr(&${arena}[DCA1_CELL_MASK+DCA1_TARGET_CAP+packet],high);
}

@compute @workgroup_size(64)
fn compileSparseCM12DynamicTRA(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let rank=4u*wid.x+lane/16u;
  let count=min(DCA1_SOURCE_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.surfaceSourceCount)}]));
  if(rank>=count){return;}let packet=atomicLoad(&${arena}[DCA1_SURFACE_LIST+rank]);
  ${p}DynamicClosureApplyTRA(packet,${p}DynamicClosureSurfaceMask(packet),lane&15u);
}
@compute @workgroup_size(64)
fn compileSparseCM12DynamicVEX(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let rank=4u*wid.x+lane/16u;
  let count=min(DCA1_SOURCE_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.densitySourceCount)}]));
  if(rank>=count){return;}let packet=atomicLoad(&${arena}[DCA1_DENSITY_LIST+rank]);
  ${p}DynamicClosureApplyVEX(packet,${p}DynamicClosureDensityMask(packet),lane&15u);
}

@compute @workgroup_size(1)
fn sealSparseCM12DynamicClosureTargets(){
  let rows=min(DCA1_TARGET_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.rowTouchedCount)}]));
  let cells=min(DCA1_TARGET_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.cellTouchedCount)}]));
  atomicStore(&${arena}[${indirect(SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT.rowScatter, 0)}],(rows+3u)/4u);
  atomicStore(&${arena}[${indirect(SPARSE_CM12_DYNAMIC_CLOSURE_INDIRECT.vexSeed, 0)}],(cells+3u)/4u);
}

// The row masks are read-only here and deliberately survive both phases. The
// host orders snapshot scatter/finalize before refinement scatter/finalize;
// clearSparseCM12DynamicRows reclaims the masks only at the next frame start.
var<workgroup>dca1ScatterCount:u32;
var<workgroup>dca1ScatterPackets:array<u32,4>;
var<workgroup>dca1ScatterMasks:array<u32,24>;
@compute @workgroup_size(256)
fn scatterSparseCM12DynamicGammaSnapshotRows(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let group=lane/64u;let rank=4u*wid.x+group;let cellLane=lane&63u;
  if(lane==0u){dca1ScatterCount=min(DCA1_TARGET_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.rowTouchedCount)}]));}
  workgroupBarrier();let validRank=rank<dca1ScatterCount;
  if(validRank&&cellLane==0u){dca1ScatterPackets[group]=atomicLoad(
    &${arena}[DCA1_ROW_TOUCHED+rank]);}workgroupBarrier();
  let packet=dca1ScatterPackets[group];
  if(validRank&&cellLane<6u){dca1ScatterMasks[6u*group+cellLane]=atomicLoad(
    &${arena}[DCA1_ROW_MASK+cellLane*DCA1_TARGET_CAP+packet]);}
  workgroupBarrier();if(!validRank){return;}
  for(var axis=0u;axis<3u;axis+=1u){
    let low=dca1ScatterMasks[6u*group+2u*axis];
    let high=dca1ScatterMasks[6u*group+2u*axis+1u];
    if(((select(low,high,cellLane>=32u)>>(cellLane&31u))&1u)!=0u){
      ${p}DynamicClosureScatterGammaSnapshotRow(packet,axis,cellLane);
    }
  }
}
@compute @workgroup_size(256)
fn scatterSparseCM12DynamicGammaRefinementRows(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let group=lane/64u;let rank=4u*wid.x+group;let cellLane=lane&63u;
  if(lane==0u){dca1ScatterCount=min(DCA1_TARGET_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.rowTouchedCount)}]));}
  workgroupBarrier();let validRank=rank<dca1ScatterCount;
  if(validRank&&cellLane==0u){dca1ScatterPackets[group]=atomicLoad(
    &${arena}[DCA1_ROW_TOUCHED+rank]);}workgroupBarrier();
  let packet=dca1ScatterPackets[group];
  if(validRank&&cellLane<6u){dca1ScatterMasks[6u*group+cellLane]=atomicLoad(
    &${arena}[DCA1_ROW_MASK+cellLane*DCA1_TARGET_CAP+packet]);}
  workgroupBarrier();if(!validRank){return;}
  for(var axis=0u;axis<3u;axis+=1u){
    let low=dca1ScatterMasks[6u*group+2u*axis];
    let high=dca1ScatterMasks[6u*group+2u*axis+1u];
    if(((select(low,high,cellLane>=32u)>>(cellLane&31u))&1u)!=0u){
      ${p}DynamicClosureScatterGammaRefinementRow(packet,axis,cellLane);
    }
  }
}
@compute @workgroup_size(256)
fn seedSparseCM12DynamicVEXFrontier(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let rank=4u*wid.x+lane/64u;let worker=lane&63u;
  let count=min(DCA1_TARGET_CAP,atomicLoad(&${arena}[
    ${h(SPARSE_CM12_DYNAMIC_CLOSURE_HEADER.cellTouchedCount)}]));
  if(rank>=count){return;}let packet=atomicLoad(&${arena}[DCA1_CELL_TOUCHED+rank]);
  let low=atomicLoad(&${arena}[DCA1_CELL_MASK+packet]);
  let high=atomicLoad(&${arena}[DCA1_CELL_MASK+DCA1_TARGET_CAP+packet]);
  ${p}DynamicClosureSeedVEXFrontier(packet,vec2u(low,high),worker);
}
`;
}
