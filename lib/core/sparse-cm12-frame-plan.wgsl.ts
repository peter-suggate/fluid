import {
  SPARSE_CM12_FRAME_PLAN_BRICK,
  SPARSE_CM12_FRAME_PLAN_BRICK_FLAG,
  SPARSE_CM12_FRAME_PLAN_BRICK_WORDS,
  SPARSE_CM12_FRAME_PLAN_FAULT,
  SPARSE_CM12_FRAME_PLAN_FLAG,
  SPARSE_CM12_FRAME_PLAN_HEADER,
  SPARSE_CM12_FRAME_PLAN_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_INVALID,
  SPARSE_CM12_FRAME_PLAN_MAGIC,
  SPARSE_CM12_FRAME_PLAN_SLOT,
  SPARSE_CM12_FRAME_PLAN_SLOT_FLAG,
  SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS,
  SPARSE_CM12_FRAME_PLAN_STAGE_COUNT,
  SPARSE_CM12_FRAME_PLAN_TILE,
  SPARSE_CM12_FRAME_PLAN_TILE_EDGE,
  SPARSE_CM12_FRAME_PLAN_TILE_WORDS,
  SPARSE_CM12_FRAME_PLAN_VERSION,
  type SparseCM12FramePlanLayout,
} from "./sparse-cm12-frame-plan";

export interface SparseCM12FramePlanWGSLOptions {
  readonly layout: SparseCM12FramePlanLayout;
  /** Existing array<atomic<u32>> storage binding. */
  readonly arenaName?: string;
  readonly prefix?: string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/** Binding-free FPL1 scheduler and native overlay-access library. */
export function createSparseCM12FramePlanWGSL(
  options: SparseCM12FramePlanWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "framePlan", "arenaName");
  const p = identifier(options.prefix ?? "cm12FramePlan", "prefix");
  const h = (word: number) => `${layout.baseWords + word}u`;
  const defaultPacketLow = Array.from({ length: SPARSE_CM12_FRAME_PLAN_STAGE_COUNT },
    (_, stage) => (stage % layout.packetCount) << (5 * stage))
    .reduce((sum, value) => sum | value, 0) >>> 0;
  return /* wgsl */ `
const ${p}Invalid:u32=0x${SPARSE_CM12_FRAME_PLAN_INVALID.toString(16)}u;
const ${p}Magic:u32=0x${SPARSE_CM12_FRAME_PLAN_MAGIC.toString(16)}u;
const ${p}Version:u32=${SPARSE_CM12_FRAME_PLAN_VERSION}u;
const ${p}StageCount:u32=${SPARSE_CM12_FRAME_PLAN_STAGE_COUNT}u;
const ${p}PacketCount:u32=${layout.packetCount}u;
const ${p}BrickCapacity:u32=${layout.brickCapacity}u;
const ${p}TilesPerBrick:u32=${layout.tilesPerBrick}u;
const ${p}SlotWords:u32=${layout.slotWords}u;
const ${p}BrickOffset:u32=${layout.brickBaseOffsetWords}u;
const ${p}TileOffset:u32=${layout.tileBaseOffsetWords}u;
const ${p}StageListOffset:u32=${layout.stageListBaseOffsetWords}u;

fn ${p}Load(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn ${p}Store(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn ${p}SlotBase(slot:u32)->u32{
  return select(${layout.slot0BaseWords}u,${layout.slot1BaseWords}u,slot==1u);
}
fn ${p}BrickAt(slot:u32,brick:u32)->u32{
  return ${p}SlotBase(slot)+${p}BrickOffset+${SPARSE_CM12_FRAME_PLAN_BRICK_WORDS}u*brick;
}
fn ${p}TileAt(slot:u32,brick:u32,tile:u32)->u32{
  return ${p}SlotBase(slot)+${p}TileOffset+${SPARSE_CM12_FRAME_PLAN_TILE_WORDS}u
    *(brick*${p}TilesPerBrick+tile);
}
fn ${p}PacketAt(slot:u32,packet:u32)->u32{
  return ${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.stagePackets}u+4u*packet;
}
fn ${p}PacketListAt(slot:u32,packet:u32,brick:u32)->u32{
  return ${p}SlotBase(slot)+${p}StageListOffset+packet*${p}BrickCapacity+brick;
}
fn ${p}CurrentSlot()->u32{return ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.currentSlot)});}
fn ${p}NextSlot()->u32{return 1u-${p}CurrentSlot();}
fn ${p}CandidateGeneration()->u32{
  return ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.candidateGeneration)});
}
fn ${p}HeaderValid()->bool{
  return arrayLength(&${arena})>=${layout.totalWords}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.magic)})==${p}Magic
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.version)})==${p}Version
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.headerWords)})
      ==${SPARSE_CM12_FRAME_PLAN_HEADER_WORDS}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.slotHeaderWords)})
      ==${SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.brickWords)})
      ==${SPARSE_CM12_FRAME_PLAN_BRICK_WORDS}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.tileWords)})
      ==${SPARSE_CM12_FRAME_PLAN_TILE_WORDS}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.stageCount)})==${p}StageCount
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.tileEdge)})
      ==${SPARSE_CM12_FRAME_PLAN_TILE_EDGE}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.brickCapacity)})==${p}BrickCapacity
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.tilesPerBrick)})==${p}TilesPerBrick
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.packetCount)})==${p}PacketCount
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.indirectWords)})
      ==${3 * layout.packetCount}u
    &&${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.totalWords)})==${layout.totalWords}u;
}
fn ${p}GlobalFail(code:u32,brick:u32,tile:u32,stage:u32){
  let won=atomicCompareExchangeWeak(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_HEADER.faultCode)}],
    ${SPARSE_CM12_FRAME_PLAN_FAULT.none}u,code).exchanged;
  atomicOr(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)}],
    ${SPARSE_CM12_FRAME_PLAN_FLAG.globalFault}u);
  if(won){
    ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.firstFaultBrick)},brick);
    ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.firstFaultTile)},tile);
    ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.firstFaultStage)},stage);
  }
}
fn ${p}BrickFail(slot:u32,brick:u32,tile:u32,stage:u32,code:u32){
  if(brick>=${p}BrickCapacity){${p}GlobalFail(code,brick,tile,stage);return;}
  let record=${p}BrickAt(slot,brick);
  let won=atomicCompareExchangeWeak(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_BRICK.faultCode}u],
    ${SPARSE_CM12_FRAME_PLAN_FAULT.none}u,code).exchanged;
  atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_BRICK.flags}u],
    ${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.localFault}u);
  if(won){
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.firstFaultTile}u,tile);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.firstFaultStage}u,stage);
    atomicCompareExchangeWeak(&${arena}[
      ${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.firstFaultBrick}u],
      ${p}Invalid,brick);
    atomicAdd(&${arena}[${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.localFaultBrickCount}u],1u);
    atomicOr(&${arena}[${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.flags}u],
      ${SPARSE_CM12_FRAME_PLAN_SLOT_FLAG.localFaults}u);
    atomicOr(&${arena}[${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)}],
      ${SPARSE_CM12_FRAME_PLAN_FLAG.localFaults}u);
  }
}
fn ${p}SetStageMask(record:u32,stage:u32,tile:u32){
  let at=record+${SPARSE_CM12_FRAME_PLAN_BRICK.stageMasks}u+2u*stage;
  atomicOr(&${arena}[at+select(0u,1u,tile>=32u)],1u<<(tile&31u));
}
fn ${p}StageMask(record:u32,stage:u32)->vec2u{
  let at=record+${SPARSE_CM12_FRAME_PLAN_BRICK.stageMasks}u+2u*stage;
  return vec2u(${p}Load(at),${p}Load(at+1u));
}
fn ${p}MaskContains(mask:vec2u,tile:u32)->bool{
  return (mask[select(0u,1u,tile>=32u)]&(1u<<(tile&31u)))!=0u;
}
fn ${p}PackedStageMask(record:u32,shift:u32)->u32{
  return (${p}Load(record+${SPARSE_CM12_FRAME_PLAN_TILE.packedStageMasks}u)>>shift)&63u;
}
fn ${p}BrickStagePacket(record:u32,stage:u32)->u32{
  return (${p}Load(record+${SPARSE_CM12_FRAME_PLAN_BRICK.packedPacketIndicesLow}u)
    >>(5u*stage))&31u;
}
fn ${p}MergePackedDepths(at:u32,packedDepths:u32){
  for(var stage=0u;stage<${p}StageCount;stage+=1u){
    let shift=4u*stage;let depth=(packedDepths>>shift)&15u;
    loop{
      let old=${p}Load(at);let previous=(old>>shift)&15u;if(depth<=previous){break;}
      let desired=(old&~(15u<<shift))|(depth<<shift);
      if(atomicCompareExchangeWeak(&${arena}[at],old,desired).exchanged){break;}
    }
  }
}
fn ${p}SetNextTopologyGeneration(topologyGeneration:u32){
  let flags=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)});
  if((flags&${SPARSE_CM12_FRAME_PLAN_FLAG.candidateOpen}u)==0u){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidTransition}u,${p}Invalid,
      ${p}Invalid,${p}Invalid);return;
  }
  let slot=${p}NextSlot();
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.candidateTopologyGeneration)},
    topologyGeneration);
  ${p}Store(${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.topologyGeneration}u,
    topologyGeneration);
}
fn ${p}SetNextFrameAuthority(frameGeneration:u32,parity:u32){
  let flags=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)});
  let slot=${p}NextSlot();
  if((flags&${SPARSE_CM12_FRAME_PLAN_FLAG.candidateOpen}u)==0u
    ||frameGeneration>=0x40000000u||parity>1u){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidTransition}u,${p}Invalid,
      ${p}Invalid,${p}Invalid);return;
  }
  // Bit 31 distinguishes the valid generation-zero bootstrap from an omitted
  // authority write. Bits 1..30 hold the physics-frame generation; bit 0 is
  // the accepted scalar parity.
  ${p}Store(${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.reserved}u,
    0x80000000u|(frameGeneration<<1u)|parity);
}
fn ${p}SetNextBrickLogicalKey(brick:u32,logicalBrickKey:u32){
  let slot=${p}NextSlot();
  if(brick>=${p}BrickCapacity){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidBrick}u,brick,
      ${p}Invalid,${p}Invalid);return;
  }
  let record=${p}BrickAt(slot,brick);
  if(${p}Load(record+${SPARSE_CM12_FRAME_PLAN_BRICK.generation}u)
    !=${p}CandidateGeneration()){
    ${p}BrickFail(slot,brick,${p}Invalid,${p}Invalid,
      ${SPARSE_CM12_FRAME_PLAN_FAULT.localGeneration}u);return;
  }
  ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.logicalBrickKey}u,logicalBrickKey);
}

@compute @workgroup_size(1)
fn beginSparseCM12FramePlanNext(){
  if(!${p}HeaderValid()){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidHeader}u,${p}Invalid,
      ${p}Invalid,${p}Invalid);return;
  }
  let flags=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)});
  if((flags&${SPARSE_CM12_FRAME_PLAN_FLAG.globalFault}u)!=0u){return;}
  if((flags&${SPARSE_CM12_FRAME_PLAN_FLAG.candidateOpen}u)!=0u){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidTransition}u,${p}Invalid,
      ${p}Invalid,${p}Invalid);return;
  }
  let bootstrapping=(flags&${SPARSE_CM12_FRAME_PLAN_FLAG.accepted}u)==0u;
  let accepted=${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedGeneration)});
  if(!bootstrapping&&accepted>=0xfffffffdu){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.generationExhausted}u,${p}Invalid,
      ${p}Invalid,${p}Invalid);return;
  }
  // Construction accepts generation/parity zero. The first physics frame then
  // advances to generation/parity one, matching the resident A->B update. A
  // synthetic generation-one bootstrap would phase-shift every field bank.
  let generation=select(accepted+1u,0u,bootstrapping);
  let slot=${p}NextSlot();let base=${p}SlotBase(slot);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.candidateGeneration)},generation);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.candidateTopologyGeneration)},
    ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedTopologyGeneration)}));
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)},
    ${SPARSE_CM12_FRAME_PLAN_FLAG.initialized | SPARSE_CM12_FRAME_PLAN_FLAG.candidateOpen}u);
  ${p}Store(base+${SPARSE_CM12_FRAME_PLAN_SLOT.generation}u,generation);
  ${p}Store(base+${SPARSE_CM12_FRAME_PLAN_SLOT.topologyGeneration}u,
    ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.candidateTopologyGeneration)}));
  ${p}Store(base+${SPARSE_CM12_FRAME_PLAN_SLOT.flags}u,
    ${SPARSE_CM12_FRAME_PLAN_SLOT_FLAG.open}u);
  for(var word=${SPARSE_CM12_FRAME_PLAN_SLOT.localFaultBrickCount}u;
      word<${SPARSE_CM12_FRAME_PLAN_SLOT_HEADER_WORDS}u;word+=1u){${p}Store(base+word,0u);}
  ${p}Store(base+${SPARSE_CM12_FRAME_PLAN_SLOT.firstFaultBrick}u,${p}Invalid);
  for(var packetIndex=0u;packetIndex<${p}PacketCount;packetIndex+=1u){
    let packet=${p}PacketAt(slot,packetIndex);
    ${p}Store(packet+2u,1u);${p}Store(packet+3u,1u);
  }
}

@compute @workgroup_size(64)
fn initializeSparseCM12FramePlanNext(
  @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  let brick=wid.x;if(brick>=${p}BrickCapacity){return;}
  let slot=${p}NextSlot();let generation=${p}CandidateGeneration();
  let record=${p}BrickAt(slot,brick);
  if(lid.x==0u){
    for(var word=0u;word<${SPARSE_CM12_FRAME_PLAN_BRICK_WORDS}u;word+=1u){
      ${p}Store(record+word,0u);
    }
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.generation}u,generation);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.topologyGeneration}u,
      ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.candidateTopologyGeneration)}));
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.flags}u,
      ${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.initialized}u);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.logicalBrickKey}u,brick);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.packedPacketIndicesLow}u,
      ${defaultPacketLow}u);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.producerGeneration}u,generation);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.packetEpoch}u,generation);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.firstFaultTile}u,${p}Invalid);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.firstFaultStage}u,${p}Invalid);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskLow}u,
      ${layout.tilesPerBrick >= 32 ? 0xffff_ffff : (2 ** layout.tilesPerBrick - 1) >>> 0}u);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.validTileMaskHigh}u,
      ${layout.tilesPerBrick === 64 ? 0xffff_ffff : 0}u);
    for(var packet=0u;packet<${p}PacketCount;packet+=1u){
      ${p}Store(${p}PacketListAt(slot,packet,brick),${p}Invalid);
    }
  }
  if(lid.x<${p}TilesPerBrick){
    let tile=${p}TileAt(slot,brick,lid.x);
    ${p}Store(tile+${SPARSE_CM12_FRAME_PLAN_TILE.generation}u,generation);
    ${p}Store(tile+${SPARSE_CM12_FRAME_PLAN_TILE.packedStageMasks}u,0u);
    ${p}Store(tile+${SPARSE_CM12_FRAME_PLAN_TILE.packedCauseMasks}u,0u);
    ${p}Store(tile+${SPARSE_CM12_FRAME_PLAN_TILE.packedClosureDepths}u,0u);
  }
}

// Producer API. Exactly one lane owns (brick,tile); cross-brick producers OR a
// bounded brick mask and the target brick's lane resolves it in the fixed
// closure pass below. There is no per-tile append or global event journal.
fn ${p}MarkOwnedNextTile(brick:u32,tile:u32,directStages:u32,closureStages:u32,
  originCauses:u32,inheritedCauses:u32,packedDepths:u32){
  let slot=${p}NextSlot();let generation=${p}CandidateGeneration();
  if(brick>=${p}BrickCapacity){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidBrick}u,brick,tile,${p}Invalid);return;
  }
  if(tile>=${p}TilesPerBrick){
    ${p}BrickFail(slot,brick,tile,${p}Invalid,
      ${SPARSE_CM12_FRAME_PLAN_FAULT.invalidTile}u);return;
  }
  if(((directStages|closureStages)&~63u)!=0u){
    ${p}BrickFail(slot,brick,tile,${p}Invalid,
      ${SPARSE_CM12_FRAME_PLAN_FAULT.invalidStage}u);return;
  }
  let brickRecord=${p}BrickAt(slot,brick);let tileRecord=${p}TileAt(slot,brick,tile);
  if(${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.generation}u)!=generation
    ||${p}Load(tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.generation}u)!=generation){
    ${p}BrickFail(slot,brick,tile,${p}Invalid,
      ${SPARSE_CM12_FRAME_PLAN_FAULT.localGeneration}u);return;
  }
  let masks=(directStages&63u)|((closureStages&63u)<<6u);
  atomicOr(&${arena}[tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.packedStageMasks}u],masks);
  atomicOr(&${arena}[tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.packedCauseMasks}u],
    (originCauses&0xffffu)|((inheritedCauses&0xffffu)<<16u));
  ${p}MergePackedDepths(
    tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.packedClosureDepths}u,packedDepths);
  atomicOr(&${arena}[brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.causeMask}u],
    originCauses|inheritedCauses);
  ${p}MergePackedDepths(
    brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.packedClosureDepths}u,packedDepths);
  let stages=directStages|closureStages;
  for(var stage=0u;stage<${p}StageCount;stage+=1u){
    if((stages&(1u<<stage))!=0u){${p}SetStageMask(brickRecord,stage,tile);}
  }
}
fn ${p}RequestNextClosureMask(brick:u32,stage:u32,mask:vec2u,
  inheritedCause:u32,depth:u32){
  let slot=${p}NextSlot();
  if(brick>=${p}BrickCapacity||stage>=${p}StageCount||depth>15u){
    ${p}BrickFail(slot,brick,${p}Invalid,stage,
      select(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidStage}u,
        ${SPARSE_CM12_FRAME_PLAN_FAULT.invalidTile}u,depth>15u));return;
  }
  let record=${p}BrickAt(slot,brick);
  let at=record+${SPARSE_CM12_FRAME_PLAN_BRICK.stageMasks}u+2u*stage;
  atomicOr(&${arena}[at],mask.x);atomicOr(&${arena}[at+1u],mask.y);
  atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_BRICK.causeMask}u],inheritedCause);
  let shift=4u*stage;
  loop{
    let old=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_BRICK.packedClosureDepths}u);
    let previous=(old>>shift)&15u;if(depth<=previous){break;}
    let desired=(old&~(15u<<shift))|(depth<<shift);
    if(atomicCompareExchangeWeak(&${arena}[
      record+${SPARSE_CM12_FRAME_PLAN_BRICK.packedClosureDepths}u],old,desired).exchanged){break;}
  }
}
fn ${p}SetNextBrickPacket(brick:u32,stage:u32,packet:u32){
  let slot=${p}NextSlot();
  if(brick>=${p}BrickCapacity||stage>=${p}StageCount
    ||packet>=${layout.packetCount}u||packet>=32u){
    ${p}BrickFail(slot,brick,${p}Invalid,stage,
      ${SPARSE_CM12_FRAME_PLAN_FAULT.invalidPacket}u);return;
  }
  let record=${p}BrickAt(slot,brick);let shift=5u*stage;
  let at=record+${SPARSE_CM12_FRAME_PLAN_BRICK.packedPacketIndicesLow}u;
  loop{
    let old=${p}Load(at);let desired=(old&~(31u<<shift))|(packet<<shift);
    if(atomicCompareExchangeWeak(&${arena}[at],old,desired).exchanged){break;}
  }
}

@compute @workgroup_size(64)
fn resolveSparseCM12FramePlanNextClosure(
  @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  let brick=wid.x;if(brick>=${p}BrickCapacity||lid.x>=${p}TilesPerBrick){return;}
  let slot=${p}NextSlot();let record=${p}BrickAt(slot,brick);
  let causes=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_BRICK.causeMask}u);
  let depths=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_BRICK.packedClosureDepths}u);
  var closure=0u;
  let tileRecord=${p}TileAt(slot,brick,lid.x);
  let direct=${p}PackedStageMask(tileRecord,0u);
  for(var stage=0u;stage<${p}StageCount;stage+=1u){
    if(${p}MaskContains(${p}StageMask(record,stage),lid.x)
      &&(direct&(1u<<stage))==0u){closure|=1u<<stage;}
  }
  if(closure!=0u){${p}MarkOwnedNextTile(brick,lid.x,0u,closure,0u,causes,depths);}
}

var<workgroup> ${p}SealFault:atomic<u32>;
@compute @workgroup_size(64)
fn sealSparseCM12FramePlanNextBricks(
  @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  let brick=wid.x;if(brick>=${p}BrickCapacity){return;}
  let slot=${p}NextSlot();let generation=${p}CandidateGeneration();
  let record=${p}BrickAt(slot,brick);
  if(lid.x==0u){atomicStore(&${p}SealFault,0u);}workgroupBarrier();
  if(lid.x<${p}TilesPerBrick){
    let tile=${p}TileAt(slot,brick,lid.x);
    if(${p}Load(tile+${SPARSE_CM12_FRAME_PLAN_TILE.generation}u)!=generation){
      atomicOr(&${p}SealFault,1u);
    }
    let scheduled=${p}PackedStageMask(tile,0u)|${p}PackedStageMask(tile,6u);
    for(var stage=0u;stage<${p}StageCount;stage+=1u){
      let listed=${p}MaskContains(${p}StageMask(record,stage),lid.x);
      if(listed!=((scheduled&(1u<<stage))!=0u)){atomicOr(&${p}SealFault,2u);}
    }
    if(${p}PackedStageMask(tile,24u)!=0u){atomicOr(&${p}SealFault,4u);}
  }
  workgroupBarrier();
  if(lid.x==0u){
    let priorFlags=${p}Load(record+${SPARSE_CM12_FRAME_PLAN_BRICK.flags}u);
    if((priorFlags&${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.sealed}u)!=0u){
      atomicOr(&${p}SealFault,8u);
    }
    let localFault=atomicLoad(&${p}SealFault);
    if(localFault!=0u||(priorFlags&${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.localFault}u)!=0u){
      ${p}BrickFail(slot,brick,${p}Invalid,${p}Invalid,
        ${SPARSE_CM12_FRAME_PLAN_FAULT.missingCoverage}u);
      for(var packet=0u;packet<${p}PacketCount;packet+=1u){
        ${p}Store(${p}PacketListAt(slot,packet,brick),${p}Invalid);
      }
      atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_BRICK.flags}u],
        ${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.sealed}u);
      atomicAdd(&${arena}[
        ${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.sealedBrickCount}u],1u);
      return;
    }
    var hasWork=false;var expected=0u;
    for(var stage=0u;stage<${p}StageCount;stage+=1u){
      let mask=${p}StageMask(record,stage);if((mask.x|mask.y)!=0u){expected|=1u<<stage;}
    }
    for(var packet=0u;packet<${p}PacketCount;packet+=1u){
      var scheduled=false;
      for(var stage=0u;stage<${p}StageCount;stage+=1u){
        let mask=${p}StageMask(record,stage);
        scheduled=scheduled||((mask.x|mask.y)!=0u&&${p}BrickStagePacket(record,stage)==packet);
      }
      ${p}Store(${p}PacketListAt(slot,packet,brick),select(${p}Invalid,brick,scheduled));
      if(scheduled){atomicAdd(&${arena}[${p}PacketAt(slot,packet)],1u);hasWork=true;}
    }
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.expectedStageMask}u,expected);
    ${p}Store(record+${SPARSE_CM12_FRAME_PLAN_BRICK.flags}u,
      ${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.initialized
        | SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.sealed
        | SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.coverageComplete}u
      |select(0u,${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.hasWork}u,hasWork));
    atomicAdd(&${arena}[${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.sealedBrickCount}u],1u);
  }
}

@compute @workgroup_size(1)
fn finalizeSparseCM12FramePlanNext(){
  let slot=${p}NextSlot();let base=${p}SlotBase(slot);
  let generation=${p}CandidateGeneration();
  if(${p}Load(base+${SPARSE_CM12_FRAME_PLAN_SLOT.sealedBrickCount}u)!=${p}BrickCapacity){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.missingCoverage}u,${p}Invalid,
      ${p}Invalid,${p}Invalid);return;
  }
  if((${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)})
    &${SPARSE_CM12_FRAME_PLAN_FLAG.globalFault}u)!=0u){return;}
  for(var packetIndex=0u;packetIndex<${p}PacketCount;packetIndex+=1u){
    let packet=${p}PacketAt(slot,packetIndex);
    // The deterministic list has one slot per brick. Dirty count remains word
    // zero for census/overlay; dispatch x spans brickCapacity and clean entries
    // return before touching physics.
    ${p}Store(packet+1u,${p}BrickCapacity);
    ${p}Store(packet+2u,1u);${p}Store(packet+3u,1u);
    ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.fixedPacketCounts)}+packetIndex,
      ${p}Load(packet));
    let indirect=${h(SPARSE_CM12_FRAME_PLAN_HEADER.fixedIndirectArguments)}+3u*packetIndex;
    ${p}Store(indirect,${p}BrickCapacity);
    ${p}Store(indirect+1u,1u);${p}Store(indirect+2u,1u);
  }
  let localFaults=${p}Load(base+${SPARSE_CM12_FRAME_PLAN_SLOT.localFaultBrickCount}u)!=0u;
  let frameAuthority=${p}Load(base+${SPARSE_CM12_FRAME_PLAN_SLOT.reserved}u);
  if((frameAuthority&0x80000000u)==0u){
    ${p}GlobalFail(${SPARSE_CM12_FRAME_PLAN_FAULT.invalidTransition}u,${p}Invalid,
      ${p}Invalid,${p}Invalid);return;
  }
  ${p}Store(base+${SPARSE_CM12_FRAME_PLAN_SLOT.flags}u,
    ${SPARSE_CM12_FRAME_PLAN_SLOT_FLAG.brickSealed
      | SPARSE_CM12_FRAME_PLAN_SLOT_FLAG.indirectSealed
      | SPARSE_CM12_FRAME_PLAN_SLOT_FLAG.accepted}u
    |select(0u,${SPARSE_CM12_FRAME_PLAN_SLOT_FLAG.localFaults}u,localFaults));
  let old=${p}CurrentSlot();
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.previousSlot)},old);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.currentSlot)},slot);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedGeneration)},generation);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedFrameGeneration)},
    (frameAuthority&0x7fffffffu)>>1u);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedParity)},frameAuthority&1u);
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedTopologyGeneration)},
    ${p}Load(base+${SPARSE_CM12_FRAME_PLAN_SLOT.topologyGeneration}u));
  ${p}Store(${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)},
    ${SPARSE_CM12_FRAME_PLAN_FLAG.initialized | SPARSE_CM12_FRAME_PLAN_FLAG.accepted}u
    |select(0u,${SPARSE_CM12_FRAME_PLAN_FLAG.localFaults}u,localFaults));
}

fn ${p}AcceptedFrameGeneration()->u32{
  return ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedFrameGeneration)});
}
fn ${p}AcceptedParity()->u32{
  return ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedParity)});
}

fn ${p}CurrentPacketBrickInvocation(packet:u32,invocation:u32)->u32{
  if(packet>=${p}PacketCount||invocation>=${p}BrickCapacity){return ${p}Invalid;}
  return ${p}Load(${p}PacketListAt(${p}CurrentSlot(),packet,invocation));
}
fn ${p}CurrentStageInPacket(brick:u32,stage:u32,packet:u32)->bool{
  if(brick>=${p}BrickCapacity||stage>=${p}StageCount||packet>=${p}PacketCount){return false;}
  let record=${p}BrickAt(${p}CurrentSlot(),brick);
  return ${p}BrickStagePacket(record,stage)==packet;
}
fn ${p}CurrentTileScheduled(brick:u32,tile:u32,stage:u32)->bool{
  if(brick>=${p}BrickCapacity||tile>=${p}TilesPerBrick||stage>=${p}StageCount){return false;}
  return ${p}MaskContains(${p}StageMask(${p}BrickAt(${p}CurrentSlot(),brick),stage),tile);
}
fn ${p}MarkCurrentTileExecuted(brick:u32,tile:u32,stage:u32){
  let slot=${p}CurrentSlot();
  if(!${p}CurrentTileScheduled(brick,tile,stage)){
    ${p}BrickFail(slot,brick,tile,stage,${SPARSE_CM12_FRAME_PLAN_FAULT.missingCoverage}u);return;
  }
  let record=${p}TileAt(slot,brick,tile);
  atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_TILE.packedStageMasks}u],1u<<(12u+stage));
  atomicOr(&${arena}[${p}BrickAt(slot,brick)+${SPARSE_CM12_FRAME_PLAN_BRICK.executedStageMask}u],
    1u<<stage);
}
// Producer-local fail-closed seam. Unlike MarkCurrentTileExecuted this never
// fabricates coverage for work whose provenance or authority failed.
fn ${p}MarkCurrentTileFault(brick:u32,tile:u32,stage:u32){
  let slot=${p}CurrentSlot();
  if(brick>=${p}BrickCapacity||tile>=${p}TilesPerBrick||stage>=${p}StageCount){
    ${p}BrickFail(slot,brick,tile,stage,${SPARSE_CM12_FRAME_PLAN_FAULT.invalidStage}u);return;
  }
  ${p}BrickFail(slot,brick,tile,stage,${SPARSE_CM12_FRAME_PLAN_FAULT.missingCoverage}u);
}
fn ${p}MarkCurrentTileSkipped(brick:u32,tile:u32,stage:u32){
  let slot=${p}CurrentSlot();
  if(brick>=${p}BrickCapacity||tile>=${p}TilesPerBrick||stage>=${p}StageCount){
    ${p}BrickFail(slot,brick,tile,stage,${SPARSE_CM12_FRAME_PLAN_FAULT.invalidStage}u);return;
  }
  let record=${p}TileAt(slot,brick,tile);
  atomicOr(&${arena}[record+${SPARSE_CM12_FRAME_PLAN_TILE.packedStageMasks}u],1u<<(18u+stage));
  atomicOr(&${arena}[${p}BrickAt(slot,brick)+${SPARSE_CM12_FRAME_PLAN_BRICK.skippedStageMask}u],
    1u<<stage);
}

override CM12_FRAME_PLAN_VERIFY_STAGE:u32=0u;
var<workgroup> ${p}VerifyFault:atomic<u32>;
@compute @workgroup_size(64)
fn verifySparseCM12FramePlanCurrentStage(
  @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_id)lid:vec3u){
  let brick=wid.x;if(brick>=${p}BrickCapacity){return;}
  let slot=${p}CurrentSlot();
  if(lid.x==0u){atomicStore(&${p}VerifyFault,0u);}workgroupBarrier();
  if(CM12_FRAME_PLAN_VERIFY_STAGE>=${p}StageCount){
    if(lid.x==0u){${p}BrickFail(slot,brick,${p}Invalid,CM12_FRAME_PLAN_VERIFY_STAGE,
      ${SPARSE_CM12_FRAME_PLAN_FAULT.invalidStage}u);}return;
  }
  if(lid.x<${p}TilesPerBrick
    &&${p}CurrentTileScheduled(brick,lid.x,CM12_FRAME_PLAN_VERIFY_STAGE)){
    let tile=${p}TileAt(slot,brick,lid.x);
    let covered=${p}PackedStageMask(tile,12u)|${p}PackedStageMask(tile,18u);
    if((covered&(1u<<CM12_FRAME_PLAN_VERIFY_STAGE))==0u){
      atomicOr(&${p}VerifyFault,1u);
      atomicOr(&${arena}[tile+${SPARSE_CM12_FRAME_PLAN_TILE.packedStageMasks}u],
        1u<<(24u+CM12_FRAME_PLAN_VERIFY_STAGE));
    }
  }
  workgroupBarrier();
  if(lid.x==0u){
    if(atomicLoad(&${p}VerifyFault)!=0u){
      ${p}BrickFail(slot,brick,${p}Invalid,CM12_FRAME_PLAN_VERIFY_STAGE,
        ${SPARSE_CM12_FRAME_PLAN_FAULT.missingCoverage}u);
    }else{
      ${p}Store(${p}BrickAt(slot,brick)+${SPARSE_CM12_FRAME_PLAN_BRICK.consumerGeneration}u,
        ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.acceptedGeneration)}));
    }
  }
}

struct ${p}OverlayTile{
  valid:bool,generation:u32,directStages:u32,closureStages:u32,
  executedStages:u32,skippedStages:u32,uncoveredStages:u32,
  originCauses:u32,inheritedCauses:u32,packedClosureDepths:u32,
  packetIndices:vec2u,brickFlags:u32,brickFault:u32,logicalBrickKey:u32,
  topologyGeneration:u32,producerGeneration:u32,consumerGeneration:u32,packetEpoch:u32
}
fn ${p}OverlayTileAt(brick:u32,tile:u32,previous:bool)->${p}OverlayTile{
  var result:${p}OverlayTile;
  result.valid=false;result.generation=0u;result.directStages=0u;result.closureStages=0u;
  result.executedStages=0u;result.skippedStages=0u;result.uncoveredStages=63u;
  result.originCauses=0u;result.inheritedCauses=0u;result.packedClosureDepths=0u;
  result.packetIndices=vec2u(0u);result.brickFlags=0u;result.brickFault=
    ${SPARSE_CM12_FRAME_PLAN_FAULT.invalidHeader}u;
  result.logicalBrickKey=${p}Invalid;result.topologyGeneration=0u;
  result.producerGeneration=0u;result.consumerGeneration=0u;result.packetEpoch=0u;
  if(!${p}HeaderValid()
    ||(${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.flags)})
      &${SPARSE_CM12_FRAME_PLAN_FLAG.globalFault}u)!=0u
    ||brick>=${p}BrickCapacity||tile>=${p}TilesPerBrick){return result;}
  let slot=select(${p}CurrentSlot(),
    ${p}Load(${h(SPARSE_CM12_FRAME_PLAN_HEADER.previousSlot)}),previous);
  let brickRecord=${p}BrickAt(slot,brick);let tileRecord=${p}TileAt(slot,brick,tile);
  let generation=${p}Load(tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.generation}u);
  let packed=${p}Load(tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.packedStageMasks}u);
  let causes=${p}Load(tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.packedCauseMasks}u);
  result.generation=generation;result.directStages=packed&63u;
  result.closureStages=(packed>>6u)&63u;result.executedStages=(packed>>12u)&63u;
  result.skippedStages=(packed>>18u)&63u;result.uncoveredStages=(packed>>24u)&63u;
  result.originCauses=causes&0xffffu;result.inheritedCauses=causes>>16u;
  result.packedClosureDepths=${p}Load(
    tileRecord+${SPARSE_CM12_FRAME_PLAN_TILE.packedClosureDepths}u)&0x00ffffffu;
  result.packetIndices=vec2u(
    ${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.packedPacketIndicesLow}u),
    ${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.packedPacketIndicesHigh}u));
  result.brickFlags=${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.flags}u);
  result.brickFault=${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.faultCode}u);
  result.logicalBrickKey=${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.logicalBrickKey}u);
  result.topologyGeneration=${p}Load(
    brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.topologyGeneration}u);
  result.producerGeneration=${p}Load(
    brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.producerGeneration}u);
  result.consumerGeneration=${p}Load(
    brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.consumerGeneration}u);
  result.packetEpoch=${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.packetEpoch}u);
  let slotGeneration=${p}Load(${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.generation}u);
  let slotFlags=${p}Load(${p}SlotBase(slot)+${SPARSE_CM12_FRAME_PLAN_SLOT.flags}u);
  result.valid=generation==slotGeneration
    &&generation==${p}Load(brickRecord+${SPARSE_CM12_FRAME_PLAN_BRICK.generation}u)
    &&(slotFlags&${SPARSE_CM12_FRAME_PLAN_SLOT_FLAG.accepted}u)!=0u
    &&(result.brickFlags&${SPARSE_CM12_FRAME_PLAN_BRICK_FLAG.sealed}u)!=0u
    &&result.brickFault==${SPARSE_CM12_FRAME_PLAN_FAULT.none}u;
  return result;
}
`;
}
