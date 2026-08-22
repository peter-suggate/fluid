import type { SparseCM12TransportPacketAuthorityLayout } from
  "./sparse-cm12-transport-packet-authority";

export function createSparseCM12TransportPacketAuthorityWGSL(options: {
  readonly layout: SparseCM12TransportPacketAuthorityLayout;
  readonly arenaName?: string;
}): string {
  const arena = options.arenaName ?? "activity";
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(arena)) throw new TypeError("arenaName");
  const l = options.layout;
  return /* wgsl */ `
const CM12_TPA_CAPACITY:u32=${l.packetCapacity}u;
const CM12_TPA_FAMILIES:u32=3u;
const CM12_TPA_FAMILY_STRIDE:u32=${l.familyStrideWords}u;
const CM12_TPA_HEADER:u32=${l.baseWords}u;

fn cm12TransportFamilyHeader(family:u32)->u32{
  return CM12_TPA_HEADER+family*CM12_TPA_FAMILY_STRIDE;}
fn cm12TransportFamilyStamp(family:u32)->u32{
  return cm12TransportFamilyHeader(family)+8u;}
fn cm12TransportFamilyMaskLow(family:u32)->u32{
  return cm12TransportFamilyStamp(family)+CM12_TPA_CAPACITY;}
fn cm12TransportFamilyMaskHigh(family:u32)->u32{
  return cm12TransportFamilyMaskLow(family)+CM12_TPA_CAPACITY;}
fn cm12TransportFamilyList(family:u32)->u32{
  return cm12TransportFamilyMaskHigh(family)+CM12_TPA_CAPACITY;}

fn cm12TransportPacketCount(family:u32)->u32{
  return select(0u,atomicLoad(&${arena}[cm12TransportFamilyHeader(family)+1u]),
    family<CM12_TPA_FAMILIES);}
fn cm12TransportPacketId(rank:u32,family:u32)->u32{
  let count=cm12TransportPacketCount(family);
  return select(0xffffffffu,
    atomicLoad(&${arena}[cm12TransportFamilyList(family)+rank]),
    family<CM12_TPA_FAMILIES&&rank<count);
}
fn cm12TransportPacketMask(packet:u32,family:u32)->vec2u{
  if(packet>=CM12_TPA_CAPACITY||family>=CM12_TPA_FAMILIES){return vec2u(0u);}
  return vec2u(atomicLoad(&${arena}[cm12TransportFamilyMaskLow(family)+packet]),
    atomicLoad(&${arena}[cm12TransportFamilyMaskHigh(family)+packet]));
}
fn cm12TransportPacketLaneSelected(mask:vec2u,lane:u32)->bool{
  return lane<64u&&((mask[lane>>5u]>>(lane&31u))&1u)!=0u;
}

// The packet authority is sealed before the three conservative dispatches.
// One invocation therefore snapshots its list entry, mask, and immutable TEI
// descriptor for the whole workgroup. Hot lanes do arithmetic only; they do
// not repeat atomic scheduling loads or the four-word packet decode.
var<workgroup>cm12TransportStagedPacketId:u32;
var<workgroup>cm12TransportStagedPacketMask:vec2u;
var<workgroup>cm12TransportStagedPacket:CM12TransportPacket;
var<workgroup>cm12TransportStagedPacketOriginFine:vec3u;
var<workgroup>cm12TransportStagedTopologySlot:u32;
fn cm12StageTransportPacket(packetRank:u32,lane:u32,family:u32){
  if(lane==0u){
    let packet=cm12TransportPacketId(packetRank,family);
    let slot=acceptedTopologySlot();
    cm12TransportStagedTopologySlot=slot;
    cm12TransportStagedPacketId=packet;
    cm12TransportStagedPacketMask=cm12TransportPacketMask(packet,family);
    cm12TransportStagedPacket=cm12TeiPacket(packet,slot);
    cm12TransportStagedPacketOriginFine=vec3u(0xffffffffu);
    if(family<2u){cm12TransportStagedPacketOriginFine=
      cm12TeiPacketFineOrigin(packet,slot);}
  }
  workgroupBarrier();
}
fn cm12TransportStagedExecutionCell(lane:u32)->u32{
  let packet=cm12TransportStagedPacket;
  let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  if(!cm12TransportPacketLaneSelected(cm12TransportStagedPacketMask,lane)
    ||packet.first==0xffffffffu||any(q>=packet.counts)){return 0xffffffffu;}
  return packet.first+q.x+packet.strideY*q.y+packet.strideZ*q.z;
}

@compute @workgroup_size(1)
fn beginSparseCM12TransportPacketAuthority(){
  for(var family=0u;family<CM12_TPA_FAMILIES;family+=1u){
    let header=cm12TransportFamilyHeader(family);
    atomicStore(&${arena}[header],max(1u,fsm1Generation()));
    atomicStore(&${arena}[header+1u],0u);atomicStore(&${arena}[header+2u],0u);
    atomicStore(&${arena}[header+3u],CM12_TPA_CAPACITY);
    atomicStore(&${arena}[header+4u],0u);atomicStore(&${arena}[header+5u],1u);
    atomicStore(&${arena}[header+6u],1u);atomicStore(&${arena}[header+7u],0u);
  }
}
fn cm12PublishDirectTransportPacket(packet:u32,mask:vec2u,family:u32){
  if(packet>=CM12_TPA_CAPACITY||(mask.x|mask.y)==0u){return;}
  let header=cm12TransportFamilyHeader(family);
  atomicOr(&${arena}[cm12TransportFamilyMaskLow(family)+packet],mask.x);
  atomicOr(&${arena}[cm12TransportFamilyMaskHigh(family)+packet],mask.y);
  let generation=atomicLoad(&${arena}[header]);
  let previous=atomicExchange(&${arena}[
    cm12TransportFamilyStamp(family)+packet],generation);
  if(previous==0u){let at=atomicAdd(&${arena}[header+1u],1u);
    if(at<CM12_TPA_CAPACITY){atomicStore(
      &${arena}[cm12TransportFamilyList(family)+at],packet);}
    else{atomicOr(&${arena}[header+2u],2u);}}
}
// Compile the prior frame's final scalar facts straight into the three packet
// execution families. Each invocation owns one stable TEI packet. Spatial-tile
// records are used only as the immutable packet-neighbour index; no dirty fact
// is copied into their address space.
@compute @workgroup_size(64)
fn compileSparseCM12TransportPacketsFromFinalScalarMasks(
 @builtin(global_invocation_id)gid:vec3u){
  let packetId=gid.x;if(packetId>=CM12_TPA_CAPACITY){return;}
  let slot=acceptedTopologySlot();let packet=cm12TeiPacket(packetId,slot);
  if(packet.first==0xffffffffu){return;}
  let leaf=packetId/64u;let leafDescriptor=cm12TeiLoadLeaf(slot,leaf);
  let origin=cm12TeiPacketFineOrigin(packetId,slot);
  var packetMask=vec2u(0u);
  for(var lane=0u;lane<64u;lane+=1u){
    let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
    if(any(q>=packet.counts)){continue;}
    if(lane<32u){packetMask.x|=1u<<lane;}else{packetMask.y|=1u<<(lane-32u);}
  }
  var dirty=!fsm1Published()||fsm1TopologyGeneration()
    !=atomicLoad(&topologyArena[topologyWorklistBase()]);
  dirty=dirty||incrementalActivityBrickVelocityDirty(leaf);
  let tileCounts=(leafDescriptor.scale*packet.counts+vec3u(3u))/4u;
  for(var tz=0u;!dirty&&tz<tileCounts.z;tz+=1u){
    for(var ty=0u;!dirty&&ty<tileCounts.y;ty+=1u){
      for(var tx=0u;!dirty&&tx<tileCounts.x;tx+=1u){
        let tileOrigin=origin+4u*vec3u(tx,ty,tz);
        for(var dz=-1;!dirty&&dz<=1;dz+=1){for(var dy=-1;!dirty&&dy<=1;dy+=1){
          for(var dx=-1;!dirty&&dx<=1;dx+=1){
            let q=vec3i(tileOrigin)+4*vec3i(dx,dy,dz);
            if(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))){continue;}
            let neighbor=cm12TeiSpatialTile(cm12TransportSpatialTileId(vec3u(q)),slot);
            if(neighbor.packetId==0xffffffffu){continue;}
            let mask=fsm1Nonexact(neighbor.packetId)&~fsm1Bulk(neighbor.packetId)
              &neighbor.laneMask;
            dirty=(mask.x|mask.y)!=0u;
          }
        }}
      }
    }
  }
  if(!dirty){return;}
  for(var family=0u;family<CM12_TPA_FAMILIES;family+=1u){
    cm12PublishDirectTransportPacket(packetId,packetMask,family);}
}
@compute @workgroup_size(64)
fn clearSparseCM12TransportPacketAuthority(@builtin(global_invocation_id)gid:vec3u){
  let packet=gid.x;if(packet>=CM12_TPA_CAPACITY){return;}
  for(var family=0u;family<CM12_TPA_FAMILIES;family+=1u){
    atomicStore(&${arena}[cm12TransportFamilyStamp(family)+packet],0u);
    atomicStore(&${arena}[cm12TransportFamilyMaskLow(family)+packet],0u);
    atomicStore(&${arena}[cm12TransportFamilyMaskHigh(family)+packet],0u);
  }
}
@compute @workgroup_size(1)
fn finalizeSparseCM12TransportPacketAuthority(){
  for(var family=0u;family<CM12_TPA_FAMILIES;family+=1u){
    let header=cm12TransportFamilyHeader(family);
    let fault=atomicLoad(&${arena}[header+2u]);
    let count=select(atomicLoad(&${arena}[header+1u]),0u,fault!=0u);
    atomicStore(&${arena}[header+4u],count);
    atomicStore(&${arena}[header+5u],1u);atomicStore(&${arena}[header+6u],1u);
  }
}
fn cm12TransportExecutionCell(packetRank:u32,lane:u32,family:u32)->u32{
  let packet=cm12TransportPacketId(packetRank,family);
  let mask=cm12TransportPacketMask(packet,family);
  if(packet==0xffffffffu||!cm12TransportPacketLaneSelected(mask,lane)){
    return 0xffffffffu;}
  return cm12TeiPacketCell(packet,lane,acceptedTopologySlot());
}
// Replays one stable SRR spatial tile (logical brick * tiles-per-logical +
// local 4^3 tile) as a 64-lane workgroup through TEI's precompiled
// tile-to-cell mapping. Lane numbering is the packet's; the cell SET is the
// baseline work-tile's.
fn cm12TransportSpatialTileCell(tile:u32,lane:u32)->u32{
  let record=cm12TeiSpatialTile(tile,acceptedTopologySlot());
  if(record.packetId==0xffffffffu||!cm12TeiSpatialTileSelectsLane(record,lane)){
    return 0xffffffffu;}
  return cm12TeiPacketCell(record.packetId,lane,acceptedTopologySlot());
}
// Stable spatial-tile id of the fine coordinate's min corner, in the same id
// same stable space used by cm12TeiSpatialTile.
fn cm12TransportSpatialTileId(fine:vec3u)->u32{
  // Spatial tiles are 4 fine cells per edge at every brick resolution.
  let local=(fine%BRICK_FINE_RESOLUTION)/4u;
  return cm12TeiLogicalKey(fine/BRICK_FINE_RESOLUTION)
    *CM12_TEI_SPATIAL_TILES_PER_LOGICAL+local.x+CM12_TEI_SPATIAL_TILES_PER_AXIS
    *(local.y+CM12_TEI_SPATIAL_TILES_PER_AXIS*local.z);
}
`;
}
