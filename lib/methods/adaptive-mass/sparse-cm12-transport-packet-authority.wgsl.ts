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
const CM12_TPA_DIRECT_PACKETS_PER_LEAF:u32=${l.dispatchPacketsPerLeaf}u;
const CM12_TPA_DIRECT_PACKET_COUNT:u32=${l.dispatchPacketCount}u;
const CM12_TPA_DIRECT_WIDTH:u32=${l.dispatchWidth}u;
const CM12_TPA_COMPILER_WIDTH:u32=${l.compilerDispatchWidth}u;
const CM12_TPA_INDIRECT:u32=${l.indirectBaseWords}u;
const CM12_TPA_TRANSPORT_LOW:u32=${l.transportMaskLowBaseWords}u;
const CM12_TPA_TRANSPORT_HIGH:u32=${l.transportMaskHighBaseWords}u;
const CM12_TPA_SHARPENING_LOW:u32=${l.sharpeningMaskLowBaseWords}u;
const CM12_TPA_SHARPENING_HIGH:u32=${l.sharpeningMaskHighBaseWords}u;
const CM12_TPA_PACKET_LIST:u32=${l.packetListBaseWords}u;

fn cm12TransportPacketLaneSelected(mask:vec2u,lane:u32)->bool{
  return lane<64u&&((mask[lane>>5u]>>(lane&31u))&1u)!=0u;
}
fn cm12TransportDirectOrdinal(wid:vec3u)->u32{
  return wid.x+CM12_TPA_DIRECT_WIDTH*wid.y;
}
fn cm12TransportDirectStablePacket(ordinal:u32)->u32{
  let leaf=ordinal/CM12_TPA_DIRECT_PACKETS_PER_LEAF;
  return 64u*leaf+(ordinal-leaf*CM12_TPA_DIRECT_PACKETS_PER_LEAF);
}
fn cm12TransportCompactOrdinal(packet:u32)->u32{
  let leaf=packet/64u;let local=packet-64u*leaf;
  if(local>=CM12_TPA_DIRECT_PACKETS_PER_LEAF){return 0xffffffffu;}
  let ordinal=leaf*CM12_TPA_DIRECT_PACKETS_PER_LEAF+local;
  return select(0xffffffffu,ordinal,ordinal<CM12_TPA_DIRECT_PACKET_COUNT);
}
fn cm12TransportPacketCount()->u32{
  return min(atomicLoad(&${arena}[CM12_TPA_INDIRECT]),CM12_TPA_DIRECT_PACKET_COUNT);}
fn cm12TransportPacketOrdinal(rank:u32)->u32{
  return select(0xffffffffu,atomicLoad(&${arena}[CM12_TPA_PACKET_LIST+rank]),
    rank<cm12TransportPacketCount());
}
fn cm12TransportPacketId(rank:u32)->u32{
  let ordinal=cm12TransportPacketOrdinal(rank);
  return select(0xffffffffu,cm12TransportDirectStablePacket(ordinal),
    ordinal<CM12_TPA_DIRECT_PACKET_COUNT);
}
fn cm12TransportPacketMaskAt(ordinal:u32)->vec2u{
  if(ordinal>=CM12_TPA_DIRECT_PACKET_COUNT){return vec2u(0u);}
  return vec2u(atomicLoad(&${arena}[CM12_TPA_TRANSPORT_LOW+ordinal]),
    atomicLoad(&${arena}[CM12_TPA_TRANSPORT_HIGH+ordinal]));
}
fn cm12TransportSharpeningMaskAt(ordinal:u32)->vec2u{
  if(ordinal>=CM12_TPA_DIRECT_PACKET_COUNT){return vec2u(0u);}
  return vec2u(atomicLoad(&${arena}[CM12_TPA_SHARPENING_LOW+ordinal]),
    atomicLoad(&${arena}[CM12_TPA_SHARPENING_HIGH+ordinal]));
}
fn cm12TransportPublishSharpeningMask(packet:u32,mask:vec2u){
  let ordinal=cm12TransportCompactOrdinal(packet);
  if(ordinal==0xffffffffu){return;}
  atomicStore(&${arena}[CM12_TPA_SHARPENING_LOW+ordinal],mask.x);
  atomicStore(&${arena}[CM12_TPA_SHARPENING_HIGH+ordinal],mask.y);
}
var<workgroup>cm12SharpeningLow:atomic<u32>;
var<workgroup>cm12SharpeningHigh:atomic<u32>;
fn cm12TransportPublishSharpeningLane(packet:u32,lane:u32,
 selected:bool,cellScale:u32){
  if(lane==0u){
    atomicStore(&cm12SharpeningLow,0u);atomicStore(&cm12SharpeningHigh,0u);
  }
  workgroupBarrier();
  if(selected){
    var low=0u;var high=0u;
    if(cellScale==1u){low=0xffffffffu;high=0xffffffffu;
    }else if(cellScale==2u){
      let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
      let base=(q.x&~1u)+4u*(q.y&~1u)+16u*(q.z&~1u);
      for(var dz=0u;dz<2u;dz+=1u){for(var dy=0u;dy<2u;dy+=1u){
        for(var dx=0u;dx<2u;dx+=1u){let member=base+dx+4u*dy+16u*dz;
          if(member<32u){low|=1u<<member;}else{high|=1u<<(member-32u);}
        }
      }}
    }else if(lane<32u){low=1u<<lane;}else{high=1u<<(lane-32u);}
    _=atomicOr(&cm12SharpeningLow,low);_=atomicOr(&cm12SharpeningHigh,high);
  }
  workgroupBarrier();
  if(lane==0u){cm12TransportPublishSharpeningMask(packet,vec2u(
    atomicLoad(&cm12SharpeningLow),atomicLoad(&cm12SharpeningHigh)));}
}

// The packet authority is sealed before the three conservative dispatches.
// One invocation therefore snapshots its list entry, mask, and immutable TEI
// descriptor for the whole workgroup. Hot lanes do arithmetic only; they do
// not repeat atomic scheduling loads or the four-word packet decode.
var<workgroup>cm12TransportStagedPacketId:u32;
var<workgroup>cm12TransportStagedPacketOrdinal:u32;
var<workgroup>cm12TransportStagedPacketMask:vec2u;
var<workgroup>cm12TransportStagedPacket:CM12TransportPacket;
var<workgroup>cm12TransportStagedPacketOriginFine:vec3i;
var<workgroup>cm12TransportStagedTopologySlot:u32;
fn cm12StageTransportPacket(packetRank:u32,lane:u32){
  if(lane==0u){
    let ordinal=cm12TransportPacketOrdinal(packetRank);
    let packet=select(0xffffffffu,cm12TransportDirectStablePacket(ordinal),
      ordinal<CM12_TPA_DIRECT_PACKET_COUNT);
    let slot=acceptedTopologySlot();
    cm12TransportStagedTopologySlot=slot;
    cm12TransportStagedPacketOrdinal=ordinal;
    cm12TransportStagedPacketId=packet;
    cm12TransportStagedPacketMask=cm12TransportPacketMaskAt(ordinal);
    cm12TransportStagedPacket=cm12TeiPacket(packet,slot);
    cm12TransportStagedPacketOriginFine=cm12TeiPacketFineOrigin(packet,slot);
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

// Compile the prior frame's final scalar facts straight into the three packet
// transforms. Each invocation owns one compact TEI packet. Spatial-tile
// records are used only as the immutable packet-neighbour index; no dirty fact
// is copied into their address space.
@compute @workgroup_size(64)
fn compileSparseCM12TransportPacketsFromFinalScalarMasks(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let group=wid.x+CM12_TPA_COMPILER_WIDTH*wid.y;
  let ordinal=64u*group+lane;
  if(ordinal>=CM12_TPA_DIRECT_PACKET_COUNT){return;}
  atomicStore(&${arena}[CM12_TPA_TRANSPORT_LOW+ordinal],0u);
  atomicStore(&${arena}[CM12_TPA_TRANSPORT_HIGH+ordinal],0u);
  atomicStore(&${arena}[CM12_TPA_SHARPENING_LOW+ordinal],0u);
  atomicStore(&${arena}[CM12_TPA_SHARPENING_HIGH+ordinal],0u);
  let packetId=cm12TransportDirectStablePacket(ordinal);
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
        let tileOrigin=origin+vec3i(4u*vec3u(tx,ty,tz));
        for(var dz=-1;!dirty&&dz<=1;dz+=1){for(var dy=-1;!dirty&&dy<=1;dy+=1){
          for(var dx=-1;!dirty&&dx<=1;dx+=1){
            let q=tileOrigin+4*vec3i(dx,dy,dz);
            let neighbor=cm12TeiSpatialTile(cm12TransportSpatialTileId(q),slot);
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
  atomicStore(&${arena}[CM12_TPA_TRANSPORT_LOW+ordinal],packetMask.x);
  atomicStore(&${arena}[CM12_TPA_TRANSPORT_HIGH+ordinal],packetMask.y);
  let at=atomicAdd(&${arena}[CM12_TPA_INDIRECT],1u);
  if(at<CM12_TPA_DIRECT_PACKET_COUNT){
    atomicStore(&${arena}[CM12_TPA_PACKET_LIST+at],ordinal);}
}
fn cm12TransportExecutionCell(packetRank:u32,lane:u32)->u32{
  let ordinal=cm12TransportPacketOrdinal(packetRank);
  let packet=select(0xffffffffu,cm12TransportDirectStablePacket(ordinal),
    ordinal<CM12_TPA_DIRECT_PACKET_COUNT);
  let mask=cm12TransportPacketMaskAt(ordinal);
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
fn cm12TransportSpatialTileId(fine:vec3i)->u32{
  // Spatial tiles are leaf-local. Resolve the signed logical coordinate first;
  // converting a negative world origin to u32 aliases it with unrelated data.
  let brickWidth=i32(BRICK_FINE_RESOLUTION);
  let logical=vec3i(cm12WorldFloorToSpan(fine.x,brickWidth)/brickWidth,
    cm12WorldFloorToSpan(fine.y,brickWidth)/brickWidth,
    cm12WorldFloorToSpan(fine.z,brickWidth)/brickWidth);
  let leaf=cm12WorldOwnerAt(logical);
  if(leaf==0xffffffffu){return 0xffffffffu;}
  let leafFine=cm12WorldLeafCoordinate(leaf)*brickWidth;
  let signedRelative=fine-leafFine;
  if(any(signedRelative<vec3i(0))){return 0xffffffffu;}
  let relative=vec3u(signedRelative);
  let logicalLocal=relative/BRICK_FINE_RESOLUTION;
  let span=brickSpan(leaf);
  if(any(logicalLocal>=vec3u(span))){return 0xffffffffu;}
  let logicalIndex=logicalLocal.x+span*(logicalLocal.y+span*logicalLocal.z);
  if(logicalIndex>=CM12_TEI_LOGICAL_SLOTS_PER_LEAF){return 0xffffffffu;}
  let local=(relative%BRICK_FINE_RESOLUTION)/4u;
  let localIndex=local.x+CM12_TEI_SPATIAL_TILES_PER_AXIS
    *(local.y+CM12_TEI_SPATIAL_TILES_PER_AXIS*local.z);
  return leaf*CM12_TEI_SPATIAL_TILES_PER_LEAF
    +logicalIndex*CM12_TEI_SPATIAL_TILES_PER_LOGICAL+localIndex;
}
`;
}
