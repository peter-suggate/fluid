import type { SparseCM12BrickTileFaceAddressLayout } from
  "./sparse-cm12-brick-tile-face-address-program";

/** Production split-face preparation/projection over the accepted ITR1 slot. */
export function createSparseCM12BrickTileFaceAddressWGSL(options: Readonly<{
  layout: SparseCM12BrickTileFaceAddressLayout;
  arenaName?: string;
}>): string {
  const l = options.layout, arena = options.arenaName ?? "topologyArena";
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(arena)) throw new TypeError("BFA1 arena name");
  return /* wgsl */ `
const BFA1_INTERIOR_COUNT:u32=${l.interiorTileCount}u;
const BFA1_SEAM_COUNT:u32=${l.seamAddressCount}u;
const BFA1_SEAM_PACKET_COUNT:u32=${l.seamPacketCount}u;
const BFA1_INTERIOR_BASE:u32=${l.interiorBaseWords}u;
const BFA1_SEAM_BASE:u32=${l.seamBaseWords}u;
const BFA1_DISPATCH_WIDTH:u32=${l.dispatchWidth}u;
const BFA1_INVALID:u32=0xffffffffu;
fn bfa1Load(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn bfa1Ordinal(wid:vec3u)->u32{return wid.x+BFA1_DISPATCH_WIDTH*wid.y;}
fn bfa1StablePacket(stableTile:u32)->u32{
  let leaf=stableTile/8u;return 64u*leaf+(stableTile-leaf*8u);
}
fn bfa1OwnerLocalCoordinate(stableTile:u32,axis:u32,lane:u32)->u32{
  let localTile=stableTile%8u;
  if(axis==0u){return 4u*(localTile&1u)+(lane&3u);}
  if(axis==1u){return 4u*((localTile>>1u)&1u)+((lane>>2u)&3u);}
  return 4u*((localTile>>2u)&1u)+((lane>>4u)&3u);
}
fn bfa1Prepare(row:u32){if(row!=BFA1_INVALID&&rowAccepted(row)){
  prepareTransportFaceRow(row);}}
fn bfa1Project(row:u32){if(row!=BFA1_INVALID&&rowAccepted(row)&&pcmRowContains(row)){
  projectPressureRow(row);state[sourceFaceVelocity()+row]=state[destinationFaceVelocity()+row];}}
@compute @workgroup_size(64)
fn prepareSparseCM12InteriorFaceTiles(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let ordinal=bfa1Ordinal(wid);if(ordinal>=BFA1_INTERIOR_COUNT){return;}
  let stableTile=bfa1Load(BFA1_INTERIOR_BASE+ordinal);
  let packet=bfa1StablePacket(stableTile);
  for(var axis=0u;axis<3u;axis+=1u){
    let row=itr1StableRowForOwner(packet,axis,lane);
    if(bfa1OwnerLocalCoordinate(stableTile,axis,lane)>0u
      &&row!=BFA1_INVALID&&rowKind(row)==0u){bfa1Prepare(row);}}
}
@compute @workgroup_size(64)
fn prepareSparseCM12SeamFacePackets(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let packetOrdinal=bfa1Ordinal(wid);if(packetOrdinal>=BFA1_SEAM_PACKET_COUNT){return;}
  let ordinal=64u*packetOrdinal+lane;if(ordinal>=BFA1_SEAM_COUNT){return;}
  let packed=bfa1Load(BFA1_SEAM_BASE+ordinal);let stableTile=packed>>9u;
  let address=packed&0x1ffu;let family=address/64u;let ownerLane=address&63u;
  let packet=bfa1StablePacket(stableTile);var row=BFA1_INVALID;
  if(family<3u){row=itr1StableRowForOwner(packet,family,ownerLane);
  }else{row=itr1StablePositiveSparseAirRowAndBucketOwner(
    packet,family-3u,ownerLane).x;}
  bfa1Prepare(row);
}
@compute @workgroup_size(64)
fn projectSparseCM12InteriorFaceTiles(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let ordinal=bfa1Ordinal(wid);if(ordinal>=BFA1_INTERIOR_COUNT){return;}
  let stableTile=bfa1Load(BFA1_INTERIOR_BASE+ordinal);
  let packet=bfa1StablePacket(stableTile);
  for(var axis=0u;axis<3u;axis+=1u){
    let row=itr1StableRowForOwner(packet,axis,lane);
    if(bfa1OwnerLocalCoordinate(stableTile,axis,lane)>0u
      &&row!=BFA1_INVALID&&rowKind(row)==0u){bfa1Project(row);}}
}
@compute @workgroup_size(64)
fn projectSparseCM12SeamFacePackets(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let packetOrdinal=bfa1Ordinal(wid);if(packetOrdinal>=BFA1_SEAM_PACKET_COUNT){return;}
  let ordinal=64u*packetOrdinal+lane;if(ordinal>=BFA1_SEAM_COUNT){return;}
  let packed=bfa1Load(BFA1_SEAM_BASE+ordinal);let stableTile=packed>>9u;
  let address=packed&0x1ffu;let family=address/64u;let ownerLane=address&63u;
  let packet=bfa1StablePacket(stableTile);var row=BFA1_INVALID;
  if(family<3u){row=itr1StableRowForOwner(packet,family,ownerLane);
  }else{row=itr1StablePositiveSparseAirRowAndBucketOwner(
    packet,family-3u,ownerLane).x;}
  bfa1Project(row);
}
`;
}
