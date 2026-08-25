import {
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_EDGE,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS,
  type SparseCM12TransportExecutionImageLayout,
} from "./sparse-cm12-transport-execution-image";

export function createSparseCM12TransportExecutionImageWGSL(options: {
  readonly layout: SparseCM12TransportExecutionImageLayout;
  readonly packedLogicalOwner16BaseWords: number;
  readonly imageName?: string;
}): string {
  const { layout } = options;
  const image = options.imageName ?? "fineMetadata";
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(image)) {
    throw new TypeError("transport execution image binding must be a WGSL identifier");
  }
  const l = SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF;
  const sd = SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_SCALE_DESCRIPTOR;
  return /* wgsl */ `
const CM12_TEI_INVALID:u32=${SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID}u;
const CM12_TEI_LEAF_WORDS:u32=${SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_LEAF_WORDS}u;
const CM12_TEI_PACKET_WORDS:u32=${SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS}u;
const CM12_TEI_SPATIAL_TILE_WORDS:u32=${SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_SPATIAL_TILE_WORDS}u;
const CM12_TEI_PACKETS_PER_LEAF:u32=${SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKETS_PER_LEAF}u;
const CM12_TEI_PACKET_EDGE:u32=${SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_EDGE}u;
const CM12_TEI_SCALE_LOG2_SHIFT:u32=${sd.scaleLog2Shift}u;
const CM12_TEI_SCALE_LOG2_MASK:u32=${sd.scaleLog2Mask}u;
const CM12_TEI_SCALE_LOG2_ENCODED:u32=${sd.encodedMask}u;
const CM12_TEI_LEAF_CAPACITY:u32=${layout.leafCapacity}u;
const CM12_TEI_PACKET_CAPACITY:u32=${layout.packetCapacity}u;
const CM12_TEI_SPATIAL_TILE_CAPACITY:u32=${layout.spatialTileCapacity}u;
const CM12_TEI_SPATIAL_TILES_PER_LOGICAL:u32=${layout.spatialTilesPerLogicalBrick}u;
const CM12_TEI_SPATIAL_TILES_PER_AXIS:u32=${layout.spatialTilesPerLogicalBrickAxis}u;
const CM12_TEI_SPATIAL_TILES_PER_LEAF:u32=${layout.spatialTilesPerLeaf}u;
const CM12_TEI_LOGICAL_SLOTS_PER_LEAF:u32=${layout.logicalSlotsPerLeaf}u;
const CM12_TEI_SLOT0_LEAVES:u32=${layout.slotLeafBaseOffsets[0]}u;
const CM12_TEI_SLOT1_LEAVES:u32=${layout.slotLeafBaseOffsets[1]}u;
const CM12_TEI_SLOT0_PACKETS:u32=${layout.slotPacketBaseOffsets[0]}u;
const CM12_TEI_SLOT1_PACKETS:u32=${layout.slotPacketBaseOffsets[1]}u;
const CM12_TEI_SLOT0_SPATIAL_TILES:u32=${layout.slotSpatialTileBaseOffsets[0]}u;
const CM12_TEI_SLOT1_SPATIAL_TILES:u32=${layout.slotSpatialTileBaseOffsets[1]}u;
const CM12_TEI_SLOT0:u32=${layout.slotBaseWords[0]}u;
const CM12_TEI_SLOT1:u32=${layout.slotBaseWords[1]}u;
const CM12_TEI_PACKED_OWNER16:u32=${options.packedLogicalOwner16BaseWords}u;

struct CM12TransportLeaf{generation:u32,flags:u32,first:u32,count:u32,
  originKey:u32,valid:vec3u,scale:u32,scaleLog2:u32}
struct CM12TransportOwner{cell:u32,widths:vec3u,volume:u32}
struct CM12TransportPacket{first:u32,counts:vec3u,strideY:u32,strideZ:u32}
struct CM12TransportSpatialTile{packetId:u32,laneMask:vec2u}

var<workgroup>cm12TeiCacheKeys:array<u32,27>;
var<workgroup>cm12TeiCache0:array<vec4u,27>;
var<workgroup>cm12TeiCache1:array<vec4u,27>;
var<workgroup>cm12TeiCacheCenter:vec3u;
var<workgroup>cm12TeiCacheSlot:u32;

fn cm12TeiSlotBase(slot:u32)->u32{
  return select(CM12_TEI_SLOT0,CM12_TEI_SLOT1,(slot&1u)!=0u);}
fn cm12TeiLeafBase(slot:u32)->u32{
  return select(CM12_TEI_SLOT0_LEAVES,CM12_TEI_SLOT1_LEAVES,(slot&1u)!=0u);}
fn cm12TeiPacketBase(slot:u32)->u32{
  return select(CM12_TEI_SLOT0_PACKETS,CM12_TEI_SLOT1_PACKETS,(slot&1u)!=0u);}
fn cm12TeiSpatialTileBase(slot:u32)->u32{return select(
  CM12_TEI_SLOT0_SPATIAL_TILES,CM12_TEI_SLOT1_SPATIAL_TILES,(slot&1u)!=0u);}
fn cm12TeiOwnerBrick(key:u32)->u32{
  return cm12WorldOwnerAt(vec3i(cm12TeiLogicalCoordinate(key)));
}
fn cm12TeiScaleLog2(scale:u32,descriptor:u32)->u32{
  if(scale==0u){return 0u;}
  if((descriptor&CM12_TEI_SCALE_LOG2_ENCODED)!=0u){
    return (descriptor>>CM12_TEI_SCALE_LOG2_SHIFT)&CM12_TEI_SCALE_LOG2_MASK;}
  // Compatibility with packet-axis-only word 7 from the original TEI2
  // producer.  New producer/consumer pairs take the encoded branch.
  return u32(firstLeadingBit(scale));
}
fn cm12TeiLoadLeaf(slot:u32,brick:u32)->CM12TransportLeaf{
  if(brick>=CM12_TEI_LEAF_CAPACITY){return CM12TransportLeaf(
    0u,0u,CM12_TEI_INVALID,0u,CM12_TEI_INVALID,vec3u(0u),0u,0u);}
  let at=cm12TeiLeafBase(slot)+CM12_TEI_LEAF_WORDS*brick;
  let generation=${image}[at+${l.generation}u];
  // Delta publication intentionally leaves unchanged records stamped by the
  // generation that last changed their owning leaf.  Zero alone is invalid;
  // requiring equality with the slot transaction generation would turn every
  // untouched leaf into a hole after an O(changed-leaf) patch.
  if(generation==0u){return CM12TransportLeaf(
    0u,0u,CM12_TEI_INVALID,0u,CM12_TEI_INVALID,vec3u(0u),0u,0u);}
  let packed=${image}[at+${l.validDimensions}u];
  let scale=${image}[at+${l.scale}u];
  return CM12TransportLeaf(generation,${image}[at+${l.flags}u],
    ${image}[at+${l.cellFirst}u],${image}[at+${l.cellCount}u],
    ${image}[at+${l.originKey}u],vec3u(packed&31u,(packed>>5u)&31u,
      (packed>>10u)&31u),scale,cm12TeiScaleLog2(
        scale,${image}[at+${l.scaleDescriptor}u]));
}
fn cm12TeiLogicalCoordinate(key:u32)->vec3u{
  let dims=vec3u(${layout.logicalBrickDimensions[0]}u,
    ${layout.logicalBrickDimensions[1]}u,${layout.logicalBrickDimensions[2]}u);
  let xy=dims.x*dims.y;let z=key/xy;let rem=key-z*xy;let y=rem/dims.x;
  return vec3u(rem-y*dims.x,y,z);
}
fn cm12TeiLogicalKey(q:vec3u)->u32{return q.x+${layout.logicalBrickDimensions[0]}u
  *(q.y+${layout.logicalBrickDimensions[1]}u*q.z);}
fn cm12TeiStageDirectory(tileOrigin:vec3u,lane:u32,slot:u32){
  if(lane==0u){cm12TeiCacheCenter=tileOrigin/BRICK_FINE_RESOLUTION;
    cm12TeiCacheSlot=slot&1u;}
  if(lane<27u){let ox=i32(lane%3u)-1;let oy=i32((lane/3u)%3u)-1;
    let oz=i32(lane/9u)-1;let coordinate=vec3i(tileOrigin/BRICK_FINE_RESOLUTION)
      +vec3i(ox,oy,oz);let dims=vec3i(${layout.logicalBrickDimensions[0]},
        ${layout.logicalBrickDimensions[1]},${layout.logicalBrickDimensions[2]});
    var key=CM12_TEI_INVALID;var leaf=CM12TransportLeaf(
      0u,0u,CM12_TEI_INVALID,0u,CM12_TEI_INVALID,vec3u(0u),0u,0u);
    if(all(coordinate>=vec3i(0))&&all(coordinate<dims)){
      key=cm12TeiLogicalKey(vec3u(coordinate));
      leaf=cm12TeiLoadLeaf(slot,cm12TeiOwnerBrick(key));}
    cm12TeiCacheKeys[lane]=key;
    cm12TeiCache0[lane]=vec4u(leaf.generation,leaf.flags,leaf.first,leaf.count);
    cm12TeiCache1[lane]=vec4u(leaf.originKey,
      leaf.valid.x|(leaf.valid.y<<5u)|(leaf.valid.z<<10u),leaf.scale,leaf.scaleLog2);}
  workgroupBarrier();
}
fn cm12TeiLeafAtLogical(coordinate:vec3u)->CM12TransportLeaf{
  if(EXP_DYNAMIC_WORLD_GROWTH){
    return cm12TeiLoadLeaf(cm12TeiCacheSlot,cm12WorldOwnerAt(vec3i(coordinate)));
  }
  let dims=vec3u(${layout.logicalBrickDimensions[0]}u,
    ${layout.logicalBrickDimensions[1]}u,${layout.logicalBrickDimensions[2]}u);
  if(any(coordinate>=dims)){return CM12TransportLeaf(
    0u,0u,CM12_TEI_INVALID,0u,CM12_TEI_INVALID,vec3u(0u),0u,0u);}
  let delta=vec3i(coordinate)-vec3i(cm12TeiCacheCenter);var a=vec4u(0u);
  var b=vec4u(CM12_TEI_INVALID,0u,0u,0u);
  if(all(delta>=vec3i(-1))&&all(delta<=vec3i(1))){
    let index=u32(delta.x+1)+3u*u32(delta.y+1)+9u*u32(delta.z+1);
    a=cm12TeiCache0[index];b=cm12TeiCache1[index];
  }else{let leaf=cm12TeiLoadLeaf(cm12TeiCacheSlot,
      cm12TeiOwnerBrick(cm12TeiLogicalKey(coordinate)));
    a=vec4u(leaf.generation,leaf.flags,leaf.first,leaf.count);
    b=vec4u(leaf.originKey,leaf.valid.x|(leaf.valid.y<<5u)|(leaf.valid.z<<10u),
      leaf.scale,leaf.scaleLog2);}
  return CM12TransportLeaf(a.x,a.y,a.z,a.w,b.x,
    vec3u(b.y&31u,(b.y>>5u)&31u,(b.y>>10u)&31u),b.z,b.w);
}
fn cm12TeiOwnerAtFine(q:vec3i)->CM12TransportOwner{
  if((!EXP_DYNAMIC_WORLD_GROWTH&&(any(q<vec3i(0))||any(q>=vec3i(p.dimensions.xyz))))
    ||(EXP_DYNAMIC_WORLD_GROWTH&&q.y<0)){return CM12TransportOwner(
    CM12_TEI_INVALID,vec3u(0u),0u);}
  let owner=select(cm12TeiOwnerBrick(cm12TeiLogicalKey(
    vec3u(q)/BRICK_FINE_RESOLUTION)),cm12WorldOwnerAt(vec3i(
      cm12WorldFloorToSpan(q.x,i32(BRICK_FINE_RESOLUTION))/i32(BRICK_FINE_RESOLUTION),
      cm12WorldFloorToSpan(q.y,i32(BRICK_FINE_RESOLUTION))/i32(BRICK_FINE_RESOLUTION),
      cm12WorldFloorToSpan(q.z,i32(BRICK_FINE_RESOLUTION))/i32(BRICK_FINE_RESOLUTION))),
    EXP_DYNAMIC_WORLD_GROWTH);
  let leaf=cm12TeiLoadLeaf(cm12TeiCacheSlot,owner);
  if((leaf.flags&0x80000000u)==0u||leaf.scale==0u){return CM12TransportOwner(
    CM12_TEI_INVALID,vec3u(0u),0u);}
  let origin=select(vec3i(cm12TeiLogicalCoordinate(leaf.originKey)),
    cm12WorldLeafCoordinate(owner),EXP_DYNAMIC_WORLD_GROWTH)
    *i32(BRICK_FINE_RESOLUTION);
  let scale=1u<<leaf.scaleLog2;
  let relative=q-origin;
  if(any(relative<vec3i(0))){return CM12TransportOwner(
    CM12_TEI_INVALID,vec3u(0u),0u);}
  let local=vec3u(relative)/scale;
  if(any(local>=leaf.valid)){return CM12TransportOwner(CM12_TEI_INVALID,vec3u(0u),0u);}
  let offset=local.x+leaf.valid.x*(local.y+leaf.valid.y*local.z);
  if(offset>=leaf.count){return CM12TransportOwner(CM12_TEI_INVALID,vec3u(0u),0u);}
  let lower=origin+vec3i(vec3u(relative)&~vec3u(scale-1u));
  let widths=select(min(vec3u(scale),p.dimensions.xyz-vec3u(lower)),vec3u(scale),
    EXP_DYNAMIC_WORLD_GROWTH&&owner>=CM12_WDR_INITIAL_LEAVES);
  return CM12TransportOwner(leaf.first+offset,widths,widths.x*widths.y*widths.z);
}
fn cm12TeiPacket(packet:u32,slot:u32)->CM12TransportPacket{
  if(packet>=CM12_TEI_PACKET_CAPACITY){return CM12TransportPacket(
    CM12_TEI_INVALID,vec3u(0u),0u,0u);}
  let at=cm12TeiPacketBase(slot)+CM12_TEI_PACKET_WORDS*packet;
  let first=${image}[at+1u];let counts=${image}[at+2u];let strides=${image}[at+3u];
  if(${image}[at]==0u||(counts&0x80000000u)==0u){
    return CM12TransportPacket(CM12_TEI_INVALID,vec3u(0u),0u,0u);}
  return CM12TransportPacket(first,vec3u(counts&31u,(counts>>5u)&31u,
    (counts>>10u)&31u),strides&0xffffu,strides>>16u);
}
fn cm12TeiPacketCell(packet:u32,lane:u32,slot:u32)->u32{
  let item=cm12TeiPacket(packet,slot);let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  if(item.first==CM12_TEI_INVALID||any(q>=item.counts)){return CM12_TEI_INVALID;}
  return item.first+q.x+item.strideY*q.y+item.strideZ*q.z;
}
fn cm12TeiLeafLocalPacketAddress(leaf:u32,resolution:u32,local:vec3u)->vec2u{
  if(leaf>=CM12_TEI_LEAF_CAPACITY||resolution==0u){return vec2u(CM12_TEI_INVALID);}
  let packetAxis=max(1u,(resolution+CM12_TEI_PACKET_EDGE-1u)/CM12_TEI_PACKET_EDGE);
  let coordinate=local/CM12_TEI_PACKET_EDGE;
  if(any(coordinate>=vec3u(packetAxis))){return vec2u(CM12_TEI_INVALID);}
  let packet=leaf*CM12_TEI_PACKETS_PER_LEAF+coordinate.x+packetAxis
    *(coordinate.y+packetAxis*coordinate.z);
  let q=local%CM12_TEI_PACKET_EDGE;
  let lane=q.x+CM12_TEI_PACKET_EDGE*(q.y+CM12_TEI_PACKET_EDGE*q.z);
  return select(vec2u(packet,lane),vec2u(CM12_TEI_INVALID),
    packet>=CM12_TEI_PACKET_CAPACITY||lane>=64u);
}
fn cm12TeiPacketFineOrigin(packet:u32,slot:u32)->vec3u{
  if(packet>=CM12_TEI_PACKET_CAPACITY){return vec3u(CM12_TEI_INVALID);}
  let brick=packet/CM12_TEI_PACKETS_PER_LEAF;
  let localPacket=packet%CM12_TEI_PACKETS_PER_LEAF;
  let leaf=cm12TeiLoadLeaf(slot,brick);
  if((leaf.flags&0x80000000u)==0u||leaf.scale==0u){return vec3u(CM12_TEI_INVALID);}
  let scale=1u<<leaf.scaleLog2;
  let resolution=leaf.flags&31u;let packetAxis=max(1u,(resolution+3u)/4u);
  if(localPacket>=packetAxis*packetAxis*packetAxis){return vec3u(CM12_TEI_INVALID);}
  let pz=localPacket/(packetAxis*packetAxis);
  let remainder=localPacket-pz*packetAxis*packetAxis;
  let py=remainder/packetAxis;let px=remainder-py*packetAxis;
  return vec3u(cm12WorldLeafCoordinate(brick))*BRICK_FINE_RESOLUTION
    +scale*CM12_TEI_PACKET_EDGE*vec3u(px,py,pz);
}
// Cell scale (fine cells per cell edge) of the staged leaf owning a fine
// coordinate; reads the 27-leaf window, so call after cm12TeiStageDirectory.
fn cm12TeiStagedScaleAtFine(fine:vec3u)->u32{
  return 1u<<cm12TeiLeafAtLogical(fine/BRICK_FINE_RESOLUTION).scaleLog2;}
fn cm12TeiSpatialTile(tile:u32,slot:u32)->CM12TransportSpatialTile{
  if(tile>=CM12_TEI_SPATIAL_TILE_CAPACITY){return CM12TransportSpatialTile(
    CM12_TEI_INVALID,vec2u(0u));}
  let at=cm12TeiSpatialTileBase(slot)+CM12_TEI_SPATIAL_TILE_WORDS*tile;
  if(${image}[at]==0u){return CM12TransportSpatialTile(
    CM12_TEI_INVALID,vec2u(0u));}
  return CM12TransportSpatialTile(${image}[at+1u],vec2u(${image}[at+2u],${image}[at+3u]));
}
fn cm12TeiSpatialTileSelectsLane(tile:CM12TransportSpatialTile,lane:u32)->bool{
  return lane<64u&&((tile.laneMask[lane>>5u]>>(lane&31u))&1u)!=0u;}

fn cm12TeiWriteLeaf(slot:u32,brick:u32,generation:u32,candidate:bool){
  var originKey=brick;
  if(!EXP_DYNAMIC_WORLD_GROWTH||brick<CM12_WDR_INITIAL_LEAVES){
    originKey=topology[p.topologyOffsets2.z+2u*brick+1u];
  }
  let span=brickSpan(brick);var resolution=acceptedBrickResolution(brick);
  if(candidate){resolution=scheduledBrickResolution(brick);}
  let spanFine=BRICK_FINE_RESOLUTION*span;let scale=spanFine/resolution;
  let origin=cm12WorldLeafCoordinate(brick)*i32(BRICK_FINE_RESOLUTION);
  let extent=select(min(vec3u(spanFine),p.dimensions.xyz-min(vec3u(origin),p.dimensions.xyz)),
    vec3u(spanFine),EXP_DYNAMIC_WORLD_GROWTH&&brick>=CM12_WDR_INITIAL_LEAVES);
  let valid=(extent+vec3u(scale-1u))/scale;
  let range=templateBrickCellRange(brick,resolution);
  let flags=resolution|(u32(firstLeadingBit(span))<<8u)
    |select(0u,0x80000000u,select(brickActive(brick),
      scheduledBrickActive(brick),candidate));
  let at=cm12TeiLeafBase(slot)+CM12_TEI_LEAF_WORDS*brick;
  ${image}[at]=generation;${image}[at+1u]=flags;${image}[at+2u]=range.x;
  ${image}[at+3u]=range.y;${image}[at+4u]=originKey;
  ${image}[at+5u]=valid.x|(valid.y<<5u)|(valid.z<<10u);
  ${image}[at+6u]=scale;${image}[at+7u]=CM12_TEI_SCALE_LOG2_ENCODED
    |(u32(firstLeadingBit(scale))<<CM12_TEI_SCALE_LOG2_SHIFT)
    |max(1u,(resolution+3u)/4u);
}
fn cm12TeiWritePacket(slot:u32,packet:u32,generation:u32,candidate:bool){
  let at=cm12TeiPacketBase(slot)+CM12_TEI_PACKET_WORDS*packet;
  ${image}[at]=generation;${image}[at+1u]=CM12_TEI_INVALID;
  ${image}[at+2u]=0u;${image}[at+3u]=0u;
  let brick=packet/CM12_TEI_PACKETS_PER_LEAF;
  let localPacket=packet%CM12_TEI_PACKETS_PER_LEAF;
  if(brick>=CM12_TEI_LEAF_CAPACITY
    ||!select(brickActive(brick),scheduledBrickActive(brick),candidate)){return;}
  var resolution=acceptedBrickResolution(brick);
  if(candidate){resolution=scheduledBrickResolution(brick);}
  let packetAxis=max(1u,(resolution+3u)/CM12_TEI_PACKET_EDGE);
  if(localPacket>=packetAxis*packetAxis*packetAxis){return;}
  let pz=localPacket/(packetAxis*packetAxis);
  let remainder=localPacket-pz*packetAxis*packetAxis;
  let py=remainder/packetAxis;let px=remainder-py*packetAxis;
  let local=CM12_TEI_PACKET_EDGE*vec3u(px,py,pz);
  let spanFine=BRICK_FINE_RESOLUTION*brickSpan(brick);let scale=spanFine/resolution;
  let leafOrigin=cm12WorldLeafCoordinate(brick)*i32(BRICK_FINE_RESOLUTION);
  let extent=select(min(vec3u(spanFine),p.dimensions.xyz-min(vec3u(leafOrigin),p.dimensions.xyz)),
    vec3u(spanFine),EXP_DYNAMIC_WORLD_GROWTH&&brick>=CM12_WDR_INITIAL_LEAVES);
  let dimensions=(extent+vec3u(scale-1u))/scale;
  if(any(local>=dimensions)){return;}
  let counts=min(vec3u(CM12_TEI_PACKET_EDGE),dimensions-local);
  let range=templateBrickCellRange(brick,resolution);
  let first=range.x+local.x+dimensions.x*(local.y+dimensions.y*local.z);
  if(first>=range.x+range.y){return;}
  ${image}[at+1u]=first;
  ${image}[at+2u]=0x80000000u|counts.x|(counts.y<<5u)|(counts.z<<10u);
  ${image}[at+3u]=dimensions.x|((dimensions.x*dimensions.y)<<16u);
}
fn cm12TeiWriteSpatialTile(slot:u32,tile:u32,generation:u32,candidate:bool){
  let at=cm12TeiSpatialTileBase(slot)+CM12_TEI_SPATIAL_TILE_WORDS*tile;
  ${image}[at]=generation;${image}[at+1u]=CM12_TEI_INVALID;
  ${image}[at+2u]=0u;${image}[at+3u]=0u;
  let brick=tile/CM12_TEI_SPATIAL_TILES_PER_LEAF;
  let leafTile=tile%CM12_TEI_SPATIAL_TILES_PER_LEAF;
  let logicalLocal=leafTile/CM12_TEI_SPATIAL_TILES_PER_LOGICAL;
  let localTile=leafTile%CM12_TEI_SPATIAL_TILES_PER_LOGICAL;
  let span=brickSpan(brick);
  if(logicalLocal>=span*span*span){return;}
  let lz=logicalLocal/(span*span);let rem=logicalLocal-lz*span*span;
  let ly=rem/span;let lx=rem-ly*span;
  let logicalCoordinate=cm12WorldLeafCoordinate(brick)+vec3i(vec3u(lx,ly,lz));
  let tx=localTile%CM12_TEI_SPATIAL_TILES_PER_AXIS;
  let ty=(localTile/CM12_TEI_SPATIAL_TILES_PER_AXIS)%CM12_TEI_SPATIAL_TILES_PER_AXIS;
  let tz=localTile/(CM12_TEI_SPATIAL_TILES_PER_AXIS*CM12_TEI_SPATIAL_TILES_PER_AXIS);
  let origin=logicalCoordinate*i32(BRICK_FINE_RESOLUTION)+vec3i(4u*vec3u(tx,ty,tz));
  if(brick>=CM12_TEI_LEAF_CAPACITY
    ||!select(brickActive(brick),scheduledBrickActive(brick),candidate)
    ||(!EXP_DYNAMIC_WORLD_GROWTH&&(any(origin<vec3i(0))
      ||any(vec3u(origin)>=p.dimensions.xyz)))||origin.y<0){return;}
  var resolution=acceptedBrickResolution(brick);
  if(candidate){resolution=scheduledBrickResolution(brick);}
  let spanFine=BRICK_FINE_RESOLUTION*brickSpan(brick);let scale=spanFine/resolution;
  let leafOrigin=cm12WorldLeafCoordinate(brick)*i32(BRICK_FINE_RESOLUTION);
  let extent=select(min(vec3u(spanFine),p.dimensions.xyz-min(vec3u(leafOrigin),p.dimensions.xyz)),
    vec3u(spanFine),EXP_DYNAMIC_WORLD_GROWTH&&brick>=CM12_WDR_INITIAL_LEAVES);
  let dimensions=(extent+vec3u(scale-1u))/scale;
  let home=vec3u(origin-leafOrigin)/scale;
  let packetAxis=max(1u,(resolution+3u)/CM12_TEI_PACKET_EDGE);
  let packetCoordinate=home/CM12_TEI_PACKET_EDGE;
  if(any(packetCoordinate>=vec3u(packetAxis))){return;}
  let localPacket=packetCoordinate.x+packetAxis
    *(packetCoordinate.y+packetAxis*packetCoordinate.z);
  let packetId=brick*CM12_TEI_PACKETS_PER_LEAF+localPacket;
  ${image}[at+1u]=packetId;
  var mask=vec2u(0u);
  for(var lane=0u;lane<64u;lane+=1u){
    let q=origin+vec3i(vec3u(lane&3u,(lane>>2u)&3u,lane>>4u));
    if((!EXP_DYNAMIC_WORLD_GROWTH&&(any(q<vec3i(0))
      ||any(vec3u(q)>=p.dimensions.xyz)))||q.y<0){continue;}
    let relative=vec3u(q-leafOrigin);
    if(any(relative%vec3u(scale)!=vec3u(0u))){continue;}
    let local=relative/scale;if(any(local>=dimensions)){continue;}
    let cellPacket=local/CM12_TEI_PACKET_EDGE;
    let cellPacketId=cellPacket.x+packetAxis*(cellPacket.y+packetAxis*cellPacket.z);
    if(cellPacketId!=localPacket){continue;}
    let cellLane=(local.x&3u)+4u*((local.y&3u)+4u*(local.z&3u));
    mask[cellLane>>5u]|=1u<<(cellLane&31u);
  }
  ${image}[at+2u]=mask.x;${image}[at+3u]=mask.y;
}
fn cm12TeiWriteSlotHeader(slot:u32,generation:u32){
  let at=cm12TeiSlotBase(slot);
  ${image}[at]=generation;${image}[at+1u]=1u;
  ${image}[at+2u]=CM12_TEI_LEAF_CAPACITY;
  ${image}[at+3u]=CM12_TEI_PACKET_CAPACITY;
  ${image}[at+4u]=CM12_TEI_SPATIAL_TILE_CAPACITY;
}

// One workgroup owns one changed leaf.  Its 64 lanes are exactly the leaf's
// stable packet arena; the same lanes cover every finest spatial tile owned by
// the leaf.  No unchanged leaf or world-sized interior is visited.
fn cm12TeiCompileTopologyDelta(lane:u32,rank:u32,candidate:bool){
  let brick=topologyDeltaLeafInvocation(rank);if(brick==CM12_TEI_INVALID){return;}
  let slot=shadowTopologySlot();
  let generation=select(atomicLoad(&activity[12]),
    atomicLoad(&topologyArena[topologyWorklistBase()+1u]),candidate);
  if(rank==0u&&lane==0u){cm12TeiWriteSlotHeader(slot,generation);}
  if(lane==0u){cm12TeiWriteLeaf(slot,brick,generation,candidate);}
  cm12TeiWritePacket(slot,brick*CM12_TEI_PACKETS_PER_LEAF+lane,generation,candidate);

  let span=brickSpan(brick);
  let tileCount=min(span*span*span,CM12_TEI_LOGICAL_SLOTS_PER_LEAF)
    *CM12_TEI_SPATIAL_TILES_PER_LOGICAL;
  for(var within=lane;within<tileCount;within+=64u){
    cm12TeiWriteSpatialTile(slot,
      brick*CM12_TEI_SPATIAL_TILES_PER_LEAF+within,generation,candidate);
  }
}

@compute @workgroup_size(64)
fn compileSparseCM12TransportExecutionImageShadow(
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  cm12TeiCompileTopologyDelta(lane,wid.x,true);
}

@compute @workgroup_size(64)
fn replaySparseCM12TransportExecutionImageRetired(
 @builtin(local_invocation_index)lane:u32,@builtin(workgroup_id)wid:vec3u){
  // After acceptance, shadowTopologySlot names the retired slot; after
  // rejection it still names the isolated candidate slot.  Replaying accepted
  // records therefore both seeds the next shadow and rolls a rejected patch
  // back to an exact mirror without a second selector or full refresh.
  cm12TeiCompileTopologyDelta(lane,wid.x,false);
}
`;
}
