import {
  SPARSE_CM12_DIRTY_FACE_ROW_MASK_FAMILY_COUNT,
  SPARSE_CM12_DIRTY_FACE_ROW_MASK_WORDS_PER_PACKET,
  type SparseCM12DirtyFaceRowMaskLayout,
} from "./sparse-cm12-dirty-face-row-masks";

/** DFRM1 compiler and direct projection consumer. */
export function createSparseCM12DirtyFaceRowMaskWGSL(options: Readonly<{
  layout: SparseCM12DirtyFaceRowMaskLayout;
  arenaName?: string;
}>): string {
  const layout = options.layout;
  const arena = options.arenaName ?? "activity";
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(arena)) {
    throw new TypeError("DFRM1 arena name");
  }
  return /* wgsl */ `
const DFRM1_INVALID:u32=0xffffffffu;
const DFRM1_MASK_BASE:u32=${layout.maskBaseWords}u;
const DFRM1_ACCEPTED_PRESSURE_BITS:u32=${layout.acceptedPressureBitsBaseWords}u;
const DFRM1_CELL_CAPACITY:u32=${layout.cellCapacity}u;
const DFRM1_PACKET_CAPACITY:u32=${layout.packetCapacity}u;
const DFRM1_DISPATCH_PACKETS_PER_LEAF:u32=${layout.dispatchPacketsPerLeaf}u;
const DFRM1_DISPATCH_PACKET_COUNT:u32=${layout.dispatchPacketCount}u;
const DFRM1_DISPATCH_WIDTH:u32=${layout.dispatchWidth}u;
const DFRM1_FAMILY_COUNT:u32=${SPARSE_CM12_DIRTY_FACE_ROW_MASK_FAMILY_COUNT}u;
const DFRM1_WORDS_PER_PACKET:u32=${SPARSE_CM12_DIRTY_FACE_ROW_MASK_WORDS_PER_PACKET}u;

var<workgroup> dfrm1Ballot:array<atomic<u32>,12>;

fn dfrm1Ordinal(wid:vec3u)->u32{return wid.x+DFRM1_DISPATCH_WIDTH*wid.y;}
fn dfrm1StablePacket(ordinal:u32)->u32{
  let leaf=ordinal/DFRM1_DISPATCH_PACKETS_PER_LEAF;
  let local=ordinal-leaf*DFRM1_DISPATCH_PACKETS_PER_LEAF;
  return 64u*leaf+local;
}
fn dfrm1CompactOrdinal(packet:u32)->u32{
  let leaf=packet/64u;let local=packet-leaf*64u;
  if(local>=DFRM1_DISPATCH_PACKETS_PER_LEAF){return DFRM1_INVALID;}
  let ordinal=leaf*DFRM1_DISPATCH_PACKETS_PER_LEAF+local;
  return select(DFRM1_INVALID,ordinal,ordinal<DFRM1_DISPATCH_PACKET_COUNT);
}
fn dfrm1MaskWord(ordinal:u32,family:u32,lane:u32)->u32{
  return DFRM1_MASK_BASE+DFRM1_WORDS_PER_PACKET*ordinal
    +2u*family+select(0u,1u,lane>=32u);
}
fn dfrm1RowSelected(address:vec2u)->bool{return address.x!=DFRM1_INVALID
  &&rowAccepted(address.x)&&incrementalActivityBrickDirty(address.y);}
fn dfrm1MarkAddress(address:vec2u,family:u32){
  if(address.x==DFRM1_INVALID||family>=DFRM1_FAMILY_COUNT){return;}
  let ordinal=dfrm1CompactOrdinal(address.x);if(ordinal==DFRM1_INVALID){return;}
  _=atomicOr(&${arena}[dfrm1MaskWord(ordinal,family,address.y)],
    1u<<(address.y&31u));
}
fn dfrm1MarkRow(row:u32){
  // The classifier calls this before publishing the candidate PCM bit.  Gate
  // on immutable topology acceptance here; the projector applies finalized
  // pressure membership before touching a row.
  if(row==DFRM1_INVALID||!rowAccepted(row)){return;}
  let begin=rowTermOffset(row);let end=begin+rowTermCount(row);
  var ownerCell=DFRM1_INVALID;var family=rowAxis(row);
  for(var term=begin;term<end;term+=1u){if(termCoefficient(term)>0.0){
    ownerCell=termCell(term);break;}}
  if(ownerCell==DFRM1_INVALID){
    if(rowKind(row)!=3u||end!=begin+1u||termCoefficient(begin)>=0.0){return;}
    ownerCell=termCell(begin);family+=3u;
  }
  dfrm1MarkAddress(cm12FinalScalarPacketAddress(ownerCell),family);
}

@compute @workgroup_size(64)
fn compileSparseCM12DirtyFaceRowMasks(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let ordinal=dfrm1Ordinal(wid);if(ordinal>=DFRM1_DISPATCH_PACKET_COUNT){return;}
  if(lane<DFRM1_WORDS_PER_PACKET){atomicStore(&dfrm1Ballot[lane],0u);}workgroupBarrier();
  let packet=dfrm1StablePacket(ordinal);
  if(packet<DFRM1_PACKET_CAPACITY){
    for(var axis=0u;axis<3u;axis+=1u){
      let address=itr1StableRowAndBucketOwner(packet,axis,lane);
      if(dfrm1RowSelected(address)){
        prepareTransportFaceRow(address.x);
        let word=2u*axis+select(0u,1u,lane>=32u);
        _=atomicOr(&dfrm1Ballot[word],1u<<(lane&31u));
      }
      let sparseAir=itr1StablePositiveSparseAirRowAndBucketOwner(packet,axis,lane);
      if(dfrm1RowSelected(sparseAir)){
        prepareTransportFaceRow(sparseAir.x);
        let word=2u*(3u+axis)+select(0u,1u,lane>=32u);
        _=atomicOr(&dfrm1Ballot[word],1u<<(lane&31u));
      }
    }
  }
  workgroupBarrier();
  if(lane<DFRM1_WORDS_PER_PACKET){atomicStore(&${arena}[DFRM1_MASK_BASE
      +DFRM1_WORDS_PER_PACKET*ordinal+lane],atomicLoad(&dfrm1Ballot[lane]));}
}

@compute @workgroup_size(64)
fn markSparseCM12DirtyFaceRowsFromPressure(@builtin(global_invocation_id)gid:vec3u){
  let cell=pressureCellInvocation(gid.x);if(cell==DFRM1_INVALID||cell>=DFRM1_CELL_CAPACITY){return;}
  let pressureBits=bitcast<u32>(state[p.stateOffsets2.x+cell]);
  let previous=atomicExchange(&${arena}[DFRM1_ACCEPTED_PRESSURE_BITS+cell],pressureBits);
  if(previous==pressureBits){return;}
  for(var incidence=incidenceBegin(cell);incidence<incidenceEnd(cell);incidence+=1u){
    dfrm1MarkRow(incidenceRow(incidence));
  }
}

@compute @workgroup_size(64)
fn projectSparseCM12DirtyFaceRows(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let ordinal=dfrm1Ordinal(wid);if(ordinal>=DFRM1_DISPATCH_PACKET_COUNT){return;}
  let packet=dfrm1StablePacket(ordinal);if(packet>=DFRM1_PACKET_CAPACITY){return;}
  if(lane<DFRM1_WORDS_PER_PACKET){atomicStore(&dfrm1Ballot[lane],atomicLoad(&${arena}[
      DFRM1_MASK_BASE+DFRM1_WORDS_PER_PACKET*ordinal+lane]));}
  workgroupBarrier();
  for(var family=0u;family<DFRM1_FAMILY_COUNT;family+=1u){
    let mask=atomicLoad(&dfrm1Ballot[2u*family+select(0u,1u,lane>=32u)]);
    if(((mask>>(lane&31u))&1u)==0u){continue;}
    let axis=family%3u;var row=DFRM1_INVALID;
    if(family<3u){row=itr1StableRowForOwner(packet,axis,lane);
    }else{row=itr1StablePositiveSparseAirRowAndBucketOwner(packet,axis,lane).x;}
    if(row!=DFRM1_INVALID&&pcmRowContains(row)){
      projectPressureRow(row);
      state[sourceFaceVelocity()+row]=state[destinationFaceVelocity()+row];
    }
  }
}
`;
}
