import { type SparseCM12IboTRASupplementLayout } from
  "./sparse-cm12-ibo-tra-supplement";

const identifier = (value: string) => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError("ITR1 invalid identifier");
  return value;
};

/** Exact IBO row mapping and compact direct gamma-mask execution. */
export function createSparseCM12IboTRASupplementWGSL(options: Readonly<{
  layout: SparseCM12IboTRASupplementLayout;
  arenaName?: string;
  hookPrefix?: string;
  /** Absolute shared-arena word base of the composed IBO image. */
  baseWords?: number;
}>): string {
  const l = options.layout, arena = identifier(options.arenaName ?? "topologyArena");
  const p = identifier(options.hookPrefix ?? "cm12Resident");
  const base = options.baseWords ?? 0;
  if (!Number.isSafeInteger(base) || base < 0
    || base + l.totalWords > 0xffff_ffff) throw new RangeError("ITR1 baseWords is invalid");
  return /* wgsl */ `
const ITR1_INVALID:u32=0xffffffffu;
const ITR1_BASE:u32=${base}u;
const ITR1_DIRECTORY:u32=${base + l.directoryBaseWords}u;
const ITR1_TEMPLATE_COUNT:u32=${l.templateCount}u;
fn itr1Load(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn itr1Directory(templateId:u32)->vec4u{
  if(templateId>=ITR1_TEMPLATE_COUNT){return vec4u(ITR1_INVALID);}
  let at=ITR1_DIRECTORY+4u*templateId;
  return vec4u(itr1Load(at),itr1Load(at+1u),itr1Load(at+2u),itr1Load(at+3u));
}
fn itr1Plane(axis:u32,q:u32)->vec2u{
  if(axis==0u){let word=0x11111111u<<q;return vec2u(word);}
  if(axis==1u){let word=0x000f000fu<<(4u*q);return vec2u(word);}
  if(q==0u){return vec2u(0x0000ffffu,0u);}
  if(q==1u){return vec2u(0xffff0000u,0u);}
  if(q==2u){return vec2u(0u,0x0000ffffu);}
  return vec2u(0u,0xffff0000u);
}
fn itr1MaskWithout(mask:vec2u,plane:vec2u)->vec2u{return mask&~plane;}
fn itr1ShiftWithin(mask:vec2u,axis:u32)->vec2u{
  if(axis==0u){return vec2u(mask.x<<1u,mask.y<<1u);}
  if(axis==1u){return vec2u(mask.x<<4u,mask.y<<4u);}
  return vec2u(mask.x<<16u,(mask.y<<16u)|(mask.x>>16u));
}
fn itr1ShiftCross(mask:vec2u,axis:u32)->vec2u{
  if(axis==0u){return vec2u(mask.x>>3u,mask.y>>3u);}
  if(axis==1u){return vec2u(mask.x>>12u,mask.y>>12u);}
  return vec2u(mask.y>>16u,0u);
}
fn itr1Bit(lane:u32)->vec2u{return vec2u(select(0u,1u<<(lane&31u),lane<32u),
  select(0u,1u<<(lane&31u),lane>=32u));}
fn itr1LocalFromOrdinal(ordinal:u32,dims:vec3u)->vec3u{
  let z=ordinal/(dims.x*dims.y);let remain=ordinal-z*dims.x*dims.y;
  let y=remain/dims.x;return vec3u(remain-y*dims.x,y,z);
}
fn itr1Boundary(local:vec3u,axis:u32,resolution:u32)->u32{
  let u=select(local.x,local.y,axis==0u);
  let v=select(local.z,local.y,axis==2u);
  return u+resolution*v;
}
fn itr1ApplyCanonical(packet:u32,mask:vec2u,slot:u32){
  let d=${p}IBOTRAPacketDescriptor(packet,slot);if(d.x==ITR1_INVALID){return;}
  let resolution=d.y;let dims=vec3u(d.z&1023u,(d.z>>10u)&1023u,(d.z>>20u)&1023u);
  let packetAxis=max(1u,(resolution+3u)/4u);let localPacket=d.w;
  let pz=localPacket/(packetAxis*packetAxis);let remain=localPacket-pz*packetAxis*packetAxis;
  let py=remain/packetAxis;let px=remain-py*packetAxis;let origin=4u*vec3u(px,py,pz);
  for(var axis=0u;axis<3u;axis+=1u){
    var own=mask;if(origin[axis]==0u){own=itr1MaskWithout(own,itr1Plane(axis,0u));}
    var positive=mask;let last=dims[axis]-origin[axis]-1u;
    if(origin[axis]+4u>=dims[axis]){positive=itr1MaskWithout(positive,itr1Plane(axis,last));}
    let within=itr1ShiftWithin(itr1MaskWithout(positive,itr1Plane(axis,3u)),axis);
    cm12TransportMarkGammaMask(packet,axis,own.x|within.x,own.y|within.y);
    if(origin[axis]+4u<dims[axis]){let cross=itr1ShiftCross(positive&itr1Plane(axis,3u),axis);
      let stride=select(select(1u,packetAxis,axis==1u),packetAxis*packetAxis,axis==2u);
      cm12TransportMarkGammaMask(packet+stride,axis,cross.x,cross.y);}
  }
}
fn itr1ApplyBoundary(packet:u32,mask:vec2u,worker16:u32,slot:u32){
  for(var lane=worker16;lane<64u;lane+=16u){
    if(((mask[lane>>5u]>>(lane&31u))&1u)==0u){continue;}
    let address=${p}IBOTRAPacketLocal(packet,lane,slot);if(address.w==ITR1_INVALID){continue;}
    let local=address.xyz;let leaf=address.w;let dims=${p}IBOLeafDimensions(slot,leaf);
    for(var side=0u;side<6u;side+=1u){let axis=side>>1u;
      if(local[axis]!=select(0u,dims[axis]-1u,(side&1u)!=0u)){continue;}
      let refCount=${p}IBOFaceRefCount(slot,leaf,side);
      for(var refLocal=0u;refLocal<refCount;refLocal+=1u){let faceRef=${p}IBORef(slot,leaf,side,refLocal);
        let directory=itr1Directory(faceRef.x);let boundary=itr1Boundary(local,axis,directory.w);
        let begin=itr1Load(ITR1_BASE+directory.x+boundary);
        let end=itr1Load(ITR1_BASE+directory.x+boundary+1u);
        for(var at=begin;at<end;at+=1u){let entry=itr1Load(ITR1_BASE+directory.y+at);
          let localRow=entry&0xfffu;let ownerTerm=entry>>12u;
          if(ownerTerm==0xfu){continue;}
          let packed=${p}IBOTemplateRowWord(faceRef.x,localRow,1u);let first=packed&0x007fffffu;
          let normalized=${p}IBOTemplateTermWord(faceRef.x,first+ownerTerm,0u);
          let ownerLeaf=select(leaf,faceRef.y,(normalized&0x80000000u)!=0u);
          let ownerDims=${p}IBOLeafDimensions(slot,ownerLeaf);
          let ownerLocal=itr1LocalFromOrdinal(normalized&0x7fffffffu,ownerDims);
          let owner=${p}IBOTRAPacketForLocal(ownerLeaf,ownerLocal,slot);
          let rowAxis=${p}IBOTemplateRowWord(faceRef.x,localRow,2u)>>30u;
          cm12TransportMarkGammaMask(
            owner.x,rowAxis,itr1Bit(owner.y).x,itr1Bit(owner.y).y);
        }} }
  }
}
@compute @workgroup_size(64)
fn compileSparseCM12GammaRowMasks(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  let packet=cm12TransportPacketId(wid.x,2u);
  if(packet==ITR1_INVALID){return;}
  let mask=tpm1SurfaceMask(packet);let slot=${p}IBOAcceptedSlot();
  if(lane==0u){itr1ApplyCanonical(packet,mask,slot);}
  if(lane<16u){itr1ApplyBoundary(packet,mask,lane,slot);}
}
fn itr1StableRowAndBucketOwner(packet:u32,axis:u32,lane:u32)->vec2u{
  let slot=${p}IBOAcceptedSlot();let address=${p}IBOTRAPacketLocal(packet,lane,slot);
  if(address.w==ITR1_INVALID){return vec2u(ITR1_INVALID);}let local=address.xyz;let leaf=address.w;
  let descriptor=${p}IBOLeafDescriptorId(slot,leaf);
  if(local[axis]>0u){let dims=${p}IBOLeafDimensions(slot,leaf);
    if(axis==0u){return vec2u(${p}IBOCanonicalRowBase(descriptor,0u)+(local.x-1u)
      +(dims.x-1u)*(local.y+dims.y*local.z),leaf);}
    if(axis==1u){return vec2u(${p}IBOCanonicalRowBase(descriptor,1u)+local.x
      +dims.x*((local.y-1u)+(dims.y-1u)*local.z),leaf);}
    return vec2u(${p}IBOCanonicalRowBase(descriptor,2u)+local.x
      +dims.x*(local.y+dims.y*(local.z-1u)),leaf);}
  let dims=${p}IBOLeafDimensions(slot,leaf);
  for(var side=0u;side<6u;side+=1u){if((side>>1u)!=axis
      ||local[axis]!=select(0u,dims[axis]-1u,(side&1u)!=0u)){continue;}
    let count=${p}IBOFaceRefCount(slot,leaf,side);
    for(var refLocal=0u;refLocal<count;refLocal+=1u){let faceRef=${p}IBORef(slot,leaf,side,refLocal);
      let directory=itr1Directory(faceRef.x);let boundary=itr1Boundary(local,axis,directory.w);
      let localRow=itr1Load(ITR1_BASE+directory.z+boundary);if(localRow!=ITR1_INVALID){
        return vec2u(faceRef.z+${p}IBOTemplateRowWord(faceRef.x,localRow,0u),min(leaf,faceRef.y));}}
  }return vec2u(ITR1_INVALID);
}
fn itr1StableRowForOwner(packet:u32,axis:u32,lane:u32)->u32{
  return itr1StableRowAndBucketOwner(packet,axis,lane).x;
}
fn itr1StablePositiveSparseAirRowAndBucketOwner(
 packet:u32,axis:u32,lane:u32)->vec2u{
  let slot=${p}IBOAcceptedSlot();let address=${p}IBOTRAPacketLocal(packet,lane,slot);
  if(address.w==ITR1_INVALID){return vec2u(ITR1_INVALID);}
  let local=address.xyz;let leaf=address.w;let dims=${p}IBOLeafDimensions(slot,leaf);
  if(local[axis]+1u!=dims[axis]){return vec2u(ITR1_INVALID);}
  let side=2u*axis+1u;let count=${p}IBOFaceRefCount(slot,leaf,side);
  let ordinal=local.x+dims.x*(local.y+dims.y*local.z);
  for(var refLocal=0u;refLocal<count;refLocal+=1u){
    let faceRef=${p}IBORef(slot,leaf,side,refLocal);
    if(faceRef.y!=ITR1_INVALID){continue;}
    let directory=itr1Directory(faceRef.x);let boundary=itr1Boundary(local,axis,directory.w);
    let begin=itr1Load(ITR1_BASE+directory.x+boundary);
    let end=itr1Load(ITR1_BASE+directory.x+boundary+1u);
    for(var at=begin;at<end;at+=1u){let entry=itr1Load(ITR1_BASE+directory.y+at);
      let localRow=entry&0xfffu;
      let metadata=${p}IBOTemplateRowWord(faceRef.x,localRow,2u);
      if(((metadata>>28u)&3u)!=3u){continue;}
      let packed=${p}IBOTemplateRowWord(faceRef.x,localRow,1u);
      let first=packed&0x007fffffu;
      let normalized=${p}IBOTemplateTermWord(faceRef.x,first,0u);
      let coefficient=bitcast<f32>(${p}IBOTemplateTermWord(faceRef.x,first,1u));
      if(normalized==ordinal&&coefficient<0.0){
        return vec2u(faceRef.z+${p}IBOTemplateRowWord(faceRef.x,localRow,0u),leaf);
      }
    }
  }
  return vec2u(ITR1_INVALID);
}
var<workgroup>itr1GammaRowMask:array<u32,6>;
fn itr1ScatterGammaMask(wid:vec3u,lane:u32,inputRho:u32,inputGamma:u32){
  let ordinal=cm12TransportDirectOrdinal(wid);
  if(ordinal>=CM12_TPA_DIRECT_PACKET_COUNT){return;}
  if(lane<6u){itr1GammaRowMask[lane]=atomicLoad(
    &activity[CM12_TPA_GAMMA_ROW_MASK+6u*ordinal+lane]);}
  workgroupBarrier();let packet=cm12TransportDirectStablePacket(ordinal);
  for(var axis=0u;axis<3u;axis+=1u){
    let mask=itr1GammaRowMask[2u*axis+select(0u,1u,lane>=32u)];
    if(((mask>>(lane&31u))&1u)==0u){continue;}
    let row=itr1StableRowForOwner(packet,axis,lane);
    if(row!=ITR1_INVALID){scatterGammaRow(row,inputRho,inputGamma);}
  }
}
@compute @workgroup_size(64)
fn scatterSparseCM12GammaSnapshotRows(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  itr1ScatterGammaMask(wid,lane,destinationDensity(),destinationGamma());
}
@compute @workgroup_size(64)
fn scatterSparseCM12GammaRefinementRows(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  itr1ScatterGammaMask(wid,lane,p.stateOffsets2.x,p.stateOffsets2.y);
}
`;
}
