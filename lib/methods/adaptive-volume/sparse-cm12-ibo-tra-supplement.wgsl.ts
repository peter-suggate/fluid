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
  let at=ITR1_DIRECTORY+5u*templateId;
  return vec4u(itr1Load(at),itr1Load(at+1u),itr1Load(at+2u),itr1Load(at+3u));
}
fn itr1SparseAirOwnerBase(templateId:u32)->u32{
  if(templateId>=ITR1_TEMPLATE_COUNT){return ITR1_INVALID;}
  return itr1Load(ITR1_DIRECTORY+5u*templateId+4u);
}
fn itr1Boundary(local:vec3u,axis:u32,resolution:u32)->u32{
  let u=select(local.x,local.y,axis==0u);
  let v=select(local.z,local.y,axis==2u);
  return u+resolution*v;
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
// BFA1's ordinary seam catalogue contains the positive row owner. For a
// cross-leaf face that owner is necessarily on local coordinate zero, so the
// relevant IBO reference is the negative side of the known axis. Keep this
// direct path separate from the general owner query: asking a six-side search
// to rediscover a topology address already encoded by BFA1 creates a large
// Metal optimizer graph for every seam preparation/projection pipeline.
fn itr1StableNegativeBoundaryRowForOwner(packet:u32,axis:u32,lane:u32)->u32{
  let slot=${p}IBOAcceptedSlot();let address=${p}IBOTRAPacketLocal(packet,lane,slot);
  if(address.w==ITR1_INVALID||axis>=3u||address[axis]!=0u){return ITR1_INVALID;}
  let local=address.xyz;let leaf=address.w;let side=2u*axis;
  let count=${p}IBOFaceRefCount(slot,leaf,side);
  for(var refLocal=0u;refLocal<count;refLocal+=1u){
    let faceRef=${p}IBORef(slot,leaf,side,refLocal);
    let directory=itr1Directory(faceRef.x);let boundary=itr1Boundary(local,axis,directory.w);
    let localRow=itr1Load(ITR1_BASE+directory.z+boundary);
    if(localRow!=ITR1_INVALID){return faceRef.z+
      ${p}IBOTemplateRowWord(faceRef.x,localRow,0u);}
  }
  return ITR1_INVALID;
}
fn itr1StablePositiveSparseAirRowAndBucketOwner(
 packet:u32,axis:u32,lane:u32)->vec2u{
  let slot=${p}IBOAcceptedSlot();let address=${p}IBOTRAPacketLocal(packet,lane,slot);
  if(address.w==ITR1_INVALID){return vec2u(ITR1_INVALID);}
  let local=address.xyz;let leaf=address.w;let dims=${p}IBOLeafDimensions(slot,leaf);
  if(local[axis]+1u!=dims[axis]){return vec2u(ITR1_INVALID);}
  let side=2u*axis+1u;let count=${p}IBOFaceRefCount(slot,leaf,side);
  for(var refLocal=0u;refLocal<count;refLocal+=1u){
    let faceRef=${p}IBORef(slot,leaf,side,refLocal);
    if(faceRef.y!=ITR1_INVALID){continue;}
    let directory=itr1Directory(faceRef.x);let boundary=itr1Boundary(local,axis,directory.w);
    let sparseAirBase=itr1SparseAirOwnerBase(faceRef.x);
    if(sparseAirBase==ITR1_INVALID){continue;}
    let localRow=itr1Load(ITR1_BASE+sparseAirBase+boundary);
    if(localRow!=ITR1_INVALID){return vec2u(
      faceRef.z+${p}IBOTemplateRowWord(faceRef.x,localRow,0u),leaf);}
  }
  return vec2u(ITR1_INVALID);
}
`;
}
