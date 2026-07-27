/** Class-specialized direct structured-velocity transport for sparse fine phi. */
import {
  OCTREE_AIR_SUPPORT_LAYOUT_VERSION,
  OCTREE_AIR_SUPPORT_TAG,
  OCTREE_AIR_SUPPORT_VALID,
} from "./webgpu-octree-air-velocity-support";

export const structuredFineLevelSetTransportWGSL = /* wgsl */ `
struct P {
  brickDims:vec3u,r:u32,sampleDims:vec3u,samplesPerBrick:u32,origin:vec3f,h:f32,
  pageCapacity:u32,generation:u32,segments:u32,maxLeaf:u32,dimensions:vec3u,rowCapacity:u32,
  physical:f32,dt:f32,bandCells:u32,closed:u32,openTop:u32,maxSlots:u32,authorityWords:u32,rowStride:u32,
  valuesOffset:u32,ownerOffset:u32,neighborOffset:u32,metadataOffset:u32,areaOffset:u32,inverseOffset:u32,
  fractionOffset:u32,pressureScaleOffset:u32,normalOffset:u32,centroidOffset:u32,rowNeighborOffset:u32,
  rowReciprocalOffset:u32,rowOwnerMetadataOffset:u32,rowHandleOffset:u32,rowSignOffset:u32,
  rowCatalogOffset:u32,rowAxisOffset:u32,rowFamilyPrefixOffset:u32,rowFamilyHandleOffset:u32,
  rowFamilySlotOffset:u32,tetraVertexCount:u32,maxBacktrace:u32,airOwnerOffset:u32,volumeOffset:u32,
  slotGeometryOffset:u32,tetraHeaderOffset:u32,tetraVertexOffset:u32,tetraOffset:u32,templateHeaderOffset:u32,
  airSupportEnabled:u32,selectorTagOffset:u32,regularTagOffset:u32,airControlOffset:u32,
  supportVectorOffset:u32,supportCapacity:u32,selectorStride:u32
}
struct Control { outside:u32,nonfinite:u32,processed:u32,committed:u32,extended:u32,maxDisplacement:u32,
  authorityUnavailable:u32,velocityUnavailable:u32,invalidStatus:u32,nonpositive:u32,reasons:u32,
  firstStatus:u32,firstIndex:u32,firstX:u32,firstY:u32,firstZ:u32 }
struct SlotGeometry { neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f }
struct AirOwner {tag:u32,cell:u32,size:u32,caseTransform:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read_write>metadata:array<u32>;
@group(0)@binding(2)var<storage,read>worklist:array<u32>;
@group(0)@binding(3)var<storage,read>flags:array<u32>;
@group(0)@binding(4)var<storage,read_write>phi:array<f32>;
@group(0)@binding(5)var<storage,read_write>nextPhi:array<f32>;
@group(0)@binding(7)var<storage,read_write>control:Control;
@group(0)@binding(8)var<storage,read_write>delta:array<u32>;
@group(0)@binding(9)var<storage,read>accepted:array<u32>;
@group(0)@binding(12)var<storage,read>rowVelocity:array<vec4f>;
@group(0)@binding(13)var<storage,read_write>state:array<u32>;
@group(0)@binding(6)var<storage,read_write>catalog:array<u32>;
@group(0)@binding(14)var<storage,read>metrics:array<vec4u>;
@group(0)@binding(20)var<storage,read>airSupport:array<vec4f>;
@group(0)@binding(21)var<storage,read>boundary:array<u32>;

const INVALID:u32=0xffffffffu; const VALID:u32=1u;
const PAGE_INTERFACE:u32=2u; const PAGE_DIRTY:u32=8u;
const REGULAR_COMMON:u32=0u; const TRANSITION_COMMON:u32=1u;
const REGULAR_RARE:u32=2u; const TRANSITION_RARE:u32=3u;
const HEADER_WORDS:u32=7u; const HEADER_BASE:u32=4u; const STATUS_BASE:u32=128u;
const SUPPORT_TAG:u32=${OCTREE_AIR_SUPPORT_TAG}u;
const SUPPORT_VALID:u32=${OCTREE_AIR_SUPPORT_VALID}u;
const SUPPORT_VERSION:u32=${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u;
// FLT_MAX is the fail-closed value returned by sampleFine. Keep the bound
// strict so a missing sparse trilinear stencil cannot be committed as phi.
fn finite(v:f32)->bool{return v==v&&abs(v)<3.402823e38;}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(3.402823e38));}
fn inTransportBand(value:f32)->bool{return finite(value)&&abs(value)<=f32(p.bandCells)*p.h;}
fn structuredValid()->bool{return arrayLength(&accepted)>=6u&&accepted[0]==0u&&accepted[2]>0u&&accepted[3]!=0u&&accepted[4]<=1u;}
fn activeBank()->u32{return accepted[4]&1u;} fn rbase()->u32{return state[33u]*p.rowStride;}
fn airWord(word:u32)->u32{if(word>=4u*arrayLength(&airSupport)){return INVALID;}return bitcast<vec4u>(airSupport[word/4u])[word&3u];}
fn cellCoord(cell:u32)->vec3u{return vec3u(cell%p.dimensions.x,(cell/p.dimensions.x)%p.dimensions.y,
  cell/(p.dimensions.x*p.dimensions.y));}
fn airOwner(cell:u32)->AirOwner{if(cell>=p.dimensions.x*p.dimensions.y*p.dimensions.z){return AirOwner(INVALID,INVALID,0u,INVALID);}
  let at=p.airOwnerOffset+4u*cell;if(at+3u>=4u*arrayLength(&airSupport)){return AirOwner(INVALID,INVALID,0u,INVALID);}
  let result=AirOwner(airWord(at),airWord(at+1u),airWord(at+2u),airWord(at+3u));let supportCount=airWord(p.airControlOffset+6u);
  let tagValid=result.tag!=INVALID&&select(result.tag<state[32],(result.tag&0x7fffffffu)<supportCount,(result.tag&SUPPORT_TAG)!=0u);
  if(tagValid&&result.cell<p.dimensions.x*p.dimensions.y*p.dimensions.z&&result.size>0u&&result.caseTransform!=INVALID){return result;}
  return AirOwner(INVALID,INVALID,0u,INVALID);}
fn ownerCellAtPosition(x:vec3f)->u32{let q=vec3u(clamp(floor(x/p.physical),vec3f(0),vec3f(p.dimensions)-vec3f(1)));
  return q.x+p.dimensions.x*(q.y+p.dimensions.y*q.z);}
fn airOwnerAtPosition(x:vec3f)->AirOwner{return airOwner(ownerCellAtPosition(x));}
fn airPublicationValid()->bool{let at=p.airControlOffset;if(p.airSupportEnabled!=1u||at+16u>4u*arrayLength(&airSupport)
  ||p.airOwnerOffset+4u*p.dimensions.x*p.dimensions.y*p.dimensions.z>4u*arrayLength(&airSupport)
  ||arrayLength(&boundary)<7u||boundary[0]!=0u||boundary[1]!=INVALID||boundary[2]!=accepted[2]
  ||boundary[4]!=accepted[3]||boundary[5]!=activeBank()||boundary[6]!=boundary[4]){return false;}
  let count=airWord(at+6u);let capacity=airWord(at+7u);let faces=airWord(at+10u);let seeds=airWord(at+11u);
  return airWord(at)==0u&&airWord(at+1u)==INVALID&&airWord(at+2u)==accepted[3]
    &&airWord(at+3u)==activeBank()&&airWord(at+4u)==boundary[4]&&airWord(at+5u)==accepted[2]
    &&capacity==p.supportCapacity&&count<=capacity&&airWord(at+13u)==SUPPORT_VALID
    &&airWord(at+14u)==SUPPORT_VERSION&&airWord(at+15u)==p.generation&&seeds<=faces
    &&(count==0u||(airWord(at+8u)>0u&&faces>0u&&seeds>0u));}
fn statusBase(page:u32)->u32{return STATUS_BASE+3u*page;}
fn payloadBase()->u32{return STATUS_BASE+3u*p.pageCapacity;}
fn headerBase(cls:u32)->u32{return HEADER_BASE+HEADER_WORDS*cls;}

// The codebase's standard widening shape (see the scanSparseSeedRecords /
// scanSparseGroups / offsetSparseGroups family in the fine topology module):
// a 256-lane per-page reduce, one block prefix sum, then a scatter keyed by
// the same page index the retired single-thread loops walked. Because the
// scatter address is a pure function of the page index, the compacted streams
// keep the exact ascending order those loops produced.
//
// Scratch lives past the class payload in the governor arena: SCAN_BLOCK_WORDS
// per 256-page block, then one global record, then one per-item word. Slots
// 0..3 hold the four workset class counts (then their block offsets), 4..10 the
// status reduction, 11 the interface-delta count (then its block offset).
//
// The per-item word carries each item's in-block exclusive prefix from the
// reduce dispatch to the scatter, so the scatter needs no barriers and no
// second classification. The workset and delta compactions are separated by
// the whole transport and commit chain, so they share the one array.
const SCAN_BLOCK_WORDS:u32=16u;
const SCAN_SLOT_OUTSIDE:u32=4u; const SCAN_SLOT_NONFINITE:u32=5u; const SCAN_SLOT_PROCESSED:u32=6u;
const SCAN_SLOT_EXTENDED:u32=7u; const SCAN_SLOT_INVALID:u32=8u; const SCAN_SLOT_DISPLACEMENT:u32=9u;
const SCAN_SLOT_FIRST:u32=10u; const SCAN_SLOT_DELTA:u32=11u;
const SCAN_TOTAL_CLASS_BASE:u32=0u; const SCAN_TOTAL_CLASS_COUNT:u32=4u;
fn scanBlockCount()->u32{return (p.pageCapacity+255u)/256u;}
fn scanBase()->u32{return payloadBase()+p.pageCapacity;}
fn scanBlockWord(block:u32,slot:u32)->u32{return scanBase()+SCAN_BLOCK_WORDS*block+slot;}
fn scanTotalWord(slot:u32)->u32{return scanBase()+SCAN_BLOCK_WORDS*scanBlockCount()+slot;}
fn scanItemWord(item:u32)->u32{return scanTotalWord(SCAN_BLOCK_WORDS)+item;}
var<workgroup> identityScanLanes:array<u32,256>; var<workgroup> identityScanTotal:u32;
// Blelloch exclusive scan over one 256-lane block; the inclusive block total
// stays readable in identityScanTotal, published behind a barrier.
fn scanIdentityBlock(local:u32,value:u32)->u32{
 identityScanLanes[local]=value;workgroupBarrier();var step=1u;
 loop{if(step>=256u){break;}let index=(local+1u)*step*2u-1u;
  if(index<256u){identityScanLanes[index]+=identityScanLanes[index-step];}workgroupBarrier();step*=2u;}
 if(local==255u){identityScanTotal=identityScanLanes[255u];identityScanLanes[255u]=0u;}workgroupBarrier();step=128u;
 loop{let index=(local+1u)*step*2u-1u;if(index<256u){let lower=identityScanLanes[index-step];
   identityScanLanes[index-step]=identityScanLanes[index];identityScanLanes[index]+=lower;}workgroupBarrier();
  if(step==1u){break;}step/=2u;}return identityScanLanes[local];
}

var<workgroup> governorSpeed:array<f32,128>; var<workgroup> governorInvalid:array<u32,128>;
@compute @workgroup_size(128)
fn planStructuredFineTransportSubsteps(@builtin(local_invocation_index)lid:u32) {
  var maximum=0.; var invalid=0u; let valid=structuredValid()&&airPublicationValid();
  let rows=select(0u,min(accepted[2],p.rowCapacity),valid);
  let mapNeeded=valid&&(state[36]!=accepted[3]||state[37]!=activeBank()||state[38]!=p.generation);
  for(var row=lid;row<rows;row+=128u){
    let velocity=rowVelocity[activeBank()*p.rowStride+row];
    if(!finite3(velocity.xyz)||velocity.w<=0.){invalid=1u;}else{maximum=max(maximum,length(velocity.xyz));}
    if(mapNeeded){if(4u*row+3u<arrayLength(&catalog)&&row<arrayLength(&metrics)){
      let at=4u*row;let m=metrics[row];catalog[at]=m.x;catalog[at+1u]=m.y;catalog[at+2u]=m.z;catalog[at+3u]=m.w;
    }else{invalid=1u;}}
  }
  let supportCount=select(0u,airWord(p.airControlOffset+6u),valid);
  for(var support=lid;support<supportCount;support+=128u){let at=p.supportVectorOffset/4u+support;
    if(at>=arrayLength(&airSupport)){invalid=1u;continue;}let velocity=airSupport[at];
    if(!finite3(velocity.xyz)||velocity.w<=0.){invalid=1u;}else{maximum=max(maximum,length(velocity.xyz));}}
  governorSpeed[lid]=maximum; governorInvalid[lid]=invalid; workgroupBarrier();
  for(var width=64u;width>0u;width>>=1u){if(lid<width){
    governorSpeed[lid]=max(governorSpeed[lid],governorSpeed[lid+width]);
    governorInvalid[lid]|=governorInvalid[lid+width];}workgroupBarrier();}
  if(lid==0u){
    let displacement=max(1u,u32(ceil(governorSpeed[0]*p.dt/max(p.h,1e-20))));
    let required=max(max(1u,p.segments),displacement);
    let scheduleValid=valid&&governorInvalid[0]==0u&&required<=64u&&displacement<=p.maxBacktrace;
    let activeSteps=select(0u,required,scheduleValid);
    state[0]=select(1u,0u,scheduleValid); state[1]=activeSteps;
    state[2]=bitcast<u32>(select(0.,p.dt/f32(activeSteps),activeSteps>0u)); state[3]=displacement;
    state[32]=select(0u,accepted[2],scheduleValid); state[33]=select(0u,activeBank(),scheduleValid);
    state[34]=select(0u,accepted[3],scheduleValid); state[35]=select(0u,accepted[5],scheduleValid);
    state[39]=select(0u,airWord(p.airControlOffset+6u),scheduleValid);
    let pages=min(worklist[1],p.pageCapacity);
    state[36]=select(state[36],accepted[3],scheduleValid&&mapNeeded);
    state[37]=select(state[37],activeBank(),scheduleValid&&mapNeeded);
    state[38]=select(state[38],p.generation,scheduleValid&&mapNeeded);
    state[40]=select(0u,pages,scheduleValid);state[41]=1u;state[42]=1u;
  }
}

fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let rem=key-z*xy;let y=rem/p.brickDims.x;return vec3u(rem-y*p.brickDims.x,y,z);}
fn localCoord(i:u32)->vec3u{let z=i/(p.r*p.r);let rem=i-z*p.r*p.r;let y=rem/p.r;return vec3u(rem-y*p.r,y,z);}
var<workgroup> classifyRare:array<u32,64>;var<workgroup> classifyInvalid:array<u32,64>;
@compute @workgroup_size(64)
fn classifyStructuredFineTransportBlocks(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){let work=wg.x;var id=INVALID;var rare=0u;var invalid=0u;if(state[0]==0u&&work<min(worklist[1],p.pageCapacity)){id=worklist[7u+work];if(id>=p.pageCapacity||metadata[id*10u+2u]!=p.generation){id=INVALID;}}if(id!=INVALID){for(var local=lid;local<p.samplesPerBrick;local+=64u){let index=id*p.samplesPerBrick+local;let sampleValid=(flags[index]&VALID)!=0u;if(sampleValid){let value=phi[index];invalid|=select(1u,0u,finite(value));rare|=select(0u,1u,value>=0.);}else{rare=1u;}}}classifyRare[lid]=rare;classifyInvalid[lid]=invalid;workgroupBarrier();for(var width=32u;width>0u;width>>=1u){if(lid<width){classifyRare[lid]|=classifyRare[lid+width];classifyInvalid[lid]|=classifyInvalid[lid+width];}workgroupBarrier();}if(lid==0u&&work<p.pageCapacity){let base=statusBase(work);let valid=id!=INVALID&&classifyInvalid[0]==0u;let cls=select(REGULAR_COMMON,REGULAR_RARE,classifyRare[0]!=0u);state[base]=cls;state[base+1u]=select(INVALID,work,valid);state[base+2u]=0xffff0000u;}}

struct WorksetRecord{present:u32,cls:u32,id:u32}
fn worksetValid()->bool{let valid=state[0]==0u&&accepted[2]==state[32]&&activeBank()==state[33]&&accepted[3]==state[34];return valid;}
// One workset item per live page, read in the same index order the retired
// serial loops walked. classifyStructuredFineTransportBlocks only ever stores
// REGULAR_COMMON or REGULAR_RARE; the clamp reproduces the bounds behaviour of
// the retired counts[state[base]] indexing for any other value.
fn worksetRecord(work:u32)->WorksetRecord{
  if(!worksetValid()||work>=min(worklist[1],p.pageCapacity)){return WorksetRecord(0u,0u,INVALID);}
  let base=statusBase(work);let id=state[base+1u];
  if(id==INVALID){return WorksetRecord(0u,0u,INVALID);}
  return WorksetRecord(1u,min(state[base],3u),id);
}
var<workgroup> worksetClassTotals:array<u32,4>;
// This item's exclusive prefix inside its own class, plus the block's four
// class totals. Classes are disjoint per page, so one prefix word per item is
// enough. Two 16-bit fields per scanned word carry two classes at once: a
// 256-lane block contributes at most 256 to a field, so the low field cannot
// carry into the high one and each field's prefix is that class's own count.
// Two scans therefore replace four.
fn scanWorksetBlock(local:u32,record:WorksetRecord)->u32{
  let member=record.present!=0u;
  let low=select(0u,select(1u,65536u,record.cls==1u),member&&record.cls<2u);
  let high=select(0u,select(1u,65536u,record.cls==3u),member&&record.cls>=2u);
  let lowPrefix=scanIdentityBlock(local,low);
  if(local==0u){worksetClassTotals[0]=identityScanTotal&65535u;worksetClassTotals[1]=identityScanTotal>>16u;}
  workgroupBarrier();
  let highPrefix=scanIdentityBlock(local,high);
  if(local==0u){worksetClassTotals[2]=identityScanTotal&65535u;worksetClassTotals[3]=identityScanTotal>>16u;}
  workgroupBarrier();
  let packed=select(highPrefix,lowPrefix,record.cls<2u);
  return select(packed>>16u,packed&65535u,(record.cls&1u)==0u);
}
@compute @workgroup_size(256)
fn reduceStructuredFineTransportWorksetBlocks(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  let work=wg.x*256u+lid;
  state[scanItemWord(work)]=scanWorksetBlock(lid,worksetRecord(work));
  if(lid==0u){for(var cls=0u;cls<4u;cls+=1u){state[scanBlockWord(wg.x,cls)]=worksetClassTotals[cls];}}
}
// One workgroup scans every block offset for all four classes, chunked so a
// single dispatch covers any page capacity, and publishes each class's stream
// base — the ascending class order the retired header loop accumulated.
@compute @workgroup_size(256)
fn scanStructuredFineTransportWorksetGroups(@builtin(local_invocation_index)lid:u32){
  let blocks=scanBlockCount();var base=0u;
  for(var cls=0u;cls<4u;cls+=1u){
    var carry=0u;
    for(var chunk=0u;chunk<blocks;chunk+=256u){
      let block=chunk+lid;var value=0u;if(block<blocks){value=state[scanBlockWord(block,cls)];}
      let prefix=scanIdentityBlock(lid,value);
      if(block<blocks){state[scanBlockWord(block,cls)]=carry+prefix;}
      carry+=identityScanTotal;workgroupBarrier();
    }
    if(lid==0u){state[scanTotalWord(SCAN_TOTAL_CLASS_BASE+cls)]=base;
      state[scanTotalWord(SCAN_TOTAL_CLASS_COUNT+cls)]=carry;}
    base+=carry;
  }
}
@compute @workgroup_size(1)
fn publishStructuredFineTransportWorksets(){
  // Each workgroup owns one page; its 64 lanes cover that page's fine samples.
  // Section 5 assigns the backtraced value to every starting fine cell, so the
  // indirect X dimension is the page count, not a second division by 64.
  let valid=worksetValid();
  var counts=array<u32,4>(0u,0u,0u,0u);
  for(var cls=0u;cls<4u;cls+=1u){counts[cls]=state[scanTotalWord(SCAN_TOTAL_CLASS_COUNT+cls)];}
  for(var cls=0u;cls<4u;cls+=1u){let base=headerBase(cls);state[base]=state[34];state[base+1u]=counts[cls];state[base+2u]=p.pageCapacity;state[base+3u]=select(0u,1u,valid);state[base+4u]=counts[cls];state[base+5u]=1u;state[base+6u]=1u;}
}
// Barrier-free: the in-block prefix was carried by the reduce dispatch, so the
// scatter is one address computation and one store per live page.
@compute @workgroup_size(256)
fn compactStructuredFineTransportWorksets(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  let work=wg.x*256u+lid;
  let record=worksetRecord(work);
  if(record.present==0u){return;}
  let at=state[scanTotalWord(SCAN_TOTAL_CLASS_BASE+record.cls)]+state[scanBlockWord(wg.x,record.cls)]
    +state[scanItemWord(work)];
  state[payloadBase()+at]=record.id;
}
fn workItem(cls:u32,index:u32)->u32{let base=headerBase(cls);if(state[base+3u]==0u||state[base]!=state[34]||index>=state[base+1u]){return INVALID;}var prefix=0u;for(var c=0u;c<cls;c+=1u){prefix+=state[headerBase(c)+1u];}return state[payloadBase()+prefix+index];}
fn canonicalVelocity(row:u32)->vec4f{if(row>=state[32]||rbase()+row>=arrayLength(&rowVelocity)){return vec4f(0,0,0,-1);}
  let value=rowVelocity[rbase()+row];return select(vec4f(0,0,0,-1),vec4f(value.xyz,1),value.w>0.&&finite3(value.xyz));}
fn taggedVelocity(tag:u32)->vec4f{if(tag==INVALID){return vec4f(0,0,0,-1);}if((tag&SUPPORT_TAG)==0u){return canonicalVelocity(tag);}
  let support=tag&0x7fffffffu;if(support>=state[39]||support>=p.supportCapacity){return vec4f(0,0,0,-1);}
  let at=p.supportVectorOffset/4u+support;if(at>=arrayLength(&airSupport)){return vec4f(0,0,0,-1);}let value=airSupport[at];
  return select(vec4f(0,0,0,-1),vec4f(value.xyz,1),value.w>0.&&finite3(value.xyz));}
// Characteristics may leave the cell-centred octree domain before their final
// fine-phi gather. Continue the already extrapolated Section 5 velocity with a
// zero-normal-gradient boundary value; both cube and power-tetra interpolation
// must use the same nearest valid octree sample point.
fn velocityDomainPoint(x:vec3f)->vec3f{return clamp(x,p.origin+vec3f(.5*p.physical),
  p.origin+vec3f(p.dimensions)*p.physical-vec3f(.5*p.physical));}
struct RegularAttempt{value:vec4f,exact:u32}
fn regularSampleExact(x:vec3f,anchor:AirOwner)->RegularAttempt{
  let invalid=RegularAttempt(vec4f(0,0,0,-1),1u);
  let clamped=velocityDomainPoint(x);
  let h=f32(anchor.size)*p.physical;if(!finite(h)||h<=0.){return invalid;}
  let center=vec3f(cellCoord(anchor.cell))*p.physical+.5*h;var lowOffset=vec3i(0);for(var axis=0u;axis<3u;axis+=1u){if(clamped[axis]<center[axis]){lowOffset[axis]=-1;}}
  let lowCenter=center+vec3f(lowOffset)*h;let t=clamp((clamped-lowCenter)/h,vec3f(0),vec3f(1));var result=vec3f(0);
  var velocityValid=true;
  for(var corner=0u;corner<8u;corner+=1u){let offset=lowOffset+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));let requested=center+vec3f(offset)*h;
    let samplePoint=clamp(requested,vec3f(.5*h),vec3f(p.dimensions)*p.physical-vec3f(.5*h));let owner=airOwnerAtPosition(samplePoint);
    let ownerCenter=vec3f(cellCoord(owner.cell))*p.physical+.5*f32(owner.size)*p.physical;let tolerance=max(1e-5,h*2e-5);
    if(owner.tag==INVALID||owner.size!=anchor.size||any(abs(ownerCenter-samplePoint)>vec3f(tolerance))){return RegularAttempt(vec4f(0,0,0,-1),0u);}
    let w=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);
    if(w==0.){continue;}let value=taggedVelocity(owner.tag);let good=value.w>=0.&&finite3(value.xyz);velocityValid=velocityValid&&good;
    if(good){result+=w*value.xyz;}}
  if(velocityValid&&finite3(result)){return RegularAttempt(vec4f(result,1),1u);}return invalid;}
fn bitsf(at:u32)->f32{return bitcast<f32>(catalog[at]);}
fn signs(code:u32)->vec3f{let b=code&7u;return vec3f(select(1.,-1.,(b&1u)!=0u),select(1.,-1.,(b&2u)!=0u),select(1.,-1.,(b&4u)!=0u));}
fn inverseTransform(v:vec3f,code:u32)->vec3f{let q=v*signs(code);let k=(code/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn powerTransform(v:vec3f,code:u32)->vec3f{let k=(code/8u)%6u;var q=v;if(k==1u){q=v.xzy;}else if(k==2u){q=v.yxz;}else if(k==3u){q=v.yzx;}else if(k==4u){q=v.zxy;}else if(k==5u){q=v.zyx;}return q*signs(code);}
fn caseHeader(caseId:u32)->vec2u{let at=p.volumeOffset+p.templateHeaderOffset+4u*caseId;return vec2u(catalog[at],catalog[at+1u]);}
fn geom(global:u32)->SlotGeometry{let at=p.slotGeometryOffset+12u*global;return SlotGeometry(vec4f(bitsf(at),bitsf(at+1u),bitsf(at+2u),bitsf(at+3u)),vec4f(bitsf(at+4u),bitsf(at+5u),bitsf(at+6u),bitsf(at+7u)),vec4f(bitsf(at+8u),bitsf(at+9u),bitsf(at+10u),bitsf(at+11u)));}
fn selectorGeometryValid(selector:vec4f)->bool{return finite3(selector.xyz)&&finite(selector.w)&&selector.w>0.
  &&(length(selector.xyz)>=1e-7||(all(selector.xyz==vec3f(0.))&&selector.w==1.));}
fn selectorVelocity(owner:AirOwner,selectorIndex:u32,selector:vec4f)->vec4f{if(owner.tag==INVALID||selectorIndex>=p.selectorStride||!selectorGeometryValid(selector)){return vec4f(0,0,0,-1);}
  if(all(selector.xyz==vec3f(0.))){return taggedVelocity(owner.tag);}let extent=f32(owner.size)*p.physical;let neighborExtent=selector.w*extent;
  if(!finite(neighborExtent)||neighborExtent<=0.){return vec4f(0,0,0,-1);}let center=vec3f(cellCoord(owner.cell))*p.physical+.5*extent;
  let selectorCenter=center+inverseTransform(selector.xyz,(owner.caseTransform>>16u)&63u)*extent;let lower=vec3f(.5*neighborExtent);
  let upper=vec3f(p.dimensions)*p.physical-lower;let tolerance=max(1e-5,neighborExtent*2e-5);
  if(all(selectorCenter>=lower-vec3f(tolerance))&&all(selectorCenter<=upper+vec3f(tolerance))){let other=airOwnerAtPosition(selectorCenter);
    if(other.tag==INVALID||abs(f32(other.size)*p.physical-neighborExtent)>tolerance){return vec4f(0,0,0,-1);}
    let otherCenter=vec3f(cellCoord(other.cell))*p.physical+.5*f32(other.size)*p.physical;
    if(any(abs(otherCenter-selectorCenter)>vec3f(tolerance))){return vec4f(0,0,0,-1);}return taggedVelocity(other.tag);}
  return taggedVelocity(owner.tag);}
fn tetraWeights(point:vec3f,x:vec3f,y:vec3f,z:vec3f)->vec4f{let d=dot(x,cross(y,z));if(!finite(d)||abs(d)<1e-10){return vec4f(-2);}let a0=dot(point,cross(y,z))/d;let a1=dot(x,cross(point,z))/d;let a2=dot(x,cross(y,point))/d;return vec4f(1.-a0-a1-a2,a0,a1,a2);}
fn transitionSample(x:vec3f)->vec4f{
  let invalid=vec4f(0,0,0,-1);if(!finite3(x)){return invalid;}let sampleX=velocityDomainPoint(x);
  let owner=airOwnerAtPosition(sampleX);if(owner.tag==INVALID){return invalid;}
  let caseId=owner.caseTransform&0xffffu;let transform=(owner.caseTransform>>16u)&63u;if(transform>=48u){return invalid;}
  if(caseId==0u){let regular=regularSampleExact(sampleX,owner);if(regular.exact!=0u){return regular.value;}}
  let words=arrayLength(&catalog);if(p.tetraVertexOffset>words||p.tetraVertexCount==0u
      ||p.tetraVertexCount>(words-p.tetraVertexOffset)/4u){return invalid;}
  if(caseId>(0xffffffffu-p.tetraHeaderOffset)/3u){return invalid;}let thAt=p.tetraHeaderOffset+3u*caseId;
  if(thAt>words||words-thAt<3u||p.tetraOffset>words){return invalid;}let first=catalog[thAt];let count=catalog[thAt+1u];
  if(first>words-p.tetraOffset||count>words-p.tetraOffset-first){return invalid;}
  let extent=f32(owner.size)*p.physical;if(!finite(extent)||extent<=0.){return invalid;}
  let center=vec3f(cellCoord(owner.cell))*p.physical+.5*extent;let local=powerTransform((sampleX-center)/extent,transform);
  for(var ti=0u;ti<count;ti+=1u){let packed=catalog[p.tetraOffset+first+ti];let s=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
    if(any(s>=vec3u(p.tetraVertexCount))){return invalid;}let va=p.tetraVertexOffset+4u*s.x;let vb=p.tetraVertexOffset+4u*s.y;let vc=p.tetraVertexOffset+4u*s.z;
    let a=vec4f(bitsf(va),bitsf(va+1u),bitsf(va+2u),bitsf(va+3u));let b=vec4f(bitsf(vb),bitsf(vb+1u),bitsf(vb+2u),bitsf(vb+3u));let c=vec4f(bitsf(vc),bitsf(vc+1u),bitsf(vc+2u),bitsf(vc+3u));
    if(!selectorGeometryValid(a)||!selectorGeometryValid(b)||!selectorGeometryValid(c)){return invalid;}
    let weights=tetraWeights(local,a.xyz,b.xyz,c.xyz);if(all(weights>=vec4f(-2e-6))&&all(weights<=vec4f(1.000002))){var w=max(weights,vec4f(0));let total=dot(w,vec4f(1));if(!finite(total)||total<=0.){return invalid;}w/=total;var result=vec3f(0);if(w.x>0.){let v=taggedVelocity(owner.tag);if(v.w<0.||!finite3(v.xyz)){return invalid;}result+=w.x*v.xyz;}if(w.y>0.){let v=selectorVelocity(owner,s.x,a);if(v.w<0.||!finite3(v.xyz)){return invalid;}result+=w.y*v.xyz;}if(w.z>0.){let v=selectorVelocity(owner,s.y,b);if(v.w<0.||!finite3(v.xyz)){return invalid;}result+=w.z*v.xyz;}if(w.w>0.){let v=selectorVelocity(owner,s.z,c);if(v.w<0.||!finite3(v.xyz)){return invalid;}result+=w.w*v.xyz;}return select(invalid,vec4f(result,1),finite3(result));}}
  return invalid;}

fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}
fn pageOf(key:u32)->u32{let at=7u+p.pageCapacity+key;if(key>=p.brickDims.x*p.brickDims.y*p.brickDims.z||at>=arrayLength(&worklist)){return INVALID;}let id=worklist[at];return select(INVALID,id,id<p.pageCapacity&&metadata[id*10u+1u]==key&&metadata[id*10u+2u]==p.generation);}
fn sampleIndex(q:vec3u)->u32{let b=q/p.r;let id=pageOf(packBrick(b));if(id==INVALID){return INVALID;}let l=q-b*p.r;return id*p.samplesPerBrick+l.x+p.r*(l.y+p.r*l.z);}
fn sampleFine(x0:vec3f)->f32{let high=vec3f(p.sampleDims)-vec3f(1.0001);let grid=clamp((x0-p.origin)/p.h-vec3f(.5),vec3f(0),high);let base=vec3u(floor(grid));let t=fract(grid);var sum=0.;var weight=0.;for(var corner=0u;corner<8u;corner+=1u){let o=vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u);let q=min(base+o,p.sampleDims-vec3u(1));let at=sampleIndex(q);if(at==INVALID||at>=arrayLength(&phi)||(flags[at]&VALID)==0u){continue;}let value=phi[at];if(!finite(value)){continue;}let w=select(1.-t.x,t.x,o.x!=0u)*select(1.-t.y,t.y,o.y!=0u)*select(1.-t.z,t.z,o.z!=0u);sum+=w*value;weight+=w;}return select(3.402823e38,sum/max(weight,1e-20),weight>0.999);}
fn insideDomain(x:vec3f)->bool{return all(x>=p.origin)&&all(x<=p.origin+vec3f(p.sampleDims)*p.h);}
fn clampDomain(x:vec3f)->vec3f{var hi=p.origin+vec3f(p.sampleDims)*p.h;if(p.openTop!=0u){hi.y=max(hi.y,x.y);}return clamp(x,p.origin,hi);}
var<workgroup> reduceOutside:array<u32,64>;var<workgroup> reduceNonfinite:array<u32,64>;var<workgroup> reduceProcessed:array<u32,64>;var<workgroup> reduceExtended:array<u32,64>;var<workgroup> reduceDisplacement:array<u32,64>;var<workgroup> reduceFirstBad:array<u32,64>;
fn finishPage(work:u32,lid:u32,outside:u32,bad:u32,processed:u32,extended:u32,displacement:u32,firstBad:u32){reduceOutside[lid]=outside;reduceNonfinite[lid]=bad;reduceProcessed[lid]=processed;reduceExtended[lid]=extended;reduceDisplacement[lid]=displacement;reduceFirstBad[lid]=firstBad;workgroupBarrier();for(var width=32u;width>0u;width>>=1u){if(lid<width){reduceOutside[lid]+=reduceOutside[lid+width];reduceNonfinite[lid]+=reduceNonfinite[lid+width];reduceProcessed[lid]+=reduceProcessed[lid+width];reduceExtended[lid]+=reduceExtended[lid+width];reduceDisplacement[lid]=max(reduceDisplacement[lid],reduceDisplacement[lid+width]);reduceFirstBad[lid]=min(reduceFirstBad[lid],reduceFirstBad[lid+width]);}workgroupBarrier();}if(lid==0u&&work<p.pageCapacity){let base=statusBase(work);state[base]=(reduceOutside[0]&65535u)|(min(reduceNonfinite[0],65535u)<<16u);state[base+1u]=(reduceProcessed[0]&65535u)|(min(reduceExtended[0],65535u)<<16u);state[base+2u]=(reduceDisplacement[0]&65535u)|(min(reduceFirstBad[0],65535u)<<16u);}}
struct Departure{x:vec3f,good:u32,extended:u32,maximumSpeed:f32}
struct SampleOutcome{outside:u32,bad:u32,processed:u32,extended:u32,displacement:u32}
// Section 5 advances each characteristic through the selected m substeps.
// The governor proves state[1] <= 64 before publishing any transport work.
fn regularCommonDeparture(origin:vec3f)->Departure{var x=origin;var good=1u;var maximum=0.;for(var stage=0u;stage<state[1];stage+=1u){let v=transitionSample(x);good&=select(0u,1u,v.w>=0.&&finite3(v.xyz));if(good!=0u){maximum=max(maximum,length(v.xyz));x-=bitcast<f32>(state[2])*v.xyz;}}return Departure(x,good,0u,maximum);}
fn regularRareDeparture(origin:vec3f,air:bool)->Departure{var x=origin;var good=1u;var extended=0u;var maximum=0.;for(var stage=0u;stage<state[1];stage+=1u){let v=transitionSample(x);good&=select(0u,1u,v.w>=0.&&finite3(v.xyz));if(good!=0u){extended+=select(0u,1u,air);maximum=max(maximum,length(v.xyz));x-=bitcast<f32>(state[2])*v.xyz;}}return Departure(x,good,extended,maximum);}
fn transitionCommonDeparture(origin:vec3f)->Departure{var x=origin;var good=1u;var maximum=0.;for(var stage=0u;stage<state[1];stage+=1u){let v=transitionSample(x);good&=select(0u,1u,v.w>=0.&&finite3(v.xyz));if(good!=0u){maximum=max(maximum,length(v.xyz));x-=bitcast<f32>(state[2])*v.xyz;}}return Departure(x,good,0u,maximum);}
fn transitionRareDeparture(origin:vec3f,air:bool)->Departure{var x=origin;var good=1u;var extended=0u;var maximum=0.;for(var stage=0u;stage<state[1];stage+=1u){let v=transitionSample(x);good&=select(0u,1u,v.w>=0.&&finite3(v.xyz));if(good!=0u){extended+=select(0u,1u,air);maximum=max(maximum,length(v.xyz));x-=bitcast<f32>(state[2])*v.xyz;}}return Departure(x,good,extended,maximum);}
fn finishSample(index:u32,old:f32,d:Departure)->SampleOutcome{var x=d.x;var outside=0u;if(!insideDomain(x)){outside=1u;if(p.closed!=0u){x=clampDomain(x);}}var sampled=3.402823e38;if(d.good!=0u){sampled=sampleFine(x);}let acceptedSample=d.good!=0u&&finite(sampled);if(acceptedSample){nextPhi[index]=sampled;}else{nextPhi[index]=old;}return SampleOutcome(outside,select(1u,0u,acceptedSample),select(0u,1u,acceptedSample),d.extended,u32(ceil(d.maximumSpeed*p.dt/p.h)));}
fn pageSample(id:u32,local:u32)->vec4u{let brick=unpackBrick(metadata[id*10u+1u]);let index=id*p.samplesPerBrick+local;let q=brick*p.r+localCoord(local);return vec4u(index,q);}
fn runRegularCommon(work:u32,lid:u32){let original=workItem(REGULAR_COMMON,work);var out=SampleOutcome(0u,0u,0u,0u,0u);var firstBad=INVALID;if(original!=INVALID){let id=worklist[7u+original];if(id<p.pageCapacity&&metadata[id*10u+2u]==p.generation){for(var local=lid;local<p.samplesPerBrick;local+=64u){let iq=pageSample(id,local);let index=iq.x;if((flags[index]&VALID)==0u){continue;}let old=phi[index];if(any(iq.yzw>=p.sampleDims)||!finite(old)){nextPhi[index]=old;out.bad+=1u;firstBad=min(firstBad,local);continue;}if(!inTransportBand(old)){nextPhi[index]=old;continue;}let origin=p.origin+(vec3f(iq.yzw)+.5)*p.h;let s=finishSample(index,old,regularCommonDeparture(origin));out.outside+=s.outside;out.bad+=s.bad;firstBad=min(firstBad,select(INVALID,local,s.bad!=0u));out.processed+=s.processed;out.displacement=max(out.displacement,s.displacement);}}}finishPage(original,lid,out.outside,out.bad,out.processed,out.extended,out.displacement,firstBad);}
fn runRegularRare(work:u32,lid:u32){let original=workItem(REGULAR_RARE,work);var out=SampleOutcome(0u,0u,0u,0u,0u);var firstBad=INVALID;if(original!=INVALID){let id=worklist[7u+original];if(id<p.pageCapacity&&metadata[id*10u+2u]==p.generation){for(var local=lid;local<p.samplesPerBrick;local+=64u){let iq=pageSample(id,local);let index=iq.x;if((flags[index]&VALID)==0u){continue;}let old=phi[index];if(any(iq.yzw>=p.sampleDims)||!finite(old)){nextPhi[index]=old;out.bad+=1u;firstBad=min(firstBad,local);continue;}if(!inTransportBand(old)){nextPhi[index]=old;continue;}let origin=p.origin+(vec3f(iq.yzw)+.5)*p.h;let s=finishSample(index,old,regularRareDeparture(origin,old>=0.));out.outside+=s.outside;out.bad+=s.bad;firstBad=min(firstBad,select(INVALID,local,s.bad!=0u));out.processed+=s.processed;out.extended+=s.extended;out.displacement=max(out.displacement,s.displacement);}}}finishPage(original,lid,out.outside,out.bad,out.processed,out.extended,out.displacement,firstBad);}
fn runTransitionCommon(work:u32,lid:u32){let original=workItem(TRANSITION_COMMON,work);var out=SampleOutcome(0u,0u,0u,0u,0u);var firstBad=INVALID;if(original!=INVALID){let id=worklist[7u+original];if(id<p.pageCapacity&&metadata[id*10u+2u]==p.generation){for(var local=lid;local<p.samplesPerBrick;local+=64u){let iq=pageSample(id,local);let index=iq.x;if((flags[index]&VALID)==0u){continue;}let old=phi[index];if(any(iq.yzw>=p.sampleDims)||!finite(old)){nextPhi[index]=old;out.bad+=1u;firstBad=min(firstBad,local);continue;}if(!inTransportBand(old)){nextPhi[index]=old;continue;}let origin=p.origin+(vec3f(iq.yzw)+.5)*p.h;let s=finishSample(index,old,transitionCommonDeparture(origin));out.outside+=s.outside;out.bad+=s.bad;firstBad=min(firstBad,select(INVALID,local,s.bad!=0u));out.processed+=s.processed;out.displacement=max(out.displacement,s.displacement);}}}finishPage(original,lid,out.outside,out.bad,out.processed,out.extended,out.displacement,firstBad);}
fn runTransitionRare(work:u32,lid:u32){let original=workItem(TRANSITION_RARE,work);var out=SampleOutcome(0u,0u,0u,0u,0u);var firstBad=INVALID;if(original!=INVALID){let id=worklist[7u+original];if(id<p.pageCapacity&&metadata[id*10u+2u]==p.generation){for(var local=lid;local<p.samplesPerBrick;local+=64u){let iq=pageSample(id,local);let index=iq.x;if((flags[index]&VALID)==0u){continue;}let old=phi[index];if(any(iq.yzw>=p.sampleDims)||!finite(old)){nextPhi[index]=old;out.bad+=1u;firstBad=min(firstBad,local);continue;}if(!inTransportBand(old)){nextPhi[index]=old;continue;}let origin=p.origin+(vec3f(iq.yzw)+.5)*p.h;let s=finishSample(index,old,transitionRareDeparture(origin,old>=0.));out.outside+=s.outside;out.bad+=s.bad;firstBad=min(firstBad,select(INVALID,local,s.bad!=0u));out.processed+=s.processed;out.extended+=s.extended;out.displacement=max(out.displacement,s.displacement);}}}finishPage(original,lid,out.outside,out.bad,out.processed,out.extended,out.displacement,firstBad);}
@compute @workgroup_size(64)fn transportRegularCommonPhi(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){runRegularCommon(wg.x,lid);}
@compute @workgroup_size(64)fn transportTransitionCommonPhi(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){runTransitionCommon(wg.x,lid);}
@compute @workgroup_size(64)fn transportRegularRarePhi(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){runRegularRare(wg.x,lid);}
@compute @workgroup_size(64)fn transportTransitionRarePhi(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){runTransitionRare(wg.x,lid);}
struct StatusSummary{outside:u32,nonfinite:u32,processed:u32,extended:u32,invalidStatus:u32,maxDisplacement:u32,firstWork:u32}
// 64 lanes, matching this file's finishPage reduction: seven u32 lanes cost
// 1,792 bytes of workgroup storage rather than 7,168, and the tree is six
// steps instead of eight. Every operator below is an integer add, a max, or a
// min -- all associative and commutative, with u32 wraparound preserved -- so
// folding 256 pages as four serial steps then a 64-lane tree yields the same
// words as a 256-lane tree. No float is involved, so there is no reassociation.
var<workgroup> summaryOutside:array<u32,64>;var<workgroup> summaryNonfinite:array<u32,64>;
var<workgroup> summaryProcessed:array<u32,64>;var<workgroup> summaryExtended:array<u32,64>;
var<workgroup> summaryInvalid:array<u32,64>;var<workgroup> summaryDisplacement:array<u32,64>;
var<workgroup> summaryFirstWork:array<u32,64>;
fn mergeStatusSummary(a:StatusSummary,b:StatusSummary)->StatusSummary{
  return StatusSummary(a.outside+b.outside,a.nonfinite+b.nonfinite,a.processed+b.processed,
    a.extended+b.extended,a.invalidStatus+b.invalidStatus,
    max(a.maxDisplacement,b.maxDisplacement),min(a.firstWork,b.firstWork));
}
// One page's contribution to the transport summary. Every counter is an
// integer add and the displacement a max, so the reduction is order-free; the
// retained first rejection is the smallest page index that both carries a
// packed local failure and still resolves to a page of this generation, which
// is exactly what the retired loop's early-first-wins guard selected.
fn statusSummary(i:u32,count:u32)->StatusSummary{
  if(i>=count){return StatusSummary(0u,0u,0u,0u,0u,0u,INVALID);}
  let base=statusBase(i);let errors=state[base];let work=state[base+1u];
  if(work==INVALID){return StatusSummary(0u,1u,0u,0u,1u,0u,INVALID);}
  let packed=state[base+2u];let firstLocal=packed>>16u;var first=INVALID;
  if(firstLocal!=65535u){let id=worklist[7u+i];
    if(id<p.pageCapacity&&metadata[id*10u+2u]==p.generation){first=i;}}
  return StatusSummary(errors&65535u,errors>>16u,work&65535u,work>>16u,0u,packed&65535u,first);
}
fn reduceStatusBlock(lid:u32,value:StatusSummary){
  summaryOutside[lid]=value.outside;summaryNonfinite[lid]=value.nonfinite;summaryProcessed[lid]=value.processed;
  summaryExtended[lid]=value.extended;summaryInvalid[lid]=value.invalidStatus;
  summaryDisplacement[lid]=value.maxDisplacement;summaryFirstWork[lid]=value.firstWork;workgroupBarrier();
  for(var width=32u;width>0u;width>>=1u){if(lid<width){
    summaryOutside[lid]+=summaryOutside[lid+width];summaryNonfinite[lid]+=summaryNonfinite[lid+width];
    summaryProcessed[lid]+=summaryProcessed[lid+width];summaryExtended[lid]+=summaryExtended[lid+width];
    summaryInvalid[lid]+=summaryInvalid[lid+width];
    summaryDisplacement[lid]=max(summaryDisplacement[lid],summaryDisplacement[lid+width]);
    summaryFirstWork[lid]=min(summaryFirstWork[lid],summaryFirstWork[lid+width]);}workgroupBarrier();}
}
// A rejected schedule publishes an empty summary, so its page statuses are
// never folded in — the same shape the retired loop's early return had.
fn summaryPageCount()->u32{return select(0u,min(worklist[1],p.pageCapacity),state[0]==0u&&state[1]!=0u);}
@compute @workgroup_size(64)
fn reduceStructuredFineTransportStatus(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  // The block partition stays 256 pages wide; 64 lanes fold it on a uniform
  // trip count so the reduction's barriers stay in uniform control flow.
  let count=summaryPageCount();var value=StatusSummary(0u,0u,0u,0u,0u,0u,INVALID);
  for(var chunk=0u;chunk<256u;chunk+=64u){
    value=mergeStatusSummary(value,statusSummary(wg.x*256u+chunk+lid,count));}
  reduceStatusBlock(lid,value);
  if(lid==0u){
    state[scanBlockWord(wg.x,SCAN_SLOT_OUTSIDE)]=summaryOutside[0];
    state[scanBlockWord(wg.x,SCAN_SLOT_NONFINITE)]=summaryNonfinite[0];
    state[scanBlockWord(wg.x,SCAN_SLOT_PROCESSED)]=summaryProcessed[0];
    state[scanBlockWord(wg.x,SCAN_SLOT_EXTENDED)]=summaryExtended[0];
    state[scanBlockWord(wg.x,SCAN_SLOT_INVALID)]=summaryInvalid[0];
    state[scanBlockWord(wg.x,SCAN_SLOT_DISPLACEMENT)]=summaryDisplacement[0];
    state[scanBlockWord(wg.x,SCAN_SLOT_FIRST)]=summaryFirstWork[0];}
}
@compute @workgroup_size(64)fn summarizeStructuredFineTransport(@builtin(local_invocation_index)lid:u32){
  // Chunked on a uniform trip count so the reduction's barriers stay in
  // uniform control flow; one workgroup then covers any block count.
  let blocks=scanBlockCount();var value=StatusSummary(0u,0u,0u,0u,0u,0u,INVALID);
  for(var chunk=0u;chunk<blocks;chunk+=64u){let block=chunk+lid;if(block<blocks){
    value=mergeStatusSummary(value,StatusSummary(
      state[scanBlockWord(block,SCAN_SLOT_OUTSIDE)],state[scanBlockWord(block,SCAN_SLOT_NONFINITE)],
      state[scanBlockWord(block,SCAN_SLOT_PROCESSED)],state[scanBlockWord(block,SCAN_SLOT_EXTENDED)],
      state[scanBlockWord(block,SCAN_SLOT_INVALID)],state[scanBlockWord(block,SCAN_SLOT_DISPLACEMENT)],
      state[scanBlockWord(block,SCAN_SLOT_FIRST)]));}}
  reduceStatusBlock(lid,value);
  if(lid!=0u){return;}
  control=Control(0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,INVALID,INVALID,0u,0u,0u);
  if(state[0]!=0u||state[1]==0u){control.authorityUnavailable=1u;control.maxDisplacement=state[3];return;}
  control.outside=summaryOutside[0];control.nonfinite=summaryNonfinite[0];
  control.processed=summaryProcessed[0];control.extended=summaryExtended[0];
  control.invalidStatus=summaryInvalid[0];control.maxDisplacement=summaryDisplacement[0];
  let firstWork=summaryFirstWork[0];
  if(firstWork!=INVALID){let id=worklist[7u+firstWork];let firstLocal=state[statusBase(firstWork)+2u]>>16u;
    let q=unpackBrick(metadata[id*10u+1u])*p.r+localCoord(firstLocal);let x=p.origin+(vec3f(q)+vec3f(.5))*p.h;
    control.firstIndex=firstLocal;control.firstX=bitcast<u32>(x.x);control.firstY=bitcast<u32>(x.y);control.firstZ=bitcast<u32>(x.z);}
  control.velocityUnavailable=control.nonfinite;
  // Section 5 dynamic topology. A departure point that leaves the resident band
  // yields no advected value, and finishSample already carries the old sample
  // forward rather than inventing one, so nextPhi is defined for every valid
  // sample of a transported page. The paper repairs precisely this population
  // downstream: "new cells may emerge in the narrow band which require valid
  // signed distance values ... we run fast marching on this SPGrid, activating
  // new cells wherever necessary". Any band with an outer edge starves some
  // samples every step, so that is normal operation rather than a reason to
  // discard the whole update. What must still veto is a page that was never
  // transported at all -- its nextPhi would be read undefined -- which classify
  // records as an INVALID status and summarizes into invalidStatus. A step that
  // advected nothing at all also stays uncommitted, which is the wholly starved
  // case the CPU oracle pins.
  control.committed=select(0u,1u,control.processed>0u&&control.invalidStatus==0u);
}
var<workgroup> neighborPages:array<u32,6>;var<workgroup> pageChanged:array<u32,64>;var<workgroup> pageOldInterface:array<u32,64>;var<workgroup> pageNewInterface:array<u32,64>;
fn neighborIndex(id:u32,local:u32,q:vec3u,dir:u32)->u32{var n=vec3i(q);let axis=dir/2u;n[axis]+=select(-1,1,(dir&1u)!=0u);if(n[axis]>=0&&n[axis]<i32(p.r)){return id*p.samplesPerBrick+u32(n.x)+p.r*(u32(n.y)+p.r*u32(n.z));}let other=neighborPages[dir];if(other==INVALID){return INVALID;}n[axis]=select(i32(p.r)-1,0,n[axis]>=i32(p.r));return other*p.samplesPerBrick+u32(n.x)+p.r*(u32(n.y)+p.r*u32(n.z));}
@compute @workgroup_size(64)fn commitStructuredFineTransport(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){let work=wg.x;var live=control.committed!=0u&&work<min(worklist[1],p.pageCapacity);var id=INVALID;if(live){id=worklist[7u+work];live=id<p.pageCapacity&&metadata[id*10u+2u]==p.generation;}if(lid<6u){var n=INVALID;if(live){let q=metadata[id*10u+4u+lid];if(q<p.pageCapacity&&metadata[q*10u+2u]==p.generation){n=q;}}neighborPages[lid]=n;}workgroupBarrier();var changed=0u;var oldInterface=0u;var newInterface=0u;if(live){for(var local=lid;local<p.samplesPerBrick;local+=64u){let index=id*p.samplesPerBrick+local;if((flags[index]&VALID)==0u){continue;}let old=phi[index];let fresh=nextPhi[index];if(!finite(old)||!finite(fresh)){changed=1u;continue;}changed|=select(0u,1u,(old<0.)!=(fresh<0.));let q=localCoord(local);for(var dir=0u;dir<6u;dir+=1u){let ni=neighborIndex(id,local,q,dir);if(ni==INVALID||(flags[ni]&VALID)==0u){continue;}let oldN=phi[ni];let newN=nextPhi[ni];oldInterface|=select(0u,1u,finite(oldN)&&(old<0.)!=(oldN<0.));newInterface|=select(0u,1u,finite(newN)&&(fresh<0.)!=(newN<0.));}}}pageChanged[lid]=changed;pageOldInterface[lid]=oldInterface;pageNewInterface[lid]=newInterface;workgroupBarrier();for(var width=32u;width>0u;width>>=1u){if(lid<width){pageChanged[lid]|=pageChanged[lid+width];pageOldInterface[lid]|=pageOldInterface[lid+width];pageNewInterface[lid]|=pageNewInterface[lid+width];}workgroupBarrier();}if(live&&lid==0u){let before=(metadata[id*10u+3u]&PAGE_INTERFACE)!=0u||pageOldInterface[0]!=0u;let after=pageNewInterface[0]!=0u;let membership=pageChanged[0]!=0u||before||after;metadata[id*10u+3u]=VALID|select(0u,PAGE_INTERFACE,after)|select(0u,PAGE_DIRTY,membership);delta[8u+id]=select(INVALID,metadata[id*10u+1u],membership);}workgroupBarrier();if(live){for(var local=lid;local<p.samplesPerBrick;local+=64u){let index=id*p.samplesPerBrick+local;if((flags[index]&VALID)!=0u){phi[index]=nextPhi[index];}}}}
// The changed brick key this live slot contributes, or INVALID. Slots are read
// in page-index order exactly as the retired compaction loop walked them.
fn deltaChangedKey(work:u32)->u32{
  if(work>=min(worklist[1],p.pageCapacity)){return INVALID;}
  let id=worklist[7u+work];if(id>=p.pageCapacity){return INVALID;}
  return delta[8u+id];
}
// Header and compacted stream only: the per-page key range written by
// commitStructuredFineTransport stays untouched, as before.
@compute @workgroup_size(256)fn clearStructuredFineDelta(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  let i=wg.x*256u+lid;if(i>=8u+2u*p.pageCapacity){return;}
  if(i<8u||i>=8u+p.pageCapacity){delta[i]=0u;}
}
@compute @workgroup_size(256)fn reduceStructuredFineDeltaBlocks(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  let work=wg.x*256u+lid;
  state[scanItemWord(work)]=scanIdentityBlock(lid,select(0u,1u,deltaChangedKey(work)!=INVALID));
  if(lid==0u){state[scanBlockWord(wg.x,SCAN_SLOT_DELTA)]=identityScanTotal;}
}
@compute @workgroup_size(256)fn publishStructuredFineDelta(@builtin(local_invocation_index)lid:u32){
  let blocks=scanBlockCount();var carry=0u;
  for(var chunk=0u;chunk<blocks;chunk+=256u){
    let block=chunk+lid;var value=0u;if(block<blocks){value=state[scanBlockWord(block,SCAN_SLOT_DELTA)];}
    let prefix=scanIdentityBlock(lid,value);
    if(block<blocks){state[scanBlockWord(block,SCAN_SLOT_DELTA)]=carry+prefix;}
    carry+=identityScanTotal;workgroupBarrier();}
  if(lid!=0u){return;}
  let count=carry;let liveCount=min(worklist[1],p.pageCapacity);
  delta[0]=count;delta[1]=p.generation;delta[2]=control.committed;delta[3]=liveCount;delta[4]=select(0u,u32(ceil(f32(count)/64.)),count>0u);delta[5]=1u;delta[6]=1u;delta[7]=0u;
}
@compute @workgroup_size(256)fn compactStructuredFineDelta(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  let work=wg.x*256u+lid;
  let key=deltaChangedKey(work);
  if(key==INVALID){return;}
  delta[8u+p.pageCapacity+state[scanBlockWord(wg.x,SCAN_SLOT_DELTA)]+state[scanItemWord(work)]]=key;
}
`;
