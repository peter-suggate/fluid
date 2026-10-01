import { compileMixedTiers, type UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { WebGPUUniformVelocityExtrapolator } from "./webgpu-uniform-velocity-extrapolation";
import { UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL,uniformMixedFaceDispatchWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";

type Hierarchy=ReturnType<WebGPUUniformVelocityExtrapolator["prepareMixedContinuation"]>;
interface Fields {
 physical:GPUTexture;phase:GPUTexture;negative:GPUBuffer;output:GPUTexture;outputNegative:GPUBuffer;
 scratch:GPUBufferBinding;params:GPUBuffer;
}
/** umFarValue with its coarse state and source-bound loads supplied. */
function farValueWGSL(name:string,state:string,lower:string,upper:string):string{
 return /* wgsl */`fn ${name}(face:UMFace)->f32{
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var distances:array<f32,8>;var values:array<f32,8>;var best=UM_INF;
 for(var k=0u;k<umCounts.w;k++){
  distances[k]=UM_INF;let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if(any(weights<=vec3f(0))){continue;}
  let p=clamp(base+bit,vec3i(0),ueFarTop(face.axis));let state=${state};
  if((u32(round(state.w))&(1u<<face.axis))==0u){continue;}
  let lower=${lower};let upper=${upper};if(lower==0u){continue;}
  var lo=vec3f(umSourcePoint(lower))+vec3f(0.5);var hi=vec3f(umSourcePoint(upper))+vec3f(0.5);lo[face.axis]+=0.5;hi[face.axis]+=0.5;
  let scale=h.xyz/min(h.x,min(h.y,h.z));let delta=(point-clamp(point,lo,hi))*scale;
  let distance=(delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
  best=min(best,distance);distances[k]=distance;values[k]=state[face.axis];
 }
 var sum=0.0;var count=0.0;
 for(var k=0u;k<umCounts.w;k++){if(distances[k]<0.5*UM_INF&&abs(distances[k]-best)<=1e-6*max(1.0,best)){sum+=values[k];count+=1.0;}}
 return select(0.0,sum/count,count>0.0);
}
`;
}
/** (value, distance) slots per canonical patch and component: h patches
 * tile-major (tile, component, 64 cells), width-4 patches in a component-major
 * n/4 layer, then the negative domain-wall planes of each width, then a
 * two-word finite-cell mask per tile (UE_FLAGS). The address is arithmetic
 * on (anchor, axis, width); see umSlot. */
function extensionSlots(d:readonly number[]):number{
 const t=d.map(n=>n/4),plane=(a:readonly number[])=>a[1]!*a[2]!+a[0]!*a[2]!+a[0]!*a[1]!;
 return 3*d[0]!*d[1]!*d[2]!+3*t[0]!*t[1]!*t[2]!+plane(d)+plane(t)+2*t[0]!*t[1]!*t[2]!;
}
/** Regular-bulk mode restricts physical source faces directly to the 4h
 * nearest-source hierarchy, then retains supported h faces on publication.
 * The legacy mode first performs Godunov sweeps on canonical mixed patches;
 * its two transient vec2f slot arrays borrow the native FIM arena. */
export class UniformMixedExtension {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 private readonly slotBytes:number;
 private readonly resources:GPUBindGroupLayout;
 /** GPU-counted launches (UNIFORM_MIXED_COUNTED), each of a fixed grid: no
  * launch size, variant or skip reads the host membership. Regular h owners. */
 private readonly regularPipelines=new Map<string,GPUComputePipeline>();
 /** Seam tiers: h per owner, 4h one tile job per seam tile (…Coarse entries). */
 private readonly seamPipelines=new Map<string,GPUComputePipeline[]>();
 /** Every regular 4h owner, 64 per job, fused tier or not: the per-owner code
  * is the tier launch's (only the owner lookup differs), so one launch serves. */
 private readonly regularCoarseListPipelines=new Map<string,GPUComputePipeline>();
 private restrictPipeline?:GPUComputePipeline;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,readonly hierarchy:Hierarchy,private readonly regularBulk=false,directRestriction=false){
  if(ownership.capacity.lattice.dimensions.some(n=>n%4!==0))throw new Error("Mixed extension requires a 4-aligned lattice");
  if(!directRestriction)throw new Error("Mixed extension restricts the face-plane anchors directly");
  this.slotBytes=Math.ceil(8*extensionSlots(ownership.capacity.lattice.dimensions)/256)*256;
  this.scratchBytes=2*this.slotBytes;
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   ...[4,5,9].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   {binding:10,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"uint",viewDimension:"3d"}},
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[8,11].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"rgba32float" as const,viewDimension:"3d" as const}})),
   {binding:12,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32uint",viewDimension:"3d"}},
   {binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
  ]});
 }
 bind(f:Fields):readonly [GPUBindGroup,GPUBindGroup]{
  const d=this.ownership.capacity.lattice.dimensions;
  for(const [i,t] of [f.physical,f.phase,f.output].entries())if(t.format!==(i===1?"r32float":"rgba32float")||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]))throw new Error("Mixed extension requires native canonical fields");
  if(f.physical===f.output||f.negative===f.outputNegative)throw new Error("Mixed extension outputs must be disjoint");
  if((f.scratch.size??f.scratch.buffer.size-(f.scratch.offset??0))<this.scratchBytes)throw new Error("Mixed extension scratch is too small");
  const size=this.slotBytes,offset=f.scratch.offset??0;
  return [0,1].map(parity=>this.device.createBindGroup({layout:this.resources,entries:[
   ...[parity,parity^1].map((slot,binding)=>({binding,resource:{buffer:f.scratch.buffer,offset:offset+slot*size,size}})),
   {binding:4,resource:f.physical.createView()},{binding:5,resource:f.phase.createView()},
   {binding:6,resource:{buffer:f.negative}},{binding:7,resource:{buffer:f.params,size:16}},
   {binding:8,resource:f.output.createView()},{binding:9,resource:this.hierarchy.output.createView()},
   {binding:10,resource:this.hierarchy.outputOrigins.createView()},
   {binding:11,resource:this.hierarchy.input.createView()},{binding:12,resource:this.hierarchy.inputOrigins.createView()},
   {binding:13,resource:{buffer:f.outputNegative}},
  ]})) as unknown as readonly [GPUBindGroup,GPUBindGroup];
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> stateIn:array<vec2f>;
@group(1) @binding(1) var<storage,read_write> stateOut:array<vec2f>;
@group(1) @binding(4) var physical:texture_3d<f32>;
@group(1) @binding(5) var phase:texture_3d<f32>;
@group(1) @binding(6) var<storage,read> negative:array<f32>;
@group(1) @binding(7) var<uniform> h:vec4f;
@group(1) @binding(8) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(9) var coarse:texture_3d<f32>;
@group(1) @binding(10) var origins:texture_3d<u32>;
@group(1) @binding(11) var coarseOut:texture_storage_3d<rgba32float,write>;
@group(1) @binding(12) var originsOut:texture_storage_3d<rgba32uint,write>;
@group(1) @binding(13) var<storage,read_write> boundary:array<f32>;
${uniformMixedFaceAddressWGSL}
const UM_INF=1e20;
// A closed positive domain wall is the mirror of a negative wall plane: seeded
// every pass, never swept or restricted, published as its physical value. The
// negative planes have no coarse +face, so the far lookup never taps the
// positive wall layer on its normal axis either. h.w: the top is open.
fn umClosedWall(face:UMFace)->bool{
 return face.sign>0&&face.anchor[face.axis]==i32(UM_D[face.axis])-1&&!(face.axis==1u&&h.w>0.5);
}
fn ueFarTop(axis:u32)->vec3i{
 var top=vec3i(UM_T)-vec3i(1);if(!(axis==1u&&h.w>0.5)){top[axis]=max(top[axis]-1,0);}return top;
}
// One (value, distance) slot per canonical patch and component. h patches are
// tile-major, (tile, component, 64 cells): an h tile's lanes read one 512-byte
// run per component. Width-4 patches (anchor/4 is their tile) sit in a
// component-major n/4 layer, so neighbouring 4h owners are neighbouring words.
// Negative domain-wall patches (anchor[axis] = -1) have planes of their own.
// The address needs the patch width, never a topology load.
const UE_TILES=UM_T.x*UM_T.y*UM_T.z;
const UE_COARSE=192u*UE_TILES;
const UE_UNIT_WALL=UE_COARSE+3u*UE_TILES;
const UE_COARSE_WALL=UE_UNIT_WALL+UM_D.y*UM_D.z+UM_D.x*UM_D.z+UM_D.x*UM_D.y;
// Per tile finite-cell masks (ueMaskIn), two words per tile.
const UE_FLAGS=UE_COARSE_WALL+UM_T.y*UM_T.z+UM_T.x*UM_T.z+UM_T.x*UM_T.y;
fn ueCoarseWallIndex(t:vec3u,axis:u32)->u32{
 if(axis==0u){return t.y+UM_T.y*t.z;}
 if(axis==1u){return UM_T.y*UM_T.z+t.x+UM_T.x*t.z;}
 return UM_T.y*UM_T.z+UM_T.x*UM_T.z+t.x+UM_T.x*t.y;
}
fn umSlot(anchor:vec3i,axis:u32,width:u32)->u32{
 let p=vec3u(max(anchor,vec3i(0)));
 if(anchor[axis]<0){return select(UE_COARSE_WALL+ueCoarseWallIndex(p/4u,axis),UE_UNIT_WALL+umNegativeBoundaryIndex(p,axis),width==1u);}
 let tile=umTileAt(p/4u);let l=p%4u;
 return select(UE_COARSE+axis*UE_TILES+tile,(3u*tile+axis)*64u+l.x+4u*(l.y+4u*l.z),width==1u);
}
// The shared sampler source needs this symbol; no entry of this module samples.
fn umLoadMixedFace(p:vec3i,axis:u32)->f32{return stateIn[umSlot(p,axis,1u)].x;}
// A slot is written only by the owner holding its (clamped) anchor cell. Off
// the extension support no pass writes it: its seed would be distance INF (a
// source face needs a liquid owner, which the support dilates around), and a
// value beside an infinite distance is never consumed. A slot is read only as
// the patch of that width at that anchor, which its owner writes every pass.
fn umSlotState(anchor:vec3i,axis:u32,width:u32)->vec2f{
 let tile=umTileAt(vec3u(clamp(anchor,vec3i(0),vec3i(UM_D)-vec3i(1)))/4u);
 // Branch-free: the support word and the slot load together.
 let state=stateIn[umSlot(clamp(anchor,vec3i(-1),vec3i(UM_D)-vec3i(1)),axis,width)];
 return select(vec2f(0.0,UM_INF),state,(umTileSupport(tile)&2u)!=0u);
}
${uniformMixedVelocitySamplingSource()}
fn umSource(face:UMFace,owner:UMOwner)->bool{
 if(textureLoad(phase,vec3i(umOrigin(owner)),0).x>0.5){return true;}
 return face.neighbor.width!=0u&&textureLoad(phase,vec3i(umOrigin(face.neighbor)),0).x>0.5;
}
fn umPhysical(face:UMFace)->f32{
 if(face.anchor[face.axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(face.anchor,vec3i(0))),face.axis)];}
 return textureLoad(physical,face.anchor,0)[face.axis];
}
struct UMNeighbor {value:f32,distance:f32,spacing:f32}
fn umNeighbor(point:vec3f,center:vec3f,component:u32,step:u32,width:u32)->UMNeighbor{
 if(any(point<vec3f(0))||any(point>vec3f(UM_D))){return UMNeighbor(0,UM_INF,1);}
 let tile=umTileAt(min(vec3u(point),UM_D-vec3u(1))/4u);
 // Both direct cases read the width-w patch slot below the plane (w is 1 on
 // the all-h stencil path). Its anchor needs no topology, so the slot,
 // support and width words issue together instead of as a dependent chain.
 var offset=vec3f(0.5*f32(width));offset[component]=1.0;
 // The first case below always reads a unit patch (every cell of an all-h
 // stencil, and so the one below the plane, is h); the second reads the
 // width-w patch. Both issue here, before the topology words resolve.
 let anchor=vec3i(round(point-offset));let unit=umSlotState(anchor,component,1u);
 var direct=unit;if(width!=1u){direct=umSlotState(anchor,component,width);}
 // A unit request whose plane is interior to a unit tile has unit cells on
 // both sides: the case below with lowWidth = highWidth = 1, one load.
 let interior=width==1u&&(u32(round(point[component]))&3u)!=0u&&umTileWidth(tile)==1u;
 if(umRegularFine||interior||umTileMaximumWidth(tile)==1u){return UMNeighbor(unit.x,unit.y,h[step]);}
 // The point is a width-w lattice patch centre (the requesting face has
 // width w). When the owners on both sides of its plane are no finer than w
 // and one of them has width w (or the plane is a domain wall and the inner
 // owner has width w), the lower owner's positive face has patches of width
 // min(low, high) = w, one of them centred here: the search below selects
 // exactly it.
 var below=vec3i(floor(point));let plane=i32(round(point[component]));below[component]=plane-1;
 var above=below;above[component]=plane;
 let lowWidth=select(0u,umTileWidth(umTileAt(vec3u(max(below,vec3i(0)))/4u)),plane>0);
 let highWidth=select(0u,umTileWidth(umTileAt(min(vec3u(above),UM_D-vec3u(1))/4u)),plane<i32(UM_D[component]));
 if(select((lowWidth==width&&(highWidth==0u||highWidth>=width))||(highWidth==width&&lowWidth>width),highWidth==width,lowWidth==0u)){
  // The point is center ± width along step only: the spacing is exactly
  // width*h, as on the fast path above. sqrt(dot) can land an ulp above it,
  // and the root cutoff below sits exactly on whole-cell distances, so which
  // path a face took (the +side tile decides) would decide its reach.
  return UMNeighbor(direct.x,direct.y,abs(point[step]-center[step])*h[step]);
 }
 let site=umVelocitySite(point,component);
 // Candidate real faces; choose geometrically, never read an unowned fine
 // texel. A requested plane inside a coarser cell has two incident faces. A
 // wider patch's centre on finer owners sits on fine cell boundaries in both
 // tangential axes: every tied cell's face is a candidate. Ties are resolved
 // by value, never by order: floor() (and a strict first-wins compare) takes
 // the +side (-side) cell of a tie, and a mirror maps it to the other side.
 let owner=umOwnerAt(min(vec3i(floor(point)),vec3i(UM_D)-vec3i(1)));
 let u=(component+1u)%3u;let v=(component+2u)%3u;
 var faces:array<UMFace,4>;var n=0u;
 if(site.interior){
  for(var side=0u;side<2u;side++){
   let sign=select(-1,1,side==1u);let first=umFace(owner,component,sign,0u);
   let local=clamp(point-vec3f(umOrigin(owner)),vec3f(0),vec3f(f32(owner.width)-1e-4));
   faces[n]=umFace(owner,component,sign,u32(local[u])/first.width+(owner.width/first.width)*(u32(local[v])/first.width));n++;
  }
 }else if(site.width<width){
  for(var k=0u;k<4u;k++){
   var tied=point;tied[u]+=select(-0.5,0.5,(k&1u)!=0u);tied[v]+=select(-0.5,0.5,(k&2u)!=0u);
   let s=umVelocitySite(tied,component);if(!s.interior){faces[n]=s.face;n++;}
  }
 }else{faces[0]=site.face;n=1u;}
 // Nearest faces, then the least slot distance among them, then the mean
 // value of the faces holding it.
 var spatials:array<f32,4>;var slots:array<vec2f,4>;var nearest=UM_INF;
 for(var i=0u;i<n;i++){
  spatials[i]=UM_INF;let face=faces[i];if(face.width==0u){continue;}
  let location=umFaceCenter(face);let delta=(location-center)*h.xyz;
  if(abs(location[step]-center[step])<1e-5){continue;}
  spatials[i]=dot(delta,delta);slots[i]=umSlotState(face.anchor,component,face.width);nearest=min(nearest,spatials[i]);
 }
 if(nearest>=0.5*UM_INF){return UMNeighbor(0,UM_INF,1);}
 var distance=UM_INF;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)){distance=min(distance,slots[i].y);}}
 var sum=0.0;var count=0.0;
 for(var i=0u;i<n;i++){if(spatials[i]<=nearest*(1.0+1e-5)&&abs(slots[i].y-distance)<=1e-6*max(1.0,distance)){sum+=slots[i].x;count+=1.0;}}
 return UMNeighbor(select(0.0,sum/count,count>0.0),distance,sqrt(nearest));
}
// Callers evaluate only faces ueFaceLive admits: a face none of whose
// readable slots is finite would return old here.
fn umExtended(face:UMFace,owner:UMOwner)->vec2f{
 let old=stateIn[umSlot(face.anchor,face.axis,face.width)];
 if(old.y==0.0||(umTileSupport(owner.tile)&2u)==0u||umClosedWall(face)){return old;}let center=umFaceCenter(face);let width=f32(face.width);
 var low:array<UMNeighbor,3>;var high:array<UMNeighbor,3>;var minima:array<f32,3>;var spacing:array<f32,3>;
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=width;
  low[axis]=umNeighbor(center-delta,center,face.axis,axis,face.width);high[axis]=umNeighbor(center+delta,center,face.axis,axis,face.width);
  let lower=low[axis].distance<=high[axis].distance;
  minima[axis]=select(high[axis].distance,low[axis].distance,lower);
  spacing[axis]=select(high[axis].spacing,low[axis].spacing,lower);
 }
 var sorted=minima;var lengths=spacing;
 for(var i=0u;i<2u;i++){for(var j=i+1u;j<3u;j++){if(sorted[j]<sorted[i]){
  let d=sorted[i];sorted[i]=sorted[j];sorted[j]=d;let s=lengths[i];lengths[i]=lengths[j];lengths[j]=s;
 }}}
 if(sorted[0]>=0.5*UM_INF){return old;}var root=sorted[0]+lengths[0];
 for(var count=2u;count<=3u;count++){
  if(sorted[count-1u]>=0.5*UM_INF||root<=sorted[count-1u]){break;}
  var a=0.0;var b=0.0;var c=-1.0;
  for(var i=0u;i<count;i++){let w=1.0/(lengths[i]*lengths[i]);a+=w;b+=sorted[i]*w;c+=sorted[i]*sorted[i]*w;}
  root=(b+sqrt(max(b*b-a*c,0.0)))/a;
 }
 if(root>2.0*width*min(h.x,min(h.y,h.z))||root>old.y){return old;}
 var weighted=0.0;var total=0.0;let epsilon=max(root,2.0*width*min(h.x,min(h.y,h.z)))*1.1920929e-7;
 for(var axis=0u;axis<3u;axis++){
  if(minima[axis]>=root-epsilon){continue;}var value=0.0;var count=0.0;
  if(abs(low[axis].distance-minima[axis])<=epsilon){value+=low[axis].value;count+=1.0;}
  if(abs(high[axis].distance-minima[axis])<=epsilon){value+=high[axis].value;count+=1.0;}
  let weight=(root-minima[axis])/(spacing[axis]*spacing[axis]);weighted+=weight*value/max(count,1.0);total+=weight;
 }
 return vec2f(select(0.0,weighted/total,total>0.0),root);
}
// Frozen equal-width stencils certify one unit patch at every fine face.
fn umUnitExtensionFace(owner:UMOwner,axis:u32,sign:i32)->UMFace{
 let origin=vec3i(umOrigin(owner));var neighbor=origin;neighbor[axis]+=sign;
 var anchor=origin;if(sign<0){anchor[axis]-=1;}
 return UMFace(umOwnerAt(neighbor),anchor,1u,1u,axis,sign);
}
fn umSeedState(face:UMFace,owner:UMOwner)->vec2f{
 return select(vec2f(0.0,UM_INF),vec2f(umPhysical(face),0.0),umSource(face,owner));
}
// Seed and sweep outputs. The seed writes every slot and mask into BOTH
// slot arrays. A sweep result whose distance is 0 or INF is its seed value
// (distances only fall, a sweep root is positive, and an INF slot keeps its
// seed value), and so is every slot a sweep skips; the array a sweep writes
// holds that seed value already (from the seed, or from a pass that skipped
// it too). A sweep therefore stores only distances outside {0, INF} and
// never revisits a domain-wall patch (seed-only). Exact for any sweep count.
fn ueSeedStore(slot:u32,value:vec2f){stateOut[slot]=value;stateIn[slot]=value;}
fn ueSweepStore(slot:u32,value:vec2f){if(value.y<0.5*UM_INF&&value.y!=0.0){stateOut[slot]=value;}}
// Negative domain-wall patches are never swept: the seed writes them once.
// Returns whether its distance is finite.
fn umSeedWall(face:UMFace,owner:UMOwner)->bool{
 let source=umSource(face,owner);
 ueSeedStore(umSlot(face.anchor,face.axis,face.width),vec2f(umPhysical(face),select(UM_INF,0.0,source)));
 return source;
}
// Per tile, in the slot array a pass wrote: a 64-bit mask of the tile's
// cells holding the (clamped) anchor of a finite slot, as four exact 16-bit
// floats in two words (UE_FLAGS+2 tile). Only supported tiles' masks are
// current (ueMaskIn gates on support). Masks only grow: a pass's mask is its
// input mask or'ed with the anchors of the finite slots it wrote.
fn ueBit(anchor:vec3i)->vec2u{
 let l=vec3u(clamp(anchor,vec3i(0),vec3i(UM_D)-vec3i(1)))%4u;let b=l.x+4u*(l.y+4u*l.z);
 return select(vec2u(1u<<(b&31u),0u),vec2u(0u,1u<<(b&31u)),b>=32u);
}
fn ueMaskIn(tile:u32)->vec2u{
 if((umTileSupport(tile)&2u)==0u){return vec2u(0u);}
 let a=stateIn[UE_FLAGS+2u*tile];let b=stateIn[UE_FLAGS+2u*tile+1u];
 return vec2u(u32(a.x)|(u32(a.y)<<16u),u32(b.x)|(u32(b.y)<<16u));
}
fn ueMaskOut(tile:u32,m:vec2u,seed:bool){
 let a=vec2f(f32(m.x&0xffffu),f32(m.x>>16u));let b=vec2f(f32(m.y&0xffffu),f32(m.y>>16u));
 stateOut[UE_FLAGS+2u*tile]=a;stateOut[UE_FLAGS+2u*tile+1u]=b;
 if(seed){stateIn[UE_FLAGS+2u*tile]=a;stateIn[UE_FLAGS+2u*tile+1u]=b;}
}
// Tile jobs stage the 3x3x3 tiles around theirs: width (0 outside the
// lattice) and input mask. The job is live when any mask is nonzero.
var<workgroup> ueLive:atomic<u32>;
var<workgroup> ueNew:array<atomic<u32>,2>;
var<workgroup> ueFast:atomic<u32>;
var<workgroup> ueGeneral:atomic<u32>;
var<workgroup> ueQueue:array<u32,192>;
var<workgroup> ueWidths:array<u32,27>;
var<workgroup> ueMasks:array<vec2u,27>;
// xyz: the job tile; w: its index, or UM_TILES without a supported owner.
var<workgroup> ueJob:vec4u;
fn ueStaged(t:vec3i)->u32{let r=vec3u(t-vec3i(ueJob.xyz)+vec3i(1));return r.x+3u*(r.y+3u*r.z);}
fn ueCellFinite(cell:vec3i)->bool{
 let t=cell/4;let m=ueMasks[ueStaged(t)];let l=vec3u(cell-4*t);let b=l.x+4u*(l.y+4u*l.z);
 return ((select(m.x,m.y,b>=32u)>>(b&31u))&1u)!=0u;
}
// Stages the neighbourhood of lane 0's owner's tile (every lane of a tile
// job holds an owner of that tile, or none). Workgroup-uniform result.
fn ueStage(owner:UMOwner,lane:u32,seed:bool)->bool{
 if(lane==0u){
  ueJob=vec4u(umTileCoord(owner.tile),select(UM_TILES,owner.tile,owner.width!=0u&&(umTileSupport(owner.tile)&2u)!=0u));
  atomicStore(&ueLive,0u);atomicStore(&ueNew[0],0u);atomicStore(&ueNew[1],0u);atomicStore(&ueFast,0u);atomicStore(&ueGeneral,0u);
 }
 let job=workgroupUniformLoad(&ueJob);
 if(job.w>=UM_TILES){return false;}
 if(seed){return true;}
 if(lane<27u){
  let t=vec3i(job.xyz)+vec3i(umCorner(lane,3u))-vec3i(1);var width=0u;var mask=vec2u(0u);
  if(all(t>=vec3i(0))&&all(t<vec3i(UM_T))){let tile=umTileAt(vec3u(t));width=umTileWidth(tile);mask=ueMaskIn(tile);}
  ueWidths[lane]=width;ueMasks[lane]=mask;if(any(mask!=vec2u(0u))){atomicOr(&ueLive,1u);}
 }
 return workgroupUniformLoad(&ueLive)!=0u;
}
// Whether umExtended(face) can differ from its old slot: some slot it may
// read is finite. 0: none (umExtended returns old); 1: some, every read in a
// unit tile; 2: some, and a read may reach a 4h tile (the general path).
// Every slot a request reads is anchored in the cells tested here:
// - old: the face's anchor.
// - request point P = centre +- width e_step inside the lattice, q its
//   (clamped) anchor round(P - offset), Tq the tile of q (in the job's 3x3x3:
//   q is within one width of the face's anchor, whose tile is the job's or,
//   for a 4h owner's patch, whose +axis neighbour is staged).
// - width 1, Tq unit: the unit slot at q only (lowWidth is 1: the direct
//   case; a domain-wall patch at q[axis] = -1 is attributed to q).
// - width 1, Tq 4h: the unit slot at q, and the site owner O (the 4h owner
//   of Tq) patches covering q: its +axis patch in Tq's top layer at q's
//   columns (unit patches) or at O's origin columns (a width-4 patch), and
//   when P's plane is strictly inside O (plane % 4 != 0) its -axis patch in
//   the layer below O at the same columns (a domain wall: layer 0 of Tq);
//   at plane 0 the site is O's -axis domain-wall patch (layer 0 of Tq).
// - width 4: P is a 4h-aligned patch centre; every direct, site and tied
//   candidate lies in Tq (never interior: its plane is a multiple of four).
fn ueFaceLive(anchor:vec3i,axis:u32,width:u32)->u32{
 var live=ueCellFinite(clamp(anchor,vec3i(0),vec3i(UM_D)-vec3i(1)));var coarse=width!=1u;
 var center=vec3f(anchor)+vec3f(0.5*f32(width));center[axis]=f32(anchor[axis]+1);
 var offset=vec3f(0.5*f32(width));offset[axis]=1.0;
 for(var n=0u;n<6u;n++){
  var point=center;point[n/2u]+=select(-f32(width),f32(width),(n&1u)!=0u);
  if(any(point<vec3f(0))||any(point>vec3f(UM_D))){continue;}
  let q=clamp(vec3i(round(point-offset)),vec3i(0),vec3i(UM_D)-vec3i(1));let t=q/4;
  if(width!=1u){live=live||any(ueMasks[ueStaged(t)]!=vec2u(0u));continue;}
  if(ueWidths[ueStaged(t)]==1u){live=live||ueCellFinite(q);continue;}
  coarse=true;
  let o=4*t;var top=q;top[axis]=o[axis]+3;var corner=o;corner[axis]=o[axis]+3;
  live=live||ueCellFinite(q)||ueCellFinite(top)||ueCellFinite(corner);
  let plane=i32(round(point[axis]));
  if(plane%4!=0||plane==0){
   var below=q;below[axis]=max(o[axis]-1,0);var belowCorner=o;belowCorner[axis]=below[axis];
   live=live||ueCellFinite(below)||ueCellFinite(belowCorner);
  }
 }
 return select(0u,select(1u,2u,coarse),live);
}
// Queue a live item: unit-only items from the front, the rest from the back,
// so the general sampler runs in as few SIMD groups as possible.
fn ueEnqueue(item:u32,kind:u32){
 if(kind==1u){ueQueue[atomicAdd(&ueFast,1u)]=item;}
 else if(kind==2u){ueQueue[191u-atomicAdd(&ueGeneral,1u)]=item;}
}
fn ueQueueItem(i:u32,fast:u32)->u32{if(i<fast){return ueQueue[i];}return ueQueue[191u-(i-fast)];}
// The anchor bit of a stored (finite, nonzero) sweep result. Lanes gather
// their bits and or them into ueNew once.
fn ueNote(value:vec2f,anchor:vec3i)->vec2u{
 if(value.y<0.5*UM_INF&&value.y!=0.0){return ueBit(anchor);}return vec2u(0u);
}
fn ueGather(bits:vec2u){if(any(bits!=vec2u(0u))){atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);}}
// The job's output mask: its input mask or the finite slots it wrote. A
// zero mask is already in the output array (the seed wrote it, and masks
// only grow).
fn ueFlush(){
 let m=ueMasks[13]|vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1]));
 if(any(m!=vec2u(0u))){ueMaskOut(ueJob.w,m,false);}
}
// The regular 4h launch takes the ownership's packed regular 4h list (one
// lane per owner, no seam tiles to filter), including a tier small enough to
// be fused elsewhere: the owners are the tier list's regular 4h owners.
override ueRegularCoarseList:bool=false;
fn ueOwner(gid:vec3u)->UMOwner{
 if(ueRegularCoarseList){return umRegularCoarseOwner(gid.x+umDispatchX*64u*gid.y);}
 return umOwner(gid);
}
// Off the extension support seed and sweeps write nothing: every reader of
// such a slot goes through umSlotState (or, in restrictBand, skips the tile).
// One entry per launch (regular h, regular 4h list, seam h, seam 4h), so a
// split trace prices each launch.
// The seed of one owner; returns the anchors of its finite slots.
fn ueSeedOwner(owner:UMOwner)->vec2u{
 let origin=umOrigin(owner);var bits=vec2u(0u);
 // A unit owner's faces are the unit patches whatever its neighbours.
 if(umRegularFine||owner.width==1u){
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);if(umSeedWall(face,owner)){bits|=ueBit(face.anchor);}}
   let face=umUnitExtensionFace(owner,axis,1);
   let value=umSeedState(face,owner);ueSeedStore(umSlot(face.anchor,axis,1u),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}
  }
  return bits;
 }
 // Every patch owns its component slot: no anchor packing across axes.
 for(var axis=0u;axis<3u;axis++){
  if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);if(umSeedWall(face,owner)){bits|=ueBit(face.anchor);}}
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,1,part);
   let value=umSeedState(face,owner);ueSeedStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}
  }
 }
 return bits;
}
@compute @workgroup_size(64) fn seedList(@builtin(global_invocation_id) gid:vec3u){
 let owner=ueOwner(gid);if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u){return;}
 ueMaskOut(owner.tile,ueSeedOwner(owner),true);
}
// A regular 4h owner's faces are single width-4 patches: every slot they
// read lies in its tile or a face neighbour (ueFaceLive, width 4).
@compute @workgroup_size(64) fn sweepList(@builtin(global_invocation_id) gid:vec3u){
 let owner=ueOwner(gid);if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u){return;}
 let t=vec3i(umTileCoord(owner.tile));let own=ueMaskIn(owner.tile);var live=any(own!=vec2u(0u));
 for(var n=0u;n<6u;n++){
  if(live){break;}
  var s=t;s[n/2u]+=select(-1,1,(n&1u)!=0u);
  if(all(s>=vec3i(0))&&all(s<vec3i(UM_T))){live=any(ueMaskIn(umTileAt(vec3u(s)))!=vec2u(0u));}
 }
 if(!live){return;}
 var bits=own;
 for(var axis=0u;axis<3u;axis++){
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,1,part);let value=umExtended(face,owner);
   ueSweepStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}
  }
 }
 if(any(bits!=vec2u(0u))){ueMaskOut(owner.tile,bits,false);}
}
${["","Seam"].map(suffix=>/* wgsl */`
@compute @workgroup_size(64) fn seed${suffix}(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 // Every lane of a job holds an owner of one tile (or none).
 let owner=ueOwner(gid);
 if(!ueStage(owner,lane,true)){return;}
 let bits=ueSeedOwner(owner);atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);
 workgroupBarrier();
 if(lane==0u){ueMaskOut(ueJob.w,vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1])),true);}
}
// Unit tile: one item per (owner, axis); a unit owner's positive patch is
// anchored at its origin. A regular h tile has no 4h tile in its stencil
// (every live face is fast): each lane evaluates its own live faces, with no
// queue. Seam h tiles queue live faces, general ones last.
@compute @workgroup_size(64) fn sweep${suffix}(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=ueOwner(gid);
 if(!ueStage(owner,lane,false)){return;}
 var bits=vec2u(0u);
 if(umRegularFine){
  for(var axis=0u;axis<3u;axis++){
   if(ueFaceLive(vec3i(umOrigin(owner)),axis,1u)==0u){continue;}
   let face=umUnitExtensionFace(owner,axis,1);let value=umExtended(face,owner);
   ueSweepStore(umSlot(face.anchor,axis,1u),value);bits|=ueNote(value,face.anchor);
  }
 }else{
  for(var axis=0u;axis<3u;axis++){ueEnqueue(lane+64u*axis,ueFaceLive(vec3i(umOrigin(owner)),axis,1u));}
 }
 workgroupBarrier();
 let fast=atomicLoad(&ueFast);let total=fast+atomicLoad(&ueGeneral);
 for(var i=lane;i<total;i+=64u){
  let item=ueQueueItem(i,fast);var o=owner;o.lane=item%64u;o.index=owner.index-owner.lane+o.lane;
  let face=umUnitExtensionFace(o,item/64u,1);let value=umExtended(face,o);
  ueSweepStore(umSlot(face.anchor,face.axis,1u),value);bits|=ueNote(value,face.anchor);
 }
 ueGather(bits);
 workgroupBarrier();
 if(lane==0u){ueFlush();}
}`).join("\n")}
// Coarse seam owners: one group per tile, one lane per patch and component.
// A 4h owner beside h tiles has sixteen patches per face; evaluating them
// serially in one lane made the coarse seam tier a latency chain.
fn umSeamTileOwner(group:vec3u,local:vec3u)->UMOwner{
 var owner=umTileJobOwner(group);if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u){return UMOwner();}
 let q=local/owner.width;let side=4u/owner.width;owner.lane=q.x+side*(q.y+side*q.z);owner.index+=owner.lane;return owner;
}
// 4h seam tiles hold one owner with at most sixteen patches per positive
// face. One lane per (axis, patch); each writes its own slot directly.
fn umCoarseCell(local:vec3u)->u32{return local.x+4u*(local.y+4u*local.z);}
@compute @workgroup_size(64) fn seedCoarse(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umSeamTileOwner(group,vec3u(0));
 if(!ueStage(owner,lane,true)){return;}
 var bits=vec2u(0u);
 if(lane<48u){
  let axis=lane/16u;let face=umFace(owner,axis,1,lane%16u);
  if(face.width!=0u){let value=umSeedState(face,owner);ueSeedStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits=ueBit(face.anchor);}}
 }else if(lane<51u){
  let axis=lane-48u;if(umOrigin(owner)[axis]==0u){let face=umFace(owner,axis,-1,0u);if(umSeedWall(face,owner)){bits=ueBit(face.anchor);}}
 }
 atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);
 workgroupBarrier();
 if(lane==0u){ueMaskOut(ueJob.w,vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1])),true);}
}
// The positive patches umFace would return, from the staged +axis neighbour
// width (umFace: min of the owner's and the neighbour's, the owner's
// outside the lattice): lane = 16 axis + part.
@compute @workgroup_size(64) fn sweepCoarse(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umSeamTileOwner(group,vec3u(0));
 if(!ueStage(owner,lane,false)){return;}
 if(lane<48u){
  let axis=lane/16u;let part=lane%16u;var probe=vec3i(umOrigin(owner));probe[axis]+=4;
  var width=4u;if(probe[axis]<i32(UM_D[axis])){width=min(4u,ueWidths[ueStaged(probe/4)]);}
  let side=4u/width;
  if(part<side*side){
   var anchor=probe;anchor[(axis+1u)%3u]+=i32((part%side)*width);anchor[(axis+2u)%3u]+=i32((part/side)*width);anchor[axis]-=1;
   ueEnqueue(lane,ueFaceLive(anchor,axis,width));
  }
 }
 workgroupBarrier();
 let fast=atomicLoad(&ueFast);let total=fast+atomicLoad(&ueGeneral);var bits=vec2u(0u);
 for(var i=lane;i<total;i+=64u){
  let item=ueQueueItem(i,fast);let face=umFace(owner,item/16u,1,item%16u);let value=umExtended(face,owner);
  ueSweepStore(umSlot(face.anchor,face.axis,face.width),value);bits|=ueNote(value,face.anchor);
 }
 ueGather(bits);
 workgroupBarrier();
 if(lane==0u){ueFlush();}
}
fn umSourceIndex(p:vec3i)->u32{return u32(p.x+i32(UM_D.x)*(p.y+i32(UM_D.y)*p.z))+1u;}
fn umSourcePoint(i:u32)->vec3i{let at=i-1u;return vec3i(vec3u(at%UM_D.x,(at/UM_D.x)%UM_D.y,at/(UM_D.x*UM_D.y)));}
// One lane per (tile, component) of a 4x4x4 tile block: component-major lane
// order, so a SIMD group restricts one component.
var<workgroup> rbValues:array<f32,192>;
var<workgroup> rbLower:array<u32,192>;
var<workgroup> rbUpper:array<u32,192>;
@compute @workgroup_size(64,3) fn restrictBand(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) id:vec3u){
 let gid=4u*group+vec3u(id.x%4u,(id.x/4u)%4u,id.x/16u);let component=id.y;let at=id.x+64u*component;
 let inside=all(gid<UM_T);let origin=vec3i(gid)*4;let tile=umTileAt(min(gid,UM_T-vec3u(1)));
 // Off the extension support every face keeps its seed distance, and a
 // source face needs a liquid owner, which the support dilates around: no
 // face here is finite, so the restriction is the empty one written below.
 // Every slot read below is anchored in this tile: a supported tile whose
 // final mask is empty restricts to the same empty state.
 let supported=inside&&any(ueMaskIn(tile)!=vec2u(0u));
 rbValues[at]=0.0;rbLower[at]=0u;rbUpper[at]=0u;
 if(supported&&!(gid[component]==UM_T[component]-1u&&!(component==1u&&h.w>0.5))){
  var location=vec3f(origin)+vec3f(2);location[component]+=2.0;
  var best=UM_INF;var sum=0.0;var count=0.0;var lo=vec3i(UM_D);var hi=vec3i(-1);
  ${this.regularBulk?"let coarse=false;let patchWidth=1u;":`// A 4h tile's one owner has a single positive patch (width 4: only the
  // first plane anchor resolves to it) or sixteen unit patches at the plane
  // anchors; the fallback footprint holds no other patch. An h owner always
  // has one unit patch at its positive anchor, even beside a coarse
  // neighbour. No per-anchor owner or neighbour lookup is needed.
  let coarse=umTileWidth(tile)!=1u;var patchWidth=1u;if(coarse){patchWidth=umFace(umOwnerAt(origin),component,1,0u).width;}`}
  // Restrict real MAC patches, with the native vertical footprint fallback.
  for(var fallback=0u;fallback<select(1u,2u,component==1u&&!coarse);fallback++){
   if(count>0.0){break;}
   // Preserve z/y/x order among the 16 face-plane anchors.
   for(var k=0u;k<select(select(16u,64u,fallback!=0u),1u,patchWidth==4u);k++){
    var local=umCorner(k,4u);
    if(fallback==0u){local=vec3u(k%4u,k/4u,3u);if(component==0u){local=vec3u(3u,k%4u,k/4u);}else if(component==1u){local=vec3u(k%4u,3u,k/4u);}}
    let p=origin+vec3i(local);
    ${this.regularBulk?"let o=umOwnerAt(p);let face=umPositiveFaceAtAnchor(o,component,p);if(face.width==0u){continue;}":"let face=UMFace(UMOwner(),p,patchWidth,select(1u,16u,coarse&&patchWidth==1u),component,1);"}
    ${this.regularBulk?"if(!umSource(face,o)){continue;}":"let slot=stateIn[umSlot(face.anchor,component,face.width)];if(slot.y>=0.5*UM_INF){continue;}"}
    let delta=(umFaceCenter(face)-location)*h.xyz;let distance=dot(delta,delta);let epsilon=1e-6*max(1.0,distance);
    if(distance<best-epsilon){best=distance;sum=0.0;count=0.0;lo=vec3i(UM_D);hi=vec3i(-1);}
    if(abs(distance-best)<=epsilon){sum+=${this.regularBulk?"umPhysical(face)":"slot.x"};count+=1.0;
     // The whole tangential patch contributes its original support bounds.
     var end=face.anchor+vec3i(i32(face.width)-1);end[component]=face.anchor[component];lo=min(lo,face.anchor);hi=max(hi,end);}
   }
  }
  if(count>0.0){rbValues[at]=sum/count;rbLower[at]=umSourceIndex(lo);rbUpper[at]=umSourceIndex(hi);}
 }
 workgroupBarrier();
 if(component==0u&&inside){
  var values=vec4f(0);var lower=vec4u(0);var upper=vec4u(0);var mask=0u;
  // A restricted component's lower source index is at least 1.
  for(var c=0u;c<3u;c++){values[c]=rbValues[id.x+64u*c];lower[c]=rbLower[id.x+64u*c];upper[c]=rbUpper[id.x+64u*c];if(lower[c]!=0u){mask|=1u<<c;}}
  values.w=f32(mask);textureStore(coarseOut,vec3i(gid),values);textureStore(originsOut,vec3i(gid),lower);textureStore(originsOut,vec3i(gid)+vec3i(0,0,i32(UM_T.z)),upper);
 }
}
${farValueWGSL("umFarValue","textureLoad(coarse,p,0)","textureLoad(origins,p,0)[face.axis]","textureLoad(origins,p+vec3i(0,0,i32(UM_T.z)),0)[face.axis]")}
// A regular fine publish workgroup is one tile; every far tap of its faces is
// one of the 27 coarse cells around it, staged once in workgroup memory.
var<workgroup> farTile:u32;
var<workgroup> farState:array<vec4f,27>;
// Per coarse cell and component: the source bound box, decoded once (lo.w
// < 0 marks no bound for that component).
var<workgroup> farLo:array<vec4f,81>;
var<workgroup> farHi:array<vec3f,81>;
fn umFarSlot(p:vec3i)->u32{let local=vec3u(p-vec3i(umTileCoord(farTile))+vec3i(1));return local.x+3u*(local.y+3u*local.z);}
// The same selection with staged taps and source boxes decoded once per tile.
fn umFarDistance(face:UMFace,point:vec3f,p:vec3i)->f32{
 let at=3u*umFarSlot(p)+face.axis;let lo=farLo[at];if(lo.w<0.0){return UM_INF;}
 let scale=h.xyz/min(h.x,min(h.y,h.z));let delta=(point-clamp(point,lo.xyz,farHi[at]))*scale;
 return (delta.x*delta.x+delta.z*delta.z)+delta.y*delta.y;
}
fn umFarValueStaged(face:UMFace)->f32{
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;
 let base=vec3i(floor(q));let fraction=fract(q);var best=UM_INF;
 // A constant bound unrolls into registers: each distance is evaluated once.
 var distances:array<f32,8>;var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  distances[k]=UM_INF;let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if(any(weights<=vec3f(0))){continue;}
  let p=clamp(base+bit,vec3i(0),ueFarTop(face.axis));
  distances[k]=umFarDistance(face,point,p);values[k]=farState[umFarSlot(p)][face.axis];best=min(best,distances[k]);
 }
 var sum=0.0;var count=0.0;
 for(var k=0u;k<8u;k++){if(distances[k]<0.5*UM_INF&&abs(distances[k]-best)<=1e-6*max(1.0,best)){sum+=values[k];count+=1.0;}}
 return select(0.0,sum/count,count>0.0);
}
// Bit a set: the tile's in-domain staged cells differ in component a.
var<workgroup> farMixed:atomic<u32>;
// When every staged cell of a component holds one state, each tap umFarValueStaged
// keeps has the same box and value: its distances are equal (all tie), or all
// infinite. Its result is then this same k-ordered sum over the kept taps.
fn umFarValueUniform(face:UMFace,value:f32,valid:bool)->f32{
 if(!valid){return 0.0;}
 let point=umFaceCenter(face);var q=point/4.0-vec3f(0.5);q[face.axis]-=0.5;let fraction=fract(q);
 var sum=0.0;var count=0.0;
 for(var k=0u;k<8u;k++){
  let bit=vec3i(umCorner(k,2u));let weights=select(vec3f(1)-fraction,fraction,bit==vec3i(1));if(any(weights<=vec3f(0))){continue;}
  sum+=value;count+=1.0;
 }
 return select(0.0,sum/count,count>0.0);
}
fn umPublished(face:UMFace)->f32{
 if(face.anchor[face.axis]<0||umClosedWall(face)){return umPhysical(face);}
 ${this.regularBulk?"if(umSource(face,umOwnerAt(face.anchor))){return umPhysical(face);}":"let slot=umSlotState(face.anchor,face.axis,face.width);if(slot.y<0.5*UM_INF){return slot.x;}"}
 return umFarValue(face);
}
@compute @workgroup_size(64) fn publishFine(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=umOwner(gid);
 if(lane==0u){farTile=select(UM_TILES,owner.tile,owner.width!=0u);atomicStore(&farMixed,0u);}
 if(workgroupUniformLoad(&farTile)>=UM_TILES){return;}
 for(var item=lane;item<81u;item+=64u){
  let cell=item/3u;let axis=item%3u;
  let p=vec3i(umTileCoord(farTile))+vec3i(umCorner(cell,3u))-vec3i(1);
  if(all(p>=vec3i(0))&&all(p<vec3i(UM_T))){
   let state=textureLoad(coarse,p,0);if(axis==0u){farState[cell]=state;}
   let lower=textureLoad(origins,p,0)[axis];let upper=textureLoad(origins,p+vec3i(0,0,i32(UM_T.z)),0)[axis];
   var lo=vec3f(umSourcePoint(lower))+vec3f(0.5);var hi=vec3f(umSourcePoint(upper))+vec3f(0.5);lo[axis]+=0.5;hi[axis]+=0.5;
   let valid=(u32(round(state.w))&(1u<<axis))!=0u&&lower!=0u;
   farLo[item]=vec4f(lo,select(-1.0,1.0,valid));farHi[item]=hi;
  }
 }
 workgroupBarrier();
 // Compare every in-domain staged cell with the tile's own (cell 13), bitwise.
 for(var item=lane;item<81u;item+=64u){
  let cell=item/3u;let axis=item%3u;let reference=39u+axis;
  let p=vec3i(umTileCoord(farTile))+vec3i(umCorner(cell,3u))-vec3i(1);
  if(all(p>=vec3i(0))&&all(p<vec3i(UM_T))){
   let a=farLo[item];let b=farLo[reference];
   var same=(a.w<0.0)==(b.w<0.0);
   if(same&&b.w>=0.0){
    same=all(bitcast<vec3u>(a.xyz)==bitcast<vec3u>(b.xyz))&&all(bitcast<vec3u>(farHi[item])==bitcast<vec3u>(farHi[reference]))
     &&bitcast<u32>(farState[cell][axis])==bitcast<u32>(farState[13u][axis]);
   }
   if(!same){atomicOr(&farMixed,1u<<axis);}
  }
 }
 workgroupBarrier();
 let mixed=atomicLoad(&farMixed);
 let origin=umOrigin(owner);var value=vec4f(0);
 for(var axis=0u;axis<3u;axis++){
  if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);boundary[umNegativeBoundaryIndex(origin,axis)]=umPhysical(face);}
  let face=umUnitExtensionFace(owner,axis,1);${this.regularBulk?"let slot=select(vec2f(0,UM_INF),vec2f(umPhysical(face),0),umSource(face,owner));":"let slot=umSlotState(face.anchor,axis,1u);"}
  if(umClosedWall(face)){value[axis]=umPhysical(face);}
  else if(slot.y<0.5*UM_INF){value[axis]=slot.x;}
  else if((mixed&(1u<<axis))==0u){value[axis]=umFarValueUniform(face,farState[13u][axis],farLo[39u+axis].w>=0.0);}
  else{value[axis]=umFarValueStaged(face);}
 }
 value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);
}
var<workgroup> publishCoarseState:array<vec2f,192>;
@compute @workgroup_size(64) fn publishCoarse(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var i=lane;i<192u;i+=64u){publishCoarseState[i]=vec2f(0);}
 workgroupBarrier();
 let owner=umSeamTileOwner(group,vec3u(0));
 if(owner.width!=0u){
  let origin=umOrigin(owner);
  if(lane<48u){
   let axis=lane/16u;let part=lane%16u;var local=vec3u(0);local[axis]=3u;local[(axis+1u)%3u]=part%4u;local[(axis+2u)%3u]=part/4u;
   let face=umPositiveFaceAtAnchor(owner,axis,vec3i(origin+local));if(face.width!=0u){publishCoarseState[umCoarseCell(local)+64u*axis]=vec2f(umPublished(face),1);}
  }else if(lane<51u){let axis=lane-48u;if(origin[axis]==0u){boundary[umNegativeBoundaryIndex(origin,axis)]=umPublished(umFace(owner,axis,-1,0u));}}
 }
 workgroupBarrier();
 let x=publishCoarseState[lane];let y=publishCoarseState[lane+64u];let z=publishCoarseState[lane+128u];
 if(owner.width!=0u&&x.y+y.y+z.y>0.0){let anchor=vec3i(umOrigin(owner)+umCorner(lane,4u));textureStore(output,anchor,vec4f(x.x,y.x,z.x,textureLoad(physical,anchor,0).w));}
}
// h seam owners publish through publishFine (the same values: umPublished
// per unit face, with the staged far value); 4h seam tiles through publishCoarse.
${["publishList"].map(name=>uniformMixedFaceDispatchWGSL(name,"umPublished(face)",false,"value.w=textureLoad(physical,ownedFace.anchor,0).w;","ueOwner").replace(" let origin=umOrigin(owner);",` let origin=umOrigin(owner);
 // A regular 4h owner (one-width 3x3x3 stencil) has one width-4 patch per
 // positive face, anchored at origin+3e_axis, and each of those texels holds
 // only that component (no other positive face shares the anchor). Its far
 // value (umFarValue) has one tap of positive weight: the face centre maps
 // to the owner's own coarse cell with zero fraction on every axis, so the
 // mean over the tied nearest taps is that cell's component when it is known
 // (mask bit and a source bound) and zero otherwise. The same values as the
 // general dispatch below, without the face search.
 if(ueRegularCoarseList){
  let cell=vec3i(origin/4u);let state=textureLoad(coarse,cell,0);let lower=textureLoad(origins,cell,0);let mask=u32(round(state.w));
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let i=umNegativeBoundaryIndex(origin,axis);boundary[i]=negative[i];}
   var anchor=vec3i(origin);anchor[axis]+=3;
   let slot=umSlotState(anchor,axis,4u);let physicalAt=textureLoad(physical,anchor,0);var value=vec4f(0);
   // umClosedWall: a closed positive domain wall publishes its physical value.
   if(anchor[axis]==i32(UM_D[axis])-1&&!(axis==1u&&h.w>0.5)){value[axis]=physicalAt[axis];}
   else if(slot.y<0.5*UM_INF){value[axis]=slot.x;}
   else if((mask&(1u<<axis))!=0u&&lower[axis]!=0u){value[axis]=state[axis];}
   value.w=physicalAt.w;textureStore(output,anchor,value);
  }
  return;
 }
 // A unit owner's faces are the unit patches whatever its neighbours.
 if(umRegularFine||owner.width==1u){
  var value=vec4f(0);
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);boundary[umNegativeBoundaryIndex(origin,axis)]=umPublished(face);}
   value[axis]=umPublished(umUnitExtensionFace(owner,axis,1));
  }
  value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);return;
 }`)).join("\n")}
`,["seed","sweep","seedList","sweepList","seedSeam","sweepSeam","seedCoarse","sweepCoarse","publishList","publishFine","publishCoarse"])});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  this.restrictPipeline=await this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"restrictBand",constants:{umDispatchX:this.ownership.dispatchX}}});
  await Promise.all((this.regularBulk?["publish"]:["seed","sweep","publish"]).map(async entryPoint=>{
   const compile=(width:number,regular:boolean,counted:number,name=entryPoint,list=false)=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:name,constants:{umDispatchX:this.ownership.dispatchX,umCellWidth:width,umRegularTiles:+regular,umInterfaceTiles:+!regular,umRegularFine:+(regular&&width===1),ueRegularCoarseList:+list,umCountedJobs:counted}}});
   const C=UNIFORM_MIXED_COUNTED;
   this.regularPipelines.set(entryPoint,await compile(1,true,C.owners,entryPoint==="publish"?"publishFine":entryPoint));
   this.regularCoarseListPipelines.set(entryPoint,await compile(4,true,C.regularCoarse,`${entryPoint}List`,true));
   this.seamPipelines.set(entryPoint,await compileMixedTiers(w=>w===1?compile(1,false,C.owners,entryPoint==="publish"?"publishFine":`${entryPoint}Seam`):compile(4,false,C.tiles,`${entryPoint}Coarse`)));
  }));
 }
 encode(encoder:GPUCommandEncoder,groups:readonly [GPUBindGroup,GPUBindGroup],sweeps=2):void{
  if(this.seamPipelines.size!==(this.regularBulk?1:3)||!this.restrictPipeline)throw new Error("Mixed extension is not initialized");
  // One pass: seed, sweeps, restriction, the hierarchy continuation, publish.
  let open:GPUComputePassEncoder|undefined;
  const run=(entry:string,group:GPUBindGroup)=>{
   if(!open){open=encoder.beginComputePass({label:"Uniform mixed extension"});open.setBindGroup(0,this.ownership.bindGroup);}
   const pass=open;pass.setBindGroup(1,group);
   if(entry==="restrictBand"){pass.setPipeline(this.restrictPipeline!);pass.dispatchWorkgroups(...this.ownership.capacity.tileDimensions.map(n=>Math.ceil(n/4)) as [number,number,number]);}
   else {
    // Four counted launches, each empty (not skipped) when its GPU count is zero.
    const o=this.ownership;
    o.dispatchTierCounted(pass,this.regularPipelines.get(entry)!,0);
    o.dispatchRegularCoarseCounted(pass,this.regularCoarseListPipelines.get(entry)!);
    o.dispatchTiersCounted(pass,this.seamPipelines.get(entry)!,[false,true]);
   }
  };
  const end=()=>{open?.end();open=undefined;};
  if(!Number.isSafeInteger(sweeps)||sweeps<0)throw new Error("Invalid mixed extension sweep count");
  // Bulk extension uses only the regular hierarchy. Its restriction reads
  // physical supported faces directly, and publication preserves them at h.
  if(!this.regularBulk){run("seed",groups[1]);for(let i=0;i<sweeps;i++)run("sweep",groups[i%2]!);}
  const final=groups[sweeps%2]!;run("restrictBand",final);
  this.hierarchy.encode(open!);open!.setBindGroup(0,this.ownership.bindGroup);run("publish",final);end();
 }
}
