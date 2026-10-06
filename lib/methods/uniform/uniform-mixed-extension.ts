import { uniformCompiledExtensionNeighborWGSL } from "./uniform-compiled-extension.wgsl";
import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
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
/** (value, distance) slots per canonical patch and component at `fineTiles`
 * h tiles. Fixed by the lattice: width-4 patches in a component-major n/4
 * layer, the negative domain-wall planes of each width, a two-word
 * finite-cell mask per tile (UE_FLAGS) and the no-patch slot. Then the unit
 * patches by fine-tile rank, 240 an h tile. See umSlot. */
function extensionSlots(d:readonly number[],fineTiles:number):number{
 const t=d.map(n=>n/4),plane=(a:readonly number[])=>a[1]!*a[2]!+a[0]!*a[2]!+a[0]!*a[1]!;
 return 5*t[0]!*t[1]!*t[2]!+plane(d)+plane(t)+1+240*fineTiles;
}
/** Seam 4h tiles per sweepSeams job (ueSweepCoarsePack). */
const UE_PACK_TILES=2;
/** Regular-bulk mode restricts physical source faces directly to the 4h
 * nearest-source hierarchy, then retains supported h faces on publication.
 * The legacy mode first performs Godunov sweeps on canonical mixed patches;
 * its two transient vec2f slot arrays borrow the caller's stage scratch. */
export class UniformMixedExtension {
 readonly allocatedBytes=0;
 /** One slot array at `fineTiles` h tiles. */
 private slotBytesAt(fineTiles:number):number{return Math.ceil(8*extensionSlots(this.ownership.capacity.lattice.dimensions,fineTiles)/256)*256;}
 /** bind()'s scratch at `fineTiles` h tiles: both slot arrays. bind() takes
  * the ownership's capacity, so a capacity change binds again. */
 scratchBytesAt(fineTiles:number):number{return 2*this.slotBytesAt(fineTiles);}
 private readonly resources:GPUBindGroupLayout;
 /** GPU-counted launches (UNIFORM_MIXED_COUNTED), each of a fixed grid: no
  * launch size, variant or skip reads the host membership. Regular h owners. */
 private readonly regularPipelines=new Map<string,GPUComputePipeline>();
 /** Every regular 4h owner, 64 per job, fused tier or not. */
 private readonly listPipelines=new Map<string,GPUComputePipeline>();
 /** Seed/publication share a seam launch. Sweeps separate h seams from
  * regular and packed seam 4h jobs, so h jobs compile with one staged tile. */
 private readonly seamPipelines=new Map<string,GPUComputePipeline>();
 private fineSeamSweep?:GPUComputePipeline;
 private restrictPipeline?:GPUComputePipeline;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,readonly hierarchy:Hierarchy,private readonly regularBulk=false,directRestriction=false){
  if(ownership.capacity.lattice.dimensions.some(n=>n%4!==0))throw new Error("Mixed extension requires a 4-aligned lattice");
  if(!directRestriction)throw new Error("Mixed extension restricts the face-plane anchors directly");
  this.resources=uniformDetailBindLayout(device,{entries:[
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
 bind(f:Fields):readonly [UniformDetailGroup,UniformDetailGroup]{
  const d=this.ownership.capacity.lattice.dimensions;
  for(const [i,t] of [f.physical,f.phase,f.output].entries())if(t.format!==(i===1?"r32float":"rgba32float")||uniformDetailExtent(t).some((n,a)=>n!==d[a]))throw new Error("Mixed extension requires native canonical fields");
  if(f.physical===f.output||f.negative===f.outputNegative)throw new Error("Mixed extension outputs must be disjoint");
  const size=this.slotBytesAt(this.ownership.capacity.fineTiles),offset=f.scratch.offset??0;
  if((f.scratch.size??f.scratch.buffer.size-offset)<2*size)throw new Error("Mixed extension scratch is too small");
  return [0,1].map(parity=>uniformDetailGroup(this.device,{layout:this.resources,entries:[
   ...[parity,parity^1].map((slot,binding)=>({binding,resource:{buffer:f.scratch.buffer,offset:offset+slot*size,size}})),
   {binding:4,resource:f.physical},{binding:5,resource:f.phase},
   {binding:6,resource:{buffer:f.negative}},{binding:7,resource:{buffer:f.params,size:16}},
   {binding:8,resource:f.output},{binding:9,resource:this.hierarchy.output},
   {binding:10,resource:this.hierarchy.outputOrigins},
   {binding:11,resource:this.hierarchy.input},{binding:12,resource:this.hierarchy.inputOrigins},
   {binding:13,resource:{buffer:f.outputNegative}},
  ]})) as unknown as readonly [UniformDetailGroup,UniformDetailGroup];
 }
 async initialize():Promise<void>{
  const module=uniformDetailModule(this.device,{label:"Uniform mixed extension",code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
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
// One (value, distance) slot per canonical patch and component. The sections
// the lattice fixes come first. Width-4 patches (anchor/4 is their tile) sit
// in a component-major n/4 layer, so neighbouring 4h owners are neighbouring
// words. Negative domain-wall patches (anchor[axis] = -1) have planes of
// their own. Their addresses need the patch width, never a topology load.
const UE_TILES=UM_T.x*UM_T.y*UM_T.z;
const UE_COARSE=0u;
const UE_UNIT_WALL=UE_COARSE+3u*UE_TILES;
const UE_COARSE_WALL=UE_UNIT_WALL+UM_D.y*UM_D.z+UM_D.x*UM_D.z+UM_D.x*UM_D.y;
// Per tile finite-cell masks (ueMaskIn), two words per tile.
const UE_FLAGS=UE_COARSE_WALL+UM_T.y*UM_T.z+UM_T.x*UM_T.z+UM_T.x*UM_T.y;
// The slot of a unit address that names no patch: never written, and read
// only by loads issued ahead of the test that discards them.
const UE_NONE=UE_FLAGS+2u*UE_TILES;
// Unit patches by fine-tile rank (h tiles are packed first: a tile word is
// 64 * rank), 240 slots an h tile. (component, 64 cells) for the patches
// anchored in the tile: its lanes read one 512-byte run per component. Then
// per component the sixteen unit patches of a 4h tile below it on that axis
// (a 4h seam owner's, anchored in that tile's top layer), by column. The
// address takes the anchor tile's word, and the word of the tile above when
// the anchor tile is 4h (upper; 0 without one).
const UE_UNIT=UE_NONE+1u;
fn ueUnitSlotIn(word:u32,upper:u32,l:vec3u,axis:u32)->u32{
 if((word&0x80000000u)!=0u){return UE_UNIT+15u*((word&0x3fffffffu)>>2u)+axis*64u+l.x+4u*(l.y+4u*l.z);}
 return select(UE_NONE,UE_UNIT+15u*((upper&0x3fffffffu)>>2u)+192u+16u*axis+l[(axis+1u)%3u]+4u*l[(axis+2u)%3u],l[axis]==3u&&(upper&0x80000000u)!=0u);
}
// A unit owner's positive patch is anchored at its origin: no tile word.
fn ueOwnSlot(owner:UMOwner,axis:u32)->u32{return UE_UNIT+15u*((owner.index-owner.lane)>>2u)+axis*64u+owner.lane;}
// The sweep tile jobs stage the words of the 3x3x3 tiles around theirs
// (ueStage), which hold every anchor their faces read (ueFaceLive): those
// jobs take a word from workgroup memory instead of a topology load.
// The tile above a 4h tile may lie outside the 3x3x3: its word is loaded.
// ueMixedJobs: the launch also holds jobs that stage nothing (sweepSeams'
// regular 4h jobs); ueStagedJob then says whether this invocation's job did.
override ueStagedTiles:bool=false;
override ueMixedJobs:bool=false;
var<private> ueStagedJob:bool=false;
// The mixed launch packs its seam 4h tiles UE_PACK a job (ueSweepCoarsePack):
// pack slot s stages its 3x3x3 at 27 s. There a lane's staged tile is its
// pick (xyz: the tile; w: 27 s), set by the job it is serving (a one-tile
// job: ueStage, w = 0); every other launch reads ueJob and slot 0.
const UE_PACK=${UE_PACK_TILES}u;
// Fine-only sweeps use one neighborhood; packed coarse sweeps need two.
override ueStagePack:u32=${UE_PACK_TILES}u;
var<workgroup> ueWords:array<u32,27u*ueStagePack>;
// xyz: the job tile; w: its index, or UM_TILES without a supported owner
// (a sweep's: without an open one).
var<workgroup> ueJob:vec4u;
var<private> uePick:vec4u;
var<private> ueRecipe:vec2u;
var<workgroup> ueJobRecipe:vec2u;
fn ueJobTile()->vec3i{if(ueMixedJobs){return vec3i(uePick.xyz);}return vec3i(ueJob.xyz);}
fn ueJobBase()->u32{if(ueMixedJobs){return uePick.w;}return 0u;}
fn ueTileWord(t:vec3u)->u32{
 if(ueStagedTiles&&(!ueMixedJobs||ueStagedJob)){let r=vec3u(vec3i(t)-ueJobTile()+vec3i(1));if(all(r<=vec3u(2u))){return ueWords[ueJobBase()+r.x+3u*(r.y+3u*r.z)];}}
 return umTopology[umTileAt(t)];
}
fn ueUnitSlot(p:vec3u,axis:u32)->u32{
 let t=p/4u;let l=p%4u;let word=ueTileWord(t);var upper=0u;
 if((word&0x80000000u)==0u&&l[axis]==3u&&t[axis]+1u<UM_T[axis]){var above=t;above[axis]+=1u;upper=ueTileWord(above);}
 return ueUnitSlotIn(word,upper,l,axis);
}
fn ueCoarseWallIndex(t:vec3u,axis:u32)->u32{
 if(axis==0u){return t.y+UM_T.y*t.z;}
 if(axis==1u){return UM_T.y*UM_T.z+t.x+UM_T.x*t.z;}
 return UM_T.y*UM_T.z+UM_T.x*UM_T.z+t.x+UM_T.x*t.y;
}
fn umSlot(anchor:vec3i,axis:u32,width:u32)->u32{
 let p=vec3u(max(anchor,vec3i(0)));
 if(anchor[axis]<0){return select(UE_COARSE_WALL+ueCoarseWallIndex(p/4u,axis),UE_UNIT_WALL+umNegativeBoundaryIndex(p,axis),width==1u);}
 if(width==1u){return ueUnitSlot(p,axis);}
 return UE_COARSE+axis*UE_TILES+umTileAt(p/4u);
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
 // The support word loads beside the slot's address (a unit slot's tile word).
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
 // the all-h stencil path). Its anchor needs no topology, so the support and
 // width words issue beside the slot's address instead of behind its value.
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
// One patch from its six neighbours (low and high along each axis): the
// Godunov root over the nearer of each pair, cut off at two patch widths and
// at the old distance, then the mean value of the neighbours that reach it.
fn ueCombine(old:vec2f,width:f32,low:array<UMNeighbor,3>,high:array<UMNeighbor,3>)->vec2f{
 var minima:array<f32,3>;var spacing:array<f32,3>;
 for(var axis=0u;axis<3u;axis++){
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
// Callers evaluate only faces ueFaceLive admits: a face none of whose
// readable slots is finite would return old here.
fn umExtended(face:UMFace,owner:UMOwner)->vec2f{
 // A unit owner's face is its own positive patch (every caller's).
 var slot=0u;if(owner.width==1u){slot=ueOwnSlot(owner,face.axis);}else{slot=umSlot(face.anchor,face.axis,face.width);}
 let old=stateIn[slot];
 if(old.y==0.0||(umTileSupport(owner.tile)&2u)==0u||umClosedWall(face)){return old;}let center=umFaceCenter(face);let width=f32(face.width);
 var low:array<UMNeighbor,3>;var high:array<UMNeighbor,3>;
 for(var axis=0u;axis<3u;axis++){
  var delta=vec3f(0);delta[axis]=width;
  low[axis]=umNeighbor(center-delta,center,face.axis,axis,face.width);high[axis]=umNeighbor(center+delta,center,face.axis,axis,face.width);
 }
 return ueCombine(old,width,low,high);
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
// The last float also carries the tile's open bit (65536): the seed left a
// positive patch that is not a source. A seed is a source (distance 0, which
// no sweep changes) or INF, so a tile without the bit stores nothing in any
// sweep and its mask stays its seed mask in both arrays: its sweep job ends
// at this word (ueOpen). The sweeps' tile work is the tiles holding a patch
// still to reach, not every tile of liquid.
fn ueBit(anchor:vec3i)->vec2u{
 let l=vec3u(clamp(anchor,vec3i(0),vec3i(UM_D)-vec3i(1)))%4u;let b=l.x+4u*(l.y+4u*l.z);
 return select(vec2u(1u<<(b&31u),0u),vec2u(0u,1u<<(b&31u)),b>=32u);
}
fn ueMaskIn(tile:u32)->vec2u{
 if((umTileSupport(tile)&2u)==0u){return vec2u(0u);}
 let a=stateIn[UE_FLAGS+2u*tile];let b=stateIn[UE_FLAGS+2u*tile+1u];
 return vec2u(u32(a.x)|(u32(a.y)<<16u),u32(b.x)|((u32(b.y)&0xffffu)<<16u));
}
fn ueOpen(tile:u32)->bool{return stateIn[UE_FLAGS+2u*tile+1u].y>=65536.0;}
fn ueMaskOut(tile:u32,m:vec2u,seed:bool,open:bool){
 let a=vec2f(f32(m.x&0xffffu),f32(m.x>>16u));let b=vec2f(f32(m.y&0xffffu),f32((m.y>>16u)|select(0u,0x10000u,open)));
 stateOut[UE_FLAGS+2u*tile]=a;stateOut[UE_FLAGS+2u*tile+1u]=b;
 if(seed){stateIn[UE_FLAGS+2u*tile]=a;stateIn[UE_FLAGS+2u*tile+1u]=b;}
}
// Tile jobs stage the 3x3x3 tiles around theirs: width (0 outside the
// lattice) and input mask. A 4h tile job is live when any mask is nonzero;
// an h tile job when one of its cells is (ueCellLive). ueLive: bit 0 live;
// an h tile job also notes, per face direction k = 2 axis + side, a 4h face
// neighbour (bit 1 + k) and whether a request into it can read a finite
// slot (bit 7 + k).
var<workgroup> ueLive:atomic<u32>;
var<workgroup> ueNew:array<atomic<u32>,2>;
// Width is already encoded in ueWords. Outside entries are zero: callers
// either ask only whether they are fine, or have checked the domain bounds.
// Avoid a duplicate 27-word width array for each packed job.
fn ueStagedWidth(index:u32)->u32{return select(4u,1u,(ueWords[index]&0x80000000u)!=0u);}
var<workgroup> ueMasks:array<vec2u,27u*ueStagePack>;
fn ueStaged(t:vec3i)->u32{let r=vec3u(t-ueJobTile()+vec3i(1));return ueJobBase()+r.x+3u*(r.y+3u*r.z);}
fn ueCellFinite(cell:vec3i)->bool{
 let t=cell/4;let m=ueMasks[ueStaged(t)];let l=vec3u(cell-4*t);let b=l.x+4u*(l.y+4u*l.z);
 return ((select(m.x,m.y,b>=32u)>>(b&31u))&1u)!=0u;
}
// The cells of a face neighbour's mask in its layer against the job's tile.
fn ueFacing(offset:vec3i)->vec2u{
 if(offset.x>0){return vec2u(0x11111111u);}if(offset.x<0){return vec2u(0x88888888u);}
 if(offset.y>0){return vec2u(0x000f000fu);}if(offset.y<0){return vec2u(0xf000f000u);}
 if(offset.z>0){return vec2u(0x0000ffffu,0u);}return vec2u(0u,0xffff0000u);
}
// A face direction: 2 axis + side, the request order.
fn ueDirection(offset:vec3i)->u32{
 if(offset.x!=0){return select(0u,1u,offset.x>0);}if(offset.y!=0){return select(2u,3u,offset.y>0);}return select(4u,5u,offset.z>0);
}
// The tile faces a cell lies against, by direction.
fn ueEdge(local:vec3u)->u32{
 var edge=0u;
 for(var axis=0u;axis<3u;axis++){if(local[axis]==0u){edge|=1u<<(2u*axis);}if(local[axis]==3u){edge|=2u<<(2u*axis);}}
 return edge;
}
// Whether a unit face anchored at a cell of the job's h tile can differ from
// its old slot (ueFaceLive, for the three components at once). A unit face's
// requests are anchored at the six cells around its anchor (a request past a
// domain wall reads nothing; one against the low wall of its own axis is
// anchored at the cell itself). Where the request's tile is unit, the one
// slot it reads is that cell's: the job tile's mask, spread one cell along
// each axis inside the tile, and the facing layer of each unit face
// neighbour moved onto the layer against it. Where it is a 4h face
// neighbour, every cell ueFaceLive tests lies in that tile, in the tile
// below it on another axis, or is the face's own anchor: when either tile's
// mask is set (ueStage notes it) the whole layer against the neighbour is
// live. Exact while every face neighbour is unit, a superset beside a 4h one
// (a face it admits beyond ueFaceLive's returns its old slot: no store).
fn ueCellLive(local:vec3u,edge:u32,flags:u32)->bool{
 if((edge&(flags>>1u)&(flags>>7u))!=0u){return true;}
 let m=ueMasks[13];
 var d=m|((m&vec2u(0x77777777u))<<vec2u(1u))|((m&vec2u(0xeeeeeeeeu))>>vec2u(1u))|((m&vec2u(0x0fff0fffu))<<vec2u(4u))|((m&vec2u(0xfff0fff0u))>>vec2u(4u))
  |vec2u((m.x<<16u)|(m.x>>16u)|(m.y<<16u),(m.y<<16u)|(m.x>>16u)|(m.y>>16u));
 if(ueStagedWidth(12)==1u){d|=(ueMasks[12]&vec2u(0x88888888u))>>vec2u(3u);}
 if(ueStagedWidth(14)==1u){d|=(ueMasks[14]&vec2u(0x11111111u))<<vec2u(3u);}
 if(ueStagedWidth(10)==1u){d|=(ueMasks[10]&vec2u(0xf000f000u))>>vec2u(12u);}
 if(ueStagedWidth(16)==1u){d|=(ueMasks[16]&vec2u(0x000f000fu))<<vec2u(12u);}
 if(ueStagedWidth(4)==1u){d.x|=ueMasks[4].y>>16u;}
 if(ueStagedWidth(22)==1u){d.y|=(ueMasks[22].x&0xffffu)<<16u;}
 let b=local.x+4u*(local.y+4u*local.z);
 return ((select(d.x,d.y,b>=32u)>>(b&31u))&1u)!=0u;
}
// Stages the neighbourhood of lane 0's owner's tile (every lane of a tile
// job holds an owner of that tile, or none). Workgroup-uniform result.
// fine: an h tile job, live when some ueCellLive of it can hold: its own
// mask is set, or the facing layer of a unit face neighbour, or a 4h face
// neighbour's mask, or the mask of the tile below one on another axis (an
// edge neighbour). A corner tile's mask reaches none of its faces.
fn ueStage(owner:UMOwner,lane:u32,seed:bool,fine:bool)->bool{
 if(lane==0u){
  var index=UM_TILES;
  if(owner.width!=0u&&(umTileSupport(owner.tile)&2u)!=0u&&(seed||ueOpen(owner.tile))){index=owner.tile;}
  ueJob=vec4u(umTileCoord(owner.tile),index);
  if(!seed&&(ueMixedJobs||ueSweepKind==1u)&&index<UM_TILES){ueJobRecipe=umTileStencil(index);}
  atomicStore(&ueLive,0u);atomicStore(&ueNew[0],0u);atomicStore(&ueNew[1],0u);
 }
 let job=workgroupUniformLoad(&ueJob);
 if(ueMixedJobs){uePick=vec4u(job.xyz,0u);ueRecipe=ueJobRecipe;}
 if(ueSweepKind==1u){ueRecipe=ueJobRecipe;}
 if(job.w>=UM_TILES){return false;}
 if(seed){return true;}
 if(lane<27u){
  let t=vec3i(job.xyz)+vec3i(umCorner(lane,3u))-vec3i(1);var width=0u;var mask=vec2u(0u);var word=0u;
  if(all(t>=vec3i(0))&&all(t<vec3i(UM_T))){let tile=umTileAt(vec3u(t));word=umTopology[tile];width=select(4u,1u,(word&0x80000000u)!=0u);mask=ueMaskIn(tile);}
  ueWords[lane]=word;ueMasks[lane]=mask;
  let offset=vec3i(umCorner(lane,3u))-vec3i(1);let far=abs(offset.x)+abs(offset.y)+abs(offset.z);
  let filled=any(mask!=vec2u(0u));var live=0u;
  if(!fine||far==0){live=select(0u,1u,filled);}
  else if(far==1){
   let k=ueDirection(offset);
   if(width==4u){live=(2u<<k)|select(0u,128u<<k,filled);}
   else{live=select(0u,1u,any((mask&ueFacing(offset))!=vec2u(0u)));}
  }else if(far==2&&filled){
   for(var axis=0u;axis<3u;axis++){if(offset[axis]<0){var d=offset;d[axis]=0;live|=128u<<ueDirection(d);}}
  }
  if(live!=0u){atomicOr(&ueLive,live);}
 }
 let live=workgroupUniformLoad(&ueLive);return (live&1u)!=0u||((live>>1u)&(live>>7u)&63u)!=0u;
}
// Whether umExtended(face) can differ from its old slot: some slot it may
// read is finite (otherwise it returns old).
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
fn ueFaceLive(anchor:vec3i,axis:u32,width:u32)->bool{
 var live=ueCellFinite(clamp(anchor,vec3i(0),vec3i(UM_D)-vec3i(1)));
 var center=vec3f(anchor)+vec3f(0.5*f32(width));center[axis]=f32(anchor[axis]+1);
 for(var n=0u;n<6u;n++){
  var point=center;point[n/2u]+=select(-f32(width),f32(width),(n&1u)!=0u);
  if(any(point<vec3f(0))||any(point>vec3f(UM_D))){continue;}
  live=live||ueRequestLive(point,axis,width);
 }
 return live;
}
// One request of ueFaceLive (P inside the lattice): whether a slot it may
// read is finite.
fn ueRequestLive(point:vec3f,axis:u32,width:u32)->bool{
 var offset=vec3f(0.5*f32(width));offset[axis]=1.0;
 return ueRequestAnchorLive(vec3i(round(point-offset)),axis,width);
}
fn ueRequestAnchorLive(anchor:vec3i,axis:u32,width:u32)->bool{
 let q=clamp(anchor,vec3i(0),vec3i(UM_D)-vec3i(1));let t=q/4;
 if(width!=1u){return any(ueMasks[ueStaged(t)]!=vec2u(0u));}
 if(ueStagedWidth(ueStaged(t))==1u){return ueCellFinite(q);}
 let o=4*t;var top=q;top[axis]=o[axis]+3;var corner=o;corner[axis]=o[axis]+3;
 var live=ueCellFinite(q)||ueCellFinite(top)||ueCellFinite(corner);
 let plane=anchor[axis]+1;
 if(plane%4!=0||plane==0){
  var below=q;below[axis]=max(o[axis]-1,0);var belowCorner=o;belowCorner[axis]=below[axis];
  live=live||ueCellFinite(below)||ueCellFinite(belowCorner);
 }
 return live;
}
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
 if(any(m!=vec2u(0u))){ueMaskOut(ueJob.w,m,false,true);}
}
// Seam requests use compiled lattice recipes in the requesting lane. Their
// geometry is now cheap enough to avoid the shared request/answer queue,
// its per-face atomic reservation and two barriers (3844 bytes at pack 2).
// A positive patch by its anchor (no neighbour lookup: no caller reads it).
fn uePatch(anchor:vec3i,axis:u32,width:u32)->UMFace{return UMFace(UMOwner(),anchor,width,1u,axis,1);}
// umExtended's request n = 2 step + side of a patch centred at center.
fn ueRequestPoint(center:vec3f,width:u32,n:u32)->vec3f{
 var delta=vec3f(0);delta[n/2u]=f32(width);
 if((n&1u)==0u){return center-delta;}
 return center+delta;
}
fn ueOutside(point:vec3f)->bool{return any(point<vec3f(0))||any(point>vec3f(UM_D));}
${uniformCompiledExtensionNeighborWGSL}
// Immutable request geometry comes from the layout recipe; liveness still
// comes from this sweep's staged masks. Classify and consume in one loop,
// retaining the old per-axis combination and candidate reduction order.
fn ueExtend(face:UMFace,old:vec2f,need:u32)->vec2f{
 let center=umFaceCenter(face);let origin=4*ueJobTile();
 var low:array<UMNeighbor,3>;var high:array<UMNeighbor,3>;
 for(var n=0u;n<6u;n++){
  var anchor=face.anchor;anchor[n/2u]+=select(-i32(face.width),i32(face.width),(n&1u)!=0u);
  var answer=UMNeighbor(0,UM_INF,1);
  if(ueRequestInside(anchor,face.axis)){
   let search=(need&(1u<<n))!=0u&&!ueTopologyDirectAt(ueRecipe,anchor-origin,face.axis,face.width);
   if(search){
    if(ueRequestAnchorLive(anchor,face.axis,face.width)){
     var point=vec3f(anchor)+vec3f(0.5*f32(face.width));point[face.axis]=f32(anchor[face.axis]+1);
     answer=ueCompiledNeighbor(point,center,face.axis,n/2u,face.width);
    }
   }else{
    let state=umSlotState(anchor,face.axis,face.width);
    answer=UMNeighbor(state.x,state.y,f32(face.width)*h[n/2u]);
   }
  }
  if((n&1u)==0u){low[n/2u]=answer;}else{high[n/2u]=answer;}
 }
 return ueCombine(old,f32(face.width),low,high);
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
//
// Launches. A launch of a few jobs costs the latency of its longest per-lane
// chain whatever its job count, so each seam tier with a launch of its own
// added that latency to the pass however few tiles it held. Every pass here
// (seed, a sweep, publish) is a Jacobi pass over disjoint outputs, so the
// seam jobs of a pass share one launch and their chains overlap: a job per
// seam h tile, then per seam 4h tile. A sweep's launch first holds the
// regular 4h owners, 64 a job, whose chain is as long: the first h tile then
// adds no launch to a sweep. The regular h kernels stay apart, and the list
// kernel while no h tile can exist: a kernel holding every job kind runs its
// bulk slower (the regular h sweep 1.8x, the list sweep 1.3x).
// ueLaunch: the job counts (seam h tiles, seam 4h tiles, regular 4h packs), loaded
// once per workgroup; a job's kind is workgroup-uniform (its index against
// those counts).
var<workgroup> ueLaunch:vec3u;
fn ueLaunchCounts()->vec3u{
 let header=7u*UM_TILES+16u;
 return vec3u(umSupport[header],umSupport[header+1u],(umSupport[8u*UM_TILES+20u]+63u)/64u);
}
// Seam h tile job: one lane per owner.
fn ueSeamFineJob(job:u32,lane:u32)->UMOwner{
 let tile=umSupport[7u*UM_TILES+20u+job];return UMOwner(tile,lane,1u,(umTopology[tile]&0x3fffffffu)+lane);
}
// Seam 4h tile job (listed after the seam h tiles): the tile's one owner, or
// none off the extension support.
fn ueSeamCoarseJob(job:u32)->UMOwner{
 let header=7u*UM_TILES+16u;let fine=umSupport[header];let tile=umSupport[header+4u+fine+job];
 if((umTileSupport(tile)&2u)==0u){return UMOwner();}
 return UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
}
// The seed of one owner; returns the anchors of its finite slots (xy) and
// whether it left a patch that is not a source (z: the tile's open bit).
fn ueSeedOwner(owner:UMOwner)->vec3u{
 let origin=umOrigin(owner);var bits=vec2u(0u);var open=0u;
 // A unit owner's faces are the unit patches whatever its neighbours.
 if(umRegularFine||owner.width==1u){
  for(var axis=0u;axis<3u;axis++){
   if(origin[axis]==0u){let face=umUnitExtensionFace(owner,axis,-1);if(umSeedWall(face,owner)){bits|=ueBit(face.anchor);}}
   let face=umUnitExtensionFace(owner,axis,1);
   let value=umSeedState(face,owner);ueSeedStore(ueOwnSlot(owner,axis),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}else{open=1u;}
  }
  return vec3u(bits,open);
 }
 // Every patch owns its component slot: no anchor packing across axes.
 for(var axis=0u;axis<3u;axis++){
  if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);if(umSeedWall(face,owner)){bits|=ueBit(face.anchor);}}
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let face=umFace(owner,axis,1,part);
   let value=umSeedState(face,owner);ueSeedStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits|=ueBit(face.anchor);}else{open=1u;}
  }
 }
 return vec3u(bits,open);
}
@compute @workgroup_size(64) fn seedList(@builtin(global_invocation_id) gid:vec3u){
 let owner=ueOwner(gid);if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u){return;}
 let seeded=ueSeedOwner(owner);ueMaskOut(owner.tile,seeded.xy,true,seeded.z!=0u);
}
// A regular 4h owner's faces are single width-4 patches: every slot they
// read lies in its tile or a face neighbour (ueFaceLive, width 4).
fn ueSweepPack(owner:UMOwner){
 if(owner.width==0u||(umTileSupport(owner.tile)&2u)==0u||!ueOpen(owner.tile)){return;}
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
 if(any(bits!=vec2u(0u))){ueMaskOut(owner.tile,bits,false,true);}
}
@compute @workgroup_size(64) fn sweepList(@builtin(global_invocation_id) gid:vec3u){
 ueSweepPack(ueOwner(gid));
}
// An h tile job: every lane holds an owner of the one tile (or none).
fn ueSeedFine(owner:UMOwner,lane:u32){
 if(!ueStage(owner,lane,true,true)){return;}
 let bits=ueSeedOwner(owner);atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);if(bits.z!=0u){atomicOr(&ueLive,1u);}
 workgroupBarrier();
 if(lane==0u){ueMaskOut(ueJob.w,vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1])),true,atomicLoad(&ueLive)!=0u);}
}
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 ueSeedFine(ueOwner(gid),lane);
}
// A regular h tile has no 4h tile in its stencil: each lane evaluates the
// faces of its owner (a unit owner's positive patch is anchored at its
// origin) when its cell is live, and every neighbour is a unit slot.
@compute @workgroup_size(64) fn sweep(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let owner=ueOwner(gid);
 if(!ueStage(owner,lane,false,true)){return;}
 var bits=vec2u(0u);let local=umCorner(owner.lane,4u);let live=ueCellLive(local,ueEdge(local),atomicLoad(&ueLive));
 for(var axis=0u;axis<3u;axis++){
  if(!live){continue;}
  let face=umUnitExtensionFace(owner,axis,1);let value=umExtended(face,owner);
  ueSweepStore(ueOwnSlot(owner,axis),value);bits|=ueNote(value,face.anchor);
 }
 ueGather(bits);
 workgroupBarrier();
 if(lane==0u){ueFlush();}
}
// A seam h tile: each lane evaluates its three unit patches. No values
// remain live across a queue barrier; each patch is planned and consumed.
fn ueSweepSeamFine(owner:UMOwner,lane:u32){
 if(!ueStage(owner,lane,false,true)){return;}
 let origin=vec3i(umOrigin(owner));var bits=vec2u(0u);
 let flags=atomicLoad(&ueLive);let local=umCorner(owner.lane,4u);let edge=ueEdge(local);let live=ueCellLive(local,edge,flags);
 for(var axis=0u;axis<3u;axis++){
  if(!live){continue;}
  let face=uePatch(origin,axis,1u);let old=stateIn[ueOwnSlot(owner,axis)];
  if(old.y==0.0||umClosedWall(face)){continue;}
  let value=ueExtend(face,old,edge&(flags>>1u)&~(1u<<(2u*axis)));
  ueSweepStore(ueOwnSlot(owner,axis),value);bits|=ueNote(value,origin);
 }
 ueGather(bits);
 workgroupBarrier();
 if(lane==0u){ueFlush();}
}
// Coarse seam owners: one job per tile, one lane per patch and component.
// A 4h owner beside h tiles has sixteen patches per face; evaluating them
// serially in one lane made the coarse seam tier a latency chain.
// 4h seam tiles hold one owner with at most sixteen patches per positive
// face. One lane per (axis, patch); each writes its own slot directly.
fn umCoarseCell(local:vec3u)->u32{return local.x+4u*(local.y+4u*local.z);}
fn ueSeedCoarse(owner:UMOwner,lane:u32){
 if(!ueStage(owner,lane,true,false)){return;}
 var bits=vec2u(0u);
 if(lane<48u){
  let axis=lane/16u;let face=umFace(owner,axis,1,lane%16u);
  if(face.width!=0u){let value=umSeedState(face,owner);ueSeedStore(umSlot(face.anchor,axis,face.width),value);if(value.y<0.5*UM_INF){bits=ueBit(face.anchor);}else{atomicOr(&ueLive,1u);}}
 }else if(lane<51u){
  let axis=lane-48u;if(umOrigin(owner)[axis]==0u){let face=umFace(owner,axis,-1,0u);if(umSeedWall(face,owner)){bits=ueBit(face.anchor);}}
 }
 atomicOr(&ueNew[0],bits.x);atomicOr(&ueNew[1],bits.y);
 workgroupBarrier();
 if(lane==0u){ueMaskOut(ueJob.w,vec2u(atomicLoad(&ueNew[0]),atomicLoad(&ueNew[1])),true,atomicLoad(&ueLive)!=0u);}
}
// The positive patch umFace would return for a 4h seam owner, from the
// staged +axis neighbour width (umFace: min of the owner's and the
// neighbour's, the owner's outside the lattice): item = 16 axis + part.
// Width 0: no such part.
fn ueCoarsePatch(origin:vec3i,item:u32)->UMFace{
 let axis=item/16u;let part=item%16u;var probe=origin;probe[axis]+=4;
 var width=4u;if(probe[axis]<i32(UM_D[axis])){width=min(4u,ueStagedWidth(ueStaged(probe/4)));}
 let side=4u/width;if(part>=side*side){return UMFace();}
 var anchor=probe;anchor[(axis+1u)%3u]+=i32((part%side)*width);anchor[(axis+2u)%3u]+=i32((part/side)*width);anchor[axis]-=1;
 return uePatch(anchor,axis,width);
}
// A pack of seam 4h tiles (list entries first .. first + UE_PACK, below
// count). A seam 4h tile fills 3 to 48 of a job's 64 lanes and every job
// pays its staging, barriers and flush whatever it holds: a pack shares
// them. Slot s holds one tile: its header (uePackJobs, as ueJob), its staged
// 3x3x3 at 27 s, its live bit (some staged mask nonzero) and its new anchor
// bits. The pack's patches are numbered in (slot, component, part) order
// over the live slots, a lane each: lane l evaluates patch l (and l + 64
// where the pack holds more). Each slot's mask is flushed by one lane.
// A patch's item is 48 slot + 16 axis + part (ueCoarsePatch's item of its
// slot).
const UE_PACK_ROUNDS=${Math.ceil(48*UE_PACK_TILES/64)}u;
const UE_PACK_NONE=0xffffffffu;
var<workgroup> uePackJobs:array<vec4u,UE_PACK>;
var<workgroup> uePackRecipes:array<vec2u,UE_PACK>;
var<workgroup> uePackLive:array<atomic<u32>,UE_PACK>;
var<workgroup> uePackNew:array<atomic<u32>,${2*UE_PACK_TILES}>;
// The patches of a slot's component: sixteen under an h tile (the staged
// +axis neighbour, ueCoarsePatch's width), one otherwise; none in a slot
// without a live job.
fn uePackParts(slot:u32,axis:u32)->u32{
 if(uePackJobs[slot].w>=UM_TILES||atomicLoad(&uePackLive[slot])==0u){return 0u;}
 return select(1u,16u,ueStagedWidth(27u*slot+13u+select(select(9u,3u,axis==1u),1u,axis==0u))==1u);
}
fn uePackItem(at:u32)->u32{
 var base=0u;
 for(var k=0u;k<3u*UE_PACK;k++){
  let parts=uePackParts(k/3u,k%3u);
  if(at<base+parts){return 48u*(k/3u)+16u*(k%3u)+(at-base);}
  base+=parts;
 }
 return UE_PACK_NONE;
}
fn ueSweepCoarsePack(first:u32,count:u32,lane:u32){
 if(lane<UE_PACK){
  var job=vec4u(0u,0u,0u,UM_TILES);
  if(first+lane<count){
   let owner=ueSeamCoarseJob(first+lane);
   if(owner.width!=0u&&ueOpen(owner.tile)){job=vec4u(umTileCoord(owner.tile),owner.tile);}
  }
  if(job.w<UM_TILES){uePackRecipes[lane]=umTileStencil(job.w);}
  uePackJobs[lane]=job;atomicStore(&uePackLive[lane],0u);atomicStore(&uePackNew[2u*lane],0u);atomicStore(&uePackNew[2u*lane+1u],0u);
 }
 workgroupBarrier();
 for(var i=lane;i<27u*UE_PACK;i+=64u){
  let slot=i/27u;let job=uePackJobs[slot];var mask=vec2u(0u);var word=0u;
  if(job.w<UM_TILES){
   let t=vec3i(job.xyz)+vec3i(umCorner(i%27u,3u))-vec3i(1);
   if(all(t>=vec3i(0))&&all(t<vec3i(UM_T))){let tile=umTileAt(vec3u(t));word=umTopology[tile];mask=ueMaskIn(tile);}
   if(any(mask!=vec2u(0u))){atomicOr(&uePackLive[slot],1u);}
  }
  ueWords[i]=word;ueMasks[i]=mask;
 }
 workgroupBarrier();
 for(var round=0u;round<UE_PACK_ROUNDS;round++){
  let item=uePackItem(64u*round+lane);if(item==UE_PACK_NONE){continue;}
  let slot=item/48u;let job=uePackJobs[slot];uePick=vec4u(job.xyz,27u*slot);ueRecipe=uePackRecipes[slot];
  let face=ueCoarsePatch(vec3i(job.xyz)*4,item%48u);
  if(face.width==0u||!ueFaceLive(face.anchor,face.axis,face.width)){continue;}
  let old=stateIn[umSlot(face.anchor,face.axis,face.width)];
  if(old.y==0.0||umClosedWall(face)){continue;}
  let value=ueExtend(face,old,63u);
  ueSweepStore(umSlot(face.anchor,face.axis,face.width),value);let bits=ueNote(value,face.anchor);
  if(any(bits!=vec2u(0u))){atomicOr(&uePackNew[2u*slot],bits.x);atomicOr(&uePackNew[2u*slot+1u],bits.y);}
 }
 workgroupBarrier();
 if(lane<UE_PACK){
  let job=uePackJobs[lane];
  if(job.w<UM_TILES&&atomicLoad(&uePackLive[lane])!=0u){
   let m=ueMasks[27u*lane+13u]|vec2u(atomicLoad(&uePackNew[2u*lane]),atomicLoad(&uePackNew[2u*lane+1u]));
   if(any(m!=vec2u(0u))){ueMaskOut(job.w,m,false,true);}
  }
 }
}
// Fixed direct grid-stride launches over the GPU job counts: any grid is
// complete. The barrier after a job keeps the next job's staging apart.
@compute @workgroup_size(64) fn seedSeams(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ueLaunch=ueLaunchCounts();}
 let counts=workgroupUniformLoad(&ueLaunch);
 for(var job=group.x;job<counts.x+counts.y;job+=groups.x){
  if(job<counts.x){ueSeedFine(ueSeamFineJob(job,lane),lane);}
  else{ueSeedCoarse(ueSeamCoarseJob(job-counts.x),lane);}
  workgroupBarrier();
 }
}
// 1: fine seams; 2: all coarse jobs. 0 retains the merged reference used by
// the differential sweep test; production always specializes to 1 or 2.
override ueSweepKind:u32=0u;
@compute @workgroup_size(64) fn sweepSeams(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ueLaunch=ueLaunchCounts();}
 let counts=workgroupUniformLoad(&ueLaunch);let fine=counts.z+counts.x;
 if(ueSweepKind==1u){
  ueStagedJob=true;
  for(var job=group.x;job<counts.x;job+=groups.x){ueSweepSeamFine(ueSeamFineJob(job,lane),lane);workgroupBarrier();}
  return;
 }
 if(ueSweepKind==2u){
  for(var job=group.x;job<counts.z+(counts.y+UE_PACK-1u)/UE_PACK;job+=groups.x){
   ueStagedJob=job>=counts.z;
   if(job<counts.z){ueSweepPack(umRegularCoarseOwner(64u*job+lane));}
   else{ueSweepCoarsePack(UE_PACK*(job-counts.z),counts.y,lane);}
   workgroupBarrier();
  }
  return;
 }
 for(var job=group.x;job<fine+(counts.y+UE_PACK-1u)/UE_PACK;job+=groups.x){
  ueStagedJob=job>=counts.z;
  if(job<counts.z){ueSweepPack(umRegularCoarseOwner(64u*job+lane));}
  else if(job<fine){ueSweepSeamFine(ueSeamFineJob(job-counts.z,lane),lane);}
  else{ueSweepCoarsePack(UE_PACK*(job-fine),counts.y,lane);}
  workgroupBarrier();
 }
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
  // Unit slots take the tile's word, and the word of the tile above a 4h
  // tile with unit patches (that tile is h): loaded once for the anchors.
  let word=umTopology[tile];let coarse=(word&0x80000000u)==0u;var patchWidth=1u;var upper=0u;
  if(coarse){patchWidth=umFace(umOwnerAt(origin),component,1,0u).width;if(patchWidth==1u){var above=gid;above[component]+=1u;upper=umTopology[umTileAt(above)];}}`}
  // Restrict real MAC patches, with the native vertical footprint fallback.
  for(var fallback=0u;fallback<select(1u,2u,component==1u&&!coarse);fallback++){
   if(count>0.0){break;}
   // Preserve z/y/x order among the 16 face-plane anchors.
   for(var k=0u;k<select(select(16u,64u,fallback!=0u),1u,patchWidth==4u);k++){
    var local=umCorner(k,4u);
    if(fallback==0u){local=vec3u(k%4u,k/4u,3u);if(component==0u){local=vec3u(3u,k%4u,k/4u);}else if(component==1u){local=vec3u(k%4u,3u,k/4u);}}
    let p=origin+vec3i(local);
    ${this.regularBulk?"let o=umOwnerAt(p);let face=umPositiveFaceAtAnchor(o,component,p);if(face.width==0u){continue;}":"let face=UMFace(UMOwner(),p,patchWidth,select(1u,16u,coarse&&patchWidth==1u),component,1);"}
    ${this.regularBulk?"if(!umSource(face,o)){continue;}":"let slot=stateIn[select(UE_COARSE+component*UE_TILES+tile,ueUnitSlotIn(word,upper,local,component),patchWidth==1u)];if(slot.y>=0.5*UM_INF){continue;}"}
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
// An h publish job is one tile; every far tap of its faces is one of the 27
// coarse cells around it, staged once in workgroup memory.
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
fn uePublishFine(owner:UMOwner,lane:u32){
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
  let face=umUnitExtensionFace(owner,axis,1);${this.regularBulk?"let slot=select(vec2f(0,UM_INF),vec2f(umPhysical(face),0),umSource(face,owner));":"let slot=select(vec2f(0.0,UM_INF),stateIn[ueOwnSlot(owner,axis)],(umTileSupport(owner.tile)&2u)!=0u);"}
  if(umClosedWall(face)){value[axis]=umPhysical(face);}
  else if(slot.y<0.5*UM_INF){value[axis]=slot.x;}
  else if((mixed&(1u<<axis))==0u){value[axis]=umFarValueUniform(face,farState[13u][axis],farLo[39u+axis].w>=0.0);}
  else{value[axis]=umFarValueStaged(face);}
 }
 value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);
}
@compute @workgroup_size(64) fn publishFine(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 uePublishFine(umOwner(gid),lane);
}
var<workgroup> publishCoarseState:array<vec2f,192>;
fn uePublishCoarse(owner:UMOwner,lane:u32){
 for(var i=lane;i<192u;i+=64u){publishCoarseState[i]=vec2f(0);}
 workgroupBarrier();
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
// h seam owners publish through uePublishFine (the same values: umPublished
// per unit face, with the staged far value); 4h seam tiles through uePublishCoarse.
@compute @workgroup_size(64) fn publishSeams(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){ueLaunch=ueLaunchCounts();}
 let counts=workgroupUniformLoad(&ueLaunch);
 for(var job=group.x;job<counts.x+counts.y;job+=groups.x){
  if(job<counts.x){uePublishFine(ueSeamFineJob(job,lane),lane);}
  else{uePublishCoarse(ueSeamCoarseJob(job-counts.x),lane);}
  workgroupBarrier();
 }
}
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
`,["seed","sweep","seedList","sweepList","publishList","publishFine"])});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  this.restrictPipeline=await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"restrictBand",constants:{umDispatchX:this.ownership.dispatchX}}});
  await Promise.all((this.regularBulk?["publish"]:["seed","sweep","publish"]).map(async entryPoint=>{
   const compile=(width:number,counted:number,name:string,list=false)=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:name,constants:{umDispatchX:this.ownership.dispatchX,umCellWidth:width,umRegularTiles:1,umRegularFine:+(width===1),ueRegularCoarseList:+list,umCountedJobs:counted,ueStagedTiles:+(entryPoint==="sweep"&&!list)}}});
   const C=UNIFORM_MIXED_COUNTED;
   this.regularPipelines.set(entryPoint,await compile(1,C.owners,entryPoint==="publish"?"publishFine":entryPoint));
   this.listPipelines.set(entryPoint,await compile(4,C.regularCoarse,`${entryPoint}List`,true));
   // The seam launch reads no launch constant; a sweep's tile jobs stage their tiles.
   this.seamPipelines.set(entryPoint,await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:`${entryPoint}Seams`,constants:entryPoint==="sweep"?{ueStagedTiles:1,ueMixedJobs:1,ueSweepKind:2}:{}}}));
   if(entryPoint==="sweep")this.fineSeamSweep=await uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint:"sweepSeams",constants:{ueStagedTiles:1,ueMixedJobs:0,ueSweepKind:1,ueStagePack:1}}});
  }));
 }
 encode(encoder:GPUCommandEncoder,groups:readonly [UniformDetailGroup,UniformDetailGroup],sweeps=2):void{
  if(this.seamPipelines.size!==(this.regularBulk?1:3)||!this.restrictPipeline)throw new Error("Mixed extension is not initialized");
  // One pass: seed, sweeps, restriction, the hierarchy continuation, publish.
  let open:GPUComputePassEncoder|undefined;
  const run=(entry:string,group:UniformDetailGroup)=>{
   if(!open){open=encoder.beginComputePass({label:"Uniform mixed extension"});open.setBindGroup(0,this.ownership.bindGroup);}
   const pass=open;pass.setBindGroup(1,group.group);
   if(entry==="restrictBand"){pass.setPipeline(uniformDetailPick(this.restrictPipeline!));pass.dispatchWorkgroups(...this.ownership.capacity.tileDimensions.map(n=>Math.ceil(n/4)) as [number,number,number]);}
   else {
    // Counted launches, each empty (not skipped) when its GPU count is zero
    // unless no h tile can exist.
    const o=this.ownership,tiles=o.capacity.tiles,packs=Math.ceil(tiles/64);
    o.dispatchTierCounted(pass,this.regularPipelines.get(entry)!,0);
    // Fine seams reserve only their own staged neighborhood. Regular 4h
    // owners share the coarse seam launch to avoid a third sweep dispatch.
    // A coarse-only capacity continues to use the existing list kernel.
    if(entry==="sweep"&&!o.coarseOnly){
     o.dispatchCounted(pass,this.fineSeamSweep!,o.capacity.fineTiles,"seamFine");
     o.dispatchCounted(pass,this.seamPipelines.get(entry)!,Math.ceil(tiles/UE_PACK_TILES)+packs,"seams");return;
    }
    // A regular 4h owner (a 3x3x3 stencil of 4h tiles) seeds from and
    // publishes at tile origins and +face anchors only: the base blocks.
    // Its sweep touches no field.
    const based=entry!=="sweep";
    if(based)pass.setBindGroup(1,group.base);
    o.dispatchRegularCoarseCounted(pass,this.listPipelines.get(entry)!);
    if(based)pass.setBindGroup(1,group.group);
    if(based)o.dispatchCounted(pass,this.seamPipelines.get(entry)!,tiles,"hanging");
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
