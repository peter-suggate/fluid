import { uniformCompiledVertexResolveWGSL } from "./uniform-compiled-topology";
import { uniformDetailBindLayout, uniformDetailExtent, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, uniformDetailGroup, type UniformDetailGroup } from "./uniform-detail-fields";
import {UNIFORM_DETAIL_4H_LOAD,UNIFORM_DETAIL_CANONICAL_LOAD} from "../../core/uniform-detail-abi";
import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {UNIFORM_MIXED_COUNTED,uniformMixedCountedEntriesWGSL,uniformMixedPageCount,uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformSurfaceFillWGSL} from "./uniform-surface-volume.wgsl";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,uniformMixedVertexBuriedWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

/** Native four-cell band and a global normal shift found by two secant
 * Newton rounds, reduced with physical owner volumes. Fields and scratch are borrowed. With static
 * solids a unit owner's capacity is its native open fraction: closed cells
 * neither seed nor carry the band, and slopes use only open-sided axes.
 *
 * Only the seed visits every owner. Each dilation moves the band at most one
 * owner, so at most one tile: a step visits the tiles within one tile of a
 * tile already carrying band, and every other owner keeps the zero the seed
 * wrote into both parities. Apply visits one tile around the final band
 * tiles (a vertex's incident cells lie in adjacent tiles); metric and measure
 * add every tile with a wider owner in its stencil. An owner of a skipped
 * tile has all-zero corner scales, so both its samples are one
 * constant: the seed writes those workgroup partials once, in the measure's
 * own reduction order, and the refinements rebuild only workgroups that can
 * differ. Every value equals the dense evaluation's.
 *
 * Residency (uniformMixedResidencyWord): owner passes stride the h tiles and
 * the 4h tiles of resident pages (UNIFORM_MIXED_COUNTED.residentAll), tile
 * passes the tiles of resident pages. An absent page's 4h owners are far air
 * (V=0, corner phi at least 16h, audited by the next census): unseeded, no
 * band, no liquid or volume, so their dense rows are zeros the sums lose
 * nothing without (only the grouping of the partial sums changes). clearBand
 * clears the band, visit and measure flags densely; a measured tile with a
 * neighbour in an absent page would read an owner the seed skipped, so it
 * sets the sticky closure bit (a fatal at the next census).
 *
 * bind(phi,volume,phi,scratch) corrects phi in place, touching only band
 * vertices. In scalar-target mode volume is the measured geometry output:
 * read once at initialization, then a 32-byte budget owns the target. The
 * correction band spans four h cells and the shift is bounded by 4h, including
 * large timesteps; the required shift remains visible in diagnostics. */
export class UniformMixedSurfaceVolume {
 /** NB budget: target, initialized, cumulative outflow, last normal shift,
  * volume before correction, initial volume, last measured volume, target. */
 readonly budget?:GPUBuffer;
 private readonly budgetParams?:GPUBuffer;
 private budgetCaptured=false;
 get allocatedBytes():number{return this.scalarTarget?48:0;}
 /** Bytes of bind()'s scratch at that many h tiles. The sections the lattice
  * fixes come first: the corner scales (a word a vertex), the partial rows
  * and their chunk sums, the Newton state and three flags a tile. Then the
  * band's two parities, by owner index. */
 scratchBytesAt(fineTiles:number):number{return 4*(this.vertices+4*(this.groups+this.chunks)+8+3*this.tiles+2*(63*fineTiles+this.tiles));}
 get scratchBytes():number{return this.scratchBytesAt(this.ownership.capacity.fineTiles);}
 private readonly vertices:number;
 private readonly groups:number;
 private readonly chunks:number;
 private readonly tiles:number;
 /** Residency pages: the bound of the page launches. */
 private readonly pages:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly applyResources:GPUBindGroupLayout;
 private readonly applyGroups=new WeakMap<UniformDetailGroup,UniformDetailGroup>();
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** resolved: the caller runs UniformMixedPhiResolve on phi before encode.
  * Every owner corner is then a stored or resolved texel; resolveScale
  * completes the corner scales the same way after metric. */
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly resolved=false,private readonly scalarTarget=false){
  const lattice=ownership.capacity.lattice;
  this.vertices=lattice.dimensions.reduce((n,d)=>n*(d+1),1);
  this.tiles=ownership.capacity.tiles;this.pages=uniformMixedPageCount(lattice);
  // Partial rows: the residentAll launch's job bound (svJobs): every tile h,
  // plus a job per page. Rows sized to the tiles alone let an all-h layout's
  // page jobs write the chunk sums, and its extra chunk the Newton state.
  this.groups=this.tiles+this.pages;this.chunks=Math.ceil(this.groups/64);
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const scratch={binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}};
  if(scalarTarget){
   this.budget=device.createBuffer({label:"NB scalar volume budget",size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   this.budgetParams=device.createBuffer({label:"NB volume budget parameters",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  }
  this.resources=uniformDetailBindLayout(device,{entries:[texture(0),texture(1),scratch,...(scalarTarget?[texture(4),{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}},{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[])]});
  this.applyResources=uniformDetailBindLayout(device,{entries:[{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32float",viewDimension:"3d"}},scratch]});
 }
 bind(phi:GPUTexture,volume:GPUTexture,output:GPUTexture,scratch:GPUBufferBinding,velocity?:GPUTexture):UniformDetailGroup{
  const d=this.ownership.capacity.lattice.dimensions;
  for(const [i,t] of [phi,volume,output].entries())if(t.format!=="r32float"||uniformDetailExtent(t).some((n,a)=>n!==d[a]!+(i===1?0:1)))throw new Error("Mixed surface constraint requires native vertex and cell fields");
  if(output!==phi)throw new Error("Mixed surface constraint corrects phi in place");
  if((scratch.size??scratch.buffer.size-(scratch.offset??0))<this.scratchBytes)throw new Error("Mixed surface constraint needs sufficient scratch");
  if(this.scalarTarget&&!velocity)throw new Error("NB volume budget requires boundary velocity");
  const work={...scratch,size:this.scratchBytes};
  const group=uniformDetailGroup(this.device,{layout:this.resources,entries:[{binding:0,resource:phi},{binding:1,resource:volume},{binding:3,resource:work},...(this.scalarTarget?[{binding:4,resource:velocity!},{binding:5,resource:{buffer:this.budget!}},{binding:6,resource:{buffer:this.budgetParams!}}]:[])]});
  this.applyGroups.set(group,uniformDetailGroup(this.device,{layout:this.applyResources,entries:[{binding:2,resource:phi},{binding:3,resource:work}]}));
  return group;
 }
 async initialize():Promise<void>{
  const P=this.vertices,R=P+4*this.groups,S=R+4*this.chunks,T=this.tiles;
  const topology=uniformMixedTopologyWGSL(this.ownership.capacity,0);
  const shared=(sampling:string)=>/* wgsl */`
const UM_H=vec3f(${this.ownership.capacity.lattice.cellSize_m.join(",")});
${sampling}
fn umVertexIndex(p:vec3u)->u32{let d=UM_D+vec3u(1);return p.x+d.x*(p.y+d.y*p.z);}
// Band tiles, visited tiles, measured tiles.
const SV_BAND:u32=${S+8}u;const SV_VISIT:u32=${S+8+T}u;const SV_MEASURE:u32=${S+8+2*T}u;
// The band by owner index: two parities (svBandAt), the last section.
const SV_OWNERS:u32=${S+8+3*T}u;
fn svVisited(o:UMOwner)->bool{return scratch[SV_VISIT+o.tile]!=0.0;}
// Every incident owner of a unit-stencil owner's corners has unit width: the
// corners are stored, and a corner belongs to the lowest incident cell, so
// the owner holds its positive corner and its corners on negative walls.
fn svUnit(o:UMOwner)->bool{return umTileMaximumWidth(o.tile)==1u;}
// An owner's corners are width-aligned vertices of its tile closure: stored
// (uniform or unit stencil) or, when resolved, completed in place (mixed).
fn svDirect(o:UMOwner)->bool{return ${this.resolved}||svUnit(o);}
fn svAuthority(o:UMOwner,k:u32)->bool{
 if(o.width==4u){return (umCoarseCornerMask(o.tile)&(1u<<k))!=0u;}
 let corner=umCorner(k,2u);
 if(svUnit(o)){return all((corner!=vec3u(0))|(umOrigin(o)==vec3u(0)));}
 return umVertexAuthority(umOrigin(o)+corner*o.width).index==o.index;
}
// Tile of a unit-stencil owner's corner authority, clamp(p-1)'s owner.
fn svUnitAuthorityTile(p:vec3u)->u32{return umTileAt(vec3u(clamp(vec3i(p)-vec3i(1),vec3i(0),vec3i(UM_D)-vec3i(1)))/4u);}
fn svMeasured(t:u32)->bool{return scratch[SV_MEASURE+t]!=0.0;}`;
  const sourceScale=uniformMixedVertexSamplingSource("",false).replace(/\bum(Vertex\w*|SampleVertex|LoadVertex)\b/g,name=>name.replace("um","umScale"));
  // Owner entries stride every live owner (all), resolveScale every seam
  // tile (fused): GPU-counted fixed grids, sized by capacity alone.
  const module=uniformDetailModule(this.device,{label:"Uniform mixed surface volume",code:uniformMixedCountedEntriesWGSL(topology+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> scratch:array<f32>;
override parity:u32=0u;
${this.scalarTarget?`@group(1) @binding(4) var velocity:texture_3d<f32>;
@group(1) @binding(5) var<storage,read_write> budget:array<f32>;
@group(1) @binding(6) var<uniform> budgetParams:vec4f;
override captureBudget:bool=false;`:""}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
// A tile corner, and below a 4h owner's own texel (its tile origin): the
// base blocks (UNIFORM_DETAIL_4H_LOAD).
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
fn umLoadVertexW(p:vec3u,width:u32)->f32{if(width==4u){return umLoadCorner(p);}return umLoadVertex(p);}
${shared(uniformMixedVertexSamplingSource("",this.resolved,undefined,"umLoadCorner"))}
fn umScaleLoadVertex(p:vec3u)->f32{return scratch[umVertexIndex(p)];}
${sourceScale}
${uniformSurfaceFillWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}${this.solid?uniformMixedVertexBuriedWGSL:""}
// What an owner can hold: its open fraction (an h cell's, a 4h owner's tile's).
fn umCapacity(o:UMOwner)->f32{if(!umSolidEnabled()){return 1.0;}if(o.width==1u){return umCellOpen(vec3i(umOrigin(o)));}return umTileOpen(o.tile);}
// A cut 4h owner's filled volume in h cells: the tetrahedral fill of each h
// cell of its tile under the owner's trilinear field, times the cell's open
// fraction.
fn svOpenFill(v:array<f32,8>,origin:vec3i)->f32{
 var sum=0.0;
 for(var s=0u;s<64u;s++){
  let cell=vec3u(s&3u,(s>>2u)&3u,s>>4u);let open=umCellOpen(origin+vec3i(cell));
  if(open<=0.0){continue;}
  let base=vec3f(cell);var c:array<f32,8>;var negative=0u;
  for(var k=0u;k<8u;k++){
   let f=0.25*(base+vec3f(umCorner(k,2u)));
   let low=mix(mix(v[0],v[1],f.x),mix(v[2],v[3],f.x),f.y);let high=mix(mix(v[4],v[5],f.x),mix(v[6],v[7],f.x),f.y);
   c[k]=mix(low,high,f.z);negative+=select(0u,1u,c[k]<0.0);
  }
  if(negative==8u){sum+=open;}else if(negative!=0u){sum+=open*fill(c);}
 }
 return sum;
}
// Partial rows: one per job of the residentAll launch (h tiles, then resident pages).
fn svJobs()->u32{return umCounts.x+umResidentPageCount();}
// A band parity's base: the second follows the live owners of the first.
fn svBandAt(side:u32)->u32{return SV_OWNERS+side*(64u*umCounts.x+umCounts.y);}
fn svVolume(o:UMOwner)->f32{${this.scalarTarget?"return 0.0;":`let p=vec3i(umOrigin(o));if(o.width==4u){return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,p,0).x;}return textureLoad(volume,p,0).x;`}}
fn umShiftLimit()->f32{return min(UM_H.x,min(UM_H.y,UM_H.z))*${this.scalarTarget?"4.0":"f32(select(4u,1u,umCounts.x>0u))"};}
// A lane's row: filled mass at the shift and one secant step ahead, volume.
var<workgroup> sums:array<vec4f,64>;
var<workgroup> measureLive:atomic<u32>;
var<workgroup> measureBand:atomic<u32>;
// Seed's corner values die at the measureBand uniform barrier. Reuse the
// same 512 bytes for its 64 two-float reduction rows after that barrier.
var<workgroup> seedScratch:array<f32,128>;
fn sumGroup(l:u32){workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l<stride){sums[l]+=sums[l+stride];}workgroupBarrier();}}
fn storeSum(at:u32,value:vec4f){for(var c=0u;c<4u;c++){scratch[at+c]=value[c];}}
fn loadSum(at:u32)->vec4f{return vec4f(scratch[at],scratch[at+1u],scratch[at+2u],scratch[at+3u]);}
// The workgroup's partial from its lanes' rows. A group without liquid or
// volume sums to zero: skip its tree.
fn storePartial(l:u32,group:vec3u){
 workgroupBarrier();
 if(any(sums[l]!=vec4f(0))){atomicOr(&measureLive,1u);}
 let at=${P}u+4u*(group.x+umDispatchX*group.y);
 if(workgroupUniformLoad(&measureLive)==0u){if(l<4u){scratch[at+l]=0.0;}return;}
 sumGroup(l);
 if(l==0u){storeSum(at,sums[0]);}
}
// Clears the tile flags; tile 0's thread also starts the Newton state:
// shift, the bracket's low and high, converged, secant step.
@compute @workgroup_size(64) fn clearBand(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x+umDispatchX*64u*gid.y;if(t<UM_TILES){scratch[SV_BAND+t]=0.0;scratch[SV_VISIT+t]=0.0;scratch[SV_MEASURE+t]=0.0;}
 if(t==0u){let limit=umShiftLimit();scratch[${S}u]=0.0;scratch[${S+1}u]=-limit;scratch[${S+2}u]=limit;scratch[${S+3}u]=0.0;scratch[${S+4}u]=limit/64.0;}
}
// One owner's corner (general owners reconstruct hanging corners).
fn svCorner(o:UMOwner,k:u32)->f32{
 let vertex=umOrigin(o)+umCorner(k,2u)*o.width;
 if(svDirect(o)){return umLoadVertexW(vertex,o.width);}
 return umVertexValue(vertex);
}
// A seed workgroup that is one unit-stencil fine tile stages its 5^3 stored
// vertices once; every lane then reads its eight corners from them.
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let job=group.x+umDispatchX*group.y;
 let staged=job<umCounts.x&&umTileMaximumWidth(umTopology[UM_TILES+job])==1u;
 // A workgroup runs several jobs (counted launch): clear the flags per job.
 if(l==0u){atomicStore(&measureLive,0u);atomicStore(&measureBand,0u);}
 if(staged){let base=umTileCoord(umTopology[UM_TILES+job])*4u;for(var i=l;i<125u;i+=64u){seedScratch[i]=umLoadVertex(base+umCorner(i,5u));}}
 workgroupBarrier();
 let o=umResidentAllOwner(gid);var inside=false;var seeded=false;
 if(o.width!=0u){
  if(umCapacity(o)>0.0){
   var low=1e30;var high=-1e30;
   if(staged){
    let local=umCorner(o.lane,4u);
    for(var k=0u;k<8u;k++){let c=local+umCorner(k,2u);let v=seedScratch[c.x+5u*(c.y+5u*c.z)];low=min(low,v);high=max(high,v);}
   }else{
   if(svDirect(o)){for(var k=0u;k<8u;k++){let v=umLoadVertexW(umOrigin(o)+umCorner(k,2u)*o.width,o.width);low=min(low,v);high=max(high,v);}}
   else{for(var k=0u;k<umCounts.w;k++){let v=svCorner(o,k);low=min(low,v);high=max(high,v);}}
   }
   seeded=low<=0.0&&high>=0.0;inside=high<0.0;
  }
  let band=select(0.0,5.0,seeded);scratch[svBandAt(0u)+o.index]=band;scratch[svBandAt(1u)+o.index]=band;
 }
 if(seeded){scratch[SV_BAND+o.tile]=1.0;atomicOr(&measureBand,1u);}
 // A seeded group is always measured. Otherwise every corner scale of every
 // lane stays zero: each sample is the owner's whole mass or nothing. Both
 // samples are then one value per lane, and the measure tree adds every
 // component in the same order: reduce (sample, volume) once and replicate.
 if(workgroupUniformLoad(&measureBand)!=0u){return;}
 var row=vec2f(0);
 if(o.width!=0u){
  let mass=f32(o.width*o.width*o.width)*umCapacity(o);row=vec2f(select(0.0,mass,inside),svVolume(o)*f32(o.width*o.width*o.width));
 }
 seedScratch[2u*l]=row.x;seedScratch[2u*l+1u]=row.y;
 if(any(row!=vec2f(0))){atomicOr(&measureLive,1u);}
 let at=${P}u+4u*(group.x+umDispatchX*group.y);
 if(workgroupUniformLoad(&measureLive)==0u){if(l<4u){scratch[at+l]=0.0;}return;}
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){seedScratch[2u*l]+=seedScratch[2u*(l+stride)];seedScratch[2u*l+1u]+=seedScratch[2u*(l+stride)+1u];}workgroupBarrier();}
 let total=vec2f(seedScratch[0],seedScratch[1]);
 // The partial: (s, s, volume, 0).
 if(l<4u){scratch[at+l]=select(select(0.0,total.y,l==2u),total.x,l<2u);}
}
// Visited: tiles within one tile of band. Every corner scale of an owner
// outside them is zero. Dilate tests the same rule inline; grow writes the
// final visit flags once the band is complete.
fn svNearBand(t:u32,k:u32)->bool{
 let q=vec3i(umTileCoord(t))+vec3i(umCorner(k,3u))-vec3i(1);
 return all(q>=vec3i(0))&&all(q<vec3i(UM_T))&&scratch[SV_BAND+umTileAt(vec3u(q))]!=0.0;
}
@compute @workgroup_size(64) fn grow(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let t=umResidentPageTile(group.x,lane);if(t>=UM_TILES){return;}
 let centre=vec3i(umTileCoord(t));var visit=0.0;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let q=centre+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
  if(scratch[SV_BAND+umTileAt(vec3u(q))]!=0.0){visit=1.0;}
 }}}
 scratch[SV_VISIT+t]=visit;
}
// Measured (metric's set): the visited tiles plus every tile with a wider
// owner in its stencil within two tiles of band. A visited tile's hanging
// corners interpolate the corners of an adjacent wider owner, whose
// authorities lie one tile farther. Within two tiles of band is within one
// of a visited tile (the in-domain step toward the band tile is one), so
// this reads the final visit flags' 27 neighbours, not band's 125.
@compute @workgroup_size(64) fn measureGrow(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let t=umResidentPageTile(group.x,lane);if(t>=UM_TILES){return;}
 var measured=scratch[SV_VISIT+t];
 if(measured==0.0&&umTileMaximumWidth(t)>1u){
  let centre=vec3i(umTileCoord(t));
  for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
   let q=centre+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
   if(scratch[SV_VISIT+umTileAt(vec3u(q))]!=0.0){measured=1.0;}
  }}}
 }
 scratch[SV_MEASURE+t]=measured;
 // Closure: every owner a measured tile's passes read lies within one tile.
 if(measured!=0.0){
  let centre=vec3i(umTileCoord(t));let a=max(centre-vec3i(1),vec3i(0));let b=min(centre+vec3i(1),vec3i(UM_T)-vec3i(1));
  var inside=true;
  for(var z=a.z;z<=b.z;z++){for(var y=a.y;y<=b.y;y++){for(var x=a.x;x<=b.x;x++){inside=inside&&umTileResident(umTileAt(vec3u(vec3i(x,y,z))));}}}
  if(!inside){umSupport[UM_RESIDENCY+1u]=1u;}
 }
}
// NB measures this band in h cells, independent of owner width.
// A step dilates the owners of visited tiles (grow's rule on the band flags
// as this step reads them). Band flags another workgroup sets during the
// step only add tiles whose neighbourhood is zero in the input parity: their
// owners write the zero the output parity already holds (seed, monotone band).
// An h tile's lanes test its 27 neighbour tiles together, a 4h lane its own.
// A 4h owner's face is one patch, or the sixteen unit owners of an h tile's
// facing layer: that tile's lanes, with one ownership lookup a side.
var<workgroup> svVisit:atomic<u32>;
@compute @workgroup_size(64) fn dilate(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let fine=group.x<umCounts.x;
 if(l==0u){atomicStore(&svVisit,0u);}
 workgroupBarrier();
 let o=umResidentAllOwner(gid);
 if(fine&&l<27u&&umTileResident(o.tile)&&svNearBand(o.tile,l)){atomicOr(&svVisit,1u);}
 var visited=workgroupUniformLoad(&svVisit)!=0u;
 if(o.width==0u){return;}
 if(!fine){for(var k=0u;k<27u&&!visited;k++){visited=svNearBand(o.tile,k);}}
 if(!visited){return;}let input=svBandAt(parity);let out=svBandAt(parity^1u);var band=scratch[input+o.index];
 if(o.width==1u){
  // A unit owner's every face is one unit patch onto the adjacent owner.
  // Inside its own tile that owner is lane +-1, +-4 or +-16 (umOwnerAt's
  // index); only tile-face neighbours look ownership up.
  let origin=vec3i(umOrigin(o));let local=umCorner(o.lane,4u);
  for(var k=0u;k<6u;k++){
   let axis=k/2u;let up=(k&1u)==1u;let stride=select(select(16u,4u,axis==1u),1u,axis==0u);
   if(select(local[axis]>0u,local[axis]<3u,up)){band=max(band,scratch[input+select(o.index-stride,o.index+stride,up)]-1.0);continue;}
   var q=origin;q[axis]+=select(-1,1,up);let other=umOwnerAt(q);if(other.width!=0u){band=max(band,scratch[input+other.index]-${this.scalarTarget?"f32(max(o.width,other.width))":"1.0"});}
  }
 }else{
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let other=umFace(o,axis,select(-1,1,side==1u),0u).neighbor;if(other.width==0u){continue;}
  if(other.width==o.width){band=max(band,scratch[input+other.index]-${this.scalarTarget?"f32(max(o.width,other.width))":"1.0"});continue;}
  let stride=vec3u(1u,4u,16u);let layer=input+other.index-other.lane+select(3u,0u,side==1u)*stride[axis];
  for(var part=0u;part<16u;part++){band=max(band,scratch[layer+(part%4u)*stride[(axis+1u)%3u]+(part/4u)*stride[(axis+2u)%3u]]-${this.scalarTarget?"f32(o.width)":"1.0"});}
 }}}
 let value=select(0.0,max(0.0,band),umCapacity(o)>0.0);scratch[out+o.index]=value;
 if(value>0.0){scratch[SV_BAND+o.tile]=1.0;}
}
// Scales over the measured tiles, which hold every visited tile. A corner
// whose authority lies outside them has no band in any incident cell: every
// owner there writes the same zero. Nothing reads scales farther away.
// A measured unit-stencil fine tile stages the bands of its 5^3 cell block
// (its cells and the next layer up); its authority corners read only those.
var<workgroup> metricBand:array<f32,125>;
@compute @workgroup_size(64) fn metric(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let job=group.x+umDispatchX*group.y;let fineTile=umTopology[UM_TILES+min(job,UM_TILES-1u)];
 let staged=job<umCounts.x&&umTileMaximumWidth(fineTile)==1u&&svMeasured(fineTile);let block=umTileCoord(fineTile)*4u;
 if(staged){for(var i=l;i<125u;i+=64u){
  let c=block+umCorner(i,5u);var band=0.0;
  if(all(c<UM_D)){let m=c%4u;band=scratch[SV_OWNERS+(umTopology[umTileAt(c/4u)]&0x3fffffffu)+m.x+4u*(m.y+4u*m.z)];}
  metricBand[i]=band;
 }}
 workgroupBarrier();
 let o=umResidentAllOwner(gid);if(o.width==0u||!svMeasured(o.tile)){return;}let unit=svUnit(o);
 for(var k=0u;k<umCounts.w;k++){
  let p=umOrigin(o)+umCorner(k,2u)*o.width;
  if(unit){
   if(!svAuthority(o,k)){if(!svMeasured(svUnitAuthorityTile(p))){scratch[umVertexIndex(p)]=0.0;}continue;}
  }else{
   let authority=umVertexAuthority(p);
   if(authority.index!=o.index){if(!svMeasured(authority.tile)){scratch[umVertexIndex(p)]=0.0;}continue;}
  }
  var band=0.0;
  if(staged){
   // Block coordinates below zero lie outside the lattice (wall corners).
   let local=vec3i(p)-vec3i(block)-vec3i(1);
   for(var j=0u;j<umCounts.w;j++){let c=local+vec3i(umCorner(j,2u));if(all(c>=vec3i(0))){band=max(band,metricBand[u32(c.x+5*(c.y+5*c.z))]);}}
  }else{
   for(var j=0u;j<umCounts.w;j++){let c=umOwnerAt(vec3i(p)+vec3i(umCorner(j,2u))-vec3i(1));if(c.width!=0u){band=max(band,scratch[SV_OWNERS+c.index]);}}
  }
  if(band<=0.0){scratch[umVertexIndex(p)]=0.0;continue;}
  var gradient=vec3f(0);
  ${this.solid?`// Native centred slopes need an open incident cell on both sides.
  var openLow=vec3<bool>(false);var openHigh=vec3<bool>(false);
  for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);if(umCellOpen(vec3i(p)+vec3i(corner)-vec3i(1))>0.0){
   for(var a=0u;a<3u;a++){if(corner[a]==0u){openLow[a]=true;}else{openHigh[a]=true;}}}}`:""}
  // Both ends are width-aligned lattice vertices in tiles whose stencil holds
  // this owner: stored or resolved texels, so the samples are direct loads.
  // A wide owner's wall vertex takes the one-sided slope. Dropping the wall
  // axis (the native rule) leaves a floor vertex under a layer thinner than
  // one owner at the 0.1 floor: it moved a tenth as far as the rest, and the
  // global shift piled that layer's volume onto steep and tall parts.
  for(var axis=0u;axis<3u;axis++){
   if((p[axis]==0u||p[axis]==UM_D[axis])&&o.width==1u){continue;}
   ${this.solid?"if(o.width==1u&&!(openLow[axis]&&openHigh[axis])){continue;}":""}
   var lo=p;var hi=p;lo[axis]=p[axis]-min(p[axis],o.width);hi[axis]=min(UM_D[axis],p[axis]+o.width);
   // A wide slope is one-sided at a buried end: phi there is not state.
   ${this.solid?"if(o.width!=1u){if(umVertexBuried(vec3i(lo))){lo=p;}if(umVertexBuried(vec3i(hi))){hi=p;}if(lo[axis]==hi[axis]){continue;}}":""}
   gradient[axis]=(${this.resolved?"umLoadVertexW(hi,o.width)-umLoadVertexW(lo,o.width)":"umSampleVertex(vec3f(hi))-umSampleVertex(vec3f(lo))"})/(f32(hi[axis]-lo[axis])*UM_H[axis]);
  }
  scratch[umVertexIndex(p)]=band*0.2*max(0.1,length(gradient));
 }
}
${this.resolved?`// Completes the unstored scale texels of mixed-stencil tiles after metric,
// as UniformMixedPhiResolve does for phi (one group per seam tile, compiled
// authority and staged 4h lattice): reconstruction reads only stored scales
// and this writes only unstored ones.
var<workgroup> svResolveLattice:array<f32,27>;
@compute @workgroup_size(125) fn resolveScale(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let header=7u*UM_TILES+16u;let job=group.x+umDispatchX*group.y;
 let valid=job<umSupport[header]+umSupport[header+1u];
 let tile=select(0u,umSupport[header+4u+select(0u,job,valid)],valid);let base=vec3i(umTileCoord(tile));
 if(valid&&lane>=9u&&lane<35u){let v=(base+vec3i(umCorner(lane-8u,3u))-vec3i(1))*4;if(all(v>=vec3i(0))){svResolveLattice[lane-8u]=umScaleLoadVertex(vec3u(v));}}
 workgroupBarrier();
 if(!valid){return;}
${uniformCompiledVertexResolveWGSL("svResolveLattice","umScaleVertexWeight","umScaleVertexSum8")}let at2=umVertexIndex(p);
 if(bitcast<u32>(value)!=bitcast<u32>(scratch[at2])){scratch[at2]=value;}
}
`:""}// Corner values (raw, then scale) of a general owner; runtime-bounded so
// the reconstruction is not expanded eight times.
fn umMeasureCorners(o:UMOwner)->array<f32,16>{
 var corners:array<f32,16>;
 for(var k=0u;k<umCounts.w;k++){let vertex=umOrigin(o)+umCorner(k,2u)*o.width;corners[k]=umVertexValue(vertex);corners[k+8u]=umScaleVertexValue(vertex);}
 return corners;
}
// Each lane writes its filled mass at the shift and one secant step ahead,
// and its volume, into its own workgroup row. A group with no visited tile
// keeps the constant partial its seed wrote; a converged solve skips the
// round. An unvisited lane's scales are zero: its texels may hold an earlier
// frame's scales or interpolate them.
@compute @workgroup_size(64) fn measure(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=umResidentAllOwner(gid);
 if(l==0u){atomicStore(&measureLive,0u);atomicStore(&measureBand,0u);}
 workgroupBarrier();
 if(o.width!=0u&&svVisited(o)&&scratch[${S+3}u]==0.0){atomicOr(&measureBand,1u);}
 if(workgroupUniformLoad(&measureBand)==0u){return;}
 var row=vec4f(0);
 if(o.width!=0u){
  let origin=umOrigin(o);let capacity=umCapacity(o);let mass=f32(o.width*o.width*o.width)*capacity;row.z=svVolume(o)*f32(o.width*o.width*o.width);
  var raw:array<f32,8>;var scale:array<f32,8>;var low=1e30;var high=-1e30;
  let shift=scratch[${S}u];let step=scratch[${S+4}u];
  if(svDirect(o)){
   // Every corner is a stored (or resolved) vertex: eight direct loads.
   for(var k=0u;k<8u;k++){let vertex=origin+umCorner(k,2u)*o.width;raw[k]=umLoadVertexW(vertex,o.width);scale[k]=umScaleLoadVertex(vertex);}
  }else{
   let corners=umMeasureCorners(o);for(var k=0u;k<8u;k++){raw[k]=corners[k];scale[k]=corners[k+8u];}
  }
  if(!svVisited(o)){scale=array<f32,8>();}
  for(var k=0u;k<8u;k++){low=min(low,raw[k]-(shift+step)*scale[k]);high=max(high,raw[k]-shift*scale[k]);}
  // An owner inside or outside the surface at both shifts has a constant
  // fraction; only cut owners evaluate the two shifted fills.
  if(high<0.0){row.x=mass;row.y=mass;}
  else if(low<0.0){
   for(var sample=0u;sample<2u;sample++){
    let at=shift+f32(sample)*step;var values:array<f32,8>;var negative=0u;
    for(var k=0u;k<8u;k++){values[k]=raw[k]-at*scale[k];negative+=select(0u,1u,values[k]<0.0);}
    // A cut 4h owner fills its open cells, not a share of its box.
    if(negative==8u){row[sample]=mass;}
    else if(negative!=0u){if(o.width==4u&&capacity<0.99999){row[sample]=svOpenFill(values,vec3i(umOrigin(o)));}else{row[sample]=fill(values)*mass;}}
   }
  }
 }
 sums[l]=row;
 storePartial(l,group);
}
// A fixed grid strides the chunks of 64 measure partials the GPU count
// holds (at most ${this.chunks}); solve reads exactly those chunk sums.
var<workgroup> svReduceJobs:u32;
@compute @workgroup_size(64) fn reduce(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 // The job count lives in storage: made workgroup-uniform for the barriers.
 if(l==0u){svReduceJobs=svJobs();}
 let count=workgroupUniformLoad(&svReduceJobs);let chunks=(count+63u)/64u;
 for(var chunk=group.x;chunk<chunks;chunk+=groups.x){
  let i=chunk*64u+l;
  sums[l]=vec4f(0);if(i<count){sums[l]=loadSum(${P}u+4u*i);}sumGroup(l);
  if(l==0u){storeSum(${R}u+4u*chunk,sums[0]);}
  workgroupBarrier();
 }
}
// Filled volume never decreases with the shift. Secant Newton, clamped into
// the bracket the measured side narrows; converged once within tolerance or
// when no shift inside the limit can change the filled volume.
@compute @workgroup_size(64) fn solve(@builtin(local_invocation_index) l:u32){
 if(l==0u){svReduceJobs=svJobs();}
 let count=(workgroupUniformLoad(&svReduceJobs)+63u)/64u;
 var sum=vec4f(0);for(var i=l;i<count;i+=64u){sum+=loadSum(${R}u+4u*i);}sums[l]=sum;sumGroup(l);
 if(l==0u&&scratch[${S+3}u]==0.0){
  let filled=sums[0].x;
  ${this.scalarTarget?`if(captureBudget){budget[0]=filled;budget[1]=1.0;budget[5]=filled;scratch[${S+3}u]=1.0;return;}
  if(scratch[${S}u]==0.0){budget[4]=filled;}
  budget[6]=filled;budget[7]=budget[0];`:""}
  let desired=${this.scalarTarget?"budget[0]":"sums[0].z"};let shift=scratch[${S}u];let step=scratch[${S+4}u];
  let rise=sums[0].y-filled;
  if(abs(filled-desired)<=max(1e-5,1e-7*abs(desired))||rise*(2.0*umShiftLimit()/step)<=1e-6){scratch[${S+3}u]=1.0;}
  else{
   var low=scratch[${S+1}u];var high=scratch[${S+2}u];
   if(filled<desired){low=shift;}else{high=shift;}
   scratch[${S+1}u]=low;scratch[${S+2}u]=high;scratch[${S}u]=clamp(shift+(desired-filled)*step/max(rise,1e-30),low,high);
  }
  ${this.scalarTarget?`budget[3]=scratch[${S}u];`:""}
 }
}
${this.scalarTarget?/* wgsl */`
// Initial budget uses geometry alone, before any transport or metric scratch.
@compute @workgroup_size(64) fn measureInitial(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=umResidentAllOwner(gid);var amount=0.0;
 if(l==0u){atomicStore(&measureLive,0u);}workgroupBarrier();
 if(o.width!=0u){
  // The frame has just measured occupancy from the initialized surface.
  // NB binds that geometry output here, never transported cell mass.
  let p=vec3i(umOrigin(o));var occupied=0.0;
  if(o.width==4u){occupied=${UNIFORM_DETAIL_4H_LOAD}textureLoad(volume,p,0).x;}
  else{occupied=textureLoad(volume,p,0).x;}
  amount=occupied*f32(o.width*o.width*o.width);
 }
 sums[l]=vec4f(amount,amount,0,0);storePartial(l,group);
}
// Boundary-only accounting. Integrate the top face's swept liquid column
// with the frozen outward velocity, in h-cell volume units. Subdivide the
// column geometrically for large steps; this does not advance the solver.
@compute @workgroup_size(64) fn outflow(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let i=gid.x;var lost=0.0;
 if(i<UM_D.x*UM_D.z&&budgetParams.z>0.0&&umTileResident(umTileAt(vec3u(i%UM_D.x,UM_D.y-1u,i/UM_D.x)/4u))){
  let c=vec3u(i%UM_D.x,UM_D.y-1u,i/UM_D.x);let o=umOwnerAt(vec3i(c));
  let face=vec3i(umOrigin(o)+vec3u(0u,o.width-1u,0u));
  var vy=0.0;if(o.width==4u){vy=${UNIFORM_DETAIL_4H_LOAD}textureLoad(velocity,face,0).y;}else{vy=textureLoad(velocity,face,0).y;}
  let speed=max(0.0,vy);
  let travel=min(f32(UM_D.y),speed*budgetParams.x/UM_H.y);
  for(var j=0u;f32(j)<travel;j++){
   let depth=min(1.0,travel-f32(j));let y=f32(UM_D.y-j);var v:array<f32,8>;
   for(var k=0u;k<8u;k++){let corner=umCorner(k,2u);v[k]=umSampleVertex(vec3f(f32(c.x+corner.x),y-depth+depth*f32(corner.y),f32(c.z+corner.z)));}
   lost+=fill(v)*depth*umCellOpen(vec3i(vec3u(c.x,UM_D.y-1u-j,c.z)));
  }
 }
 sums[l]=vec4f(lost,0,0,0);sumGroup(l);
 if(l==0u){scratch[${P}u+group.x]=sums[0].x;}
}
@compute @workgroup_size(64) fn updateBudget(@builtin(local_invocation_index) l:u32){
 var lost=0.0;
 if(budgetParams.z>0.0){for(var i=l;i<(UM_D.x*UM_D.z+63u)/64u;i+=64u){lost+=scratch[${P}u+i];}}
 sums[l]=vec4f(lost,0,0,0);sumGroup(l);
 if(l==0u){let out=min(budget[0],sums[0].x);budget[0]=max(0.0,budget[0]+budgetParams.y-out);budget[2]+=out;budget[3]=0.0;}
}`:""}
`,["seed","dilate","metric","measure","grow","measureGrow",...(this.scalarTarget?["measureInitial"]:[]),...(this.resolved?["resolveScale"]:[])])});
  // Apply: in place on the band's authoritative vertices (every other scale
  // is zero).
  const applyModule=uniformDetailModule(this.device,{label:"Uniform mixed surface volume apply",code:uniformMixedCountedEntriesWGSL(topology+/* wgsl */`
@group(1) @binding(2) var phi:texture_storage_3d<r32float,read_write>;
fn umLoadVertex(p:vec3u)->f32{return ${UNIFORM_DETAIL_CANONICAL_LOAD}textureLoad(phi,vec3i(p)).x;}
@group(1) @binding(3) var<storage,read_write> scratch:array<f32>;
${shared(uniformMixedVertexSamplingSource("",false))}
@compute @workgroup_size(64) fn apply(@builtin(global_invocation_id) gid:vec3u){
 let o=umResidentAllOwner(gid);if(o.width==0u||!svVisited(o)){return;}
 for(var k=0u;k<umCounts.w;k++){if(!svAuthority(o,k)){continue;}let p=umOrigin(o)+umCorner(k,2u)*o.width;
  // Unvisited owners' scales are zero and were not rewritten this frame; a
  // zero shift or scale leaves the texel as it is.
  let shift=scratch[${S}u]*scratch[umVertexIndex(p)];
  if(shift!=0.0){textureStore(phi,vec3i(p),vec4f(umLoadVertex(p)-shift));}
 }
}`,["apply"])});
  const modules=[module,applyModule];
  for(const m of modules){const errors=(await m.getCompilationInfo()).messages.filter(e=>e.type==="error");if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join("\n"));}
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[])]});
  const constants={umDispatchX:this.ownership.dispatchX};
  const resident={umCountedJobs:UNIFORM_MIXED_COUNTED.residentAll},pages={umCountedJobs:UNIFORM_MIXED_COUNTED.residentPages};
  const create=(key:string,entryPoint:string,extra:Record<string,number>={})=>uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{layout,compute:{module,entryPoint,constants:{...constants,...extra,...s}}})).then(p=>{this.pipelines.set(key,p);});
  await Promise.all([
   ...["clearBand","reduce","solve"].map(entry=>create(entry,entry)),
   ...(this.scalarTarget?[create("capture","solve",{captureBudget:1}),create("measureInitial","measureInitial",resident),create("outflow","outflow"),create("updateBudget","updateBudget")]:[]),
   ...["seed","metric","measure"].map(entry=>create(entry,entry,resident)),
   ...(this.resolved?[create("resolveScale","resolveScale",{umCountedJobs:UNIFORM_MIXED_COUNTED.fused})]:[]),
   create("dilate0","dilate",{parity:0,...resident}),create("dilate1","dilate",{parity:1,...resident}),
   create("grow","grow",pages),create("measureGrow","measureGrow",pages),
   uniformDetailPipeline(this.device,this.ownership,{layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.applyResources]}),compute:{module:applyModule,entryPoint:"apply",constants:{...constants,...resident}}}).then(p=>{this.pipelines.set("apply",p);}),
  ]);
 }
 /** rounds: secant Newton rounds (measure, reduce, solve); a converged solve skips the rest's work.
  * apply false measures the volume and the shift that would restore it, and leaves phi alone. */
 encode(encoder:GPUCommandEncoder,group:UniformDetailGroup,rounds=2,capture=false,apply=true):void{
  if(!Number.isInteger(rounds)||rounds<1)throw new Error(`Mixed surface constraint needs a positive whole round count, not ${rounds}`);
  if(this.pipelines.size!==((this.resolved?12:11)+(this.scalarTarget?4:0)))throw new Error("Mixed surface constraint is not initialized");
  const applyGroup=this.applyGroups.get(group);if(!applyGroup)throw new Error("Mixed surface constraint group was not bound by this stage");
  const pass=encoder.beginComputePass({label:"Uniform mixed global surface volume"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  // The apply modules hold no solid library: only the constraint's own pipelines have twins.
  const run=(entry:string)=>{const own=this.pipelines.get(entry)!;const pipeline=uniformDetailPick(entry==="apply"?own:this.solid?.select(own)??own);pass.setPipeline(pipeline);
   if(entry==="solve"||entry==="capture")pass.dispatchWorkgroups(1);
   else if(entry==="reduce")this.ownership.dispatchCounted(pass,pipeline,this.chunks);
   else if(entry==="resolveScale")this.ownership.dispatchFusedCounted(pass,pipeline);
   else if(entry==="grow"||entry==="measureGrow")this.ownership.dispatchCounted(pass,pipeline,this.pages);
   else if(entry==="clearBand"){const groups=Math.ceil(this.tiles/64);pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));}
   else this.ownership.dispatchAllCounted(pass,pipeline);};
  if(capture){run("clearBand");run("measureInitial");run("reduce");run("capture");pass.end();return;}
  run("clearBand");run("seed");for(let i=0;i<4;i++)run(`dilate${i%2}`);run("grow");run("measureGrow");run("metric");if(this.resolved)run("resolveScale");
  for(let i=0;i<rounds;i++){run("measure");run("reduce");run("solve");}
  if(apply){pass.setBindGroup(1,applyGroup.group);run("apply");}pass.end();
 }
 /** Capture the initialized surface once; thereafter only explicit sources
  * and boundary outflow change the target. No CPU readback feeds correction. */
 beginStep(encoder:GPUCommandEncoder,group:UniformDetailGroup,dt:number,addedCells:number,openTop:boolean):void{
  if(!this.scalarTarget)throw new Error("Scalar budget is only available for NB-FLIP");
  if(!this.budgetCaptured){this.encode(encoder,group,1,true);this.budgetCaptured=true;}
  if(!openTop&&addedCells===0)return;
  this.device.queue.writeBuffer(this.budgetParams!,0,new Float32Array([dt,addedCells,+openTop,0]));
  const pass=encoder.beginComputePass({label:"NB scalar volume sources and outflow"});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group.group);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  const run=(entry:string,n:number)=>{const own=this.pipelines.get(entry)!;pass.setPipeline(uniformDetailPick(this.solid?.select(own)??own));pass.dispatchWorkgroups(n);};
  if(openTop)run("outflow",Math.ceil(this.ownership.capacity.lattice.dimensions[0]*this.ownership.capacity.lattice.dimensions[2]/64));
  run("updateBudget",1);pass.end();
 }
 destroy():void{this.budget?.destroy();this.budgetParams?.destroy();}

}
