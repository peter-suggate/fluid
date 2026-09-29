import type {UniformMixedOwnership} from "./uniform-mixed-ownership";
import {uniformMixedTopologyWGSL} from "./uniform-mixed-topology.wgsl";
import {uniformMixedVertexSamplingSource} from "./uniform-mixed-vertex-sampling.wgsl";
import {uniformSurfaceFillWGSL} from "./uniform-surface-volume.wgsl";
import {uniformMixedSolidPipeline,uniformMixedSolidWGSL,type UniformMixedSolid} from "./uniform-mixed-solid.wgsl";

/** Native four-cell band and a global normal shift found by two secant
 * Newton rounds, reduced with physical owner volumes. Every field is borrowed. With static
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
 * differ. Every value and reduction order equals the dense evaluation's.
 *
 * bind(phi,volume,phi,scratch) corrects phi in place, touching only band
 * vertices; a separate output receives every authoritative vertex. */
export class UniformMixedSurfaceVolume {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 private readonly cells:number;
 private readonly vertices:number;
 private readonly groups:number;
 private readonly chunks:number;
 private readonly tiles:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly copyResources:GPUBindGroupLayout;
 private readonly inPlaceResources:GPUBindGroupLayout;
 private readonly applyGroups=new WeakMap<GPUBindGroup,{group:GPUBindGroup;inPlace:boolean}>();
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 /** resolved: the caller runs UniformMixedPhiResolve on phi before encode.
  * Every owner corner is then a stored or resolved texel; resolveScale
  * completes the corner scales the same way after metric. */
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly resolved=false){
  this.cells=ownership.layout.lattice.dimensions.reduce((n,d)=>n*d,1);
  this.vertices=ownership.layout.lattice.dimensions.reduce((n,d)=>n*(d+1),1);
  this.groups=Math.ceil(this.cells/64);this.chunks=Math.ceil(this.groups/64);
  this.tiles=ownership.layout.tiles.length;
  this.scratchBytes=4*(this.cells*2+this.vertices+4*(this.groups+this.chunks)+8+3*this.tiles);
  const texture=(binding:number)=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}});
  const scratch={binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}};
  this.resources=device.createBindGroupLayout({entries:[texture(0),texture(1),scratch]});
  this.copyResources=device.createBindGroupLayout({entries:[texture(0),{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},scratch]});
  this.inPlaceResources=device.createBindGroupLayout({entries:[{binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32float",viewDimension:"3d"}},scratch]});
 }
 bind(phi:GPUTexture,volume:GPUTexture,output:GPUTexture,scratch:GPUBufferBinding):GPUBindGroup{
  const d=this.ownership.layout.lattice.dimensions;
  for(const [i,t] of [phi,volume,output].entries())if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((n,a)=>n!==d[a]!+(i===1?0:1)))throw new Error("Mixed surface constraint requires native vertex and cell fields");
  if((scratch.size??scratch.buffer.size-(scratch.offset??0))<this.scratchBytes)throw new Error("Mixed surface constraint needs sufficient scratch");
  const work={...scratch,size:this.scratchBytes};
  const group=this.device.createBindGroup({layout:this.resources,entries:[{binding:0,resource:phi.createView()},{binding:1,resource:volume.createView()},{binding:3,resource:work}]});
  const inPlace=phi===output;
  this.applyGroups.set(group,{inPlace,group:inPlace
   ?this.device.createBindGroup({layout:this.inPlaceResources,entries:[{binding:2,resource:phi.createView()},{binding:3,resource:work}]})
   :this.device.createBindGroup({layout:this.copyResources,entries:[{binding:0,resource:phi.createView()},{binding:2,resource:output.createView()},{binding:3,resource:work}]})});
  return group;
 }
 async initialize():Promise<void>{
  const N=this.cells,V=this.vertices,P=2*N+V,R=P+4*this.groups,S=R+4*this.chunks,T=this.tiles;
  const topology=uniformMixedTopologyWGSL(this.ownership.layout,0);
  const shared=(sampling:string)=>/* wgsl */`
const UM_H=vec3f(${this.ownership.layout.lattice.cellSize_m.join(",")});
${sampling}
fn umVertexIndex(p:vec3u)->u32{let d=UM_D+vec3u(1);return p.x+d.x*(p.y+d.y*p.z);}
// Band tiles, visited tiles, measured tiles.
const SV_BAND:u32=${S+8}u;const SV_VISIT:u32=${S+8+T}u;const SV_MEASURE:u32=${S+8+2*T}u;
fn svVisited(o:UMOwner)->bool{return scratch[SV_VISIT+o.tile]!=0.0;}
// Every incident owner of a unit-stencil owner's corners has unit width: the
// corners are stored, and a corner belongs to the lowest incident cell, so
// the owner holds its positive corner and its corners on negative walls.
fn svUnit(o:UMOwner)->bool{return umTileMaximumWidth(o.tile)==1u;}
// An owner's corners are width-aligned vertices of its tile closure: stored
// (uniform or unit stencil) or, when resolved, completed in place (mixed).
fn svDirect(o:UMOwner)->bool{return ${this.resolved}||svUnit(o);}
fn svAuthority(o:UMOwner,k:u32)->bool{
 let corner=umCorner(k,2u);
 if(svUnit(o)){return all((corner!=vec3u(0))|(umOrigin(o)==vec3u(0)));}
 return umVertexAuthority(umOrigin(o)+corner*o.width).index==o.index;
}
// Tile of a unit-stencil owner's corner authority, clamp(p-1)'s owner.
fn svUnitAuthorityTile(p:vec3u)->u32{return umTileAt(vec3u(clamp(vec3i(p)-vec3i(1),vec3i(0),vec3i(UM_D)-vec3i(1)))/4u);}
fn svMeasured(t:u32)->bool{return scratch[SV_MEASURE+t]!=0.0;}`;
  const sourceScale=uniformMixedVertexSamplingSource("",false).replace(/\bum(Vertex\w*|SampleVertex|LoadVertex)\b/g,name=>name.replace("um","umScale"));
  const module=this.device.createShaderModule({code:topology+/* wgsl */`
@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> scratch:array<f32>;
override parity:u32=0u;
override svMeasureReach:bool=false;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
${shared(uniformMixedVertexSamplingSource("",this.resolved))}
fn umScaleLoadVertex(p:vec3u)->f32{return scratch[${2*N}u+umVertexIndex(p)];}
${sourceScale}
${uniformSurfaceFillWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined)}
fn umCapacity(o:UMOwner)->f32{return select(1.0,umCellOpen(vec3i(umOrigin(o))),umSolidEnabled()&&o.width==1u);}
fn umLiveCells()->u32{return umCounts.x*64u+umCounts.y;}
fn svVolume(origin:vec3u)->f32{return textureLoad(volume,vec3i(origin),0).x;}
fn umShiftLimit()->f32{return min(UM_H.x,min(UM_H.y,UM_H.z))*f32(select(4u,1u,umCounts.x>0u));}
// A lane's row: filled mass at the shift and one secant step ahead, volume.
var<workgroup> sums:array<vec4f,64>;
var<workgroup> measureLive:atomic<u32>;
var<workgroup> measureBand:atomic<u32>;
var<workgroup> seedSums:array<vec2f,64>;
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
// Newton state: shift, the bracket's low and high, converged, secant step.
@compute @workgroup_size(1) fn begin(){let limit=umShiftLimit();scratch[${S}u]=0.0;scratch[${S+1}u]=-limit;scratch[${S+2}u]=limit;scratch[${S+3}u]=0.0;scratch[${S+4}u]=limit/64.0;}
@compute @workgroup_size(64) fn clearBand(@builtin(global_invocation_id) gid:vec3u){let t=gid.x+umDispatchX*64u*gid.y;if(t<UM_TILES){scratch[SV_BAND+t]=0.0;}}
// One owner's corner (general owners reconstruct hanging corners).
fn svCorner(o:UMOwner,k:u32)->f32{
 let vertex=umOrigin(o)+umCorner(k,2u)*o.width;
 if(svDirect(o)){return umLoadVertex(vertex);}
 return umVertexValue(vertex);
}
// A seed workgroup that is one unit-stencil fine tile stages its 5^3 stored
// vertices once; every lane then reads its eight corners from them.
var<workgroup> seedCorners:array<f32,125>;
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let job=group.x+umDispatchX*group.y;
 let staged=job<umCounts.x&&umTileMaximumWidth(umTopology[UM_TILES+job])==1u;
 if(staged){let base=umTileCoord(umTopology[UM_TILES+job])*4u;for(var i=l;i<125u;i+=64u){seedCorners[i]=umLoadVertex(base+umCorner(i,5u));}}
 workgroupBarrier();
 let o=umAllOwner(gid);var inside=false;var seeded=false;
 if(o.width!=0u){
  if(umCapacity(o)>0.0){
   var low=1e30;var high=-1e30;
   if(staged){
    let local=umCorner(o.lane,4u);
    for(var k=0u;k<8u;k++){let c=local+umCorner(k,2u);let v=seedCorners[c.x+5u*(c.y+5u*c.z)];low=min(low,v);high=max(high,v);}
   }else{
   if(svDirect(o)){for(var k=0u;k<8u;k++){let v=umLoadVertex(umOrigin(o)+umCorner(k,2u)*o.width);low=min(low,v);high=max(high,v);}}
   else{for(var k=0u;k<umCounts.w;k++){let v=svCorner(o,k);low=min(low,v);high=max(high,v);}}
   }
   seeded=low<=0.0&&high>=0.0;inside=high<0.0;
  }
  let band=select(0.0,5.0,seeded);scratch[o.index]=band;scratch[${N}u+o.index]=band;
 }
 if(seeded){scratch[SV_BAND+o.tile]=1.0;atomicOr(&measureBand,1u);}
 // A seeded group is always measured. Otherwise every corner scale of every
 // lane stays zero: each sample is the owner's whole mass or nothing. Both
 // samples are then one value per lane, and the measure tree adds every
 // component in the same order: reduce (sample, volume) once and replicate.
 if(workgroupUniformLoad(&measureBand)!=0u){return;}
 var row=vec2f(0);
 if(o.width!=0u){
  let mass=f32(o.width*o.width*o.width)*umCapacity(o);row=vec2f(select(0.0,mass,inside),svVolume(umOrigin(o))*f32(o.width*o.width*o.width));
 }
 seedSums[l]=row;
 if(any(row!=vec2f(0))){atomicOr(&measureLive,1u);}
 let at=${P}u+4u*(group.x+umDispatchX*group.y);
 if(workgroupUniformLoad(&measureLive)==0u){if(l<4u){scratch[at+l]=0.0;}return;}
 for(var stride=32u;stride>0u;stride/=2u){if(l<stride){seedSums[l]+=seedSums[l+stride];}workgroupBarrier();}
 let total=seedSums[0];
 // The partial: (s, s, volume, 0).
 if(l<4u){scratch[at+l]=select(select(0.0,total.y,l==2u),total.x,l<2u);}
}
// Visited: tiles within one tile of band. Every corner scale of an owner
// outside them is zero. With svMeasureReach, measured (metric's set): the
// visited tiles plus every tile with a wider owner in its stencil within two
// tiles of band. A visited tile's hanging corners interpolate the corners of
// an adjacent wider owner, whose authorities lie one tile farther.
@compute @workgroup_size(64) fn grow(@builtin(global_invocation_id) gid:vec3u){
 let t=gid.x+umDispatchX*64u*gid.y;if(t>=UM_TILES){return;}
 let centre=vec3i(umTileCoord(t));var visit=0.0;
 for(var z=-1;z<=1;z++){for(var y=-1;y<=1;y++){for(var x=-1;x<=1;x++){
  let q=centre+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
  if(scratch[SV_BAND+umTileAt(vec3u(q))]!=0.0){visit=1.0;}
 }}}
 scratch[SV_VISIT+t]=visit;
 if(svMeasureReach){
  var measured=visit;
  if(measured==0.0&&umTileMaximumWidth(t)>1u){
   for(var z=-2;z<=2;z++){for(var y=-2;y<=2;y++){for(var x=-2;x<=2;x++){
    let q=centre+vec3i(x,y,z);if(any(q<vec3i(0))||any(q>=vec3i(UM_T))){continue;}
    if(scratch[SV_BAND+umTileAt(vec3u(q))]!=0.0){measured=1.0;}
   }}}
  }
  scratch[SV_MEASURE+t]=measured;
 }
}
@compute @workgroup_size(64) fn dilate(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u||!svVisited(o)){return;}let input=parity*${N}u;let out=(parity^1u)*${N}u;var band=scratch[input+o.index];
 if(o.width==1u){
  // A unit owner's every face is one unit patch onto the adjacent owner.
  // Inside its own tile that owner is lane +-1, +-4 or +-16 (umOwnerAt's
  // index); only tile-face neighbours look ownership up.
  let origin=vec3i(umOrigin(o));let local=umCorner(o.lane,4u);
  for(var k=0u;k<6u;k++){
   let axis=k/2u;let up=(k&1u)==1u;let stride=select(select(16u,4u,axis==1u),1u,axis==0u);
   if(select(local[axis]>0u,local[axis]<3u,up)){band=max(band,scratch[input+select(o.index-stride,o.index+stride,up)]-1.0);continue;}
   var q=origin;q[axis]+=select(-1,1,up);let other=umOwnerAt(q);if(other.width!=0u){band=max(band,scratch[input+other.index]-1.0);}
  }
 }else{
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let sign=select(-1,1,side==1u);let first=umFace(o,axis,sign,0u);
  for(var part=0u;part<first.count;part++){let other=umFace(o,axis,sign,part).neighbor;if(other.width!=0u){band=max(band,scratch[input+other.index]-1.0);}}
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
  if(all(c<UM_D)){let m=c%4u;band=scratch[(umTopology[umTileAt(c/4u)]&0x3fffffffu)+m.x+4u*(m.y+4u*m.z)];}
  metricBand[i]=band;
 }}
 workgroupBarrier();
 let o=umAllOwner(gid);if(o.width==0u||!svMeasured(o.tile)){return;}let unit=svUnit(o);
 for(var k=0u;k<umCounts.w;k++){
  let p=umOrigin(o)+umCorner(k,2u)*o.width;
  if(unit){
   if(!svAuthority(o,k)){if(!svMeasured(svUnitAuthorityTile(p))){scratch[${2*N}u+umVertexIndex(p)]=0.0;}continue;}
  }else{
   let authority=umVertexAuthority(p);
   if(authority.index!=o.index){if(!svMeasured(authority.tile)){scratch[${2*N}u+umVertexIndex(p)]=0.0;}continue;}
  }
  var band=0.0;
  if(staged){
   // Block coordinates below zero lie outside the lattice (wall corners).
   let local=vec3i(p)-vec3i(block)-vec3i(1);
   for(var j=0u;j<umCounts.w;j++){let c=local+vec3i(umCorner(j,2u));if(all(c>=vec3i(0))){band=max(band,metricBand[u32(c.x+5*(c.y+5*c.z))]);}}
  }else{
   for(var j=0u;j<umCounts.w;j++){let c=umOwnerAt(vec3i(p)+vec3i(umCorner(j,2u))-vec3i(1));if(c.width!=0u){band=max(band,scratch[c.index]);}}
  }
  if(band<=0.0){scratch[${2*N}u+umVertexIndex(p)]=0.0;continue;}
  var gradient=vec3f(0);
  ${this.solid?`// Native centred slopes need an open incident cell on both sides.
  var openLow=vec3<bool>(false);var openHigh=vec3<bool>(false);
  for(var j=0u;j<8u;j++){let corner=umCorner(j,2u);if(umCellOpen(vec3i(p)+vec3i(corner)-vec3i(1))>0.0){
   for(var a=0u;a<3u;a++){if(corner[a]==0u){openLow[a]=true;}else{openHigh[a]=true;}}}}`:""}
  // Both ends are width-aligned lattice vertices in tiles whose stencil holds
  // this owner: stored or resolved texels, so the samples are direct loads.
  for(var axis=0u;axis<3u;axis++){
   if(p[axis]==0u||p[axis]==UM_D[axis]){continue;}
   ${this.solid?"if(o.width==1u&&!(openLow[axis]&&openHigh[axis])){continue;}":""}
   var lo=p;var hi=p;lo[axis]=p[axis]-min(p[axis],o.width);hi[axis]=min(UM_D[axis],p[axis]+o.width);
   gradient[axis]=(${this.resolved?"umLoadVertex(hi)-umLoadVertex(lo)":"umSampleVertex(vec3f(hi))-umSampleVertex(vec3f(lo))"})/(f32(hi[axis]-lo[axis])*UM_H[axis]);
  }
  scratch[${2*N}u+umVertexIndex(p)]=band*0.2*max(0.1,length(gradient));
 }
}
${this.resolved?`// Completes the unstored scale texels of mixed-stencil tiles after metric,
// as UniformMixedPhiResolve does for phi (one group per seam tile, incident
// tile words and 4h lattice staged): reconstruction reads only stored scales
// and this writes only unstored ones.
var<workgroup> svResolveWords:array<u32,8>;var<workgroup> svResolveLattice:array<f32,27>;
@compute @workgroup_size(125) fn resolveScale(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let header=7u*UM_TILES+16u;let job=group.x+umDispatchX*group.y;
 let valid=job<umSupport[header]+umSupport[header+1u];
 let tile=select(0u,umSupport[header+4u+select(0u,job,valid)],valid);let base=vec3i(umTileCoord(tile));
 if(valid&&lane<8u){let t=base+vec3i(umCorner(lane,2u))-vec3i(1);svResolveWords[lane]=select(0x80000000u,umTopology[umTileAt(vec3u(max(t,vec3i(0))))],all(t>=vec3i(0)));}
 else if(valid&&lane>=8u&&lane<35u){let v=(base+vec3i(umCorner(lane-8u,3u))-vec3i(1))*4;if(all(v>=vec3i(0))){svResolveLattice[lane-8u]=umScaleLoadVertex(vec3u(v));}}
 workgroupBarrier();
 if(!valid){return;}
 let local=umCorner(lane,5u);let p=vec3u(base)*4u+local;
 if(any((local==vec3u(4u))&(p!=UM_D))||all(p%4u==vec3u(0))){return;}
 var best=0xffffffffu;var at=vec3u(0);
 for(var k=0u;k<8u;k++){
  let c=umCorner(k,2u);let word=svResolveWords[k];
  if(any((c==vec3u(0))&(local!=vec3u(0)))||(word&0xc0000000u)!=0u){continue;}
  if((word&0x3fffffffu)<best){best=word&0x3fffffffu;at=c;}
 }
 if(best==0xffffffffu){return;}
 let t=vec3f(local+(vec3u(1)-at)*4u)/4.0;var values:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let corner=umCorner(k,2u);let weight=umScaleVertexWeight(t,corner);let m=at+corner;
  if(weight>0.0){values[k]=weight*svResolveLattice[m.x+3u*(m.y+3u*m.z)];}
 }
 let value=umScaleVertexSum8(values);let at2=${2*N}u+umVertexIndex(p);
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
 let o=umAllOwner(gid);
 if(l==0u){atomicStore(&measureLive,0u);atomicStore(&measureBand,0u);}
 workgroupBarrier();
 if(o.width!=0u&&svVisited(o)&&scratch[${S+3}u]==0.0){atomicOr(&measureBand,1u);}
 if(workgroupUniformLoad(&measureBand)==0u){return;}
 var row=vec4f(0);
 if(o.width!=0u){
  let origin=umOrigin(o);let mass=f32(o.width*o.width*o.width)*umCapacity(o);row.z=svVolume(origin)*f32(o.width*o.width*o.width);
  var raw:array<f32,8>;var scale:array<f32,8>;var low=1e30;var high=-1e30;
  let shift=scratch[${S}u];let step=scratch[${S+4}u];
  if(svDirect(o)){
   // Every corner is a stored (or resolved) vertex: eight direct loads.
   for(var k=0u;k<8u;k++){let vertex=origin+umCorner(k,2u)*o.width;raw[k]=umLoadVertex(vertex);scale[k]=umScaleLoadVertex(vertex);}
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
    var fraction=0.0;if(negative==8u){fraction=1.0;}else if(negative!=0u){fraction=fill(values);}
    row[sample]=fraction*mass;
   }
  }
 }
 sums[l]=row;
 storePartial(l,group);
}
@compute @workgroup_size(64) fn reduce(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 let i=group.x*64u+l;let count=(umLiveCells()+63u)/64u;
 sums[l]=vec4f(0);if(i<count){sums[l]=loadSum(${P}u+4u*i);}sumGroup(l);
 if(l==0u){storeSum(${R}u+4u*group.x,sums[0]);}
}
// Filled volume never decreases with the shift. Secant Newton, clamped into
// the bracket the measured side narrows; converged once within tolerance or
// when no shift inside the limit can change the filled volume.
@compute @workgroup_size(64) fn solve(@builtin(local_invocation_index) l:u32){
 let count=(umLiveCells()+4095u)/4096u;
 var sum=vec4f(0);for(var i=l;i<count;i+=64u){sum+=loadSum(${R}u+4u*i);}sums[l]=sum;sumGroup(l);
 if(l==0u&&scratch[${S+3}u]==0.0){
  let filled=sums[0].x;let desired=sums[0].z;let shift=scratch[${S}u];let step=scratch[${S+4}u];
  let rise=sums[0].y-filled;
  if(abs(filled-desired)<=max(1e-5,1e-7*abs(desired))||rise*(2.0*umShiftLimit()/step)<=1e-6){scratch[${S+3}u]=1.0;}
  else{
   var low=scratch[${S+1}u];var high=scratch[${S+2}u];
   if(filled<desired){low=shift;}else{high=shift;}
   scratch[${S+1}u]=low;scratch[${S+2}u]=high;scratch[${S}u]=clamp(shift+(desired-filled)*step/max(rise,1e-30),low,high);
  }
 }
}
`});
  // Apply: in place on the band's authoritative vertices (every other scale
  // is zero), or every authoritative vertex into a separate output.
  const applyModule=(inPlace:boolean)=>this.device.createShaderModule({code:topology+/* wgsl */`
${inPlace?`@group(1) @binding(2) var phi:texture_storage_3d<r32float,read_write>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p)).x;}`:`@group(1) @binding(0) var phi:texture_3d<f32>;
@group(1) @binding(2) var output:texture_storage_3d<r32float,write>;
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}`}
@group(1) @binding(3) var<storage,read_write> scratch:array<f32>;
${shared(uniformMixedVertexSamplingSource("",false))}
@compute @workgroup_size(64) fn apply(@builtin(global_invocation_id) gid:vec3u){
 let o=umAllOwner(gid);if(o.width==0u${inPlace?"||!svVisited(o)":""}){return;}let visited=svVisited(o);
 for(var k=0u;k<umCounts.w;k++){if(!svAuthority(o,k)){continue;}let p=umOrigin(o)+umCorner(k,2u)*o.width;
  // Unvisited owners' scales are zero and were not rewritten this frame. In
  // place, a zero shift or scale leaves the texel as it is.
  let shift=select(0.0,scratch[${S}u]*scratch[${2*N}u+umVertexIndex(p)],visited);
  ${inPlace?"if(shift!=0.0){textureStore(phi,vec3i(p),vec4f(umLoadVertex(p)-shift));}":"textureStore(output,vec3i(p),vec4f(umLoadVertex(p)-shift));"}
 }
}`});
  const modules=[module,applyModule(true),applyModule(false)];
  for(const m of modules){const errors=(await m.getCompilationInfo()).messages.filter(e=>e.type==="error");if(errors.length)throw new Error(errors.map(e=>`${e.lineNum}: ${e.message}`).join("\n"));}
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.bindLayout]:[])]});
  const constants={umDispatchX:this.ownership.dispatchX};
  const create=(key:string,entryPoint:string,extra:Record<string,number>={})=>uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{...constants,...extra,...s}}})).then(p=>{this.pipelines.set(key,p);});
  await Promise.all([
   ...["begin","clearBand","seed","metric","measure","reduce","solve",...(this.resolved?["resolveScale"]:[])].map(entry=>create(entry,entry)),
   create("dilate0","dilate",{parity:0}),create("dilate1","dilate",{parity:1}),
   create("grow","grow"),create("growMeasure","grow",{svMeasureReach:1}),
   ...([["applyInPlace",1,this.inPlaceResources],["applyCopy",2,this.copyResources]] as const).map(([key,m,resources])=>
    this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,resources]}),compute:{module:modules[m]!,entryPoint:"apply",constants}}).then(p=>{this.pipelines.set(key,p);})),
  ]);
 }
 encode(encoder:GPUCommandEncoder,group:GPUBindGroup):void{
  if(this.pipelines.size!==(this.resolved?14:13))throw new Error("Mixed surface constraint is not initialized");
  const apply=this.applyGroups.get(group);if(!apply)throw new Error("Mixed surface constraint group was not bound by this stage");
  const pass=encoder.beginComputePass({label:"Uniform mixed global surface volume"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.solid.bindGroup);
  // The apply modules hold no solid library: only the constraint's own pipelines have twins.
  const run=(entry:string)=>{const own=this.pipelines.get(entry)!;const pipeline=entry.startsWith("apply")?own:this.solid?.select(own)??own;pass.setPipeline(pipeline);
   if(entry==="begin"||entry==="solve")pass.dispatchWorkgroups(1);
   else if(entry==="reduce")pass.dispatchWorkgroups(Math.ceil(this.ownership.layout.cellCount/4096));
   else if(entry==="resolveScale")this.ownership.dispatchFused(pass,pipeline);
   else if(entry==="grow"||entry==="growMeasure"||entry==="clearBand"){const groups=Math.ceil(this.tiles/64);pass.dispatchWorkgroups(Math.min(groups,this.ownership.dispatchX),Math.ceil(groups/this.ownership.dispatchX));}
   else this.ownership.dispatchAll(pass,pipeline);};
  run("begin");run("clearBand");run("seed");for(let i=0;i<4;i++){run("grow");run(`dilate${i%2}`);}run("growMeasure");run("metric");if(this.resolved)run("resolveScale");
  for(let i=0;i<2;i++){run("measure");run("reduce");run("solve");}
  pass.setBindGroup(1,apply.group);run(apply.inPlace?"applyInPlace":"applyCopy");pass.end();
 }
}
