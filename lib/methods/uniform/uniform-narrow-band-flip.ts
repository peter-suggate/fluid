import { narrowBandRedistanceWGSL } from "./uniform-narrow-band-redistance.wgsl";
import { narrowBandMembershipWGSL } from "./uniform-narrow-band-membership.wgsl";
import { NARROW_BAND_TRACE_LIMIT, narrowBandTraceWGSL } from "./uniform-narrow-band-advection.wgsl";
import { uniformMixedSourceWGSL, UNIFORM_PARAMS_BYTES } from "./uniform-mixed-source.wgsl";
import { UniformNarrowBandOrder } from "./uniform-narrow-band-order";
import { NARROW_BAND_SURFACE_RADIUS, narrowBandSurfaceWGSL, narrowBandCoarseTransferWGSL, narrowBandFineTransferWGSL } from "./uniform-narrow-band-surface.wgsl";
import { UNIFORM_DETAIL_4H_LOAD } from "../../core/uniform-detail-abi";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPipeline, uniformDetailPick, type UniformDetailGroup } from "./uniform-detail-fields";
import { uniformMixedTopologyWGSL, uniformMixedCertifiedEntriesWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformMixedFaceAddressWGSL } from "./uniform-mixed-face-dispatch.wgsl";
import { uniformMixedVertexSamplingSource } from "./uniform-mixed-vertex-sampling.wgsl";
import { uniformMixedVelocitySamplingSource } from "./uniform-mixed-velocity-sampling.wgsl";
import { uniformMixedSolidWGSL, uniformMixedSolidPipeline, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";
import { UniformMixedMomentumCache, type UniformMixedHangingTaps } from "./uniform-mixed-momentum-cache";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import type { GPUFluidParticleSource } from "../../core/webgpu-particle-overlay";

/** Ferstl et al. 2016: persistent samples in a band under the surface carry
 * the liquid's velocity there and overrule the advected level set (Eq. 4).
 * Every sample is liquid; none is set aside as spray.
 * R=4h, r=2h: preserve the axial quadratic footprint; diagonal kernel
 * support beyond the 2h inner collar is truncated (see the alignment notes).
 * Only the inner overlap is routinely reseeded; the outer surface is retained,
 * except where a source adds liquid, which gets its whole band that step.
 * R controls lifetime on either grid width; swept crossing cells control h coverage.
 * Coupled samples compare identical liquid-face extrapolations across pressure.
 * before.w tracks transfer state: 0 coupled, 1 ballistic. A sample no resolved
 * liquid cell supports is ballistic: it keeps its staggered momentum, receives
 * gravity directly, and couples again with that momentum once supported. */
export class UniformNarrowBandFlip {
 readonly capacity:number;
 readonly particles:readonly [GPUBuffer,GPUBuffer];
 readonly state:GPUBuffer;
 private readonly baseAllocatedBytes:number;
 get allocatedBytes():number{return this.baseAllocatedBytes+this.order.allocatedBytes+this.moveDispatch.size;}
 private readonly order:UniformNarrowBandOrder;
 private readonly moveDispatch:GPUBuffer;
 private moveDispatchPipeline!:GPUComputePipeline;
 private moveDispatchGroup!:GPUBindGroup;
 readonly surfaceSource:{contourVertexPhi:true;vertexPhi:GPUTexture;openFraction:GPUTexture;cellSize_m:readonly [number,number,number]};
 get coarseOnly():boolean{return this.ownership.capacity.fineTiles===0;}
 count=0;
 reseedClipped=0;
 /** Positive distances in h cells, before union and on the accepted surface.
  * Unresolved samples are reported separately from the resolved liquid band. */
 diagnostics={beforeMaxOutside:0,afterMaxOutside:0,outsideSurface:0,unsupported:0,deepInterior:0};
 /** Completed frame samples, exposed for transfer/conservation verification. */
 get activeParticles():GPUBuffer{return this.particles[this.parity]!;}
 /** The completed frame's samples for the particle layer: positions in h cells,
  * live prefix counted by the receipt's first word. Eight samples seed a cell,
  * so a sphere is a little under a quarter cell in radius. update leaves each
  * record the grid velocity it took and its depth; the inner collar is the
  * surface's share of the band. */
 get particleSource():GPUFluidParticleSource{
  const h=this.ownership.capacity.lattice.cellSize_m;
  return {buffer:this.activeParticles,strideFloats:12,capacity:this.capacity,positionScale_m:[h[0]!,h[1]!,h[2]!],
   radius_m:0.22*Math.min(...h),liveCount:{buffer:this.state,byteOffset:0},
   grid:{velocityFloat:8,depthFloat:7,ballisticFloat:11,surfaceDepth:2}};
 }
 private readonly bins:GPUBuffer;
 private readonly next:GPUBuffer;
 private readonly params:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 private groups:Record<string,readonly [UniformDetailGroup,UniformDetailGroup]>={};
 private parity=0;
 private started=false;
 private bandCurrent=false;
 private refinementPipeline?:GPUComputePipeline;
 private refinementLayout?:GPUBindGroupLayout;
 private readonly refinementParams:GPUBuffer;
 private readonly cache:UniformMixedMomentumCache;
 private readonly coarseVelocity:GPUTexture;
 private cacheGroups:Record<string,UniformDetailGroup>={};
 private stageTaps?:{builder:UniformMixedHangingTaps;group:UniformDetailGroup;unit:GPUTexture};
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,readonly coarseParticles=false,private readonly sourceParams?:GPUBuffer){
  const cells=ownership.capacity.lattice.dimensions.reduce((n,v)=>n*v,1);
  const vertices=ownership.capacity.lattice.dimensions.reduce((n,v)=>n*(v+1),1);
  // The band is a surface, but what a scene's surface grows to is not known
  // here: half a sample per cell is the budget a 128-cubed grid has always
  // had, kept in proportion on larger ones.
  this.capacity=Math.min(cells*8,Math.max(1_048_576,cells>>1),Math.floor(device.limits.maxStorageBufferBindingSize/48));
  const buffer=(label:string,size:number,uniform=false)=>device.createBuffer({label:`Narrow-band FLIP ${label}`,size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.particles=[buffer("particles A",this.capacity*48),buffer("particles B",this.capacity*48)];
  this.moveDispatch=device.createBuffer({label:"Narrow-band FLIP live particle dispatch",size:12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
  this.state=buffer("receipt",48);this.bins=buffer("cell counts, sort cursors, coverage, surface tiles, depth guard and band tiles",(3*cells+vertices)*4+ownership.capacity.tiles*36+16);this.next=buffer("cell starts and compact neighbor samples",cells*4+this.capacity*32);this.params=buffer("parameters",32,true);
  this.order=new UniformNarrowBandOrder(device,ownership.capacity.lattice.dimensions,this.particles,this.bins,this.next,this.state);
  this.refinementParams=buffer("refinement prediction",32,true);
  this.cache=new UniformMixedMomentumCache(device,ownership);
  const size=ownership.capacity.lattice.dimensions.map(n=>n/4+2);
  this.coarseVelocity=device.createTexture({label:"Narrow-band FLIP stage sampling cache",size,dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
  const dims=ownership.capacity.lattice.dimensions;
  const field=(label:string,size:readonly number[])=>device.createTexture({label,size:[...size],dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
  this.surfaceSource={contourVertexPhi:true,vertexPhi:field("Narrow-band particle render surface",coarseParticles?dims.map(n=>n+1):[1,1,1]),openFraction:field("Narrow-band render solid fractions",coarseParticles?dims:[1,1,1]),cellSize_m:dims.map((_,a)=>ownership.capacity.lattice.cellSize_m[a]!) as [number,number,number]};
  this.baseAllocatedBytes=this.capacity*132+(3*cells+vertices)*4+ownership.capacity.tiles*32+128+size.reduce((n,v)=>n*v,16)+(coarseParticles?4*(cells+dims.reduce((n,v)=>n*(v+1),1)):8);
  this.resources=uniformDetailBindLayout(device,{entries:[
   ...[0,1,2,3,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:(binding===5?"read-only-storage":"storage") as GPUBufferBindingType}})),
   {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...[7,8,10,11,15].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   ...[12,13,14].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
   {binding:9,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}},
   ...(sourceParams?[{binding:16,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform" as const}}]:[]),
  ]});
 }
 bind(fields:{phi:GPUTexture;phiScratch:GPUTexture;phase:GPUTexture;velocity:GPUTexture;velocityScratch:GPUTexture;departure:GPUTexture;negative:GPUBuffer;negativeScratch:GPUBuffer;negativeDeparture:GPUBuffer;coarseExtended:GPUTexture;unitExtended:GPUTexture;hanging:UniformMixedHangingTaps}):void{
  const f=fields;
  // Snapshot, update and seed sample velocityScratch after it is rewritten.
  // Its unit taps are filled first, against this stage's own 4h cache and
  // into this stage's own texture (viscosity still reads the frame's), so a
  // seam tap is one load there as it is in advection.
  // The transfer writes the stored velocity: momentum was its last reader,
  // and the pressure split rewrites every face of it after the forces.
  // One texture for the solver's life: a detail field is created before the
  // first layout, and the frame binds its stages again when capacity grows.
  const unit=this.stageTaps?.unit??f.hanging.createTaps("Narrow-band FLIP stage unit taps");
  this.stageTaps={builder:f.hanging,unit,group:f.hanging.bind({extended:f.velocityScratch,negative:f.negativeScratch,coarse:this.coarseVelocity},unit)};
  for(const [name,velocity,negative,output] of [["bootstrap",f.velocityScratch,f.negativeScratch,f.departure],["advectParticles",f.velocityScratch,f.negativeScratch,f.departure],["seed",f.departure,f.negativeDeparture,f.velocityScratch],["transfer",f.departure,f.negativeDeparture,f.velocity],["snapshot",f.velocityScratch,f.negativeScratch,f.departure],["surface",f.velocity,f.negative,f.departure],["redistance",f.velocity,f.negative,f.departure],["update",f.velocityScratch,f.negativeScratch,f.departure]] as const){
   this.cacheGroups[name]=this.cache.bind({extended:velocity,negative,coarseExtended:this.coarseVelocity});
   this.groups[name]=[0,1].map(parity=>uniformDetailGroup(this.device,{layout:this.resources,entries:[
    {binding:0,resource:{buffer:this.particles[parity]!}},{binding:1,resource:{buffer:this.particles[1-parity]!}},
    {binding:2,resource:{buffer:this.bins}},{binding:3,resource:{buffer:this.next}},{binding:4,resource:{buffer:this.state}},
    {binding:5,resource:{buffer:negative}},{binding:6,resource:{buffer:this.params}},
    {binding:7,resource:name==="redistance"?f.phiScratch:f.phi},{binding:8,resource:velocity},{binding:9,resource:output},
    {binding:10,resource:name==="advectParticles"?f.coarseExtended:this.coarseVelocity},
    {binding:11,resource:name==="snapshot"||name==="update"?unit:f.unitExtended},{binding:12,resource:this.surfaceSource.vertexPhi},{binding:13,resource:this.surfaceSource.openFraction},{binding:14,resource:name==="redistance"?f.phi:f.phiScratch},
    {binding:15,resource:f.phase},
    ...(this.sourceParams?[{binding:16,resource:{buffer:this.sourceParams,size:UNIFORM_PARAMS_BYTES}}]:[]),
   ]})) as [UniformDetailGroup,UniformDetailGroup];
  }
 }
 async initialize():Promise<void>{
  await Promise.all([this.cache.initialize(),this.order.initialize()]);
  // One particle per lane: the old fixed 65,536-lane grid made each lane
  // trace many unrelated particles in series. The GPU receipt is current
  // even during bootstrap or source growth; no CPU count/readback is used.
  const moveModule=this.device.createShaderModule({label:"Narrow-band FLIP live work",code:/* wgsl */`
@group(0) @binding(0) var<storage,read> state:array<u32>;
@group(0) @binding(1) var<storage,read_write> work:array<u32>;
@compute @workgroup_size(1) fn main(){
 let groups=(min(state[0],${this.capacity}u)+63u)/64u;
 work[0]=min(groups,65535u);work[1]=(groups+65534u)/65535u;work[2]=1u;
}`});
  this.moveDispatchPipeline=await this.device.createComputePipelineAsync({layout:"auto",compute:{module:moveModule,entryPoint:"main"}});
  this.moveDispatchGroup=this.device.createBindGroup({layout:this.moveDispatchPipeline.getBindGroupLayout(0),entries:[this.state,this.moveDispatch].map((buffer,binding)=>({binding,resource:{buffer}}))});
  this.refinementLayout=this.device.createBindGroupLayout({entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
  ]});
  const refine=this.device.createShaderModule({label:"FLIP surface refinement prediction",code:/* wgsl */`
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(0) @binding(0) var<storage,read> samples:array<Particle>;
@group(0) @binding(1) var<storage,read> count:array<u32>;
@group(0) @binding(2) var<storage,read_write> joins:array<atomic<u32>>;
struct Params{dimsDt:vec4f,h:vec4f}
@group(0) @binding(3) var<uniform> p:Params;
@group(0) @binding(4) var<storage,read> bins:array<u32>;
struct Box{low:vec3u,high:vec3u}
fn sweptSurface(sample:Particle)->Box{
 let dims=vec3u(p.dimsDt.xyz);let tiles=dims/4u;
 let start=sample.position.xyz;let motion=select(sample.before.xyz,sample.velocity.xyz,sample.before.w>0.5);
 let travel=p.dimsDt.w*motion/p.h.xyz;
 let c=vec3u(clamp(floor(start),vec3f(0),vec3f(dims)-1.0));
 let bank=2u*dims.x*dims.y*dims.z+2u*tiles.x*tiles.y*tiles.z+(dims.x+1u)*(dims.y+1u)*(dims.z+1u);
 // The nearest-crossing bank holds a tile only within two tiles of a crossing (its reach bit).
 let t=c/4u;let reach=bins[bank+dims.x*dims.y*dims.z+4u+2u*tiles.x*tiles.y*tiles.z+t.x+tiles.x*(t.y+tiles.y*t.z)];
 var nearest=0xffffffffu;if((reach&1u)!=0u){nearest=bins[bank+c.x+dims.x*(c.y+dims.y*c.z)];}
 var low=start-${NARROW_BAND_SURFACE_RADIUS};var high=start+${NARROW_BAND_SURFACE_RADIUS};
 // Interior samples request the actual crossing cell, not their 4h collar.
 // Escaped samples still request their own swept support for reconstruction.
 if(sample.velocity.w<=0.0&&nearest!=0xffffffffu){
  low=vec3f(vec3u(nearest%dims.x,(nearest/dims.x)%dims.y,nearest/(dims.x*dims.y)));high=low+1.0;
 }
 let lo=vec3u(clamp(floor((low+min(travel,vec3f(0))-p.h.w)/4.0),vec3f(0),vec3f(tiles)-1.0));
 let hi=vec3u(clamp(floor((high+max(travel,vec3f(0))+p.h.w-1e-5)/4.0),vec3f(0),vec3f(tiles)-1.0));
 return Box(lo,max(lo,hi));
}
@compute @workgroup_size(64) fn refine(@builtin(global_invocation_id) gid:vec3u){
 let tiles=vec3u(p.dimsDt.xyz)/4u;
 for(var i=gid.x;i<min(count[0],arrayLength(&samples));i+=65536u){
  let sample=samples[i];if(sample.velocity.w < -1.0||sample.before.w==1.0){continue;}
  let box=sweptSurface(sample);
  // The samples keep their cell order, so the next one is usually a
  // neighbor: one with a superset sweep can mark for both.
  if(i+1u<min(count[0],arrayLength(&samples))){
   let other=samples[i+1u];
   if(other.velocity.w>=-1.0&&other.before.w!=1.0){let b=sweptSurface(other);if(all(b.low<=box.low)&&all(b.high>=box.high)){continue;}}
  }
  for(var z=box.low.z;z<=box.high.z;z++){for(var y=box.low.y;y<=box.high.y;y++){for(var x=box.low.x;x<=box.high.x;x++){
   let tile=x+tiles.x*(y+tiles.y*z);atomicOr(&joins[tile/32u],1u<<(tile%32u));
  }}}
 }
}`});
  const refineErrors=(await refine.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(refineErrors.length)throw new Error(refineErrors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  this.refinementPipeline=await this.device.createComputePipelineAsync({layout:this.device.createPipelineLayout({bindGroupLayouts:[this.refinementLayout]}),compute:{module:refine,entryPoint:"refine"}}).catch(e=>{throw new Error(`Refinement pipeline: ${e.message} ${e.reason}`,{cause:e});});
  const sourceCode=(unitTaps:boolean)=>uniformMixedCertifiedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
override nbCoarseOnly:bool=false;
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(1) @binding(0) var<storage,read_write> source:array<Particle>;
@group(1) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(1) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
@group(1) @binding(3) var<storage,read_write> links:array<u32>;
@group(1) @binding(4) var<storage,read_write> state:array<atomic<u32>>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params{hDt:vec4f,settings:vec4f}
@group(1) @binding(6) var<uniform> params:Params;
@group(1) @binding(7) var phi:texture_3d<f32>;
@group(1) @binding(8) var velocity:texture_3d<f32>;
@group(1) @binding(9) var output:texture_storage_3d<rgba32float,write>;
@group(1) @binding(10) var coarseVelocity:texture_3d<f32>;
@group(1) @binding(11) var unitVelocity:texture_3d<f32>;
@group(1) @binding(12) var surfacePhi:texture_storage_3d<r32float,write>;
@group(1) @binding(13) var surfaceOpen:texture_storage_3d<r32float,write>;
@group(1) @binding(14) var outputPhi:texture_storage_3d<r32float,write>;
@group(1) @binding(15) var phase:texture_3d<f32>;
// The spatial order leaves the samples sorted by cell, a tile's 64 cells
// together and x fastest inside it. A cell's samples are one contiguous run
// from its start, and the cells of an x row inside a tile follow one another,
// so a gather reads memory in order and follows no list. The bins hold each
// cell's count at its order; the head of the link arena holds its start.
const NB_CELLS:u32=UM_D.x*UM_D.y*UM_D.z;
fn nbOrder(c:vec3u)->u32{let t=c/4u;let l=c%4u;return 64u*umTileAt(t)+l.x+4u*l.y+16u*l.z;}
// The samples [x,y) of cells a to b, two cells of one x row of one tile.
fn nbRun(a:vec3i,b:vec3i)->vec2u{let last=nbOrder(vec3u(b));return vec2u(links[nbOrder(vec3u(a))],links[last]+atomicLoad(&bins[last]));}
fn nbCellRun(c:vec3i)->vec2u{return nbRun(c,c);}
// Compact read-mostly neighbor data shares the link arena in disjoint banks.
// This stays within the ten-storage-buffer limit, including solid bindings.
fn nbPosition(i:u32)->vec3f{let a=NB_CELLS+4u*i;return bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));}
fn nbMotion(i:u32)->vec4f{let a=NB_CELLS+4u*(arrayLength(&particles)+i);return bitcast<vec4f>(vec4u(links[a],links[a+1u],links[a+2u],links[a+3u]));}
fn nbStoreMotion(i:u32,p:vec4f){let a=NB_CELLS+4u*(arrayLength(&particles)+i);let v=bitcast<vec4u>(p);links[a]=v.x;links[a+1u]=v.y;links[a+2u]=v.z;links[a+3u]=v.w;}
// A tile bank no kernel uses any longer keeps its place before the others.
const NB_COVERAGE:u32=2u*NB_CELLS;
const NB_SURFACE_TILES:u32=NB_COVERAGE+UM_T.x*UM_T.y*UM_T.z;
${uniformMixedFaceAddressWGSL}
fn umLoadVertex(p:vec3u)->f32{return textureLoad(phi,vec3i(p),0).x;}
fn umLoadCorner(p:vec3u)->f32{return ${UNIFORM_DETAIL_4H_LOAD}textureLoad(phi,vec3i(p),0).x;}
${uniformMixedVertexSamplingSource("",true,undefined,"umLoadCorner")}
fn umLoadMixedFace(anchor:vec3i,axis:u32)->f32{
 if(anchor[axis]<0){return negative[umNegativeBoundaryIndex(vec3u(max(anchor,vec3i(0))),axis)];}
 return textureLoad(velocity,anchor,0)[axis];
}
fn umLoadCoarseFace(index:vec3i,axis:u32)->f32{return textureLoad(coarseVelocity,index+vec3i(1),0)[axis];}
${uniformMixedVelocitySamplingSource(false,true,"velocity",undefined,unitTaps?"unitVelocity":undefined)}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.solid?.coarse?.count)}
${this.sourceParams?uniformMixedSourceWGSL(16):""}
fn cellIndex(p:vec3i)->u32{return u32(p.x)+UM_D.x*(u32(p.y)+UM_D.y*u32(p.z));}
fn sampleVelocity(p:vec3f)->vec3f{
 if(nbCoarseOnly){return vec3f(umSampleVelocity4(p,0u),umSampleVelocity4(p,1u),umSampleVelocity4(p,2u));}
 return umSampleVelocity(p);
}
fn bandPhi(p:vec3f)->f32{
 var value=0.0;
 if(nbCoarseOnly){
  let q=clamp(p/4.0,vec3f(0),vec3f(UM_T));let base=min(vec3u(floor(q)),UM_T-1u);let f=q-vec3f(base);
  for(var k=0u;k<8u;k++){let bit=umCorner(k,2u);let w=select(vec3f(1)-f,f,bit!=vec3u(0));value+=w.x*w.y*w.z*umLoadCorner(4u*(base+bit));}
 }else{value=umSampleVertex(p);}
 return value/min(params.hDt.x,min(params.hDt.y,params.hDt.z));
}
${narrowBandMembershipWGSL}
${narrowBandRedistanceWGSL}
// The same liquid/retained-mass owners used by velocity extrapolation.
// Negative phi at a particle is insufficient: a sub-cell droplet may have
// no resolved cell at all, so the grid cannot evolve its momentum.
fn gridSupported(p:vec3f)->bool{
 let base=vec3i(floor(p-0.5));let fraction=fract(p-0.5);
 for(var k=0u;k<8u;k++){
  let bit=umCorner(k,2u);let w=select(vec3f(1)-fraction,fraction,bit!=vec3u(0));
  if(w.x*w.y*w.z<=1e-6){continue;}
  let c=clamp(base+vec3i(bit),vec3i(0),vec3i(UM_D)-1);let owner=umOwnerAt(c);
  if(textureLoad(phase,vec3i(umOrigin(owner)),0).x>0.5){return true;}
 }
 return false;
}
// Liquid a source makes this step, in h cells: the minimum of \`value\` and
// the source's own distance. The source writes the level set after advection;
// here it follows the particle union the same way, and decides which cells
// are seeded as at initialization.
fn nbSourcePhi(p:vec3f,value:f32)->f32{
 ${this.sourceParams?"let h=min(params.hDt.x,min(params.hDt.y,params.hDt.z));return umSourceuvSourcePhi(p,value*h)/h;":"return value;"}
}
// A sample made at the end of a step joins the tail of the epoch. The next
// move counts, marks and orders every sample, so nothing is filed for it here.
fn store(p:Particle){
 let i=atomicAdd(&state[1],1u);if(i>=arrayLength(&particles)){atomicAdd(&state[2],1u);return;}
 particles[i]=p;
}
// The tiles a sample's surface can reach.
fn markSurfaceTiles(position:vec3f){
 let lo=max(vec3i(floor((position-2.0)/4.0)),vec3i(0));let hi=min(vec3i(floor((position+2.0)/4.0)),vec3i(UM_T)-1);
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  atomicStore(&bins[NB_SURFACE_TILES+umTileAt(vec3u(vec3i(x,y,z)))],1u);
 }}}
}
${narrowBandTraceWGSL("sampleVelocity")}
@compute @workgroup_size(64) fn advectParticles(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+65535u*64u*gid.y;if(i>=min(atomicLoad(&state[0]),arrayLength(&source))){return;}
  var p=source[i];var q=p.position.xyz;
  // Solid collision, rather than grid ownership, limits particle motion.
  if(umCellOpen(vec3i(q))<0.5){source[i].position.x=-1.0;return;}
  let ballistic=p.before.w==1.0||!gridSupported(q);p.before.w=max(p.before.w,select(0.0,1.0,ballistic));
  // Uniform stores velocity half a step ahead of position. Ballistic drift
  // uses that midpoint velocity; update supplies the next full gravity kick.
  var speed=p.velocity.xyz/params.hDt.xyz;
  if(!ballistic){speed=sampleVelocity(q)/params.hDt.xyz;}
  let steps=nbTraceSteps(speed,params.hDt.w);
  // Trajectory refinement leaves the global pressure timestep unchanged.
  if(steps>${NARROW_BAND_TRACE_LIMIT}u){atomicStore(&state[3],1u);source[i].position.x=-1.0;return;}
  let base=params.hDt.w/f32(steps);var units=1u;
  for(var s=0u;s<steps;s+=units){
   var end=q+base*p.velocity.xyz/params.hDt.xyz;
   if(ballistic){units=1u;}else{
    // The first subdivision starts where the speed was sampled.
    if(s>0u){speed=sampleVelocity(q)/params.hDt.xyz;}
    let span=nbTraceSpan(q,base,speed,steps-s);end=span.xyz;units=u32(span.w);
   }
   if(params.settings.y>0.5&&end.y>=f32(UM_D.y)){q=end;break;}
   let endpoint=clamp(end,vec3f(0.01),vec3f(UM_D)-0.01);
   let travel=endpoint-q;let walk=max(1u,u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))));
   if(walk>256u){atomicStore(&state[3],1u);break;}
   let start=q;var hit=false;
   for(var j=1u;j<=walk;j++){
    let point=mix(start,endpoint,f32(j)/f32(walk));
    if(umCellOpen(vec3i(point))<0.5){p.velocity=vec4f(0);hit=true;break;}q=point;
   }
   for(var axis=0u;axis<3u;axis++){if(endpoint[axis]!=end[axis]){p.velocity[axis]=0.0;}}
   if(hit){break;}
  }
  if(any(q<vec3f(0))||any(q>=vec3f(UM_D))){source[i].position.x=-1.0;return;}
  // The sample moves where it is stored: the spatial order that follows
  // reads every slot, drops the ones marked here (a negative x) and takes
  // each cell's count from the bins. Nothing gathers before it.
  // A trajectory can cross from h into a coarse owner during this step.
  p.position=vec4f(q,p.position.w);source[i]=p;
  atomicAdd(&bins[nbOrder(vec3u(q))],1u);markSurfaceTiles(q);
}
// End-of-step sampling uses the reconstructed surface and projected velocity.
// Protect the outer h; retire only the deep interior and crowded overlap.
fn nbKept(i:u32)->bool{
 // The depth update just stored, of this position in this band.
 let depth=source[i].velocity.w;let order=nbOrder(vec3u(source[i].position.xyz));
 // Seeding tops a cell up to its count: the samples it keeps and the crowded ones it sheds.
 if(depth<-4.0){atomicSub(&bins[order],1u);return false;}
 // Preserve surface samples even when compressed. Thin structures cannot
 // survive an arbitrary per-cell cap. Only deep overlap samples retire: the
 // source is still in cell order, so a sample's place in its run is its rank.
 return depth>=-1.0||i-links[order]<16u;
}
// Each lane compacts one contiguous block of the ordered source, so the
// survivors stay beside their neighbors, and reserves its slots 32 samples
// at a time instead of contending for the cursor once a sample.
@compute @workgroup_size(64) fn resample(@builtin(global_invocation_id) gid:vec3u){
 let n=min(atomicLoad(&state[0]),arrayLength(&source));let block=(n+65535u)/65536u;
 let end=min(n,(gid.x+1u)*block);
 for(var first=gid.x*block;first<end;first+=32u){
  let size=min(32u,end-first);var keep=0u;
  for(var k=0u;k<size;k++){if(nbKept(first+k)){keep|=1u<<k;}}
  var next=atomicAdd(&state[1],countOneBits(keep));
  for(var k=0u;k<size;k++){if((keep&(1u<<k))!=0u){particles[next]=source[first+k];next++;}}
 }
}
// Fit the outer lattice to the sphere reconstruction's nodal interface.
// Keep tangential quarter-cell positions: their known offset determines the
// seed depth and preserves a resting plane without any volume correction.
fn seedShell(lattice:vec3f)->vec4f{
 var q=lattice;var d=particleDepth(q);
 for(var i=0u;i<4u;i++){
  let rise=d+NB_SEED_DEPTH;if(rise<=1e-3){break;}
  let g=vec3f(particleDepth(q+vec3f(0.5,0,0))-particleDepth(q-vec3f(0.5,0,0)),particleDepth(q+vec3f(0,0.5,0))-particleDepth(q-vec3f(0,0.5,0)),particleDepth(q+vec3f(0,0,0.5))-particleDepth(q-vec3f(0,0,0.5)));
  let g2=dot(g,g);if(g2<1e-8){break;}
  let next=clamp(q-min(rise,0.5)*g*inverseSqrt(g2),vec3f(0.01),vec3f(UM_D)-0.01);
  let depth=particleDepth(next);if(depth>=d){break;}
  q=next;d=depth;
 }
 return vec4f(q,d);
}
fn seedCell(c:vec3u,initial:bool){
 // No tile past the reach is within six cells of the surface.
 if((nbBandReach(c)&1u)==0u){return;}
 let d=particleDepth(vec3f(c)+0.5);if(d>0.9||d < -4.5){return;}
 if(umCellOpen(vec3i(c))<0.5){return;}
 let count=atomicLoad(&bins[nbOrder(c)]);if(count>=8u){return;}
 for(var k=count;k<8u;k++){
  let q=vec3f(c)+0.25+0.5*vec3f(umCorner(k,2u));let distance=particleDepth(q);
  if(distance>0.0||distance < -4.0){continue;}
  // Paper Section 3.3: replenish only the inner [-R,-h] band. The outer
  // layer keeps its transported samples instead of being recreated; a change
  // of ownership or an uncovered tile cannot bootstrap one from the bulk
  // field. Section 3.2: liquid made from a description gets its whole band,
  // at initialization and where a source has just made it.
  if(!initial&&distance>-1.0&&nbSourcePhi(q,1.0)>=0.0){continue;}
  if(distance<=-NB_SURFACE_RADIUS){let v=sampleVelocity(q);store(Particle(vec4f(q,1),vec4f(v,distance),vec4f(v,0)));continue;}
  let shell=seedShell(q);let p=shell.xyz;
  if(shell.w>0.0||umCellOpen(vec3i(p))<0.5){continue;}
  // A shell sample can land in another cell, whose own seeding decision
  // does not see it: no count changes here, whatever the dispatch order.
  let v=sampleVelocity(p);store(Particle(vec4f(p,1),vec4f(v,particleDepth(p)),vec4f(v,0)));
 }
}
// Seed the geometric particle band, including its overlap on 4h owners: one
// workgroup for each tile of the band search's list, the only ones seedCell
// passes.
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+group.x]);
 seedCell(4u*vec3u(tile%UM_T.x,(tile/UM_T.x)%UM_T.y,tile/(UM_T.x*UM_T.y))+vec3u(lane%4u,(lane/4u)%4u,lane/16u),false);
}

@compute @workgroup_size(64) fn seedInitial(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<UM_D.x*UM_D.y*UM_D.z;i+=65536u){seedCell(nbCell(i),true);}
}
fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
// Also the receipt's census of the samples against the surface they made.
@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id) gid:vec3u){
 var outside=0.0;var counts=vec3u(0);
 if(gid.x==0u){atomicStore(&bins[NB_BAND+3u],0u);}
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  let q=particles[i].position.xyz;let supported=gridSupported(q);let d=bandPhi(q);outside=max(outside,d);
  counts+=vec3u(u32(d>0.5),u32(!supported),u32(particleDepth(q)<-4.0));
  particles[i].before.w=select(1.0,0.0,supported);
  nbStoreMotion(i,vec4f(particles[i].velocity.xyz,particles[i].before.w));
 }
 atomicMax(&state[5],bitcast<u32>(outside));
 for(var k=0u;k<3u;k++){if(counts[k]!=0u){atomicAdd(&state[6u+k],counts[k]);}}
}
fn transferFace(face:UMFace)->f32{
 let original=umLoadMixedFace(face.anchor,face.axis);
 if(face.width==0u||face.anchor[face.axis]<0||face.neighbor.width==0u){return original;}
 let q=umFaceCenter(face);let depth=particleDepth(q);let blend=select(0.0,1.0,depth>=-2.0);
 if(blend==0.0||depth>1.5){return original;}
 // Cells starting exactly at q+1.5 have zero kernel weight throughout.
 let width=1.0;let lo=max(vec3i(floor(q-1.5*width)),vec3i(0));let hi=min(vec3i(ceil(q+1.5*width))-1,vec3i(UM_D)-1);
 var momentum=0.0;var total=0.0;
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let run=nbCellRun(vec3i(x,y,z));
  for(var i=run.x;i<run.y;i++){
   let motion=nbMotion(i);if(motion.w==1.0){continue;}
   let r=(q-nbPosition(i))/width;let w=weight(r.x)*weight(r.y)*weight(r.z);
   total+=w;momentum+=w*motion[face.axis];
  }
 }}}
 if(total<1e-5){return original;}return mix(original,momentum/total,blend);
}
${narrowBandFineTransferWGSL}
// The 4h owners, one a lane. An owner whose whole stencil is 4h has one
// patch a face, and a face centred in a tile past the reach lies more than
// two cells from the surface: it keeps its velocity without a gather. The
// rest are listed for transferSeams. Negative walls are untouched: their
// boundary condition is Uniform's.
@compute @workgroup_size(64) fn transferOwners(@builtin(global_invocation_id) gid:vec3u){
 let owner=umAllOwner(vec3u(umCounts.x*64u+gid.x,0,0));if(owner.width==0u){return;}
 let origin=umOrigin(owner);var far=umTileMaximumWidth(owner.tile)==4u&&umTileMinimumWidth(owner.tile)==4u;
 for(var axis=0u;axis<3u;axis++){
  var c=origin+2u;c[axis]=min(origin[axis]+4u,UM_D[axis]-1u);far=far&&(nbBandReach(c)&1u)==0u;
 }
 if(!far){atomicStore(&bins[NB_BAND_SEAM+atomicAdd(&bins[NB_BAND+3u],1u)],owner.tile);return;}
 for(var axis=0u;axis<3u;axis++){
  var anchor=vec3i(origin);anchor[axis]+=3;let original=textureLoad(velocity,anchor,0);
  var value=vec4f(0,0,0,original.w);value[axis]=original[axis];textureStore(output,anchor,value);
 }
}
// A listed 4h owner a workgroup, a lane for each patch of each positive face:
// one beside h cells has sixteen a face, each with its own gather.
var<workgroup> nbSeam:array<vec2f,64>;
@compute @workgroup_size(64) fn transferSeams(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=atomicLoad(&bins[NB_BAND_SEAM+group.x]);let owner=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);
 let part=lane/3u;let axis=lane%3u;var anchor=vec3i(0);var result=vec2f(0);
 if(lane<48u){let face=umFace(owner,axis,1,part);if(face.width!=0u){anchor=face.anchor;result=vec2f(transferFace(face),1);}}
 nbSeam[lane]=result;workgroupBarrier();
 if(result.y==0.0){return;}
 // One writer a texel: the lowest axis with a patch at this anchor.
 let offset=anchor-vec3i(umOrigin(owner));var value=vec4f(0);var writer=true;
 for(var other=0u;other<3u;other++){
  if(other==axis){value[other]=result.x;continue;}
  let face=umPositiveFaceAtAnchor(owner,other,anchor);if(face.width==0u){continue;}
  if(other<axis){writer=false;}
  let u=(other+1u)%3u;let v=(other+2u)%3u;let side=4u/face.width;
  value[other]=nbSeam[3u*(u32(offset[u])/face.width+side*(u32(offset[v])/face.width))+other].x;
 }
 if(writer){value.w=textureLoad(velocity,anchor,0).w;textureStore(output,anchor,value);}
}

@compute @workgroup_size(64) fn snapshot(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  let mode=particles[i].before.w;var v=vec3f(0);if(mode!=1.0){v=sampleVelocity(particles[i].position.xyz);}
  particles[i].before=vec4f(v,mode);
 }
}
@compute @workgroup_size(64) fn update(@builtin(global_invocation_id) gid:vec3u){
 if(gid.x==0u){let n=min(atomicLoad(&state[1]),arrayLength(&particles));atomicStore(&state[0],n);atomicStore(&state[1],n);}
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  let mode=particles[i].before.w;
  var pic=vec3f(0);var v=vec3f(0);
  if(mode!=1.0){pic=sampleVelocity(particles[i].position.xyz);let flip=particles[i].velocity.xyz+pic-particles[i].before.xyz;v=mix(pic,flip,params.settings.x);}
  if(mode==1.0){
   v=particles[i].velocity.xyz+vec3f(0,params.settings.z*params.hDt.w,0);
   let q=particles[i].position.xyz;
   for(var axis=0u;axis<3u;axis++){
    if((q[axis]<=0.011&&v[axis]<0.0)||(q[axis]>=f32(UM_D[axis])-0.011&&v[axis]>0.0&&!(axis==1u&&params.settings.y>0.5))){v[axis]=0.0;}
   }
  }
  if(!all(abs(v)<vec3f(1e10))){atomicStore(&state[3],2u);continue;}
  particles[i].velocity=vec4f(v,particleDepth(particles[i].position.xyz));
  // Retain accepted grid motion for next frame's residency prediction.
  // FLIP residual velocity transfers momentum, but does not trace positions.
  particles[i].before=vec4f(pic,select(0.0,1.0,mode==1.0));
 }
}
// A lane reduces its own samples and touches the shared receipt once.
@compute @workgroup_size(64) fn diagnoseBefore(@builtin(global_invocation_id) gid:vec3u){
 var outside=0.0;
 for(var i=gid.x;i<min(atomicLoad(&state[1]),arrayLength(&particles));i+=65536u){
  outside=max(outside,bandPhi(particles[i].position.xyz));
 }
 atomicMax(&state[4],bitcast<u32>(outside));
}
@compute @workgroup_size(64) fn commitCount(@builtin(global_invocation_id) gid:vec3u){if(gid.x==0u){let n=min(atomicLoad(&state[1]),arrayLength(&particles));atomicStore(&state[0],n);atomicStore(&state[1],n);}}
${narrowBandCoarseTransferWGSL}
${narrowBandSurfaceWGSL}
`,[]);
  // Advection borrows the exact extended-field unit taps Uniform already
  // prepared; snapshot, update and seed fill their own from the field they
  // sample (stageTaps). The other stages read their own fresh 4h cache and
  // resolve fine seam taps in place.
  const modules=[false,true].map(unitTaps=>uniformDetailModule(this.device,{label:`Uniform narrow-band FLIP ${unitTaps?"extended":"stage"}`,code:uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(sourceCode(unitTaps),["depthSeeds"],"atomicLoad(&bins[NB_BAND])"),["seed"],"atomicLoad(&bins[NB_BAND+1u])"),["depthNearest"],"atomicLoad(&bins[NB_BAND+1u])"),["buildDistance"],"atomicLoad(&bins[NB_BAND+2u])"),["transfer"],"8u*umCounts.x"),["transferOwners"],"(umCounts.y+63u)/64u"),["transferSeams"],"atomicLoad(&bins[NB_BAND+3u])"),["couple"],"(umCounts.y+3u)/4u"),["coupleFine"],"umCounts.x"),["redistanceFine"],"umCounts.x")}));
  for(const module of modules){
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  }
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[])]});
  await Promise.all([false,true].flatMap(coarse=>["advectParticles","seed","seedInitial","classify","transfer","transferOwners","transferSeams","transferCoarse","snapshot","update","surface","couple","coupleFine","commitCount","resample","diagnoseBefore","buildDistance","depthTiles","depthSeeds","depthLists","depthNearest","redistanceFine","redistanceCoarse"].map(async entryPoint=>{
   if((entryPoint==="transferCoarse"&&!coarse)||(entryPoint==="surface"&&!this.coarseParticles))return;
   const module=modules[["advectParticles","snapshot","update","seed"].includes(entryPoint)?1:0]!;
   const p=await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{label:`Narrow-band FLIP ${entryPoint} ${coarse?"4h":"mixed"}`,layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:(entryPoint==="couple"||entryPoint==="transfer")?3:0,nbCoarseOnly:+coarse,...s}}})).catch(e=>{throw new Error(`${entryPoint} ${coarse}: ${e.message} ${e.reason}`,{cause:e});});
   this.pipelines.set(`${entryPoint}:${coarse}`,p);
  })));

 }
 private dispatch(encoder:GPUCommandEncoder,entry:string,groupName?:string):void{
  const pass=encoder.beginComputePass({label:`Narrow-band FLIP ${entry}`});
  pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.groups[groupName??(entry==="resample"?"update":entry==="diagnoseBefore"||entry==="buildDistance"||entry.startsWith("depth")?"surface":entry.startsWith("transfer")||entry==="classify"?"transfer":entry==="couple"||entry==="coupleFine"||entry==="commitCount"?"surface":entry)]![this.parity]!.group);
  if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  const selected=this.pipelines.get(`${entry}:${this.coarseOnly}`)!;
  const pipeline=this.solid?.select(selected)??selected;
  if(entry==="couple"||entry==="transfer")this.ownership.dispatchAllCounted(pass,pipeline);
  else {
   pass.setPipeline(uniformDetailPick(pipeline));
   if(entry==="transferCoarse")pass.dispatchWorkgroups(256,Math.ceil(this.ownership.capacity.tiles/256),3);
   else if(entry==="advectParticles")pass.dispatchWorkgroupsIndirect(this.moveDispatch,0);
   else pass.dispatchWorkgroups(1024);
  }
  pass.end();
 }
 move(encoder:GPUCommandEncoder,dt:number,openTop:boolean,gravity=0):void{
  this.bandCurrent=false;
  this.device.queue.writeBuffer(this.params,0,new Float32Array([...this.ownership.capacity.lattice.cellSize_m,dt,0.95,+openTop,gravity,0]));
  const cells=this.ownership.capacity.lattice.dimensions.reduce((n,v)=>n*v,1);
  encoder.clearBuffer(this.state,16,32);
  this.bootstrap(encoder);
  // The samples move in place, counting themselves into clear bins and
  // marking the tiles they reach; the order packs them into the other buffer.
  encoder.clearBuffer(this.state,4,8);encoder.clearBuffer(this.bins,0,cells*8);
  encoder.clearBuffer(this.bins,cells*8+this.ownership.capacity.tiles*4,this.ownership.capacity.tiles*4);
  const work=encoder.beginComputePass({label:"Narrow-band FLIP live particle work"});work.setPipeline(this.moveDispatchPipeline);work.setBindGroup(0,this.moveDispatchGroup);work.dispatchWorkgroups(1);work.end();
  this.dispatch(encoder,"advectParticles");
  this.order.encode(encoder,1-this.parity);
  this.dispatch(encoder,"commitCount");
 }
 private bootstrap(encoder:GPUCommandEncoder):void{
  if(this.started)return;
  const cells=this.ownership.capacity.lattice.dimensions.reduce((n,v)=>n*v,1);
  encoder.clearBuffer(this.state,4,8);encoder.clearBuffer(this.bins,0,cells*8);
  encoder.clearBuffer(this.bins,cells*8+this.ownership.capacity.tiles*4,this.ownership.capacity.tiles*4);
  this.measureBand(encoder);this.dispatch(encoder,"buildDistance");this.cache.encode(encoder,this.cacheGroups.bootstrap!);
  this.dispatch(encoder,"seedInitial","bootstrap");this.dispatch(encoder,"commitCount");
  this.parity=1-this.parity;this.started=true;
 }
 /** Called only for enabled automatic refinement. Explicit Requested/Full
  * layouts keep their authored semantics; particle lifetime is independent. */
 refine(encoder:GPUCommandEncoder,target:{buffer:GPUBuffer;wordOffset:number},dt:number,padding=1):void{
  const lattice=this.ownership.capacity.lattice;
  // The first frame also needs a sweep; bootstrap from the head's extension
  // before choosing its grid layout, instead of after the first census.
  if(!this.started){this.device.queue.writeBuffer(this.params,0,new Float32Array([...lattice.cellSize_m,dt,0.95,0,0,0]));this.bootstrap(encoder);}
  this.device.queue.writeBuffer(this.refinementParams,0,new Float32Array([...lattice.dimensions,dt,...lattice.cellSize_m,padding]));
  const group=this.device.createBindGroup({layout:this.refinementLayout!,entries:[
   {binding:0,resource:{buffer:this.activeParticles}},{binding:1,resource:{buffer:this.state}},
   {binding:2,resource:{buffer:target.buffer,offset:target.wordOffset*4,size:4*Math.ceil(this.ownership.capacity.tiles/32)}},
   {binding:3,resource:{buffer:this.refinementParams}},
   {binding:4,resource:{buffer:this.bins}},
  ]});
  const pass=encoder.beginComputePass({label:"Narrow-band FLIP surface refinement"});pass.setPipeline(this.refinementPipeline!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1024);pass.end();
 }
 reconstruct(encoder:GPUCommandEncoder):void{this.bandCurrent=false;this.measureBand(encoder);this.dispatch(encoder,"diagnoseBefore");if(!this.ownership.coarseOnly)this.dispatch(encoder,"coupleFine");this.dispatch(encoder,"couple");}
 transfer(encoder:GPUCommandEncoder):void{
  // Each FLIP stage samples a different field. Rebuild the small 4h cache
  // after every grid write; keep Uniform's extended-field caches intact.
  this.cache.encode(encoder,this.cacheGroups.seed!);this.dispatch(encoder,"classify");if(this.coarseOnly)this.dispatch(encoder,"transferCoarse");else{this.dispatch(encoder,"transfer");this.dispatch(encoder,"transferOwners");this.dispatch(encoder,"transferSeams");}}
 snapshot(encoder:GPUCommandEncoder):void{this.cache.encode(encoder,this.cacheGroups.snapshot!);this.encodeStageTaps(encoder);this.dispatch(encoder,"snapshot");}
 private encodeStageTaps(encoder:GPUCommandEncoder):void{if(this.stageTaps&&this.ownership.capacity.fineTiles>0)this.stageTaps.builder.encode(encoder,this.stageTaps.group);}
 /** The crossing set is unchanged by this redistance, so its search also
  * serves end-step particle membership; forces and pressure never edit phi. */
 redistance(encoder:GPUCommandEncoder):void{
  this.measureBand(encoder,"redistance");this.dispatch(encoder,"buildDistance","redistance");
  if(!this.ownership.coarseOnly)this.dispatch(encoder,"redistanceFine","redistance");
  this.dispatch(encoder,"redistanceCoarse","redistance");this.bandCurrent=true;
 }
 private measureBand(encoder:GPUCommandEncoder,group?:string):void{
  // The search's tile banks follow the cell and vertex banks: counts, masks and reach start clear.
  const d=this.ownership.capacity.lattice.dimensions,cells=d.reduce((n,v)=>n*v,1),tiles=this.ownership.capacity.tiles;
  encoder.clearBuffer(this.bins,(3*cells+d.reduce((n,v)=>n*(v+1),1)+2*tiles)*4,(4+3*tiles)*4);
  for(const entry of ["depthTiles","depthSeeds","depthLists","depthNearest"])this.dispatch(encoder,entry,group);
 }
 update(encoder:GPUCommandEncoder):void{
  if(!this.bandCurrent){this.measureBand(encoder);this.dispatch(encoder,"buildDistance");}this.bandCurrent=false;this.cache.encode(encoder,this.cacheGroups.update!);this.encodeStageTaps(encoder);this.dispatch(encoder,"update");
  // The ordered epoch compacts into the other buffer; the bins keep its counts for the seeding.
  this.parity=1-this.parity;
  encoder.clearBuffer(this.state,4,4);
  this.dispatch(encoder,"resample");this.dispatch(encoder,"seed","update");
  this.dispatch(encoder,"commitCount");
  if(this.coarseParticles)this.dispatch(encoder,"surface");this.parity=1-this.parity;
 }
 noteReceipt(words:Uint32Array):void{
  this.count=words[1]!;this.reseedClipped=words[2]!;
  const distances=new Float32Array(words.buffer,words.byteOffset,words.length);
  this.diagnostics={beforeMaxOutside:distances[4]!,afterMaxOutside:distances[5]!,outsideSurface:words[6]!,unsupported:words[7]!,deepInterior:words[8]!};
  // Persistent samples own the budget first. At capacity only optional
  // reseeding is deferred; never destroy a surface to make room for it.
  if(words[3])throw new Error(`Narrow-band FLIP ${words[3]===1?`trajectory exceeded ${NARROW_BAND_TRACE_LIMIT} cells`:"nonfinite velocity"}`);
 }
 destroy():void{this.order.destroy();this.moveDispatch.destroy();this.surfaceSource.vertexPhi.destroy();this.surfaceSource.openFraction.destroy();this.coarseVelocity.destroy();this.stageTaps?.unit.destroy();this.refinementParams.destroy();for(const b of [...this.particles,this.state,this.bins,this.next,this.params])b.destroy();}
}
