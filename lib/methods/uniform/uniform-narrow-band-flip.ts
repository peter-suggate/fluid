import { narrowBandActivityWGSL, NARROW_BAND_ACTIVITY_WORDS_PER_TILE } from "./uniform-narrow-band-activity.wgsl";
import { narrowBandRedistanceWGSL } from "./uniform-narrow-band-redistance.wgsl";
import { narrowBandMembershipWGSL } from "./uniform-narrow-band-membership.wgsl";
import { NARROW_BAND_TRACE_LIMIT, narrowBandTraceWGSL } from "./uniform-narrow-band-advection.wgsl";
import { uniformMixedSourceWGSL, UNIFORM_PARAMS_BYTES } from "./uniform-mixed-source.wgsl";
import { UniformNarrowBandOrder } from "./uniform-narrow-band-order";
import { NARROW_BAND_SURFACE_RADIUS, narrowBandSurfaceWGSL, narrowBandCoarseTransferWGSL, narrowBandFineTransferSource, narrowBandTransferPropertiesSource } from "./uniform-narrow-band-surface.wgsl";
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
 get allocatedBytes():number{return this.baseAllocatedBytes+this.order.allocatedBytes+this.particleDispatch.size;}
 private readonly order:UniformNarrowBandOrder;
 private readonly particleDispatch:GPUBuffer;
 private particleDispatchPipeline!:GPUComputePipeline;
 private particleDispatchGroup!:GPUBindGroup;
 readonly surfaceSource:{contourVertexPhi:true;vertexPhi:GPUTexture;openFraction:GPUTexture;cellSize_m:readonly [number,number,number]};
 get coarseOnly():boolean{return this.ownership.capacity.fineTiles===0;}
 count=0;
 reseedClipped=0;
 /** Positive distances in h cells, before union and on the accepted surface.
  * Unresolved samples are reported separately from the resolved liquid band. */
 diagnostics={beforeMaxOutside:NaN,afterMaxOutside:0,outsideSurface:0,unsupported:0,deepInterior:0};
 /** The separation before the union is a diagnostic alone, a sweep of every
  * sample that nothing in the step reads: a step measures it only for a
  * reader that set this before stepping. Unmeasured, it reports NaN. */
 measureTraces=false;
 /** Completed frame samples, exposed for transfer/conservation verification.
  * Resampling preserves the survivors' cell order, but seeding appends an
  * unsorted tail. The complete buffer is not yet a cell-sorted epoch. */
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
 private seedAll=false;
 private adaptive=false;
 setAdaptive(enabled:boolean):void{enabled=enabled&&this.activitySupported;if(this.adaptive&&!enabled)this.seedAll=true;this.adaptive=enabled;}
 get adaptiveSurface():boolean{return this.adaptive;}
 private get activityWord():number{const d=this.ownership.capacity.lattice.dimensions;return 3*d.reduce((a,b)=>a*b,1)+d.reduce((a,b)=>a*(b+1),1)+9*this.ownership.capacity.tiles+4;}
 private refinementPipeline?:GPUComputePipeline;
 private refinementLayout?:GPUBindGroupLayout;
 private readonly refinementGroups:({particles:GPUBuffer;target:GPUBuffer;wordOffset:number;size:number;group:GPUBindGroup}|undefined)[]=[];
 private readonly refinementParams:GPUBuffer;
 private readonly cache:UniformMixedMomentumCache;
 private readonly coarseVelocity:GPUTexture;
 private cacheGroups:Record<string,UniformDetailGroup>={};
 private stageTaps?:{builder:UniformMixedHangingTaps;group:UniformDetailGroup;unit:GPUTexture};
 constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,readonly coarseParticles=false,private readonly sourceParams?:GPUBuffer,private readonly activitySupported=false){
  const cells=ownership.capacity.lattice.dimensions.reduce((n,v)=>n*v,1);
  const vertices=ownership.capacity.lattice.dimensions.reduce((n,v)=>n*(v+1),1);
  // The band is a surface, but what a scene's surface grows to is not known
  // here. Budget one sample per domain cell, with a two-million floor: twice
  // the previous allocation, still capped by full-domain seeding and the
  // device's per-buffer limit. Sources take priority if that budget fills.
  this.capacity=Math.min(cells*8,Math.max(2_097_152,cells),Math.floor(device.limits.maxStorageBufferBindingSize/48));
  const buffer=(label:string,size:number,uniform=false)=>device.createBuffer({label:`Narrow-band FLIP ${label}`,size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  this.particles=[buffer("particles A",this.capacity*48),buffer("particles B",this.capacity*48)];
  this.particleDispatch=device.createBuffer({label:"Narrow-band FLIP live particle dispatch",size:12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.INDIRECT});
  this.state=buffer("receipt",48);this.bins=buffer("cell counts, sort cursors, coverage, surface tiles, depth guard and band tiles",(3*cells+vertices)*4+ownership.capacity.tiles*(36+(activitySupported?4*NARROW_BAND_ACTIVITY_WORDS_PER_TILE:0))+16);this.next=buffer("cell starts and compact neighbor samples",cells*4+this.capacity*32);this.params=buffer("parameters",48,true);
  this.order=new UniformNarrowBandOrder(device,ownership.capacity.lattice.dimensions,this.particles,this.bins,this.next,this.state);
  this.refinementParams=buffer("refinement prediction",32,true);
  this.cache=new UniformMixedMomentumCache(device,ownership);
  const size=ownership.capacity.lattice.dimensions.map(n=>n/4+2);
  this.coarseVelocity=device.createTexture({label:"Narrow-band FLIP stage sampling cache",size,dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
  const dims=ownership.capacity.lattice.dimensions;
  const field=(label:string,size:readonly number[])=>device.createTexture({label,size:[...size],dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
  this.surfaceSource={contourVertexPhi:true,vertexPhi:field("Narrow-band particle render surface",coarseParticles?dims.map(n=>n+1):[1,1,1]),openFraction:field("Narrow-band render solid fractions",coarseParticles?dims:[1,1,1]),cellSize_m:dims.map((_,a)=>ownership.capacity.lattice.cellSize_m[a]!) as [number,number,number]};
  this.baseAllocatedBytes=this.capacity*132+(3*cells+vertices)*4+ownership.capacity.tiles*(32+(activitySupported?4*NARROW_BAND_ACTIVITY_WORDS_PER_TILE:0))+144+size.reduce((n,v)=>n*v,16)+(coarseParticles?4*(cells+dims.reduce((n,v)=>n*(v+1),1)):8);
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
  const subgroups=this.device.features.has("subgroups")&&(this.device.adapterInfo.subgroupMinSize??0)>=32;
  await Promise.all([this.cache.initialize(),this.order.initialize()]);
  // One particle per lane for tracing and sorted grid sampling. Refresh
  // from the current GPU receipt at each stage, including bootstrap and
  // source growth; no CPU count/readback is used.
  const particleModule=this.device.createShaderModule({label:"Narrow-band FLIP live work",code:/* wgsl */`
@group(0) @binding(0) var<storage,read> state:array<u32>;
@group(0) @binding(1) var<storage,read_write> work:array<u32>;
@compute @workgroup_size(1) fn main(){
 let groups=(min(state[0],${this.capacity}u)+63u)/64u;
 work[0]=min(groups,65535u);work[1]=(groups+65534u)/65535u;work[2]=1u;
}`});
  this.particleDispatchPipeline=await this.device.createComputePipelineAsync({layout:"auto",compute:{module:particleModule,entryPoint:"main"}});
  this.particleDispatchGroup=this.device.createBindGroup({layout:this.particleDispatchPipeline.getBindGroupLayout(0),entries:[this.state,this.particleDispatch].map((buffer,binding)=>({binding,resource:{buffer}}))});
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
  // Resampling preserves neighboring survivors; the seed tail is unsorted.
  // A superset sweep can mark for both, regardless of their order.
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
  const sourceCode=(unitTaps:boolean)=>uniformMixedCertifiedEntriesWGSL((this.device.features.has("subgroups")?"enable subgroups;\n":"")+uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
override nbCoarseOnly:bool=false;
const nbSparseBins:bool=${this.order.sparse};
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
@group(1) @binding(0) var<storage,read_write> source:array<Particle>;
@group(1) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(1) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
@group(1) @binding(3) var<storage,read_write> links:array<u32>;
@group(1) @binding(4) var<storage,read_write> state:array<atomic<u32>>;
@group(1) @binding(5) var<storage,read> negative:array<f32>;
struct Params{hDt:vec4f,settings:vec4f,activity:vec4f}
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
fn nbRun(a:vec3i,b:vec3i)->vec2u{${this.order.sparse?"if(atomicLoad(&bins[NB_COVERAGE+umTileAt(vec3u(a)/4u)])==0u){return vec2u(0);}":""}
 let last=nbOrder(vec3u(b));return vec2u(links[nbOrder(vec3u(a))],links[last]+atomicLoad(&bins[last]));}
fn nbCellRun(c:vec3i)->vec2u{return nbRun(c,c);}
// Compact read-mostly neighbor data shares the link arena in disjoint banks.
// This stays within the ten-storage-buffer limit, including solid bindings.
fn nbPosition(i:u32)->vec3f{let a=NB_CELLS+4u*i;return bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));}
fn nbMotion(i:u32)->vec4f{let a=NB_CELLS+4u*(arrayLength(&particles)+i);return bitcast<vec4f>(vec4u(links[a],links[a+1u],links[a+2u],links[a+3u]));}
fn nbStoreMotion(i:u32,p:vec4f){let a=NB_CELLS+4u*(arrayLength(&particles)+i);let v=bitcast<vec4u>(p);links[a]=v.x;links[a+1u]=v.y;links[a+2u]=v.z;links[a+3u]=v.w;}
// Live particle counts per tile drive sparse cell-bin clearing and ordering.
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
${narrowBandActivityWGSL}
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
// Gather/addition/velocity support reaches neighboring tiles. Erasure heat
// belongs to occupied tiles: dilating it erodes unseeded neighbors.
fn markSurfaceTiles(position:vec3f,heat:f32){
 if(nbAdaptive()){let tile=umTileAt(min(vec3u(position)/4u,UM_T-1u));atomicMax(&bins[NB_ACTIVITY_SURFACE_HEAT+tile],bitcast<u32>(max(heat,0.0)));}
 let lo=max(vec3i(floor((position-2.0)/4.0)),vec3i(0));let hi=min(vec3i(floor((position+2.0)/4.0)),vec3i(UM_T)-1);
 for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
  let tile=umTileAt(vec3u(vec3i(x,y,z)));
  atomicStore(&bins[NB_SURFACE_TILES+tile],1u);
  if(nbAdaptive()){atomicMax(&bins[NB_ACTIVITY_THETA+tile],bitcast<u32>(max(heat,0.0)));}
 }}}
}
// Sorted cell runs let one tile combine repeated particle support writes.
var<workgroup> nbMarkedMask:atomic<u32>;
var<workgroup> nbMarkedHeat:array<atomic<u32>,27>;
var<workgroup> nbOwnHeat:atomic<u32>;
var<workgroup> nbMarkedPopulation:u32;
@compute @workgroup_size(64) fn markParticleTiles(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let tile=group.x+65535u*group.y;if(tile>=NB_BAND_TILES){return;}
 let first=64u*tile;
 if(lane==0u){
  nbMarkedPopulation=0u;
  ${this.order.sparse?"if(atomicLoad(&bins[NB_COVERAGE+tile])!=0u)":""}{nbMarkedPopulation=links[first+63u]+atomicLoad(&bins[first+63u])-links[first];}
 }
 if(workgroupUniformLoad(&nbMarkedPopulation)==0u){return;}
 if(lane==0u){atomicStore(&nbMarkedMask,0u);atomicStore(&nbOwnHeat,0u);}
 if(lane<27u){atomicStore(&nbMarkedHeat[lane],0u);}workgroupBarrier();
 let local=umCorner(lane,4u);let coord=vec3i(umTileCoord(tile));let c=4u*vec3u(coord)+local;
 let low=max(coord+vec3i(local/2u)-1,vec3i(0));let high=min(coord+vec3i(local/2u),vec3i(UM_T)-1);
 let order=first+lane;let start=links[order];let stop=start+atomicLoad(&bins[order]);
 var heat=0.0;var live=false;
 for(var i=start;i<stop;i++){
  let a=NB_CELLS+4u*i;let q=bitcast<vec3f>(vec3u(links[a],links[a+1u],links[a+2u]));let h=max(0.0,bitcast<f32>(links[a+3u]));
  // Preserve the original floating-point boundary decisions exactly.
  let lo=max(vec3i(floor((q-2.0)/4.0)),vec3i(0));let hi=min(vec3i(floor((q+2.0)/4.0)),vec3i(UM_T)-1);
  if(any(lo!=low)||any(hi!=high)){markSurfaceTiles(q,h);continue;}
  live=true;heat=max(heat,h);
 }
 if(live){
  var mask=0u;
  for(var z=low.z;z<=high.z;z++){for(var y=low.y;y<=high.y;y++){for(var x=low.x;x<=high.x;x++){
   let o=vec3u(vec3i(x,y,z)-coord+1);let bit=o.x+3u*(o.y+3u*o.z);mask|=1u<<bit;
   if(nbAdaptive()){atomicMax(&nbMarkedHeat[bit],bitcast<u32>(heat));}
  }}}
  atomicOr(&nbMarkedMask,mask);if(nbAdaptive()){atomicMax(&nbOwnHeat,bitcast<u32>(heat));}
 }
 workgroupBarrier();
 if(lane==0u&&nbAdaptive()){atomicMax(&bins[NB_ACTIVITY_SURFACE_HEAT+tile],atomicLoad(&nbOwnHeat));}
 if(lane<27u&&((atomicLoad(&nbMarkedMask)>>lane)&1u)!=0u){
  let t=coord+vec3i(umCorner(lane,3u))-1;let markedTile=umTileAt(vec3u(t));
  atomicStore(&bins[NB_SURFACE_TILES+markedTile],1u);
  if(nbAdaptive()){atomicMax(&bins[NB_ACTIVITY_THETA+markedTile],atomicLoad(&nbMarkedHeat[lane]));}
 }
}
${narrowBandTraceWGSL("sampleVelocity")}
@compute @workgroup_size(64) fn advectParticles(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+65535u*64u*gid.y;if(i>=min(atomicLoad(&state[0]),arrayLength(&source))){return;}
  var p=source[i];var q=p.position.xyz;
  // Solid collision, rather than grid ownership, limits particle motion.
  if(umCellOpen(vec3i(q))<0.5){source[i].position.x=-1.0;return;}
  let ballistic=p.before.w==1.0||!gridSupported(q);p.before.w=max(p.before.w,select(0.0,1.0,ballistic));
  if(nbAdaptive()){
   // A coupled sample has the heat of the tile it is in: the census's, cooled
   // over the retirement time and spread one tile (activitySpread). Carried
   // with the liquid instead, it held every tile that liquid reached in that
   // time at h, whatever the census admitted there. A ballistic sample has
   // no tile's support to lose and keeps its own.
   p.position.w=max(nbTargetHeat(q),select(0.0,p.position.w,ballistic));
   if(p.position.w<=0.0){source[i].position.x=-1.0;return;}
  }
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
   // With no embedded solids the segment stays inside the convex container
   // after endpoint clamping. There are no voxel crossings to test; retaining
   // the RK3 endpoint also avoids accumulating rounded collision increments.
   if(!umSolidEnabled()){q=endpoint;}else{
   let travel=endpoint-q;let walk=max(1u,u32(ceil(2.0*max(abs(travel.x),max(abs(travel.y),abs(travel.z))))));
   if(walk>256u){atomicStore(&state[3],1u);break;}
   // A voxel face stops a particle as a container wall does: the crossing
   // axis ends at the face and loses its velocity, the others carry on.
   let stride=travel/f32(walk);
   for(var j=1u;j<=walk;j++){
    for(var axis=0u;axis<3u;axis++){
     var point=q;point[axis]+=stride[axis];
     if(umCellOpen(vec3i(point))<0.5){
      let cell=floor(q[axis]);point[axis]=select(cell+0.01,cell+0.99,stride[axis]>0.0);p.velocity[axis]=0.0;
     }
     q=point;
    }
   }
   }
   for(var axis=0u;axis<3u;axis++){if(endpoint[axis]!=end[axis]){p.velocity[axis]=0.0;}}
  }
  if(any(q<vec3f(0))||any(q>=vec3f(UM_D))){source[i].position.x=-1.0;return;}
  // The sample moves where it is stored: the spatial order that follows
  // reads every slot, drops the ones marked here (a negative x) and takes
  // each cell's count from the bins. Nothing gathers before it.
  // A trajectory can cross from h into a coarse owner during this step.
  p.position=vec4f(q,p.position.w);source[i]=p;
  atomicAdd(&bins[nbOrder(vec3u(q))],1u);${this.order.sparse?"atomicAdd(&bins[NB_COVERAGE+umTileAt(vec3u(q)/4u)],1u);":""}
}
// End-of-step sampling uses the reconstructed surface and projected velocity.
// Protect the outer h; retire only the deep interior and crowded overlap.
fn nbKept(i:u32)->bool{
 // The depth update just stored, of this position in this band.
 let depth=source[i].velocity.w;let order=nbOrder(vec3u(source[i].position.xyz));
 // Seeding tops a cell up to its count: the samples it keeps and the crowded ones it sheds.
 if(depth<-4.0||(nbAdaptive()&&source[i].position.w<=0.0)){atomicSub(&bins[order],1u);return false;}
 // Preserve surface samples even when compressed. Thin structures cannot
 // survive an arbitrary per-cell cap. Only deep overlap samples retire: the
 // source is still in cell order, so a sample's place in its run is its rank.
 return depth>=-1.0||i-links[order]<16u;
}
// Compact positions/motions expire after update. Borrow that part of links
// for per-particle ranks and per-workgroup counts/offsets; cell starts remain
// intact while nbKept uses the sorted rank to retain the overlap samples.
fn nbCompactCounts()->u32{return NB_CELLS+arrayLength(&source);}
fn nbCompactOffsets()->u32{return nbCompactCounts()+(arrayLength(&source)+63u)/64u;}
var<workgroup> nbCompactScan:array<u32,256>;
${subgroups?/* wgsl */`
// One subgroup owns a 64-particle block, two consecutive samples per lane.
// Its integer scan gives the same stable ranks without shared-memory rounds.
@compute @workgroup_size(32) fn resampleCount(@builtin(workgroup_id) group:vec3u,@builtin(subgroup_invocation_id) lane:u32){
 let job=group.x+65535u*group.y;let i=64u*job+2u*lane;
 let n=min(atomicLoad(&state[0]),arrayLength(&source));
 var a=0u;var b=0u;if(i<n){a=u32(nbKept(i));}if(i+1u<n){b=u32(nbKept(i+1u));}
 let rank=subgroupExclusiveAdd(a+b);let count=subgroupAdd(a+b);
 if(i<n){links[NB_CELLS+i]=select(0xffffffffu,rank,a!=0u);}
 if(i+1u<n){links[NB_CELLS+i+1u]=select(0xffffffffu,rank+a,b!=0u);}
 if(lane==0u&&i<n){links[nbCompactCounts()+job]=count;}
}
`:/* wgsl */`
@compute @workgroup_size(64) fn resampleCount(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) lane:u32){
 let i=gid.x+65535u*64u*gid.y;let n=min(atomicLoad(&state[0]),arrayLength(&source));
 var kept=0u;if(i<n){kept=u32(nbKept(i));}
 nbCompactScan[lane]=kept;workgroupBarrier();
 for(var stride=1u;stride<64u;stride*=2u){
  var add=0u;if(lane>=stride){add=nbCompactScan[lane-stride];}
  workgroupBarrier();nbCompactScan[lane]+=add;workgroupBarrier();
 }
 if(i<n){links[NB_CELLS+i]=select(0xffffffffu,nbCompactScan[lane]-kept,kept!=0u);}
 if(lane==0u&&i<n){links[nbCompactCounts()+i/64u]=nbCompactScan[63];}
}
`}
// One small scan of block counts, not a second particle sort. Each lane
// sums its contiguous range before the workgroup scan, then writes offsets.
@compute @workgroup_size(256) fn resamplePrefix(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){atomicStore(&bins[NB_BAND+3u],0u);}
 let blocks=(min(atomicLoad(&state[0]),arrayLength(&source))+63u)/64u;
 let span=(blocks+255u)/256u;let first=lane*span;let end=min(blocks,first+span);
 var total=0u;for(var b=first;b<end;b++){total+=links[nbCompactCounts()+b];}
 nbCompactScan[lane]=total;workgroupBarrier();
 for(var stride=1u;stride<256u;stride*=2u){
  var add=0u;if(lane>=stride){add=nbCompactScan[lane-stride];}
  workgroupBarrier();nbCompactScan[lane]+=add;workgroupBarrier();
 }
 var offset=nbCompactScan[lane]-total;
 for(var b=first;b<end;b++){links[nbCompactOffsets()+b]=offset;offset+=links[nbCompactCounts()+b];}
 if(lane==255u){atomicStore(&state[1],nbCompactScan[255]);}
}
@compute @workgroup_size(64) fn resample(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x+65535u*64u*gid.y;
 if(i>=min(atomicLoad(&state[0]),arrayLength(&source))){return;}
 let rank=links[NB_CELLS+i];if(rank==0xffffffffu){return;}
 particles[links[nbCompactOffsets()+i/64u]+rank]=source[i];
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
fn seedSite(c:vec3u,k:u32,initial:bool,activated:bool)->vec4f{
  let q=vec3f(c)+0.25+0.5*vec3f(umCorner(k,2u));let distance=particleDepth(q);
  if(distance>0.0||distance < -4.0){return vec4f(-1,0,0,0);}
  // Paper Section 3.3: replenish only the inner [-R,-h] band. The outer
  // layer keeps its transported samples instead of being recreated; a change
  // of ownership or an uncovered tile cannot bootstrap one from the bulk
  // field. Section 3.2: liquid made from a description gets its whole band,
  // at initialization and where a source has just made it.
  if(!initial&&!activated&&(u32(params.settings.w)&2u)==0u&&distance>-1.0&&nbSourcePhi(q,1.0)>=0.0){return vec4f(-1,0,0,0);}
  // The heat-one overlap supplies support without authority to erase
  // liquid. Its outer shell only seeds on resolved fine owners.
  if(nbAdaptive()&&distance>-1.0&&umOwnerAt(vec3i(c)).width!=1u){return vec4f(-1,0,0,0);}
  if(distance<=-NB_SURFACE_RADIUS){return vec4f(q,distance);}
  let shell=seedShell(q);let p=shell.xyz;
  if(shell.w>0.0||umCellOpen(vec3i(p))<0.5){return vec4f(-1,0,0,0);}
  // A shell sample can land in another cell, whose own seeding decision
  // does not see it: no count changes here, whatever the dispatch order.
  return vec4f(p,particleDepth(p));
}
fn seedCell(c:vec3u,initial:bool){
 // No tile past the reach is within six cells of the surface.
 if((nbBandReach(c)&1u)==0u){return;}
 let heat=nbTargetHeat(vec3f(c)+0.5);if(nbAdaptive()&&heat<=0.0){return;}
 let count=atomicLoad(&bins[nbOrder(c)]);if(count>=8u){return;}
 if(umCellOpen(vec3i(c))<0.5){return;}
 let d=particleDepth(vec3f(c)+0.5);if(d>0.9||d < -4.5){return;}
 let activated=nbAdaptive()&&atomicLoad(&bins[NB_ACTIVITY_NEW+nbBandTile(c)])!=0u;
 var seeds:array<vec4f,8>;var seedCount=0u;
 for(var k=count;k<8u;k++){
  let site=seedSite(c,k,initial,activated);if(site.x<0.0){continue;}
  seeds[seedCount]=site;seedCount++;
 }
 // Reserve the cell's accepted candidates together. Neighboring new samples
 // then remain contiguous through the next frame's particle tracing.
 if(seedCount==0u){return;}
 let first=atomicAdd(&state[1],seedCount);let capacity=arrayLength(&particles);
 let kept=min(seedCount,capacity-min(first,capacity));
 if(kept<seedCount){atomicAdd(&state[2],seedCount-kept);}
 for(var k=0u;k<kept;k++){
  let q=seeds[k];let v=sampleVelocity(q.xyz);
  particles[first+k]=Particle(vec4f(q.xyz,heat),vec4f(v,q.w),vec4f(v,0));
 }
}
// Seed the geometric particle band, including its overlap on 4h owners: one
// workgroup for each tile of the band search's list, the only ones seedCell
// passes.
fn nbSeedCell(group:vec3u,lane:u32)->vec3u{
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+group.x]);
 return 4u*vec3u(tile%UM_T.x,(tile/UM_T.x)%UM_T.y,tile/(UM_T.x*UM_T.y))+vec3u(lane%4u,(lane/4u)%4u,lane/16u);
}
// Reserve the remaining slots for new liquid before optional overlap
// reseeding. A single racing dispatch lets the growing pool exhaust the
// budget before the nozzle is visited, punching holes in a continuous jet.
// One cell of source distance encloses every quarter-cell seed; the same
// predicate partitions both passes so a source cell is never seeded twice.
fn nbSourceCell(c:vec3u)->bool{return nbSourcePhi(vec3f(c)+0.5,1.0)<1.0;}
// Compaction has consumed the old cell starts. Reuse that table as a list
// of cells needing candidates, one entry per possible cell at most.
@compute @workgroup_size(64) fn seedCells(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let c=nbSeedCell(group,lane);
 if((nbBandReach(c)&1u)==0u){return;}
 let heat=nbTargetHeat(vec3f(c)+0.5);if(nbAdaptive()&&heat<=0.0){return;}
 if(atomicLoad(&bins[nbOrder(c)])>=8u||umCellOpen(vec3i(c))<0.5){return;}
 let depth=particleDepth(vec3f(c)+0.5);if(depth>0.9||depth < -4.5){return;}
 let slot=atomicAdd(&bins[NB_BAND+3u],1u);
 links[slot]=cellIndex(vec3i(c))|select(0u,0x80000000u,nbSourceCell(c));
}
fn seedCandidate(group:vec3u,lane:u32,sourcePass:bool)->Particle{
 var empty=Particle();empty.position.x=-1.0;
 let slot=8u*group.x+lane/8u;if(slot>=atomicLoad(&bins[NB_BAND+3u])){return empty;}
 let entry=links[slot];if(((entry&0x80000000u)!=0u)!=sourcePass){return empty;}
 let c=nbCell(entry&0x7fffffffu);let k=lane%8u;
 if(k<atomicLoad(&bins[nbOrder(c)])){return empty;}
 let activated=nbAdaptive()&&atomicLoad(&bins[NB_ACTIVITY_NEW+nbBandTile(c)])!=0u;
 let site=seedSite(c,k,false,activated);if(site.x<0.0){return empty;}
 let heat=nbTargetHeat(vec3f(c)+0.5);let v=sampleVelocity(site.xyz);
 return Particle(vec4f(site.xyz,heat),vec4f(v,site.w),vec4f(v,0));
}
fn storeSeedCandidate(p:Particle,lane:u32){
 ${subgroups?/* wgsl */`
 let valid=p.position.x>=0.0;let rank=subgroupExclusiveAdd(u32(valid));let count=subgroupAdd(u32(valid));var first=0u;
 if(lane==0u&&count!=0u){first=atomicAdd(&state[1],count);}first=subgroupBroadcastFirst(first);
 if(valid){let at=first+rank;if(at<arrayLength(&particles)){particles[at]=p;}else{atomicAdd(&state[2],1u);}}
 `:"if(p.position.x>=0.0){store(p);}"}
}
@compute @workgroup_size(64) fn seedSources(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32${subgroups?",@builtin(subgroup_invocation_id) subgroupLane:u32":""}){
 storeSeedCandidate(seedCandidate(group,lane,true),${subgroups?"subgroupLane":"0u"});
}
@compute @workgroup_size(64) fn seed(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32${subgroups?",@builtin(subgroup_invocation_id) subgroupLane:u32":""}){
 storeSeedCandidate(seedCandidate(group,lane,false),${subgroups?"subgroupLane":"0u"});
}

@compute @workgroup_size(64) fn seedInitial(@builtin(global_invocation_id) gid:vec3u){
 for(var i=gid.x;i<UM_D.x*UM_D.y*UM_D.z;i+=65536u){seedCell(nbCell(i),true);}
}
fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
// Also the receipt's census of the samples against the surface they made.
@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id) gid:vec3u){
 var outside=0.0;var counts=vec3u(0);
 if(gid.x==0u){atomicStore(&bins[NB_BAND+3u],0u);atomicStore(&state[9],0u);}
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
 if(total<1e-5){return original;}return mix(original,momentum/total,blend*nbTransferBlend(q,depth,total));
}
${narrowBandTransferPropertiesSource(subgroups)}
${narrowBandFineTransferSource(subgroups,true)}
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
 let i=gid.x+65535u*64u*gid.y;
 if(i<min(atomicLoad(&state[1]),arrayLength(&particles))){
  let mode=particles[i].before.w;var v=vec3f(0);if(mode!=1.0){v=sampleVelocity(particles[i].position.xyz);}
  particles[i].before=vec4f(v,mode);
 }
}
@compute @workgroup_size(64) fn update(@builtin(global_invocation_id) gid:vec3u){
 if(all(gid==vec3u(0))){let n=min(atomicLoad(&state[1]),arrayLength(&particles));atomicStore(&state[0],n);atomicStore(&state[1],n);}
 let i=gid.x+65535u*64u*gid.y;
 if(i<min(atomicLoad(&state[1]),arrayLength(&particles))){
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
  if(!all(abs(v)<vec3f(1e10))){atomicStore(&state[3],2u);return;}
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
  const modules=[false,true].map(unitTaps=>uniformDetailModule(this.device,{label:`Uniform narrow-band FLIP ${unitTaps?"extended":"stage"}`,code:uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(uniformMixedCertifiedEntriesWGSL(sourceCode(unitTaps),["surfaceSplat"],"atomicLoad(&bins[NB_BAND+3u])"),["depthSeeds"],"atomicLoad(&bins[NB_BAND])"),["seed","seedSources"],"(atomicLoad(&bins[NB_BAND+3u])+7u)/8u"),["depthNearest"],"atomicLoad(&bins[NB_BAND+1u])"),["buildDistance"],"atomicLoad(&bins[NB_BAND+2u])"),["transferProperties"],"umCounts.x"),["transfer"],subgroups?"select(16u*umCounts.x,(atomicLoad(&state[9])+1u)/2u,nbTransferPropertiesFit())":"select(8u*umCounts.x,(atomicLoad(&state[9])+3u)/4u,nbTransferPropertiesFit())"),["transferOwners"],"(umCounts.y+63u)/64u"),["transferSeams"],"atomicLoad(&bins[NB_BAND+3u])"),["couple"],"(umCounts.y+63u)/64u"),["coupleFine"],"umCounts.x"),["redistanceFine"],"umCounts.x"),["seedCells"],"atomicLoad(&bins[NB_BAND+1u])")}));
  for(const module of modules){
   const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
   if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  }
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.solid.tileLayout]:[])]});
  await Promise.all([false,true].flatMap(coarse=>["activity","activitySpread","advectParticles","markParticleTiles","surfaceCells","surfaceSplat","seedCells","seed","seedSources","seedInitial","classify","transferProperties","transfer","transferOwners","transferSeams","transferCoarse","snapshot","update","surface","couple","coupleFine","commitCount","resampleCount","resamplePrefix","resample","diagnoseBefore","buildDistance","depthTiles","depthSeeds","depthLists","depthNearest","redistanceFine","redistanceCoarse"].map(async entryPoint=>{
   if((entryPoint==="activity"||entryPoint==="activitySpread")&&!this.activitySupported)return;
   if((entryPoint==="transferCoarse"&&!coarse)||(entryPoint==="surface"&&!this.coarseParticles))return;
   const module=modules[["advectParticles","snapshot","update","seed","seedSources"].includes(entryPoint)?1:0]!;
   const p=await uniformMixedSolidPipeline(this.solid,s=>uniformDetailPipeline(this.device,this.ownership,{label:`Narrow-band FLIP ${entryPoint} ${coarse?"4h":"mixed"}`,layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:(entryPoint==="couple"||entryPoint==="transfer")?3:0,nbCoarseOnly:+coarse,nbActivityEnabled:+this.activitySupported,...s}}})).catch(e=>{throw new Error(`${entryPoint} ${coarse}: ${e.message} ${e.reason}`,{cause:e});});
   this.pipelines.set(`${entryPoint}:${coarse}`,p);
  })));

 }
 private dispatch(encoder:GPUCommandEncoder,entry:string,groupName?:string):void{
  this.dispatchBatch(encoder,[entry],groupName,entry);
 }
 private dispatchBatch(encoder:GPUCommandEncoder,entries:readonly string[],groupName:string|undefined,label:string,shared?:GPUComputePassEncoder):void{
  const pass=shared??encoder.beginComputePass({label:`Narrow-band FLIP ${label}`});
  pass.setBindGroup(0,this.ownership.bindGroup);if(this.solid)pass.setBindGroup(2,this.solid.tileGroup);
  let bound:GPUBindGroup|undefined;
  for(const entry of entries){
  const group=this.groups[groupName??(entry.startsWith("resample")?"update":entry==="markParticleTiles"||entry==="surfaceCells"||entry==="surfaceSplat"||entry==="diagnoseBefore"||entry==="buildDistance"||entry.startsWith("depth")?"surface":entry.startsWith("transfer")||entry==="classify"?"transfer":entry==="couple"||entry==="coupleFine"||entry==="commitCount"?"surface":entry)]![this.parity]!.group;
  if(group!==bound){pass.setBindGroup(1,group);bound=group;}
  const selected=this.pipelines.get(`${entry}:${this.coarseOnly}`)!;
  const pipeline=this.solid?.select(selected)??selected;
  if(entry==="couple"||entry==="transfer")this.ownership.dispatchAllCounted(pass,pipeline);
  else {
   pass.setPipeline(uniformDetailPick(pipeline));
   if(entry==="markParticleTiles")pass.dispatchWorkgroups(Math.min(this.ownership.capacity.tiles,65535),Math.ceil(this.ownership.capacity.tiles/65535));
   else if(entry==="transferCoarse")pass.dispatchWorkgroups(256,Math.ceil(this.ownership.capacity.tiles/256),3);
   else if(entry==="resamplePrefix")pass.dispatchWorkgroups(1);
   else if(entry==="advectParticles"||entry==="snapshot"||entry==="update"||entry==="resampleCount"||entry==="resample")pass.dispatchWorkgroupsIndirect(this.particleDispatch,0);
   else pass.dispatchWorkgroups(1024);
  }
  }
  if(!shared)pass.end();
 }
 private encodeParticleWork(encoder:GPUCommandEncoder,shared?:GPUComputePassEncoder):void{
  const work=shared??encoder.beginComputePass({label:"Narrow-band FLIP live particle work"});work.setPipeline(this.particleDispatchPipeline);work.setBindGroup(0,this.particleDispatchGroup);work.dispatchWorkgroups(1);if(!shared)work.end();
 }
 move(encoder:GPUCommandEncoder,dt:number,openTop:boolean,gravity=0):void{
  this.bandCurrent=false;
  this.device.queue.writeBuffer(this.params,0,new Float32Array([...this.ownership.capacity.lattice.cellSize_m,dt,0.95,+openTop,gravity,+this.adaptive+(this.seedAll?2:0)]));
  const cells=this.ownership.capacity.lattice.dimensions.reduce((n,v)=>n*v,1);
  encoder.clearBuffer(this.state,16,32);
  this.bootstrap(encoder);
  this.seedAll=false;
  if(this.adaptive)encoder.clearBuffer(this.bins,(this.activityWord+5*this.ownership.capacity.tiles)*4,2*this.ownership.capacity.tiles*4);
  // The samples move in place, counting themselves into clear bins and
  // marking the tiles they reach; the order packs them into the other buffer.
  encoder.clearBuffer(this.state,4,8);this.order.prepare(encoder,1-this.parity);
  encoder.clearBuffer(this.bins,cells*8+this.ownership.capacity.tiles*4,this.ownership.capacity.tiles*4);
  this.encodeParticleWork(encoder);
  this.dispatch(encoder,"advectParticles");
  this.order.encode(encoder,1-this.parity,this.particleDispatch);
  this.dispatchBatch(encoder,["markParticleTiles","commitCount"],"surface","particle tile marking");
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
  if(!this.started&&!this.adaptive){this.device.queue.writeBuffer(this.params,0,new Float32Array([...lattice.cellSize_m,dt,0.95,0,0,0]));this.bootstrap(encoder);}
  this.device.queue.writeBuffer(this.refinementParams,0,new Float32Array([...lattice.dimensions,dt,...lattice.cellSize_m,padding]));
  // One group per particle bank: the bank alternates every step, the census's target does not.
  const particles=this.activeParticles,size=4*Math.ceil(this.ownership.capacity.tiles/32);
  let held=this.refinementGroups[this.parity];
  if(held?.particles!==particles||held.target!==target.buffer||held.wordOffset!==target.wordOffset||held.size!==size){
   held=this.refinementGroups[this.parity]={particles,target:target.buffer,wordOffset:target.wordOffset,size,group:this.device.createBindGroup({layout:this.refinementLayout!,entries:[
    {binding:0,resource:{buffer:particles}},{binding:1,resource:{buffer:this.state}},
    {binding:2,resource:{buffer:target.buffer,offset:target.wordOffset*4,size}},
    {binding:3,resource:{buffer:this.refinementParams}},
    {binding:4,resource:{buffer:this.bins}},
   ]})};
  }
  const group=held.group;
  const pass=encoder.beginComputePass({label:"Narrow-band FLIP surface refinement"});pass.setPipeline(this.refinementPipeline!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1024);pass.end();
 }
 /** The head census runs before remapping and advection. Copy its scores
  * into the existing arena, then publish heat in separate neighbour passes. */
 activity(encoder:GPUCommandEncoder,importance:{buffer:GPUBuffer;offset:number},dt:number,fadeSeconds=0.5):void{
  if(!this.adaptive)return;
  const tiles=this.ownership.capacity.tiles;
  this.device.queue.writeBuffer(this.params,32,new Float32Array([2/Math.max(0.05,fadeSeconds),0,0,0]));
  this.device.queue.writeBuffer(this.params,0,new Float32Array([...this.ownership.capacity.lattice.cellSize_m,dt,0.95,0,0,1]));
  encoder.copyBufferToBuffer(importance.buffer,importance.offset,this.bins,this.activityWord*4,tiles*8);
  this.dispatchBatch(encoder,["activity","activitySpread"],"bootstrap","activity");
 }
 reconstruct(encoder:GPUCommandEncoder):void{this.bandCurrent=false;this.measureBand(encoder);if(this.measureTraces)this.dispatch(encoder,"diagnoseBefore");
  const cells=this.ownership.capacity.lattice.dimensions.reduce((a,b)=>a*b,1),vertices=this.ownership.capacity.lattice.dimensions.reduce((a,b)=>a*(b+1),1);
  encoder.clearBuffer(this.bins,(2*cells+2*this.ownership.capacity.tiles)*4,vertices*4);
  this.dispatchBatch(encoder,["surfaceCells","surfaceSplat",...(!this.ownership.coarseOnly?["coupleFine"]:[]),"couple"],"surface","surface reconstruction");}
 transfer(encoder:GPUCommandEncoder):void{
  // Classification reads phase and phi. Transfer loads canonical face
  // texels directly in both mixed and all-4h modes; neither samples a cache.
  this.dispatchBatch(encoder,["classify",...(this.coarseOnly?["transferCoarse"]:["transferProperties","transfer","transferOwners","transferSeams"])],"transfer","transfer");}
 snapshot(encoder:GPUCommandEncoder):void{
  const pass=encoder.beginComputePass({label:"Narrow-band FLIP snapshot"});
  this.cache.encode(encoder,this.cacheGroups.snapshot!,pass);this.encodeStageTaps(encoder,pass);this.encodeParticleWork(encoder,pass);
  this.dispatchBatch(encoder,["snapshot"],undefined,"snapshot",pass);pass.end();
 }
 private encodeStageTaps(encoder:GPUCommandEncoder,shared?:GPUComputePassEncoder):void{if(this.stageTaps&&this.ownership.capacity.fineTiles>0)this.stageTaps.builder.encode(encoder,this.stageTaps.group,shared);}
 /** Fixed-band redistance preserves the crossing set. Adaptive retirement
  * can move a hanging zero when it normalizes coarse corner magnitudes;
  * rebuild its search/metric cache before end-step particle membership. */
 redistance(encoder:GPUCommandEncoder):void{
  this.measureBand(encoder,"redistance");this.dispatchBatch(encoder,["buildDistance",...(!this.ownership.coarseOnly?["redistanceFine"]:[]),"redistanceCoarse"],"redistance","redistance");this.bandCurrent=!this.adaptive;
 }
 private measureBand(encoder:GPUCommandEncoder,group?:string):void{
  // The search's tile banks follow the cell and vertex banks: counts, masks and reach start clear.
  const d=this.ownership.capacity.lattice.dimensions,cells=d.reduce((n,v)=>n*v,1),tiles=this.ownership.capacity.tiles;
  encoder.clearBuffer(this.bins,(3*cells+d.reduce((n,v)=>n*(v+1),1)+2*tiles)*4,(4+3*tiles)*4);
  this.dispatchBatch(encoder,["depthTiles","depthSeeds","depthLists","depthNearest"],group,"crossing search");
 }
 update(encoder:GPUCommandEncoder):void{
  if(!this.bandCurrent){this.measureBand(encoder);this.dispatch(encoder,"buildDistance");}this.bandCurrent=false;
  const pass=encoder.beginComputePass({label:"Narrow-band FLIP update"});
  this.cache.encode(encoder,this.cacheGroups.update!,pass);this.encodeStageTaps(encoder,pass);this.encodeParticleWork(encoder,pass);
  this.dispatchBatch(encoder,["update"],undefined,"update",pass);pass.end();
  // The ordered epoch compacts into the other buffer; the bins keep its counts for the seeding.
  this.parity=1-this.parity;
  this.dispatchBatch(encoder,["resampleCount","resamplePrefix","resample","seedCells","seedSources","seed"],"update","resample and seed");
  this.dispatch(encoder,"commitCount");
  if(this.coarseParticles)this.dispatch(encoder,"surface");this.parity=1-this.parity;
 }
 noteReceipt(words:Uint32Array):void{
  this.count=words[1]!;this.reseedClipped=words[2]!;
  const distances=new Float32Array(words.buffer,words.byteOffset,words.length);
  this.diagnostics={beforeMaxOutside:this.measureTraces?distances[4]!:NaN,afterMaxOutside:distances[5]!,outsideSurface:words[6]!,unsupported:words[7]!,deepInterior:words[8]!};
  // Persistent samples own the budget first. At capacity only optional
  // reseeding is deferred; never destroy a surface to make room for it.
  if(words[3])throw new Error(`Narrow-band FLIP ${words[3]===1?`trajectory exceeded ${NARROW_BAND_TRACE_LIMIT} cells`:"nonfinite velocity"}`);
 }
 destroy():void{this.order.destroy();this.particleDispatch.destroy();this.surfaceSource.vertexPhi.destroy();this.surfaceSource.openFraction.destroy();this.coarseVelocity.destroy();this.stageTaps?.unit.destroy();this.refinementParams.destroy();for(const b of [...this.particles,this.state,this.bins,this.next,this.params])b.destroy();}
}
