import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import type { UniformPagePool } from "./uniform-page-pool";
import { UNIFORM_LIVE_GROUP_GROUPS, UniformLiveVisit, uniformLiveDispatch, uniformLiveListsWGSL } from "./uniform-live-lists";

/** Potential writers and their dependency neighbourhoods for one complete
 * frame. These are conservative work domains, not permissions to discard
 * numerical changes. Consumers retain their exact revision checks. */
export const UNIFORM_REGIONAL_WORK = {
 surface: 0, topology: 1, velocity: 2,
 surfaceHalo: 3, topologyHalo: 4, velocityHalo: 5, support: 6,
} as const;
export type UniformRegionalWorkKind = keyof typeof UNIFORM_REGIONAL_WORK;
const CHANNELS=7, HEADER=12;

export function uniformRegionalPlanLayout(regions:number){
 const flags=HEADER, lists=flags+regions, fine=lists+CHANNELS*regions, coarse=fine+64*regions, history=coarse+64*regions;
 // World xyz, phi revision, six exact motion extrema, topology revision.
 return {regions,flags,lists,fine,coarse,history,words:history+11*regions};
}

/** Direct launches only. One metadata pass builds all lists at the frame
 * boundary; none of its consumers rebuilds or resizes the plan. A bounded
 * direct grid strides the actual list, including unexpected activation.
 * With a page pool both launches stride its blocks (uniform-live-lists.ts):
 * a free region's seed is a fixed point after one visit (its history's
 * revision zero), and its expand writes nothing. */
export class UniformRegionalFramePlan {
 readonly layout:ReturnType<typeof uniformRegionalPlanLayout>;
 readonly buffer:GPUBuffer;
 readonly allocatedBytes:number;
 readonly groups:number;
 enabled=true;
 /** Diagnostic ablation of the shared stage schedule, independent of domains. */
 scheduleStages=true;
 coalescePasses=true;
 fuseAirClear=true;
 /** Diagnostic ablation; consumers must use full setup when disabled. */
 retainGeometryCache=true;
 active=false;
 /** The frame host proved phi, topology, solids and derived outputs held.
  * Consumers omit whole stages; the next active frame rebuilds the lists. */
 retainsGeometry=false;
 serial=0;
 /** Stage-owned domains may live in an existing arena to avoid another
  * storage binding. Their producer runs once under this same frame seal. */
 private readonly domains=new WeakMap<object,number>();
 invalidateDomain(owner:object):void{this.domains.delete(owner);}
 encodeDomain(encoder:GPUCommandEncoder,owner:object,encode:(encoder:GPUCommandEncoder)=>void):boolean{
  if(!this.active)return false;
  if(this.domains.get(owner)!==this.serial){encode(encoder);this.domains.set(owner,this.serial);}
  return true;
 }
 private readonly params:GPUBuffer;
 private readonly resources:GPUBindGroupLayout;
 private readonly group:GPUBindGroup;
 private seed!:GPUComputePipeline;
 private expand!:GPUComputePipeline;
 private generation=-1;
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,
  revisions:{buffer:GPUBuffer;liveOffset:number;topologyOffset:number;regions:number},tiers:GPUBuffer|undefined,
  private readonly currentGeneration:()=>number,
  private readonly geometryCache?:{buffer:GPUBuffer;mapping:number;owners:number;remap:number;regions:number},
  private readonly pool?:UniformPagePool){
  const r=revisions.regions;this.layout=uniformRegionalPlanLayout(r);this.groups=Math.max(1,Math.min(pool?UNIFORM_LIVE_GROUP_GROUPS:128,r));
  this.seedVisit=pool&&new UniformLiveVisit(pool);
  this.buffer=device.createBuffer({label:"Uniform shared regional frame plan",size:4*this.layout.words,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  this.params=device.createBuffer({label:"Uniform regional frame plan policy",size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.allocatedBytes=this.buffer.size+this.params.size;
  this.resources=device.createBindGroupLayout({entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   ...[1,2,...(tiers?[3]:[])].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}})),
   {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...(geometryCache?[{binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}]:[]),
   ...(pool?[{binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}]:[]),
  ]});
  this.group=device.createBindGroup({layout:this.resources,entries:[
   {binding:0,resource:{buffer:this.buffer}},{binding:1,resource:{buffer:revisions.buffer}},
   {binding:2,resource:{buffer:ownership.regionVelocity}},...(tiers?[{binding:3,resource:{buffer:tiers}}]:[]),
   {binding:4,resource:{buffer:this.params}},
   ...(geometryCache?[{binding:5,resource:{buffer:geometryCache.buffer}}]:[]),
   ...(pool?[{binding:6,resource:{buffer:pool.state}}]:[]),
  ]});
  this.source={revision:revisions.liveOffset/4,topology:revisions.topologyOffset/4,tiers:!!tiers};
 }
 private readonly source:{revision:number;topology:number;tiers:boolean};
 private readonly seedVisit?:UniformLiveVisit;
 async initialize():Promise<void>{
  const l=this.layout,r=l.regions;
  const module=this.device.createShaderModule({label:"Uniform shared regional frame plan",code:uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> plan:array<atomic<u32>>;
@group(1) @binding(1) var<storage,read> revisions:array<u32>;
@group(1) @binding(2) var<storage,read> velocities:array<u32>;
${this.source.tiers?"@group(1) @binding(3) var<storage,read> tiers:array<u32>;":""}
@group(1) @binding(4) var<uniform> policy:vec4u;
const RP_REGIONS:u32=${r}u;
${this.pool?uniformLiveListsWGSL(1,6,this.pool.layout)+`fn rpVisits(sweep:bool)->u32{return 8u*ulVisitCount(sweep);}
fn rpRegion(i:u32,sweep:bool)->u32{return ulVisitRegion(i,sweep);}`:`fn ulSweep(groups:vec3u)->bool{return false;}
fn rpVisits(sweep:bool)->u32{return RP_REGIONS;}
fn rpRegion(i:u32,sweep:bool)->u32{return i;}`}
fn rpRetiredHalo(at:u32){
 if(atomicLoad(&plan[at+3u])==0u){return;}
 let old=vec3i(vec3u(atomicLoad(&plan[at]),atomicLoad(&plan[at+1u]),atomicLoad(&plan[at+2u])));
 for(var k=0u;k<27u;k++){let tile=umTileKey(4*(old+vec3i(umCorner(k,3u))-vec3i(1)));if(umKeyStored(tile)){atomicOr(&plan[${l.flags}u+umPageOfTile(tile)],7u);}}
}
${this.geometryCache?/* wgsl */`
@group(1) @binding(5) var<storage,read_write> rpGeometry:array<atomic<u32>>;
var<workgroup> rpFree:array<u32,64>;
var<workgroup> rpFreeCount:atomic<u32>;
var<workgroup> rpClaimCount:atomic<u32>;
// A bounded cache of whole physical blocks. Retain surviving assignments;
// a page retirement releases its slot, and new live blocks fill vacancies.
// Only this workgroup writes the mapping. The other seed groups merely OR
// flags, so cache allocation shares their launch without a global barrier.
fn rpCache(lane:u32){
 if(lane==0u){atomicStore(&rpFreeCount,0u);atomicStore(&rpClaimCount,0u);}workgroupBarrier();
 if(lane<${this.geometryCache.regions/8}u){
  let owner=atomicLoad(&rpGeometry[${this.geometryCache.owners}u+8u*lane]);
  if(owner>=RP_REGIONS||!umTileLive(umPageTile(owner,0u))){
   if(owner<RP_REGIONS){for(var k=0u;k<8u;k++){atomicStore(&rpGeometry[${this.geometryCache.mapping}u+owner+k],0u);}}
   for(var k=0u;k<8u;k++){atomicStore(&rpGeometry[${this.geometryCache.owners}u+8u*lane+k],0xffffffffu);}
   let free=atomicAdd(&rpFreeCount,1u);rpFree[free]=lane;
  }
 }
 storageBarrier();workgroupBarrier();
 let free=atomicLoad(&rpFreeCount);if(free==0u){return;}
 for(var block=1u+lane;block<UM_TILES/512u-1u;block+=64u){
  if(umBlockWord(block)==0xffffffffu||atomicLoad(&rpGeometry[${this.geometryCache.mapping}u+8u*block])!=0u){continue;}
  let rank=atomicAdd(&rpClaimCount,1u);if(rank>=free){continue;}let slot=rpFree[rank];
  for(var k=0u;k<8u;k++){
   let region=8u*block+k;
   atomicStore(&rpGeometry[${this.geometryCache.mapping}u+region],8u*slot+k+1u);
   atomicStore(&rpGeometry[${this.geometryCache.owners}u+8u*slot+k],region);
   // A new assignment invalidates every snapshot in its slot before reuse.
   atomicStore(&rpGeometry[${this.geometryCache.remap}u+region],1u);
   atomicOr(&plan[${l.flags}u+region],7u);
  }
 }
}
`:""}
@compute @workgroup_size(64) fn seed(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let sweep=ulSweep(groups);if(group.y!=0u){return;}
 ${this.geometryCache?"if(group.x==0u&&policy.y!=0u){rpCache(lane);}":""}
 for(var visit=id.x;visit<rpVisits(sweep);visit+=64u*groups.x){
  let region=rpRegion(visit,sweep);let at=${l.history}u+11u*region;
  if(!umTileLive(umPageTile(region,0u))){rpRetiredHalo(at);atomicStore(&plan[at+3u],0u);continue;}
  var flags=0u;let world=umPageCoord(region);
  if(any(world!=vec3u(atomicLoad(&plan[at]),atomicLoad(&plan[at+1u]),atomicLoad(&plan[at+2u])))){rpRetiredHalo(at);flags=7u;}
  for(var k=0u;k<3u;k++){atomicStore(&plan[at+k],world[k]);}
  let revision=revisions[${this.source.revision}u+region];let previous=atomicLoad(&plan[at+3u]);
  if(previous==0u||revision==0u||revision>=0xfffffffeu||revision!=previous){flags|=1u;}
  atomicStore(&plan[at+3u],revision);
  let topology=revisions[${this.source.topology}u+region];
  if(topology==0u||topology>=0xfffffffeu||topology!=atomicLoad(&plan[at+10u])){flags|=7u;}
  atomicStore(&plan[at+10u],topology);
  for(var k=0u;k<6u;k++){let value=velocities[6u*region+k];if(value!=atomicLoad(&plan[at+4u+k])){flags|=4u;}atomicStore(&plan[at+4u+k],value);}
  // Non-COLD regions may write later this frame even with unchanged inputs.
  if(${this.source.tiers?"tiers[region]!=0u":"true"}){flags|=5u;}
  if(policy.x!=0u){flags=7u;}atomicOr(&plan[${l.flags}u+region],flags);
 }
}
// Read only the immutable seed bits. Each region appends itself once to
// each list, so overlapping neighbourhoods cannot duplicate work.
var<workgroup> rpHalo:atomic<u32>;
var<workgroup> rpCounts:array<atomic<u32>,2>;
var<workgroup> rpBases:array<u32,2>;
var<workgroup> rpLive:u32;
var<workgroup> rpMask:u32;
@compute @workgroup_size(64) fn expand(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var visit=group.x;visit<rpVisits(false);visit+=groups.x){
  let region=rpRegion(visit,false);
  if(lane==0u){rpLive=u32(umTileLive(umPageTile(region,0u)));atomicStore(&rpHalo,atomicLoad(&plan[${l.flags}u+region])&7u);}
  if(workgroupUniformLoad(&rpLive)==0u){continue;}
  let own=atomicLoad(&plan[${l.flags}u+region])&7u;
  if(lane<27u){let p=vec3i(umPageCoord(region));let tile=umTileKey(4*(p+vec3i(umCorner(lane,3u))-vec3i(1)));if(umKeyStored(tile)){atomicOr(&rpHalo,atomicLoad(&plan[${l.flags}u+umPageOfTile(tile)])&7u);}}
  workgroupBarrier();
  if(lane==0u){let halo=atomicLoad(&rpHalo);rpMask=own|(halo<<3u)|select(0u,64u,(halo&5u)!=0u);atomicStore(&plan[${l.flags}u+region],rpMask);atomicStore(&rpCounts[0],0u);atomicStore(&rpCounts[1],0u);}
  let mask=workgroupUniformLoad(&rpMask);
  if(lane<7u&&(mask&(1u<<lane))!=0u){let slot=atomicAdd(&plan[lane],1u);atomicStore(&plan[${l.lists}u+lane*RP_REGIONS+slot],region);}
  if((mask&8u)!=0u){
   let tile=umPageTile(region,lane);let bank=u32(umTileWidth(tile)!=1u);
   let rank=atomicAdd(&rpCounts[bank],1u);workgroupBarrier();
   // Reserve each region's whole tile range once, then emit in parallel.
   if(lane<2u){rpBases[lane]=atomicAdd(&plan[7u+lane],atomicLoad(&rpCounts[lane]));}workgroupBarrier();
   atomicStore(&plan[select(${l.fine}u,${l.coarse}u,bank==1u)+rpBases[bank]+rank],tile);
  }
  workgroupBarrier();
 }
}

`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources]});
  [this.seed,this.expand]=await Promise.all(["seed","expand"].map(entryPoint=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint}})));
 }
 beginFrame():void{this.active=false;this.retainsGeometry=false;this.serial++;}
 encode(encoder:GPUCommandEncoder,retained=false):void{
  if(!this.enabled)return;
  this.retainsGeometry=retained;if(retained)return;
  const generation=this.currentGeneration();this.device.queue.writeBuffer(this.params,0,new Uint32Array([+(generation!==this.generation),+this.retainGeometryCache,0,0]));this.generation=generation;
  encoder.clearBuffer(this.buffer,0,4*(HEADER+this.layout.regions));
  const pass=encoder.beginComputePass({label:"Uniform shared regional frame plan"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,this.group);
  pass.setPipeline(this.seed);uniformLiveDispatch(pass,Math.min(128,Math.ceil(this.layout.regions/64)),this.seedVisit?.sweep()??false);pass.setPipeline(this.expand);pass.dispatchWorkgroups(this.groups);pass.end();this.active=true;
 }
 /** A consumer selects this list once for its frame. The list is immutable
  * until the next head, including through reconstruction and projection. */
 wgsl(group:number,binding:number,name="regionalPlan"):string{return uniformRegionalPlanWGSL(this.layout.regions,group,binding,name);}
 destroy():void{this.buffer.destroy();this.params.destroy();}
}

export function uniformRegionalPlanWGSL(regions:number,group:number,binding:number,name="regionalPlan"):string{
 const l=uniformRegionalPlanLayout(regions);
 return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> ${name}:array<u32>;
override regionalListed:bool=false;
fn regionalCount(channel:u32)->u32{return select(${regions}u,${name}[channel],regionalListed);}
fn regionalAt(channel:u32,slot:u32)->u32{if(!regionalListed){return slot;}return ${name}[${l.lists}u+channel*${regions}u+slot];}
fn regionalHas(channel:u32,region:u32)->bool{return !regionalListed||(${name}[${l.flags}u+region]&(1u<<channel))!=0u;}
`;
}
