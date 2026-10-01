import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { UNIFORM_MIXED_COUNTED, uniformMixedCountedEntriesWGSL, uniformMixedPageCount, uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { uniformVolumeCorrectionWGSL } from "./uniform-volume-correction.wgsl";
import { uniformMixedDetachedMassWGSL } from "./uniform-mixed-detached-mass.wgsl";
import { uniformMixedSolidPipeline, uniformMixedSolidWGSL, type UniformMixedSolid } from "./uniform-mixed-solid.wgsl";

export interface UniformMixedPressureAuthorityFields {
 centerPhi:GPUTexture;volume:GPUTexture;targetFill:GPUTexture;
 phi:GPUBufferBinding;
 /** Phase flag for momentum's supported physical donor gather. */
 phase:GPUTexture;
 correction:GPUTexture;
 scratch:GPUBufferBinding;
 /** dt, deficit balance (0 on, -1 off), unused, dust threshold. */
 params:GPUBuffer;
 /** Coarse mode only: the h centre phi and volume in simulation ownership,
  * where every cut tile is h (promotion certificate). */
 fine?:{centerPhi:GPUTexture;volume:GPUTexture};
}
/** One pressure interface for RHS, projection, momentum support and extension.
 * Excess/deficit balance uses physical cell mass; coarse cells are not counted
 * as one fine cell. All fields and reduction scratch are caller-owned.
 * With static solids, fine owners follow native pressurePhi exactly: rho'=V/open
 * claims rows, closed cells continue the open liquid interface (CM11a's one
 * layer of solid unknowns), and balance counts open liquid capacity only.
 * Coarse mode (band pressure's all-4h owners) reads the static all-4h solid
 * record the same way: capacity is the owner's mean open fraction, a closed
 * owner continues its open liquid neighbours, and deficits are of target*cap.
 * A cut 4h owner's phi is native mgDownsampleTopology's open-child vote
 * (h to 2h to 4h, positive preferred) of the h pressurePhi: its 4h centre
 * phi samples vertices buried in the solid, which are not state. The vote
 * runs first, one workgroup per cut tile (a lane per h cell), into the
 * scratch after the balance words; build reads it for the owner and for a
 * closed owner's cut neighbours.
 * Resident mode (the split's all-4h pressure ownership, whose support carries
 * the simulation's residency certificate) strides the resident pages' owners
 * only (residentAll): an absent page is certified far air (V=0, corner phi at
 * least 16h), so its owners add nothing to the balance and no reader needs
 * their phase or correction. Their phi is still read (neighbour faces, the
 * native continuation lattice), and the pressure root's phi words are shared
 * scratch, so an "absent" pass writes umAuthority(o,0) for each owner of an
 * absent page: the same value the dense build writes there. */
/** Owner entries: GPU-counted launches over every tier (umAllOwner). */
const counted:readonly string[]=["build","resolve"];

export class UniformMixedPressureAuthority {
 readonly allocatedBytes=0;
 readonly scratchBytes:number;
 /** Owners this ownership can hold: the construction cellCount. */
 private readonly owners:number;
 /** Balance partials (one per counted job) and their chunk sums. */
 private readonly groups:number;
 private readonly chunks:number;
 private readonly resources:GPUBindGroupLayout;
 private readonly pipelines=new Map<string,GPUComputePipeline>();
 constructor(private readonly device:GPUDevice,readonly ownership:UniformMixedOwnership,private readonly solid?:UniformMixedSolid,private readonly coarse=false,private readonly resident=false){
  if(coarse&&(!solid?.coarse||ownership.layout.tiles.some(word=>(word&0xc0000000)!==0)))throw new Error("Coarse mixed pressure authority requires the all-4h solid record and all-4h ownership");
  // A capacity bound, read once: the unified frame reserves its simulation
  // ownership all-h (cellCount = 64 per tile) and pressure ownership is the
  // fixed all-4h hierarchy, so no later generation has more owners. build's
  // counted jobs (umCounts.x + ceil(umCounts.y/64), which chunks reads back
  // from the GPU) therefore never exceed groups.
  this.owners=ownership.layout.cellCount;
  this.groups=Math.ceil(this.owners/64);this.chunks=Math.ceil(this.groups/1024);
  // Two vec2f words per partial; coarse: one vec2f cut-vote slot per owner
  // after the balance words.
  this.scratchBytes=8*(1+2*(this.groups+this.chunks)+(coarse?this.owners:0));
  this.resources=device.createBindGroupLayout({entries:[
   ...[0,1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
   ...[3,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
   ...[4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only" as const,format:"r32float" as const,viewDimension:"3d" as const}})),
   {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
   ...(coarse?[8,9].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})):[]),
  ]});
 }
 bind(f:UniformMixedPressureAuthorityFields):GPUBindGroup{
  if((f.scratch.size??f.scratch.buffer.size-(f.scratch.offset??0))<this.scratchBytes)throw new Error("Mixed pressure authority scratch is too small");
  const d=this.ownership.capacity.lattice.dimensions;
  for(const t of [f.centerPhi,f.volume,f.targetFill,f.phase,f.correction])if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==d[a]))throw new Error("Mixed pressure authority requires native scalar fields");
  if([f.centerPhi,f.volume,f.targetFill].some(t=>t===f.phase||t===f.correction)||f.phase===f.correction)throw new Error("Mixed pressure authority outputs must be disjoint");
  if(!!f.fine!==this.coarse)throw new Error("Coarse mixed pressure authority needs the h simulation centre phi and volume, and only it");
  if(f.fine)for(const t of [f.fine.centerPhi,f.fine.volume])if(t.format!=="r32float"||[t.width,t.height,t.depthOrArrayLayers].some((v,a)=>v!==d[a])||t===f.phase||t===f.correction)throw new Error("Coarse mixed pressure authority h fields must be native scalar inputs");
  return this.device.createBindGroup({layout:this.resources,entries:[
   ...[f.centerPhi,f.volume,f.targetFill].map((t,binding)=>({binding,resource:t.createView()})),
   {binding:3,resource:{...f.phi,size:4*this.owners}},
   {binding:4,resource:f.phase.createView()},{binding:5,resource:f.correction.createView()},
   {binding:6,resource:{...f.scratch,size:this.scratchBytes}},{binding:7,resource:{buffer:f.params,size:16}},
   ...(f.fine?[f.fine.centerPhi,f.fine.volume].map((t,i)=>({binding:8+i,resource:t.createView()})):[]),
  ]});
 }
 async initialize():Promise<void>{
  // build (and its phase variant) and resolve stride the GPU-counted owners
  // of every tier (umAllOwner); build writes one balance partial per job.
  const module=this.device.createShaderModule({code:uniformMixedCountedEntriesWGSL(uniformMixedTopologyWGSL(this.ownership.capacity,0)+/* wgsl */`
@group(1) @binding(0) var centerPhi:texture_3d<f32>;
@group(1) @binding(1) var volume:texture_3d<f32>;
@group(1) @binding(2) var targetFill:texture_3d<f32>;
@group(1) @binding(3) var<storage,read_write> phi:array<f32>;
@group(1) @binding(4) var phase:texture_storage_3d<r32float,write>;
@group(1) @binding(5) var correction:texture_storage_3d<r32float,write>;
@group(1) @binding(6) var<storage,read_write> balance:array<vec2f>;
@group(1) @binding(7) var<uniform> params:vec4f;
${uniformVolumeCorrectionWGSL}
${uniformMixedSolidWGSL(this.solid?2:undefined,this.coarse?this.solid!.coarse!.count:undefined)}
const UM_HMIN=${Math.min(...this.ownership.capacity.lattice.cellSize_m)};
// pressureSurfacePhi for one fine cell (solid scenes only): centre phi.
fn umSurfacePhiCell(p:vec3i)->f32{return textureLoad(centerPhi,p,0).x;}
// pressurePhi: a closed cell continues its open liquid neighbours.
fn umPressurePhiCell(p:vec3i)->f32{
 if(umCellOpen(p)>1e-5){return umSurfacePhiCell(p);}
 var terms:array<f32,6>;var weights:array<f32,6>;
 for(var i=0u;i<6u;i++){var q=p;q[i/2u]+=select(-1,1,(i&1u)!=0u);let open=umCellOpen(q);
  terms[i]=0.0;weights[i]=0.0;if(open<=1e-5){continue;}
  let phi=umSurfacePhiCell(q);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}}
 let weight=((weights[0]+weights[1])+(weights[4]+weights[5]))+(weights[2]+weights[3]);
 let sum=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);
 return select(0.5*UM_HMIN,sum/max(weight,1e-9),weight>0.0);
}
fn umCapacity(o:UMOwner)->f32{${this.coarse?"return umSolidCoarse(o.index).x;":"return select(1.0,umCellOpen(vec3i(umOrigin(o))),umSolidEnabled()&&o.width==1u);"}}
${uniformMixedDetachedMassWGSL(o=>`textureLoad(centerPhi,vec3i(umOrigin(${o})),0).x<0.0`,o=>`textureLoad(volume,vec3i(umOrigin(${o})),0).x`,"params.w")}
// Surface phi of an owner: its centre phi.
fn umOwnerSurfacePhi(o:UMOwner,v:f32,cap:f32)->f32{return textureLoad(centerPhi,vec3i(umOrigin(o)),0).x;}
${this.coarse?`@group(1) @binding(8) var fineCenterPhi:texture_3d<f32>;
@group(1) @binding(9) var fineVolume:texture_3d<f32>;
// umSurfacePhiCell on the h simulation fields.
fn umFineSurfacePhiCell(p:vec3i)->f32{return textureLoad(fineCenterPhi,p,0).x;}
// umPressurePhiCell on the h simulation fields.
fn umFinePressurePhiCell(p:vec3i)->f32{
 if(umCellOpen(p)>1e-5){return umFineSurfacePhiCell(p);}
 var terms:array<f32,6>;var weights:array<f32,6>;
 for(var i=0u;i<6u;i++){var q=p;q[i/2u]+=select(-1,1,(i&1u)!=0u);let open=umCellOpen(q);
  terms[i]=0.0;weights[i]=0.0;if(open<=1e-5){continue;}
  let phi=umFineSurfacePhiCell(q);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}}
 let weight=((weights[0]+weights[1])+(weights[4]+weights[5]))+(weights[2]+weights[3]);
 let sum=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);
 return select(0.5*UM_HMIN,sum/max(weight,1e-9),weight>0.0);
}
fn umSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
// Native restrictSurfacePhi on a split owner (umPreferPositivePhi on).
fn umOpenVote(values:array<f32,8>,opens:array<f32,8>)->f32{
 var open:array<f32,8>;var flags:array<f32,8>;var positive:array<f32,8>;var positiveFlags:array<f32,8>;var negativeFlags:array<f32,8>;
 for(var k=0u;k<8u;k++){let isOpen=opens[k]>1e-5;let value=values[k];
  open[k]=select(0.0,value,isOpen);flags[k]=select(0.0,1.0,isOpen);
  positive[k]=select(0.0,value,value>=0.0&&isOpen);positiveFlags[k]=select(0.0,1.0,value>=0.0&&isOpen);negativeFlags[k]=select(0.0,1.0,value<0.0&&isOpen);}
 let openCount=umSum8(flags);let positiveCount=umSum8(positiveFlags);
 let sum=select(umSum8(values),umSum8(open)*8.0/max(openCount,1.0),openCount>0.0);
 return select(sum/8.0,umSum8(positive)/max(positiveCount,1.0),positiveCount>0.0&&umSum8(negativeFlags)>0.0);
}
const UM_CUT_BASE=${1+2*(this.groups+this.chunks)}u;
var<workgroup> cutCount:atomic<u32>;
var<workgroup> cutListed:u32;
var<workgroup> cutTiles:array<u32,64>;
var<workgroup> cutValues:array<f32,64>;
var<workgroup> cutOpens:array<f32,64>;
var<workgroup> cutMid:array<f32,8>;
var<workgroup> cutMidOpen:array<f32,8>;
// A cut owner's phi: h pressurePhi voted to 2h, then to 4h. A fixed grid of
// workgroups strides over the lattice 64 tiles at a time: each lane tests one
// tile, the cut ones are compacted, then each is one workgroup job (lane
// 8k+j is h cell j of 2h block k). Uncut tiles cost one lane test.
@compute @workgroup_size(64) fn cut(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) l:u32){
 for(var start=64u*group.x;start<UM_TILES;start+=64u*groups.x){
  if(l==0u){atomicStore(&cutCount,0u);}
  workgroupBarrier();
  let t=start+l;
  if(t<UM_TILES&&umSolidCut(t)){cutTiles[atomicAdd(&cutCount,1u)]=t;}
  workgroupBarrier();
  if(l==0u){cutListed=atomicLoad(&cutCount);}
  let count=workgroupUniformLoad(&cutListed);
  for(var i=0u;i<count;i++){cutTile(cutTiles[i],l);workgroupBarrier();}
 }
}
fn cutTile(t:u32,l:u32){
 let origin=vec3i(4u*umTileCoord(t));let k=l>>3u;let j=l&7u;
 let p=origin+2*vec3i(vec3u(k&1u,(k>>1u)&1u,k>>2u))+vec3i(vec3u(j&1u,(j>>1u)&1u,j>>2u));
 cutValues[l]=umFinePressurePhiCell(p);cutOpens[l]=umCellOpen(p);
 workgroupBarrier();
 if(l<8u){var values:array<f32,8>;var opens:array<f32,8>;
  for(var i=0u;i<8u;i++){values[i]=cutValues[8u*l+i];opens[i]=cutOpens[8u*l+i];}
  cutMid[l]=umOpenVote(values,opens);cutMidOpen[l]=umSum8(opens)/8.0;}
 workgroupBarrier();
 if(l==0u){balance[UM_CUT_BASE+(umTopology[t]&0x3fffffffu)]=vec2f(umOpenVote(cutMid,cutMidOpen),0.0);}
}
fn umCutPhi(o:UMOwner)->f32{return balance[UM_CUT_BASE+o.index].x;}
// An open owner's phi: cut owners vote their h cells.
fn umOpenOwnerPhi(o:UMOwner,v:f32,cap:f32)->f32{
 if(umSolidCut(umTileAt(umOrigin(o)/4u))){return umCutPhi(o);}
 return umOwnerSurfacePhi(o,v,cap);
}`:""}
fn umAuthority(o:UMOwner,v:f32)->f32{
 ${this.coarse?`let cap=umCapacity(o);if(cap>1e-5){return umOpenOwnerPhi(o,v,cap);}
 // A closed owner continues its open liquid neighbours, capacity-weighted.
 var terms:array<f32,6>;var weights:array<f32,6>;
 for(var i=0u;i<6u;i++){
  terms[i]=0.0;weights[i]=0.0;var q=vec3i(umOrigin(o));q[i/2u]+=select(-i32(o.width),i32(o.width),(i&1u)!=0u);
  if(any(q<vec3i(0))||any(q>=vec3i(UM_D))){continue;}
  let n=umOwnerAt(q);let open=umCapacity(n);if(n.width==0u||open<=1e-5){continue;}
  let phi=umOpenOwnerPhi(n,textureLoad(volume,q,0).x,open);if(phi<0.0){terms[i]=open*phi;weights[i]=open;}
 }
 let weight=((weights[0]+weights[1])+(weights[4]+weights[5]))+(weights[2]+weights[3]);
 let sum=((terms[0]+terms[1])+(terms[4]+terms[5]))+(terms[2]+terms[3]);
 return select(0.5*UM_HMIN*f32(o.width),sum/max(weight,1e-9),weight>0.0);`:`if(umSolidEnabled()&&o.width==1u){return umPressurePhiCell(vec3i(umOrigin(o)));}
 return umOwnerSurfacePhi(o,v,1.0);`}
}
fn umDeficit(o:UMOwner,v:f32,distance:f32)->f32{
 let cap=umCapacity(o);
 if(cap<=1e-5||v>cap||distance>=0.0){return 0.0;}
 return max(0.0,textureLoad(targetFill,vec3i(umOrigin(o)),0).x${this.coarse?"*cap":""}-v);
}
// Diverging transport dilutes bulk liquid and leaves its V in a skin of air
// owners above phi. That stranded V (uncut air owners) is a second budget,
// spent only on deep bulk deficits (uncut, more than a 4h width below the
// surface), so the contraction refilling the diluted liquid draws the skin
// back down. It relaxes with a 0.25 s half-life, not the excess's 1/30 s:
// a resting pool's skin persists for seconds, while a moving body's V lags
// its phi for a few frames. Spent at the excess rate, or on cut and shallow
// owners, the lag contracted transient flow (cm12-figure-9's falling ball
// stretched and its dam front frayed). Cut owners keep the excess-only rate.
fn umFill(o:UMOwner)->f32{return textureLoad(targetFill,vec3i(umOrigin(o)),0).x${this.coarse?"*umCapacity(o)":""};}
fn umStranded(o:UMOwner,v:f32,distance:f32)->f32{
 let cap=umCapacity(o);
 if(cap<=1e-5||distance<0.0||umDetachedMass(o)||umFill(o)>0.0){return 0.0;}
 return uvVolumeCorrectionFractionAt(params.x/7.5)*min(v,cap);
}
fn umBulkDeficit(o:UMOwner,v:f32,distance:f32)->f32{return select(0.0,umDeficit(o,v,distance),umFill(o)>=umCapacity(o)&&distance< -4.0*UM_HMIN);}
// false: the phase-only build (phi and phase; no balance reduction).
override umAuthorityBalance:bool=true;
var<workgroup> sums:array<vec4f,64>;
fn umReduce(l:u32){workgroupBarrier();for(var stride=32u;stride>0u;stride/=2u){if(l<stride){sums[l]+=sums[l+stride];}workgroupBarrier();}}
@compute @workgroup_size(64) fn build(@builtin(global_invocation_id) gid:vec3u,@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) group:vec3u){
 let o=${this.resident?"umResidentAllOwner":"umAllOwner"}(gid);var values=vec4f(0);
 if(o.width!=0u){let origin=vec3i(umOrigin(o));let v=textureLoad(volume,origin,0).x;let distance=umAuthority(o,v);
  phi[o.index]=distance;
  textureStore(phase,origin,vec4f(select(0.0,1.0,distance<0.0||umDetachedMass(o))));
  if(umAuthorityBalance){
  let cap=umCapacity(o);
  values=vec4f(uvVolumeCorrectionAmountAt(v,cap,params.x),umDeficit(o,v,distance),umStranded(o,v,distance),umBulkDeficit(o,v,distance))*f32(o.width*o.width*o.width);
  // Native balance counts open liquid rows only.
  if(umSolidEnabled()&&(cap<=1e-5||distance>=0.0)){values=vec4f(0,0,values.z,0);}
  }
 }
 if(!umAuthorityBalance){return;}
 sums[l]=values;umReduce(l);let index=group.x+umDispatchX*group.y;
 if(l==0u&&index<${this.groups}u){balance[1u+2u*index]=sums[0].xy;balance[2u+2u*index]=sums[0].zw;}
}
@compute @workgroup_size(64) fn chunks(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 var value=vec4f(0);for(var i=group.x*1024u+l;i<min(${this.resident?"umCounts.x+umResidentPageCount()":"(umCounts.x*64u+umCounts.y+63u)/64u"},(group.x+1u)*1024u);i+=64u){value+=vec4f(balance[1u+2u*i],balance[2u+2u*i]);}
 sums[l]=value;umReduce(l);if(l==0u){balance[${1+2*this.groups}u+2u*group.x]=sums[0].xy;balance[${2+2*this.groups}u+2u*group.x]=sums[0].zw;}
}
@compute @workgroup_size(64) fn reduce(@builtin(local_invocation_index) l:u32){
 var value=vec4f(0);for(var i=l;i<${this.chunks}u;i+=64u){value+=vec4f(balance[${1+2*this.groups}u+2u*i],balance[${2+2*this.groups}u+2u*i]);}
 sums[l]=value;umReduce(l);if(l==0u){
  // rate: every deficit, from excess. bulk: what remains of uncut deficits,
  // from stranded V.
  var rate=0.0;var bulk=0.0;let s=sums[0];
  if(params.y>=0.0&&s.y>0.0){rate=min(1.0,s.x/s.y);}
  if(params.y>=0.0&&s.w>0.0){bulk=min(1.0-rate,s.z/s.w);}
  balance[0]=vec2f(rate,bulk);}
}
@compute @workgroup_size(64) fn resolve(@builtin(global_invocation_id) gid:vec3u){
 let o=${this.resident?"umResidentAllOwner":"umAllOwner"}(gid);if(o.width==0u){return;}let origin=vec3i(umOrigin(o));let v=textureLoad(volume,origin,0).x;
 let distance=phi[o.index];
 let amount=uvVolumeCorrectionAmountAt(v,umCapacity(o),params.x)-balance[0].x*umDeficit(o,v,distance)-balance[0].y*umBulkDeficit(o,v,distance);
 textureStore(correction,origin,vec4f(amount/max(params.x,1e-12)));
}
${this.resident?`// One group per page: the 4h owners of an absent page take the dense
// build's phi (V=0 there).
@compute @workgroup_size(64) fn absent(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) l:u32){
 let page=group.x;if(page>=UM_PAGES||umPageResident(page)){return;}
 let tile=umPageTile(page,l);if(tile>=UM_TILES||umTileWidth(tile)!=4u){return;}
 let o=UMOwner(tile,0u,4u,umTopology[tile]&0x3fffffffu);phi[o.index]=umAuthority(o,0.0);
}`:""}
`,counted)});
  const info=await module.getCompilationInfo(),errors=info.messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
  const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.resources,...(this.solid?[this.coarse?this.solid.coarse!.bindLayout:this.solid.bindLayout]:[])]});
  const mode=this.resident?UNIFORM_MIXED_COUNTED.residentAll:UNIFORM_MIXED_COUNTED.all;
  await Promise.all(["build","chunks","reduce","resolve",...(this.coarse?["cut"]:[]),...(this.resident?["absent"]:[])].map(async entryPoint=>{this.pipelines.set(entryPoint,await uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint,constants:{umDispatchX:this.ownership.dispatchX,...(counted.includes(entryPoint)?{umCountedJobs:mode}:{}),...s}}})));}));
  this.pipelines.set("phase",await uniformMixedSolidPipeline(this.solid,s=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint:"build",constants:{umDispatchX:this.ownership.dispatchX,umCountedJobs:mode,umAuthorityBalance:0,...s}}})));
 }
 /** balance=false writes phi and phase only: the extension's authority,
  * when this same stage (same ownership, same origin texels) rebuilds the
  * correction and its balance scratch before their readers (band rows, RHS). */
 private variant(pipeline:GPUComputePipeline):GPUComputePipeline{return this.solid?.select(pipeline)??pipeline;}
 encode(encoder:GPUCommandEncoder,group:GPUBindGroup,balance=true):void{
  if(this.pipelines.size!==5+(this.coarse?1:0)+(this.resident?1:0))throw new Error("Mixed pressure authority is not initialized");
  const pass=encoder.beginComputePass({label:balance?"Uniform mixed pressure authority and volume correction":"Uniform mixed pressure authority phase"});pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);if(this.solid)pass.setBindGroup(2,this.coarse?this.solid.coarse!.bindGroup:this.solid.bindGroup);
  if(this.coarse){pass.setPipeline(this.variant(this.pipelines.get("cut")!));pass.dispatchWorkgroups(Math.max(1,Math.min(1024,Math.ceil(this.ownership.capacity.tiles/64))));}
  if(this.resident){pass.setPipeline(this.variant(this.pipelines.get("absent")!));pass.dispatchWorkgroups(uniformMixedPageCount(this.ownership.capacity.lattice));}
  if(!balance){this.ownership.dispatchAllCounted(pass,this.variant(this.pipelines.get("phase")!));pass.end();return;}
  for(const entry of ["build","chunks","reduce","resolve"]){const pipeline=this.variant(this.pipelines.get(entry)!);pass.setPipeline(pipeline);
   if(entry==="chunks")pass.dispatchWorkgroups(this.chunks);else if(entry==="reduce")pass.dispatchWorkgroups(1);else this.ownership.dispatchAllCounted(pass,pipeline);
  }
  pass.end();
 }
}
