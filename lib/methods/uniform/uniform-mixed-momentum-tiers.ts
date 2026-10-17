import { UNIFORM_DETAIL_RING_4H_LOAD } from "../../core/uniform-detail-abi";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "./uniform-mixed-topology.wgsl";
import { UNIFORM_REGION_HEAT_BUDGET_H, uniformRegionHeatWGSL } from "./uniform-region-heat";

/** Full: every face traces. Retain: a face keeps its extended value.
 * Affine: a face traces the region's affine model (no sampler).
 * Pending: between the decide and affine passes only. */
export const UNIFORM_MOMENTUM_TIER={full:0,retain:1,affine:2,pending:3} as const;
/** f32 words a 16h region of the affine model record: the model's velocity
 * at the region centre (m/s, 3), spare, the gradient (m/s per cell, row a =
 * component, 9), eligible (1, else 0), the 27-region speed bound of this
 * frame (m/s), spare, the RMS residual of each component over the region's
 * own faces (m/s, 3), spare, then the residual of each of the
 * UNIFORM_MOMENTUM_RESIDUAL_PARTS parts of the faces the trace reads (m/s). */
export const UNIFORM_MOMENTUM_MODEL_WORDS=36;
/** Workgroups one region's residual is spread over (latency, not work: a
 * lone region's fit is otherwise one workgroup's serial loads). */
export const UNIFORM_MOMENTUM_RESIDUAL_PARTS=16;
/** Cells (of the region's width) beyond its box that an affine region's
 * residual covers: a one-step departure moves at most 1.5 h from its face
 * centre (the tier's speed bound) and the sampler reads faces within one
 * cell of it, so 2.5 h at width 1 and 1.375 cells at width 4. */
export const UNIFORM_MOMENTUM_AFFINE_HALO=3;
/** The velocity error (m/s) an affine step may make: the displacement a
 * COLD region may hold back (UNIFORM_REGION_HEAT_BUDGET_H h) over one step. */
const affineBudgetWGSL=(dt:string,hmin:string)=>`${UNIFORM_REGION_HEAT_BUDGET_H}*${hmin}/${dt}`;
/** The pending regions worth a residual: eligible, with an own-face RMS
 * residual (a lower bound on the residual) within the affine budget, and the
 * one-substep and gradient conditions the bound needs. The same test gates
 * the residual's gather and the affine decision, so a region the residual
 * skipped is never read stale. */
function affineCandidateWGSL(name:string,models:string):string{const W=UNIFORM_MOMENTUM_MODEL_WORDS;return /* wgsl */`
fn ${name}(region:u32,dt:f32,hmin:f32)->bool{
 let b=${W}u*region;let rms=max(${models}[b+16u],max(${models}[b+17u],${models}[b+18u]));
 let g=vec3f(abs(${models}[b+4u])+abs(${models}[b+5u])+abs(${models}[b+6u]),abs(${models}[b+7u])+abs(${models}[b+8u])+abs(${models}[b+9u]),abs(${models}[b+10u])+abs(${models}[b+11u])+abs(${models}[b+12u]));
 return ${models}[b+13u]==1.0&&rms<=${affineBudgetWGSL("dt","hmin")}&&dt*max(g.x,max(g.y,g.z))<=hmin&&dt*(${models}[b+14u]+rms)<=1.5*hmin;
}`;}
/** The momentum's affine fit and residual grids: a lane tests one (region,
 * part) item, so an idle frame costs each workgroup one gather; consecutive
 * regions land in different workgroups. */
export function uniformMomentumFitGrid(regions:number,parts:number):number{return Math.min(65535,Math.ceil(parts*regions/256));}
/** The tiers' own pass (decide, the momentum's fit, affine). */
export const UNIFORM_MOMENTUM_TIER_PASS:GPUComputePassDescriptor={label:"Uniform mixed momentum tiers"};

/** The momentum tier of every 16h region, from the frame's region work heat
 * (UniformRegionHeat), decided on the GPU each frame before momentum reads
 * it.
 *
 * COLD: retain. Every face keeps its extended value (the heat holds a COLD
 * region's displacement within UNIFORM_REGION_HEAT_BUDGET_H h over the
 * frames it stays COLD, and triggers on bodies, sources, identity and
 * unknown evidence).
 *
 * WARM: affine where the error is bounded, else full. A WARM region whose
 * 27 regions are all stored (no wall, SOLID or AIR page: no departure is
 * held) at one width has a least squares model M(p) = u0 + G (p - centre)
 * of its own faces; r is max |u - M| over every face its traces can read
 * (its own and a halo of UNIFORM_MOMENTUM_AFFINE_HALO cells, measured). A
 * face then runs the momentum's one midpoint step on M: with
 * dt (max|u| + r) <= 1.5 h both traces take that one step, and the sampler
 * reproduces affine fields (a convex combination with linear precision), so
 * a step perturbed by r per sample moves the departure at most
 * dt r (1 + k) e^{2k} / h, k = dt max_a sum_b |G_ab| / h <= 1, and the value
 * differs by at most e = r (1 + k (1 + k) e^{2k}). The region is affine
 * when dt e <= UNIFORM_REGION_HEAT_BUDGET_H h: the displacement error of
 * one step stays within what a COLD region may hold back.
 *
 * HOT: full. */
export class UniformMixedMomentumTiers {
  /** The tier a region (u32). */
  readonly record:GPUBuffer;
  /** The affine model record (UNIFORM_MOMENTUM_MODEL_WORDS f32 a region). */
  readonly models:GPUBuffer;
  readonly allocatedBytes:number;
  private readonly layout:GPUBindGroupLayout;
  private decidePipeline?:GPUComputePipeline;
  private affinePipeline?:GPUComputePipeline;
  private readonly regions:number;
  private readonly groups:number;
  /** heat: the frame's region work tiers (UniformRegionHeat.tiers). */
  constructor(private readonly device:GPUDevice,private readonly ownership:UniformMixedOwnership,private readonly heat:GPUBuffer){
    this.regions=ownership.capacity.tiles/64;
    this.groups=Math.min(65535,Math.ceil(this.regions/64));
    this.record=device.createBuffer({label:"Uniform mixed momentum tiers",size:4*this.regions,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.models=device.createBuffer({label:"Uniform mixed momentum affine models",size:4*UNIFORM_MOMENTUM_MODEL_WORDS*this.regions,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.allocatedBytes=this.record.size+this.models.size;
    this.layout=device.createBindGroupLayout({label:"Uniform mixed momentum tiers",entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
  }
  /** params: the momentum parameters (h.xyz, dt). */
  bind(params:GPUBuffer):GPUBindGroup{
    return this.device.createBindGroup({label:"Uniform mixed momentum tiers",layout:this.layout,entries:[
      {binding:0,resource:{buffer:this.record}},{binding:1,resource:{buffer:this.ownership.regionVelocity}},{binding:2,resource:{buffer:this.heat}},
      {binding:3,resource:{buffer:params,size:32}},{binding:5,resource:{buffer:this.models}},
    ]});
  }
  async initialize():Promise<void>{
    const T=UNIFORM_MOMENTUM_TIER,W=UNIFORM_MOMENTUM_MODEL_WORDS;
    const code=uniformMixedTopologyWGSL(this.ownership.capacity,0)+uniformRegionHeatWGSL(1,2,this.regions)+/* wgsl */`
@group(1) @binding(0) var<storage,read_write> tiers:array<u32>;
@group(1) @binding(1) var<storage,read> regionVelocity:array<u32>;
struct UMMomentumParams {hDt:vec4f,flags:vec4u}
@group(1) @binding(3) var<uniform> momentum:UMMomentumParams;
@group(1) @binding(5) var<storage,read_write> models:array<f32>;
const MT_MAX:f32=3.402823e38;
// The plan's float-flip order (uniform-mixed-frame-plan.ts umVelocityOrder) undone.
fn mtDecode(w:u32)->f32{return bitcast<f32>(select(~w,w^0x80000000u,(w&0x80000000u)!=0u));}
${affineCandidateWGSL("mtCandidate","models")}
@compute @workgroup_size(64) fn decide(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 for(var region=gid.x;region<UM_PAGES;region+=64u*groups.x){
  // The momentum's fit sets eligible again where it models this region.
  models[${W}u*region+13u]=0.0;
  let heat=umRegionTier(region);
  // COLD retains every frame: the departure takes this frame's extended
  // value, which the last projection corrected. Holding the first COLD
  // frame's departure instead froze the pressure problem that frame posed.
  if(heat==UM_TIER_COLD){tiers[region]=${T.retain}u;continue;}
  if(heat!=UM_TIER_WARM){tiers[region]=${T.full}u;continue;}
  // The 27 regions' speed bound (an unstored or empty one is at rest).
  var speed=0.0;let origin=16u*umPageCoord(region);
  for(var k=0u;k<27u;k++){
   let tile=umTileKey(vec3i(origin/4u)+4*(vec3i(umCorner(k,3u))-vec3i(1)));if(!umKeyStored(tile)){continue;}
   let n=umPageOfTile(tile);
   for(var j=0u;j<6u;j++){let w=regionVelocity[6u*n+j];if(w==0xffffffffu){speed=MT_MAX*0.5;}else if(w!=0u){speed=max(speed,abs(mtDecode(w)));}}
  }
  models[${W}u*region+14u]=speed;tiers[region]=${T.pending}u;
 }
}
@compute @workgroup_size(64) fn affine(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) groups:vec3u){
 let h=momentum.hDt.xyz;let dt=momentum.hDt.w;let hmin=min(h.x,min(h.y,h.z));
 for(var region=gid.x;region<UM_PAGES;region+=64u*groups.x){
  if(tiers[region]!=${T.pending}u){continue;}
  let b=${W}u*region;var fail=!mtCandidate(region,dt,hmin);
  // r: the fit's residual over every face the trace can read.
  var r=0.0;
  for(var part=0u;part<${UNIFORM_MOMENTUM_RESIDUAL_PARTS}u;part++){let q=models[b+20u+part];fail=fail||!(q<=MT_MAX);r=max(r,q);}
  let g=vec3f(abs(models[b+4u])+abs(models[b+5u])+abs(models[b+6u]),abs(models[b+7u])+abs(models[b+8u])+abs(models[b+9u]),abs(models[b+10u])+abs(models[b+11u])+abs(models[b+12u]));
  let kappa=dt*max(g.x,max(g.y,g.z))/hmin;let speed=models[b+14u];
  fail=fail||!(r<=MT_MAX)||!(kappa<=1.0)||!(dt*(speed+r)<=1.5*hmin);
  let e=r*(1.0+kappa*(1.0+kappa)*exp(2.0*kappa))+64.0*1.1920929e-7*(speed+r);
  tiers[region]=select(${T.full}u,${T.affine}u,!fail&&e<=${affineBudgetWGSL("dt","hmin")});
 }
}`;
    const module=this.device.createShaderModule({label:"Uniform mixed momentum tiers",code});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>`${m.lineNum}: ${m.message}`).join("\n"));
    const layout=this.device.createPipelineLayout({bindGroupLayouts:[this.ownership.bindLayout,this.layout]});
    [this.decidePipeline,this.affinePipeline]=await Promise.all(["decide","affine"].map(entryPoint=>this.device.createComputePipelineAsync({layout,compute:{module,entryPoint}})));
  }
  /** One lane a region, in the momentum's tier pass ahead of its fit. */
  encodeDecide(pass:GPUComputePassEncoder,group:GPUBindGroup):void{
    if(!this.decidePipeline)throw new Error("Momentum tiers are not initialized");
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setPipeline(this.decidePipeline);pass.dispatchWorkgroups(this.groups);
  }
  /** After the momentum's fit and residual, ahead of its launches. */
  encodeAffine(pass:GPUComputePassEncoder,group:GPUBindGroup):void{
    if(!this.affinePipeline)throw new Error("Momentum tiers are not initialized");
    pass.setBindGroup(0,this.ownership.bindGroup);pass.setBindGroup(1,group);pass.setPipeline(this.affinePipeline);pass.dispatchWorkgroups(this.groups);
  }
  destroy():void{this.record.destroy();this.models.destroy();}
}
/** The momentum's read of the tiers (bound read-only at `binding`) and of
 * the affine models (read-write at `models`: the fit writes them). */
export function uniformMomentumTierWGSL(binding:number,models:number):string{const W=UNIFORM_MOMENTUM_MODEL_WORDS,T=UNIFORM_MOMENTUM_TIER;return /* wgsl */`
@group(1) @binding(${binding}) var<storage,read> umTiers:array<u32>;
@group(1) @binding(${models}) var<storage,read_write> umModels:array<f32>;
fn umMomentumRetained(owner:UMOwner)->bool{return umTiers[umPageOfTile(owner.tile)]==${T.retain}u;}
fn umMomentumAffine(owner:UMOwner)->bool{return umTiers[umPageOfTile(owner.tile)]==${T.affine}u;}
// The region's model at p (cells).
fn umAffineAt(region:u32,p:vec3f)->vec3f{
 let b=${W}u*region;let x=p-(16.0*vec3f(umPageCoord(region))+vec3f(8.0));
 return vec3f(umModels[b],umModels[b+1u],umModels[b+2u])+vec3f(
  dot(vec3f(umModels[b+4u],umModels[b+5u],umModels[b+6u]),x),
  dot(vec3f(umModels[b+7u],umModels[b+8u],umModels[b+9u]),x),
  dot(vec3f(umModels[b+10u],umModels[b+11u],umModels[b+12u]),x));
}
// The momentum's one midpoint step (uniformVelocityDepartureWGSL: the tier
// admits only a single substep, and no departure is held) on the model.
fn umAffineMomentum(owner:UMOwner,face:UMFace)->f32{
 let region=umPageOfTile(owner.tile);let h=momentum.hDt.xyz;let dt=momentum.hDt.w;let p=umFaceCenter(face);
 let at=p-0.5*umAffineAt(region,p)*dt/h;let point=p-umAffineAt(region,at)*dt/h;
 return umAffineAt(region,point)[face.axis];
}
var<workgroup> umFitSums:array<vec4f,256>;
var<workgroup> umFitSquares:array<f32,256>;
${affineCandidateWGSL("umAffineCandidate","umModels")}
var<workgroup> umFitList:array<u32,256>;
var<workgroup> umFitCount:atomic<u32>;
var<workgroup> umFitFlag:atomic<u32>;
var<workgroup> umFitTotal:u32;
// A cell's positive face (cell at its low corner, width 1 or 4): its value
// and centre (cells from the region centre).
fn umFitFace(cell:vec3i,axis:u32,width:u32,centre:vec3f)->vec4f{
 if(width==4u){
  var anchor=cell;anchor[axis]+=3;var p=vec3f(cell)+vec3f(2.0);p[axis]=f32(cell[axis]+4);
  return vec4f(p-centre,${UNIFORM_DETAIL_RING_4H_LOAD}textureLoad(extended,anchor,0)[axis]);
 }
 var p=vec3f(cell)+vec3f(0.5);p[axis]=f32(cell[axis]+1);
 return vec4f(p-centre,textureLoad(extended,cell,0)[axis]);
}
// The (region, part) items of pending regions among a workgroup's 256: item
// = group + groups x (lane + 256 round), region = item % pages.
fn umFitGather(round:u32,group:u32,groups:u32,lane:u32,parts:u32,modelled:bool)->u32{
 if(lane==0u){atomicStore(&umFitCount,0u);}
 workgroupBarrier();
 let item=group+groups*(lane+256u*round);
 if(item<parts*UM_PAGES&&umTiers[item%UM_PAGES]==${T.pending}u){
  let h=momentum.hDt.xyz;
  if(!modelled||umAffineCandidate(item%UM_PAGES,momentum.hDt.w,min(h.x,min(h.y,h.z)))){umFitList[atomicAdd(&umFitCount,1u)]=item;}
 }
 workgroupBarrier();
 if(lane==0u){umFitTotal=atomicLoad(&umFitCount);}
 return workgroupUniformLoad(&umFitTotal);
}
fn umFitRegion(region:u32)->vec3u{let origin=16u*umPageCoord(region);return vec3u(origin.x,origin.y,origin.z);}
// The affine model of each pending region (UniformMixedMomentumTiers), a
// workgroup per (region, component); part 3 checks eligibility. The faces
// of a component form a regular n^3 grid (n = 16 / width, spacing width), so
// about the grid's own centre the least squares normal equations are
// diagonal: c = mean v, g_j = sum v x_j / sum x_j^2.
@compute @workgroup_size(256) fn momentumAffineFit(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 for(var round=0u;256u*groups.x*round<4u*UM_PAGES;round++){
  let count=umFitGather(round,group.x,groups.x,lane,4u,false);
  for(var j=0u;j<count;j++){
   let item=workgroupUniformLoad(&umFitList[j]);let region=item%UM_PAGES;let part=item/UM_PAGES;
   let origin=umFitRegion(region);let width=umTileWidth(umPageTile(region,0u));let centre=vec3f(origin)+vec3f(8.0);
   let n=16u/width;var sums=vec4f(0);var squares=0.0;
   if(lane==0u){atomicStore(&umFitFlag,0u);}
   workgroupBarrier();
   if(part==3u){
    // Eligibility: the 27 regions stored, every tile at the centre's width.
    for(var k=lane;k<27u*64u;k+=256u){
     let d=vec3i(i32((k/64u)%3u)-1,i32((k/192u)%3u)-1,i32(k/576u)-1);
     let key=umTileKey(vec3i(origin/4u)+4*d);
     if(!umKeyStored(key)||umTileWidth(umPageTile(umPageOfTile(key),k%64u))!=width){atomicOr(&umFitFlag,1u);}
    }
   }else{
    // The grid's centre is the region's, but half a width up its own axis.
    var shift=vec3f(0);shift[part]=0.5*f32(width);
    for(var k=lane;k<n*n*n;k+=256u){
     let s=umFitFace(vec3i(origin)+i32(width)*vec3i(umCorner(k,n)),part,width,centre);
     sums+=s.w*vec4f(1.0,s.xyz-shift);squares+=s.w*s.w;
    }
   }
   umFitSums[lane]=sums;umFitSquares[lane]=squares;
   for(var stride=128u;stride>0u;stride>>=1u){
    workgroupBarrier();
    if(lane<stride){umFitSums[lane]+=umFitSums[lane+stride];umFitSquares[lane]+=umFitSquares[lane+stride];}
   }
   workgroupBarrier();
   let b=${W}u*region;
   if(lane==0u&&part==3u){umModels[b+13u]=select(0.0,1.0,atomicLoad(&umFitFlag)==0u);}
   if(lane<3u&&part<3u){
    let q=umFitSums[0];let w=f32(width);
    // sum x_j^2 over the grid: n^2 lines of n points at spacing w.
    let second=f32(n*n)*w*w*f32(n*(n*n-1u))/12.0;let count=f32(n*n*n);let g=q.yzw/second;
    // The own faces' RMS residual: sum v^2 - (sum v)^2 / N - |sum v x|^2 / S.
    if(lane==0u){umModels[b+16u+part]=sqrt(max(umFitSquares[0]-q.x*q.x/count-dot(q.yzw,q.yzw)/second,0.0)/count);}
    // The model about the region centre: c at the shifted grid centre.
    if(lane==0u){umModels[b+part]=q.x/count-0.5*w*g[part];}
    umModels[b+4u+3u*part+lane]=g[lane];
   }
   workgroupBarrier();
  }
 }
}
// The residual of each pending region's model over the faces its trace
// reads (its own and a halo of ${UNIFORM_MOMENTUM_AFFINE_HALO} cells of its width, all within its 27
// regions), a workgroup per (region, part) of those faces.
@compute @workgroup_size(256) fn momentumAffineResidual(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) groups:vec3u,@builtin(local_invocation_index) lane:u32){
 const parts=${UNIFORM_MOMENTUM_RESIDUAL_PARTS}u;
 for(var round=0u;256u*groups.x*round<parts*UM_PAGES;round++){
  let count=umFitGather(round,group.x,groups.x,lane,parts,true);
  for(var j=0u;j<count;j++){
   let item=workgroupUniformLoad(&umFitList[j]);let region=item%UM_PAGES;let part=item/UM_PAGES;
   let origin=umFitRegion(region);let width=umTileWidth(umPageTile(region,0u));let centre=vec3f(origin)+vec3f(8.0);
   let b=${W}u*region;
   if(lane==0u){atomicStore(&umFitFlag,0u);}
   workgroupBarrier();
   let side=16u/width+2u*${UNIFORM_MOMENTUM_AFFINE_HALO}u;let low=vec3i(origin)-vec3i(i32(width*${UNIFORM_MOMENTUM_AFFINE_HALO}u));
   let total=side*side*side;let chunk=(total+parts-1u)/parts;let end=min(total,(part+1u)*chunk);
   var residual=0.0;
   // A component at a time: its model words are uniform scalars.
   for(var a=0u;a<3u;a++){
    let c=umModels[b+a];let g=vec3f(umModels[b+4u+3u*a],umModels[b+5u+3u*a],umModels[b+6u+3u*a]);
    for(var k=part*chunk+lane;k<end;k+=256u){
     let s=umFitFace(low+i32(width)*vec3i(umCorner(k,side)),a,width,centre);
     let gap=abs(s.w-c-dot(g,s.xyz));residual=select(3.402823e38,max(residual,gap),gap<=3.402823e38);
    }
   }
   atomicMax(&umFitFlag,bitcast<u32>(residual));
   workgroupBarrier();
   if(lane==0u){umModels[b+20u+part]=bitcast<f32>(atomicLoad(&umFitFlag));}
   workgroupBarrier();
  }
 }
}
`;}
