import { UniformPageGeneration } from "./uniform-page-generation";
import { UNIFORM_PHI_REDISTANCE_READ_REACH } from "./uniform-phi-band";

export interface UniformPageSupportFields {
  volume: number;
  phi: number;
  velocity: readonly [number, number, number];
}

/** GPU producer of desired residency, independent of a world-sized catalogue.
 * Retains every nonzero V and every phi below the positive interface band, then
 * adds characteristic/stencil support. This is an allocation policy component,
 * not a proof that the current fluid operators can run with absent pages.
 *
 * Source records must already cover the source's complete interface-band footprint.
 * The caller must supply conservative force/midpoint allowances; start velocity
 * alone is not generally a bound on an RK2 characteristic. Other operators may
 * require a larger read reach than the default phi advection/redistance reach.
 */
export class UniformPageSupport {
  readonly sources: GPUBuffer;
  readonly summaries: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly group: GPUBindGroup;
  private stepConfigured = false;
  private constructor(private readonly device: GPUDevice, readonly pool: UniformPageGeneration,
    private readonly pipelines: readonly GPUComputePipeline[], layout: GPUBindGroupLayout) {
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    this.sources=device.createBuffer({label:"Uniform source support commands",size:pool.layout.requestWords*4,usage:storage});
    this.summaries=device.createBuffer({label:"Uniform resident page support summaries",size:pool.options.capacity*16,usage:storage});
    this.params=device.createBuffer({label:"Uniform support step bounds",size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.dispatch=device.createBuffer({label:"Uniform resident support dispatch",size:12,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.INDIRECT});
    this.group=device.createBindGroup({layout,entries:[pool.accepted,pool.fields,this.summaries,pool.requests,this.sources,this.params]
      .map((buffer,binding)=>({binding,resource:{buffer}}))});
  }
  static async create(device: GPUDevice, pool: UniformPageGeneration, fields: UniformPageSupportFields) {
    const stride=pool.options.initialCell.length;
    if(![fields.volume,fields.phi,...fields.velocity].every(i=>Number.isInteger(i)&&i>=0&&i<stride))
      throw new RangeError("Support field index outside persistent field record");
    const layout=device.createBindGroupLayout({entries:[0,1,2,3,4,5].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,
      buffer:{type:binding===5?"uniform" as const:[0,1,4].includes(binding)?"read-only-storage" as const:"storage" as const}}))});
    const module=device.createShaderModule({label:"Uniform page support requests",code:uniformPageSupportWGSL(pool,fields)});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const pipelines=await Promise.all(["classify","emit"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
    return new UniformPageSupport(device,pool,pipelines,layout);
  }
  /** External step inputs only; no GPU receipt is needed to set these. */
  setStep(spacing: readonly [number,number,number], dt: number, positivePhiBand: number,
    extraTravelCells: readonly [number,number,number], readReachCells=UNIFORM_PHI_REDISTANCE_READ_REACH): void {
    if(!spacing.every(n=>Number.isFinite(n)&&n>0)||!Number.isFinite(dt)||dt<0
      ||!Number.isFinite(positivePhiBand)||positivePhiBand<=0
      ||!extraTravelCells.every(n=>Number.isFinite(n)&&n>=0)||!Number.isFinite(readReachCells)||readReachCells<0)
      throw new RangeError("Invalid page support bounds");
    const data=new Float32Array([...spacing,dt,...extraTravelCells,readReachCells,positivePhiBand,0,0,0]);
    if(!data.every(Number.isFinite)||data[0]===0||data[1]===0||data[2]===0||data[8]===0)throw new RangeError("Support bounds do not fit f32");
    this.device.queue.writeBuffer(this.params,0,data);
    this.stepConfigured = true;
  }
  encode(encoder: GPUCommandEncoder): void {
    if (!this.stepConfigured) throw new Error("Configure external step bounds before generating support");
    encoder.clearBuffer(this.summaries);
    encoder.copyBufferToBuffer(this.pool.accepted,8*4,this.dispatch,0,12);
    const classify=encoder.beginComputePass({label:"Classify resident page support"});
    classify.setPipeline(this.pipelines[0]!);classify.setBindGroup(0,this.group);classify.dispatchWorkgroupsIndirect(this.dispatch,0);classify.end();
    const emit=encoder.beginComputePass({label:"Emit signed frontier requests"});
    emit.setPipeline(this.pipelines[1]!);emit.setBindGroup(0,this.group);emit.dispatchWorkgroups(1);emit.end();
  }
  get allocatedBytes(): number { return this.sources.size + this.summaries.size + this.params.size + this.dispatch.size; }
  destroy(): void {for(const b of [this.sources,this.summaries,this.params,this.dispatch])b.destroy();}
}

function uniformPageSupportWGSL(pool: UniformPageGeneration, fields: UniformPageSupportFields): string {
  const {edge,requestCapacity,initialCell}=pool.options;
  return /* wgsl */ `
struct Bounds{spacingDt:vec4f,extraReach:vec4f,band:vec4f}
@group(0) @binding(0) var<storage,read> accepted:array<u32>;
@group(0) @binding(1) var<storage,read> fields:array<f32>;
@group(0) @binding(2) var<storage,read_write> summary:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> requests:array<u32>;
@group(0) @binding(4) var<storage,read> sources:array<u32>;
@group(0) @binding(5) var<uniform> bounds:Bounds;
const REQUEST_CAP:u32=${requestCapacity}u;
fn finite(v:f32)->bool{return v==v && abs(v)<=3.402823e38;}
@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id)id:vec3u){
 if(id.y>=accepted[1]||id.x>=${edge ** 3}u){return;}
 let slot=accepted[${pool.layout.activeBase}u+id.y];let at=(slot*${edge ** 3}u+id.x)*${initialCell.length}u;
 let v=fields[at+${fields.volume}u];let phi=fields[at+${fields.phi}u];
 let velocity=vec3f(fields[at+${fields.velocity[0]}u],fields[at+${fields.velocity[1]}u],fields[at+${fields.velocity[2]}u]);
 if(!finite(v)||!finite(phi)||!finite(velocity.x)||!finite(velocity.y)||!finite(velocity.z)){
  atomicOr(&summary[4u*slot],4u);return;
 }
 var roles=0u;if(v!=0.0){roles|=1u;}if(phi<bounds.band.x){roles|=2u;}
 atomicOr(&summary[4u*slot],roles);
 let travel=abs(velocity)*bounds.spacingDt.w/bounds.spacingDt.xyz;
 for(var axis=0u;axis<3u;axis++){
  if(!finite(travel[axis])){atomicOr(&summary[4u*slot],4u);return;}
  atomicMax(&summary[4u*slot+1u+axis],bitcast<u32>(travel[axis]));
 }
}
fn fail(reason:u32){requests[0]=REQUEST_CAP+1u;requests[1]=reason;}
fn appendSupport(q:vec3i,roles:u32,travel:vec3f)->bool{
 let radiusF=ceil((travel+bounds.extraReach.xyz+vec3f(bounds.extraReach.w))/f32(${edge}));
 if(any(radiusF>vec3f(f32(REQUEST_CAP)))||any(radiusF!=radiusF)){fail(1u);return false;}
 let radius=vec3i(radiusF);
 for(var z=-radius.z;z<=radius.z;z++){for(var y=-radius.y;y<=radius.y;y++){for(var x=-radius.x;x<=radius.x;x++){
  if(requests[0]>=REQUEST_CAP){fail(1u);return false;}
  let delta=vec3i(x,y,z);var n=q;
  for(var axis=0u;axis<3u;axis++){
   if((delta[axis]>0 && q[axis]>2147483647-delta[axis])||(delta[axis]<0 && q[axis]<(-2147483647-1)-delta[axis])){fail(2u);return false;}
   n[axis]+=delta[axis];
  }
  let at=4u+4u*requests[0];let words=bitcast<vec3u>(n);
  requests[at]=words.x;requests[at+1u]=words.y;requests[at+2u]=words.z;
  requests[at+3u]=select(8u,roles,all(delta==vec3i(0)));requests[0]++;
 }}}return true;
}
@compute @workgroup_size(1) fn emit(){
 requests[0]=0u;requests[1]=0u;
 if(sources[0]>REQUEST_CAP){fail(1u);return;}
 // The maximum over every resident page covers characteristics that sample
 // velocities in neighboring support, not just the liquid seed's own page.
 var travel=vec3f(0);
 for(var i=0u;i<accepted[1];i++){
  let slot=accepted[${pool.layout.activeBase}u+i];
  if((atomicLoad(&summary[4u*slot])&4u)!=0u){fail(3u);return;}
  for(var axis=0u;axis<3u;axis++){travel[axis]=max(travel[axis],bitcast<f32>(atomicLoad(&summary[4u*slot+1u+axis])));}
 }
 for(var i=0u;i<accepted[1];i++){
  let slot=accepted[${pool.layout.activeBase}u+i];let roles=atomicLoad(&summary[4u*slot]);if(roles==0u){continue;}
  let at=16u+16u*slot;let q=bitcast<vec3i>(vec3u(accepted[at],accepted[at+1u],accepted[at+2u]));
  if(!appendSupport(q,roles,travel)){return;}
 }
 for(var i=0u;i<sources[0];i++){
  let at=4u+4u*i;if(sources[at+3u]==0u){continue;}
  let q=bitcast<vec3i>(vec3u(sources[at],sources[at+1u],sources[at+2u]));
  if(!appendSupport(q,sources[at+3u]|16u,travel)){return;}
 }
}
`;
}
