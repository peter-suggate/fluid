import { UniformPageGeneration } from "./uniform-page-generation";
import { UNIFORM_PHI_REDISTANCE_READ_REACH } from "./uniform-phi-band";

export interface UniformPageSupportFields {
  volume: number;
  phi: number;
  velocity: readonly [number, number, number];
  /** V's units must be explicit; open-cell fractions also require the aperture field. */
  volumeKind: "physical" | "full-cell-fraction" | "open-cell-fraction";
  openCellFraction?: number;
}

/** GPU producer of desired residency, independent of a world-sized catalogue.
 * Summarizes canonical occupied cells and 4³ tiles, then closes local forward
 * destination and backward-read bounds over all intersecting resident velocities.
 * Retains every positive V and every phi below the positive interface band. This is an allocation policy component,
 * not a proof that the current fluid operators can run with absent pages.
 *
 * Missing pages use the pool's declared ambient initial velocity for prediction.
 * Consumers still need actual sample-validity checks before committing a step;
 * this predictor does not certify an evolving velocity extension or pressure solve.
 * Request header words 1/2 report fault / maximum closure iterations. Faults:
 * 1 request capacity, 2 signed coordinate overflow, 3 invalid fields, 4 closure limit.
 *
 * Source records must already cover the source's complete interface-band footprint.
 * The caller must supply conservative force/midpoint allowances; start velocity
 * alone is not generally a bound on an RK2 characteristic. Other operators may
 * require a larger read reach than the default phi advection/redistance reach.
 */
export class UniformPageSupport {
  /** Per-slot words: roles, occupied lo/hi (half-open), negative/positive travel,
   * physical volume, max fraction, tile count, interface lo/hi, disagreement count,
   * seed cell count, reserved through word 31, then 4³ tile masks. Retired slots
   * are not cleared: consumers must use the accepted generation's compact list. */
  readonly summaryWords: number;
  readonly sources: GPUBuffer;
  readonly summaries: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly dispatch: GPUBuffer;
  private readonly dispatchArgs: GPUBuffer;
  private readonly group: GPUBindGroup;
  private stepConfigured = false;
  private ambientPhi = 0;
  private constructor(private readonly device: GPUDevice, readonly pool: UniformPageGeneration,
    private readonly pipelines: readonly GPUComputePipeline[], layout: GPUBindGroupLayout) {
    const storage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
    this.sources=device.createBuffer({label:"Uniform source support commands",size:pool.layout.requestWords*4,usage:storage});
    this.summaryWords=32+(pool.options.edge/4)**3/32;
    this.summaries=device.createBuffer({label:"Uniform resident page support summaries",size:pool.options.capacity*this.summaryWords*4,usage:storage});
    this.params=device.createBuffer({label:"Uniform support step bounds",size:48,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    this.dispatch=device.createBuffer({label:"Uniform resident support dispatch",size:12,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.INDIRECT});
    this.dispatchArgs=device.createBuffer({label:"Uniform support dispatch writer",size:12,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    this.group=device.createBindGroup({layout,entries:[pool.accepted,pool.fields,this.summaries,pool.requests,this.sources,this.params,this.dispatchArgs]
      .map((buffer,binding)=>({binding,resource:{buffer}}))});
  }
  static async create(device: GPUDevice, pool: UniformPageGeneration, fields: UniformPageSupportFields) {
    const stride=pool.options.initialCell.length;
    if(![fields.volume,fields.phi,...fields.velocity,...(fields.openCellFraction===undefined?[]:[fields.openCellFraction])].every(i=>Number.isInteger(i)&&i>=0&&i<stride))
      throw new RangeError("Support field index outside persistent field record");
    if(!["physical","full-cell-fraction","open-cell-fraction"].includes(fields.volumeKind)
      ||(fields.volumeKind==="open-cell-fraction" && fields.openCellFraction===undefined))
      throw new RangeError("Specify the physical volume convention and open-cell fraction adapter");
    if(pool.options.initialCell[fields.volume]!==0 || !(pool.options.initialCell[fields.phi]!>0))
      throw new RangeError("New support pages must initialize to mass-free ambient air");
    const layout=device.createBindGroupLayout({entries:[0,1,2,3,4,5,6].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,
      buffer:{type:binding===5?"uniform" as const:[0,1,4].includes(binding)?"read-only-storage" as const:"storage" as const}}))});
    const module=device.createShaderModule({label:"Uniform page support requests",code:uniformPageSupportWGSL(pool,fields)});
    const diagnostics=await module.getCompilationInfo();
    const errors=diagnostics.messages.filter(message=>message.type==="error");
    if(errors.length)throw new Error(errors.map(message=>`${message.lineNum}: ${message.message}`).join("\n"));
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const pipelines=await Promise.all(["prepare","classify","emit"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
    const support=new UniformPageSupport(device,pool,pipelines,layout);
    support.ambientPhi=pool.options.initialCell[fields.phi]!;
    return support;
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
    const cellVolume=Math.fround(Math.fround(data[0]!*data[1]!)*data[2]!);
    if(!Number.isFinite(cellVolume)||cellVolume<=0)throw new RangeError("Cell volume does not fit f32");
    if(data[8]!>this.ambientPhi)throw new RangeError("Ambient phi would recursively seed support");
    this.device.queue.writeBuffer(this.params,0,data);
    this.stepConfigured = true;
  }
  encode(encoder: GPUCommandEncoder): void {
    if (!this.stepConfigured) throw new Error("Configure external step bounds before generating support");
    const prepare=encoder.beginComputePass({label:"Prepare resident support dispatch"});
    prepare.setPipeline(this.pipelines[0]!);prepare.setBindGroup(0,this.group);prepare.dispatchWorkgroups(1);prepare.end();
    encoder.copyBufferToBuffer(this.dispatchArgs,0,this.dispatch,0,12);
    const classify=encoder.beginComputePass({label:"Summarize occupied cells and local motion"});
    classify.setPipeline(this.pipelines[1]!);classify.setBindGroup(0,this.group);classify.dispatchWorkgroupsIndirect(this.dispatch,0);classify.end();
    const emit=encoder.beginComputePass({label:"Close local trajectory support"});
    emit.setPipeline(this.pipelines[2]!);emit.setBindGroup(0,this.group);emit.dispatchWorkgroups(1);emit.end();
  }

  get allocatedBytes(): number { return this.sources.size + this.summaries.size + this.params.size + this.dispatch.size + this.dispatchArgs.size; }
  destroy(): void {for(const b of [this.sources,this.summaries,this.params,this.dispatch,this.dispatchArgs])b.destroy();}
}

function uniformPageSupportWGSL(pool: UniformPageGeneration, fields: UniformPageSupportFields): string {
  const {edge,requestCapacity,initialCell}=pool.options;
  const maskWords=(edge/4)**3/32, summaryWords=32+maskWords;
  const aperture=fields.volumeKind==="open-cell-fraction"?`fields[at+${fields.openCellFraction}u]`:"1.0";
  const physical=fields.volumeKind==="physical"?"v":"v*open*cellVolume";
  const initialVelocity=fields.velocity.map(i=>`${initialCell[i]!}`).map(n=>n.includes(".")||n.includes("e")?n:n+".0").join(",");
  return /* wgsl */ `
struct Bounds{spacingDt:vec4f,extraReach:vec4f,band:vec4f}
@group(0) @binding(0) var<storage,read> accepted:array<u32>;
@group(0) @binding(1) var<storage,read> fields:array<f32>;
@group(0) @binding(2) var<storage,read_write> summary:array<u32>;
@group(0) @binding(3) var<storage,read_write> requests:array<u32>;
@group(0) @binding(4) var<storage,read> sources:array<u32>;
@group(0) @binding(5) var<uniform> bounds:Bounds;
@group(0) @binding(6) var<storage,read_write> dispatch:array<u32>;
const REQUEST_CAP:u32=${requestCapacity}u;
const EDGE:u32=${edge}u;
const SUMMARY:u32=${summaryWords}u;
struct Reduction {
 lo:vec3u, hi:vec3u, interfaceLo:vec3u, interfaceHi:vec3u,
 negative:vec3f, positive:vec3f, volume:f32, maximum:f32,
 roles:u32, disagreement:u32, cells:u32,
}
var<workgroup> reduction:array<Reduction,64>;
var<workgroup> tileMask:array<atomic<u32>,${maskWords}>;
fn finite(v:f32)->bool{return v==v && abs(v)<=3.402823e38;}
@compute @workgroup_size(1) fn prepare(){dispatch[0]=accepted[1];dispatch[1]=1u;dispatch[2]=1u;}
@compute @workgroup_size(64) fn classify(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){
 let slot=accepted[${pool.layout.activeBase}u+group.x];
 if(lane<${maskWords}u){atomicStore(&tileMask[lane],0u);}
 workgroupBarrier();
 var r:Reduction;r.lo=vec3u(EDGE);r.interfaceLo=vec3u(EDGE);
 let cellVolume=bounds.spacingDt.x*bounds.spacingDt.y*bounds.spacingDt.z;
 for(var cell=lane;cell<${edge**3}u;cell+=64u){
  let at=(slot*${edge**3}u+cell)*${initialCell.length}u;
  let v=fields[at+${fields.volume}u];let phi=fields[at+${fields.phi}u];let open=${aperture};
  let velocity=vec3f(fields[at+${fields.velocity[0]}u],fields[at+${fields.velocity[1]}u],fields[at+${fields.velocity[2]}u]);
  let travel=velocity*bounds.spacingDt.w/bounds.spacingDt.xyz;
  if(!finite(v)||v<0.0||!finite(phi)||!finite(open)||open<0.0||open>1.0
    ||!finite(travel.x)||!finite(travel.y)||!finite(travel.z)
    ||!finite(velocity.x)||!finite(velocity.y)||!finite(velocity.z)){r.roles|=4u;continue;}
  r.negative=max(r.negative,max(-travel,vec3f(0)));r.positive=max(r.positive,max(travel,vec3f(0)));
  let physical=${physical};r.volume+=physical;r.maximum=max(r.maximum,physical/cellVolume);
  let p=vec3u(cell%EDGE,(cell/EDGE)%EDGE,cell/(EDGE*EDGE));
  var roles=0u;if(v>0.0){roles|=1u;}if(phi<bounds.band.x){roles|=2u;}
  if(phi<0.0 && v==0.0){r.disagreement++;}
  if(abs(phi)<bounds.band.x){r.interfaceLo=min(r.interfaceLo,p);r.interfaceHi=max(r.interfaceHi,p+vec3u(1));}
  if(roles!=0u){
   r.roles|=roles;r.lo=min(r.lo,p);r.hi=max(r.hi,p+vec3u(1));r.cells++;
   let t=p/4u;let tile=t.x+${edge/4}u*(t.y+${edge/4}u*t.z);
   atomicOr(&tileMask[tile/32u],1u<<(tile%32u));
  }
 }
 reduction[lane]=r;workgroupBarrier();
 for(var width=32u;width>0u;width/=2u){
  if(lane<width){let b=reduction[lane+width];var a=reduction[lane];
   a.lo=min(a.lo,b.lo);a.hi=max(a.hi,b.hi);a.interfaceLo=min(a.interfaceLo,b.interfaceLo);a.interfaceHi=max(a.interfaceHi,b.interfaceHi);
   a.negative=max(a.negative,b.negative);a.positive=max(a.positive,b.positive);
   a.volume+=b.volume;a.maximum=max(a.maximum,b.maximum);a.roles|=b.roles;a.disagreement+=b.disagreement;a.cells+=b.cells;
   reduction[lane]=a;
  }workgroupBarrier();
 }
 let base=slot*SUMMARY;
 if(lane==0u){for(var i=0u;i<32u;i++){summary[base+i]=0u;}}
 if(lane<${maskWords}u){summary[base+32u+lane]=atomicLoad(&tileMask[lane]);}
 if(lane==0u){let a=reduction[0];summary[base]=a.roles;
  if(!finite(a.volume)||!finite(a.maximum)){summary[base]|=4u;}
  for(var axis=0u;axis<3u;axis++){
   summary[base+1u+axis]=a.lo[axis];summary[base+4u+axis]=a.hi[axis];
   summary[base+7u+axis]=bitcast<u32>(a.negative[axis]);summary[base+10u+axis]=bitcast<u32>(a.positive[axis]);
   summary[base+16u+axis]=a.interfaceLo[axis];summary[base+19u+axis]=a.interfaceHi[axis];
  }
  summary[base+13u]=bitcast<u32>(a.volume);summary[base+14u]=bitcast<u32>(a.maximum);
  var tiles=0u;for(var i=0u;i<${maskWords}u;i++){tiles+=countOneBits(atomicLoad(&tileMask[i]));}
  summary[base+15u]=tiles;summary[base+22u]=a.disagreement;summary[base+23u]=a.cells;
 }
}
fn fail(reason:u32){requests[0]=REQUEST_CAP+1u;requests[1]=reason;}
fn coord(slot:u32)->vec3i{let at=16u+16u*slot;return bitcast<vec3i>(vec3u(accepted[at],accepted[at+1u],accepted[at+2u]));}
struct Box{lo:vec3i,hi:vec3i,valid:bool}
// Cells are half-open [lo,hi). The upper page endpoint is inclusive. Integer
// page keys never pass through f32, even for a source millions of pages away.
fn pageBox(q:vec3i,lo:vec3f,hi:vec3f)->Box{
 let low=floor(lo/f32(EDGE));let high=ceil(hi/f32(EDGE))-vec3f(1);
 if(any(abs(low)>vec3f(f32(REQUEST_CAP)))||any(abs(high)>vec3f(f32(REQUEST_CAP)))
   ||any(low!=low)||any(high!=high)){fail(1u);return Box(q,q,false);}
 let l=vec3i(low);let h=vec3i(high);
 for(var axis=0u;axis<3u;axis++){
  if((l[axis]<0 && q[axis]<(-2147483647-1)-l[axis])||(h[axis]>0 && q[axis]>2147483647-h[axis])){fail(2u);return Box(q,q,false);}
 }
 return Box(q+l,q+h,true);
}
fn appendBox(box:Box,roles:u32)->bool{
 // Iterate offsets so an upper coordinate of INT_MAX never wraps the loop.
 let span=vec3u(box.hi-box.lo)+vec3u(1);
 if(any(span>vec3u(REQUEST_CAP))||span.x>REQUEST_CAP/span.y||span.x*span.y>REQUEST_CAP/span.z){fail(1u);return false;}
 for(var z=0u;z<span.z;z++){for(var y=0u;y<span.y;y++){for(var x=0u;x<span.x;x++){
  if(requests[0]>=REQUEST_CAP){fail(1u);return false;}
  let n=bitcast<vec3u>(box.lo+vec3i(vec3u(x,y,z)));let at=4u+4u*requests[0];
  requests[at]=n.x;requests[at+1u]=n.y;requests[at+2u]=n.z;requests[at+3u]=roles;requests[0]++;
 }}}return true;
}
fn appendSupport(q:vec3i,roles:u32,lo:vec3f,hi:vec3f)->bool{
 let ambient=vec3f(${initialVelocity})*bounds.spacingDt.w/bounds.spacingDt.xyz;
 var negative=max(-ambient,vec3f(0));var positive=max(ambient,vec3f(0));
 // The caller supplies actual-stage uncertainty. Forward destinations and their
 // backward queries are both covered; a forward sweep alone is insufficient.
 let reach=2.0*bounds.extraReach.xyz+vec3f(bounds.extraReach.w);
 var box=pageBox(q,lo-negative-positive-reach,hi+positive+negative+reach);
 if(!box.valid){return false;}
 var converged=false;
 for(var iteration=0u;iteration<8u;iteration++){
  requests[2]=max(requests[2],iteration+1u);
  for(var i=0u;i<accepted[1];i++){
   let slot=accepted[${pool.layout.activeBase}u+i];let n=coord(slot);
   if(any(n<box.lo)||any(n>box.hi)){continue;}
   let base=SUMMARY*slot;
   for(var axis=0u;axis<3u;axis++){
    negative[axis]=max(negative[axis],bitcast<f32>(summary[base+7u+axis]));
    positive[axis]=max(positive[axis],bitcast<f32>(summary[base+10u+axis]));
   }
  }
  let next=pageBox(q,lo-negative-positive-reach,hi+positive+negative+reach);
  if(!next.valid){return false;}
  if(all(next.lo==box.lo)&&all(next.hi==box.hi)){converged=true;break;}
  box=next;
 }
 if(!converged){fail(4u);return false;}
 if(!appendBox(box,8u)){return false;}
 let destination=pageBox(q,lo-negative-bounds.extraReach.xyz,hi+positive+bounds.extraReach.xyz);
 if(!destination.valid||!appendBox(destination,32u)){return false;}
 return appendBox(Box(q,q,true),roles);
}
@compute @workgroup_size(1) fn emit(){
 requests[0]=0u;requests[1]=0u;requests[2]=0u;requests[3]=0u;
 if(sources[0]>REQUEST_CAP){fail(1u);return;}
 for(var i=0u;i<accepted[1];i++){
  let slot=accepted[${pool.layout.activeBase}u+i];if((summary[SUMMARY*slot]&4u)!=0u){fail(3u);return;}
 }
 for(var i=0u;i<accepted[1];i++){
  let slot=accepted[${pool.layout.activeBase}u+i];let base=SUMMARY*slot;let roles=summary[base];if(roles==0u){continue;}
  let lo=vec3f(vec3u(summary[base+1u],summary[base+2u],summary[base+3u]));
  let hi=vec3f(vec3u(summary[base+4u],summary[base+5u],summary[base+6u]));
  if(!appendSupport(coord(slot),roles,lo,hi)){return;}
 }
 for(var i=0u;i<sources[0];i++){
  let at=4u+4u*i;if(sources[at+3u]==0u){continue;}
  let q=bitcast<vec3i>(vec3u(sources[at],sources[at+1u],sources[at+2u]));
  if(!appendSupport(q,sources[at+3u]|16u,vec3f(0),vec3f(f32(EDGE)))){return;}
 }
}
`;
}
