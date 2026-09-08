import { gpuCompilationManagerFor } from "../../lib/core/gpu-compilation-manager";
import type { TensorCSL4Field, Triple } from "./tensor-csl4-oracle";

/** Isolated smooth-field research candidate, never a production solver path.
 *
 * Periodic axis slots are [V, h*D, mean], x-fast over (3nx,3ny,3nz).
 * Their Cartesian product stores 27 dimensionless mixed functionals per cell.
 * Every directional pass measures the CURRENT quartic; there is no authored
 * seed, external cell-mean target, positivity clipping or surface operation.
 *
 * Accepted storage is fixed. Two scratch banks perform X/Y/Z translation;
 * device Bernstein admission precedes a conditional copy and generation commit.
 * Failed candidates leave accepted bytes unchanged. Advance submits GPU work
 * without reading field data or receipts. Readbacks are explicitly QA methods.
 * Bernstein bounds are evaluated in f32, not outward-rounded intervals. Their
 * sufficient real-arithmetic range property is not an interval proof in f32.
 * Only uniform translation on a complete periodic grid is admitted here.
 */
export const TENSOR_CSL4_GPU_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read> source:array<f32>;
@group(0) @binding(1) var<storage,read_write> destination:array<f32>;
@group(0) @binding(2) var<storage,read> p:array<f32>;
@group(0) @binding(3) var<storage,read_write> receipt:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(5) var<storage,read> queries:array<f32>;
@group(0) @binding(6) var<storage,read_write> output:array<f32>;
fn u(i:u32)->u32{return bitcast<u32>(p[i]);}
fn dims()->vec3u{return vec3u(u(0u),u(1u),u(2u));}
fn spacing()->vec3f{return vec3f(p[4],p[5],p[6]);}
fn wrapped(i:i32,n:u32)->u32{return u32(((i%i32(n))+i32(n))%i32(n));}
fn slotId(q:vec3u)->u32{let n=3u*dims();return q.x+n.x*(q.y+n.y*q.z);}
fn slotCoord(id:u32)->vec3u{let n=3u*dims();return vec3u(id%n.x,(id/n.x)%n.y,id/(n.x*n.y));}
fn cellCoord(id:u32)->vec3i{let n=dims();return vec3i(i32(id%n.x),i32((id/n.x)%n.y),i32(id/(n.x*n.y)));}
fn constraintSlot(cell:i32,n:u32,k:u32)->u32{
 if(k<2u){return 3u*wrapped(cell,n)+k;}
 if(k<4u){return 3u*wrapped(cell+1,n)+k-2u;}
 return 3u*wrapped(cell,n)+2u;
}
fn moment(cell:vec3i,x:u32,y:u32,z:u32)->f32{let n=dims();
 return source[slotId(vec3u(constraintSlot(cell.x,n.x,x),constraintSlot(cell.y,n.y,y),constraintSlot(cell.z,n.z,z)))];}
fn fail(code:u32,cell:u32){atomicOr(&receipt[0],code);atomicMin(&receipt[1],cell);}
fn finite(v:f32)->bool{return abs(v)<3.4e38;}
// Hermite cubic plus its SAME quartic volume residual. Factoring avoids
// subtracting the large expanded cardinal weights at every transport step.
fn volumeResidual(v:array<f32,5>)->f32{return v[4]-0.5*(v[0]+v[2])-(v[1]-v[3])/12.0;}
fn pointValueFromLeft(v:array<f32,5>,t:f32,derivative:bool)->f32{
 if(t==0.0){return v[select(0u,1u,derivative)];}
 if(t==1.0){return v[select(2u,3u,derivative)];}
 let u=1.0-t;let difference=v[2]-v[0];let residual=volumeResidual(v);
 if(derivative){return difference*6.0*t*u+v[1]*u*(1.0-3.0*t)
  +v[3]*t*(3.0*t-2.0)+residual*60.0*t*u*(1.0-2.0*t);}
 return v[0]+difference*t*t*(3.0-2.0*t)+v[1]*t*u*u-v[3]*t*t*u
  +residual*30.0*t*t*u*u;
}
fn prefixIntegralFromLeft(v:array<f32,5>,t:f32)->f32{
 if(t==0.0){return 0.0;}if(t==1.0){return v[4];}
 let t2=t*t;let t3=t2*t;
 return v[0]*t+(v[2]-v[0])*t3*(1.0-0.5*t)
  +v[1]*t2*(0.5+t*(-2.0/3.0+0.25*t))+v[3]*t3*(-1.0/3.0+0.25*t)
  +volumeResidual(v)*t3*(10.0+t*(-15.0+6.0*t));
}
fn reflected(v:array<f32,5>)->array<f32,5>{return array<f32,5>(v[2],-v[3],v[0],-v[1],v[4]);}
fn pointValue(v:array<f32,5>,t:f32,derivative:bool)->f32{
 if(t<=0.5){return pointValueFromLeft(v,t,derivative);}
 let value=pointValueFromLeft(reflected(v),1.0-t,derivative);return select(value,-value,derivative);
}
fn prefixIntegral(v:array<f32,5>,t:f32)->f32{
 if(t<=0.5){return prefixIntegralFromLeft(v,t);}
 return v[4]-prefixIntegralFromLeft(reflected(v),1.0-t);
}
fn transportedMean(left:array<f32,5>,right:array<f32,5>,t:f32)->f32{
 // Prefix is linear in all five current moments. Contracting their difference
 // avoids subtracting two bulk-sized fluxes; constants give an exact zero.
 var difference:array<f32,5>;for(var i=0u;i<5u;i++){difference[i]=right[i]-left[i];}
 if(t<=0.5){return left[4]+prefixIntegralFromLeft(difference,t);}
 return right[4]-prefixIntegralFromLeft(reflected(difference),1.0-t);
}
fn lineFunctional(v:array<f32,5>,a:f32,b:f32,kind:u32)->f32{
 if(kind==2u){
  if(a>=0.5){let reverse=reflected(v);return prefixIntegralFromLeft(reverse,1.0-a)-prefixIntegralFromLeft(reverse,1.0-b);}
  return prefixIntegral(v,b)-prefixIntegral(v,a);
 }
 return pointValue(v,a,kind==1u);
}
fn lineMoments(q:vec3u,axis:u32,cell:i32)->array<f32,5>{var v:array<f32,5>;
 for(var k=0u;k<5u;k++){var at=q;at[axis]=constraintSlot(cell,dims()[axis],k);v[k]=source[slotId(at)];}return v;}
@compute @workgroup_size(1) fn beginOperation(){
 atomicStore(&receipt[0],0u);atomicStore(&receipt[1],0xffffffffu);atomicStore(&receipt[2],0u);atomicStore(&receipt[3],0u);
 if(u(7u)==0u&&atomicLoad(&control[1])==0u){fail(8u,0u);}
}
fn translateSlot(id:u32,axis:u32){
 if(id>=27u*u(3u)||atomicLoad(&receipt[0])!=0u){return;}
 let q=slotCoord(id);let kind=q[axis]%3u;let cell=i32(q[axis]/3u);
 let n=dims()[axis];let h=spacing()[axis];
 let raw=p[8u+axis]/h;let shift=raw-floor(raw/f32(n))*f32(n);
 let whole=i32(floor(shift));let fraction=shift-f32(whole);
 // Integer translations are a permutation of the shared DOFs, exactly.
 if(fraction==0.0){var at=q;at[axis]=3u*wrapped(cell-whole,n)+kind;destination[id]=source[slotId(at)];return;}
 let donor=cell-whole-1;let t=1.0-fraction;let left=lineMoments(q,axis,donor);var value=0.0;
 if(kind==2u){let right=lineMoments(q,axis,donor+1);
  value=transportedMean(left,right,t);}
 else{value=pointValue(left,t,kind==1u);}
 if(!finite(value)){fail(2u,id);return;}destination[id]=value;
}
@compute @workgroup_size(64) fn translateX(@builtin(global_invocation_id)gid:vec3u){translateSlot(gid.x,0u);}
@compute @workgroup_size(64) fn translateY(@builtin(global_invocation_id)gid:vec3u){translateSlot(gid.x,1u);}
@compute @workgroup_size(64) fn translateZ(@builtin(global_invocation_id)gid:vec3u){translateSlot(gid.x,2u);}
fn bernstein(v:array<f32,5>,k:u32)->f32{
 if(k==0u){return v[0];}if(k==1u){return v[0]+0.25*v[1];}
 if(k==2u){return -2.0*v[0]-0.25*v[1]-2.0*v[2]+0.25*v[3]+5.0*v[4];}
 if(k==3u){return v[2]-0.25*v[3];}return v[2];
}
@compute @workgroup_size(64) fn admitRange(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=u(3u)||atomicLoad(&receipt[0])==8u){return;}
 let cell=cellCoord(gid.x);var a:array<f32,125>;var b:array<f32,125>;
 for(var z=0u;z<5u;z++){for(var y=0u;y<5u;y++){for(var x=0u;x<5u;x++){
  a[x+5u*(y+5u*z)]=moment(cell,x,y,z);}}}
 var stride=1u;
 for(var axis=0u;axis<3u;axis++){
  for(var base=0u;base<125u;base+=5u*stride){for(var offset=0u;offset<stride;offset++){
   var v:array<f32,5>;for(var j=0u;j<5u;j++){v[j]=a[base+offset+j*stride];}
   for(var j=0u;j<5u;j++){b[base+offset+j*stride]=bernstein(v,j);}}}
  a=b;stride*=5u;
 }
 var lower=3.4e38;var upper=-3.4e38;var valid=true;
 for(var k=0u;k<125u;k++){let v=a[k];lower=min(lower,v);upper=max(upper,v);
  if(!finite(v)){fail(2u,gid.x);valid=false;}else if(v<0.0||v>1.0){valid=false;}}
 output[4u*gid.x]=lower;output[4u*gid.x+1u]=upper;
 output[4u*gid.x+2u]=moment(cell,4u,4u,4u)*p[4]*p[5]*p[6];
 output[4u*gid.x+3u]=select(0.0,1.0,valid);
 if(!valid){fail(1u,gid.x);atomicAdd(&receipt[2],1u);}
}
@compute @workgroup_size(64) fn commitValues(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=27u*u(3u)||atomicLoad(&receipt[0])!=0u){return;}destination[gid.x]=source[gid.x];
}
@compute @workgroup_size(1) fn finalizeGeneration(){if(atomicLoad(&receipt[0])!=0u){return;}
 atomicStore(&control[1],1u);atomicAdd(&control[0],1u);atomicStore(&receipt[3],1u);
}
fn contract(cell:vec3i,a:vec3f,b:vec3f,kind:vec3u)->f32{
 var zValues:array<f32,5>;for(var z=0u;z<5u;z++){var yValues:array<f32,5>;
  for(var y=0u;y<5u;y++){var xValues:array<f32,5>;
   for(var x=0u;x<5u;x++){xValues[x]=moment(cell,x,y,z);}
   yValues[y]=lineFunctional(xValues,a.x,b.x,kind.x);}
  zValues[z]=lineFunctional(yValues,a.y,b.y,kind.y);}
 return lineFunctional(zValues,a.z,b.z,kind.z);
}
@compute @workgroup_size(64) fn queryField(@builtin(global_invocation_id)gid:vec3u){
 if(gid.x>=u(11u)){return;}if(atomicLoad(&control[1])==0u){fail(8u,gid.x);return;}
 let at=16u*gid.x;let h=spacing();let lo=vec3f(queries[at],queries[at+1u],queries[at+2u])/h;
 if(queries[at+3u]==0.0){
  var cell=vec3i(floor(lo));if(queries[at+8u]!=0.0){cell=vec3i(vec3f(queries[at+9u],queries[at+10u],queries[at+11u]));}
  let t=lo-vec3f(cell);
  output[4u*gid.x]=contract(cell,t,t,vec3u(0u));
  output[4u*gid.x+1u]=contract(cell,t,t,vec3u(1u,0u,0u))/h.x;
  output[4u*gid.x+2u]=contract(cell,t,t,vec3u(0u,1u,0u))/h.y;
  output[4u*gid.x+3u]=contract(cell,t,t,vec3u(0u,0u,1u))/h.z;return;
 }
 let hi=vec3f(queries[at+4u],queries[at+5u],queries[at+6u])/h;
 let first=vec3i(floor(lo));let last=vec3i(ceil(hi));var sum=0.0;
 for(var z=first.z;z<last.z;z++){for(var y=first.y;y<last.y;y++){for(var x=first.x;x<last.x;x++){
  let cell=vec3i(x,y,z);let a=max(lo,vec3f(cell))-vec3f(cell);let b=min(hi,vec3f(cell)+vec3f(1.0))-vec3f(cell);
  if(any(b<=a)){continue;}
  if(all(a==vec3f(0.0))&&all(b==vec3f(1.0))){sum+=moment(cell,4u,4u,4u);}
  else{sum+=contract(cell,a,b,vec3u(2u));}
 }}}
 output[4u*gid.x]=sum*h.x*h.y*h.z;
}
`;

export type TensorCSL4GPUQuery = { point: Triple; cell?: Triple } | { lower: Triple; upper: Triple };
export interface TensorCSL4GPUReceipt {
  fault: number; firstCell: number; rejectedCells: number; committed: boolean;
  generation: number; initialized: boolean;
}
type Compiled = { layout: GPUBindGroupLayout; pipelines: ReadonlyMap<string, GPUComputePipeline> };
const compilation = new WeakMap<GPUDevice, Promise<Compiled>>();
const ENTRIES = ["beginOperation", "translateX", "translateY", "translateZ", "admitRange", "commitValues", "finalizeGeneration", "queryField"] as const;
async function compile(device: GPUDevice): Promise<Compiled> {
  const existing = compilation.get(device); if (existing) return existing;
  const pending = (async () => {
    const manager = gpuCompilationManagerFor(device);
    const layout = device.createBindGroupLayout({ entries: Array.from({ length: 7 }, (_, binding) => ({
      binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: [0, 2, 5].includes(binding) ? "read-only-storage" as const : "storage" as const },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const module = manager.createShaderModule({ label: "Research periodic tensor CSL4", code: TENSOR_CSL4_GPU_WGSL });
    const pipelines = new Map<string, GPUComputePipeline>();
    for (const entryPoint of ENTRIES) {
      try { pipelines.set(entryPoint, await manager.compileComputePipeline({ layout: pipelineLayout,
        label: `Research tensor CSL4 ${entryPoint}`, compute: { module, entryPoint } })); }
      catch (error) {
        const messages = (await module.getCompilationInfo()).messages.filter(message => message.type === "error")
          .map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n");
        throw new Error(`Tensor CSL4 ${entryPoint} compilation failed: ${String(error)}\n${messages}`, { cause: error });
      }
    }
    return { layout, pipelines };
  })();
  compilation.set(device, pending);
  void pending.catch(() => { if (compilation.get(device) === pending) compilation.delete(device); });
  return pending;
}

export class GPUTensorCSL4 {
  private busy = false;
  private destroyed = false;
  private constructor(readonly device: GPUDevice, readonly dimensions: Triple, readonly lengths: Triple,
    private readonly compiled: Compiled, private readonly accepted: GPUBuffer,
    private readonly scratch: readonly [GPUBuffer, GPUBuffer], private readonly receipt: GPUBuffer,
    private readonly control: GPUBuffer, private readonly ranges: GPUBuffer, private readonly dummy: GPUBuffer) {}

  static async create(device: GPUDevice, initial: TensorCSL4Field): Promise<GPUTensorCSL4> {
    const n = initial.dimensions.reduce((a, b) => a * b, 1);
    if (!initial.dimensions.every(value => Number.isSafeInteger(value) && value >= 2 && value <= 32)
      || n > 32768 || !initial.lengths.every((value, axis) => Number.isFinite(value) && value > 0
        && Number.isFinite(Math.fround(value)) && Math.fround(value / initial.dimensions[axis]!) > 0)
      || initial.data.length !== 27 * n || !initial.data.every(Number.isFinite)) throw new Error("Invalid bounded tensor CSL4 field");
    const packed = Float32Array.from(initial.data);
    if (!packed.every(Number.isFinite)) throw new Error("Tensor CSL4 initial field exceeds float32");
    const compiled = await compile(device);
    const buffers: GPUBuffer[] = [];
    const allocate = (label: string, size: number) => { const buffer = device.createBuffer({ label, size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }); buffers.push(buffer); return buffer; };
    try {
      const accepted = allocate("Tensor CSL4 accepted27", packed.byteLength);
      const scratch = [allocate("Tensor CSL4 scratchA27", packed.byteLength), allocate("Tensor CSL4 scratchB27", packed.byteLength)] as const;
      const field = new GPUTensorCSL4(device, [...initial.dimensions] as Triple, [...initial.lengths] as Triple,
        compiled, accepted, scratch, allocate("Tensor CSL4 admission receipt", 16), allocate("Tensor CSL4 generation", 16),
        allocate("Tensor CSL4 candidate Bernstein bounds", 16 * n), allocate("Tensor CSL4 unused query", 64));
      device.queue.writeBuffer(scratch[0], 0, packed);
      await field.operation([0, 0, 0], true);
      return field;
    } catch (error) { for (const buffer of buffers) buffer.destroy(); throw error; }
  }
  private parameters(displacement: Triple, initialize: boolean, queryCount = 0): GPUBuffer {
    const values = new Float32Array(16), words = new Uint32Array(values.buffer);
    words.set(this.dimensions, 0); words[3] = this.dimensions.reduce((a, b) => a * b, 1);
    values.set(this.lengths.map((length, axis) => length / this.dimensions[axis]!), 4);
    words[7] = Number(initialize); values.set(displacement, 8); words[11] = queryCount;
    const buffer = this.device.createBuffer({ label: "Tensor CSL4 immutable operation", size: values.byteLength,
      usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
    new Float32Array(buffer.getMappedRange()).set(values); buffer.unmap(); return buffer;
  }
  private binding(source: GPUBuffer, destination: GPUBuffer, p: GPUBuffer,
    output = this.ranges, queries = this.dummy, receipt = this.receipt): GPUBindGroup {
    return this.device.createBindGroup({ layout: this.compiled.layout,
      entries: [source, destination, p, receipt, this.control, queries, output]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }
  private dispatch(encoder: GPUCommandEncoder, name: string, binding: GPUBindGroup, count: number): void {
    const pass = encoder.beginComputePass({ label: `Research tensor CSL4 ${name}` });
    pass.setPipeline(this.compiled.pipelines.get(name)!); pass.setBindGroup(0, binding);
    pass.dispatchWorkgroups(Math.ceil(count / (name === "beginOperation" || name === "finalizeGeneration" ? 1 : 64))); pass.end();
  }
  private assertIdle(): void { if (this.busy || this.destroyed) throw new Error("Tensor CSL4 field is busy or destroyed"); }
  private async operation(displacement: Triple, initialize: boolean): Promise<void> {
    this.assertIdle(); this.busy = true;
    const parameters = this.parameters(displacement, initialize);
    try {
      const count = this.dimensions.reduce((a, b) => a * b, 1);
      const encoder = this.device.createCommandEncoder({ label: "Research tensor CSL4 atomic translation" });
      const candidate = this.binding(this.scratch[0], this.accepted, parameters);
      this.dispatch(encoder, "beginOperation", candidate, 1);
      if (!initialize) {
        this.dispatch(encoder, "translateX", this.binding(this.accepted, this.scratch[0], parameters), 27 * count);
        this.dispatch(encoder, "translateY", this.binding(this.scratch[0], this.scratch[1], parameters), 27 * count);
        this.dispatch(encoder, "translateZ", this.binding(this.scratch[1], this.scratch[0], parameters), 27 * count);
      }
      this.dispatch(encoder, "admitRange", candidate, count);
      this.dispatch(encoder, "commitValues", candidate, 27 * count);
      this.dispatch(encoder, "finalizeGeneration", candidate, 1);
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
    } finally { parameters.destroy(); this.busy = false; }
  }
  async advance(displacement: Triple): Promise<void> {
    if (!displacement.every((value, axis) => Number.isFinite(value) && Number.isFinite(Math.fround(value))
      && Math.abs(value) <= 1024 * this.lengths[axis]!)) {
      throw new Error("Invalid bounded uniform displacement");
    }
    await this.operation(displacement, false);
  }
  private async read(buffer: GPUBuffer): Promise<ArrayBuffer> {
    this.assertIdle(); this.busy = true;
    const readback = this.device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 0, readback, 0, buffer.size);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const result = readback.getMappedRange().slice(0); readback.unmap(); return result;
    } finally { readback.destroy(); this.busy = false; }
  }
  async readReceiptForQA(): Promise<TensorCSL4GPUReceipt> {
    this.assertIdle(); this.busy = true;
    const readback = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(this.receipt, 0, readback, 0, 16); encoder.copyBufferToBuffer(this.control, 0, readback, 16, 16);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()).slice(); readback.unmap();
      return { fault: words[0]!, firstCell: words[1]!, rejectedCells: words[2]!, committed: words[3] === 1,
        generation: words[4]!, initialized: words[5] === 1 };
    } finally { readback.destroy(); this.busy = false; }
  }
  async readCurrentRecordsForQA(): Promise<Float32Array> { return new Float32Array(await this.read(this.accepted)); }
  async readCandidateRangesForQA(): Promise<Float32Array> { return new Float32Array(await this.read(this.ranges)); }
  /** Result stride4: q, gradient xyz for points; physical amount,0,0,0 for boxes. */
  async queryForQA(requests: readonly TensorCSL4GPUQuery[]): Promise<Float32Array> {
    this.assertIdle(); if (requests.length === 0) return new Float32Array();
    if (requests.length > 4096) throw new Error("Too many tensor QA queries");
    const values = new Float32Array(16 * requests.length);
    for (let i = 0; i < requests.length; i++) {
      const query = requests[i]!;
      const lower = "point" in query ? query.point : query.lower;
      if (!lower.every((value, axis) => Number.isFinite(value) && Math.abs(value) <= 1024 * this.lengths[axis]!)) throw new Error("Invalid tensor QA point");
      values.set(lower, 16 * i);
      if ("upper" in query) {
        if (!query.upper.every((value, axis) => Number.isFinite(value) && value >= lower[axis]!
          && value - lower[axis]! <= this.lengths[axis]!)) throw new Error("Invalid periodic QA box");
        values[16 * i + 3] = 1; values.set(query.upper, 16 * i + 4);
      } else if (query.cell) {
        if (!query.cell.every((value, axis) => Number.isInteger(value) && lower[axis]! >= value * this.lengths[axis]! / this.dimensions[axis]!
          && lower[axis]! <= (value + 1) * this.lengths[axis]! / this.dimensions[axis]!)) throw new Error("Point lies outside explicit tensor cell");
        values[16 * i + 8] = 1; values.set(query.cell, 16 * i + 9);
      }
    }
    this.busy = true; const temporary: GPUBuffer[] = [];
    try {
      const allocate = (size: number, usage: GPUBufferUsageFlags) => { const buffer = this.device.createBuffer({ size, usage }); temporary.push(buffer); return buffer; };
      const parameters = this.parameters([0, 0, 0], false, requests.length); temporary.push(parameters);
      const queries = allocate(values.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST); this.device.queue.writeBuffer(queries, 0, values);
      const output = allocate(16 * requests.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const receipt = allocate(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = allocate(16 + output.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const encoder = this.device.createCommandEncoder();
      this.dispatch(encoder, "queryField", this.binding(this.accepted, this.scratch[1], parameters, output, queries, receipt), requests.length);
      encoder.copyBufferToBuffer(receipt, 0, readback, 0, 16); encoder.copyBufferToBuffer(output, 0, readback, 16, output.size);
      this.device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const mapped = readback.getMappedRange(), fault = new Uint32Array(mapped, 0, 4)[0];
      const result = new Float32Array(mapped, 16, 4 * requests.length).slice(); readback.unmap();
      if (fault) throw new Error(`Tensor CSL4 QA query rejected: fault=${fault}`); return result;
    } finally { for (const buffer of temporary) buffer.destroy(); this.busy = false; }
  }
  destroy(): void {
    this.assertIdle(); this.destroyed = true;
    for (const buffer of [this.accepted, ...this.scratch, this.receipt, this.control, this.ranges, this.dummy]) buffer.destroy();
  }
}
