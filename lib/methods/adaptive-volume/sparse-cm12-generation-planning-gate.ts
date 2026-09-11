import { writeGPUBufferView } from "../../core/webgpu-buffer-upload";

/** Conservative device preflight. False positives may request a detailed
 * snapshot; a negative receipt must rule out every ordinary generation cause. */
export class SparseCM12GenerationPlanningGate {
  private constructor(
    private readonly device: GPUDevice,
    private readonly pipeline: GPUComputePipeline,
    private readonly group: GPUBindGroup,
    private readonly buffers: readonly GPUBuffer[],
    private readonly count: number,
  ) {}

  static async create(device: GPUDevice, activity: GPUBuffer, metadata: Uint32Array,
    headerWords: number, recordWords: number): Promise<SparseCM12GenerationPlanningGate> {
    // Metadata: bit 0 = backed, bit 1 = unclipped, bits 8..31 = brick span.
    const shader = device.createShaderModule({ label: "CM12 generation planning preflight", code: `
@group(0)@binding(0)var<storage,read>a:array<u32>;
@group(0)@binding(1)var<storage,read>metadata:array<u32>;
@group(0)@binding(2)var<storage,read_write>receipt:array<atomic<u32>>;
struct Parameters{limits:vec4u,tuning:vec4f}
@group(0)@binding(3)var<uniform>p:Parameters;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3u){
  let leaf=id.x;if(leaf>=p.limits.x){return;}
  let at=${headerWords}u+${recordWords}u*leaf;
  let flags=metadata[leaf];let span=flags>>8u;
  let isActive=a[at+10u]!=0u;let current=a[at+12u];
  let requested=select(current,a[at+47u],a[at+47u]!=0u);
  let activation=(a[at+9u]&0x80000000u)!=0u;
  let frozenFrontier=(a[at+9u]&0x00020000u)!=0u;
  if(frozenFrontier){atomicOr(&receipt[0],1u);}
  if(p.limits.w!=0u){return;}
  let reasons=a[at+1u];let travel=bitcast<f32>(a[at+33u]);
  if(((isActive||activation)&&(flags&1u)==0u&&requested!=current)
    ||(isActive&&span>1u&&((reasons&256u)!=0u||travel>=p.tuning.x))){
    atomicOr(&receipt[0],1u);
  }
  if(isActive&&current==1u&&(flags&2u)==0u&&2u*span<=p.limits.y
    &&((a[at+2u]>>8u)&255u)>=max(64u,p.limits.z)
    &&bitcast<f32>(a[at+4u])>=0.9999
    &&(reasons&(1u|16u|256u|512u))==0u&&travel<0.125){
    atomicAdd(&receipt[1],1u);
  }
}` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto",
      compute: { module: shader, entryPoint: "main" } });
    const descriptors = device.createBuffer({ label: "CM12 planning leaf metadata",
      size: Math.max(4, metadata.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const receipt = device.createBuffer({ label: "CM12 planning request receipt",
      size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const parameters = device.createBuffer({ label: "CM12 planning gate parameters",
      size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const readback = device.createBuffer({ label: "CM12 planning request readback",
      size: 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    if (metadata.byteLength) writeGPUBufferView(device.queue, descriptors, 0, metadata);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
      [activity, descriptors, receipt, parameters].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    return new SparseCM12GenerationPlanningGate(device, pipeline, group,
      [descriptors, receipt, parameters, readback], metadata.length);
  }

  async needed(maximumSpan: number, demoteEpochs: number, finestTravel: number,
    frozenFrontierOnly = false): Promise<boolean> {
    const [, receipt, parameters, readback] = this.buffers;
    const data = new ArrayBuffer(32);
    new Uint32Array(data).set([this.count, maximumSpan, demoteEpochs, Number(frozenFrontierOnly)]);
    new Float32Array(data)[4] = finestTravel;
    this.device.queue.writeBuffer(parameters!, 0, data);
    const encoder = this.device.createCommandEncoder({ label: "CM12 planning preflight" });
    encoder.clearBuffer(receipt!);
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline); pass.setBindGroup(0, this.group);
    pass.dispatchWorkgroups(Math.ceil(this.count / 64)); pass.end();
    encoder.copyBufferToBuffer(receipt!, 0, readback!, 0, 8);
    this.device.queue.submit([encoder.finish()]);
    await readback!.mapAsync(GPUMapMode.READ);
    try {
      const words = new Uint32Array(readback!.getMappedRange());
      return words[0] !== 0 || words[1]! >= 8;
    } finally { readback!.unmap(); }
  }

  destroy(): void { for (const buffer of this.buffers) buffer.destroy(); }
  get allocatedBytes(): number { return this.buffers.reduce((sum, buffer) => sum + buffer.size, 0); }
}
