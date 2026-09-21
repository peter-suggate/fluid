import { UniformTexturePages } from "./uniform-texture-pages";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { uniformSurfaceVolumeWGSL } from "./uniform-surface-volume.wgsl";

const entries = ["begin", "seed", "dilate", "metric", "measure", "reduce", "solve", "apply"] as const;
/** GPU-only, bounded global normal shift. Scratch is allocated during initialization. */
export class UniformSurfaceVolumeCorrection {
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelines: Partial<Record<typeof entries[number], GPUComputePipeline>> = {};
  private buffers?: GPUBuffer[];
  private groups?: [GPUBindGroup, GPUBindGroup];
  private output?: GPUTexture;
  private readonly cellCount: number;
  private readonly vertexCount: number;
  constructor(private readonly device: GPUDevice, private readonly dims: readonly [number, number, number],
    private readonly h: readonly [number, number, number], private readonly phi: GPUTexture,
    private readonly volume: GPUTexture, private readonly capacity: GPUTexture, private readonly fieldPages?: UniformTexturePages) {
    this.cellCount = dims[0]*dims[1]*dims[2];
    this.vertexCount = (dims[0]+1)*(dims[1]+1)*(dims[2]+1);
    this.layout = device.createBindGroupLayout({label:"Surface volume constraint", entries:[
      {binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:"uniform"}},
      ...[1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      ...[5,6,7,8,9].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
      ...(fieldPages?.layout([]) ?? []),
    ]});
  }
  async initialize(signal?: AbortSignal) {
    this.allocate();
    const compiler = gpuCompilationManagerFor(this.device);
    const module = compiler.createShaderModule({label:"Total surface volume",code:this.fieldPages?.shader(uniformSurfaceVolumeWGSL,new Map([[1,this.phi],[2,this.volume],[3,this.capacity],[4,this.output!]]),false,this.fieldPages.nativeStorage) ?? uniformSurfaceVolumeWGSL});
    const layout = this.device.createPipelineLayout({bindGroupLayouts:[this.layout]});
    for (const entryPoint of entries) this.pipelines[entryPoint] = await compiler.compileComputePipeline({
      label:`Surface volume ${entryPoint}`,layout,compute:{module,entryPoint},
    },{priority:"visible",signal});
  }
  private allocate() {
    if(this.buffers) return;
    const device=this.device;
    const descriptor:GPUTextureDescriptor={label:"Total surface volume corrected phi",dimension:"3d",format:"r32float",
      size:this.dims.map(n=>n+1),usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC};
    this.output=this.fieldPages?this.fieldPages.createTextureLike(this.phi,descriptor):device.createTexture(descriptor);
    const buffer=(label:string,size:number,uniform=false)=>device.createBuffer({label,size,
      usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)});
    const params=buffer("Surface volume params",32,true);
    const band=buffer("Surface geometric band",this.vertexCount*4);
    const next=buffer("Surface geometric band scratch",this.vertexCount*4);
    const partial=buffer("Surface volume partial sums",Math.ceil(this.cellCount/64)*80);
    const reduced=buffer("Surface volume reduced sums",Math.ceil(this.cellCount/4096)*80);
    const state=buffer("Surface volume shift and receipt",32);
    this.buffers=[params,band,next,partial,reduced,state];
    const data=new ArrayBuffer(32);new Uint32Array(data).set(this.dims);new Float32Array(data).set([...this.h,Math.min(...this.h)],4);
    device.queue.writeBuffer(params,0,data);
    const view=(t:GPUTexture)=>this.fieldPages?.view(t) ?? t.createView();
    const group=(a:GPUBuffer,b:GPUBuffer)=>(this.fieldPages ?? device).createBindGroup({layout:this.layout,entries:[
      {binding:0,resource:{buffer:params}},{binding:1,resource:view(this.phi)},
      {binding:2,resource:view(this.volume)},{binding:3,resource:view(this.capacity)},
      {binding:4,resource:view(this.output!)},
      ...[a,b,partial,reduced,state].map((buffer,i)=>({binding:i+5,resource:{buffer}})),
    ]});
    this.groups=[group(band,next),group(next,band)];
  }
  get allocatedBytes(): number { return (this.buffers?.reduce((sum,b)=>sum+b.size,0)??0)+(this.output?this.vertexCount*4:0); }
  /** Read-only diagnostic buffer: [shift, search range, target V, prior surface V]. */
  get diagnostics(): GPUBuffer | undefined { return this.buffers?.[5]; }
  encode(encoder: GPUCommandEncoder) {
    if(!this.buffers) throw new Error("Surface correction is not initialized");
    encoder.clearBuffer(this.buffers![1]!);
    const run=(entry:typeof entries[number],count:number,group=0)=>{
      const pass=encoder.beginComputePass({label:`Total surface volume: ${entry}`});
      pass.setPipeline(this.pipelines[entry]!);pass.setBindGroup(0,this.groups![group]!);
      pass.dispatchWorkgroups(Math.min(count,65535),Math.ceil(count/65535));pass.end();
    };
    run("begin",1);run("seed",Math.ceil(this.cellCount/64));
    for(let i=0;i<4;i++) run("dilate",Math.ceil(this.vertexCount/64),i%2);
    run("metric",Math.ceil(this.vertexCount/64));
    // Two 17-sample monotone volume curves refine the scalar root, using
    // deterministic reductions. No per-step CPU readback or iterative solve.
    for(let i=0;i<2;i++) {
      run("measure",Math.ceil(this.cellCount/64));run("reduce",Math.ceil(this.cellCount/4096));run("solve",1);
    }
    run("apply",Math.ceil(this.vertexCount/64));
    if(this.fieldPages) this.fieldPages.copy(encoder,this.output!,this.phi);
    else encoder.copyTextureToTexture({texture:this.output!},{texture:this.phi},this.dims.map(n=>n+1));
  }
  destroy() { this.output?.destroy(); for(const buffer of this.buffers??[]) buffer.destroy(); }
}
