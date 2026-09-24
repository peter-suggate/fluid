import { UniformTexturePages } from "./uniform-texture-pages";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";
import { createUniformSurfaceVolumeWGSL } from "./uniform-surface-volume.wgsl";

import { uniformAbOn } from "./uniform-ab-switch";

/** In place over the bounded vertex box. Outside it metric wrote no scale, so
 * the dense pass stored phi - shift*0 there: phi itself, bar a -0 turning +0. */
const inPlaceApplyWGSL = /* wgsl */ `
@group(0) @binding(0) var phi:texture_storage_3d<r32float,read_write>;
@group(0) @binding(1) var<storage,read> scale:array<u32>;
@group(0) @binding(2) var<storage,read> state:array<vec4f>;
@group(0) @binding(3) var<storage,read> work:array<u32>;
@group(0) @binding(4) var<uniform> dims:vec4u;
@compute @workgroup_size(64) fn apply(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)l:u32){
 let logical=(w.x+w.y*65535u)*64u+l;if(logical>=work[15]){return;}
 let d=vec3u(work[12],work[13],work[14])+vec3u(1);
 let q=vec3u(work[8],work[9],work[10])+vec3u(logical%d.x,(logical/d.x)%d.y,logical/(d.x*d.y));
 let i=q.x+(dims.x+1u)*(q.y+(dims.y+1u)*q.z);
 textureStore(phi,q,vec4f(textureLoad(phi,q).x-state[0].x*bitcast<f32>(scale[i])));
}`;

const entries = ["finishWork", "begin", "seed", "seedBox", "seedTiles", "dilate", "metric", "measure", "reduce", "solve", "apply"] as const;
/** surfaceseed inputs: the phi census box (phi region words 7..12) and the
 * transport receiver tile list with its indirect dispatch. */
export interface UniformSurfaceSeedWindow {
  readonly phiRegion: GPUBuffer;
  readonly receivers: GPUBuffer;
  readonly receiverWord: number;
  readonly tiles: number;
  readonly receiverDispatch: GPUBuffer;
}
/** GPU-only, bounded global normal shift. Scratch is allocated during initialization. */
export class UniformSurfaceVolumeCorrection {
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelines: Partial<Record<typeof entries[number], GPUComputePipeline>> = {};
  private buffers?: GPUBuffer[];
  private groups?: [GPUBindGroup, GPUBindGroup];
  private output?: GPUTexture;
  private work?: GPUBuffer;
  private workDispatch?: GPUBuffer;
  private reverseParams?: GPUBuffer;
  private inPlace?: { pipeline: GPUComputePipeline; group: GPUBindGroup };
  private readonly inPlaceApply: boolean;
  private lastWindowedSeed = false;
  private readonly cellCount: number;
  private readonly vertexCount: number;
  constructor(private readonly device: GPUDevice, private readonly dims: readonly [number, number, number],
    private readonly h: readonly [number, number, number], private readonly phi: GPUTexture,
    private readonly volume: GPUTexture, private readonly capacity: GPUTexture, private readonly fieldPages?: UniformTexturePages, private readonly compactWork = uniformAbOn("surfacewindow"), private readonly capacityComponent: "x" | "w" = "x", private readonly seedWindow?: UniformSurfaceSeedWindow) {
    if(seedWindow && !compactWork)throw new Error("A windowed surface seed needs the compact work bounds");
    this.cellCount = dims[0]*dims[1]*dims[2];
    this.vertexCount = (dims[0]+1)*(dims[1]+1)*(dims[2]+1);
    this.inPlaceApply = compactWork && uniformAbOn("surfaceapply");
    if(this.inPlaceApply && (!(phi.usage & GPUTextureUsage.STORAGE_BINDING) || (fieldPages && (fieldPages.scratchMetadata(phi)!==0 || phi.width!==dims[0]+1))))
      throw new Error("In-place surface correction needs a native, storage-bound vertex phi");
    this.layout = device.createBindGroupLayout({label:"Surface volume constraint", entries:[
      {binding:0, visibility:GPUShaderStage.COMPUTE, buffer:{type:"uniform"}},
      ...[1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const,viewDimension:"3d" as const}})),
      {binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}},
      ...[5,6,7,8,9].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
      ...(compactWork ? [{binding:10,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}}] : []),
      ...(seedWindow ? [{binding:11,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage" as const}}] : []),
      ...(fieldPages?.layout([]) ?? []),
    ]});
  }
  async initialize(signal?: AbortSignal) {
    this.allocate();
    const compiler = gpuCompilationManagerFor(this.device);
    const bandWords=Math.ceil(this.vertexCount/4)*4;
    const shader=createUniformSurfaceVolumeWGSL(this.compactWork,this.capacityComponent,
      this.seedWindow ? this.seedWindow.receiverWord-this.receiverRange()!.offset/4 : undefined);
    const source=this.fieldPages?.scratch ? shader
      .replace(/&band\[([^\]]+)\]/g,`&uniformScratch[select(${bandWords}u,0u,p.dims.w!=0u)+($1)]`)
      .replace(/&nextBand\[([^\]]+)\]/g,`&uniformScratch[select(0u,${bandWords}u,p.dims.w!=0u)+($1)]`)
      : shader;
    const module = compiler.createShaderModule({label:"Total surface volume",code:this.fieldPages?.shader(source,new Map([[1,this.phi],[2,this.volume],[3,this.capacity],[4,this.output!]]),false,this.fieldPages.nativeStorage,false,new Set([1,2,3])) ?? source});
    const layout = this.device.createPipelineLayout({bindGroupLayouts:[this.layout]});
    for (const entryPoint of entries.filter(entry=>(this.compactWork || entry!=="finishWork") && (!!this.seedWindow || (entry!=="seedBox" && entry!=="seedTiles")))) this.pipelines[entryPoint] = await compiler.compileComputePipeline({
      label:`Surface volume ${entryPoint}`,layout,compute:{module,entryPoint},
    },{priority:"visible",signal});
    if(this.inPlaceApply){
      const pipeline=await compiler.compileComputePipeline({label:"Surface volume apply in place",layout:"auto",
        compute:{module:compiler.createShaderModule({label:"Total surface volume in place",code:inPlaceApplyWGSL}),entryPoint:"apply"}},{priority:"visible",signal});
      const scale=this.fieldPages?.scratch ? {buffer:this.fieldPages.scratch.buffer,offset:0,size:this.vertexCount*4} : {buffer:this.buffers![2]!};
      this.inPlace={pipeline,group:this.device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
        {binding:0,resource:this.phi.createView()},{binding:1,resource:scale},{binding:2,resource:{buffer:this.buffers![5]!}},
        {binding:3,resource:{buffer:this.work!}},{binding:4,resource:{buffer:this.buffers![0]!,offset:0,size:16}}]})};
    }
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
    const band=buffer("Surface geometric band",this.fieldPages?.scratch?16:this.vertexCount*4);
    const next=buffer("Surface geometric band scratch",this.fieldPages?.scratch?16:this.vertexCount*4);
    const partialBytes=Math.ceil(this.cellCount/64)*80;
    const partialOffset=Math.ceil(Math.ceil(this.vertexCount/4)*32/256)*256;
    const partial=buffer("Surface volume partial sums",this.fieldPages?.scratch?16:partialBytes);
    const reduced=buffer("Surface volume reduced sums",Math.ceil(this.cellCount/4096)*80);
    const state=buffer("Surface volume shift and receipt",32);
    this.buffers=[params,band,next,partial,reduced,state];
    if(this.compactWork){
      this.work=buffer("Surface correction work bounds",256);
      this.workDispatch=device.createBuffer({label:"Surface correction bounded dispatch",size:72,
        usage:GPUBufferUsage.INDIRECT|GPUBufferUsage.COPY_DST});
    }
    const data=new ArrayBuffer(32);new Uint32Array(data).set(this.dims);new Float32Array(data).set([...this.h,Math.min(...this.h)],4);
    device.queue.writeBuffer(params,0,data);
    if(this.fieldPages?.scratch){
      this.reverseParams=buffer("Surface volume reverse parity params",32,true);
      new Uint32Array(data)[3]=1;device.queue.writeBuffer(this.reverseParams,0,data);
    }
    const view=(t:GPUTexture)=>this.fieldPages?.view(t) ?? t.createView();
    const group=(a:GPUBuffer,b:GPUBuffer,parameters=params)=>{
     const descriptor:GPUBindGroupDescriptor={layout:this.layout,entries:[
      {binding:0,resource:{buffer:parameters}},{binding:1,resource:view(this.phi)},
      {binding:2,resource:view(this.volume)},{binding:3,resource:view(this.capacity)},
      {binding:4,resource:view(this.output!)},
      ...[a,b,partial,reduced,state].map((buffer,i)=>({binding:i+5,resource:i===2&&this.fieldPages?.scratch
       ? {buffer:this.fieldPages.scratch.buffer,offset:partialOffset,size:partialBytes} : {buffer}})),
      ...(this.work ? [{binding:10,resource:{buffer:this.work}}] : []),
      ...(this.seedWindow ? [{binding:11,resource:{buffer:this.seedWindow.receivers,...this.receiverRange()!}}] : []),
     ]};
     return this.fieldPages?.createBindGroup(descriptor,true,partialOffset) ?? device.createBindGroup(descriptor);
    };
    this.groups=[group(band,next),group(next,band,this.reverseParams??params)];
  }
  private receiverRange(): {offset:number;size:number} | undefined {
    if(!this.seedWindow)return undefined;
    const offset=Math.floor(this.seedWindow.receiverWord*4/256)*256;
    return {offset,size:this.seedWindow.receiverWord*4-offset+4*(1+this.seedWindow.tiles)};
  }
  get allocatedBytes(): number { return (this.buffers?.reduce((sum,b)=>sum+b.size,0)??0)+(this.output?this.vertexCount*4:0)+(this.reverseParams?.size??0)+(this.work?.size??0)+(this.workDispatch?.size??0); }
  /** Read-only diagnostic buffer: [shift, search range, target V, prior surface V]. */
  get diagnostics(): GPUBuffer | undefined { return this.buffers?.[5]; }
  get workSourceForQA(): GPUBuffer | undefined { return this.work; }
  /** After encode: the tiles whose cells may hold V or a phi corner below zero,
   * outside the sharpening list. Box words at byte 176 of `work` (origin, dims),
   * indirect dispatch at byte 48 of `dispatch`. */
  /** After a windowed-seed encode: `balanceWindow` joined with the phi window's
   * cells. Box words at byte 212 of `work`, indirect dispatch at byte 60. */
  get classifyWindow(): { work: GPUBuffer; dispatch: GPUBuffer } | undefined {
    return this.lastWindowedSeed ? this.balanceWindow : undefined;
  }
  get balanceWindow(): { work: GPUBuffer; dispatch: GPUBuffer } | undefined {
    return this.work && this.workDispatch ? { work: this.work, dispatch: this.workDispatch } : undefined;
  }
  /** `windowedSeed`: the host certifies this step's surfaceseed premises
   * (receiver-only V writes, a dust-floored phi census, no source or edit). */
  encode(encoder: GPUCommandEncoder, windowedSeed = false) {
    if(windowedSeed && !this.seedWindow)throw new Error("Windowed surface seed was not configured");
    this.lastWindowedSeed = windowedSeed;
    if(!this.buffers) throw new Error("Surface correction is not initialized");
    const clearNext=()=>{
      if(this.fieldPages?.scratch)encoder.clearBuffer(this.fieldPages.scratch.buffer,0,this.vertexCount*4);
      else encoder.clearBuffer(this.buffers![2]!);
    };
    if(this.fieldPages?.scratch)encoder.clearBuffer(this.fieldPages.scratch.buffer,Math.ceil(this.vertexCount/4)*16,this.vertexCount*4);
    else encoder.clearBuffer(this.buffers![1]!);
    // Dilation only rewrites the bounded region. Both parity buffers must be
    // zero outside it; the shared arena held unrelated stage data beforehand.
    if(this.compactWork)clearNext();
    const run=(entry:typeof entries[number],count:number,group=0)=>{
      const pass=encoder.beginComputePass({label:`Total surface volume: ${entry}`});
      pass.setPipeline(this.pipelines[entry]!);pass.setBindGroup(0,this.groups![group]!);
      const offset=entry==="measure"||entry==="dilate"?0:entry==="metric"?12:entry==="reduce"?24:entry==="seedBox"?36:undefined;
      if(this.workDispatch && offset!==undefined)pass.dispatchWorkgroupsIndirect(this.workDispatch,offset);
      else pass.dispatchWorkgroups(Math.min(count,65535),Math.ceil(count/65535));
      pass.end();
    };
    if(windowedSeed){
      encoder.copyBufferToBuffer(this.seedWindow!.phiRegion,28,this.work!,112,24);
      run("begin",1);
      encoder.copyBufferToBuffer(this.work!,164,this.workDispatch!,36,12);
      run("seedBox",0);
      const pass=encoder.beginComputePass({label:"Total surface volume: seedTiles"});
      pass.setPipeline(this.pipelines.seedTiles!);pass.setBindGroup(0,this.groups![0]!);
      pass.dispatchWorkgroupsIndirect(this.seedWindow!.receiverDispatch,0);pass.end();
    } else { run("begin",1);run("seed",Math.ceil(this.cellCount/64)); }
    if(this.work){
      run("finishWork",1);
      for(let k=0;k<3;k++)encoder.copyBufferToBuffer(this.work,64+16*k,this.workDispatch!,12*k,12);
      encoder.copyBufferToBuffer(this.work,200,this.workDispatch!,48,12);
      if(windowedSeed)encoder.copyBufferToBuffer(this.work,236,this.workDispatch!,60,12);
    }
    // The cell band ping-pongs and ends in the primary buffer. Metric writes
    // the per-vertex scale into the other buffer, which the bounded dilation
    // left holding cell values at indices the bounded metric never rewrites,
    // so it is cleared first; measure and apply then read it as `band`.
    for(let i=0;i<4;i++) run("dilate",Math.ceil(this.cellCount/64),i%2);
    if(this.compactWork&&!this.inPlace)clearNext();
    run("metric",Math.ceil(this.vertexCount/64));
    // Two 17-sample monotone volume curves refine the scalar root, using
    // deterministic reductions. No per-step CPU readback or iterative solve.
    for(let i=0;i<2;i++) {
      run("measure",Math.ceil(this.cellCount/64),1);run("reduce",Math.ceil(this.cellCount/4096));run("solve",1);
    }
    if(this.inPlace){
      const pass=encoder.beginComputePass({label:"Total surface volume: apply"});
      pass.setPipeline(this.inPlace.pipeline);pass.setBindGroup(0,this.inPlace.group);
      pass.dispatchWorkgroupsIndirect(this.workDispatch!,12);pass.end();
      return;
    }
    run("apply",Math.ceil(this.vertexCount/64),1);
    if(this.fieldPages) this.fieldPages.copy(encoder,this.output!,this.phi);
    else encoder.copyTextureToTexture({texture:this.output!},{texture:this.phi},this.dims.map(n=>n+1));
  }
  destroy() { this.work?.destroy();this.workDispatch?.destroy();this.output?.destroy();this.reverseParams?.destroy(); for(const buffer of this.buffers??[]) buffer.destroy(); }
}
