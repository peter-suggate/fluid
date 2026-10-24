import type { UniformPagePool } from "./uniform-page-pool";
import { UNIFORM_LIVE_KEY_GROUPS, UniformLiveVisit, uniformLiveDispatch, uniformLiveListsWGSL } from "./uniform-live-lists";

/** A span of block-indexed words: block b's words are [from + words·b,
 * from + words·(b+1)) of the source, copied to the same block's range at
 * `to` of the target. */
export interface UniformBlockSpan{readonly from:number;readonly to:number;readonly words:number}
const SPANS=2;
const COPY_PASS:GPUComputePassDescriptor={label:"Uniform live block copy"};
type Binding=GPUBuffer|GPUBufferBinding;
const bindingOf=(b:Binding):GPUBufferBinding=>"buffer" in b?b:{buffer:b};
const same=(a:Binding,b:Binding):boolean=>a===b||("buffer" in a&&"buffer" in b&&a.buffer===b.buffer&&(a.offset??0)===(b.offset??0)&&a.size===b.size);

/** A whole-array copy of block-indexed words as a visit
 * (uniform-live-lists.ts): the visited blocks' words, never the
 * reservation's. For a copy whose target words of a block that holds no page
 * are a fixed point (the source's words there do not change while the block
 * stays free), so the one visit after the block leaves leaves what the whole
 * copy would. A sweep (UniformLiveVisit, and the first encode over new
 * buffers) copies every block. Up to two spans a copy; a fixed grid strides
 * the visited blocks' words. One instance a copy site: its spans and
 * bindings are its own. */
export class UniformLiveBlockCopy{
 private readonly visit:UniformLiveVisit;
 private readonly params:GPUBuffer;
 private readonly layout:GPUBindGroupLayout;
 private readonly words=new Uint32Array(4*SPANS);
 private readonly next=new Uint32Array(4*SPANS);
 private pipeline?:GPUComputePipeline;
 private bound?:{source:Binding;target:Binding;group:GPUBindGroup};
 constructor(private readonly device:GPUDevice,private readonly pool:UniformPagePool,private readonly label:string,spans:readonly UniformBlockSpan[]){
  this.visit=new UniformLiveVisit(pool);
  this.params=device.createBuffer({label,size:16*SPANS,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.layout=device.createBindGroupLayout({label,entries:[
   {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
   {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
   {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}}]});
  this.setSpans(spans);
 }
 async initialize():Promise<void>{
  const module=this.device.createShaderModule({label:this.label,code:uniformLiveBlockCopyWGSL(this.pool)});
  this.pipeline=await this.device.createComputePipelineAsync({label:this.label,layout:this.device.createPipelineLayout({bindGroupLayouts:[this.layout]}),compute:{module,entryPoint:"copy"}});
 }
 /** Moves the spans (a host write only when they change). */
 setSpans(spans:readonly UniformBlockSpan[]):void{
  if(spans.length<1||spans.length>SPANS)throw new Error(`${this.label}: ${spans.length} spans, 1..${SPANS}`);
  const next=this.next;next.fill(0);
  for(let i=0;i<spans.length;i++){const s=spans[i]!;next[4*i]=s.from;next[4*i+1]=s.to;next[4*i+2]=s.words;}
  let changed=false;for(let i=0;i<next.length;i++)if(next[i]!==this.words[i])changed=true;
  if(!changed)return;
  this.words.set(next);this.device.queue.writeBuffer(this.params,0,this.words);
 }
 /** Copies the spans' visited blocks from source to target. */
 encode(encoder:GPUCommandEncoder,source:Binding,target:Binding):void{
  if(!this.pipeline)throw new Error(`${this.label} is not initialized`);
  let sweep=this.visit.sweep();
  if(!this.bound||!same(this.bound.source,source)||!same(this.bound.target,target)){
   this.bound={source,target,group:this.device.createBindGroup({label:this.label,layout:this.layout,entries:[
    {binding:0,resource:{buffer:this.pool.state}},{binding:1,resource:bindingOf(source)},{binding:2,resource:bindingOf(target)},{binding:3,resource:{buffer:this.params}}]})};
   sweep=true;
  }
  const pass=encoder.beginComputePass(COPY_PASS);
  pass.setPipeline(this.pipeline);pass.setBindGroup(0,this.bound.group);
  uniformLiveDispatch(pass,UNIFORM_LIVE_KEY_GROUPS,sweep);pass.end();
 }
 destroy():void{this.params.destroy();}
}

/** A lane a word of the visited blocks' spans, in visit order. */
export function uniformLiveBlockCopyWGSL(pool:UniformPagePool):string{
 return /* wgsl */`
${uniformLiveListsWGSL(0,0,pool.layout)}
@group(0) @binding(1) var<storage,read> src:array<u32>;
@group(0) @binding(2) var<storage,read_write> dst:array<u32>;
@group(0) @binding(3) var<uniform> spans:array<vec4u,${SPANS}>;
@compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) gid:vec3u,@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) groups:vec3u){
 if(wid.y!=0u){return;}
 let sweep=ulSweep(groups);let a=spans[0];let b=spans[1];let words=a.z+b.z;
 let total=ulVisitCount(sweep)*words;
 for(var j=gid.x;j<total;j+=64u*groups.x){
  let block=ulVisitBlock(j/words,sweep);let w=j%words;
  if(w<a.z){dst[a.y+a.z*block+w]=src[a.x+a.z*block+w];}
  else{let v=w-a.z;dst[b.y+b.z*block+v]=src[b.x+b.z*block+v];}
 }
}`;
}
