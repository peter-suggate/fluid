/** Buffer copies and clears of the advance's encoders as compute launches.
 *
 * On Dawn's Metal backend a copyBufferToBuffer or clearBuffer between two
 * compute passes is a blit encoder between two compute encoders, and that
 * switch drains the GPU: ~40 us per blit run (600 dependent launches on an
 * M1 Max, 104 passes, one copy between each: 6.8 ms against 2.3 ms with no
 * copy; consecutive passes share a Metal encoder, a dependent launch ~5 us).
 * A NB-FLIP frame issued 40 such runs. A copy or clear whose buffers can be
 * bound as storage is instead a word-for-word copy (or zero fill) in a
 * compute launch: the same bytes, so results are bit-identical. Everything
 * else (mappable readback targets, uniform-only buffers, buffers beyond the
 * storage binding limit, unaligned ranges) stays a native blit.
 *
 * Consecutive ones are one launch where they are independent: an encoder's
 * copies and clears wait in a fixed queue until its next other command, and
 * then run in order as batches of up to BATCH_RECORDS records over up to
 * BATCH_BUFFERS distinct buffers, a batch ending where a record would read
 * or write what an earlier one of it writes (or write what one reads).
 *
 * adopt() installs the forms on one encoder (an own property per recording
 * method; the prototype's are the native ones). Per-batch state is
 * immutable and made at first sight: a bind group per buffer tuple, a
 * 512-byte record table in a uniform buffer read at a dynamic offset,
 * matched word for word. A steady frame allocates nothing; a full table
 * store leaves the batch to native blits.
 *
 * After the regional frame plan is sealed, logical stages may share one
 * WebGPU compute pass too: dispatch ordering and its resource barriers stay
 * intact, while per-pass validation/encoding and redundant binding resets
 * disappear. Native commands, explicit timestamps and submission close it. */
const passEpochs=new WeakMap<GPUComputePassEncoder,number>();
/** Binding caches must distinguish logical stages sharing a native pass. */
export function uniformComputePassEpoch(pass:GPUComputePassEncoder):number{return passEpochs.get(pass)??0;}
const TABLES=1024;
const TABLE_BYTES=512;
const BATCH_RECORDS=15;
const TABLE_WORDS=4+8*BATCH_RECORDS;
const QUEUE=64;
const LANES=256;
const MAX_GROUPS=1024;
const COPY=0,FILL=1;
const PASS:GPUComputePassDescriptor={label:"Uniform compute copy"};
type Native={
 copyBufferToBuffer:(s:GPUBuffer,so:number,d:GPUBuffer,dO:number,n:number)=>void;
 clearBuffer:(b:GPUBuffer,o?:number,n?:number)=>void;
 beginComputePass:(d?:GPUComputePassDescriptor)=>GPUComputePassEncoder;
 beginRenderPass:(d:GPURenderPassDescriptor)=>GPURenderPassEncoder;
 finish:(d?:GPUCommandBufferDescriptor)=>GPUCommandBuffer;
 resolveQuerySet:(q:GPUQuerySet,f:number,c:number,d:GPUBuffer,o:number)=>void;
 copyBufferToTexture:(a:unknown,b:unknown,c:unknown)=>void;
 copyTextureToBuffer:(a:unknown,b:unknown,c:unknown)=>void;
 copyTextureToTexture:(a:unknown,b:unknown,c:unknown)=>void;
};
/** A buffer tuple's node: the next slot's buffers, then (at the last slot) its bind group and tables. */
interface Node{next:WeakMap<GPUBuffer,Node>;group?:GPUBindGroup;tables:{words:Uint32Array;offset:number}[]}

export class UniformComputeBlit {
 /** Enabled only for the sealed frame schedule. Dispatch boundaries still
  * synchronize dependent storage; native commands and timestamped passes fence. */
 private sharing=false;
 private pass?:GPUComputePassEncoder;
 private passEncoder?:GPUCommandEncoder;
 private endPass?:()=>void;
 private boundMask=0;
 private pendingMask=0;
 setSharedSchedule(enabled:boolean):void{if(this.sharing!==enabled){this.closePass();this.sharing=enabled;}}
 private closePass():void{this.endPass?.();this.pass=undefined;this.passEncoder=undefined;this.endPass=undefined;this.boundMask=0;this.pendingMask=0;}
 private compute(encoder:GPUCommandEncoder,descriptor?:GPUComputePassDescriptor):GPUComputePassEncoder{
  if(!this.sharing||descriptor?.timestampWrites){this.closePass();return this.native!.beginComputePass.call(encoder,descriptor);}
  if(this.passEncoder!==encoder)this.closePass();
  if(!this.pass){
   const pass=this.native!.beginComputePass.call(encoder,{label:"Uniform shared frame schedule"});
   this.endPass=pass.end.bind(pass);pass.end=()=>{};this.pass=pass;this.passEncoder=encoder;
   const setGroup=pass.setBindGroup.bind(pass),dispatch=pass.dispatchWorkgroups.bind(pass);
   // Reset only resources the next stage does not replace. Clearing every
   // slot at every stage adds hundreds of host/driver commands per frame.
   pass.setBindGroup=((index:number,group:GPUBindGroup|null,offsets?:Iterable<number>|Uint32Array,start?:number,length?:number)=>{
    const bit=1<<index;this.pendingMask&=~bit;
    if(group)this.boundMask|=bit;else this.boundMask&=~bit;
    if(start===undefined)setGroup(index,group,offsets as number[]|undefined);
    else setGroup(index,group,offsets as Uint32Array,start,length!);
   }) as GPUComputePassEncoder["setBindGroup"];
   pass.dispatchWorkgroups=(x:number,y?:number,z?:number)=>{
    for(let bits=this.pendingMask;bits;bits&=bits-1){const index=31-Math.clz32(bits&-bits);setGroup(index,null);}
    this.boundMask&=~this.pendingMask;this.pendingMask=0;dispatch(x,y,z);
   };
  }
  passEpochs.set(this.pass,(passEpochs.get(this.pass)??0)+1);
  this.pendingMask=this.boundMask;
  return this.pass;
 }
 private readonly buffers:number;
 private readonly tables:GPUBuffer;
 private tableCount=0;
 /** A distinct unused-slot buffer a slot: Dawn refuses writable bindings that alias. */
 private readonly dummies:GPUBuffer[];
 private readonly layout:GPUBindGroupLayout;
 private pipeline?:GPUComputePipeline;
 private readonly limit:number;
 private readonly root:Node={next:new WeakMap(),tables:[]};
 private readonly dynamic=[0];
 private native?:Native;
 private rawNative?:Native;
 // The queue: kind, source (COPY), word ranges, target.
 private readonly kind=new Uint8Array(QUEUE);
 private readonly source:(GPUBuffer|undefined)[]=new Array(QUEUE).fill(undefined);
 private readonly target:(GPUBuffer|undefined)[]=new Array(QUEUE).fill(undefined);
 private readonly from=new Uint32Array(QUEUE);
 private readonly to=new Uint32Array(QUEUE);
 private readonly words=new Uint32Array(QUEUE);
 private queued=0;
 private queuedOn?:GPUCommandEncoder;
 // A batch under assembly: its buffers (slot order) and table.
 private readonly slots:GPUBuffer[];
 private readonly table=new Uint32Array(TABLE_WORDS);
 /** Records launched, batches launched, and copies or clears left native (diagnostics). */
 launched=0;
 batches=0;
 nativeCount=0;
 private readonly forms:Record<keyof Native,unknown>;
 constructor(private readonly device:GPUDevice){
  this.limit=device.limits.maxStorageBufferBindingSize;
  this.buffers=Math.min(8,device.limits.maxStorageBuffersPerShaderStage);
  this.tables=device.createBuffer({label:"Uniform compute blit tables",size:TABLES*TABLE_BYTES,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  this.dummies=Array.from({length:this.buffers},()=>device.createBuffer({label:"Uniform compute blit unused slot",size:16,usage:GPUBufferUsage.STORAGE}));
  this.slots=this.dummies.slice();
  this.layout=device.createBindGroupLayout({label:"Uniform compute blit",entries:[
   ...Array.from({length:this.buffers},(_,binding):GPUBindGroupLayoutEntry=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}})),
   {binding:this.buffers,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform",hasDynamicOffset:true,minBindingSize:4*TABLE_WORDS}}]});
  const blit=this;
  this.forms={
   copyBufferToBuffer:function(this:GPUCommandEncoder,s:GPUBuffer,so:number,d:GPUBuffer,dO:number,n:number){blit.encodeCopy(this,s,so,d,dO,n);},
   clearBuffer:function(this:GPUCommandEncoder,b:GPUBuffer,o?:number,n?:number){blit.encodeClear(this,b,o,n);},
   beginComputePass:function(this:GPUCommandEncoder,d?:GPUComputePassDescriptor){blit.flush(this);return blit.compute(this,d);},
   beginRenderPass:function(this:GPUCommandEncoder,d:GPURenderPassDescriptor){blit.flush(this);blit.closePass();return blit.native!.beginRenderPass.call(this,d);},
   finish:function(this:GPUCommandEncoder,d?:GPUCommandBufferDescriptor){blit.flush(this);blit.closePass();return blit.native!.finish.call(this,d);},
   resolveQuerySet:function(this:GPUCommandEncoder,q:GPUQuerySet,f:number,c:number,d:GPUBuffer,o:number){blit.flush(this);blit.closePass();blit.native!.resolveQuerySet.call(this,q,f,c,d,o);},
   copyBufferToTexture:function(this:GPUCommandEncoder,a:unknown,b:unknown,c:unknown){blit.flush(this);blit.closePass();blit.native!.copyBufferToTexture.call(this,a,b,c);},
   copyTextureToBuffer:function(this:GPUCommandEncoder,a:unknown,b:unknown,c:unknown){blit.flush(this);blit.closePass();blit.native!.copyTextureToBuffer.call(this,a,b,c);},
   copyTextureToTexture:function(this:GPUCommandEncoder,a:unknown,b:unknown,c:unknown){blit.flush(this);blit.closePass();blit.native!.copyTextureToTexture.call(this,a,b,c);},
  };
 }
 async initialize():Promise<void>{
  const n=this.buffers,cases=(body:(i:number)=>string)=>Array.from({length:n},(_,i)=>i<n-1?` case ${i}u:{${body(i)}}`:` default:{${body(i)}}`).join("");
  const code=/* wgsl */`
${Array.from({length:n},(_,i)=>`@group(0) @binding(${i}) var<storage,read_write> blit${i}:array<u32>;`).join("\n")}
struct BlitRecord{a:vec4u,b:vec4u}
struct BlitTable{head:vec4u,records:array<BlitRecord,${BATCH_RECORDS}>}
@group(0) @binding(${n}) var<uniform> table:BlitTable;
fn blitLoad(slot:u32,i:u32)->u32{switch slot{${cases(i=>`return blit${i}[i];`)}}}
fn blitStore(slot:u32,i:u32,v:u32){switch slot{${cases(i=>`blit${i}[i]=v;`)}}}
// A record: a (source slot, word), (target slot, word); b (words, kind, first word of the batch's sequence).
// A lane's words rise, so its record only moves on.
@compute @workgroup_size(${LANES}) fn main(@builtin(global_invocation_id) id:vec3u,@builtin(num_workgroups) groups:vec3u){
 let count=table.head.x;let total=table.head.y;var r=0u;
 for(var i=id.x;i<total;i+=groups.x*${LANES}u){
  loop{if(r+1u<count&&i>=table.records[r+1u].b.z){r++;}else{break;}}
  let a=table.records[r].a;let b=table.records[r].b;let j=i-b.z;
  var v=0u;if(b.y==${COPY}u){v=blitLoad(a.x,a.y+j);}
  blitStore(a.z,a.w+j,v);
 }
}`;
  const module=this.device.createShaderModule({label:"Uniform compute copy",code});
  this.pipeline=await this.device.createComputePipelineAsync({label:"Uniform compute copy",layout:this.device.createPipelineLayout({bindGroupLayouts:[this.layout]}),compute:{module,entryPoint:"main"}});
 }
 /** Install the compute forms on `encoder`, the frame's one open encoder.
  * Before initialize() completes the encoder keeps its native forms. */
 adopt(encoder:GPUCommandEncoder,instrument?:(encoder:GPUCommandEncoder)=>GPUCommandEncoder):GPUCommandEncoder{
  if(encoder.finish===this.forms.finish)return encoder;
  if(!this.pipeline)return instrument?.(encoder)??encoder;
  if(!this.rawNative){
   const proto=Object.getPrototypeOf(encoder) as Record<string,unknown>,native:Record<string,unknown>={};
   for(const key of Object.keys(this.forms)){const f=proto[key];if(typeof f!=="function")throw new Error(`Uniform compute blit: the encoder has no ${key}`);native[key]=f;}
   this.rawNative=native as unknown as Native;
  }
  this.native=this.rawNative;
  if(instrument){
   // Instrument physical passes after scheduling. A recorder outside this
   // layer timestamps each logical pass and disables sharing. Keep a stable
   // view of the native methods so its encoder-break copies cannot recurse
   // into the forms installed below or become deferred compute copies.
   const native=this.rawNative;
   const view=new Proxy(encoder,{get(target,key){
    const value=key in native?native[key as keyof Native]:Reflect.get(target,key,target);
    return typeof value==="function"?value.bind(target):value;
   }});
   const measured=instrument(view),methods:Record<string,unknown>={};
   for(const key of Object.keys(this.forms))methods[key]=measured[key as keyof Native].bind(measured);
   this.native=methods as unknown as Native;
  }
  const target=encoder as unknown as Record<string,unknown>;
  for(const key in this.forms)target[key]=this.forms[key as keyof Native];
  return encoder;
 }
 /** Close physical work before a timing boundary; never split untimed stages. */
 finishStage(encoder:GPUCommandEncoder):void{this.flush(encoder);this.closePass();}
 private bindable(buffer:GPUBuffer):boolean{return (buffer.usage&GPUBufferUsage.STORAGE)!==0&&buffer.size<=this.limit&&buffer.size%4===0;}
 private enqueue(encoder:GPUCommandEncoder,kind:number,source:GPUBuffer|undefined,from:number,target:GPUBuffer,to:number,words:number):void{
  if(this.queued>0&&this.queuedOn!==encoder)throw new Error("Uniform compute blit: copies queued on another encoder");
  if(this.queued===QUEUE)this.flush(encoder);
  const k=this.queued++;this.queuedOn=encoder;
  this.kind[k]=kind;this.source[k]=source;this.target[k]=target;this.from[k]=from;this.to[k]=to;this.words[k]=words;
 }
 private encodeCopy(encoder:GPUCommandEncoder,source:GPUBuffer,sourceOffset:number,destination:GPUBuffer,destinationOffset:number,size:number):void{
  if(size===0)return;
  if(sourceOffset%4===0&&destinationOffset%4===0&&size%4===0&&this.bindable(source)&&this.bindable(destination)){this.enqueue(encoder,COPY,source,sourceOffset/4,destination,destinationOffset/4,size/4);return;}
  this.flush(encoder);this.closePass();this.nativeCount++;this.native!.copyBufferToBuffer.call(encoder,source,sourceOffset,destination,destinationOffset,size);
 }
 private encodeClear(encoder:GPUCommandEncoder,buffer:GPUBuffer,offset=0,size?:number):void{
  const bytes=size??buffer.size-offset;if(bytes===0)return;
  if(offset%4===0&&bytes%4===0&&this.bindable(buffer)){this.enqueue(encoder,FILL,undefined,0,buffer,offset/4,bytes/4);return;}
  this.flush(encoder);this.closePass();this.nativeCount++;this.native!.clearBuffer.call(encoder,buffer,offset,size);
 }
 /** Whether record k's ranges meet record j's write (or k's write meets j's read). */
 private conflicts(k:number,j:number):boolean{
  const meet=(a:GPUBuffer|undefined,a0:number,b:GPUBuffer|undefined,b0:number,n:number,m:number)=>a!==undefined&&a===b&&a0<b0+m&&b0<a0+n;
  const tk=this.target[k],tj=this.target[j],nk=this.words[k],nj=this.words[j];
  return meet(tj,this.to[j],tk,this.to[k],nj,nk)||meet(tj,this.to[j],this.source[k],this.from[k],nj,nk)||meet(this.source[j],this.from[j],tk,this.to[k],nj,nk);
 }
 private slotOf(buffer:GPUBuffer,used:number):number{for(let s=0;s<used;s++)if(this.slots[s]===buffer)return s;return -1;}
 /** Launch the queue in order: each batch the longest independent run that fits. */
 private flush(encoder:GPUCommandEncoder):void{
  const n=this.queued;if(n===0)return;
  this.queued=0;this.queuedOn=undefined;
  let i=0;
  while(i<n){
   let used=0,j=i,total=0;
   for(;j<n&&j-i<BATCH_RECORDS;j++){
    let conflict=false;for(let k=i;k<j;k++)if(this.conflicts(k,j)){conflict=true;break;}
    if(conflict)break;
    const s=this.source[j],t=this.target[j]!;
    const needS=s!==undefined&&this.slotOf(s,used)<0?1:0,needT=this.slotOf(t,used)<0&&t!==s?1:0;
    if(used+needS+needT>this.buffers)break;
    if(needS)this.slots[used++]=s!;if(needT)this.slots[used++]=t;
    const r=4+8*(j-i);
    this.table[r]=s!==undefined?this.slotOf(s,used):0;this.table[r+1]=this.from[j]!;this.table[r+2]=this.slotOf(t,used);this.table[r+3]=this.to[j]!;
    this.table[r+4]=this.words[j]!;this.table[r+5]=this.kind[j]!;this.table[r+6]=total;this.table[r+7]=0;
    total+=this.words[j]!;
   }
   this.table[0]=j-i;this.table[1]=total;this.table[2]=0;this.table[3]=0;
   for(let s=used;s<this.buffers;s++)this.slots[s]=this.dummies[s]!;
   this.launch(encoder,i,j,total);
   i=j;
  }
  for(let k=0;k<n;k++){this.source[k]=undefined;this.target[k]=undefined;}
 }
 private launch(encoder:GPUCommandEncoder,first:number,end:number,total:number):void{
  let node=this.root;
  for(let s=0;s<this.buffers;s++){const b=this.slots[s]!;let next=node.next.get(b);if(!next){next={next:new WeakMap(),tables:[]};node.next.set(b,next);}node=next;}
  if(!node.group)node.group=this.device.createBindGroup({label:"Uniform compute blit",layout:this.layout,entries:[
   ...this.slots.map((buffer,binding)=>({binding,resource:{buffer}})),{binding:this.buffers,resource:{buffer:this.tables,size:4*TABLE_WORDS}}]});
  const length=4+8*(end-first);let offset=-1;
  for(const known of node.tables){
   let same=known.words.length===length;for(let w=0;same&&w<length;w++)same=known.words[w]===this.table[w];
   if(same){offset=known.offset;break;}
  }
  if(offset<0){
   if(this.tableCount===TABLES){
    this.closePass();
    // No table left: these records as the native blits they were.
    for(let k=first;k<end;k++){this.nativeCount++;if(this.kind[k]===FILL)this.native!.clearBuffer.call(encoder,this.target[k]!,4*this.to[k]!,4*this.words[k]!);else this.native!.copyBufferToBuffer.call(encoder,this.source[k]!,4*this.from[k]!,this.target[k]!,4*this.to[k]!,4*this.words[k]!);}
    return;
   }
   offset=TABLE_BYTES*this.tableCount++;
   const words=this.table.slice(0,length);node.tables.push({words,offset});
   this.device.queue.writeBuffer(this.tables,offset,words);
  }
  const pass=this.compute(encoder,PASS);pass.setPipeline(this.pipeline!);this.dynamic[0]=offset;pass.setBindGroup(0,node.group,this.dynamic);
  pass.dispatchWorkgroups(Math.min(MAX_GROUPS,Math.ceil(total/LANES)));pass.end();
  this.launched+=end-first;this.batches++;
 }
 destroy():void{this.tables.destroy();for(const d of this.dummies)d.destroy();}
}
