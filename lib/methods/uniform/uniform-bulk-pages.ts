import {BULK16_PAGE,BULK16_PAGE_WORDS,bulk16FaceIndexWGSL,bulk16PageLayout} from "./uniform-bulk-pages-reference";

export const BULK16_STATUS={capacity:0,free:1,head:2,pending:3,pops:4,live:5,fatal:6,fatalSlot:7,generation:8,version:9,words:32} as const;
export const BULK16_FATAL={exhausted:1,invalidSlot:2,invalidTransition:4} as const;
type Layout=ReturnType<typeof bulk16PageLayout>;

/** The live list for a reader compiled without the capacity (the
 * renderer's static modules), bulk16PageLayout's offsets taken from the
 * status capacity word. A binding shorter than the status words (no store
 * bound) has no live slot. A page is a window page: the frame page is not
 * yet removed. */
export function bulk16LiveWGSL(name:string):string{return /* wgsl */`
fn ${name}Live()->u32{if(arrayLength(&${name})<${BULK16_STATUS.words}u){return 0u;}return ${name}[${BULK16_STATUS.live}u];}
fn ${name}LivePage(i:u32)->vec3i{
 let capacity=${name}[${BULK16_STATUS.capacity}u];
 let records=((${BULK16_STATUS.words}u+3u*capacity+63u)/64u)*64u;
 let at=records+${BULK16_PAGE_WORDS}u*${name}[${BULK16_STATUS.words}u+capacity+i]+${BULK16_PAGE.page}u;
 return bitcast<vec3i>(vec3u(${name}[at],${name}[at+1u],${name}[at+2u]));
}
`;}

/** The mutable slot allocator. Retired slots are quarantined until the next
 * begin dispatch; allocation never grows a buffer or evicts another page.
 * Callers publish directory changes only after the transfer dispatch ends. */
export function bulk16LifecycleWGSL(layout:Layout,group=0,binding=0):string{
 const H=BULK16_STATUS;
 return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read_write> bulkPages:array<atomic<u32>>;
const BP_CAPACITY=${layout.capacity}u;const BP_FREE=${layout.free}u;const BP_LIVE=${layout.live}u;const BP_PENDING=${layout.pending}u;const BP_RECORDS=${layout.records}u;
const BP_STRIDE=${BULK16_PAGE_WORDS}u;
${bulk16FaceIndexWGSL}
fn bpAt(slot:u32,word:u32)->u32{return BP_RECORDS+BP_STRIDE*slot+word;}
fn bpRead(slot:u32,word:u32)->u32{return atomicLoad(&bulkPages[bpAt(slot,word)]);}
fn bpWrite(slot:u32,word:u32,value:u32){atomicStore(&bulkPages[bpAt(slot,word)],value);}
fn bpFail(reason:u32,slot:u32){
 if(atomicOr(&bulkPages[${H.fatal}u],reason)==0u){atomicStore(&bulkPages[${H.fatalSlot}u],slot);}
}
fn bpClaim(page:vec3i)->u32{
 let pop=atomicAdd(&bulkPages[${H.pops}u],1u);
 if(pop>=atomicLoad(&bulkPages[${H.free}u])){bpFail(${BULK16_FATAL.exhausted}u,0u);return 0u;}
 let head=atomicLoad(&bulkPages[${H.head}u]);let slot=atomicLoad(&bulkPages[BP_FREE+(head+pop)%BP_CAPACITY]);
 if(slot==0u||slot>BP_CAPACITY){bpFail(${BULK16_FATAL.invalidSlot}u,slot);return 0u;}
 // Clear the prior incarnation, including pressures and seam heads, before
 // this invocation fills the new page and publishes its directory word.
 for(var w=0u;w<BP_STRIDE;w++){bpWrite(slot,w,0u);}
 for(var a=0u;a<3u;a++){bpWrite(slot,a,bitcast<u32>(page[a]));}
 bpWrite(slot,${BULK16_PAGE.generation}u,atomicLoad(&bulkPages[${H.generation}u]));
 bpWrite(slot,${BULK16_PAGE.open}u,255u);bpWrite(slot,${BULK16_PAGE.flags}u,1u);
 return slot;
}
fn bpRetire(slot:u32){
 if(slot==0u||slot>BP_CAPACITY){bpFail(${BULK16_FATAL.invalidSlot}u,slot);return;}
 loop{
  let changed=atomicCompareExchangeWeak(&bulkPages[bpAt(slot,${BULK16_PAGE.flags}u)],1u,2u);
  if(changed.exchanged){break;}
  if(changed.old_value!=1u){bpFail(${BULK16_FATAL.invalidTransition}u,slot);return;}
 }
 let i=atomicAdd(&bulkPages[${H.pending}u],1u);
 atomicStore(&bulkPages[BP_PENDING+i],slot);
}
`;
}

/** Fixed storage and lifecycle dispatches. Field transport and the coupled
 * pressure operator bind state directly; these dispatches only manage slot
 * identity, quarantine, the ascending live list and sticky failures. */
export class UniformBulkPages{
 readonly layout:Layout;
 /** The store's one binding: its own buffer, or (adopt) a range of the
  * pressure arena's pool B, which the solve's kernels then read without a
  * binding of their own (pNative is at the per-stage storage limit). */
 private store:{buffer:GPUBuffer;offset:number;owned:boolean};
 private begin?:GPUComputePipeline;
 private finish?:GPUComputePipeline;
 private group?:GPUBindGroup;
 get state():GPUBuffer{return this.store.buffer;}
 get binding():GPUBufferBinding{return {buffer:this.store.buffer,offset:this.store.offset,size:this.layout.bytes};}
 get allocatedBytes():number{return this.store.owned?this.store.buffer.size:0;}
 constructor(private readonly device:GPUDevice,capacity:number){
  this.layout=bulk16PageLayout(capacity);
  if(this.layout.bytes>Math.min(device.limits.maxBufferSize,device.limits.maxStorageBufferBindingSize))throw new Error(`BULK16 capacity ${capacity} needs ${this.layout.bytes} bytes in one storage binding`);
  this.store={buffer:device.createBuffer({label:"Uniform whole-page BULK16 slots",size:this.layout.bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST}),offset:0,owned:true};
  this.seed();
 }
 private seed():void{
  const capacity=this.layout.capacity;
  const seed=new Uint32Array(this.layout.free+capacity);seed[BULK16_STATUS.capacity]=capacity;seed[BULK16_STATUS.free]=capacity;seed[BULK16_STATUS.version]=0xb0160001;
  for(let i=0;i<capacity;i++)seed[this.layout.free+i]=i+1;
  this.device.queue.writeBuffer(this.store.buffer,this.store.offset,seed);
 }
 /** Before any bind (initialize, the readers' groups): move the store into
  * host (layout.bytes at a storage offset), seeded as the constructor's was. */
 adopt(host:GPUBufferBinding):void{
  if(this.group)throw new Error("BULK16 store adopted after its lifecycle bound it");
  const offset=host.offset??0;
  if((host.size??host.buffer.size-offset)<this.layout.bytes||offset%this.device.limits.minStorageBufferOffsetAlignment!==0)throw new Error("BULK16 store host range is too small or misaligned");
  if(this.store.owned)this.store.buffer.destroy();
  this.store={buffer:host.buffer,offset,owned:false};
  this.seed();
 }
 async initialize():Promise<void>{
  const H=BULK16_STATUS;
  const module=this.device.createShaderModule({label:"Uniform BULK16 slot lifecycle",code:bulk16LifecycleWGSL(this.layout)+/* wgsl */`
var<workgroup> bpScan:array<u32,256>;
@compute @workgroup_size(256) fn begin(@builtin(local_invocation_index) lane:u32){
 let pending=atomicLoad(&bulkPages[${H.pending}u]);let free=atomicLoad(&bulkPages[${H.free}u]);let head=atomicLoad(&bulkPages[${H.head}u]);
 for(var i=lane;i<pending;i+=256u){
  let slot=atomicLoad(&bulkPages[BP_PENDING+i]);bpWrite(slot,${BULK16_PAGE.flags}u,0u);
  atomicStore(&bulkPages[BP_FREE+(head+free+i)%BP_CAPACITY],slot);
 }
 storageBarrier();
 if(lane==0u){atomicStore(&bulkPages[${H.free}u],free+pending);atomicStore(&bulkPages[${H.pending}u],0u);atomicStore(&bulkPages[${H.pops}u],0u);atomicAdd(&bulkPages[${H.generation}u],1u);}
}
@compute @workgroup_size(256) fn finish(@builtin(local_invocation_index) lane:u32){
 let per=(BP_CAPACITY+255u)/256u;let first=min(lane*per,BP_CAPACITY)+1u;let last=min((lane+1u)*per,BP_CAPACITY)+1u;
 var count=0u;
 for(var slot=first;slot<last;slot++){count+=u32(bpRead(slot,${BULK16_PAGE.flags}u)==1u);}
 bpScan[lane]=count;workgroupBarrier();
 for(var stride=1u;stride<256u;stride*=2u){
  var add=0u;if(lane>=stride){add=bpScan[lane-stride];}
  workgroupBarrier();bpScan[lane]+=add;workgroupBarrier();
 }
 var at=bpScan[lane]-count;
 for(var slot=first;slot<last;slot++){if(bpRead(slot,${BULK16_PAGE.flags}u)==1u){atomicStore(&bulkPages[BP_LIVE+at],slot);at++;}}
 if(lane==255u){
  let free=atomicLoad(&bulkPages[${H.free}u]);let pops=min(free,atomicLoad(&bulkPages[${H.pops}u]));
  atomicStore(&bulkPages[${H.head}u],(atomicLoad(&bulkPages[${H.head}u])+pops)%BP_CAPACITY);
  atomicStore(&bulkPages[${H.free}u],free-pops);atomicStore(&bulkPages[${H.live}u],bpScan[255]);
 }
}
`});
  const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
  if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
  const layout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
  const pipelineLayout=this.device.createPipelineLayout({bindGroupLayouts:[layout]});
  [this.begin,this.finish]=await Promise.all(["begin","finish"].map(entryPoint=>this.device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
  this.group=this.device.createBindGroup({layout,entries:[{binding:0,resource:this.binding}]});
 }
 private encode(encoder:GPUCommandEncoder,pipeline:GPUComputePipeline|undefined,label:string):void{
  if(!pipeline||!this.group)throw new Error("BULK16 lifecycle is not initialized");
  const pass=encoder.beginComputePass({label});pass.setPipeline(pipeline);pass.setBindGroup(0,this.group);pass.dispatchWorkgroups(1);pass.end();
 }
 encodeBegin(encoder:GPUCommandEncoder):void{this.encode(encoder,this.begin,"BULK16 reclaim quarantine");}
 encodeFinish(encoder:GPUCommandEncoder):void{this.encode(encoder,this.finish,"BULK16 publish live slots");}
 encodeReceipt(encoder:GPUCommandEncoder,target:GPUBuffer,offset=0):void{encoder.copyBufferToBuffer(this.store.buffer,this.store.offset,target,offset,4*BULK16_STATUS.words);}
 destroy():void{if(this.store.owned)this.store.buffer.destroy();}
}

export function bulk16Failure(receipt:ArrayLike<number>):string|undefined{
 const fatal=receipt[BULK16_STATUS.fatal]??0;if(!fatal)return;
 const slot=receipt[BULK16_STATUS.fatalSlot]??0;
 if(fatal&BULK16_FATAL.exhausted)return `BULK16 fixed slot capacity ${receipt[BULK16_STATUS.capacity]} exhausted`;
 return `BULK16 invalid slot lifecycle (flags ${fatal}, slot ${slot})`;
}
