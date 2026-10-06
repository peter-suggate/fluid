/** Research-only capture of queue writes/submissions and a complete GPU-state
 * checkpoint. Replays an accepted frame without rerunning CPU scheduling.
 * Checkpoint restoration is explicitly outside the measured GPU interval.
 */
import assert from "node:assert/strict";
import {GPUStageTimestampRecorder} from "../lib/core/performance-trace";
type Build=(instrument?:(encoder:GPUCommandEncoder)=>GPUCommandEncoder)=>GPUCommandBuffer;
export class UniformFrozenGPUFrame {
 private readonly buffers=new Set<GPUBuffer>();
 private readonly textures=new Set<GPUTexture>();
 private readonly staging:GPUBuffer[]=[];
 private readonly snapshots:(GPUBuffer|GPUTexture)[]=[];
 private readonly commands:Build[]=[];
 private readonly encoded=new WeakMap<GPUCommandBuffer,Build>();
 private recording=false;
 private restore:Build|undefined;
 private checkpointBuffers=new Set<GPUBuffer>();
 private checkpointTextures=new Set<GPUTexture>();
 readonly statistics={buffers:0,textures:0,checkpointBytes:0,submissions:0,writes:0,commandBuffers:0};
 private readonly makeBuffer:GPUDevice["createBuffer"];
 private readonly makeTexture:GPUDevice["createTexture"];
 private readonly submit:GPUQueue["submit"];
 constructor(private readonly device:GPUDevice){
  const patch=(o:object,k:string,v:unknown)=>Object.defineProperty(o,k,{value:v,configurable:true,writable:true});
  this.makeBuffer=device.createBuffer.bind(device);this.makeTexture=device.createTexture.bind(device);this.submit=device.queue.submit.bind(device.queue);
  const write=device.queue.writeBuffer.bind(device.queue),writeTexture=device.queue.writeTexture.bind(device.queue);
  const makeEncoder=device.createCommandEncoder.bind(device);
  // WebGPU command buffers are single-use. Retain the encoding operations,
  // then create fresh command buffers for every replay.
  const copyArgument=(value:any):any=>{
   if(ArrayBuffer.isView(value))return "slice" in value?(value as any).slice():new DataView(value.buffer.slice(value.byteOffset,value.byteOffset+value.byteLength));
   if(value instanceof ArrayBuffer)return value.slice(0);
   if(Array.isArray(value))return value.map(copyArgument);
   if(value&&Object.getPrototypeOf(value)===Object.prototype)return Object.fromEntries(Object.entries(value).map(([k,v])=>[k,copyArgument(v)]));
   return value;
  };
  type Action={method:string;args:any[];nested?:Action[]};
  const replay=(target:any,actions:Action[])=>{for(const a of actions){const result=target[a.method](...a.args);if(a.nested)replay(result,a.nested);}};
  const wrap=(target:any,actions:Action[],finish?:(d?:GPUCommandBufferDescriptor)=>GPUCommandBuffer):any=>new Proxy(target,{get:(t,key)=>{
   const value=Reflect.get(t,key,t);if(typeof value!=="function")return value;
   if(key==="finish"&&finish)return finish;
   return (...args:any[])=>{const action:Action={method:String(key),args:args.map(copyArgument)};actions.push(action);const result=value.apply(t,args);
    if(key==="beginComputePass"||key==="beginRenderPass"){action.nested=[];return wrap(result,action.nested);}return result;};
  }});
  patch(device,"createCommandEncoder",(descriptor?:GPUCommandEncoderDescriptor)=>{
   const d=copyArgument(descriptor),encoder=makeEncoder(descriptor),actions:Action[]=[];
   return wrap(encoder,actions,(finishDescriptor?:GPUCommandBufferDescriptor)=>{
    const fd=copyArgument(finishDescriptor),buffer=encoder.finish(finishDescriptor);
    this.encoded.set(buffer,(instrument)=>{const raw=makeEncoder(d),e=instrument?instrument(raw):raw;replay(e,actions);return e.finish(fd);});return buffer;
   });
  });
  patch(device,"createBuffer",(d:GPUBufferDescriptor)=>{
   const mappable=!!(d.usage&(GPUBufferUsage.MAP_READ|GPUBufferUsage.MAP_WRITE));
   const b=this.makeBuffer({...d,usage:d.usage|(mappable?0:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST)});
   if(!mappable){this.buffers.add(b);const destroy=b.destroy.bind(b);patch(b,"destroy",()=>{this.buffers.delete(b);destroy();});}return b;
  });
  patch(device,"createTexture",(d:GPUTextureDescriptor)=>{
   assert.ok(!d.format.startsWith("depth")&&(d.mipLevelCount??1)===1&&(d.sampleCount??1)===1,"Frozen fixture only supports single-mip colour textures");
   const t=this.makeTexture({...d,usage:d.usage|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});this.textures.add(t);
   const destroy=t.destroy.bind(t);patch(t,"destroy",()=>{this.textures.delete(t);destroy();});return t;
  });
  patch(device.queue,"submit",(commands:Iterable<GPUCommandBuffer>)=>{const list=[...commands];if(this.recording){for(const c of list){const build=this.encoded.get(c);assert.ok(build,"Captured command buffer was not recorded");this.commands.push(build);}this.statistics.submissions++;}this.submit(list);});
  patch(device.queue,"writeTexture",(...args:Parameters<GPUQueue["writeTexture"]>)=>{assert.ok(!this.recording,"Texture uploads during the captured frame need an explicit replay implementation");return writeTexture(...args);});
  patch(device.queue,"writeBuffer",(buffer:GPUBuffer,offset:number,data:GPUAllowSharedBufferSource,dataOffset=0,size?:number)=>{
   if(this.recording){
    const typed=ArrayBuffer.isView(data),unit=typed&&"BYTES_PER_ELEMENT" in data?Number(data.BYTES_PER_ELEMENT):1;
    const bytes=typed?new Uint8Array(data.buffer,data.byteOffset,data.byteLength):new Uint8Array(data);
    const begin=dataOffset*unit,length=size===undefined?bytes.byteLength-begin:size*unit;
    assert.equal(length%4,0);assert.equal(offset%4,0);
    const source=this.makeBuffer({size:Math.max(4,length),usage:GPUBufferUsage.COPY_SRC,mappedAtCreation:true});
    new Uint8Array(source.getMappedRange()).set(bytes.subarray(begin,begin+length));source.unmap();this.staging.push(source);
    const e=device.createCommandEncoder();if(length)e.copyBufferToBuffer(source,0,buffer,offset,length);const command=e.finish();this.commands.push(this.encoded.get(command)!);this.statistics.writes++;
   }
   return write(buffer,offset,data,dataOffset,size);
  });
 }
 checkpoint():void{
  assert.ok(!this.restore&&!this.recording);
  const save=this.device.createCommandEncoder(),restore=this.device.createCommandEncoder();
  for(const b of this.buffers){
   const copy=this.makeBuffer({size:b.size,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});this.snapshots.push(copy);
   save.copyBufferToBuffer(b,0,copy,0,b.size);restore.copyBufferToBuffer(copy,0,b,0,b.size);
   this.statistics.buffers++;this.statistics.checkpointBytes+=b.size;
  }
  for(const t of this.textures){
   const extent=[t.width,t.height,t.depthOrArrayLayers];
   const copy=this.makeTexture({size:extent,dimension:t.dimension,format:t.format,usage:GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});this.snapshots.push(copy);
   save.copyTextureToTexture({texture:t},{texture:copy},extent);restore.copyTextureToTexture({texture:copy},{texture:t},extent);
   this.statistics.textures++;this.statistics.checkpointBytes+=extent.reduce((a,b)=>a*b,1)*(t.format.startsWith("rgba")?16:4);
  }
  this.checkpointBuffers=new Set(this.buffers);this.checkpointTextures=new Set(this.textures);
  const command=restore.finish();this.restore=this.encoded.get(command)!;this.submit([save.finish()]);
 }
 begin():void{assert.ok(this.restore&&!this.recording);this.recording=true;}
 end():void{
  this.recording=false;assert.ok(this.commands.length>0);
  assert.ok(this.buffers.size===this.checkpointBuffers.size&&[...this.buffers].every(b=>this.checkpointBuffers.has(b)),"GPU buffer allocation/lifetime changed during capture");
  assert.ok(this.textures.size===this.checkpointTextures.size&&[...this.textures].every(t=>this.checkpointTextures.has(t)),"GPU texture allocation/lifetime changed during capture");
  this.statistics.commandBuffers=this.commands.length;
 }
 async measure(samples=24,warmup=4):Promise<number[]>{
  assert.ok(this.restore&&!this.recording);
  const d=this.device,times:number[]=[];
  // Use the repository's completion-frontier timestamps. Independent start/end
  // markers can execute ahead of unrelated work on Metal and undercount a frame.
  await GPUStageTimestampRecorder.prepare(d);
  for(let i=0;i<samples+warmup;i++){
   this.submit([this.restore()]);await d.queue.onSubmittedWorkDone();
   const trace=new GPUStageTimestampRecorder(d,i,"physics","Frozen complete frame");trace.begin();
   try{
    const tape=this.commands.map(build=>build(e=>trace.instrument(e))),end=d.createCommandEncoder();
    trace.completePhase(end,{id:"other",label:"Frozen complete frame"});trace.resolve(end);
    this.submit([...tape,end.finish()]);
    const sample=await trace.read();assert.ok(sample&&sample.total_ms>0,"Frozen replay needs a valid completion-frontier timestamp");
    if(i>=warmup)times.push(sample.total_ms);
   }finally{trace.destroy();}
  }
  return times;
 }

 destroy():void{for(const x of [...this.snapshots,...this.staging])x.destroy();}
}
