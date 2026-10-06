/** Diagnostic only: split dispatches into timestamped passes for one frame.
 * This deliberately changes command overhead and is never a frame-time arm. */
export class UniformKernelTimestamps {
 readonly device:GPUDevice;
 private active=false;
 private readonly labels:string[]=[];
 private readonly shaders=new WeakMap<GPUShaderModule,string>();
 private readonly pipelines=new WeakMap<GPUComputePipeline,string>();
 private readonly queries:GPUQuerySet;
 constructor(private readonly raw:GPUDevice,private readonly capacity=4096){
  this.queries=raw.createQuerySet({type:"timestamp",count:capacity});
  this.device=new Proxy(raw,{get:(target,key)=>{
   if(key==="createShaderModule")return (d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);this.shaders.set(m,d.label??"shader");return m;};
   if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(d);this.pipelines.set(p,`${this.shaders.get(d.compute.module)??d.label??"pipeline"} :: ${d.compute.entryPoint}`);return p;};
   if(key==="createComputePipeline")return(d:GPUComputePipelineDescriptor)=>{const p=target.createComputePipeline(d);this.pipelines.set(p,`${this.shaders.get(d.compute.module)??d.label??"pipeline"} :: ${d.compute.entryPoint}`);return p;};
   if(key==="createCommandEncoder")return(d?:GPUCommandEncoderDescriptor)=>this.encoder(target.createCommandEncoder(d));
   const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
 }
 start():void{this.labels.length=0;this.active=true;}
 private encoder(encoder:GPUCommandEncoder):GPUCommandEncoder{
  return new Proxy(encoder,{get:(target,key)=>{
   if(key==="beginComputePass")return(d?:GPUComputePassDescriptor)=>{
    if(!this.active)return target.beginComputePass(d);
    const original=d?.timestampWrites;
    if(original?.beginningOfPassWriteIndex!==undefined)target.beginComputePass({timestampWrites:{querySet:original.querySet,beginningOfPassWriteIndex:original.beginningOfPassWriteIndex}}).end();
    const groups=new Map<number,unknown[]>();let pipeline:GPUComputePipeline|undefined;
    return new Proxy({} as GPUComputePassEncoder,{get:(_pass,property)=>{
     if(property==="setPipeline")return(p:GPUComputePipeline)=>{pipeline=p;};
     if(property==="setBindGroup")return(...args:unknown[])=>{const copy=[...args];if(Array.isArray(copy[2]))copy[2]=[...copy[2]];else if(copy[2] instanceof Uint32Array)copy[2]=copy[2].slice();groups.set(args[0] as number,copy);};
     if(property==="dispatchWorkgroups"||property==="dispatchWorkgroupsIndirect")return(...args:unknown[])=>{
      if(!pipeline)throw new Error("Dispatch without pipeline");
      const index=2*this.labels.length;if(index+1>=this.capacity)throw new Error("Kernel timestamp capacity exceeded");
      this.labels.push(this.pipelines.get(pipeline)??pipeline.label??d?.label??"unlabelled");
      const pass=target.beginComputePass({...d,timestampWrites:{querySet:this.queries,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
      pass.setPipeline(pipeline);for(const g of groups.values())Reflect.apply(pass.setBindGroup,pass,g);
      Reflect.apply(pass[property],pass,args);pass.end();
     };
     if(property==="end")return()=>{if(original?.endOfPassWriteIndex!==undefined)target.beginComputePass({timestampWrites:{querySet:original.querySet,endOfPassWriteIndex:original.endOfPassWriteIndex}}).end();};
     if(property==="pushDebugGroup"||property==="popDebugGroup"||property==="insertDebugMarker")return()=>{};
     if(property==="label")return d?.label;
     throw new Error(`Unsupported diagnostic pass method ${String(property)}`);
    }});
   };
   const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
 }
 async finish(){
  this.active=false;const count=2*this.labels.length;
  if(count===0)throw new Error("No diagnostic dispatches recorded");
  const output=this.raw.createBuffer({size:count*8,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
  const read=this.raw.createBuffer({size:count*8,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   const e=this.raw.createCommandEncoder();e.resolveQuerySet(this.queries,0,count,output,0);e.copyBufferToBuffer(output,0,read,0,count*8);this.raw.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
   const times=new BigUint64Array(read.getMappedRange());
   if(times.every(t=>t===0n))throw new Error("Kernel timestamps were not written");
   const sums=new Map<string,{label:string;calls:number;ms:number}>();
   this.labels.forEach((label,i)=>{const row=sums.get(label)??{label,calls:0,ms:0};row.calls++;row.ms+=Number(times[2*i+1]!-times[2*i]!)/1e6;sums.set(label,row);});read.unmap();
   return {scope:"Diagnostic isolated dispatches; not comparable to normal frame time",dispatches:this.labels.length,kernels:[...sums.values()].sort((a,b)=>b.ms-a.ms)};
  }finally{read.destroy();output.destroy();}
 }
 destroy():void{this.queries.destroy();}
}
