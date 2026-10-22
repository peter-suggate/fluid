/** Diagnostic only: split selected multi-dispatch passes for per-dispatch
 * timestamps. Reapply the original pipeline and bindings at every boundary.
 * Pass splitting adds overhead; use this to locate work, not to report speed. */
export class NarrowBandDispatchProbe {
 readonly device:GPUDevice;
 enabled=false;
 private readonly names=new WeakMap<GPUComputePipeline,string>();
 constructor(raw:GPUDevice){
  this.device=new Proxy(raw,{get:(target,key)=>{
   if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
    const pipeline=await target.createComputePipelineAsync(d);
    const c=d.compute.constants;
    const tags=["umRegularFine","umMergedTiles","umFastSurface","umSolidFree","umSolidTwinned","umCountedJobs"].filter(key=>c?.[key]!==undefined).map(key=>`${key}=${c![key]}`).join(" ");
    this.names.set(pipeline,`${d.compute.entryPoint??"main"}${c?.umCellWidth?` width=${c.umCellWidth}`:""}${c?.ueSweepKind?` sweep=${c.ueSweepKind}`:""}${tags?` ${tags}`:""}`);
    return pipeline;
   };
   if(key==="createCommandEncoder")return (d?:GPUCommandEncoderDescriptor)=>{
    const encoder=target.createCommandEncoder(d);
    return new Proxy(encoder,{get:(e,k)=>{
     if(k==="beginComputePass")return (descriptor?:GPUComputePassDescriptor)=>{
      if(!this.enabled||process.env.NB_SPLIT_ALL!=="1"&&!['Uniform mixed extension','Uniform pressure arena solve','Narrow-band FLIP resample and seed','Narrow-band FLIP surface reconstruction','Narrow-band FLIP crossing search','Narrow-band FLIP surface arena search','Narrow-band FLIP surface arena metric','Narrow-band FLIP surface arena store','Narrow-band FLIP coarse retirement','Narrow-band FLIP transfer','Narrow-band FLIP snapshot','Narrow-band FLIP update','Uniform dynamic ownership census','Uniform shared frame plan','Uniform local speed certificate','Uniform mixed layout build','Uniform mixed remap','Uniform pressure arena marks and rows','Uniform mixed surface advect','Narrow-band FLIP advectParticles','Uniform mixed momentum','Uniform mixed extension slot snapshot','Uniform mixed geometric fill','Uniform mixed pressure V-cycle','Uniform mixed pressure Full-cycle'].includes(descriptor?.label??""))return e.beginComputePass(descriptor);
      let pipeline:GPUComputePipeline|undefined;
      const bindings=new Map<number,unknown[]>();
      const dispatch=(kind:"dispatchWorkgroups"|"dispatchWorkgroupsIndirect",args:unknown[])=>{
       if(!pipeline)throw new Error("Dispatch probe requires a pipeline");
       const grid=process.env.NB_SPLIT_GRID==="1"&&kind==="dispatchWorkgroups"?` [${args.map(a=>a??1).join("x")}]`:"";
       const pass=e.beginComputePass({...descriptor,label:`${descriptor!.label} / ${this.names.get(pipeline)??pipeline.label}${grid}`});
       pass.setPipeline(pipeline);
       for(const [index,values] of bindings)Reflect.apply(pass.setBindGroup,pass,[index,...values]);
       Reflect.apply(pass[kind],pass,args);pass.end();
      };
      return {
       setPipeline:(p:GPUComputePipeline)=>{pipeline=p;},
       setBindGroup:(index:number,...values:unknown[])=>{bindings.set(index,values);},
       dispatchWorkgroups:(...args:unknown[])=>dispatch("dispatchWorkgroups",args),
       dispatchWorkgroupsIndirect:(...args:unknown[])=>dispatch("dispatchWorkgroupsIndirect",args),
       end:()=>{},pushDebugGroup:()=>{},popDebugGroup:()=>{},insertDebugMarker:()=>{},
      } as unknown as GPUComputePassEncoder;
     };
     const v=Reflect.get(e,k,e);return typeof v==="function"?v.bind(e):v;
    }});
   };
   const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;
  }});
 }
}
