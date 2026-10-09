/** Diagnostic ablations of the actual fine P2G shader. Each variant borrows
 * identical inputs and writes only the normal transfer output. The caller
 * MUST run production transfer last before the solver reads that output.
 * Ablations are not valid simulation algorithms or additive cost accounting. */
export class NarrowBandTransferProbe {
 readonly device:GPUDevice;
 variant:string|undefined;
 private readonly sources=new WeakMap<GPUShaderModule,string>();
 private readonly variants=new WeakMap<GPUComputePipeline,Map<string,GPUComputePipeline>>();
 readonly names=["baseline","unprepared","no-gather","cheap-weights","no-reduction"];
 constructor(raw:GPUDevice){
  const replace=(code:string,from:string,to:string)=>{
   if(!code.includes(from))throw new Error("P2G probe source no longer matches: "+from);
   return code.replace(from,to);
  };
  const modify=(code:string,name:string)=>{
   if(name==="unprepared")return replace(code,"return 416u*umCounts.x<=NB_CELLS;","return false;");
   if(name==="no-gather")return replace(code,"let gather=any(blend>vec3f(0))||any(nextBlend>vec3f(0));","let gather=false;");
   if(name==="no-reduction")return replace(code,"for(var stride=8u;stride>0u;stride/=2u)","for(var stride=0u;stride>0u;stride/=2u)");
   if(name==="cheap-weights"){
    const start="let wc=vec3f(weight(d.x),weight(d.y),weight(d.z));";
    const end="totalNext+=wn;momentumNext+=wn*motion.xyz;";
    const a=code.indexOf(start),b=code.indexOf(end,a);if(a<0||b<0)throw new Error("P2G probe weight block missing");
    // Retain both position/motion reads, four vector accumulators, and the
    // reduction, but replace eight spline evaluations with a cheap weight.
    return code.slice(0,a)+`let w=vec3f(1.0)/(vec3f(1.0)+abs(d));
    total+=w;momentum+=w*motion.xyz;
    let wn=w*0.5;totalNext+=wn;momentumNext+=wn*motion.xyz;`+code.slice(b+end.length);
   }
   return code;
  };
  this.device=new Proxy(raw,{get:(target,key)=>{
   if(key==="createShaderModule")return (d:GPUShaderModuleDescriptor)=>{
    const module=target.createShaderModule(d);this.sources.set(module,d.code);return module;
   };
   if(key==="createComputePipelineAsync")return async(d:GPUComputePipelineDescriptor)=>{
    const pipeline=await target.createComputePipelineAsync(d);
    if(d.compute.entryPoint==="transfer"){
     const source=this.sources.get(d.compute.module);if(!source)throw new Error("P2G probe shader source missing");
     const variants=new Map<string,GPUComputePipeline>([["baseline",pipeline]]);
     for(const name of this.names.slice(1)){
      const module=target.createShaderModule({label:`P2G probe ${name}`,code:modify(source,name)});
      const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
      if(errors.length)throw new Error(errors.map(e=>e.message).join("\n"));
      variants.set(name,await target.createComputePipelineAsync({...d,label:`P2G probe ${name}`,compute:{...d.compute,module}}));
     }
     this.variants.set(pipeline,variants);
    }
    return pipeline;
   };
   if(key==="createCommandEncoder")return (d?:GPUCommandEncoderDescriptor)=>{
    const encoder=target.createCommandEncoder(d);
    return new Proxy(encoder,{get:(e,k)=>{
     if(k==="beginComputePass")return (descriptor?:GPUComputePassDescriptor)=>{
      const name=this.variant;
      const pass=e.beginComputePass(name?{...descriptor,label:`P2G probe ${name}`}:descriptor);
      return new Proxy(pass,{get:(p,property)=>{
       if(property==="setPipeline"&&name)return (pipeline:GPUComputePipeline)=>{
        const alternative=this.variants.get(pipeline)?.get(name);
        if(!alternative)throw new Error(`P2G probe pipeline missing for ${name}`);
        p.setPipeline(alternative);
       };
       const v=Reflect.get(p,property,p);return typeof v==="function"?v.bind(p):v;
      }});
     };
     const v=Reflect.get(e,k,e);return typeof v==="function"?v.bind(e):v;
    }});
   };
   const v=Reflect.get(target,key,target);return typeof v==="function"?v.bind(target):v;
  }});
 }
}
