/** Execute the same deferred closure in compute; experimental host adapter only. */
export function computeShadingExperiment(raw:GPUDevice):GPUDevice {
 const [groupX,groupY]=(process.env.FLUID_SHADING_WORKGROUP??'8x8').split('x').map(Number);
 if(!Number.isSafeInteger(groupX)||!Number.isSafeInteger(groupY)||groupX<1||groupY<1||groupX*groupY>256)throw new Error('Invalid compute shading workgroup');
 const layouts=new Map<GPUBindGroupLayout,GPUBindGroupLayoutDescriptor>();
 const pipelineLayouts=new Map<GPUPipelineLayout,GPUPipelineLayoutDescriptor>();
 const groups=new Map<GPUBindGroup,GPUBindGroupDescriptor>();
 const modules=new Map<GPUShaderModule,GPUShaderModuleDescriptor>();
 const pipelines=new Map<GPURenderPipeline,{pipeline:GPUComputePipeline;layout:GPUBindGroupLayout}>();
 return new Proxy(raw,{get(t,k){
  if(k==='createShaderModule')return (d:GPUShaderModuleDescriptor)=>{const m=t.createShaderModule(d);modules.set(m,d);return m;};
  if(k==='createBindGroupLayout')return (d:GPUBindGroupLayoutDescriptor)=>{const descriptor={...d,entries:Array.from(d.entries,e=>({...e,visibility:e.visibility|GPUShaderStage.COMPUTE}))};const l=t.createBindGroupLayout(descriptor);layouts.set(l,descriptor);return l;};
  if(k==='createPipelineLayout')return (d:GPUPipelineLayoutDescriptor)=>{const l=t.createPipelineLayout(d);pipelineLayouts.set(l,d);return l;};
  if(k==='createBindGroup')return (d:GPUBindGroupDescriptor)=>{const g=t.createBindGroup(d);groups.set(g,d);return g;};
  if(k==='createRenderPipelineAsync')return async(d:GPURenderPipelineDescriptor)=>{
   const original=await t.createRenderPipelineAsync(d);
   if(d.fragment?.entryPoint!=='dryLightingMain'||d.label?.endsWith('x1'))return original;
   const descriptor=modules.get(d.fragment.module)!;
   let code=descriptor.code;
   const start=code.indexOf('@fragment fn dryLightingMain'),end=code.indexOf('@fragment fn drySkyLightingMain',start);
   let entry=code.slice(start,end).replace('@fragment fn dryLightingMain','fn dryComputeLighting').replace('->@location(0) vec4f','->vec4f').replaceAll('discard;','return vec4f(0.0);');
   code+=`\n${entry}\n@group(2) @binding(31) var computeShadingOutput:texture_storage_2d<rgba16float,write>;
@compute @workgroup_size(${groupX},${groupY}) fn dryComputeMain(@builtin(global_invocation_id) id:vec3u){
 let dims=textureDimensions(computeShadingOutput);if(any(id.xy>=dims)){return;}
 let geometry=drySplitGeometryAt(vec2i(id.xy));if(!(geometry.w>0.0&&geometry.w<DRY_MISS)){return;}
 let pixel=vec2f(id.xy)+vec2f(.5);let uv=vec2f(pixel.x/f32(dims.x),1.0-pixel.y/f32(dims.y));
 textureStore(computeShadingOutput,vec2i(id.xy),dryComputeLighting(VertexOut(vec4f(pixel,0.0,1.0),uv)));
}`;
   const module=t.createShaderModule({...descriptor,code,label:'Compute deferred experiment'});
   const base=Array.from(pipelineLayouts.get(d.layout as GPUPipelineLayout)!.bindGroupLayouts);
   const layout=t.createBindGroupLayout({entries:[...layouts.get(base[2]!)!.entries,{binding:31,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba16float'}}]});
   base[2]=layout;
   const pipeline=await t.createComputePipelineAsync({label:'Compute deferred experiment',layout:t.createPipelineLayout({bindGroupLayouts:base}),compute:{module,entryPoint:'dryComputeMain'}});
   pipelines.set(original,{pipeline,layout});return original;
  };
  if(k==='createCommandEncoder')return (d?:GPUCommandEncoderDescriptor)=>{
   const encoder=t.createCommandEncoder(d);
   return new Proxy(encoder,{get(e,key){
    if(key==='beginRenderPass')return (desc:GPURenderPassDescriptor)=>{
     if(desc.label!=='Sparse voxel deferred dry lighting')return e.beginRenderPass(desc);
     const attachments=Array.from(desc.colorAttachments);const view=attachments[0]!.view;
     const bound=new Map<number,GPUBindGroup>();let selected:GPURenderPipeline|undefined;
     let computePass:GPUComputePassEncoder|undefined,renderPass:GPURenderPassEncoder|undefined;
     return {
      setBindGroup(index:number,group:GPUBindGroup){bound.set(index,group);},
      setPipeline(pipeline:GPURenderPipeline){selected=pipeline;},
      draw(...args:Parameters<GPURenderPassEncoder['draw']>){
       if(!selected)throw new Error('No deferred shading pipeline');const active=pipelines.get(selected);
       if(active){
        if(renderPass)throw new Error('Mixed render and compute shading draws');
        computePass??=e.beginComputePass({label:desc.label,timestampWrites:desc.timestampWrites});
        computePass.setPipeline(active.pipeline);
        for(const [index,group] of bound){if(index===2){const descriptor=groups.get(group)!;computePass.setBindGroup(index,t.createBindGroup({...descriptor,layout:active.layout,entries:[...descriptor.entries,{binding:31,resource:view}]}));}else computePass.setBindGroup(index,group);}
        computePass.dispatchWorkgroups(Math.ceil(1920/groupX),Math.ceil(1080/groupY));
       }else{
        if(computePass)throw new Error('Mixed render and compute shading draws');
        renderPass??=e.beginRenderPass(desc);renderPass.setPipeline(selected);for(const [index,group] of bound)renderPass.setBindGroup(index,group);renderPass.draw(...args);
       }
      },
      end(){if(computePass)computePass.end();else if(renderPass)renderPass.end();else e.beginRenderPass(desc).end();},
     };
    };
    const v=Reflect.get(e,key,e);return typeof v==='function'?v.bind(e):v;
   }});
  };
  const v=Reflect.get(t,k,t);return typeof v==='function'?v.bind(t):v;
 }});
}
