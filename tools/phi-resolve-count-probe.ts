/** Diagnostic command-stream captures of actual phi-resolve counts.
 * Wrap the device before constructing solvers, set enabled for selected frames,
 * then read() after those frames complete. Copies occur immediately after each
 * resolve pass, where the pass's read-only counts still describe its dispatch.
 */
import assert from 'node:assert/strict';
export class PhiResolveCountProbe {
 enabled=false;
 readonly device:GPUDevice;
 private readonly groups=new WeakMap<GPUBindGroup,GPUBindGroupDescriptor>();
 private readonly modules=new WeakMap<GPUShaderModule,string>();
 private readonly pipelines=new WeakMap<GPUComputePipeline,{entry:string;tiles:number}>();
 private readonly captures:{buffer:GPUBuffer;entry:string;tiles:number;dispatch:number[];listed:boolean}[]=[];
 constructor(private readonly raw:GPUDevice){
  const self=this;
  this.device=new Proxy(raw,{get(target,key){
   if(key==='createBindGroup')return(d:GPUBindGroupDescriptor)=>{const g=target.createBindGroup(d);self.groups.set(g,d);return g;};
   if(key==='createShaderModule')return(d:GPUShaderModuleDescriptor)=>{const m=target.createShaderModule(d);self.modules.set(m,d.code);return m;};
   if(key==='createComputePipelineAsync')return async(d:GPUComputePipelineDescriptor)=>{const p=await target.createComputePipelineAsync(d);self.remember(p,d);return p;};
   if(key==='createComputePipeline')return(d:GPUComputePipelineDescriptor)=>{const p=target.createComputePipeline(d);self.remember(p,d);return p;};
   if(key==='createCommandEncoder')return(d?:GPUCommandEncoderDescriptor)=>self.encoder(target.createCommandEncoder(d));
   const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
  }});
 }
 private remember(p:GPUComputePipeline,d:GPUComputePipelineDescriptor){
  const entry=d.compute.entryPoint??'';if(entry!=='resolve'&&entry!=='resolveListed')return;
  const source=this.modules.get(d.compute.module);if(!source?.includes('umResolveLattice'))return;
  const match=source.match(/fn UM_TILES\(\)->u32\{return (\d+)u;\}/)??source.match(/const UM_TILES:u32=(\d+)u;/);assert.ok(match,'phi resolve tile count');
  this.pipelines.set(p,{entry,tiles:Number(match[1])});
 }
 private encoder(encoder:GPUCommandEncoder):GPUCommandEncoder{
  const self=this;return new Proxy(encoder,{get(target,key){
   if(key==='beginComputePass')return(d?:GPUComputePassDescriptor)=>{
    const pass=target.beginComputePass(d);if(!self.enabled||d?.label!=='Uniform mixed phi resolve')return pass;
    const bound=new Map<number,GPUBindGroup>();let pipeline:GPUComputePipeline|undefined;const dispatch:number[]=[];
    return new Proxy(pass,{get(pt,pk){
     if(pk==='setBindGroup')return(index:number,g:GPUBindGroup|null,...args:unknown[])=>{if(g)bound.set(index,g);(pt.setBindGroup as Function)(index,g,...args);};
     if(pk==='setPipeline')return(p:GPUComputePipeline)=>{pipeline=p;pt.setPipeline(p);};
     if(pk==='dispatchWorkgroups')return(x:number,y=1,z=1)=>{dispatch.push(x,y,z);pt.dispatchWorkgroups(x,y,z);};
     if(pk==='end')return()=>{
      pt.end();const meta=pipeline&&self.pipelines.get(pipeline);assert.ok(meta,'captured phi resolve pipeline');
      const resource=(group:number,binding:number)=>{const g=bound.get(group);assert.ok(g);const e=self.groups.get(g)?.entries;assert.ok(e);const r=Array.from(e).find(e=>e.binding===binding)?.resource as GPUBufferBinding;assert.ok(r?.buffer);return r;};
      const counts=resource(0,1),support=resource(0,2),read=self.raw.createBuffer({size:96,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
      target.copyBufferToBuffer(counts.buffer,counts.offset??0,read,0,48);
      target.copyBufferToBuffer(support.buffer,(support.offset??0)+4*(7*meta.tiles+16),read,48,16);
      const listed=meta.entry==='resolveListed';if(listed){const c=resource(2,0);target.copyBufferToBuffer(c.buffer,c.offset??0,read,64,32);}
      self.captures.push({buffer:read,...meta,dispatch,listed});
     };
     const v=Reflect.get(pt,pk,pt);return typeof v==='function'?v.bind(pt):v;
    }});
   };
   const v=Reflect.get(target,key,target);return typeof v==='function'?v.bind(target):v;
  }});
 }
 async read(){
  const out=[];for(const c of this.captures.splice(0)){
   await c.buffer.mapAsync(GPUMapMode.READ);const w=new Uint32Array(c.buffer.getMappedRange());
   const fine=w[0]!,coarse=w[1]!,seamFine=w[12]!,seamCoarse=w[13]!,changed=c.listed?w[16]:undefined,dilated=c.listed?w[21]:undefined;
   out.push({entry:c.entry,tiles:c.tiles,dispatch:c.dispatch,fine,coarse,seamFine,seamCoarse,changed,dilated,jobs:c.listed?dilated:seamFine+seamCoarse});
   c.buffer.unmap();c.buffer.destroy();
  }return out;
 }
}
