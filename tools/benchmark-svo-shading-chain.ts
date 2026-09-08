/** Isolate real production passes only after the depth-3 probe proves mesh publication. */
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {DEFAULT_SVO_RENDER_TUNING} from '../lib/svo/pipeline/svo-render-tuning';
import type {SparseVoxelDrySceneRenderer} from '../lib/svo/pipeline/webgpu-svo-dry-scene';
import {packSvoDryViewUniforms} from './svo-dry-frame-harness';

export async function benchmarkShadingChain(device:GPUDevice,renderer:SparseVoxelDrySceneRenderer,uniforms:GPUBuffer,view:Parameters<typeof packSvoDryViewUniforms>[0]) {
 const width=1920,height=1080,cycles=Number(process.env.FLUID_SHADING_CYCLES??16);
 assert.ok(Number.isSafeInteger(cycles)&&cycles>=3&&cycles<=256);
 if(process.env.FLUID_SHADING_CAMERA==='orbit')view={...view,camera:{...view.camera,azimuth_rad:view.camera.azimuth_rad+.25,elevation_rad:view.camera.elevation_rad+.1},cameraMoving:true};
 const directory=process.env.FLUID_SHADING_OUT??'/tmp/svo-shading/depth3';mkdirSync(directory,{recursive:true});
 const target=device.createTexture({size:[width,height],format:'rgba16float',usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});
 renderer.ensureSize(width,height);device.queue.writeBuffer(uniforms,0,packSvoDryViewUniforms({...view,width,height}));
 const query=device.createQuerySet({type:'timestamp',count:256});
 const resolve=device.createBuffer({size:4096,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});
 const read=device.createBuffer({size:4096,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 const shade=(label:string)=>/cone|directional-light cache|deferred|radiance reconstruction/i.test(label);
 const scopes:Record<string,(label:string)=>boolean>={frame:()=>true,shading:shade,raster:l=>!shade(l),mesh:l=>l==='Voxel surface mesh rasterization',deferred:l=>l==='Sparse voxel deferred dry lighting',cones:l=>/cone|directional-light cache/i.test(l),sky:l=>l==='Sparse voxel deferred sky lighting'};
 const results:unknown[]=[];
 async function run(scope:string){
  const selected=device.createCommandEncoder(),discarded=device.createCommandEncoder();let count=0;const labels:string[]=[];
  const routed=new Proxy(discarded,{get(t,k){
   if(k==='beginRenderPass'||k==='beginComputePass')return (d:GPURenderPassDescriptor & GPUComputePassDescriptor)=>{
    if(!scopes[scope](d.label??''))return k==='beginRenderPass'?t.beginRenderPass(d):t.beginComputePass(d);
    labels.push(d.label??'');const descriptor={...d,timestampWrites:{querySet:query,beginningOfPassWriteIndex:count*2,endOfPassWriteIndex:count*2+1}};count++;
    return k==='beginRenderPass'?selected.beginRenderPass(descriptor):selected.beginComputePass(descriptor);
   };
   // Clears/copies feed the selected passes too. All resources already hold a complete frame.
   if(k==='clearBuffer'||k==='copyBufferToBuffer'||k==='copyTextureToTexture')return (...args:unknown[])=>Reflect.apply(Reflect.get(selected,k,selected),selected,args);
   const v=Reflect.get(t,k,t);return typeof v==='function'?v.bind(t):v;
  }});
  assert.ok(renderer.encode(routed,target));discarded.finish();assert.ok(count>0&&count<=128);
  selected.resolveQuerySet(query,0,count*2,resolve,0);selected.copyBufferToBuffer(resolve,0,read,0,count*16);
  device.queue.submit([selected.finish()]);await read.mapAsync(GPUMapMode.READ);
  const ticks=new BigUint64Array(read.getMappedRange());let start=ticks[0],end=ticks[1];
  for(let i=0;i<count;i++){if(ticks[i*2]<start)start=ticks[i*2];if(ticks[i*2+1]>end)end=ticks[i*2+1];}
  const ms=Number(end-start)/1e6;read.unmap();return {ms,labels};
 }
 const arms=[
  {name:'baseline'},
  {name:'no-cache',cache:false},
  {name:'no-ao',ao:false},
  {name:'no-shadows',shadows:false},
  {name:'no-visibility',ao:false,shadows:false},
  {name:'quarter',scale:0.25},
  {name:'bilateral',reconstruction:'joint-bilateral'},
  {name:'baseline-repeat'},
 ] as const;
 const selectedArms=arms.filter(a=>!process.env.FLUID_SHADING_ARMS||process.env.FLUID_SHADING_ARMS.split(',').includes(a.name));
 assert.ok(selectedArms.length>0,'No requested shading arms matched');
 try {
  for(const arm of selectedArms){
   const a=arm as {name:string;cache?:boolean;ao?:boolean;shadows?:boolean;scale?:0.25|0.5;reconstruction?:'joint-bilateral'};
   renderer.setRenderTuning({...DEFAULT_SVO_RENDER_TUNING,coneRadianceReconstruction:a.reconstruction??'full-res-relight'});
   renderer.setLightingOptions({coneLightingScale:a.scale??0.5,globalIlluminationEnabled:false,coneTracingMode:'cones',shadowsEnabled:a.shadows??true,ambientOcclusionEnabled:a.ao??true});
   renderer.setVoxelLightCacheEnabled(a.cache??true);await renderer.ensureConeLightingPrepass();
   for(let n=0;n<8;n++)await run('frame');
   const timings:Record<string,unknown>={};
   for(const scope of Object.keys(scopes)){
    const samples:number[]=[];let labels:string[]=[];
    for(let n=-3;n<cycles;n++){const r=await run(scope);if(n>=0)samples.push(r.ms);labels=r.labels;}
    const sorted=[...samples].sort((a,b)=>a-b);timings[scope]={median_ms:sorted[Math.floor(sorted.length/2)],p95_ms:sorted[Math.ceil(sorted.length*.95)-1],samples,labels};
   }
   await run('frame');
   const pitch=Math.ceil(width*8/256)*256;const pixels=device.createBuffer({size:pitch*height,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   const e=device.createCommandEncoder();e.copyTextureToBuffer({texture:target},{buffer:pixels,bytesPerRow:pitch},[width,height]);device.queue.submit([e.finish()]);await pixels.mapAsync(GPUMapMode.READ);
   writeFileSync(join(directory,`${a.name}.rgba16f`),new Uint8Array(pixels.getMappedRange()));pixels.unmap();pixels.destroy();
   const result={arm:a,shader:process.env.FLUID_SHADING_SHADER??'baseline',compute:process.env.FLUID_SHADING_COMPUTE==='1',compiledCache:process.env.FLUID_SHADING_REFERENCE==='1',productionSpecialization:process.env.FLUID_SHADING_REFERENCE!=='1'&&process.env.FLUID_SHADING_DISABLE_CACHE!=='1',scene:view.scene.sceneId,camera:view.camera,width,height,timings};results.push(result);writeFileSync(join(directory,'results.json'),JSON.stringify(results,null,2));console.log(JSON.stringify({phase:'shading-study',...result}));
  }
 }finally{query.destroy();resolve.destroy();read.destroy();target.destroy();}
}
