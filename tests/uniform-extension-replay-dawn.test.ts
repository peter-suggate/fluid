import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readRgbaTexture3D } from "../lib/harness/webgpu-smoke-readbacks";
import { WebGPUUniformVelocityExtrapolator } from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("mixed extension reuses source selections exactly after velocities change", {timeout:120_000}, async()=>{
 await withUniformDevice("Mixed extension source replay",async device=>{
  const dims=[64,48,32] as const, root=dims.map(n=>n/4), count=root.reduce((a,b)=>a*b,1);
  const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
  const texture=(size:readonly number[],format:GPUTextureFormat)=>{
   const t=device.createTexture({size:[...size],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;
  };
  const buffer=(size:number)=>{const b=device.createBuffer({size,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});buffers.push(b);return b;};
  const params=buffer(208),active=buffer(64),scratch=buffer(4);
  const values=new Float32Array(52);values.set([...dims,1/30,.05,.1,.2,0]);device.queue.writeBuffer(params,0,values);
  const region=new Uint32Array(16);region.set(dims,10);region.set(root,13);device.queue.writeBuffer(active,0,region);
  const density=texture(dims,"r32float"),open=texture(dims,"rgba32float"),velocity=texture(dims,"rgba32float"),predicted=texture(dims,"rgba32float");
  const transport=texture(dims.map(n=>n+2),"rgba32float"),predictedTransport=texture(dims.map(n=>n+2),"rgba32float");
  const fixtureDevice=new Proxy(device,{get(target,key){
   if(key==="createTexture")return (d:GPUTextureDescriptor)=>target.createTexture({...d,usage:d.usage|GPUTextureUsage.COPY_DST});
   const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;
  }});
  const extension=new WebGPUUniformVelocityExtrapolator(fixtureDevice,dims,[.05,.1,.2],params,density,open,velocity,predicted,transport,predictedTransport,active,scratch,undefined,true,true,undefined,undefined,undefined,false,true);
  try{
   await extension.initialize();const hierarchy=extension.prepareMixedContinuation();
   const field=new Float32Array(count*4),origins=new Uint32Array(count*8);
   const write=()=>{device.queue.writeTexture({texture:hierarchy.input},field,{bytesPerRow:root[0]!*16,rowsPerImage:root[1]!},root);};
   const encode=async(replay:boolean,read=false)=>{
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();hierarchy.encode(pass,replay);pass.end();device.queue.submit([encoder.finish()]);
    if(read)return readRgbaTexture3D(device,hierarchy.output,root[0]!,root[1]!,root[2]!);
    await device.queue.onSubmittedWorkDone();
   };
   for(let sample=0;sample<7;sample++){
    for(let i=0;i<count;i++){
     const x=i%root[0]!,y=Math.floor(i/root[0]!)%root[1]!,z=Math.floor(i/(root[0]!*root[1]!));
     const mask=sample===0?0:sample===1?7:((i*1103515245+sample*31)>>>7)&7;
     for(let a=0;a<3;a++){
      field[4*i+a]=Math.sin(i*.173+a+sample);const point=[4*x+2,4*y+2,4*z+2];point[a]=Math.min(point[a]!+1,dims[a]!-1);
      const source=point[0]!+dims[0]*(point[1]!+dims[1]*point[2]!)+1;
      origins[4*i+a]=origins[4*(i+count)+a]=(mask&(1<<a))?source:0;
     }
     field[4*i+3]=mask;origins[4*i+3]=origins[4*(i+count)+3]=0;
    }
    device.queue.writeTexture({texture:hierarchy.inputOrigins},origins,{bytesPerRow:root[0]!*16,rowsPerImage:root[1]!},[root[0]!,root[1]!,2*root[2]!]);write();await encode(false);
    for(let i=0;i<count;i++)for(let a=0;a<3;a++)field[4*i+a]=Math.cos(i*.097-3*a+sample)*7;
    write();const actual=await encode(true,true);const expected=await encode(false,true);
    assert.ok(actual&&expected);assert.deepEqual(new Uint32Array(actual.buffer),new Uint32Array(expected.buffer),`source pattern ${sample}: every value and known mask`);
   }
  }finally{extension.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());}
 });
});
