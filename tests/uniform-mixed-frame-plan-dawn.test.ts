import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedFramePlan} from "../lib/methods/uniform/uniform-mixed-frame-plan";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("shared frame census follows live ownership and rebuilds fine/shell support from current fields",async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed frame plan");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const lattice={dimensions:[32,32,32] as const,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const region={id:"coarse",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,min_m:{x:12,y:12,z:12},max_m:{x:16,y:16,z:16}};
  const ownership=new UniformMixedOwnership(device,createUniformMixedLayout(lattice,[]));
  const volume=device.createTexture({size:[32,32,32],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const phi=device.createTexture({size:[33,33,33],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const velocity=device.createTexture({size:[32,32,32],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const negative=device.createBuffer({size:3*32*32*4,usage:GPUBufferUsage.STORAGE});
  const readback=device.createBuffer({size:ownership.support.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  // Copying support for this test is the only reason the production buffer
  // would need COPY_SRC. Use a compute copy so its allocation contract stays lean.
  const copier=device.createShaderModule({code:"@group(0) @binding(0) var<storage,read_write> a:array<u32>; @group(0) @binding(1) var<storage,read_write> b:array<u32>; @compute @workgroup_size(64) fn copy(@builtin(global_invocation_id) id:vec3u){if(id.x<arrayLength(&a)){b[id.x]=a[id.x];}}"});
  const copyPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:copier,entryPoint:"copy"}});
  const copy=device.createBuffer({size:ownership.support.size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const copyGroup=device.createBindGroup({layout:copyPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ownership.support}},{binding:1,resource:{buffer:copy}}]});
  const guarded=new Proxy(device,{get(target,key){if(key==="createTexture")return()=>{throw new Error("frame plan allocated a field");};if(key==="createBuffer")return(desc:GPUBufferDescriptor)=>{assert.ok(desc.size<=32,"only bounded parameter storage is allowed");return target.createBuffer(desc);};const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
  const plan=new UniformMixedFramePlan(guarded,ownership,volume,phi,velocity,negative);await plan.initialize();
  try{
   const release=ownership.acquireFrame();
   assert.throws(()=>ownership.update(createUniformMixedLayout(lattice,[])),/immutable/);
   assert.throws(()=>ownership.acquireFrame(),/active frame/);release();release();
   for(const scenario of ["fine-liquid","coarse-liquid","empty","surface","fast-moving"] as const){
    ownership.update(createUniformMixedLayout(lattice,scenario==="coarse-liquid"||scenario==="fast-moving"?[region]:[]));
    const v=new Float32Array(32**3),p=new Float32Array(33**3).fill(100);
    if(scenario.includes("liquid"))v[12+32*(12+32*12)]=.25;
    if(scenario==="surface")p[13+33*(13+33*13)]=-1;
    device.queue.writeTexture({texture:volume},v,{bytesPerRow:128,rowsPerImage:32},[32,32,32]);
    device.queue.writeTexture({texture:phi},p,{bytesPerRow:132,rowsPerImage:33},[33,33,33]);
    const velocities=new Float32Array(32**3*4);if(scenario==="fast-moving")for(let i=0;i<velocities.length;i+=4)velocities[i]=16;
    device.queue.writeTexture({texture:velocity},velocities,{bytesPerRow:512,rowsPerImage:32},[32,32,32]);
    const encoder=device.createCommandEncoder();plan.encode(encoder,undefined,1);const pass=encoder.beginComputePass();pass.setPipeline(copyPipeline);pass.setBindGroup(0,copyGroup);pass.dispatchWorkgroups(Math.ceil(copy.size/256));pass.end();encoder.copyBufferToBuffer(copy,0,readback,0,copy.size);device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);const words=new Uint32Array(readback.getMappedRange()).slice();readback.unmap();
    for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
     const distance=Math.max(Math.abs(x-3),Math.abs(y-3),Math.abs(z-3));
     const expected=scenario==="empty"||scenario==="fast-moving"?0:(distance<=2?1:0)|(distance<=3?2:0);
     assert.equal(words[3*512+x+8*(y+8*z)],expected,`${scenario}: ${x},${y},${z}`);
    }
    const count=ownership.layout.tiles.length,fast=words[4*count+1]!,general=words[4*count+2]!;
    assert.equal(fast+general,ownership.layout.fineTiles.length);
    assert.equal(words[4*count+3],1,"speed bound is published only after the census");
    assert.equal(new Float32Array(words.buffer)[4*count],scenario==="fast-moving"?16:0);
    const tiers=[ownership.layout.fineTiles,ownership.layout.transitionTiles,ownership.layout.coarseTiles];
    let seamOffset=7*count+20;
    tiers.forEach((tiles,tier)=>{
     const expected=[...tiles].filter(tile=>(ownership.layout.stencils[2*tile]!>>>27)!==(ownership.layout.stencils[2*tile+1]!>>>27));
     assert.equal(words[7*count+16+tier],expected.length);
     assert.deepEqual([...words.slice(seamOffset,seamOffset+expected.length)],expected,"frame census must preserve immutable pressure work");
     seamOffset+=expected.length;
    });
    const fastTiles=new Set(words.slice(5*count+16,5*count+16+fast));
    const generalTiles=new Set(words.slice(6*count+16,6*count+16+general));
    const nonfine=[...ownership.layout.tiles].flatMap((word,t)=>word&0x80000000?[]:[t]);
    const coord=(t:number)=>[t%8,Math.floor(t/8)%8,Math.floor(t/64)];
    const radius=scenario==="fast-moving"?7:2;
    for(const tile of ownership.layout.fineTiles){
     const a=coord(tile),distance=Math.min(Infinity,...nonfine.map(t=>Math.max(...coord(t).map((v,i)=>Math.abs(v-a[i]!)))));
     assert.equal(fastTiles.has(tile),distance>radius,`${scenario}: regular certificate for ${tile}`);
     assert.equal(generalTiles.has(tile),distance<=radius);
    }
   }
   assert.deepEqual(errors,[]);
  }finally{plan.destroy();ownership.destroy();volume.destroy();phi.destroy();velocity.destroy();negative.destroy();readback.destroy();copy.destroy();}
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
