import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { sampledSurfaceTexturesWGSL } from "../lib/svo/features/materials/sampled-surface-textures.wgsl";

test("voxel image attributes filter in linear space, repeat at seams, and use explicit ray LOD", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  await withUniformDevice("voxel image attributes",async device=>{
    const words=new Uint32Array(4096),f=new Float32Array(words.buffer);
    f[3]=1;words.set([8,8,8],4);words.set([1,1,1],8);words[16]=18;
    const map=18+729*2,node=map+32,material=node+729*2;
    words[12]=map;words[13]=node;words[14]=material;
    // U=x, V=z; unperturbed +Y; tangent +X, bitangent -Z.
    f.set([1,0,0,0,0,0,1,0,1,0,0,1],map+16);words[map+28]=0;
    for(let i=0;i<729;i++){words[node+2*i]=1;words[node+2*i+1]=0xffff8000;}
    f.set([1,1,1,.8],material);words[material+4]=0;words[material+5]=1;f[material+6]=1;f[material+7]=0;
    words.set([2,2,2],material+8);
    const data=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(data,0,words);
    const texture=device.createTexture({size:[2,2,2],mipLevelCount:2,format:"rgba8unorm-srgb",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    // Black/white checker, with a linear-space average encoded into its mip.
    device.queue.writeTexture({texture},new Uint8Array([0,0,0,128,255,255,255,128,255,255,255,128,0,0,0,128]),{bytesPerRow:8},[2,2,1]);
    device.queue.writeTexture({texture,mipLevel:1},new Uint8Array([188,188,188,128]),{bytesPerRow:4},[1,1,1]);
    for(let mip=0;mip<2;mip++)device.queue.writeTexture({texture,mipLevel:mip,origin:[0,0,1]},new Uint8Array(Array.from({length:mip?1:4},()=>[188,188,255,255]).flat()),{bytesPerRow:mip?4:8},[mip?1:2,mip?1:2,1]);
    const output=device.createBuffer({size:6*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var surfaceTextureImages:texture_2d_array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
fn drySceneWord(at:u32)->u32{return data[at];}
${sampledSurfaceTexturesWGSL}
@compute @workgroup_size(1) fn main(){
  let points=array<vec4f,6>(vec4f(.25,2,.25,.01),vec4f(.5,2,.25,.01),vec4f(1.25,2,.25,.01),vec4f(0,2,.25,.01),vec4f(.25,2,.25,2),vec4f(-1,2,.25,.01));
  for(var i=0u;i<6u;i++){let s=sampledImageSurface(0u,points[i].xyz,points[i].w);output[i*2u]=vec4f(s.base,s.roughness);output[i*2u+1u]=vec4f(s.normal,f32(s.valid));}
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data}},{binding:1,resource:texture.createView({dimension:"2d-array"})},{binding:2,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);const actual=new Float32Array(read.getMappedRange());
    for(const [i,value] of [0,.5,0,.5,.503,0].entries())assert.ok(Math.abs(actual[i*8]!-value)<.005,`case${i}: ${actual[i*8]} vs ${value}`);
    for(let i=0;i<5;i++){assert.ok(Math.abs(actual[i*8+3]!-.8*128/255)<.001);assert.ok(actual[i*8+5]!>.999);assert.equal(actual[i*8+7],1);}
    assert.equal(actual[47],0,"outside the FVX remains unmapped");
    read.unmap();for(const buffer of[data,output,read])buffer.destroy();texture.destroy();
  });
});

test("GPU source-material blending is continuous without blending UV charts", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  await withUniformDevice("voxel image material transition",async device=>{
    const words=new Uint32Array(4096),f=new Float32Array(words.buffer);
    f[3]=1;words.set([8,8,8],4);words.set([1,1,1],8);words[16]=18;
    const map=18+729*2,node=map+64,material=node+729*2;
    words[12]=map;words[13]=node;words[14]=material;
    // Two charts of material0 deliberately address different texels. The
    // strongest chart wins within that material; only material values blend.
    for(const [id,u,mat] of [[1,.25,0],[2,.25,1],[3,.75,0]]){
      f.set([0,0,0,u!,0,0,0,.25,1,0,0,1],map+id!*16);words[map+id!*16+12]=mat!;
    }
    const oct=(x:number,y:number,z:number)=>{const sum=Math.abs(x)+Math.abs(y)+Math.abs(z);return ((Math.round((x/sum*.5+.5)*65535)&65535)|((Math.round((y/sum*.5+.5)*65535)&65535)<<16))>>>0;};
    for(let z=0;z<9;z++)for(let y=0;y<9;y++)for(let x=0;x<9;x++){
      const at=node+2*(x+9*(y+9*z)),right=(x&1)!==0;
      words[at]=right?2:(y&1)?3:1;words[at+1]=right?oct(.6,.8,0):oct(0,1,0);
    }
    for(let m=0;m<2;m++){
      const at=material+m*12;f.set([1,1,1,m?.8:.2],at);words[at+4]=m;words[at+5]=0xffffffff;
      f[at+6]=1;f[at+7]=m;words.set([2,2,1],at+8);
    }
    const locations=[2,2.25,2.4999,2.5,2.5001,2.75,2.9999,3,3.0001];
    const data=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(data,0,words);
    const texture=device.createTexture({size:[2,2,2],format:"rgba8unorm-srgb",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    device.queue.writeTexture({texture},new Uint8Array([255,0,0,255,0,255,0,255,255,0,0,255,0,255,0,255]),{bytesPerRow:8},[2,2,1]);
    device.queue.writeTexture({texture,origin:[0,0,1]},new Uint8Array(Array.from({length:4},()=>[0,0,255,255]).flat()),{bytesPerRow:8},[2,2,1]);
    const output=device.createBuffer({size:locations.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    try{
      const module=device.createShaderModule({code:`
@group(0) @binding(0) var<storage,read> data:array<u32>;
@group(0) @binding(1) var surfaceTextureImages:texture_2d_array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
fn drySceneWord(at:u32)->u32{return data[at];}
${sampledSurfaceTexturesWGSL}
@compute @workgroup_size(1) fn main(){
  let locations=array<f32,${locations.length}>(${locations.join(",")});
  for(var i=0u;i<${locations.length}u;i++){let s=sampledImageSurface(0u,vec3f(locations[i],2.25,2.25),.01);output[i*2u]=vec4f(s.base,s.roughness);output[i*2u+1u]=vec4f(s.normal,s.metallic);}
}`});
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
      const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
      const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:data}},{binding:1,resource:texture.createView({dimension:"2d-array"})},{binding:2,resource:{buffer:output}}]});
      const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ);const actual=new Float32Array(read.getMappedRange());
      locations.forEach((x,i)=>{
        const t=x<=3?x-2:4-x,expected=[1-t,0,t,.2+.6*t],n=[.6*t,1-.2*t,0],length=Math.hypot(...n);
        expected.forEach((v,c)=>assert.ok(Math.abs(actual[i*8+c]!-v)<2e-5,`point${x}, attribute${c}`));
        n.forEach((v,c)=>assert.ok(Math.abs(actual[i*8+4+c]!-v/length)<5e-5,`point${x}, normal${c}`));
        assert.ok(Math.abs(actual[i*8+7]!-t)<2e-5);
      });
      assert.ok(Math.abs(actual[2*8]!-actual[4*8]!)<.00021,"no material jump across the cell midpoint");
      assert.ok(Math.abs(actual[6*8]!-actual[8*8]!)<2e-6,"matching attributes across adjacent cell faces");
    }finally{if(read.mapState==="mapped")read.unmap();data.destroy();output.destroy();read.destroy();texture.destroy();}
  });
});
