import assert from "node:assert/strict";
import test from "node:test";
import { createWaterReflectionWGSL, waterReflectionWGSL } from "../lib/core/water-reflection.wgsl";
import { waterReflectionBoundsWGSL } from "../lib/core/water-reflection-bounds.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

test("GPU reflection depth bounds preserve full-resolution ray hits across empty tiles, silhouettes and odd dimensions", {skip:!process.env.WEBGPU_NODE_MODULE}, async()=>{
  await withUniformDevice("conservative reflection depth bounds",async device=>{
    const boundsLayout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float"}},
      {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{format:"rg32float",access:"write-only"}},
    ]});
    const boundsModule=device.createShaderModule({code:waterReflectionBoundsWGSL});
    assert.deepEqual((await boundsModule.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const boundPipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[boundsLayout]}),compute:{module:boundsModule,entryPoint:"main"}});
    const layout=device.createBindGroupLayout({entries:[
      {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},
      ...[1,2].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float" as const}})),
      {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},
      {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}},
    ]});
    const reference=waterReflectionWGSL.replace("override screenSpaceReflections:u32=1u;","").replaceAll("waterReflectionHit","referenceHit").replaceAll("waterReflectionSample","referenceSample");
    const module=device.createShaderModule({code:`
struct U {cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,viewport:vec4f}
@group(0) @binding(0) var<uniform> u:U;
@group(0) @binding(1) var depths:texture_2d<f32>;
@group(0) @binding(2) var depthBounds:texture_2d<f32>;
@group(0) @binding(3) var<storage,read> rays:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4f>;
fn cameraTanHalfFov()->f32{return u.viewport.z;}
fn project(p:vec3f)->vec2f{return vec2f(.5+.5*p.x/(p.z*u.viewport.z*u.viewport.w),.5-.5*p.y/(p.z*u.viewport.z));}
fn waterReflectionDepthDimensions()->vec2u{return textureDimensions(depths);}
fn waterReflectionDepthBounds(tile:vec2i)->vec2f{return textureLoad(depthBounds,tile,0).xy;}
fn waterReflectionDepth(uv:vec2f)->f32{let d=textureLoad(depths,clamp(vec2i(uv*vec2f(textureDimensions(depths))),vec2i(0),vec2i(textureDimensions(depths))-1),0).a;return select(65504.0,d,d>0.0);}
fn waterEnvironmentRadiance(d:vec3f)->vec3f{return vec3f(.1);}
fn waterReflectionColor(uv:vec2f)->vec3f{return vec3f(uv,0);}
${createWaterReflectionWGSL(true)}
${reference}
@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=512u){return;}
  let origin=rays[id.x*2u].xyz;let direction=rays[id.x*2u+1u].xyz;
  output[id.x*2u]=waterReflectionHit(origin,direction,.1);
  output[id.x*2u+1u]=referenceHit(origin,direction,.1);
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"main"}});
    for(const [width,height] of [[127,93],[128,96],[1201,801],[1280,800]] as const){
      const source=device.createTexture({size:[width,height],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      const bounds=device.createTexture({size:[Math.ceil(width/8),Math.ceil(height/8)],format:"rg32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING});
      const uniforms=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      const rays=device.createBuffer({size:512*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
      const output=device.createBuffer({size:512*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
      try{
        device.queue.writeBuffer(uniforms,0,new Float32Array([0,0,0,0,0,0,1,0,16,16,16,0,width,height,.8,width/height]));
        let seed=37;const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
        const pixels=new Float32Array(width*height*4),rayValues=new Float32Array(512*8);
        for(let y=0;y<height;y++)for(let x=0;x<width;x++){
          const r=[((x+.5)/width-.5)*1.6*width/height,(.5-(y+.5)/height)*1.6,1];
          const z=10/(.6*r[0]!+.8),depth=z*Math.hypot(...r);
          pixels[(y*width+x)*4+3]=y>height/3&&y<height/2?0:(x+y)%4===0&&depth>0?depth:65504;
        }
        for(let i=0;i<512;i++){
          const origin=[(random()-.5)*.6,(random()-.5)*.6,2+random()*(i<256?2:12)];
          let target=[(random()-.5)*30,(random()-.5)*20,10];
          if(i<256){
            for(;;){
              const x=5+Math.floor(random()*(width-10)),y=5+Math.floor(random()*(height-10));
              const depth=pixels[(y*width+x)*4+3]!;if(!(depth>0&&depth<50))continue;
              const r=[((x+.5)/width-.5)*1.6*width/height,(.5-(y+.5)/height)*1.6,1];
              target=r.map(v=>v*depth/Math.hypot(...r));break;
            }
          }
          // Long almost-axis-aligned walks exercise repeated f32 crossings at
          // full resolution, including exact corners and tiny secondary steps.
          if(i>=384){
            const axis=i%2,sign=i%4<2?1:-1;
            target=[origin[0]!,origin[1]!,origin[2]!];
            target[axis]=origin[axis]!+sign*30;
            target[1-axis]=origin[1-axis]!+(i%3===0?0:1e-7);
          }
          const d=target.map((v,k)=>v-origin[k]!);const length=Math.hypot(...d);
          rayValues.set([...origin,0,...d.map(v=>v/length),0],i*8);
        }
        const build=device.createBindGroup({layout:boundsLayout,entries:[{binding:0,resource:source.createView()},{binding:1,resource:bounds.createView()}]});
        const query=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:uniforms}},{binding:1,resource:source.createView()},{binding:2,resource:bounds.createView()},{binding:3,resource:{buffer:rays}},{binding:4,resource:{buffer:output}}]});
        device.queue.writeBuffer(rays,0,rayValues);
        for(const viewportScale of [1,2])for(const empty of [false,true]){
          device.queue.writeBuffer(uniforms,0,new Float32Array([0,0,0,0,0,0,1,0,16,16,16,0,width*viewportScale,height*viewportScale,.8,width/height]));
          device.queue.writeTexture({texture:source},empty?new Float32Array(pixels.length):pixels,{bytesPerRow:width*16},[width,height]);
          const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(boundPipeline);p.setBindGroup(0,build);p.dispatchWorkgroups(bounds.width,bounds.height);p.end();
          const q=e.beginComputePass();q.setPipeline(pipeline);q.setBindGroup(0,query);q.dispatchWorkgroups(16);q.end();device.queue.submit([e.finish()]);
          const actual=await readMixedBuffer(device,output);
          let hits=0;
          for(let i=0;i<512;i++){
            if(actual[i*8+7]!>0)hits++;
            for(let k=0;k<4;k++)assert.ok(Math.abs(actual[i*8+k]!-actual[i*8+4+k]!)<1e-5,`${width}x${height}, empty ${empty}, ray ${i}, component ${k}: ${actual[i*8+k]} vs ${actual[i*8+4+k]}`);
          }
          if(empty)assert.equal(hits,0);else assert.ok(hits>100,`oracle includes substantial actual geometry hits: ${hits}`);
        }
      }finally{source.destroy();bounds.destroy();uniforms.destroy();rays.destroy();output.destroy();}
    }
  });
});
