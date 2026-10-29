import assert from "node:assert/strict";
import test from "node:test";
import { waterReflectionFilterWGSL } from "../lib/core/water-reflection-filter.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("water reflection footprint preserves zero/subpixel detail and continuous kernel variance", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("continuous water reflection footprint", async device => {
    const size=9, roughness=[0,.04,...[.25,.5,.9999,1,1.0001,1.5,2].map(radius=>Math.sqrt(radius/4.5))];
    const module=device.createShaderModule({code:`
const screenSpaceReflections=1u;
struct U{cameraPosition:vec4f} const u=U(vec4f(0,1,1,0));
var<private> fixtureRoughness:f32;
fn waterRoughness()->f32{return fixtureRoughness;}
fn cameraTanHalfFov()->f32{return .5;}
fn waterEnvironmentRadiance(direction:vec3f)->vec3f{return vec3f(0);}
${waterReflectionFilterWGSL}
@group(0) @binding(0) var colors:texture_2d_array<f32>;
@group(0) @binding(1) var positions:texture_2d<f32>;
@group(0) @binding(2) var normals:texture_2d<f32>;
@group(0) @binding(3) var<storage,read> roughness:array<f32>;
@group(0) @binding(4) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 fixtureRoughness=roughness[id.z];let uv=(vec2f(id.xy)+.5)/9.;
 output[(id.z*9u+id.y)*9u+id.x]=vec4f(filteredWaterReflection(colors,positions,normals,uv,vec3f(0,0,1),vec3f(0,1,0),vec3f(0,1,0),.1),1);
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const textures=Array.from({length:3},()=>device.createTexture({size:[size,size],format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST}));
    const colors=new Float32Array(size*size*4),positions=new Float32Array(colors.length),normals=new Float32Array(colors.length);
    for(let i=0;i<size*size;i++){colors.set([i===40?1:0,0,0,1],i*4);positions.set([0,0,1,1],i*4);normals.set([0,1,0,1],i*4);}
    [colors,positions,normals].forEach((data,i)=>device.queue.writeTexture({texture:textures[i]!},data,{bytesPerRow:size*16},[size,size]));
    const input=device.createBuffer({size:roughness.length*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,new Float32Array(roughness));
    const bytes=size*size*roughness.length*16;
    const output=device.createBuffer({size:bytes,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...textures.map((t,binding)=>({binding,resource:t.createView({dimension:binding===0?"2d-array":"2d"})})),{binding:3,resource:{buffer:input}},{binding:4,resource:{buffer:output}}]});
    const run=async()=>{const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(size,size,roughness.length);p.end();e.copyBufferToBuffer(output,0,read,0,bytes);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const r=new Float32Array(read.getMappedRange()).slice();read.unmap();return r;};
    try{
      const result=await run();
      for(let z=0;z<roughness.length;z++){
        const radius=roughness[z]!**2*4.5;let sum=0,mx=0,my=0;
        for(let y=0;y<size;y++)for(let x=0;x<size;x++){const value=result[(z*size*size+y*size+x)*4]!;sum+=value;mx+=value*(x-4)**2;my+=value*(y-4)**2;}
        assert.ok(Math.abs(sum-1)<1e-6,`constant energy for radius ${radius}`);
        assert.ok(Math.abs(mx-.5*radius**2)<2e-6,`x second moment for radius ${radius}: ${mx}`);
        assert.ok(Math.abs(my-.5*radius**2)<2e-6,`y second moment for radius ${radius}: ${my}`);
      }
      assert.equal(result[40*4],1,"analytic zero footprint preserves the original impulse exactly");
      const minRadius=.04**2*4.5;
      assert.ok(Math.abs(result[(81+40)*4]!-(1-.75*minRadius**2))<1e-7,"authored minimum roughness .04 preserves subpixel detail at finite hit distance");
      assert.ok(result[(81+40)*4]!>.9999,"minimum roughness no longer forces the old .25 impulse peak");
      for(let i=0;i<81;i++)colors.set([2,3,4,1],i*4);
      device.queue.writeTexture({texture:textures[0]!},colors,{bytesPerRow:size*16},[size,size]);
      const constant=await run();
      for(let i=0;i<constant.length;i+=4)for(let c=0;c<3;c++)assert.ok(Math.abs(constant[i+c]!-(c+2))<1e-6,"constant radiance remains unchanged");
    }finally{textures.forEach(t=>t.destroy());input.destroy();output.destroy();read.destroy();}
  });
});
