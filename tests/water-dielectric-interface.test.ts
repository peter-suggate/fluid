import assert from "node:assert/strict";
import test from "node:test";
import { dielectricInterfaceReflectance, waterDielectricInterfaceWGSL } from "../lib/core/water-dielectric-interface";
import { compositeShader } from "../lib/core/webgpu-water-pipeline";

test("dielectric exit reflectance is analytic, continuous at critical angle, and energy conserving",()=>{
  for(const eta of [1/1.5,1/1.333,1,1.333,1.5]){
    assert.ok(Math.abs(dielectricInterfaceReflectance(1,eta)-((eta-1)/(eta+1))**2)<1e-12);
    for(let i=0;i<=1000;i++){
      const r=dielectricInterfaceReflectance(i/1000,eta),t=1-r;
      assert.ok(r>=0&&r<=1);assert.equal(r+t,1);
      for(const front of [0,.02,.5,1])for(const attenuation of [0,.2,1]){
        const reflected=front+(1-front)*attenuation*r,transmitted=(1-front)*attenuation*t;
        assert.ok(reflected+transmitted<=1+1e-12);
      }
    }
    if(eta>1){const critical=Math.sqrt(1-1/eta**2);assert.equal(dielectricInterfaceReflectance(critical-1e-5,eta),1);let last=0;for(const offset of [1e-2,1e-4,1e-6,1e-8,1e-10]){const value=dielectricInterfaceReflectance(critical+offset,eta);assert.ok(value>last);last=value;}assert.ok(last>.999);}
  }
  assert.doesNotMatch(compositeShader,/if\(tir\)|waterEnvironmentRadiance\(outgoing\),\.88/);
  assert.equal((compositeShader.match(/exitReflectance=dielectricInterfaceReflectance/g)??[]).length,2,"both water intervals partition the exit reflection");
});

test("GPU exact Fresnel matches CPU at water exit, TIR, reciprocal entrance and index-matched boundaries",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  const {withUniformDevice}=await import("./helpers/uniform-geometric");
  const inputs:number[]=[];for(const eta of [1/1.333,1,1.333,1.5]){for(let i=0;i<=40;i++)inputs.push(i/40,eta);if(eta>1){const critical=Math.sqrt(1-1/eta**2);for(const offset of [-.001,-.00001,.00001,.001])inputs.push(critical+offset,eta);}}
  const data=new Float32Array(inputs),count=data.length/2;
  await withUniformDevice("dielectric exit partition",async device=>{
    const module=device.createShaderModule({code:`${waterDielectricInterfaceWGSL}
      @group(0) @binding(0) var<storage,read> inputs:array<vec2f>;
      @group(0) @binding(1) var<storage,read_write> results:array<vec4f>;
      @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let v=inputs[id.x];let r=dielectricInterfaceReflectance(v.x,v.y);let t=1.0-r;results[id.x]=vec4f(r,t,.02+.98*.6*(r+t),mix(.15,3.2,r));}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),output=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:count*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});device.queue.writeBuffer(input,0,data);
    try{const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]}));pass.dispatchWorkgroups(count);pass.end();encoder.copyBufferToBuffer(output,0,read,0,count*16);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange()).slice();read.unmap();for(let i=0;i<count;i++){const expected=dielectricInterfaceReflectance(data[i*2]!,data[i*2+1]!);assert.ok(Math.abs(result[i*4]!-expected)<.001,`Fresnel ${i}: ${result[i*4]} vs ${expected}`);assert.ok(Math.abs(result[i*4]!+result[i*4+1]!-1)<1e-6);assert.ok(Math.abs(result[i*4+2]!-.608)<1e-6);assert.ok(result[i*4+3]!>=.14999&&result[i*4+3]!<=3.20001);}}finally{input.destroy();output.destroy();read.destroy();}
  });
});
