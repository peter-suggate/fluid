import assert from "node:assert/strict";
import test from "node:test";
import { evaluateSvoEnvironmentDiffuseIrradiance, packSvoEnvironmentLightingRecords, svoEnvironmentLightingRecord, svoEnvironmentLightingWGSL } from "../lib/svo/features/radiance/svo-environment-lighting";
import { withUniformDevice } from "./helpers/uniform-geometric";

const normals = [[0,1,0],[1,0,0],[0,-1,0],[.6,.8,0]] as const;
const constant = svoEnvironmentLightingRecord("garden",1,{lowerRadianceLinear:[.2,.4,.8],upperRadianceLinear:[.2,.4,.8],accentRadianceLinear:[0,0,0],diffuseScale:1});
const gradient = svoEnvironmentLightingRecord("garden",1,{lowerRadianceLinear:[.1,.2,.3],upperRadianceLinear:[.7,.5,.9],accentRadianceLinear:[0,0,0],diffuseScale:1});

test("constant environment radiance integrates to pi radiance, giving albedo times radiance after Lambert division",()=>{
  for(const n of normals){
    const irradiance=evaluateSvoEnvironmentDiffuseIrradiance(constant,n);
    irradiance.forEach((v,c)=>{
      assert.ok(Math.abs(v-Math.PI*constant.lowerRadianceLinear[c]!)<1e-12);
      assert.ok(Math.abs(.5*v/Math.PI-.5*constant.lowerRadianceLinear[c]!)<1e-12);
    });
  }
});

test("vertical-gradient irradiance agrees with a solid-angle cosine quadrature",()=>{
  // Integrate the sky itself over the visible hemisphere. This is independent
  // of the analytic coefficients in the implementation and exposes missing pi.
  for(const normal of normals){
    const tangent=Math.abs(normal[1])>.9?[1,0,0]:[normal[1],-normal[0],0];
    const length=Math.hypot(...tangent);for(let i=0;i<3;i++)tangent[i]!/=length;
    const bitangent=[normal[1]*tangent[2]!-normal[2]*tangent[1]!,normal[2]*tangent[0]!-normal[0]*tangent[2]!,normal[0]*tangent[1]!-normal[1]*tangent[0]!];
    const sum=[0,0,0],rings=128,sectors=128;
    for(let r=0;r<rings;r++){
      const cosine=(r+.5)/rings,sine=Math.sqrt(1-cosine*cosine);
      for(let a=0;a<sectors;a++){
        const angle=(a+.5)*2*Math.PI/sectors;
        const worldY=normal[1]*cosine+sine*(tangent[1]!*Math.cos(angle)+bitangent[1]!*Math.sin(angle));
        for(let c=0;c<3;c++){
          const radiance=gradient.lowerRadianceLinear[c]!*(1-worldY)/2+gradient.upperRadianceLinear[c]!*(1+worldY)/2;
          sum[c]!+=radiance*cosine*2*Math.PI/(rings*sectors);
        }
      }
    }
    evaluateSvoEnvironmentDiffuseIrradiance(gradient,normal).forEach((v,c)=>assert.ok(Math.abs(v-sum[c]!)<2e-5));
  }
});

test("GPU diffuse environment uses the same irradiance units and Lambert energy as CPU",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
  await withUniformDevice("environment irradiance units",async device=>{
    const records=[constant,gradient],packed=new Uint32Array(48);
    records.forEach((record,index)=>packed.set(packSvoEnvironmentLightingRecords([record]),index*24));
    const input=device.createBuffer({size:packed.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,packed);
    const output=device.createBuffer({size:8*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
    const module=device.createShaderModule({code:`${svoEnvironmentLightingWGSL}
@group(0) @binding(0) var<storage,read> records:array<SvoEnvironmentLightingRecord>;
@group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
const normals=array<vec3f,4>(${normals.map(n=>`vec3f(${n.join(",")})`).join(",")});
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
  let e=svoEnvironmentDiffuseIrradiance(records[id.x/4u],normals[id.x%4u]);
  output[id.x*2u]=vec4f(e,1);output[id.x*2u+1u]=vec4f(.5*e/3.141592653589793,1);
}`});
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(8);pass.end();encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const values=new Float32Array(read.getMappedRange());
    for(let i=0;i<8;i++)evaluateSvoEnvironmentDiffuseIrradiance(records[Math.floor(i/4)]!,normals[i%4]!).forEach((v,c)=>{
      assert.ok(Math.abs(values[i*8+c]!-v)<1e-6);assert.ok(Math.abs(values[i*8+4+c]!-.5*v/Math.PI)<1e-6);
    });
    read.unmap();for(const b of[input,output,read])b.destroy();
  });
});
