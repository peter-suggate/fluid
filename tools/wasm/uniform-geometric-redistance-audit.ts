/** Isolate the production closest-point shader on a captured scene-stage input. */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../../lib/harness/webgpu-smoke-isolation";

const capture=JSON.parse(readFileSync(process.argv[2]??"/tmp/fluid-uniform-ceiling-release-30.json","utf8"));
const [nx,ny]=capture.input.dimensions as [number,number];
const [hx,hy]=capture.input.cellSize as [number,number];
const source=readFileSync("lib/methods/uniform/uniform-volume.wgsl.ts","utf8");
const reference=readFileSync("lib/methods/uniform/webgpu-uniform-reference.wgsl.ts","utf8");
const extract=(text:string,name:string)=>{
  const start=text.indexOf(`fn ${name}(`);assert.ok(start>=0,name);
  let depth=0;let opened=false;
  for(let end=start;end<text.length;end++){
    if(text[end]==="{"){depth++;opened=true;}
    if(text[end]==="}"&&--depth===0&&opened)return text.slice(start,end+1);
  }
  throw Error(`Unclosed ${name}`);
};
const vertexCount=(nx+1)*(ny+1);
const shader=`
const UNIFORM_REFERENCE_DIMENSION:u32 = ${process.argv.includes("--2d")?2:3}u;
@group(0) @binding(0) var uvPhiIn:texture_3d<f32>;
@group(0) @binding(1) var uvPhiOut:texture_storage_3d<r32float,write>;
@group(0) @binding(2) var<storage,read_write> audit:array<vec4f>;
fn dims()->vec3i{return vec3i(${nx},${ny},1);}
fn activeVertexId(gid:vec3u)->vec3i{return vec3i(gid);}
${extract(reference,"d4Sum8")}
${extract(source,"uvCorner")}
${extract(source,"uvPhi")}
${extract(source,"uvGradient")}
@compute @workgroup_size(4,4,4)
${extract(source,"uvRedistancePhi").replaceAll("params.cellGravity.xyz",`vec3f(${hx},${hy},${hx})`).replace("let norm=dot(g/h,g/h);",`let norm=dot(g/h,g/h);let record=(gid.x+${nx+1}u*(gid.y+${ny+1}u*gid.z))*16u+2u*i;audit[record]=vec4f(q,uvPhi(q));audit[record+1u]=vec4f(g,norm);`)}`;
await acquireWebGPUExclusiveLock("dawn-test","uniform geometric redistance scene audit");
let device:GPUDevice|undefined;
try{
  const source=process.env.WEBGPU_NODE_MODULE;assert.ok(source);
  const dawn=await import(pathToFileURL(source).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,["backend=metal"]);
  const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const shaderModule=device.createShaderModule({code:shader});
  const info=await shaderModule.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==="error"),[]);
  const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:shaderModule,entryPoint:"uvRedistancePhi"}});
  const input=device.createTexture({size:[nx+1,ny+1,2],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  const output=device.createTexture({size:[nx+1,ny+1,2],dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});
  const size=vertexCount*2*16*16;
  const audit=device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const mapped=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const row=Math.ceil((nx+1)*4/256)*256;
  const final=device.createBuffer({size:row*(ny+1)*2,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  const bind=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:input.createView()},{binding:1,resource:output.createView()},{binding:2,resource:{buffer:audit}}]});
  device.queue.writeTexture({texture:input},new Float32Array(capture.gpu.advectedPhi3d??[...capture.gpu.advectedPhi,...capture.gpu.advectedPhi]),{bytesPerRow:(nx+1)*4,rowsPerImage:ny+1},[nx+1,ny+1,2]);
  const encoder=device.createCommandEncoder();const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(Math.ceil((nx+1)/4),Math.ceil((ny+1)/4),1);pass.end();
  encoder.copyBufferToBuffer(audit,0,mapped,0,size);encoder.copyTextureToBuffer({texture:output},{buffer:final,bytesPerRow:row,rowsPerImage:ny+1},[nx+1,ny+1,2]);device.queue.submit([encoder.finish()]);
  await Promise.all([mapped.mapAsync(GPUMapMode.READ),final.mapAsync(GPUMapMode.READ)]);
  const data=[...new Float32Array(mapped.getMappedRange())];const values=new Float32Array(final.getMappedRange());
  const phi=Array.from({length:vertexCount},(_,i)=>values[Math.floor(i/(nx+1))*row/4+i%(nx+1)]);
  const result={phi,productionMaxError:Math.max(...phi.map((v,i)=>Math.abs(v-capture.gpu.phi[i]))),iterations:Array.from({length:8},(_,i)=>data.slice((7+(nx+1)*4)*64+i*8,(7+(nx+1)*4)*64+i*8+8))};
  writeFileSync("/tmp/fluid-uniform-redistance-audit.json",JSON.stringify(result,null,2));console.log(JSON.stringify({...result,phi:undefined}));
  mapped.unmap();final.unmap();
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
