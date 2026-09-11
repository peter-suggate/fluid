import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// An affine field's mean on each cubic cell is its value at the centre.
// Exercise the actual production stencil on a synthetic symmetric 2:1 grid.
const source = readFileSync(new URL("../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
function production(name: string): string {
  const result = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(result, name); return result;
}
assert.ok(process.env.WEBGPU_NODE_MODULE);
await acquireWebGPUExclusiveLock("dawn-probe", "transport-affine-consistency");
const live = new Set<GPU>();
Object.assign(globalThis, { cm12ConsistencyGPUs: live });
let gpu: GPU | undefined, device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const shader = device.createShaderModule({ code: `
const INVALID=0xffffffffu;
fn cm12RecordFailure(code:u32,cell:u32,data:vec4u){_=code;_=cell;_=data;}
struct TransportStencil{cells:array<u32,8>,weights:array<f32,8>}
struct Owner{cell:u32}
@group(0)@binding(0)var<storage,read>queries:array<vec4f>;
@group(0)@binding(1)var<storage,read_write>output:array<vec4f>;
fn cm12ClampToResidentWorld(q:vec3f,m:vec3f)->vec3f{return clamp(q,m,vec3f(32)-m);}
fn cellTransportActive(c:u32)->bool{return c!=INVALID;}
fn minimum(c:u32)->vec3u{return vec3u(c%32u,(c/32u)%32u,c/1024u);}
fn scale(q:vec3u)->u32{return select(2u,1u,all(q>=vec3u(8))&&all(q<vec3u(24)));}
fn cellWidths(c:u32)->vec3f{return vec3f(f32(scale(minimum(c))));}
fn cellMinimumWidth(c:u32)->f32{return cellWidths(c).x;}
fn cellCenter(c:u32)->vec3f{return vec3f(minimum(c))+.5*cellWidths(c);}
fn cm12TransportOwnerAtFine(q:vec3i,direct:bool)->Owner{
  _=direct;let v=vec3u(clamp(q,vec3i(0),vec3i(31)));let w=scale(v);let m=(v/w)*w;
  return Owner(m.x+32u*(m.y+32u*m.z));
}
${production("transportSourceSamplingSpans")}
${production("effectiveTransportStencilAtSpansMode")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)id:vec3u){
  if(id.x>=arrayLength(&queries)){return;}
  let q=queries[id.x];let cell=cm12TransportOwnerAtFine(vec3i(floor(q.xyz)),true).cell;
  let spans=transportSourceSamplingSpans(cell,true);
  let position=q.xyz+vec3f(q.w,0,0);
  let s=effectiveTransportStencilAtSpansMode(position,spans,true);
  var moment=0.0;var total=0.0;var own=0.0;
  for(var k=0u;k<8u;k++){
    let c=s.cells[k];let w=s.weights[k];if(c==INVALID){continue;}
    moment+=w*cellCenter(c).x;total+=w;
    if(c==cell){own+=w;}
  }
  output[id.x]=vec4f(moment,total,own,spans.x);
}` });
  assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => m.message), []);
  const points: number[][] = [];
  const origins = [[3,13,13], [7,13,13], [8.5,13.5,13.5], [25,13,13], [23.5,13.5,13.5]];
  for(const origin of origins) for(const displacement of [-.4,-.1,0,.1,.4]) points.push([...origin,displacement]);
  const input = device.createBuffer({size:points.length*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  const output = device.createBuffer({size:input.size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const copy = device.createBuffer({size:input.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(input,0,new Float32Array(points.flat()));
  const pipeline = await device.createComputePipelineAsync({layout:"auto",compute:{module:shader,entryPoint:"main"}});
  const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[input,output].map((buffer,binding)=>({binding,resource:{buffer}}))});
  const encoder=device.createCommandEncoder(), pass=encoder.beginComputePass();
  pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
  encoder.copyBufferToBuffer(output,0,copy,0,copy.size);device.queue.submit([encoder.finish()]);
  await copy.mapAsync(GPUMapMode.READ);
  const data=new Float32Array(copy.getMappedRange());
  const rows=points.map((q,i)=>({source:q.slice(0,3),displacement:q[3],expected:q[0]!+q[3]!,
    actual:data[4*i],partition:data[4*i+1],selfWeight:data[4*i+2],spacing:data[4*i+3]}));
  assert.ok(rows.every(r=>Number.isFinite(r.actual)&&Math.abs(r.partition!-1)<1e-6));
  console.log(JSON.stringify({affineConsistent:rows.every(r=>Math.abs(r.actual!-r.expected)<1e-5),rows},null,2));
  copy.unmap();input.destroy();output.destroy();copy.destroy();assert.deepEqual(errors,[]);
} finally { device?.destroy(); await releaseWebGPUExclusiveLock(); if(gpu)live.delete(gpu); }
