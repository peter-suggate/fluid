import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL("../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const fn = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(fn, name);
  return fn.replace(/\$\{implicitSharpeningOwnerArithmeticForQA[\s\S]*?\}/g,
    "effectiveTransportStencilAtSpans(position,transportSourceSamplingSpans(cell,false))");
};

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("sharpening transports sub-quantum shares symmetrically instead of pinning them to the donor", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "fractional-sharpening-receipts");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const shader = device.createShaderModule({code: `
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
@group(0)@binding(2)var<storage,read_write>result:array<f32>;
const INVALID=0xffffffffu;
const CM12_LIQUID_ISOVALUE=0.5;
struct Params{counts:vec4u,stateOffsets5:vec4u}
const p=Params(vec4u(9,0,0,0),vec4u(9,0,0,0));
struct TransportStencil{cells:array<u32,8>,weights:array<f32,8>}
fn cellTransportActive(c:u32)->bool{_=c;return true;}
fn conditionedDensity(c:u32)->f32{return state[c];}
fn cellVolume(c:u32)->f32{_=c;return 1.0;}
fn cm12PhysicalMassFixedScale()->f32{return 65536.0;}
fn traceSharpeningMass(c:u32)->vec3f{_=c;return vec3f(1.5);}
fn transportSourceSamplingSpans(c:u32,b:bool)->vec3f{_=c;_=b;return vec3f(1);}
fn effectiveTransportStencilAtSpans(q:vec3f,w:vec3f)->TransportStencil{
  _=q;_=w;var s:TransportStencil;
  for(var i=0u;i<8u;i++){s.cells[i]=i+1u;s.weights[i]=0.125;}
  return s;
}
fn cm12Phase1QACaptureSharpening(c:u32,q:vec3f,s:TransportStencil,r:f32,d:f32,m:i32){_=c;_=q;_=s;_=r;_=d;_=m;}
fn cm12RecordFailure(a:u32,b:u32,c:vec4u){_=a;_=b;_=c;}
${production("addSharpeningReceipt")}
${source.includes("fn addSharpeningFractionalReceipt(") ? production("addSharpeningFractionalReceipt") : ""}
${production("sharpeningReceipt")}
${production("scatterSharpeningCell")}
@compute @workgroup_size(1)
fn main(){
  scatterSharpeningCell(0u);
  storageBarrier();
  for(var c=0u;c<9u;c++){result[c]=state[c]+state[9u+c]+sharpeningReceipt(c);}
}`});
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({layout:"auto",compute:{module:shader,entryPoint:"main"}});
    const buffers = [18*4,7*9*4,9*4].map(size => device!.createBuffer({size,
      usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC}));
    const readback = device.createBuffer({size:9*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try {
      const group = device.createBindGroup({layout:pipeline.getBindGroupLayout(0),
        entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
      for(const quanta of [1000,4,0.25]) {
        const mass=quanta/65536, input=new Float32Array(18);input[0]=mass;input[9]=-mass;
        device.queue.writeBuffer(buffers[0],0,input);
        const encoder=device.createCommandEncoder();encoder.clearBuffer(buffers[1]);
        const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();
        encoder.copyBufferToBuffer(buffers[2],0,readback,0,readback.size);device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const result=[...new Float32Array(readback.getMappedRange())];readback.unmap();
        assert.equal(result[0],0, `${quanta} quanta: the full debit must leave the donor`);
        for(const share of result.slice(1)) assert.equal(share,mass/8, "equal recipients must receive equal shares");
        assert.equal(result.reduce((sum,value)=>sum+value,0),mass,"sharpening must conserve total mass");
      }
    } finally {readback.destroy();buffers.forEach(buffer=>buffer.destroy());}
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();assert.ok(gpu);}
});
