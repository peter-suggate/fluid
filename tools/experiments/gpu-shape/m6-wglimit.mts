// GPU-shape M6 (lane B2): the per-core workgroup-memory limit, measured (docs/plans/gpu-shape/m6-residency.md).
//
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/m6-wglimit.mts
//
// A light kernel (few registers) of L lanes with W bytes of workgroup memory, every byte touched, then a fixed ALU
// spin (~100 us) so groups overlap fully. Each group adds one to a live counter at its start, keeps the running max,
// and subtracts one at its end: the peak is the groups resident at once (over the GPU's 32 cores). 4096 groups.
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(new URL(".", import.meta.url).pathname, "../../..");
const { createProcessRetainedDawnGPU } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const dawn = await import(pathToFileURL(resolve(root, "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
const device: GPUDevice = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const errors: string[] = []; device.addEventListener?.("uncapturederror", (e: any) => errors.push(String(e.error?.message)));

const buf = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
const read = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const kernel = (L: number, W: number, spin: number) => /* wgsl */`
@group(0) @binding(0) var<storage,read_write> c:array<atomic<u32>,16>;
${W > 0 ? `var<workgroup> pad:array<u32,${W / 4}>;` : ""}
@compute @workgroup_size(${L}) fn main(@builtin(local_invocation_index) lid:u32,@builtin(workgroup_id) wid:vec3u){
 if(lid==0u){let n=atomicAdd(&c[0],1u)+1u;atomicMax(&c[1],n);}
 var x=f32(lid)*1e-3+f32(wid.x)*1e-6;
 ${W > 0 ? `for(var j=lid;j<${W / 4}u;j+=${L}u){pad[j]=j^wid.x;}workgroupBarrier();x+=f32(pad[(lid*7u)%${W / 4}u]&1u);` : ""}
 for(var k=0u;k<${spin}u;k++){x=fma(x,0.999,0.001);}
 if(x==12345.0){atomicAdd(&c[3],1u);}
 workgroupBarrier();
 if(lid==0u){atomicSub(&c[0],1u);atomicAdd(&c[2],1u);}
}`;
const run = async (L: number, W: number, groups = 4096, spin = 40000) => {
  const module = device.createShaderModule({ code: kernel(L, W, spin) });
  const pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
  const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
  device.queue.writeBuffer(buf, 0, new Uint32Array(16));
  const enc = device.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(pipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(groups); pass.end();
  enc.copyBufferToBuffer(buf, 0, read, 0, 64); device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ); const w = new Uint32Array(read.getMappedRange().slice(0)); read.unmap();
  if (errors.length) throw new Error(errors[0]);
  return { peak: w[1]!, done: w[2]! };
};
// Calibrate the spin so a group takes ~100 us (a 64-lane group alone).
console.log("L\tW\tpeak\tper core\tW x per core");
for (const L of [64, 256, 512]) for (const W of [0, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 28672, 32768]) {
  if (W / 4 < 1 && W > 0) continue;
  const r = await run(L, W);
  console.log(`${L}\t${W}\t${r.peak}\t${(r.peak / 32).toFixed(2)}\t${((W * r.peak) / 32 / 1024).toFixed(1)} KB${r.done !== 4096 ? ` done ${r.done}` : ""}`);
}
process.exit(0);
