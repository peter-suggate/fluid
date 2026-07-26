import { pathToFileURL } from "node:url";
import { GPUPerformanceTraceRecorder } from "../lib/performance-trace";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/benchmark-octree-pressure-page-shapes.ts");
try {
const modulePath = process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`;
const dawn = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, dawn.globals);
const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
const adapter = await gpu.requestAdapter();
if (!adapter || !adapter.features.has("timestamp-query")) {
  throw new Error("Pressure page-shape benchmark requires a timestamp-query adapter");
}
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });

const PAGES = 256;
const REPEATS = 40;
const WARMUPS = 3;
const SAMPLES = 21;

type Shape = readonly [number, number, number];
type Variant = { shape: Shape; haloElements: number; pipeline: GPUComputePipeline;
  params: GPUBuffer; output: GPUBuffer; group: GPUBindGroup };

function shader([px, py, pz]: Shape): string {
  const hx = px + 2, hy = py + 2, hz = pz + 2;
  const pageElements = px * py * pz, haloElements = hx * hy * hz;
  return `
struct Params{pageCount:u32}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
var<workgroup> slots:array<u32,${haloElements}>;
var<workgroup> a:array<f32,${haloElements}>;
var<workgroup> b:array<f32,${haloElements}>;
var<workgroup> rhs:array<f32,${haloElements}>;
var<workgroup> diagonal:array<f32,${haloElements}>;
fn hi(local:u32)->u32{let x=local%${px}u;let yz=local/${px}u;
 return x+1u+${hx}u*((yz%${py}u)+1u+${hy}u*((yz/${py}u)+1u));}
fn applied(index:u32,useB:bool)->f32{let source=select(a[index],b[index],useB);
 var value=diagonal[index]*source;
 let offsets=array<i32,6>(-1,1,-${hx},${hx},-${hx * hy},${hx * hy});
 for(var k=0u;k<6u;k+=1u){let n=u32(i32(index)+offsets[k]);
  if(slots[n]!=0xffffffffu){value-=select(a[n],b[n],useB);}}
 return value;}
fn sweep(lid:u32,useB:bool,weight:f32){for(var local=lid;local<${pageElements}u;local+=128u){let index=hi(local);
 let source=select(a[index],b[index],useB);let next=source+weight*(rhs[index]-applied(index,useB))/diagonal[index];
 if(useB){a[index]=next;}else{b[index]=next;}}}
@compute @workgroup_size(128) fn benchmark(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 for(var index=lid;index<${haloElements}u;index+=128u){slots[index]=index;a[index]=f32(index&7u)*0.01;
  b[index]=a[index];rhs[index]=0.25;diagonal[index]=7.0;}workgroupBarrier();
 sweep(lid,false,0.42);workgroupBarrier();sweep(lid,true,0.57);workgroupBarrier();
 sweep(lid,false,0.78);workgroupBarrier();sweep(lid,true,1.03);workgroupBarrier();
 if(lid==0u&&wid.x<params.pageCount){output[wid.x]=a[hi(0u)]+f32(slots[hi(0u)]&1u);}}
`;
}

function variant(shape: Shape): Variant {
  const haloElements = (shape[0] + 2) * (shape[1] + 2) * (shape[2] + 2);
  const module = device.createShaderModule({ code: shader(shape) });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "benchmark" } });
  const params = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: PAGES * 4, usage: GPUBufferUsage.STORAGE });
  device.queue.writeBuffer(params, 0, new Uint32Array([PAGES]));
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: output } },
  ] });
  return { shape, haloElements, pipeline, params, output, group };
}

async function measure(v: Variant): Promise<number> {
  const encoder = device.createCommandEncoder();
  const recorder = new GPUPerformanceTraceRecorder(device, 1, "physics",
    `octree-pressure-page-${v.shape.join("x")}`,
    [{ id: "pressure-solve", label: "Pressure page smoother" }]);
  recorder.boundary(encoder, "Pressure page smoother begin");
  const pass = encoder.beginComputePass();
  pass.setPipeline(v.pipeline); pass.setBindGroup(0, v.group);
  for (let repeat = 0; repeat < REPEATS; repeat += 1) pass.dispatchWorkgroups(PAGES);
  pass.end();
  recorder.boundary(encoder, "Pressure page smoother complete");
  recorder.resolve(encoder);
  device.queue.submit([encoder.finish()]);
  const trace = await recorder.read();
  if (!trace) throw new Error("Pressure page-shape trace was unavailable");
  return trace.total_ms * 1000 / REPEATS / PAGES;
}

const variants = [variant([4, 4, 4]), variant([8, 8, 4])];
for (let warmup = 0; warmup < WARMUPS; warmup += 1) for (const v of variants) await measure(v);
const samples = new Map<Variant, number[]>(variants.map((v) => [v, []]));
for (let sample = 0; sample < SAMPLES; sample += 1) {
  for (const v of sample % 2 === 0 ? variants : [...variants].reverse()) samples.get(v)!.push(await measure(v));
}
const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
for (const v of variants) {
  const cells = v.shape[0] * v.shape[1] * v.shape[2], sharedBytes = v.haloElements * 5 * 4;
  const med = median(samples.get(v)!);
  console.log(JSON.stringify({ shape: v.shape.join("x"), cells, haloElements: v.haloElements,
    haloAmplification: v.haloElements / cells, sharedBytes,
    occupancy32KiB: Math.floor(32 * 1024 / sharedBytes), medianMicrosecondsPerPage: med,
    estimate6912RowsMicroseconds: med * Math.ceil(6912 / cells) }));
  v.params.destroy(); v.output.destroy();
}
device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
