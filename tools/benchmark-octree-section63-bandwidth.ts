import { pathToFileURL } from "node:url";
import { GPUPerformanceTraceRecorder } from "../lib/core/performance-trace";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const rows = Number(process.env.FLUID_SECTION63_BENCHMARK_ROWS ?? 1_048_576);
const neighbours = 18;
const repeats = Number(process.env.FLUID_SECTION63_BENCHMARK_REPEATS ?? 8);
if (!Number.isSafeInteger(rows) || rows < 262_144 || !Number.isSafeInteger(repeats) || repeats < 1) {
  throw new RangeError("Section 6.3 benchmark requires >=262144 rows and a positive repeat count");
}

await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/benchmark-octree-section63-bandwidth.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  if (!adapter?.features.has("timestamp-query")) throw new Error("Section 6.3 benchmark requires timestamp-query");
  const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const make = (label: string, size: number) => device.createBuffer({ label, size, usage: storage });
  const params = device.createBuffer({ label: "Section 6.3 bandwidth parameters", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([rows, neighbours]));
  const values = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) values[row] = Math.fround((row & 1023) / 1024);
  const valueBuffer = make("Section 6.3 values", values.byteLength);
  device.queue.writeBuffer(valueBuffer, 0, values);
  const ids = new Uint32Array(rows * neighbours), coefficients = new Float32Array(ids.length);
  // A full-period odd affine permutation defeats cache-line locality while
  // retaining deterministic input across variants and machines.
  for (let row = 0; row < rows; row += 1) for (let slot = 0; slot < neighbours; slot += 1) {
    const at = row * neighbours + slot;
    ids[at] = ((row * (2 * slot + 3) + 104_729 * (slot + 1)) >>> 0) % rows;
    coefficients[at] = Math.fround(0.03125 + (slot & 3) * 0.0078125);
  }
  const idBuffer = make("Resolved-row random neighbour IDs", ids.byteLength);
  const coefficientBuffer = make("Resolved-row per-cell coefficients", coefficients.byteLength);
  device.queue.writeBuffer(idBuffer, 0, ids); device.queue.writeBuffer(coefficientBuffer, 0, coefficients);
  const catalog = new Float32Array(1_608 * 19);
  for (let entry = 0; entry < 1_608; entry += 1) for (let channel = 0; channel < 19; channel += 1) {
    catalog[entry * 19 + channel] = channel === 0 ? 1 : Math.fround(0.03125 + (channel & 3) * 0.0078125);
  }
  const catalogBuffer = make("Section 6.3 resident 19-channel catalog", catalog.byteLength);
  device.queue.writeBuffer(catalogBuffer, 0, catalog);
  const output = make("Section 6.3 bandwidth output", values.byteLength);
  const shader = device.createShaderModule({ label: "Section 6.3 gather versus stream benchmark", code: `
struct Params{rows:u32,neighbours:u32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>values:array<f32>;
@group(0)@binding(2)var<storage,read>ids:array<u32>;
@group(0)@binding(3)var<storage,read>coefficients:array<f32>;
@group(0)@binding(4)var<storage,read>catalog:array<f32>;
@group(0)@binding(5)var<storage,read_write>output:array<f32>;
@compute @workgroup_size(256)fn resolvedGather(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=p.rows){return;}var v=values[row];let base=row*p.neighbours;
 for(var slot=0u;slot<18u;slot+=1u){v-=coefficients[base+slot]*values[ids[base+slot]];}output[row]=v;}
@compute @workgroup_size(256)fn section63Stream(@builtin(global_invocation_id)g:vec3u){let row=g.x;if(row>=p.rows){return;}var v=catalog[(row%1608u)*19u]*values[row];let base=(row/256u)*256u;
 for(var slot=0u;slot<18u;slot+=1u){let local=(row+slot+1u)&255u;v-=catalog[(row%1608u)*19u+1u+slot]*values[min(base+local,p.rows-1u)];}output[row]=v;}` });
  const pipeline = (entryPoint: string) => device.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint } });
  const resolvedPipeline = pipeline("resolvedGather"), streamPipeline = pipeline("section63Stream");
  const bind = (pipeline: GPUComputePipeline, entries: readonly (readonly [number, GPUBuffer])[]) =>
    device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map(([binding, buffer]) =>
      ({ binding, resource: { buffer } })) });
  const variants = [{ name: "resolved-gather", pipeline: resolvedPipeline,
    group: bind(resolvedPipeline, [[0, params], [1, valueBuffer], [2, idBuffer], [3, coefficientBuffer], [5, output]]) },
  { name: "section63-page-stream", pipeline: streamPipeline,
    group: bind(streamPipeline, [[0, params], [1, valueBuffer], [4, catalogBuffer], [5, output]]) }];
  const measure = async (variant: typeof variants[number]) => {
    const encoder = device.createCommandEncoder();
    const trace = new GPUPerformanceTraceRecorder(device, 1, "physics", variant.name,
      [{ id: "pressure-solve", label: variant.name }]);
    trace.boundary(encoder, `${variant.name} begin`); const pass = encoder.beginComputePass();
    pass.setPipeline(variant.pipeline); pass.setBindGroup(0, variant.group);
    for (let repeat = 0; repeat < repeats; repeat += 1) pass.dispatchWorkgroups(Math.ceil(rows / 256));
    pass.end(); trace.boundary(encoder, `${variant.name} complete`); trace.resolve(encoder);
    device.queue.submit([encoder.finish()]); const result = await trace.read();
    if (!result) throw new Error(`No timestamp result for ${variant.name}`);
    return result.total_ms / repeats;
  };
  for (const variant of variants) await measure(variant);
  const result = Object.fromEntries(await Promise.all(variants.map(async (variant) => [variant.name, await measure(variant)])));
  console.log(JSON.stringify({ rows, neighbours, repeats, cachePressureMiB: {
    resolvedRows: (ids.byteLength + coefficients.byteLength) / 2 ** 20,
    section63Catalog: catalog.byteLength / 2 ** 20,
  }, millisecondsPerApply: result,
  speedup: result["resolved-gather"] / result["section63-page-stream"] }));
  for (const buffer of [params, valueBuffer, idBuffer, coefficientBuffer, catalogBuffer, output]) buffer.destroy();
  device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
