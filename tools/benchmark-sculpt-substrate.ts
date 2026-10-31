/** Run with WEBGPU_NODE_MODULE and tools/gpu/fluid-gpu-run.sh timing.
 * Measures the bake and field lookup separately from startup and CPU packing. */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { WebgpuSculptSubstrate } from "../lib/svo/features/scene-publication/webgpu-sculpt-substrate";
import { SCULPT_ARENA_WORDS } from "../lib/svo/features/scene-publication/svo-sculpt-substrate";
import { packSvoSculptCuts, svoSculptCutsWGSL } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";
import { subcellEllipsoidPatch } from "../lib/core/solid-patch/ellipsoid-bounds";
const iterations = 24, queries = 65536;
const lattice = { origin_m: [0, 0, 0] as const, cellSize_m: [.01, .01, .01] as const };
const cut = (i: number) => {
  const x = (i % 8) / 8, y = (Math.floor(i / 8) % 8) / 8, z = (Math.floor(i / 64) % 8) / 8;
  return { ...subcellEllipsoidPatch([x - 3, y - 3, z - 3], [x + 3, y + 3, z + 3], "clear"), affectsScenery: true };
};
const stats = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return { median_ms: sorted[Math.floor(sorted.length / 2)], p95_ms: sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * .95))] };
};
await withUniformDevice("sculpt history scaling", async device => {
  assert.ok(device.features.has("timestamp-query"), "timestamp-query is needed for GPU measurements");
  const reports: unknown[] = [];
  const output = device.createBuffer({ size: queries * 4, usage: GPUBufferUsage.STORAGE });
  const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> arena:array<u32>;
@group(0) @binding(1) var<storage,read_write> results:array<f32>;
fn load(i:u32)->u32{return arena[i];}
${svoSculptCutsWGSL("load")}
@compute @workgroup_size(64) fn sampleField(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${queries}u){return;}
 let p=vec3f(f32(id.x%64u),f32((id.x/64u)%32u),f32(id.x/2048u))*.002-vec3f(.04);
 results[id.x]=svoSculptCutDistance(p);
}` });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "sampleField" } });
  try {
    // Repeat in reverse order to expose warmup/clock effects.
    for (const historyLength of [32, 4096, 4096, 32]) {
      const arena = device.createBuffer({ size: SCULPT_ARENA_WORDS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const timestamps = device.createQuerySet({ type: "timestamp", count: iterations * 4 });
      const resolved = device.createBuffer({ size: iterations * 4 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const read = device.createBuffer({ size: resolved.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      let timestampIndex: number | undefined;
      const measuredDevice = new Proxy(device, { get(target, key) {
        if (key === "createCommandEncoder") return (descriptor?: GPUCommandEncoderDescriptor) => {
          const encoder = target.createCommandEncoder(descriptor);
          return new Proxy(encoder, { get(enc, property) {
            if (property === "beginComputePass") return (desc?: GPUComputePassDescriptor) => enc.beginComputePass({ ...desc,
              ...(desc?.label === "Bake accepted sculpt pages" && timestampIndex !== undefined
                ? { timestampWrites: { querySet: timestamps, beginningOfPassWriteIndex: timestampIndex, endOfPassWriteIndex: timestampIndex + 1 } } : {}) });
            const value = Reflect.get(enc, property, enc); return typeof value === "function" ? value.bind(enc) : value;
          } });
        };
        const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
      } });
      try {
        const field = new WebgpuSculptSubstrate(measuredDevice); await field.initialize();
        let history = Array.from({ length: historyLength }, (_, i) => cut(i));
        field.publish(arena, 0, packSvoSculptCuts(history, lattice)); await device.queue.onSubmittedWorkDone();
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [arena, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const packing: number[] = [], planning: number[] = [];
        for (let i = 0; i < iterations; i++) {
          history = [...history, cut(i)]; const start = performance.now();
          const words = packSvoSculptCuts(history, lattice); const packed = performance.now();
          timestampIndex = i * 4; field.publish(arena, 0, words); planning.push(performance.now() - packed); packing.push(packed - start);
          assert.equal(field.lastUpdate.records, field.lastUpdate.pages);
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass({ timestampWrites: {
            querySet: timestamps, beginningOfPassWriteIndex: i * 4 + 2, endOfPassWriteIndex: i * 4 + 3 } });
          pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(queries / 64); pass.end();
          device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
        }
        const encoder = device.createCommandEncoder(); encoder.resolveQuerySet(timestamps, 0, iterations * 4, resolved, 0);
        encoder.copyBufferToBuffer(resolved, 0, read, 0, resolved.size); device.queue.submit([encoder.finish()]);
        await read.mapAsync(GPUMapMode.READ); const times = new BigUint64Array(read.getMappedRange());
        const bake = Array.from({ length: iterations }, (_, i) => Number(times[i * 4 + 1]! - times[i * 4]!) / 1e6);
        const sample = Array.from({ length: iterations }, (_, i) => Number(times[i * 4 + 3]! - times[i * 4 + 2]!) / 1e6);
        reports.push({ historyLength, pages: field.pageCount, proxies: field.proxyCount, update: field.lastUpdate,
          gpuBake: stats(bake.slice(4)), gpuSample65536: stats(sample.slice(4)), cpuPack: stats(packing.slice(4)), cpuPublish: stats(planning.slice(4)) });
        read.unmap();
      } finally { arena.destroy(); timestamps.destroy(); resolved.destroy(); read.destroy(); }
    }
    const report = { note: "GPU timestamp microbenchmark; excludes scene reconstruction, physics and presentation. CPU includes metadata planning, upload and submission. Four warmup iterations discarded.", reports };
    console.log(JSON.stringify(report, null, 2));
    const destination = process.argv[2]; if (destination) writeFileSync(destination, JSON.stringify(report, null, 2) + "\n");
  } finally { output.destroy(); }
}, ["timestamp-query"], ["disable-dawn-features=timestamp_quantization"]);
