/** Fixed-input correctness/timing comparison, independent of scene trajectory.
 * Export the pre-change uniform-particle-canonical.wgsl.ts to a temporary .ts
 * file and pass --baseline=/absolute/path.ts. Run under the timing GPU lock.
 * No production switch to the old algorithm. A 30 s watchdog bounds this probe. */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { particleCanonicalWGSL } from "../lib/methods/uniform/uniform-particle-canonical.wgsl";
import { narrowBandParticleWGSL } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { canonicalMergeCPU, compareParticleWords } from "./particle-canonical-reference";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";
import { GPUPassProfile } from "./gpu-pass-profile";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";

const baseline = process.argv.find(a => a.startsWith("--baseline="))?.slice(11);
assert.ok(baseline, "--baseline must name the exported pre-change canonical shader module");
const old = await import(pathToFileURL(resolve(baseline)).href) as { particleCanonicalWGSL(apic: boolean): string };
assert.ok(old.particleCanonicalWGSL(false).includes("fn nbParticleRank("), "baseline must be the former all-pairs rank");
const watchdog = setTimeout(() => { console.error("Canonical probe exceeded 30 s"); process.exit(2); }, 30_000);
watchdog.unref(); paceDawnEventPolling();
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
try {
  await withUniformDevice("canonical fixed-input comparison", async device => {
    const profile = new GPUPassProfile(device);
    try {
      const programs = await Promise.all([false, true].map(async merge => {
        const module = device.createShaderModule({ code: `${narrowBandParticleWGSL(false)}
@group(0) @binding(0) var<storage,read_write> source:array<Particle>;
@group(0) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(2) var<storage,read_write> links:array<u32>;
const NB_LINKS:u32=0u;
${merge ? particleCanonicalWGSL(false) : old.particleCanonicalWGSL(false)}
var<workgroup> nbTestCount:u32;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){nbTestCount=arrayLength(&source);}let count=workgroupUniformLoad(&nbTestCount);
 for(var i=lane;i<count;i+=256u){
  links[4u*i]=i;
  ${merge ? "" : "let bits=bitcast<vec3u>(source[i].position.xyz);links[4u*i+1u]=bits.x;links[4u*i+2u]=bits.y;links[4u*i+3u]=bits.z;"}
 }
 storageBarrier();workgroupBarrier();
 ${merge ? `var bank=0u;
 for(var width=1u;width<count;width*=2u){
  for(var i=lane;i<count;i+=256u){nbCanonicalMerge(i,0u,count,width,bank);}
  storageBarrier();workgroupBarrier();bank^=1u;
 }
 for(var i=lane;i<count;i+=256u){particles[i]=source[links[4u*i+bank]];}` : `
 for(var i=lane;i<count;i+=256u){particles[nbParticleRank(i,0u,count,i,bitcast<vec3u>(source[i].position.xyz))]=source[i];}`}
}` });
        assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
        return device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
      }));
      for (const count of [8, 256, 5319, 10638]) {
        let random = 0x12345678;
        const records = Array.from({ length: count }, (_, i) => {
          random ^= random << 13; random ^= random >>> 17; random ^= random << 5;
          const f = new Float32Array(12);
          // Half the positions coincide; all remain in the same cell.
          f.set(i % 2 ? [(random & 255) / 256, ((random >>> 8) & 255) / 256, ((random >>> 16) & 255) / 256] : [0.25, 0.25, 0.25]);
          for (let k = 3; k < 12; k++) f[k] = (i * 17 + k * 31) % 113;
          return new Uint32Array(f.buffer);
        });
        const expected = [...records].sort(compareParticleWords).flatMap(r => [...r]);
        assert.deepEqual(canonicalMergeCPU(records, [count]).records.flatMap(r => [...r]), expected);
        const bytes = 48 * count;
        const source = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const links = device.createBuffer({ size: 16 * count, usage: GPUBufferUsage.STORAGE });
        const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          device.queue.writeBuffer(source, 0, Uint32Array.from(records.flatMap(r => [...r])));
          const groups = programs.map(p => device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [source, output, links].map((buffer, binding) => ({ binding, resource: { buffer } })) }));
          const timings: number[][] = [[], []];
          for (let repeat = 0; repeat < 7; repeat++) for (let arm = 0; arm < 2; arm++) {
            const index = (arm + repeat) % 2;
            if (repeat) profile.start();
            const e = profile.device.createCommandEncoder(), pass = e.beginComputePass({ label: index ? "parallel merge" : "all-pairs rank" });
            pass.setPipeline(programs[index]!); pass.setBindGroup(0, groups[index]!); pass.dispatchWorkgroups(1); pass.end();
            e.copyBufferToBuffer(output, 0, read, 0, bytes); device.queue.submit([e.finish()]);
            await read.mapAsync(GPUMapMode.READ);
            assert.deepEqual([...new Uint32Array(read.getMappedRange())], expected, `${count} records, arm ${index}`); read.unmap();
            if (repeat) timings[index]!.push((await profile.finish())[0]!.ms);
          }
          const medians = timings.map(t => [...t].sort((a, b) => a - b).slice(2, 4).reduce((a, b) => a + b, 0) / 2);
          console.log(JSON.stringify({ count, everyOutputMatchesCPU: true, samples: timings, oldMedian_ms: medians[0], mergeMedian_ms: medians[1], speedup: medians[0]! / medians[1]! }));
        } finally { read.destroy(); links.destroy(); output.destroy(); source.destroy(); }
      }
    } finally { profile.destroy(); }
  }, ["timestamp-query"], ["disable-dawn-features=timestamp_quantization"]);
} finally { clearTimeout(watchdog); }
