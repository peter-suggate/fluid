import assert from "node:assert/strict";
import test from "node:test";
import { particleCanonicalWGSL } from "../lib/methods/uniform/uniform-particle-canonical.wgsl";
import { narrowBandParticleWGSL, narrowBandParticleStride } from "../lib/methods/uniform/uniform-narrow-band-apic.wgsl";
import { canonicalMergeCPU, compareParticleWords } from "../tools/particle-canonical-reference";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("GPU parallel canonical merge preserves FLIP/APIC payloads, dense cells and append-order independence", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("canonical particle cells", async device => {
    for (const apic of [false, true]) {
      const stride = narrowBandParticleStride(apic), counts = [0, 1, 8, 16, 65, 257, 5319, 10638], ranges: number[] = [];
      let total = 0;
      for (const n of counts) { ranges.push(total, n); total += n; }
      const records = Array.from({ length: total }, (_, i) => {
        const f = new Float32Array(stride);
        for (let k = 0; k < stride; k++) f[k] = k < 3 ? Math.floor(i / 7) % 11 : (i * 17 + k * 31) % 113 - 56;
        // Position ties, identical records, and pairs differing only in the
        // final payload word (APIC lifetime included).
        if (i % 5 === 0) { f.fill(1); f[stride - 1] = i % 101; }
        return new Uint32Array(f.buffer);
      });
      const expected = new Uint32Array(total * stride);
      for (let c = 0; c < counts.length; c++) {
        const start = ranges[2 * c]!, n = counts[c]!;
        records.slice(start, start + n).sort(compareParticleWords).forEach((r, j) => expected.set(r, (start + j) * stride));
      }
      const source = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const output = device.createBuffer({ size: expected.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const cells = device.createBuffer({ size: 4 * ranges.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(cells, 0, new Uint32Array(ranges));
      const prefix = 17, scratch = device.createBuffer({ size: 4 * (prefix + 4 * total + 17), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const read = device.createBuffer({ size: Math.max(output.size, scratch.size), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const module = device.createShaderModule({ code: `${narrowBandParticleWGSL(apic)}
@group(0) @binding(0) var<storage,read_write> source:array<Particle>;
@group(0) @binding(1) var<storage,read_write> particles:array<Particle>;
@group(0) @binding(2) var<storage,read> cells:array<vec2u>;
@group(0) @binding(3) var<storage,read_write> links:array<u32>;
const NB_LINKS:u32=${prefix}u;
${particleCanonicalWGSL(apic)}
var<workgroup> range:vec2u;
@compute @workgroup_size(256) fn main(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 if(lane==0u){range=cells[group.x];}let cell=workgroupUniformLoad(&range);
 for(var k=lane;k<cell.y;k+=256u){let at=cell.x+k;links[NB_LINKS+4u*at]=at;}
 storageBarrier();workgroupBarrier();var bank=0u;
 for(var width=1u;width<cell.y;width*=2u){
  for(var k=lane;k<cell.y;k+=256u){nbCanonicalMerge(cell.x+k,cell.x,cell.y,width,bank);}
  storageBarrier();workgroupBarrier();bank^=1u;
 }
 for(var k=lane;k<cell.y;k+=256u){let at=cell.x+k;particles[at]=source[links[NB_LINKS+4u*at+bank]];}
}` });
        assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error").map(m => `${m.lineNum}: ${m.message}`), []);
        const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [source, output, cells, scratch].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        for (const reverse of [false, true]) {
          const input = new Uint32Array(expected.length), inputRecords: Uint32Array[] = [];
          for (let c = 0; c < counts.length; c++) {
            const start = ranges[2 * c]!, n = counts[c]!;
            for (let j = 0; j < n; j++) { const r = records[start + (reverse ? n - 1 - j : j)]!; input.set(r, (start + j) * stride); inputRecords.push(r); }
          }
          assert.deepEqual(canonicalMergeCPU(inputRecords, counts).records.flatMap(r => [...r]), [...expected]);
          device.queue.writeBuffer(source, 0, input);
          device.queue.writeBuffer(output, 0, new Uint32Array(expected.length).fill(0xdeadbeef));
          device.queue.writeBuffer(scratch, 0, new Uint32Array(scratch.size / 4).fill(0xdeadbeef));
          const e = device.createCommandEncoder(), p = e.beginComputePass();
          p.setPipeline(pipeline); p.setBindGroup(0, group); p.dispatchWorkgroups(counts.length); p.end();
          e.copyBufferToBuffer(output, 0, read, 0, output.size); device.queue.submit([e.finish()]);
          await read.mapAsync(GPUMapMode.READ); const result = new Uint32Array(read.getMappedRange()).slice(0, expected.length); read.unmap();
          assert.deepEqual(result, expected, `${apic ? "APIC" : "FLIP"}, reversed=${reverse}`);
          const check = device.createCommandEncoder(); check.copyBufferToBuffer(scratch, 0, read, 0, scratch.size); device.queue.submit([check.finish()]);
          await read.mapAsync(GPUMapMode.READ); const words = new Uint32Array(read.getMappedRange());
          assert.ok(words.subarray(0, prefix).every(w => w === 0xdeadbeef));
          assert.ok(words.subarray(prefix + 4 * total, scratch.size / 4).every(w => w === 0xdeadbeef));
          for (let i = 0; i < total; i++) { assert.equal(words[prefix + 4 * i + 2], 0xdeadbeef); assert.equal(words[prefix + 4 * i + 3], 0xdeadbeef); }
          read.unmap();
        }
      } finally { read.destroy(); scratch.destroy(); cells.destroy(); output.destroy(); source.destroy(); }
    }
  });
});
