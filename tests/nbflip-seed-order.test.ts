import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { narrowBandSeedJitterWGSL } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { withUniformDevice } from "./helpers/uniform-geometric";

// A frame's reseed sites follow from the cell, the site and the frame alone:
// seedCandidate's lanes read a seed list that seedCells fills in atomicAdd
// order, and a salt that read that order (the slot) or the list's count made
// every run seed differently.
const N = 512;
const shader = (salt: string) => /* wgsl */`
${narrowBandSeedJitterWGSL}
@group(0) @binding(0) var<storage,read> list:array<vec4u>;
@group(0) @binding(1) var<storage,read_write> sites:array<vec4f>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) gid:vec3u){
 let slot=gid.x;if(slot>=${N}u){return;}
 let entry=list[slot];let c=entry.xyz;let k=entry.w&7u;
 // Filed at the entry's own index (w's upper bits).
 sites[entry.w>>16u]=vec4f(vec3f(c)+0.5*vec3f(vec3u(k&1u,(k>>1u)&1u,k>>2u))+0.5*nbSeedJitter(c,k,${salt}),0.0);
}`;

async function seed(device: GPUDevice, entries: Uint32Array<ArrayBuffer>, salt: string): Promise<Float32Array> {
  const manager = gpuCompilationManagerFor(device);
  const module = manager.createShaderModule({ code: shader(salt) });
  const pipeline = await manager.compileComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
  const list = device.createBuffer({ size: entries.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(list, 0, entries);
  const sites = device.createBuffer({ size: 16 * N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16 * N, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: list } }, { binding: 1, resource: { buffer: sites } }] }));
    pass.dispatchWorkgroups(N / 64); pass.end();
    encoder.copyBufferToBuffer(sites, 0, read, 0, 16 * N); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); return new Float32Array(read.getMappedRange()).slice();
  } finally { if (read.mapState === "mapped") read.unmap(); list.destroy(); sites.destroy(); read.destroy(); }
}

test("reseed sites are independent of the seed list's fill order", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("seed order", async device => {
    // Entry i: a cell, a site k and the frame; w = i << 16 | frame << 3 | k.
    const entries = (frame: number) => {
      const e = new Uint32Array(4 * N);
      for (let i = 0; i < N; i++) e.set([37 + (i % 13), 5 + ((i >> 4) % 7), 90 + (i >> 7), (i << 16) | (frame << 3) | (i & 7)], 4 * i);
      return e;
    };
    // A deterministic shuffle of the list's slots.
    const shuffled = (e: Uint32Array<ArrayBuffer>) => {
      const order = [...Array(N).keys()];let s = 12345;
      for (let i = N - 1; i > 0; i--) { s = (s * 1103515245 + 12345) >>> 0; const j = s % (i + 1); [order[i], order[j]] = [order[j]!, order[i]!]; }
      const out = new Uint32Array(4 * N);order.forEach((from, to) => out.set(e.subarray(4 * from, 4 * from + 4), 4 * to));return out;
    };
    const frame = (w: string) => `(${w}>>3u)&8191u`;
    const salt = frame("entry.w");
    const a = await seed(device, entries(7), salt), b = await seed(device, shuffled(entries(7)), salt);
    assert.deepEqual(new Uint32Array(b.buffer), new Uint32Array(a.buffer), "the same frame's sites in any slot order");
    const next = await seed(device, entries(8), salt);
    let moved = 0;for (let i = 0; i < 4 * N; i += 4) if (next[i] !== a[i] || next[i + 1] !== a[i + 1] || next[i + 2] !== a[i + 2]) moved++;
    assert.ok(moved > 0.99 * N, `another frame reseeds a cell at new sites (${moved} of ${N} moved)`);
    for (let i = 0; i < 4 * N; i += 4) for (let axis = 0; axis < 3; axis++) {
      const c = entries(7)[i + axis]!, k = (entries(7)[i + 3]! & 7) >> axis & 1, f = a[i + axis]! - c - 0.5 * k;
      assert.ok(f >= 0 && f <= 0.5, `site ${i / 4} axis ${axis}: in its eighth (${f})`);
    }
    // The former salt (slot plus one and the list's count) moves sites with the slot order.
    const slotSalt = `slot+1u+${N}u`;
    const c = await seed(device, entries(7), slotSalt), d = await seed(device, shuffled(entries(7)), slotSalt);
    assert.notDeepEqual(new Uint32Array(d.buffer), new Uint32Array(c.buffer), "a slot-salted seeding depends on the fill order");
  });
  // The seeding itself salts with the frame alone (seedCandidate).
  const source = readFileSync(new URL("../lib/methods/uniform/uniform-narrow-band-flip.ts", import.meta.url), "utf8");
  const sets = [...source.matchAll(/nbSeedSalt=([^;]*);/g)].map(m => m[1]);
  assert.deepEqual(sets, ["params.live.z"], "seedCandidate's salt is the frame index");
});
