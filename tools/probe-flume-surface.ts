/** GPU regression for the flume's field and the voxel-normal striping bug.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-flume-surface.ts
 */
import assert from "node:assert/strict";
import { createNbflipThemeParkSlide } from "../lib/core/nbflip-theme-park-slide-scene";
import { sweptFlumeDistance, sweptFlumeSurfaceWGSL } from "../lib/core/swept-flume-surface";
import { createSolidWorld, materializedSolidWorldPages } from "../lib/core/solid-world";
import { createWebgpuSolidWorldPageLayout, packWebgpuSolidWorldPages } from "../lib/core/webgpu-solid-world-pages";
import { solidWorldProxyWGSL } from "../lib/core/webgpu-sparse-scene-proxies";
import { withUniformDevice } from "../tests/helpers/uniform-geometric";

const surface = createNbflipThemeParkSlide().solidSurface!;
assert.equal(surface.kind, "swept-flume");
if (surface.kind !== "swept-flume") throw new Error("Missing flume surface");
const world = createSolidWorld([{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [32, 24, 6], materialId: 21 }]);
const pages = materializedSolidWorldPages(world);
const layout = createWebgpuSolidWorldPageLayout({ baseWords: 0, authoredPageCount: pages.length, includesMaterial: true });
const origin = [-3.84, 0, -3.2] as const, h = .02;
const image = packWebgpuSolidWorldPages(layout, { pages, regions: [] }, [0, 0, 0], { origin_m: origin, cellSize_m: [h, h, h] });
const points: number[][] = [], expectedNormals: number[][] = [];
// Opposite faces and translated cell centres; tiny offsets used to change
// the arbitrary nearest-cell-face axis even though the wall stays planar.
for (let x = 5; x < 27; x++) for (const [z, sign] of [[.5, -1], [5.5, 1]]) for (const epsilon of [-1e-6, 0, 1e-6]) {
  points.push([origin[0] + (x + .5) * h + epsilon, 12.5 * h - epsilon, origin[2] + z * h]);
  expectedNormals.push([0, 0, sign]);
}
// Reproducible points around each piece, covering both sides and arc wrap seams.
const wallCount = points.length;
for (const p of surface.pieces) for (let i = 0; i <= 16; i++) {
  const d = p.length * i / 16, a = p.heading + p.curvature * d;
  const x = p.x + (p.curvature ? (Math.sin(a) - Math.sin(p.heading)) / p.curvature : d * Math.cos(a));
  const z = p.z + (p.curvature ? (Math.cos(p.heading) - Math.cos(a)) / p.curvature : d * Math.sin(a));
  for (const across of [-.42, -.36, -.24, 0, .24, .36, .42]) for (const height of [-.04, .08, .32, .5, .72]) {
    points.push([x - across * Math.sin(a), p.top - p.fall * i / 16 + height, z + across * Math.cos(a)]);
  }
}
const input = new Float32Array(points.flatMap(p => [...p, 0]));
await withUniformDevice("Flume surface and stable wall normals", async device => {
  const buffers: GPUBuffer[] = [];
  const buffer = (size: number, usage: number) => { const b = device.createBuffer({ size, usage }); buffers.push(b); return b; };
  try {
    const solids = buffer(image.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const queries = buffer(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const output = buffer(input.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const read = buffer(output.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    device.queue.writeBuffer(solids, 0, new Uint32Array(image));
    device.queue.writeBuffer(queries, 0, input);
    const module = device.createShaderModule({ code: `
const INVALID_INDEX=0xffffffffu;
@group(0) @binding(0) var<storage,read_write> maintenance:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> points:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> results:array<vec4f>;
${solidWorldProxyWGSL(layout)}
${sweptFlumeSurfaceWGSL(surface)}
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){
 let p=points[id.x].xyz;
 results[id.x]=vec4f(sampleSolidWorld(p,vec3f(.02)).normal,swAuthoredDistance(p));
}` });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [solids, queries, output].map((b, binding) => ({ binding, resource: { buffer: b } })) });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(points.length); pass.end();
    encoder.copyBufferToBuffer(output, 0, read, 0, output.size); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(read.getMappedRange().slice(0)); read.unmap();
    let minimumAlignment = 1, maxFieldError = 0;
    points.forEach((_, i) => {
      if (i < wallCount) {
        const alignment = expectedNormals[i]!.reduce((sum, v, a) => sum + v * data[4 * i + a]!, 0);
        minimumAlignment = Math.min(minimumAlignment, alignment);
        assert.ok(alignment > .99999, `wall normal ${i}: ${data.slice(4 * i, 4 * i + 3)}`);
      }
      const expected = sweptFlumeDistance(surface, input[4 * i]!, input[4 * i + 1]!, input[4 * i + 2]!);
      const error = Math.abs(expected - data[4 * i + 3]!);
      maxFieldError = Math.max(maxFieldError, error);
      assert.ok(error < 2e-5, `CPU/GPU field disagreement at ${i}: ${error}`);
    });
    console.log(JSON.stringify({ passed: true, wallSamples: wallCount, fieldSamples: points.length, minimumAlignment, maxFieldError }));
  } finally { for (const b of buffers) b.destroy(); }
});
