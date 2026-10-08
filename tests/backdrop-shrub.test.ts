import assert from "node:assert/strict";
import test from "node:test";
import { BACKDROP_SHRUB_LEAVES, BACKDROP_SHRUB_STEMS, BACKDROP_SHRUB_REACH, BACKDROP_SHRUB_RISE,
  backdropShrubDistance, backdropShrubWGSL } from "../lib/svo/features/backdrop/backdrop-shrub";
import { BACKDROP_SCATTER_CLASSES } from "../lib/svo/features/backdrop/backdrop-detail";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";

test("background shrubs have a fixed connected shoot budget within existing scatter bounds", () => {
  assert.equal(BACKDROP_SHRUB_LEAVES.length, 15);
  assert.equal(BACKDROP_SHRUB_STEMS.length, 3);
  assert.equal(BACKDROP_SCATTER_CLASSES[2].shape, 2);
  assert.equal(BACKDROP_SCATTER_CLASSES[3].shape, 2);
  for (const leaf of BACKDROP_SHRUB_LEAVES) {
    for (let axis = 0; axis < 3; axis++) {
      const extent = Math.hypot(leaf.u[axis]! * leaf.radii[0], leaf.v[axis]! * leaf.radii[1], leaf.w[axis]! * leaf.radii[2]);
      assert.ok(leaf.centre[axis]! + extent < (axis === 1 ? BACKDROP_SHRUB_RISE : BACKDROP_SHRUB_REACH));
      assert.ok(leaf.centre[axis]! - extent > (axis === 1 ? 0 : -BACKDROP_SHRUB_REACH));
    }
  }
  for (const stem of BACKDROP_SHRUB_STEMS) {
    for (let i = 0; i <= 100; i++) {
      const p = stem.a.map((v, axis) => v + (stem.b[axis]! - v) * i / 100) as [number, number, number];
      assert.ok(backdropShrubDistance(p) < 0, "stems remain connected from ground to terminal spray");
    }
  }
});

test("shrub packets retain open silhouettes at coarse cells with less surface work than rock puffs", () => {
  const rock = (x: number, y: number, z: number) => Math.min(Math.hypot(x, y-.55, z)-1,
    Math.hypot(x-.6, y-.4, z)-.65, Math.hypot(x-.6*Math.cos(2.3), y-.4, z-.6*Math.sin(2.3))-.65);
  for (const h of [.25, .1, .05]) {
    let shrubEdges = 0, rockEdges = 0, filled = 0, silhouette = 0;
    for (let x = -1.3; x < 1.3; x += h) for (let y = .1; y < 1.6; y += h) {
      let hit = false;
      for (let z = -1.3; z < 1.3; z += h) {
        const p: [number, number, number] = [x+.037, y+.023, z+.011];
        const a = backdropShrubDistance(p) < 0, b = rock(...p) < 0;
        hit ||= a;
        filled += Number(a);
        for (let axis = 0; axis < 3; axis++) {
          const q = [...p] as [number, number, number]; q[axis] += h;
          shrubEdges += Number(a !== (backdropShrubDistance(q) < 0));
          rockEdges += Number(b !== (rock(...q) < 0));
        }
      }
      silhouette += Number(hit);
    }
    assert.ok(filled > 10, `packets survive cells ${h} radii wide`);
    assert.ok(silhouette*h*h > .65 && silhouette*h*h < 2, `open but readable silhouette at ${h}`);
    assert.ok(shrubEdges < rockEdges * 1.1, `surface budget at ${h}: shrub ${shrubEdges}, puff ${rockEdges}`);
  }
});

test("Dawn background shrub field matches CPU leaves, woody stems and empty gaps", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  let device: GPUDevice | undefined;
  try {
    device = (await createDawnRenderDevice()).device;
    device.pushErrorScope("validation");
    const count = 8192;
    const data = new Float32Array(Array.from({ length: count }, (_, i) => [
      (((i*73)%1009)/1009-.5)*3, ((i*193)%1013)/1013*2-.2, (((i*311)%1019)/1019-.5)*3, 0]).flat());
    const module = device.createShaderModule({ code: `${backdropShrubWGSL}
@group(0) @binding(0) var<storage,read> points:array<vec4f>;
@group(0) @binding(1) var<storage,read_write> answers:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 if(id.x<arrayLength(&points)){answers[id.x]=backdropShrubDistance(points[id.x].xyz);}
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const input = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({ size: count*4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: count*4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, data);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [input, output].map((buffer,binding) => ({ binding, resource: { buffer } })) });
    const encoder=device.createCommandEncoder(), pass=encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0,group); pass.dispatchWorkgroups(count/64); pass.end();
    encoder.copyBufferToBuffer(output,0,read,0,count*4); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const result=new Float32Array(read.getMappedRange().slice(0)); read.unmap();
    for(let i=0;i<count;i++) {
      const expected=backdropShrubDistance([data[4*i],data[4*i+1],data[4*i+2]]);
      assert.ok(Math.abs(result[i]-expected)<5e-7, `sample ${i}: ${result[i]} != ${expected}`);
    }
    for(const buffer of [input,output,read]) buffer.destroy();
    assert.equal(await device.popErrorScope(),null);
  } finally { device?.destroy(); }
});
