import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const source = readFileSync(new URL("../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const election = source.match(/fn sparseWorldFrontierAllocationOwner\([\s\S]*?\n}/)?.[0];
assert.ok(election);
const sweptMask = source.match(/fn cm12SweptPageSupportMask\([\s\S]*?\n}/)?.[0]; assert.ok(sweptMask);

(modulePath ? test : test.skip)("frontier allocation elects every unique signed, diagonal and macro-adjacent request", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "frontier-allocation-election");
  let device: GPUDevice | undefined;
  const buffers: GPUBuffer[] = [];
  const live = new Set<GPU>();
  Object.assign(globalThis, { frontierAllocationGPU: live });
  try {
    const dawn = await import(pathToFileURL(modulePath!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    device = await (await gpu.requestAdapter())!.requestDevice();
    const bricks = [
      { q: [-2, -2, -2], span: 2, active: true, support: 0x7ffffff, blocked: 0 },
      { q: [0, -1, -1], span: 1, active: true, support: 0x7ffffff, blocked: 1 << 24 },
      { q: [-1, 1, -1], span: 1, active: true, support: 0x7ffffff, blocked: 1 << 19 },
      { q: [1, 0, 0], span: 1, active: true, support: 0x7ffffff, blocked: 0 },
      // An inactive backed leaf carries no allocation invocation, even if a
      // previous activity word remains. It must not win another source's key.
      { q: [0, 1, 0], span: 1, active: false, support: 0x7ffffff, blocked: 0 },
      { q: [0, 0, 1], span: 1, active: true, support: 0x7ffffff, blocked: 0x7ffffff },
      { q: [1, -1, 1], span: 1, active: true, support: 0x1249249, blocked: 0 },
    ];
    const own = (q: number[]) => bricks.findIndex(b => q.every((v, axis) => v >= b.q[axis]! && v < b.q[axis]! + b.span));
    const activity = new Uint32Array(bricks.length * 48);
    const geometry = new Int32Array(bricks.length * 4);
    bricks.forEach((b, id) => {
      activity[48 * id + 1] = id === 1 ? 0 : 64; activity[48 * id + 3] = b.support;
      activity[48 * id + 10] = Number(b.active); activity[48 * id + 4] = b.blocked;
      geometry.set([...b.q, b.span], 4 * id);
    });
    const expected = new Map<string, number>();
    const targets = new Map<number, string>();
    for (const [id, b] of bricks.entries()) for (let bit = 0; bit < 27; bit++) {
      if (!b.active || bit === 13 || !(b.support & (1 << bit)) || (b.blocked & (1 << bit))) continue;
      const offset = [bit % 3 - 1, Math.floor(bit / 3) % 3 - 1, Math.floor(bit / 9) - 1];
      const q = b.q.map((v, axis) => v + offset[axis]!); if (own(q) >= 0) continue;
      const key = q.join(","); targets.set(27 * id + bit, key);
      expected.set(key, Math.min(expected.get(key) ?? Infinity, id + 1));
    }
    const shader = device.createShaderModule({ code: `
const INVALID=0xffffffffu;
@group(0)@binding(0)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read>geometry:array<vec4i>;
@group(0)@binding(2)var<storage,read_write>result:array<u32>;
fn activityRecord(brick:u32)->u32{return 48u*brick;}
fn brickActive(brick:u32)->bool{return atomicLoad(&activity[activityRecord(brick)+10u])!=0u;}
fn cm12WorldLeafCoordinate(brick:u32)->vec3i{return geometry[brick].xyz;}
fn cm12WorldOwnerAt(q:vec3i)->u32{
  for(var brick=0u;brick<arrayLength(&geometry);brick++){
    let b=geometry[brick];if(all(q>=b.xyz)&&all(q<b.xyz+vec3i(b.w))){return brick;}
  }return INVALID;
}
fn cm12FluidNeighborReachable(q:vec3i,offset:vec3i)->bool{
  let brick=cm12WorldOwnerAt(q);let bit=u32(offset.x+1)+3u*u32(offset.y+1)+9u*u32(offset.z+1);
  return (atomicLoad(&activity[activityRecord(brick)+4u])&(1u<<bit))==0u;
}
${election}
${sweptMask}
@compute @workgroup_size(64)
fn sweep(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=27u){return;}
  let offset=vec3i(i32(gid.x%3u)-1,i32((gid.x/3u)%3u)-1,i32(gid.x/9u)-1);
  result[gid.x]=cm12SweptPageSupportMask(offset);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&result)){return;}
  let brick=gid.x/27u;let bit=gid.x%27u;
  if(!brickActive(brick)||bit==13u){return;}
  if((atomicLoad(&activity[activityRecord(brick)+3u])&(1u<<bit))==0u){return;}
  let offset=vec3i(i32(bit%3u)-1,i32((bit/3u)%3u)-1,i32(bit/9u)-1);
  let origin=cm12WorldLeafCoordinate(brick);let q=origin+offset;
  if(cm12WorldOwnerAt(q)!=INVALID||!cm12FluidNeighborReachable(origin,offset)){return;}
  if(sparseWorldFrontierAllocationOwner(brick,q)){result[gid.x]=brick+1u;}
}` });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const create = (size: number, usage: GPUBufferUsageFlags) => { const b = device!.createBuffer({ size, usage }); buffers.push(b); return b; };
    const a = create(activity.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const g = create(geometry.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const r = create(4 * 27 * bricks.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const rb = create(r.size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
    device.queue.writeBuffer(a, 0, activity); device.queue.writeBuffer(g, 0, geometry);
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [a,g,r].map((buffer,binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(27 * bricks.length / 64)); pass.end();
    encoder.copyBufferToBuffer(r, 0, rb, 0, r.size); device.queue.submit([encoder.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const results = new Uint32Array(rb.getMappedRange());
    const actual = new Map<string, number>();
    results.forEach((winner, index) => { if (!winner) return; const key = targets.get(index); assert.ok(key); assert.equal(actual.has(key), false, `duplicate insertion for ${key}`); actual.set(key, winner); });
    assert.deepEqual(actual, expected, "every reachable requested coordinate has exactly one caller");
    assert.ok(expected.size > 50); rb.unmap();
    const sweepPipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "sweep" } });
    const sweepGroup = device.createBindGroup({ layout: sweepPipeline.getBindGroupLayout(0), entries: [{ binding: 2, resource: { buffer: r } }] });
    const se = device.createCommandEncoder(), sp = se.beginComputePass();
    sp.setPipeline(sweepPipeline); sp.setBindGroup(0, sweepGroup); sp.dispatchWorkgroups(1); sp.end();
    se.copyBufferToBuffer(r, 0, rb, 0, r.size); device.queue.submit([se.finish()]); await rb.mapAsync(GPUMapMode.READ);
    const masks = new Uint32Array(rb.getMappedRange());
    for (let bit = 0; bit < 27; bit++) {
      const offset = [bit % 3 - 1, Math.floor(bit / 3) % 3 - 1, Math.floor(bit / 9) - 1]; let expectedMask = 0;
      for (let z = Math.min(0,offset[2]!); z <= Math.max(0,offset[2]!); z++)
        for (let y = Math.min(0,offset[1]!); y <= Math.max(0,offset[1]!); y++)
          for (let x = Math.min(0,offset[0]!); x <= Math.max(0,offset[0]!); x++)
            if (x || y || z) expectedMask |= 1 << (x+1+3*(y+1)+9*(z+1));
      assert.equal(masks[bit], expectedMask, `swept support for ${offset}`);
    }
    const maskBit = (q: number[]) => q[0]!+1+3*(q[1]!+1)+9*(q[2]!+1);
    for (const transform of [(q: number[]) => [-q[0]!,q[1]!,q[2]!],
      (q: number[]) => [q[2]!,q[0]!,q[1]!]]) {
      for (let bit = 0; bit < 27; bit++) {
        const q = [bit % 3 - 1, Math.floor(bit / 3) % 3 - 1, Math.floor(bit / 9) - 1];
        let transformedMask = 0;
        for (let other = 0; other < 27; other++) if (masks[bit]! & (1 << other)) {
          const delta = [other % 3 - 1, Math.floor(other / 3) % 3 - 1, Math.floor(other / 9) - 1];
          transformedMask |= 1 << maskBit(transform(delta));
        }
        assert.equal(masks[maskBit(transform(q))], transformedMask,
          "signed reflection and axis permutation must preserve swept support");
      }
    }
    // A z wall rejects both routes with +z, leaving the open +y receiver.
    assert.equal(masks[25]! & ~((1 << 22) | (1 << 25)), 1 << 16);
    rb.unmap();
  } finally { buffers.forEach(b => b.destroy()); device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
