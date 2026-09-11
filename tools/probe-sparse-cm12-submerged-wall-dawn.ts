import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const body = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(body, name);
  return body;
};

// Six incident faces around a rho=.475 cell. Exercise the actual production
// membership predicates; only topology/geometry accessors are fixture-backed.
const cases = [
  { name: "interior", walls: 0, expected: 1 },
  { name: "wall", walls: 1, expected: 1 },
  { name: "floor edge", walls: 2, expected: 1 },
  { name: "floor corner", walls: 3, expected: 1 },
  { name: "open air port", walls: 3, open: 1, expected: 0 },
  { name: "partially open port", walls: 3, open: .25, expected: 0 },
  { name: "separating ceiling", walls: 3, separating: 1, expected: 0 },
  { name: "previous air cell", walls: 3, previous: 0, expected: 0 },
  { name: "air neighbour", walls: 3, airNeighbour: 1, expected: 0 },
  { name: "fully enclosed without liquid neighbours", walls: 6, expected: 0 },
  { name: "empty row", walls: 3, empty: 1, expected: 0 },
  { name: "unaccepted air port", walls: 3, open: 1, unaccepted: 1, expected: 1 },
];

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "submerged pressure membership distinguishes walls, air ports and separating contact",
  async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "submerged-wall");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const module = device.createShaderModule({ code: `
const CM12_LIQUID_ISOVALUE=0.5;
struct Case{walls:f32,open:f32,separating:f32,previous:f32,air:f32,empty:f32,unaccepted:f32,pad:f32}
@group(0)@binding(0)var<storage,read>cases:array<Case>;
@group(0)@binding(1)var<storage,read_write>result:array<u32>;
var<private>q:Case;
fn pcmCellContains(id:u32)->bool{return select(q.air==0.0,q.previous>0.0,id==0u);}
fn incidenceBegin(id:u32)->u32{_=id;return 0u;}
fn incidenceEnd(id:u32)->u32{_=id;return 6u;}
fn incidenceRow(at:u32)->u32{return at;}
fn rowAccepted(row:u32)->bool{return !(row==0u&&q.unaccepted>0.0);}
fn rowTermRange(row:u32)->vec2u{
  var count=select(2u,1u,row<u32(q.walls));
  if(row==0u&&q.empty>0.0){count=0u;}
  return vec2u(2u*row,2u*row+count);
}
fn rowOpenFraction(row:u32)->f32{return select(1.0,select(0.0,q.open,row==0u),row<u32(q.walls));}
fn rowSeparatingFromClosedWorld(row:u32)->bool{return row==0u&&q.separating>0.0;}
fn termCell(term:u32)->u32{return select(0u,1u+term/2u,(term&1u)!=0u);}
fn cellActive(id:u32)->bool{_=id;return true;}
fn cellOpenVolume(id:u32)->f32{_=id;return 1.0;}
${production("pressureCellSubmerged")}
${production("pressureCellMembershipFromDensity")}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&cases)){return;}q=cases[gid.x];
  result[gid.x]=u32(pressureCellMembershipFromDensity(0u,0.475));
}
` });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const input = device.createBuffer({ size: cases.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const output = device.createBuffer({ size: cases.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        device.queue.writeBuffer(input, 0, new Float32Array(cases.flatMap(c => [
          c.walls, c.open ?? 0, c.separating ?? 0, c.previous ?? 1,
          c.airNeighbour ?? 0, c.empty ?? 0, c.unaccepted ?? 0, 0,
        ])));
        const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
        const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
          entries: [input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const actual = new Uint32Array(readback.getMappedRange());
        cases.forEach((c, i) => assert.equal(actual[i], c.expected, c.name));
        assert.deepEqual(errors, []);
      } finally {
        if (readback.mapState === "mapped") readback.unmap();
        readback.destroy(); input.destroy(); output.destroy();
      }
    } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
  });
