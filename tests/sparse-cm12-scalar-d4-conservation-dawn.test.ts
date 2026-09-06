import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(process.env.CM12_D4_SOURCE ?? new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = source.match(/fn preserveHorizontalD4\([\s\S]*?\n}/)?.[0];
assert.ok(production);
const live = new Set<GPU>();

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "scalar D4 preserves non-quantized symmetric fields and orbit mass", async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "scalar-d4-conservation");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice();
      assert.ok(device);
      // Odd extent includes reflection axes and the center, while x=z gives
      // duplicate members on diagonal orbits. Layer2 contains inactive support.
      const n = 9, count = n * n * 4;
      const rho = new Float32Array(count), gamma = new Float32Array(count);
      const active = (x: number, y: number, z: number) => y !== 2 || (x !== 4 && z !== 4);
      for (let z = 0; z < n; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < n; x++) {
        const id = x + n * (y + 4 * z), fx = Math.min(x, n - 1 - x), fz = Math.min(z, n - 1 - z);
        if (!active(x, y, z)) { gamma[id] = 1; continue; }
        // Layer0 is exactly D4 already, and contains representable density
        // smaller than half of the old1/65536 quantum. It must be unchanged.
        rho[id] = y === 0 ? (1 + Math.min(fx, fz) + .13 * Math.max(fx, fz)) * 1e-6
          : Math.fround(.1 + .71 * ((id * 17) % 53) / 53);
        gamma[id] = y === 0 ? 1 + (1 + fx + fz) * 1e-6
          : Math.fround(.5 + 1.7 * ((id * 29) % 37) / 37);
      }
      const code = `
const INVALID=0xffffffffu;
const CM12_SPARSE_TRANSPORT_FIXED=65536.0;
struct Params{dimensions:vec4u,stateOffsets5:vec4u}
const p=Params(vec4u(9,4,9,0),vec4u(${2 * count},${3 * count},0,0));
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
fn cellCenter(id:u32)->vec3f{return vec3f(f32(id%9u),f32((id/9u)%4u),f32(id/36u))+.5;}
fn cellActive(id:u32)->bool{let q=vec3u(cellCenter(id));return q.y!=2u||(q.x!=4u&&q.z!=4u);}
fn acceptedTemplateCellInvocation(id:u32)->u32{return select(INVALID,id,id<${count}u);}
fn ownerCellAt(q:vec3i)->u32{if(any(q<vec3i(0))||any(q>=vec3i(9,4,9))){return INVALID;}
 let id=u32(q.x+9*(q.y+4*q.z));return select(INVALID,id,cellActive(id));}
fn destinationDensity()->u32{return 0u;}
fn destinationGamma()->u32{return ${count}u;}
@compute @workgroup_size(64)
${production}
`;
      const shader = device.createShaderModule({ code });
      assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "preserveHorizontalD4" } });
      const buffer = device.createBuffer({ size: count * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const copy = device.createBuffer({ size: count * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      device.queue.writeBuffer(buffer, 0, rho); device.queue.writeBuffer(buffer, count * 4, gamma);
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] }));
      pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
      encoder.copyBufferToBuffer(buffer, count * 8, copy, 0, count * 8); device.queue.submit([encoder.finish()]);
      await copy.mapAsync(GPUMapMode.READ);
      const values = new Float32Array(copy.getMappedRange()).slice();
      for (let z = 0; z < n; z++) for (let y = 0; y < 4; y++) for (let x = 0; x < n; x++) {
        const id = x + n * (y + 4 * z);
        for (let field = 0; field < 2; field++) {
          const result = values[field * count + id]!;
          if (y === 0) assert.equal(result, [rho, gamma][field]![id], "an already symmetric scalar is an identity, including sub-quantum mass");
          for (const [tx, tz] of [[n - 1 - x, z], [x, n - 1 - z], [z, x]])
            assert.equal(result, values[field * count + tx! + n * (y + 4 * tz!)]!, "every D4 transform is bit-identical");
        }
      }
      for (let field = 0; field < 2; field++) {
        const before = [rho, gamma][field]!.reduce((sum, v) => sum + v, 0);
        const after = values.subarray(field * count, (field + 1) * count).reduce((sum, v) => sum + v, 0);
        assert.ok(Math.abs(after - before) < 2e-7 * before, `field${field} mass is conserved within float roundoff: ${before} -> ${after}`);
      }
      copy.unmap(); copy.destroy(); buffer.destroy();
    } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
  });
