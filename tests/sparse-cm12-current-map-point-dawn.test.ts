import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12CurrentMapLayout, createSparseCM12CurrentMapWGSL } from "../lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();

(dawnModule ? test : test.skip)("point-only GPU map matches full point output through nonlinear accepted and candidate chains",
  { timeout: 60_000 }, async t => {
    await acquireWebGPUExclusiveLock("dawn-test", "current-map-point-parity");
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    const buffers: GPUBuffer[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const layout = createSparseCM12CurrentMapLayout(4, [4, 4, 4], 4, .5, 4);
      const points: number[][] = [];
      for (const bank of [0, 1]) {
        for (let i = 0; i < 257; i++) points.push([
          -5 + 14 * ((i * 67) % 257) / 256,
          -5 + 14 * ((i * 101) % 257) / 256,
          -5 + 14 * ((i * 191) % 257) / 256, bank,
        ]);
        // Knot, collar, and physical boundary values exercise identical fast paths.
        for (const x of [-4.00001, -4, -3.99999, -.5, 0, .5, 2, 4, 8, 8.00001]) points.push([x, .7, 2.1, bank]);
      }
      const module = device.createShaderModule({ code: /* wgsl */ `
struct Parameters{frame:vec4f}
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<storage,read> queries:array<vec4f>;
@group(0) @binding(2) var<storage,read_write> output:array<vec4f>;
const p:Parameters=Parameters(vec4f(0.0,1.0,0.0,0.0));
fn cm12RetainedDensityAcceptedBank()->u32{return 0u;}
fn cm12CurrentMapNativeVelocity(point:vec3f)->vec4f{return vec4f(0.0);}
fn cm12CurrentMapInitializeVelocity(id:u32,sample:vec4f){}
fn cm12CurrentMapVelocityBoundary(point:vec3f,velocity:vec3f)->vec3f{return velocity;}
fn cm12CurrentMapCoefficientBoundaryPoint(point:vec3f)->vec3f{return point;}
fn cm12CurrentMapCoefficientBoundaryValue(point:vec3f,value:vec3f)->vec3f{return value;}
fn cm12CurrentMapFail(code:u32,id:u32){}
fn cm12CurrentMapFailed()->bool{return false;}
${createSparseCM12CurrentMapWGSL(layout)}
@compute @workgroup_size(64)
fn query(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=arrayLength(&queries)){return;}
  let point=queries[gid.x].xyz;let bank=u32(queries[gid.x].w);
  let full=cm12CurrentMapEvaluate(point,bank);
  output[5u*gid.x]=vec4f(full.point,0.0);
  for(var axis=0u;axis<3u;axis++){output[5u*gid.x+axis+1u]=vec4f(full.jacobian[axis],0.0);}
  output[5u*gid.x+4u]=vec4f(cm12CurrentMapEvaluatePoint(point,bank),0.0);
}` });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(x => x.type === "error").map(x => `${x.lineNum}:${x.linePos} ${x.message}`), []);
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "query" } });
      const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = device!.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const state = allocate(4 * layout.endWords, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const queries = allocate(16 * points.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const output = allocate(80 * points.length, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const readback = allocate(output.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(queries, 0, new Float32Array(points.flat()));
      const values = new Float32Array(layout.endWords);
      const bases = [...layout.coefficientBaseWords, ...Array.from({ length: 4 }, (_, slot) => layout.chainBaseWords + 3 * layout.nodeCount * slot)];
      bases.forEach((base, generation) => {
        for (let id = 0; id < layout.nodeCount; id++) {
          const x = id % layout.nodeDimensions[0], y = Math.floor(id / layout.nodeDimensions[0]) % layout.nodeDimensions[1];
          const z = Math.floor(id / (layout.nodeDimensions[0] * layout.nodeDimensions[1]));
          for (let axis = 0; axis < 3; axis++) values[base + 3 * id + axis] = .08 * Math.sin(
            .37 * x + .21 * y - .19 * z + 1.17 * axis + .53 * generation);
        }
      });
      const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
        [state, queries, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      let totalDifferences = 0;
      for (const count of [0, 1, 4]) {
        values[layout.chainCountBaseWords] = count; device.queue.writeBuffer(state, 0, values);
        const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(points.length / 64)); pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, output.size); device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(readback.getMappedRange()).slice(); readback.unmap();
        const floats = new Float32Array(words.buffer);
        let differences = 0, maximumAbsolute = 0, maximumUlps = 0;
        points.forEach((point, i) => { for (let axis = 0; axis < 3; axis++) {
          if (words[20 * i + axis] !== words[20 * i + 16 + axis]) differences++;
          maximumAbsolute = Math.max(maximumAbsolute, Math.abs(floats[20 * i + axis]! - floats[20 * i + 16 + axis]!));
          const ordered = (word: number) => (word & 0x80000000) ? 0x80000000 - (word & 0x7fffffff) : 0x80000000 + word;
          maximumUlps = Math.max(maximumUlps, Math.abs(ordered(words[20 * i + axis]!) - ordered(words[20 * i + 16 + axis]!)));
        } });
        totalDifferences += differences;
        t.diagnostic(`chain=${count}, queries=${points.length}, differing point words=${differences}, maximum absolute=${maximumAbsolute}, maximum ULPs=${maximumUlps}`);
      }
      assert.equal(totalDifferences, 0, "GPU bitwise parity diagnostic");
      assert.deepEqual(errors, []);
    } finally {
      for (const buffer of buffers) buffer.destroy(); device?.destroy();
      if (gpu) live.delete(gpu); await releaseWebGPUExclusiveLock();
    }
  });
