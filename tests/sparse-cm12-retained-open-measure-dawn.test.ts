import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createSolidWorld } from "../lib/core/solid-world";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { createCM12ResourceRecorder } from "../lib/methods/adaptive-mass/sparse-cm12-resource-recipe";
import { retainedSceneDensity } from "../lib/methods/adaptive-mass/sparse-cm12-retained-scene-density";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import type { SparseCM12RetainedDensityResidentLayout } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";

// A permitted reciprocal-and-fuse lowering of 1-q8/255. All arithmetic inside
// Math.fround is exact binary64 here; this is one correctly rounded f32 FMA.
test("closed q8 reciprocal subtraction reproduces the negative conserved receipt", () => {
  const reciprocal = Math.fround(1 / 255);
  const closed = Math.fround(1 - 255 * reciprocal);
  assert.equal(closed, -5.9138983488082886e-8);
  assert.equal(Math.fround(.5829853415489197 * closed / 4), -8.619290170486238e-9,
    "two old closed supports in the eight-support native cell reproduce frame432 exactly");
  for (let q8 = 0; q8 <= 255; q8++) assert.ok(Math.fround((255 - q8) * reciprocal) >= 0);
  assert.equal(Math.fround((255 - 255) * reciprocal), 0);
});

const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "generated retained initialization preserves compiled moments and q8 restrictions stay nonnegative", async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "retained-open-measure");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
      const adapter = await gpu.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice(); assert.ok(device);
      const recorder = createCM12ResourceRecorder(device.limits);
      const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [{
        key: 0, coordinate: [0, 0, 0], resolution: 1,
        density: new Float64Array([0]), gamma: new Float64Array([1]),
      }], 1, 8);
      const field = retainedSceneDensity({ generation: 1, transitionWidth: .125,
        domain: { lower: [0, 0, 0], upper: [1, 1, 1] },
        primitives: [{ kind: "quadratic-height", center: [0, .3, 0], curvature: [0, 0, 0] }] });
      const resident = await WebGPUSparseCM12Resident.create(recorder.device, atlas,
        buildSparseAtlasCompositeGrid(atlas), .125, createSolidWorld(), new Set([0]),
        undefined, undefined, 8, () => {}, 0, undefined, field);
      await resident.waitForSimulationPipelines();
      const data = resident as unknown as { retainedDensityLayout: SparseCM12RetainedDensityResidentLayout;
        retainedDensitySupportCount: number; state: GPUBuffer };
      const support = data.retainedDensityLayout.support!;
      const modules = recorder.finish({}).operations.filter(op => op.method === "createShaderModule")
        .map(op => (op.args[0] as GPUShaderModuleDescriptor).code);
      const actualFunction = (name: string) => {
        const expression = new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`);
        const match = modules.map(code => code.match(expression)).find(Boolean);
        assert.ok(match, `generated production function ${name}`); return match[0];
      };
      const count = 512, resultCount = 1792;
      const code = `
@group(0) @binding(0) var<storage,read_write> state:array<f32>;
@group(0) @binding(1) var<storage,read> input:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
var<private> cm12FailureKernel:u32;
struct Params { failure:vec4u }
const p=Params(vec4u(0u));
const CM12_RETAINED_SUPPORT_COUNT=${data.retainedDensitySupportCount}u;
const CM12_RETAINED_DENSE_SUPPORT_COUNT=${count}u;
const CM12_RETAINED_OPEN_BASE=${support.openFractionBaseWords}u;
fn cm12RetainedDensityEnabled()->bool{return true;}
fn cm12SolidVoxelFractionQ8(q:vec3i)->u32{return input[u32(q.x)/2u];}
fn cellCenter(cell:u32)->vec3f{return vec3f(2.0*f32(cell)+1.0,1.0,1.0);}
fn cellWidths(_cell:u32)->vec3f{return vec3f(2.0);}
fn solidVoxelCellOpenOffset()->u32{return 0u;}
${actualFunction("cm12RetainedDensityCoefficientBase")}
${actualFunction("cm12RetainedDensitySupportOpen")}
@compute @workgroup_size(64)
${actualFunction("initializeRetainedDensityOpenSupport")}
${actualFunction("refreshSparseCM12SolidWorldCell")}
@compute @workgroup_size(64)
fn inspect(@builtin(global_invocation_id)gid:vec3u){
  let i=gid.x;
  if(i<${count}u){
    output[i]=state[CM12_RETAINED_OPEN_BASE+i];
    output[${count}u+i]=state[${support.seedMeanBaseWords}u+i];
  }
  if(i>=256u){return;}
  refreshSparseCM12SolidWorldCell(i);
  output[1024u+i]=state[i];
  output[1280u+i]=cm12RetainedDensitySupportOpen(${count}u+i,vec3i(i32(2u*i),0,0));
  output[1536u+i]=fma(-f32(input[i]),bitcast<f32>(input[256]),1.0);
}
`;
      const module = device.createShaderModule({ code });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error"), []);
      const layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      const pipeline = (entryPoint: string) => device!.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } });
      const initialize = pipeline("initializeRetainedDensityOpenSupport"), inspect = pipeline("inspect");
      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
      const state = device.createBuffer({ size: data.state.size, usage: storage });
      const input = device.createBuffer({ size: 257 * 4, usage: storage });
      const output = device.createBuffer({ size: resultCount * 4, usage: storage });
      const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const q8 = Uint32Array.from({ length: 257 }, (_, i) => i % 256);
      q8[256] = new Uint32Array(new Float32Array([1 / 255]).buffer)[0]!;
      const open = Float32Array.from({ length: count }, (_, i) => (255 - i % 256) / 255);
      const seed = Float32Array.from(open, value => .3 * value);
      device.queue.writeBuffer(input, 0, q8);
      device.queue.writeBuffer(state, 4 * support.openFractionBaseWords, open);
      device.queue.writeBuffer(state, 4 * support.seedMeanBaseWords, seed);
      const binding = device.createBindGroup({ layout, entries: [
        { binding: 0, resource: { buffer: state } }, { binding: 1, resource: { buffer: input } },
        { binding: 2, resource: { buffer: output } },
      ] });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(); pass.setBindGroup(0, binding);
      pass.setPipeline(initialize); pass.dispatchWorkgroups(Math.ceil(data.retainedDensitySupportCount / 64));
      pass.setPipeline(inspect); pass.dispatchWorkgroups(count / 64); pass.end();
      encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange()).slice(); readback.unmap();
      assert.deepEqual(result.subarray(0, count), open, "compiled open moments survive initialization word for word");
      assert.deepEqual(result.subarray(count, 2 * count), seed, "compiled clipped seed integrals survive initialization");
      for (let i = 0; i < 256; i++) {
        for (const offset of [1024, 1280]) {
          assert.ok(result[offset + i]! >= 0, `q8=${i}: open restriction is nonnegative`);
          assert.ok(Math.abs(result[offset + i]! - (255 - i) / 255) < 2e-7);
        }
      }
      assert.equal(result[1279], 0, "fully closed native cell has exact zero capacity");
      assert.equal(result[1535], 0, "fully closed dynamic support has exact zero capacity");
      assert.ok(result[1791]! < 0, "explicit reciprocal/FMA counterexample remains visible; no predicate suppresses it");
      assert.equal(Math.fround(result[1791]! * .5829853415489197 / 4), -8.619290170486238e-9);
      resident.destroy(); readback.destroy(); output.destroy(); input.destroy(); state.destroy();
    } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
  });
