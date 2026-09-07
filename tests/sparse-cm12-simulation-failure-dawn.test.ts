import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { CM12_FAILURE_WORDS, decodeCM12SimulationFailure } from "../lib/methods/adaptive-mass/sparse-cm12-simulation-failure";
import { cm12SimulationFailureWGSL, guardCM12SimulationDispatches } from "../lib/methods/adaptive-mass/sparse-cm12-simulation-failure.wgsl";
import { createSparseCM12RowAccessWGSL, SPARSE_CM12_ATOMIC_ARENA_READERS } from "../lib/methods/adaptive-mass/sparse-cm12-row-access.wgsl";

const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("corrupt incidence latches provenance and halts later dispatches and frames", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "cm12-simulation-failure");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    assert.ok(device);
    const accessor = createSparseCM12RowAccessWGSL(SPARSE_CM12_ATOMIC_ARENA_READERS, true,
      "cm12RecordFailure(1u,cell,vec4u(begin,end,maximum,0u));");
    const bounded = accessor.match(/fn boundedIncidenceEnd\([\s\S]*?\n}/)![0];
    const deficitSupport = readFileSync(new URL(
      "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8")
      .match(/fn validateDensityDeficitSupport\([\s\S]*?\n}/)![0];
    const code = guardCM12SimulationDispatches(`
@group(0)@binding(0)var<storage,read_write>topologyArena:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>activity:array<atomic<u32>>;
struct Params { failure:vec4u }
@group(0)@binding(2)var<uniform>p:Params;
const BRICK_FINE_RESOLUTION=8u;
fn cm12FCCandidateGeneration()->u32{return 17u;}
${cm12SimulationFailureWGSL}
${bounded}
${deficitSupport}
@compute @workgroup_size(1) fn emptyDeficit(){
  validateDensityDeficitSupport(123u,0.0,0.5,bitcast<f32>(atomicLoad(&activity[0])));
}
@compute @workgroup_size(64) fn healthy(@builtin(global_invocation_id) gid:vec3u){
  if(gid.x==0u){atomicAdd(&topologyArena[0],1u);
    if(atomicLoad(&activity[0])==0u){cm12RecordFailure(4u,999u,vec4u(0u));}}
  workgroupBarrier();
}
@compute @workgroup_size(64) fn corrupt(@builtin(local_invocation_index) lane:u32){
  if(lane==0u){_=boundedIncidenceEnd(123u,900u,2u);}
  workgroupBarrier();
}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) gid:vec3u,){
  if(gid.x==0u){atomicAdd(&topologyArena[0],1u);cm12RecordFailure(4u,999u,vec4u(0u));}
  workgroupBarrier();
}`);
    const module = device.createShaderModule({ code });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const topology = device.createBuffer({ size: (CM12_FAILURE_WORDS + 1) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const activity = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(activity, 0, new Uint32Array([42]));
    const gate = device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const encoder = device.createCommandEncoder();
    for (const entryPoint of ["healthy", "corrupt", "publish", "corrupt", "publish"]) {
      encoder.copyBufferToBuffer(topology, 4, gate, 0, 4);
      const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
      const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: topology } }, { binding: 1, resource: { buffer: activity } },
        { binding: 2, resource: { buffer: gate } },
      ] });
      const pass = encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0, bindings);pass.dispatchWorkgroups(1);pass.end();
    }
    const readback = device.createBuffer({ size: topology.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(topology, 0, readback, 0, topology.size);
    device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(readback.getMappedRange()).slice();
    assert.equal(words[0], 1, "healthy work runs; publication after failure must not run");
    const failure = decodeCM12SimulationFailure(words.slice(1))!;
    assert.equal(failure.kernel, "corrupt");assert.equal(failure.frame, 42);
    assert.equal(failure.ownerId, 123);assert.equal(failure.rawWords[0], 1);
    assert.deepEqual(failure.operands, [900, 2, 384, 0]);
    readback.unmap();
    const deficitPipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "emptyDeficit" } });
    const deficitBindings = device.createBindGroup({ layout: deficitPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: topology } }, { binding: 1, resource: { buffer: activity } },
      { binding: 2, resource: { buffer: gate } },
    ] });
    device.queue.writeBuffer(gate, 0, new Uint32Array(4));
    // An empty extrapolated air trace is harmless. No epsilon may exempt a
    // donor with actual mass, even far below the solver's dry-cell threshold.
    for (const density of [0, 1e-12, 1, NaN]) {
      device.queue.writeBuffer(activity, 0, new Float32Array([density]));
      const check = device.createCommandEncoder();
      check.clearBuffer(topology);
      const pass = check.beginComputePass();
      pass.setPipeline(deficitPipeline);pass.setBindGroup(0, deficitBindings);pass.dispatchWorkgroups(1);pass.end();
      check.copyBufferToBuffer(topology, 0, readback, 0, topology.size);
      device.queue.submit([check.finish()]);await readback.mapAsync(GPUMapMode.READ);
      const receipt = decodeCM12SimulationFailure(new Uint32Array(readback.getMappedRange()).slice(1));
      if (density === 0) assert.equal(receipt, undefined);
      else {
        assert.equal(receipt?.code, "EMPTY_DEFICIT_STENCIL");
        assert.equal(receipt?.ownerId, 123);
        assert.deepEqual(receipt?.operandNames, ["visibleWeight", "deficit", "donorDensity", "reserved"]);
        assert.deepEqual(receipt?.operands, [0, 0.5, Math.fround(density), 0]);
      }
      readback.unmap();
    }
    readback.destroy();topology.destroy();activity.destroy();gate.destroy();
  } finally { device?.destroy();live.clear();await releaseWebGPUExclusiveLock(); }
});
