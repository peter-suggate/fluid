import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createSparseCM12CurrentMapLayout, createSparseCM12CurrentMapWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl";
import { createSparseCM12CurrentMapVelocityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-current-map-velocity.wgsl";
import { createSparseCM12CurrentMapCompletionSpecs, createSparseCM12CurrentMapCompletionWGSL,
  instrumentSparseCM12CurrentMapCompletionWGSL, SPARSE_CM12_CURRENT_MAP_COMPLETION_WORDS,
  SPARSE_CM12_CURRENT_MAP_COMMIT_COMPLETION_SLOT,
  type SparseCM12CurrentMapCompletionSpec } from
  "../lib/methods/adaptive-mass/sparse-cm12-current-map-completion.wgsl";

test("completion specification covers every current-map transaction pass and repeated velocity sweeps", () => {
  const map = createSparseCM12CurrentMapLayout(128, [32,24,32]);
  const specs = createSparseCM12CurrentMapCompletionSpecs(map);
  assert.equal(specs.length, 17);
  assert.ok(specs.length <= SPARSE_CM12_CURRENT_MAP_COMPLETION_WORDS);
  assert.equal(specs.find(spec => spec.name === "compileRetainedDensityNativeIntegrals")!.expectedCompletions, "p.counts.x");
  assert.equal(specs.filter(spec => spec.name.startsWith("extendCurrentMapVelocity"))
    .reduce((sum, spec) => sum + (spec.expectedCompletions as number), 0), 64 * map.nodeCount);
  const stubs = specs.filter(spec => ["validateCurrentMapMaterialCoverage", "integrateCurrentMapFineMeasure",
    "validateCurrentMapCoverage", "compileRetainedDensityNativeIntegrals"].includes(spec.name))
    .map(spec => `@compute @workgroup_size(${spec.workgroupSize})\nfn ${spec.name}(@builtin(global_invocation_id)gid:vec3u){if(gid.x>0u){return;}}`).join("\n");
  const source = createSparseCM12CurrentMapWGSL(map) + createSparseCM12CurrentMapVelocityWGSL(map) + stubs;
  const instrumented = instrumentSparseCM12CurrentMapCompletionWGSL(source, specs);
  assert.equal((instrumented.match(/fn cm12CurrentMapCompletionWorker_/g) ?? []).length, 17);
  assert.match(instrumented, /cm12CompletionGid.x<p.counts.x/);
  assert.match(createSparseCM12CurrentMapCompletionWGSL(specs), /observed!=p.counts.x/);
  assert.throws(() => instrumentSparseCM12CurrentMapCompletionWGSL(source.replace("fn certifyCurrentMap(", "fn omittedCertificate("), specs), /Expected one.*certifyCurrentMap/);
});

test("worker extraction preserves early returns and ignores braces in nested comments", () => {
  const spec: SparseCM12CurrentMapCompletionSpec = { name: "proof", slot: 0,
    invocations: 1, dispatches: 1, workgroupSize: 1, expectedCompletions: 1 };
  const source = `@compute @workgroup_size(1)\nfn proof(){\n// }\n/* { /* } */ } */\nif(true){return;}\n}\nfn after()->u32{return 7u;}`;
  const instrumented = instrumentSparseCM12CurrentMapCompletionWGSL(source, [spec]);
  assert.match(instrumented, /fn cm12CurrentMapCompletionWorker_proof\(\)/);
  assert.match(instrumented, /if\(true\)\{return;\}/);
  assert.match(instrumented, /workgroupBarrier\(\)/);
  assert.ok(instrumented.endsWith("fn after()->u32{return 7u;}"));
  assert.throws(() => instrumentSparseCM12CurrentMapCompletionWGSL(source + source, [spec]), /found 2/);
});

test("cooperative measure retains its own uniform barriers and reports one completed support", () => {
  const map = createSparseCM12CurrentMapLayout(0, [4,3,4], 4);
  const specs = createSparseCM12CurrentMapCompletionSpecs(map, "p.counts.x", { cooperativeMeasure: true });
  const measure = specs.find(spec => spec.name === "integrateCurrentMapFineMeasureCooperative")!;
  assert.equal(measure.instrumentation, "manual"); assert.equal(measure.expectedCompletions, 48);
  const source = `@compute @workgroup_size(64)\nfn ${measure.name}(@builtin(workgroup_id)group:vec3u,@builtin(local_invocation_index)lane:u32){workgroupBarrier();if(lane==0u){cm12CurrentMapMeasureCompleted(group.x);}}`;
  assert.equal(instrumentSparseCM12CurrentMapCompletionWGSL(source, [measure]), source);
  assert.throws(() => instrumentSparseCM12CurrentMapCompletionWGSL("", [measure]), /Expected one/);
  const helper = createSparseCM12CurrentMapCompletionWGSL(specs);
  assert.match(helper, /fn cm12CurrentMapMeasureCompleted\(ordinal:u32\)/);
  assert.match(helper, /ordinal<48u/);
  assert.match(helper, /fn cm12CurrentMapCompletionValidPrefix\(limit:u32\)/);
});

test("cached ranges require a completed producer before the native-admission prefix", () => {
  const map = createSparseCM12CurrentMapLayout(0, [4,3,4], 4);
  const specs = createSparseCM12CurrentMapCompletionSpecs(map, "p.counts.x",
    { cooperativeMeasure: true, cachedMeasureRanges: true });
  const producer = specs.find(spec => spec.name === "compileCurrentMapFineMeasureRanges")!;
  const measure = specs.find(spec => spec.name === "integrateCurrentMapFineMeasureCooperative")!;
  const native = specs.find(spec => spec.name === "compileRetainedDensityNativeIntegrals")!;
  assert.equal(specs.length, 18); assert.equal(producer.invocations, 48);
  assert.equal(producer.expectedCompletions, 48); assert.equal(producer.workgroupSize, 64);
  assert.equal(producer.instrumentation, "worker"); assert.equal(producer.slot + 1, measure.slot);
  assert.ok(measure.slot < native.slot);
  assert.ok(specs.every(spec => spec.slot < SPARSE_CM12_CURRENT_MAP_COMMIT_COMPLETION_SLOT));
  const source = `@compute @workgroup_size(64)\nfn compileCurrentMapFineMeasureRanges(@builtin(global_invocation_id)gid:vec3u){if(gid.x>=48u){return;}}`;
  const instrumented = instrumentSparseCM12CurrentMapCompletionWGSL(source, [producer]);
  assert.match(instrumented, /fn cm12CurrentMapCompletionWorker_compileCurrentMapFineMeasureRanges/);
  assert.match(instrumented, /cm12CompletionGid.x<48u/);
  assert.throws(() => instrumentSparseCM12CurrentMapCompletionWGSL("", [producer]), /Expected one/);
  assert.throws(() => createSparseCM12CurrentMapCompletionWGSL([{ ...producer, slot: 31 }]), /Invalid/);
});

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();
(dawnModule ? test : test.skip)("GPU completion census rejects missing, partial, duplicate and stale work before commit",
  { timeout: 60_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", "current-map-completion");
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    const buffers: GPUBuffer[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const specs: readonly SparseCM12CurrentMapCompletionSpec[] = [
        { name: "work", slot: 0, invocations: "p.counts.x", dispatches: 1, workgroupSize: 64, expectedCompletions: "p.counts.x" },
        { name: "seal", slot: 1, invocations: 1, dispatches: 1, workgroupSize: 1, expectedCompletions: 1 },
      ];
      const module = device.createShaderModule({ code: createSparseCM12CurrentMapCompletionWGSL(specs)
        + instrumentSparseCM12CurrentMapCompletionWGSL(/* wgsl */ `
struct Parameters{counts:vec4u}
@group(0) @binding(0) var<storage,read_write> topologyArena:array<atomic<u32>>;
@group(0) @binding(1) var<uniform> p:Parameters;
fn cm12FailureBase()->u32{return 32u;}
fn cm12CurrentMapFailed()->bool{return atomicLoad(&topologyArena[32])!=0u;}
fn cm12RecordFailure(_reason:u32,owner:u32,operands:vec4u){
  atomicStore(&topologyArena[32],operands.x);atomicStore(&topologyArena[33],owner);
  atomicStore(&topologyArena[34],operands.z);atomicStore(&topologyArena[35],operands.w);
}
@compute @workgroup_size(64)
fn work(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=p.counts.x){return;}
  // Determining that this cell needs no write is completed work too.
  if(gid.x%2u==0u){return;}
  atomicStore(&topologyArena[64u+gid.x],gid.x+1u);
}
@compute @workgroup_size(1)
fn seal(){atomicStore(&topologyArena[49],1u);}
@compute @workgroup_size(1)
fn commit(){if(cm12CurrentMapCompletionValid()){atomicAdd(&topologyArena[48],1u);}}
@compute @workgroup_size(1)
fn preflight(){if(cm12CurrentMapCompletionValidPrefix(1u)){atomicStore(&topologyArena[50],1u);}}
`, specs) });
      assert.deepEqual((await module.getCompilationInfo()).messages.filter(message => message.type === "error")
        .map(message => `${message.lineNum}: ${message.message}`), []);
      const layout = device.createBindGroupLayout({ entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ] });
      const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
      const pipelines = await Promise.all(["work", "seal", "commit", "preflight"].map(entryPoint => device!.createComputePipelineAsync({
        layout: pipelineLayout, compute: { module, entryPoint } })));
      const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = device!.createBuffer({ size, usage }); buffers.push(buffer); return buffer;
      };
      const arena = allocate(4 * 160, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
      const parameters = allocate(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      const readback = allocate(4 * 160, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      device.queue.writeBuffer(parameters, 0, new Uint32Array([65,0,0,0]));
      const bindings = device.createBindGroup({ layout, entries: [arena, parameters]
        .map((buffer, binding) => ({ binding, resource: { buffer } })) });
      for (const [name, workgroups, repeats, expectedCount] of [
        ["complete", 2, 1, 65], ["partial after successful prior frame", 1, 1, 64],
        ["missing", 0, 0, 0], ["duplicate", 2, 2, 130], ["complete again", 2, 1, 65],
      ] as const) {
        const encoder = device.createCommandEncoder(); encoder.clearBuffer(arena);
        const pass = encoder.beginComputePass(); pass.setBindGroup(0, bindings);
        pass.setPipeline(pipelines[0]!);
        for (let iteration = 0; iteration < repeats; iteration++) pass.dispatchWorkgroups(workgroups);
        if (expectedCount === 65) { pass.setPipeline(pipelines[3]!); pass.dispatchWorkgroups(1); }
        pass.setPipeline(pipelines[1]!); pass.dispatchWorkgroups(1);
        pass.setPipeline(pipelines[2]!); pass.dispatchWorkgroups(1); pass.end();
        encoder.copyBufferToBuffer(arena, 0, readback, 0, 4 * 160);
        device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
        const result = new Uint32Array(readback.getMappedRange()).slice(); readback.unmap();
        assert.equal(result[0], expectedCount, name); assert.equal(result[1], 1, name);
        if (expectedCount === 65) {
          assert.equal(result[32], 0, name); assert.equal(result[48], 1, name);
          assert.equal(result[50], 1, "prefix admits complete prior work before the later seal runs");
          assert.equal(result[64 + 63], 64, "last meaningful output written before completion");
        } else {
          assert.deepEqual([...result.slice(32,36)], [124,0,expectedCount,65], name);
          assert.equal(result[48], 0, `${name}: accepted publication unchanged`);
        }
      }
      await device.queue.onSubmittedWorkDone();
    } finally {
      buffers.forEach(buffer => buffer.destroy()); device?.destroy(); if (gpu) live.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
