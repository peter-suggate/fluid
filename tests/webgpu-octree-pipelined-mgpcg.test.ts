import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_PIPELINED_PCG_CONTROL_BYTES,
  OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION,
  OCTREE_PIPELINED_PCG_INDIRECT_STRIDE_BYTES,
  OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID,
  OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS,
  OCTREE_PIPELINED_MGPCG_MAX_PRODUCTION_STORAGE_BINDINGS,
  WebGPUOctreePipelinedMGPCG,
  addCompensatedF32,
  compensatedF32Value,
  type CompensatedF32,
  mergeCompensatedF32,
  octreeCompensatedF32WGSL,
  octreePipelinedMGPCGActivityShader,
  octreePipelinedMGPCGShader,
  planOctreePipelinedMGPCG,
} from "../lib/webgpu-octree-pipelined-mgpcg";
import {
  createGPULogicalActivityAdoptionContext,
  gpuLogicalActivityPipelineRegistration,
  gpuLogicalActivityTaskDescriptions,
} from "../lib/gpu-logical-activity-adoption";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { decodeOctreePressureSolveWork } from "../lib/webgpu-octree-work-accounting";
import { usePerformanceInstrumentationStore } from "../lib/stores/performance-instrumentation-store";

test("pipelined planner exposes residual and direct-curvature reductions", () => {
  const plan = planOctreePipelinedMGPCG({ rowCapacity: 6_912, maximumIterations: 10 });
  assert.deepEqual(plan.rowDispatch, [108, 1, 1]);
  assert.deepEqual(plan.reductionDispatch, [54, 1, 1]);
  assert.equal(plan.reductionLanes, 128);
  assert.equal(plan.hardIterationCeiling, 16,
    "the non-encoded hard ceiling remains diagnostic metadata");
  assert.equal(plan.initialReductionCount, 1);
  assert.equal(plan.reductionsPerOuterIteration, 2);
  assert.equal(plan.maximumReductionCount, 20);
  assert.equal(plan.mergedScalarsPerReduction, 3);
  assert.equal(plan.dispatchRecordCount,
    10 * OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION);
  assert.equal(plan.indirectBytes,
    plan.dispatchRecordCount * OCTREE_PIPELINED_PCG_INDIRECT_STRIDE_BYTES);
  assert.equal(plan.ownedBytes, OCTREE_PIPELINED_PCG_CONTROL_BYTES + 64
    + plan.indirectBytes + plan.reductionPartialBytes);
  assert.throws(() => planOctreePipelinedMGPCG({ rowCapacity: 0 }), /positive integer/);
  assert.throws(() => planOctreePipelinedMGPCG({ rowCapacity: 1, maximumIterations: 3 }),
    /remain in \[4,10\]/);
  assert.throws(() => planOctreePipelinedMGPCG({ rowCapacity: 1, maximumIterations: 11 }),
    /remain in \[4,10\]/);
  assert.throws(() => planOctreePipelinedMGPCG({ rowCapacity: 1,
    maximumIterations: 4, hardIterationCeiling: 15 }), /fixed at 16/);
});

test("pipelined solve row count comes only from the accepted structured authority", () => {
  assert.match(octreePipelinedMGPCGShader,
    /fn rows\(\) -> u32 \{[^}]*acceptedAuthority\[2\][^}]*\}/);
  assert.doesNotMatch(octreePipelinedMGPCGShader,
    /acceptedAuthority\[2\] != rowCount\[0\]|rowCount\[0\] == 0u/,
    "a cleared next-candidate compaction buffer must not invalidate the accepted solve");
});

test("compensated f32 helpers retain low cancellation terms", () => {
  let compensated: CompensatedF32 = [0, 0];
  let naive = Math.fround(0);
  for (const value of [1e8, 1, 1, -1e8]) {
    compensated = addCompensatedF32(compensated, value);
    naive = Math.fround(naive + Math.fround(value));
  }
  assert.equal(naive, 0);
  assert.equal(compensatedF32Value(compensated), 2);
  assert.equal(compensatedF32Value(mergeCompensatedF32(
    addCompensatedF32([0, 0], 1e8),
    addCompensatedF32(addCompensatedF32([0, 0], -1e8), 3),
  )), 3);
});

test("WGSL implements direct curvature, compensated scalars, and GPU tail zeroing", () => {
  assert.match(octreeCompensatedF32WGSL,
    /struct CompensatedF32 \{ hi: f32, lo: f32 \}/);
  assert.match(octreePipelinedMGPCGShader,
    /local\.gamma = addCompensatedF32\(local\.gamma, r \* u\)/);
  assert.match(octreePipelinedMGPCGShader,
    /local\.delta = addCompensatedF32\(local\.delta, u \* w\)/);
  assert.match(octreePipelinedMGPCGShader,
    /local\.rr = addCompensatedF32\(local\.rr, r \* r\)/);
  assert.doesNotMatch(octreePipelinedMGPCGShader,
    /delta - beta \* gamma \/ previousAlpha/);
  assert.match(octreePipelinedMGPCGShader,
    /fn reduceDirectionCurvaturePartials[\s\S]*d \* image/);
  assert.match(octreePipelinedMGPCGShader,
    /fn finishDirectionCurvature[\s\S]*atomicAdd\(&control\[3\], 1u\)[\s\S]*let direct = compensatedValue\(curvature\)/);
  assert.match(octreePipelinedMGPCGShader,
    /fn zeroRemainingAfterUpdate[\s\S]*zeroDispatch\(iteration, 3u\)[\s\S]*zeroDispatch\(future, 0u\)[\s\S]*zeroDispatch\(future, 1u\)/);
  assert.match(octreePipelinedMGPCGShader,
    /pressureOut\[row\] = select\(seed, candidate, success && finite\(candidate\)\)/);
  assert.match(octreePipelinedMGPCGShader,
    /fn diagonalAt\(row: u32\)[\s\S]*section63Coefficients\[\(acceptedBank\(\) \* capacity\(\) \+ row\) \* 19u\]/);
  assert.match(octreePipelinedMGPCGShader, /let value = -rhs\[row\] - directionImage\[row\]/);
  assert.doesNotMatch(octreePipelinedMGPCGShader, /LeafHeader|headers\[/,
    "the outer solve must consume the Section 6.3 diagonal channel and RHS authorities directly");
  assert.doesNotMatch(octreePipelinedMGPCGShader, /\bf64\b|fallback|legacy/i);
});

test("sole target shader is 128-lane subgroup f32 and fail-closed", () => {
  assert.match(octreePipelinedMGPCGShader, /enable subgroups;/);
  assert.match(octreePipelinedMGPCGShader,
    /@compute @workgroup_size\(128\)\nfn reduceMergedPartials/);
  assert.match(octreePipelinedMGPCGShader,
    /@compute @workgroup_size\(128\)\nfn finishMergedReduction/);
  // docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md Part D item 4: the reduction is an
  // explicit width-halving shared-memory tree in BOTH the hierarchical and the
  // persistent executor, because `subgroupAdd`'s association order is
  // implementation-defined and a pinned tree is what makes the Gate A
  // bit-exact comparison applicable across the executor selection.
  assert.doesNotMatch(octreePipelinedMGPCGShader, /subgroupAdd\(/);
  assert.match(octreePipelinedMGPCGShader,
    /merged\[lane\] = local;\n\s*for \(var width = REDUCTION_LANES \/ 2u; width > 0u; width >>= 1u\) \{\n\s*workgroupBarrier\(\);\n\s*if \(lane < width\) \{\n\s*merged\[lane\] = mergeScalars\(merged\[lane\], merged\[lane \+ width\]\);/);
  assert.match(octreePipelinedMGPCGShader, /fn zeroRemainingAfterUpdate\(iteration: u32\)/);
  assert.doesNotMatch(octreePipelinedMGPCGShader, /\bf16\b|portable|fallback|legacy/i);
});

test("MGPCG activity variant is conditional, exhaustive, bounded, and storage-safe", () => {
  const disabled = createGPULogicalActivityAdoptionContext({
    moduleId: OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID,
    profile: { enabled: false, generation: 4 },
    identity: "subgroup",
    subgroupsAlreadyEnabled: true,
  });
  const disabledSource = octreePipelinedMGPCGActivityShader(disabled);
  assert.equal(disabledSource, octreePipelinedMGPCGShader);
  assert.equal(disabled.module(disabledSource, "mgpcg-test").code, octreePipelinedMGPCGShader);

  const enabled = createGPULogicalActivityAdoptionContext({
    moduleId: OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID,
    profile: { enabled: true, generation: 5 },
    identity: "subgroup",
    subgroupsAlreadyEnabled: true,
  });
  const activitySource = octreePipelinedMGPCGActivityShader(enabled);
  const descriptors = Object.values(OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS);
  assert.equal(descriptors.length, 12);
  assert.equal((activitySource.match(/fluidGpuLogicalActivitySubgroup\(/g) ?? []).length,
    descriptors.length,
  "every compute entry point has one uniform active-lane ballot at entry");
  assert.ok((activitySource.match(/fluidGpuLogicalActivityWorkgroup\(/g) ?? []).length
    >= descriptors.length,
  "every compute entry point retains scalar all-control-path exit checkpoints");
  assert.equal((activitySource.match(/@builtin\(num_workgroups\) activityNumWorkgroups/g) ?? []).length,
    descriptors.length, "every activity entry point exposes its complete dispatch population");
  assert.doesNotMatch(activitySource, /activityWorkgroupId\.x < 16u/,
    "the shader must not retain prefix-only workgroup sampling");
  assert.match(activitySource, /< rows\(\) && !stopped\(\)/,
    "recurring row work records only meaningful live dispatch groups");
  assert.match(activitySource,
    /activityWorkgroupId\.x \* 64u \+ activityLocalInvocationIndex < rows\(\)/,
    "row kernels ballot the actual active invocation population, including partial tails");
  assert.match(activitySource, /subgroupLane, subgroupSize/,
    "existing subgroup builtins are reused by reduction entry points");
  for (const descriptor of descriptors) {
    assert.equal(descriptor.taskId, enabled.taskId(descriptor.task));
    assert.equal(descriptor.checkpoints.enter, enabled.checkpointId(descriptor.task, "enter"));
    assert.equal(descriptor.checkpoints.exit, enabled.checkpointId(descriptor.task, "exit"));
    assert.match(activitySource, new RegExp(`${descriptor.taskId}u`));
  }
  const variant = enabled.module(activitySource, "mgpcg-test");
  assert.equal((variant.code.match(/enable subgroups;/g) ?? []).length, 1);
  assert.match(variant.code, /subgroupBallot\(activePredicate\)/);
  assert.match(variant.code, /@builtin\(subgroup_invocation_id\) activitySubgroupLane/);
  assert.match(variant.code, /@group\(3\) @binding\(0\)/);
  assert.equal(OCTREE_PIPELINED_MGPCG_MAX_PRODUCTION_STORAGE_BINDINGS, 7);
  assert.ok(OCTREE_PIPELINED_MGPCG_MAX_PRODUCTION_STORAGE_BINDINGS + 1 <= 10,
    "the activity buffer must keep every MGPCG entry point within the storage limit");
});

test("construction rejects an unproven preconditioner before allocating GPU state", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  let allocations = 0;
  const buffer = (size: number) => ({ size }) as GPUBuffer;
  const device = {
    createBuffer() { allocations += 1; return buffer(4); },
  } as unknown as GPUDevice;
  assert.throws(() => new WebGPUOctreePipelinedMGPCG(device, {
    coefficients: buffer(152), rhs: buffer(4), rowCount: buffer(4), rowDispatch: buffer(12),
    acceptedAuthority: buffer(24),
    operator: { convergenceTail: "gpu-zero-indirect", encodedDispatchCount: 1, encode() {} },
    vectors: {
      pressure: buffer(4), residual: buffer(4), preconditioned: buffer(4),
      preconditionedImage: buffer(4), direction: buffer(4), directionImage: buffer(4),
    },
    preconditioner: {
      operatorOrder: 1, isSymmetricPositiveDefinite: false,
      allocatedBytes: 0, encodedCorrectionDispatchCount: 1,
      encodeSetup() {}, encodeCorrection() {},
    } as never,
  }, { rowCapacity: 1 }), /fixed-schedule SPD/);
  assert.equal(allocations, 0);
});

test("construction rejects a device without subgroups instead of compiling a fallback", () => {
  const buffer = (size: number) => ({ size }) as GPUBuffer;
  let allocations = 0;
  const device = {
    features: new Set(),
    limits: {
      maxComputeInvocationsPerWorkgroup: 1024,
      maxComputeWorkgroupStorageSize: 32 * 1024,
    },
    createBuffer() { allocations += 1; return buffer(4); },
  } as unknown as GPUDevice;
  assert.throws(() => new WebGPUOctreePipelinedMGPCG(device, {
    coefficients: buffer(152), rhs: buffer(4), rowCount: buffer(4), rowDispatch: buffer(12),
    acceptedAuthority: buffer(24),
    operator: { convergenceTail: "gpu-zero-indirect", encodedDispatchCount: 1, encode() {} },
    vectors: {
      pressure: buffer(4), residual: buffer(4), preconditioned: buffer(4),
      preconditionedImage: buffer(4), direction: buffer(4), directionImage: buffer(4),
    },
    preconditioner: {
      operatorOrder: 1, isSymmetricPositiveDefinite: true,
      convergenceTail: "gpu-zero-indirect",
      allocatedBytes: 0, encodedCorrectionDispatchCount: 1,
      encodeSetup() {}, encodeCorrection() {},
    },
  }, { rowCapacity: 1 }), /requires the target subgroup feature/);
  assert.equal(allocations, 0);
});

test("orchestrator encodes direct direction curvature after every live update", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  type MockBuffer = GPUBuffer & { label: string; destroyed?: boolean };
  const buffers: MockBuffer[] = [];
  const pipelines: string[] = [];
  const createdPipelines: GPUComputePipeline[] = [];
  const modules: string[] = [];
  const stageBindings = new Map<string, number[]>();
  const buffer = (size: number, label = "external") =>
    ({ size, label, destroy() { this.destroyed = true; } }) as MockBuffer;
  const device = {
    features: new Set(["subgroups"]),
    limits: {
      maxComputeInvocationsPerWorkgroup: 1024,
      maxComputeWorkgroupStorageSize: 32 * 1024,
    },
    queue: { writeBuffer() {} },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const result = buffer(Number(descriptor.size), String(descriptor.label));
      buffers.push(result);
      return result;
    },
    createShaderModule(descriptor: GPUShaderModuleDescriptor) {
      modules.push(String(descriptor.label));
      return { label: descriptor.label };
    },
    createComputePipeline(descriptor: GPUComputePipelineDescriptor) {
      const entryPoint = String(descriptor.compute.entryPoint);
      const pipeline = { entryPoint, getBindGroupLayout() { return {}; } } as unknown as GPUComputePipeline;
      createdPipelines.push(pipeline);
      return pipeline;
    },
    createBindGroup(descriptor: GPUBindGroupDescriptor) {
      stageBindings.set(
        String(descriptor.label),
        Array.from(descriptor.entries, (entry) => entry.binding),
      );
      return {};
    },
  } as unknown as GPUDevice;
  let corrections = 0, setups = 0;
  const source = {
    coefficients: buffer(2 * 19 * 4 * 64), rhs: buffer(4 * 64),
    rowCount: buffer(4), rowDispatch: buffer(12),
    acceptedAuthority: buffer(24),
    operator: {
      convergenceTail: "gpu-zero-indirect" as const,
      encodedDispatchCount: 4,
      encode() { pipelines.push("resolved-row-operator"); },
    },
    vectors: {
      pressure: buffer(4 * 64), residual: buffer(4 * 64),
      preconditioned: buffer(4 * 64), preconditionedImage: buffer(4 * 64),
      direction: buffer(4 * 64), directionImage: buffer(4 * 64),
    },
    preconditioner: {
      operatorOrder: 1 as const,
      isSymmetricPositiveDefinite: true as const,
      convergenceTail: "gpu-zero-indirect" as const,
      allocatedBytes: 0,
      encodedSetupDispatchCount: 1,
      encodedCorrectionDispatchCount: 2,
      encodeSetup() { setups += 1; },
      encodeCorrection() { corrections += 1; },
    },
  };
  usePerformanceInstrumentationStore.getState().setMode("activity");
  const solver = new WebGPUOctreePipelinedMGPCG(
    device, source, { rowCapacity: 64, maximumIterations: 4 },
  );
  usePerformanceInstrumentationStore.getState().setMode("off");
  assert.equal(createdPipelines.length, Object.keys(OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS).length);
  assert.ok(createdPipelines.every((pipeline) =>
    gpuLogicalActivityPipelineRegistration(pipeline)?.moduleId
      === OCTREE_PIPELINED_MGPCG_ACTIVITY_MODULE_ID),
  "every MGPCG compute pipeline must be registered to the activity binding context");
  const generation = gpuLogicalActivityPipelineRegistration(createdPipelines[0])!.generation;
  const taskDescriptions = gpuLogicalActivityTaskDescriptions(generation);
  for (const descriptor of Object.values(OCTREE_PIPELINED_MGPCG_ACTIVITY_TASKS)) {
    assert.deepEqual(taskDescriptions[descriptor.taskId], {
      id: descriptor.id,
      label: descriptor.label,
      phaseId: "pressure-solve",
      checkpoints: descriptor.checkpoints,
    });
  }
  let current = "";
  let passIndex = 0;
  const pipelinePasses: Array<{ name: string; pass: number }> = [];
  const indirectOffsets: number[] = [];
  const encoder = {
    clearBuffer() {},
    beginComputePass() {
      const openedPass = ++passIndex;
      return {
        setPipeline(pipeline: { entryPoint: string }) {
          current = pipeline.entryPoint;
          pipelines.push(current);
          pipelinePasses.push({ name: current, pass: openedPass });
        },
        setBindGroup() {},
        dispatchWorkgroups() {},
        dispatchWorkgroupsIndirect(_source: GPUBuffer, offset: number) {
          indirectOffsets.push(offset);
        },
        end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  solver.encode(new PassBroker(encoder), {
    pressureSeed: buffer(4 * 64),
    pressureOut: buffer(4 * 64),
  });

  assert.equal(setups, 1);
  assert.equal(corrections, 5, "initial M plus one fixed M application per outer iteration");
  assert.equal(pipelines.filter((name) => name === "finishMergedReduction").length, 5,
    "initialization plus each outer iteration has one residual reduction finish");
  assert.equal(pipelines.filter((name) => name === "finishDirectionCurvature").length, 4);
  assert.equal(pipelines.filter((name) => name === "resolved-row-operator").length, 6,
    "one initial A(x) plus A(Mr) for the initial and every outer recurrence");
  assert.equal(pipelines.filter((name) => name === "reduceMergedPartials").length, 5);
  assert.equal(pipelines.filter((name) => name === "reduceDirectionCurvaturePartials").length, 4);
  const initialReduction = pipelinePasses.findIndex(({ name }) => name === "reduceMergedPartials");
  const initialDirections = pipelinePasses.findIndex(({ name }) => name === "initializeDirections");
  const initialFinish = pipelinePasses.findIndex(({ name }) => name === "finishMergedReduction");
  assert.ok(initialReduction >= 0 && initialFinish > initialReduction && initialDirections > initialFinish);
  assert.notEqual(pipelinePasses[initialReduction].pass, pipelinePasses[initialFinish].pass,
    "the global reduction finish must begin after the partials boundary");
  assert.notEqual(pipelinePasses[initialFinish].pass, pipelinePasses[initialDirections].pass,
    "the GPU-zeroed initial direction tail must be consumed in a later pass");
  assert.deepEqual(indirectOffsets, [
    0, 0, 32, 48,
    0, 32, 48, 16,
    64, 96, 112, 80,
    128, 160, 176, 144,
    192, 224, 240, 208,
    0,
  ]);
  assert.equal(solver.reductionsPerOuterIteration, 2);
  assert.deepEqual(modules, [
    "Octree pipelined MGPCG · M1 Max 128-lane subgroups",
  ], "only the measured target module is compiled");
  assert.deepEqual(Object.fromEntries(stageBindings), {
    "Pipelined MGPCG · initializeControlAndDispatch": [0, 2, 13, 15],
    "Pipelined MGPCG · validateAuthority": [0, 2, 13],
    "Pipelined MGPCG · initializeState": [0, 1, 2, 5, 7, 8, 13, 16],
    "Pipelined MGPCG · formInitialResidual": [0, 2, 8, 12, 13, 16],
    "Pipelined MGPCG · reduceMergedPartials": [0, 2, 8, 9, 10, 13, 14, 16],
    "Pipelined MGPCG · finishMergedReduction": [0, 2, 13, 14, 15],
    "Pipelined MGPCG · reduceDirectionCurvaturePartials": [0, 2, 11, 12, 13, 14],
    "Pipelined MGPCG · finishDirectionCurvature": [0, 2, 13, 14, 15],
    "Pipelined MGPCG · initializeDirections": [0, 2, 9, 10, 11, 12, 13],
    "Pipelined MGPCG · advancePCGState": [0, 2, 7, 8, 11, 12, 13],
    "Pipelined MGPCG · updateDirections": [0, 2, 9, 11, 13],
    "Pipelined MGPCG · finalizeAndPublish": [0, 2, 5, 6, 7, 13],
  }, "every stage bind group must contain exactly its statically reachable shader bindings");
  assert.ok(buffers.some((candidate) =>
    candidate.label === "Pipelined MGPCG GPU-zeroed outer tail"));
  solver.destroy();
  assert.ok(buffers.every((candidate) => candidate.destroyed));
});

test("Dawn executes a manufactured one-row solve through direct-curvature PCG", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU recurrence validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  assert.equal(adapter.features.has("subgroups"), true,
    "the M1 Max production target must expose subgroups");
  const device = await adapter.requestDevice({
    requiredFeatures: ["subgroups"],
  });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const make = (size: number, data?: GPUAllowSharedBufferSource, usage = storage) => {
    const buffer = device.createBuffer({ size, usage });
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const section63Data = new Float32Array(2 * 19);
  section63Data[0] = 2;
  section63Data[19] = 2;
  const section63Coefficients = make(section63Data.byteLength, section63Data);
  const rhs = make(4, new Float32Array([-2]));
  const rowCount = make(4, new Uint32Array([1]));
  const acceptedAuthority = make(
    24,
    new Uint32Array([0, 0xffff_ffff, 1, 1, 1, 1]),
  );
  const vector = () => make(4);
  const vectors = {
    pressure: vector(), residual: vector(), preconditioned: vector(),
    preconditionedImage: vector(), direction: vector(), directionImage: vector(),
  };
  const copyModule = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<storage,read> control:array<u32>;
@compute @workgroup_size(1) fn copyResidual() {
  if(control[0]==0u&&control[1]==0u){output[0]=input[0];}
}
@compute @workgroup_size(1) fn applyOperator() {
  if(control[0]==0u&&control[1]==0u){output[0]=2.0*input[0];}
}` });
  const copyPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: copyModule, entryPoint: "copyResidual" },
  });
  const operatorPipeline = device.createComputePipeline({
    layout: "auto", compute: { module: copyModule, entryPoint: "applyOperator" },
  });
  const encodePipeline = (
    pipeline: GPUComputePipeline,
    broker: PassBroker,
    input: GPUBuffer,
    output: GPUBuffer,
    solverControl: GPUBuffer,
  ) => {
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: input } },
        { binding: 1, resource: { buffer: output } },
        { binding: 2, resource: { buffer: solverControl } },
      ],
    });
    const pass = broker.compute();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
  };
  const preconditioner = {
    operatorOrder: 1 as const,
    isSymmetricPositiveDefinite: true as const,
    convergenceTail: "gpu-zero-indirect" as const,
    allocatedBytes: 0,
    encodedCorrectionDispatchCount: 1,
    encodeSetup() {},
    encodeCorrection(broker: PassBroker, input: {
      rhs: GPUBuffer; correction: GPUBuffer; solverControl: GPUBuffer;
    }) {
      encodePipeline(
        copyPipeline, broker, input.rhs, input.correction, input.solverControl,
      );
    },
  };
  for (const _target of [0]) {
    device.pushErrorScope("validation");
    const solver = new WebGPUOctreePipelinedMGPCG(device, {
      coefficients: section63Coefficients,
      rhs,
      rowCount,
      rowDispatch: make(12, new Uint32Array([1, 1, 1]),
        GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST),
      acceptedAuthority,
      operator: {
        convergenceTail: "gpu-zero-indirect" as const,
        encodedDispatchCount: 1,
        encode(broker, input, output, solverControl) {
          encodePipeline(operatorPipeline, broker, input, output, solverControl);
        },
      },
      preconditioner,
      vectors,
    }, {
      rowCapacity: 1,
      maximumIterations: 4,
      relativeTolerance: 1e-6,
    });
    const pressureSeed = make(4, new Float32Array([0])), pressureOut = make(4);
    const pressureReadback = make(
      4, undefined, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    );
    const controlReadback = make(
      OCTREE_PIPELINED_PCG_CONTROL_BYTES,
      undefined,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    );
    const indirectReadback = make(
      solver.plan.indirectBytes,
      undefined,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    );
    const encoder = device.createCommandEncoder(), broker = new PassBroker(encoder);
    solver.encode(broker, { pressureSeed, pressureOut });
    broker.copyBufferToBuffer(pressureOut, 0, pressureReadback, 0, 4);
    broker.copyBufferToBuffer(
      solver.control, 0, controlReadback, 0, OCTREE_PIPELINED_PCG_CONTROL_BYTES,
    );
    broker.copyBufferToBuffer(solver.workAccountingBuffers.indirectTail, 0,
      indirectReadback, 0, solver.plan.indirectBytes);
    device.queue.submit([broker.finish()]);
    await Promise.all([
      pressureReadback.mapAsync(GPUMapMode.READ),
      controlReadback.mapAsync(GPUMapMode.READ),
      indirectReadback.mapAsync(GPUMapMode.READ),
    ]);
    const pressure = new Float32Array(pressureReadback.getMappedRange())[0];
    const control = [...new Uint32Array(controlReadback.getMappedRange())];
    const indirectWords = new Uint32Array(indirectReadback.getMappedRange());
    const validation = await device.popErrorScope();
    assert.equal(validation, null, validation?.message);
    assert.equal(control[0], 0);
    assert.equal(control[1], 1);
    assert.equal(control[2], 1);
    assert.equal(control[3], 2, "initial and terminal residual reductions are both counted");
    assert.equal(control[5],
      2 + 4 * (solver.plan.maximumIterations - control[2]!),
      "the GPU must prove every encoded post-convergence record is zero-work");
    assert.equal(control[20], 1);
    assert.ok(Math.abs(pressure - 1) < 1e-6);
    const acceptedRows = new Uint32Array([0, 0xffff_ffff, 1, 9, 0, 1]);
    const classWorksets = [new Uint32Array([9, 1, 1, 3, 1, 1, 1]),
      ...Array.from({ length: 3 }, () => new Uint32Array([9, 0, 1, 3, 0, 1, 1]))];
    const spgridDispatch = new Uint32Array(26);
    for (let level = 0; level < 2; level += 1) {
      const base = level * 12; spgridDispatch[base] = 1; spgridDispatch[base + 8] = 1;
      spgridDispatch.set([1, 1, 1], base + 2);
      spgridDispatch.set([0, 1, 1], base + 5);
      spgridDispatch.set([1, 1, 1], base + 9);
    }
    for (let iteration = 0; iteration < solver.plan.maximumIterations; iteration += 1) {
      const base = (iteration * 4 + 1) * 4;
      assert.deepEqual([...indirectWords.subarray(base, base + 4)], [0, 0, 0, 0],
        "post-convergence direct-curvature record is canonical no-work");
    }
    spgridDispatch[24] = 1; spgridDispatch[25] = 1;
    const telemetry = decodeOctreePressureSolveWork({ acceptedRows, classWorksets,
      hybridBandWorksets: classWorksets.map((header) => header.slice()),
      spgridDispatch, outerControl: Uint32Array.from(control),
      outerIndirectTail: Uint32Array.from(indirectWords) }, {
      rowCapacity: 1, maximumOuterIterations: solver.plan.maximumIterations,
      maximumNeighborSlots: 6,
      classApplyEncodedDispatches: 4, hybridBoundarySweeps: 2,
      hybridEncodedCorrectionDispatches: 1, spgridLevelCount: 2,
      spgridEncodedCorrectionDispatches: 10, spgridLevelCapacities: [2, 2],
      reductionPartialCount: 1, reductionLanes: 128,
    });
    assert.equal(telemetry.blocker, null);
    assert.equal(telemetry.report?.outer.executedIterations, 1);
    assert.equal(telemetry.report?.outer.outerReductionPasses, 1);
    assert.equal(telemetry.report?.outer.tailExecutedRowLanes, 0,
      "the GPU-zeroed convergence tail reports no row work");
    pressureReadback.unmap();
    controlReadback.unmap();
    indirectReadback.unmap();
    solver.destroy();
    for (const buffer of [
      pressureSeed, pressureOut, pressureReadback, controlReadback, indirectReadback,
    ]) buffer.destroy();
  }
  for (const buffer of [
    section63Coefficients, rhs, rowCount, acceptedAuthority,
    ...Object.values(vectors),
  ]) buffer.destroy();
  device.destroy();
});
