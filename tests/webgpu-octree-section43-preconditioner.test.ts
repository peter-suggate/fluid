import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  WebGPUOctreeSection43HybridPreconditioner,
  octreeSection43HybridPreconditionerShader,
} from "../lib/webgpu-octree-section43-preconditioner";
import { PassBroker } from "../lib/webgpu-pass-broker";

test("Section 4.3 row domain follows the accepted structured authority", () => {
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn rows\(\) -> u32 \{[^}]*accepted\[2\][^}]*\}/);
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader,
    /fn rows\(\) -> u32 \{[^}]*rowCount\[0\]/,
    "the mutable candidate compaction count is not the accepted pressure-row ABI");
});

test("Section 4.3 hybrid publishes the shell once and schedules the parallel §4.3 correction", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const order: string[] = [];
  let bindGroupCount = 0;
  const buffer = (size: number) => ({
    size,
    destroy() {},
  }) as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer(descriptor: GPUBufferDescriptor) {
      return buffer(Number(descriptor.size));
    },
    createShaderModule() { return {}; },
    createComputePipeline(descriptor: GPUComputePipelineDescriptor) {
      return {
        entryPoint: String(descriptor.compute.entryPoint),
        getBindGroupLayout() { return {}; },
      };
    },
    createBindGroup() { bindGroupCount += 1; return {}; },
  } as unknown as GPUDevice;
  const rowCount = buffer(4);
  const inner = {
    operatorOrder: 1 as const,
    isSymmetricPositiveDefinite: true as const,
    convergenceTail: "gpu-zero-indirect" as const,
    allocatedBytes: 0,
    encodedSetupDispatchCount: 2,
    encodedCorrectionDispatchCount: 3,
    smootherContract: {
      kind: "chebyshev" as const,
      degree: 2 as const,
      spectralBounds: "transactional-scaled-gershgorin" as const,
      lowerFraction: 1 / 30,
    },
    encodeSetup() { order.push("L1-setup"); },
    encodeCorrection() { order.push("L1-correction"); },
  };
  const hybrid = new WebGPUOctreeSection43HybridPreconditioner(device, {
    rowCount,
    firstOrderVCycle: inner,
    secondOrderOperator: {
      convergenceTail: "gpu-zero-indirect",
      encodedDispatchCount: 4,
      encodedMergedBandDispatchCount: 1,
      encode() { order.push("L2-apply"); },
      encodeWorksets() { order.push("L2-band-apply"); },
      encodeMergedBandWorkset() { order.push("L2-band-merged"); },
    },
    section63: { coefficients: buffer(2 * 64 * 19 * 4), control: buffer(32),
      topology: buffer(4), state: buffer(4), geometry: buffer(64 * 16),
      metrics: buffer(64 * 16), layout: buffer(64) },
  }, {
    rowCapacity: 64,
  });
  const encoder = {
    beginComputePass() {
      return {
        setPipeline(pipeline: { entryPoint: string }) {
          order.push(pipeline.entryPoint);
        },
        setBindGroup() {},
        dispatchWorkgroups() {},
        dispatchWorkgroupsIndirect() {},
        end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  const broker = new PassBroker(encoder);
  hybrid.encodeSetup(broker, { solverControl: buffer(128), rowCount });
  const correction = {
    rhs: buffer(64 * 4), correction: buffer(64 * 4),
    solverControl: buffer(128),
    rowCount,
  };
  hybrid.encodeCorrection(broker, correction);

  // Every band sweep consumes a precomputed L2 image of the state it reads, so
  // the merged compact-band apply must immediately precede each smooth and
  // target the same source. Sweep parity is the paper's alternating schedule.
  const bandSweep = (fromB: boolean) =>
    ["L2-band-merged", fromB ? "smoothBtoA" : "smoothAtoB"];
  const preHalf = Array.from({ length: 7 },
    (_unused, index) => bandSweep(((index + 1) & 1) === 1)).flat();
  const postHalf = Array.from({ length: 8 },
    (_unused, index) => bandSweep((index & 1) === 0)).flat();
  assert.deepEqual(order, [
    "resetBandWorksets",
    "prepareCorrectionDispatches",
    "classifyBand",
    "dilateBandAtoB",
    "dilateBandBtoA",
    "dilateBandAtoB",
    "compactBandIntersections",
    "finalizeBandWorksets",
    "L1-setup",
    // §4.3(1) J^k(0, q) -> p1, leaving p1 in hybridA.
    "prepareCorrectionDispatches",
    "smoothZeroToB",
    ...preHalf,
    // §4.3(2) r1 = q - L2 p1 over every accepted row, then M1 r1.
    "L2-apply",
    "formInnerResidual",
    "L1-correction",
    // §4.3(3) p2 = p1 + M1 r1, then the matching k sweeps ending in hybridB.
    "addInnerCorrection",
    ...postHalf,
    "publishCorrection",
  ]);
  assert.equal(hybrid.boundarySmoothingIterations, 8);
  assert.equal(hybrid.encodedSetupDispatchCount, 10);
  assert.equal(hybrid.encodedPassTransitionCount, 4,
    "setup keeps two publication boundaries; correction adds its own gate and the L2 apply gate");
  // Gate, four row stages, 15 band smooths, 15 merged band applies, one exact
  // four-class L2 apply, and the inner page-parallel V-cycle.
  assert.equal(hybrid.encodedCorrectionDispatchCount, 5 + 2 * 15 + 4 + 3);
  assert.equal(hybrid.workAccountingPlan.mergedBandApplies, 15);
  assert.equal(hybrid.workAccountingPlan.avoidedBandDispatches, 45);
  assert.equal(order.includes("L2-band-apply"), false,
    "the compact shell no longer invokes the four-class workset path");
  const cachedBindGroups = bindGroupCount;
  hybrid.encodeCorrection(broker, correction);
  assert.equal(bindGroupCount, cachedBindGroups,
    "an identical recurring correction must reuse every immutable bind group");
  hybrid.destroy();
});

test("Section 4.3 hybrid shader keeps a race-free zero sweep and shared shell operator", () => {
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn stopped\(\) -> bool[\s\S]*control\[0\][\s\S]*control\[1\]/,
    "the fixed encoded preconditioner tail must do no row work after convergence");
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn prepareCorrectionDispatches[\s\S]*gatedRowDispatch\[0\] = select\(0u, \(rows\(\) \+ 63u\) \/ 64u, solveLive\)[\s\S]*word % 3u == 0u/,
    "the shell publishes zero-x row and class records after convergence");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader, /sourceRowDispatch/,
    "the accepted solve must not consume the mutable next-candidate row record");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader, /fn failed\(\)/);
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn smoothZeroValue\(row: u32\)[\s\S]*params\.damping \* rhs\[row\] \/ diagonalAt\(row\)/);
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn diagonalAt\(row: u32\)[\s\S]*section63Coefficients\[\(accepted\[4\]/,
    "the shell must consume channel zero from the accepted Section 6.3 coefficient bank");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader,
    /ResolvedParams|resolvedRows|resolvedNeighbor/,
    "the production shell must not retain the retired resolved-row ABI");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader, /LeafHeader|headers\[/,
    "the zero-initialized legacy row-header diagonal must not remain a solver input");
  assert.doesNotMatch(
    octreeSection43HybridPreconditionerShader.match(
      /fn smoothZeroValue[\s\S]*?return next;\n\}/,
    )?.[0] ?? "",
    /hybridValue|applyHybridA/,
    "the first sweep must not read zeroes written by another workgroup",
  );
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn smoothValue[\s\S]*rhs\[row\] - operatorImage\[row\]/);
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn formInnerResidual[\s\S]*rhs\[row\] - operatorImage\[row\]/);
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn addInnerCorrection[\s\S]*hybridA\[row\] = value; hybridB\[row\] = value;/,
    "the domain-wide M1 correction must initialize both post-shell ping-pong states");
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn compactBandIntersections[\s\S]*atomicAdd\(&bandWorksets\[classBase \+ 1u\]/,
    "the fixed shell is compacted into exact row-class intersections");
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn section63Class[\s\S]*transformAndFlags&0x3f00u[\s\S]*physicalBoundary\|\|section63Coefficients/,
    "closed physical boundaries must remain in the paper's compact smoothing shell");
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn dilatedBand[\s\S]*pageSlot[\s\S]*GHOST[\s\S]*coefficientForDirection/,
    "the three-layer shell must follow physical SPGrid adjacency and destination-owned fine aliases");
  assert.match(octreeSection43HybridPreconditionerShader,
    /fn smoothBtoA[\s\S]*bandRow\(global\.x\)/,
    "repeated shell smooths dispatch only compact band rows");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader,
    /persistentCorrection|persistentArena|pSmoothLevel|pApplyM1/,
    "the serial one-workgroup transcription must not remain a hidden alternate authority");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader,
    /fn finalizeBandWorksets[\s\S]*for \(var row/,
    "band publication must not walk every accepted row from a single lane");
});

test("Dawn executes compact Section 4.3 band publication and shell bindings", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    | GPUBufferUsage.COPY_SRC;
  const make = (size: number, data?: GPUAllowSharedBufferSource,
    usage = storage) => {
    const result = device.createBuffer({ size, usage });
    if (data) device.queue.writeBuffer(result, 0, data);
    return result;
  };
  const resolved = new Uint32Array(64);
  resolved[0] = 1 << 24; resolved[2] = 1;
  const resolvedParams = new Uint32Array([1, 32, 6, 1, 8, 72, 32, 0]);
  const rowCount = make(4, new Uint32Array([1]));
  const accepted = make(32, new Uint32Array([0, 0xffff_ffff, 1, 1, 0, 0, 0, 0]));
  const solverControl = make(128);
  const operator = { convergenceTail: "gpu-zero-indirect" as const,
    encodedDispatchCount: 4, encodedMergedBandDispatchCount: 1 as const,
    encode() {}, encodeWorksets() {}, encodeMergedBandWorkset() {} };
  const inner = { operatorOrder: 1 as const, isSymmetricPositiveDefinite: true as const,
    convergenceTail: "gpu-zero-indirect" as const,
    allocatedBytes: 0, encodedSetupDispatchCount: 0, encodedCorrectionDispatchCount: 1,
    smootherContract: { kind: "chebyshev" as const, degree: 2 as const,
      spectralBounds: "transactional-scaled-gershgorin" as const,
      lowerFraction: 1 / 30 },
    encodeSetup() {}, encodeCorrection() {} };
  device.pushErrorScope("validation");
  const hybrid = new WebGPUOctreeSection43HybridPreconditioner(device, {
    rowCount, firstOrderVCycle: inner,
    secondOrderOperator: operator,
    section63: { coefficients: make(152, new Float32Array([1])), control: accepted,
      topology: make(4), state: make(4), geometry: make(16), metrics: make(16),
      layout: make(64, new Uint32Array(16),
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST) },
  }, { rowCapacity: 1,
  });
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  hybrid.encodeSetup(broker, { solverControl, rowCount });
  hybrid.encodeCorrection(broker, { rhs: make(4), correction: make(4),
    solverControl, rowCount });
  device.queue.submit([broker.finish()]);
  await device.queue.onSubmittedWorkDone();
  const error = await device.popErrorScope();
  assert.equal(error, null, error?.message);
  hybrid.destroy();
  device.destroy();
});
