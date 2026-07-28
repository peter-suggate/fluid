import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS,
} from "../lib/webgpu-octree-section43-contract";
import {
  WebGPUOctreeSection43HybridPreconditioner,
  octreeSection43HybridPreconditionerShader,
} from "../lib/webgpu-octree-section43-preconditioner";
import {
  OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION,
  octreePipelinedMGPCGShader,
  planOctreePipelinedMGPCG,
} from "../lib/webgpu-octree-pipelined-mgpcg";
import { PassBroker } from "../lib/webgpu-pass-broker";

type MockBuffer = GPUBuffer & { readonly label: string; destroyed?: boolean };
type IndirectDispatch = { readonly label: string; readonly offset: number };

test("the serial one-workgroup §4.3 transcription is deleted, not merely unused", () => {
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader,
    /fn persistentCorrection/,
    "the 128-lane serial executor must not remain as an alternate authority");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader,
    /var<workgroup> persistentPage/,
    "its page-resident scratch must not keep an eleventh storage-class reservation");
  assert.doesNotMatch(octreeSection43HybridPreconditionerShader,
    /for\(var page=0u;page<pPageCount/,
    "no shell kernel may walk SPGrid pages serially");
  assert.doesNotMatch(WebGPUOctreeSection43HybridPreconditioner
    .prototype.encodeCorrection.toString(), /persistentCorrection/);
});

test("Section 4.3 correction encodes the paper's parallel schedule", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const buffers: MockBuffer[] = [];
  const parameterWrites: Uint32Array[] = [];
  const groupDescriptors = new Map<object, GPUBindGroupDescriptor>();
  const correctionGroups: object[] = [];
  const correctionEntryPoints: string[] = [];
  const indirect: IndirectDispatch[] = [];
  const bandApplySources: string[] = [];
  let innerCorrections = 0;
  let outerOperatorApplies = 0;
  let directDispatches = 0;
  let observingCorrection = false;
  const buffer = (size: number, label = "external") => ({
    size, label, destroy() { this.destroyed = true; },
  }) as MockBuffer;
  const device = {
    queue: {
      writeBuffer(_target: GPUBuffer, _offset: number, data: GPUAllowSharedBufferSource) {
        if (ArrayBuffer.isView(data)) {
          parameterWrites.push(new Uint32Array(
            data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength),
          ));
        }
      },
    },
    createBuffer(descriptor: GPUBufferDescriptor) {
      const result = buffer(Number(descriptor.size), String(descriptor.label));
      buffers.push(result);
      return result;
    },
    createShaderModule() { return {}; },
    createComputePipeline(descriptor: GPUComputePipelineDescriptor) {
      return {
        entryPoint: String(descriptor.compute.entryPoint),
        getBindGroupLayout() { return {}; },
      };
    },
    createBindGroup(descriptor: GPUBindGroupDescriptor) {
      const group = {};
      groupDescriptors.set(group, descriptor);
      return group;
    },
  } as unknown as GPUDevice;
  const rowCount = buffer(4);
  const firstOrderVCycle = {
    operatorOrder: 1 as const,
    isSymmetricPositiveDefinite: true as const,
    convergenceTail: "gpu-zero-indirect" as const,
    allocatedBytes: 0,
    encodedSetupDispatchCount: 0,
    encodedCorrectionDispatchCount: 1,
    smootherContract: {
      kind: "chebyshev" as const,
      degree: 2 as const,
      spectralBounds: "transactional-scaled-gershgorin" as const,
      lowerFraction: 1 / 30,
    },
    encodeSetup() {},
    encodeCorrection() { innerCorrections += 1; },
  };
  const hybrid = new WebGPUOctreeSection43HybridPreconditioner(device, {
    rowCount,
    firstOrderVCycle,
    secondOrderOperator: {
      convergenceTail: "gpu-zero-indirect",
      encodedDispatchCount: 1,
      encodedMergedBandDispatchCount: 1,
      encode() { outerOperatorApplies += 1; },
      encodeWorksets() { assert.fail("the compact shell must not schedule four-class band work"); },
      encodeMergedBandWorkset(
        _broker: PassBroker, input: GPUBuffer, output: GPUBuffer,
        _control: GPUBuffer, _worksets: GPUBuffer, mergedDispatch: GPUBuffer,
        _layout: GPUBuffer, mergedDispatchOffsetBytes: number,
      ) {
        bandApplySources.push((input as MockBuffer).label);
        assert.equal((output as MockBuffer).label,
          "Section 4.3 hybrid resolved L2 image");
        assert.equal((mergedDispatch as MockBuffer).label,
          "Section 4.3 convergence-gated compact class dispatches");
        assert.equal(mergedDispatchOffsetBytes, 4 * 12,
          "the shell smooths over the fifth union class record");
      },
    },
    section63: {
      coefficients: buffer(2 * 64 * 19 * 4), control: buffer(32),
      topology: buffer(4), state: buffer(4), geometry: buffer(64 * 16),
      metrics: buffer(64 * 16), layout: buffer(64),
    },
  }, { rowCapacity: 64 });
  const encoder = {
    beginComputePass() {
      return {
        setPipeline(pipeline: { entryPoint: string }) {
          if (observingCorrection) correctionEntryPoints.push(pipeline.entryPoint);
        },
        setBindGroup(_index: number, group: object) {
          if (observingCorrection) correctionGroups.push(group);
        },
        dispatchWorkgroups() {
          if (observingCorrection) directDispatches += 1;
        },
        dispatchWorkgroupsIndirect(target: MockBuffer, offset: number) {
          if (observingCorrection) indirect.push({ label: target.label, offset });
        },
        end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  const broker = new PassBroker(encoder);
  observingCorrection = true;
  hybrid.encodeCorrection(broker, {
    rhs: buffer(64 * 4), correction: buffer(64 * 4),
    solverControl: buffer(128), rowCount,
  });
  observingCorrection = false;

  const sweeps = hybrid.boundarySmoothingIterations;
  assert.equal(OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS, 8);
  assert.equal(sweeps, 8, "the correction must use the production k=8 matched halves");
  assert.ok(parameterWrites.some((words) => words[1] === 8),
    "the GPU correction parameters must receive the exact k=8 policy");

  // The zero-vector first sweep needs no operator image; every later sweep in
  // both halves consumes one, so the shell performs exactly 2k-1 band applies.
  assert.equal(bandApplySources.length, 2 * sweeps - 1);
  assert.equal(hybrid.workAccountingPlan.mergedBandApplies, 2 * sweeps - 1);
  assert.deepEqual(bandApplySources, [
    ...Array.from({ length: sweeps - 1 }, (_unused, index) =>
      (((index + 1) & 1) === 1 ? "B" : "A")),
    ...Array.from({ length: sweeps }, (_unused, index) =>
      ((index & 1) === 0 ? "B" : "A")),
  ].map((state) => `Section 4.3 hybrid L2 iterate ${state}`),
  "each apply must target the state its following smooth reads");
  assert.equal(innerCorrections, 1,
    "the page-parallel first-order V-cycle runs once between the matched halves");
  assert.equal(outerOperatorApplies, 1,
    "the inner residual needs one exact row-wide L2 apply of p1");

  assert.deepEqual(correctionEntryPoints, [
    "prepareCorrectionDispatches",
    "smoothZeroToB",
    ...Array.from({ length: sweeps - 1 }, (_unused, index) =>
      (((index + 1) & 1) === 1 ? "smoothBtoA" : "smoothAtoB")),
    "formInnerResidual",
    "addInnerCorrection",
    ...Array.from({ length: sweeps }, (_unused, index) =>
      ((index & 1) === 0 ? "smoothBtoA" : "smoothAtoB")),
    "publishCorrection",
  ]);
  assert.equal(directDispatches, 1,
    "only the convergence-gate publisher is a direct singleton dispatch");
  assert.equal(indirect.length, 2 * sweeps + 3);
  const rowRecord = "Section 4.3 convergence-gated row dispatch";
  const bandRecord = "Section 4.3 convergence-gated compact class dispatches";
  assert.deepEqual(indirect.filter((entry) => entry.label === rowRecord),
    Array.from({ length: 4 }, () => ({ label: rowRecord, offset: 0 })),
    "the four row-wide stages consume the convergence-gated accepted-row record");
  assert.deepEqual(indirect.filter((entry) => entry.label === bandRecord),
    Array.from({ length: 2 * sweeps - 1 },
      () => ({ label: bandRecord, offset: 4 * 12 })),
    "every shell smooth consumes the convergence-gated union class record");
  // Gate + four row stages + 2k-1 smooths + 2k-1 merged applies + one exact
  // apply + the inner V-cycle. A converged solve encodes all of them and
  // dispatches zero workgroups.
  assert.equal(hybrid.encodedCorrectionDispatchCount,
    5 + 2 * (2 * sweeps - 1) + 1 + 1);
  assert.equal(hybrid.encodedCorrectionDispatchCount,
    directDispatches + indirect.length + bandApplySources.length
      + outerOperatorApplies + innerCorrections);

  const storageBindings = new Set(Array.from(
    octreeSection43HybridPreconditionerShader.matchAll(
      /@group\(0\)\s*@binding\((\d+)\)\s*var<storage\b/g,
    ), (match) => Number(match[1])));
  for (const group of new Set(correctionGroups)) {
    const used = [...groupDescriptors.get(group)!.entries]
      .filter((entry) => storageBindings.has(entry.binding)).length;
    assert.ok(used <= 10,
      `a correction stage uses ${used} storage bindings; the portable limit is 10`);
  }
  hybrid.destroy();
  assert.ok(buffers.every((candidate) => candidate.destroyed));
});

test("the parallel Section 4.3 correction does not change the ten-step outer convergence gate", () => {
  const plan = planOctreePipelinedMGPCG({ rowCapacity: 64, maximumIterations: 10 });
  assert.equal(plan.maximumIterations, 10);
  assert.equal(plan.dispatchRecordCount,
    10 * OCTREE_PIPELINED_PCG_DISPATCHES_PER_ITERATION);
  assert.match(octreePipelinedMGPCGShader,
    /fn zeroRemainingAfterUpdate\(iteration: u32\)[\s\S]*zeroDispatch\(iteration, 3u\)[\s\S]*for \(var future = iteration \+ 1u; future < maximumIterations\(\);/,
    "the same-step residual gate must still zero every future outer record");
  assert.match(WebGPUOctreeSection43HybridPreconditioner.prototype.encodeCorrection.toString(),
    /dispatch|run/,
    "Section 4.3 remains an ordinary outer-preconditioner correction call");
});
