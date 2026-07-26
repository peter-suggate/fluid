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

function wgslFunctionBody(shader: string, entryPoint: string): string {
  const signature = new RegExp(`\\bfn\\s+${entryPoint}\\s*\\(`);
  const match = signature.exec(shader);
  assert.ok(match, `WGSL entry point ${entryPoint} must exist`);
  const open = shader.indexOf("{", match.index);
  assert.ok(open >= 0, `WGSL entry point ${entryPoint} must have a body`);
  let depth = 0;
  for (let index = open; index < shader.length; index += 1) {
    if (shader[index] === "{") depth += 1;
    if (shader[index] === "}") depth -= 1;
    if (depth === 0) return shader.slice(open + 1, index);
  }
  assert.fail(`WGSL entry point ${entryPoint} has an unterminated body`);
}

test("Section 4.3 correction is one persistent dispatch with no host-side numerical loop", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const buffers: MockBuffer[] = [];
  const parameterWrites: Uint32Array[] = [];
  const groupDescriptors = new Map<object, GPUBindGroupDescriptor>();
  const correctionGroups: object[] = [];
  const correctionEntryPoints: string[] = [];
  let innerCorrections = 0;
  let outerOperatorApplies = 0;
  let mergedBandApplies = 0;
  let correctionDispatches = 0;
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
  const hybrid = new WebGPUOctreeSection43HybridPreconditioner(device, {
    rowCount,
    firstOrderVCycle: {
      operatorOrder: 1,
      isSymmetricPositiveDefinite: true,
      convergenceTail: "gpu-zero-indirect",
      allocatedBytes: 0,
      encodedSetupDispatchCount: 0,
      encodedCorrectionDispatchCount: 1,
      smootherContract: {
        kind: "chebyshev",
        degree: 2,
        spectralBounds: "transactional-scaled-gershgorin",
        lowerFraction: 1 / 30,
      },
      encodeSetup() {},
      encodeCorrection() { innerCorrections += 1; },
    },
    secondOrderOperator: {
      convergenceTail: "gpu-zero-indirect",
      encodedDispatchCount: 1,
      encodedMergedBandDispatchCount: 1,
      encode() { outerOperatorApplies += 1; },
      encodeWorksets() { assert.fail("persistent correction must not schedule class work on host"); },
      encodeMergedBandWorkset() { mergedBandApplies += 1; },
    },
    section63: {
      coefficients: buffer(2 * 64 * 19 * 4), control: buffer(32),
      topology: buffer(4), state: buffer(4), geometry: buffer(64 * 16),
      metrics: buffer(64 * 16), layout: buffer(64), dispatch: buffer(12),
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
          if (observingCorrection) correctionDispatches += 1;
        },
        dispatchWorkgroupsIndirect() {
          if (observingCorrection) correctionDispatches += 1;
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

  assert.equal(OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS, 8);
  assert.equal(hybrid.boundarySmoothingIterations, 8,
    "the persistent correction must use the production k=8 matched halves");
  assert.ok(parameterWrites.some((words) => words[1] === 8),
    "the GPU correction parameters must receive the exact k=8 policy");
  assert.equal(hybrid.encodedCorrectionDispatchCount, 1);
  assert.equal(correctionDispatches, 1,
    "one preconditioner application must encode one GPU correction dispatch");
  assert.equal(innerCorrections, 0,
    "the host must not enter the first-order V-cycle during correction encoding");
  assert.equal(outerOperatorApplies, 0,
    "the host must not enter a full L2 operator during correction encoding");
  assert.equal(mergedBandApplies, 0,
    "the host must not enter a smoothing loop during correction encoding");
  const hostCorrection = WebGPUOctreeSection43HybridPreconditioner
    .prototype.encodeCorrection.toString();
  assert.doesNotMatch(hostCorrection, /\b(?:for|while)\s*\(/,
    "the persistent correction's numerical loops belong in WGSL");

  assert.equal(correctionEntryPoints.length, 1);
  const correctionBody = wgslFunctionBody(
    octreeSection43HybridPreconditionerShader,
    correctionEntryPoints[0]!,
  );
  const markers = [
    "§4.3(1)",
    "§4.3(2)",
    "§4.3(3)",
  ] as const;
  let previous = -1;
  for (const marker of markers) {
    assert.equal(correctionBody.split(marker).length - 1, 1,
      `${marker} must mark exactly one phase boundary`);
    const index = correctionBody.indexOf(marker);
    assert.ok(index > previous,
      `${marker} must occur in exact pre-smooth, L1, post-smooth order`);
    previous = index;
  }
  const preSmooth = correctionBody.slice(
    correctionBody.indexOf(markers[0]), correctionBody.indexOf(markers[1]),
  );
  const postSmooth = correctionBody.slice(
    correctionBody.indexOf(markers[2]),
  );
  assert.match(preSmooth,
    /for\s*\(var sweep\s*=\s*1u;\s*sweep\s*<\s*params\.smoothingIterations;/,
    "the explicit zero-vector first sweep plus seven loop sweeps must produce pre k=8");
  assert.match(postSmooth,
    /for\s*\(var sweep\s*=\s*0u;\s*sweep\s*<\s*params\.smoothingIterations;/,
    "the post half must execute all eight matching sweeps");

  const storageBindings = new Set(Array.from(
    octreeSection43HybridPreconditionerShader.matchAll(
      /@group\(0\)\s*@binding\((\d+)\)\s*var<storage\b/g,
    ), (match) => Number(match[1])));
  const usedGroups = [...new Set(correctionGroups)];
  assert.equal(usedGroups.length, 1,
    "the persistent correction dispatch must bind one fused resource set");
  const storageBindingCount = usedGroups.reduce((count, group) => count
    + [...groupDescriptors.get(group)!.entries]
      .filter((entry) => storageBindings.has(entry.binding)).length, 0);
  assert.ok(storageBindingCount <= 10,
    `persistent correction uses ${storageBindingCount} storage bindings; target limit is 10`);
  hybrid.destroy();
  assert.ok(buffers.every((candidate) => candidate.destroyed));
});

test("persistent Section 4.3 correction does not change the ten-step outer convergence gate", () => {
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
