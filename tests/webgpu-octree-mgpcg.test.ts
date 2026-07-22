import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_MGPCG_PRECONDITIONER_KIND,
  OCTREE_SECTION43_DEFAULT_PCG_ITERATIONS,
  OCTREE_SECTION43_BOUNDARY_BAND_LAYERS,
  OCTREE_SECTION43_MAXIMUM_PCG_ITERATIONS,
  OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS,
  WebGPUOctreeMGPCG,
  octreeMGPCGShader,
  normalizeOctreeSection43BoundarySmoothing,
  normalizeOctreeSection43IterationCap,
  planOctreeMGPCG,
} from "../lib/webgpu-octree-mgpcg";
import { octreePowerOperatorShader } from "../lib/webgpu-octree-power-operator";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

test("MGPCG allocation is bounded by compact rows and sparse levels", () => {
  const small = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16 });
  const wide = planOctreeMGPCG({ dimensions: [1024, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16 });
  assert.equal(small.rowCapacity, wide.rowCapacity);
  assert.ok(wide.hierarchyLevelCount > small.hierarchyLevelCount);
  assert.equal(wide.hierarchyBytes, wide.hierarchyLevelCount * wide.rowCapacity * 5 * 4);
  assert.ok(!("cellCount" in wide), "planner must not expose a finest-domain allocation");
});

test("Section 4.3 hybrid is opt-in, separately sized, and requires an explicit SPD L1 V-cycle", () => {
  const aggregate = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16 });
  const hybrid = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16,
    preconditionerKind: "section43-hybrid" });
  assert.equal(aggregate.preconditionerKind, "aggregate");
  assert.equal(aggregate.hybridBytes, 0);
  assert.equal(hybrid.preconditionerKind, "section43-hybrid");
  assert.equal(hybrid.hierarchyBytes, 0, "aggregate hierarchy is not the hybrid's L1 V-cycle");
  assert.equal(hybrid.hybridBytes, hybrid.vectorBytes * 6);
  assert.throws(() => new WebGPUOctreeMGPCG({} as GPUDevice, {
    leafHeaders: {} as GPUBuffer, leafEntries: {} as GPUBuffer, rowCount: {} as GPUBuffer,
  }, { dimensions: [64, 48, 32], rowCapacity: 10_000, maximumLeafSize: 16,
    maximumIterations: 16, preconditionerKind: "section43-hybrid" }),
  /requires an explicit SPD first-order V-cycle/);
});

test("Section 4.3 hybrid has a three-layer band and symmetry-locked paired L2 smoothing", () => {
  assert.equal(OCTREE_SECTION43_BOUNDARY_BAND_LAYERS, 3);
  assert.equal(OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS, 8);
  assert.equal(OCTREE_SECTION43_MAXIMUM_PCG_ITERATIONS, 128);
  assert.equal(OCTREE_SECTION43_DEFAULT_PCG_ITERATIONS, 128);
  assert.equal(normalizeOctreeSection43IterationCap(undefined), 128);
  assert.equal(normalizeOctreeSection43IterationCap(7), 8);
  assert.equal(normalizeOctreeSection43IterationCap(400), 128);
  assert.equal(normalizeOctreeSection43BoundarySmoothing(undefined), 8);
  assert.equal(normalizeOctreeSection43BoundarySmoothing(1), 2);
  assert.equal(normalizeOctreeSection43BoundarySmoothing(7), 8,
    "odd inputs must round to an even ping/pong schedule");
  assert.equal(normalizeOctreeSection43BoundarySmoothing(32), 16);
  assert.match(octreeMGPCGShader, /boundaryGap=h\.diagonal-offDiagonalSum/);
  assert.match(octreeMGPCGShader, /headers\[e\.row\]\.size!=h\.size/);
  assert.match(octreeMGPCGShader, /dilateHybridBandAtoB/);
  assert.match(octreeMGPCGShader, /dilateHybridBandBtoA/);
  assert.match(octreeMGPCGShader, /formHybridL1Residual/);
  assert.match(octreeMGPCGShader, /addHybridL1Correction/);
  const source = WebGPUOctreeMGPCG.toString();
  assert.match(source, /firstOrderVCycle|encodeCorrection/);
  assert.match(source, /boundarySmoothingIterations/);
  assert.match(WebGPUOctreeProjection.toString(), /3 graph-ring band approximation/,
    "the visible solver label must not describe graph dilation as an exact three-voxel paper band");
});

test("Section 4.3 tuning keeps equal pre/post sweeps, exact dispatch accounting, and one pass", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 } });
  let passes = 0, dispatches = 0, currentStage = "";
  const events: string[] = [];
  const buffer = (size: number) => ({ size, usage: 7, destroy() {} }) as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: ({ size }: { size: number }) => buffer(size),
    createShaderModule: () => ({}),
    createComputePipeline: ({ label, compute }: { label: string; compute: { entryPoint: string } }) => ({
      label, entryPoint: compute.entryPoint, getBindGroupLayout: () => ({}),
    }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const correctionDispatches = 5;
  const firstOrderVCycle = {
    operatorOrder: 1 as const, isSymmetricPositiveDefinite: true as const, allocatedBytes: 0,
    encodedCorrectionPassCount: correctionDispatches,
    encodedCorrectionDispatchCount: correctionDispatches,
    encodedSetupDispatchCount: 3,
    encodedPassTransitionCount: 1,
    encodeSetup(_encoder: GPUCommandEncoder, _input: unknown, pass?: GPUComputePassEncoder) {
      assert.ok(pass, "MGPCG must lend its active pass to V-cycle setup");
      for (let dispatch = 0; dispatch < 3; dispatch += 1) pass.dispatchWorkgroups(1);
    },
    encodeCorrection(_encoder: GPUCommandEncoder, _input: unknown, pass?: GPUComputePassEncoder) {
      assert.ok(pass, "MGPCG must lend its active pass to every V-cycle correction");
      for (let dispatch = 0; dispatch < correctionDispatches; dispatch += 1) pass.dispatchWorkgroups(1);
    },
  };
  const source = {
    leafHeaders: buffer(48 * 128), leafEntries: buffer(8 * 512), rowCount: buffer(64), firstOrderVCycle,
  };
  const maximumIterations = 2;
  const solver = new WebGPUOctreeMGPCG(device, source, {
    dimensions: [16, 16, 16], rowCapacity: 128, maximumLeafSize: 4, maximumIterations,
    preconditionerKind: "section43-hybrid", boundarySmoothingIterations: 7,
  });
  const encoder = {
    clearBuffer() {},
    beginComputePass: () => {
      passes += 1;
      return {
        setPipeline(pipeline: { entryPoint: string }) { currentStage = pipeline.entryPoint; },
        setBindGroup() {},
        dispatchWorkgroups() { dispatches += 1; events.push(currentStage); },
        end() {},
      };
    },
  } as unknown as GPUCommandEncoder;
  solver.encode(encoder, buffer(512), buffer(512));

  const sweeps = solver.boundarySmoothingIterations;
  assert.equal(sweeps, 8, "constructor must apply even normalization before recording the schedule");
  const starts = events.flatMap((stage, index) => stage === "clearHybridPreconditioner" ? [index] : []);
  assert.equal(starts.length, maximumIterations + 1, "initial M plus one M per fixed PCG iteration");
  for (const start of starts) {
    const form = events.indexOf("formHybridL1Residual", start);
    const add = events.indexOf("addHybridL1Correction", form);
    const publish = events.indexOf("publishHybridPreconditioner", add);
    assert.ok(start < form && form < add && add < publish);
    const pre = events.slice(start, form).filter((stage) => stage.startsWith("smoothHybrid"));
    const post = events.slice(add, publish).filter((stage) => stage.startsWith("smoothHybrid"));
    assert.equal(pre.length, sweeps);
    assert.equal(post.length, sweeps, "post-smoothing must exactly match pre-smoothing");
    assert.equal(pre[0], "smoothHybridAtoB");
    assert.equal(pre.at(-1), "smoothHybridBtoA", "even pre-sweeps must publish canonical A");
    assert.equal(post[0], "smoothHybridBtoA");
    assert.equal(post.at(-1), "smoothHybridAtoB", "even post-sweeps must publish canonical B");
  }
  const preconditionerDispatches = 4 + 2 * sweeps + correctionDispatches;
  const expectedDispatches = 3 + 7 + preconditionerDispatches + 1
    + maximumIterations * (6 + preconditionerDispatches) + 2;
  assert.equal(dispatches, expectedDispatches);
  assert.equal(solver.encodedDispatchCount, expectedDispatches);
  assert.equal(passes, 1);
  assert.equal(solver.encodedPassTransitionCount, 1);
  solver.destroy();
});

test("authoritative power projection constructs and selects the Section 4.3 L1 V-cycle", () => {
  const source = WebGPUOctreeProjection.toString();
  assert.match(source, /new WebGPUOctreeFirstOrderVCycle/);
  assert.match(source, /preconditionerKind:\s*this\.powerPolicy\.authoritative\s*\?\s*"section43-hybrid"/);
  assert.match(source, /firstOrderVCycle\?\.encodeCapture\(encoder\)/,
    "L1 rows must be captured before power publication replaces the shared CSR");
});

test("matrix-free aggregate PCG uses additive transfers and GPU-only convergence", () => {
  assert.equal(OCTREE_MGPCG_PRECONDITIONER_KIND, "additive-geometric-aggregate-diagonal");
  assert.match(octreeMGPCGShader, /value-=e\.coefficient\*fieldValue/);
  assert.match(octreeMGPCGShader, /restrictResidual/);
  assert.match(octreeMGPCGShader, /prolongateCorrection/);
  assert.match(octreeMGPCGShader, /residual\[row\]\*preconditioned\[row\]/);
  assert.match(octreeMGPCGShader, /atomicStore\(&control\[1\],1u\)/);
  assert.match(octreeMGPCGShader, /fn reduceUpdatedResidual/);
  const encodeSource = WebGPUOctreeMGPCG.prototype.encode.toString();
  const update = encodeSource.indexOf("rows(this.stages.update)");
  const residualGate = encodeSource.indexOf("single(this.stages.updatedResidualReduction)", update);
  const nextPreconditioner = encodeSource.indexOf("this.applyPreconditioner", update);
  assert.ok(update >= 0 && update < residualGate && residualGate < nextPreconditioner,
    "PCG must test the updated residual before another preconditioner application");
  assert.equal(octreeMGPCGShader.match(/atomicAdd\(&control\[2\],1u\)/g)?.length, 1,
    "an immediately converged pressure update must still count as one iteration");
  assert.doesNotMatch(WebGPUOctreeProjection.prototype.encode.toString(), /mapAsync|getMappedRange/);
  assert.match(octreeMGPCGShader,
    /var observed=atomicLoad\(&hierarchy\[at\]\);[\s\S]*retry<16u[\s\S]*observed=claim\.old_value/,
    "the rollback aggregate hierarchy must retry a spurious weak-CAS failure at the same slot");
});

test("aggregate preconditioner is not mislabeled as the paper Section 4.3 hybrid V-cycle", () => {
  const source = WebGPUOctreeProjection.toString();
  assert.doesNotMatch(source, /matrix-free MGPCG/);
  assert.doesNotMatch(source, /"multigrid"/);
  assert.doesNotMatch(octreeMGPCGShader, /GhostValuePropagate|GhostValueAccumulate/);
});

test("power projection publication is gated by MGPCG success", () => {
  assert.match(octreePowerOperatorShader, /preparePowerProjectionMGPCG/);
  assert.match(octreePowerOperatorShader, /atomicLoad\(&solverControl\[0\]\)!=0u/);
  assert.match(octreePowerOperatorShader, /atomicLoad\(&solverControl\[1\]\)==0u/);
  assert.match(WebGPUOctreeProjection.prototype.encode.toString(), /this\.mgpcg.*encode/);
});

test("fixed PCG replay retains immutable bind groups instead of rebuilding descriptors per dispatch", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 } });
  let bindGroups = 0, passes = 0, dispatches = 0;
  const buffer = (size: number) => ({ size, usage: 7, destroy() {} }) as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: ({ size }: { size: number }) => buffer(size),
    createShaderModule: () => ({}),
    createComputePipeline: ({ label, compute }: { label: string; compute: { entryPoint: string } }) => ({
      label, entryPoint: compute.entryPoint, getBindGroupLayout: () => ({}),
    }),
    createBindGroup: () => { bindGroups += 1; return {}; },
  } as unknown as GPUDevice;
  const solver = new WebGPUOctreeMGPCG(device, {
    leafHeaders: buffer(48 * 256), leafEntries: buffer(8 * 1024), rowCount: buffer(64),
  }, { dimensions: [16, 16, 16], rowCapacity: 256, maximumLeafSize: 4, maximumIterations: 128 });
  const encoder = {
    clearBuffer() {},
    beginComputePass: () => { passes += 1; return { setPipeline() {}, setBindGroup() {},
      dispatchWorkgroups() { dispatches += 1; }, end() {} }; },
  } as unknown as GPUCommandEncoder;
  const pressureA = buffer(1024), pressureB = buffer(1024);
  solver.encode(encoder, pressureA, pressureB);
  const firstGroups = bindGroups, firstPasses = passes, firstDispatches = dispatches;
  assert.equal(firstPasses, solver.encodedPassTransitionCount,
    "the complete pressure schedule should use one ordered compute pass");
  assert.equal(solver.encodedPassCount, solver.encodedDispatchCount,
    "the legacy count remains an exact dispatch-count alias");
  assert.equal(firstDispatches, solver.encodedDispatchCount,
    "reported pressure dispatch count must equal the command stream actually emitted");
  solver.encode(encoder, pressureA, pressureB);
  assert.equal(passes, firstPasses * 2, "a second replay should add only one pass transition");
  assert.ok(firstDispatches > firstGroups * 30, `${firstDispatches} dispatches should share ${firstGroups} descriptors`);
  assert.equal(bindGroups, firstGroups, "a second fixed replay must allocate no bind groups");
  solver.destroy();
});
