import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_SECTION43_BOUNDARY_BAND_LAYERS,
  OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS,
  OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS,
  OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS,
  OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY,
  OCTREE_PERSISTENT_MGPCG_STATE_CHANNELS,
  WebGPUOctreeMGPCG,
  octreeMGPCGAuthorityGateShader,
  octreeMGPCGShader,
  normalizeOctreeSection43BoundarySmoothing,
  octreeSection43IterationBudget,
  planOctreeMGPCG,
  selectOctreeMGPCGExecutionMode,
} from "../lib/webgpu-octree-mgpcg";
import { octreePowerOperatorShader } from "../lib/webgpu-octree-power-operator";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { solveSPGridBottomLDLT } from "../lib/webgpu-octree-spgrid-vcycle";
import { PassBroker } from "../lib/webgpu-pass-broker";

const dot = (left: readonly number[], right: readonly number[]) =>
  left.reduce((sum, value, row) => sum + value * right[row], 0);
const multiply = (matrix: readonly (readonly number[])[], vector: readonly number[]) =>
  matrix.map((row) => dot(row, vector));

/** CPU transcription of Section 4.3 steps 1--3. This is intentionally an
 * algebra oracle rather than another production preconditioner. */
function applySection43Oracle(l2: readonly (readonly number[])[], l1: readonly (readonly number[])[],
  band: readonly boolean[], rhs: readonly number[], iterations = 8, omega = 2 / 3): number[] {
  const pressure = new Array<number>(rhs.length).fill(0);
  const smooth = () => {
    const product = multiply(l2, pressure);
    for (let row = 0; row < pressure.length; row += 1) {
      if (band[row]) pressure[row] += omega * (rhs[row] - product[row]) / l2[row][row];
    }
  };
  for (let iteration = 0; iteration < iterations; iteration += 1) smooth();
  const residual = multiply(l2, pressure).map((value, row) => rhs[row] - value);
  const correction = solveSPGridBottomLDLT(l1, residual);
  for (let row = 0; row < pressure.length; row += 1) pressure[row] += correction[row];
  for (let iteration = 0; iteration < iterations; iteration += 1) smooth();
  return pressure;
}

/** Ordinary preconditioned-CG transcription for the staged device recurrence. */
function applyPCGOracle(matrix: readonly (readonly number[])[], inverse: readonly number[],
  rhs: readonly number[], iterations: number, tolerance = 1e-12) {
  const pressure = rhs.map(() => 0); const residual = [...rhs];
  let preconditioned = residual.map((value, row) => inverse[row] * value);
  let direction = [...preconditioned], rz = dot(residual, preconditioned);
  const history: number[] = [];
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    const product = multiply(matrix, direction);
    const alpha = rz / dot(direction, product);
    pressure.forEach((_, row) => { pressure[row] += alpha * direction[row]; residual[row] -= alpha * product[row]; });
    const relative = Math.sqrt(dot(residual, residual) / dot(rhs, rhs)); history.push(relative);
    if (relative <= tolerance) break;
    preconditioned = residual.map((value, row) => inverse[row] * value);
    const nextRz = dot(residual, preconditioned), beta = nextRz / rz;
    direction = preconditioned.map((value, row) => value + beta * direction[row]); rz = nextRz;
  }
  return { pressure, residual, history };
}

test("Section 4.3 MGPCG allocation is bounded by compact rows", () => {
  const small = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000 });
  const wide = planOctreeMGPCG({ dimensions: [1024, 48, 32], rowCapacity: 10_000 });
  assert.equal(small.rowCapacity, wide.rowCapacity);
  assert.equal(wide.allocatedBytes, small.allocatedBytes);
  assert.equal(wide.hybridBytes, wide.vectorBytes * 6);
  assert.equal(wide.allocatedBytes, wide.vectorBytes * 11 + 64 + 64,
    "large-domain storage must contain only compact PCG and hybrid vectors");
  assert.equal(wide.persistentStateBytes, wide.vectorBytes * OCTREE_PERSISTENT_MGPCG_STATE_CHANNELS);
  assert.equal(small.persistentRequiredByCapacity, false);
  const persistent = planOctreeMGPCG({ dimensions: [4, 4, 4], rowCapacity: 64 });
  assert.equal(persistent.persistentRequiredByCapacity, true);
  assert.equal(persistent.allocatedBytes, 64, "small-domain MGPCG owns only its shared solve-control record");
  assert.ok(!("cellCount" in wide), "planner must not expose a finest-domain allocation");
  assert.ok(!("hierarchyBytes" in wide), "the deleted aggregate hierarchy must not be allocated");
});

test("persistent MGPCG selection is immutable-capacity gated and proof carrying", () => {
  const exact = {
    maximumRowCapacity: OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY,
    encodedDispatchCount: 1 as const, dispatchShape: [1, 1, 1] as const,
    invariantProof: {
      ghostRows: "spgrid-identical" as const,
      transfers: "validated-adjoint-pair" as const,
      invalidRows: "uniform-fail-closed-before-arithmetic" as const,
    },
    encodeSolve() {},
  };
  assert.equal(selectOctreeMGPCGExecutionMode(6_912, undefined), "staged-pcg-large-domain",
    "the mini-dam capacity must use the parallel PCG large-domain lane");
  assert.throws(() => selectOctreeMGPCGExecutionMode(64, undefined), /requires a persistent proof-carrying executor/);
  assert.equal(selectOctreeMGPCGExecutionMode(64, exact), "persistent-small-domain");
  assert.equal(selectOctreeMGPCGExecutionMode(OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY + 1, exact),
    "staged-pcg-large-domain", "large-domain policy is selected from immutable capacity, not live rows");
  assert.throws(() => selectOctreeMGPCGExecutionMode(64, { ...exact, maximumRowCapacity: 32 }), /incompatible/);
  assert.throws(() => selectOctreeMGPCGExecutionMode(64, { ...exact,
    invariantProof: { ...exact.invariantProof, transfers: "unproven" as never } }), /invariant proof/);
  assert.throws(() => selectOctreeMGPCGExecutionMode(0, exact), /positive integer/);
});

test("small-domain construction rejects missing proof before allocating GPU state", () => {
  let allocations = 0;
  const device = { createBuffer() { allocations += 1; return {} as GPUBuffer; } } as unknown as GPUDevice;
  const buffer = (size: number) => ({ size }) as GPUBuffer;
  const cycle = { operatorOrder: 1 as const, isSymmetricPositiveDefinite: true as const, allocatedBytes: 0,
    encodedCorrectionDispatchCount: 1, encodeSetup() {}, encodeCorrection() {} };
  assert.throws(() => new WebGPUOctreeMGPCG(device, {
    leafHeaders: buffer(48 * 64), leafEntries: buffer(8), rowCount: buffer(64), firstOrderVCycle: cycle,
  }, { dimensions: [4, 4, 4], rowCapacity: 64 }), /requires a persistent proof-carrying executor/);
  assert.equal(allocations, 0);
});

test("large-domain PCG responds to the preconditioned residual and solves an SPD M·L2 operator", () => {
  assert.equal(OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS, 10);
  const matrix = [
    [3, -1, 0, 0], [-1, 5, -1, 0], [0, -1, 7, -2], [0, 0, -2, 9],
  ];
  const inverse = [0.25, 0.2, 0.125, 0.1];
  const rhs = [1.2, -0.7, 2.1, 0.4];
  const solved = applyPCGOracle(matrix, inverse, rhs, OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS);
  assert.ok(solved.history.at(-1)! < 1e-12,
    `PCG recurrence left relative residual ${solved.history.at(-1)}`);
  assert.ok(solved.history.length <= matrix.length,
    "exact-arithmetic PCG must terminate in at most the matrix dimension");
  const exactResidual = multiply(matrix, solved.pressure).map((value, row) => rhs[row] - value);
  assert.ok(exactResidual.every((value, row) => Math.abs(value - solved.residual[row]) < 1e-12),
    "x and r updates must consume the same L2 direction product");
});

test("large-domain PCG fuses scalar reductions with adjacent vector work", () => {
  assert.doesNotMatch(octreeMGPCGShader, /Chebyshev|spectrum|initializeChebyshevPower/,
    "large-domain PCG must not retain spectral estimation or a fixed polynomial");
  assert.match(octreeMGPCGShader,
    /preparePCGStep[\s\S]*let q=applyA\(row,true\);product\[row\]=q;dq\+=direction\[row\]\*q/,
    "A*d and d·A*d must share one workgroup dispatch");
  assert.match(octreeMGPCGShader,
    /updatePCGStateAndReduceResidual[\s\S]*pressure\[row\]=x;residual\[row\]=r;rr\+=r\*r/,
    "x/r update and residual reduction must share one workgroup dispatch");
  assert.match(octreeMGPCGShader,
    /finishPCGIteration[\s\S]*rz\+=residual\[row\]\*preconditioned\[row\][\s\S]*preconditioned\[row\]\+beta\*direction\[row\]/,
    "r·z reduction and direction update must share one workgroup dispatch");
  assert.match(octreeMGPCGShader,
    /clearHybridPreconditioner[\s\S]*row>=liveRows\(\)[\s\S]*hybridA\[row\]=0\.0/,
    "preconditioner initialization must touch one live vector only");
  const clear = octreeMGPCGShader.slice(octreeMGPCGShader.indexOf("fn clearHybridPreconditioner"),
    octreeMGPCGShader.indexOf("fn classifyHybridBand"));
  assert.doesNotMatch(clear, /hybridB|hybridRhs|hybridCorrection/,
    "vectors overwritten later in the preconditioner must not be cleared");
  const encode = WebGPUOctreeMGPCG.prototype.encode.toString();
  const prepare = encode.indexOf("single(this.stages.preparePCGStep)");
  const deviceGate = encode.indexOf("single(this.stages.updatePCGState)", prepare);
  const finalize = encode.indexOf("single(this.stages.finalize)", deviceGate);
  assert.ok(prepare >= 0 && prepare < deviceGate && deviceGate < finalize,
    "every PCG update must reach the device residual gate before fail-closed publication");
  assert.doesNotMatch(encode, /mapAsync|getMappedRange/,
    "PCG convergence and acceptance must remain GPU-only");
});

test("an exact persistent executor is the sole small-domain solve and allocates no large-domain tail", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 } });
  let passes = 0, dispatches = 0, persistentCalls = 0, allocations = 0, pipelines = 0;
  const buffer = (size: number) => ({ size, usage: 7, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} }, createBuffer: ({ size }: { size: number }) => { allocations += 1; return buffer(size); },
    createShaderModule: () => ({}), createComputePipeline: ({ compute }: { compute: { entryPoint: string } }) => {
      pipelines += 1; return { entryPoint: compute.entryPoint, getBindGroupLayout: () => ({}) };
    }, createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const encoder = { clearBuffer() {}, beginComputePass: () => { passes += 1; return {
    setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { dispatches += 1; }, end() {},
  }; } } as unknown as GPUCommandEncoder;
  const persistentMGPCG = {
    maximumRowCapacity: 64,
    encodedDispatchCount: 1 as const, dispatchShape: [1, 1, 1] as const,
    invariantProof: { ghostRows: "spgrid-identical" as const, transfers: "validated-adjoint-pair" as const,
      invalidRows: "uniform-fail-closed-before-arithmetic" as const },
    encodeSolve(broker: PassBroker) {
      persistentCalls += 1; broker.compute().dispatchWorkgroups(1);
    },
  };
  const cycle = {
    operatorOrder: 1 as const, isSymmetricPositiveDefinite: true as const, allocatedBytes: 0,
    encodedCorrectionDispatchCount: 57, encodedSetupDispatchCount: 3, persistentMGPCG,
    encodeSetup(broker: PassBroker) {
      const pass = broker.compute();
      for (let i = 0; i < 3; i += 1) pass.dispatchWorkgroups(1);
      broker.fence("test setup publication");
    },
    encodeCorrection() { throw new Error("host-recorded correction must not run for a persistent solve"); },
  };
  const solver = new WebGPUOctreeMGPCG(device, {
    leafHeaders: buffer(48 * 64), leafEntries: buffer(8 * 512), rowCount: buffer(64), firstOrderVCycle: cycle,
  }, { dimensions: [4, 4, 4], rowCapacity: 64 });
  solver.encode(new PassBroker(encoder), buffer(4 * 64), buffer(4 * 64), buffer(64));
  assert.equal(solver.executionMode, "persistent-small-domain");
  assert.equal(persistentCalls, 1);
  assert.equal(dispatches, 5);
  assert.equal(solver.encodedDispatchCount, 5);
  assert.equal(passes, 2);
  assert.equal(allocations, 1, "small-domain host ownership is only the shared solve-control buffer");
  assert.equal(pipelines, 1, "small-domain construction compiles only the mandatory native-L2 authority gate");
  solver.destroy();
});

test("Section 4.3 hybrid is the only preconditioner and requires an explicit SPD L1 V-cycle", () => {
  const hybrid = planOctreeMGPCG({ dimensions: [64, 48, 32], rowCapacity: 10_000 });
  assert.equal(hybrid.hybridBytes, hybrid.vectorBytes * 6);
  assert.throws(() => new WebGPUOctreeMGPCG({} as GPUDevice, {
    leafHeaders: {} as GPUBuffer, leafEntries: {} as GPUBuffer, rowCount: {} as GPUBuffer,
    firstOrderVCycle: undefined as never,
  }, { dimensions: [64, 48, 32], rowCapacity: 10_000 }),
  /requires an explicit SPD first-order V-cycle/);
});

test("MGPCG requires the stable device topology-reuse tail publication", () => {
  const shader = octreeMGPCGShader.replace(/\s+/g, "");
  assert.match(shader,
    /letcountWords=arrayLength\(&counts\);if\(countWords<8u\|\|counts\[countWords-8u\]!=0u\)/,
    "PCG must consume the stable structural-failure tail outside recurring scan/task scratch");
});

test("Section 4.3 hybrid has a three-layer band and symmetry-locked paired L2 smoothing", () => {
  assert.equal(OCTREE_SECTION43_BOUNDARY_BAND_LAYERS, 3);
  assert.equal(OCTREE_SECTION43_BOUNDARY_SMOOTHING_ITERATIONS, 8);
  assert.equal(OCTREE_SECTION43_SMALL_DOMAIN_PCG_ITERATIONS, 12);
  assert.equal(octreeSection43IterationBudget(64), 12,
    "one-row-per-lane fixtures retain the exact persistent PCG budget");
  assert.equal(octreeSection43IterationBudget(16 ** 3), 10,
    "normal production capacities use bounded parallel PCG");
  assert.equal(octreeSection43IterationBudget(OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY + 1), 10,
    "capacities beyond the persistent lane select bounded parallel PCG");
  assert.throws(() => octreeSection43IterationBudget(0), /positive integer/);
  assert.equal(normalizeOctreeSection43BoundarySmoothing(undefined), 8);
  assert.equal(normalizeOctreeSection43BoundarySmoothing(1), 2);
  assert.equal(normalizeOctreeSection43BoundarySmoothing(7), 8,
    "odd inputs must round to an even ping/pong schedule");
  assert.equal(normalizeOctreeSection43BoundarySmoothing(32), 16);
  assert.match(octreeMGPCGShader, /boundaryGap=h\.diagonal-offDiagonalSum/);
  assert.match(octreeMGPCGShader, /\(h\.pad0&ROW_BOUNDARY\)!=0u\|\|boundaryGap/,
    "closed and cut solid rows must enter the paper's boundary smoother even without a Dirichlet gap");
  assert.match(octreePowerOperatorShader,
    /\(face\.flags&\(BOUNDARY\|OPEN_BOUNDARY\)\)!=0u\|\|face\.openFraction<1\.0[\s\S]*arena\[base\+3u\]=rowFlags/,
    "authoritative face assembly must publish explicit boundary incidence for MGPCG");
  assert.match(octreeMGPCGShader, /headers\[e\.row\]\.size!=h\.size/);
  assert.match(octreeMGPCGShader, /dilateHybridBandAtoB/);
  assert.match(octreeMGPCGShader, /dilateHybridBandBtoA/);
  assert.match(octreeMGPCGShader,
    /clearHybridPreconditioner[\s\S]*row>=liveRows\(\)\|\|stopped\(\)/,
    "PCG preconditioning must not clear row-capacity hybrid vectors");
  assert.match(octreeMGPCGShader, /formHybridL1Residual/);
  assert.match(octreeMGPCGShader, /addHybridL1Correction/);
  const source = WebGPUOctreeMGPCG.toString();
  assert.match(source, /firstOrderVCycle|encodeCorrection/);
  assert.match(source, /boundarySmoothingIterations/);
  assert.doesNotMatch(WebGPUOctreeProjection.toString(), /exact three-voxel|three-voxel paper band/,
    "the visible solver must not describe graph dilation as an exact paper band");
});

test("Section 4.3 Jacobi--M1--Jacobi composition is linear, symmetric, and positive", () => {
  // Deliberately use unequal diagonals and a disconnected smoothing mask: a
  // diagonal-only or all-domain oracle would not exercise the paper's banded
  // L2 relaxation argument. L1 is a different SPD first-order operator.
  const l2 = [
    [5, -1, 0, 0, 0], [-1, 6, -2, 0, 0], [0, -2, 7, -1, 0],
    [0, 0, -1, 5, -1], [0, 0, 0, -1, 3],
  ];
  const l1 = [
    [4, -1, 0, 0, 0], [-1, 5, -1, 0, 0], [0, -1, 4, -1, 0],
    [0, 0, -1, 4, -1], [0, 0, 0, -1, 3],
  ];
  const band = [true, true, false, true, false];
  const x = [0.7, -1.3, 0.25, 2.1, -0.4], y = [-0.2, 0.9, 1.7, -0.6, 0.3];
  const mx = applySection43Oracle(l2, l1, band, x);
  const my = applySection43Oracle(l2, l1, band, y);
  assert.ok(Math.abs(dot(x, my) - dot(y, mx)) < 1e-11,
    "matching pre/post Jacobi around an SPD M1 must produce a symmetric map");
  assert.ok(dot(x, mx) > 0 && dot(y, my) > 0,
    "the Section 4.3 map must remain a valid positive PCG preconditioner");
  const combined = applySection43Oracle(l2, l1, band, x.map((value, row) => value - 0.37 * y[row]));
  assert.ok(combined.every((value, row) => Math.abs(value - mx[row] + 0.37 * my[row]) < 1e-11),
    "the fixed hybrid schedule must be linear");
});

test("large-domain command stream contains only bounded parallel PCG", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8 } });
  let dispatches = 0, corrections = 0, currentStage = ""; const events: string[] = [];
  const buffer = (size: number) => ({ size, usage: 7, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} }, createBuffer: ({ size }: { size: number }) => buffer(size),
    createShaderModule: () => ({}),
    createComputePipeline: ({ compute }: { compute: { entryPoint: string } }) => ({
      entryPoint: compute.entryPoint, getBindGroupLayout: () => ({}),
    }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const setupDispatches = 2, correctionDispatches = 3;
  const cycle = {
    operatorOrder: 1 as const, isSymmetricPositiveDefinite: true as const, allocatedBytes: 0,
    encodedCorrectionDispatchCount: correctionDispatches,
    encodedSetupDispatchCount: setupDispatches,
    encodeSetup(broker: PassBroker) { const pass = broker.compute();
      for (let i = 0; i < setupDispatches; i += 1) pass.dispatchWorkgroups(1);
      broker.fence("test setup publication"); },
    encodeCorrection(broker: PassBroker) {
      corrections += 1; const pass = broker.compute();
      for (let i = 0; i < correctionDispatches; i += 1) pass.dispatchWorkgroups(1);
    },
  };
  const capacity = OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY + 1;
  const solver = new WebGPUOctreeMGPCG(device, {
    leafHeaders: buffer(48 * capacity), leafEntries: buffer(8 * capacity), rowCount: buffer(64),
    firstOrderVCycle: cycle,
  }, { dimensions: [128, 64, 32], rowCapacity: capacity });
  const encoder = { clearBuffer() {}, beginComputePass: () => ({
    setPipeline(pipeline: { entryPoint: string }) { currentStage = pipeline.entryPoint; }, setBindGroup() {},
    dispatchWorkgroups() { dispatches += 1; events.push(currentStage); }, end() {},
  }) } as unknown as GPUCommandEncoder;
  solver.encode(new PassBroker(encoder), buffer(4 * capacity), buffer(4 * capacity), buffer(64));
  assert.equal(solver.executionMode, "staged-pcg-large-domain");
  assert.equal(dispatches, solver.encodedDispatchCount,
    "the reported fixed schedule must include every hierarchy and V-cycle dispatch");
  assert.equal(corrections, OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS,
    "M must run for the initial residual and between bounded PCG iterations");
  assert.equal(events.filter((stage) => stage === "classifyHybridBand").length, 1,
    "the fixed Section 4.3 band is prepared once and reused by every PCG preconditioner");
  assert.equal(events.filter((stage) => stage === "preparePCGStep").length,
    OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS);
  assert.equal(events.filter((stage) => stage === "updatePCGStateAndReduceResidual").length,
    OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS);
  assert.equal(events.filter((stage) => stage === "finishPCGIteration").length,
    OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS - 1);
  assert.equal(events.some((stage) => /Chebyshev|Spectrum|Power/.test(stage)), false,
    "the encoded large-domain command stream must have no polynomial or spectral-estimation tail");
  assert.equal(solver.iterationBudget, OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS,
    "diagnostics must report the actual bounded PCG budget");
  solver.destroy();
});

test("Section 4.3 remains a separately selected pressure implementation", () => {
  const source = WebGPUOctreeProjection.toString();
  assert.match(source, /this\.pressureSolverMode\s*=\s*options\.powerPressureSolver/);
  assert.match(source, /if\s*\(this\.pressureSolverMode\s*===\s*"galerkin"\)/);
  assert.match(source, /new WebGPUOctreeSPGridVCycle/);
  assert.doesNotMatch(source, /WebGPUOctreeFirstOrderVCycle|aggregate-galerkin/);
  assert.match(WebGPUOctreeProjection.prototype.encode.toString(), /firstOrderVCycle\?\.encodeCapture\(pressureBroker\)/,
    "L1 rows must be captured before power publication replaces the shared CSR");
});

test("MGPCG rejects missing or mismatched native L2 authority before either solve lane", () => {
  assert.match(octreeMGPCGAuthorityGateShader,
    /\(flags & ASSEMBLED\) != 0u && \(flags & ~\(ASSEMBLED \| PROJECTED\)\) == 0u/);
  assert.match(octreeMGPCGAuthorityGateShader,
    /authorityControl\[2\] != rowCounts\[0\][\s\S]*atomicOr\(&solverControl\[0\], INVALID_ROW_ERROR\)/);
  const encodeSource = WebGPUOctreeMGPCG.prototype.encode.toString();
  const clear = encodeSource.indexOf("broker.clearBuffer(this.control)");
  const gate = encodeSource.indexOf("authorityGate.dispatchWorkgroups(1)", clear);
  const setup = encodeSource.indexOf("this.source.firstOrderVCycle.encodeSetup", gate);
  assert.ok(clear >= 0 && gate > clear && setup > gate,
    "the native L2 gate must poison solver control before L1 setup or pressure arithmetic");
});

test("large-domain Section 4.3 PCG uses GPU-only fail-closed convergence", () => {
  assert.match(octreeMGPCGShader, /value-=e\.coefficient\*fieldValue/);
  assert.doesNotMatch(octreeMGPCGShader, /restrictResidual|prolongateCorrection|buildHierarchyMap/);
  assert.match(octreeMGPCGShader, /atomicStore\(&control\[1\],1u\)/);
  assert.match(octreeMGPCGShader, /fn updatePCGStateAndReduceResidual/);
  const encodeSource = WebGPUOctreeMGPCG.prototype.encode.toString();
  const update = encodeSource.indexOf("single(this.stages.preparePCGStep)");
  const residualGate = encodeSource.indexOf("single(this.stages.updatePCGState)", update);
  const finalize = encodeSource.indexOf("single(this.stages.finalize)", residualGate);
  assert.ok(update >= 0 && update < residualGate && residualGate < finalize,
    "PCG must test its updated residual before fail-closed publication");
  assert.equal(octreeMGPCGShader.match(/atomicAdd\(&control\[2\],1u\)/g)?.length, 1,
    "each completed PCG update must increment the device iteration count once");
  assert.doesNotMatch(octreeMGPCGShader, /Chebyshev|spectrum/,
    "bounded parallel PCG must not retain the retired polynomial path");
  assert.doesNotMatch(WebGPUOctreeProjection.prototype.encode.toString(), /mapAsync|getMappedRange/);
  assert.doesNotMatch(octreeMGPCGShader, /hierarchy:array|aggregateKey|solveCoarseAggregates/);
});

test("power projection publication is gated by selected solver success", () => {
  assert.match(octreePowerOperatorShader, /preparePowerProjectionMGPCG/);
  assert.match(octreePowerOperatorShader, /solverControl\[0\]!=0u/);
  assert.match(octreePowerOperatorShader, /solverControl\[1\]==0u/);
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode, /if\s*\(this\.galerkin\)[\s\S]*else if\s*\(this\.mgpcg\)/);
  assert.doesNotMatch(encode, /catch[\s\S]*(?:mgpcg|galerkin)\.encode/,
    "neither explicit mode may fall back to the other");
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
  const capacity = OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY + 1;
  const solver = new WebGPUOctreeMGPCG(device, {
    leafHeaders: buffer(48 * capacity), leafEntries: buffer(8 * capacity), rowCount: buffer(64),
    firstOrderVCycle: {
      operatorOrder: 1, isSymmetricPositiveDefinite: true, allocatedBytes: 0,
      encodedCorrectionDispatchCount: 1, encodedSetupDispatchCount: 1,
      encodeSetup(broker) { broker.compute().dispatchWorkgroups(1); broker.fence("test setup publication"); },
      encodeCorrection(broker) { broker.compute().dispatchWorkgroups(1); },
    },
  }, { dimensions: [129, 64, 1], rowCapacity: capacity });
  const encoder = {
    clearBuffer() {},
    beginComputePass: () => { passes += 1; return { setPipeline() {}, setBindGroup() {},
      dispatchWorkgroups() { dispatches += 1; }, end() {} }; },
  } as unknown as GPUCommandEncoder;
  const pressureA = buffer(4 * capacity), pressureB = buffer(4 * capacity);
  const authorityControl = buffer(64);
  solver.encode(new PassBroker(encoder), pressureA, pressureB, authorityControl);
  const firstGroups = bindGroups, firstPasses = passes, firstDispatches = dispatches;
  assert.equal(firstPasses, solver.encodedPassTransitionCount,
    "hierarchy publication and the pressure solve must use separate ordered passes");
  assert.equal(firstDispatches, solver.encodedDispatchCount,
    "reported pressure dispatch count must equal the command stream actually emitted");
  solver.encode(new PassBroker(encoder), pressureA, pressureB, authorityControl);
  assert.equal(passes, firstPasses * 2, "a second replay should add the same two pass transitions");
  assert.ok(firstDispatches > firstGroups * 10, `${firstDispatches} dispatches should share ${firstGroups} descriptors`);
  assert.equal(bindGroups, firstGroups, "a second fixed replay must allocate no bind groups");
  solver.destroy();
});

test("large-domain PCG auto-layout lists exactly match transitive WGSL resources", () => {
  const functions = new Map<string, string>();
  const declaration = /\bfn\s+([A-Za-z_]\w*)\s*\(/g;
  for (let match = declaration.exec(octreeMGPCGShader); match; match = declaration.exec(octreeMGPCGShader)) {
    const open = octreeMGPCGShader.indexOf("{", match.index);
    assert.notEqual(open, -1, `missing body for ${match[1]}`);
    let depth = 0, end = -1;
    for (let index = open; index < octreeMGPCGShader.length; index += 1) {
      if (octreeMGPCGShader[index] === "{") depth += 1;
      if (octreeMGPCGShader[index] === "}") depth -= 1;
      if (depth === 0) { end = index + 1; break; }
    }
    assert.notEqual(end, -1, `unterminated body for ${match[1]}`);
    functions.set(match[1], octreeMGPCGShader.slice(open, end));
    declaration.lastIndex = end;
  }
  const resources = new Map<string, number>();
  for (const match of octreeMGPCGShader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  )) resources.set(match[2], Number(match[1]));
  const transitiveBindings = (entryPoint: string) => {
    const reached = new Set<string>(), bindings = new Set<number>();
    const visit = (name: string) => {
      if (reached.has(name)) return;
      reached.add(name);
      const body = functions.get(name);
      assert.ok(body, `missing transitive WGSL function ${name}`);
      for (const [resource, binding] of resources) {
        if (new RegExp(`\\b${resource}\\b`).test(body)) bindings.add(binding);
      }
      for (const candidate of functions.keys()) {
        if (candidate !== name && new RegExp(`\\b${candidate}\\s*\\(`).test(body)) visit(candidate);
      }
    };
    visit(entryPoint);
    return [...bindings].sort((left, right) => left - right);
  };
  const declared = new Map<string, number[]>();
  for (const match of WebGPUOctreeMGPCG.toString().matchAll(
    /pipeline\("([^"]+)",\s*\[([^\]]*)\]\)/g,
  )) {
    declared.set(match[1], match[2].split(",").map(Number).sort((left, right) => left - right));
  }
  assert.equal(declared.size, 17, "every large-domain PCG entry point must declare one immutable binding list");
  for (const [entryPoint, bindings] of declared) {
    assert.deepEqual(bindings, transitiveBindings(entryPoint),
      `${entryPoint} must bind exactly its transitively reachable auto-layout resources`);
  }
  assert.deepEqual(declared.get("formInitialResidual"), [0, 1, 2, 3, 5, 7, 11, 12],
    "applyA reaches fieldValue's direction binding even when this entry passes useDirection=false");

  const authorityBindings = [...octreeMGPCGAuthorityGateShader.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var(?:<[^>]+>)?\s+([A-Za-z_]\w*)/g,
  )].map((match) => Number(match[1])).sort((left, right) => left - right);
  assert.deepEqual(authorityBindings, [0, 1, 2],
    "the native-L2 authority gate has exactly its control, row-count, and solver-control resources");
  assert.match(WebGPUOctreeMGPCG.toString(),
    /entries:\s*\[\s*\{\s*binding:\s*0[\s\S]*\{\s*binding:\s*1[\s\S]*\{\s*binding:\s*2/,
    "the authority-gate auto layout receives every declared transitive resource");
});

test("Dawn accepts the authoritative large-domain PCG entry points", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  device.pushErrorScope("validation");
  const shaderModule = device.createShaderModule({ code: octreeMGPCGShader });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  assert.deepEqual(errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
  for (const entryPoint of [
    "initializeMGPCG", "formInitialResidual", "clearHybridPreconditioner",
    "classifyHybridBand", "dilateHybridBandAtoB", "dilateHybridBandBtoA", "smoothHybridAtoB",
    "smoothHybridBtoA", "formHybridL1Residual", "addHybridL1Correction", "publishHybridPreconditioner",
    "initializePCGReduction", "preparePCGStep", "updatePCGStateAndReduceResidual", "finishPCGIteration",
    "finalizeMGPCG", "publishMGPCG",
  ]) device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message); device.destroy();
});

test("Dawn large-domain PCG solves a known SPD system", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU solve checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const capacity = OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY + 1;
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const makeBuffer = (size: number, data?: Uint32Array<ArrayBuffer>) => {
    const result = device.createBuffer({ size, usage: storage }); if (data) device.queue.writeBuffer(result, 0, data); return result;
  };
  const headerWords = new Uint32Array(capacity * 12), headerFloats = new Float32Array(headerWords.buffer);
  for (let row = 0; row < capacity; row += 1) {
    const base = row * 12; headerWords[base] = row; headerWords[base + 3] = 1;
    headerFloats[base + 4] = 2; headerFloats[base + 5] = -1;
  }
  const counts = new Uint32Array(16); counts[0] = capacity;
  const leafHeaders = makeBuffer(headerWords.byteLength, headerWords), leafEntries = makeBuffer(8);
  const rowCount = makeBuffer(counts.byteLength, counts), pressureIn = makeBuffer(capacity * 4);
  const pressureOut = makeBuffer(capacity * 4);
  const authorityWords = new Uint32Array(16);
  authorityWords[0] = 0x8000_0000;
  authorityWords[2] = capacity;
  const authorityControl = makeBuffer(authorityWords.byteLength, authorityWords);
  const copyModule = device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage,read> rhs:array<f32>;
    @group(0) @binding(1) var<storage,read_write> correction:array<f32>;
    @group(0) @binding(2) var<storage,read> live:array<u32>;
    @compute @workgroup_size(64) fn copyRhs(@builtin(global_invocation_id) gid:vec3u){
      if(gid.x<live[0]){correction[gid.x]=rhs[gid.x];}
    }` });
  const copyPipeline = device.createComputePipeline({ layout: "auto", compute: { module: copyModule, entryPoint: "copyRhs" } });
  const cycle = {
    operatorOrder: 1 as const, isSymmetricPositiveDefinite: true as const, allocatedBytes: 0,
    encodedCorrectionDispatchCount: 1, encodedSetupDispatchCount: 0,
    encodeSetup(broker: PassBroker) { broker.compute(); broker.fence("test setup publication"); },
    encodeCorrection(broker: PassBroker, input: { rhs: GPUBuffer; correction: GPUBuffer }) {
      const group = device.createBindGroup({ layout: copyPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: input.rhs } }, { binding: 1, resource: { buffer: input.correction } },
        { binding: 2, resource: { buffer: rowCount } },
      ] });
      const pass = broker.compute();
      pass.setPipeline(copyPipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(capacity / 64));
    },
  };
  const solver = new WebGPUOctreeMGPCG(device, {
    leafHeaders, leafEntries, rowCount, firstOrderVCycle: cycle,
  }, {
    dimensions: [129, 64, 1], rowCapacity: capacity,
    boundarySmoothingIterations: 2, relativeTolerance: 1e-4,
  });
  assert.equal(solver.executionMode, "staged-pcg-large-domain");
  const pressureReadback = device.createBuffer({ size: capacity * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const controlReadback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.pushErrorScope("validation"); const encoder = device.createCommandEncoder();
  solver.encode(new PassBroker(encoder), pressureIn, pressureOut, authorityControl);
  encoder.copyBufferToBuffer(pressureOut, 0, pressureReadback, 0, capacity * 4);
  encoder.copyBufferToBuffer(solver.control, 0, controlReadback, 0, 64);
  device.queue.submit([encoder.finish()]);
  await Promise.all([pressureReadback.mapAsync(GPUMapMode.READ), controlReadback.mapAsync(GPUMapMode.READ)]);
  const pressure = new Float32Array(pressureReadback.getMappedRange());
  const control = new Uint32Array(controlReadback.getMappedRange());
  const diagnostics = new Float32Array(control.buffer, control.byteOffset, control.length);
  const validationError = await device.popErrorScope(); assert.equal(validationError, null, validationError?.message);
  assert.equal(control[0], 0, `device failure flag ${control[0]} at stage ${control[10]}, row ${control[11]}`);
  assert.equal(control[1], 1, "PCG must satisfy the device residual gate");
  assert.ok(control[2] >= 1 && control[2] <= OCTREE_SECTION43_LARGE_DOMAIN_PCG_ITERATIONS,
    `unexpected device iteration count ${control[2]}`);
  assert.ok(diagnostics[4] <= 1e-8 * diagnostics[5], `accepted rr=${diagnostics[4]}, bb=${diagnostics[5]}`);
  for (const row of [0, 1, Math.floor(capacity / 2), capacity - 1]) assert.ok(Math.abs(pressure[row] - 0.5) < 2e-4,
    `row ${row} produced ${pressure[row]} instead of the exact 0.5 pressure`);
  pressureReadback.unmap(); controlReadback.unmap();
  solver.destroy(); for (const buffer of [leafHeaders, leafEntries, rowCount, pressureIn, pressureOut,
    pressureReadback, controlReadback]) buffer.destroy(); device.destroy();
});
