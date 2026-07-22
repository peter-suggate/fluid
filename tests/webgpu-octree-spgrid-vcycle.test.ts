import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  SPGRID_CELL_FLAG,
  SPGRID_MAXIMUM_ROW_CAPACITY,
  WebGPUOctreeSPGridVCycle,
  buildSPGridPyramidOracle,
  octreeSPGridVCycleShader,
  planOctreeSPGridVCycle,
  prolongSPGrid,
  restrictSPGrid,
  solveSPGridBottomLDLT,
} from "../lib/webgpu-octree-spgrid-vcycle";

const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, value, index) => sum + value * b[index], 0);

test("native sparse pyramid allocation is bounded by row capacity, not dense domain volume", () => {
  const narrow = planOctreeSPGridVCycle({ dimensions: [64, 48, 32], rowCapacity: 5_000 });
  const wide = planOctreeSPGridVCycle({ dimensions: [1_024, 48, 32], rowCapacity: 5_000 });
  assert.equal(narrow.levelStride, wide.levelStride);
  assert.equal(narrow.transferStride, 40_000);
  assert.ok(wide.levelCount > narrow.levelCount);
  assert.equal(wide.stateBytes, 14 * wide.levelCount * wide.levelStride * 4);
  assert.ok(!("cellCount" in wide));
  assert.throws(() => planOctreeSPGridVCycle({ dimensions: [16, 16, 16],
    rowCapacity: SPGRID_MAXIMUM_ROW_CAPACITY + 1 }), /bounded/);
});

test("native pyramid distinguishes active, ghost, and multigrid-only cells", () => {
  const pyramid = buildSPGridPyramidOracle([
    { origin: [0, 0, 0], size: 2 },
    { origin: [2, 0, 0], size: 1 },
  ], 3);
  assert.ok(pyramid.levels[0].flags.some((flag) => (flag & SPGRID_CELL_FLAG.ghost) !== 0));
  assert.ok(pyramid.levels[0].flags.some((flag) => (flag & SPGRID_CELL_FLAG.active) !== 0));
  assert.ok(pyramid.levels[1].flags.some((flag) => (flag & SPGRID_CELL_FLAG.multigridOnly) !== 0));
  assert.ok(pyramid.transfers[0].some((record) => record.weight === 1), "persisting ghost cells copy exactly");
  assert.ok(pyramid.transfers[0].some((record) => record.weight > 0 && record.weight < 1), "finest cells use trilinear records");
  for (const level of pyramid.levels) for (const flags of level.flags) {
    const classes = [SPGRID_CELL_FLAG.active, SPGRID_CELL_FLAG.ghost, SPGRID_CELL_FLAG.multigridOnly]
      .filter((classification) => (flags & classification) !== 0);
    assert.equal(classes.length, 1, "every sparse cell has one exclusive storage class");
  }
});

test("stored trilinear restriction and prolongation are exact adjoints", () => {
  const pyramid = buildSPGridPyramidOracle([
    { origin: [1, 1, 1], size: 1 }, { origin: [3, 1, 1], size: 1 }, { origin: [0, 2, 0], size: 2 },
  ], 3);
  const records = pyramid.transfers[0], fine = pyramid.levels[0].coordinates.map((_, i) => 0.25 + i * 0.7);
  const coarse = pyramid.levels[1].coordinates.map((_, i) => -1.1 + i * 0.31);
  const restricted = restrictSPGrid(fine, records, coarse.length);
  const prolonged = prolongSPGrid(coarse, records, fine.length);
  assert.ok(Math.abs(dot(restricted, coarse) - dot(fine, prolonged)) < 1e-12);
  for (let fineIndex = 0; fineIndex < fine.length; fineIndex += 1) {
    const rowSum = records.filter((record) => record.fine === fineIndex).reduce((sum, record) => sum + record.weight, 0);
    assert.ok(Math.abs(rowSum - 1) < 1e-12, `transfer row ${fineIndex} must preserve constants`);
  }
});

test("fixed LDLT bottom operation is exact, linear, symmetric, and positive", () => {
  const operator = [[4, -1, 0], [-1, 4, -1], [0, -1, 3]];
  const x = [1, -2, 0.5], y = [-0.25, 3, 2];
  const mx = solveSPGridBottomLDLT(operator, x), my = solveSPGridBottomLDLT(operator, y);
  const product = operator.map((row) => dot(row, mx));
  assert.deepEqual(product.map((value, i) => Math.abs(value - x[i]) < 1e-12), [true, true, true]);
  assert.ok(Math.abs(dot(x, my) - dot(y, mx)) < 1e-12);
  assert.ok(dot(x, mx) > 0 && dot(y, my) > 0);
  const sum = solveSPGridBottomLDLT(operator, x.map((value, i) => value + 2 * y[i]));
  assert.ok(sum.every((value, i) => Math.abs(value - mx[i] - 2 * my[i]) < 1e-12));
  assert.throws(() => solveSPGridBottomLDLT([[0]], [1]), /not SPD/);
});

test("GPU source stores one transfer record and consumes it in both adjoint directions", () => {
  assert.match(octreeSPGridVCycleShader, /appendTransfer\(l,fine,c,weight\)/);
  assert.match(octreeSPGridVCycleShader, /fn restrictResidual/);
  assert.match(octreeSPGridVCycleShader, /fn prolongCorrection/);
  assert.match(octreeSPGridVCycleShader, /w\*loadf\(RESIDUAL,l,f\)/);
  assert.match(octreeSPGridVCycleShader, /w\*loadf\(A,l\+1u,c\)/);
  assert.match(octreeSPGridVCycleShader, /const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u/);
  assert.match(octreeSPGridVCycleShader, /fn mergeClass/);
  assert.match(octreeSPGridVCycleShader, /fn smoothable/);
  assert.match(octreeSPGridVCycleShader, /fn exactBottom/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /while\s*\([^)]*atomicLoad/, "no cross-workgroup spin barrier is permitted");
});

test("one correction uses one compute-pass transition, ordered dispatches, and cached descriptors", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  let passes = 0, dispatches = 0, groups = 0;
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} }, createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}), createComputePipeline: ({ label }: { label: string }) => ({ label, getBindGroupLayout: () => ({}) }),
    createBindGroup: () => { groups += 1; return {}; },
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, { leafHeaders: buffer(48 * 128), leafEntries: buffer(8 * 512) },
    { dimensions: [16, 16, 16], rowCapacity: 128, maximumLevels: 5, finestCellWidth: 1 });
  const encoder = {
    clearBuffer() {}, copyBufferToBuffer() {}, beginComputePass: () => {
      passes += 1; return { setPipeline() {}, setBindGroup() {},
        dispatchWorkgroups() { dispatches += 1; }, dispatchWorkgroupsIndirect() { dispatches += 1; }, end() {} };
    },
  } as unknown as GPUCommandEncoder;
  const input = { rowCount: buffer(64), solverControl: buffer(64), rhs: buffer(512), correction: buffer(512) };
  cycle.encodeSetup(encoder, input);
  assert.equal(dispatches, cycle.encodedSetupDispatchCount);
  const setupPasses = passes, before = dispatches;
  cycle.encodeCorrection(encoder, input);
  assert.equal(passes - setupPasses, 1);
  assert.equal(dispatches - before, cycle.encodedCorrectionDispatchCount);
  assert.equal(cycle.encodedCorrectionPassCount, cycle.encodedCorrectionDispatchCount,
    "legacy pass-count accounting means ordered dispatches");
  assert.equal(cycle.encodedPassTransitionCount, 1);
  assert.equal(cycle.diagnostics.bottomOperation, "exact-single-cell");
  assert.equal(cycle.diagnostics.coarsestDegreesOfFreedom, 1);
  const firstGroups = groups;
  cycle.encodeCorrection(encoder, input);
  assert.equal(groups, firstGroups, "repeated solves allocate no new bind groups");
  cycle.destroy();
});

test("shared-pass ABI emits no nested pass or command-encoder clear", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} },
    createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage), createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, { leafHeaders: buffer(48 * 32), leafEntries: buffer(256) },
    { dimensions: [8, 8, 8], rowCapacity: 32, finestCellWidth: 1 });
  let nested = 0, clears = 0, dispatches = 0;
  const encoder = { beginComputePass() { nested += 1; throw new Error("nested pass"); }, clearBuffer() { clears += 1; } } as unknown as GPUCommandEncoder;
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { dispatches += 1; },
    dispatchWorkgroupsIndirect() { dispatches += 1; }, end() {} } as unknown as GPUComputePassEncoder;
  const input = { rowCount: buffer(64), solverControl: buffer(64), rhs: buffer(128), correction: buffer(128) };
  cycle.encodeSetup(encoder, input, pass); const setupDispatches = dispatches;
  cycle.encodeCorrection(encoder, input, pass);
  assert.equal(nested, 0); assert.equal(clears, 0);
  assert.equal(dispatches - setupDispatches, cycle.encodedCorrectionDispatchCount);
  cycle.destroy();
});

test("Dawn accepts the native sparse V-cycle shader", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice();
  device.pushErrorScope("validation");
  const shaderModule = device.createShaderModule({ code: octreeSPGridVCycleShader });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  assert.deepEqual(errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
  for (const entryPoint of ["clearTopology", "clearState", "clearDispatch", "clearCorrection", "emitCells",
    "buildTransfers", "buildStencil", "ensureDiagonal", "finalizeIndirect", "zeroVectors", "seedRhs", "applyA",
    "applyB", "jacobiAtoB", "jacobiBtoA", "formResidual", "restrictResidual", "exactBottom",
    "prolongCorrection", "publish"]) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message); device.destroy();
});
