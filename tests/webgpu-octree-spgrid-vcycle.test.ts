import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_SPGRID_VCYCLE_BINDINGS,
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
  assert.equal(wide.stateBytes, 27 * wide.levelCount * wide.levelStride * 4);
  assert.equal(wide.dispatchBytes, wide.levelCount * 32 + 32);
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
  assert.match(octreeSPGridVCycleShader, /clearCorrection[\s\S]*r<rows\(\)&&!stopped\(\)/,
    "post-convergence correction clears must be write-free");
  assert.match(octreeSPGridVCycleShader, /zeroVectors[\s\S]*i>=count\(l\)\|\|stopped\(\)/,
    "post-convergence sparse-level clears must be write-free");
  const bindings = [...octreeSPGridVCycleShader.matchAll(/@group\(0\) @binding\((\d+)\)/g)]
    .map((match) => Number(match[1]));
  assert.equal(new Set(bindings).size, bindings.length,
    "every WGSL binding must have exactly one module-scope declaration");
  assert.match(octreeSPGridVCycleShader, /appendTransfer\(l,fine,c,weight\)/);
  assert.match(octreeSPGridVCycleShader, /fn restrictResidual/);
  assert.match(octreeSPGridVCycleShader, /fn prolongCorrection/);
  assert.match(octreeSPGridVCycleShader, /fn ghostValueAccumulate/);
  assert.match(octreeSPGridVCycleShader, /fn ghostValuePropagate/);
  assert.match(octreeSPGridVCycleShader, /if\(w!=1\.0\).*atomicAddF\(at\(RHS,l\+1u,c\),loadf\(RESIDUAL,l,f\)\)/s,
    "GhostValueAccumulate applies E transpose through a unit copy record");
  assert.match(octreeSPGridVCycleShader, /if\(w!=1\.0\).*storef\(A,l,f,loadf\(A,l\+1u,c\)\)/s,
    "GhostValuePropagate applies E through the same unit copy record");
  assert.match(octreeSPGridVCycleShader, /w\*loadf\(RESIDUAL,l,f\)/);
  assert.match(octreeSPGridVCycleShader, /w\*loadf\(A,l\+1u,c\)/);
  assert.match(octreeSPGridVCycleShader, /const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u/);
  assert.match(octreeSPGridVCycleShader, /fn mergeClass/);
  assert.match(octreeSPGridVCycleShader, /fn contactCoord/);
  assert.match(octreeSPGridVCycleShader, /fn insertOwned/);
  assert.match(octreeSPGridVCycleShader, /fn emitGhostAliases/);
  assert.match(octreeSPGridVCycleShader, /const XYPP=9u.*const YZPP=17u/s,
    "the production stencil retains all twelve directed octree-edge contacts");
  assert.match(octreeSPGridVCycleShader, /let edgeChannels=array<u32,12>/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /select\([^\n]*insert\([^\n]*insertOwned/,
    "WGSL select eagerly evaluates both insertion paths");
  assert.match(octreeSPGridVCycleShader, /fn smoothable/);
  assert.match(octreeSPGridVCycleShader, /fn exactBottom/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /while\s*\([^)]*atomicLoad/, "no cross-workgroup spin barrier is permitted");
});

test("every SPGrid auto-layout binds the complete reachable resource ABI", () => {
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.resetInvalidBuffers, [4, 5],
    "the conditional cold reset reaches only the topology and state arenas");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.clearCorrection, [0, 3, 7, 9],
    "correction clearing observes the solver stop gate before writing output");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.zeroVectors, [0, 4, 5, 6, 7],
    "vector clearing observes the solver stop gate before touching sparse slots");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.seedRhs, [0, 1, 3, 4, 5, 7, 8],
    "native-level RHS seeding reads the captured leaf size and must bind headers");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.publish, [0, 1, 3, 4, 5, 7, 9],
    "native-level correction publication likewise reads the captured leaf size");
  for (const [entryPoint, bindings] of Object.entries(OCTREE_SPGRID_VCYCLE_BINDINGS)) {
    assert.equal(new Set(bindings).size, bindings.length, `${entryPoint} must not bind a resource twice`);
    assert.ok(bindings.length <= 10, `${entryPoint} exceeds the portable storage-buffer stage budget`);
  }
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

test("setup retires the prior live generation without unconditional full-buffer clears", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const created: Array<{ label?: string; size: number; usage: number; buffer: GPUBuffer }> = [];
  const buffer = (size: number, usage = 31, label?: string) => ({ size, usage, label, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} },
    createBuffer: ({ label, size, usage }: { label?: string; size: number; usage: number }) => {
      const gpuBuffer = buffer(size, usage, label); created.push({ label, size, usage, buffer: gpuBuffer }); return gpuBuffer;
    }, createShaderModule: () => ({}),
    createComputePipeline: ({ label }: { label: string }) => ({ label, getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, { leafHeaders: buffer(48 * 32), leafEntries: buffer(256) },
    { dimensions: [8, 8, 8], rowCapacity: 32, finestCellWidth: 1 });
  const events: string[] = []; let dispatches = 0, current = "";
  const encoder = {
    clearBuffer() { throw new Error("warm SPGrid setup must not clear full buffers"); },
    beginComputePass() { events.push("begin"); return {
      setPipeline(pipeline: { label: string }) { current = pipeline.label.replace("SPGrid V-cycle · ", ""); }, setBindGroup() {},
      dispatchWorkgroups() { dispatches += 1; events.push(`direct:${current}`); },
      dispatchWorkgroupsIndirect(source: GPUBuffer, offset: number) {
        dispatches += 1; events.push(`indirect:${current}:${offset}:${String((source as GPUBuffer & { label?: string }).label)}`);
      },
      end() { events.push("end"); } }; },
    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number) {
      assert.equal(sourceOffset, 0); assert.equal(destinationOffset, 0); assert.equal(size, cycle.plan.dispatchBytes);
      events.push(`copy:${String((source as GPUBuffer & { label?: string }).label)}->${String((destination as GPUBuffer & { label?: string }).label)}`);
    },
  } as unknown as GPUCommandEncoder;
  const input = { rowCount: buffer(64), solverControl: buffer(64), rhs: buffer(128), correction: buffer(128) };
  cycle.encodeSetup(encoder, input);
  assert.equal(dispatches, cycle.encodedSetupDispatchCount);
  assert.equal(events[0], "begin");
  assert.equal(events[1], `indirect:resetInvalidBuffers:${cycle.plan.levelCount * 32 + 8}:SPGrid live indirect dispatches`);
  assert.deepEqual(events.slice(2, 2 + cycle.plan.levelCount), Array.from({ length: cycle.plan.levelCount }, (_, level) =>
    `indirect:retireSlots:${level * 32 + 8}:SPGrid live indirect dispatches`));
  assert.equal(events[2 + cycle.plan.levelCount], "direct:retireRows");
  assert.equal(events[3 + cycle.plan.levelCount], "direct:resetSetupMetadata");
  assert.ok(events.indexOf("direct:finalizeLifecycle") < events.indexOf("end"));
  assert.deepEqual(events.slice(-2), ["end",
    "copy:SPGrid worklist counts and published dispatches->SPGrid live indirect dispatches"]);
  const metadata = created.find((entry) => entry.label === "SPGrid worklist counts and published dispatches")!;
  const indirect = created.find((entry) => entry.label === "SPGrid live indirect dispatches")!;
  assert.equal(metadata.usage & GPUBufferUsage.COPY_SRC, GPUBufferUsage.COPY_SRC);
  assert.equal(metadata.usage & GPUBufferUsage.INDIRECT, 0, "writable metadata must never be an indirect source");
  assert.equal(indirect.usage, GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);
  assert.equal(indirect.usage & GPUBufferUsage.STORAGE, 0, "indirect arguments must never be writable storage");
  cycle.destroy();
});

test("failed generations publish a conditional full reset and successful generations retire exact live slots", () => {
  assert.match(octreeSPGridVCycleShader, /fn previousValid\(\)->bool/);
  assert.match(octreeSPGridVCycleShader,
    /fn retireSlots[\s\S]*!previousValid\(\)\|\|i>=count\(l\)[\s\S]*for\(var c=0u;c<27u;c\+=1u\)/);
  assert.match(octreeSPGridVCycleShader,
    /fn resetInvalidBuffers[\s\S]*i<arrayLength\(&topology\)[\s\S]*i<arrayLength\(&state\)/);
  assert.match(octreeSPGridVCycleShader,
    /fn finalizeLifecycle[\s\S]*atomicLoad\(&control\[0\]\)==0u[\s\S]*atomicStore\(&dispatchMeta\[base\],1u\)[\s\S]*let words=max\(arrayLength\(&topology\),arrayLength\(&state\)\)/);
  assert.doesNotMatch(WebGPUOctreeSPGridVCycle.prototype.encodeSetup.toString(), /clearBuffer/,
    "successful recurring generations must not write full sparse capacity");
});

test("correction consumes per-level live slot and transfer dispatch offsets", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const buffer = (size: number, usage = 31, label?: string) => ({ size, usage, label, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} },
    createBuffer: ({ label, size, usage }: { label?: string; size: number; usage: number }) => buffer(size, usage, label),
    createShaderModule: () => ({}), createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, { leafHeaders: buffer(48 * 32), leafEntries: buffer(256) },
    { dimensions: [8, 8, 8], rowCapacity: 32, finestCellWidth: 1 });
  const offsets: number[] = []; const sources = new Set<GPUBuffer>(); let direct = 0;
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { direct += 1; },
    dispatchWorkgroupsIndirect(source: GPUBuffer, offset: number) { sources.add(source); offsets.push(offset); }, end() {} } as unknown as GPUComputePassEncoder;
  const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const input = { rowCount: buffer(64), solverControl: buffer(64), rhs: buffer(128), correction: buffer(128) };
  cycle.encodeCorrection(encoder, input);
  assert.equal(direct, 3, "only correction clear, RHS seed, and publication use row-capacity dispatches");
  assert.equal(offsets.length, cycle.encodedCorrectionDispatchCount - direct);
  assert.equal(sources.size, 1, "all correction work must consume the dedicated indirect buffer");
  assert.ok(offsets.includes(8) && offsets.includes(20), "level zero uses distinct slot and transfer records");
  assert.ok(offsets.includes((cycle.plan.levelCount - 1) * 32 + 8), "the bottom level uses its own live slot record");
  assert.ok(offsets.every((offset) => offset % 32 === 8 || offset % 32 === 20));
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
  for (const entryPoint of ["resetInvalidBuffers", "retireSlots", "retireRows", "resetSetupMetadata", "finalizeLifecycle",
    "clearCorrection", "emitCells", "emitGhostAliases",
    "buildTransfers", "buildStencil", "ensureDiagonal", "finalizeIndirect", "zeroVectors", "seedRhs", "applyA",
    "applyB", "jacobiAtoB", "jacobiBtoA", "formResidual", "restrictResidual", "ghostValueAccumulate", "exactBottom",
    "prolongCorrection", "ghostValuePropagate", "publish"]) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message); device.destroy();
});
