import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  applyFirstOrderVCycleOracle,
  firstOrderGhostFluidBoundaryCoefficient,
  firstOrderOctreeAxisCoefficient,
  galerkinCoarsen,
  ghostValueAccumulate,
  ghostValuePropagate,
  octreeFirstOrderVCycleShader,
  planOctreeFirstOrderVCycle,
  WebGPUOctreeFirstOrderVCycle,
} from "../lib/webgpu-octree-first-order-vcycle";

const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, value, i) => sum + value * b[i], 0);

test("sparse L1 pyramid allocation is bounded by live rows and levels", () => {
  const small = planOctreeFirstOrderVCycle({ dimensions: [64, 48, 32], rowCapacity: 5_000 });
  const wide = planOctreeFirstOrderVCycle({ dimensions: [1024, 48, 32], rowCapacity: 5_000 });
  assert.equal(small.hierarchyStride, wide.hierarchyStride);
  assert.ok(wide.levelCount > small.levelCount);
  assert.equal(wide.stateBytes, 8 * wide.levelCount * wide.hierarchyStride * 4);
  assert.ok(!("cellCount" in wide));
});

test("GhostValueAccumulate is the exact adjoint of GhostValuePropagate", () => {
  const map = [0, 0, 1, 1, 1, 2];
  const coarse = [0.25, -2, 1.5], fine = [3, -1, 4, 2, -3, 0.5];
  const propagated = ghostValuePropagate(coarse, map);
  const accumulated = ghostValueAccumulate(fine, map, coarse.length);
  assert.ok(Math.abs(dot(propagated, fine) - dot(coarse, accumulated)) < 1e-12);
});

test("Galerkin L1 coarse operators retain positive energy", () => {
  const fine = [
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 2],
  ];
  const coarse = galerkinCoarsen(fine, [0, 0, 1, 1], 2);
  for (const value of [[1, 0], [0, 1], [1, -2], [-0.25, 3]]) {
    const product = coarse.map((row) => dot(row, value));
    assert.ok(dot(value, product) > 0);
  }
});

test("mixed-size T-junction L1 uses Cartesian shared-face area over center distance", () => {
  const h = 0.25, coarseOrigin = [0, 0, 0] as const;
  const fineOrigins = [[2, 0, 0], [2, 1, 0], [2, 0, 1], [2, 1, 1]] as const;
  const coefficients = fineOrigins.map((origin) => firstOrderOctreeAxisCoefficient(coarseOrigin, 2, origin, 1, h));
  for (const coefficient of coefficients) assert.ok(Math.abs(coefficient - h / 1.5) < 1e-12);
  assert.ok(Math.abs(coefficients.reduce((sum, value) => sum + value, 0) - 4 * h / 1.5) < 1e-12);
  assert.equal(firstOrderOctreeAxisCoefficient(coarseOrigin, 2, [2, 2, 0], 1, h), 0,
    "an edge-only contact is not an L1 face neighbor");
  assert.equal(firstOrderOctreeAxisCoefficient(coarseOrigin, 2, fineOrigins[0], 1, h),
    firstOrderOctreeAxisCoefficient(fineOrigins[0], 1, coarseOrigin, 2, h), "pair weight is reciprocal");
});

test("first-order free-surface anchor uses the ghost-fluid zero-crossing distance", () => {
  assert.equal(firstOrderGhostFluidBoundaryCoefficient(2, 4, -1, 3), 2,
    "theta=1/4 places the p=0 anchor one quarter of the center distance from liquid");
  assert.equal(firstOrderGhostFluidBoundaryCoefficient(2, 4, -1, 3, 0.5), 1,
    "the same Cartesian solid aperture weights the L1 boundary diagonal");
  assert.throws(() => firstOrderGhostFluidBoundaryCoefficient(1, 1, 1, -1), /invalid/);
});

test("paired-smoothing Galerkin V-cycle is symmetric and positive", () => {
  const operator = [
    [3, -1, 0, 0, 0, 0],
    [-1, 3, -1, 0, 0, 0],
    [0, -1, 3, -1, 0, 0],
    [0, 0, -1, 3, -1, 0],
    [0, 0, 0, -1, 3, -1],
    [0, 0, 0, 0, -1, 2],
  ];
  const maps = [[0, 0, 1, 1, 2, 2], [0, 0, 1]];
  const x = [1, -2, 0.5, 3, -1, 0.25], y = [-1, 0.5, 2, -0.5, 1.5, 4];
  const mx = applyFirstOrderVCycleOracle(operator, maps, x);
  const my = applyFirstOrderVCycleOracle(operator, maps, y);
  assert.ok(Math.abs(dot(x, my) - dot(y, mx)) < 1e-10);
  assert.ok(dot(x, mx) > 0);
  assert.ok(dot(y, my) > 0);
});

test("GPU L1 path filters axis faces and exposes adjacent-level ghost transfers", () => {
  assert.match(octreeFirstOrderVCycleShader, /fn axisCoefficient/);
  assert.match(octreeFirstOrderVCycleShader, /params\.weights\.z\*f32\(area\)/);
  assert.match(octreeFirstOrderVCycleShader, /fn firstOrderCoefficient/);
  assert.match(octreeFirstOrderVCycleShader, /return e\.coefficient/,
    "captured L1 aperture/GFM coefficients, not L2 power coefficients, drive the V-cycle");
  assert.match(octreeFirstOrderVCycleShader, /firstOrderDiagonal\(row\)-coupled/,
    "the exact first-order free-surface anchor comes from the captured L1 row, not L2");
  assert.match(octreeFirstOrderVCycleShader, /fn ghostAccumulate/);
  assert.match(octreeFirstOrderVCycleShader, /fn ghostPropagate/);
  assert.match(octreeFirstOrderVCycleShader, /atomicAddFloat\(at\(RHS,l\+1u,parent\)/);
  assert.match(octreeFirstOrderVCycleShader, /atomicAddFloat\(at\(SOLUTION_A,l,child\)/);
});

test("Dawn L1 V-cycle application has symmetric positive energy", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU V-cycle checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice();
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const headers = device.createBuffer({ size: 96, usage: storage, mappedAtCreation: true });
  {
    const memory = headers.getMappedRange(), words = new Uint32Array(memory), floats = new Float32Array(memory);
    words[0] = 0; words[1] = 0; words[2] = 1; words[3] = 1; floats[4] = 2;
    words[12] = 1; words[13] = 1; words[14] = 1; words[15] = 1; floats[16] = 2; headers.unmap();
  }
  const entries = device.createBuffer({ size: 16, usage: storage, mappedAtCreation: true });
  {
    const memory = entries.getMappedRange(), words = new Uint32Array(memory), floats = new Float32Array(memory);
    words[0] = 1; floats[1] = 1; words[2] = 0; floats[3] = 1; entries.unmap();
  }
  const counts = device.createBuffer({ size: 64, usage: storage }); device.queue.writeBuffer(counts, 0, new Uint32Array([2]));
  const control = device.createBuffer({ size: 64, usage: storage });
  const makeVector = (value?: readonly number[]) => {
    const buffer = device.createBuffer({ size: 8, usage: storage });
    if (value) device.queue.writeBuffer(buffer, 0, new Float32Array(value)); return buffer;
  };
  const x = [1, -2], y = [-0.5, 3], rhsX = makeVector(x), rhsY = makeVector(y), outX = makeVector(), outY = makeVector();
  const readback = device.createBuffer({ size: 80, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const cycle = new WebGPUOctreeFirstOrderVCycle(device, { leafHeaders: headers, leafEntries: entries },
    { dimensions: [4, 1, 1], rowCapacity: 2, maximumLevels: 3, finestCellWidth: 1 });
  device.pushErrorScope("validation"); const encoder = device.createCommandEncoder();
  cycle.encodeCapture(encoder);
  cycle.encodeSetup(encoder, { solverControl: control, rowCount: counts });
  cycle.encodeCorrection(encoder, { rhs: rhsX, correction: outX, solverControl: control, rowCount: counts });
  cycle.encodeCorrection(encoder, { rhs: rhsY, correction: outY, solverControl: control, rowCount: counts });
  encoder.copyBufferToBuffer(control, 0, readback, 0, 64); encoder.copyBufferToBuffer(outX, 0, readback, 64, 8);
  encoder.copyBufferToBuffer(outY, 0, readback, 72, 8); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  assert.equal(await device.popErrorScope(), null); await readback.mapAsync(GPUMapMode.READ);
  const mapped = readback.getMappedRange(); assert.equal(new Uint32Array(mapped, 0, 1)[0], 0);
  const mx = Array.from(new Float32Array(mapped, 64, 2)), my = Array.from(new Float32Array(mapped, 72, 2));
  assert.ok(Math.abs(dot(x, my) - dot(y, mx)) < 1e-5); assert.ok(dot(x, mx) > 0); assert.ok(dot(y, my) > 0); readback.unmap();
  cycle.destroy(); for (const buffer of [headers, entries, counts, control, rhsX, rhsY, outX, outY, readback]) buffer.destroy(); device.destroy();
});
