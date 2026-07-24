import assert from "node:assert/strict";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import {
  buildFixedUniformOctreePowerGalerkinHierarchy,
  refreshOctreePowerGalerkinOperators,
} from "../lib/octree-power-galerkin";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { WebGPUOctreePowerGalerkin } from "../lib/webgpu-octree-power-galerkin";
import { WEBGPU_EXCLUSIVE_LOCK } from "./webgpu-smoke-isolation";

const dimensions = [24, 18, 16] as const;
const repetitions = Math.max(1, Number(process.env.FLUID_GALERKIN_BENCH_REPETITIONS ?? 8));
const cycles = Math.max(1, Number(process.env.FLUID_GALERKIN_BENCH_CYCLES ?? 10));
const warmups = Math.max(0, Number(process.env.FLUID_GALERKIN_BENCH_WARMUPS ?? 2));
const retainedNativeGPUs: GPU[] = [];

try {
  await mkdir(WEBGPU_EXCLUSIVE_LOCK);
} catch (error) {
  let owner = "unknown owner";
  try { owner = await readFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, "utf8"); } catch { /* diagnostic only */ }
  throw new Error(`Refusing concurrent Galerkin benchmark; lock exists (${owner})`, { cause: error });
}

try {
  await writeFile(`${WEBGPU_EXCLUSIVE_LOCK}/owner.json`, JSON.stringify({
    pid: process.pid,
    startedAt: new Date().toISOString(),
    kind: "power-galerkin-benchmark",
    target: "tools/benchmark-power-galerkin.ts",
  }));

  const modulePath = resolve(process.env.WEBGPU_NODE_MODULE ?? "node_modules/webgpu/index.js");
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  retainedNativeGPUs.push(nativeGpu);
  const adapter = await nativeGpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU adapter unavailable");
  assert.ok(adapter.features.has("timestamp-query"), "timestamp-query is required");
  const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });

  const hierarchy = buildFixedUniformOctreePowerGalerkinHierarchy(dimensions, {
    transfer: "trilinear",
    coarsestNodeLimit: 64,
  });
  const finest = hierarchy.levels[0];
  const operators = refreshOctreePowerGalerkinOperators(
    hierarchy,
    new Float64Array(finest.nodeCount).fill(1),
    new Float64Array(hierarchy.fineOffDiagonalEntries.length),
  );
  const solver = new WebGPUOctreePowerGalerkin(device, hierarchy, operators, {
    cycles,
    smoothingIterations: Math.max(2, Number(process.env.FLUID_GALERKIN_BENCH_SMOOTHING ?? 2)),
    damping: 0.5,
    relativeTolerance: 1e-4,
  });

  const [nx, ny] = dimensions;
  const activeCells: number[] = [];
  for (let z = 0; z < 8; z += 1) {
    for (let y = 0; y < 17; y += 1) {
      for (let x = 0; x < 12; x += 1) activeCells.push(x + nx * (y + ny * z));
    }
  }
  assert.equal(activeCells.length, 1_632);
  const rowForCell = new Map(activeCells.map((cell, row) => [cell, row]));
  const neighbours = activeCells.map((cell) => {
    const x = cell % nx;
    const y = Math.floor(cell / nx) % ny;
    const z = Math.floor(cell / (nx * ny));
    return [
      x > 0 ? cell - 1 : -1,
      x + 1 < dimensions[0] ? cell + 1 : -1,
      y > 0 ? cell - nx : -1,
      y + 1 < dimensions[1] ? cell + nx : -1,
      z > 0 ? cell - nx * ny : -1,
      z + 1 < dimensions[2] ? cell + nx * ny : -1,
    ].flatMap((candidate) => {
      const row = rowForCell.get(candidate);
      return row === undefined ? [] : [row];
    });
  });
  const entryCount = neighbours.reduce((sum, row) => sum + row.length, 0);
  const headers = new Uint32Array(activeCells.length * 12);
  const headerFloats = new Float32Array(headers.buffer);
  const entries = new Uint32Array(entryCount * 2);
  const entryFloats = new Float32Array(entries.buffer);
  let entry = 0;
  activeCells.forEach((cell, row) => {
    const base = 12 * row;
    headers[base] = cell;
    headers[base + 1] = entry;
    headers[base + 2] = neighbours[row].length;
    headers[base + 3] = 1;
    headerFloats[base + 4] = neighbours[row].length + 1;
    headerFloats[base + 5] = Math.sin((row + 1) * 0.013) * 1e-3;
    for (const column of neighbours[row]) {
      entries[2 * entry] = column;
      entryFloats[2 * entry + 1] = 1;
      entry += 1;
    }
  });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const upload = (label: string, data: Uint32Array) => {
    const buffer = device.createBuffer({ label, size: data.byteLength, usage: storage });
    const bytes = new Uint8Array(data.byteLength);
    bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };
  const leafHeaders = upload("Galerkin benchmark live headers", headers);
  const leafEntries = upload("Galerkin benchmark live entries", entries);
  const authority = upload("Galerkin benchmark authority", new Uint32Array([
    0x8000_0000, 0xffff_ffff, activeCells.length, 0,
  ]));
  const correction = device.createBuffer({
    label: "Galerkin benchmark correction",
    size: activeCells.length * 4,
    usage: storage,
  });

  const encodeSolve = (encoder: GPUCommandEncoder) => {
    const broker = new PassBroker(encoder);
    solver.encode(broker, {
      liveOperator: { leafHeaders, leafEntries, authorityControl: authority },
      correction,
    });
    broker.fence("isolated Galerkin solve complete");
  };
  for (let warmup = 0; warmup < warmups; warmup += 1) {
    const encoder = device.createCommandEncoder();
    encodeSolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const controlReadback = device.createBuffer({
    size: 64,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder({ label: "Power Galerkin isolated benchmark" });
  const begin = encoder.beginComputePass({
    timestampWrites: { querySet, beginningOfPassWriteIndex: 0 },
  });
  begin.end();
  for (let iteration = 0; iteration < repetitions; iteration += 1) encodeSolve(encoder);
  const end = encoder.beginComputePass({
    timestampWrites: { querySet, beginningOfPassWriteIndex: 1 },
  });
  end.end();
  encoder.copyBufferToBuffer(solver.control, 0, controlReadback, 0, 64);
  encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
  encoder.copyBufferToBuffer(resolveBuffer, 0, readback, 0, 16);
  const wallStart = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const queueWall_ms = performance.now() - wallStart;
  await Promise.all([
    readback.mapAsync(GPUMapMode.READ),
    controlReadback.mapAsync(GPUMapMode.READ),
  ]);
  const timestamps = new BigUint64Array(readback.getMappedRange());
  const timestampDelta = Number(timestamps[1] - timestamps[0]);
  const gpuTotal_ms = timestampDelta > 0 ? timestampDelta / 1e6 : null;
  const control = new Uint32Array(controlReadback.getMappedRange());
  const relativeResidual = new Float32Array(new Uint32Array([control[6]]).buffer)[0];
  console.log(JSON.stringify({
    benchmark: "fixed-native-l2-galerkin",
    dimensions,
    activeRows: activeCells.length,
    fixedRows: finest.nodeCount,
    levels: hierarchy.levels.map((level) => level.nodeCount),
    cycles: solver.plan.cycles,
    dispatchesPerSolve: solver.plan.encodedDispatches,
    warmups,
    repetitions,
    gpuTotal_ms,
    gpuPerSolve_ms: gpuTotal_ms === null ? null : gpuTotal_ms / repetitions,
    queueWall_ms,
    queueWallPerSolve_ms: queueWall_ms / repetitions,
    controlFlags: control[0],
    converged: control[1] === 1,
    relativeResidual,
  }));
  readback.unmap();
  controlReadback.unmap();
  querySet.destroy();
  resolveBuffer.destroy();
  readback.destroy();
  controlReadback.destroy();
  solver.destroy();
  leafHeaders.destroy();
  leafEntries.destroy();
  authority.destroy();
  correction.destroy();
  device.destroy();
} finally {
  await rm(WEBGPU_EXCLUSIVE_LOCK, { recursive: true, force: true });
}
