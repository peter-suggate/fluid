import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { initializeRigidBodies } from "../lib/rigid-body";
import type { GPUInitializationProgress, GPUSolverInstance, MethodParamValues } from "../lib/methods/types";
import { octreeMethod } from "../lib/methods/octree";
import { fluidExecutionDeviceFeatures } from "../lib/gpu-startup";
import { viewportFailureIndicator } from "../lib/viewport-failure-diagnostics";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { decodeOctreeAirSupportGPUFirstError } from "../lib/webgpu-octree-air-velocity-support-gpu";
import { createSmokeScenario, isSmokeScenarioId } from "./webgpu-smoke-scenarios";
import {
  parseWebGPUBringupStage,
  reachedSolverResourceBoundary,
  stageIncludesComputeSentinel,
  stageIncludesSparseT0,
} from "./webgpu-bringup-stages";

const COMPUTE_SENTINEL_WORD = 0x4f43_5452;
const EXCLUSIVE_LOCK = "/tmp/fluid-webgpu-exclusive.lock";

class SolverResourceBoundary extends Error {
  constructor() { super("solver resources compiled and allocated before sparse t=0 warmup"); }
}

/**
 * Wait for already-created resources and their error events to reach JS.
 * onSubmittedWorkDone does not submit a command buffer, and the following
 * event-loop turn lets Dawn dispatch an uncapturederror queued by validation.
 */
async function flushGPUErrorDelivery(device: GPUDevice): Promise<void> {
  await device.queue.onSubmittedWorkDone();
  await new Promise<void>((resolve) => setImmediate(resolve));
}

async function acquireExclusiveGPUProcessLock() {
  try {
    await mkdir(EXCLUSIVE_LOCK);
  } catch (error) {
    let owner = "unknown owner";
    try { owner = await readFile(`${EXCLUSIVE_LOCK}/owner.json`, "utf8"); } catch { /* best-effort diagnostic */ }
    throw new Error(`Refusing concurrent GPU bring-up; ${EXCLUSIVE_LOCK} already exists (${owner}). Remove it only after confirming no Dawn or browser GPU run is active.`, { cause: error });
  }
  await writeFile(`${EXCLUSIVE_LOCK}/owner.json`, JSON.stringify({ pid: process.pid, startedAt: new Date().toISOString(), kind: "dawn-bringup" }));
}

async function assertComputeSentinel(device: GPUDevice): Promise<number> {
  const output = device.createBuffer({ label: "Bring-up compute sentinel", size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ label: "Bring-up compute sentinel readback", size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const shaderModule = device.createShaderModule({
      label: "Bring-up compute sentinel",
      code: "@group(0) @binding(0) var<storage, read_write> output: array<u32>; @compute @workgroup_size(1) fn sentinel() { output[0] = 0x4f435452u; }",
    });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint: "sentinel" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder({ label: "Bring-up compute sentinel" });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const observed = new Uint32Array(readback.getMappedRange())[0];
    readback.unmap();
    if (observed !== COMPUTE_SENTINEL_WORD) throw new Error(`compute sentinel returned 0x${observed.toString(16).padStart(8, "0")} instead of 0x${COMPUTE_SENTINEL_WORD.toString(16)}`);
    return observed;
  } finally {
    output.destroy(); readback.destroy();
  }
}

/** Failure-only inspection of one accepted structured family handle. This is
 * deliberately absent from the recurring simulation schedule. */
async function readStructuredHandleFailure(device: GPUDevice, projection: unknown, handle: number) {
  const internals = projection as { structuredVelocity?: { source: {
    plan: { rowCapacity: number; slotCapacity: number; authorityWords: number;
      maximumCaseSlots: number; offsets: Record<string, number> };
    control: GPUBuffer; authority: GPUBuffer; rowGeometry: GPUBuffer; rowVelocities: GPUBuffer;
  } }; structuredBoundary?: { solidNormalVelocities: GPUBuffer };
    structuredDynamics?: { transportMetrics: GPUBuffer; selectorStride: number;
      selectorOffsetWords: number; dimensions: readonly [number, number, number] };
    powerTopology?: { metrics: GPUBuffer } };
  const source = internals.structuredVelocity?.source;
  const boundary = internals.structuredBoundary;
  const dynamics = internals.structuredDynamics;
  const topology = internals.powerTopology;
  if (!source || !boundary || !dynamics || !topology || !Number.isSafeInteger(handle)
    || handle < 0 || handle >= source.plan.slotCapacity) return undefined;
  const controlReadback = device.createBuffer({ label: "Structured failure control readback", size: 44,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const controlEncoder = device.createCommandEncoder({ label: "Read structured failure control" });
  controlEncoder.copyBufferToBuffer(source.control, 0, controlReadback, 0, 44);
  device.queue.submit([controlEncoder.finish()]);
  await controlReadback.mapAsync(GPUMapMode.READ);
  const control = Array.from(new Uint32Array(controlReadback.getMappedRange()));
  controlReadback.unmap(); controlReadback.destroy();
  const bank = control[4] ?? 2, slots = control[5] ?? 0;
  const failureDetail = new Float32Array(new Uint32Array(control.slice(6, 10)).buffer);
  if (bank > 1 || handle >= slots) return { handle, control, blocker: "handle is outside the accepted bank" };
  const plan = source.plan, offsets = plan.offsets;
  const authorityBase = bank * plan.authorityWords * 4;
  const first = device.createBuffer({ label: "Structured failure handle readback", size: 128,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const firstEncoder = device.createCommandEncoder({ label: "Read structured failure handle" });
  const scalarOffsets = ["values", "ownerRows", "neighborRows", "metadata", "areas",
    "inverseDistances", "fractions", "pressureScales"];
  scalarOffsets.forEach((name, index) => firstEncoder.copyBufferToBuffer(source.authority,
    authorityBase + (offsets[name]! + handle) * 4, first, index * 4, 4));
  firstEncoder.copyBufferToBuffer(source.authority,
    authorityBase + (offsets.normals! + 4 * handle) * 4, first, 32, 16);
  firstEncoder.copyBufferToBuffer(source.authority,
    authorityBase + (offsets.centroids! + 4 * handle) * 4, first, 48, 16);
  firstEncoder.copyBufferToBuffer(boundary.solidNormalVelocities,
    (bank * plan.slotCapacity + handle) * 4, first, 64, 4);
  firstEncoder.copyBufferToBuffer(dynamics.transportMetrics, handle * 16, first, 80, 16);
  device.queue.submit([firstEncoder.finish()]);
  await first.mapAsync(GPUMapMode.READ);
  const firstBytes = first.getMappedRange().slice(0);
  first.unmap(); first.destroy();
  const firstWords = new Uint32Array(firstBytes), firstFloats = new Float32Array(firstBytes);
  const owner = firstWords[1]!, neighbor = firstWords[2]!;
  if (owner >= (control[2] ?? 0) || owner >= plan.rowCapacity) {
    return { handle, control, owner, neighbor, words: Array.from(firstWords.slice(0, 24)) };
  }
  const rowReadback = device.createBuffer({ label: "Structured failure owner-row readback", size: 512,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const rowEncoder = device.createCommandEncoder({ label: "Read structured failure owner row" });
  rowEncoder.copyBufferToBuffer(source.rowGeometry, (bank * plan.rowCapacity + owner) * 16,
    rowReadback, 0, 16);
  rowEncoder.copyBufferToBuffer(source.rowVelocities, (bank * plan.rowCapacity + owner) * 16,
    rowReadback, 16, 16);
  rowEncoder.copyBufferToBuffer(topology.metrics, owner * 16, rowReadback, 32, 16);
  rowEncoder.copyBufferToBuffer(source.authority,
    authorityBase + (offsets.rowAxisNeighbors! + 6 * owner) * 4, rowReadback, 48, 24);
  rowEncoder.copyBufferToBuffer(source.authority,
    authorityBase + (offsets.rowNeighbors! + plan.maximumCaseSlots * owner) * 4,
    rowReadback, 72, plan.maximumCaseSlots * 4);
  device.queue.submit([rowEncoder.finish()]);
  await rowReadback.mapAsync(GPUMapMode.READ);
  const rowBytes = rowReadback.getMappedRange().slice(0);
  rowReadback.unmap(); rowReadback.destroy();
  const rowWords = new Uint32Array(rowBytes), rowFloats = new Float32Array(rowBytes);
  const selectorIndex = Math.trunc(failureDetail[0] ?? -1);
  const selectorStride = dynamics.selectorStride;
  const geometryBytes = control[2]! * 16;
  const topologyReadback = device.createBuffer({
    label: "Structured failure selector and accepted geometry readback",
    size: Math.max(4, geometryBytes + 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const topologyEncoder = device.createCommandEncoder({ label: "Read structured failure selector topology" });
  topologyEncoder.copyBufferToBuffer(source.rowGeometry,
    bank * plan.rowCapacity * 16, topologyReadback, 0, geometryBytes);
  if (selectorIndex >= 0 && selectorIndex < selectorStride) {
    topologyEncoder.copyBufferToBuffer(dynamics.transportMetrics,
      (dynamics.selectorOffsetWords + owner * selectorStride + selectorIndex) * 4,
      topologyReadback, geometryBytes, 4);
  }
  device.queue.submit([topologyEncoder.finish()]);
  await topologyReadback.mapAsync(GPUMapMode.READ);
  const topologyWords = new Uint32Array(topologyReadback.getMappedRange().slice(0));
  topologyReadback.unmap(); topologyReadback.destroy();
  const ownerCell = rowWords[0]!, ownerSize = rowWords[1]!;
  const dimensions = dynamics.dimensions;
  const ownerQ = [ownerCell % dimensions[0], Math.trunc(ownerCell / dimensions[0]) % dimensions[1],
    Math.trunc(ownerCell / (dimensions[0] * dimensions[1]))];
  const nearbyRows = Array.from({ length: control[2]! }, (_, row) => {
    const cell = topologyWords[row * 4]!, size = topologyWords[row * 4 + 1]!;
    const q = [cell % dimensions[0], Math.trunc(cell / dimensions[0]) % dimensions[1],
      Math.trunc(cell / (dimensions[0] * dimensions[1]))];
    return { row, cell, size, q };
  }).filter(({ q, size }) => size <= 2 * ownerSize
    && q.every((value, axis) => Math.abs(value - ownerQ[axis]!) <= 2 * ownerSize));
  return { handle, control: control.slice(0, 6),
    sampleFailureDetail: { vector: Array.from(failureDetail), dynamicClass: control[10] },
    family: firstWords[3]! & 7, orientation: (firstWords[3]! >>> 3) & 7,
    owner, neighbor, value: firstFloats[0], area: firstFloats[4], inverseDistance: firstFloats[5],
    aperture: firstFloats[6], pressureScale: firstFloats[7],
    normal: Array.from(firstFloats.slice(8, 11)), centroid: Array.from(firstFloats.slice(12, 15)),
    solidNormalVelocity: firstFloats[16], transportMetric: Array.from(firstFloats.slice(20, 24)),
    ownerGeometry: Array.from(rowWords.slice(0, 4)), ownerVelocity: Array.from(rowFloats.slice(4, 8)),
    ownerMetric: { caseId: rowWords[8], transformAndFlags: rowWords[9], volume: rowFloats[10], error: rowWords[11] },
    ownerAxisNeighbors: Array.from(rowWords.slice(12, 18)),
    ownerRowNeighbors: Array.from(rowWords.slice(18, 18 + plan.maximumCaseSlots)),
    selectorMappedRow: selectorIndex >= 0 && selectorIndex < selectorStride
      ? topologyWords[control[2]! * 4] : undefined,
    nearbyRows };
}

/** Failure-only inspection of the structured row that emitted one air-support
 * candidate. The producer reports a candidate item, not a row; decode that
 * bounded schedule identity before reading row-local evidence. */
async function readAirSupportRowFailure(device: GPUDevice, projection: unknown, item: number) {
  const internals = projection as { structuredVelocity?: { source: { plan: { rowCapacity: number };
    control: GPUBuffer; rowGeometry: GPUBuffer; rowVelocities: GPUBuffer } };
    airVelocitySupport?: { faceA: GPUBuffer; faceB: GPUBuffer; directAirVectors: GPUBuffer;
      scratch: GPUBuffer; plan: { candidateStride: number; fineCandidateOffset: number;
        offsets: { directoryFlags: number }; domainVolume: number } };
    powerTopology?: { metrics: GPUBuffer } };
  const source = internals.structuredVelocity?.source, support = internals.airVelocitySupport;
  const topology = internals.powerTopology;
  if (!source || !support || !topology || !Number.isSafeInteger(item) || item < 0
    || item >= support.plan.fineCandidateOffset) return undefined;
  const row = Math.floor(item / support.plan.candidateStride);
  const candidateLocal = item % support.plan.candidateStride;
  if (!Number.isSafeInteger(row) || row < 0
    || row >= source.plan.rowCapacity) return undefined;
  const first = device.createBuffer({ label: "Air-support failure row readback", size: 480,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Read air-support failure row" });
  encoder.copyBufferToBuffer(source.control, 0, first, 0, 24);
  encoder.copyBufferToBuffer(source.rowGeometry, row * 16, first, 32, 16);
  encoder.copyBufferToBuffer(source.rowVelocities, row * 16, first, 48, 16);
  encoder.copyBufferToBuffer(topology.metrics, row * 16, first, 64, 16);
  encoder.copyBufferToBuffer(support.directAirVectors, row * 16, first, 80, 16);
  encoder.copyBufferToBuffer(support.faceA, row * 12 * 16, first, 96, 12 * 16);
  encoder.copyBufferToBuffer(support.faceB, row * 12 * 16, first, 288, 12 * 16);
  device.queue.submit([encoder.finish()]);
  await first.mapAsync(GPUMapMode.READ);
  const bytes = first.getMappedRange().slice(0); first.unmap(); first.destroy();
  const words = new Uint32Array(bytes), floats = new Float32Array(bytes);
  const bank = words[4] ?? 0, cell = words[8] ?? 0;
  const flagReadback = device.createBuffer({ label: "Air-support failure demand flag", size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const flagEncoder = device.createCommandEncoder({ label: "Read air-support failure demand flag" });
  flagEncoder.copyBufferToBuffer(support.scratch,
    (support.plan.offsets.directoryFlags + cell) * 4, flagReadback, 0, 4);
  device.queue.submit([flagEncoder.finish()]); await flagReadback.mapAsync(GPUMapMode.READ);
  const demandFlags = new Uint32Array(flagReadback.getMappedRange())[0];
  flagReadback.unmap(); flagReadback.destroy();
  const face = (offset: number) => Array.from({ length: 12 }, (_, index) => ({
    value: floats[offset / 4 + 4 * index], layer: words[offset / 4 + 4 * index + 1],
    source: words[offset / 4 + 4 * index + 2], valid: words[offset / 4 + 4 * index + 3],
  }));
  return { item, row, candidateLocal, bank, demandFlags, geometry: Array.from(words.slice(8, 12)),
    velocity: Array.from(floats.slice(12, 16)), metric: Array.from(words.slice(16, 20)),
    staged: Array.from(floats.slice(20, 24)), faceA: face(96), faceB: face(288) };
}

/** Failure-only inspection of one ordinary-face row from the Section 5 march. */
async function readAirSupportFaceFailure(device: GPUDevice, projection: unknown, faceRow: number) {
  const internals = projection as { structuredVelocity?: { source: { plan: { rowCapacity: number };
    control: GPUBuffer; rowGeometry: GPUBuffer } };
    airVelocitySupport?: { arena: GPUBuffer; faceA: GPUBuffer; faceB: GPUBuffer;
      faceAdjacency: GPUBuffer; directAirVectors: GPUBuffer; recordArena: GPUBuffer;
      plan: { faceCellCapacity: number; faceAdjacencyStride: number;
        support: { supportVectorOffsetWords: number };
        records: { recordOffsetWords: number } } } };
  const source = internals.structuredVelocity?.source, support = internals.airVelocitySupport;
  if (!source || !support || !Number.isSafeInteger(faceRow) || faceRow < 0
    || faceRow >= support.plan.faceCellCapacity) return undefined;
  const controlReadback = device.createBuffer({ label: "Air-support face control readback", size: 24,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const controlEncoder = device.createCommandEncoder({ label: "Read air-support face control" });
  controlEncoder.copyBufferToBuffer(source.control, 0, controlReadback, 0, 24);
  device.queue.submit([controlEncoder.finish()]); await controlReadback.mapAsync(GPUMapMode.READ);
  const control = Array.from(new Uint32Array(controlReadback.getMappedRange()));
  controlReadback.unmap(); controlReadback.destroy();
  const directRows = control[2] ?? 0;
  if (faceRow >= directRows + support.plan.faceCellCapacity) return undefined;
  const readback = device.createBuffer({ label: "Air-support face failure readback", size: 640,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Read air-support face failure" });
  if (faceRow < directRows) {
    const bank = control[4] ?? 0;
    encoder.copyBufferToBuffer(source.rowGeometry,
      (bank * source.plan.rowCapacity + faceRow) * 16, readback, 32, 16);
    encoder.copyBufferToBuffer(support.directAirVectors, faceRow * 16, readback, 576, 16);
  } else {
    const supportIndex = faceRow - directRows;
    encoder.copyBufferToBuffer(support.recordArena,
      (support.plan.records.recordOffsetWords + 8 * supportIndex) * 4, readback, 32, 32);
    encoder.copyBufferToBuffer(support.arena,
      (support.plan.support.supportVectorOffsetWords + 4 * supportIndex) * 4,
      readback, 576, 16);
  }
  encoder.copyBufferToBuffer(support.faceA, faceRow * 12 * 16, readback, 192, 12 * 16);
  encoder.copyBufferToBuffer(support.faceB, faceRow * 12 * 16, readback, 384, 12 * 16);
  encoder.copyBufferToBuffer(support.faceAdjacency,
    faceRow * support.plan.faceAdjacencyStride * 4, readback, 64,
    Math.min(32, support.plan.faceAdjacencyStride) * 4);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0); readback.unmap(); readback.destroy();
  const words = new Uint32Array(bytes), floats = new Float32Array(bytes);
  const face = (offset: number) => Array.from({ length: 12 }, (_, index) => ({
    value: floats[offset / 4 + 4 * index], distance: floats[offset / 4 + 4 * index + 1],
    source: words[offset / 4 + 4 * index + 2], valid: words[offset / 4 + 4 * index + 3],
  }));
  const rejectedVector = Array.from(floats.slice(144, 148));
  const packedValidity = Math.trunc(rejectedVector[2] ?? 0);
  const rejectedDetail = (rejectedVector[3] ?? 0) < 0 ? {
    localFace: Math.trunc(rejectedVector[0] ?? 0), axis: Math.trunc(rejectedVector[1] ?? 0),
    quadrant: packedValidity & 15, positiveValid: (packedValidity >>> 4) & 1,
    negativeValid: (packedValidity >>> 5) & 1,
  } : undefined;
  return { faceRow, supportIndex: faceRow >= directRows ? faceRow - directRows : undefined,
  control, identity: Array.from(words.slice(8, 16)), rejectedVector, rejectedDetail,
  adjacency: Array.from(words.slice(16, 48)), faceA: face(192), faceB: face(384) };
}

function decodePrecedingRejectedDetail(packed: number | undefined) {
  const value = Number(packed ?? 0) >>> 0;
  if ((value & 0x8000_0000) === 0) return undefined;
  const validity = (value >>> 10) & 0x3f;
  return { localFace: value & 0xff, axis: (value >>> 8) & 3,
    quadrant: validity & 15, positiveValid: (validity >>> 4) & 1,
    negativeValid: (validity >>> 5) & 1,
    changedWaves: Array.from({ length: 6 }, (_, layer) => ((value >>> (16 + layer)) & 1) !== 0) };
}

function decodePrecedingRejectedIdentity(packed: number | undefined, detail: number | undefined) {
  const value = Number(packed ?? 0) >>> 0;
  if (((Number(detail ?? 0) >>> 0) & 0x8000_0000) === 0) return undefined;
  return { directRows: value & 0x3fff, cell: (value >>> 14) & 0x3fff,
    size: 1 << ((value >>> 28) & 15) };
}

function solverValues(): MethodParamValues {
  const values = octreeMethod.presetFor("balanced");
  values.globalFineLevelSetFactor = process.env.FLUID_FINE_FACTOR
    ?? process.env.FLUID_OCTREE_GLOBAL_FINE_FACTOR
    ?? "4";
  return values;
}

await acquireExclusiveGPUProcessLock();
let device: GPUDevice | undefined;
let solver: GPUSolverInstance | undefined;
let lost: GPUDeviceLostInfo | undefined;
try {
  const stage = parseWebGPUBringupStage(process.env.FLUID_BRINGUP_STAGE);
  console.log(JSON.stringify({ phase: "bringup-start", stage, pid: process.pid, browserGPURequiredClosed: true }));
  const modulePath = process.env.WEBGPU_NODE_MODULE;
  const webgpuModule = modulePath ? await import(pathToFileURL(modulePath).href) : await import("webgpu");
  const { create, globals } = webgpuModule as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  Reflect.deleteProperty(globalThis, "Worker");
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });

  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("Dawn did not expose a WebGPU adapter");
  const requiredFeatures = fluidExecutionDeviceFeatures(adapter.features);
  device = await adapter.requestDevice({ requiredFeatures, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  void device.lost.then((info) => { lost = info; });
  console.log(JSON.stringify({ phase: "adapter-device", stage, passed: true, features: requiredFeatures }));
  if (stage === "adapter-device") process.exitCode = 0;
  else {
    if (stageIncludesComputeSentinel(stage)) {
      const observed = await assertComputeSentinel(device);
      console.log(JSON.stringify({ phase: "compute-sentinel", stage, passed: true, value: observed }));
    }
    if (stage !== "compute-sentinel") {
      const scenarioId = process.env.FLUID_BRINGUP_SCENE ?? "dam-break-ui";
      if (!isSmokeScenarioId(scenarioId)) throw new Error(`Unknown FLUID_BRINGUP_SCENE: ${scenarioId}`);
      const scenario = createSmokeScenario(scenarioId);
      scenario.scene.voxelDomain.finestCellSize_m = Number(process.env.FLUID_VOXEL_CELL_SIZE ?? 0.05);
      const validationErrors: string[] = [];
      let lastInitializationCompleted = 0;
      device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
      const progress = (snapshot: GPUInitializationProgress) => {
        const boundary = snapshot.completed > lastInitializationCompleted ? "completed" : "starting";
        console.log(JSON.stringify({ record: "solver-initialization", boundary, stage, ...snapshot }));
        lastInitializationCompleted = Math.max(lastInitializationCompleted, snapshot.completed);
        if (stage === "solver-resources" && reachedSolverResourceBoundary(snapshot)) throw new SolverResourceBoundary();
      };
      try {
        solver = await octreeMethod.createSolverAsync!(device, scenario.scene, "balanced", solverValues(), undefined, progress);
      } catch (error) {
        if (!(error instanceof SolverResourceBoundary)) throw error;
        await flushGPUErrorDelivery(device);
        if (lost) throw new Error(`device lost before solver resource boundary: ${lost.reason} ${lost.message}`);
        if (validationErrors.length > 0) {
          throw new Error(`solver-resources validation failed: ${validationErrors.join("; ")}`);
        }
        console.log(JSON.stringify({ phase: "solver-resources", stage, passed: true, stoppedBeforeTask: "solver.warmup" }));
      }
      if (stageIncludesSparseT0(stage)) {
        if (!solver?.initialSparseAuthorityReady) throw new Error("solver returned without complete sparse t=0 authority");
        await flushGPUErrorDelivery(device);
        if (lost) throw new Error(`device lost during sparse t=0 publication: ${lost.reason} ${lost.message}`);
        if (validationErrors.length > 0) {
          throw new Error(`sparse-t0 validation failed: ${validationErrors.join("; ")}`);
        }
        console.log(JSON.stringify({ phase: "sparse-t0", stage, passed: true, scenario: scenarioId,
          startupMode: "phase-fenced", grid: [solver.info.nx, solver.info.ny, solver.info.nz],
          allocatedBytes: solver.info.allocatedBytes }));
      }
      if (stage === "one-step") {
        const requestedTime_s = scenario.scene.numerics.maxDt_s;
        const accepted = solver!.advanceTo(requestedTime_s, initializeRigidBodies(scenario.scene.rigidBodies));
        if (!accepted) throw new Error("solver refused the first bounded advance");
        // `advanceTo` only submits. The UI admits a generation after its GPU
        // completion fence, so the isolated Dawn checkpoint must not inspect
        // the host-side submitted clock and call that a completed step.
        await device.queue.onSubmittedWorkDone();
        solver!.info.completedTime_s = Math.max(solver!.info.completedTime_s ?? 0,
          solver!.info.submittedTime_s ?? 0);
        const info = { ...await solver!.readStats() };
        await flushGPUErrorDelivery(device);
        if (lost) throw new Error(`device lost during first step: ${lost.reason} ${lost.message}`);
        if ((info.encodedSteps ?? 0) !== 1 || (info.submittedTime_s ?? 0) < requestedTime_s) {
          throw new Error(`one-step checkpoint did not complete exactly one submission: steps=${info.encodedSteps}, submitted=${info.submittedTime_s}`);
        }
        if ((info.completedTime_s ?? 0) + 1e-9 < requestedTime_s) {
          throw new Error(`one-step checkpoint did not reach its GPU completion fence: completed=${info.completedTime_s}, requested=${requestedTime_s}`);
        }
        const authorityFailure = viewportFailureIndicator(info, undefined, scenario.scene);
        if (authorityFailure?.tone === "rejected") {
          const projection = (solver as unknown as { octreeProjection?: {
            readGlobalFineLevelSetDiagnostics(): Promise<{ structuredVelocityControl?: readonly number[];
              structuredBoundaryControl?: readonly number[]; airSupportControl?: readonly number[] }>;
            readPowerFrontierFailure(): Promise<unknown>;
            readMGPCGDiagnostics(): Promise<Uint32Array>;
          } }).octreeProjection;
          const [fineFailure, frontierFailure, mgpcgFailure] = projection
            ? await Promise.all([projection.readGlobalFineLevelSetDiagnostics(),
              projection.readPowerFrontierFailure(), projection.readMGPCGDiagnostics()])
            : [undefined, undefined, undefined] as const;
          const packedFailure = Number(fineFailure?.structuredVelocityControl?.[1] ?? 0xffff_ffff) >>> 0;
          const structuredHandleFailure = projection && (packedFailure >>> 24) >= 1
            && (packedFailure >>> 24) <= 3
            ? await readStructuredHandleFailure(device, projection, packedFailure & 0x00ff_ffff)
            : undefined;
          const packedAirSupportFailure = Number((fineFailure as { firstAirSupportFailure?: readonly number[] }
            | undefined)?.firstAirSupportFailure?.[1] ?? 0xffff_ffff) >>> 0;
          const { stage: airSupportFailureStage, item: firstAirSupportItem } =
            decodeOctreeAirSupportGPUFirstError(packedAirSupportFailure);
          const precedingAirSupportTerminal = (fineFailure as {
            precedingAirSupportTerminal?: readonly number[] } | undefined)?.precedingAirSupportTerminal;
          const airSupportRowFailure = projection && airSupportFailureStage === 1
            ? await readAirSupportRowFailure(device, projection, firstAirSupportItem)
            : undefined;
          const airSupportFaceFailure = projection && airSupportFailureStage >= 7
            ? await readAirSupportFaceFailure(device, projection, firstAirSupportItem)
            : undefined;
          throw new Error(`one-step authority rejected: ${authorityFailure.stage}: ${authorityFailure.detail}; power=`
            + JSON.stringify({ authoritative: info.powerDiagramAuthoritative,
              rows: info.pressureRequiredRows, overflow: info.pressureCapacityOverflow,
              generation: info.powerDiagramGeneration, structuredVelocityValid: info.structuredVelocityValid,
              structuredVelocityRows: info.structuredVelocityRows,
              structuredVelocityGeneration: info.structuredVelocityGeneration,
              structuredBoundaryValid: info.structuredBoundaryValid,
              structuredBoundaryGeneration: info.structuredBoundaryGeneration,
              velocityControl: fineFailure?.structuredVelocityControl,
              boundaryControl: fineFailure?.structuredBoundaryControl,
              airSupportControl: fineFailure?.airSupportControl,
              coarseControl: (fineFailure as { coarseControl?: readonly number[] } | undefined)?.coarseControl,
              fineRestrictionControl: (fineFailure as { fineRestrictionControl?: readonly number[] } | undefined)
                ?.fineRestrictionControl,
              fineTopologyControl: (fineFailure as { topologyControl?: readonly number[] } | undefined)
                ?.topologyControl,
              fineWorklistHeader: (fineFailure as { worklistHeader?: readonly number[] } | undefined)
                ?.worklistHeader,
              fineVolumeControl: (fineFailure as { fineVolumeControl?: readonly number[] } | undefined)
                ?.fineVolumeControl,
              firstCoarsePhi: (fineFailure as { firstCoarsePhi?: readonly number[] } | undefined)?.firstCoarsePhi,
              finePublication: fineFailure ? {
                configuredGeneration: (fineFailure as { configuredFineGeneration?: number }).configuredFineGeneration,
                scheduledGeneration: (fineFailure as { scheduledFineGeneration?: number }).scheduledFineGeneration,
                currentIsA: (fineFailure as { currentFineIsA?: boolean }).currentFineIsA,
              } : undefined,
              precedingAirSupportTerminal,
              precedingRejectedDetail: decodePrecedingRejectedDetail(precedingAirSupportTerminal?.[1]),
              precedingRejectedIdentity: decodePrecedingRejectedIdentity(
                precedingAirSupportTerminal?.[0], precedingAirSupportTerminal?.[1]),
              firstAirSupportFailure: (fineFailure as { firstAirSupportFailure?: readonly number[] } | undefined)
                ?.firstAirSupportFailure,
              airSupportFailureStage,
              firstAirSupportItem,
              airSupportRowFailure,
              airSupportFaceFailure,
              structuredHandleFailure,
              frontier: frontierFailure,
              mgpcg: mgpcgFailure ? Array.from(mgpcgFailure) : undefined }));
        }
        if (validationErrors.length > 0 || info.gpuValidationError) throw new Error(`one-step validation failed: ${[...validationErrors, info.gpuValidationError].filter(Boolean).join("; ")}`);
        const acceptedAirSupport = await (solver as unknown as { octreeProjection?: {
          readGlobalFineLevelSetDiagnostics(): Promise<{ airSupportControl?: readonly number[] }>;
        } }).octreeProjection?.readGlobalFineLevelSetDiagnostics();
        console.log(JSON.stringify({ phase: "one-step", stage, passed: true,
          encodedSteps: info.encodedSteps, submittedTime_s: info.submittedTime_s,
          completedTime_s: info.completedTime_s, nonFiniteCount: info.nonFiniteCount,
          stabilityFlags: info.stabilityFlags,
          airSupportMarchDepth: acceptedAirSupport?.airSupportControl?.[12],
        }));
      }
    }
  }
} finally {
  solver?.destroy();
  device?.destroy();
  Reflect.deleteProperty(globalThis, "navigator");
  await rm(EXCLUSIVE_LOCK, { recursive: true, force: true });
}
