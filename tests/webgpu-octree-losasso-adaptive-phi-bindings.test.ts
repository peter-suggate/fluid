import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { WebGPUOctreeLosassoAdaptivePhi } from "../lib/webgpu-octree-losasso-adaptive-phi";
import type {
  WebGPUOctreeLosassoAdaptiveNodalVelocitySource,
  WebGPUOctreeLosassoAdaptivePhiGraphBankSource,
} from "../lib/webgpu-octree-losasso-adaptive-phi";
import {
  octreeLosassoAdaptivePhiAcceptedScheduleWGSL,
  octreeLosassoAdaptivePhiBacktraceWGSL,
  octreeLosassoAdaptivePhiCorrectionWGSL,
  octreeLosassoAdaptivePhiCommitWGSL,
  octreeLosassoAdaptivePhiEvidenceWGSL,
  octreeLosassoAdaptivePhiGhostWGSL,
  octreeLosassoAdaptivePhiHandoffWGSL,
  octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
  octreeLosassoAdaptivePhiRedistanceBtoAWGSL,
  octreeLosassoAdaptivePhiRedistanceFinishWGSL,
  octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
  octreeLosassoAdaptivePhiRedistanceProjectBWGSL,
  octreeLosassoAdaptivePhiPredictorSnapshotWGSL,
  octreeLosassoAdaptivePhiReverseBacktraceWGSL,
  octreeLosassoAdaptivePhiScheduleWGSL,
  octreeLosassoAdaptivePhiTransportWGSL,
  octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
  octreeLosassoAdaptivePhiWorklistConstrainedWGSL,
  octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
  octreeLosassoAdaptivePhiWorklistFinalizeWGSL,
  octreeLosassoAdaptivePhiWorklistIndependentWGSL,
  octreeLosassoAdaptivePhiWorklistInflowWGSL,
  octreeLosassoAdaptivePhiWorklistPrepareWGSL,
  octreeLosassoAdaptivePhiWorklistProjectWGSL,
  octreeLosassoAdaptivePhiWorklistReachWGSL,
  octreeLosassoAdaptivePhiWorklistReceiptWGSL,
  octreeLosassoAdaptivePhiWGSL,
} from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";
import type { LosassoSurfaceGraphBankSource }
  from "../lib/webgpu-octree-losasso-surface-graph";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const requestedStage = process.env.FLUID_ADAPTIVE_PHI_BINDING_STAGE ?? "C";
if (requestedStage !== "A" && requestedStage !== "B" && requestedStage !== "C") {
  throw new Error("FLUID_ADAPTIVE_PHI_BINDING_STAGE must be A, B, or C");
}
const stage = requestedStage as "A" | "B" | "C";
const mark = (phase: string, detail: Record<string, unknown> = {}): void => {
  process.stderr.write(`${JSON.stringify({ stage, phase, ...detail })}\n`);
};

type DiagnosticPhi = {
  readonly acceptedScheduleControl: GPUBuffer;
  readonly advanceParams: GPUBuffer;
  readonly transportControl: GPUBuffer;
  readonly transportDispatch: GPUBuffer;
  readonly transportWorklist: GPUBuffer;
  readonly transportBandMask: GPUBuffer;
  readonly transportDepartures: GPUBuffer;
  readonly backtrace: GPUComputePipeline;
  readonly reverseBacktrace: GPUComputePipeline;
  readonly predictorSnapshot: GPUComputePipeline;
  readonly distanceA: GPUBuffer;
  readonly transport: GPUComputePipeline;
  readonly correction: GPUComputePipeline;
  readonly worklist: Readonly<Record<"prepareTransportBand" | "markTransportReach" | "markTransportInflow"
    | "publishTransportIndependent" | "markTransportConstrained" | "publishTransportConstrained"
    | "finalizeTransportDispatch" | "publishTransportPartition"
    | "projectTransportedBand", GPUComputePipeline>>;
  writeParams(target: GPUBuffer, dt: number, inflow: undefined, candidate: boolean): void;
  flushReferenceVolumeDelta(): void;
  encodeAcceptedLiveSchedule(broker: PassBroker): void;
  mainBuffers(bank: WebGPUOctreeLosassoAdaptivePhiGraphBankSource,
    distanceInput: GPUBuffer | undefined,
    velocity: WebGPUOctreeLosassoAdaptiveNodalVelocitySource,
    parameterBuffer: GPUBuffer): GPUBuffer[];
  runMain(broker: PassBroker, name: string, buffers: readonly GPUBuffer[], workgroups: number): void;
  runMainBufferIndirect(broker: PassBroker, name: string, buffers: readonly GPUBuffer[],
    dispatch: GPUBuffer, offset: number): void;
  runBufferIndirect(broker: PassBroker, pipeline: GPUComputePipeline,
    buffers: readonly GPUBuffer[], dispatch: GPUBuffer, offset: number): void;
  run(broker: PassBroker, pipeline: GPUComputePipeline,
    buffers: readonly GPUBuffer[], workgroups: number): void;
};

const floatBits = (value: number): number => {
  const f = new Float32Array([value]);
  return new Uint32Array(f.buffer)[0]!;
};

const entryBindings = (shader: string, root: string): number[] => {
  const declarations = [...shader.matchAll(/@binding\((\d+)\)var(?:<[^>]+>)?\s*(\w+)/g)]
    .map((match) => ({ binding: Number(match[1]), name: match[2]! }));
  const functions = new Map<string, string>();
  for (const match of shader.matchAll(/\bfn\s+(\w+)\s*\(/g)) {
    const open = shader.indexOf("{", match.index! + match[0].length);
    let depth = 1, at = open + 1;
    while (depth > 0 && at < shader.length) {
      if (shader[at] === "{") depth += 1;
      else if (shader[at] === "}") depth -= 1;
      at += 1;
    }
    functions.set(match[1]!, shader.slice(open + 1, at - 1));
  }
  const reachable = new Set<string>();
  const visit = (name: string): void => {
    if (reachable.has(name)) return;
    const body = functions.get(name); assert.ok(body, `missing WGSL function ${name}`);
    reachable.add(name);
    for (const call of body.matchAll(/\b(\w+)\s*\(/g)) {
      if (functions.has(call[1]!)) visit(call[1]!);
    }
  };
  visit(root);
  const source = [...reachable].map((name) => functions.get(name)!).join("\n");
  return declarations.filter(({ name }) => new RegExp(`\\b${name}\\b`).test(source))
    .map(({ binding }) => binding).sort((a, b) => a - b);
};

test("volume evidence entry points have exact transitive host binding sets", () => {
  const bindingSet = (WebGPUOctreeLosassoAdaptivePhi.prototype as unknown as {
    bindingSet(label: string): readonly number[];
  }).bindingSet;
  const cases = [
    [octreeLosassoAdaptivePhiWGSL, "bootstrapNodal"],
    [octreeLosassoAdaptivePhiWGSL, "bootstrapNodalLattice"],
    [octreeLosassoAdaptivePhiWGSL, "bootstrapCellCentred"],
    [octreeLosassoAdaptivePhiWGSL, "capturePreRedistanceVolumes"],
    [octreeLosassoAdaptivePhiWGSL, "derivePostRedistanceVolumes"],
    [octreeLosassoAdaptivePhiVolumeEvidenceWGSL, "prepareVolumeEvidence"],
    [octreeLosassoAdaptivePhiVolumeEvidenceWGSL, "validateVolumeEvidence"],
    [octreeLosassoAdaptivePhiVolumeEvidenceWGSL, "finalizeVolumeEvidence"],
    [octreeLosassoAdaptivePhiBacktraceWGSL, "backtraceIndependent"],
    [octreeLosassoAdaptivePhiReverseBacktraceWGSL, "reverseBacktraceIndependent"],
    [octreeLosassoAdaptivePhiTransportWGSL, "transportIndependent"],
    [octreeLosassoAdaptivePhiPredictorSnapshotWGSL, "snapshotProjectedPredictor"],
    [octreeLosassoAdaptivePhiCorrectionWGSL, "correctTransportIndependent"],
  ] as const;
  for (const [shader, entry] of cases) {
    assert.deepEqual([...bindingSet.call({}, `Adaptive phi - ${entry}`)].sort((a, b) => a - b),
      entryBindings(shader, entry), `${entry} host bindings must equal its transitive shader ABI`);
  }
});

test("standalone volume-evidence shader modules declare every symbolic constant", () => {
  for (const [label, shader] of [
    ["main", octreeLosassoAdaptivePhiWGSL],
    ["volume evidence", octreeLosassoAdaptivePhiVolumeEvidenceWGSL],
    ["transport correction", octreeLosassoAdaptivePhiCorrectionWGSL],
  ] as const) {
    const declared = new Set([...shader.matchAll(/\bconst\s+([A-Z][A-Z0-9_]*)\s*:/g)]
      .map((match) => match[1]!));
    const referenced = new Set([...shader.matchAll(/\b[A-Z][A-Z0-9_]*\b/g)]
      .map((match) => match[0]));
    assert.deepEqual([...referenced].filter((name) => !declared.has(name)), [],
      `${label} must not assume constants from another WGSL module`);
  }
});

test("volume-evidence shader identifiers avoid WGSL reserved words", () => {
  for (const [label, shader] of [
    ["main", octreeLosassoAdaptivePhiWGSL],
    ["volume evidence", octreeLosassoAdaptivePhiVolumeEvidenceWGSL],
  ] as const) for (const reserved of ["meta", "target"] as const) {
    assert.doesNotMatch(shader,
      new RegExp(`(?:\\b(?:let|var)\\s+${reserved}\\b|[,{]\\s*${reserved}\\s*:)`),
      `${label} must not declare reserved WGSL identifier ${reserved}`);
  }
});

test("volume evidence stays within the Dawn ten-storage-buffer limit", () => {
  for (const entry of ["prepareVolumeEvidence", "validateVolumeEvidence",
    "finalizeVolumeEvidence"] as const) {
    const storageBindings = entryBindings(octreeLosassoAdaptivePhiVolumeEvidenceWGSL, entry)
      .filter((binding) => binding !== 0);
    assert.ok(storageBindings.length <= 10,
      `${entry} uses ${storageBindings.length} storage buffers`);
  }
});

test(`adaptive accepted compact band stage ${stage} encodes and submits`, {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for adaptive phi binding validation",
}, async () => {
  mark("dawn-import-start");
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  mark("device-ready");
  void device.lost.then((info) => {
    mark("device-lost", { reason: info.reason, message: info.message });
  });

  if (process.env.FLUID_ADAPTIVE_PHI_COMPILE_BISECT === "1") {
    const modules = [
      { label: "accepted-schedule", code: octreeLosassoAdaptivePhiAcceptedScheduleWGSL,
        names: ["scheduleAcceptedWork"] },
      { label: "main", code: octreeLosassoAdaptivePhiWGSL, names: [
        "prepareBootstrap", "bootstrapNodal", "bootstrapNodalLattice", "bootstrapCellCentred",
        "applyReferenceVolumeDelta", "projectAccepted", "finalizeBootstrap",
        "captureReferenceVolume", "measureDerivations", "prepareAdvance",
        "captureTransportReceipt", "prepareCandidateRepair", "canonicalizeCandidatePhi",
        "projectTransported", "capturePreRedistanceVolumes", "deriveRows",
        "deriveLeafVolumes", "derivePostRedistanceVolumes", "finalizeAccepted", "finalizeCandidateRepair",
      ] },
      { label: "volume-evidence", code: octreeLosassoAdaptivePhiVolumeEvidenceWGSL,
        names: ["prepareVolumeEvidence", "validateVolumeEvidence", "finalizeVolumeEvidence"] },
      { label: "worklist-prepare", code: octreeLosassoAdaptivePhiWorklistPrepareWGSL,
        names: ["prepareTransportBand"] },
      { label: "worklist-reach", code: octreeLosassoAdaptivePhiWorklistReachWGSL,
        names: ["markTransportReach"] },
      { label: "worklist-inflow", code: octreeLosassoAdaptivePhiWorklistInflowWGSL,
        names: ["markTransportInflow"] },
      { label: "worklist-independent", code: octreeLosassoAdaptivePhiWorklistIndependentWGSL,
        names: ["publishTransportIndependent"] },
      { label: "worklist-constraint-mark", code: octreeLosassoAdaptivePhiWorklistConstraintMarkWGSL,
        names: ["markTransportConstrained"] },
      { label: "worklist-constrained", code: octreeLosassoAdaptivePhiWorklistConstrainedWGSL,
        names: ["publishTransportConstrained"] },
      { label: "worklist-finalize", code: octreeLosassoAdaptivePhiWorklistFinalizeWGSL,
        names: ["finalizeTransportDispatch"] },
      { label: "worklist-receipt", code: octreeLosassoAdaptivePhiWorklistReceiptWGSL,
        names: ["publishTransportPartition"] },
      { label: "worklist-project", code: octreeLosassoAdaptivePhiWorklistProjectWGSL,
        names: ["projectTransportedBand"] },
      { label: "redistance-initialize", code: octreeLosassoAdaptivePhiRedistanceInitializeWGSL,
        names: ["initializeRedistance"] },
      { label: "redistance-a-to-b", code: octreeLosassoAdaptivePhiRedistanceAtoBWGSL,
        names: ["redistanceAtoB"] },
      { label: "redistance-project-b", code: octreeLosassoAdaptivePhiRedistanceProjectBWGSL,
        names: ["projectDistanceB"] },
      { label: "redistance-b-to-a", code: octreeLosassoAdaptivePhiRedistanceBtoAWGSL,
        names: ["redistanceBtoA"] },
      { label: "redistance-project-a", code: octreeLosassoAdaptivePhiRedistanceProjectAWGSL,
        names: ["projectDistanceA"] },
      { label: "redistance-finish", code: octreeLosassoAdaptivePhiRedistanceFinishWGSL,
        names: ["finishRedistance"] },
      { label: "backtrace", code: octreeLosassoAdaptivePhiBacktraceWGSL,
        names: ["backtraceIndependent"] },
      { label: "reverse-backtrace", code: octreeLosassoAdaptivePhiReverseBacktraceWGSL,
        names: ["reverseBacktraceIndependent"] },
      { label: "transport", code: octreeLosassoAdaptivePhiTransportWGSL,
        names: ["transportIndependent"] },
      { label: "predictor-snapshot", code: octreeLosassoAdaptivePhiPredictorSnapshotWGSL,
        names: ["snapshotProjectedPredictor"] },
      { label: "correction", code: octreeLosassoAdaptivePhiCorrectionWGSL,
        names: ["correctTransportIndependent"] },
      { label: "handoff", code: octreeLosassoAdaptivePhiHandoffWGSL,
        names: ["prepareCandidateHandoff", "handoffCandidate", "projectCandidate",
          "finalizeCandidateHandoff"] },
      { label: "commit", code: octreeLosassoAdaptivePhiCommitWGSL,
        names: ["commitCandidate", "stampCandidateBootstrap", "syncAcceptedCommit",
          "stampCandidateRepair", "stampAcceptedAdvance"] },
      { label: "ghost", code: octreeLosassoAdaptivePhiGhostWGSL,
        names: ["deriveGhosts"] },
      { label: "evidence", code: octreeLosassoAdaptivePhiEvidenceWGSL,
        names: ["prepareTopologyEvidence", "publishTopologyEvidenceRows",
          "finishTopologyEvidence"] },
      { label: "schedule", code: octreeLosassoAdaptivePhiScheduleWGSL,
        names: ["scheduleCandidateSource", "scheduleCandidateRepair"] },
    ];
    const startLabel = process.env.FLUID_ADAPTIVE_PHI_COMPILE_START;
    if (startLabel && !modules.some((spec) => spec.label === startLabel)) {
      throw new Error(`unknown adaptive phi compile start module: ${startLabel}`);
    }
    let reachedStart = startLabel === undefined;
    for (const spec of modules) {
      reachedStart ||= spec.label === startLabel;
      if (!reachedStart) continue;
      mark(`module-start:${spec.label}`);
      const shaderModule = device.createShaderModule({
        label: `Adaptive phi ${spec.label} compile bisection`, code: spec.code,
      });
      for (const name of spec.names) {
        mark(`compile-start:${spec.label}:${name}`);
        await device.createComputePipelineAsync({
          label: `Adaptive phi compile bisection - ${name}`,
          layout: "auto", compute: { module: shaderModule, entryPoint: name },
        });
        mark(`compile-ok:${spec.label}:${name}`);
      }
      mark(`module-ok:${spec.label}`);
    }
    device.destroy();
    return;
  }

  const liveNodeCount = 8;
  const nodeCapacity = 128, leafCapacity = 128, pressureRowCapacity = 128;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buffer = (size: number) => device.createBuffer({ size, usage });
  const bank = (): LosassoSurfaceGraphBankSource => ({
    control: buffer(256), leaves: buffer(64 * leafCapacity),
    leafDirectory: buffer(16 * leafCapacity), nodes: buffer(16 * nodeCapacity),
    nodeDirectory: buffer(8 * nodeCapacity), constraints: buffer(48 * nodeCapacity),
    adjacency: buffer(48 * nodeCapacity), incidentLeaves: buffer(32 * nodeCapacity),
    leafLocator: buffer(64),
    phi: buffer(8 * nodeCapacity),
    nodalVelocity: buffer(32 * nodeCapacity), nodeValidity: buffer(4 * nodeCapacity),
    pressureRowToGraphLeaf: buffer(4 * pressureRowCapacity), pressureRowCapacity,
    leafCapacity, nodeCapacity, leafDispatchOffsetBytes: 17 * 4,
    nodeDispatchOffsetBytes: 20 * 4,
  });
  const graph = { accepted: bank(), candidate: bank() };
  const velocityReceipt = buffer(52);
  const phi = new WebGPUOctreeLosassoAdaptivePhi(device, graph, {
    nodeCapacity, leafCapacity, dimensions: [4, 4, 4], maximumLeafSpan: 4,
    cellSize: .05, redistanceIterations: 1,
  });
  mark("phi-initialize-start");
  await phi.initialize();
  mark("phi-initialize-complete");

  // One coherent span-4 leaf over the complete domain. Its eight independent
  // corner nodes give the compact-band classifiers real, non-zero work rather
  // than relying on a zero-count indirect dispatch.
  const graphControl = new Uint32Array(64);
  graphControl.set([1, 1, liveNodeCount, 1, 0, 1, 1, 0, liveNodeCount], 0);
  graphControl[14] = leafCapacity;
  graphControl[15] = nodeCapacity;
  graphControl.set([1, 1, 1], 17);
  graphControl.set([1, 1, 1], 20);
  graphControl[28] = 1;
  graphControl[30] = 1;
  graphControl[31] = 64;
  device.queue.writeBuffer(graph.accepted.control, 0, graphControl);

  const leaves = new Uint32Array(16 * leafCapacity);
  leaves.set([0, 0, 0, 4], 0);
  leaves.set([0, 1, 2, 3], 8);
  leaves.set([4, 5, 6, 7], 12);
  device.queue.writeBuffer(graph.accepted.leaves, 0, leaves);
  device.queue.writeBuffer(graph.accepted.leafDirectory, 0, Uint32Array.of(0, 4, 0, 1));

  const nodeItems = [0, 4, 20, 24, 100, 104, 120, 124];
  const nodes = new Uint32Array(4 * nodeCapacity);
  const nodeDirectory = new Uint32Array(2 * nodeCapacity);
  for (let i = 0; i < liveNodeCount; i += 1) {
    nodes.set([nodeItems[i]!, 1, 4, i], 4 * i);
    nodeDirectory.set([nodeItems[i]!, i], 2 * i);
  }
  device.queue.writeBuffer(graph.accepted.nodes, 0, nodes);
  device.queue.writeBuffer(graph.accepted.nodeDirectory, 0, nodeDirectory);

  const adjacency = new Uint32Array(12 * nodeCapacity); adjacency.fill(0xffff_ffff);
  for (let i = 0; i < liveNodeCount; i += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const positive = (i & (1 << axis)) === 0;
      adjacency[12 * i + axis + (positive ? 3 : 0)] = i ^ (1 << axis);
      adjacency[12 * i + 6 + axis + (positive ? 3 : 0)] = floatBits(.2);
    }
  }
  device.queue.writeBuffer(graph.accepted.adjacency, 0, adjacency);
  device.queue.writeBuffer(graph.accepted.incidentLeaves, 0, new Uint32Array(8 * nodeCapacity));

  const acceptedPhi = new Float32Array(2 * nodeCapacity);
  for (let i = 0; i < liveNodeCount; i += 1) {
    const value = (i & 2) === 0 ? -.025 : .025;
    acceptedPhi[2 * i] = value; acceptedPhi[2 * i + 1] = value;
  }
  device.queue.writeBuffer(graph.accepted.phi, 0, acceptedPhi);
  const velocity = new Uint32Array(8 * nodeCapacity);
  for (let i = 0; i < liveNodeCount; i += 1) {
    velocity[8 * i + 3] = 7; velocity[8 * i + 7] = 7;
  }
  device.queue.writeBuffer(graph.accepted.nodalVelocity, 0, velocity);
  device.queue.writeBuffer(graph.accepted.nodeValidity, 0,
    new Uint32Array(nodeCapacity).fill(0x77));
  const velocityReceiptWords = new Uint32Array(13);
  velocityReceiptWords[0] = 1; velocityReceiptWords[1] = floatBits(.35);
  velocityReceiptWords[6] = liveNodeCount; velocityReceiptWords[7] = 1;
  device.queue.writeBuffer(velocityReceipt, 0, velocityReceiptWords);

  const phiControl = new Uint32Array(20);
  phiControl.set([0x4150_4849, 1, 1, 1, 8, 1, 0, 1], 0);
  phiControl[16] = floatBits(.004);
  device.queue.writeBuffer(phi.source.control, 0, phiControl);

  const internal = phi as unknown as DiagnosticPhi;
  const readAcceptedSchedule = async (): Promise<Uint32Array> => {
    const readback = device.createBuffer({ size: 80,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const scheduleBroker = new PassBroker(device.createCommandEncoder());
    internal.encodeAcceptedLiveSchedule(scheduleBroker);
    scheduleBroker.copyBufferToBuffer(phi.source.nodeDispatch, 0, readback, 0, 12);
    scheduleBroker.copyBufferToBuffer(phi.source.leafDispatch, 0, readback, 12, 12);
    scheduleBroker.copyBufferToBuffer(phi.source.rowDispatch, 0, readback, 24, 12);
    scheduleBroker.copyBufferToBuffer(phi.source.faceDispatch, 0, readback, 36, 12);
    scheduleBroker.copyBufferToBuffer(internal.acceptedScheduleControl, 0,
      readback, 48, 32);
    device.pushErrorScope("validation");
    device.queue.submit([scheduleBroker.finish()]);
    await device.queue.onSubmittedWorkDone();
    const validation = await device.popErrorScope();
    assert.equal(validation?.message, undefined);
    await readback.mapAsync(GPUMapMode.READ);
    const words = Uint32Array.from(new Uint32Array(readback.getMappedRange()));
    readback.unmap(); readback.destroy();
    return words;
  };
  const liveSchedule = await readAcceptedSchedule();
  assert.deepEqual(Array.from(liveSchedule.subarray(0, 12)), [
    1, 1, 1, // 8 live nodes, not the two-workgroup node capacity
    1, 1, 1, // one live leaf, not the two-workgroup leaf capacity
    1, 1, 1, // one live pressure row, not the two-workgroup row capacity
    0, 1, 1, // no face authority in this focused binding fixture
  ]);
  assert.deepEqual(Array.from(liveSchedule.subarray(12, 20)),
    [1, 1, 0, 1, liveNodeCount, 1, 1, 0]);
  assert.equal(phi.plan.nodeDispatch[0], 2);
  assert.equal(phi.plan.leafDispatch[0], 2);
  assert.equal(phi.plan.pressureRowDispatch[0], 2);

  const staleControl = phiControl.slice(); staleControl[1] = 2;
  device.queue.writeBuffer(phi.source.control, 0, staleControl);
  const rejectedSchedule = await readAcceptedSchedule();
  assert.deepEqual(Array.from(rejectedSchedule.subarray(0, 12)), [
    0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1,
  ]);
  assert.notEqual(rejectedSchedule[14], 0, "generation mismatch must fail closed");
  assert.equal(rejectedSchedule[15], 0, "invalid tuple must not publish a schedule");
  device.queue.writeBuffer(phi.source.control, 0, phiControl);

  internal.writeParams(internal.advanceParams, .004, undefined, false);
  internal.flushReferenceVolumeDelta();
  const velocitySource = {
    control: graph.accepted.control,
    values: graph.accepted.nodalVelocity,
    receipt: velocityReceipt,
  };
  const buffers = internal.mainBuffers(graph.accepted, undefined, velocitySource,
    internal.advanceParams);

  device.pushErrorScope("validation");
  mark("encode-start");
  const broker = new PassBroker(device.createCommandEncoder());
  internal.runMain(broker, "prepareAdvance", buffers, 1);
  mark("prepare-encoded");
  internal.runMain(broker, "applyReferenceVolumeDelta", buffers, 1);
  mark("reference-delta-encoded");
  const publishBuffers = [internal.advanceParams, graph.accepted.control,
    graph.accepted.constraints, phi.source.control, graph.accepted.phi,
    internal.transportControl, internal.transportWorklist, internal.transportBandMask];
  internal.run(broker, internal.worklist.prepareTransportBand,
    [internal.transportControl, velocityReceipt], 1);
  internal.run(broker, internal.worklist.markTransportReach,
    [internal.advanceParams, graph.accepted.control, phi.source.control,
      graph.accepted.phi, internal.transportControl, internal.transportBandMask], 1);
  internal.run(broker, internal.worklist.markTransportInflow,
    [internal.advanceParams, graph.accepted.control, graph.accepted.nodes,
      internal.transportBandMask], 1);
  internal.run(broker, internal.worklist.markTransportConstrained,
    [graph.accepted.control, graph.accepted.constraints, phi.source.control,
      internal.transportBandMask], 1);
  internal.run(broker, internal.worklist.publishTransportIndependent, publishBuffers, 1);
  internal.run(broker, internal.worklist.publishTransportConstrained, publishBuffers, 1);
  mark("classify-encoded");
  internal.run(broker, internal.worklist.finalizeTransportDispatch,
    [internal.advanceParams, phi.source.control, internal.transportControl], 1);
  internal.run(broker, internal.worklist.publishTransportPartition,
    [internal.transportControl, phi.source.receipts], 1);
  mark("worklist-finalize-encoded");
  if (stage === "B" || stage === "C") {
    broker.copyBufferToBuffer(internal.transportControl, 20, internal.transportDispatch, 0, 24);
    internal.runBufferIndirect(broker, internal.backtrace, [internal.advanceParams,
      graph.accepted.control, graph.accepted.leaves, graph.accepted.nodes,
      graph.accepted.incidentLeaves, phi.source.control, graph.accepted.nodalVelocity,
      internal.transportControl, internal.transportWorklist, internal.transportDepartures],
    internal.transportDispatch, 0);
    internal.runBufferIndirect(broker, internal.transport, [internal.advanceParams,
      graph.accepted.control, graph.accepted.leaves, graph.accepted.nodes,
      graph.accepted.constraints, phi.source.control,
      graph.accepted.phi, internal.transportControl, internal.transportWorklist,
      internal.transportDepartures], internal.transportDispatch, 0);
    mark("transport-indirect-encoded", { offset: 20 });
  }
  if (stage === "C") {
    internal.runBufferIndirect(broker, internal.worklist.projectTransportedBand,
      [internal.advanceParams, graph.accepted.control, graph.accepted.constraints,
        phi.source.control, graph.accepted.phi, internal.transportControl,
        internal.transportWorklist], internal.transportDispatch, 12);
    internal.runBufferIndirect(broker, internal.predictorSnapshot, [internal.advanceParams,
      graph.accepted.control, phi.source.control, graph.accepted.phi, internal.distanceA],
    phi.source.nodeDispatch, 0);
    internal.runBufferIndirect(broker, internal.reverseBacktrace, [internal.advanceParams,
      graph.accepted.control, graph.accepted.leaves, graph.accepted.nodes,
      graph.accepted.incidentLeaves, phi.source.control, graph.accepted.nodalVelocity,
      internal.transportControl, internal.transportWorklist, internal.transportDepartures],
    internal.transportDispatch, 0);
    internal.runBufferIndirect(broker, internal.correction, [internal.advanceParams,
      graph.accepted.control, graph.accepted.leaves, graph.accepted.nodes,
      graph.accepted.constraints, phi.source.control, graph.accepted.phi,
      internal.transportControl, internal.transportWorklist, internal.transportDepartures,
      internal.distanceA], internal.transportDispatch, 0);
    internal.runBufferIndirect(broker, internal.worklist.projectTransportedBand,
      [internal.advanceParams, graph.accepted.control, graph.accepted.constraints,
        phi.source.control, graph.accepted.phi, internal.transportControl,
        internal.transportWorklist], internal.transportDispatch, 12);
    internal.runMain(broker, "captureTransportReceipt", buffers, 1);
    phi.encodeAcceptedFinalize(broker);
    mark("constraint-and-finalize-encoded", { offset: 32 });
  }
  const commands = broker.finish();
  mark("broker-finish-complete");
  device.queue.submit([commands]);
  mark("submit-complete");
  await device.queue.onSubmittedWorkDone();
  mark("submitted-work-done");
  const error = await device.popErrorScope();
  mark("scope-popped", { validation: error?.message });
  assert.equal(error?.message, undefined);
  phi.destroy(); device.destroy();
});
