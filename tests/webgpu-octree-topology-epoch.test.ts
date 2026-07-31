import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_TOPOLOGY_EPOCH_ERROR,
  WebGPUOctreeTopologyEpoch,
  octreeTopologyEpochHistoryShader,
  octreeTopologyEpochWGSL,
  publishReadyOctreeTopology,
  validateOctreeTopologyCandidate,
  type OctreeTopologyEpochState,
} from "../lib/webgpu-octree-topology-epoch";
import { WebGPUOctreeSimulationOwnerPages } from "../lib/webgpu-octree-owner-pages";
import { PassBroker } from "../lib/webgpu-pass-broker";

function state(): OctreeTopologyEpochState {
  return {
    activeEpoch: 7,
    activeTable: 0,
    activeGeneration: 40,
    candidateEpoch: 0,
    candidateTable: 1,
    candidateGeneration: 0,
    candidateCount: 0,
    candidateError: 0,
    flipReady: false,
    topologyHash: 0,
    catalogVersion: 5,
  };
}

test("candidate validation never mutates active authority before the next-substep flip", () => {
  const decision = validateOctreeTopologyCandidate(state(), {
    epoch: 8,
    generation: 41,
    errorFlags: 0,
    count: 19,
    capacity: 32,
    encodedSupportCells: 9,
    requiredSupportCells: 9,
    topologyHash: 0x1234,
    catalogVersion: 5,
  }, 41, 5);
  assert.equal(decision.accepted, true);
  assert.equal(decision.state.activeEpoch, 7);
  assert.equal(decision.state.activeTable, 0);
  assert.equal(decision.state.candidateTable, 1);
  assert.equal(decision.state.flipReady, true);
  const published = publishReadyOctreeTopology(decision.state);
  assert.equal(published.activeEpoch, 8);
  assert.equal(published.activeTable, 1);
  assert.equal(published.activeGeneration, 41);
  assert.equal(published.flipReady, false);
});

test("stale, incomplete, unsupported, and mismatched candidates fail closed", () => {
  const decision = validateOctreeTopologyCandidate(state(), {
    epoch: 7,
    generation: 40,
    errorFlags: 2,
    count: 0,
    capacity: 0,
    encodedSupportCells: 3,
    requiredSupportCells: 4,
    topologyHash: 0,
    catalogVersion: 4,
  }, 41, 5);
  assert.equal(decision.accepted, false);
  assert.equal(decision.state.flipReady, false);
  for (const bit of [
    OCTREE_TOPOLOGY_EPOCH_ERROR.staleEpoch,
    OCTREE_TOPOLOGY_EPOCH_ERROR.generationMismatch,
    OCTREE_TOPOLOGY_EPOCH_ERROR.publisherRejected,
    OCTREE_TOPOLOGY_EPOCH_ERROR.supportClosure,
    OCTREE_TOPOLOGY_EPOCH_ERROR.catalogVersion,
    OCTREE_TOPOLOGY_EPOCH_ERROR.emptyPublication,
  ]) assert.notEqual(decision.error & bit, 0);
  assert.deepEqual(publishReadyOctreeTopology(decision.state), decision.state);
});

test("a rejected candidate preserves accepted authority and the next valid attempt recovers", () => {
  const accepted = state();
  const rejected = validateOctreeTopologyCandidate(accepted, {
    epoch: 8, generation: 41, errorFlags: 1, count: 19, capacity: 32,
    encodedSupportCells: 9, requiredSupportCells: 9, topologyHash: 0x1111, catalogVersion: 5,
  }, 41, 5);
  const afterReject = publishReadyOctreeTopology(rejected.state);
  assert.equal(afterReject.activeEpoch, accepted.activeEpoch);
  assert.equal(afterReject.activeTable, accepted.activeTable);
  assert.equal(afterReject.activeGeneration, accepted.activeGeneration);

  const retry = validateOctreeTopologyCandidate(afterReject, {
    epoch: 8, generation: 42, errorFlags: 0, count: 19, capacity: 32,
    encodedSupportCells: 9, requiredSupportCells: 9, topologyHash: 0x2222, catalogVersion: 5,
  }, 42, 5);
  assert.equal(retry.accepted, true);
  assert.equal(publishReadyOctreeTopology(retry.state).activeGeneration, 42);
});

test("GPU row publication is candidate-isolated and requires the successful epoch token", () => {
  assert.match(octreeTopologyEpochWGSL, /epoch\.reserved\[0u\]=0u/);
  assert.match(octreeTopologyEpochWGSL, /epoch\.reserved\[0u\]=1u/);
  assert.match(octreeTopologyEpochWGSL, /let commit=epoch\.reserved\[0u\]==1u/);
  assert.doesNotMatch(octreeTopologyEpochWGSL, /expectedGeneration/,
    "a shared host uniform cannot identify separate invocations in one command buffer");
  assert.match(octreeTopologyEpochWGSL, /age>=0x80000000u/);
  assert.match(octreeTopologyEpochWGSL,
    /prepareCandidateRowCommitDispatch[\s\S]*select\(1u,\(epoch\.rowCount\+63u\)\/64u/,
    "accepted row publication must dispatch from the GPU-validated live count");
  assert.match(octreeTopologyEpochWGSL,
    /acceptedHeaders\[base\+word\]=candidateHeaders\[base\+word\]/);
  assert.match(octreeTopologyEpochWGSL, /pressureA\[row\]=seed;pressureB\[row\]=seed/);
  assert.match(octreeTopologyEpochWGSL,
    /retained=select\(0u,acceptedStructured\[2u\][\s\S]*rowCountControl\[0u\]=select\(retained,epoch\.rowCount,commit\)/,
    "a rejected candidate must restore the row count from accepted Section 4 authority, not rejection scratch");
});

test("ready topology singleton handoffs retain only the storage-to-indirect boundary", () => {
  const events: string[] = [];
  let passIndex = 0;
  const encoder = {
    beginComputePass(descriptor?: GPUComputePassDescriptor) {
      const index = ++passIndex;
      events.push(`begin:${descriptor?.label}`);
      return {
        setPipeline(pipeline: unknown) { events.push(`pipeline:${pipeline}`); },
        setBindGroup(_index: number, group: unknown) { events.push(`group:${group}`); },
        dispatchWorkgroups(count: number) { events.push(`dispatch:${count}`); },
        dispatchWorkgroupsIndirect(buffer: unknown, offset: number) {
          events.push(`indirect:${buffer}@${offset}`);
        },
        end() { events.push(`end:${index}`); },
      };
    },
  } as unknown as GPUCommandEncoder;
  const broker = new PassBroker(encoder, { isolateLabels: false });
  const epoch = {
    destroyed: false,
    validateExpectedGeneration() {},
    commitGatePipeline: "gate-pipeline",
    commitGateGroup: "gate-group",
    prepareCommitRowsPipeline: "prepare-pipeline",
    prepareCommitRowsGroup: "prepare-group",
    commitRowsPipeline: "rows-pipeline",
    commitRowsGroup: "rows-group",
    commitRowsDispatch: "rows-dispatch",
  } as unknown as WebGPUOctreeTopologyEpoch;
  WebGPUOctreeTopologyEpoch.prototype.encodeReadyCommitGate.call(epoch, broker, 7);

  assert.equal(broker.computePassCount, 2,
    "gate+prepare share one pass; the indirect row consumer opens the second");
  assert.equal(broker.hasOpenComputePass, true,
    "accepted rows must leave the coupled storage commit pass open");

  const ownerPages = {
    destroyed: false,
    topologyResidency: {},
    topologyGroup: "owner-group",
    commit: "owner-pipeline",
  } as unknown as WebGPUOctreeSimulationOwnerPages;
  WebGPUOctreeSimulationOwnerPages.prototype.encodeReadyCommit.call(ownerPages, broker);
  assert.equal(broker.computePassCount, 2,
    "owner publication must join the accepted-row storage pass");
  broker.fence("downstream publication boundary");

  assert.deepEqual(events, [
    "begin:Open coupled topology ready-commit gate",
    "pipeline:gate-pipeline", "group:gate-group", "dispatch:1",
    "pipeline:prepare-pipeline", "group:prepare-group", "dispatch:1",
    "end:1",
    "begin:Commit accepted topology row identities and pressure seed",
    "pipeline:rows-pipeline", "group:rows-group", "indirect:rows-dispatch@0",
    "group:owner-group", "pipeline:owner-pipeline", "dispatch:1",
    "end:2",
  ]);
});

test("coupled validation decodes descriptor and SPGrid candidate control ABIs exactly", () => {
  assert.match(octreeTopologyEpochWGSL,
    /descriptor\[2u\]!=0u\|\|descriptor\[4u\]!=0u\|\|descriptor\[1u\]!=count/,
    "descriptor errorCount and flags reject; sameOrCoarser is a legitimate population count");
  assert.doesNotMatch(octreeTopologyEpochWGSL, /descriptor\[6u\](?:!=0u|\|=error)/);
  assert.match(octreeTopologyEpochWGSL, /descriptor\[2u\]\|=error;descriptor\[4u\]\|=error/);
  assert.match(octreeTopologyEpochWGSL, /spgrid\[10u\]!=generation/,
    "SPGrid sourceGeneration must match the coupled topology generation");
  assert.match(octreeTopologyEpochWGSL,
    /spgrid\[4u\]!=spgrid\[0u\][\s\S]*spgrid\[2u\]!=spgrid\[3u\][\s\S]*spgrid\[2u\]>spgrid\[1u\]/,
    "SPGrid ready generation and changed-page validation must be coherent");
  assert.doesNotMatch(octreeTopologyEpochWGSL,
    /spgrid\[2u\]!=spgrid\[1u\]|spgrid\[1u\]!=0u&&spgrid\[2u\]==0u/,
    "an unchanged Section 4.3 hierarchy is a valid zero-page delta");
});

test("every topology-epoch pipeline binds its own reflected auto layout and exact ABI", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const buffer = () => ({ size: 4096, destroy() {} }) as unknown as GPUBuffer;
  const layouts = new Map<string, object>();
  const groups: Array<{ layout: object; bindings: number[] }> = [];
  const device = {
    queue: { writeBuffer() {} },
    createBuffer: () => buffer(),
    createShaderModule: () => ({}),
    createComputePipeline: ({ label }: { label: string }) => {
      const layout = { label }; layouts.set(label, layout);
      return { getBindGroupLayout: () => layout };
    },
    createBindGroup: ({ layout, entries }: { layout: object; entries: GPUBindGroupEntry[] }) => {
      groups.push({ layout, bindings: entries.map((entry) => entry.binding) }); return {};
    },
  } as unknown as GPUDevice;
  const resources = {
    ownerArena: buffer(), ownerCandidate: buffer(), frontier: buffer(),
    descriptorCandidateControl: buffer(), topologyCandidateControl: buffer(),
    structuredCandidateControl: buffer(), structuredAcceptedControl: buffer(), boundaryCandidateControl: buffer(),
    spgridCandidateControl: buffer(), candidateLeafHeaders: buffer(), acceptedLeafHeaders: buffer(),
    candidatePressure: buffer(), candidatePressureHistory: buffer(),
    acceptedPressureHistory: buffer(), pressureA: buffer(), pressureB: buffer(),
    rowCountControl: buffer(),
  };
  const epoch = new WebGPUOctreeTopologyEpoch(device, resources,
    { rowCapacity: 128, slotCapacity: 256, catalogVersion: 5 });
  assert.deepEqual(groups.map((group) => group.layout), [
    layouts.get("validateInactiveTopologyEpoch"),
    layouts.get("beginReadyTopologyCommit"),
    layouts.get("prepareCandidateRowCommitDispatch"),
    layouts.get("commitCandidateRows"),
  ]);
  assert.deepEqual(groups.map((group) => group.bindings), [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 3, 4, 5, 6, 7, 8, 9],
    [1, 16],
    [1, 10, 11, 12, 13, 14, 15, 17],
  ]);
  epoch.destroy();
});

test("pressure history carry is an exact opt-in topology-commit variant", () => {
  assert.equal(octreeTopologyEpochHistoryShader(false), octreeTopologyEpochWGSL);
  const enabled = octreeTopologyEpochHistoryShader(true);
  assert.match(enabled, /@binding\(18\).*candidatePressureHistory/);
  assert.match(enabled, /@binding\(19\).*acceptedPressureHistory/);
  assert.match(enabled,
    /acceptedPressureHistory\[row\]=candidatePressureHistory\[row\]/);
});
