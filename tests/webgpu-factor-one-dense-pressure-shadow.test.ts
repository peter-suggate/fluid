import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  FACTOR_ONE_DENSE_CELL_ROLE,
  buildFactorOneDensePressureHierarchy,
  factorOneDenseCoordinate,
  planFactorOneDensePressureHierarchy,
  type FactorOneDenseCoordinate,
  type FactorOneDenseHierarchy,
} from "../lib/factor-one-dense-pressure-hierarchy";
import {
  WebGPUFactorOneDensePressureShadow,
  factorOneDensePressureShadowShader,
  type FactorOneDenseSparseShadowPlan,
} from "../lib/webgpu-factor-one-dense-pressure-shadow";
import { PassBroker } from "../lib/webgpu-pass-broker";

const spgridSource = readFileSync(
  new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url),
  "utf8",
);

const floatBits = (value: number): number =>
  new Uint32Array(Float32Array.of(value).buffer)[0]!;

const capturedMiniHierarchy = (): FactorOneDenseHierarchy => {
  const plan = planFactorOneDensePressureHierarchy([16, 16, 16]);
  const counts = [1475, 214, 35, 8] as const;
  return buildFactorOneDensePressureHierarchy([16, 16, 16],
    counts.flatMap((count, level) =>
      Array.from({ length: count }, (_, local) => ({
        level,
        coordinate: factorOneDenseCoordinate(
          plan, level, plan.levelBases[level]! + local,
        ),
        role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly,
      }))));
};

const oddHierarchy = () => buildFactorOneDensePressureHierarchy([5, 3, 2], [
  { level: 0, coordinate: [0, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 0 },
  { level: 0, coordinate: [1, 0, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.ghost, owner: 3 },
  { level: 0, coordinate: [4, 2, 1], role: FACTOR_ONE_DENSE_CELL_ROLE.active, owner: 7 },
  { level: 0, coordinate: [3, 2, 0], role: FACTOR_ONE_DENSE_CELL_ROLE.multigridOnly },
]);

test("dense shadow shader retains isolated transactional publication entry points", () => {
  for (const entryPoint of [
    "clearCandidate", "scatterCandidate", "buildCandidateWorklists",
    "finalizeCandidate", "copyCandidateToAccepted", "finalizeAccepted",
  ]) assert.match(factorOneDensePressureShadowShader,
    new RegExp(`fn ${entryPoint}\\b`));
  assert.doesNotMatch(factorOneDensePressureShadowShader,
    /atomic(?:Store|Or)\(&?(?:solverControl|capture)/,
    "shadow failure paths must never mutate sparse lifecycle control");
  assert.match(factorOneDensePressureShadowShader,
    /capture\[6\].*C_CANDIDATE_EPOCH/,
    "accepted publication must require the sparse published generation");
  assert.match(factorOneDensePressureShadowShader, /solverControl\[0\]!=0u/);
  assert.doesNotMatch(factorOneDensePressureShadowShader, /solverControl\[1\]/,
    "the prior solve's legitimate converged word must not reject the next dense epoch");
});

test("SPGrid attaches and publishes the dense image only for geometric factor-1 setup", () => {
  assert.match(spgridSource,
    /if \(options\.geometricAggregateTransfers === true\) \{[\s\S]*new WebGPUFactorOneDensePressureShadow/);
  assert.match(spgridSource,
    /validateCandidateHierarchy[\s\S]*denseShadow\?\.encodeCandidate/);
  assert.match(spgridSource,
    /finalizeLifecycle[\s\S]*denseShadow\?\.encodeCommit/);
  assert.match(spgridSource,
    /get factorOneDenseAcceptedView\(\): FactorOneDenseAcceptedView \| undefined/);
  assert.match(spgridSource,
    /denseShadowSetupDispatchCount = this\.denseShadow\?\.encodedSetupDispatchCount \?\? 0/);
  assert.match(spgridSource,
    /this\.encodedSetupDispatchCount = 3 \+ 2 \+ 1 \+ 17 \+ 1 \+ 5 \+ 2\s*\+ denseShadowSetupDispatchCount/);
});

test("dense shadow GPU publishes odd and captured-mini coordinate images", {
  skip: !process.env.WEBGPU_NODE_MODULE
    && "set WEBGPU_NODE_MODULE for factor-1 dense shadow GPU checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();

  const run = async (
    dimensions: FactorOneDenseCoordinate,
    hierarchy: FactorOneDenseHierarchy,
    generation: number,
  ) => {
    const capacities = hierarchy.plan.levelVolumes.map((volume) => {
      let capacity = 1;
      while (capacity < volume) capacity *= 2;
      return capacity;
    });
    const offsets: number[] = [];
    let total = 0;
    for (const capacity of capacities) {
      offsets.push(total);
      total += capacity;
    }
    const sparsePlan: FactorOneDenseSparseShadowPlan = {
      levelCount: hierarchy.plan.levelCount,
      levelCapacities: capacities,
      levelOffsets: offsets,
      totalLevelSlots: total,
      worklistBaseWords: 0,
    };
    const state = new Uint32Array(26 * total);
    const topology = new Uint32Array(total);
    const dispatch = new Uint32Array(hierarchy.plan.levelCount * 12 + 5);
    hierarchy.occupiedLocalIndices.forEach((worklist, level) => {
      const sparseBase = offsets[level]!;
      const denseBase = hierarchy.plan.levelBases[level]!;
      dispatch[level * 12] = worklist.length;
      worklist.forEach((local, item) => {
        const sparseSlot = local;
        topology[sparseBase + item] = sparseSlot;
        state[sparseBase + sparseSlot] = local + 1;
        state[total + sparseBase + sparseSlot] = hierarchy.flags[denseBase + local]!;
        state[2 * total + sparseBase + sparseSlot] =
          floatBits(1 + level + Number(local) / Math.max(1, hierarchy.plan.levelVolumes[level]!));
        state[24 * total + sparseBase + sparseSlot] = hierarchy.owners[denseBase + local]!;
      });
      state[25 * total + sparseBase] = floatBits(1.5 + level / 8);
    });
    dispatch[hierarchy.plan.levelCount * 12] = 1;
    const levelDelta = new Uint32Array(hierarchy.plan.levelCount * 8);
    for (let level = 0; level < hierarchy.plan.levelCount; level += 1) {
      levelDelta[level * 8 + 1] = 3;
      levelDelta[level * 8 + 7] = state[25 * total + offsets[level]!]!;
    }
    const capture = new Uint32Array(12);
    capture[0] = generation;
    capture[6] = generation;
    const solverControl = new Uint32Array(8);
    const storageCopy = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      | GPUBufferUsage.COPY_SRC;
    const buffer = (label: string, data: Uint32Array, usage = storageCopy) => {
      const result = device.createBuffer({
        label,
        size: Math.max(4, data.byteLength),
        usage,
      });
      if (data.byteLength > 0) {
        device.queue.writeBuffer(
          result, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength,
        );
      }
      return result;
    };
    const acceptedTopology = buffer("dense shadow fixture accepted topology", topology);
    const acceptedState = buffer("dense shadow fixture accepted state", state);
    const acceptedDispatch = buffer("dense shadow fixture accepted dispatch", dispatch);
    const candidateTopology = buffer("dense shadow fixture candidate topology", topology);
    const candidateState = buffer("dense shadow fixture candidate state", state);
    const candidateDispatch = buffer("dense shadow fixture candidate dispatch", dispatch);
    const deltaBuffer = buffer("dense shadow fixture level delta", levelDelta);
    const captureBuffer = buffer("dense shadow fixture capture", capture);
    const solverBuffer = buffer("dense shadow fixture solver control", solverControl);
    device.pushErrorScope("validation");
    const shadow = new WebGPUFactorOneDensePressureShadow(device, {
      dimensions,
      sparsePlan,
      acceptedTopology,
      acceptedState,
      acceptedDispatch,
      candidateTopology,
      candidateState,
      candidateDispatch,
      levelDelta: deltaBuffer,
      captureControl: captureBuffer,
    });
    const constructionError = await device.popErrorScope();
    assert.equal(constructionError?.message ?? null, null);
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    shadow.encodeCandidate(broker, solverBuffer);
    shadow.encodeCommit(broker, solverBuffer);
    device.queue.submit([broker.finish()]);
    const differential = await shadow.readDifferential();
    assert.equal(differential.acceptedValid, 1);
    assert.equal(differential.acceptedEpoch, generation);
    assert.equal(differential.acceptedError, 0);
    assert.deepEqual(differential.occupiedCounts,
      hierarchy.occupiedLocalIndices.map((worklist) => worklist.length));
    assert.deepEqual(differential.discrepancies, []);
    shadow.destroy();
    for (const resource of [
      acceptedTopology, acceptedState, acceptedDispatch,
      candidateTopology, candidateState, candidateDispatch,
      deltaBuffer, captureBuffer, solverBuffer,
    ]) resource.destroy();
  };

  await run([5, 3, 2], oddHierarchy(), 7);
  await run([16, 16, 16], capturedMiniHierarchy(), 11);
  device.destroy();
});
