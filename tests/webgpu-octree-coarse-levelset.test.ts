import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_FINE_PHI_CONTRIBUTION_BYTES,
  WebGPUOctreeCoarseLevelSet,
  octreeCoarsePhiBootstrapActivityShader,
  octreeCoarsePhiBootstrapShader,
  planOctreeCoarsePhi,
} from "../lib/webgpu-octree-coarse-levelset";
import { OCTREE_COARSE_PHI_BYTES, OCTREE_COARSE_PHI_FLAG } from "../lib/octree-coarse-levelset";
import { OCTREE_FINE_SEED_STATE } from "../lib/octree-fine-seed-leaves";
import {
  createGPULogicalActivityAdoptionContext,
  gpuLogicalActivityTaskDescriptions,
} from "../lib/gpu-logical-activity-adoption";
import {
  assertWGSLActivityBindingEligibility,
  auditWGSLComputeBindingReachability,
} from "../lib/wgsl-binding-reachability";
import { usePerformanceInstrumentationStore } from "../lib/stores/performance-instrumentation-store";
import { PassBroker } from "../lib/webgpu-pass-broker";

// Standalone Dawn execution below does not install the frame-owned recorder.
// Activity source and compilation are exercised explicitly in the focused test.
usePerformanceInstrumentationStore.getState().setMode("timeline");

test("coarse phi owns only compact row records", () => {
  const small = planOctreeCoarsePhi(16);
  const large = planOctreeCoarsePhi(32);
  assert.deepEqual(small, {
    rowCapacity: 16,
    recordBytes: 16 * OCTREE_COARSE_PHI_BYTES,
    allocatedBytes: 16 * OCTREE_COARSE_PHI_BYTES,
  });
  assert.equal(large.allocatedBytes, small.allocatedBytes * 2);
  assert.equal(OCTREE_FINE_PHI_CONTRIBUTION_BYTES, 16,
    "the production power schedule retains its aggregate ABI");
});

test("bootstrap shader contains no recurring correction machinery", () => {
  assert.match(octreeCoarsePhiBootstrapShader, /fn bootstrapCoarsePhiFromSurfaceLeaves/);
  assert.match(octreeCoarsePhiBootstrapShader,
    new RegExp(`leaf\\.flags&${OCTREE_FINE_SEED_STATE.live}u`));
  assert.doesNotMatch(octreeCoarsePhiBootstrapShader,
    /FineContribution|Control|correctCoarsePhi|finalizeCoarsePhi|rowStatus|atomic/);
});

test("coarse-phi bootstrap activity is conditional and within the storage-binding budget", () => {
  const audit = auditWGSLComputeBindingReachability(
    octreeCoarsePhiBootstrapShader,
    "bootstrapCoarsePhiFromSurfaceLeaves",
  );
  assert.deepEqual(audit.reachableFunctions, ["bootstrapCoarsePhiFromSurfaceLeaves", "finite"]);
  assert.deepEqual(audit.storage.map(({ binding, name }) => ({ binding, name })), [
    { binding: 0, name: "fineSeedLeaves" },
    { binding: 1, name: "coarse" },
  ]);
  assert.deepEqual(assertWGSLActivityBindingEligibility(audit), {
    entryPoint: "bootstrapCoarsePhiFromSurfaceLeaves",
    productionStorageCount: 2,
    activityStorageCount: 1,
    totalStorageCount: 3,
    maximumStorageCount: 10,
  });

  const disabled = createGPULogicalActivityAdoptionContext({
    moduleId: "octree/coarse-levelset",
    profile: { enabled: false, generation: 0x4400 },
  });
  assert.equal(octreeCoarsePhiBootstrapActivityShader(disabled), octreeCoarsePhiBootstrapShader);

  const generation = 0x4401;
  const enabled = createGPULogicalActivityAdoptionContext({
    moduleId: "octree/coarse-levelset",
    profile: { enabled: true, generation },
  });
  enabled.describeTask("bootstrap-coarse-phi", {
    id: "gpu.physics.coarse-phi.bootstrap",
    label: "Coarse phi · bootstrap surface rows",
    phaseId: "fine-sdf-advection",
  });
  const shader = enabled.module(octreeCoarsePhiBootstrapActivityShader(enabled)).code;
  assert.match(shader, /@group\(3\) @binding\(0\)/);
  assert.equal((shader.match(/fluidGpuLogicalActivityWorkgroup\(/g) ?? []).length, 2,
    "one helper definition and one bootstrap checkpoint are generated");
  assert.deepEqual(gpuLogicalActivityTaskDescriptions(generation), {
    [enabled.taskId("bootstrap-coarse-phi")]: {
      id: "gpu.physics.coarse-phi.bootstrap",
      label: "Coarse phi · bootstrap surface rows",
      phaseId: "fine-sdf-advection",
    },
  });
});

test("Dawn bootstraps compact coarse phi from adapter-style live rows", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for coarse-phi GPU checks",
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
  const compilation = await device.createShaderModule({ code: octreeCoarsePhiBootstrapShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const activity = createGPULogicalActivityAdoptionContext({
    moduleId: "octree/coarse-levelset/dawn",
    profile: { enabled: true, generation: 0x4402 },
  });
  const activityCompilation = await device.createShaderModule({
    code: activity.module(octreeCoarsePhiBootstrapActivityShader(activity)).code,
  }).getCompilationInfo();
  assert.deepEqual(activityCompilation.messages.filter((message) => message.type === "error"), []);

  const coarse = new WebGPUOctreeCoarseLevelSet(device, 2);
  const leafData = new ArrayBuffer(2 * 64);
  const leafWords = new Uint32Array(leafData);
  const leafFloats = new Float32Array(leafData);
  for (let row = 0; row < 2; row += 1) {
    const base = row * 16;
    leafWords[base] = row * 8;
    leafWords[base + 3] = 8;
    leafWords[base + 4] = OCTREE_FINE_SEED_STATE.live;
    leafFloats[base + 8] = row === 0 ? -2 : 3;
  }
  const leaves = device.createBuffer({
    size: leafData.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint8Array(leaves.getMappedRange()).set(new Uint8Array(leafData));
  leaves.unmap();
  const readback = device.createBuffer({
    size: coarse.plan.recordBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  const liveRowDispatch = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(liveRowDispatch, 0, new Uint32Array([1, 1, 1]));
  coarse.encodeBootstrapFromSurfaceLeaves(broker, leaves, liveRowDispatch);
  broker.fence("coarse phi bootstrap readback");
  encoder.copyBufferToBuffer(coarse.records, 0, readback, 0, coarse.plan.recordBytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const result = readback.getMappedRange().slice(0);
  readback.unmap();
  const floats = new Float32Array(result);
  const words = new Uint32Array(result);
  assert.deepEqual([floats[0], floats[1], floats[2]], [-2, -2, -2]);
  assert.deepEqual([floats[4], floats[5], floats[6]], [3, 3, 3]);
  assert.equal(words[3] & (OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite),
    OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite);
  assert.equal(words[7] & (OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite),
    OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite);

  coarse.destroy();
  leaves.destroy();
  liveRowDispatch.destroy();
  readback.destroy();
  device.destroy();
});
