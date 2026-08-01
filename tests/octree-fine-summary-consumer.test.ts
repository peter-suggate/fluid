import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  
  octreeProjectionShader,
  
} from "../lib/webgpu-octree";
import {
  FINE_LEVELSET_SUMMARY_CENTER_COMPLETE,
  FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY,
  
  fineLevelSetSummaryRefinementSignal,
  planFineLevelSetSummaryLeafLookup,
} from "../lib/webgpu-octree-fine-levelset-summary";

test("factor-4 and factor-8 octree leaves map to one aligned dyadic summary node", () => {
  assert.deepEqual(planFineLevelSetSummaryLeafLookup([8, 8, 8], [8, 8, 8], [4, 2, 0], 2), {
    level: 1, key: 518, brickSide: 2, expectedBrickCount: 8, expectedSampleCount: 512,
  });
  assert.deepEqual(planFineLevelSetSummaryLeafLookup([16, 16, 16], [8, 8, 8], [4, 2, 0], 2), {
    level: 2, key: 4614, brickSide: 4, expectedBrickCount: 64, expectedSampleCount: 4096,
  });
  assert.throws(() => planFineLevelSetSummaryLeafLookup([16, 8, 16], [8, 8, 8], [0, 0, 0], 2),
    /equal integer brick count/);
});

test("a published zero crossing always refines while absent coverage can never authorize coarsening", () => {
  const lookup = { expectedBrickCount: 64, expectedSampleCount: 4096 };
  const base = { published: true, directoryFlags: 0, found: true, entryFlags: 0,
    minimumPhi: 2, maximumPhi: 4, minimumAbsolutePhi: 2, brickCount: 64, sampleCount: 4096 };
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, minimumPhi: -1, maximumPhi: 1,
    brickCount: 1, sampleCount: 64 }, lookup, 0.5), "refine",
  "even a partial node's observed sign crossing is sufficient evidence");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, brickCount: 63 }, lookup, 0.5), "invalid");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, published: false }, lookup, 0.5), "invalid");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, directoryFlags: 1 }, lookup, 0.5), "invalid");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base,
    entryFlags: FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY, brickCount: 0, sampleCount: 0,
  }, lookup, 0.5), "complete-no-crossing", "coarse authority is an ABI flag, not an entry error");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base,
    entryFlags: FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY | 1,
  }, lookup, 0.5), "invalid", "low entry-flag bits remain fail-closed errors");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base,
    entryFlags: FINE_LEVELSET_SUMMARY_CENTER_COMPLETE,
  }, lookup, 0.5), "complete-no-crossing",
  "centre-stencil completeness is evidence, not an entry error");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, minimumAbsolutePhi: 0.25 }, lookup, 0.5), "refine");
  assert.equal(fineLevelSetSummaryRefinementSignal(base, lookup, 0.5), "complete-no-crossing");
});

test("Dawn compiles summary-consuming refinement at the portable ten-storage-buffer limit", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU summary-consumer checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "write-only", format: "rgba32float", viewDimension: "3d" } },
    ...[2, 3, 4, 5, 8, 9, 10, 11, 13].map((binding) => ({ binding, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" as const } })),
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
    { binding: 12, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
    { binding: 14, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: "read-write", format: "r32float", viewDimension: "3d" } },
    { binding: 15, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
  ] });
  const shaderModule = device.createShaderModule({ code: octreeProjectionShader });
  device.pushErrorScope("validation");
  // Compile the summary consumer plus the recurring dirty-coarse and
  // GPU-authored frontier scheduling entry points at the portable limit.
  for (const entryPoint of [
    "refineTopology",
    "refineTopologyCoarseDelta",
    "balanceTopologyCoarseDelta",
    "prepareFrontierDispatch",
    "sortFrontierCandidatesLocal",
  ]) {
    device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint, constants: {
        targetRefinementSize: entryPoint.includes("Coarse") ? 16 : 2,
      } } });
  }
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  await device.queue.onSubmittedWorkDone();
  device.destroy();
});
