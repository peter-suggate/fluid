import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { octreeProjectionShader } from "../lib/webgpu-octree";
import {
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
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, brickCount: 63 }, lookup, 0.5), "fallback");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, published: false }, lookup, 0.5), "fallback");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, directoryFlags: 1 }, lookup, 0.5), "fallback");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, minimumAbsolutePhi: 0.25 }, lookup, 0.5), "refine");
  assert.equal(fineLevelSetSummaryRefinementSignal(base, lookup, 0.5), "complete-no-crossing");
});

test("summary sizing aliases binding 4 without adding storage bindings or pressure semantics", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const layout = source.slice(source.indexOf("this.layout = device.createBindGroupLayout"),
    source.indexOf("this.pipelineLayout =", source.indexOf("this.layout = device.createBindGroupLayout")));
  assert.equal((layout.match(/buffer: \{ type: \"(?:read-only-)?storage\" \}/g) ?? []).length, 10);
  assert.doesNotMatch(layout, /binding: 16/);
  assert.match(octreeProjectionShader,
    /fn fineSummaryWord\(index: u32\) -> u32 \{ return bitcast<u32>\(pressureIn\[index\]\); \}/);

  const sizing = source.slice(source.indexOf("const dispatchCoarse ="), source.indexOf("pass.end();",
    source.indexOf("const dispatchCoarse =")));
  const uses = sizing.split("\n").filter((line) => line.includes("fineSummarySizingGroup"));
  assert.equal(uses.length, 3);
  assert.match(uses[0], /refineCoarsePipelines/);
  assert.match(uses[1], /dispatchCandidates\(level\.full, level\.active/);
  assert.match(uses[2], /dispatchRetiredCandidates\(level\.retired/);
  const callGraph = octreeProjectionShader.slice(octreeProjectionShader.indexOf("struct FineLeafSummary"),
    octreeProjectionShader.indexOf("fn balanceTopologyAt"));
  assert.equal((callGraph.match(/pressureIn\[/g) ?? []).length, 1,
    "the only pressureIn access reachable from summary-bound refinement is the raw bitcast reader");
  assert.doesNotMatch(callGraph, /pressureOut\[/);
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
  const module = device.createShaderModule({ code: octreeProjectionShader });
  device.pushErrorScope("validation");
  // All six wrappers have the same transitive resource set; the static gate
  // above proves their dispatch set and this representative pipeline asks
  // Dawn to validate that resource set against the portable limit.
  for (const entryPoint of ["refineTopology"]) {
    device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint, constants: { targetRefinementSize: 2 } } });
  }
  assert.equal(await device.popErrorScope(), null);
  await device.queue.onSubmittedWorkDone();
  device.destroy();
});
