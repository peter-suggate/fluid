import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { octreeProjectionShader, WebGPUOctreeProjection } from "../lib/webgpu-octree";
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
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, brickCount: 63 }, lookup, 0.5), "fallback");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, published: false }, lookup, 0.5), "fallback");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base, directoryFlags: 1 }, lookup, 0.5), "fallback");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base,
    entryFlags: FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY, brickCount: 0, sampleCount: 0,
  }, lookup, 0.5), "complete-no-crossing", "coarse authority is an ABI flag, not an entry error");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base,
    entryFlags: FINE_LEVELSET_SUMMARY_COARSE_AUTHORITY | 1,
  }, lookup, 0.5), "fallback", "low entry-flag bits remain fail-closed errors");
  assert.equal(fineLevelSetSummaryRefinementSignal({ ...base,
    entryFlags: FINE_LEVELSET_SUMMARY_CENTER_COMPLETE,
  }, lookup, 0.5), "complete-no-crossing",
  "centre-stencil completeness is evidence, not an entry error");
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

  const sizing = source.slice(source.indexOf("const dispatchCoarse ="),
    source.indexOf("broker.fence(\"octree topology and frontier publication complete\")",
      source.indexOf("const dispatchCoarse =")));
  const uses = sizing.split("\n").filter((line) => line.includes("fineSummarySizingGroup"));
  assert.equal(uses.length, 5);
  assert.match(uses[0], /refineCoarsePipelines/);
  assert.match(uses[1], /dispatchCandidates\(level\.full, level\.delta/);
  assert.match(uses[2], /candidates\.setBindGroup/);
  assert.match(uses[3], /candidateSort\.setBindGroup/);
  assert.match(uses[4], /merge\.setBindGroup/);
  const callGraph = octreeProjectionShader.slice(octreeProjectionShader.indexOf("struct FineLeafSummary"),
    octreeProjectionShader.indexOf("fn balanceTopologyAt"));
  assert.equal((callGraph.match(/pressureIn\[/g) ?? []).length, 1,
    "the only pressureIn access reachable from summary-bound refinement is the raw bitcast reader");
  assert.doesNotMatch(callGraph, /pressureOut\[/);
  assert.match(callGraph,
    /result\.centerPhi = bitcast<f32>\(fineSummaryWord\(base \+ 7u\)\);[\s\S]*result\.centerValid = \(entryFlags & 0x3fc00000u\) == 0x3fc00000u/,
    "word 7 is the exact current fine phi at every dyadic summary node centre");
  assert.match(callGraph,
    /let fineComplete = fineSummaryWord\(base \+ 5u\) == expectedBricks[\s\S]*result\.complete = result\.coarseAuthority \|\| fineComplete;[\s\S]*result\.centerValid = \(entryFlags & 0x3fc00000u\) == 0x3fc00000u/,
    "exact centre evidence is independent of whole sparse-brick completeness");
  assert.doesNotMatch(callGraph, /result\.centerValid =[^;]*!result\.coarseAuthority/,
    "a unified fine+coarse entry must retain its exact fine phase classifier");
  assert.match(octreeProjectionShader,
    /if\(fine\.found\)\{[\s\S]*if\(fine\.centerValid\)\{wet=fine\.centerPhi<0\.0;\}/,
    "recurring pressure leaves of every size consume their current complete centre stencil");
});

test("incremental topology separates structural refinement from wet frontier decisions", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const signature = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn classifyTopologyTileSignature"),
    octreeProjectionShader.indexOf("fn currentRigidWord"),
  );
  const dirty = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn buildDirtyTileDelta"),
    octreeProjectionShader.indexOf("// -----------------------------------------------------------------------------",
      octreeProjectionShader.indexOf("fn buildDirtyTileDelta")));
  assert.match(signature,
    /let structuralDecision = cell \^ \(owner\.size[\s\S]*let frontierDecision = structuralDecision \^ select[\s\S]*TILE_SIGNATURE_VALID_MAGIC/,
    "persistent signatures must independently hash structural identity and wet/dry membership");
  assert.match(signature,
    /let structuralUnchanged = valid && all\([\s\S]*let frontierUnchanged = frontierValid && all\([\s\S]*TILE_SIGNATURE_STRUCTURAL_CHANGED[\s\S]*TILE_SIGNATURE_FRONTIER_CHANGED/,
    "raw fine-value changes must not alter the structural signature bit");
  assert.match(dirty,
    /let changed = compaction\[tileSignatureChangedBase\(\) \+ slot\][\s\S]*changed & TILE_SIGNATURE_STRUCTURAL_CHANGED[\s\S]*appendDirtyTileRing\(tileIndex/,
    "only structural signature changes may enter reset/refine/balance");
  assert.match(dirty,
    /fn buildDirtyFrontierDelta\(\)[\s\S]*changed & TILE_SIGNATURE_FRONTIER_CHANGED[\s\S]*tileFrontierChangeFlagsBase\(\) \+ tileIndex[\s\S]*compaction\[dirtyListBase\(\) \+ dirtyCount\] = tileIndex/,
    "wet membership changes must feed only the compact liquid frontier");
  assert.doesNotMatch(dirty,
    /if \(wet\) \{ compaction\[tileChangeFlagsBase\(\)/,
    "wet-only changes must not enter the structural topology/residency stamps");
  assert.match(dirty,
    /compaction\[signature \+ 4u\] = 0u[\s\S]*compaction\[frontierSignature \+ 4u\] = 0u[\s\S]*appendDirtyTileRing\(tileIndex/,
    "retirement must invalidate both persistent signatures and dirty surviving neighbors");
  assert.doesNotMatch(dirty, /deltaWord|fineDelta|finePageTopologyTile/,
    "fine payload-refresh deltas must not masquerade as structural dirtiness");

  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString().replace(/\s+/g, "");
  assert.match(rebuild,
    /this\.topologyDecisionGroup/);
  assert.match(rebuild,
    /signatures\.setPipeline\(this\.classifyTopologyTileSignaturePipeline\).*signatures\.dispatchWorkgroupsIndirect\(this\.solveDispatch,48\).*mark\.setPipeline\(this\.buildDirtyTileDeltaPipeline\).*mark\.dispatchWorkgroups\(1\).*frontierDelta\.setPipeline\(this\.buildDirtyFrontierDeltaPipeline\)/,
    "decision comparison must be active-tile-sized before the compact singleton publication");
  assert.match(source,
    /this\.topologyDecisionGroup = this\.createProjectionGroup\([\s\S]*globalFineSummaries\?\.directory[\s\S]*topologyTileStateBuffer[\s\S]*coarseDirectory/,
    "the stable decision group must retain fine classification, residency, and coarse fallback");
  assert.doesNotMatch(source, /FLUID_OCTREE_(?:INCREMENTAL|CHANGE_DRIVEN)/,
    "the exact incremental path has no legacy runtime escape hatch");
  assert.doesNotMatch(source, /if \(this\.globalFineSummaries\) return false/,
    "fine-summary publication must no longer force a full active-tile rebuild");
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
