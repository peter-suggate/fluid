import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS,
  OCTREE_PROJECTION_ACTIVITY_MODULE_ID,
  OCTREE_PROJECTION_ACTIVITY_TASKS,
  OCTREE_PROJECTION_CORE_BUFFER_LAYOUT,
  OCTREE_PROJECTION_CORE_TEXTURE_LAYOUT,
  OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT,
  octreeProjectionActivityShader,
  octreeProjectionShader,
} from "../lib/webgpu-octree";
import {
  auditExplicitComputeBufferBindings,
  auditWGSLComputeBindingReachability,
} from "../lib/wgsl-binding-reachability";
import { createGPULogicalActivityAdoptionContext } from "../lib/gpu-logical-activity-adoption";
import { GPU_LOGICAL_ACTIVITY_STRATIFIED_SAMPLE_COUNT } from "../lib/gpu-logical-activity";

const computeEntryPoints = [...octreeProjectionShader.matchAll(
  /@compute\b(?:\s*@[A-Za-z_]\w*\s*\([^)]*\))*\s*fn\s+([A-Za-z_]\w*)\s*\(/g,
)].map((match) => match[1]!);

function explicitBufferAudit(entries: readonly {
  readonly binding: number;
  readonly type: "uniform" | "storage" | "read-only-storage";
}[]) {
  return auditExplicitComputeBufferBindings(entries.map(({ binding, type }) => ({
    binding,
    visibility: 0x4,
    buffer: { type },
  })));
}

test("octree projection layout families reserve an activity-safe storage slot", () => {
  const core = explicitBufferAudit(OCTREE_PROJECTION_CORE_BUFFER_LAYOUT);
  const frontierSort = explicitBufferAudit(OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT);

  assert.equal(core.storageCount, 9);
  assert.deepEqual(core.storageBindings, [2, 3, 4, 5, 8, 10, 11, 13, 15]);
  assert.deepEqual(core.uniformBindings, [6]);
  assert.ok(core.storageCount + 1 <= 10,
    "the recurring topology family must retain one storage slot for activity instrumentation");

  assert.equal(frontierSort.storageCount, 4);
  assert.deepEqual(frontierSort.storageBindings, [2, 3, 9, 13]);
  assert.deepEqual(frontierSort.uniformBindings, [6, 7]);
  assert.ok(frontierSort.storageCount + 1 <= 10,
    "the exceptional large-frontier sort must also remain activity eligible");
});

test("only the exact frontier sort family reaches scratch binding nine", () => {
  assert.equal(computeEntryPoints.length, new Set(computeEntryPoints).size,
    "compute entry points must remain uniquely auditable");
  const scratchConsumers = computeEntryPoints.filter((entryPoint) =>
    auditWGSLComputeBindingReachability(octreeProjectionShader, entryPoint)
      .bindings.some(({ binding }) => binding === 9));

  assert.deepEqual(scratchConsumers, ["sortFrontierCandidates"]);
  assert.equal(OCTREE_PROJECTION_CORE_BUFFER_LAYOUT.some(({ binding }) => Number(binding) === 9), false);
  assert.equal(OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT.some(({ binding }) => binding === 9), true);
});

test("every projection entry point is covered by its explicit layout family", () => {
  // The family is buffers plus textures. Taking only the buffer layout made
  // every texture the shader legitimately reaches read as uncovered, so the
  // texture entries are now published beside the buffers and consumed here.
  const coreBindings = new Set<number>([
    ...OCTREE_PROJECTION_CORE_BUFFER_LAYOUT.map(({ binding }) => binding),
    ...OCTREE_PROJECTION_CORE_TEXTURE_LAYOUT.map(({ binding }) => binding),
  ]);
  const sortBindings = new Set<number>(
    OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT.map(({ binding }) => binding),
  );

  for (const entryPoint of computeEntryPoints) {
    const audit = auditWGSLComputeBindingReachability(octreeProjectionShader, entryPoint);
    const layout = entryPoint === "sortFrontierCandidates" ? sortBindings : coreBindings;
    const uncovered = audit.bindings.filter(({ binding }) => !layout.has(binding));
    assert.deepEqual(uncovered, [],
      `${entryPoint} reaches bindings absent from its explicit pipeline family`);
  }
});

test("projection activity variant is conditional, bounded, exhaustive, and storage-safe", () => {
  const disabled = createGPULogicalActivityAdoptionContext({
    moduleId: OCTREE_PROJECTION_ACTIVITY_MODULE_ID,
    profile: { enabled: false, generation: 8 },
  });
  assert.equal(octreeProjectionActivityShader(disabled), octreeProjectionShader);

  const enabled = createGPULogicalActivityAdoptionContext({
    moduleId: OCTREE_PROJECTION_ACTIVITY_MODULE_ID,
    profile: { enabled: true, generation: 9 },
  });
  const source = octreeProjectionActivityShader(enabled);
  const variant = enabled.module(source, "projection-activity-test");
  assert.equal(OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS.length, computeEntryPoints.length);
  assert.equal((source.match(/fluidGpuLogicalActivityWorkgroup\(/g) ?? []).length,
    computeEntryPoints.length, "each dispatch entry point gets one progress checkpoint");
  assert.equal((variant.code.match(/struct FluidGpuLogicalActivityBuffer/g) ?? []).length, 1);
  // "Bounded" moved from an inlined `< 128u` guard in each entry point to the
  // shared helper's stratified sampling, so it is asserted on the emitted
  // module rather than on the pre-variant source. The property is stronger
  // than the old guard: records per dispatch are capped by the sample count
  // regardless of how many workgroups the dispatch has.
  assert.match(variant.code,
    /let sampleCount = min\(FLUID_GPU_ACTIVITY_STRATIFIED_SAMPLE_COUNT, total\);/,
    "records per dispatch must be bounded by the stratified sample count, not the dispatch size");
  assert.equal(GPU_LOGICAL_ACTIVITY_STRATIFIED_SAMPLE_COUNT, 16);
  assert.ok(Object.values(OCTREE_PROJECTION_ACTIVITY_TASKS)
    .every(({ phaseId }) => phaseId === "coarse-grid"));
  for (const entryPoint of computeEntryPoints) {
    const audit = auditWGSLComputeBindingReachability(variant.code, entryPoint);
    assert.ok(audit.storageCount <= 10, `${entryPoint} exceeds the activity storage budget`);
    assert.ok(audit.bindings.some(({ group, binding }) => group === 3 && binding === 0),
      `${entryPoint} does not reach the activity recorder`);
  }
});

test("Dawn accepts every activity-enabled split projection pipeline layout", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU layout checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([
    `backend=${process.env.WEBGPU_BACKEND ?? "metal"}`,
  ]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  const activity = createGPULogicalActivityAdoptionContext({
    moduleId: OCTREE_PROJECTION_ACTIVITY_MODULE_ID,
    profile: { enabled: true, generation: 11 },
  });
  const layout = (entries: readonly {
    readonly binding: number;
    readonly type: "uniform" | "storage" | "read-only-storage";
  }[], terrain = false) => {
    const primary = device.createBindGroupLayout({ entries: [
      ...entries.map(({ binding, type }) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type },
      })),
      ...(terrain ? [{ binding: 12, visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" as const, viewDimension: "2d" as const } }] : []),
    ] });
    return device.createPipelineLayout({ bindGroupLayouts: [
      primary,
      device.createBindGroupLayout({ entries: [] }),
      device.createBindGroupLayout({ entries: [] }),
      device.createBindGroupLayout({ entries: [{ binding: 0,
        visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] }),
    ] });
  };
  const coreLayout = layout(OCTREE_PROJECTION_CORE_BUFFER_LAYOUT, true);
  const sortLayout = layout(OCTREE_PROJECTION_FRONTIER_SORT_BUFFER_LAYOUT);
  const shaderModule = device.createShaderModule({
    code: activity.module(octreeProjectionActivityShader(activity), "projection-dawn").code,
  });

  device.pushErrorScope("validation");
  for (const entryPoint of computeEntryPoints) {
    const frontierSort = entryPoint === "sortFrontierCandidates";
    const constants: Record<string, number> | undefined = entryPoint.includes("TopologyCoarse")
      ? { targetRefinementSize: 16 } : undefined;
    device.createComputePipeline({ layout: frontierSort ? sortLayout : coreLayout, compute: {
      module: shaderModule,
      entryPoint,
      ...(constants ? { constants } : {}),
    } });
  }
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  device.destroy();
});
