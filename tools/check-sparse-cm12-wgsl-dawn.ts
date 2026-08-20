#!/usr/bin/env node
/**
 * Compile the Sparse CM12 resident shader, and only the shader.
 *
 * A WGSL error in this module surfaces during solver construction, which needs
 * a scene, an atlas, a packed topology and several megabytes of buffers — so
 * the cheapest question ("does it parse and does every entry point resolve?")
 * is normally answered by the most expensive test available. This asks it
 * directly: one device, one `createShaderModule`, one pipeline per entry point.
 *
 * Creating the pipelines matters as much as compiling the module. A module can
 * compile while an entry point fails to specialise — an override constant with
 * no default, a binding the layout does not carry — and that failure would
 * otherwise only appear on the frame that first dispatches it.
 */
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createWebgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";
import { createSparseCM12IncrementalActivityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity";
import { createSparseCM12CanonicalMembershipLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-canonical-membership";
import { createSparseCM12FramePlanLayout } from "../lib/core/sparse-cm12-frame-plan";
import { createSparseCM12FramePlanPresentationLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import { createSparseCM12FrameControl } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import {
  createSparseCM12ResidentScalarAuthorityLayout,
  createWebgpuSparseCM12ScalarAuthorityPlannerWGSL,
} from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-scalar-authority";
import { createSparseCM12ScalarWorkAuthority } from
  "../lib/methods/adaptive-mass/sparse-cm12-scalar-work-authority";
import { createSparseCM12SRR1IngressLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-srr1-runtime-adapter";
import { createSparseCM12PressureTopologyRepairLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair";
import { createSparseCM12ResidentPersistentPressureCacheLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-persistent-pressure-cache";
import { createSparseCM12FaceProjectionAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-face-projection-authority";
import { createSparseCM12VexActivityBatchLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-activity-batch";

const staticConcurrencyCheck = process.argv.includes("--static-concurrency-check");
const productionOnly = process.argv.includes("--production-only");
if (staticConcurrencyCheck) {
  const source = await readFile(fileURLToPath(import.meta.url), "utf8");
  const fanout = /\bPromise\.(?:all|allSettled|any|race)\s*\(/;
  if (fanout.test(source)) {
    throw new Error("Sparse CM12 Dawn checker must not fan out native pipeline compilation");
  }
  const compileCallCount = [...source.matchAll(/\.createComputePipelineAsync\s*\(/g)].length;
  if (compileCallCount < 4) {
    throw new Error(`Sparse CM12 Dawn checker concurrency guard found only ${compileCallCount} compile sites`);
  }
  console.log(JSON.stringify({
    phase: "sparse-cm12-wgsl-static-concurrency-check",
    passed: true,
    policy: "sequential-native-pipeline-compilation",
    compileCallCount,
  }));
  process.exit(0);
}

const dawnModule = process.env.WEBGPU_NODE_MODULE;
if (!dawnModule) {
  console.error("WEBGPU_NODE_MODULE is required; run via npm run check:sparse-cm12:wgsl");
  process.exit(2);
}

/** Entry points named in the shader source, in declaration order. */
function entryPoints(source: string): readonly string[] {
  return [...source.matchAll(/@compute\s+@workgroup_size\([^)]*\)\s+fn\s+([A-Za-z0-9_]+)/g)]
    .map((match) => match[1]!);
}

async function main(): Promise<void> {
  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-resident");
  let gpu: GPU | undefined;
  let device: GPUDevice | undefined;
  try {
    const { create, globals } = await import(dawnModule!) as {
      create: (flags: string[]) => GPU; globals: { GPUBufferUsage: unknown };
    };
    Object.assign(globalThis, globals);
    gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    // Ten storage buffers in one stage is past the WebGPU default, and the
    // shipping solver already requests the adapter's own ceiling. A check on a
    // default device would fail on the layout rather than on the shader.
    device = await adapter.requestDevice({
      requiredLimits: requiredFluidDeviceLimits(adapter.limits),
    });
    device.pushErrorScope("validation");

    const storage = { type: "storage" } as const;
    const bindGroupLayout = device.createBindGroupLayout({
      label: "Sparse CM12 WGSL check layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
        ...[2, 3, 4].map((binding) => ({ binding,
          visibility: GPUShaderStage.COMPUTE, buffer: storage })),
        ...[11, 12, 13].map((binding) => ({ binding,
          visibility: GPUShaderStage.COMPUTE, buffer: storage })),
        { binding: 14, visibility: GPUShaderStage.COMPUTE,
          buffer: storage },
        ...[15, 16].map((binding) => ({ binding,
          visibility: GPUShaderStage.COMPUTE, buffer: storage })),
      ],
    });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    let compiledEntryPoints = 0;
    const variants: readonly (readonly [4 | 8 | 16, 4 | 8 | 16])[] = productionOnly
      ? [[16, 16]]
      : [[4, 4], [8, 4], [8, 8], [16, 4], [16, 8], [16, 16]];
    for (const [brickFineResolution, presentationPageResolution] of variants) {
      const variant = `B${brickFineResolution}/P${presentationPageResolution}`;
      const temporal = { headerBaseWords: 64,
        cellListBaseWords: 72,
        rowListBaseWords: 1096,
        cellFlagABaseWords: 2120,
        cellFlagBBaseWords: 3144,
        totalWords: 4168 };
      const pressure = { edgeCoefficientBaseWords: 4096, cellSlotBaseWords: 8192,
        rowSlotBaseWords: 9216, cellChangeBaseWords: 10240,
        rowChangeBaseWords: 11264, brickStateBaseWords: 12288,
        rowTopologyStampBaseWords: 12352, aggregateEdgeForFineEdgeBaseWords: 13376,
        aggregateEdgeSourceBaseWords: 14400,
        hierarchyEdgeForAggregateBaseWords: [14464],
        headerBaseWords: 15488, totalWords: 15496 };
      const activity = createSparseCM12IncrementalActivityLayout({
        baseWords: temporal.totalWords, stableTileCount: 512, brickCount: 8,
      });
      const canonicalMembership = createSparseCM12CanonicalMembershipLayout({
        baseWords: activity.totalWords, cellCapacity: 1024, rowCapacity: 2048,
      });
      const framePlan = brickFineResolution === presentationPageResolution
        ? createSparseCM12FramePlanLayout({
          baseWords: Math.ceil(canonicalMembership.totalWords / 64) * 64,
          brickCapacity: 8, brickFineResolution, packetCount: 6,
        }) : undefined;
      const presentation = framePlan
        ? createSparseCM12FramePlanPresentationLayout({
          baseWords: framePlan.totalWords, pageCapacity: 8,
          brickFineResolution, pageResolution: presentationPageResolution,
          packetIndex: 5,
        }) : undefined;
      const productionB16P16 = brickFineResolution === 16
        && presentationPageResolution === 16;
      const frameControl = productionB16P16 ? createSparseCM12FrameControl({
        baseWords: 32768, cellWorkgroups: 16, rowWorkgroups: 32,
        bodyCapacity: 0, d4Capable: true, rigidCapable: false,
        boundaryCapable: false,
      }) : undefined;
      const scalarAuthority = productionB16P16 && presentation
        ? createSparseCM12SRR1IngressLayout({
          baseWords: presentation.totalWords, tileCapacity: 512,
        }) : undefined;
      const pressureTopologyRepair = productionB16P16 && frameControl
        ? createSparseCM12PressureTopologyRepairLayout({
          baseWords: frameControl.layout.totalWords,
          brickCapacity: 8, rowCapacity: 2048,
          brickFineResolution: 16, presentationPageResolution: 16,
        }) : undefined;
      const persistentPressureCache = pressureTopologyRepair
        ? createSparseCM12ResidentPersistentPressureCacheLayout({
          baseWords: pressureTopologyRepair.totalWords,
          cellCount: 1024, rowCount: 2048, directedEdgeCount: 1024,
          brickCount: 8, aggregateEdgeCount: 32,
          hierarchyLevelCounts: [8], hierarchyEdgeLevelCounts: [32],
        }) : undefined;
      const faceProjectionAuthority = persistentPressureCache
        ? createSparseCM12FaceProjectionAuthorityLayout({
          baseWords: persistentPressureCache.bufferSizeWords,
          rowCapacity: 2048, cellCapacity: 1024,
        }) : undefined;
      const vexActivityBatch = productionB16P16 && scalarAuthority
        ? createSparseCM12VexActivityBatchLayout({
          activityTailWords: scalarAuthority.totalWords,
          stateTailFloats: 65536,
          cellCapacity: 1024,
        }) : undefined;
      const source = createWebgpuSparseCM12ResidentWGSL(
        brickFineResolution,
        presentationPageResolution,
        temporal, pressure, activity, canonicalMembership,
        framePlan, presentation, frameControl?.layout, scalarAuthority,
        pressureTopologyRepair,
        persistentPressureCache,
        faceProjectionAuthority,
        vexActivityBatch,
      );
      const shaderModule = device.createShaderModule({
        label: `Sparse CM12 resident WGSL check ${variant}`,
        code: source,
      });
      const info = await shaderModule.getCompilationInfo();
      const errors = info.messages.filter((message) => message.type === "error");
      if (errors.length > 0) {
        for (const error of errors) {
          console.error(`${variant} ${error.lineNum}:${error.linePos} ${error.message}`);
          const lines = source.split("\n");
          const begin = Math.max(0, error.lineNum - 5);
          const end = Math.min(lines.length, error.lineNum + 4);
          for (let line = begin; line < end; line += 1) {
            console.error(`${line + 1}: ${lines[line]}`);
          }
        }
        throw new Error(`${variant}: ${errors.length} WGSL compilation error(s)`);
      }

      const names = entryPoints(source);
      if (names.length === 0) throw new Error("no compute entry points found");
      // Keep Metal/Dawn native compilation strictly bounded. Launching every
      // entry point at once retains one native waiter per request; six resident
      // variants can otherwise accumulate hundreds of threads and enough
      // memory pressure to trip the WindowServer watchdog. Declaration order
      // also makes the first failing entry point deterministic.
      for (const entryPoint of names) {
        await device.createComputePipelineAsync({
          label: `Sparse CM12 ${variant} WGSL check ${entryPoint}`,
          layout, compute: { module: shaderModule, entryPoint },
        });
      }
      // The journal's snapshot variant is the one specialisation the shipping
      // encode compiles, so a check that skipped it would miss exactly the
      // override-constant failures this script exists to catch.
      await device.createComputePipelineAsync({
        label: `Sparse CM12 ${variant} journalIteration snapshot variant`,
        layout,
        compute: { module: shaderModule, entryPoint: "journalIteration",
          constants: { JOURNAL_SNAPSHOT: 1 } },
      });
      if (framePlan) {
        await device.createComputePipelineAsync({
          label: `Sparse CM12 ${variant} FPL1 presentation verification`,
          layout,
          compute: { module: shaderModule,
            entryPoint: "verifySparseCM12FramePlanCurrentStage",
            constants: { CM12_FRAME_PLAN_VERIFY_STAGE: 5 } },
        });
      }
      compiledEntryPoints += names.length;
    }
    // The SAW1 planner is intentionally isolated from the resident bind group;
    // compile it separately so an ABI error cannot hide behind resident-only
    // entry-point coverage.
    const scalarResident = createSparseCM12ResidentScalarAuthorityLayout({
      baseWords: 4096, tileCapacity: 512,
    });
    const scalar = createSparseCM12ScalarWorkAuthority({ tileCapacity: 512 });
    const plannerSource = createWebgpuSparseCM12ScalarAuthorityPlannerWGSL(
      scalar.layout, scalarResident.candidateWords,
    );
    const plannerModule = device.createShaderModule({
      label: "Sparse CM12 SAW1 dedicated planner WGSL check", code: plannerSource,
    });
    const plannerInfo = await plannerModule.getCompilationInfo();
    const plannerErrors = plannerInfo.messages.filter((message) => message.type === "error");
    if (plannerErrors.length > 0) {
      for (const error of plannerErrors) {
        console.error(`SAW1 ${error.lineNum}:${error.linePos} ${error.message}`);
      }
      throw new Error(`SAW1: ${plannerErrors.length} WGSL compilation error(s)`);
    }
    const plannerBindGroupLayout = device.createBindGroupLayout({
      label: "Sparse CM12 SAW1 checker layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE,
          buffer: { type: "read-only-storage" } },
      ],
    });
    const plannerLayout = device.createPipelineLayout({
      bindGroupLayouts: [plannerBindGroupLayout],
    });
    const plannerEntries = entryPoints(plannerSource);
    for (const entryPoint of plannerEntries) {
      await device.createComputePipelineAsync({
        label: `Sparse CM12 SAW1 WGSL check ${entryPoint}`,
        layout: plannerLayout, compute: { module: plannerModule, entryPoint },
      });
    }
    compiledEntryPoints += plannerEntries.length;

    const scope = await device.popErrorScope();
    if (scope) throw new Error(`validation error: ${scope.message}`);
    console.log(`Sparse CM12 resident WGSL: ${compiledEntryPoints} entry points compiled across ${variants.length} B/P variants`);
  } finally {
    if (device) {
      try { await device.queue.onSubmittedWorkDone(); } catch { /* Device fault already reported. */ }
      device.destroy();
      // Keep the Dawn instance strongly reachable until device retirement has
      // reached the native event pump, matching the temporal harness lifecycle.
      await new Promise<void>((resolve) => setImmediate(resolve));
    }
    gpu = undefined;
    await releaseWebGPUExclusiveLock();
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
