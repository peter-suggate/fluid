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
import { createSparseCM12PressureTopologyRepairLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair";
import { createSparseCM12ResidentPersistentPressureCacheLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-persistent-pressure-cache";
import { createSparseCM12PressureExecutionImageLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-execution-image";
import { createSparseCM12TopologyEffectsAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-topology-effects-authority";
import { createSparseCM12DirtyFaceRowMaskLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-dirty-face-row-masks";
import { createSparseCM12VelocityExtensionResidentLayouts } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import type { SparseCM12InternedBoundaryLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import type { SparseCM12InternedRefLookupLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-ref-lookup";
import { createSparseCM12TransportProducerMaskLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-producer-masks";
import { createSparseCM12TransportExecutionImageLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image";
import { createSparseCM12TransportPacketAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-packet-authority";
import { SPARSE_CM12_LENSES } from
  "../lib/methods/adaptive-mass/sparse-cm12-stage-lenses";
import { stageLensProgramWGSL } from "../lib/core/webgpu-stage-lens-overlay";

const staticConcurrencyCheck = process.argv.includes("--static-concurrency-check");
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
    const variants: readonly (readonly [8, 8])[] = [[8, 8]];
    for (const [brickFineResolution, presentationPageResolution] of variants) {
      const variant = `B${brickFineResolution}/P${presentationPageResolution}`;
      const pressure = { aggregateEdgeForFineEdgeBaseWords: 13376,
        aggregateEdgeSourceBaseWords: 14400,
        hierarchyEdgeForAggregateBaseWords: [14464],
        headerBaseWords: 15488, totalWords: 15497 };
      const activity = createSparseCM12IncrementalActivityLayout({
        baseWords: 4168, brickCount: 8,
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
      const productionMatchedProfile = true;
      const frameControl = productionMatchedProfile ? createSparseCM12FrameControl({
        baseWords: 32768, cellWorkgroups: 16, rowWorkgroups: 32,
        bodyCapacity: 0, d4Capable: true, rigidCapable: false,
        boundaryCapable: false,
        brickFineResolution, presentationPageResolution,
      }) : undefined;
      const pressureTopologyRepair = productionMatchedProfile && frameControl
        ? createSparseCM12PressureTopologyRepairLayout({
          baseWords: frameControl.layout.totalWords,
          brickCapacity: 8,
          brickFineResolution,
          presentationPageResolution,
        }) : undefined;
      const persistentPressureCache = pressureTopologyRepair
        ? createSparseCM12ResidentPersistentPressureCacheLayout({
          baseWords: pressureTopologyRepair.totalWords,
          cellCount: 1024, rowCount: 2048, directedEdgeCount: 1024,
          brickCount: 8, aggregateEdgeCount: 32,
          hierarchyLevelCounts: [8], hierarchyEdgeLevelCounts: [32],
        }) : undefined;
      const pressureExecutionImage = createSparseCM12PressureExecutionImageLayout({
        baseWords: 200000, cellCapacity: 1024, brickCapacity: 8,
        hierarchyCapacity: 8, brickFineResolution: 8,
        presentationPageResolution: 8,
      });
      const topologyEffects = createSparseCM12TopologyEffectsAuthorityLayout({
        baseWords: persistentPressureCache!.controlEndWords,
        ptrCapacity: 8, ptrLeafCapacity: 1,
      });
      const velocityExtension = productionMatchedProfile && presentation
        ? createSparseCM12VelocityExtensionResidentLayouts({
          activityTailWords: presentation.totalWords,
          stateTailFloats: 65536,
          cellCapacity: 1024,
          packetCapacity: 8 * 64,
          brickFineResolution,
        }) : undefined;
      const dirtyFaceRows = velocityExtension
        ? createSparseCM12DirtyFaceRowMaskLayout({
          baseWords: velocityExtension.activity.totalWords,
          cellCapacity: 1024,
          packetCapacity: velocityExtension.activity.packetCapacity,
          dispatchPacketsPerLeaf: velocityExtension.activity.dispatchPacketsPerLeaf,
          dispatchPacketCount: velocityExtension.activity.dispatchPacketCount,
        }) : undefined;
      const iboLayout: SparseCM12InternedBoundaryLayout = {
        leafCapacity: 8, canonicalCapacity: 24, templateCount: 8,
        templatePayloadWords: 256, canonicalBaseWords: 64,
        templateDirectoryBaseWords: 448, templatePayloadBaseWords: 512,
        immutableWords: 1024, immutableBytes: 4096,
        slotBaseWords: [1024, 2048], slotLeafBaseWords: [1088, 2112],
        slotRefBaseWords: [1280, 2304], wordsPerSlot: 1024,
        bytesPerSlot: 4096, totalWords: 3072, totalBytes: 12288,
      };
      const refLookupLayout: SparseCM12InternedRefLookupLayout = {
        baseWords: 768, canonicalCapacity: 24, sideDirectoryCount: 144,
        directoryBaseWords: 768, templateDirectoryBaseWords: 840,
        entryBaseWords: 844, templateCount: 4, templateEntryCount: 8,
        entryCount: 32, fallbackAnchorBaseWords: 860, fallbackAnchorCount: 24,
        levelsPerLeaf: 3,
        maximumEntriesPerSide: 3, totalWords: 1024, totalBytes: 1024,
      };
      const internedBoundaryImage = { layout: iboLayout, refLookupLayout,
          traSupplementLayout: { baseWords: 1024, templateCount: 4,
            directoryBaseWords: 1040, totalWords: 2048, totalBytes: 4096 },
          baseWords: 131072,
          semanticAuthority: {
            geometryBaseWords: 135168, geometryOffsetBaseWords: 8,
            geometryNeighborBaseWords: 17, authorityBaseWords: 135424,
            leafCapacity: 8, immutableContentHash: 1,
            immutableCertificateHash: 1,
          } };
      const transportProducerMasks = createSparseCM12TransportProducerMaskLayout({
        baseWords: 120000, packetCapacity: 512,
      });
      const transportExecutionImage = createSparseCM12TransportExecutionImageLayout({
        brickFineResolution, logicalBrickDimensions: [2, 2, 2], leafCapacity: 8,
      });
      const logicalOwnerDirectory = {
        layout: {
          brickFineResolution,
          presentationPageResolution,
          logicalBrickDimensions: [2, 2, 2] as const,
          logicalBrickCount: 8,
          residentBrickCount: 8,
          maximumSpanLog: 0,
          atlasGeneration: 1,
          recordBaseWords: 16,
          totalWords: 32,
          totalBytes: 128,
        },
        baseWords: 100000,
        packedOwner16BaseWords: 100032,
      };
      const transportPacketAuthority = createSparseCM12TransportPacketAuthorityLayout({
        baseWords: 160000, packetCapacity: transportExecutionImage.packetCapacity,
        dispatchPacketsPerLeaf: velocityExtension!.activity.dispatchPacketsPerLeaf,
        dispatchPacketCount: velocityExtension!.activity.dispatchPacketCount,
      });
      const source = createWebgpuSparseCM12ResidentWGSL(
        brickFineResolution,
        presentationPageResolution,
        pressure, activity, canonicalMembership,
        framePlan, presentation, frameControl?.layout, pressureTopologyRepair,
        persistentPressureCache,
        velocityExtension,
        pressureExecutionImage,
        logicalOwnerDirectory, 0, undefined,
        transportExecutionImage, transportPacketAuthority,
        transportProducerMasks,
        undefined, undefined, internedBoundaryImage,
        topologyEffects, undefined, dirtyFaceRows,
        250000,
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
    // Lens programs, composed exactly as the overlay composes them. Naga
    // already parses these; Dawn is what the browser actually runs, and the
    // failures the two disagree about — a struct a publication declares twice,
    // a binding the preamble numbers past the backend's ceiling — are the ones
    // that would otherwise first appear when a user opens the flyout.
    let compiledLensPrograms = 0;
    for (const lens of SPARSE_CM12_LENSES) {
      for (const program of Object.keys(lens.programs)) {
        const name = `${lens.id}#${program}`;
        const lensModule = device.createShaderModule({
          label: `Sparse CM12 lens WGSL check ${name}`,
          code: stageLensProgramWGSL(lens, program),
        });
        const lensInfo = await lensModule.getCompilationInfo();
        const lensErrors = lensInfo.messages.filter((message) => message.type === "error");
        if (lensErrors.length > 0) {
          for (const error of lensErrors) {
            console.error(`${name} ${error.lineNum}:${error.linePos} ${error.message}`);
          }
          throw new Error(`${name}: ${lensErrors.length} WGSL compilation error(s)`);
        }
        compiledLensPrograms += 1;
      }
    }

    const scope = await device.popErrorScope();
    if (scope) throw new Error(`validation error: ${scope.message}`);
    console.log(`Sparse CM12 compiled-transport WGSL: ${compiledEntryPoints} entry points compiled across ${variants.length} B/P variants, ${compiledLensPrograms} stage-lens programs`);
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
