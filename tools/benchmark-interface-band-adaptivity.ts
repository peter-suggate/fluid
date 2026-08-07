/**
 * Objective Dawn gate on the octree's adaptive update criteria.
 *
 * Recreates the conditions the surface artifact was first measured under -- the
 * shipped `dam-break-ui` water box at `globalFineLevelSetFactor` 1, stepped to
 * the frame where the free surface goes blobby -- and scores the published
 * coarse level-set directory against the refinement rule Losasso et al. (2004)
 * section 6 state: refine a band about the interface, and coarsen only as you
 * move away from it. See `lib/octree-interface-band-audit.ts` for the predicate.
 *
 * The gate is `bandCoarseRows`: at factor 1 this directory is the entire
 * surface representation, so a leaf coarser than one cell sitting inside the
 * authored band is a facet the renderer has no choice but to draw. It must be
 * zero. `straddlingCoarseRows` states the same rule more sharply but rarely
 * fires, because the publisher stores one phi per row rather than a nodal
 * field -- see lib/octree-interface-band-audit.ts.
 *
 * The band/grading arms are not part of the gate. They exist because a dial
 * that cannot move the topology is a dial that is not wired to anything, and
 * that is exactly the failure this lane was built to catch: sweeping
 * `interfaceRefinementBandCells` 4 -> 12 once produced byte-identical trees.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/benchmark-interface-band-adaptivity.ts
 *
 * Environment:
 *   FLUID_SCENE            scene definition id (default water-box-dam-break)
 *   FLUID_SAMPLE_TIMES_S   comma-separated sample times (default 0,0.168,0.336)
 *   FLUID_MAX_DT           fixed step (default 0.008)
 *   FLUID_BAND_ARMS        comma-separated band:grading arms (default 1:1,4:1,4:3)
 *   FLUID_BAND_GATE        0 to report without failing the process
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import { getScenePreset } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import {
  auditOctreeInterfaceBand,
  type OctreeInterfaceBandAudit,
} from "../lib/octree-interface-band-audit";
import type { OctreeTopologyLeafCensus } from "../lib/webgpu-octree";
import { readBufferBinding } from "./webgpu-smoke-readbacks";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

interface Arm {
  readonly bandCells: number;
  readonly gradingLayers: number;
}

interface Sample {
  readonly t_s: number;
  readonly audit: OctreeInterfaceBandAudit;
  readonly leafCountsBySize: Readonly<Record<string, number>>;
  readonly topologyLeaves: number;
  readonly residentOwnerPages: number;
  readonly directory: Readonly<Record<string, number>>;
  readonly stats: Readonly<Record<string, number>>;
  readonly receipt: Readonly<Record<string, unknown>>;
}

const sceneId = process.env.FLUID_SCENE === "dam-break-ui" || !process.env.FLUID_SCENE
  ? "water-box-dam-break" : process.env.FLUID_SCENE;
const dt = Number(process.env.FLUID_MAX_DT ?? 0.008);
const sampleTimes = (process.env.FLUID_SAMPLE_TIMES_S ?? "0,0.168,0.336")
  .split(",").map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value) && value >= 0)
  .sort((left, right) => left - right);
const arms: readonly Arm[] = (process.env.FLUID_BAND_ARMS ?? "1:1,4:1,4:3")
  .split(",").map((entry) => {
    const [band, grading] = entry.split(":").map((value) => Number(value.trim()));
    assert.ok(Number.isFinite(band) && Number.isFinite(grading),
      `Malformed band arm: ${entry}`);
    return { bandCells: band!, gradingLayers: grading! };
  });
assert.ok(sampleTimes.length > 0, "At least one sample time is required");
assert.ok(dt > 0, "Step size must be positive");

await acquireWebGPUExclusiveLock(
  "dawn-benchmark", "tools/benchmark-interface-band-adaptivity.ts",
);
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    ...(process.env.FLUID_WEBGPU_ADAPTER
      ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
  ]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const requiredFeatures: GPUFeatureName[] = ["subgroups"];
  if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures, requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const report: Array<Record<string, unknown>> = [];
  for (const arm of arms) {
    const scene = getScenePreset(sceneId).create();
    scene.numerics.fixedDt_s = dt;
    scene.numerics.maxDt_s = dt;
    const solver = await octreeMethod.createSolverAsync!(device, scene, "balanced", {
      ...octreeMethod.presetFor("balanced"),
      // Factor 1 is the product default and the only configuration in which
      // this directory is the whole surface; the artifact lives here.
      globalFineLevelSetFactor: "1",
      interfaceRefinementBandCells: arm.bandCells,
      surfaceRefinementGradingLayers: arm.gradingLayers,
      secondaryParticles: "off",
    }, undefined, () => {}) as GPUSolverInstance;
    await device.queue.onSubmittedWorkDone();
    const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as
      [number, number, number];
    const projection = (solver as unknown as {
      octreeProjection?: {
        readSolveDiagnostics(): Promise<void>;
        readTopologyLeafCensus(): Promise<OctreeTopologyLeafCensus>;
      };
    }).octreeProjection;
    assert.ok(projection, "octree projection is unavailable");

    const samples: Sample[] = [];
    let step = 0;
    for (const target of sampleTimes) {
      const wanted = Math.round(target / dt);
      while (step < wanted) {
        step += 1;
        while (!solver.advanceTo(step * dt, [])) {
          await new Promise((resolve) => setImmediate(resolve));
        }
      }
      await device.queue.onSubmittedWorkDone();
      await projection.readSolveDiagnostics();
      const source = solver.coarseLevelSetSource;
      assert.ok(source, "factor-1 octree published no coarse level set");
      const bytes = await readBufferBinding(device, source.directory,
        source.directory.size ?? source.directory.buffer.size - (source.directory.offset ?? 0));
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      // The refinement predicate reads this directory as its ONLY phi
      // evidence. Its header says whether it is published at all this
      // generation, and how many B4 entries carry a phi interval.
      // Factor 1 has no fine-band summary object: its refinement evidence is
      // published by the dense coarse tracker into the same directory ABI.
      const projectionInternals = projection as unknown as {
        globalFineSummaries?: { directory: GPUBuffer };
        coarseOnlySummary?: { directory: GPUBuffer };
      };
      const summaryDirectory = projectionInternals.globalFineSummaries?.directory
        ?? projectionInternals.coarseOnlySummary?.directory;
      const summaryHeader = summaryDirectory
        ? new Uint32Array((await readBufferBinding(device,
          { buffer: summaryDirectory }, 64)).buffer.slice(0, 64))
        : undefined;
      // The dirty-tile transaction decides WHICH tiles the refinement ladder
      // is allowed to visit. When it rejects, dirtyCount and the retired count
      // are both forced to zero and no tile is refined at all -- which is
      // indistinguishable from the criteria declining, unless you read this.
      const compactionBuffer = (projection as unknown as {
        compaction?: GPUBuffer; dirtyFailureOffsetBytes?: number;
      });
      const dirtyHeader = compactionBuffer.compaction
        ? new Uint32Array((await readBufferBinding(device,
          { buffer: compactionBuffer.compaction }, 64)).buffer.slice(0, 64))
        : undefined;
      const dirtyFailure = compactionBuffer.compaction
        && compactionBuffer.dirtyFailureOffsetBytes !== undefined
        ? new Uint32Array((await readBufferBinding(device,
          { buffer: compactionBuffer.compaction, offset: compactionBuffer.dirtyFailureOffsetBytes },
          32)).buffer.slice(0, 32))
        : undefined;
      const census = await projection.readTopologyLeafCensus();
      const stats = await solver.readStats();
      // The factor-1 surface publisher no-ops its whole lattice when the
      // air-support publication it consumes is not valid for that advance. A
      // stalled advance is invisible in the directory header -- it still reads
      // PUBLISHED at the current generation -- so the receipt is the only place
      // the stall is counted.
      const receipt = await (solver as GPUSolverInstance & {
        readCoarseSurfaceTrackerReceipt?(): Promise<Record<string, number> | undefined>;
      }).readCoarseSurfaceTrackerReceipt?.();
      samples.push({
        stats: {
          pressureRequiredRows: stats.pressureRequiredRows ?? 0,
          pressureRowCapacity: stats.pressureRowCapacity ?? 0,
          pressureCapacityOverflow: stats.pressureCapacityOverflow ? 1 : 0,
          frontierCapacityOverflow: stats.frontierCapacityOverflow ? 1 : 0,
          maximumNeighborDelta: stats.maximumNeighborDelta ?? 0,
          // A reused topology is the previous epoch carried forward; a rebuild
          // recomputes every leaf from the current criteria. If the collapse
          // frames are the rebuilds, the criteria -- not the transport -- own
          // the artifact.
          topologyReused: (solver.info as unknown as Record<string, unknown>).topologyReused ? 1 : 0,
          topologyReuseCount: Number(
            (solver.info as unknown as Record<string, unknown>).topologyReuseCount ?? 0),
          summaryState: summaryHeader?.[0] ?? -1,
          summaryCount: summaryHeader?.[2] ?? -1,
          summaryCapacity: summaryHeader?.[3] ?? -1,
          summaryPublished: summaryHeader?.[9] ?? -1,
          summaryMaximumLevel: summaryHeader?.[7] ?? -1,
          dirtyTiles: dirtyHeader?.[0] ?? -1,
          retiredTiles: dirtyHeader?.[4] ?? -1,
          totalTiles: dirtyHeader?.[5] ?? -1,
          dirtyFailureReason: dirtyFailure?.[0] ?? -1,
          dirtyFailureStage: dirtyFailure?.[1] ?? -1,
          dirtyFailureSlot: dirtyFailure?.[2] ?? -1,
          dirtyFailureTile: dirtyFailure?.[3] ?? -1,
          dirtyFailureActive: dirtyFailure?.[4] ?? -1,
          dirtyFailureCapacity: dirtyFailure?.[6] ?? -1,
        },
        t_s: Number((step * dt).toFixed(6)),
        receipt: receipt ? {
          advances: receipt.advances, completions: receipt.completions,
          predictedCells: receipt.predictedCells, domainVolume: receipt.domainVolume,
          published: receipt.published ? 1 : 0, error: receipt.error,
          airUnpublishedAdvances: receipt.airUnpublishedAdvances,
          airErrorWord: receipt.airErrorWord, airValidWord: receipt.airValidWord,
          airFirstErrorStage: receipt.airFirstErrorStage,
          airFirstErrorItem: receipt.airFirstErrorItem,
        } : {},
        audit: auditOctreeInterfaceBand(words, dimensions, arm.bandCells),
        leafCountsBySize: census.leafCountsBySize,
        topologyLeaves: census.topologyLeaves,
        // Splitting a block allocates owner pages for its children. A pool at
        // its ceiling drops the split silently, which looks exactly like a
        // refinement criterion declining to fire.
        residentOwnerPages: census.residentOwnerPages,
        // A directory that saturates its capacity, or one whose generation
        // lags, coarsens for a reason that has nothing to do with the
        // refinement criteria; separate those causes before blaming the band.
        directory: {
          state: (words[0] ?? 0) >>> 0,
          generation: (words[1] ?? 0) >>> 0,
          rowCount: words[2] ?? 0,
          rowCapacity: source.rowCapacity,
          sourceGeneration: source.generation,
        },
      });
    }
    report.push({
      band: arm.bandCells, grading: arm.gradingLayers, dimensions,
      samples: samples.map((sample) => ({
        t_s: sample.t_s,
        topologyLeaves: sample.topologyLeaves,
        residentOwnerPages: sample.residentOwnerPages,
        leafCountsBySize: sample.leafCountsBySize,
        directory: sample.directory,
        stats: sample.stats,
        receipt: sample.receipt,
        rows: sample.audit.rows,
        straddlingRowsBySize: sample.audit.straddlingRowsBySize,
        straddlingCoarseRows: sample.audit.straddlingCoarseRows,
        straddlingCoarseCells: sample.audit.straddlingCoarseCells,
        flaggedInterfaceRows: sample.audit.flaggedInterfaceRows,
        bandCoarseRows: sample.audit.bandCoarseRows,
        bandCoarseRowsWaterSide: sample.audit.bandCoarseRowsWaterSide,
        bandCoarseRowsAirSide: sample.audit.bandCoarseRowsAirSide,
        monotonicityBreaks: sample.audit.monotonicityBreaks,
        maximumSizeByDistanceCells: sample.audit.maximumSizeByDistanceCells,
        straddlingCoarseRowsByOriginY: sample.audit.straddlingCoarseRowsByOriginY,
        worstStraddlingRows: sample.audit.worstStraddlingRows,
      })),
    });
    solver.destroy();
  }

  const worst = report.flatMap((arm) => (arm.samples as Array<Record<string, number>>)
    .map((sample) => sample.bandCoarseRows!));
  const gateFailed = worst.some((value) => value > 0);
  // Two arms whose every topology number is identical mean the dial is inert,
  // which is a distinct defect from the gate and has to be visible separately.
  const fingerprints = report.map((arm) => JSON.stringify(
    (arm.samples as Array<Record<string, unknown>>)
      .map((sample) => [sample.topologyLeaves, sample.leafCountsBySize])));
  const inertArms = fingerprints.length > 1
    && fingerprints.every((value) => value === fingerprints[0]);
  console.log(JSON.stringify({
    phase: "interface-band-adaptivity",
    scene: sceneId, dt, sampleTimes,
    validationErrors,
    gate: {
      invariant: "no leaf coarser than one cell may sit within the band of the interface",
      worstBandCoarseRows: Math.max(...worst),
      failed: gateFailed,
      inertArms,
    },
    arms: report,
  }, null, 1));
  device.destroy();
  if (gateFailed && process.env.FLUID_BAND_GATE !== "0") {
    process.exitCode = 1;
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
