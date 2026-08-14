import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
// Composition root for this entry point: importing the method catalog installs
// the simulation methods and the octree coarse-dynamics lanes, without which
// constructing a solver throws rather than silently running the wrong backend.
import "../lib/methods";
import { losassoMethod } from "../lib/methods/losasso/method";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { createMinimalPowerDamBreakScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import { compareScalarFields } from "../lib/harness/webgpu-smoke-scenarios";
import { readCubicVolumeField } from "../lib/harness/webgpu-smoke-readbacks";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

interface Arm {
  readonly label: string;
  readonly gated: boolean;
  readonly adaptivity: 0 | 1;
  readonly maximumLeafSize: 2 | 4 | 8 | 16 | 32;
  readonly interfaceBandCells: number;
}

interface TopologyCensus {
  readonly generation: number;
  readonly residentOwnerPages: number;
  readonly topologyLeaves: number;
  readonly representedCells: number;
  readonly leafCountsBySize: Readonly<Record<string, number>>;
  readonly coarseLeafCountsByOriginY: readonly number[];
}

interface Result {
  readonly arm: Arm;
  readonly construction_ms: number;
  readonly simulationWall_ms: number;
  readonly wallPerStep_ms: number;
  readonly initialTopology: TopologyCensus;
  readonly topology: TopologyCensus;
  readonly field: Float32Array;
  readonly fieldSummary: Awaited<ReturnType<typeof readCubicVolumeField>>["summary"];
  readonly pressureRows: number;
  readonly pressureIterations: number;
  readonly representedVolumeCellSum: number;
  readonly traceTotal_ms: number | null;
  readonly tracePhases_ms: Readonly<Record<string, number>> | null;
  readonly pressureProfile: readonly Readonly<Record<string, unknown>>[] | null;
}

const parseArm = (source: string): Arm => {
  const [mode, leafText, bandText] = source.split(":");
  assert.ok(mode === "off" || mode === "on" || mode === "zero",
    `arm "${source}" must start with off, on, or zero`);
  const maximumLeafSize = Number(leafText);
  assert.ok(maximumLeafSize === 2 || maximumLeafSize === 4
    || maximumLeafSize === 8 || maximumLeafSize === 16
    || maximumLeafSize === 32,
  `arm "${source}" must use leaf size 2, 4, 8, 16, or 32`);
  const interfaceBandCells = bandText === undefined ? 3 : Number(bandText);
  assert.ok(Number.isInteger(interfaceBandCells)
    && interfaceBandCells >= 0 && interfaceBandCells <= 32,
  `arm "${source}" must use an interface band from 0 through 32`);
  return {
    label: source,
    gated: mode === "on",
    adaptivity: mode === "zero" ? 0 : 1,
    maximumLeafSize,
    interfaceBandCells,
  };
};

const rounded = (value: number, digits = 3) => Number(value.toFixed(digits));
const fieldAdvance = (field: Float32Array, threshold = 0.01) => {
  let maximumX = -1, maximumZ = -1, leadingMass = 0;
  for (let z = 0; z < 16; z += 1) for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      const value = field[x + 16 * (y + 16 * z)]!;
      if (value >= threshold) { maximumX = Math.max(maximumX, x); maximumZ = Math.max(maximumZ, z); }
      if (x >= 10) leadingMass += value;
    }
  }
  return { maximumX, maximumZ, leadingMass: rounded(leadingMass, 6) };
};
const layerAdvance = (field: Float32Array, y: number, threshold = 0.01) => {
  let maximumX = -1, maximumZ = -1, leadingMass = 0;
  for (let z = 0; z < 16; z += 1) for (let x = 0; x < 16; x += 1) {
    const value = field[x + 16 * (y + 16 * z)]!;
    if (value >= threshold) {
      maximumX = Math.max(maximumX, x);
      maximumZ = Math.max(maximumZ, z);
    }
    if (x >= 10) leadingMass += value;
  }
  return { maximumX, maximumZ, leadingMass: rounded(leadingMass, 6) };
};
const tracePhaseTotals = (trace: {
  readonly phases: readonly { readonly id: string; readonly duration_ms: number }[];
} | undefined): Readonly<Record<string, number>> | null => {
  if (!trace) return null;
  const totals: Record<string, number> = {};
  for (const phase of trace.phases) {
    totals[phase.id] = (totals[phase.id] ?? 0) + phase.duration_ms;
  }
  return Object.fromEntries(Object.entries(totals)
    .map(([id, duration_ms]) => [id, rounded(duration_ms)]));
};
const traceRequested = process.env.FLUID_MINI_DAM_TRACE === "1";
const auditEveryStep = process.env.FLUID_MINI_DAM_AUDIT_EVERY_STEP === "1";
const surfaceTrackingFactor = Number(process.env.FLUID_MINI_DAM_FINE_FACTOR ?? 4);
assert.ok(surfaceTrackingFactor === 1 || surfaceTrackingFactor === 4
  || surfaceTrackingFactor === 8,
"FLUID_MINI_DAM_FINE_FACTOR must be 1, 4, or 8");
const steps = Number(process.env.FLUID_MINI_DAM_STEPS ?? 62);
assert.ok(Number.isSafeInteger(steps) && steps >= 0);
const arms = (process.env.FLUID_MINI_DAM_ARMS ?? "off:2,on:2,off:4,on:4")
  .split(",")
  .filter(Boolean)
  .map(parseArm);
assert.ok(arms.length > 0);
assert.equal(new Set(arms.map((arm) => arm.label)).size, arms.length,
  "FLUID_MINI_DAM_ARMS must contain unique configurations; this Dawn build cannot safely rebuild an identical specialization in one process");

await acquireWebGPUExclusiveLock(
  "dawn-benchmark",
  "tools/benchmark-mini-dam-fluid-gate.ts",
);
try {
  usePerformanceInstrumentationStore.getState().setEnabled(traceRequested);
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
    ...(process.env.FLUID_WEBGPU_DAWN_FEATURES
      ? [`enable-dawn-features=${process.env.FLUID_WEBGPU_DAWN_FEATURES}`] : []),
  ]);
  Object.defineProperty(globalThis, "navigator", {
    configurable: true,
    value: { gpu },
  });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  assert.ok(adapter.features.has("subgroups"),
    "mini-dam octree benchmark requires subgroups");
  const requestedFeatures: GPUFeatureName[] = ["subgroups"];
  if (adapter.features.has("timestamp-query")) requestedFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: requestedFeatures,
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const results: Result[] = [];
  for (const arm of arms) {
    const scene = createMinimalPowerDamBreakScene();
    const values = {
      ...losassoMethod.presetFor("balanced"),
      maximumLeafSize: String(arm.maximumLeafSize),
      interfaceRefinementBandCells: arm.interfaceBandCells,
      globalFineLevelSetFactor: String(surfaceTrackingFactor),
      octreeAdaptivity: arm.adaptivity,
      fluidGatedBoundaryRefinement: arm.gated,
      secondaryParticles: "off",
    };
    const constructionStarted = performance.now();
    const solver = await losassoMethod.createSolverAsync!(
      device,
      scene,
      "balanced",
      values,
      undefined,
      () => {},
    ) as GPUSolverInstance;
    await device.queue.onSubmittedWorkDone();
    const construction_ms = performance.now() - constructionStarted;
    assert.deepEqual(
      [solver.info.nx, solver.info.ny, solver.info.nz],
      [16, 16, 16],
    );
    const projection = (solver as unknown as {
      octreeProjection?: {
        readSolveDiagnostics(): Promise<void>;
        readTopologyLeafCensus(): Promise<TopologyCensus>;
      };
    }).octreeProjection;
    assert.ok(projection, "octree projection was not exposed");
    const initialTopology = await projection.readTopologyLeafCensus();
    const pressureProfile: Array<Readonly<Record<string, unknown>>> = [];

    const simulationStarted = performance.now();
    for (let step = 1; step <= steps; step += 1) {
      const requestedTime_s = step * scene.numerics.fixedDt_s!;
      while (!solver.advanceTo(requestedTime_s, [])) {
        await new Promise((resolve) => setImmediate(resolve));
      }
      if (auditEveryStep) {
        const sample = await solver.readStats();
        pressureProfile.push({
          step,
          snapshotStep: sample.structuredSnapshotStep ?? 0,
          snapshotIterations: sample.structuredSnapshotExecutedSolveIterations ?? 0,
          snapshotConverged: sample.structuredSnapshotSolveConverged ?? false,
          authorityLagSteps: sample.structuredAuthorityLagSteps ?? 0,
          predictionFailures: sample.stepPredictionFailures ?? [],
          capacityOverflow: sample.pressureCapacityOverflow ?? false,
        });
      }
    }
    await device.queue.onSubmittedWorkDone();
    const simulationWall_ms = performance.now() - simulationStarted;

    await projection.readSolveDiagnostics();
    const topology = await projection.readTopologyLeafCensus();
    const fieldReadback = await readCubicVolumeField(device, solver, true);
    const info = await solver.readStats();
    results.push({
      arm,
      construction_ms,
      simulationWall_ms,
      wallPerStep_ms: steps > 0 ? simulationWall_ms / steps : 0,
      initialTopology,
      topology,
      field: fieldReadback.field,
      fieldSummary: fieldReadback.summary,
      pressureRows: info.pressureRequiredRows ?? 0,
      pressureIterations: info.quadtreePressureIterationsUsed ?? 0,
      representedVolumeCellSum: info.representedVolumeCellSum ?? 0,
      traceTotal_ms: info.physicsTrace?.total_ms ?? null,
      tracePhases_ms: tracePhaseTotals(info.physicsTrace),
      pressureProfile: auditEveryStep ? pressureProfile : null,
    });
    solver.destroy();
  }

  const report = results.map((result) => {
    const zeroRefinement = results.find((candidate) =>
      candidate.arm.adaptivity === 0
      && candidate.arm.maximumLeafSize === result.arm.maximumLeafSize
      && candidate.arm.interfaceBandCells === result.arm.interfaceBandCells);
    const ungatedAdaptive = results.find((candidate) =>
      candidate.arm.adaptivity === 1
      && !candidate.arm.gated
      && candidate.arm.maximumLeafSize === result.arm.maximumLeafSize
      && candidate.arm.interfaceBandCells === result.arm.interfaceBandCells);
    const reference = zeroRefinement ?? ungatedAdaptive;
    const fieldDifference = reference
      ? compareScalarFields(
        result.field,
        reference.field,
        16,
        16,
        16,
      )
      : null;
    const gateDifference = result.arm.gated && ungatedAdaptive
      ? compareScalarFields(result.field, ungatedAdaptive.field, 16, 16, 16)
      : null;
    return {
      arm: result.arm.label,
      gated: result.arm.gated,
      adaptivity: result.arm.adaptivity,
      maximumLeafSize: result.arm.maximumLeafSize,
      interfaceBandCells: result.arm.interfaceBandCells,
      construction_ms: rounded(result.construction_ms),
      simulationWall_ms: rounded(result.simulationWall_ms),
      wallPerStep_ms: rounded(result.wallPerStep_ms),
      traceTotal_ms: result.traceTotal_ms === null
        ? null : rounded(result.traceTotal_ms),
      tracePhases_ms: result.tracePhases_ms,
      pressureProfile: result.pressureProfile,
      initialTopologyLeaves: result.initialTopology.topologyLeaves,
      initialLeafCountsBySize: result.initialTopology.leafCountsBySize,
      initialCoarseLeavesByOriginY: result.initialTopology.coarseLeafCountsByOriginY,
      topologyLeaves: result.topology.topologyLeaves,
      leafCountsBySize: result.topology.leafCountsBySize,
      coarseLeavesByOriginY: result.topology.coarseLeafCountsByOriginY,
      pressureRows: result.pressureRows,
      pressureIterations: result.pressureIterations,
      representedVolumeCellSum: rounded(result.representedVolumeCellSum, 6),
      fieldCellSum: rounded(result.fieldSummary.cellSum, 6),
      fieldAdvance: fieldAdvance(result.field),
      bottomAdvance: layerAdvance(result.field, 0),
      layerOneAdvance: layerAdvance(result.field, 1),
      fieldDifference,
      gateDifference,
      speedupVsControl: reference
        ? rounded(reference.simulationWall_ms / result.simulationWall_ms, 4)
        : null,
    };
  });
  console.table(report);
  console.log(JSON.stringify({
    phase: "mini-dam-fluid-gate-benchmark",
    steps,
    surfaceTrackingFactor,
    results: report,
    validationErrors,
  }));
  if (steps === 1) {
    for (const result of results.filter(({ arm }) =>
      arm.gated && arm.maximumLeafSize === 2 && arm.interfaceBandCells === 3)) {
      assert.deepEqual(result.topology.leafCountsBySize, {
        1: 2_560,
        2: 192,
      }, "the first recurring mini-dam topology must preserve its dry size-two region");
      assert.equal(result.topology.topologyLeaves, 2_752,
        "the first recurring publication must not regress to 4,096 unit leaves");
      assert.ok(result.pressureIterations < 10,
        "the corrected adaptive topology must converge inside the encoded pressure tail");
    }
    if (surfaceTrackingFactor === 4) {
      const zeroRefinement = results.find(({ arm }) => arm.adaptivity === 0);
      const gatedAdaptive = results.find(({ arm }) => arm.adaptivity === 1 && arm.gated
        && arm.maximumLeafSize === 32 && arm.interfaceBandCells === 3);
      if (zeroRefinement && gatedAdaptive) {
        const zeroAdvance = fieldAdvance(zeroRefinement.field);
        const adaptiveAdvance = fieldAdvance(gatedAdaptive.field);
        assert.deepEqual(
          [adaptiveAdvance.maximumX, adaptiveAdvance.maximumZ],
          [zeroAdvance.maximumX, zeroAdvance.maximumZ],
          "the first adaptive factor-4 step must preserve the zero-refinement front extent",
        );
        assert.ok(adaptiveAdvance.maximumX >= 10 && adaptiveAdvance.maximumZ >= 10,
          "the paper-resolution mini16 surface must resolve the front beyond its authored face immediately");
        const difference = compareScalarFields(
          gatedAdaptive.field, zeroRefinement.field, 16, 16, 16);
        assert.ok(difference.meanAbsoluteError < 0.001,
          `first-step adaptive factor-4 MAE ${difference.meanAbsoluteError} is not close to zero refinement`);
        assert.ok(Math.abs(adaptiveAdvance.leadingMass - zeroAdvance.leadingMass)
          <= zeroAdvance.leadingMass * 0.02,
        "first-step adaptive factor-4 leading mass must stay within 2% of zero refinement");
        assert.ok(gatedAdaptive.topology.topologyLeaves
          < zeroRefinement.topology.topologyLeaves,
        "first-step factor-4 adaptivity must retain fewer leaves than zero refinement");
      }
    }
  }
  if (steps === 44 && surfaceTrackingFactor === 1) {
    const zeroRefinement = results.find(({ arm }) => arm.adaptivity === 0);
    const ungatedAdaptive = results.find(({ arm }) => arm.adaptivity === 1 && !arm.gated);
    const gatedAdaptive = results.find(({ arm }) => arm.adaptivity === 1 && arm.gated);
    if (zeroRefinement && ungatedAdaptive && gatedAdaptive) {
      const gateDifference = compareScalarFields(
        gatedAdaptive.field, ungatedAdaptive.field, 16, 16, 16);
      assert.equal(gateDifference.meanAbsoluteError, 0,
        "the boundary gate must not alter the factor-1 adaptive mini16 field");
      const zeroDifference = compareScalarFields(
        gatedAdaptive.field, zeroRefinement.field, 16, 16, 16);
      assert.ok(zeroDifference.meanAbsoluteError < 0.001,
        `factor-1 adaptive mini16 MAE ${zeroDifference.meanAbsoluteError} is not close to zero refinement`);
      assert.equal(zeroDifference.wetIntersectionOverUnion, 1,
        "factor-1 adaptive mini16 must preserve the zero-refinement wet support");
      assert.ok(zeroDifference.centroidDistanceCells !== null
        && zeroDifference.centroidDistanceCells < 0.01,
        `factor-1 adaptive mini16 centroid drifted ${zeroDifference.centroidDistanceCells} cells`);
      const adaptiveAdvance = fieldAdvance(gatedAdaptive.field);
      const zeroAdvance = fieldAdvance(zeroRefinement.field);
      assert.deepEqual(
        [adaptiveAdvance.maximumX, adaptiveAdvance.maximumZ],
        [zeroAdvance.maximumX, zeroAdvance.maximumZ],
        "factor-1 adaptive mini16 must advance as far as zero refinement",
      );
      assert.ok(Math.abs(adaptiveAdvance.leadingMass - zeroAdvance.leadingMass)
        <= Math.max(0.01, zeroAdvance.leadingMass * 0.01),
      "factor-1 adaptive mini16 leading mass must stay within 1% of zero refinement");
      assert.ok(gatedAdaptive.topology.topologyLeaves < zeroRefinement.topology.topologyLeaves,
        "factor-1 adaptive mini16 must retain fewer leaves than zero refinement");
      if (auditEveryStep) {
        assert.deepEqual(gatedAdaptive.pressureProfile, zeroRefinement.pressureProfile,
          "factor-1 adaptive mini16 pressure receipt must match zero refinement");
      }
      const maximumSlowdown = Number(process.env.FLUID_MINI_DAM_MAX_SLOWDOWN ?? 1.1);
      assert.ok(gatedAdaptive.simulationWall_ms
        <= zeroRefinement.simulationWall_ms * maximumSlowdown,
      `factor-1 adaptive mini16 wall time regressed by ${rounded(100 * (gatedAdaptive.simulationWall_ms / zeroRefinement.simulationWall_ms - 1), 2)}%`);
    }
  }
  if (steps === 8 && surfaceTrackingFactor === 4) {
    for (const adaptive of results.filter(({ arm }) => arm.adaptivity === 1
      && arm.gated && arm.maximumLeafSize === 32 && arm.interfaceBandCells === 4)) {
      const bottom = layerAdvance(adaptive.field, 0);
      const layerOne = layerAdvance(adaptive.field, 1);
      assert.ok(bottom.leadingMass >= 1.85,
        `factor-4 mini16 bottom front stalled at ${bottom.leadingMass} leading cells`);
      assert.ok(layerOne.leadingMass >= 1.65,
        `factor-4 mini16 first layer stalled at ${layerOne.leadingMass} leading cells`);
      assert.ok(bottom.leadingMass - layerOne.leadingMass >= 0.15,
        "factor-4 mini16 must develop its early floor-running front");
      assert.ok(adaptive.pressureIterations < 10,
        "factor-4 mini16 must not exhaust its pressure tail after the topology update");
    }
  }
  if ((steps === 62 || steps === 400) && surfaceTrackingFactor === 4) {
    for (const gated of results.filter(({ arm }) =>
      arm.gated && arm.maximumLeafSize === 32 && arm.interfaceBandCells === 3)) {
      const control = results.find(({ arm }) => !arm.gated
        && arm.maximumLeafSize === gated.arm.maximumLeafSize
        && arm.interfaceBandCells === gated.arm.interfaceBandCells);
      if (!control) continue;
      const difference = compareScalarFields(gated.field, control.field, 16, 16, 16);
      assert.ok(difference.meanAbsoluteError < 1e-6,
        `gated dry-boundary coarsening changed the ${steps}-step mini16 field by MAE ${difference.meanAbsoluteError}`);
      assert.equal(difference.wetIntersectionOverUnion, 1,
        `gated dry-boundary coarsening changed the ${steps}-step mini16 wet support`);
      assert.ok(difference.centroidDistanceCells !== null
        && difference.centroidDistanceCells < 1e-5,
      `gated dry-boundary coarsening moved the ${steps}-step mini16 centroid by ${difference.centroidDistanceCells} cells`);
      assert.equal(gated.pressureRows, control.pressureRows,
        `boundary coarsening must not change the ${steps}-step mini16 pressure discretization`);
      assert.equal(gated.pressureIterations, control.pressureIterations,
        `boundary coarsening must not change the ${steps}-step mini16 pressure convergence profile`);
      if (auditEveryStep) {
        const mismatch = gated.pressureProfile?.findIndex((sample, index) =>
          JSON.stringify(sample) !== JSON.stringify(control.pressureProfile?.[index])) ?? -1;
        assert.equal(mismatch, -1,
          `boundary coarsening changed the mini16 pressure receipt at step ${mismatch + 1}: control=${JSON.stringify(control.pressureProfile?.[mismatch])} gated=${JSON.stringify(gated.pressureProfile?.[mismatch])}`);
      }
      assert.equal(gated.representedVolumeCellSum, control.representedVolumeCellSum,
        "boundary coarsening must not change represented mini16 volume");
      assert.ok(gated.topology.topologyLeaves < control.topology.topologyLeaves,
        "the gated arm must actually remove dry mini16 boundary leaves");
      const maximumSlowdown = Number(process.env.FLUID_MINI_DAM_MAX_SLOWDOWN ?? 1.1);
      assert.ok(gated.simulationWall_ms <= control.simulationWall_ms * maximumSlowdown,
        `default mini16 wall time regressed by ${rounded(100 * (gated.simulationWall_ms / control.simulationWall_ms - 1), 2)}%`);
    }
  }
  assert.deepEqual(validationErrors, [],
    `WebGPU validation errors: ${validationErrors.join("; ")}`);
  device.destroy();
} finally {
  usePerformanceInstrumentationStore.getState().setEnabled(false);
  await releaseWebGPUExclusiveLock();
}
