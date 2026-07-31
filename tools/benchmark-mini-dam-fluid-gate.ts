import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import { createMinimalPowerDamBreakScene } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { compareScalarFields } from "./webgpu-smoke-scenarios";
import { readCubicVolumeField } from "./webgpu-smoke-readbacks";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

interface Arm {
  readonly label: string;
  readonly gated: boolean;
  readonly maximumLeafSize: 2 | 4 | 8 | 16;
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
}

const parseArm = (source: string): Arm => {
  const [mode, leafText, bandText] = source.split(":");
  assert.ok(mode === "off" || mode === "on",
    `arm "${source}" must start with off or on`);
  const maximumLeafSize = Number(leafText);
  assert.ok(maximumLeafSize === 2 || maximumLeafSize === 4
    || maximumLeafSize === 8 || maximumLeafSize === 16,
  `arm "${source}" must use leaf size 2, 4, 8, or 16`);
  const interfaceBandCells = bandText === undefined ? 3 : Number(bandText);
  assert.ok(Number.isInteger(interfaceBandCells)
    && interfaceBandCells >= 0 && interfaceBandCells <= 32,
  `arm "${source}" must use an interface band from 0 through 32`);
  return {
    label: source,
    gated: mode === "on",
    maximumLeafSize,
    interfaceBandCells,
  };
};

const rounded = (value: number, digits = 3) => Number(value.toFixed(digits));
const steps = Number(process.env.FLUID_MINI_DAM_STEPS ?? 62);
assert.ok(Number.isSafeInteger(steps) && steps > 0);
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
    if (arm.gated) {
      process.env.FLUID_OCTREE_FLUID_GATED_BOUNDARIES = "1";
    } else {
      process.env.FLUID_OCTREE_FLUID_GATED_BOUNDARIES = "0";
    }
    const scene = createMinimalPowerDamBreakScene();
    const values = {
      ...octreeMethod.presetFor("balanced"),
      maximumLeafSize: String(arm.maximumLeafSize),
      interfaceRefinementBandCells: arm.interfaceBandCells,
      globalFineLevelSetFactor: "4",
      secondaryParticles: "off",
    };
    const constructionStarted = performance.now();
    const solver = await octreeMethod.createSolverAsync!(
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

    const simulationStarted = performance.now();
    for (let step = 1; step <= steps; step += 1) {
      const requestedTime_s = step * scene.numerics.fixedDt_s!;
      while (!solver.advanceTo(requestedTime_s, [])) {
        await new Promise((resolve) => setImmediate(resolve));
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
      wallPerStep_ms: simulationWall_ms / steps,
      initialTopology,
      topology,
      field: fieldReadback.field,
      fieldSummary: fieldReadback.summary,
      pressureRows: info.pressureRequiredRows ?? 0,
      pressureIterations: info.quadtreePressureIterationsUsed ?? 0,
      representedVolumeCellSum: info.representedVolumeCellSum ?? 0,
      traceTotal_ms: info.physicsTrace?.total_ms ?? null,
    });
    solver.destroy();
  }

  const report = results.map((result) => {
    const control = results.find((candidate) =>
      !candidate.arm.gated
      && candidate.arm.maximumLeafSize === result.arm.maximumLeafSize
      && candidate.arm.interfaceBandCells === result.arm.interfaceBandCells);
    const fieldDifference = control
      ? compareScalarFields(
        result.field,
        control.field,
        16,
        16,
        16,
      )
      : null;
    return {
      arm: result.arm.label,
      gated: result.arm.gated,
      maximumLeafSize: result.arm.maximumLeafSize,
      interfaceBandCells: result.arm.interfaceBandCells,
      construction_ms: rounded(result.construction_ms),
      simulationWall_ms: rounded(result.simulationWall_ms),
      wallPerStep_ms: rounded(result.wallPerStep_ms),
      traceTotal_ms: result.traceTotal_ms === null
        ? null : rounded(result.traceTotal_ms),
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
      fieldDifference,
      speedupVsControl: control
        ? rounded(control.simulationWall_ms / result.simulationWall_ms, 4)
        : null,
    };
  });
  console.table(report);
  console.log(JSON.stringify({
    phase: "mini-dam-fluid-gate-benchmark",
    steps,
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
  }
  assert.deepEqual(validationErrors, [],
    `WebGPU validation errors: ${validationErrors.join("; ")}`);
  device.destroy();
} finally {
  delete process.env.FLUID_OCTREE_FLUID_GATED_BOUNDARIES;
  await releaseWebGPUExclusiveLock();
}
