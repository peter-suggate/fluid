import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";

const POWER_ASSEMBLED = 0x8000_0000;
const POWER_PROJECTED = 0x4000_0000;

// Aanjaneya et al. (2017), Section 4.3 keeps the second-order power operator
// as the pressure system while optimizing its parallel relaxation and
// preconditioning. This gate protects that ownership: Chebyshev is an
// implementation accelerator, never permission to solve the legacy axis rows.

interface PowerStageRecord {
  phase: "power-stage-audit";
  step: number;
  generation: number;
  seedFlags: number;
  seedValid: boolean;
  reverseFlags: number;
  reverseValid: boolean;
  operatorFlags: number;
  operatorFirstError: number;
  operatorProjectedCount: number;
  operatorAssemblyFlags: number;
  operatorAssemblyFirstError: number;
  operatorEntryCount: number;
  operatorRowCount: number;
  operatorFaceCount: number;
  operatorIncidenceCount: number;
}

interface RunningRecord {
  phase: "running";
  steps: number;
  pressureRelativeResidual: number;
}

interface ResultRecord {
  phase: "result";
  steps: number;
  pressureRequiredRows: number;
  pressureRequiredEntries: number;
  pressureCapacityOverflow: boolean;
  powerDiagramProjection: string;
  powerDiagramReady: boolean;
  powerDiagramAuthoritative: boolean;
  powerDiagramFallbackReason?: string;
  quadtreePressureIterationsUsed: number;
  quadtreePressureIterationBudget: number;
  quadtreePressureIterationHardBudget: number;
  pressureResidual: number;
  pressureRelativeResidual: number;
  nonFiniteCount: number;
  validationErrors: unknown[];
}

function records(stdout: string): Array<Record<string, unknown>> {
  return stdout.split("\n").flatMap(line => {
    try { return [JSON.parse(line) as Record<string, unknown>]; }
    catch { return []; }
  });
}

test("authoritative power CSR publication remains upstream of parallel Chebyshev and power projection", () => {
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  const powerRows = encode.indexOf("this.encodePowerAssemblyMirror(encoder)");
  const chebyshev = encode.indexOf("this.iterateChebyshevPipeline");
  const powerProjection = encode.indexOf("this.encodePowerProjectionMirror(encoder");
  assert.ok(powerRows >= 0 && chebyshev > powerRows && powerProjection > chebyshev,
    "power rows must publish before Chebyshev and the converged pressure must project power faces afterward");
  assert.match(encode, /const useChebyshev=this\.leafSolver===["']chebyshev["']/,
    "power authority must not select a serial replacement for Chebyshev");
  assert.match(encode, /pressure\.dispatchWorkgroupsIndirect\(this\.solveDispatch,0\)/,
    "Chebyshev must remain row-parallel over the compact publication dispatch");
});

test("three production dam-break generations keep power CSR authoritative through Chebyshev", {
  skip: !process.env.WEBGPU_NODE_MODULE
    ? "set WEBGPU_NODE_MODULE for the production Chebyshev/power gate"
    : process.env.FLUID_POWER_CHEBYSHEV_ACCEPTANCE !== "1"
      && "set FLUID_POWER_CHEBYSHEV_ACCEPTANCE=1 after the coarse-sign publication gate passes",
  timeout: 90_000,
}, () => {
  const child = spawnSync(process.execPath, ["--import", "tsx", "tools/run-webgpu-smoke.ts"], {
    cwd: process.cwd(), encoding: "utf8", timeout: 75_000, killSignal: "SIGKILL",
    maxBuffer: 32 * 1024 * 1024,
    env: { ...process.env, FLUID_SCENE: "dam-break-ui", FLUID_METHOD: "octree",
      FLUID_TARGET_S: "0.012", FLUID_ORACLE_STEPS: "3", FLUID_VOXEL_CELL_SIZE: "0.02",
      FLUID_PRESSURE_CYCLES: "128", FLUID_OCTREE_LEAF_SOLVER: "chebyshev",
      FLUID_CPU_ORACLE: "0", FLUID_FIELD_STATS: "0", FLUID_DISABLE_TIMESTAMPS: "1",
      FLUID_OCTREE_FACE_TRANSPORT: "1", FLUID_OCTREE_POWER_PROJECTION: "authoritative",
      FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4", FLUID_POWER_GENERATION_AUDIT: "1",
      FLUID_POWER_GENERATION_AUDIT_LOG: "0", FLUID_POWER_STAGE_AUDIT: "1", FLUID_REPORT_EVERY: "1" },
  });
  assert.equal(child.error, undefined, `Chebyshev/power process failed: ${child.error?.message ?? "unknown"}`);
  assert.equal(child.status, 0, `Chebyshev/power smoke failed:\n${child.stderr}\n${child.stdout.slice(-12_000)}`);
  const output = records(child.stdout);
  const stages = output.filter(record => record.phase === "power-stage-audit") as unknown as PowerStageRecord[];
  const running = output.filter(record => record.phase === "running") as unknown as RunningRecord[];
  const result = output.findLast(record => record.phase === "result") as unknown as ResultRecord | undefined;
  assert.equal(stages.length, 3, "three-step gate did not audit every power generation");
  assert.equal(running.length, 3, "three-step gate did not publish every Chebyshev residual");
  assert.ok(result, "three-step gate emitted no result record");

  let previousGeneration = 0;
  for (const stage of stages) {
    assert.ok(stage.generation > previousGeneration, `step ${stage.step} did not advance power generation`);
    previousGeneration = stage.generation;
    assert.equal(stage.seedFlags, 0, `step ${stage.step} seed flags are nonzero`);
    assert.equal(stage.seedValid, true, `step ${stage.step} power seed is invalid`);
    assert.equal(stage.reverseFlags, 0, `step ${stage.step} reverse-publication flags are nonzero`);
    assert.equal(stage.reverseValid, true, `step ${stage.step} reverse power-to-axis publication is invalid`);
    assert.equal(stage.operatorAssemblyFlags, POWER_ASSEMBLED,
      `step ${stage.step} did not assemble authoritative power CSR rows`);
    assert.equal(stage.operatorAssemblyFirstError, 0xffff_ffff,
      `step ${stage.step} power assembly reported an error`);
    assert.equal(stage.operatorFlags, POWER_ASSEMBLED | POWER_PROJECTED,
      `step ${stage.step} did not project from the assembled power operator`);
    assert.equal(stage.operatorFirstError, 0xffff_ffff,
      `step ${stage.step} power projection reported an error`);
    assert.ok(stage.operatorRowCount > 0 && stage.operatorEntryCount > 0,
      `step ${stage.step} collapsed power rows or entries`);
    assert.ok(stage.operatorFaceCount > 0 && stage.operatorIncidenceCount > 0,
      `step ${stage.step} collapsed power faces or incidence`);
    assert.equal(stage.operatorProjectedCount, stage.operatorFaceCount,
      `step ${stage.step} projected only part of the authoritative face set`);
  }
  for (const sample of running) {
    assert.ok(Number.isFinite(sample.pressureRelativeResidual)
      && sample.pressureRelativeResidual >= 0 && sample.pressureRelativeResidual < 1,
    `step ${sample.steps} Chebyshev solve did not reduce residual below the unsolved RHS norm`);
  }
  assert.equal(result.steps, 3);
  assert.equal(result.powerDiagramProjection, "authoritative");
  assert.equal(result.powerDiagramReady, true);
  assert.equal(result.powerDiagramAuthoritative, true);
  assert.equal(result.powerDiagramFallbackReason, undefined,
    "production pressure fell back from power CSR to axis-row authority");
  assert.ok(result.pressureRequiredRows > 0 && result.pressureRequiredEntries > 0,
    "Chebyshev compact row publication collapsed");
  assert.equal(result.pressureCapacityOverflow, false);
  assert.equal(result.quadtreePressureIterationsUsed, 32,
    "128 equivalent sweeps must remain 32 parallel Chebyshev polynomial passes");
  assert.equal(result.quadtreePressureIterationBudget, 32);
  assert.equal(result.quadtreePressureIterationHardBudget, 32);
  assert.ok(Number.isFinite(result.pressureResidual) && Number.isFinite(result.pressureRelativeResidual));
  assert.equal(result.nonFiniteCount, 0);
  assert.deepEqual(result.validationErrors, []);
});
