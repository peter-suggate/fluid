import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import type { PerformanceTrace } from "../lib/performance-trace";
import {
  powerDamComputePassStage,
  powerDamPerformanceFailures,
  powerDamResultFromLine,
  summarizePowerDamPerformance,
} from "../tools/power-dam-performance-report";

const physicsTrace = (phases: PerformanceTrace["phases"]): PerformanceTrace => ({
  sampleId: 1,
  domain: "gpu",
  lane: "physics",
  context: "octree:test",
  capturedAt_ms: 1,
  total_ms: phases.reduce((sum, phase) => sum + phase.duration_ms, 0),
  phases,
});

test("UI throughput command enforces the final Power Liquids authority gates", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["benchmark:power-dam-ui"];
  assert.match(command, /FLUID_MAX_ADVANCE_MS=\$\{FLUID_MAX_ADVANCE_MS:-15\}/);
  assert.match(command, /FLUID_MAX_DISPATCHES_PER_ADVANCE=\$\{FLUID_MAX_DISPATCHES_PER_ADVANCE:-300\}/);
  assert.match(command, /FLUID_MAX_PASSES_PER_ADVANCE=\$\{FLUID_MAX_PASSES_PER_ADVANCE:-60\}/);
  assert.doesNotMatch(command, /FLUID_MAX_PRESSURE_NON_SOLVE_MS/,
    "the throughput command has no sampled physics trace; profile acceptance owns the pressure-system gate");
  assert.match(packageJson.scripts["profile:power-dam-ui"],
    /FLUID_MAX_PRESSURE_NON_SOLVE_MS=\$\{FLUID_MAX_PRESSURE_NON_SOLVE_MS:-4\}/);
  assert.match(packageJson.scripts["profile:power-dam-ui"], /FLUID_ENGINE_SPLIT=fine/,
    "the sampled pressure-system gate requires fine checkpoint attribution");
  assert.match(packageJson.scripts["test:power-liquids-structure"],
    /^npx tsc --noEmit && npm run test:water-shaders && node --import tsx --test .*power-liquids-clean-cut\.test\.ts.*webgpu-pass-broker\.test\.ts.*power-dam-performance-report\.test\.ts$/,
    "phase acceptance must fail before Dawn on type, WGSL, clean-cut, pass-broker, or budget-contract regressions");
  assert.match(packageJson.scripts["acceptance:power-liquids-phase"],
    /^npm run test:power-liquids-structure && npm run test:webgpu:dam-ui-construction && npm run test:webgpu:dam-ui-two-step/,
    "clean-cut source, type, and shader gates must pass before the exact UI graph and numerical acceptance");
  assert.match(packageJson.scripts["acceptance:power-liquids-phase"],
    /npm run benchmark:power-dam-ui && npm run profile:power-dam-ui/,
    "throughput and sampled pressure-system ratchets must both execute in phase acceptance");
  assert.doesNotMatch(packageJson.scripts["acceptance:power-liquids-phase"],
    /test:webgpu:dam-ui-performance/,
    "the gated profile wrapper replaces the duplicate ungated profiler invocation");
});

test("power dam throughput summary normalizes command costs per advance", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "minimal-power-dam-break", method: "octree", phase: "result",
    steps: 4, simulationWall_ms: 240, validationErrors: [],
    gpuCommandAudit: {
      dispatches: 400, indirectDispatches: 240, computePasses: 80,
      clearBuffer: { calls: 8, bytes: 4_000_000 },
      copyBufferToBuffer: { calls: 12, bytes: 8_000_000 },
      computePassesByLabel: {
        "Octree owner pages": { calls: 12, bytes: 0 },
        "Octree MGPCG solve": { calls: 8, bytes: 0 },
        "Fine redistance": { calls: 60, bytes: 0 },
      },
      dispatchesByPassLabel: {
        "Octree MGPCG solve": { calls: 200, bytes: 0 },
        "SPGrid persistent small-domain MGPCG": { calls: 100, bytes: 0 },
      },
    },
  });
  assert.equal(summary.advanceWall_ms, 60);
  assert.deepEqual(summary.commands, {
    dispatchesPerAdvance: 100, indirectDispatchesPerAdvance: 60,
    computePassesPerAdvance: 20,
    computePassesByStage: {
      "Octree owner pages": 3, "MGPCG solve": 2, "Fine redistance / volume": 15,
    },
    computePassesByLabel: {
      "Octree owner pages": 3, "Octree MGPCG solve": 2, "Fine redistance": 15,
    },
    computePassAttributionComplete: true,
    unattributedComputePassesPerAdvance: 0,
    unownedComputePassLabels: [],
    mgpcgDispatchesPerAdvance: 75,
    mgpcgDispatchFraction: 0.75,
    clearBytesPerAdvance: 1_000_000,
    copyBytesPerAdvance: 2_000_000,
  });
});

test("compute-pass attribution aggregates indexed native labels into stable owning stages", () => {
  const computePassesByLabel: Record<string, { calls: number; bytes: number }> = {
    "Resolve Section 5 catalog adjacency": { calls: 2, bytes: 0 },
    "Validate and publish Section 5 catalog adjacency": { calls: 2, bytes: 0 },
  };
  computePassesByLabel["Prepare global fine trajectory chunk 1/4"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Prepare global fine trajectory chunk 2/4"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Rank fixed octree fine-seed candidate records"] = { calls: 2, bytes: 0 };
  computePassesByLabel["Finalize octree fine-seed candidate publication"] = { calls: 2, bytes: 0 };
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 100, gpuCommandAudit: { computePasses: 60, computePassesByLabel },
  });
  assert.deepEqual(summary.commands?.computePassesByStage, {
    "Section 5 face band · catalog adjacency": 2,
    "Fine transport · prepare trajectory chunks": 2,
    "Fine seed adapter": 2,
  });
  assert.equal(summary.commands?.computePassesByLabel["Resolve Section 5 catalog adjacency"], 1);
});

test("compute-pass attribution is a closed ownership table", () => {
  assert.equal(powerDamComputePassStage("Prepare exact owner-page delta dispatch"), "Octree owner pages");
  assert.equal(powerDamComputePassStage("A newly introduced mystery pass"), undefined);

  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 1, gpuCommandAudit: {
      computePasses: 4,
      computePassesByLabel: { "A newly introduced mystery pass": { calls: 4, bytes: 0 } },
    },
  });
  assert.equal(summary.commands?.computePassAttributionComplete, false);
  assert.deepEqual(summary.commands?.unownedComputePassLabels, ["A newly introduced mystery pass"]);
  assert.deepEqual(summary.commands?.computePassesByStage, {});
  assert.deepEqual(powerDamPerformanceFailures(summary, {
    maximumComputePassesPerAdvance: 60,
  }), ["compute passes/stage attribution incomplete (2.0 unattributed/advance)"]);
});

test("generic physics attribution is exact and aggregates repeated semantic phases", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 62,
    simulationWall_ms: 4_200,
    physicsTrace: physicsTrace([
      { id: "coarse-grid", label: "Adaptive coarse-grid topology", duration_ms: 9 },
      { id: "pressure-system", label: "Power operator", duration_ms: 20 },
      { id: "pressure-system", label: "Pressure rows", duration_ms: 12 },
      { id: "pressure-solve", label: "MGPCG", duration_ms: 7.36 },
      { id: "velocity-extrapolation", label: "Closest point", duration_ms: 23 },
      { id: "adaptive-publication", label: "Publication", duration_ms: 13.64 },
    ]),
  });
  assert.equal(summary.physicsTrace?.total_ms, 85);
  assert.equal(summary.physicsTrace?.accounted_ms, 85);
  assert.equal(summary.physicsTrace?.exact, true);
  assert.equal(summary.physicsTrace?.phaseTotals_ms["pressure-system"], 32);
  assert.equal(summary.physicsTrace?.phaseTotals_ms["velocity-extrapolation"], 23);
  assert.equal(summary.advanceWall_ms, 4_200 / 62);
});

test("NDJSON parsing ignores logs and selects octree result records", () => {
  assert.equal(powerDamResultFromLine("SAFETY: close browser tabs"), undefined);
  assert.equal(powerDamResultFromLine(JSON.stringify({ phase: "result", method: "uniform" })), undefined);
  assert.equal(powerDamResultFromLine(JSON.stringify({
    phase: "result", method: "octree", scenario: "dam-break-ui", steps: 62, simulationWall_ms: 4200,
  }))?.scenario, "dam-break-ui");
});

test("performance limits gate throughput and generic pressure-system attribution independently", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 120,
    gpuCommandAudit: {
      dispatches: 300, computePasses: 50,
      computePassesByLabel: { "Power faces": { calls: 50, bytes: 0 } },
    },
    physicsTrace: physicsTrace([
      { id: "pressure-system", label: "Pressure operator", duration_ms: 22 },
      { id: "pressure-solve", label: "MGPCG", duration_ms: 8 },
      { id: "other", label: "Other measured work", duration_ms: 50 },
    ]),
  });
  assert.deepEqual(powerDamPerformanceFailures(summary, {
    maximumAdvanceWall_ms: 59,
    maximumDispatchesPerAdvance: 149,
    maximumComputePassesPerAdvance: 24,
    maximumPressureNonSolve_ms: 21,
  }), [
    "advance wall 60.00 ms exceeds 59.00 ms",
    "dispatches/advance 150.0 exceeds 149.0",
    "compute passes/advance 25.0 exceeds 24.0",
    "pressure-system 22.00 ms exceeds 21.00 ms",
  ]);
});

test("compute-pass budget rejects missing and unnamed stage ownership", () => {
  const missing = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 1, gpuCommandAudit: { computePasses: 4 },
  });
  assert.deepEqual(powerDamPerformanceFailures(missing, {
    maximumComputePassesPerAdvance: 60,
  }), ["compute passes/stage attribution incomplete (2.0 unattributed/advance)"]);

  const unnamed = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 2,
    simulationWall_ms: 1, gpuCommandAudit: {
      computePasses: 4,
      computePassesByLabel: { "<unlabeled compute pass>": { calls: 4, bytes: 0 } },
    },
  });
  assert.deepEqual(powerDamPerformanceFailures(unnamed, {
    maximumComputePassesPerAdvance: 60,
  }), ["compute passes/stage attribution incomplete (2.0 unattributed/advance)"]);
});

test("configured structural gates fail closed when their audit source is absent", () => {
  const summary = summarizePowerDamPerformance({
    scenario: "dam-break-ui", method: "octree", phase: "result", steps: 1,
    simulationWall_ms: 1,
  });
  assert.deepEqual(powerDamPerformanceFailures(summary, {
    maximumDispatchesPerAdvance: 300,
    maximumComputePassesPerAdvance: 60,
    maximumPressureNonSolve_ms: 4,
  }), [
    "dispatches/advance unavailable for configured gate",
    "compute passes/advance unavailable for configured gate",
    "pressure-system phase unavailable for configured gate",
  ]);
});
