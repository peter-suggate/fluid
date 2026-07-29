import assert from "node:assert/strict";
import test from "node:test";
import type { SceneDescription } from "../lib/model";
import {
  defineSceneDiagnosticHook,
  defineSceneDiagnostics,
  diagnosticRequest,
  evaluateDeclarativeDiagnosticRules,
  evaluateSceneDiagnostics,
  getDiagnosticValue,
  pathCheck,
  scalarCheck,
  type SceneDiagnosticEvidence,
} from "../lib/scene-diagnostics";

const scene = {
  sceneId: "diagnostic-fixture",
  container: { width_m: 2 },
} as unknown as SceneDescription;

const evidence: SceneDiagnosticEvidence = {
  methods: {
    octree: {
      diagnostics: {
        run: { acceptedSteps: 2 },
        stability: { maximumCfl: 0.75, samples: [{ drift: 0.01 }] },
      },
    },
    uniform: {
      diagnostics: {
        run: { acceptedSteps: 2 },
        stability: { maximumCfl: 1.25, samples: [{ drift: 0.03 }] },
      },
    },
  },
};

test("scene diagnostics evaluate scalar and path checks per method", () => {
  const contract = defineSceneDiagnostics({
    diagnostics: [diagnosticRequest("stability.envelope", { checkpoints: true })],
    checks: [
      scalarCheck({
        id: "run.exact-steps",
        select: { method: "*", path: "run.acceptedSteps" },
        operator: "equal",
        expected: 2,
      }),
      scalarCheck({
        id: "octree.cfl-limit",
        select: { method: "octree", path: "stability.maximumCfl" },
        operator: "less-than-or-equal",
        expected: 1,
      }),
      pathCheck({
        id: "legacy-field-removed",
        select: { method: "*", path: "legacy.rawBuffer" },
        operator: "absent",
      }),
    ],
  });

  const evaluation = evaluateSceneDiagnostics(contract, { scene, evidence });
  assert.equal(evaluation.passed, true);
  assert.equal(evaluation.failedErrorCount, 0);
  assert.deepEqual(
    evaluation.findings.map(({ checkId, method, passed }) => ({ checkId, method, passed })),
    [
      { checkId: "run.exact-steps", method: "octree", passed: true },
      { checkId: "run.exact-steps", method: "uniform", passed: true },
      { checkId: "octree.cfl-limit", method: "octree", passed: true },
      { checkId: "legacy-field-removed", method: "octree", passed: true },
      { checkId: "legacy-field-removed", method: "uniform", passed: true },
    ],
  );
});

test("missing methods and non-numeric values produce ordinary findings", () => {
  const contract = defineSceneDiagnostics({
    diagnostics: [],
    checks: [
      scalarCheck({
        id: "missing-method",
        select: { method: "tall-cell", path: "run.acceptedSteps" },
        operator: "equal",
        expected: 2,
      }),
      scalarCheck({
        id: "not-a-number",
        severity: "warning",
        select: { method: "octree", path: "stability" },
        operator: "finite",
      }),
    ],
  });

  const evaluation = evaluateSceneDiagnostics(contract, { scene, evidence });
  assert.equal(evaluation.passed, false);
  assert.equal(evaluation.failedErrorCount, 1);
  assert.equal(evaluation.failedWarningCount, 1);
  assert.equal(evaluation.findings[0]?.actual, undefined);
  assert.deepEqual(evaluation.findings[0]?.path, ["run", "acceptedSteps"]);
});

test("explicit paths traverse arrays without exposing runner internals", () => {
  assert.deepEqual(
    getDiagnosticValue(evidence, "uniform", ["stability", "samples", 0, "drift"]),
    { found: true, value: 0.03, path: ["stability", "samples", 0, "drift"] },
  );
  assert.equal(getDiagnosticValue(evidence, "uniform", ["stability", "samples", 2]).found, false);
});

test("pure hooks return stable namespaced findings", () => {
  const contract = defineSceneDiagnostics({
    diagnostics: [],
    checks: [],
    hooks: [defineSceneDiagnosticHook({
      id: "dam-spread",
      evaluate: ({ scene: authoredScene, evidence: collected, getValue }) => {
        const value = getValue(collected, "octree", "stability.samples.0.drift");
        return [{
          id: "initial-drift",
          passed: value.found && typeof value.value === "number" && value.value <= 0.02,
          message: `${authoredScene.sceneId} should begin inside its drift envelope`,
          method: "octree",
          path: value.path,
          actual: value.value,
          expected: { maximum: 0.02 },
        }];
      },
    })],
  });

  const evaluation = evaluateSceneDiagnostics(contract, { scene, evidence });
  assert.equal(evaluation.passed, true);
  assert.equal(evaluation.findings[0]?.checkId, "dam-spread.initial-drift");
  assert.equal(evaluation.findings[0]?.method, "octree");
});

test("contracts reject duplicate and unstable IDs before a run", () => {
  assert.throws(() => defineSceneDiagnostics({
    diagnostics: [],
    checks: [
      pathCheck({ id: "duplicate", select: { method: "octree", path: "a" }, operator: "present" }),
      pathCheck({ id: "duplicate", select: { method: "octree", path: "b" }, operator: "present" }),
    ],
  }), /Duplicate diagnostic check ID: duplicate/);

  assert.throws(() => diagnosticRequest("not stable"), /Invalid diagnostic request ID/);
  assert.throws(() => defineSceneDiagnostics({
    diagnostics: [],
    checks: [scalarCheck({
      id: "bad-finite",
      select: { method: "octree", path: "value" },
      operator: "finite",
      expected: 1,
    })],
  }), /cannot set expected/);
});

test("hook exceptions become a stable error finding", () => {
  const contract = defineSceneDiagnostics({
    diagnostics: [],
    checks: [],
    hooks: [defineSceneDiagnosticHook({
      id: "custom-check",
      evaluate: () => {
        throw new Error("bad derived metric");
      },
    })],
  });

  const evaluation = evaluateSceneDiagnostics(contract, { scene, evidence });
  assert.equal(evaluation.passed, false);
  assert.equal(evaluation.findings[0]?.checkId, "custom-check.execution");
  assert.match(evaluation.findings[0]?.message ?? "", /bad derived metric/);
});

test("declarative catalog rules support wildcards, transforms, predicates, references, and values", () => {
  const evaluation = evaluateDeclarativeDiagnosticRules([
    { id: "bounded-drift", metric: "methods.*.drift.abs", operator: "at-most", expected: 0.02 },
    { id: "matching-steps", metric: "methods.octree.samples", operator: "equal",
      expected: { selector: "methods.octree.steps" } },
    { id: "grid-shape", metric: "methods.octree.grid", operator: "equal", expected: [2, 3, 4] },
    { id: "late-speed", metric: "methods.*.speed", operator: "at-least", expected: 0.1,
      when: [{ metric: "methods.*.time_s", operator: "at-least", expected: 0.3 }] },
  ], {
    methods: {
      octree: { drift: -0.01, samples: 2, steps: 2, grid: [2, 3, 4], speed: 0.2, time_s: 0.5 },
      uniform: { drift: 0.015, speed: 0, time_s: 0.2 },
    },
  });

  assert.equal(evaluation.passed, true);
  assert.deepEqual(evaluation.findings.map(({ checkId, method }) => [checkId, method]), [
    ["bounded-drift", "octree"],
    ["bounded-drift", "uniform"],
    ["matching-steps", "octree"],
    ["grid-shape", "octree"],
    ["late-speed", "octree"],
  ]);
});
