import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/model";
import {
  createSceneDiagnosticRuntime,
  type DiagnosticPackImplementation,
  type NormalizedSceneDiagnosticEvidence,
  type SceneHookImplementation,
} from "../lib/scene-diagnostic-runtime";
import type { SceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke";

function lane(
  overrides: Partial<Pick<SceneWebGPUSmokeLane, "diagnostics" | "acceptance" | "hooks">> = {},
): SceneWebGPUSmokeLane {
  return {
    id: "test",
    description: "runtime fixture",
    methods: [
      { id: "octree", quality: "balanced", overrides: {} },
      { id: "uniform", quality: "balanced", overrides: {} },
    ],
    stop: { simulatedTime_s: 0.1 },
    oracle: { enabled: false, matchedSteps: 1 },
    collect: {
      evidenceCollectors: [],
      fieldStats: "final",
      stabilityEnvelope: false,
      spatialField: false,
      sparsePublication: false,
      raster: "none",
      globalFineGeneration: false,
      powerGenerationAudit: false,
      boundaryThetaHistogram: false,
      structuredValidation: false,
      performanceProfile: false,
      gpuCommandAudit: false,
    },
    diagnostics: overrides.diagnostics ?? [],
    acceptance: overrides.acceptance ?? [],
    hooks: overrides.hooks ?? [],
  };
}

function evidence(
  octreeAvailable: readonly string[] = ["checkpoint fields", "stability envelope"],
): NormalizedSceneDiagnosticEvidence {
  return {
    methods: {
      octree: {
        available: octreeAvailable,
        diagnostics: { run: { steps: 2 }, stability: { maximumCfl: 0.8 } },
      },
      uniform: {
        available: ["stability envelope"],
        diagnostics: { run: { steps: 2 }, stability: { maximumCfl: 1.2 } },
      },
    },
  };
}

test("runtime dispatches packs and hooks by registry ID with normalized context", () => {
  const pack: DiagnosticPackImplementation<"core-webgpu-health"> = {
    id: "core-webgpu-health",
    requires: ["stability envelope"],
    evaluate: ({ selectedMethods, getMethod }) => selectedMethods.map((method) => ({
      id: "cfl-finite",
      method,
      passed: Number.isFinite((getMethod(method)?.diagnostics.stability as { maximumCfl: number }).maximumCfl),
      message: `${method} CFL should be finite`,
    })),
  };
  const hook: SceneHookImplementation<"minimal-dam-motion"> = {
    id: "minimal-dam-motion",
    requires: ["checkpoint fields"],
    evaluate: ({ parameters, selectedMethods }) => [{
      id: "minimum-speed",
      method: selectedMethods[0],
      passed: parameters.minimumPeakSpeed_m_s === 0.1,
      message: "scene-owned speed threshold is applied",
      expected: parameters.minimumPeakSpeed_m_s,
    }],
  };
  const runtime = createSceneDiagnosticRuntime({
    packs: { "core-webgpu-health": pack },
    hooks: { "minimal-dam-motion": hook },
  });
  const result = runtime.evaluate({
    scene: defaultScene,
    lane: lane({
      diagnostics: [{ id: "core-webgpu-health" }],
      acceptance: [{ id: "exact-steps", metric: "methods.*.run.steps", operator: "equal", expected: 2 }],
      hooks: [{
        id: "minimal-dam-motion",
        methods: ["octree"],
        requires: ["checkpoint fields"],
        parameters: { minimumPeakSpeed_m_s: 0.1 },
      }],
    }),
    evidence: evidence(),
  });

  assert.equal(result.passed, true);
  assert.deepEqual(result.findings.map(({ checkId, method }) => ({ checkId, method })), [
    { checkId: "exact-steps", method: "octree" },
    { checkId: "exact-steps", method: "uniform" },
    { checkId: "pack.core-webgpu-health.cfl-finite", method: "octree" },
    { checkId: "pack.core-webgpu-health.cfl-finite", method: "uniform" },
    { checkId: "hook.minimal-dam-motion.minimum-speed", method: "octree" },
  ]);
});

test("missing implementations and method evidence fail closed without throwing", () => {
  const runtime = createSceneDiagnosticRuntime({ packs: {}, hooks: {} });
  const result = runtime.evaluate({
    scene: defaultScene,
    lane: lane({
      diagnostics: [{ id: "equilibrium" }],
      hooks: [{ id: "ocean-wave-profile", methods: ["octree"], requires: ["checkpoint fields"] }],
    }),
    evidence: evidence(),
  });

  assert.equal(result.passed, false);
  assert.deepEqual(result.findings.map(({ checkId }) => checkId), [
    "pack.equilibrium.implementation",
    "hook.ocean-wave-profile.implementation",
  ]);
});

test("every lane method must publish normalized evidence", () => {
  const runtime = createSceneDiagnosticRuntime({ packs: {}, hooks: {} });
  const onlyOctree: NormalizedSceneDiagnosticEvidence = {
    methods: { octree: evidence().methods.octree! },
  };
  const result = runtime.evaluate({ scene: defaultScene, lane: lane(), evidence: onlyOctree });

  assert.equal(result.passed, false);
  assert.equal(result.runtimeFindings[0]?.checkId, "runtime.method-evidence");
  assert.equal(result.runtimeFindings[0]?.method, "uniform");
});

test("missing required evidence prevents implementation execution", () => {
  let called = false;
  const hook: SceneHookImplementation<"ocean-wave-profile"> = {
    id: "ocean-wave-profile",
    requires: ["checkpoint fields"],
    evaluate: () => {
      called = true;
      return [];
    },
  };
  const runtime = createSceneDiagnosticRuntime({ packs: {}, hooks: { "ocean-wave-profile": hook } });
  const result = runtime.evaluate({
    scene: defaultScene,
    lane: lane({
      hooks: [{ id: "ocean-wave-profile", methods: ["octree"], requires: ["checkpoint fields"] }],
    }),
    evidence: evidence([]),
  });

  assert.equal(called, false);
  assert.equal(result.passed, false);
  assert.equal(result.findings[0]?.checkId, "hook.ocean-wave-profile.requirements");
  assert.equal(result.findings[0]?.method, "octree");
});

test("hook implementation requirements must be declared by the scene", () => {
  const hook: SceneHookImplementation<"water-raster-integrity"> = {
    id: "water-raster-integrity",
    requires: ["front/back raster"],
    evaluate: () => [],
  };
  const runtime = createSceneDiagnosticRuntime({ packs: {}, hooks: { "water-raster-integrity": hook } });
  const result = runtime.evaluate({
    scene: defaultScene,
    lane: lane({
      hooks: [{ id: "water-raster-integrity", methods: ["octree"], requires: [] }],
    }),
    evidence: evidence(),
  });

  assert.equal(result.passed, false);
  assert.equal(result.findings[0]?.checkId, "hook.water-raster-integrity.requirements-declaration");
});

test("bad plugin output and implementation exceptions become stable findings", () => {
  const badId: DiagnosticPackImplementation<"performance"> = {
    id: "performance",
    evaluate: () => [{ id: "not stable", passed: true, message: "invalid" }],
  };
  const throwing: SceneHookImplementation<"hose-jet-drift"> = {
    id: "hose-jet-drift",
    evaluate: () => { throw new Error("derived field is malformed"); },
  };
  const runtime = createSceneDiagnosticRuntime({
    packs: { performance: badId },
    hooks: { "hose-jet-drift": throwing },
  });
  const result = runtime.evaluate({
    scene: defaultScene,
    lane: lane({
      diagnostics: [{ id: "performance", methods: ["octree"] }],
      hooks: [{ id: "hose-jet-drift", methods: ["octree"], requires: [] }],
    }),
    evidence: evidence(),
  });

  assert.equal(result.passed, false);
  assert.deepEqual(result.findings.map(({ checkId }) => checkId), [
    "pack.performance.finding-id",
    "hook.hose-jet-drift.execution",
  ]);
});

test("malformed finding severity cannot accidentally turn a failure into a pass", () => {
  const malformed = {
    id: "performance",
    evaluate: () => [{ id: "budget", passed: false, severity: "fatal", message: "over budget" }],
  } as unknown as DiagnosticPackImplementation<"performance">;
  const runtime = createSceneDiagnosticRuntime({ packs: { performance: malformed }, hooks: {} });
  const result = runtime.evaluate({
    scene: defaultScene,
    lane: lane({ diagnostics: [{ id: "performance", methods: ["octree"] }] }),
    evidence: evidence(),
  });

  assert.equal(result.passed, false);
  assert.equal(result.findings[0]?.checkId, "pack.performance.finding-contract");
});

test("registry keys are checked once at construction", () => {
  assert.throws(() => createSceneDiagnosticRuntime({
    packs: { equilibrium: { id: "performance", evaluate: () => [] } } as never,
    hooks: {},
  }), /key equilibrium differs from implementation ID performance/);
});
