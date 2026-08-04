import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/model";
import { sceneDiagnosticPackImplementations } from "../lib/scene-diagnostic-pack-implementations";
import { createSceneDiagnosticRuntime, type NormalizedSceneDiagnosticEvidence } from "../lib/scene-diagnostic-runtime";
import type { SceneWebGPUDiagnosticPack, SceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke";

const expectedPackIds = [
  "authoritative-water-raster",
  "core-webgpu-health",
  "cross-method-field-parity",
  "deep-compression",
  "equilibrium",
  "exhaustive-power-generation",
  "global-fine-publication",
  "inflow-activity",
  "octree-authority",
  "performance",
  "quadtree-dam-parity",
  "settling",
  "structured-power",
  "tall-cell-restricted",
  "volume-and-topology",
] as const;

function lane(diagnostic: SceneWebGPUDiagnosticPack): SceneWebGPUSmokeLane {
  return {
    id: "pack-test",
    description: "pack fixture",
    methods: [{ id: "octree", quality: "balanced", overrides: {} }],
    stop: { simulatedTime_s: 0.008, exactSteps: 2, maxDt_s: 0.004 },
    oracle: { enabled: false, matchedSteps: 1 },
    collect: {
      evidenceCollectors: [],
      fieldStats: "final", stabilityEnvelope: false, spatialField: false,
      sparsePublication: false, raster: "none", globalFineGeneration: false,
      powerGenerationAudit: false, boundaryThetaHistogram: false,
      structuredValidation: false,
      performanceProfile: false, gpuCommandAudit: false,
    },
    diagnostics: [diagnostic],
    acceptance: [],
    hooks: [],
  };
}

test("the built-in registry is exhaustive over every declared diagnostic pack ID", () => {
  assert.deepEqual(Object.keys(sceneDiagnosticPackImplementations).sort(), [...expectedPackIds].sort());
  for (const id of expectedPackIds) assert.equal(sceneDiagnosticPackImplementations[id].id, id);
});

test("core health consumes only normalized run/solver evidence and exact lane settings", () => {
  const evidence: NormalizedSceneDiagnosticEvidence = {
    methods: {
      octree: {
        available: ["run", "solver"],
        diagnostics: {
          run: { steps: 2 },
          solver: {
            encodedSteps: 2,
            submittedTime_s: 0.008,
            completedTime_s: 0.008,
            nonFiniteCount: 0,
            maxSpeed_m_s: 1.25,
          },
          validationErrors: [],
        },
      },
    },
  };
  const runtime = createSceneDiagnosticRuntime({ packs: sceneDiagnosticPackImplementations, hooks: {} });
  const result = runtime.evaluate({ scene: defaultScene, lane: lane({ id: "core-webgpu-health" }), evidence });

  assert.equal(result.passed, true);
  assert.equal(result.packFindings.length, 8);
  assert.ok(result.packFindings.every(({ checkId }) => checkId.startsWith("pack.core-webgpu-health.")));
});

test("volume pack fails closed on malformed or out-of-envelope summaries", () => {
  const evidence: NormalizedSceneDiagnosticEvidence = {
    methods: {
      octree: {
        available: ["field summary"],
        diagnostics: {
          field: {
            grid: [4, 4, 4],
            matched: { summary: { minimum: -0.2, maximum: 1, componentCount: 1 } },
            final: { summary: { minimum: 0, maximum: Number.NaN, componentCount: 1 } },
          },
        },
      },
    },
  };
  const runtime = createSceneDiagnosticRuntime({ packs: sceneDiagnosticPackImplementations, hooks: {} });
  const result = runtime.evaluate({ scene: defaultScene, lane: lane({ id: "volume-and-topology" }), evidence });

  assert.equal(result.passed, false);
  assert.deepEqual(result.packFindings.filter(({ passed }) => !passed).map(({ checkId }) => checkId), [
    "pack.volume-and-topology.matched-minimum",
    "pack.volume-and-topology.final-maximum",
  ]);
});

test("octree authority does not invent dam-envelope failures for timing-only evidence", () => {
  const evidence: NormalizedSceneDiagnosticEvidence = {
    methods: {
      octree: {
        available: ["solver", "octree authority"],
        diagnostics: {
          solver: {
            gridKind: "octree",
            quadtreeMaximumNeighborRatio: 2,
            quadtreeTopologyReadbackBytes: 0,
            powerDiagramReady: true,
            powerDiagramAuthoritative: true,
            pressureSolver: "Octree power MGPCG · row-parallel exact-reduction executor · Section 4.3 fixed schedule",
          },
          octree: {
            powerTopology: {
              descriptor: { errorCount: 0 },
              topology: { invalidCount: 0 },
            },
          },
        },
      },
    },
  };
  const runtime = createSceneDiagnosticRuntime({ packs: sceneDiagnosticPackImplementations, hooks: {} });
  const result = runtime.evaluate({ scene: defaultScene, lane: lane({ id: "octree-authority" }), evidence });

  assert.equal(result.passed, true);
  assert.equal(result.packFindings.some(({ checkId }) => checkId.includes(".dam-")), false);
});

test("water raster authority follows the retained accepted fine-grid generation", () => {
  const generation = { generation: 7 };
  const raster = {
    frontInterfacePixels: 10,
    backInterfacePixels: 8,
    backOnlyInterfacePixels: 0,
    reverseView: { backOnlyInterfacePixels: 0 },
    surfaceGeometrySource: "global-fine-coarse",
    globalFineCrossingPublished: true,
    presentationFallbackActive: false,
    globalFineAuthorityLatch: 1,
    globalFineAuthorityTransition: { validGeneration: 7 },
  };
  const evidence: NormalizedSceneDiagnosticEvidence = {
    methods: {
      octree: {
        available: ["global fine generation", "front/back raster"],
        diagnostics: {
          initialGlobalFineGeneration: generation,
          finalGlobalFineGeneration: generation,
          initialGlobalFineRaster: raster,
          finalGlobalFineRaster: raster,
        },
      },
    },
  };
  const runtime = createSceneDiagnosticRuntime({ packs: sceneDiagnosticPackImplementations, hooks: {} });
  const result = runtime.evaluate({
    scene: defaultScene,
    lane: lane({ id: "authoritative-water-raster" }),
    evidence,
  });

  assert.equal(result.passed, true);
  assert.ok(result.packFindings
    .filter(({ checkId }) => checkId.endsWith("-authority"))
    .every(({ passed }) => passed));
});
