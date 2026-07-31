import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import {
  planOctreeAnalyticBootstrapBounds,
  sampleOctreeAnalyticBootstrapPhi,
} from "../lib/octree-analytic-bootstrap";
import { octreeFineSeedAdapterShader } from "../lib/webgpu-octree-fine-seed-adapter";
import { makeFineLevelSetTopologyWGSL } from "../lib/webgpu-octree-fine-levelset-topology";
import { octreeProjectionShader } from "../lib/webgpu-octree";

const bootstrapInput = {
  dimensions: [16, 16, 16] as const,
  containerSize: [0.8, 0.8, 0.8] as const,
  tileSizeCells: 8,
  initialCondition: "dam-break" as const,
  fillFraction: 0.22,
  interfaceBandCells: 3,
};

test("minimal dam t=0 phi has three exposed faces and no closed-wall free surfaces", () => {
  const plan = planOctreeAnalyticBootstrapBounds(bootstrapInput);
  const minimum = [
    -0.5 * bootstrapInput.containerSize[0],
    0,
    -0.5 * bootstrapInput.containerSize[2],
  ] as const;
  const maximum = [
    minimum[0] + plan.damBreak.width * bootstrapInput.containerSize[0],
    plan.damBreak.height * bootstrapInput.containerSize[1],
    minimum[2] + plan.damBreak.depth * bootstrapInput.containerSize[2],
  ] as const;

  for (let axis = 0; axis < 3; axis += 1) {
    const throughClosedWall = maximum.map((value) => value - 0.05) as [number, number, number];
    throughClosedWall[axis] = minimum[axis] - 0.05;
    assert.ok(sampleOctreeAnalyticBootstrapPhi(bootstrapInput, throughClosedWall) < 0,
      `closed contact wall ${axis} invented a liquid/air interface`);

    const beyondExposedFace = maximum.map((value) => value - 0.05) as [number, number, number];
    beyondExposedFace[axis] = maximum[axis] + 0.01;
    assert.ok(sampleOctreeAnalyticBootstrapPhi(bootstrapInput, beyondExposedFace) > 0,
      `exposed reservoir face ${axis} did not retain positive air`);
  }
});

test("all GPU t=0 dam classifiers use the same one-sided closed-wall phi", () => {
  const topologyShader = makeFineLevelSetTopologyWGSL("");
  for (const [label, shader, point] of [
    ["coarse topology", octreeProjectionShader, "world"],
    ["fine seed adapter", octreeFineSeedAdapterShader, "world"],
    ["global-fine topology", topologyShader, "point"],
  ] as const) {
    const compact = shader.replace(/\s+/g, "");
    assert.match(compact, new RegExp(`letq=${point}-exposedMaximum;returnlength\\(max\\(q,vec3f\\(0(?:\\.0)?\\)\\)\\)\\+min\\(max\\(q\\.x,max\\(q\\.y,q\\.z\\)\\),0(?:\\.0)?\\)`),
      `${label} drifted from the CPU closed-wall signed-distance convention`);
    assert.doesNotMatch(compact, /abs\([^)]*-exposedMaximum/,
      `${label} restored a box SDF and therefore a false free surface on a closed wall`);
  }
});

interface MinimalDamResult {
  phase: string;
  steps?: number;
  nonFiniteCount?: number;
  validationErrors?: unknown[];
  globalFineGenerationCheckpoints?: Array<{
    globalFineGeneration?: {
      generation: number;
      coarseGeneration?: number;
      publicationValid: boolean;
      activePages?: number;
      redistanceInitialPages?: number;
      redistanceFinalPages?: number;
    };
  }>;
}

function resultRecord(stdout: string): MinimalDamResult | undefined {
  return stdout.split("\n").flatMap((line) => {
    try {
      const value = JSON.parse(line) as MinimalDamResult;
      return value.phase === "result" ? [value] : [];
    } catch {
      return [];
    }
  }).at(-1);
}

test("Dawn keeps the band-1 minimal dam current through generation 15", {
  skip: !process.env.WEBGPU_NODE_MODULE
    ? "set WEBGPU_NODE_MODULE for the focused minimal-dam generation gate"
    : process.env.FLUID_MINIMAL_DAM_GENERATION_ACCEPTANCE !== "1"
      && "set FLUID_MINIMAL_DAM_GENERATION_ACCEPTANCE=1 for the focused 13-step gate",
  timeout: 180_000,
}, () => {
  // Do not inherit unrelated FLUID_* profiling, fallback, or validation
  // switches from a developer shell. This gate is the exact smallest clock
  // that reaches the former step-13 / generation-15 rejection.
  const cleanEnvironment = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !key.startsWith("FLUID_")),
  );
  const child = spawnSync(process.execPath, ["--import", "tsx", "tools/run-webgpu-smoke.ts"], {
    cwd: process.cwd(),
    encoding: "utf8",
    timeout: 150_000,
    killSignal: "SIGKILL",
    maxBuffer: 32 * 1024 * 1024,
    env: {
      ...cleanEnvironment,
      NODE_ENV: process.env.NODE_ENV ?? "test",
      ...(process.env.FLUID_WEBGPU_BACKEND
        ? { FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND } : {}),
      ...(process.env.FLUID_WEBGPU_ADAPTER
        ? { FLUID_WEBGPU_ADAPTER: process.env.FLUID_WEBGPU_ADAPTER } : {}),
      FLUID_SCENE: "minimal-power-dam-break",
      FLUID_METHOD: "octree",
      FLUID_QUALITY: "balanced",
      FLUID_TARGET_S: "0.052",
      FLUID_MAX_DT: "0.004",
      FLUID_ORACLE_STEPS: "13",
      FLUID_EXPECT_EXACT_STEPS: "13",
      FLUID_REQUIRE_SPATIAL_FIELD: "1",
      FLUID_CHECKPOINT_EVERY_S: "0.004",
      FLUID_VOXEL_CELL_SIZE: "0.05",
      FLUID_EXPECT_GRID: "16,16,16",
      // This is a focused publication invariant, not the authored two-second
      // scene acceptance lane. Keep the short exact run while still collecting
      // the generation checkpoints below.
      FLUID_PERFORMANCE_PROFILE: "1",
      FLUID_STABILITY_ENVELOPE: "1",
      FLUID_CPU_ORACLE: "0",
      FLUID_FIELD_STATS: "1",
      FLUID_RASTER_CHECKPOINTS: "0",
      FLUID_GLOBAL_FINE_GENERATION_TRANSITION: "1",
      FLUID_MAXIMUM_LEAF_SIZE: "32",
      FLUID_OCTREE_INTERFACE_BAND: "1",
      FLUID_OCTREE_GLOBAL_FINE_FACTOR: "4",
      FLUID_POWER_GENERATION_AUDIT: "1",
      FLUID_POWER_GENERATION_AUDIT_LOG: "0",
      FLUID_POWER_AUDIT_EVERY_STEPS: "1",
    },
  });
  assert.equal(child.error, undefined, `focused Dawn process failed: ${child.error?.message ?? "unknown"}`);
  assert.equal(child.status, 0,
    `minimal dam exposed a rejected or stale publication:\n${child.stderr}\n${child.stdout.slice(-12_000)}`);

  const result = resultRecord(child.stdout);
  assert.ok(result, "focused minimal-dam run emitted no result JSON");
  assert.equal(result.steps, 13);
  // The smoke omits undefined optional fields from its JSON result and
  // validates them with the same nullish-zero convention before reporting.
  assert.equal(result.nonFiniteCount ?? 0, 0);
  assert.deepEqual(result.validationErrors ?? [], []);
  const generations = (result.globalFineGenerationCheckpoints ?? [])
    .map((checkpoint) => checkpoint.globalFineGeneration);
  assert.equal(generations.length, 13, "every advance must expose one accepted generation checkpoint");
  generations.forEach((generation, index) => {
    assert.ok(generation, `checkpoint ${index + 1} omitted global-fine authority`);
    assert.equal(generation.publicationValid, true, `checkpoint ${index + 1} exposed an unpublished fine slot`);
    assert.equal(generation.coarseGeneration, generation.generation,
      `checkpoint ${index + 1} exposed a stale fine/coarse generation pair`);
    assert.ok((generation.activePages ?? 0) > 0,
      `checkpoint ${index + 1} exposed no live fine band`);
    assert.ok((generation.redistanceInitialPages ?? 0) > 0,
      `checkpoint ${index + 1} exposed no fine-band output worklist`);
    assert.ok((generation.redistanceFinalPages ?? 0) > 0,
      `checkpoint ${index + 1} exposed no fine-band support worklist`);
    assert.equal(generation.redistanceInitialPages, generation.redistanceFinalPages,
      `checkpoint ${index + 1} carried stale valid phi outside the recomputed band output halo`);
    if (index > 0) assert.ok(generation.generation > generations[index - 1]!.generation,
      `checkpoint ${index + 1} did not advance the immutable publication`);
  });
});
