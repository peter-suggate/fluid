import assert from "node:assert/strict";
import test from "node:test";
import {
  fluidPipelinePhaseCosts,
  measureFluidPipelineBand,
  measureFluidPipelineStage,
  type FluidPipelineContext,
} from "../lib/fluid-pipeline";
import {
  UNIFORM_ADVANCE_PHASE,
  UNIFORM_FLUID_PIPELINE,
} from "../lib/webgpu-uniform-reference";
import type { PerformanceTrace } from "../lib/performance-trace";

/**
 * The colocation contract, held mechanically: the graph a user reads and the
 * seams the encoder emits are the same partition. A stage that names a label
 * the solver never emits shows a permanent dash; a label no stage owns is
 * advance time the diagram silently drops. Both are drift, and both are exactly
 * what colocating graph and encoder in one file is supposed to prevent.
 */

const PHASE_LABELS = Object.values(UNIFORM_ADVANCE_PHASE).map((phase) => phase.label);

const context = (patch: Partial<FluidPipelineContext> = {}): FluidPipelineContext => ({
  values: { densityPostProcessing: "scene" },
  info: null,
  sceneId: "dam-break",
  bodyCount: 0,
  hasTerrain: false,
  hasInflow: false,
  running: true,
  ...patch,
});

test("every stage is measurable, or explicitly shared or structural", () => {
  for (const stage of UNIFORM_FLUID_PIPELINE.stages) {
    assert.ok(
      stage.phaseLabels.length > 0 || stage.costInsideStage || stage.spendsNoFrameTime,
      `${stage.id} has no phase labels, no host stage, and claims frame time`,
    );
  }
});

test("every stage label is a phase the encoder emits", () => {
  for (const stage of UNIFORM_FLUID_PIPELINE.stages) {
    for (const label of stage.phaseLabels) {
      assert.ok(PHASE_LABELS.includes(label),
        `${stage.id} names "${label}", which is not in UNIFORM_ADVANCE_PHASE`);
    }
  }
});

test("the stages partition the advance: every phase label owned exactly once", () => {
  const owners = new Map<string, string[]>();
  for (const stage of UNIFORM_FLUID_PIPELINE.stages) {
    for (const label of stage.phaseLabels) {
      owners.set(label, [...owners.get(label) ?? [], stage.id]);
    }
  }
  for (const label of PHASE_LABELS) {
    const owning = owners.get(label) ?? [];
    assert.equal(owning.length, 1,
      `"${label}" is owned by [${owning.join(", ")}]; the partition needs exactly one owner`);
  }
});

test("every stage sits in a declared band and every band holds a stage", () => {
  const bandIds = new Set(UNIFORM_FLUID_PIPELINE.bands.map((band) => band.id));
  const used = new Set<string>();
  for (const stage of UNIFORM_FLUID_PIPELINE.stages) {
    assert.ok(bandIds.has(stage.band), `${stage.id} sits in undeclared band "${stage.band}"`);
    used.add(stage.band);
  }
  for (const band of UNIFORM_FLUID_PIPELINE.bands) {
    assert.ok(used.has(band.id), `band "${band.id}" holds no stage`);
  }
});

test("costInsideStage always names a stage in the graph", () => {
  const ids = new Set(UNIFORM_FLUID_PIPELINE.stages.map((stage) => stage.id));
  for (const stage of UNIFORM_FLUID_PIPELINE.stages) {
    if (stage.costInsideStage) {
      assert.ok(ids.has(stage.costInsideStage),
        `${stage.id} claims to run inside unknown stage "${stage.costInsideStage}"`);
    }
  }
});

test("state and chip stay total over solver-less and gated contexts", () => {
  const contexts = [
    context(),
    context({ bodyCount: 3, hasTerrain: true, hasInflow: true }),
    context({ sceneId: "symmetric-expansion", running: false }),
    context({ values: { densityPostProcessing: "on" } }),
  ];
  for (const stage of UNIFORM_FLUID_PIPELINE.stages) {
    for (const candidate of contexts) {
      assert.ok(["on", "off", "unavailable"].includes(stage.state(candidate)));
      assert.equal(typeof stage.chip(candidate), "string");
    }
  }
});

test("gated stages read their context", () => {
  const stageById = new Map(UNIFORM_FLUID_PIPELINE.stages.map((stage) => [stage.id, stage]));
  assert.equal(stageById.get("solid-excess")?.state(context()), "unavailable");
  assert.equal(stageById.get("solid-excess")?.state(context({ hasTerrain: true })), "on");
  assert.equal(stageById.get("rigid-coupling")?.state(context()), "unavailable");
  assert.equal(stageById.get("rigid-coupling")?.state(context({ bodyCount: 2 })), "on");
  // The Sec. 3.8 gate resolves "scene" exactly as solver construction does.
  assert.equal(stageById.get("density-post-process")?.state(context()), "off");
  assert.equal(
    stageById.get("density-post-process")?.state(context({ sceneId: "symmetric-expansion" })), "on");
  assert.equal(
    stageById.get("density-post-process")?.state(context({ sceneId: "minimal-power-dam-break-64" })), "on");
  assert.equal(
    stageById.get("density-post-process")?.state(context({ values: { densityPostProcessing: "on" } })), "on");
});

/** A synthetic averaged hardware partition carrying a chosen subset of phases. */
const syntheticTrace = (
  phases: ReadonlyArray<{ label: string; duration_ms: number; encodedFraction?: number }>,
): PerformanceTrace => ({
  sampleId: 1,
  domain: "gpu",
  lane: "physics",
  context: "test",
  measurementSource: "gpu-hardware-timestamp",
  capturedAt_ms: 0,
  total_ms: phases.reduce((sum, phase) => sum + phase.duration_ms, 0),
  span_ms: phases.reduce((sum, phase) => sum + phase.duration_ms, 0),
  phases: phases.map((phase, index) => ({
    id: "other",
    label: phase.label,
    duration_ms: phase.duration_ms,
    start_ms: index,
    end_ms: index + 1,
    ...(phase.encodedFraction !== undefined ? { encodedFraction: phase.encodedFraction } : {}),
  })),
} as unknown as PerformanceTrace);

test("measured stages sum their own labels; the full partition sums to the advance", () => {
  const costs = fluidPipelinePhaseCosts(syntheticTrace(
    PHASE_LABELS.map((label, index) => ({ label, duration_ms: index + 1 }))));
  const total_ms = PHASE_LABELS.reduce((sum, _, index) => sum + index + 1, 0);
  let stageSum = 0;
  const liveContext = context({
    bodyCount: 1, hasTerrain: true, values: { densityPostProcessing: "on" },
  });
  for (const stage of UNIFORM_FLUID_PIPELINE.stages) {
    const cost = measureFluidPipelineStage(
      stage, UNIFORM_FLUID_PIPELINE.stages, costs, total_ms, stage.state(liveContext));
    assert.equal(cost.kind, "measured", `${stage.id} should measure from a full partition`);
    stageSum += cost.duration_ms ?? 0;
  }
  assert.ok(Math.abs(stageSum - total_ms) < 1e-9,
    `stage costs sum to ${stageSum}, advance is ${total_ms}; the partition leaks`);
  const bandSum = UNIFORM_FLUID_PIPELINE.bands.reduce((sum, band) => sum
    + (measureFluidPipelineBand(band.id, UNIFORM_FLUID_PIPELINE, costs, total_ms).duration_ms ?? 0), 0);
  assert.ok(Math.abs(bandSum - total_ms) < 1e-9, "band costs must also sum to the advance");
});

test("a closed gate wins over a stale label in the averaged window", () => {
  const rigid = UNIFORM_FLUID_PIPELINE.stages.find((stage) => stage.id === "rigid-coupling");
  assert.ok(rigid);
  const costs = fluidPipelinePhaseCosts(syntheticTrace([
    { label: UNIFORM_ADVANCE_PHASE.rigidCoupling.label, duration_ms: 2, encodedFraction: 0.5 },
  ]));
  // The window still carries the label from samples taken before the last body
  // was deleted; the state says the stage no longer encodes. Now wins.
  const cost = measureFluidPipelineStage(
    rigid, UNIFORM_FLUID_PIPELINE.stages, costs, 2, "unavailable");
  assert.equal(cost.kind, "withheld");
  assert.equal(cost.duration_ms, 0);
});

test("intermittent phases price the expected advance, badged by encoded fraction", () => {
  const costs = fluidPipelinePhaseCosts(syntheticTrace([
    { label: UNIFORM_ADVANCE_PHASE.solidExcess.label, duration_ms: 4, encodedFraction: 0.25 },
  ]));
  const cost = costs.get(UNIFORM_ADVANCE_PHASE.solidExcess.label);
  assert.equal(cost?.expected_ms, 1);
  assert.equal(cost?.encodedFraction, 0.25);
  const stage = UNIFORM_FLUID_PIPELINE.stages.find((candidate) => candidate.id === "solid-excess");
  assert.ok(stage);
  const measured = measureFluidPipelineStage(
    stage, UNIFORM_FLUID_PIPELINE.stages, costs, 10, "on");
  assert.equal(measured.kind, "measured");
  assert.equal(measured.duration_ms, 1);
  assert.equal(measured.encodedFraction, 0.25);
});

test("once a partition arrives, an absent gated stage reads idle, not unmeasured", () => {
  const stage = UNIFORM_FLUID_PIPELINE.stages.find((candidate) => candidate.id === "density-post-process");
  assert.ok(stage);
  const costs = fluidPipelinePhaseCosts(syntheticTrace([
    { label: UNIFORM_ADVANCE_PHASE.densityAdvection.label, duration_ms: 3 },
  ]));
  assert.equal(measureFluidPipelineStage(
    stage, UNIFORM_FLUID_PIPELINE.stages, costs, 3, "on").kind, "idle");
  assert.equal(measureFluidPipelineStage(
    stage, UNIFORM_FLUID_PIPELINE.stages, new Map(), 0, "on").kind, "unmeasured");
});
