import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  ADAPTIVE_MASS_ADVANCE_PHASE,
  ADAPTIVE_MASS_FLUID_PIPELINE,
  ADAPTIVE_MASS_GPU_WORK_CHUNKS,
  ADAPTIVE_MASS_RESIDENT_STAGE_PHASE,
} from "../lib/methods/adaptive-mass/adaptive-mass-frame-pipeline";
import {
  SPARSE_CM12_RESIDENT_STAGES,
  SPARSE_CM12_RESIDENT_WORK_CHUNKS,
} from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

/**
 * The SIM panel puts a figure on a stage only when that stage's declared seam
 * label is the label the encoder actually emits. Nothing at runtime enforces
 * that identity — a renamed phase or an unstaged dispatch degrades silently to
 * a node reading "—" — so it is pinned here.
 */

const residentStageSource = (): string => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf("  encode(\n    encoder: GPUCommandEncoder,");
  const end = source.indexOf(
    "  /** Publish generation zero without executing a physics step",
    begin,
  );
  assert.ok(begin >= 0 && end > begin, "the resident advance must be inspectable");
  return source.slice(begin, end);
};

test("the resident encoder closes every declared stage exactly once, in order", () => {
  const body = residentStageSource();
  const closed = [...body.matchAll(/\n {4}stage\("([a-z0-9-]+)"/g)]
    .map((match) => match[1]);
  assert.deepEqual(closed, [...SPARSE_CM12_RESIDENT_STAGES],
    "encode order and the stage ABI must be the same list");
});

test("no dispatch escapes the stage partition", () => {
  const body = residentStageSource();
  assert.doesNotMatch(body, /\n {4}dispatch\(/,
    "a dispatch outside a stage callback would be timed under its neighbour");
});

test("every stage the encoder emits has a phase of its own", () => {
  const labels = SPARSE_CM12_RESIDENT_STAGES
    .map((stage) => ADAPTIVE_MASS_RESIDENT_STAGE_PHASE[stage].label);
  assert.equal(new Set(labels).size, labels.length,
    "two stages sharing a label would sum into one unattributable figure");
});

test("the SIM diagram names every advance seam, and only those", () => {
  const emitted = ADAPTIVE_MASS_GPU_WORK_CHUNKS.map((chunk) => chunk.phase.label);
  const named = ADAPTIVE_MASS_FLUID_PIPELINE.stages.flatMap((stage) => stage.phaseLabels);
  assert.equal(new Set(named).size, named.length, "no two nodes may claim one seam");
  assert.deepEqual([...named].sort(), [...emitted].sort());
});

test("resident work chunks are closed exactly once", () => {
  const body = residentStageSource();
  const closed = [...body.matchAll(/closeWorkChunk\("([a-z0-9-]+)"\)/g)]
    .map((match) => match[1]);
  assert.deepEqual(closed, [...SPARSE_CM12_RESIDENT_WORK_CHUNKS]);
});

test("every timestamp phase has one concrete owner", () => {
  const labels = ADAPTIVE_MASS_GPU_WORK_CHUNKS.map((chunk) => chunk.phase.label);
  assert.equal(new Set(labels).size, labels.length);
  for (const chunk of ADAPTIVE_MASS_GPU_WORK_CHUNKS) {
    const stage = ADAPTIVE_MASS_FLUID_PIPELINE.stages.find(
      (candidate) => candidate.id === chunk.rollupStage);
    assert.ok(stage, `${chunk.id} names missing rollup ${chunk.rollupStage}`);
    assert.ok(stage.phaseLabels.includes(chunk.phase.label),
      `${chunk.id} is absent from its ${chunk.rollupStage} rollup`);
  }
});

test("a node without a seam of its own says whose passes carry it", () => {
  const byId = new Map(ADAPTIVE_MASS_FLUID_PIPELINE.stages.map((stage) => [stage.id, stage]));
  for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
    if (stage.phaseLabels.length > 0) {
      assert.equal(stage.costInsideStage, undefined,
        `${stage.id} owns a seam and must not also borrow one`);
      continue;
    }
    assert.ok(stage.costInsideStage, `${stage.id} would read as unmeasured forever`);
    assert.ok(byId.get(stage.costInsideStage)?.phaseLabels.length,
      `${stage.id} defers to a host that has no seam either`);
  }
});

test("every diagram node sits in a declared band", () => {
  const bands = new Set(ADAPTIVE_MASS_FLUID_PIPELINE.bands.map((band) => band.id));
  for (const stage of ADAPTIVE_MASS_FLUID_PIPELINE.stages) {
    assert.ok(bands.has(stage.band), `${stage.id} is in the orphan band ${stage.band}`);
  }
  const populated = new Set(ADAPTIVE_MASS_FLUID_PIPELINE.stages.map((stage) => stage.band));
  for (const band of bands) assert.ok(populated.has(band), `band ${band} draws nothing`);
});

test("pressure topology presents its live row census as labeled lines", () => {
  const stage = ADAPTIVE_MASS_FLUID_PIPELINE.stages.find(
    (candidate) => candidate.id === "pressure-topology",
  );
  assert.ok(stage);
  const context = {
    info: {
      adaptiveAcceptedRowCount: 12_480,
      adaptivePressureActiveRowCount: 5_940,
      adaptiveAcceptedSameLevelCoarseRowCount: 7_216,
      adaptiveAcceptedMixedSeamRowCount: 384,
    },
  } as Parameters<typeof stage.chip>[0];
  assert.equal(stage.chip(context), [
    "Accepted rows: 12,480",
    "Active in solve: 5,940",
    "Same-level coarse: 7,216",
    "Mixed seams: 384",
  ].join("\n"));
});

test("the CPU-lane brackets stay outside the diagram", () => {
  // Host planning and submission are real CPU intervals but no GPU stage owns
  // them; a node naming either would read 0 ms under a hardware partition.
  const named = new Set(ADAPTIVE_MASS_FLUID_PIPELINE.stages.flatMap((stage) => stage.phaseLabels));
  for (const phase of [
    ADAPTIVE_MASS_ADVANCE_PHASE.commandEncoding,
    ADAPTIVE_MASS_ADVANCE_PHASE.upload,
    ADAPTIVE_MASS_ADVANCE_PHASE.queueCompletion,
  ]) assert.ok(!named.has(phase.label), `${phase.label} is not GPU stage time`);
});
