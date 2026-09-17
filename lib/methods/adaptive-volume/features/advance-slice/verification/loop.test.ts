import assert from "node:assert/strict";
import test from "node:test";
import { ADVANCE_STAGE_ORDER } from "../advance-work";
import {
  ADVANCE_LOOP_STEP_NAMES, ADVANCE_REPRESENT_SLICE, ADVANCE_SURFACE_VIEWS,
  ADVANCE_TRANSPORT_EXPERIMENTS, ADVANCE_TRANSPORT_EXPERIMENT_ORDER,
  ADVANCE_DEFAULT_TRANSPORT_EXPERIMENT, isAdvanceTransportExperiment,
} from "../definition";
import {
  ADVANCE_LOOP_STEPS, advanceStageLoopStep, advanceStageSlice,
} from "../loop";
import { SPARSE_CM12_STAGES, sparseCM12Stage } from "../../../sparse-cm12-stages";
import type { SparseCM12ResidentStageId } from "../../../webgpu-sparse-cm12-resident";

test("every encoded stage declares one reading, and every retired one declares none", () => {
  const encoded = new Set<string>(ADVANCE_STAGE_ORDER);
  for (const stage of Object.keys(SPARSE_CM12_STAGES) as SparseCM12ResidentStageId[]) {
    assert.equal(Boolean(sparseCM12Stage(stage).slice), encoded.has(stage),
      encoded.has(stage)
        ? `${stage} is encoded and must declare a slice reading`
        : `${stage} is retired from the graph and must not offer a lens`);
  }
});

test("the loop's four readings partition the advance in encode order", () => {
  assert.deepEqual(ADVANCE_LOOP_STEPS.map(step => step.n), [1, 2, 3, 4]);
  assert.deepEqual(ADVANCE_LOOP_STEPS.map(step => step.name),
    [1, 2, 3, 4].map(n => ADVANCE_LOOP_STEP_NAMES[n as 1 | 2 | 3 | 4]));

  const represent = ADVANCE_LOOP_STEPS[0]!;
  assert.deepEqual([represent.from, represent.to], [0, 0],
    "step 1 is the state the advance starts from, so it encodes no stage");

  /* Exactly one step, and the covering steps laid end to end with no gap: a
   * stage counted twice would be highlighted under two readings, and a gap is
   * a stage the strip draws with no reading at all. */
  const covering = ADVANCE_LOOP_STEPS.filter(step => step.from >= 1);
  let expectedFrom = 1;
  for (const step of covering) {
    assert.equal(step.from, expectedFrom,
      `${step.name} must begin where the previous reading ended`);
    assert.ok(step.to >= step.from, `${step.name} covers no stage`);
    expectedFrom = step.to + 1;
  }
  assert.equal(expectedFrom - 1, ADVANCE_STAGE_ORDER.length,
    "the readings must reach the last stage the encoder writes");

  ADVANCE_STAGE_ORDER.forEach((stage, index) => {
    const owners = covering.filter(step => index + 1 >= step.from && index + 1 <= step.to);
    assert.equal(owners.length, 1, `${stage} belongs to ${owners.length} readings, not one`);
    assert.equal(owners[0]!.n, advanceStageSlice(stage).loopStep);
    assert.equal(advanceStageLoopStep(stage).n, owners[0]!.n);
  });
});

test("a declared mark can be named, coloured and read out", () => {
  const marks = [...ADVANCE_REPRESENT_SLICE.keys,
    ...ADVANCE_STAGE_ORDER.flatMap(stage => advanceStageSlice(stage).keys)];
  assert.ok(marks.length > 20, "the whole declared set is under test, not a corner of it");
  const ids = new Set<string>();
  for (const mark of marks) {
    assert.ok(mark.label.length > 0, `${mark.id} has no name`);
    assert.ok(mark.note.length > 12,
      `${mark.id} has nothing to say, so it has no reason to appear in the probe`);
    ids.add(mark.id);
  }
  for (const stage of ADVANCE_STAGE_ORDER) {
    const keys = advanceStageSlice(stage).keys;
    assert.equal(new Set(keys.map(key => key.id)).size, keys.length,
      `${stage} declares the same mark id twice, so one predicate answers for both`);
    assert.ok(advanceStageSlice(stage).caption.length > 40,
      `${stage} needs a caption a reader can act on`);
  }
});

test("the surface and transport choices are declared once, with their defaults", () => {
  assert.deepEqual(ADVANCE_SURFACE_VIEWS.map(view => view.id),
    ["shared-rdf", "plic", "direct-level-set"]);
  assert.deepEqual(ADVANCE_SURFACE_VIEWS.filter(view => view.selectable).map(view => view.id),
    ["shared-rdf", "plic"],
    "the direct level set is what the level-set transport publishes, not a choice");
  assert.ok(ADVANCE_SURFACE_VIEWS.every(view => view.hint.length > 0));

  assert.deepEqual([...ADVANCE_TRANSPORT_EXPERIMENT_ORDER],
    Object.keys(ADVANCE_TRANSPORT_EXPERIMENTS));
  for (const id of ADVANCE_TRANSPORT_EXPERIMENT_ORDER) {
    const experiment = ADVANCE_TRANSPORT_EXPERIMENTS[id];
    assert.equal(experiment.id, id, "an arm filed under the wrong key is the wrong arm");
    assert.ok(experiment.defaultPressureBudget >= 4);
    assert.ok(experiment.pressureTolerance > 0);
    assert.equal(isAdvanceTransportExperiment(id), true);
  }
  assert.equal(ADVANCE_TRANSPORT_EXPERIMENTS.baseline.defaultPressureBudget, 28);
  assert.equal(ADVANCE_TRANSPORT_EXPERIMENTS["cellwise-remap"].option?.mode, "cellwise-remap");
  assert.equal(ADVANCE_TRANSPORT_EXPERIMENTS["level-set-volume"].option, undefined,
    "only the cellwise arm is parameterised; the others ride their bare selector");
  assert.equal(isAdvanceTransportExperiment("swept"), false);
  assert.equal(isAdvanceTransportExperiment(null), false);
  assert.equal(
    ADVANCE_TRANSPORT_EXPERIMENTS[ADVANCE_DEFAULT_TRANSPORT_EXPERIMENT].id,
    ADVANCE_DEFAULT_TRANSPORT_EXPERIMENT);
});
