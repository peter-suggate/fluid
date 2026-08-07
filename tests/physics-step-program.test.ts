import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  OCTREE_STEP_PROGRAM,
  StepSequenceRecorder,
  physicsStepProgramJSON,
  stepSequenceDeviations,
  validatePhysicsStepProgram,
} from "../lib/physics-step-program";

test("the octree step program is dependency-consistent", () => {
  assert.deepEqual(validatePhysicsStepProgram(OCTREE_STEP_PROGRAM), []);
});

test("the persisted JSON contract matches the declared program", () => {
  const persisted = readFileSync(new URL("../docs/physics-step-program.json", import.meta.url), "utf8");
  assert.equal(persisted, physicsStepProgramJSON(OCTREE_STEP_PROGRAM),
    "docs/physics-step-program.json is stale; regenerate it from OCTREE_STEP_PROGRAM");
});

test("validation rejects reads with no earlier writer and unknown carried values", () => {
  const failures = validatePhysicsStepProgram({
    id: "broken", description: "", driverContract: [],
    stages: [
      { id: "a", label: "", phase: "other", reads: ["missing"], carriedReads: ["never-produced"], writes: ["x"] },
      { id: "a", label: "", phase: "other", reads: [], carriedReads: [], writes: [] },
    ],
  });
  assert.ok(failures.some((entry) => entry.includes("reads \"missing\"")));
  assert.ok(failures.some((entry) => entry.includes("carries \"never-produced\"")));
  assert.ok(failures.some((entry) => entry.includes("duplicate stage id")));
});

const FULL_SEQUENCE = [
  "ready-topology-flip", "surface-transport", "pressure-projection",
  "inactive-topology-candidate", "rigid-exchange", "sparse-brick-world", "step-snapshot",
];

test("the canonical executed sequence conforms", () => {
  assert.deepEqual(stepSequenceDeviations(FULL_SEQUENCE, OCTREE_STEP_PROGRAM), []);
});

test("optional stages may be absent; required stages may not", () => {
  const withoutRigid = FULL_SEQUENCE.filter((id) => id !== "rigid-exchange");
  assert.deepEqual(stepSequenceDeviations(withoutRigid, OCTREE_STEP_PROGRAM), []);
  const withoutFlip = FULL_SEQUENCE.filter((id) => id !== "ready-topology-flip");
  assert.ok(stepSequenceDeviations(withoutFlip, OCTREE_STEP_PROGRAM)
    .some((entry) => entry.includes("ready-topology-flip")));
});

test("reordering and unknown stages are deviations", () => {
  const swapped = ["surface-transport", "ready-topology-flip", "pressure-projection",
    "inactive-topology-candidate", "sparse-brick-world", "step-snapshot"];
  assert.ok(stepSequenceDeviations(swapped, OCTREE_STEP_PROGRAM).length > 0);
  assert.ok(stepSequenceDeviations([...FULL_SEQUENCE, "mystery"], OCTREE_STEP_PROGRAM)
    .some((entry) => entry.includes("mystery")));
});

test("the recorder validates then resets for the next step", () => {
  const recorder = new StepSequenceRecorder();
  for (const id of FULL_SEQUENCE) recorder.record(id);
  assert.deepEqual(recorder.finishStep(OCTREE_STEP_PROGRAM), []);
  // A second finish with nothing recorded reports every required stage.
  // `rigid-exchange` is scene-optional, and `step-snapshot` is optional
  // because its ring arms on consumption: a step whose predecessor's record is
  // still unread encodes no copies, and no other stage reads what it writes.
  // Pinned exactly rather than by count, so making a stage optional has to be
  // an explicit edit here.
  const empty = recorder.finishStep(OCTREE_STEP_PROGRAM);
  assert.deepEqual(empty, [
    "required stage ready-topology-flip was never executed",
    "required stage surface-transport was never executed",
    "required stage pressure-projection was never executed",
    "required stage inactive-topology-candidate was never executed",
    "required stage sparse-brick-world was never executed",
  ]);
});
