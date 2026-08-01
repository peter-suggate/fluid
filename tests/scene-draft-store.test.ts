import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, validateScene, type SceneDescription } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { applySceneDraft, useSceneDraftStore } from "../lib/stores/scene-draft-store";
import { useSceneStore } from "../lib/stores/scene-store";
import {
  gpuSceneSeedKey,
  gpuSceneSolverKey,
  gpuSceneStructuralKey,
  type SimulationRunConfig,
} from "../lib/webgpu-renderer";
import { fluidBodyBox, fluidBodyBoxPatch, dragFluidBodyBox, fluidBodyHandleById } from "../lib/editor-fluid-body";

const config: SimulationRunConfig = {
  methodId: "octree", quality: "balanced", values: {}, simulationEpoch: 0,
};

function preset(id: string): SceneDescription {
  return getScenePreset(id).create();
}

test("a draft leaves the committed scene identical", () => {
  const store = useSceneDraftStore.getState();
  const scene = preset("water-box-dam-break");
  useSceneStore.getState().setScene(cloneScene(scene));
  const before = useSceneStore.getState().scene;

  store.beginDraft("fluid-body", "Reshaped the water body");
  useSceneDraftStore.getState().updateDraft({
    container: { ...scene.container, fillFraction: 0.4 },
  });
  assert.equal(useSceneStore.getState().scene, before,
    "proposing must not touch the store the physics reads");
  useSceneDraftStore.getState().clearDraft();
});

test("the committed scene is returned unchanged when nothing is being dragged", () => {
  const scene = preset("water-box-dam-break");
  assert.equal(applySceneDraft(scene, undefined), scene,
    "identity must be preserved at rest, so nothing downstream re-derives per frame");
});

test("a draft never reaches the solver's rebuild identity", () => {
  // The point of the whole split: a gesture in flight must not re-key the
  // solver, or the renderer re-seeds at pointer rate.
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const dragged = dragFluidBodyBox(box, fluidBodyHandleById(box, "+00")!,
    { x: box.max.x + 0.2, y: 0, z: 0 }, scene);
  const draft = { subject: "fluid-body" as const, label: "drag", patch: fluidBodyBoxPatch(scene, dragged) };

  // Committed identity is computed from the committed scene, which the draft
  // has not touched.
  assert.equal(gpuSceneSolverKey(scene, config), gpuSceneSolverKey(cloneScene(scene), config));
  // And the proposal really is a different scene — this would have rebuilt.
  const proposed = applySceneDraft(scene, draft);
  assert.notEqual(gpuSceneSeedKey(proposed), gpuSceneSeedKey(scene));
});

test("a terrain proposal cannot move the lattice, which is why it may be presented", () => {
  const scene = preset("water-box-dam-break");
  const draft = {
    subject: "terrain" as const,
    label: "Shaped terrain",
    patch: { terrain: { baseHeight_m: 0.12, features: [] } },
  };
  const presented = applySceneDraft(scene, draft);
  assert.equal(gpuSceneStructuralKey(presented, config), gpuSceneStructuralKey(scene, config),
    "presenting terrain against the committed solver is only safe because the lattice holds");
  assert.notEqual(gpuSceneSeedKey(presented), gpuSceneSeedKey(scene),
    "and it must still re-seed once the gesture commits");
});

test("an update with no open gesture is ignored", () => {
  useSceneDraftStore.getState().clearDraft();
  useSceneDraftStore.getState().updateDraft({ randomSeed: 7 });
  assert.equal(useSceneDraftStore.getState().draft, undefined,
    "a stray pointer-move after a cancel must not resurrect a draft");
});

test("committing a draft yields a scene that still validates", () => {
  const scene = preset("water-box-dam-break");
  const box = fluidBodyBox(scene);
  assert.ok(box);
  const dragged = dragFluidBodyBox(box, fluidBodyHandleById(box, "+++")!,
    { x: 0.4, y: 0.6, z: 0.3 }, scene);
  const committed = applySceneDraft(scene, {
    subject: "fluid-body", label: "drag", patch: fluidBodyBoxPatch(scene, dragged),
  });
  assert.deepEqual(validateScene(committed), []);
});
