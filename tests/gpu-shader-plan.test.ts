import assert from "node:assert/strict";
import test from "node:test";
import { cloneScene, defaultScene } from "../lib/model";
import {
  planGPUShaderCapabilities,
  planGPUShaderTasks,
} from "../lib/gpu-shader-plan";

test("shader capabilities come from scene content rather than a scene ID", () => {
  const scene = cloneScene(defaultScene);
  scene.terrain = undefined;
  scene.rigidBodies = [];
  const dryTopology = planGPUShaderCapabilities(scene, {
    solver: "octree",
    fineInterface: true,
  });
  assert.equal(dryTopology.has("fluid-simulation", "adaptive-topology", "fine-interface"), true);
  assert.equal(dryTopology.has("solid-fields"), false);
  assert.equal(dryTopology.has("diagnostic-overlays"), false);

  scene.rigidBodies = [{ ...cloneScene(defaultScene).rigidBodies[0]! }];
  const coupled = planGPUShaderCapabilities(scene, { solver: "octree", fineInterface: true });
  assert.equal(coupled.has("solid-fields"), true);
});

test("shader task manifests omit unreachable programs before progress registration", async () => {
  const scene = cloneScene(defaultScene);
  scene.terrain = undefined;
  scene.rigidBodies = [];
  const capabilities = planGPUShaderCapabilities(scene, { solver: "octree" });
  const compiled: string[] = [];
  const tasks = planGPUShaderTasks(capabilities, [
    { id: "core", label: "Core", requires: ["adaptive-topology"], compile: () => { compiled.push("core"); } },
    { id: "solids", label: "Solids", requires: ["solid-fields"], compile: () => { compiled.push("solids"); } },
    { id: "overlay", label: "Overlay", requires: ["diagnostic-overlays"], compile: () => { compiled.push("overlay"); } },
  ]);
  assert.deepEqual(tasks.map(({ id }) => id), ["core"]);
  await tasks[0]!.run(new AbortController().signal);
  assert.deepEqual(compiled, ["core"]);
});

test("shader task manifests reject duplicate identities", () => {
  const capabilities = planGPUShaderCapabilities(defaultScene, { solver: "uniform" });
  assert.throws(() => planGPUShaderTasks(capabilities, [
    { id: "same", label: "One", compile() {} },
    { id: "same", label: "Two", compile() {} },
  ]), /Duplicate GPU shader task/);
});
