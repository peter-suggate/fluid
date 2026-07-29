import assert from "node:assert/strict";
import test from "node:test";
import type { SceneDescription } from "../lib/model";
import { collectSceneEvidence, type SceneEvidenceCollectorRegistry }
  from "../tools/scene-evidence-collector-runtime";
import { sceneEvidenceCollectorRegistry } from "../tools/scene-evidence-collector-implementations";

const scene = {} as SceneDescription;
const context = {
  scene,
  method: "octree" as const,
  grid: [1, 1, 1] as const,
  time_s: 0.1,
  volumeField: new Float32Array([1]),
};

test("checkpoint collector dispatch is declaration-driven and publishes capabilities", () => {
  const registry = {
    ...sceneEvidenceCollectorRegistry,
    "free-fall-contact-attribution": {
      id: "free-fall-contact-attribution",
      phase: "checkpoint",
      collect: () => ({ sample: true }),
    },
  } satisfies SceneEvidenceCollectorRegistry;
  const result = collectSceneEvidence(registry, [{
    id: "free-fall-contact-attribution",
    phase: "checkpoint",
    methods: ["octree"],
    provides: ["contact attribution"],
  }], "checkpoint", context);
  assert.deepEqual(result.values, {
    "free-fall-contact-attribution": { sample: true },
  });
  assert.deepEqual(result.available, ["contact attribution"]);
});

test("checkpoint collectors skip unselected methods and fail closed on missing evidence", () => {
  const registry = {
    ...sceneEvidenceCollectorRegistry,
    "free-fall-contact-attribution": {
      id: "free-fall-contact-attribution",
      phase: "checkpoint",
      collect: () => undefined,
    },
  } satisfies SceneEvidenceCollectorRegistry;
  assert.deepEqual(collectSceneEvidence(registry, [{
    id: "free-fall-contact-attribution", phase: "checkpoint", methods: ["uniform"], provides: ["contact attribution"],
  }], "checkpoint", context), { values: {}, available: [] });
  assert.throws(() => collectSceneEvidence(registry, [{
    id: "free-fall-contact-attribution", phase: "checkpoint", provides: ["contact attribution"],
  }], "checkpoint", context), /returned no evidence/);
});
