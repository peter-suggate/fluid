import assert from "node:assert/strict";
import test from "node:test";

import { resolveMethodValues } from "../lib/core/method-contract";
import { scenePresets } from "../lib/core/scenes";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";
import { createSparseCM12FrameControl } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { createSparseCM12PressureTopologyRepairLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair";
import { createSparseCM12FinalScalarPacketMaskLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-final-scalar-packet-masks";

test("Sparse CM12 exposes and normalizes the matched B4/P4 production profile", () => {
  const spec = adaptiveMassMethod.params.find((candidate) =>
    candidate.key === "brickFineResolution");
  assert.equal(spec?.kind, "select");
  if (spec?.kind !== "select") return;
  assert.deepEqual(spec.options.map(({ value }) => value), ["4", "8", "16"]);

  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    brickFineResolution: "4",
    presentationPageResolution: "16",
  });
  assert.equal(values.brickFineResolution, "4");
  assert.equal(values.presentationPageResolution, "4");
  assert.deepEqual({
    brickFineResolution: adaptiveMassSolverOptions(values).brickFineResolution,
    presentationPageResolution:
      adaptiveMassSolverOptions(values).presentationPageResolution,
  }, { brickFineResolution: 4, presentationPageResolution: 4 });
});

test("Sparse CM12 defaults production scenes to matched B8/P8", () => {
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
  assert.equal(values.brickFineResolution, "8");
  assert.equal(values.presentationPageResolution, "8");
  assert.deepEqual({
    brickFineResolution: adaptiveMassSolverOptions({}).brickFineResolution,
    presentationPageResolution: adaptiveMassSolverOptions({}).presentationPageResolution,
  }, { brickFineResolution: 8, presentationPageResolution: 8 });

  const productionScenes = scenePresets.filter(
    ({ methodProfile }) => methodProfile?.methodId === "adaptive-mass",
  );
  assert.ok(productionScenes.length > 0);
  for (const scene of productionScenes) {
    const sceneValues = resolveMethodValues(adaptiveMassMethod,
      scene.methodProfile!.quality, scene.methodProfile!.overrides);
    assert.equal(sceneValues.brickFineResolution, "8", scene.id);
    assert.equal(sceneValues.presentationPageResolution, "8", scene.id);
  }
});

test("Sparse CM12 production authority ABIs admit matched B4/P4", () => {
  const profile = { brickFineResolution: 4, presentationPageResolution: 4 } as const;
  const frame = createSparseCM12FrameControl({
    ...profile, cellWorkgroups: 1, rowWorkgroups: 1,
  });
  const pressureRepair = createSparseCM12PressureTopologyRepairLayout({
    ...profile, baseWords: frame.layout.totalWords, brickCapacity: 8,
  });
  const finalScalarMasks = createSparseCM12FinalScalarPacketMaskLayout({
    brickFineResolution: 4, packetCapacity: 64,
  });
  assert.ok(pressureRepair.totalWords > frame.layout.totalWords);
  assert.equal(frame.layout.brickFineResolution, 4);
  assert.equal(frame.layout.presentationPageResolution, 4);
  assert.equal(finalScalarMasks.maximumPacketsPerLeaf, 1);
});
