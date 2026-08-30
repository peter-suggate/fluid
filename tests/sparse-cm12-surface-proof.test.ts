import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { resolveMethodValues } from "../lib/core/method-contract";
import {
  adaptiveMassMethod,
  adaptiveMassSolverOptions,
} from "../lib/methods/adaptive-mass/method";
import {
  SPARSE_CM12_ACTIVITY_POLICY,
  sparseCM12ActivityPolicy,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");
const shader = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

test("surface coarsening policy is enabled, bounded, and keeps QA forcing private", () => {
  assert.equal(SPARSE_CM12_ACTIVITY_POLICY.surfaceCoarseningEnabled, true);
  assert.equal(SPARSE_CM12_ACTIVITY_POLICY.surfaceDisplacementToleranceCells, 1);
  assert.equal(SPARSE_CM12_ACTIVITY_POLICY.surfaceNormalToleranceDegrees, 30);

  const sanitized = sparseCM12ActivityPolicy({
    activitySignals: true,
    surfaceCoarseningEnabled: false,
    surfaceDisplacementToleranceCells: -4,
    surfaceNormalToleranceDegrees: 120,
    forcedSurfaceResolutionForQA: 4,
  });
  assert.equal(sanitized.activitySignals, true);
  assert.equal(sanitized.surfaceCoarseningEnabled, false);
  assert.equal(sanitized.surfaceDisplacementToleranceCells, 0);
  assert.equal(sanitized.surfaceNormalToleranceDegrees, 90);
  assert.equal(sanitized.forcedSurfaceResolutionForQA, 4);
  assert.equal(sparseCM12ActivityPolicy({
    forcedSurfaceResolutionForQA: 2,
  }).forcedSurfaceResolutionForQA, undefined);
});

test("activity plus accepted-output proof is the production default", () => {
  const selector = adaptiveMassMethod.params.find((param) =>
    param.key === "selectorMode");
  assert.equal(selector?.kind, "select");
  if (selector?.kind === "select") assert.equal(selector.default, "activity");
  assert.equal(adaptiveMassMethod.presetFor("balanced").selectorMode, "activity");

  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
  assert.equal(values.selectorMode, "activity");
  assert.equal(adaptiveMassSolverOptions(values).activityPolicy.activitySignals, true);
});

test("surface receipts are output-space, generation-stamped, and camera independent", () => {
  const presentation = resident.slice(
    resident.indexOf("private encodeFramePlanPresentation("),
    resident.indexOf("setRefinementRegionParameters("),
  );
  assert.match(presentation,
    /finalizeSparseCM12FramePlanPresentationExecution![\s\S]*publishSparseCM12SurfaceRepresentabilityReceipts![\s\S]*rejectSparseCM12FramePlanPresentationFaults!/);
  assert.match(presentation,
    /publishSparseCM12SurfaceRepresentabilityReceipts![\s\S]*dispatchWorkgroups\(bricks\)/);

  const proof = shader.slice(
    shader.indexOf("fn publishSparseCM12SurfaceRepresentabilityReceipts("),
    shader.indexOf("fn classifyPresentationBrick("),
  );
  assert.match(proof, /generationReceipt[\s\S]*==acceptedGeneration/);
  assert.match(proof, /topologyGeneration[\s\S]*==atomicLoad\(&activity\[12\]\)/);
  assert.match(proof, /surfaceProofAcceptedPhi/);
  assert.match(proof, /surfaceProofVirtualB4Density/);
  assert.match(proof, /activity\[output\+39u\].*BRICK_FINE_RESOLUTION\/2u/);
  assert.match(proof, /activity\[output\+40u\].*activity\[12\]/);

  assert.match(shader, /fn presentationLimitedSlope/);
  assert.match(shader,
    /accepted==BRICK_FINE_RESOLUTION\/2u[\s\S]*directSmoothedPresentationDensityAt/,
    "B4-to-B8 transfer must consume the same conservative field as presentation");

  const planner = shader.slice(
    shader.indexOf("fn planBrickResolution("),
    shader.indexOf("fn closePlannedResolution("),
  );
  assert.match(planner, /receiptFresh[\s\S]*activity\[output\+40u\].*activity\[12\]/);
  assert.match(planner,
    /receiptFresh&&proofEpochs[\s\S]*max\(p\.activityEpochs\.z,SURFACE_PROOF_SETTLE_EPOCHS\)/);
  assert.match(planner, /let interfaceVelocityFloor=select\(1u,velocityFloor/,
    "geometric proof must not override the moving-front transport floor");
});
