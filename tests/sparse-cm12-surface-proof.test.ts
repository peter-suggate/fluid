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
  }).forcedSurfaceResolutionForQA, 2);
  assert.equal(sparseCM12ActivityPolicy({
    forcedSurfaceResolutionForQA: 3,
  }).forcedSurfaceResolutionForQA, undefined);
});

test("coarse-first is the production default", () => {
  assert.equal(adaptiveMassMethod.params.some((param) =>
    param.key === "resolutionMode"), false,
  "fixed all-fine/all-coarse modes must not remain in the production UI");
  assert.equal("resolutionMode" in adaptiveMassSolverOptions({ resolutionMode: "all-fine" }),
    false, "stale fixed-mode state must be discarded at the production boundary");
  const selector = adaptiveMassMethod.params.find((param) =>
    param.key === "selectorMode");
  assert.equal(selector?.kind, "select");
  if (selector?.kind === "select") assert.equal(selector.default, "coarse-first");
  assert.equal(adaptiveMassMethod.presetFor("balanced").selectorMode, "coarse-first");

  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {});
  assert.equal(values.selectorMode, "coarse-first");
  assert.equal(adaptiveMassSolverOptions(values).activityPolicy?.activitySignals, true);
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
  assert.match(presentation,
    /SPARSE_CM12_PRESENTATION_COMPONENT_ISOLATION[\s\S]*FPP1 \$\{component\}/,
    "xctrace isolation must expose proof cost without changing the production pass");

  const proof = shader.slice(
    shader.indexOf("fn publishSparseCM12SurfaceRepresentabilityReceipts("),
    shader.indexOf("fn classifyPresentationBrick("),
  );
  assert.match(proof, /generationReceipt[\s\S]*==acceptedGeneration/);
  assert.match(proof, /topologyGeneration[\s\S]*==atomicLoad\(&activity\[12\]\)/);
  assert.match(proof, /surfaceProofAcceptedPhi/);
  assert.match(proof, /surfaceProofVirtualRestrictedDensity/);
  assert.match(proof, /surfaceProofRestrictionFactor/);
  assert.match(proof,
    /surfaceProofGenerationWord\(surfaceProofTarget\)[\s\S]*activity\[12\]/);

  assert.match(shader, /fn presentationLimitedSlope/);
  assert.match(shader,
    /accepted==BRICK_FINE_RESOLUTION\/2u[\s\S]*directSmoothedPresentationDensityAt/,
    "B4-to-B8 transfer must consume the same conservative field as presentation");

  const planner = shader.slice(
    shader.indexOf("fn planBrickResolution("),
    shader.indexOf("fn closePlannedResolution("),
  );
  assert.match(planner,
    /receiptFresh[\s\S]*surfaceProofGenerationWord\(nextSurfaceRung\)[\s\S]*activity\[12\]/);
  assert.match(planner,
    /receiptFresh&&proofEpochs[\s\S]*max\(p\.activityEpochs\.z,SURFACE_PROOF_SETTLE_EPOCHS\)/);
  assert.match(planner, /let interfaceVelocityFloor=select\(1u,velocityFloor/,
    "geometric proof must not override the moving-front transport floor");
  assert.match(planner,
    /acceptedSurfaceLease=current<BRICK_FINE_RESOLUTION[\s\S]*leasedExteriorSurface/,
    "every accepted coarse surface rung must survive seam-owner reclassification");
  assert.match(shader, /velocityThresholds:array<vec4f,2>/,
    "velocity floors must cover the complete B1-through-B16 ladder");
  assert.match(shader, /ACTIVITY_SURFACE_LEASE_MASK_WORD/,
    "surface retention must use a per-rung lease bitfield");
  assert.match(planner, /let thinRequiresFinest=thinFluid;/,
    "thin sheets must retain the ladder maximum");
});
