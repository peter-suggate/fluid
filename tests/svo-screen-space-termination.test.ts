import assert from "node:assert/strict";
import test from "node:test";
import {
  compareSvoScreenSpaceImages,
  createSvoScreenSpaceTraversalWGSL,
  projectedSvoNodeFootprintPixels,
  shouldTerminateSvoNodeScreenSpace,
  SVO_SCREEN_SPACE_TERMINATION_CONTRACT,
  svoScreenSpaceTerminationWGSL,
} from "../lib/svo-screen-space-termination";
import { webgpuSvoTraversalWGSL } from "../lib/webgpu-svo-traversal";
import { createSvoDrySceneFragmentWGSL, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

const unitNode = { minimum: [-0.5, -0.5, -0.5], maximum: [0.5, 0.5, 0.5] } as const;

test("screen-space termination is exactly disabled at threshold zero", () => {
  assert.equal(shouldTerminateSvoNodeScreenSpace(unitNode, [0, 0, -1_000], 21, {
    thresholdPixels: 0,
    viewportHeightPixels: 720,
  }), false);
  assert.equal(SVO_SCREEN_SPACE_TERMINATION_CONTRACT.exactShadows, true);
  assert.equal(SVO_SCREEN_SPACE_TERMINATION_CONTRACT.hasRepresentativeMaterial, false);
  assert.equal(SVO_SCREEN_SPACE_TERMINATION_CONTRACT.hasRepresentativeNormal, false);
});

test("conservative enclosing-sphere footprint shrinks with distance and grows with resolution", () => {
  const near = projectedSvoNodeFootprintPixels(unitNode, [0, 0, -10], { viewportHeightPixels: 720 });
  const far = projectedSvoNodeFootprintPixels(unitNode, [0, 0, -20], { viewportHeightPixels: 720 });
  const highResolution = projectedSvoNodeFootprintPixels(unitNode, [0, 0, -20], { viewportHeightPixels: 1440 });
  assert.ok(near > far);
  assert.ok(Math.abs(highResolution - 2 * far) < 1e-10);
  assert.equal(projectedSvoNodeFootprintPixels(unitNode, [0, 0, 0], { viewportHeightPixels: 720 }), Infinity);
});

test("screen-space threshold and minimum level jointly gate a coarse proxy", () => {
  const footprint = projectedSvoNodeFootprintPixels(unitNode, [0, 0, -1_000], { viewportHeightPixels: 720 });
  assert.ok(footprint < 1);
  assert.equal(shouldTerminateSvoNodeScreenSpace(unitNode, [0, 0, -1_000], 7, {
    thresholdPixels: 1,
    viewportHeightPixels: 720,
    minimumLevel: 8,
  }), false);
  assert.equal(shouldTerminateSvoNodeScreenSpace(unitNode, [0, 0, -1_000], 8, {
    thresholdPixels: 1,
    viewportHeightPixels: 720,
    minimumLevel: 8,
  }), true);
});

test("WGSL helper uses the same conservative sphere projection and explicit opt-in", () => {
  assert.match(svoScreenSpaceTerminationWGSL, /distanceSquared <= radiusSquared/);
  assert.match(svoScreenSpaceTerminationWGSL, /thresholdPixels > 0\.0 && level >= minimumLevel/);
  assert.match(svoScreenSpaceTerminationWGSL, /sqrt\(distanceSquared - radiusSquared\)/);
});

test("derived traversal leaves the exact canonical entry point untouched", () => {
  const derived = createSvoScreenSpaceTraversalWGSL(webgpuSvoTraversalWGSL);
  assert.match(derived, /fn svoTraversalContinuationNextScreenSpace\(/);
  assert.match(derived, /node\.links\.z == SVO_INVALID/);
  assert.match(derived, /drySvoShouldTerminateNodeScreenSpace\(proxyBounds, node\.address\.z\)/);
  assert.match(derived, /leaf-level proxy also skips the 8\^3 payload DDA/);
  assert.ok(derived.indexOf("leaf.topology.x != current.nodeIndex") < derived.indexOf("leaf-level proxy"),
    "leaf validation must precede approximation");
  assert.equal((webgpuSvoTraversalWGSL.match(/fn svoTraversalContinuationNext\(/g) ?? []).length, 1);
  assert.doesNotMatch(webgpuSvoTraversalWGSL, /SVO_STATUS_SCREEN_SPACE_PROXY/);
});

test("dry-scene proxy is compile-time opt-in and primary-only", () => {
  assert.equal(createSvoDrySceneFragmentWGSL(), svoDrySceneShader);
  const diagnostic = createSvoDrySceneFragmentWGSL(1, "canonical", "off", "inline", 1);
  assert.match(diagnostic, /svoTraversalContinuationNextScreenSpace/);
  assert.match(diagnostic, /dryTraversalCursorNextPrimary/);
  assert.match(diagnostic, /DRY_GBUFFER_FIELD_SCREEN_SPACE_PROXY/);
  assert.match(diagnostic, /fn dryTraversalCursorNext\([^]*svoTraversalContinuationNext\(/,
    "exact traversal entry remains available for shadow rays");
  assert.throws(() => createSvoDrySceneFragmentWGSL(1, "wide", "off", "inline", 1), /requires canonical inline/);
  assert.throws(() => createSvoDrySceneFragmentWGSL(1, "canonical", "off", "split", 1), /requires canonical inline/);
});

test("image comparison reports changed luminance and silhouette directions", () => {
  const reference = new Float32Array([
    1, 1, 1, 2,
    0, 0, 0, 0,
    0.5, 0.5, 0.5, 3,
  ]);
  const candidate = new Float32Array([
    0.5, 0.5, 0.5, 2,
    0.25, 0.25, 0.25, 4,
    0.5, 0.5, 0.5, 0,
  ]);
  const comparison = compareSvoScreenSpaceImages(reference, candidate, { width: 3, height: 1 });
  assert.equal(comparison.totalPixels, 3);
  assert.equal(comparison.changedPixels, 3);
  assert.equal(comparison.silhouetteDisagreementPixels, 2);
  assert.equal(comparison.silhouetteFalsePositivePixels, 1);
  assert.equal(comparison.silhouetteFalseNegativePixels, 1);
  assert.ok(comparison.depthSilhouetteDisagreementPixels >= 0);
  assert.equal(comparison.absoluteLuminanceError.maximum, 0.5);
  assert.ok(comparison.relativeLuminanceError.p95 > 0);
});

test("screen-space option validation rejects ambiguous thresholds", () => {
  assert.throws(() => shouldTerminateSvoNodeScreenSpace(unitNode, [0, 0, -10], 1, {
    thresholdPixels: -1,
    viewportHeightPixels: 720,
  }), /non-negative/);
  assert.throws(() => projectedSvoNodeFootprintPixels(unitNode, [0, 0, -10], {
    viewportHeightPixels: 0,
  }), /viewport height/);
});
