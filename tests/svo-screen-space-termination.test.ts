import assert from "node:assert/strict";
import test from "node:test";
import {
  compareSvoScreenSpaceImages,
  createSvoScreenSpaceTraversalWGSL,
  effectiveSvoScreenSpaceThresholdPixels,
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

test("authored threshold scales with render-target height", () => {
  assert.equal(effectiveSvoScreenSpaceThresholdPixels(3, 460), 3);
  assert.equal(effectiveSvoScreenSpaceThresholdPixels(3, 920), 6);
  assert.equal(effectiveSvoScreenSpaceThresholdPixels(3, 1784), 1784 * 3 / 460);
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
  assert.throws(() => createSvoDrySceneFragmentWGSL(1, "canonical", "off", "split", 1), /requires canonical inline or raster-primary split/);
  const raster = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "off", "split", 1,
    false, true, true, true);
  assert.match(raster, /fn dryPrimaryBoundsSubPixel/);
  // The threshold is a uniform lane now, not a shader constant: the panel's
  // slider and an interleaved A/B both need it to move without a pipeline
  // rebuild. The constructor argument still decides whether any of this is
  // compiled, which is what keeps the threshold-zero build the exact reference.
  assert.match(raster, /dry\.lod\.x\*uniforms\.viewport\.y\/460/);
  assert.doesNotMatch(raster, /effectiveThresholdPixels=1\*/);
  assert.match(raster, /dryPrimaryVoxelProxyHit/);
  assert.match(raster, /dryPrimaryPrimitiveProxyHit/);
  // The per-cell sub-pixel resolve belonged to the arm that still marched
  // records, and that arm no longer exists: the primary returns at the first
  // solid cell and shades the normal baked into it, so the resolve chose between
  // two identical answers at the cost of a footprint evaluation per cell.
  // Intermediate detail is the aggregate stride's job instead.
  assert.doesNotMatch(raster, /if\(dryPrimaryBoundsSubPixel\(cellBounds\)\)/);
  assert.match(raster, /let proxySpan=vec2f\(span\.x,min\(span\.y,limit\)\)/);
  assert.match(raster, /@fragment fn svoBrickLodResolveFragment/);
  assert.match(raster, /@fragment fn svoBrickExactResolveFragment/);
  assert.match(raster, /@fragment fn svoScenePrimitiveLodResolveFragment/);
  assert.match(raster, /@compute @workgroup_size\(1\) fn svoScenePrimitiveTieredComputeArgs/);
  assert.match(raster, /@compute @workgroup_size\(64\) fn svoScenePrimitiveTieredComputeResolve/);
  assert.match(raster, /dryTieredResolveQueue\.pixels\[queueIndex\]=pixel/);
  assert.match(raster, /textureStore\(dryTierPackedSurfaceWrite/);
  assert.match(raster, /DRY_GBUFFER_FIELD_RESIDENT_CELL_PROXY:u32=14u/);
  assert.match(raster, /svoPrimitiveOwnerId\(record\)/,
    "the analytic tier can suppress the redundant exact upgrade for the resident-cell owner");
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
