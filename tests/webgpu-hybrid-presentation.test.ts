import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES } from "../lib/webgpu-octree";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const uniformEulerianSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const octreeProjectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const waterPipelineSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");

test("octree smooth presentation keeps analytic solids and glass", () => {
  assert.match(rendererSource, /scene\.voxelDomain\.finestCellSize_m, Math\.min\(bodies\.length, 12\), gpuInfo\?\.quadtreeMaximumFluidScale \?\? 1,/,
    "smooth presentation must publish rigid-body count while retaining the live hierarchy scale");
  assert.match(rendererSource, /if \(sceneHasTerrain\(scene\) && scene\.terrain\)/,
    "terrain must remain part of the analytic scene for octree simulations");
  assert.match(rendererSource, /bodies\.slice\(0, 12\)\.forEach/,
    "rigid bodies must remain in the analytic scene for octree simulations");
  assert.match(waterPipelineSource, /if\s*\(\s*environmentIndex\(\)\s*!=\s*7\s*\)\s*\{/,
    "the tank glass path must be selected from scene semantics, not representation mode");
  assert.doesNotMatch(rendererSource, /voxelSceneActive|voxelScenePipeline|Compiling voxel scene materials/,
    "smooth presentation must not instantiate or encode sparse voxel cubes as production solids");
  assert.match(rendererSource, /readyGPUFluid\.initialSparseAuthorityReady[^]*globalFineWaterReady !== this\.globalFineWaterAttached/,
    "t=0 must attach global-fine geometry only after the complete initial sparse-authority fence");
  assert.doesNotMatch(rendererSource, /\(gpuInfo\?\.encodedSteps \?\? 0\) > 0/,
    "a completed physics step must not be used as a proxy for the warmed t=0 authority");
});

/**
 * The GLOBAL sparse frame is now the renderer's only consumer of a sparse
 * producer. It used to share the producer with a second attachment: the
 * expanded-record inspection overlay, which pulled `sparseVoxelRenderSource`
 * (48 bytes per resolved voxel, ~295 MB on the widened ocean scene) whenever a
 * non-smooth representation was selected, and drew additively over this frame
 * with its own depth target. Both the overlay and the producer getter that fed
 * it were removed, so what is pinned here is the single remaining attachment.
 */
test("GLOBAL consumes the sparse hierarchy through exactly one attachment", () => {
  assert.match(rendererSource, /sidecar\?\.sparseVoxelSceneSource\?\?solver\.sparseVoxelSceneSource/,
    "smooth production SVO consumes the solver hierarchy or its renderer-owned sidecar");
  assert.doesNotMatch(rendererSource, /sparseVoxelRenderSource|voxelInspectionSource|voxelDebugDepth|voxelRenderMode/,
    "no second capacity-sized inspection attachment may return to the frame");
  assert.match(rendererSource, /const drySceneReplacement = \(/);
  assert.match(uniformEulerianSource, /Initial sparse authority: \$\{descriptor\.label\}/);
  assert.match(uniformEulerianSource, /this\.octreeProjection\.encodeInitialSparseAuthorityPhase\(initialSparseScene, phase\)/);
  assert.deepEqual(OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES.map(({ id }) => id), [
    "cold-topology", "structured-authority", "surface-global-fine", "sparse-render-world",
  ], "t=0 must publish compact topology, direct structured velocity/boundary, global fine, and render authority in dependency order");
  assert.match(octreeProjectionSource, /case "cold-topology": this\.encodeColdBootstrapRebuild\(encoder\)/);
  assert.match(octreeProjectionSource,
    /case "structured-authority":[\s\S]*?this\.encode\([\s\S]*?"power-operator-only"/,
    "the structured checkpoint must publish the sole direct velocity/pressure authority");
  assert.match(octreeProjectionSource, /case "surface-global-fine": this\.encodeSurface\(encoder, 0\)/);
  assert.match(octreeProjectionSource,
    /case "sparse-render-world":[\s\S]*?this\.encodeSparseBrickWorld\(encoder\)[\s\S]*?this\.encodeInactiveTopologyCandidate\(encoder\)/,
    "the final t=0 checkpoint must publish render data before preparing the next inactive epoch");
  assert.doesNotMatch(rendererSource, /mode: "smooth", colorTarget/,
    "the renderer must never send smooth mode through the cube inspection pipeline");
});
