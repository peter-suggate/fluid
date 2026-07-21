import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const uniformEulerianSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const octreeProjectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const waterPipelineSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");

test("octree smooth presentation keeps analytic solids and glass", () => {
  assert.match(rendererSource, /scene\.nominalResolution\.length_m, Math\.min\(bodies\.length, 12\), gpuInfo\?\.quadtreeMaximumFluidScale \?\? 1,/,
    "smooth presentation must publish rigid-body count while retaining the live hierarchy scale");
  assert.match(rendererSource, /if \(sceneHasTerrain\(scene\) && scene\.terrain\)/,
    "terrain must remain part of the analytic scene for octree simulations");
  assert.match(rendererSource, /bodies\.slice\(0, 12\)\.forEach/,
    "rigid bodies must remain in the analytic scene for octree simulations");
  assert.match(waterPipelineSource, /if\s*\(\s*environmentIndex\(\)\s*!=\s*7\s*\)\s*\{/,
    "the tank glass path must be selected from scene semantics, not representation mode");
  assert.doesNotMatch(rendererSource, /voxelSceneActive|voxelScenePipeline|Compiling voxel scene materials/,
    "smooth presentation must not instantiate or encode sparse voxel cubes as production solids");
  assert.match(rendererSource, /readyGPUFluid\.initialSparseAuthorityReady[^]*adaptiveWaterReady !== this\.adaptiveWaterAttached/,
    "t=0 must attach compact pages only after the complete initial sparse-authority fence");
  assert.doesNotMatch(rendererSource, /\(gpuInfo\?\.encodedSteps \?\? 0\) > 0/,
    "a completed physics step must not be used as a proxy for the warmed t=0 authority");
});

test("raw voxel and brick-grid inspection retain the GPU sparse source", () => {
  assert.match(rendererSource, /voxelRenderMode !== "smooth" && this\.gpuFluid/);
  assert.match(rendererSource, /this\.voxelInspectionSource = requestedVoxelDebugGeneration >= 0 \? this\.gpuFluid\?\.sparseVoxelRenderSource : undefined/,
    "capacity-sized debug instance buffers attach only while inspection is visible");
  assert.match(rendererSource, /const sparseSceneSource=solver\.sparseVoxelSceneSource/,
    "smooth production SVO consumes the structural source without activating inspection records");
  assert.match(rendererSource, /if \(voxelRenderMode !== "smooth" && this\.voxelDebugDepth\)/);
  assert.match(rendererSource, /mode: voxelRenderMode/);
  assert.match(rendererSource, /colorLoadOp: "clear"/);
  assert.match(uniformEulerianSource, /Initial sparse authority: \$\{descriptor\.label\}/);
  assert.match(uniformEulerianSource, /this\.octreeProjection\.encodeInitialSparseAuthorityPhase\(initialSparseScene, phase\)/);
  assert.match(octreeProjectionSource,
    /encodeInitialSparseAuthorityPhase[\s\S]*encodeColdBootstrapRebuild\(encoder\)[\s\S]*this\.encode\(encoder[\s\S]*encodeSurface\(encoder, 0\)[\s\S]*encodeGlobalFineFaceBand\(encoder\)[\s\S]*encodeSparseBrickWorld\(encoder\)/,
    "t=0 must publish compact topology, coarse power phi, fine pages, transition face band, and indexed narrow-band authority in dependency order",
  );
  assert.doesNotMatch(rendererSource, /mode: "smooth", colorTarget/,
    "the renderer must never send smooth mode through the cube inspection pipeline");
});
