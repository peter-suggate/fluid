import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { optionalRendererPipelineRequests } from "../lib/webgpu-renderer";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");

test("paused raster-water startup requests no optional renderer pipelines", () => {
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "off", position: 0.5 }, "smooth", "raster", false, true,
  ), []);

  const initializeStart = rendererSource.indexOf("async initialize(): Promise<void>");
  const recoveryStart = rendererSource.indexOf("private scheduleDeviceRecovery", initializeStart);
  const initializeSource = rendererSource.slice(initializeStart, recoveryStart);
  assert.match(initializeSource, /new RasterWaterPipeline/,
    "authoritative water presentation remains part of startup");
  for (const optionalConstructor of [
    "new GridOverlayPipeline", "new OctreeTechniqueOverlayPipeline",
    "new OctreeTechniqueAuditOverlayPipeline", "new SparseVoxelDebugRenderer",
    "new SparseVoxelDrySceneRenderer", "new SecondaryParticleRenderPipeline",
  ]) assert.doesNotMatch(initializeSource, new RegExp(optionalConstructor), `${optionalConstructor} must be deferred`);
});

test("each optional pipeline has an explicit first-use condition", () => {
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "z", position: 0.5, mode: "structure" }, "smooth", "raster", false, false,
  ), ["grid-overlay"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "volume", position: 0.5, mode: "power-cells" }, "smooth", "raster", false, false,
  ), ["technique-overlay", "technique-audit-overlay"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, "raw-voxels", "svo", false, false,
  ), ["voxel-debug"], "inspection wins over the production SVO renderer");
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, "smooth", "svo", false, false,
  ), ["svo-dry-scene"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, "smooth", "raster", true, true,
  ), ["secondary-particles"]);
});

test("first-use compilation is single-flight and fails closed per device", () => {
  const helperStart = rendererSource.indexOf("private ensureOptionalPipeline<T>");
  const dispatcherStart = rendererSource.indexOf("private ensureRequestedOptionalPipelines", helperStart);
  const helper = rendererSource.slice(helperStart, dispatcherStart);
  assert.match(helper, /this\.optionalPipelineTasks\.has\(key\)/,
    "repeated frames cannot start duplicate compilation");
  assert.match(helper, /this\.failedOptionalPipelines\.has\(key\)/,
    "a rejected compile cannot hammer the driver every frame");
  assert.match(helper, /this\.device !== device/,
    "a pipeline compiled for a retired device cannot publish");
  assert.match(helper, /destroy\(candidate\)/,
    "superseded or failed candidates are cleaned up");
  assert.match(helper, /this\.pausedPresentationRevision \+= 1/,
    "completion asks a paused scene for exactly another presentation opportunity");
});
