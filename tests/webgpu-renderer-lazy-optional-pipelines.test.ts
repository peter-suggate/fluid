import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { optionalRendererPipelineRequests } from "../lib/webgpu-renderer";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const drySceneSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");

test("GLOBAL startup always requests the sparse renderer", () => {
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "off", position: 0.5 }, "smooth", false, true,
  ), ["svo-dry-scene"]);

  const initializeStart = rendererSource.indexOf("private async initializeInternal(): Promise<void>");
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
    { axis: "z", position: 0.5, mode: "structure" }, "smooth", false, false,
  ), ["grid-overlay", "svo-dry-scene"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "volume", position: 0.5, mode: "power-cells" }, "smooth", false, false,
  ), ["technique-overlay", "technique-audit-overlay", "svo-dry-scene"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, "raw-voxels", false, false,
  ), ["voxel-debug", "svo-dry-scene"], "structural inspection overlays the GLOBAL renderer");
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, "smooth", false, false,
  ), ["svo-dry-scene"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, "smooth", true, true,
  ), ["svo-dry-scene", "secondary-particles"]);
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

test("dry SVO startup compiles only the GLOBAL presentation pipeline", () => {
  assert.match(drySceneSource, /initialize\(progress\?: \(label: string, completed: number, total: number\) => void\)/);
  assert.match(drySceneSource, /progress\?\.\("Compiling sparse dry-scene pipeline", 0, 1\)/);
  assert.match(drySceneSource, /progress\?\.\("Sparse presentation pipeline compiled", 1, 1\)/);
  assert.doesNotMatch(drySceneSource, /SparseVoxelTemporalAccumulator|Compiling sparse temporal accumulation|svo-temporal/);
  assert.match(rendererSource, /pipeline\.initialize\(\(label, completed\) => this\.reportSvoPipelineProgress\(label, completed\)\)/,
    "the lazy optional pipeline must forward compilation stages to the viewport status flow");
  assert.match(rendererSource, /label: "Sparse garden renderer attached"[^]*completed: 3, total: 4/);
  assert.match(rendererSource, /label: "Submitting first sparse garden frame"[^]*completed: 3, total: 4/);
});

test("production exposes exact static-primary coherence behind a default-off safe-scene gate", () => {
  assert.match(rendererSource, /new SparseVoxelDrySceneRenderer\([^]*"canonical-parametric"[^]*"split", 0, "static-primary", true, true, true\)/,
    "the production renderer must retain the measured split/coherence and analytic-raster capability");
  assert.match(rendererSource, /const primaryCoherenceKey = activeSvoTuning\.stationaryPrimaryReuseEnabled[^]*!sceneRuntime\.fluidSolver \|\| !this\.simulationRunning[^]*presentationCoherenceKey[^]*sceneEpoch/,
    "the opt-in must still restrict complete caller-owned keys to static worlds and paused solvers");
  assert.match(rendererSource, /encode\(replacementEncoder, target, primaryCoherenceKey, tracePhase\)/,
    "the safe key must reach the renderer cache; running fluid scenes pass undefined");
  assert.match(drySceneSource, /this\.rayCoherenceMode, useSplit && usePrepass, primaryFrameKey/,
    "coherence must fail closed until a reduced prepass makes primary output parity-invariant");
  assert.doesNotMatch(drySceneSource, /const relightSplit|lightingMode/,
    "retired lighting-mode selection must not remain in the GLOBAL renderer");
  assert.match(drySceneSource, /this\.shadingPath === "auto-relight" && relight && this\.coneScale !== 1[^]*ensureConeLightingPrepass/,
    "selecting relight must asynchronously compile its split variant while the inline path remains available");
});
