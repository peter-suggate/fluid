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
  for (const stage of [
    "Build sparse presentation shader sources",
    "Validate sparse presentation shader modules",
    "Compile sparse primary visibility pipeline",
    "Compile sparse brick culling programs",
    "Compile split visibility and lighting programs",
    "Compile raster glass and rigid discovery programs",
    "Compile sparse cone fan-out programs",
    "Finalize sparse presentation resources",
    "Attach sparse renderer",
    "Submit first sparse frame",
  ]) assert.match(drySceneSource, new RegExp(stage));
  assert.match(drySceneSource, /report\(0\)[\s\S]*report\(1\)[\s\S]*report\(2\)[\s\S]*report\(3\)[\s\S]*report\(4\)[\s\S]*trackFamily\(SVO_PRESENTATION_STARTUP_STAGES\[4\][\s\S]*trackFamily\(SVO_PRESENTATION_STARTUP_STAGES\[5\][\s\S]*trackFamily\(SVO_PRESENTATION_STARTUP_STAGES\[6\][\s\S]*report\(7\)/,
    "each opaque browser compile family must have a truthful boundary in the owning plugin");
  assert.match(drySceneSource, /await Promise\.all\(\[[\s\S]*trackFamily\(SVO_PRESENTATION_STARTUP_STAGES\[4\][\s\S]*trackFamily\(SVO_PRESENTATION_STARTUP_STAGES\[5\][\s\S]*trackFamily\(SVO_PRESENTATION_STARTUP_STAGES\[6\]/,
    "independent sparse compile families must stay concurrent while reporting individual completion");
  assert.doesNotMatch(drySceneSource, /SparseVoxelTemporalAccumulator|Compiling sparse temporal accumulation|svo-temporal/);
  assert.match(rendererSource, /pipeline\.initialize\(\(label, completed, total\) => this\.reportSvoPipelineProgress\(label, completed, total\)\)/,
    "the lazy optional pipeline must forward compilation stages to the viewport status flow");
  assert.match(rendererSource, /label: SVO_PRESENTATION_STARTUP_STAGES\[8\][^]*completed: 8, total: SVO_PRESENTATION_STARTUP_STAGES\.length/);
  assert.match(rendererSource, /label: SVO_PRESENTATION_STARTUP_STAGES\[9\][^]*completed: 9, total: SVO_PRESENTATION_STARTUP_STAGES\.length/);
});

test("production rasterizes requested primary visibility or fails closed", () => {
  assert.match(rendererSource, /this\.requestedPrimaryTraversal === "raster"[^]*maxColorAttachmentBytesPerSample < FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE[^]*throw new RangeError\([^]*Requested SVO raster primary needs/,
    "an unsupported requested raster primary must enter the retained optional-pipeline failure channel");
  assert.match(rendererSource, /const traversal = this\.requestedPrimaryTraversal === "raster"\s*\?\s*"raster-primary" as const : "canonical-parametric" as const/);
  assert.doesNotMatch(rendererSource, /maxColorAttachmentBytesPerSample >= FLUID_RASTER_PRIMARY_COLOR_BYTES_PER_SAMPLE\s*\?\s*"raster-primary"/,
    "device limits must never silently select a different primary traversal");
  // Reuse is worth 28.5 ms of a 49.6 ms traced frame and 7.9 ms of a 29.0 ms
  // rastered one, where the impostor pass blocks it outright. Deriving the mode
  // keeps the raster path from advertising a cache that can never fill.
  assert.match(rendererSource, /const coherence = traversal === "raster-primary" \? "off" as const : "static-primary" as const/,
    "stationary primary reuse must follow the traversal, not be requested unconditionally");
  assert.match(rendererSource, /new SparseVoxelDrySceneRenderer\([^]*traversal, "off", "split", 0, coherence, true, true, true\)/,
    "the production renderer must retain the measured split/coherence and analytic-raster capability");
  assert.match(rendererSource, /const primaryCoherenceKey = activeSvoTuning\.stationaryPrimaryReuseEnabled[^]*!sceneRuntime\.fluidSolver \|\| !this\.simulationRunning[^]*presentationCoherenceKey[^]*sceneEpoch/,
    "the opt-in must still restrict complete caller-owned keys to live scenes and paused solvers");
  assert.match(rendererSource, /encode\(replacementEncoder, target, primaryCoherenceKey, tracePhase\)/,
    "the safe key must reach the renderer cache; running fluid scenes pass undefined");
  assert.match(drySceneSource, /this\.rayCoherenceMode, useSplit && usePrepass, primaryFrameKey/,
    "coherence must fail closed until a reduced prepass makes primary output parity-invariant");
  assert.doesNotMatch(drySceneSource, /const relightSplit|lightingMode/,
    "retired lighting-mode selection must not remain in the GLOBAL renderer");
  assert.match(drySceneSource, /this\.shadingPath === "auto-relight" && relight && this\.coneScale !== 1[^]*ensureConeLightingPrepass/,
    "selecting relight must asynchronously compile its requested split variant");
  assert.match(drySceneSource, /if \(this\.presentationBundleStatus\.state !== "ready"\) return false/);
  assert.match(drySceneSource, /if \(this\.coneScale !== 1 && !usePrepass\)[^]*return false[^]*if \(splitRequested && !this\.splitDiagnosticsActive && !useSplit\)[^]*return false/,
    "missing requested cone or split resources must reject rather than execute inline");
});
