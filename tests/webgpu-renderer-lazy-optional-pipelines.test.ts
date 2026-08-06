import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { optionalRendererPipelineRequests } from "../lib/webgpu-renderer";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const drySceneSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");

test("applicable GLOBAL startup requests the sparse renderer", () => {
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "off", position: 0.5 }, false, true,
  ), ["svo-dry-scene"]);

  const initializeStart = rendererSource.indexOf("private async initializeInternal(): Promise<void>");
  const recoveryStart = rendererSource.indexOf("private scheduleDeviceRecovery", initializeStart);
  const initializeSource = rendererSource.slice(initializeStart, recoveryStart);
  assert.match(initializeSource, /new RasterWaterPipeline/,
    "authoritative water presentation remains part of startup");
  for (const optionalConstructor of [
    "new GridOverlayPipeline", "new OctreeTechniqueOverlayPipeline",
    "new OctreeTechniqueAuditOverlayPipeline",
    "new SparseVoxelDrySceneRenderer", "new SecondaryParticleRenderPipeline",
  ]) assert.doesNotMatch(initializeSource, new RegExp(optionalConstructor), `${optionalConstructor} must be deferred`);
});

test("full scenes request their authored SVO shell and fluid-only scenes can skip it", () => {
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, false, false,
  ), ["svo-dry-scene"]);
  assert.match(rendererSource, /const sparsePresentationRequired = presentationMode === "full-scene"/,
    "the scene definition must own whether the dry world is part of presentation");
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, false, false, false, false, false, false,
  ), [], "a fluid-only frame requests no sparse presentation pipeline");
  assert.match(rendererSource, /solver\.sparseVoxelSceneSource[\s\S]*WebGPULiveSvoScene\.create[\s\S]*return \{solver,sidecar\}/,
    "a renderer-owned source must fill the method capability gap");
  assert.match(rendererSource, /presentationMode === "fluid-only" \|\| solver\.sparseVoxelSceneSource/,
    "fluid-only initialization must not construct a renderer-owned sparse sidecar");
  assert.match(rendererSource, /if \(sparsePresentationRequired\) requested\.push\("svo-dry-scene"\)/,
    "the authored shell must start the sparse presentation task");
});

test("each optional pipeline has an explicit first-use condition", () => {
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "z", position: 0.5, mode: "structure" }, false, false,
  ), ["grid-overlay", "svo-dry-scene"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    { axis: "volume", position: 0.5, mode: "power-cells" }, false, false,
  ), ["technique-overlay", "technique-audit-overlay", "svo-dry-scene"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, false, false,
  ), ["svo-dry-scene"]);
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, true, true,
  ), ["svo-dry-scene", "secondary-particles"]);
  // The pixel trace and the cell gather are the only diagnostics left that add
  // an optional pipeline; the expanded-record inspection overlay and its
  // "voxel-debug" request were removed with the renderer that served them.
  assert.deepEqual(optionalRendererPipelineRequests(
    undefined, false, false, true,
  ), ["svo-dry-scene", "decoration-overlay"]);
  assert.ok(!optionalRendererPipelineRequests(undefined, false, false, true).includes(
    "voxel-debug" as never), "the inspection overlay pipeline no longer exists");
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
  assert.match(rendererSource, /new SparseVoxelDrySceneRenderer\([^]*traversal, "off", "split",[^]*defaultThresholdPixels[^]*coherence, true, true, true\)/,
    "the production renderer must retain the measured split/coherence and analytic-raster capability");
  assert.match(rendererSource, /const primaryCoherenceKey = activeSvoTuning\.stationaryPrimaryReuseEnabled[^]*!sceneRuntime\.fluidSolver \|\| !this\.simulationRunning[^]*presentationCoherenceKey[^]*sceneEpoch/,
    "the opt-in must still restrict complete caller-owned keys to live scenes and paused solvers");
  assert.match(rendererSource, /encode\(replacementEncoder, target, primaryCoherenceKey, tracePhase\)/,
    "the safe key must reach the renderer cache; running fluid scenes pass undefined");
  assert.match(drySceneSource, /this\.rayCoherenceMode, useSplit && usePrepass, primaryFrameKey/,
    "coherence must fail closed until a reduced prepass makes primary output parity-invariant");
  assert.doesNotMatch(drySceneSource, /const relightSplit|lightingMode/,
    "retired lighting-mode selection must not remain in the GLOBAL renderer");
  assert.match(drySceneSource, /if \(this\.presentationBundleStatus\.state !== "ready"\) return false/);
  assert.match(drySceneSource, /if \(this\.coneScale !== 1 && !usePrepass\)[^]*return false[^]*if \(splitRequested && !this\.splitDiagnosticsActive && !useSplit\)[^]*return false/,
    "missing requested cone or split resources must reject rather than execute inline");
});
