import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const transportSource = readFileSync(new URL("../components/TransportBar.tsx", import.meta.url), "utf8");
const controllerSource = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
const solverSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const viewportSource = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
const safeModeHookSource = readFileSync(new URL("../lib/use-safe-browser-gpu-bringup.ts", import.meta.url), "utf8");
const methodTypesSource = readFileSync(new URL("../lib/methods/types.ts", import.meta.url), "utf8");

test("WebGPU transport stays locked until the fenced t=0 authority is ready", () => {
  assert.match(transportSource, /initialSceneReady = methodId !== "octree" \|\| \(gpuInfo\?\.initialSparseAuthorityReady === true[\s\S]*gpuInfo\?\.initialRasterSurfaceReady === true\)/);
  assert.match(transportSource, /transportLocked = staticRenderScene \|\| \(webgpu && \(gpuStatus\.state !== "ready" \|\| !initialSceneReady\)\)/);
  assert.match(transportSource, /disabled=\{browserPolicyPending \|\| transportLocked \|\| safeStepLocked\}[^]*simulation\.singleStep\(\)/);
  assert.match(controllerSource, /gpuInfo\?\.initialSparseAuthorityReady === true[\s\S]*gpuInfo\?\.initialRasterSurfaceReady === true/);
  assert.match(controllerSource, /backend === "webgpu" && !this\.webgpuTransportReady\(\)/);
  const phaseWarmup = solverSource.slice(
    solverSource.indexOf("private async publishInitialSparseScenePhase"),
    solverSource.indexOf("/** Publish a complete t=0 scene"),
  );
  const fence = phaseWarmup.indexOf("await this.device.queue.onSubmittedWorkDone()");
  const publication = phaseWarmup.indexOf("this.initialSparseAuthorityPublished = true", fence);
  assert.ok(fence >= 0 && publication > fence, "readiness must publish only after the final phase queue fence");
  const validation = phaseWarmup.indexOf("await this.validateInitialSparseAuthority()", fence);
  assert.ok(validation > fence && publication > validation,
    "readiness must follow the bounded fine/coarse and pressure authority proof");
  assert.match(phaseWarmup, /if \(phase === "sparse-render-world"\)[\s\S]*initialSparseAuthorityPublished = true/);
  assert.doesNotMatch(solverSource,
    /fencedInitialSparseAuthority|publishInitialSparseSceneBatched|combined Section 4\/5 publication/,
    "initial authority has one phase-fenced readiness path");
  const authoritySwitch = octreeSource.slice(
    octreeSource.indexOf("encodeInitialSparseAuthorityPhase"),
    octreeSource.indexOf("private encodeGlobalFineFaceBand", octreeSource.indexOf("encodeInitialSparseAuthorityPhase")),
  );
  for (const phase of ["topology-build", "transition-adjacency", "closest-point-extension", "power-publication"]) {
    assert.match(authoritySwitch, new RegExp(`encodeGlobalFineFaceBandPhase\\(broker, "${phase}"\\)`));
  }
  assert.match(authoritySwitch, /case "sparse-render-world": this\.encodeSparseBrickWorld\(encoder\)/);
  assert.match(solverSource, /initialSparseAuthorityReady: this\.initialSparseAuthorityPublished/);
  assert.match(rendererSource, /solver\.initialSparseAuthorityReady!==true\)\{solver\.destroy\(\);throw new Error/);
  assert.match(viewportSource, /status\.label === "WebGPU renderer ready"[\s\S]*preparing fenced t=0 solver authority/);
});

test("t=0 rejection reports exact unpublished Section 5 phi and vector rows", () => {
  const validation = solverSource.slice(
    solverSource.indexOf("private async validateInitialSparseAuthority"),
    solverSource.indexOf("private async", solverSource.indexOf("private async validateInitialSparseAuthority") + 1),
  );
  assert.match(validation,
    /firstPhiFailureSlot===0xffff_ffff[\s\S]*candidateTransition\?\.ownerFailure\?\.stage===OCTREE_FACE_BAND_OWNER_FAILURE_STAGE\.bandPhi[\s\S]*readGlobalFineCandidateBandRowFailure\([\s\S]*candidateTransition\.ownerFailure\.band\)[\s\S]*cause:"phiRow"/,
    "a band-phi rejection reads the exact unpublished row index retained by the transition diagnostic");
  assert.match(validation,
    /vectorRow!==0xffff_ffff&&vectorRow<faceBand\.rowCount[\s\S]*readGlobalFineCandidateBandRowFailure\(vectorRow\)[\s\S]*cause:"vectorReconstruction"/,
    "vector reconstruction names the exact unpublished row, metric, incidence, and provisional vector");
  assert.match(validation,
    /transientPower\.failureDomain==="face"[\s\S]*readGlobalFineTransientPowerFaceFailure\(transientPower\.firstError\)[\s\S]*readGlobalFineBandRowFailure\(transientPower\.firstError\)[\s\S]*transientPowerFailure=/,
    "transient physical-graph rejection distinguishes exact face slots from row failures");
  assert.match(validation, /faceBandProducerFailure=/);
  assert.doesNotMatch(validation, /faceBandCatalogFailure=/,
    "the packed producer diagnostic must not retain the deleted catalog-hash label");
  assert.match(octreeSource,
    /readGlobalFineCandidateBandRowFailure\(index: number\)[\s\S]*readBandRowFailure\(index, true\)/);
  assert.match(validation,
    /readGlobalFineCandidateBandFaceFailure\(faceBand\.firstPhiFailureSlot\)/,
    "a rejected candidate face slot must be read from the unpublished candidate arena");
  assert.match(validation,
    /readGlobalFineCandidateBandAcuteTetraFailure\(phiStage\)/,
    "a rejected acute tetrahedron must be read from the unpublished candidate adjacency");
  assert.match(octreeSource,
    /readGlobalFineCandidateBandFaceFailure\(slot: number\)[\s\S]*readBandFaceFailure\(slot, true\)/);
  assert.match(methodTypesSource,
    /readGlobalFineCandidateBandRowFailure\?\(index: number\): Promise<unknown> \| undefined/);
  assert.match(methodTypesSource,
    /readGlobalFineCandidateBandFaceFailure\?\(slot: number\): Promise<unknown> \| undefined/);
  assert.match(solverSource,
    /readGlobalFineCandidateBandRowFailure\(index: number\)[\s\S]*octreeProjection\?\.readGlobalFineCandidateBandRowFailure\(index\)/);
});

test("safe browser mode is a one-step lease with explicit teardown", () => {
  assert.match(viewportSource, /const acquisition = acquireBrowserGPULease\(lockManager\)[\s\S]*const lease = await acquisition/);
  const lease = viewportSource.indexOf("const lease = await acquisition");
  const initialize = viewportSource.indexOf("renderer.initialize()", lease);
  assert.ok(lease >= 0 && initialize > lease, "the cross-tab lease must be acquired before adapter initialization");
  assert.match(viewportSource, /GPU_MANUAL_STOP_EVENT/);
  assert.match(viewportSource, /pagehide/);
  assert.match(viewportSource, /await shutdownBrowserGPUSession\(renderer, pendingLease, releaseGPULease\)/);
  assert.match(viewportSource, /await shutdownBrowserGPUSession[\s\S]*state: "unavailable", label: releasedLabel/);
  assert.match(viewportSource, /state: "stopping"/);
  assert.match(viewportSource, /if \(!alive \|\| stopping \|\| stopped\) \{ if \(lease\.status === "acquired"\) lease\.release\(\)/);
  assert.match(viewportSource, /Safe WebGPU session stopped after configuration drift/);
  assert.match(viewportSource, /Safe WebGPU session stopped after a reset\/rebuild attempt/);
  assert.match(viewportSource, /safeBringup[\s\S]*setRunState\("paused"\)/);
  assert.match(transportSource, /browserSafetyLocked = safeBringupPolicy !== false/);
  assert.match(transportSource, /disabled=\{transportLocked \|\| browserSafetyLocked\}/);
  assert.match(transportSource, /safeStepRequested \|\| \(gpuInfo\?\.encodedSteps \?\? 0\) >= 1/);
  assert.match(transportSource, /STOP GPU/);
  assert.match(controllerSource, /safeBrowserBringup\(\) && runtime\.runState === "running"/);
  assert.match(controllerSource, /safeBrowserBringup\(\) && this\.safeBrowserStepConsumed/);
});

test("URL-derived safe mode is hydration-stable and fails locked", () => {
  assert.match(safeModeHookSource, /serverSnapshot = \(\): null => null/,
    "SSR and the first client render must share the same unresolved policy state");
  assert.match(safeModeHookSource, /useSyncExternalStore\(subscribeToStaticURL, browserSnapshot, serverSnapshot\)/,
    "the browser URL may only affect rendered controls after hydration");
  assert.doesNotMatch(transportSource, /typeof window === "undefined"/);
  assert.doesNotMatch(transportSource, /window\.location\.search/);
  assert.match(transportSource, /browserPolicyPending \? "Browser GPU safety policy is loading"/);
  assert.match(transportSource, /\{safeBringup && <button[^>]*>STOP GPU<\/button>\}/,
    "STOP GPU appears only after the safe policy resolves true");
});
