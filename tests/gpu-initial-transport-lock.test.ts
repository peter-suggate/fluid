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

test("WebGPU transport stays locked until the fenced structured t=0 authority is ready", () => {
  assert.match(transportSource, /initialSceneReady = methodId !== "octree" \|\| \(gpuInfo\?\.initialSparseAuthorityReady === true[\s\S]*gpuInfo\?\.initialRasterSurfaceReady === true\)/);
  assert.match(transportSource, /transportLocked = staticRenderScene \|\| \(webgpu && \(gpuStatus\.state !== "ready" \|\| !initialSceneReady\)\)/);
  assert.match(controllerSource, /backend === "webgpu" && !this\.webgpuTransportReady\(\)/);
  const phaseWarmup = solverSource.slice(
    solverSource.indexOf("private async publishInitialSparseScenePhase"),
    solverSource.indexOf("private applyGlobalFineDiagnostics"),
  );
  assert.match(phaseWarmup,
    /queue\.submit[\s\S]*await this\.device\.queue\.onSubmittedWorkDone\(\)[\s\S]*if \(phase === "sparse-render-world"\)[\s\S]*await this\.validateInitialSparseAuthority\(\)[\s\S]*initialSparseAuthorityPublished = true/,
    "readiness must follow the final fenced structured authority proof");
  const authoritySwitch = octreeSource.slice(
    octreeSource.indexOf("encodeInitialSparseAuthorityPhase"),
    octreeSource.indexOf("retireSubmittedEncoder", octreeSource.indexOf("encodeInitialSparseAuthorityPhase")),
  );
  assert.match(authoritySwitch, /case "structured-authority"[\s\S]*"power-operator-only"/);
  assert.match(authoritySwitch, /case "surface-global-fine"[\s\S]*encodeSurface\(encoder, 0\)/);
  assert.match(authoritySwitch,
    /case "sparse-render-world"[\s\S]*encodeSparseBrickWorld\(encoder\)[\s\S]*encodeInactiveTopologyCandidate\(encoder\)/);
  assert.doesNotMatch(authoritySwitch, /face-band|power-publication|closest-point-extension/i);
  assert.match(solverSource, /initialSparseAuthorityReady: this\.initialSparseAuthorityPublished/);
  const t0Validation = solverSource.slice(
    solverSource.indexOf("private async validateInitialSparseAuthority"),
    solverSource.indexOf("/** Publish a complete t=0 scene", solverSource.indexOf("private async validateInitialSparseAuthority")),
  );
  assert.match(t0Validation,
    /if \(!structuredReady \|\| !boundaryReady\)[\s\S]*throw new Error[\s\S]*this\.applyGlobalFineDiagnostics\(fine!\)/,
    "the accepted queue-fenced t=0 controls must replace stale dynamic-step diagnostics");
  assert.match(rendererSource, /solver\.initialSparseAuthorityReady!==true\)\{solver\.destroy\(\);throw new Error/);
  assert.match(viewportSource, /status\.label === "WebGPU renderer ready"[\s\S]*preparing fenced t=0 solver authority/);
});

test("t=0 rejection reports the exact structured velocity and boundary controls", () => {
  const validation = solverSource.slice(
    solverSource.indexOf("private async validateInitialSparseAuthority"),
    solverSource.indexOf("/** Publish a complete t=0 scene", solverSource.indexOf("private async validateInitialSparseAuthority")),
  );
  assert.match(validation, /fine\?\.structuredVelocityControl \?\? \[\]/);
  assert.match(validation, /fine\?\.structuredBoundaryControl \?\? \[\]/);
  assert.match(validation, /velocity\[3\] !== 0 && velocity\[4\] <= 1/);
  assert.match(validation,
    /boundary\[2\] === velocity\[2\][\s\S]*boundary\[4\] === velocity\[3\][\s\S]*boundary\[5\] === velocity\[4\][\s\S]*boundary\[6\] === velocity\[3\]/);
  assert.match(validation,
    /Paused t=0 structured authority rejected: velocity=[\s\S]*boundary=[\s\S]*frontier=[\s\S]*owner=/);
  assert.doesNotMatch(validation, /faceBand|transientPower|PowerFace/i);
  assert.doesNotMatch(methodTypesSource,
    /readGlobalFineCandidateBandRowFailure|readGlobalFineCandidateBandFaceFailure/);
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

test("Fast Refresh and RSC program reload retain the live GPU session", () => {
  assert.match(viewportSource,
    /gpuLifecycleRef\.current \?\? lifecycleWindow\.__fluidLabGPUViewportLifecycle/,
    "RSC program reload must recover the lifecycle even when React replaces its ref");
  assert.match(viewportSource,
    /retainedLifecycle\?\.canvas === canvas && retainedLifecycle\.cancelDeferredCleanup\(\)/,
    "a replay on the retained canvas must reclaim the existing renderer");
  assert.match(viewportSource,
    /retainedLifecycle\.rebind\?\.\(renderBinding\);[\s\S]*rendererRef\.current = retainedLifecycle\.renderer;[\s\S]*return retainedLifecycle\.deferCleanup/,
    "the replacement component must reuse the renderer and adopt the retained loop");
  assert.match(viewportSource,
    /process\.env\.NODE_ENV === "development" \? GPU_DEVELOPMENT_REBIND_GRACE_MS : 0/,
    "development cleanup must leave enough time for Vinext's asynchronous RSC commit");
  const immediateCleanup = viewportSource.slice(
    viewportSource.indexOf("const cleanupImmediately = () =>"),
    viewportSource.indexOf("const lifecycle: GPUViewportLifecycle"),
  );
  assert.match(immediateCleanup, /stopGPU\("WebGPU stopped during component cleanup", false\)/,
    "a real unmount must still release the GPU after the replay window");
  assert.match(viewportSource, /const pageHide = \(\) => \{ void stopGPU\("WebGPU stopped during page close", false\); \}/,
    "a real document unload must bypass the Fast Refresh grace period");
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
