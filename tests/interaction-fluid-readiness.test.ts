import assert from "node:assert/strict";
import test from "node:test";
import { createInteractionFluidDriver } from "../lib/core/simulation/interaction-fluid-driver";
import { createDiagnosticsStore } from "../lib/core/stores/diagnostics-store";
import { webGPUPlatformResourcePlugin as platform } from "../lib/core/webgpu-platform-resource";
import type { ResourcePluginDefinition } from "../lib/core/resource-plugin";
import type { GPUEulerianInfo } from "../lib/core/webgpu-eulerian";
import { SceneInteractionHost } from "../lib/interaction/runtime/host";
import { createInteractivePlayground } from "../lib/interaction/playground";

const fluid: ResourcePluginDefinition = {
  id: "test.fluid", label: "Fluid", lane: "fluid",
  provides: ["fluid-authority", "water-presentation"], blocks: "transport",
};
const presentation: ResourcePluginDefinition = {
  id: "test.presentation", label: "Presentation", lane: "svo",
  provides: ["sparse-voxel-presentation"], blocks: "transport",
};
const info = { initialSparseAuthorityReady: true, initialRasterSurfaceReady: true } as GPUEulerianInfo;

test("interactive play unlocks from published readiness after the last GPU status was progress", async () => {
  const diagnostics = createDiagnosticsStore();
  const scene = createInteractivePlayground(true);
  scene.scenery = undefined;
  const host = new SceneInteractionHost(scene, {
    fluid: createInteractionFluidDriver(diagnostics, async () => []),
  });
  const publish = diagnostics.getState().set;
  try {
    await host.ready;
    publish({ gpuStatus: { state: "ready", adapter: "test", label: "Renderer ready", resource: platform } });
    assert.equal(host.getView().readiness.state, "preparing", "the device alone is not fluid readiness");
    publish({ gpuStatus: { state: "ready", adapter: "test", label: "Solver ready", resource: fluid }, gpuInfo: info });
    publish({ gpuStatus: { state: "initializing", label: "Compile raster pipeline", resource: presentation },
      effectiveRendererStatus: { state: "pending" } });
    assert.equal(host.getView().readiness.state, "preparing");
    publish({ effectiveRendererStatus: { state: "active" } });
    assert.equal(diagnostics.getState().gpuStatus.state, "initializing");
    assert.equal(host.getView().readiness.state, "ready");
    const view = host.getView();
    assert.equal((await host.dispatch({ id: "play", generation: view.generation,
      documentRevision: view.documentRevision, command: { kind: "transport", action: "play" } })).accepted, true);

    publish({ effectiveRendererStatus: { state: "pending" } });
    assert.equal(host.getView().readiness.state, "preparing");
    publish({ effectiveRendererStatus: { state: "active" } });
    assert.equal(host.getView().readiness.state, "ready");
    publish({ gpuStatus: { state: "lost", label: "GPU device lost", resource: platform } });
    assert.equal(host.getView().readiness.state, "failed");
    assert.equal(host.getView().running, false);
  } finally { await host.dispose(); }
});

test("simulation pipeline changes notify the fluid driver even without a status message", () => {
  const diagnostics = createDiagnosticsStore();
  const publish = diagnostics.getState().set;
  for (const resource of [platform, fluid, presentation]) {
    publish({ gpuStatus: { state: "ready", adapter: "test", label: "Ready", resource } });
  }
  publish({ effectiveRendererStatus: { state: "active" } });
  const driver = createInteractionFluidDriver(diagnostics, async () => []);
  const changes: boolean[] = [];
  const detach = driver.subscribeReadiness!(() => changes.push(driver.isReady!()));
  publish({ gpuInfo: { ...info, simulationPipelinesReady: false } });
  publish({ gpuInfo: { ...info, simulationPipelinesReady: true } });
  assert.deepEqual(changes, [false, true]);
  detach();
  publish({ gpuInfo: { ...info, simulationPipelinesReady: false } });
  assert.deepEqual(changes, [false, true]);
});
