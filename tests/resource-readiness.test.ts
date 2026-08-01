import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import {
  initialResourceReadiness,
  reduceGPUResourceEvidence,
  reduceGPUResourceStatus,
  resourceInteractionGates,
  duplicateResourcePluginIds,
  resourceCapabilityUsable,
  resourceActivities,
} from "../lib/resource-readiness";
import { RESOURCE_PLUGIN_CATALOG, resourcePluginsProviding } from "../lib/resource-plugin-catalog";

test("resource plugins compose statically and claim unique identities", () => {
  assert.deepEqual(duplicateResourcePluginIds(RESOURCE_PLUGIN_CATALOG), []);
  assert.ok(resourcePluginsProviding("renderer").length > 0);
  assert.ok(resourcePluginsProviding("fluid-authority").length > 0);
  assert.ok(resourcePluginsProviding("sparse-voxel-presentation").length > 0);
  for (const plugin of RESOURCE_PLUGIN_CATALOG) {
    assert.ok(plugin.id.includes("."));
    assert.ok(plugin.provides.length > 0);
  }
});

test("independent resource lanes do not overwrite one another", () => {
  let readiness = initialResourceReadiness();
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU renderer ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU t=0 ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "initializing", label: "Compiling sparse dry-scene pipeline",
    phase: "presentation", completed: 0, total: 1, startedAt_ms: 1,
  });

  assert.equal(readiness.platform.usable, true);
  assert.equal(readiness.fluid.usable, true);
  assert.equal(readiness.svo.state, "preparing");
  assert.equal(resourceCapabilityUsable(readiness, "fluid-authority"), true);
  assert.equal(resourceCapabilityUsable(readiness, "sparse-voxel-presentation"), false);
  assert.deepEqual(resourceInteractionGates(readiness, true), {
    shellInteractive: true,
    viewportInteractive: true,
    transportInteractive: true,
  });
});

test("published resource evidence outranks a late progress message", () => {
  const readiness = reduceGPUResourceEvidence(initialResourceReadiness(), {
    initialSparseAuthorityReady: true,
    initialRasterSurfaceReady: true,
  } as never);
  assert.equal(readiness.fluid.state, "ready");
  assert.equal(readiness.fluid.usable, true);
});

test("renderer completion closes 4/4 before solver planning opens", () => {
  const platform = RESOURCE_PLUGIN_CATALOG.find(({ id }) => id === "platform.webgpu-renderer")!;
  const fluid = RESOURCE_PLUGIN_CATALOG.find(({ id }) => id === "fluid.power-octree")!;
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "initializing", label: "Renderer ready", phase: "renderer",
    completed: 4, total: 4, startedAt_ms: 1, resource: platform,
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU renderer ready", adapter: "test", resource: platform,
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "initializing", label: "Preparing fenced t=0 solver authority", phase: "planning",
    completed: 0, total: 0, startedAt_ms: 2, resource: fluid,
  });

  assert.deepEqual(resourceActivities(readiness).map(({ pluginId, completed, total }) => ({ pluginId, completed, total })), [
    { pluginId: "fluid.power-octree", completed: 0, total: 0 },
  ]);
});

test("replacement work retains interaction when a previous generation is attached", () => {
  let readiness = reduceGPUResourceStatus(initialResourceReadiness(), {
    state: "ready", label: "WebGPU renderer ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "ready", label: "WebGPU direct-field solver ready", adapter: "test",
  });
  readiness = reduceGPUResourceStatus(readiness, {
    state: "initializing", label: "Allocating replacement solver", phase: "allocation",
    completed: 1, total: 4, startedAt_ms: 1, kind: "rebuild", retainingPrevious: true,
  });
  assert.equal(readiness.fluid.state, "preparing");
  assert.equal(readiness.fluid.usable, true);
  assert.equal(resourceInteractionGates(readiness, true).transportInteractive, true);
});

test("the UI presents concurrent plugin work without treating all progress as a global block", () => {
  const fluidLab = readFileSync(new URL("../components/FluidLab.tsx", import.meta.url), "utf8");
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const transport = readFileSync(new URL("../components/TransportBar.tsx", import.meta.url), "utf8");
  assert.match(fluidLab, /resourceActivities\(resourceReadiness\)/);
  assert.match(fluidLab, /resource-activity-tray/);
  assert.match(fluidLab, /activities\.map/);
  assert.match(fluidLab, /aria-label="Resource tasks"/);
  assert.doesNotMatch(fluidLab, /viewportBlocked|blockingActivity|backgroundActivities/);
  assert.doesNotMatch(fluidLab, /<section className="viewport-shell" aria-busy=/);
  assert.doesNotMatch(fluidLab, /gpuStatus\.state === "initializing" && <GPUInitializationPanel/);
  assert.match(fluidLab, /finalizing \? "Finalizing…"/);
  assert.match(fluidLab, /GPU driver exposes no intermediate counters/);
  assert.match(viewport, /if \(rendererOnlyReady\) useDiagnosticsStore\.getState\(\)\.set\(\{ gpuStatus: status \}\)/,
    "the completed platform plugin must close before fluid planning begins");
  assert.match(viewport, /phase: "planning", completed: 0, total: 0/,
    "the solver handoff must not reuse the renderer's completed task count");
  assert.match(transport, /resourceInteractionGates\(resourceReadiness, !staticRenderScene\)/);
});
