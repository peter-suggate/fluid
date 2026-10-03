import assert from "node:assert/strict";
import test from "node:test";
import { initialResourceReadiness, reduceGPUResourceEvidence, reduceGPUResourceStatus } from "../lib/core/resource-readiness";
import type { GPUEulerianInfo } from "../lib/core/webgpu-eulerian";
import type { ResourcePluginDefinition } from "../lib/core/resource-plugin";

test("camera-dependent mesh diagnostics preserve readiness identity; lifecycle transitions still publish", () => {
  const presentation: ResourcePluginDefinition = {
    id: "test.presentation", lane: "svo", label: "Presentation",
    provides: ["sparse-voxel-presentation"], blocks: "nothing",
  };
  const fluid: ResourcePluginDefinition = {
    id: "test.fluid", lane: "fluid", label: "Fluid",
    provides: ["fluid-authority"], blocks: "transport",
  };
  let state = initialResourceReadiness();
  for (const resource of [presentation, fluid]) {
    state = reduceGPUResourceStatus(state, { state: "initializing", label: "Preparing", resource });
  }
  const info = { initialSparseAuthorityReady: true, initialRasterSurfaceReady: true } as GPUEulerianInfo;
  const ready = reduceGPUResourceEvidence(state, info, { state: "active" });
  assert.notEqual(ready, state);
  assert.equal(ready.plugins[presentation.id].activity, undefined);
  assert.equal(ready.plugins[fluid.id].usable, true);
  for (const quads of [1000, 4500, 0, 2300]) {
    assert.equal(reduceGPUResourceEvidence(ready, info, {
      state: "active", surfaceMesh: { state: "ready", quads },
    }), ready);
  }
  const pending = reduceGPUResourceEvidence(ready, info, { state: "pending", detail: "Rebuilding" });
  assert.notEqual(pending, ready);
  assert.equal(pending.svo.usable, false);
  assert.equal(reduceGPUResourceEvidence(pending, info, { state: "pending", detail: "Rebuilding" }), pending);
  const failed = reduceGPUResourceEvidence(pending, info, { state: "failed", detail: "Compile failed" });
  assert.notEqual(failed, pending);
  assert.match(failed.svo.label, /Compile failed/);
  assert.equal(reduceGPUResourceEvidence(failed, info, { state: "failed", detail: "Compile failed" }), failed);
  assert.notEqual(reduceGPUResourceEvidence(failed, info, { state: "active" }), failed);
  assert.notEqual(reduceGPUResourceEvidence(ready, info, { state: "not-required" }), ready);
});
