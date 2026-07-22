import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeMethod } from "../lib/methods/octree";

const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const mgpcgSource = readFileSync(new URL("../lib/webgpu-octree-mgpcg.ts", import.meta.url), "utf8");
const spgridCycleSource = readFileSync(new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const smokeSource = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
const methodPanelSource = readFileSync(new URL("../components/MethodPanel.tsx", import.meta.url), "utf8");
const performancePanelSource = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");

test("paper pressure hierarchy is the sole product implementation", () => {
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "powerMultigridHierarchy"), false);
  assert.match(methodSource, /powerPcgIterationCap: 128/);
  assert.doesNotMatch(methodSource, /aggregate-galerkin|powerMultigridHierarchy/);
  assert.doesNotMatch(uniformSource, /powerMultigridHierarchy/);
});

test("octree projection constructs only the paper sparse-grid pyramid", () => {
  assert.match(octreeSource, /new WebGPUOctreeSPGridVCycle/);
  assert.doesNotMatch(octreeSource, /WebGPUOctreeFirstOrderVCycle|aggregate-galerkin|additive-aggregate/);
  assert.match(octreeSource, /firstOrderVCycle: this\.firstOrderVCycle/);
  assert.match(octreeSource, /pressureIn = initialInA \? this\.pressureA : this\.pressureB/);
  assert.match(octreeSource, /pressureOut = initialInA \? this\.pressureB : this\.pressureA/);
  assert.match(mgpcgSource, /Section 4\.3 hybrid PCG requires an explicit SPD first-order V-cycle/);
  assert.doesNotMatch(mgpcgSource, /buildHierarchyMap|solveCoarseAggregates|prolongateCorrection/);
});

test("SPGrid publishes live indirect work before the dependency-ordered MGPCG solve pass", () => {
  assert.match(mgpcgSource, /get encodedDispatchCount\(\): number/);
  assert.match(mgpcgSource, /get encodedPassCount\(\): number \{ return this\.encodedDispatchCount; \}/);
  assert.match(mgpcgSource, /readonly encodedPassTransitionCount = 2/);
  assert.equal(spgridCycleSource.match(/sharedPass\?: GPUComputePassEncoder/g)?.length, 1,
    "only correction may join the active MGPCG pass");
  assert.equal(spgridCycleSource.match(/sharedPass \?\? encoder\.beginComputePass/g)?.length, 1);
  assert.equal(spgridCycleSource.match(/if \(!sharedPass\) pass\.end\(\)/g)?.length, 1);
  assert.doesNotMatch(spgridCycleSource, /encoder\.clearBuffer\(/);
  assert.match(spgridCycleSource, /dispatchWorkgroupsIndirect\(this\.indirectDispatch[\s\S]*resetInvalidBuffers/);
  assert.match(spgridCycleSource, /"retireSlots"/);
  assert.match(spgridCycleSource, /pass\.end\(\);[\s\S]*encoder\.copyBufferToBuffer\(this\.dispatchMeta, 0, this\.indirectDispatch/);
  assert.match(spgridCycleSource, /dispatchWorkgroupsIndirect\(this\.indirectDispatch/);

  const encodeStart = mgpcgSource.indexOf("  encode(encoder: GPUCommandEncoder");
  const encodeEnd = mgpcgSource.indexOf("\n  private applyPreconditioner", encodeStart);
  const encode = mgpcgSource.slice(encodeStart, encodeEnd);
  assert.equal(encode.match(/beginComputePass\(/g)?.length, 1);
  assert.match(encode, /this\.source\.firstOrderVCycle\.encodeSetup\([\s\S]*\);[\s\S]*const pass = encoder\.beginComputePass/);
  assert.doesNotMatch(encode, /encodeSetup\([\s\S]*, pass\)/);
  assert.match(encode, /pass\.end\(\)/);
});

test("pressure source no longer contains the deleted aggregate hierarchy", () => {
  assert.doesNotMatch(mgpcgSource, /buildHierarchyMap|solveCoarseAggregates/);
});

test("Dawn performance mode isolates stepping wall time from compact-field QA", () => {
  assert.match(smokeSource, /process\.env\.FLUID_PERFORMANCE_PROFILE === "1"/);
  assert.match(smokeSource, /!collectStabilityEnvelope && !performanceProfileRequested/);
  assert.match(smokeSource, /if \(!performanceProfileRequested\) failures\.push\(\.\.\.invariantFailures/);
  assert.match(smokeSource, /qualityGates: performanceProfileRequested \? "skipped" : "evaluated"/);
});

test("boundary smoothing remains a symmetry-locked paper invariant", () => {
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "powerBoundarySmoothingIterations"), false);
  assert.match(methodSource, /powerBoundarySmoothingIterations: 8/);
  assert.match(uniformSource, /powerBoundarySmoothingIterations: options\.octree\.powerBoundarySmoothingIterations/);
  assert.match(mgpcgSource, /normalizeOctreeSection43BoundarySmoothing/);
  assert.equal(mgpcgSource.match(/for \(let i = 0; i < this\.boundarySmoothingIterations; i \+= 1\)/g)?.length, 2);
  assert.doesNotMatch(smokeSource, /FLUID_POWER_BOUNDARY_SMOOTHING/);
});

test("pressure tuning UI distinguishes progress, dispatches, and compute-pass transitions", () => {
  assert.match(methodPanelSource, /PCG iterations executed \/ cap/);
  assert.match(performancePanelSource, /pressureSolvePassTransitionCount\.toLocaleString\(\).*passes/);
  assert.match(performancePanelSource, /pressureSolvePassCount\.toLocaleString\(\).*dispatches/);
});
