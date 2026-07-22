import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeMethod } from "../lib/methods/octree";

const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const mgpcgSource = readFileSync(new URL("../lib/webgpu-octree-mgpcg.ts", import.meta.url), "utf8");
const cycleSource = readFileSync(new URL("../lib/webgpu-octree-first-order-vcycle.ts", import.meta.url), "utf8");
const spgridCycleSource = readFileSync(new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const smokeSource = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
const methodPanelSource = readFileSync(new URL("../components/MethodPanel.tsx", import.meta.url), "utf8");
const performancePanelSource = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");

test("paper pressure hierarchy is an explicit A/B with the proven 128-tail default", () => {
  const hierarchy = octreeMethod.params.find((parameter) => parameter.key === "powerMultigridHierarchy");
  assert.ok(hierarchy && hierarchy.kind === "select");
  assert.equal(hierarchy.default, "aggregate-galerkin");
  assert.deepEqual(hierarchy.options.map((option) => option.value), ["aggregate-galerkin", "paper-pyramid"]);

  for (const quality of ["balanced", "high", "ultra"] as const) {
    const preset = octreeMethod.presetFor(quality);
    assert.equal(preset.powerMultigridHierarchy, "aggregate-galerkin",
      `${quality} must retain the endurance-proven rollback hierarchy`);
    assert.equal(preset.powerPcgIterationCap, 128,
      `${quality} must retain the safe recorded PCG tail`);
  }

  assert.match(methodSource,
    /powerMultigridHierarchy: powerMultigridHierarchy\(values\.powerMultigridHierarchy\)/,
    "the UI value must cross the method boundary into projection options");
  assert.match(uniformSource,
    /powerMultigridHierarchy: options\.octree\.powerMultigridHierarchy/,
    "the uniform host must not drop the hierarchy selector while constructing the octree projection");
});

test("octree projection keeps both pressure hierarchies and selects the paper pyramid only by opt-in", () => {
  assert.match(octreeSource, /powerMultigridHierarchy\?: "aggregate-galerkin" \| "paper-pyramid"/);
  assert.match(octreeSource, /WebGPUOctreeFirstOrderVCycle/,
    "the current aggregate/Galerkin hierarchy remains the rollback implementation");
  assert.ok(octreeSource.includes("WebGPUOctreeSPGridVCycle"),
    "the paper hierarchy must be a distinct implementation, not a relabelled aggregate cycle");
  assert.ok(
    /powerMultigridHierarchy[^\n]*(?:===|!==)[^\n]*"paper-pyramid"[\s\S]{0,900}WebGPUOctreeSPGridVCycle/.test(octreeSource),
    "the paper pyramid must be reachable only through the explicit A/B selection");
  assert.ok(
    /powerMultigridHierarchy[^\n]*(?:===|!==)[^\n]*"paper-pyramid"[\s\S]{0,1400}WebGPUOctreeFirstOrderVCycle/.test(octreeSource),
    "the non-paper branch must continue to construct the rollback cycle");

  // Both implementations enter the existing Section 4.3 PCG through the same
  // SPD-cycle ABI. This preserves its fail-closed solve control and pressure
  // ping/pong publication transaction instead of adding an unsafe fallback.
  assert.match(octreeSource, /firstOrderVCycle: this\.firstOrderVCycle/);
  assert.match(octreeSource, /pressureIn = initialInA \? this\.pressureA : this\.pressureB/);
  assert.match(octreeSource, /pressureOut = initialInA \? this\.pressureB : this\.pressureA/);
  assert.match(mgpcgSource, /OCTREE_MGPCG_ERROR[\s\S]*nonConvergence: 1 << 4/);
  assert.match(mgpcgSource, /Section 4\.3 hybrid PCG requires an explicit SPD first-order V-cycle/);
});

test("MGPCG exposes dispatch work separately while recording one dependency-ordered compute pass", () => {
  assert.match(mgpcgSource, /get encodedDispatchCount\(\): number/);
  assert.match(mgpcgSource, /get encodedPassCount\(\): number \{ return this\.encodedDispatchCount; \}/,
    "legacy pass-count consumers must remain an explicit dispatch-count alias during migration");
  assert.match(mgpcgSource, /readonly encodedPassTransitionCount = 1/);
  assert.match(mgpcgSource, /encodedCorrectionDispatchCount/);
  assert.match(cycleSource, /encodedCorrectionDispatchCount/);
  assert.match(cycleSource, /sharedPass\?: GPUComputePassEncoder/,
    "the nested V-cycle must accept the MGPCG pass rather than opening micro-passes");
  assert.equal(spgridCycleSource.match(/sharedPass\?: GPUComputePassEncoder/g)?.length, 2,
    "paper-pyramid setup and correction must both borrow the active MGPCG pass");
  assert.equal(spgridCycleSource.match(/sharedPass \?\? encoder\.beginComputePass/g)?.length, 2);
  assert.equal(spgridCycleSource.match(/if \(!sharedPass\) pass\.end\(\)/g)?.length, 2,
    "a borrowed pass must be ended only by MGPCG");
  assert.doesNotMatch(spgridCycleSource, /encoder\.clearBuffer\(/,
    "paper-pyramid setup/correction must use ordered compute clears inside the shared pass");

  const encodeStart = mgpcgSource.indexOf("  encode(encoder: GPUCommandEncoder");
  const encodeEnd = mgpcgSource.indexOf("\n  destroy(): void", encodeStart);
  assert.ok(encodeStart >= 0 && encodeEnd > encodeStart, "MGPCG encode body must remain inspectable");
  const encode = mgpcgSource.slice(encodeStart, encodeEnd);
  assert.equal(encode.match(/beginComputePass\(/g)?.length, 1,
    "the complete fixed MGPCG schedule must use one compute pass");
  assert.match(encode, /this\.source\.firstOrderVCycle!\.encodeSetup\([\s\S]*, pass\)/);
  assert.match(encode, /pass\.end\(\)/);
});

test("Dawn harness validates and applies the pressure hierarchy override", () => {
  assert.match(smokeSource, /process\.env\.FLUID_OCTREE_PRESSURE_HIERARCHY/);
  assert.match(smokeSource,
    /\["aggregate-galerkin", "paper-pyramid"\]\.includes\(powerMultigridHierarchyOverride\)/);
  assert.match(smokeSource,
    /throw new Error\("FLUID_OCTREE_PRESSURE_HIERARCHY must be aggregate-galerkin or paper-pyramid"\)/);
  assert.match(smokeSource,
    /values\.powerMultigridHierarchy = powerMultigridHierarchyOverride/,
    "the validated override must reach the method values used for construction");
});

test("Dawn performance mode isolates stepping wall time from compact-field QA", () => {
  assert.match(smokeSource, /process\.env\.FLUID_PERFORMANCE_PROFILE === "1"/);
  assert.match(smokeSource,
    /!collectStabilityEnvelope && !performanceProfileRequested/,
    "the matched-field readback must be omitted only by explicit timing mode");
  assert.match(smokeSource,
    /if \(!performanceProfileRequested\) failures\.push\(\.\.\.invariantFailures/,
    "performance runs must label, not accidentally execute, scene quality gates");
  assert.match(smokeSource, /qualityGates: performanceProfileRequested \? "skipped" : "evaluated"/);
});

test("boundary smoothing is one symmetry-locked even control from UI through MGPCG", () => {
  const smoothing = octreeMethod.params.find(
    (parameter) => parameter.key === "powerBoundarySmoothingIterations",
  );
  assert.ok(smoothing && smoothing.kind === "number");
  assert.equal(smoothing.default, 8, "the paper's paired boundary smoothing count is the default");
  assert.equal(smoothing.step, 2, "the UI must not offer odd ping/pong schedules");
  assert.equal(octreeMethod.presetFor("balanced").powerBoundarySmoothingIterations, 8);

  assert.match(methodSource,
    /powerBoundarySmoothingIterations: numberValue\(values, params, "powerBoundarySmoothingIterations"\)/);
  assert.match(uniformSource,
    /powerBoundarySmoothingIterations: options\.octree\.powerBoundarySmoothingIterations/);
  assert.match(octreeSource, /powerBoundarySmoothingIterations\?: number/);
  assert.match(octreeSource,
    /boundarySmoothingIterations: options\.powerBoundarySmoothingIterations/);
  assert.match(mgpcgSource,
    /this\.boundarySmoothingIterations = normalizeOctreeSection43BoundarySmoothing\([\s\S]*?options\.boundarySmoothingIterations/);
  assert.match(mgpcgSource,
    /for \(let i = 0; i < this\.boundarySmoothingIterations; i \+= 1\)/,
    "the same locked count must drive both symmetric boundary ladders");
  assert.equal(mgpcgSource.match(
    /for \(let i = 0; i < this\.boundarySmoothingIterations; i \+= 1\)/g,
  )?.length, 2, "pre and post smoothing must use the same count");

  assert.match(smokeSource, /process\.env\.FLUID_POWER_BOUNDARY_SMOOTHING/);
  assert.match(smokeSource, /powerBoundarySmoothingOverride % 2 !== 0/,
    "the Dawn harness must reject odd experiments instead of silently changing them");
  assert.match(smokeSource,
    /FLUID_POWER_BOUNDARY_SMOOTHING must be an even integer between 2 and 16/);
  assert.match(smokeSource,
    /values\.powerBoundarySmoothingIterations = powerBoundarySmoothingOverride/);
});

test("pressure tuning UI distinguishes numerical progress, dispatches, and compute-pass transitions", () => {
  assert.match(methodPanelSource, /PCG iterations executed \/ cap/);
  assert.match(performancePanelSource, /pressureSolvePassTransitionCount\.toLocaleString\(\).*passes/);
  assert.match(performancePanelSource, /pressureSolvePassCount\.toLocaleString\(\).*dispatches/);
  assert.match(performancePanelSource, /COMPUTE PASSES/);
  assert.match(performancePanelSource, /DISPATCHES/);
});
