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

test("fixed Galerkin and paper MGPCG are explicit product choices", () => {
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "powerMultigridHierarchy"), false);
  const selector = octreeMethod.params.find((parameter) => parameter.key === "powerPressureSolver");
  assert.ok(selector && selector.kind === "select");
  assert.equal(selector.default, "galerkin");
  assert.deepEqual(selector.options.map((option) => option.value), ["galerkin", "section43-mgpcg"]);
  assert.doesNotMatch(methodSource, /powerPcgIterationCap|pressureIterations/);
  assert.doesNotMatch(uniformSource, /powerPcgIterationCap: options\.octree\.powerPcgIterationCap/);
  assert.doesNotMatch(methodSource, /aggregate-galerkin|powerMultigridHierarchy/);
  assert.doesNotMatch(uniformSource, /powerMultigridHierarchy/);
});

test("octree projection constructs one immutable selected pressure implementation", () => {
  assert.match(octreeSource, /if \(this\.pressureSolverMode === "galerkin"\)/);
  assert.match(octreeSource, /new WebGPUOctreePowerGalerkin/);
  assert.match(octreeSource, /new WebGPUOctreeSPGridVCycle/);
  assert.doesNotMatch(octreeSource, /WebGPUOctreeFirstOrderVCycle|aggregate-galerkin|additive-aggregate/);
  assert.match(octreeSource, /firstOrderVCycle: this\.firstOrderVCycle/);
  assert.match(octreeSource, /pressureIn = initialInA \? this\.pressureA : this\.pressureB/);
  assert.match(octreeSource, /pressureOut = initialInA \? this\.pressureB : this\.pressureA/);
  assert.match(octreeSource, /if \(this\.galerkin\)[\s\S]*else if \(this\.mgpcg\)/);
  assert.match(octreeSource,
    /new WebGPUOctreePowerGalerkin[\s\S]*cycles:\s*scene\.sceneId === "minimal-power-dam-break"[\s\S]*hierarchy\.levels\.length === 3 \? 8 : 20[\s\S]*damping:\s*0\.25/,
    "the audited three-level mini arena must omit the measured shader-no-op cycle tail");
  assert.doesNotMatch(octreeSource, /persistentThreeLevel:\s*false/,
    "the production Galerkin lane must use the executor's measured arena-size gate");
  const selectedEncode = octreeSource.slice(
    octreeSource.indexOf("  encode(\n    encoder: GPUCommandEncoder"),
    octreeSource.indexOf("\n  encodeGlobalFineSurfaceStep", octreeSource.indexOf("  encode(\n    encoder: GPUCommandEncoder")),
  );
  assert.doesNotMatch(selectedEncode, /catch[\s\S]*(?:mgpcg|galerkin)\.encode/,
    "a failed selected solver must reject publication, never invoke the retained alternative");
  assert.match(mgpcgSource, /Section 4\.3 hybrid PCG requires an explicit SPD first-order V-cycle/);
  assert.doesNotMatch(mgpcgSource, /buildHierarchyMap|solveCoarseAggregates|prolongateCorrection/);
});

test("SPGrid publishes live indirect work before the dependency-ordered MGPCG solve pass", () => {
  assert.match(mgpcgSource, /get encodedDispatchCount\(\): number/);
  assert.doesNotMatch(mgpcgSource, /get encodedPassCount\(\): number/,
    "dispatch telemetry must not preserve the retired one-pass-per-dispatch model");
  assert.match(mgpcgSource, /readonly encodedPassTransitionCount = 2/);
  assert.doesNotMatch(spgridCycleSource, /sharedPass|GPUCommandEncoder|beginComputePass|pass\.end\(\)/,
    "SPGrid must consume one broker authority without retaining raw-pass compatibility");
  assert.match(spgridCycleSource, /encodeCorrection\(broker: PassBroker/);
  assert.match(spgridCycleSource, /const pass = broker\.compute\(\{ label: "SPGrid V-cycle · one-pass symmetric correction" \}\)/);
  assert.doesNotMatch(spgridCycleSource, /encoder\.clearBuffer\(/);
  assert.doesNotMatch(spgridCycleSource, /resetInvalidBuffers/,
    "failed setup must remain fail closed instead of scheduling an all-capacity recovery rebuild");
  assert.doesNotMatch(spgridCycleSource, /"retireSlots"/,
    "the persistent topology fingerprint must reuse unchanged levels without a retired-slot cleanup dispatch");
  assert.match(spgridCycleSource, /buildCandidateLevelDeltas/);
  assert.doesNotMatch(spgridCycleSource, /prepareSetupDispatch|setupIndirectDispatch/,
    "SPGrid setup must not retain the redundant per-level indirect command fabric");
  assert.match(spgridCycleSource, /encodeCaptureDelta/);
  assert.match(spgridCycleSource, /broker\.updateIndirectBuffer\(this\.dispatchMeta, 0, this\.indirectDispatch/);
  assert.match(spgridCycleSource, /dispatchWorkgroupsIndirect\(this\.indirectDispatch/);

  const encodeStart = mgpcgSource.indexOf("  encode(\n    broker: PassBroker");
  const encodeEnd = mgpcgSource.indexOf("\n  private applyPreconditioner", encodeStart);
  const encode = mgpcgSource.slice(encodeStart, encodeEnd);
  assert.doesNotMatch(encode, /GPUCommandEncoder|beginComputePass|pass\.end\(\)/);
  assert.match(encode, /this\.source\.firstOrderVCycle\.encodeSetup\(broker,[\s\S]*\);[\s\S]*const pass = broker\.compute/);
  assert.match(encode, /broker\.fence\("MGPCG pressure publication"\)/);
});

test("pressure source no longer contains the deleted aggregate hierarchy", () => {
  assert.doesNotMatch(mgpcgSource, /buildHierarchyMap|solveCoarseAggregates/);
});

test("Dawn performance mode isolates stepping wall time from compact-field QA", () => {
  assert.match(smokeSource, /process\.env\.FLUID_PERFORMANCE_PROFILE === "1"/);
  assert.match(smokeSource, /!collectStabilityEnvelope && !performanceProfileRequested/);
  assert.match(smokeSource, /if \(!performanceProfileRequested\) failures\.push\(\.\.\.invariantFailures/);
  assert.match(smokeSource,
    /const simulationWall_ms[\s\S]*await awaitAdvanceCompletion\(\);[\s\S]*finalPerformanceAuthorityFailures/,
    "the final packed authority gate must remain outside measured stepping wall time");
  assert.match(smokeSource, /scenarioId === "dam-break-ui" \? "final-authority-only" : "skipped"/);
});

test("boundary smoothing remains a symmetry-locked paper invariant", () => {
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "powerBoundarySmoothingIterations"), false);
  assert.match(methodSource, /powerBoundarySmoothingIterations: 8/);
  assert.match(uniformSource, /powerBoundarySmoothingIterations: options\.octree\.powerBoundarySmoothingIterations/);
  assert.match(mgpcgSource, /normalizeOctreeSection43BoundarySmoothing/);
  assert.equal(mgpcgSource.match(/for \(let i = 0; i < this\.boundarySmoothingIterations; i \+= 1\)/g)?.length, 2);
  assert.doesNotMatch(smokeSource, /FLUID_POWER_BOUNDARY_SMOOTHING/);
});
