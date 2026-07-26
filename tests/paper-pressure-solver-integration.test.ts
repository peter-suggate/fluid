import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";
import { octreeMethod } from "../lib/methods/octree";

const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const pipelinedSource = readFileSync(new URL("../lib/webgpu-octree-pipelined-mgpcg.ts", import.meta.url), "utf8");
const hybridSource = readFileSync(
  new URL("../lib/webgpu-octree-section43-preconditioner.ts", import.meta.url),
  "utf8",
);
const spgridCycleSource = readFileSync(new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const smokeSource = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");

test("paper MGPCG is the only production pressure authority", () => {
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "powerMultigridHierarchy"), false);
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "powerPressureSolver"), false);
  assert.doesNotMatch(methodSource, /powerPcgIterationCap|pressureIterations/);
  assert.doesNotMatch(uniformSource, /powerPcgIterationCap: options\.octree\.powerPcgIterationCap/);
  assert.doesNotMatch(methodSource, /galerkin|powerMultigridHierarchy/i);
  assert.doesNotMatch(uniformSource, /powerGalerkinHierarchy|buildFixedAdaptiveOctreePowerGalerkinHierarchy/);
  for (const path of [
    "../lib/octree-power-galerkin.ts",
    "../lib/webgpu-octree-power-galerkin.ts",
    "../lib/webgpu-octree-power-galerkin-persistent3.ts",
    "../lib/webgpu-octree-mgpcg.ts",
    "../tools/benchmark-power-galerkin.ts",
  ]) {
    assert.equal(existsSync(new URL(path, import.meta.url)), false,
      `${path} must be deleted at the immediate solver cutover`);
  }
});

test("octree projection constructs exactly one MGPCG implementation", () => {
  assert.doesNotMatch(octreeSource, /pressureSolverMode|WebGPUOctreePowerGalerkin|this\.galerkin/);
  assert.match(octreeSource, /new WebGPUOctreeSPGridVCycle/);
  assert.match(octreeSource,
    /new WebGPUOctreeSPGridVCycle[\s\S]*preSmoothingIterations: 4, postSmoothingIterations: 4/,
    "production M1 uses the strongest page-resident polynomial without another dispatch");
  assert.match(octreeSource, /new WebGPUOctreeSection43HybridPreconditioner/);
  assert.match(octreeSource,
    /relativeTolerance:\s*this\.solveTailPolicy\.relativeTolerance/,
    "the cutover must preserve the established f32 graphics-solve tolerance floor");
  assert.match(octreeSource, /new WebGPUOctreePipelinedMGPCG/);
  assert.doesNotMatch(octreeSource, /\bWebGPUOctreeMGPCG\b/,
    "the pre-pipelined solver must not remain schedulable");
  assert.doesNotMatch(octreeSource, /WebGPUOctreeFirstOrderVCycle|aggregate-galerkin|additive-aggregate/);
  assert.match(octreeSource, /preconditioner: this\.section43HybridPreconditioner/);
  assert.match(octreeSource, /pressureIn = initialInA \? this\.pressureA : this\.pressureB/);
  assert.match(octreeSource, /pressureOut = initialInA \? this\.pressureB : this\.pressureA/);
  assert.match(octreeSource,
    /acceptedAuthority: structuredSource\.control[\s\S]*this\.pipelinedMGPCG\.encode\(solveBroker, \{[\s\S]*pressureSeed: pressureIn,[\s\S]*pressureOut/);
  const selectedEncode = octreeSource.slice(
    octreeSource.indexOf("  encode(\n    encoder: GPUCommandEncoder"),
    octreeSource.indexOf("\n  encodeGlobalFineSurfaceStep", octreeSource.indexOf("  encode(\n    encoder: GPUCommandEncoder")),
  );
  assert.doesNotMatch(selectedEncode, /catch[\s\S]*pipelinedMGPCG\.encode/,
    "a failed solve must reject publication");
  assert.match(pipelinedSource, /requires an explicit fixed-schedule SPD first-order V-cycle/);
  assert.doesNotMatch(pipelinedSource, /leafEntries|array<LeafEntry>/,
    "the iterative owner must not retain a CSR operator binding");
  assert.match(octreeSource,
    /const section63Source = structuredSource\.section63[\s\S]*this\.resolvedLinearOperator = this\.firstOrderVCycle\.accurateOperator/,
    "the production accurate operator is the rediscretized A2 owner published by the first-order V-cycle");
  assert.doesNotMatch(octreeSource, /WebGPUOctreeResolvedRowOperator/,
    "the deleted resolved-row executor must not survive beside rediscretized A2");
  assert.match(octreeSource, /acceptedAuthority: structuredSource\.control/,
    "the pressure solve must consume the direct structured publication control without a retired authority adapter");
  assert.match(pipelinedSource,
    /this\.source\.operator\.encode\([\s\S]*vectors\.pressure[\s\S]*vectors\.directionImage/);
  assert.match(pipelinedSource,
    /this\.source\.operator\.encode\([\s\S]*vectors\.preconditioned[\s\S]*vectors\.preconditionedImage/);
  assert.doesNotMatch(pipelinedSource, /buildHierarchyMap|solveCoarseAggregates|prolongateCorrection/);
});

test("SPGrid publishes live indirect work before the dependency-ordered MGPCG solve pass", () => {
  assert.match(pipelinedSource, /get encodedDispatchCount\(\): number/);
  assert.doesNotMatch(pipelinedSource, /get encodedPassCount\(\): number/,
    "dispatch telemetry must not preserve the retired one-pass-per-dispatch model");
  assert.match(pipelinedSource, /readonly encodedPassTransitionCountPerIteration = 4/);
  assert.doesNotMatch(spgridCycleSource, /sharedPass|GPUCommandEncoder|beginComputePass|pass\.end\(\)/,
    "SPGrid must consume one broker authority without retaining raw-pass compatibility");
  assert.match(spgridCycleSource, /encodeCorrection\(broker: PassBroker/);
  assert.match(spgridCycleSource,
    /let pass = broker\.compute\(\{ label: "SPGrid V-cycle · publish convergence-gated level records" \}\)[\s\S]*broker\.fence\("SPGrid V-cycle convergence-gated indirect publication"\)[\s\S]*pass = broker\.compute\(\{ label: "SPGrid V-cycle · one-pass symmetric correction" \}\)/,
    "the exact zero-tail records must be published before the fixed symmetric correction pass");
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

  assert.doesNotMatch(pipelinedSource, /GPUCommandEncoder|beginComputePass|pass\.end\(\)/);
  assert.match(pipelinedSource,
    /preconditioner\.encodeSetup\(broker,[\s\S]*preconditioner\.encodeCorrection\(broker/);
  assert.match(pipelinedSource, /broker\.fence\("pipelined MGPCG pressure publication"\)/);
});

test("pressure source no longer contains the deleted aggregate hierarchy", () => {
  assert.doesNotMatch(pipelinedSource, /buildHierarchyMap|solveCoarseAggregates/);
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

test("the pipelined preconditioner remains one fixed proof-carrying V-cycle", () => {
  assert.equal(octreeMethod.params.some((parameter) => parameter.key === "powerBoundarySmoothingIterations"), false);
  assert.doesNotMatch(methodSource, /powerBoundarySmoothingIterations/);
  assert.doesNotMatch(uniformSource, /powerBoundarySmoothingIterations/);
  assert.match(pipelinedSource,
    /export type OctreePipelinedFixedPreconditioner = OctreeFirstOrderSPDVCycle/);
  assert.equal(pipelinedSource.match(/preconditioner\.encodeCorrection\(broker/g)?.length, 2);
  assert.match(hybridSource,
    /this\.run\(pass, "formInnerResidual", resources\);[\s\S]*firstOrderVCycle\.encodeCorrection[\s\S]*this\.run\(pass, "addInnerCorrection", resources\)/,
    "the L1 correction must remain between the matching L2 halves");
  assert.match(hybridSource,
    /iteration < this\.boundarySmoothingIterations[\s\S]*formInnerResidual[\s\S]*iteration < this\.boundarySmoothingIterations/,
    "one immutable even sweep count must control both sides of the L1 correction");
  assert.match(hybridSource,
    /normalizeOctreeSection43BoundarySmoothing\(options\.boundarySmoothingIterations\)/);
  assert.doesNotMatch(hybridSource, /fallback|selector|executionMode/i);
  assert.doesNotMatch(smokeSource, /FLUID_POWER_BOUNDARY_SMOOTHING/);
});
