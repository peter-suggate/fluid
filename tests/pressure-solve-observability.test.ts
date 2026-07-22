import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const panelSource = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");
const methodPanelSource = readFileSync(new URL("../components/MethodPanel.tsx", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");

test("pressure solve UI reports an isolated completion fence instead of a whole-advance bound", () => {
  assert.match(panelSource, /PRESSURE SOLVE · ISOLATED PROFILER SAMPLE/);
  assert.match(panelSource, /HOST COMMAND ENCODE/);
  assert.match(panelSource, /GPU KERNEL RANGE/);
  assert.match(panelSource, /COMMAND STREAM/);
  assert.match(panelSource, /SUBMIT → COMPLETE/);
  assert.match(panelSource, /exact pressure-only fence/);
  assert.match(panelSource, /\? "PRESSURE" : stage\.shortLabel/);
  assert.match(panelSource, /\? "NON-PRESSURE" : "UNATTRIBUTED"/);
  assert.match(panelSource, /queue-boundary sample below localizes the rest/);
  assert.match(panelSource, /PRODUCTION ADVANCE · QUEUE-BOUNDARY SAMPLE/);
  assert.match(panelSource, /TOPOLOGY \+ ADVECT/);
  assert.match(panelSource, /PRESSURE \+ PROJECT/);
  assert.match(panelSource, /FINE SURFACE \+ COUPLE/);
  assert.match(panelSource, /<span>FINE SURFACE<\/span>/);
  assert.match(panelSource, /fine-surface-observed-row/);
  assert.match(panelSource, /displayedPhysicsStages = fineSurfaceNeedsWallFallback/);
  assert.match(panelSource, /FINE_SURFACE_COMPONENT_KEYS\.has\(stage\.key\)/);
  assert.doesNotMatch(panelSource, /PRESSURE WALL|FINE SURFACE WALL|NON-PRESSURE WALL|UNATTRIBUTED WALL/);
});

test("production profiler splits a real advance at queue submission boundaries", () => {
  assert.match(uniformSource, /pressureWallProbeLastStep = 0/,
    "the first interactive step must publish transport diagnostics before intrusive profiling begins");
  assert.match(uniformSource, /OCTREE_INITIAL_ADVANCE_PHASE_PROFILE_STEP = 2/,
    "the first queue-attributed phase sample must follow one ordinary transport step");
  assert.match(uniformSource, /advancePhaseWallLastStep = OCTREE_INITIAL_ADVANCE_PHASE_PROFILE_STEP/);
  assert.match(uniformSource, /this\.advancePhaseWallLastStep\s*>= OCTREE_ADVANCE_PHASE_PROFILE_CADENCE_STEPS/);
  assert.match(uniformSource, /this\.advancePhaseWallLastStep = this\.info\.encodedSteps/);
  assert.match(uniformSource, /productionPhaseProbeActive/);
  assert.match(uniformSource, /submitCurrentEncoder\("topologyAdvection", true\)/);
  assert.match(uniformSource, /submitCurrentEncoder\("pressureProjection", true\)/);
  assert.match(uniformSource, /submitCurrentEncoder\("surfaceCoupling", true\)/);
  assert.match(uniformSource, /submitCurrentEncoder\("publicationDiagnostics", false\)/);
  assert.match(uniformSource, /gpuAdvancePhaseWall =/);
});

test("octree solver reports only its own host encode and bounded pass schedule", () => {
  assert.match(octreeSource, /cpuPressureSolveEncode_ms = performance\.now\(\) - pressureSolveEncodeStartedAt_ms/);
  assert.match(octreeSource, /pressureSolvePassCount = this\.mgpcg!\.encodedPassCount/);
  assert.match(octreeSource, /pressureSolvePassTransitionCount = this\.mgpcg!\.encodedPassTransitionCount/);
  assert.match(uniformSource, /cpuPressureSolveEncode_ms \+= this\.octreeProjection\.info\.cpuPressureSolveEncode_ms/);
  assert.match(uniformSource, /pressureSolvePassCount \+= this\.octreeProjection\.info\.pressureSolvePassCount/);
  assert.match(uniformSource, /pressureSolvePassTransitionCount \+= this\.octreeProjection\.info\.pressureSolvePassTransitionCount/);
});

test("isolated pressure probe drains production work and preserves authoritative pressure parity", () => {
  assert.match(octreeSource, /encodePressureWallProbe\(encoder: GPUCommandEncoder\)/);
  assert.match(octreeSource, /const pressureOut = this\.latestPressureInA \? this\.pressureB : this\.pressureA/);
  const probe = uniformSource.slice(uniformSource.indexOf("private schedulePressureWallProbe"),
    uniformSource.indexOf("private timing", uniformSource.indexOf("private schedulePressureWallProbe")));
  const firstFence = probe.indexOf("queue.onSubmittedWorkDone()");
  const encode = probe.indexOf("encodePressureWallProbe");
  const submit = probe.indexOf("queue.submit");
  const secondFence = probe.indexOf("queue.onSubmittedWorkDone()", firstFence + 1);
  assert.ok(firstFence >= 0 && firstFence < encode && encode < submit && submit < secondFence,
    "production must drain before the pressure-only submission and its own completion fence");
});

test("physics completion excludes older presentation work and pressure excludes telemetry", () => {
  assert.match(rendererSource, /queueReadyAtPromise = this\.performanceReadbacksEnabled/);
  assert.match(rendererSource, /physicsQueueWall_ms = Math\.max\(0, completedAt_ms - queueReadyAt_ms\)/);
  assert.match(rendererSource, /gpuBatchWall_ms = physicsQueueWall_ms/);
  assert.match(uniformSource, /this\.pressureWallProbePending[\s\S]*this\.readbackPending\) return this\.info/);
  assert.match(rendererSource, /gpuCompletionProductionWall_ms = Math\.max\(0, completionWall_ms - profilerWall_ms\)/);
  assert.match(panelSource, /QUEUE IDLE/);
  assert.match(panelSource, /queue-starved between advances/);
});

test("MGPCG diagnostics distinguish iterations executed from the encoded cap", () => {
  assert.match(octreeSource, /copyBufferToBuffer\(this\.mgpcg\.control, 0, readback, 32, 64\)/);
  assert.match(octreeSource, /this\.info\.pressureIterationsUsed = words\[2\]/);
  assert.match(panelSource, /dispatches · .*iterations/);
  assert.match(methodPanelSource, /PCG iterations executed \/ cap/);
});
