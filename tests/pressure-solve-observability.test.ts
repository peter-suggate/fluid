import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const panelSource = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");
const methodPanelSource = readFileSync(new URL("../components/MethodPanel.tsx", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const uniformSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const transportSource = readFileSync(new URL("../lib/webgpu-octree-fine-levelset-transport.ts", import.meta.url), "utf8");
const faceBandSource = readFileSync(new URL("../lib/webgpu-octree-face-fast-march.ts", import.meta.url), "utf8");
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
  assert.match(panelSource, /FINE SURFACE TOTAL/);
  for (const label of ["FINE PREP + SEEDS", "FINE TRANSPORT", "FINE TOPOLOGY", "REDISTANCE + VOLUME",
    "RESTRICT + COARSE φ", "PAGE SURFACE", "OTHER COUPLING"]) assert.match(panelSource, new RegExp(label.replace("+", "\\+")));
  assert.match(panelSource, /\(gpuInfo\?\.globalFineTransportEncodedPasses \?\? 0\) > 0/,
    "transport schedule must remain hidden for octree methods that initialize the optional counters to zero");
  assert.match(panelSource, /\(gpuInfo\?\.globalFineLevelSetLogicalBrickCount \?\? 0\) > 0/,
    "fine occupancy must remain hidden when no global fine lattice is allocated");
  assert.match(panelSource, /globalFineTransportQueryCapacity/);
  assert.match(panelSource, /globalFineTransportChunkCount/);
  assert.match(panelSource, /globalFineTransportSegmentCount/);
  assert.match(panelSource, /globalFineTransportVertexScratchBytes/);
  assert.match(panelSource, /no per-query vertex scratch/);
  assert.match(panelSource, /globalFineTransportPrepassScratchBytes/);
  assert.match(panelSource, /TRANSPORT SETUP/);
  assert.match(panelSource, /direct Stage-B/);
  assert.match(panelSource, /air completion/);
  assert.match(panelSource, /classify \{segment\.airBandClassify_ms/);
  assert.match(panelSource, /evaluate \{segment\.airBandEvaluate_ms/);
  assert.match(panelSource, /finalize \{segment\.airBandFinalize_ms/);
  assert.match(panelSource, /trajectory advance/);
  assert.match(panelSource, /TRANSPORT FINALIZE/);
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
  assert.match(uniformSource, /advancePhaseWallLastAdvance = OCTREE_INITIAL_ADVANCE_PHASE_PROFILE_STEP/);
  assert.match(uniformSource, /this\.profilerAdvanceCount - this\.advancePhaseWallLastAdvance/,
    "profiler cadence must count accepted browser advances, not adaptive inner substeps");
  assert.match(uniformSource, /OCTREE_ADVANCE_PHASE_PROFILE_CADENCE_ADVANCES/);
  assert.match(uniformSource, /this\.advancePhaseWallLastAdvance = this\.profilerAdvanceCount/);
  assert.match(uniformSource, /productionPhaseProbeActive/);
  assert.match(uniformSource, /submitCurrentEncoder\("topologyAdvection", true\)/);
  assert.match(uniformSource, /submitCurrentEncoder\("pressureProjection", true\)/);
  assert.match(uniformSource, /finePreparation: 0, fineTransport: 0, fineTopology: 0/);
  assert.match(uniformSource, /fineRedistance: 0, fineRestriction: 0, pageSurface: 0/);
  assert.match(octreeSource, /splitProductionPhase\("fineTransport"\)/);
  assert.match(octreeSource, /splitProductionPhase\("finePreparation"\)/);
  assert.match(octreeSource, /splitProductionPhase\("fineTopology"\)/);
  assert.match(octreeSource, /splitProductionPhase\("fineRedistance"\)/);
  assert.match(octreeSource, /splitProductionPhase\("fineRestriction"\)/);
  assert.match(octreeSource, /splitProductionPhase\("pageSurface"\)/);
  assert.match(octreeSource, /planFineLevelSetGPUTransportPasses\(/);
  assert.match(octreeSource, /globalFineTransportEncodedPasses = transportPasses\.encodedPasses/);
  assert.match(uniformSource, /globalFineTransportEncodedPasses: octree\.globalFineTransportEncodedPasses/);
  assert.match(uniformSource, /submitCurrentEncoder\("surfaceCoupling", true\)/);
  assert.match(uniformSource, /submitCurrentEncoder\("publicationDiagnostics", false\)/);
  assert.match(uniformSource, /gpuAdvancePhaseWall =/);
  assert.match(transportSource, /kind: "directStageB"/);
  for (const stage of ["classifyAirBandVelocity", "evaluateAirBandVelocity", "finalizeAirBandVelocity"]) {
    assert.match(transportSource, new RegExp(stage));
    assert.match(faceBandSource, new RegExp(`splitStage\\("${stage}"`));
  }
  assert.match(faceBandSource, /if \(!boundary\)[\s\S]*dispatch\(pass, this\.sampleClassifyPipeline[\s\S]*dispatch\(pass, this\.sampleEvaluatePipeline[\s\S]*dispatch\(pass, this\.sampleFinalizePipeline/,
    "ordinary advances must retain one unsplit air-sample compute pass");
  assert.match(transportSource, /kind: "trajectoryAdvance"/);
  assert.match(transportSource, /boundary\(detail, encoder\)/,
    "fine transport profiling must split command buffers without shader instrumentation");
  assert.match(octreeSource, /encoder = transport\.encode\(encoder/,
    "the caller must retain the replacement encoder across granular boundaries");
  assert.match(octreeSource, /globalFinePublicationByEncoder\.set\(encoder, publicationTargetIsA\)/,
    "A\/B publication must remain registered on the encoder that owns restriction\/finalization");
  assert.match(uniformSource, /submitCurrentEncoder\(phase, true, detail\)/);
  assert.match(uniformSource, /fineTransportSegments = new Map/);
  assert.match(panelSource, /no shader instrumentation or atomics are added/);
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
