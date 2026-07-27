import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
const octree = source("../lib/webgpu-octree.ts");
const solver = source("../lib/webgpu-uniform-eulerian.ts");
const panel = source("../components/PerformancePanel.tsx");
const activityGrid = source("../components/PerformanceActivityGrid.tsx");
const renderer = source("../lib/webgpu-renderer.ts");
const viewport = source("../components/WebGPUViewport.tsx");

test("authoritative power work maps onto semantic adjacent-boundary phases", () => {
  for (const mapping of [
    'mgpcgSolve: { id: "pressure-solve", label: "Selected power pressure solve" }',
    'powerDescriptorTopology: { id: "power-topology", label: "Power topology publication" }',
    'structuredAdvectionBoundaryRhs: { id: "velocity-advection", label: "Structured advection + boundary RHS" }',
    'structuredProjection: { id: "velocity-projection", label: "Structured pressure projection + CPT seeds" }',
    'fineTransport: { id: "fine-sdf-advection", label: "Factor-m fine SDF advection" }',
    'fineRedistance: { id: "fine-sdf-redistance", label: "Fine SDF redistance" }',
  ]) assert.ok(solver.includes(mapping), mapping);
  assert.match(solver, /productionBoundary: physicsTrace \|\| shouldCaptureLogicalActivity \? \(phase, completedEncoder\) => \{[^]*completePhysicsPhase\(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE\[phase\]\)/);
  assert.match(solver, /physicsTrace\?\.resolve\(encoder\)/);
});

test("all seven collapsed engine phases map to timestamp categories", () => {
  for (const phase of [
    "structureEpoch",
    "rowEngineA",
    "solveEngine",
    "rowEngineB",
    "brickEngineA",
    "closestPointWaves",
    "brickEngineB",
  ] as const) {
    assert.match(
      solver,
      new RegExp(`${phase}: \\{ id: "[^"]+", label: "\\[engine:[^\\]]+\\]`),
      `${phase} must retain a machine-visible engine timestamp category`,
    );
  }
  assert.match(
    solver,
    /encodeSurface\(encoder, dt, surfaceInflow[^]*OCTREE_SEMANTIC_TRACE_PHASE\[phase\]/,
    "surface checkpoints must use the same exhaustive semantic timestamp map",
  );
});

test("next-epoch topology and adaptive surface publication close generic phases", () => {
  assert.match(solver,
    /encodeReadyTopologyFlip\(encoder\)[^]*encodeSurface\(encoder, dt[^]*encodeInactiveTopologyCandidate\(encoder\)[^]*Inactive next-substep topology candidate/);
  assert.match(solver,
    /encodeSurface\(encoder, dt, surfaceInflow[^]*completePhysicsPhase\(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE\[phase\]\)/);
  assert.doesNotMatch(solver + octree, /timing start|timing end|beginRange\(|endRange\(/);
});

test("physics checkpoints follow performed command stages at their trailing seams", () => {
  assert.match(solver,
    /encodeReadyTopologyFlip\(encoder\);[^]*Accepted topology epoch \+ Section 5 air support/);
  assert.match(octree,
    /settled t=0 fine-demand air support published[^]*productionBoundary\("structuredProjectionTail", encoder\)/);
  assert.doesNotMatch(solver, /pressureLeafCompactionL1Capture/);
  assert.match(solver,
    /encodeBodyImpulseExchange[^]*Rigid-body impulse exchange \+ integration/);
  assert.match(solver,
    /encodeSparseBrickWorld\(encoder, dt\);[^]*Sparse-brick residency \+ publication/);
  assert.match(solver,
    /Uniform diagnostics reduction[^]*Diagnostics reduction/);
});

test("physics sampling defaults to detailed capture and remains cadence-bounded", () => {
  const store = source("../lib/stores/performance-instrumentation-store.ts");
  const tall = source("../lib/webgpu-eulerian.ts");
  assert.match(store, /mode: "activity"/);
  assert.match(store, /enabled: true/,
    "the default UI simulation path captures detailed profiling evidence");
  assert.match(solver, /const PHYSICS_TRACE_CADENCE_MS = 100/);
  assert.match(tall, /lastPhysicsTraceAt_ms>=100/);
  assert.match(tall, /Substep planning \+ advance setup/);
});

test("a traced physics step submits the same command graph as an untraced one", () => {
  assert.match(solver, /if \(physicsTrace\) encoder = physicsTrace\.instrument\(encoder\)/,
    "boundaries must ride the step's own passes, not marker passes of their own");
  assert.doesNotMatch(solver, /GPUSegmentedQueueWallPerformanceTraceRecorder|SEGMENTED_QUEUE_TRACE_CADENCE_MS/,
    "the intrusive fence-per-phase probe is exactly the UI wall-clock tax this lane removes");
  assert.doesNotMatch(solver, /profiledAdvanceCompletion/,
    "every advance now submits one command buffer synchronously");
});

test("an unusable hardware sample falls back without paying for boundaries again", () => {
  assert.match(solver, /hardwarePhysicsTraceInvalid = !trace/);
  assert.match(solver, /!this\.hardwarePhysicsTraceInvalid[^]*GPUStageTimestampRecorder\.supported/);
  assert.match(solver, /GPUQueueWallPerformanceTraceRecorder/);
  assert.match(activityGrid, /data-queue-wall-fallback=/);
  assert.doesNotMatch(panel, /INTRUSIVE QUEUE CHECKPOINT PROBE/);
});

test("logical workgroup capture survives hardware timestamp fallback", () => {
  assert.match(solver, /const shouldCaptureLogicalActivity = shouldTracePhysics[^]*this\.logicalActivity\.enabled/);
  assert.match(solver, /shouldCaptureLogicalActivity && logicalActivitySession && physicsTraceRead/);
  assert.match(solver, /Promise\.all\(\[session\.read\(\), physicsTraceRead\]\)/);
  assert.doesNotMatch(solver, /const shouldCaptureLogicalActivity = Boolean\(physicsTrace\)/);
});

test("physics timing does not create known empty queue segments", () => {
  assert.doesNotMatch(octree, /splitProductionPhase\("pressureAssemblySetup"\)/);
  assert.doesNotMatch(solver, /label: "Pressure field materialization"/);
  assert.doesNotMatch(solver, /label: "Surface coupling \+ generation commit"/);
});

test("physics UI preserves command-adjacent step labels", () => {
  assert.match(panel, /trace\.lane === "physics" && trace\.measurementSource !== "gpu-queue-wall"/);
  assert.match(panel, /return trace;/);
});

test("performance UI exposes one matrix with three independent closed ledgers and no combined total", () => {
  assert.equal((panel.match(/<PerformanceActivityGrid/g) ?? []).length, 1);
  assert.doesNotMatch(panel, /TraceLane|trace-summary|trace-stack/);
  assert.match(activityGrid, /ACCOUNTING LEDGERS/);
  assert.match(activityGrid, /performanceTraceAccounting\(trace\)\.exact/);
  assert.match(activityGrid, /data-accounting-ledgers="cpu-gpu-independent"/);
  assert.doesNotMatch(panel, /cpu\.total_ms\s*\+|physics\.total_ms\s*\+|presentation\.total_ms\s*\+/);
});

test("production SVO traversal and shading execute for every submitted presentation", () => {
  assert.doesNotMatch(renderer, /drySceneReuseKey|Sparse voxel dry scene reuse timestamp/);
  assert.match(renderer, /SVO visibility and shading are presentation work[^]*svoDryScenePipeline\?\.encode\(replacementEncoder, target, temporalFrame,/);
  assert.match(viewport, /continuousPerformancePresentation[^]*ui\.rightPanel === "performance"[^]*!continuousPerformancePresentation/,
    "the performance panel must keep a paused static dry scene submitting completion-gated presentations");
});
