import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
const octree = source("../lib/webgpu-octree.ts");
const solver = source("../lib/webgpu-uniform-eulerian.ts");
const panel = source("../components/PerformancePanel.tsx");
const renderer = source("../lib/webgpu-renderer.ts");
const viewport = source("../components/WebGPUViewport.tsx");

test("authoritative power work maps onto semantic adjacent-boundary phases", () => {
  for (const mapping of [
    'mgpcgSolve: { id: "pressure-solve", label: "Selected power pressure solve" }',
    'powerDescriptorTopologyFaces: { id: "power-topology", label: "Power topology + physical faces" }',
    'powerProjectionPublication: { id: "velocity-projection", label: "Power-face pressure projection" }',
    'faceBandClosestPointExtension: { id: "velocity-extrapolation", label: "Closest-point velocity extension" }',
    'fineTransport: { id: "fine-sdf-advection", label: "Factor-m fine SDF advection" }',
    'fineRedistance: { id: "fine-sdf-redistance", label: "Fine SDF redistance" }',
  ]) assert.ok(solver.includes(mapping), mapping);
  assert.match(solver, /productionBoundary: physicsTrace \|\| segmentedPhysicsTrace \? \(phase, completedEncoder\) => \{[^]*completePhysicsPhase\(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE\[phase\]\)/);
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

test("topology, every CFL rebuild, and adaptive surface publication close generic phases", () => {
  assert.match(solver, /encodeInlineRebuild\(encoder\)[^]*completePhysicsPhase\(encoder, this\.adaptiveProjection/);
  assert.match(solver, /substep > 0[^]*encodeInlineRebuild\(encoder\)[^]*CFL substep topology refresh/);
  assert.match(solver,
    /encodeSurface\(encoder, dt, surfaceInflow[^]*completePhysicsPhase\(completedEncoder, OCTREE_SEMANTIC_TRACE_PHASE\[phase\]\)/);
  assert.doesNotMatch(solver + octree, /timing start|timing end|beginRange\(|endRange\(/);
});

test("invalid timestamps fall back to sparse measured phase probes", () => {
  assert.match(solver, /GPUSegmentedQueueWallPerformanceTraceRecorder/);
  assert.match(solver, /hardwarePhysicsTraceInvalid = !trace/);
  assert.match(solver, /SEGMENTED_QUEUE_TRACE_CADENCE_MS/);
  assert.match(panel, /INTRUSIVE QUEUE CHECKPOINT PROBE/);
  assert.match(panel, /not GPU execution time/);
  assert.match(panel, /upper bounds/);
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

test("performance UI exposes three independent closed ledgers and no combined total", () => {
  assert.match(panel, /TraceLane trace=\{physics\}/);
  assert.match(panel, /TraceLane trace=\{presentation\}/);
  assert.match(panel, /TraceLane trace=\{cpu\}/);
  assert.match(panel, /OBSERVED TOTAL/);
  assert.match(panel, /ACCOUNTED PHASE SUM/);
  assert.match(panel, /CLOSURE ERROR/);
  assert.match(panel, /CPU and GPU are independent ledgers and are never added together/);
  assert.doesNotMatch(panel, /cpu\.total_ms\s*\+|physics\.total_ms\s*\+|presentation\.total_ms\s*\+/);
});

test("production SVO traversal and shading execute for every submitted presentation", () => {
  assert.doesNotMatch(renderer, /drySceneReuseKey|Sparse voxel dry scene reuse timestamp/);
  assert.match(renderer, /SVO visibility and shading are presentation work[^]*svoDryScenePipeline\?\.encode\(replacementEncoder, target, temporalFrame,/);
  assert.match(viewport, /continuousPerformancePresentation[^]*ui\.rightPanel === "performance"[^]*!continuousPerformancePresentation/,
    "the performance panel must keep a paused static dry scene submitting completion-gated presentations");
});
