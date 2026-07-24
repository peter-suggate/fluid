import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  OCTREE_ENGINE_PHASES,
  OCTREE_FINE_SEMANTIC_PHASES,
  octreeFineEngineSplitsEnabled,
} from "../lib/webgpu-octree";

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

test("production instrumentation closes exactly seven engine-tagged domains", () => {
  const enginePhases = [
    "structureEpoch",
    "rowEngineA",
    "solveEngine",
    "rowEngineB",
    "brickEngineA",
    "closestPointWaves",
    "brickEngineB",
  ] as const;
  assert.deepEqual(OCTREE_ENGINE_PHASES, enginePhases);
  for (const phase of enginePhases) {
    assert.match(solver, new RegExp(`${phase}: \\{ id: "[^"]+", label: "\\[engine:[^\\]]+\\]`),
      `${phase} must retain a machine-visible engine tag`);
  }
  assert.match(octree,
    /splitProductionPhase\("structureEpoch", "powerDescriptorTopologyFaces"\)/);
  assert.match(octree,
    /splitProductionPhase\("rowEngineA", "finalPressureRowAssembly", true\)/);
  assert.match(octree, /splitProductionPhase\("solveEngine", "mgpcgSolve"\)/);
  assert.match(octree, /splitProductionPhase\("rowEngineB", "powerProjectionTail"\)/);
  assert.match(octree, /splitProductionPhase\("brickEngineA", "fineTopology"\)/);
  assert.match(octree, /splitProductionPhase\("closestPointWaves", "fineRedistance"\)/);
  assert.match(octree, /splitProductionPhase\(undefined, "fineRestriction"\)/);
  assert.match(solver,
    /this\.octreeProjection && !fineEngineSplits[\s\S]*OCTREE_SEMANTIC_TRACE_PHASE\.brickEngineB/);
});

test("fine engine split mode restores historical attribution seams only on explicit opt-in", () => {
  assert.equal(OCTREE_FINE_SEMANTIC_PHASES.length, 17,
    "the 17 octree-local seams plus final frame publication reproduce the 18-way trace");
  assert.equal(octreeFineEngineSplitsEnabled({}), false);
  assert.equal(octreeFineEngineSplitsEnabled({ FLUID_ENGINE_SPLIT: "collapsed" }), false);
  assert.equal(octreeFineEngineSplitsEnabled({ FLUID_ENGINE_SPLIT: "fine" }), true);
  assert.match(octree,
    /const phase = fineEngineSplits \? finePhase : enginePhase;/);
  assert.match(octree,
    /productionBoundary && fineEngineSplits[\s\S]*faceBandTopologyBuild/);
  assert.match(octree,
    /options\?\.productionBoundary && fineEngineSplits \? undefined : pressureBroker/,
    "collapsed tracing must keep the pressure broker shared across removed seams");
  assert.match(solver,
    /this\.octreeProjection && !fineEngineSplits[\s\S]*OCTREE_SEMANTIC_TRACE_PHASE\.brickEngineB[\s\S]*Residency \+ sparse publication \+ diagnostics/,
    "fine mode must retain the historical eighteenth frame-publication seam");
});

test("topology folds into the structure engine unless fine attribution is enabled", () => {
  assert.match(solver,
    /if \(!this\.octreeProjection \|\| fineEngineSplits\) \{[^]*Adaptive coarse-grid topology/);
  assert.match(solver,
    /substep > 0[^]*encodeInlineRebuild\(encoder\)[^]*if \(fineEngineSplits\)[^]*CFL substep topology refresh/);
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
