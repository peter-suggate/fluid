import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { defaultCamera, defaultScene, type SceneDescription } from "../lib/model";
import { createMinimalPowerDamBreakScene } from "../lib/scenes";
import type { GPUEulerianInfo } from "../lib/webgpu-eulerian";
import type { WaterSurfacePresentationDiagnostics } from "../lib/webgpu-water-pipeline";
import {
  projectViewportFailure,
  viewportFailureIndicator,
  viewportFailureLocation,
} from "../lib/viewport-failure-diagnostics";

const scene: SceneDescription = {
  ...defaultScene,
  container: { ...defaultScene.container, width_m: 1, height_m: 1, depth_m: 1 },
};

function healthyInfo(patch: Partial<GPUEulerianInfo> = {}): GPUEulerianInfo {
  return {
    nx: 16, ny: 16, nz: 16, storedNy: 16, cellCount: 4096,
    equivalentUniformCells: 4096, compressionRatio: 1, regularLayers: 16,
    maximumNeighborDelta: 1, gridKind: "octree", cellSize_m: 1 / 16,
    pressureIterations: 4, pressureSolver: "Section 4.3 hybrid MGPCG",
    allocatedBytes: 1, initialSparseAuthorityReady: true,
    powerDiagramProjection: "authoritative", powerDiagramReady: true,
    powerDiagramAuthoritative: true, powerDiagramGeneration: 7,
    pressureRequiredRows: 12, pressureRequiredEntries: 24,
    pressureCapacityOverflow: false, pressureRelativeResidual: 1e-5,
    globalFineLevelSetEnabled: true, globalFineLevelSetFactor: 4,
    globalFinePublished: true, globalFineRolledBack: false,
    globalFineSeedError: 0, globalFineTopologyFlags: 0,
    globalFineDownstreamFinalizeReason: 0, globalFineGeneration: 9,
    globalFineRedistanceCommitted: true, globalFineTransportCommitted: true,
    globalFineFaceBandFlags: 0, globalFineFaceBandTransitionFlags: 0,
    globalFineFaceBandTransientPowerFlags: 0, globalFineFaceBandPointFieldFlags: 0,
    globalFineFaceBandPowerPublicationFlags: 0, globalFineCoarseLevelSetFlags: 0,
    globalFineFaceBandGeneration: 9, globalFineFaceBandValid: true,
    globalFineFaceBandTransitionValid: true, globalFineFaceBandTransientPowerValid: true,
    globalFineFaceBandPointFieldValid: true, globalFineFaceBandPowerPublicationValid: true,
    globalFineFaceBandPowerFineGeneration: 9, globalFineFaceBandPowerGeneration: 7,
    encodedSteps: 2,
    ...patch,
  } as GPUEulerianInfo;
}

const retainedWater: WaterSurfacePresentationDiagnostics = {
  surfaceGeometrySource: "retained-previous",
  globalFineAttached: true,
  globalFineAttachedGeneration: 9,
  meshPublicationGeneration: 8,
  globalFineCrossingPublished: false,
  presentationFallbackActive: true,
};

test("viewport failure alert exposes a rolled-back paper generation and retained mesh", () => {
  const failure = viewportFailureIndicator(healthyInfo({
    globalFinePublished: false,
    globalFineRolledBack: true,
    globalFineTopologyFlags: 16,
    globalFineDownstreamFinalizeReason: 12,
  }), retainedWater, scene);
  assert.equal(failure?.id, "pipeline-fine");
  assert.equal(failure?.title, "WATER UPDATE REJECTED");
  assert.match(failure?.detail ?? "", /topology 0x10/);
  assert.match(failure?.detail ?? "", /Renderer retained mesh generation 8; live generation 9 was not admitted/);
});

test("step 22 t=.088 Dawn rejection is the exact persistent UI failure packet", () => {
  // Captured by the every-step Dawn audit for minimal-power-dam-break. Keep
  // this as a CPU-only fixture: it proves that a browser readStats snapshot
  // containing the same controls cannot silently become a different alert or
  // lose the spatial witness while the rejected controls remain latched.
  const damBreakScene = createMinimalPowerDamBreakScene();
  const step22 = {
    step: 22,
    requestedTime_s: 0.088,
    submittedTime_s: 0.088,
    stats: healthyInfo({
      encodedSteps: 22,
      globalFinePublished: true,
      globalFineRolledBack: true,
      globalFineTopologyFlags: 16,
      globalFineDownstreamFinalizeReason: 8,
      globalFineTransportCommitted: false,
      globalFineTransportFirstInvalidVelocityLocalIndex: 3_232,
      globalFineTransportFirstInvalidVelocityPosition_m: {
        x: 0.5062500238418579,
        y: 0.606249988079071,
        z: 0.08125000447034834,
      },
    }),
    powerFaces: {
      flags: 8,
      firstInvalid: 669,
      invalidCount: 5,
      firstInvalidSlot: 3,
      firstInvalidNeighbor: 0,
      firstInvalidDetail: 2_048,
      firstInvalidRow: 669,
      faceCount: 0,
      incidenceCount: 0,
    },
  } as const;
  const expected = {
    id: "pipeline-fine",
    tone: "rejected",
    title: "WATER UPDATE REJECTED",
    stage: "§5 · Fine φ interface & support band",
    detail: "seed fault 0 · topology 0x10 · downstream 0x8 The rejected generation is not visible; presentation remains on the last admitted mesh.",
    location_m: {
      x: 0.10625002384185789,
      y: 0.606249988079071,
      z: -0.31874999552965166,
    },
    locationLabel: "first invalid velocity sample 3,232",
  } as const;

  assert.equal(step22.powerFaces.firstInvalid, step22.powerFaces.firstInvalidRow,
    "the ordered core power-face failure row must not be confused with a padding word");
  const uiFailure = viewportFailureIndicator(step22.stats, undefined, damBreakScene);
  // Dawn calls this same helper on a spread readStats result. Repeating that
  // boundary here catches accidental dependence on object identity or UI-only
  // store state as well as changes to the stable serialized packet.
  const dawnFailure = viewportFailureIndicator({ ...step22.stats }, undefined, damBreakScene);
  assert.deepEqual(uiFailure, expected);
  assert.deepEqual(dawnFailure, expected);
  assert.deepEqual(dawnFailure, uiFailure);

  // The alert and its crosshair remain present on every normal UI render for
  // as long as readStats continues to report the latched rejection.
  assert.deepEqual(viewportFailureIndicator(step22.stats, undefined, damBreakScene), expected);
  assert.equal(projectViewportFailure(expected.location_m, {
    ...defaultCamera,
    distance_m: 1.9,
    target_m: { x: 0, y: 0.3, z: 0 },
  }, 800, 600).visible, true, "the preset camera must draw the failure crosshair, not only its alert text");
});

test("a retained raster remains a visible failure even when the paper products are healthy", () => {
  const failure = viewportFailureIndicator(healthyInfo(), retainedWater, scene);
  assert.equal(failure?.id, "raster-retained");
  assert.equal(failure?.tone, "rejected");
  assert.equal(failure?.title, "WATER MESH STALE");
});

test("a recorded failed face is converted from grid coordinates to world space", () => {
  const result = viewportFailureLocation(healthyInfo({
    globalFineFaceBandPhiFailure: {
      cause: 0, faceIndex: 3, globalFace: 91, negativeRow: 1, positiveRow: 2,
      anchorRow: 1, centroid: [8, 4, 12], interpolantPath: 2,
      missingOrigin: [8, 4, 12], missingSize: 1, selectorOrCorner: 0, detail: 0,
    },
  }), scene);
  assert.deepEqual(result.location_m, { x: 0, y: 0.25, z: 0.25 });
  assert.equal(result.locationLabel, "first failed face 91");
});

test("an acute-grading row supplies a cell-centred world-space witness", () => {
  const rowCell = 2 + 16 * (3 + 16 * 4);
  const result = viewportFailureLocation(healthyInfo({
    globalFineFaceBandAcuteGradingFailure: { band: 5, rowCell, rowSize: 2, descriptor: 1, coarseMask: 4 },
  }), scene);
  assert.deepEqual(result.location_m, { x: -0.3125, y: 0.25, z: -0.1875 });
  assert.equal(result.locationLabel, `first failed row ${rowCell.toLocaleString()}`);
});

test("the exact transport witness maps from solver-local metres into world space", () => {
  const result = viewportFailureLocation(healthyInfo({
    globalFineTransportFirstInvalidVelocityLocalIndex: 73,
    globalFineTransportFirstInvalidVelocityPosition_m: { x: 0.6, y: 0.35, z: 0.1 },
  }), scene);
  assert.deepEqual(result.location_m, { x: 0.09999999999999998, y: 0.35, z: -0.4 });
  assert.equal(result.locationLabel, "first invalid velocity sample 73");
});

test("world-space witness projection follows the raster camera convention", () => {
  const camera = { azimuth_rad: 0, elevation_rad: 0, distance_m: 2, target_m: { x: 0, y: 0, z: 0 } };
  assert.deepEqual(projectViewportFailure({ x: 0, y: 0, z: 0 }, camera, 800, 400), {
    leftFraction: 0.5, topFraction: 0.5, visible: true,
  });
  assert.equal(projectViewportFailure({ x: 0, y: 0, z: 3 }, camera, 800, 400).visible, false);
});

test("non-octree and healthy current surfaces do not obscure the viewport", () => {
  assert.equal(viewportFailureIndicator(null, null, scene), undefined);
  const current: WaterSurfacePresentationDiagnostics = {
    ...retainedWater,
    surfaceGeometrySource: "global-fine-coarse",
    meshPublicationGeneration: 9,
    globalFineCrossingPublished: true,
    presentationFallbackActive: false,
  };
  assert.equal(viewportFailureIndicator(healthyInfo(), current, scene), undefined);
});

test("the WebGPU viewport renders the failure alert, spatial marker, and diagnostics action", () => {
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const solver = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  const styles = readFileSync(new URL("../app/globals.css", import.meta.url), "utf8");
  assert.match(viewport, /viewportFailureIndicator\(gpuInfo, waterSurfacePresentation, scene\)/);
  assert.match(viewport, /data-testid="viewport-failure-alert"/);
  assert.match(viewport, /data-testid="viewport-failure-marker"/);
  assert.match(viewport, /setDiagnosticsOpen\(true\)/);
  assert.match(styles, /\.viewport-failure-alert\s*\{/);
  assert.match(styles, /\.viewport-failure-marker\s*\{/);
  assert.match(renderer, /now_ms-this\.lastGPUReadbackAt_ms>=250/);
  assert.match(renderer, /fluid\.readStats\(\)\.then\(info=>this\.gpuInfoCallback\?\.\(\{\.\.\.info\}\)\)/,
    "the normal renderer must publish queue-fenced solver failures without a diagnostics-panel gate");
  assert.match(solver, /if\(globalFineDiagnostics\)this\.applyGlobalFineDiagnostics\(globalFineDiagnostics\)/,
    "the normal UI readStats path must apply the same fine-publication controls used by Dawn");
  assert.match(solver, /info\.globalFineTransportFirstInvalidVelocityLocalIndex = captured \? first : undefined/,
    "the normal UI readback must retain the exact invalid transport-sample index");
  assert.match(solver, /info\.globalFineTransportFirstInvalidVelocityPosition_m = captured/,
    "the normal UI readback must retain the exact invalid transport-sample location");
});

test("Dawn power failures decode the same stable viewport stage, detail, and location packet", () => {
  const smoke = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke, /viewportFailureIndicator\(\{ \.\.\.await solver\.readStats\(\) \}, undefined, scene\)/);
  assert.match(smoke, /JSON\.stringify\(\{ uiFailure, \.\.\.audit/);
  assert.match(smoke, /firstInvalidRow: face\[4\], firstInvalidPad3: face\[15\]/,
    "Dawn must use the ordered failure row rather than the stage-specific pad as its location");
});
