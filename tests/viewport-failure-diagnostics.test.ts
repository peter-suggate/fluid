import assert from "node:assert/strict";
import test from "node:test";
import { defaultCamera, defaultScene, type SceneDescription } from "../lib/model";
import { createMinimalPowerDamBreakScene } from "../lib/scenes";
import type { GPUEulerianInfo } from "../lib/webgpu-eulerian";
import type { WaterSurfacePresentationDiagnostics } from "../lib/webgpu-water-pipeline";
import { projectViewportFailure, viewportFailureIndicator,
  viewportFailureLocation } from "../lib/viewport-failure-diagnostics";

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
    powerDiagramReady: true, powerDiagramAuthoritative: true, powerDiagramGeneration: 7,
    structuredVelocityGeneration: 7, structuredVelocityValid: true,
    structuredBoundaryGeneration: 7, structuredBoundaryValid: true,
    pressureRequiredRows: 12,
    pressureCapacityOverflow: false, pressureRelativeResidual: 1e-5,
    globalFineLevelSetEnabled: true, globalFineLevelSetFactor: 4,
    globalFinePublished: true, globalFineRolledBack: false,
    globalFineSeedError: 0, globalFineTopologyFlags: 0,
    globalFineDownstreamFinalizeReason: 0, globalFineGeneration: 9,
    globalFineRedistanceCommitted: true, globalFineTransportCommitted: true,
    globalFineCoarseLevelSetFlags: 0, encodedSteps: 2,
    ...patch,
  } as GPUEulerianInfo;
}

const retainedWater: WaterSurfacePresentationDiagnostics = {
  surfaceGeometrySource: "retained-previous", globalFineAttached: true,
  globalFineAttachedGeneration: 9, meshPublicationGeneration: 8,
  globalFineCrossingPublished: false, presentationFallbackActive: true,
};

test("viewport alert exposes a rolled-back fine generation and retained mesh", () => {
  const failure = viewportFailureIndicator(healthyInfo({
    globalFinePublished: false, globalFineRolledBack: true,
    globalFineTopologyFlags: 16, globalFineDownstreamFinalizeReason: 12,
  }), retainedWater, scene);
  assert.equal(failure?.id, "pipeline-fine");
  assert.equal(failure?.title, "WATER UPDATE REJECTED");
  assert.match(failure?.detail ?? "", /topology 0x10/);
  assert.match(failure?.detail ?? "",
    /Renderer retained mesh generation 8; live generation 9 was not admitted/);
});

test("a structured-transport rejection retains its exact spatial witness", () => {
  const damBreakScene = createMinimalPowerDamBreakScene();
  const stats = healthyInfo({
    encodedSteps: 22, globalFinePublished: true, globalFineRolledBack: true,
    globalFineTopologyFlags: 16, globalFineDownstreamFinalizeReason: 8,
    globalFineTransportCommitted: false,
    globalFineTransportFirstInvalidVelocityLocalIndex: 3_232,
    globalFineTransportFirstInvalidVelocityPosition_m: {
      x: 0.5062500238418579, y: 0.606249988079071, z: 0.08125000447034834,
    },
  });
  const failure = viewportFailureIndicator(stats, undefined, damBreakScene);
  assert.equal(failure?.id, "pipeline-fine");
  assert.equal(failure?.tone, "rejected");
  assert.equal(failure?.locationLabel, "first invalid velocity sample 3,232");
  assert.deepEqual(failure?.location_m, {
    x: 0.10625002384185789, y: 0.606249988079071, z: -0.31874999552965166,
  });
  assert.equal(projectViewportFailure(failure!.location_m!, {
    ...defaultCamera, distance_m: 1.9, target_m: { x: 0, y: 0.3, z: 0 },
  }, 800, 600).visible, true);
});

test("a retained raster remains visible even when simulation products are healthy", () => {
  const failure = viewportFailureIndicator(healthyInfo(), retainedWater, scene);
  assert.equal(failure?.id, "raster-retained");
  assert.equal(failure?.tone, "rejected");
  assert.equal(failure?.title, "WATER MESH STALE");
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
  const camera = { azimuth_rad: 0, elevation_rad: 0, distance_m: 2,
    target_m: { x: 0, y: 0, z: 0 } };
  assert.deepEqual(projectViewportFailure({ x: 0, y: 0, z: 0 }, camera, 800, 400), {
    leftFraction: 0.5, topFraction: 0.5, visible: true,
  });
  assert.equal(projectViewportFailure({ x: 0, y: 0, z: 3 }, camera, 800, 400).visible, false);
});

test("non-octree and healthy current surfaces do not obscure the viewport", () => {
  assert.equal(viewportFailureIndicator(null, null, scene), undefined);
  const current: WaterSurfacePresentationDiagnostics = {
    ...retainedWater, surfaceGeometrySource: "global-fine-coarse",
    meshPublicationGeneration: 9, globalFineCrossingPublished: true,
    presentationFallbackActive: false,
  };
  assert.equal(viewportFailureIndicator(healthyInfo(), current, scene), undefined);
});
