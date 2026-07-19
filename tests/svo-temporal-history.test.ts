import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_TEMPORAL_HISTORY_THRESHOLDS,
  SVO_TEMPORAL_HIT_KEY_WORDS,
  advanceSvoTemporalConvergence,
  decodeSvoTemporalNormal,
  encodeSvoTemporalNormal,
  evaluateSvoTemporalHistory,
  packSvoTemporalHitKey,
  svoTemporalHistoryWGSL,
  unpackSvoTemporalHitKey,
  type SvoTemporalHitKey,
  type SvoTemporalReprojection,
} from "../lib/svo-temporal-history";

const close = (actual: number, expected: number, tolerance = 1e-5) => {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
};

const closeVector = (actual: readonly number[], expected: readonly number[], tolerance = 5e-5) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => close(value, expected[index], tolerance));
};

const baseKey: SvoTemporalHitKey = {
  depth_m: 2,
  geometricNormal: [0, 1, 0],
  shadingNormal: [0.1, 0.99, 0],
  materialId: 17,
  ownerId: 42,
  mediumBefore: 0,
  mediumAfter: 1,
  localTopologyGeneration: 9,
};

const staticReprojection: SvoTemporalReprojection = {
  cellSize_m: 0.05,
  deltaTime_s: 1 / 60,
  velocity_m_s: [0, 0, 0],
  motionKind: "static",
  reprojectionValid: true,
  motionValid: true,
  reprojectionError_m: 0,
};

function changed(overrides: Partial<SvoTemporalHitKey>): SvoTemporalHitKey {
  return { ...baseKey, ...overrides };
}

test("temporal hit keys pack depth, two normals, identity, media, and local generation into 24 bytes", () => {
  assert.equal(SVO_TEMPORAL_HIT_KEY_WORDS, 6);
  const key = changed({
    geometricNormal: [0.25, -0.5, -0.8291561976],
    shadingNormal: [-0.7, 0.2, 0.68556546],
    materialId: 0xabcd,
    ownerId: 0xfffe,
    mediumBefore: 12,
    mediumAfter: 0xab,
    localTopologyGeneration: 0xffff_fffe,
  });
  const packed = packSvoTemporalHitKey(key);
  assert.equal(packed.byteLength, 24);
  const unpacked = unpackSvoTemporalHitKey(packed);
  close(unpacked.depth_m, key.depth_m);
  closeVector(unpacked.geometricNormal, key.geometricNormal);
  closeVector(unpacked.shadingNormal, key.shadingNormal);
  assert.deepEqual({
    materialId: unpacked.materialId,
    ownerId: unpacked.ownerId,
    mediumBefore: unpacked.mediumBefore,
    mediumAfter: unpacked.mediumAfter,
    localTopologyGeneration: unpacked.localTopologyGeneration,
  }, {
    materialId: key.materialId,
    ownerId: key.ownerId,
    mediumBefore: key.mediumBefore,
    mediumAfter: key.mediumAfter,
    localTopologyGeneration: key.localTopologyGeneration,
  });
  closeVector(decodeSvoTemporalNormal(encodeSvoTemporalNormal([0, 0, -1])), [0, 0, -1]);
});

test("static surfaces retain history within fixed world/cell depth and reprojection thresholds", () => {
  const previous = changed({ depth_m: 2.009 });
  const result = evaluateSvoTemporalHistory(baseKey, previous, { ...staticReprojection, reprojectionError_m: 0.007 });
  assert.equal(result.reason, "accepted");
  assert.equal(result.accepted, true);
  close(result.depthTolerance_m, 0.01);
  close(result.reprojectionTolerance_m, 0.0075);

  const worldFloor = evaluateSvoTemporalHistory(baseKey, changed({ depth_m: 2.0019 }), {
    ...staticReprojection, cellSize_m: 0.001, reprojectionError_m: 0.0009,
  });
  assert.equal(worldFloor.reason, "accepted");
  close(worldFloor.depthTolerance_m, SVO_TEMPORAL_HISTORY_THRESHOLDS.minimumDepthTolerance_m);
  close(worldFloor.reprojectionTolerance_m, SVO_TEMPORAL_HISTORY_THRESHOLDS.minimumReprojectionTolerance_m);
});

test("only a changed local topology generation invalidates an otherwise unchanged region", () => {
  const changedRegion = evaluateSvoTemporalHistory(baseKey, changed({ localTopologyGeneration: 8 }), staticReprojection);
  assert.equal(changedRegion.reason, "topology-generation-change");

  // A scene may publish another generation elsewhere; no scene-global value is
  // part of this comparison, so this unchanged locality retains history.
  const unchangedRegion = evaluateSvoTemporalHistory(baseKey, changed({ localTopologyGeneration: 9 }), staticReprojection);
  assert.equal(unchangedRegion.reason, "accepted");
});

test("disocclusion, material, owner, and medium transitions have deterministic rejection reasons", () => {
  assert.equal(evaluateSvoTemporalHistory(baseKey, changed({ depth_m: 2.011 }), staticReprojection).reason, "disocclusion");
  assert.equal(evaluateSvoTemporalHistory(baseKey, changed({ materialId: 18, depth_m: 10 }), staticReprojection).reason, "material-change");
  assert.equal(evaluateSvoTemporalHistory(baseKey, changed({ ownerId: 43 }), staticReprojection).reason, "owner-change");
  assert.equal(evaluateSvoTemporalHistory(baseKey, changed({ mediumBefore: 1, mediumAfter: 0 }), staticReprojection).reason, "medium-change");
});

test("cube edge and shading-normal changes reject instead of smearing sharp features", () => {
  const adjacentCubeFace = changed({ geometricNormal: [1, 0, 0], shadingNormal: [1, 0, 0] });
  assert.equal(evaluateSvoTemporalHistory(baseKey, adjacentCubeFace, staticReprojection).reason, "geometric-normal-change");

  const shadingOnly = changed({ geometricNormal: [0, 1, 0], shadingNormal: [1, 0, 0] });
  assert.equal(evaluateSvoTemporalHistory(baseKey, shadingOnly, staticReprojection).reason, "shading-normal-change");

  const gentle = changed({ geometricNormal: [0, 0.99, 0.1], shadingNormal: [0.2, 0.97, 0] });
  assert.equal(evaluateSvoTemporalHistory(baseKey, gentle, staticReprojection).reason, "accepted");
});

test("motion and reprojection validity fail closed before history identity comparisons", () => {
  const identityChanged = changed({ materialId: 18 });
  assert.equal(evaluateSvoTemporalHistory(baseKey, identityChanged, {
    ...staticReprojection, reprojectionValid: false,
  }).reason, "invalid-reprojection");
  assert.equal(evaluateSvoTemporalHistory(baseKey, identityChanged, {
    ...staticReprojection, reprojectionError_m: 0.008,
  }).reason, "invalid-reprojection");
  assert.equal(evaluateSvoTemporalHistory(baseKey, identityChanged, {
    ...staticReprojection, motionValid: false,
  }).reason, "invalid-motion");
});

test("moving rigid and fluid history obey source-specific world/cell velocity bounds", () => {
  const common = { ...staticReprojection, cellSize_m: 0.1, deltaTime_s: 0.1 };
  const rigidAccepted = evaluateSvoTemporalHistory(baseKey, baseKey, {
    ...common, motionKind: "rigid", velocity_m_s: [1.9, 0, 0],
  });
  assert.equal(rigidAccepted.reason, "accepted");
  close(rigidAccepted.motionLimit_m, 0.2);
  assert.equal(evaluateSvoTemporalHistory(baseKey, baseKey, {
    ...common, motionKind: "rigid", velocity_m_s: [2.01, 0, 0],
  }).reason, "excessive-motion");

  const fluidAccepted = evaluateSvoTemporalHistory(baseKey, baseKey, {
    ...common, motionKind: "fluid", velocity_m_s: [0.99, 0, 0],
  });
  assert.equal(fluidAccepted.reason, "accepted");
  close(fluidAccepted.motionLimit_m, 0.1);
  assert.equal(evaluateSvoTemporalHistory(baseKey, baseKey, {
    ...common, motionKind: "fluid", velocity_m_s: [1.01, 0, 0],
  }).reason, "excessive-motion");

  const rigidWorldCap = evaluateSvoTemporalHistory(baseKey, baseKey, {
    ...common, cellSize_m: 1, motionKind: "rigid", velocity_m_s: [5, 0, 0],
  });
  assert.equal(rigidWorldCap.reason, "accepted");
  close(rigidWorldCap.motionLimit_m, 0.5);
  assert.equal(evaluateSvoTemporalHistory(baseKey, baseKey, {
    ...common, cellSize_m: 1, motionKind: "rigid", velocity_m_s: [5.01, 0, 0],
  }).reason, "excessive-motion");
});

test("paused convergence counters accumulate, saturate, and reset deterministically", () => {
  let counters = { sampleCount: 0, pausedStableFrames: 0, convergedWhilePaused: false };
  for (let index = 0; index < SVO_TEMPORAL_HISTORY_THRESHOLDS.pausedConvergenceSamples - 1; index += 1) {
    counters = advanceSvoTemporalConvergence(counters, index > 0, true);
  }
  assert.equal(counters.convergedWhilePaused, false);
  counters = advanceSvoTemporalConvergence(counters, true, true);
  assert.equal(counters.pausedStableFrames, SVO_TEMPORAL_HISTORY_THRESHOLDS.pausedConvergenceSamples);
  assert.equal(counters.convergedWhilePaused, true);

  for (let index = 0; index < 300; index += 1) counters = advanceSvoTemporalConvergence(counters, true, true);
  assert.equal(counters.sampleCount, 255);
  assert.equal(counters.pausedStableFrames, 255);
  counters = advanceSvoTemporalConvergence(counters, false, true);
  assert.deepEqual(counters, { sampleCount: 1, pausedStableFrames: 1, convergedWhilePaused: false });
  counters = advanceSvoTemporalConvergence(counters, true, false);
  assert.deepEqual(counters, { sampleCount: 2, pausedStableFrames: 0, convergedWhilePaused: false });
});

test("invalid compact keys and reprojection inputs are rejected at the contract boundary", () => {
  assert.throws(() => packSvoTemporalHitKey(changed({ geometricNormal: [0, 0, 0] })), /nonzero/);
  assert.throws(() => packSvoTemporalHitKey(changed({ ownerId: 0x1_0000 })), /uint16/);
  assert.throws(() => unpackSvoTemporalHitKey(new Uint32Array(5)), /6 words/);
  assert.equal(evaluateSvoTemporalHistory(changed({ depth_m: Number.POSITIVE_INFINITY }), baseKey, staticReprojection).reason, "invalid-current-hit");
  assert.equal(evaluateSvoTemporalHistory(baseKey, changed({ shadingNormal: [0, 0, 0] }), staticReprojection).reason, "invalid-previous-hit");
  assert.throws(() => evaluateSvoTemporalHistory(baseKey, baseKey, { ...staticReprojection, cellSize_m: 0 }), /cell size/);
  assert.throws(() => evaluateSvoTemporalHistory(baseKey, baseKey, { ...staticReprojection, reprojectionError_m: -1 }), /reprojection error/);
});

test("WGSL mirror is binding-free and carries the compact layout, thresholds, and rejection gates", () => {
  assert.doesNotMatch(svoTemporalHistoryWGSL, /@group|@binding/);
  assert.match(svoTemporalHistoryWGSL, /struct SvoTemporalHitKey \{[\s\S]*depth_m:f32,[\s\S]*geometricNormalOct:u32,[\s\S]*shadingNormalOct:u32,[\s\S]*materialOwner:u32,[\s\S]*media:u32,[\s\S]*localTopologyGeneration:u32/);
  assert.match(svoTemporalHistoryWGSL, /const SVO_TEMPORAL_MIN_DEPTH_M:f32=0\.002/);
  assert.match(svoTemporalHistoryWGSL, /const SVO_TEMPORAL_DEPTH_CELL:f32=0\.2/);
  assert.match(svoTemporalHistoryWGSL, /const SVO_TEMPORAL_RIGID_MOTION_CELL:f32=2\.0/);
  assert.match(svoTemporalHistoryWGSL, /const SVO_TEMPORAL_FLUID_MOTION_CELL:f32=1\.0/);
  assert.match(svoTemporalHistoryWGSL, /current\.localTopologyGeneration!=previous\.localTopologyGeneration/);
  assert.match(svoTemporalHistoryWGSL, /current\.media!=previous\.media/);
  assert.match(svoTemporalHistoryWGSL, /SVO_TEMPORAL_REASON_DISOCCLUSION/);
  assert.match(svoTemporalHistoryWGSL, /SVO_TEMPORAL_REASON_GEOMETRIC_NORMAL/);
  assert.match(svoTemporalHistoryWGSL, /SVO_TEMPORAL_REASON_SHADING_NORMAL/);
});
