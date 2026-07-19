import assert from "node:assert/strict";
import test from "node:test";
import { canonicalScene } from "../lib/model";
import { cameraForPreset, getScenePreset, scenePresets } from "../lib/scenes";
import { SPARSE_BRICK_GPU_LAYOUT } from "../lib/sparse-brick-octree";
import {
  canonicalSVOBaselineCase,
  SVO_BASELINE_ACCEPTANCE_AREAS,
  SVO_BASELINE_CASES,
  SVO_BASELINE_DEFAULTS,
  svoBaselineArtifactDirectory,
  svoBaselineArtifactManifestPath,
  svoBaselineArtifactPath,
  type SVOBaselineCase,
} from "../tools/svo-baseline-cases";
import {
  SVO_BASELINE_ADAPTER_ASSUMPTIONS,
  SVO_BASELINE_ARTIFACTS,
  SVO_BASELINE_RENDERER_PROFILES,
  SVO_BASELINE_REQUIRED_LIMITS,
  SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS,
  SVO_BASELINE_TOLERANCES,
  buildSVOBaselineCaptureJobs,
  buildSVOBaselineManifest,
  summarizeSVOBaselineTimings,
  validateSVOBaselineAdapterLimits,
  type SVOBaselineAdapterObservation,
} from "../tools/svo-baseline-contract";
import { createSmokeScenario, smokeScenarioIds } from "../tools/webgpu-smoke-scenarios";

test("M0 SVO baseline matrix covers every acceptance area and required variant", () => {
  assert.deepEqual(
    [...new Set(SVO_BASELINE_CASES.map((entry) => entry.acceptanceArea))].sort(),
    [...SVO_BASELINE_ACCEPTANCE_AREAS].sort(),
  );
  const variants = (area: SVOBaselineCase["acceptanceArea"]) =>
    SVO_BASELINE_CASES.filter((entry) => entry.acceptanceArea === area).map((entry) => entry.variant).sort();
  assert.deepEqual(variants("dam-break"), ["bodies", "empty"]);
  assert.deepEqual(variants("primitive-curvature-and-edges"), ["cube-highlight", "sphere-highlight"]);
  assert.deepEqual(variants("rigid-body-submersion"), ["full", "partial"]);
  assert.deepEqual(variants("thin-glass"), ["grazing-incidence", "normal-incidence"]);
});

test("M0 SVO baseline identities and sources are unique and resolve through existing catalogs", () => {
  const ids = SVO_BASELINE_CASES.map((entry) => entry.id);
  assert.equal(new Set(ids).size, ids.length);
  for (const entry of SVO_BASELINE_CASES) {
    assert.equal(entry.id, `${entry.acceptanceArea}--${entry.variant}`);
    if (entry.source.kind === "smoke-scenario") {
      assert.ok(smokeScenarioIds.includes(entry.source.id));
      assert.ok(createSmokeScenario(entry.source.id).scene.sceneId.length > 0);
    } else {
      assert.ok(scenePresets.some((preset) => preset.id === entry.source.id));
      assert.equal(getScenePreset(entry.source.id).id, entry.source.id);
    }
  }
});

test("M0 SVO baseline checkpoints are exact fixed-step checkpoints", () => {
  for (const entry of SVO_BASELINE_CASES) {
    const scene = entry.source.kind === "smoke-scenario"
      ? createSmokeScenario(entry.source.id).scene
      : getScenePreset(entry.source.id).create();
    const expected = entry.checkpoint.stepCount * scene.numerics.fixedDt_s;
    assert.ok(
      Math.abs(entry.checkpoint.simulatedTime_s - expected) <= 1e-12,
      `${entry.id}: ${entry.checkpoint.stepCount} × ${scene.numerics.fixedDt_s} != ${entry.checkpoint.simulatedTime_s}`,
    );
  }
});

test("M0 SVO baseline defaults, cameras, and output dimensions are explicit and finite", () => {
  assert.deepEqual(SVO_BASELINE_DEFAULTS, {
    methodId: "octree",
    quality: "balanced",
    renderer: "raster",
    outputResolution: { width: 1280, height: 720 },
  });
  for (const entry of SVO_BASELINE_CASES) {
    assert.equal(entry.methodId, "octree");
    assert.equal(entry.quality, "balanced");
    assert.equal(entry.renderer, "raster");
    assert.ok(entry.checkpoint.stepCount > 0 && Number.isInteger(entry.checkpoint.stepCount));
    assert.ok(entry.outputResolution.width > 0 && entry.outputResolution.height > 0);
    assert.ok([
      entry.camera.azimuth_rad,
      entry.camera.elevation_rad,
      entry.camera.distance_m,
      entry.camera.target_m.x,
      entry.camera.target_m.y,
      entry.camera.target_m.z,
    ].every(Number.isFinite));
  }

  const settled = SVO_BASELINE_CASES.find((entry) => entry.id === "settled-tank--default")!;
  assert.deepEqual(settled.camera, cameraForPreset(getScenePreset("water-box-tank-fill")));
  const grazing = SVO_BASELINE_CASES.find((entry) => entry.id === "thin-glass--grazing-incidence")!;
  assert.notDeepEqual(grazing.camera, cameraForPreset(getScenePreset("water-box-tank-fill")));
});

test("M0 SVO baseline metadata and referenced scenes have canonical stable identities", () => {
  const metadata = SVO_BASELINE_CASES.map(canonicalSVOBaselineCase);
  assert.equal(new Set(metadata).size, SVO_BASELINE_CASES.length);
  for (let index = 0; index < SVO_BASELINE_CASES.length; index += 1) {
    const entry = SVO_BASELINE_CASES[index];
    const cloned = JSON.parse(JSON.stringify(entry)) as SVOBaselineCase;
    assert.equal(canonicalSVOBaselineCase(cloned), metadata[index]);

    const sceneA = entry.source.kind === "smoke-scenario"
      ? createSmokeScenario(entry.source.id).scene
      : getScenePreset(entry.source.id).create();
    const sceneB = entry.source.kind === "smoke-scenario"
      ? createSmokeScenario(entry.source.id).scene
      : getScenePreset(entry.source.id).create();
    assert.equal(canonicalScene(sceneA), canonicalScene(sceneB), entry.id);
  }
});

test("M0 SVO artifact paths encode every comparison dimension with safe stable segments", () => {
  const baseline = SVO_BASELINE_CASES.find((entry) => entry.id === "thin-glass--grazing-incidence")!;
  assert.equal(
    svoBaselineArtifactManifestPath(baseline, {
      revision: "A1B2C3-dirty",
      adapter: "Apple M3 Max / Metal",
      renderer: "raster",
      quality: "balanced",
      internalResolution: { width: 960, height: 540 },
    }),
    "artifacts/svo-baseline/a1b2c3-dirty/apple-m3-max-metal/thin-glass--grazing-incidence/" +
      "balanced/raster/output-1280x720__internal-960x540/t100ms/manifest.json",
  );
  assert.throws(() => svoBaselineArtifactManifestPath(baseline, {
    revision: "../", adapter: "Metal", renderer: "svo", quality: "high",
    internalResolution: { width: 0, height: 540 },
  }), /path-safe|positive integer/);
  const context = {
    revision: "A1B2C3-dirty", adapter: "Apple M3 Max / Metal", renderer: "svo" as const, quality: "balanced" as const,
    internalResolution: { width: 960, height: 540 },
  };
  assert.equal(svoBaselineArtifactDirectory(baseline, context),
    "artifacts/svo-baseline/a1b2c3-dirty/apple-m3-max-metal/thin-glass--grazing-incidence/" +
      "balanced/svo/output-1280x720__internal-960x540/t100ms");
  assert.equal(svoBaselineArtifactPath(baseline, context, "depth-f32.bin"),
    `${svoBaselineArtifactDirectory(baseline, context)}/depth-f32.bin`);
  assert.throws(() => svoBaselineArtifactPath(baseline, context, "../depth.bin"), /safe path segment/);
});

test("M0 capture contract names the shipped hybrid path and every durable signal", () => {
  assert.deepEqual(SVO_BASELINE_RENDERER_PROFILES.svo, {
    id: "svo", requestedRenderMode: "svo", dryScenePath: "svo-direct", waterPath: "raster", urlRenderValue: "svo",
  });
  assert.equal(SVO_BASELINE_RENDERER_PROFILES.raster.dryScenePath, "raster");
  assert.deepEqual(Object.keys(SVO_BASELINE_ARTIFACTS).sort(), [
    "camera", "color", "depth", "energy", "geometricNormal", "identityMedia", "manifest", "scene", "timings",
  ]);
  assert.equal(SVO_BASELINE_ARTIFACTS.depth.requiredNow, false,
    "the contract is durable now while hardware readback wiring remains an explicit gap");
  assert.equal(SVO_BASELINE_TOLERANCES.identity.materialIdMismatchPixels, 0);
  assert.equal(SVO_BASELINE_TOLERANCES.identity.ownerIdMismatchPixels, 0);
  assert.equal(SVO_BASELINE_TOLERANCES.energyLinearRgb.meanLuminanceRelative, 0.05);
  assert.equal(SVO_BASELINE_TOLERANCES.performance.warmupFrames, 30);
  assert.equal(SVO_BASELINE_TOLERANCES.performance.measuredFrames, 120);
});

function baselineAdapter(overrides: Partial<Record<keyof typeof SVO_BASELINE_REQUIRED_LIMITS, number>> = {}): SVOBaselineAdapterObservation {
  return {
    name: "Apple M3 Max", backend: "metal", features: ["timestamp-query", "float32-filterable"],
    limits: { ...SVO_BASELINE_REQUIRED_LIMITS, ...overrides },
  };
}

test("M0 adapter contract records primary hardware and rejects every undersized required limit", () => {
  assert.equal(SVO_BASELINE_ADAPTER_ASSUMPTIONS[0].id, "apple-m3-max-metal");
  assert.equal(SVO_BASELINE_ADAPTER_ASSUMPTIONS[0].performanceGatesApply, true);
  assert.equal(SVO_BASELINE_REQUIRED_LIMITS.maxStorageBuffersPerShaderStage, 10);
  assert.equal(SVO_BASELINE_REQUIRED_LIMITS.maxColorAttachmentBytesPerSample, 32);
  assert.deepEqual(validateSVOBaselineAdapterLimits(baselineAdapter()), []);
  assert.deepEqual(validateSVOBaselineAdapterLimits(baselineAdapter({ maxStorageBuffersPerShaderStage: 9 })), [
    "maxStorageBuffersPerShaderStage: 9 < 10",
  ]);
});

test("M0 sparse layout assumptions pin strides, control words, and indirect offsets", () => {
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.nodeStrideBytes, SPARSE_BRICK_GPU_LAYOUT.nodeStrideBytes);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.leafStrideBytes, SPARSE_BRICK_GPU_LAYOUT.leafStrideBytes);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.geometryStrideBytes, SPARSE_BRICK_GPU_LAYOUT.geometryStrideBytes);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.velocityStrideBytes, SPARSE_BRICK_GPU_LAYOUT.velocityStrideBytes);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.materialOwnerStrideBytes, SPARSE_BRICK_GPU_LAYOUT.materialOwnerStrideBytes);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.controlStrideBytes, SPARSE_BRICK_GPU_LAYOUT.controlStrideBytes);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.payloadOffsets.controlLeafWordOffset, SPARSE_BRICK_GPU_LAYOUT.controlWords.leafWordOffset);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.payloadOffsets.controlVelocityWordOffset, SPARSE_BRICK_GPU_LAYOUT.controlWords.velocityWordOffset);
  assert.equal(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.payloadOffsets.controlMaterialOwnerWordOffset, SPARSE_BRICK_GPU_LAYOUT.controlWords.materialOwnerWordOffset);
  assert.deepEqual(SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS.indirectOffsetsBytes, {
    dispatch: SPARSE_BRICK_GPU_LAYOUT.dispatchIndirectOffsetBytes,
    draw: SPARSE_BRICK_GPU_LAYOUT.drawIndirectOffsetBytes,
  });
});

test("M0 capture jobs pair every deterministic case across raster and hybrid SVO", () => {
  const jobs = buildSVOBaselineCaptureJobs({
    baseUrl: "http://localhost:3003/",
    revision: "deadbeef",
    adapter: "Apple M3 Max Metal",
    internalResolution: { raster: { width: 1280, height: 720 }, svo: { width: 960, height: 540 } },
  });
  assert.equal(jobs.length, SVO_BASELINE_CASES.length * 2);
  assert.equal(new Set(jobs.map((job) => job.id)).size, jobs.length);
  const raster = jobs.find((job) => job.id === "thin-glass--grazing-incidence--raster")!;
  const hybrid = jobs.find((job) => job.id === "thin-glass--grazing-incidence--svo")!;
  assert.equal(new URL(raster.applicationUrl).searchParams.get("render"), "raster");
  assert.equal(new URL(hybrid.applicationUrl).searchParams.get("render"), "svo");
  assert.equal(new URL(hybrid.applicationUrl).searchParams.get("voxels"), "smooth");
  assert.equal(new URL(hybrid.applicationUrl).searchParams.get("camera.azimuth"), "1.48");
  assert.match(hybrid.artifacts.color, /\/balanced\/svo\/.*\/color\.png$/);
  assert.match(raster.artifacts.timings, /\/balanced\/raster\/.*\/timings\.json$/);
});

test("M0 timing summaries exclude warmup and retain raw timestamp availability", () => {
  const samples = Array.from({ length: 150 }, (_, index) => ({
    cpuFrame_ms: index,
    gpuRender_ms: index / 10,
    gpuDryScene_ms: index / 20,
    gpuRenderTimingAvailable: true,
  }));
  const timing = summarizeSVOBaselineTimings(samples);
  assert.equal(timing.warmupFrames, 30);
  assert.equal(timing.measuredFrames, 120);
  assert.equal(timing.gpuRender_ms?.minimum, 3);
  assert.equal(timing.gpuRender_ms?.maximum, 14.9);
  assert.ok(Math.abs(timing.gpuRender_ms!.p95 - 14.305) < 1e-12);
  const unavailable = summarizeSVOBaselineTimings(samples.map((sample, index) => ({
    ...sample, gpuRenderTimingAvailable: index !== 149,
  })));
  assert.equal(unavailable.timestampQueriesAvailable, false);
  assert.equal(unavailable.gpuRender_ms, null);
});

test("M0 manifests fail closed on missing required captures and preserve outstanding readbacks", () => {
  const [job] = buildSVOBaselineCaptureJobs({
    baseUrl: "http://localhost:3003/", revision: "deadbeef", adapter: "Apple M3 Max Metal",
    internalResolution: { raster: { width: 1280, height: 720 }, svo: { width: 960, height: 540 } },
    cases: [SVO_BASELINE_CASES[0]],
  });
  assert.throws(() => buildSVOBaselineManifest({
    job, adapter: baselineAdapter(), status: "captured", availableArtifacts: ["color"],
  }), /missing required artifact timings/);
  const manifest = buildSVOBaselineManifest({
    job, adapter: baselineAdapter(), status: "captured", rendererOwnedBytes: 1234,
    availableArtifacts: ["color", "timings", "scene", "camera"],
  });
  assert.equal(manifest.rendererOwnedBytes, 1234);
  assert.deepEqual(manifest.adapterLimitFailures, []);
  assert.deepEqual(manifest.outstandingSignals, ["depth", "geometricNormal", "identityMedia", "energy"]);
});
