import assert from "node:assert/strict";
import test from "node:test";

import {
  adaptSvoStructuralFluidVisibilityToMediaStep,
  DEFAULT_SVO_FLUID_MEDIA_HANDOFF,
  resolveSvoFluidRenderOwnership,
  SVO_FLUID_MEDIA_PATH_LIMITS,
  svoFluidMediaPathWGSL,
  traceSvoStructuralFluidMediaPath,
  type SvoFluidMediaQuery,
  type SvoFluidMediaSceneSource,
  type SvoFluidMediaWaterSource,
} from "../lib/svo-fluid-media-path";
import { WATER_OPTICS } from "../lib/webgpu-lighting";

function slabWater(top = 0, bottom = -2, generation = 7): SvoFluidMediaWaterSource {
  return ({ ray, currentMedium }) => {
    const insideFluidAtStart = currentMedium === "water";
    const candidate = currentMedium === "water"
      ? (ray.direction[1] >= 0 ? { y: top, normal: [0, 1, 0] as const } : { y: bottom, normal: [0, -1, 0] as const })
      : (ray.direction[1] < 0 && ray.origin_m[1] >= top ? { y: top, normal: [0, 1, 0] as const } : undefined);
    if (!candidate || Math.abs(ray.direction[1]) < 1e-12) {
      return { status: "miss", insideFluidAtStart, completeGeneration: generation, coarseFluidRevision: 3, steps: 1, nodeVisits: 8 };
    }
    const t_m = (candidate.y - ray.origin_m[1]) / ray.direction[1];
    return t_m >= 0 && t_m <= ray.maximumDistance_m
      ? { status: "hit", t_m, normal: [...candidate.normal], boundaryId: 91, insideFluidAtStart,
        completeGeneration: generation, coarseFluidRevision: 3, steps: 2, nodeVisits: 16 }
      : { status: "miss", insideFluidAtStart, completeGeneration: generation, coarseFluidRevision: 3, steps: 2, nodeVisits: 16 };
  };
}

function planeScene(y: number): SvoFluidMediaSceneSource {
  return ({ ray }) => {
    if (Math.abs(ray.direction[1]) < 1e-12) return { status: "miss" };
    const t_m = (y - ray.origin_m[1]) / ray.direction[1];
    return t_m >= 0 && t_m <= ray.maximumDistance_m
      ? { status: "hit", boundaries: [{ t_m, medium: "opaque", geometricNormal: [0, 1, 0], boundaryId: 12 }] }
      : { status: "miss" };
  };
}

test("legacy water remains default and the validated handoff suppresses all legacy stages atomically", () => {
  assert.equal(DEFAULT_SVO_FLUID_MEDIA_HANDOFF, "legacy-water");
  assert.deepEqual(resolveSvoFluidRenderOwnership(), {
    effective: "legacy-water", directStructuralMedia: false,
    legacyExtraction: true, legacyInterfaces: true, legacyComposite: true,
  });
  assert.equal(resolveSvoFluidRenderOwnership("direct-structural-media").fallbackReason, "direct-media-not-validated");
  assert.deepEqual(resolveSvoFluidRenderOwnership("direct-structural-media", true), {
    effective: "direct-structural-media", directStructuralMedia: true,
    legacyExtraction: false, legacyInterfaces: false, legacyComposite: false,
  });
});

test("air-to-water-to-air traversal preserves two-metre thickness, smooth normals, Beer absorption, and bounded in-scattering", () => {
  const result = traceSvoStructuralFluidMediaPath(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 5 },
    slabWater(), undefined, { continuationEpsilon_m: 0, waterInscatterLinear: [0.2, 0.3, 0.4] },
  );
  assert.equal(result.status, "water-segment");
  assert.ok(Math.abs(result.waterThickness_m - 2) < 1e-9);
  assert.deepEqual(result.entryNormal, [0, 1, 0]);
  assert.deepEqual(result.exitNormal, [0, -1, 0]);
  assert.equal(result.currentMedium, "air");
  const interfaceTransmission = (1 - WATER_OPTICS.fresnelF0) ** 2;
  for (let channel = 0; channel < 3; channel += 1) {
    const expected = interfaceTransmission * Math.exp(-WATER_OPTICS.absorption[channel] * 2);
    assert.ok(Math.abs(result.throughput[channel] - expected) < 5e-5);
    assert.ok(result.inscatteredRadiance[channel] > 0);
    assert.ok(result.inscatteredRadiance[channel] < [0.2, 0.3, 0.4][channel]);
  }
  assert.equal(result.counts.transitions, 2);
  assert.equal(result.counts.transmissions, 2);
  assert.equal(result.completeGeneration, 7);
});

test("an underwater opaque hit terminates as a submerged contact after exact water thickness", () => {
  const result = traceSvoStructuralFluidMediaPath(
    { origin_m: [0, -0.5, 0], direction: [0, -1, 0], maximumDistance_m: 4 },
    slabWater(), planeScene(-1.25), { initialMedium: "water", continuationEpsilon_m: 0 },
  );
  assert.equal(result.status, "opaque-contact");
  if (result.status !== "opaque-contact") return;
  assert.equal(result.submerged, true);
  assert.ok(Math.abs(result.waterThickness_m - 0.75) < 1e-12);
  assert.equal(result.currentMedium, "water");
  assert.ok(Math.abs(result.throughput[0] - Math.exp(-WATER_OPTICS.absorption[0] * 0.75)) < 1e-12);
});

test("water/glass boundaries inside the coincidence epsilon resolve as one water-to-glass interface", () => {
  const water = slabWater();
  const glass: SvoFluidMediaSceneSource = (query) => {
    if (query.currentMedium !== "water") return { status: "miss" };
    const exitT = (-2 - query.ray.origin_m[1]) / query.ray.direction[1];
    return { status: "hit", boundaries: [{
      t_m: exitT + 5e-6, medium: "glass", geometricNormal: [0, 1, 0], boundaryId: 72,
    }] };
  };
  const result = traceSvoStructuralFluidMediaPath(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 5 }, water, glass,
    { continuationEpsilon_m: 0, coincidentBoundaryEpsilon_m: 1e-5 },
  );
  assert.equal(result.status, "water-segment");
  assert.equal(result.currentMedium, "glass");
  assert.equal(result.counts.transitions, 2, "the coincidence group adds one interface, not water-air plus air-glass");
});

test("total internal reflection keeps the water stack and continues to a submerged solid", () => {
  const sine = 0.9;
  const result = traceSvoStructuralFluidMediaPath(
    { origin_m: [0, -0.5, 0], direction: [sine, Math.sqrt(1 - sine ** 2), 0], maximumDistance_m: 20 },
    slabWater(), planeScene(-1.5), { initialMedium: "water", continuationEpsilon_m: 1e-6 },
  );
  assert.equal(result.status, "opaque-contact");
  if (result.status !== "opaque-contact") return;
  assert.equal(result.submerged, true);
  assert.equal(result.counts.reflections, 1);
  assert.equal(result.counts.transmissions, 0);
  assert.ok(result.direction[1] < 0, "TIR reflected the path back into water");
  assert.ok(result.waterThickness_m > 1.5);
});

test("stale, nonresident, and exhausted structural queries fail closed with exact status", () => {
  const staleSource: SvoFluidMediaWaterSource = (query: SvoFluidMediaQuery) => {
    const base = slabWater()(query);
    return query.queryIndex === 0 ? base : { ...base, completeGeneration: 8 };
  };
  const stale = traceSvoStructuralFluidMediaPath(
    { origin_m: [0, 1, 0], direction: [0, -1, 0], maximumDistance_m: 5 }, staleSource,
  );
  assert.equal(stale.status, "invalid");
  if (stale.status === "invalid") assert.equal(stale.failure, "stale-generation");
  assert.deepEqual(stale.throughput, [0, 0, 0]);

  for (const status of ["nonresident", "work-exhausted"] as const) {
    const result = traceSvoStructuralFluidMediaPath(
      { origin_m: [0, -1, 0], direction: [0, -1, 0], maximumDistance_m: 2 },
      () => ({ status, insideFluidAtStart: true, completeGeneration: 7, coarseFluidRevision: 3, steps: 1, nodeVisits: 2 }),
      undefined, { initialMedium: "water" },
    );
    assert.equal(result.status, status === "nonresident" ? "invalid" : "work-exhausted");
    assert.deepEqual(result.throughput, [0, 0, 0]);
  }
});

test("the structural visibility adapter preserves generation, residency, and boundary diagnostics", () => {
  const diagnostics = {
    source: "structural-coarse" as const, completeGeneration: 9, coarseFluidRevision: 4,
    interpolationSamples: 2, topologyNodeVisits: 17, crossLeafSamples: 0, boundaryFallbackSamples: 0,
    maximumSteps: 12, maximumNodeVisits: 100, failureReason: "nonresident-leaf" as const,
  };
  assert.deepEqual(adaptSvoStructuralFluidVisibilityToMediaStep({
    status: "invalid-field", steps: 3, insideFluidAtStart: true, diagnostics,
  }), {
    status: "nonresident", reason: "nonresident-leaf", insideFluidAtStart: true,
    completeGeneration: 9, coarseFluidRevision: 4, steps: 3, nodeVisits: 17,
  });
});

test("WGSL path is binding-free, bounded, non-recursive, and carries publication and optical state", () => {
  assert.deepEqual(SVO_FLUID_MEDIA_PATH_LIMITS, { boundaryQueries: 16, transitions: 8, reflections: 4, transmissions: 8 });
  assert.match(svoFluidMediaPathWGSL, /fn svoTraceStructuralFluidMedia/);
  assert.match(svoFluidMediaPathWGSL, /svoFluidMediaQueryWater\(ray,generation\)/);
  assert.match(svoFluidMediaPathWGSL, /svoFluidMediaQueryScene\(ray\)/);
  assert.match(svoFluidMediaPathWGSL, /SVO_FLUID_MEDIA_NONRESIDENT/);
  assert.match(svoFluidMediaPathWGSL, /waterThickness_m/);
  assert.match(svoFluidMediaPathWGSL, /svoFluidMediaScatter/);
  assert.match(svoFluidMediaPathWGSL, /SVO_FLUID_MEDIA_MAX_REFLECTIONS:u32=4u/);
  assert.doesNotMatch(svoFluidMediaPathWGSL, /@group|@binding/);
  assert.doesNotMatch(svoFluidMediaPathWGSL, /fn svoTraceStructuralFluidMedia[\s\S]*svoTraceStructuralFluidMedia\(/);
});
