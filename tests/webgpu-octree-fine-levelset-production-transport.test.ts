import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";

interface FineGenerationJSON {
  generation: number;
  publicationValid: boolean;
  activePages: number;
  validSamples: number;
  finiteValidSamples: number;
  negativeValidSamples: number;
  positiveValidSamples: number;
  phiBitXor: number;
  phiBitSum: number;
  transportDepartureOutsideBand?: number;
  transportNonfiniteVelocity?: number;
  transportProcessed?: number;
  transportCommitted?: boolean;
  transportExtrapolatedVelocity?: number;
  transportFaceBandUnavailable?: number;
  transportVelocityUnavailable?: number;
}

interface FineRasterJSON {
  frontInterfacePixels: number;
  backInterfacePixels: number;
  frontInterfaceHash: number;
  backInterfaceHash: number;
  rendererValidationErrorCount?: number;
  rendererUncapturedErrorCount?: number;
  surfaceGeometrySource?: string;
  globalFineAuthorityLatch?: number;
  globalFineCrossingPublished?: boolean;
  presentationFallbackActive?: boolean;
  frontInterfaceBounds_m?: [[number, number, number], [number, number, number]];
  globalFineAuthorityTransition?: {
    validGeneration: number;
    cleanFineCoarseRequired: boolean;
    retainedGeometrySource?: string;
    retainedFrontInterfacePixels: number;
    retainedBackInterfacePixels: number;
    retainedFrontInterfaceHash: number;
    retainedBackInterfaceHash: number;
  };
}

interface SmokeResultJSON {
  phase: string;
  steps?: number;
  globalFineLevelSetFactor?: number;
  nonFiniteCount?: number;
  validationErrors?: unknown[];
  globalFineGenerationCheckpoints?: Array<{
    time_s: number;
    globalFineGeneration?: FineGenerationJSON;
    raster?: FineRasterJSON;
  }>;
}

function resultRecord(stdout: string): SmokeResultJSON | undefined {
  return stdout.split("\n").flatMap(line => {
    try {
      const value = JSON.parse(line) as SmokeResultJSON;
      return value.phase === "result" ? [value] : [];
    } catch {
      return [];
    }
  }).at(-1);
}

function assertPublishedSignedGeneration(label: string, value: FineGenerationJSON | undefined):
asserts value is FineGenerationJSON {
  assert.ok(value, `${label} generation diagnostics are absent`);
  assert.equal(value.publicationValid, true, `${label} generation is not published`);
  assert.ok(value.generation > 0 && value.activePages > 0, `${label} generation has no indexed pages`);
  assert.ok(value.validSamples > 0, `${label} generation has no valid phi samples`);
  assert.equal(value.finiteValidSamples, value.validSamples, `${label} generation contains non-finite phi`);
  assert.ok(value.negativeValidSamples > 0 && value.positiveValidSamples > 0,
    `${label} generation does not retain both signed sides of the interface`);
}

function assertCleanFineCoarseVisible(label: string, generation: FineGenerationJSON,
  raster: FineRasterJSON | undefined): asserts raster is FineRasterJSON {
  assert.ok(raster, `${label} fine raster diagnostics are absent`);
  assert.ok(raster.frontInterfacePixels > 0 && raster.backInterfacePixels > 0,
    `${label} fine-authoritative front/back interface is not visible`);
  assert.equal(raster.surfaceGeometrySource, "global-fine-coarse",
    `${label} raster did not use the clean fine/coarse publication`);
  assert.equal(raster.globalFineCrossingPublished, true,
    `${label} raster did not publish a current global crossing`);
  assert.equal(raster.presentationFallbackActive, false,
    `${label} raster used a presentation fallback`);
  assert.ok((raster.globalFineAuthorityLatch ?? 0) > 0,
    `${label} raster did not latch global fine/coarse authority`);
  assert.ok(raster.frontInterfaceBounds_m?.flat(2).every(Number.isFinite),
    `${label} raster bounds are absent or non-finite`);
  assert.equal(raster.globalFineAuthorityTransition?.cleanFineCoarseRequired, true,
    `${label} raster did not require the compact coarse member of the publication`);
  assert.equal(raster.globalFineAuthorityTransition?.validGeneration, generation.generation,
    `${label} raster did not consume its published fine generation`);
  assert.equal(raster.globalFineAuthorityTransition?.retainedGeometrySource, "retained-previous",
    `${label} unpublished generation did not select retained presentation geometry`);
  assert.equal(raster.globalFineAuthorityTransition?.retainedFrontInterfacePixels, raster.frontInterfacePixels,
    `${label} unpublished-generation probe did not retain the published front interface`);
  assert.equal(raster.globalFineAuthorityTransition?.retainedBackInterfacePixels, raster.backInterfacePixels,
    `${label} unpublished-generation probe did not retain the published back interface`);
  assert.equal(raster.globalFineAuthorityTransition?.retainedFrontInterfaceHash, raster.frontInterfaceHash,
    `${label} unpublished-generation probe changed the published front content`);
  assert.equal(raster.globalFineAuthorityTransition?.retainedBackInterfaceHash, raster.backInterfaceHash,
    `${label} unpublished-generation probe changed the published back content`);
}

function assertCommittedTransport(label: string, generation: FineGenerationJSON, requireFaceBand = false): void {
  assert.equal(generation.transportDepartureOutsideBand, 0,
    `${label} transport departed the resident narrow band`);
  assert.equal(generation.transportNonfiniteVelocity, 0,
    `${label} transport sampled a non-finite reconstructed velocity`);
  assert.equal(generation.transportFaceBandUnavailable, 0,
    `${label} transport could not resolve a containing owner/regular-face band row`);
  assert.equal(generation.transportVelocityUnavailable, 0,
    `${label} transport received an unavailable Stage-B or air-band velocity`);
  assert.ok((generation.transportProcessed ?? 0) > 0, `${label} transport processed no fine samples`);
  if (requireFaceBand) assert.ok((generation.transportExtrapolatedVelocity ?? 0) > 0,
    `${label} did not consume the regular-face fast-marched positive-air band`);
  assert.equal(generation.transportCommitted, true, `${label} transport did not commit its published generation`);
}

for (const factor of [4, 8] as const) {
  const gate = `FLUID_FINE_DAM_BREAK_FACTOR${factor}_ACCEPTANCE`;
  test(`production dam-break publishes transported factor-${factor} fine phi into moving fine/coarse rasters`, {
    skip: !process.env.WEBGPU_NODE_MODULE
      ? "set WEBGPU_NODE_MODULE for the production fine-transport acceptance"
      : process.env[gate] !== "1" && `set ${gate}=1 to run this memory-intensive production gate`,
    timeout: 180_000,
  }, () => {
    const child = spawnSync(process.execPath, ["--import", "tsx", "tools/run-webgpu-smoke.ts"], {
      cwd: process.cwd(), encoding: "utf8", timeout: 150_000, killSignal: "SIGKILL",
      maxBuffer: 32 * 1024 * 1024,
      env: { ...process.env, FLUID_SCENE: "dam-break-ui", FLUID_METHOD: "octree",
        FLUID_TARGET_S: "0.008", FLUID_ORACLE_STEPS: "2", FLUID_SURFACE_COLUMNS: "2400",
        FLUID_PRESSURE_CYCLES: "400", FLUID_STABILITY_ENVELOPE: "1", FLUID_CPU_ORACLE: "0",
        FLUID_FIELD_STATS: "0", FLUID_DISABLE_TIMESTAMPS: "1", FLUID_OCTREE_FACE_TRANSPORT: "1",
        FLUID_OCTREE_POWER_PROJECTION: "authoritative", FLUID_OCTREE_GLOBAL_FINE_FACTOR: String(factor),
        FLUID_GLOBAL_FINE_GENERATION_TRANSITION: "1", FLUID_CHECKPOINT_EVERY_S: "0.004",
        FLUID_RASTER_CHECKPOINTS: "1" },
    });
    assert.equal(child.error, undefined,
      `factor-${factor} production transport process failed: ${child.error?.message ?? "unknown"}`);
    assert.equal(child.status, 0,
      `factor-${factor} production transport smoke failed:\n${child.stderr}\n${child.stdout.slice(-8_000)}`);
    const result = resultRecord(child.stdout);
    assert.ok(result, `factor-${factor} production transport emitted no result JSON`);
    assert.equal(result.globalFineLevelSetFactor, factor, `factor-${factor} request was not honored`);
    assert.equal(result.steps, 2, `factor-${factor} acceptance must execute exactly two production steps`);
    assert.equal(result.nonFiniteCount, 0, `factor-${factor} production state contains non-finite values`);
    assert.deepEqual(result.validationErrors ?? [], [], `factor-${factor} production smoke raised validation errors`);

    const checkpoints = result.globalFineGenerationCheckpoints ?? [];
    assert.equal(checkpoints.length, 2, "two-step production run did not emit exactly two publication checkpoints");
    const [firstCheckpoint, secondCheckpoint] = checkpoints;
    const first = firstCheckpoint.globalFineGeneration;
    const second = secondCheckpoint.globalFineGeneration;
    assertPublishedSignedGeneration("step-1", first);
    assertPublishedSignedGeneration("step-2", second);
    assert.ok(second.generation > first.generation, "two production steps did not publish a newer generation");
    assert.notDeepEqual([second.phiBitXor, second.phiBitSum], [first.phiBitXor, first.phiBitSum],
      "published phi fingerprint did not change; topology publication alone is insufficient");
    // Step 1 is the bootstrap publication. It must already be signed,
    // indexable, and fine-authoritative, but intentionally has no transport
    // control record. Step 2 is the first transported publication.
    assertCommittedTransport("step-2", second, factor === 4);

    assertCleanFineCoarseVisible("step-1", first, firstCheckpoint.raster);
    assertCleanFineCoarseVisible("step-2", second, secondCheckpoint.raster);
    assert.notEqual(secondCheckpoint.raster.frontInterfaceHash,
      firstCheckpoint.raster.frontInterfaceHash, "fine-authoritative front raster content did not move");
    assert.notEqual(secondCheckpoint.raster.backInterfaceHash,
      firstCheckpoint.raster.backInterfaceHash, "fine-authoritative back raster content did not move");
  });
}
