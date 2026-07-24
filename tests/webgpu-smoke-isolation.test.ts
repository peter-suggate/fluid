import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import {
  DEFAULT_WEBGPU_SMOKE_TIMEOUT_MS,
  MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS,
  MINIMUM_WEBGPU_SMOKE_TIMEOUT_MS,
  parseWebGPUSmokeTimeout,
  WEBGPU_EXCLUSIVE_LOCK,
} from "../tools/webgpu-smoke-isolation";

function assertContainsInOrder(source: string, fragments: readonly string[], message: string): void {
  let offset = 0;
  for (const fragment of fragments) {
    const index = source.indexOf(fragment, offset);
    assert.notEqual(index, -1, `${message}: missing ${JSON.stringify(fragment)}`);
    offset = index + fragment.length;
  }
}

function normalizeWhitespace(source: string): string {
  return source.replace(/\s+/g, " ");
}

test("isolated smoke timeout is validated in the 60-240 second safety envelope", () => {
  assert.equal(parseWebGPUSmokeTimeout(undefined), DEFAULT_WEBGPU_SMOKE_TIMEOUT_MS);
  assert.equal(parseWebGPUSmokeTimeout("60000"), MINIMUM_WEBGPU_SMOKE_TIMEOUT_MS);
  assert.equal(parseWebGPUSmokeTimeout("240000"), MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS);
  for (const value of ["59999", "240001", "120000.5", "Infinity", "not-a-timeout"]) {
    assert.throws(() => parseWebGPUSmokeTimeout(value), /must be an integer from 60000 to 240000/);
  }
  assert.equal(WEBGPU_EXCLUSIVE_LOCK, "/tmp/fluid-webgpu-exclusive.lock");
});

test("isolated smoke records its owner and releases only on ordinary worker completion", async () => {
  const launcher = await readFile(new URL("../tools/run-webgpu-smoke-isolated.ts", import.meta.url), "utf8");
  const worker = await readFile(new URL("../tools/run-webgpu-smoke-isolated-worker.ts", import.meta.url), "utf8");
  assert.match(launcher, /parseWebGPUSmokeTimeout\(process\.env\.FLUID_WEBGPU_SMOKE_TIMEOUT_MS\)/,
    "timeout validation must happen before the worker imports Dawn");
  assert.match(launcher, /pid: child\.pid/);
  assert.match(launcher, /child\.kill\("SIGTERM"\)/);
  assert.match(launcher, /child\.kill\("SIGKILL"\)/);
  assert.match(launcher, /process\.exit\(124\)/);
  assert.match(launcher, /leaving \$\{WEBGPU_EXCLUSIVE_LOCK\} as owner evidence/);
  assert.match(worker, /await mkdir\(WEBGPU_EXCLUSIVE_LOCK\)/,
    "mkdir is the atomic exclusive-lock acquisition");
  assert.match(worker, /pid: process\.pid/);
  assert.match(worker, /writeFile\(`\$\{WEBGPU_EXCLUSIVE_LOCK\}\/owner\.json`/);
  assert.match(worker, /await import\("\.\/run-webgpu-smoke"\)/);
  assertContainsInOrder(normalizeWhitespace(worker), ["finally {", "await rm(WEBGPU_EXCLUSIVE_LOCK"],
    "success and ordinary smoke failures must not leave a stale lock");
});

test("one-step power/fine comparison pins exact time, spatial readback, and motion evidence", async () => {
  const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:dam-power-fine-compare-one-step"];
  assert.ok(command);
  for (const setting of [
    "FLUID_TARGET_S=0.004", "FLUID_MAX_DT=0.004", "FLUID_ORACLE_STEPS=1",
    "FLUID_EXPECT_EXACT_STEPS=1", "FLUID_MIN_PEAK_SPEED_M_S=0.01",
    "FLUID_REQUIRE_SPATIAL_FIELD=1", "FLUID_EXPECT_GRID=24,18,16",
  ]) assert.match(command, new RegExp(setting.replaceAll(".", "\\.")));
  assert.match(command, /run-webgpu-smoke-isolated\.ts$/);

  const exactTwoStep = packageJson.scripts["test:webgpu:minimal-power-dam-two-step"];
  assert.ok(exactTwoStep);
  assert.doesNotMatch(exactTwoStep, /FLUID_WEBGPU_DAWN_FEATURES=skip_validation/,
    "the exact correctness smoke must validate every pipeline and bind group before submission");
});

test("compact publication rejection reports header and control evidence before abort", async () => {
  const smoke = await readFile(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  const functionStart = smoke.indexOf("async function readCubicVolumeField(");
  const functionEnd = smoke.indexOf("\ntype GPUCommandAuditBucket", functionStart);
  assert.notEqual(functionStart, -1, "compact field readback function must exist");
  assert.notEqual(functionEnd, -1, "compact field readback function must have a bounded end");
  const readback = normalizeWhitespace(smoke.slice(functionStart, functionEnd));

  const expectedControlReadbacks = [
    ["solver.globalFineCoarseLevelSetControl", "64"],
    ["solver.globalFineTransportControl", "64"],
    ["solver.globalFineRedistanceControl", "FINE_LEVELSET_REDISTANCE_CONTROL_BYTES"],
    ["solver.globalFineVolumeControl", "64"],
    ["solver.globalFineFaceBandControl", "128"],
    ["solver.globalFineFaceBandTransitionControl", "160"],
    ["solver.globalFineFaceBandTransientPowerControl", "64"],
    ["solver.globalFineFaceBandPointFieldControl", "32"],
    ["solver.globalFineFaceBandPowerPublicationControl", "64"],
    ["solver.globalFinePowerVelocityControl", "32"],
    ["solver.globalFinePowerVelocitySampleControl", "32"],
  ] as const;
  for (const [control, byteLength] of expectedControlReadbacks) {
    assert.ok(readback.includes(`readBufferBinding(device, { buffer: ${control} }, ${byteLength})`),
      `${control} must retain its complete ${byteLength} rejection evidence`);
  }

  assertContainsInOrder(readback, [
    "catch (error) {",
    'phase: "compact-octree-field-publication-rejected"',
    "...compactOctreePublicationHeaderEvidence(compactSnapshot)",
    "throw error",
  ], "publication evidence must be emitted before the acceptance error is rethrown");

  assert.match(readback,
    /!faceBandCandidate\.valid[\s\S]*firstErrorRow = faceBandCandidate\.firstError[\s\S]*failedFaceReason === 34[\s\S]*faceBandCandidate\.stageFirstFailures\.faceEmission/,
    "an invalid candidate must distinguish an exact face slot from the incidence-count failure word");
  assert.match(readback,
    /Math\.floor\(failedFaceSlot \/ OCTREE_REGULAR_BAND_OWNED_FACES_PER_ROW\)/,
    "the failed candidate face must name its deterministic owner row");
  assert.match(readback, /failedFaceReason = faceBandCandidateControl\[7\] & 0x0fff/,
    "the packed candidate producer failure must retain the exact face-emission reason");
  assert.match(readback,
    /failedFaceSlot === 0xffff_ffff && failedFaceReason !== 34[\s\S]*faceBandCandidateControl\[28\][\s\S]*faceBandCandidateControl\[24\][\s\S]*faceBandCandidateControl\[29\][\s\S]*faceBandCandidateControl\[25\][\s\S]*faceBandCandidateControl\[30\][\s\S]*faceBandCandidateControl\[26\][\s\S]*faceBandCandidateControl\[31\][\s\S]*faceBandCandidateControl\[27\][\s\S]*count !== 0 && slot < failedFaceSlot[\s\S]*failedFaceReason = reason/,
    "a post-emission CPT rejection must identify the first exact face slot and its cause");
  assert.match(readback,
    /readGlobalFineCandidateBandRowFailure\?\.\(firstErrorRow\)[\s\S]*readGlobalFineCandidateBandFaceFailure\?\.\(failedFaceSlot\)[\s\S]*readGlobalFineCandidateBandRowFailure\?\.\(failedFaceOwnerRow\)/,
    "candidate diagnostics must read the unpublished row, face, and owner-row arenas");
  assert.match(readback,
    /failedFaceReason === 34[\s\S]*readGlobalFineCandidateBandIncidenceFailure\?\.\(faceBandCandidate\.rowCount\)/,
    "incidence-count rejection must run the exact candidate reciprocity audit");
});
