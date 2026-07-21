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
  assert.match(worker, /finally \{[\s\S]*await rm\(WEBGPU_EXCLUSIVE_LOCK/,
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
});

test("compact publication rejection reports header and control evidence before abort", async () => {
  const smoke = await readFile(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
  assert.match(smoke, /globalFineCoarseLevelSetControl[\s\S]*readBufferBinding[\s\S]*64/,
    "the coarse transaction control must be read with the rejected directory");
  assert.match(smoke,
    /globalFineTransportControl[\s\S]*readBufferBinding[\s\S]*32[\s\S]*globalFineRedistanceControl[\s\S]*readBufferBinding[\s\S]*16[\s\S]*globalFineVolumeControl[\s\S]*readBufferBinding[\s\S]*64/,
    "a required compact rejection must include complete transport, redistance, and volume controls");
  assert.match(smoke,
    /globalFineFaceBandControl[\s\S]*readBufferBinding[\s\S]*64[\s\S]*globalFineFaceBandTransitionControl[\s\S]*readBufferBinding[\s\S]*160[\s\S]*globalFineFaceBandTransientPowerControl[\s\S]*readBufferBinding[\s\S]*64[\s\S]*globalFineFaceBandPointFieldControl[\s\S]*readBufferBinding[\s\S]*32[\s\S]*globalFineFaceBandPowerPublicationControl[\s\S]*readBufferBinding[\s\S]*64[\s\S]*globalFinePowerVelocityControl[\s\S]*readBufferBinding[\s\S]*32[\s\S]*globalFinePowerVelocitySampleControl[\s\S]*readBufferBinding[\s\S]*32/,
    "a transport rejection must retain face-band, transition, transient graph, point-field, regular-to-power, Stage-A, and Stage-B producer controls");
  assert.match(smoke, /catch \(error\)[\s\S]*compact-octree-field-publication-rejected[\s\S]*compactOctreePublicationHeaderEvidence\(compactSnapshot\)[\s\S]*throw error/,
    "publication evidence must be emitted before the acceptance error is rethrown");
});
