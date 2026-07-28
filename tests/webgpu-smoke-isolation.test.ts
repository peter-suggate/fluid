import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  DEFAULT_WEBGPU_SMOKE_TIMEOUT_MS,
  MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS,
  MINIMUM_WEBGPU_SMOKE_TIMEOUT_MS,
  parseWebGPUSmokeTimeout,
  readWebGPUExclusiveLockHolder,
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

/**
 * These run against a lock of their own making, never `WEBGPU_EXCLUSIVE_LOCK`:
 * touching the real one would evict whatever Dawn run happens to hold it.
 */
test("the exclusive lock reports its holder, and whether that holder still exists", async (t) => {
  const scratch = await mkdtemp(join(tmpdir(), "fluid-lock-holder-"));
  t.after(() => rm(scratch, { recursive: true, force: true }));
  const lock = join(scratch, "exclusive.lock");

  assert.equal(await readWebGPUExclusiveLockHolder(lock), undefined,
    "a free lock must read as free, not as an unknown holder");

  await mkdir(lock);
  const midAcquisition = await readWebGPUExclusiveLockHolder(lock);
  assert.equal(midAcquisition?.alive, true,
    "a lock that exists but is unstamped is a run mid-acquisition, so it excludes us");

  const stamp = async (pid: number): Promise<void> => writeFile(`${lock}/owner.json`, JSON.stringify({
    pid, parentPid: process.ppid, startedAt: "2026-07-28T01:37:54.664Z",
    kind: "dawn-smoke", target: "tools/run-webgpu-smoke.ts",
  }));

  await stamp(process.pid);
  const live = await readWebGPUExclusiveLockHolder(lock);
  assert.equal(live?.alive, true, "a running owner is a live GPU run, to be waited for");
  assert.equal(live?.owner?.pid, process.pid);
  for (const fragment of [String(process.pid), "dawn-smoke", "tools/run-webgpu-smoke.ts"]) {
    assert.ok(live?.description.includes(fragment),
      `the holder description must name ${fragment} so the caller knows who to wait for`);
  }

  // A reaped child's pid is gone, which is the crash-evidence case: the lock
  // outlived its owner and only a human may clear it.
  const reaped = spawnSync("/usr/bin/true").pid;
  assert.ok(typeof reaped === "number" && reaped > 0);
  await stamp(reaped);
  const stale = await readWebGPUExclusiveLockHolder(lock);
  assert.equal(stale?.alive, false, "an owner that no longer exists must not read as a live run");
  assert.match(stale?.description ?? "", /no longer running/);
});

test("the xctrace profiler refuses to start against a held lock, and never orphans its solver", async () => {
  const profiler = normalizeWhitespace(await readFile(
    new URL("../tools/profile-mini-dam-xctrace.ts", import.meta.url), "utf8"));
  assertContainsInOrder(profiler, [
    "const requireExclusiveGPU",
    "await readWebGPUExclusiveLockHolder()",
    "throw new Error(holder.alive",
  ], "a capture costs minutes; a held lock must be found before any of them are spent");
  assert.ok(profiler.indexOf("await requireExclusiveGPU()")
    < profiler.indexOf('console.log("recording untraced baseline'),
    "the check must precede the first GPU run the profiler starts");
  // A worker that dies before announcing construction used to resolve
  // `constructed`, sending the profiler on to attach Instruments to a dead pid,
  // and to reject `finished` with nobody listening, which killed the profiler
  // with an unhandled rejection naming the spawn plumbing instead of the cause.
  assertContainsInOrder(profiler, [
    "if (failure) { abandonConstruction(failure); fail(failure); }",
    "constructed.catch(() => {});",
    "finished.catch(() => {});",
  ], "a worker that never constructs must reject construction, and reject it handled");
  assertContainsInOrder(profiler, [
    "traced = await handle.finished;",
    "} catch (error) {",
    "handle.stop();",
    "throw error;",
  ], "an abandoned capture must take its lock-holding solver with it");
});

test("isolated smoke records its owner and releases only on ordinary worker completion", async () => {
  const isolation = await readFile(new URL("../tools/webgpu-smoke-isolation.ts", import.meta.url), "utf8");
  const launcher = await readFile(new URL("../tools/run-webgpu-smoke-isolated.ts", import.meta.url), "utf8");
  const worker = await readFile(new URL("../tools/run-webgpu-smoke-isolated-worker.ts", import.meta.url), "utf8");
  assert.match(launcher, /parseWebGPUSmokeTimeout\(process\.env\.FLUID_WEBGPU_SMOKE_TIMEOUT_MS\)/,
    "timeout validation must happen before the worker imports Dawn");
  assert.match(launcher, /pid: child\.pid/);
  assert.match(launcher, /child\.kill\("SIGTERM"\)/);
  assert.match(launcher, /child\.kill\("SIGKILL"\)/);
  assert.match(launcher, /process\.exit\(124\)/);
  assert.match(launcher, /leaving \$\{WEBGPU_EXCLUSIVE_LOCK\} as owner evidence/);
  assert.match(isolation, /await mkdir\(WEBGPU_EXCLUSIVE_LOCK\)/,
    "mkdir is the atomic exclusive-lock acquisition");
  assert.match(isolation, /pid: process\.pid/);
  assert.match(isolation, /writeFile\(`\$\{WEBGPU_EXCLUSIVE_LOCK\}\/owner\.json`/);
  assert.match(worker, /acquireWebGPUExclusiveLock\(/);
  assert.match(worker, /await import\("\.\/run-webgpu-smoke"\)/);
  assertContainsInOrder(normalizeWhitespace(worker), ["finally {", "await releaseWebGPUExclusiveLock()"],
    "success and ordinary smoke failures must not leave a stale lock");
});

test("all standalone octree GPU benchmarks share the process-wide exclusive lock", async () => {
  const benchmarks = [
    "benchmark-octree-leaf-sizes.ts",
    "benchmark-octree-pressure-page-shapes.ts",
    "benchmark-octree-section63-bandwidth.ts",
  ];
  for (const file of benchmarks) {
    const source = normalizeWhitespace(await readFile(new URL(`../tools/${file}`, import.meta.url), "utf8"));
    assertContainsInOrder(source, [
      "await acquireWebGPUExclusiveLock(",
      "try {",
      "await import(pathToFileURL(modulePath)",
      "finally {",
      "await releaseWebGPUExclusiveLock()",
    ], `${file} must own the shared lock for its complete Dawn lifetime`);
  }
});

test("minimal dam acceptance and capture pin the exact 500-step, 2-second contract", async () => {
  const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:minimal-power-dam-break"];
  assert.ok(command);
  for (const setting of [
    "FLUID_SCENE=minimal-power-dam-break",
    "FLUID_TARGET_S=2",
    "FLUID_MAX_DT=0.004",
    "FLUID_ORACLE_STEPS=500",
    "FLUID_EXPECT_EXACT_STEPS=500",
  ]) assert.match(command, new RegExp(setting.replaceAll(".", "\\.")));
  assert.match(command, /run-webgpu-smoke-isolated\.ts$/,
    "the exact acceptance lane must acquire the exclusive GPU lock");

  const benchmark = normalizeWhitespace(await readFile(
    new URL("../tools/benchmark-power-dam.ts", import.meta.url), "utf8"));
  assertContainsInOrder(benchmark, [
    'mini: { FLUID_SCENE: "minimal-power-dam-break", FLUID_TARGET_S: "2"',
    'FLUID_MAX_DT: "0.004", FLUID_ORACLE_STEPS: "500", FLUID_EXPECT_EXACT_STEPS: "500"',
  ], "the regression capture mini lane must retain the exact acceptance contract");
  assert.match(benchmark, /run-webgpu-smoke-isolated\.ts/,
    "every regression capture subprocess must acquire the exclusive GPU lock");
});

test("M1 cutover suite does not name deleted page-pool fallback tests", async () => {
  const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:octree-m1-cutover"];
  assert.ok(command);
  assert.doesNotMatch(command, /webgpu-octree-page-pool\.test\.ts/);
  assert.match(command, /run-webgpu-exclusive\.ts/,
    "the complete Dawn cutover suite must hold one process-wide GPU lock");
});

test("direct octree Dawn test and smoke commands run beneath one lock-owning parent", async () => {
  const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  for (const name of [
    "test:webgpu:octree-power",
    "test:webgpu:octree-runtime",
    "test:webgpu:octree-m1-cutover",
    "test:webgpu:octree-displacement",
    "test:webgpu:octree-lagged-rigid",
    "test:webgpu:octree-rigid-buoyancy",
    "test:webgpu:octree-solid-topology",
  ]) {
    assert.match(packageJson.scripts[name] ?? "", /run-webgpu-exclusive\.ts/,
      `${name} must not create Dawn outside the shared lock`);
  }

  const runner = normalizeWhitespace(await readFile(
    new URL("../tools/run-webgpu-exclusive.ts", import.meta.url), "utf8"));
  assertContainsInOrder(runner, [
    "await acquireWebGPUExclusiveLock(",
    "child = spawn(process.execPath, nodeArguments",
    "finally {",
    "await releaseWebGPUExclusiveLock()",
  ], "the generic GPU command wrapper must retain its lock until its child is reaped");
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

test("compact and structured publication rejection reports exact authority evidence before abort", async () => {
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

  const generationAudit = normalizeWhitespace(smoke.slice(
    smoke.indexOf("if (captureCompactPowerStep)"),
    smoke.indexOf("if (steps === oracleSteps)", smoke.indexOf("if (captureCompactPowerStep)")),
  ));
  assertContainsInOrder(generationAudit, [
    "auditEncoder.copyBufferToBuffer",
    "audited.structuredVelocityControl",
    "audited.structuredBoundaryControl",
    "fine.worklist",
    "audited.globalFineVolumeControl",
    "audited.structuredProjectionEnergyStats",
    "device.queue.submit",
  ], "every accepted step must enqueue one exact structured/fine generation snapshot");
  assert.doesNotMatch(generationAudit, /await awaitAdvanceCompletion|readBufferBinding|mapAsync/,
    "the recurring structured audit must not fence or map the GPU queue");
  const terminalAudit = normalizeWhitespace(smoke.slice(
    smoke.indexOf("if (powerGenerationAuditSnapshot)",
      smoke.indexOf("await awaitAdvanceCompletion();", smoke.indexOf("const simulationWall_ms"))),
    smoke.indexOf("const gpuFineTimestamps", smoke.indexOf("const simulationWall_ms")),
  ));
  assertContainsInOrder(terminalAudit, [
    "powerGenerationAuditSnapshot.mapAsync",
    "unpackStructuredGenerationAuditSnapshot",
    "exactStructuredGenerationAuditFailures",
    "unpackFineLevelSetGPUVolumeControl",
    "decodeStructuredProjectionEnergy",
    'phase: "structured-generation-audit"',
    "readPowerFrontierFailure",
    "throw new Error",
  ], "terminal audit must validate every queued structured/fine generation transaction before accepting");
  assert.match(smoke,
    /const shouldSampleDetailedFields = method\.id === "octree"[\s\S]*?\? shouldSampleEnergy[\s\S]*?: shouldReport \|\| shouldSampleEnergy \|\| collectStabilityEnvelope/,
    "octree stability must not force a full spatial readback on every step");
  assert.match(smoke,
    /if \(checkpointEvery_s > 0[\s\S]*stabilityEnvelope\.spatialSampledSteps \+= 1/,
    "octree reconstructed connectivity must remain a checkpoint audit");
  for (const retired of ["globalFineFaceBand", "globalFinePowerVelocity", "powerFaceControl"]) {
    assert.doesNotMatch(smoke, new RegExp(retired), `${retired} must stay out of the smoke authority graph`);
  }
});
