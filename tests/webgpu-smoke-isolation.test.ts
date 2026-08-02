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
import { POWER_DAM_LANE_ENVIRONMENT } from "../tools/power-dam-lane-environment";

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
  // Anchored on the first recording the profiler announces, whatever that lane
  // is currently called. The invariant is ordering -- no capture may start
  // before the lock check -- and naming one baseline label made a rename look
  // like a lock-ordering regression.
  const firstRecording = profiler.indexOf('console.log("recording');
  assert.ok(firstRecording > 0, "the profiler must announce the run it starts");
  assert.ok(profiler.indexOf("await requireExclusiveGPU()") < firstRecording,
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

  // The lane table now lives beside the benchmark rather than inside it, so the
  // contract is asserted against the value the harness actually spawns with.
  // Reading the exported table instead of its source text also means a lane
  // cannot satisfy this by containing the right characters in a comment.
  assert.equal(POWER_DAM_LANE_ENVIRONMENT.mini.FLUID_SCENE, "minimal-power-dam-break");
  for (const [key, value] of Object.entries({
    FLUID_TARGET_S: "2", FLUID_MAX_DT: "0.004",
    FLUID_ORACLE_STEPS: "500", FLUID_EXPECT_EXACT_STEPS: "500",
  })) assert.equal(POWER_DAM_LANE_ENVIRONMENT.mini[key], value,
    `the regression capture mini lane must retain the exact acceptance contract for ${key}`);

  const benchmark = normalizeWhitespace(await readFile(
    new URL("../tools/benchmark-power-dam.ts", import.meta.url), "utf8"));
  const profiler = normalizeWhitespace(await readFile(
    new URL("../tools/profile-mini-dam-xctrace.ts", import.meta.url), "utf8"));
  assert.match(benchmark, /run-webgpu-smoke-isolated\.ts/,
    "every regression capture subprocess must acquire the exclusive GPU lock");
  assert.match(benchmark, /FLUID_POWER_GENERATION_AUDIT: "0"/,
    "throughput profiles must not inherit the scene's full generation/energy audit");
  // The caller may escalate to `failfast`, never de-escalate: the ternary's
  // fallback is "1", and the assignment sits after `...overrides`, so a
  // `FLUID_TRIPWIRES=0` in the environment cannot reach the child.
  assert.match(benchmark,
    /FLUID_TRIPWIRES: process\.env\.FLUID_TRIPWIRES === "failfast" \? "failfast" : "1"/,
    "disabling the full audit must retain the silent-failure gates, and the only caller override is escalation");
  assert.match(profiler, /FLUID_POWER_GENERATION_AUDIT: "0"/,
    "the xctrace graph must match the benchmark's generation-audit setting");
  assert.match(profiler, /FLUID_TRIPWIRES: "1"/,
    "the xctrace graph must retain the benchmark's per-step tripwires");
});

test("the ocean first-frame lane runs two exact advances under the browser storage tier", () => {
  assert.deepEqual(POWER_DAM_LANE_ENVIRONMENT.ocean, {
    FLUID_SCENE: "ocean-seiche",
    FLUID_TARGET_S: "0.01",
    FLUID_MAX_DT: "0.005",
    FLUID_ORACLE_STEPS: "2",
    FLUID_EXPECT_EXACT_STEPS: "2",
    FLUID_WEBGPU_MAX_STORAGE_BINDING_BYTES: "2147483648",
    FLUID_POWER_GENERATION_AUDIT: "1",
    FLUID_POWER_GENERATION_AUDIT_LOG: "1",
    FLUID_POWER_STAGE_AUDIT: "1",
    FLUID_POWER_AUDIT_EVERY_STEPS: "1",
    FLUID_STABILITY_ENVELOPE: "1",
  });
});

/**
 * Fail-fast is the contract for diagnosis: a tripped counter means the physics
 * being measured is already gone. The end-of-run walk once absorbed that --
 * `large-power-dam-break` lost its fine band at step 248 and kept stepping a
 * frozen solver to step 430 before reporting "tripped over 430 captured steps",
 * 732 trips across four ids. Under fail-fast the same run reports one
 * `air-support-failure` at step 247 and the other 731 stand revealed as
 * downstream absorption. These pin the structure that makes the run die at the
 * step that tripped. The decoded verdicts themselves are covered by the
 * control-ABI unit tests; only a Dawn lane can exercise the map/unmap cycle end
 * to end.
 */
test("silent-failure tripwires are evaluated at their own step and terminate there", async () => {
  const smoke = normalizeWhitespace(await readFile(
    new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8"));

  // One writer fills both destinations inside one encoder, so the live record
  // the verdict is taken from and the ring the whole-run report walks cannot
  // drift apart.
  assertContainsInOrder(smoke, [
    "const encodeTripwireRecordCopies = (",
    "encodeTripwireRecordCopies(encoder, sources, tripwireSnapshot, base);",
    "encodeTripwireRecordCopies(encoder, sources, tripwireLiveReadback!, 0);",
    "device.queue.submit([encoder.finish()]);",
  ], "the ring record and the live record must be written by the same encoder and writer");
  assert.match(smoke,
    /label: "Live silent-failure tripwire record", size: TRIPWIRE_RECORD\.strideBytes, usage: GPUBufferUsage\.COPY_DST \| GPUBufferUsage\.MAP_READ/,
    "the live verdict needs its own single-record readback; the ring is MAP_READ|COPY_DST and cannot be a copy source");

  const live = smoke.slice(
    smoke.indexOf("let capturedThisStep = false;"),
    smoke.indexOf("if (steps === oracleSteps)", smoke.indexOf("let capturedThisStep = false;")));
  assert.notEqual(live.length, 0, "the live tripwire evaluation must sit inside the step loop");
  assertContainsInOrder(live, [
    // Map, copy out, unmap before anything can encode into the buffer again.
    "await tripwireLiveReadback!.mapAsync(GPUMapMode.READ, 0, TRIPWIRE_RECORD.strideBytes);",
    "tripwireLiveRecord.set(new Uint8Array(",
    "} finally {",
    "tripwireLiveReadback!.unmap();",
    "const liveTrips = evaluateTripwireRecord(",
    // FLUID_TRIPWIRE_ALLOW is the only downgrade path.
    "const liveFailing = liveTrips.filter((entry) => !tripwireAllowList.has(entry.id));",
    "samplingWall_ms += liveDecode_ms;",
    "if (liveFailing.length !== 0) {",
    "throw tripwireFailure(liveFailing, `at step ${steps}`",
    "(fine generation ${fineGeneration})",
  ], "step N's tripwires must be decoded and enforced at step N, before the next advance");
  assert.doesNotMatch(live, /console\.error\(`\[tripwire/,
    "a fatal trip must throw, never warn: FLUID_TRIPWIRE_ALLOW is the only downgrade");

  // The trace has exactly one emitter per run, selected by mode: the live walk
  // under fail-fast (so it survives the fatal step), the ring walk otherwise
  // (so turning fail-fast off does not silently drop the trace). Emitting from
  // both would double every record on a run that reaches the end.
  assert.equal(smoke.split('record: "fine-topology-trace"').length - 1, 1,
    "exactly one FLUID_FINE_TOPOLOGY_TRACE emitter may exist");
  assert.match(smoke, /emitTopologyTrace && process\.env\.FLUID_FINE_TOPOLOGY_TRACE === "1"/);
  assert.match(smoke, /steps, fineGeneration, true\);/,
    "the live walk emits the topology trace");
  assert.match(smoke,
    /tripwireFineGenerations\[record\]!,\s*!tripwiresFailFast\)\);/,
    "the ring walk emits the trace exactly when the live walk did not run");

  // Whole-run reporting for allow-listed trips is unchanged, and the ring walk
  // remains a hard failure if it ever disagrees with the live one.
  assertContainsInOrder(smoke, [
    "const allowed = tripped.filter((entry) => tripwireAllowList.has(entry.id));",
    "const failing = tripped.filter((entry) => !tripwireAllowList.has(entry.id));",
    "[tripwire ${entry.id} ALLOWED]",
    "throw tripwireFailure(failing, `over ${tripwireSteps.length} captured steps`);",
    'phase: "tripwires", capturedSteps: tripwireSteps.length,',
    "required: tripwiresRequired, allowed: Array.from(tripwireAllowList),",
  ], "the end-of-run walk must keep reporting the whole run");
  assert.match(smoke, /silent-failure tripwire\(s\) tripped \$\{where\}/,
    "both termination paths must report the same forensics payload");
});

/**
 * Detection and termination-timing are separate axes, and conflating them cost
 * every benchmark number ~27%.
 *
 * The per-step verdict needs a queue fence, which removes host/GPU overlap:
 * measured 420.57 vs 331.80 ms/advance on `large-power-dam-break`. That price is
 * worth paying to find a first domino and worth nothing on a green run, where
 * there is no first trip to stop at. So `1` keeps detection unconditional and
 * pays nothing, `failfast` adds the fence, and only `0` stops gating -- which
 * the benchmark harness refuses.
 */
test("tripwire detection is unconditional; only the per-step fence is a mode", async () => {
  const smoke = normalizeWhitespace(await readFile(
    new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8"));

  assert.match(smoke,
    /tripwireMode !== "0" && tripwireMode !== "1"\s*&& tripwireMode !== "failfast"/,
    "the three modes are 0, 1 and failfast");
  assert.match(smoke, /const tripwiresFailFast = tripwireMode === "failfast";/);
  assert.match(smoke,
    /const tripwiresForcedRequired = tripwireMode === "1" \|\| tripwiresFailFast;/,
    "requiring every counter to be readable must not be weakened by escalating to failfast");

  // Off-mode must cost nothing: no second buffer, no second copy, no fence.
  assert.match(smoke, /const tripwireLiveReadback = tripwireSnapshot && tripwiresFailFast/,
    "the live readback buffer may only exist in failfast mode");
  assert.match(smoke,
    /if \(tripwiresFailFast\) \{\s*encodeTripwireRecordCopies\(encoder, sources, tripwireLiveReadback!, 0\);/,
    "the second per-step copy is failfast-only");
  assert.match(smoke, /if \(capturedThisStep && tripwiresFailFast\) \{/,
    "the per-step fence, decode and verdict are failfast-only");

  // ...but the ring capture and the end-of-run verdict are NOT gated on the
  // mode. A run that gets faster while rolling back topology still fails.
  const ringWalk = smoke.slice(smoke.indexOf("const allowed = tripped.filter("));
  assert.doesNotMatch(
    ringWalk.slice(0, ringWalk.indexOf('phase: "tripwires"')),
    /tripwiresFailFast/,
    "the end-of-run walk must fail the run in every mode; detection is not a mode");
  assert.match(smoke, /mode: tripwiresFailFast \? "failfast" : "end-of-run",/,
    "every run must record which mode produced its wall, so numbers are not compared across modes");
});

test("the ceiling profiler uses the authored band-1 fast formulation", async () => {
  assert.equal(POWER_DAM_LANE_ENVIRONMENT["ceiling-drop"].FLUID_SCENE,
    "ceiling-slab-drop");
  assert.equal(POWER_DAM_LANE_ENVIRONMENT["ceiling-drop"].FLUID_LANE, "performance");
  assert.equal(POWER_DAM_LANE_ENVIRONMENT["ceiling-drop"].FLUID_EXPECT_GRID,
    "24,16,24");
  assert.equal(POWER_DAM_LANE_ENVIRONMENT["ceiling-drop"].FLUID_MAXIMUM_LEAF_SIZE, "32");
  assert.equal(POWER_DAM_LANE_ENVIRONMENT["ceiling-drop"].FLUID_OCTREE_INTERFACE_BAND, "1");
  assert.equal(POWER_DAM_LANE_ENVIRONMENT["ceiling-drop"].FLUID_OCTREE_GLOBAL_FINE_FACTOR, "4");
  const smokeSource = await readFile(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.match(smokeSource,
    /const enforcedDiagnosticFindings = runOptions\.performanceProfile\s*\? \[\]/,
    "performance captures must rely on exact execution and packed final authority, not disabled scene collectors");
});

test("the large dam profiler pins leaf-32 and band-1", () => {
  const lane = POWER_DAM_LANE_ENVIRONMENT.large;
  assert.equal(lane.FLUID_SCENE, "large-power-dam-break");
  assert.equal(lane.FLUID_EXPECT_GRID, "64,20,64");
  assert.equal(lane.FLUID_MAXIMUM_LEAF_SIZE, "32");
  assert.equal(lane.FLUID_OCTREE_INTERFACE_BAND, "1");
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

test("compact and structured publication rejection reports exact authority evidence before abort", async () => {
  const smoke = await readFile(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  const readbacks = await readFile(new URL("../tools/webgpu-smoke-readbacks.ts", import.meta.url), "utf8");
  const functionStart = readbacks.indexOf("async function readCubicVolumeField(");
  const functionEnd = readbacks.indexOf("\nexport async function dumpFineRedistancePageDeltaForensics", functionStart);
  assert.notEqual(functionStart, -1, "compact field readback function must exist");
  assert.notEqual(functionEnd, -1, "compact field readback function must have a bounded end");
  const readback = normalizeWhitespace(readbacks.slice(functionStart, functionEnd));

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

  // Bounded by the structured audit block alone. The silent-failure tripwires
  // that follow it in the same step deliberately DO fence and map every step
  // (they must terminate the run at the step that tripped), so extending this
  // slice to the next unrelated statement would read their fence as a
  // regression in this one.
  const generationAudit = normalizeWhitespace(smoke.slice(
    smoke.indexOf("if (captureCompactPowerStep)"),
    smoke.indexOf("let capturedThisStep = false;", smoke.indexOf("if (captureCompactPowerStep)")),
  ));
  // The per-buffer copies are no longer open-coded here: one shared ABI writer
  // now serves both this harness and the browser's step-coherent snapshot ring,
  // so the two record byte-identical bytes. Asserting the call and its complete
  // source set keeps the "exact snapshot" contract while making an open-coded
  // copy that bypasses the shared writer -- and so could drift from the UI --
  // the thing that fails.
  assertContainsInOrder(generationAudit, [
    "encodeStructuredAuditRecordCopies(auditEncoder",
    "structuredVelocityControl:",
    "structuredBoundaryControl:",
    "fineWorklist:",
    "mgpcgControl:",
    "fineVolumeControl:",
    "projectionEnergyStats:",
    "device.queue.submit",
  ], "every accepted step must enqueue one exact structured/fine generation snapshot");
  assert.doesNotMatch(generationAudit, /auditEncoder\.copyBufferToBuffer/,
    "audit copies must go through the shared ABI writer the UI ring also uses");
  assert.doesNotMatch(generationAudit, /await awaitAdvanceCompletion|readBufferBinding|mapAsync/,
    "the recurring structured audit must not fence or map the GPU queue");
  assertContainsInOrder(smoke, [
    "const enableAuthoredStructuredEnergyProbe",
    "powerGenerationAuditRequested || collectStabilityEnvelope || energyEverySteps > 0",
    'process.env.FLUID_STRUCTURED_ENERGY_PROBE = "1"',
    "let solver: GPUSolverInstance",
    "finally {",
    "delete process.env.FLUID_STRUCTURED_ENERGY_PROBE",
  ], "authored audit collection must enable the energy producer before constructing structured dynamics");
  const terminalAudit = normalizeWhitespace(smoke.slice(
    smoke.indexOf("if (powerGenerationAuditSnapshot)",
      smoke.indexOf("const simulationWall_ms")),
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
