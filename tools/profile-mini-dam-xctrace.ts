/**
 * Capture a detailed CPU+GPU profile of a mini-dam-break simulation frame with
 * Instruments (`xctrace`), then reduce the trace to a report.
 *
 * Why this exists: the in-process instrumentation
 * (`FLUID_PERFORMANCE_TRACES`, `FLUID_GPU_PASS_TIMESTAMPS`) serialises passes
 * and restructures the submitted graph, so a traced run measures a *different*
 * frame than the one the benchmark gates on -- historically 2-4x apart. Metal
 * System Trace observes the untouched clean-regime frame from outside the
 * process, so what it reports is the frame that actually ships.
 *
 * Two Dawn toggles make this work:
 *  - `use_user_defined_labels_in_backend` propagates the repo's own compute-pass
 *    labels onto the Metal encoders, so GPU intervals are attributable by name.
 *  - `skip_validation` keeps the throughput regime the benchmark uses.
 *
 * Usage:
 *   node --import tsx tools/profile-mini-dam-xctrace.ts [options]
 *
 *   --lane=mini|moving-interface|ui   scene lane            (default mini)
 *   --steps=N                         advances to run       (default lane's own)
 *   --counters                        also record GPU hardware counters, which
 *                                     is what supplies occupancy / ALU /
 *                                     bandwidth per task. Counters emit ~3
 *                                     million rows per second of recording, so
 *                                     this mode does NOT record the whole run:
 *                                     it launches the solver, waits for it to
 *                                     reach steady state, and then attaches for
 *                                     --counter-seconds only.
 *   --counter-seconds=N               length of the attached counter window
 *                                     (default 2)
 *   --counter-warmup=N                seconds of stepping to let elapse before
 *                                     attaching, so the window lands in steady
 *                                     state (default: 60% of the expected run)
 *   --out=DIR                         artifact directory
 *   --baseline / --no-baseline        run an untraced pass first to quantify
 *                                     tracing distortion (default on)
 *   --keep-xml                        retain the exported XML tables
 *
 * Do NOT reach for xctrace's `--window`: it discards the metadata streams that
 * are recorded when the process starts, which erases every Metal object label
 * and the owning process from the retained intervals -- exactly the two fields
 * this report is built on. Record the whole run instead; Metal System Trace
 * costs ~1% of steady-state wall.
 */
import { execFileSync, spawn } from "node:child_process";
import { createWriteStream, mkdirSync, rmSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import {
  POWER_DAM_LANE_ENVIRONMENT,
  powerDamLaneWithSteps,
  type PowerDamRuntimeLane,
} from "./power-dam-lane-environment";
import { parseTraceTable, readTraceRows } from "./xctrace-trace-tables";
import { buildFrameReport, renderFrameReportHtml } from "./xctrace-frame-report";

const root = fileURLToPath(new URL("..", import.meta.url));
const worker = fileURLToPath(new URL("./run-webgpu-smoke-isolated-worker.ts", import.meta.url));

const flag = (name: string): string | undefined => process.argv
  .find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);

const lane = (flag("lane") ?? "mini") as PowerDamRuntimeLane;
if (!(lane in POWER_DAM_LANE_ENVIRONMENT)) {
  throw new Error(`--lane must be one of ${Object.keys(POWER_DAM_LANE_ENVIRONMENT).join(", ")}`);
}
const steps = flag("steps") === undefined ? undefined : Number(flag("steps"));
const counters = process.argv.includes("--counters");
const counterSeconds = Number(flag("counter-seconds") ?? 2);
const requestedWarmupSeconds = flag("counter-warmup") === undefined
  ? undefined : Number(flag("counter-warmup"));
const runBaseline = !process.argv.includes("--no-baseline");
const keepXml = process.argv.includes("--keep-xml");
const outputDirectory = resolve(root, flag("out") ?? "artifacts/xctrace-mini-dam");

// ---- Counter window planning ----------------------------------------------
// An attached counter run has to fit a recording window inside the solver's
// *stepping* phase, and two costs eat into that phase before a single usable
// sample lands. Both are measured on an M1 Max and both are silent: miss them
// and the capture succeeds, exports fine, and contains no GPU work of ours.

/**
 * Wall time between spawning `xctrace record --attach` and the recording
 * actually starting -- Instruments prints "Ctrl-C to stop the recording" at
 * that instant, and the `--time-limit` clock starts there. The attach must
 * therefore be issued this much *before* the window is due.
 */
const ATTACH_LATENCY_S = 2.7;
/**
 * The application-encoder metadata stream -- the join key every GPU interval
 * is attributed through -- only begins about a second into an attached
 * recording. Intervals before that arrive as bare `GPU Execution` and are not
 * attributable, so the usable part of the window is shorter than it looks.
 */
const ENCODER_STREAM_WARMUP_S = 1.1;
/**
 * Stepping that must remain after the window closes. The trace is truncated
 * the moment the attached process exits, so a window that runs to the last
 * advance loses its tail.
 */
const WINDOW_TAIL_S = 1.25;
/** Head-room over the bare minimum, absorbing attach jitter and pace error. */
const WINDOW_SLACK = 1.25;
/** Pace assumed when no untraced baseline was run to measure it. */
const ASSUMED_MS_PER_ADVANCE = 50;

export interface CounterWindowPlan {
  /** Advances the traced run must make for the window to fit. */
  readonly steps: number;
  /** Seconds after construction at which to spawn `xctrace`. */
  readonly spawnDelaySeconds: number;
  /** Seconds after construction at which recording is expected to start. */
  readonly windowStartSeconds: number;
  /** Seconds of the window whose intervals carry encoder labels. */
  readonly labelledSeconds: number;
  /** Expected duration of the stepping phase at the assumed pace. */
  readonly steppingSeconds: number;
  /** True when `steps` had to be raised above what was asked for. */
  readonly extended: boolean;
}

/**
 * Place the counter window inside the stepping phase, lengthening the run if
 * it is too short to hold one. Returning the spawn delay separately from the
 * window start is the whole point: they differ by `ATTACH_LATENCY_S`, and the
 * profiler used to conflate them, which is how a 125-advance run attached
 * 0.8 s after it had stopped stepping.
 */
export const planCounterWindow = (input: {
  readonly requestedSteps: number;
  readonly perAdvanceMs: number;
  readonly counterSeconds: number;
  readonly requestedWarmupSeconds?: number;
}): CounterWindowPlan => {
  const perAdvance = Math.max(input.perAdvanceMs, 1) / 1000;
  const minimumStepping = ATTACH_LATENCY_S + input.counterSeconds + WINDOW_TAIL_S;
  const wantedStepping = WINDOW_SLACK * minimumStepping;
  const requestedStepping = input.requestedSteps * perAdvance;
  const extended = requestedStepping < wantedStepping;
  const steps = extended
    ? Math.ceil(wantedStepping / perAdvance) : input.requestedSteps;
  const steppingSeconds = steps * perAdvance;
  const latestStart = steppingSeconds - input.counterSeconds - WINDOW_TAIL_S;
  const desiredStart = input.requestedWarmupSeconds ?? 0.6 * steppingSeconds;
  const windowStartSeconds = Math.max(ATTACH_LATENCY_S, Math.min(desiredStart, latestStart));
  return {
    steps,
    spawnDelaySeconds: Math.max(0, windowStartSeconds - ATTACH_LATENCY_S),
    windowStartSeconds,
    labelledSeconds: Math.max(0, input.counterSeconds - ENCODER_STREAM_WARMUP_S),
    steppingSeconds,
    extended,
  };
};

const requestedSteps = steps ?? Number(POWER_DAM_LANE_ENVIRONMENT[lane].FLUID_ORACLE_STEPS);
/**
 * The step count is fixed before either run starts so the baseline and the
 * traced run remain comparable, which means the plan that sizes it can only
 * use the assumed pace. Placement is re-planned against the measured pace once
 * the baseline has run.
 */
const sizingPlan = counters ? planCounterWindow({
  requestedSteps,
  perAdvanceMs: ASSUMED_MS_PER_ADVANCE,
  counterSeconds,
  requestedWarmupSeconds,
}) : undefined;
const laneSteps = sizingPlan?.steps ?? steps;

/**
 * The clean measurement regime. Every in-process probe is off: those probes
 * are what distort the frame, and Instruments replaces them from outside.
 */
const laneEnvironment = laneSteps === undefined
  ? POWER_DAM_LANE_ENVIRONMENT[lane] : powerDamLaneWithSteps(lane, laneSteps);

const profileEnvironment: Record<string, string> = {
  WEBGPU_NODE_MODULE: `${root}node_modules/webgpu/index.js`,
  FLUID_WEBGPU_BACKEND: "metal",
  FLUID_WEBGPU_ADAPTER: "Apple M1 Max",
  // `use_user_defined_labels_in_backend` is what makes GPU intervals
  // attributable to the repo's own pass names. Measured cost: none.
  FLUID_WEBGPU_DAWN_FEATURES: "skip_validation,use_user_defined_labels_in_backend",
  FLUID_METHOD: "octree",
  FLUID_QUALITY: "balanced",
  FLUID_PERFORMANCE_PROFILE: "1",
  FLUID_PERFORMANCE_TRACES: "0",
  FLUID_GPU_FINE_TIMESTAMPS: "0",
  FLUID_GPU_PASS_TIMESTAMPS: "0",
  FLUID_ALGORITHM_DIAGNOSTICS: "0",
  FLUID_GPU_COMMAND_AUDIT: "1",
  FLUID_STABILITY_ENVELOPE: "0",
  FLUID_CHECKPOINT_EVERY_S: "0",
  FLUID_CPU_ORACLE: "0",
  FLUID_FIELD_STATS: "0",
  FLUID_SPARSE_STATS: "0",
  FLUID_RASTER_CHECKPOINTS: "0",
  FLUID_WEBGPU_SMOKE_TIMEOUT_MS: "240000",
  FLUID_TRIPWIRES: "1",
  ...laneEnvironment,
};

export interface SmokeResultRecord {
  readonly simulationWall_ms?: number;
  readonly steps?: number;
  readonly construction_ms?: number;
  readonly gpuCommandAudit?: Record<string, unknown>;
}

interface WorkerHandle {
  readonly pid: number;
  /** Resolves when the solver has finished construction and is stepping. */
  readonly constructed: Promise<void>;
  readonly finished: Promise<SmokeResultRecord | undefined>;
  /**
   * Wall clocks for the stepping phase, filled in as the worker announces it.
   * A counter window is only meaningful if it lies between these two.
   */
  readonly timing: { steppingStartedAt?: number; steppingEndedAt?: number };
}

/** Spawn the worker, forwarding output to `logPath`. */
const startWorker = (
  argv: readonly string[],
  logPath: string,
  label: string,
): WorkerHandle => {
  const log = createWriteStream(logPath);
  const child = spawn(argv[0], argv.slice(1), {
    cwd: root,
    env: { ...process.env, ...profileEnvironment },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let result: SmokeResultRecord | undefined;
  let announceConstructed: () => void = () => {};
  const constructed = new Promise<void>((done) => { announceConstructed = done; });
  const timing: { steppingStartedAt?: number; steppingEndedAt?: number } = {};
  const lines = createInterface({ input: child.stdout });
  lines.on("line", (line) => {
    log.write(`${line}\n`);
    if (!line.startsWith("{")) return;
    try {
      const record = JSON.parse(line) as { phase?: string } & SmokeResultRecord;
      if (record.phase === "result") { result = record; timing.steppingEndedAt = Date.now(); }
      if (record.phase === "constructed") {
        console.log(`  ${label}: constructed in ${record.construction_ms} ms`);
        timing.steppingStartedAt = Date.now();
        announceConstructed();
      }
    } catch { /* progress lines are not all JSON */ }
  });
  child.stderr.pipe(log);
  const finished = new Promise<SmokeResultRecord | undefined>((done, fail) => {
    child.once("error", fail);
    child.once("exit", (status, signal) => {
      log.end();
      announceConstructed();
      if (signal) fail(new Error(`${label} exited from ${signal}`));
      else if (status !== 0) fail(new Error(`${label} exited with ${status}; see ${logPath}`));
      else done(result);
    });
  });
  return { pid: child.pid ?? -1, constructed, finished, timing };
};

const runWorker = async (
  argv: readonly string[],
  logPath: string,
  label: string,
): Promise<SmokeResultRecord | undefined> => startWorker(argv, logPath, label).finished;

const delay = (ms: number): Promise<void> => new Promise((done) => { setTimeout(done, ms); });

// ---- Trace tables ----------------------------------------------------------
// Each entry becomes one exported XML table reduced to NDJSON.

const TABLES: readonly {
  schema: string; file: string; stacks?: boolean; countersOnly?: boolean;
}[] = [
  { schema: "metal-gpu-intervals", file: "gpu-intervals" },
  { schema: "metal-application-encoders-list", file: "encoders" },
  { schema: "time-profile", file: "time-profile", stacks: true },
  { schema: "gpu-counter-info", file: "counter-info", countersOnly: true },
  { schema: "gpu-counter-value", file: "counter-value", countersOnly: true },
  { schema: "metal-shader-profiler-intervals", file: "shader-profile", countersOnly: true },
];

const run = (
  argv: readonly string[],
  onLine?: (line: string) => void,
): Promise<void> => new Promise((done, fail) => {
  const child = spawn(argv[0], argv.slice(1), { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
  let stderr = "";
  child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
  if (onLine) {
    // xctrace narrates its own progress on both streams; the caller needs the
    // moment recording starts, which is the only ground truth for where the
    // window landed.
    for (const stream of [child.stdout, child.stderr]) {
      createInterface({ input: stream }).on("line", onLine);
    }
  } else child.stdout.resume();
  child.once("exit", (code) => (code === 0 ? done()
    : fail(new Error(`${argv.join(" ")} failed (${code}): ${stderr.slice(0, 400)}`))));
  child.once("error", fail);
});

/**
 * Which processes the exported GPU intervals belong to. Checked before the
 * counter tables are exported: those are millions of rows and several minutes,
 * and there is no point spending either on a trace that caught none of our work.
 */
const intervalOwners = async (ndjson: string): Promise<Map<string, number>> => {
  const tally = new Map<string, number>();
  for await (const row of readTraceRows(ndjson)) {
    const owner = String(row.process ?? "(unknown)");
    tally.set(owner, (tally.get(owner) ?? 0) + 1);
  }
  return tally;
};

const main = async (): Promise<void> => {
  mkdirSync(outputDirectory, { recursive: true });
  const tracePath = `${outputDirectory}/mini-dam.trace`;
  const nodeBinary = process.execPath;

  console.log(`lane ${lane}: ${profileEnvironment.FLUID_ORACLE_STEPS} advances`
    + ` of ${profileEnvironment.FLUID_SCENE} at grid ${profileEnvironment.FLUID_EXPECT_GRID}`);
  if (sizingPlan?.extended) {
    console.log(`  run extended from ${requestedSteps} to ${sizingPlan.steps} advances:`
      + ` a ${counterSeconds} s counter window plus Instruments' ${ATTACH_LATENCY_S} s attach`
      + " does not fit inside a shorter stepping phase");
  }

  let baseline: SmokeResultRecord | undefined;
  if (runBaseline) {
    console.log("recording untraced baseline (establishes the tracing distortion factor)...");
    baseline = await runWorker(
      [nodeBinary, "--import", "tsx", worker],
      `${outputDirectory}/baseline.log`,
      "baseline",
    );
    if (baseline?.simulationWall_ms && baseline.steps) {
      console.log(`  baseline: ${(baseline.simulationWall_ms / baseline.steps).toFixed(2)}`
        + ` ms/advance over ${baseline.steps} advances`);
    }
  }

  rmSync(tracePath, { recursive: true, force: true });
  let traced: SmokeResultRecord | undefined;
  /** Known only in attach mode; in launch mode xctrace owns the process. */
  let tracedPid: number | undefined;

  if (counters) {
    // GPU counters emit ~3 M rows per second of recording, which makes a
    // whole-run capture unexportable. Instead let the solver run on its own,
    // wait for it to reach steady state, then attach for a bounded window.
    // This also puts the sample where it belongs: the mini lane's cost grows
    // steadily with step index, so the interesting frames are the late ones.
    const perAdvance = baseline?.simulationWall_ms && baseline.steps
      ? baseline.simulationWall_ms / baseline.steps : ASSUMED_MS_PER_ADVANCE;
    const plan = planCounterWindow({
      requestedSteps: Number(profileEnvironment.FLUID_ORACLE_STEPS),
      perAdvanceMs: perAdvance,
      counterSeconds,
      requestedWarmupSeconds,
    });
    if (plan.extended) {
      console.log(`  warning: at ${perAdvance.toFixed(1)} ms/advance this run steps for only`
        + ` ${(Number(profileEnvironment.FLUID_ORACLE_STEPS) * perAdvance / 1000).toFixed(1)} s,`
        + ` too short to hold a ${counterSeconds} s window;`
        + ` rerun with --steps=${plan.steps} or a smaller --counter-seconds`);
    }
    console.log(`launching solver, attaching GPU counters at`
      + ` ${plan.spawnDelaySeconds.toFixed(1)} s so recording starts`
      + ` ${plan.windowStartSeconds.toFixed(1)} s into stepping`
      + ` (Instruments takes ~${ATTACH_LATENCY_S} s to attach) for a ${counterSeconds} s window,`
      + ` of which ~${plan.labelledSeconds.toFixed(1)} s carries encoder labels...`);
    const handle = startWorker(
      [nodeBinary, "--import", "tsx", worker], `${outputDirectory}/traced.log`, "traced",
    );
    tracedPid = handle.pid;
    await handle.constructed;
    await delay(plan.spawnDelaySeconds * 1000);
    let recordingStartedAt: number | undefined;
    await run(["xcrun", "xctrace", "record",
      "--template", "Metal System Trace",
      "--instrument", "Metal GPU Counters",
      "--output", tracePath, "--no-prompt",
      "--time-limit", `${counterSeconds}s`,
      "--attach", String(handle.pid)], (line) => {
      if (recordingStartedAt === undefined && /Ctrl-C to stop the recording/.test(line)) {
        recordingStartedAt = Date.now();
      }
    });
    console.log("  counter window captured; waiting for the run to finish...");
    traced = await handle.finished;

    // Where the window actually landed. Everything about an attached counter
    // run depends on this overlapping the stepping phase, and when it does not
    // the trace still exports cleanly -- it just holds no GPU work of ours.
    const { steppingStartedAt, steppingEndedAt } = handle.timing;
    if (recordingStartedAt !== undefined && steppingStartedAt !== undefined) {
      const opened = (recordingStartedAt - steppingStartedAt) / 1e3;
      const stepped = steppingEndedAt === undefined
        ? undefined : (steppingEndedAt - steppingStartedAt) / 1e3;
      console.log(`  window: recording ran ${opened.toFixed(2)}`
        + `-${(opened + counterSeconds).toFixed(2)} s into stepping`
        + (stepped === undefined ? "" : `; the solver stepped for ${stepped.toFixed(2)} s`));
      if (stepped !== undefined && opened >= stepped) {
        throw new Error(`the counter window opened ${(opened - stepped).toFixed(2)} s after the`
          + " solver stopped stepping, so the trace holds none of its GPU work."
          + ` Rerun with more advances (--steps) or an earlier --counter-warmup`);
      }
      if (stepped !== undefined && opened + counterSeconds > stepped) {
        console.log(`  warning: the window outlasted the run by`
          + ` ${(opened + counterSeconds - stepped).toFixed(2)} s; the trace is truncated`
          + " where the process exited");
      }
    }
  } else {
    console.log("recording Metal System Trace...");
    traced = await runWorker([
      "xcrun", "xctrace", "record",
      "--template", "Metal System Trace",
      "--output", tracePath,
      "--no-prompt",
      "--target-stdout", "-",
      ...Object.entries(profileEnvironment).flatMap(([key, value]) => ["--env", `${key}=${value}`]),
      "--launch", "--", nodeBinary, "--import", "tsx", worker,
    ], `${outputDirectory}/traced.log`, "traced");
  }
  // A .trace is a bundle directory, so stat() reports the inode, not the run.
  const traceMegabytes = Number(execFileSync("du", ["-sm", tracePath])
    .toString().split("\t")[0]);
  console.log(`  trace: ${traceMegabytes} MB at ${tracePath}`);

  console.log("exporting trace tables...");
  const tables: Record<string, string> = {};
  for (const table of TABLES) {
    if (table.countersOnly && !counters) continue;
    const xml = `${outputDirectory}/${table.file}.xml`;
    const ndjson = `${outputDirectory}/${table.file}.ndjson`;
    await run(["xcrun", "xctrace", "export", "--input", tracePath,
      "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${table.schema}"]`,
      "--output", xml]);
    const batch: string[] = [];
    const sink = createWriteStream(ndjson);
    let rows = 0;
    for await (const row of parseTraceTable(xml, { stacks: table.stacks })) {
      batch.push(JSON.stringify(row));
      rows += 1;
      if (batch.length >= 4096) { sink.write(`${batch.join("\n")}\n`); batch.length = 0; }
    }
    if (batch.length > 0) sink.write(`${batch.join("\n")}\n`);
    await new Promise((done) => sink.end(done));
    if (!keepXml) rmSync(xml, { force: true });
    tables[table.file] = ndjson;
    console.log(`  ${table.schema}: ${rows} rows`);
    if (table.file === "gpu-intervals" && tracedPid !== undefined) {
      // Answer "did we catch our own GPU work" before exporting the counter
      // tables, which are millions of rows and minutes of work either way.
      const owners = await intervalOwners(ndjson);
      const mine = owners.get(`node (${tracedPid})`) ?? 0;
      if (mine === 0) {
        const seen = [...owners.entries()].sort((left, right) => right[1] - left[1])
          .slice(0, 4).map(([process, count]) => `${process} ${count}`).join(", ");
        throw new Error(`the trace holds no GPU work from the traced process (node ${tracedPid});`
          + ` its ${rows} intervals belong to ${seen || "nobody"}.`
          + " The counter window missed the stepping phase --"
          + " raise --steps or lower --counter-seconds / --counter-warmup.");
      }
      console.log(`    ${mine} of them from the traced process`);
    }
  }

  console.log("reducing to a frame report...");
  const report = await buildFrameReport({
    tables,
    lane,
    environment: profileEnvironment,
    traced,
    baseline,
    tracedPid,
  });
  await writeFile(`${outputDirectory}/summary.json`, `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(`${outputDirectory}/report.html`, renderFrameReportHtml(report));

  console.log("");
  for (const line of report.console) console.log(line);
  console.log("");
  console.log(`report:  ${outputDirectory}/report.html`);
  console.log(`summary: ${outputDirectory}/summary.json`);
  console.log(`trace:   ${tracePath}   (open with: open ${tracePath})`);
};

// Only capture when run as a program. Importing this module -- a test reaching
// for `planCounterWindow`, say -- must not start a multi-minute GPU recording.
const isMain = process.argv[1] !== undefined
  && import.meta.url === new URL(`file://${process.argv[1]}`).href;

if (isMain) await main();
