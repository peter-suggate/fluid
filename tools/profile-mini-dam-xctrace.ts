/**
 * Capture a detailed GPU-stage profile of a mini-dam-break simulation frame with
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
 *   --lane=mini|large|moving-interface|ui|ocean|ceiling-drop scene lane (default mini)
 *   --band=0|1|2|3|4                override the octree interface-band level
 *                                    after applying the lane preset
 *   --first-frame                    gate before advance 1 and reduce the
 *                                    capture to literal advance 1
 *   --counter-start-gate             gate before advance 1 only until the
 *                                    counter recorder is live, then reduce a
 *                                    representative recurring advance
 *   --counter-gate-warmup-ms=N       delay after the recorder is live before
 *                                    releasing that gate (default 1200; use 0
 *                                    when a large isolated frame would exhaust
 *                                    the bounded label stream first)
 *   --steps=N                         advances to run       (default lane's own)
 *   The report always reduces the counter window to exactly one complete,
 *   representative advance. The longer traced run exists only to warm encoder
 *   metadata and supply a choice of complete steady-state advances.
 *   GPU hardware counters and full pass-label isolation are mandatory. This
 *   is what supplies occupancy / ALU / bandwidth per task and prevents a Metal
 *   encoder from being named after only its first WebGPU stage. Counters emit ~3
 *                                     million rows per second of recording, so
 *                                     this mode does NOT record the whole run:
 *                                     it launches the solver, waits for it to
 *                                     reach steady state, and then attaches for
 *                                     --counter-seconds only.
 *   --counter-seconds=N               length of the attached counter window
 *                                     (minimum/default 3; full label metadata
 *                                     needs a longer warm-up than scoped traces)
 *   --counter-warmup=N                seconds of stepping to let elapse before
 *                                     attaching, so the window lands in steady
 *                                     state (default: 60% of the expected run)
 *   --out=DIR                         artifact directory
 *   --baseline / --no-baseline        run the clean, non-isolated shipping
 *                                     graph first to quantify total profiling
 *                                     distortion (default on)
 *   The clean baseline remains non-isolated, so the report still records the
 *   shipping wall clock separately from the fully-labelled diagnostic stream.
 *   --keep-xml                        retain the exported XML tables
 *   --counter-reduction=N             retain roughly 1/N counter rows while
 *                                     preserving exact GPU stages (default 100)
 *   --full-diagnostics                additionally capture/export Time Profiler
 *                                     and the other Metal System Trace instruments;
 *                                     the default is GPU-stage diagnostics only
 *   --gpu-report-only                 omit CPU sampling and shader-profiler
 *                                     exports from a GPU utilization report
 *   --discard-tables                  delete derived NDJSON tables after the
 *                                     self-contained HTML/JSON report is built
 *   --discard-trace                   delete the source .trace after the report
 *                                     is built; keep it on failed reductions
 *   --reuse-tables                    rebuild a rejected report from retained
 *                                     NDJSON tables without recapturing
 *
 * Every Dawn entrypoint in the repo serialises on `/tmp/fluid-webgpu-exclusive.lock`,
 * and that path is absolute, so a benchmark running out of any other checkout
 * excludes this capture too. The profiler checks the lock before it spends
 * anything, and names the holder when it finds one.
 *
 * Do NOT reach for xctrace's `--window`: it discards the metadata streams that
 * are recorded when the process starts, which erases every Metal object label
 * and the owning process from the retained intervals -- exactly the two fields
 * this report is built on. Record the whole run instead; Metal System Trace
 * costs ~1% of steady-state wall.
 */
import { execFileSync, spawn } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync, realpathSync, rmSync, statSync }
  from "node:fs";
import { writeFile } from "node:fs/promises";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import {
  POWER_DAM_LANE_ENVIRONMENT,
  powerDamLaneWithSteps,
  type PowerDamRuntimeLane,
} from "./power-dam-lane-environment";
import {
  readWebGPUExclusiveLockHolder,
  WEBGPU_EXCLUSIVE_LOCK,
} from "./webgpu-smoke-isolation";
import { parseTraceTable, readTraceRows } from "./xctrace-trace-tables";
import {
  buildFrameReport,
  GPU_FRAME_START_LABELS,
  renderFrameReportHtml,
} from "./xctrace-frame-report";

const root = fileURLToPath(new URL("..", import.meta.url));
const worker = fileURLToPath(new URL("./run-webgpu-smoke-isolated-worker.ts", import.meta.url));

const flag = (name: string): string | undefined => process.argv
  .find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);

const lane = (flag("lane") ?? "mini") as PowerDamRuntimeLane;
if (!(lane in POWER_DAM_LANE_ENVIRONMENT)) {
  throw new Error(`--lane must be one of ${Object.keys(POWER_DAM_LANE_ENVIRONMENT).join(", ")}`);
}
const bandLevel = flag("band") === undefined ? undefined : Number(flag("band"));
if (bandLevel !== undefined
  && (!Number.isInteger(bandLevel) || bandLevel < 0 || bandLevel > 4)) {
  throw new Error("--band must be an integer from 0 through 4");
}
const steps = flag("steps") === undefined ? undefined : Number(flag("steps"));
const firstFrame = process.argv.includes("--first-frame");
// Counter attach takes ~2.7 s, while the isolated Metal-encoder metadata
// stream can fill its bounded trace buffer in ~4 s. A normal warmup therefore
// cannot make labels and counters overlap reliably. This gate starts the
// recorder before recurring work without changing representative-frame
// selection (unlike --first-frame, which intentionally selects bootstrap).
const counterStartGate = process.argv.includes("--counter-start-gate");
const profileGate = firstFrame || counterStartGate;
const counters = true;
const counterSeconds = Number(flag("counter-seconds") ?? 3);
if (!Number.isFinite(counterSeconds) || counterSeconds < 3) {
  throw new Error("--counter-seconds must be at least 3 for full labels plus occupancy");
}
const counterGateWarmupMs = Number(flag("counter-gate-warmup-ms") ?? 1_200);
if (!Number.isFinite(counterGateWarmupMs) || counterGateWarmupMs < 0
  || counterGateWarmupMs > counterSeconds * 1_000) {
  throw new Error("--counter-gate-warmup-ms must be between 0 and the counter window");
}
const counterTimeLimit = Number.isInteger(counterSeconds)
  ? `${counterSeconds}s` : `${Math.round(counterSeconds * 1000)}ms`;
const requestedWarmupSeconds = flag("counter-warmup") === undefined
  ? undefined : Number(flag("counter-warmup"));
const runBaseline = !process.argv.includes("--no-baseline");
const isolatePassLabels = true;
const isolateLabelPrefix = undefined;
if (process.argv.some((argument) => argument.startsWith("--isolate-label-prefix="))) {
  throw new Error("Partial xctrace label isolation is disabled: capture every label instead");
}
if (process.argv.includes("--no-counters")
  || process.argv.includes("--no-isolate-pass-labels")) {
  throw new Error("xctrace captures require GPU counters and full label isolation");
}
const keepXml = process.argv.includes("--keep-xml");
const gpuReportOnly = process.argv.includes("--gpu-report-only");
const fullDiagnostics = process.argv.includes("--full-diagnostics");
const counterReduction = Number(flag("counter-reduction") ?? 100);
if (!Number.isFinite(counterReduction) || counterReduction < 1) {
  throw new Error("--counter-reduction must be at least 1");
}
const discardTables = process.argv.includes("--discard-tables");
const discardTrace = process.argv.includes("--discard-trace");
const reuseTables = process.argv.includes("--reuse-tables");
const outputDirectory = resolve(root, flag("out") ?? "artifacts/xctrace-mini-dam");

/** Stage timings and labels come from Metal interval tables and are never
 * downsampled. These are the only counter series consumed by the report; LLC
 * is retained for its representative-frame plot. */
export const RETAINED_GPU_COUNTERS = new Set([
  "Compute Occupancy",
  "ALU Utilization",
  "GPU Read Bandwidth",
  "GPU Write Bandwidth",
  "GPU Last Level Cache Utilization",
]);

export interface CounterExtractionPolicy {
  readonly targetReduction: number;
  readonly sourceCounterCount: number;
  readonly retainedCounterCount: number;
  readonly timestampStride: number;
  readonly retainedCounterIds: ReadonlySet<string>;
}

/** Select fewer counter names and then enough timestamps to meet the requested
 * total row reduction. On the current 31-counter device, 5 names x every 17th
 * timestamp retains 1/105.4 of the source rows. */
export const makeCounterExtractionPolicy = (
  countersById: ReadonlyMap<string, string>,
  targetReduction = 100,
  retainedCounterNames: ReadonlySet<string> = RETAINED_GPU_COUNTERS,
): CounterExtractionPolicy => {
  const retainedCounterIds = targetReduction === 1
    ? new Set(countersById.keys())
    : new Set([...countersById].filter(([, name]) => retainedCounterNames.has(name))
      .map(([id]) => id));
  if (retainedCounterIds.size === 0) {
    throw new Error("none of the report's required GPU counters were present");
  }
  const timestampStride = Math.max(1, Math.ceil(targetReduction
    * retainedCounterIds.size / Math.max(countersById.size, 1)));
  return {
    targetReduction,
    sourceCounterCount: countersById.size,
    retainedCounterCount: retainedCounterIds.size,
    timestampStride,
    retainedCounterIds,
  };
};

export class CounterRowSelector {
  private timestamp: string | undefined;
  private timestampIndex = -1;
  private retainTimestamp = false;

  public constructor(private readonly policy: CounterExtractionPolicy) {}

  public keep(row: Readonly<Record<string, unknown>>): boolean {
    const timestamp = String(row.timestamp ?? "");
    if (timestamp !== this.timestamp) {
      this.timestamp = timestamp;
      this.timestampIndex += 1;
      this.retainTimestamp = this.timestampIndex % this.policy.timestampStride === 0;
    }
    return this.retainTimestamp
      && this.policy.retainedCounterIds.has(String(row["counter-id"] ?? ""));
  }
}

interface InstrumentsScratchEntry {
  readonly path: string;
  readonly kind: "file" | "directory";
  readonly sizeBytes: number;
  readonly modifiedAt: string;
}

/** Instruments writes its raw ktrace and reduction stores outside the trace
 * bundle under opaque names. Record the exact names with the artifact: failed
 * captures otherwise leave multi-gigabyte files that are nearly impossible to
 * associate with the run that created them. */
const scratchRoots = (() => {
  const temporary = process.env.TMPDIR?.replace(/\/$/, "");
  if (!temporary || !existsSync(temporary)) return [] as readonly string[];
  const roots = [temporary,
    resolve(temporary, "../C/com.apple.dt.InstrumentsCLI/path_manager")]
    .filter((path) => existsSync(path)).map((path) => realpathSync(path));
  return [...new Set(roots)];
})();

const instrumentsScratchSnapshot = (): Map<string, InstrumentsScratchEntry> => {
  const entries = new Map<string, InstrumentsScratchEntry>();
  for (const directory of scratchRoots) {
    for (const name of readdirSync(directory)) {
      if (!/^instruments.*\.ktrace$/.test(name) && !/^xrtmp__/.test(name)) continue;
      const path = resolve(directory, name);
      try {
        const metadata = statSync(path);
        entries.set(path, {
          path,
          kind: metadata.isDirectory() ? "directory" : "file",
          sizeBytes: metadata.size,
          modifiedAt: metadata.mtime.toISOString(),
        });
      } catch { /* Instruments may remove a transient between readdir and stat. */ }
    }
  }
  return entries;
};

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
const sizingPlan = counters && !firstFrame ? planCounterWindow({
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
  // Timestamping supplies the diagnostic encoder-isolation scratch. Only one
  // filtered command buffer is retained; xctrace remains the counter source.
  FLUID_GPU_PASS_TIMESTAMPS: isolatePassLabels ? "1" : "0",
  // Diagnostic micro-stage mode: PassBroker normally keeps one compute pass
  // open across several labelled stages, so Metal only retains the first
  // label. Isolation makes each label a real encoder/counter attribution unit.
  FLUID_GPU_ISOLATE_PASS_LABELS: isolatePassLabels ? "1" : "0",
  FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES: isolatePassLabels && isolateLabelPrefix
    ? isolateLabelPrefix : "",
  // Instruments attributes counters to Metal encoders, and Dawn may merge
  // several distinct WebGPU passes into one encoder. Combined with label
  // isolation, this makes one micro-stage equal one counter attribution unit.
  FLUID_GPU_ISOLATE_PASS_ENCODERS: isolatePassLabels ? "1" : "0",
  FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS: "1",
  FLUID_GPU_PASS_TIMESTAMP_LABEL_PREFIXES:
    isolateLabelPrefix ?? "Fine JFA -,SPGrid accurate A2 -,SPGrid Section 6.3 -",
  FLUID_ALGORITHM_DIAGNOSTICS: "0",
  FLUID_GPU_COMMAND_AUDIT: "1",
  // Match benchmark-power-dam's shipping graph. Scene catalog defaults enable
  // the exhaustive generation/energy audit, which is validation work rather
  // than part of the profiled solver frame; the lightweight tripwires below
  // remain mandatory.
  FLUID_POWER_GENERATION_AUDIT: "0",
  FLUID_STABILITY_ENVELOPE: "0",
  FLUID_CHECKPOINT_EVERY_S: "0",
  FLUID_CPU_ORACLE: "0",
  FLUID_FIELD_STATS: "0",
  FLUID_SPARSE_STATS: "0",
  FLUID_RASTER_CHECKPOINTS: "0",
  FLUID_WEBGPU_SMOKE_TIMEOUT_MS: "240000",
  FLUID_TRIPWIRES: "1",
  FLUID_PROFILE_FIRST_ADVANCE_GATE: profileGate ? "1" : "0",
  ...laneEnvironment,
  // This deliberately follows the lane preset. A shell-level override cannot
  // win against startWorker's explicit environment, which previously made
  // comparative traces silently profile the lane's default band every time.
  ...(bandLevel === undefined ? {} : { FLUID_OCTREE_INTERFACE_BAND: String(bandLevel) }),
};
/** Shipping command graph used as the wall-clock control. A targeted capture
 * intentionally adds pass/Metal-encoder boundaries, so comparing it with an
 * untraced run that kept those boundaries would conceal the main measurement
 * distortion the user needs to see. */
export const makeCleanBaselineEnvironment = (
  environment: Readonly<Record<string, string>>,
): Record<string, string> => ({
  ...environment,
  // A literal-first-frame trace pauses the *traced* worker until Instruments
  // has attached. The shipping control is not attached and must never inherit
  // that gate or it waits forever for a signal nobody owns.
  FLUID_PROFILE_FIRST_ADVANCE_GATE: "0",
  FLUID_GPU_PASS_TIMESTAMPS: "0",
  FLUID_GPU_ISOLATE_PASS_LABELS: "0",
  FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES: "",
  FLUID_GPU_ISOLATE_PASS_ENCODERS: "0",
  FLUID_GPU_PASS_TIMESTAMP_COMMAND_BUFFERS: "0",
  FLUID_GPU_PASS_TIMESTAMP_LABEL_PREFIXES: "",
});
const cleanBaselineEnvironment = makeCleanBaselineEnvironment(profileEnvironment);

/**
 * `xctrace --launch` inherits this process's environment, and it cannot be
 * given a variable whose value is empty. A profile variable set to "" is an
 * instruction to the worker to run *without* that setting, so the only way to
 * deliver it through xctrace is to make it absent on both paths: omitted from
 * the forwarded `--env` list, and removed from what the worker inherits.
 */
function clearEmptyProfileVariables(): void {
  for (const [key, value] of Object.entries(profileEnvironment)) {
    if (value === "") delete process.env[key];
  }
}

export interface SmokeResultRecord {
  readonly simulationWall_ms?: number;
  readonly steps?: number;
  readonly construction_ms?: number;
  readonly gpuCommandAudit?: Record<string, unknown>;
}

export const assertCompleteOccupancyReport = (report: Pick<
  import("./xctrace-frame-report").FrameReport,
  "attribution" | "counters" | "passes" | "frames" | "timeline"
>): void => {
  const failures: string[] = [];
  if (report.attribution.mode !== "full" || report.attribution.compositeBuckets !== 0) {
    failures.push(`label isolation is ${report.attribution.mode}`
      + ` with ${report.attribution.compositeBuckets} composite buckets`);
  }
  if (report.counters.meanOccupancy === undefined
    || report.counters.partitionCount < 1
    || report.counters.occupancyTrace.length === 0) {
    failures.push("Compute Occupancy was not captured across a labelled representative advance");
  }
  if (!report.passes.some((pass) => pass.counterSamples > 0
    && pass.occupancy !== undefined)) {
    failures.push("no labelled GPU task received occupancy samples");
  }
  if (report.frames.count !== 1 || report.frames.samples.length !== 1
    || report.frames.captures.length !== 1) {
    failures.push(`report contains ${report.frames.count} analysed advances and`
      + ` ${report.frames.captures.length} captured advances instead of exactly one`);
  }
  const encoderIds = report.timeline.intervals
    .map((interval) => interval.encoderId).filter((id): id is string => id !== undefined);
  if (new Set(encoderIds).size !== encoderIds.length) {
    failures.push("one Metal encoder appears more than once in the stage timeline");
  }
  if (!GPU_FRAME_START_LABELS.some((label) => label === report.frames.anchor)) {
    failures.push(`frame anchor "${report.frames.anchor}" is periodic but not a semantic GPU start`);
  } else {
    const first = report.timeline.intervals[0];
    const semanticStart = report.timeline.intervals
      .find((interval) => interval.label === report.frames.anchor);
    // Distinct command buffers can overlap at an advance boundary. In the
    // current dam graph the deterministic fine-seed residency publication is
    // real work for the new advance and can reach the GPU just before the
    // topology gate. Keep that stage in the frame, while still requiring both
    // the observed work and the semantic boundary to occur close to t=0.
    if (first === undefined || first.start > 250 || semanticStart === undefined
      || semanticStart.start > 5_000) {
      failures.push(`frame origin does not contain semantic GPU start "${report.frames.anchor}"`
        + " within 5000 us after near-zero recurring work"
        + ` (first ${first?.label ?? "no stage"} at ${first?.start ?? "?"} us;`
        + ` semantic start at ${semanticStart?.start ?? "?"} us)`);
    }
  }
  if (failures.length > 0) {
    throw new Error("xctrace report rejected: " + failures.join("; ")
      + ". The trace and exported tables were retained for diagnosis.");
  }
};

interface WorkerHandle {
  readonly pid: number;
  /**
   * Resolves when the solver has finished construction and is stepping, and
   * rejects if it exits before getting there. It must never resolve on a
   * premature exit: what waits on it goes on to attach Instruments, and
   * attaching to a dead pid costs the whole capture and reports nothing.
   */
  readonly constructed: Promise<void>;
  /** Resolves at the profiling gate after t=0 diagnostics and before the first
   * recurring command buffer is encoded. */
  readonly beforeFirstAdvance: Promise<void>;
  readonly finished: Promise<SmokeResultRecord | undefined>;
  /**
   * Wall clocks for the stepping phase, filled in as the worker announces it.
   * A counter window is only meaningful if it lies between these two.
   */
  readonly timing: { steppingStartedAt?: number; steppingEndedAt?: number };
  /**
   * Terminate the worker if it is still running. The profiler owns a solver
   * that holds the machine-wide GPU lock, so any path that abandons a capture
   * has to take the solver with it rather than leave it stepping.
   */
  readonly stop: () => void;
}

/** How much of a failed worker's output to quote; the reason is at the end. */
const OUTPUT_TAIL_LINES = 12;

const quoteTail = (tail: readonly string[]): string => (tail.length === 0 ? ""
  : `\n${tail.map((line) => `  | ${line.slice(0, 240)}`).join("\n")}`);

/** Spawn the worker, forwarding output to `logPath`. */
const startWorker = (
  argv: readonly string[],
  logPath: string,
  label: string,
  environment: Readonly<Record<string, string>> = profileEnvironment,
): WorkerHandle => {
  const log = createWriteStream(logPath);
  const child = spawn(argv[0], argv.slice(1), {
    cwd: root,
    env: { ...process.env, ...environment },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let result: SmokeResultRecord | undefined;
  let announceConstructed: () => void = () => {};
  let abandonConstruction: (reason: Error) => void = () => {};
  const constructed = new Promise<void>((done, fail) => {
    announceConstructed = done;
    abandonConstruction = fail;
  });
  let announceBeforeFirstAdvance: () => void = () => {};
  let abandonBeforeFirstAdvance: (reason: Error) => void = () => {};
  const beforeFirstAdvance = new Promise<void>((done, fail) => {
    announceBeforeFirstAdvance = done;
    abandonBeforeFirstAdvance = fail;
  });
  const timing: { steppingStartedAt?: number; steppingEndedAt?: number } = {};
  const tail: string[] = [];
  const absorb = (line: string): void => {
    log.write(`${line}\n`);
    tail.push(line);
    if (tail.length > OUTPUT_TAIL_LINES) tail.shift();
  };
  createInterface({ input: child.stdout }).on("line", (line) => {
    absorb(line);
    if (!line.startsWith("{")) return;
    try {
      const record = JSON.parse(line) as { phase?: string } & SmokeResultRecord;
      if (record.phase === "result") { result = record; timing.steppingEndedAt = Date.now(); }
      if (record.phase === "constructed") {
        console.log(`  ${label}: constructed in ${record.construction_ms} ms`);
        timing.steppingStartedAt = Date.now();
        announceConstructed();
      }
      if (record.phase === "before-first-advance"
        && ["waiting-for-sigusr2", "waiting-for-sigusr1"].includes(
          (record as { profileGate?: string }).profileGate ?? "")) {
        announceBeforeFirstAdvance();
      }
    } catch { /* progress lines are not all JSON */ }
  });
  createInterface({ input: child.stderr }).on("line", absorb);
  let running = true;
  const finished = new Promise<SmokeResultRecord | undefined>((done, fail) => {
    child.once("error", (error) => {
      running = false;
      abandonConstruction(error);
      abandonBeforeFirstAdvance(error);
      fail(error);
    });
    // `close` rather than `exit`: it fires once the worker's output has drained
    // too, which is the only point at which the quoted tail holds the reason
    // the run ended.
    child.once("close", (status, signal) => {
      running = false;
      log.end();
      const startupFailure = tail.some((line) => line.includes("Refusing concurrent GPU execution"))
        ? `${label} never started: another GPU run holds ${WEBGPU_EXCLUSIVE_LOCK}.`
          + " Dawn runs are serialised machine-wide, across checkouts as well as within one;"
          + ` wait for the holder to finish, then rerun. See ${logPath}`
        : undefined;
      const failure = signal !== null ? new Error(`${label} exited from ${signal}${quoteTail(tail)}`)
        : status !== 0
          ? new Error((startupFailure ?? `${label} exited with ${status}; see ${logPath}`)
            + quoteTail(tail))
          : undefined;
      // Keep this exact construction-rejection sequence stable: the lock
      // lifecycle test treats it as the proof a dead worker cannot strand an
      // attach waiter. The profiling gate is rejected alongside it.
      if (failure) { abandonConstruction(failure); fail(failure); }
      if (failure) abandonBeforeFirstAdvance(failure);
      else { announceConstructed(); announceBeforeFirstAdvance(); done(result); }
    });
  });
  // Both promises can reject long before anything awaits them -- a worker that
  // dies during startup rejects `finished` while the profiler is still waiting
  // on `constructed`, and that used to take the profiler down as an unhandled
  // rejection whose stack named this line rather than the cause. Marking them
  // handled here costs nothing: the rejection is still delivered to whoever
  // awaits them.
  constructed.catch(() => {});
  beforeFirstAdvance.catch(() => {});
  finished.catch(() => {});
  return { pid: child.pid ?? -1, constructed, beforeFirstAdvance, finished, timing, stop: () => {
    // SIGTERM, not SIGKILL: the worker's handler releases the exclusive lock,
    // whereas a killed worker leaves it behind as owner evidence for a human.
    if (running) child.kill("SIGTERM");
  } };
};

const runWorker = async (
  argv: readonly string[],
  logPath: string,
  label: string,
  environment?: Readonly<Record<string, string>>,
): Promise<SmokeResultRecord | undefined> =>
  startWorker(argv, logPath, label, environment).finished;

const delay = (ms: number): Promise<void> => new Promise((done) => { setTimeout(done, ms); });

// ---- Trace tables ----------------------------------------------------------
// Each entry becomes one exported XML table reduced to NDJSON.

const TABLES: readonly {
  schema: string; file: string; stacks?: boolean; countersOnly?: boolean;
  gpuReportOptional?: boolean; fullDiagnosticsOnly?: boolean; columns: readonly string[];
}[] = [
  {
    schema: "metal-gpu-intervals", file: "gpu-intervals",
    columns: ["start", "duration", "channel-name", "event-label", "process", "encoder-id"],
  },
  {
    schema: "metal-application-encoders-list", file: "encoders",
    columns: ["encoder-id", "encoder-label", "process"],
  },
  {
    schema: "metal-application-command-buffer-submissions", file: "command-buffer-submissions",
    columns: ["start", "duration", "num-encoders", "process", "cmdbuffer-id"],
  },
  {
    schema: "metal-command-buffer-completed", file: "command-buffer-completed",
    columns: ["timestamp", "cmdbuffer-id"],
  },
  {
    schema: "time-profile", file: "time-profile", stacks: true, gpuReportOptional: true,
    fullDiagnosticsOnly: true,
    columns: ["time", "process", "thread", "thread-state"],
  },
  {
    schema: "gpu-counter-info", file: "counter-info", countersOnly: true,
    columns: ["counter-id", "name", "type", "sample-interval"],
  },
  {
    schema: "gpu-counter-value", file: "counter-value", countersOnly: true,
    columns: ["timestamp", "counter-id", "value", "ring-buffer-index"],
  },
  {
    schema: "metal-shader-profiler-intervals", file: "shader-profile", countersOnly: true,
    gpuReportOptional: true, columns: ["start", "label", "pso-label"],
  },
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

/**
 * Refuse to start while another Dawn run owns the machine.
 *
 * Every GPU entrypoint in the repo serialises on one absolute-path lock, so the
 * holder can be any checkout -- a benchmark running out of a scratch copy
 * counts. A capture is minutes of work whose first second is what fails, and it
 * fails inside the worker, where the only signal reaching the profiler is a
 * non-zero exit. Asking up front turns that into one actionable line.
 */
const requireExclusiveGPU = async (): Promise<void> => {
  const holder = await readWebGPUExclusiveLockHolder();
  if (holder === undefined) return;
  throw new Error(holder.alive
    ? `another GPU run holds ${WEBGPU_EXCLUSIVE_LOCK}: ${holder.description}.`
      + " Dawn runs are serialised machine-wide, across checkouts as well as within one;"
      + " wait for it to finish or stop it, then rerun."
    : `${WEBGPU_EXCLUSIVE_LOCK} was left behind by ${holder.description}.`
      + " Confirm no Dawn or browser GPU run is active, then remove it:"
      + ` rm -rf ${WEBGPU_EXCLUSIVE_LOCK}`);
};

const main = async (): Promise<void> => {
  mkdirSync(outputDirectory, { recursive: true });
  const tracePath = `${outputDirectory}/mini-dam.trace`;
  const scratchBefore = instrumentsScratchSnapshot();
  const observedScratch = new Map(scratchBefore);
  const observeCaptureScratch = (): void => {
    for (const [path, entry] of instrumentsScratchSnapshot()) observedScratch.set(path, entry);
  };
  const writeTempInfo = async (status: string): Promise<void> => {
    observeCaptureScratch();
    const created = [...observedScratch.entries()]
      .filter(([path]) => !scratchBefore.has(path)).map(([, entry]) => entry);
    await writeFile(`${outputDirectory}/temp-info.json`, `${JSON.stringify({
      status,
      capturedAt: new Date().toISOString(),
      tracePath,
      scratchRoots,
      created,
      cleanup: {
        note: "Delete only the exact created paths after confirming no xctrace process is active.",
        activeProcessCheck: "pgrep -lf xctrace",
      },
    }, null, 2)}\n`);
  };
  await writeTempInfo("prepared");
  // Never leave a stale successful-looking report behind when a replacement
  // capture fails the mandatory labelling/occupancy checks below.
  rmSync(`${outputDirectory}/summary.json`, { force: true });
  rmSync(`${outputDirectory}/report.html`, { force: true });
  const nodeBinary = process.execPath;

  console.log(`lane ${lane}: ${profileEnvironment.FLUID_ORACLE_STEPS} advances`
    + ` of ${profileEnvironment.FLUID_SCENE} at grid ${profileEnvironment.FLUID_EXPECT_GRID}`
    + `, interface band ${profileEnvironment.FLUID_OCTREE_INTERFACE_BAND ?? "scene default"}`);
  console.log(fullDiagnostics
    ? "  capture instruments: full Metal System Trace + GPU counters"
    : "  capture instruments: Metal Application + GPU + GPU counters (stage diagnostics only)");
  console.log(`  counter extraction: target ${counterReduction.toFixed(0)}x fewer rows;`
    + " exact Metal stage intervals remain full resolution");
  if (isolatePassLabels) {
    console.log(
      isolateLabelPrefix
        ? `  micro-stage attribution: only "${isolateLabelPrefix}" transitions are Metal-encoder-isolated`
          + "\n  EVERY OTHER ROW IN THE REPORT IS A COMPOSITE BUCKET: Metal names an encoder once,"
          + "\n  so an unscoped label carries its own dispatches AND every stage encoded after it"
          + "\n  until the next pass boundary. Widen --isolate-label-prefix to price one of those."
        : "  micro-stage attribution: label- and Metal-encoder-isolated passes"
          + " (utilization experiment; not shipping wall clock)",
    );
  }
  if (sizingPlan?.extended) {
    console.log(`  run extended from ${requestedSteps} to ${sizingPlan.steps} advances:`
      + ` a ${counterSeconds} s counter window plus Instruments' ${ATTACH_LATENCY_S} s attach`
      + " does not fit inside a shorter stepping phase");
  }

  if (reuseTables) {
    console.log("rebuilding report from retained trace tables...");
    const tables: Record<string, string> = {};
    for (const table of TABLES) {
      if (table.countersOnly && !counters) continue;
      if (table.gpuReportOptional && gpuReportOnly) continue;
      if (table.fullDiagnosticsOnly && !fullDiagnostics) continue;
      const path = `${outputDirectory}/${table.file}.ndjson`;
      if (!existsSync(path)) throw new Error(`--reuse-tables requires ${path}`);
      tables[table.file] = path;
    }
    const resultFromLog = (name: string): SmokeResultRecord | undefined => {
      const path = `${outputDirectory}/${name}.log`;
      if (!existsSync(path)) return undefined;
      let result: SmokeResultRecord | undefined;
      for (const line of readFileSync(path, "utf8").split("\n")) {
        if (!line.startsWith("{")) continue;
        try {
          const record = JSON.parse(line) as { phase?: string } & SmokeResultRecord;
          if (record.phase === "result") result = record;
        } catch { /* diagnostics include non-JSON progress text */ }
      }
      return result;
    };
    const extractionPath = `${outputDirectory}/counter-extraction.json`;
    const extraction = JSON.parse(readFileSync(extractionPath, "utf8")) as {
      sourceCounterCount: number; retainedCounterCount: number; timestampStride: number;
    };
    const report = await buildFrameReport({
      tables,
      lane,
      environment: profileEnvironment,
      traced: resultFromLog("traced"),
      baseline: resultFromLog("baseline"),
      singleFrame: !firstFrame,
      firstFrame,
      counterExtraction: extraction,
    });
    assertCompleteOccupancyReport(report);
    await writeFile(`${outputDirectory}/summary.json`, `${JSON.stringify(report, null, 2)}\n`);
    await writeFile(`${outputDirectory}/report.html`, renderFrameReportHtml(report));
    if (discardTables) {
      for (const path of Object.values(tables)) rmSync(path, { force: true });
      console.log("discarded derived trace tables; the report and source trace are retained");
    }
    for (const line of report.console) console.log(line);
    console.log(`report:  ${outputDirectory}/report.html`);
    console.log(`summary: ${outputDirectory}/summary.json`);
    await writeTempInfo("complete");
    return;
  }

  await requireExclusiveGPU();

  let baseline: SmokeResultRecord | undefined;
  if (runBaseline) {
    console.log("recording clean non-isolated baseline (shipping wall-clock control)...");
    baseline = await runWorker(
      [nodeBinary, "--import", "tsx", worker],
      `${outputDirectory}/baseline.log`,
      "baseline",
      cleanBaselineEnvironment,
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
    console.log(profileGate
      ? `launching solver behind a pre-advance gate; the ${counterSeconds} s counter`
        + ` window will warm for ${(counterGateWarmupMs / 1_000).toFixed(1)} s before advance 1 is released${firstFrame
          ? " and selected" : "; a later recurring advance will be selected"}...`
      : `launching solver, attaching GPU counters at`
      + ` ${plan.spawnDelaySeconds.toFixed(1)} s so recording starts`
      + ` ${plan.windowStartSeconds.toFixed(1)} s into stepping`
      + ` (Instruments takes ~${ATTACH_LATENCY_S} s to attach) for a ${counterSeconds} s window,`
      + ` of which ~${plan.labelledSeconds.toFixed(1)} s carries encoder labels...`);
    const handle = startWorker(
      [nodeBinary, "--import", "tsx", worker], `${outputDirectory}/traced.log`, "traced",
    );
    tracedPid = handle.pid;
    let recordingStartedAt: number | undefined;
    try {
      if (profileGate) await handle.beforeFirstAdvance;
      else {
        await handle.constructed;
        await delay(plan.spawnDelaySeconds * 1000);
      }
      let releaseGate: Promise<void> | undefined;
      try {
        const blankTemplate = resolve(execFileSync("xcode-select", ["-p"]).toString().trim(),
          "../Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate");
        if (!fullDiagnostics && !existsSync(blankTemplate)) {
          throw new Error(`Instruments Blank template not found at ${blankTemplate}`);
        }
        const instruments = fullDiagnostics
          ? ["--template", "Metal System Trace", "--instrument", "Metal GPU Counters"]
          : ["--template", blankTemplate,
            "--instrument", "Metal Application",
            "--instrument", "GPU",
            "--instrument", "Metal GPU Counters"];
        await run(["xcrun", "xctrace", "record",
          ...instruments,
          "--output", tracePath, "--no-prompt",
          "--time-limit", counterTimeLimit,
          "--attach", String(handle.pid)], (line) => {
          if (recordingStartedAt === undefined && /Ctrl-C to stop the recording/.test(line)) {
            recordingStartedAt = Date.now();
            observeCaptureScratch();
            if (profileGate) {
              // Make Instruments observe one disposable compute pass before
              // the first real advance. Without this, attached recordings can
              // retain the GPU intervals but omit every Metal encoder label
              // from the first Losasso command buffer.
              process.kill(handle.pid, "SIGUSR2");
              releaseGate = delay(counterGateWarmupMs).then(() => {
                process.kill(handle.pid, "SIGUSR1");
              });
            }
          }
        });
      } finally {
        await writeTempInfo("recording-finished");
      }
      await releaseGate;
      console.log("  counter window captured; waiting for the run to finish...");
      traced = await handle.finished;
    } catch (error) {
      // The solver outlives us otherwise, holding the machine-wide GPU lock,
      // and the next run then fails for a reason that has nothing to do with it.
      handle.stop();
      throw error;
    }

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
    clearEmptyProfileVariables();
    traced = await runWorker([
      "xcrun", "xctrace", "record",
      "--template", "Metal System Trace",
      "--output", tracePath,
      "--no-prompt",
      "--target-stdout", "-",
      // xctrace rejects `--env VAR=` outright ("cannot be parsed with provided
      // value"), so an empty value cannot be forwarded as one. Omitting the
      // variable is equivalent for every consumer -- lib/webgpu-pass-broker.ts
      // and tools/xctrace-frame-report.ts both read it as `?? ""` -- and the
      // deletion in `clearEmptyProfileVariables` stops an exported value in
      // this shell from inheriting into the launched worker under that name.
      ...Object.entries(profileEnvironment)
        .filter(([, value]) => value !== "")
        .flatMap(([key, value]) => ["--env", `${key}=${value}`]),
      "--launch", "--", nodeBinary, "--import", "tsx", worker,
    ], `${outputDirectory}/traced.log`, "traced");
  }
  // A .trace is a bundle directory, so stat() reports the inode, not the run.
  const traceMegabytes = Number(execFileSync("du", ["-sm", tracePath])
    .toString().split("\t")[0]);
  console.log(`  trace: ${traceMegabytes} MB at ${tracePath}`);

  console.log("exporting trace tables...");
  const tables: Record<string, string> = {};
  let counterPolicy: CounterExtractionPolicy | undefined;
  for (const table of TABLES) {
    if (table.countersOnly && !counters) continue;
    if (table.gpuReportOptional && gpuReportOnly) continue;
    if (table.fullDiagnosticsOnly && !fullDiagnostics) continue;
    const xml = `${outputDirectory}/${table.file}.xml`;
    const ndjson = `${outputDirectory}/${table.file}.ndjson`;
    const exportArguments = ["xctrace", "export", "--input", tracePath,
      "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${table.schema}"]`];
    let exportProcess: ReturnType<typeof spawn> | undefined;
    let exportCompleted: Promise<void> | undefined;
    let source: string | AsyncIterable<string> = xml;
    if (keepXml) {
      await run(["xcrun", ...exportArguments, "--output", xml]);
    } else {
      // xctrace writes XML to stdout when --output is omitted. Parse that
      // stream directly: counter tables approach a gigabyte even for a short
      // window, and the report never needs a materialised XML copy.
      exportProcess = spawn("xcrun", exportArguments,
        { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
      const exportStdout = exportProcess.stdout;
      const exportStderr = exportProcess.stderr;
      if (!exportStdout || !exportStderr) throw new Error("xctrace export pipes unavailable");
      exportStdout.setEncoding("utf8");
      let stderr = "";
      exportStderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
      exportCompleted = new Promise((done, fail) => {
        exportProcess?.once("exit", (code) => (code === 0 ? done()
          : fail(new Error(`xctrace export ${table.schema} failed (${code}):`
            + ` ${stderr.slice(0, 400)}`))));
        exportProcess?.once("error", fail);
      });
      source = exportStdout as AsyncIterable<string>;
    }
    const batch: string[] = [];
    const sink = createWriteStream(ndjson);
    let rows = 0;
    let sourceRows = 0;
    const counterSelector = table.file === "counter-value" && counterPolicy
      ? new CounterRowSelector(counterPolicy) : undefined;
    try {
      for await (const row of parseTraceTable(source, {
        stacks: table.stacks,
        columns: table.columns,
      })) {
        sourceRows += 1;
        if (counterSelector && !counterSelector.keep(row)) continue;
        batch.push(JSON.stringify(row));
        rows += 1;
        if (batch.length >= 4096) { sink.write(`${batch.join("\n")}\n`); batch.length = 0; }
      }
      if (exportCompleted) await exportCompleted;
    } finally {
      if (exportProcess?.exitCode === null) exportProcess.kill("SIGTERM");
    }
    if (batch.length > 0) sink.write(`${batch.join("\n")}\n`);
    await new Promise((done) => sink.end(done));
    tables[table.file] = ndjson;
    console.log(`  ${table.schema}: ${rows} rows`
      + (sourceRows === rows ? "" : ` retained from ${sourceRows}`));
    if (table.file === "counter-info") {
      const countersById = new Map<string, string>();
      const sampleIntervals = new Set<number>();
      for await (const row of readTraceRows(ndjson)) {
        countersById.set(String(row["counter-id"]), String(row.name ?? ""));
        const interval = Number(String(row["sample-interval"] ?? "0").replace(/,/g, ""));
        if (interval > 0) sampleIntervals.add(interval);
      }
      counterPolicy = makeCounterExtractionPolicy(countersById, counterReduction);
      const hardwareSampleIntervalNs = sampleIntervals.size === 1
        ? [...sampleIntervals][0] : undefined;
      await writeFile(`${outputDirectory}/counter-extraction.json`, `${JSON.stringify({
        targetReduction: counterPolicy.targetReduction,
        sourceCounterCount: counterPolicy.sourceCounterCount,
        retainedCounterCount: counterPolicy.retainedCounterCount,
        retainedCounters: [...countersById]
          .filter(([id]) => counterPolicy?.retainedCounterIds.has(id)).map(([, name]) => name),
        timestampStride: counterPolicy.timestampStride,
        hardwareSampleIntervalNs,
        retainedSampleIntervalNs: hardwareSampleIntervalNs === undefined ? undefined
          : hardwareSampleIntervalNs * counterPolicy.timestampStride,
        note: "Metal intervals, encoder labels, command-buffer boundaries, and shader samples are not downsampled.",
      }, null, 2)}\n`);
      console.log(`    retaining ${counterPolicy.retainedCounterCount}/${counterPolicy.sourceCounterCount}`
        + ` counters at every ${counterPolicy.timestampStride}th hardware timestamp`);
    }
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
    singleFrame: !firstFrame,
    firstFrame,
    counterExtraction: counterPolicy === undefined ? undefined : {
      sourceCounterCount: counterPolicy.sourceCounterCount,
      retainedCounterCount: counterPolicy.retainedCounterCount,
      timestampStride: counterPolicy.timestampStride,
    },
  });
  assertCompleteOccupancyReport(report);
  await writeFile(`${outputDirectory}/summary.json`, `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(`${outputDirectory}/report.html`, renderFrameReportHtml(report));
  if (discardTables) {
    for (const path of Object.values(tables)) rmSync(path, { force: true });
    console.log(discardTrace
      ? "discarded derived trace tables; the self-contained report is retained"
      : "discarded derived trace tables; the report and source trace are retained");
  }
  if (discardTrace) {
    rmSync(tracePath, { recursive: true, force: true });
    console.log("discarded source trace after the self-contained report was built");
  }

  console.log("");
  for (const line of report.console) console.log(line);
  console.log("");
  console.log(`report:  ${outputDirectory}/report.html`);
  console.log(`summary: ${outputDirectory}/summary.json`);
  if (!discardTrace) console.log(`trace:   ${tracePath}   (open with: open ${tracePath})`);
  await writeTempInfo("complete");
};

// Only capture when run as a program. Importing this module -- a test reaching
// for `planCounterWindow`, say -- must not start a multi-minute GPU recording.
const isMain = process.argv[1] !== undefined
  && import.meta.url === new URL(`file://${process.argv[1]}`).href;

if (isMain) {
  // The interesting failures here -- a held GPU lock, a window that missed the
  // stepping phase -- are diagnoses, not defects in this file. Print the
  // diagnosis; a stack trace through the spawn plumbing only buries it.
  await main().catch((error: unknown) => {
    console.error(`\nprofile aborted: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  });
}
