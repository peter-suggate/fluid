#!/usr/bin/env node
/**
 * Capture ALU, occupancy, bandwidth, and Metal stage timing for production SVO
 * rendering. The worker submits render-only hose-scene frames; xctrace attaches
 * from outside the process, then the established frame report reducer keeps
 * every complete frame and a representative per-frame timeline.
 *
 * Usage:
 *   node --import tsx tools/profile-svo-render-xctrace.ts
 *     [--scene=hose-tank] [--resolution=660x662]
 *     [--variant=baseline] [--traversal=hybrid|canonical|canonical-parametric|compact|wide|raster-primary]
 *     [--shading=inline|split]
 *     [--cone-scale=0.5|0.25|0.125] [--cone-tracing=cones|exact|off] [--warmups=4]
 *     [--radiance-reconstruction=nearest|gated-linear|joint-bilateral|wide-relight|full-res-relight]
 *     [--counter-seconds=3] [--counter-reduction=100] [--out=DIR]
 *     [--timing-only] [--reuse-trace] [--reuse-tables]
 */
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";
import { createInterface } from "node:readline";

import {
  CounterRowSelector,
  makeCounterExtractionPolicy,
  type CounterExtractionPolicy,
} from "./profile-mini-dam-xctrace";
import {
  acquireWebGPUExclusiveLock,
  readWebGPUExclusiveLockHolder,
  releaseWebGPUExclusiveLock,
  releaseWebGPUExclusiveLockSync,
  WEBGPU_EXCLUSIVE_LOCK,
} from "./webgpu-smoke-isolation";
import { SVO_DRY_TRAVERSAL_MODES, type SvoDryTraversalMode } from "../lib/webgpu-svo-dry-scene";
import { buildFrameReport, renderFrameReportHtml, type FrameReport } from "./xctrace-frame-report";
import { parseTraceTable, readTraceRows } from "./xctrace-trace-tables";

const root = fileURLToPath(new URL("..", import.meta.url));
const worker = fileURLToPath(new URL("./benchmark-svo-dry-frame-gpu.ts", import.meta.url));
const flag = (name: string): string | undefined => process.argv
  .find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);

const scene = flag("scene") ?? "hose-tank";
const variant = flag("variant") ?? "baseline";
if (!/^[a-zA-Z0-9._-]+$/.test(variant)) {
  throw new Error("--variant may contain only letters, digits, dot, underscore, and dash");
}
const traversal = flag("traversal") ?? "hybrid";
if (!SVO_DRY_TRAVERSAL_MODES.includes(traversal as SvoDryTraversalMode)) {
  throw new Error(`--traversal must be one of ${SVO_DRY_TRAVERSAL_MODES.join(", ")}`);
}
const shading = flag("shading") ?? "inline";
if (!["inline", "split"].includes(shading)) throw new Error("--shading must be inline or split");
const resolutionMatch = /^(\d+)x(\d+)$/.exec(flag("resolution") ?? "660x662");
if (!resolutionMatch) throw new Error("--resolution must be WIDTHxHEIGHT");
const width = Number(resolutionMatch[1]), height = Number(resolutionMatch[2]);
if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) {
  throw new Error("--resolution dimensions must be positive integers");
}
const counterSeconds = Number(flag("counter-seconds") ?? 3);
if (!Number.isFinite(counterSeconds) || counterSeconds < 3) {
  throw new Error("--counter-seconds must be at least 3 for Metal labels and GPU counters");
}
const counterReduction = Number(flag("counter-reduction") ?? 100);
if (!Number.isFinite(counterReduction) || counterReduction < 1) {
  throw new Error("--counter-reduction must be at least 1");
}
const coneTracing = flag("cone-tracing") ?? "cones";
if (!["cones", "exact", "off"].includes(coneTracing)) {
  throw new Error("--cone-tracing must be cones, exact, or off");
}
const coneScale = Number(flag("cone-scale") ?? (coneTracing === "cones" ? 0.5 : 1));
if (coneTracing === "cones" ? ![0.5, 0.25, 0.125].includes(coneScale) : coneScale !== 1) {
  throw new Error(coneTracing === "cones"
    ? "--cone-scale must be 0.5, 0.25, or 0.125 so the profiled pass graph includes the cone prepass"
    : "--cone-scale must be 1 when --cone-tracing withholds the cone stages");
}
const radianceReconstruction = flag("radiance-reconstruction") ?? "full-res-relight";
if (!["nearest", "gated-linear", "joint-bilateral", "wide-relight", "full-res-relight"].includes(radianceReconstruction)) {
  throw new Error("--radiance-reconstruction must be nearest, gated-linear, joint-bilateral, wide-relight, or full-res-relight");
}
const warmups = Number(flag("warmups") ?? 4);
if (!Number.isSafeInteger(warmups) || warmups < 1) {
  throw new Error("--warmups must be a positive integer");
}
const keepTables = process.argv.includes("--keep-tables");
const reuseTrace = process.argv.includes("--reuse-trace");
const reuseTables = process.argv.includes("--reuse-tables");
const timingOnly = process.argv.includes("--timing-only");
const outputDirectory = resolve(root, flag("out") ?? "artifacts/xctrace-hose-render-utilization");
const tracePath = resolve(outputDirectory, "svo-render.trace");
const counterTimeLimit = Number.isInteger(counterSeconds)
  ? `${counterSeconds}s` : `${Math.round(counterSeconds * 1000)}ms`;

const TABLES: readonly {
  schema: string; file: string; columns: readonly string[];
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
    schema: "gpu-counter-info", file: "counter-info",
    columns: ["counter-id", "name", "type", "sample-interval"],
  },
  {
    schema: "gpu-counter-value", file: "counter-value",
    columns: ["timestamp", "counter-id", "value", "ring-buffer-index"],
  },
];

const RENDER_GPU_COUNTERS = new Set([
  "Fragment Occupancy",
  "ALU Utilization",
  "GPU Read Bandwidth",
  "GPU Write Bandwidth",
  "GPU Last Level Cache Utilization",
]);

interface RenderWorkerResult {
  readonly phase: "result";
  readonly frames: number;
  readonly renderWall_ms: number;
  readonly medianFrame_ms: number;
  readonly p95Frame_ms: number;
}

interface SourceProvenance {
  readonly commit: string;
  readonly dirty: boolean;
  readonly changedFiles: number;
  readonly fingerprint: string;
  readonly renderFingerprint: string;
  readonly renderFiles: number;
}

/** A profile is only useful for A/B work when its exact source state is
 * recoverable. The fingerprint deliberately includes tracked working-tree
 * changes and the untracked-file manifest without copying the source into the
 * artifact. */
const sourceProvenance = (): SourceProvenance => {
  const git = (...arguments_: string[]): string => execFileSync("git", arguments_, {
    cwd: root, encoding: "utf8", maxBuffer: 128 * 1024 * 1024,
  });
  const commit = git("rev-parse", "HEAD").trim();
  const status = git("status", "--short", "--untracked-files=all");
  const trackedDiff = git("diff", "--no-ext-diff", "--binary", "HEAD");
  const renderPaths = git("ls-files", "-co", "--exclude-standard")
    .split("\n").filter((path) => /^(?:lib\/(?:webgpu-svo|webgpu-static-svo|svo-|scenes\.ts|paper-scenarios\.ts|environments\.ts|voxel-scenery\/)|tools\/benchmark-svo-dry-frame-gpu\.ts)/.test(path))
    .sort();
  const renderHash = createHash("sha256");
  for (const path of renderPaths) {
    renderHash.update(path).update("\0").update(readFileSync(resolve(root, path))).update("\0");
  }
  return {
    commit,
    dirty: status.trim().length > 0,
    changedFiles: status.split("\n").filter(Boolean).length,
    fingerprint: createHash("sha256")
      .update(commit).update("\n").update(status).update("\n").update(trackedDiff).digest("hex"),
    renderFingerprint: renderHash.digest("hex"),
    renderFiles: renderPaths.length,
  };
};

const run = (argv: readonly string[]): Promise<void> => new Promise((done, fail) => {
  const child = spawn(argv[0], argv.slice(1), { cwd: root, stdio: ["ignore", "ignore", "pipe"] });
  let stderr = "";
  child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
  child.once("error", fail);
  child.once("exit", (code) => code === 0 ? done()
    : fail(new Error(`${argv.join(" ")} failed (${code}): ${stderr.slice(-800)}`)));
});

const exportTables = async (): Promise<{
  tables: Record<string, string>;
  policy: CounterExtractionPolicy;
}> => {
  const tables: Record<string, string> = {};
  let policy: CounterExtractionPolicy | undefined;
  for (const table of TABLES.filter((candidate) => !timingOnly || !candidate.file.startsWith("counter-"))) {
    const ndjson = resolve(outputDirectory, `${table.file}.ndjson`);
    const exportArguments = ["xcrun", "xctrace", "export", "--input", tracePath,
      "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${table.schema}"]`];
    const child = spawn(exportArguments[0], exportArguments.slice(1),
      { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
    if (!child.stdout || !child.stderr) throw new Error(`could not export ${table.schema}`);
    child.stdout.setEncoding("utf8");
    let stderr = "";
    child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
    const completed = new Promise<void>((done, fail) => {
      child.once("error", fail);
      child.once("exit", (code) => code === 0 ? done()
        : fail(new Error(`xctrace export ${table.schema} failed (${code}): ${stderr.slice(-800)}`)));
    });
    const sink = createWriteStream(ndjson);
    const selector = table.file === "counter-value" && policy
      ? new CounterRowSelector(policy) : undefined;
    const batch: string[] = [];
    let sourceRows = 0, rows = 0;
    try {
      for await (const row of parseTraceTable(child.stdout as AsyncIterable<string>, {
        columns: table.columns,
      })) {
        sourceRows += 1;
        if (selector && !selector.keep(row)) continue;
        batch.push(JSON.stringify(row)); rows += 1;
        if (batch.length >= 4096) { sink.write(`${batch.join("\n")}\n`); batch.length = 0; }
      }
      await completed;
    } finally {
      if (child.exitCode === null) child.kill("SIGTERM");
    }
    if (batch.length > 0) sink.write(`${batch.join("\n")}\n`);
    await new Promise<void>((done) => sink.end(done));
    tables[table.file] = ndjson;
    console.log(`  ${table.schema}: ${rows} rows${rows === sourceRows ? "" : ` retained from ${sourceRows}`}`);
    if (table.file === "counter-info") {
      const countersById = new Map<string, string>();
      for await (const row of readTraceRows(ndjson)) {
        countersById.set(String(row["counter-id"]), String(row.name ?? ""));
      }
      policy = makeCounterExtractionPolicy(countersById, counterReduction, RENDER_GPU_COUNTERS);
      await writeFile(resolve(outputDirectory, "counter-extraction.json"), `${JSON.stringify({
        targetReduction: policy.targetReduction,
        sourceCounterCount: policy.sourceCounterCount,
        retainedCounterCount: policy.retainedCounterCount,
        retainedCounters: [...countersById]
          .filter(([id]) => policy?.retainedCounterIds.has(id)).map(([, name]) => name),
        timestampStride: policy.timestampStride,
      }, null, 2)}\n`);
    }
  }
  if (!policy) {
    if (!timingOnly) throw new Error("GPU counter metadata was not exported");
    policy = { targetReduction: 1, sourceCounterCount: 0, retainedCounterCount: 0,
      timestampStride: 1, retainedCounterIds: new Set<string>() };
  }
  return { tables, policy };
};

const assertRenderReport = (report: FrameReport): void => {
  const failures: string[] = [];
  if (report.frames.count < 1 || report.frames.captures.length < 1) {
    failures.push("no complete render frames were reduced");
  }
  if (!timingOnly && (report.counters.meanOccupancy === undefined || report.counters.meanAlu === undefined
    || report.counters.exclusiveCoverage <= 0)) {
    failures.push("occupancy/ALU counters did not cover the render window");
  }
  // Withholding the cone stages collapses the effective scale to 1: split
  // shading keeps its two full-rate passes and the inline path keeps its single
  // pass, but no cone prepass or compact cone visibility is encoded at all.
  const split = shading === "split";
  const splitRelight = split
    && (radianceReconstruction === "wide-relight" || radianceReconstruction === "full-res-relight");
  const expectedLabels = split
    ? ["Sparse voxel primary visibility", "Sparse voxel deferred dry lighting"]
    : coneTracing === "cones"
      ? ["Sparse voxel cone-lighting prepass", "Sparse voxel dry scene"]
      : ["Sparse voxel dry scene"];
  for (const label of expectedLabels) {
    const pass = report.passes.find((candidate) => candidate.label === label);
    if (!pass || (!timingOnly && (pass.counterSamples === 0 || pass.occupancy === undefined || pass.alu === undefined))) {
      failures.push(`${label} has no attributed occupancy/ALU samples`);
    }
  }
  const compactConeLabel = "Sparse voxel compact cone visibility";
  const compactCone = report.passes.some((candidate) => candidate.label === compactConeLabel);
  if (splitRelight && coneTracing === "cones" && !compactCone) {
    failures.push(`${compactConeLabel} has no attributed GPU interval`);
  }
  if (coneTracing !== "cones" && compactCone) {
    failures.push(`${compactConeLabel} ran with --cone-tracing=${coneTracing}`);
  }
  if (failures.length > 0) throw new Error(`render xctrace report rejected: ${failures.join("; ")}`);
};

const workerResultFromLog = (): RenderWorkerResult => {
  const logPath = resolve(outputDirectory, "worker.log");
  if (!existsSync(logPath)) throw new Error(`missing render worker log ${logPath}`);
  let result: RenderWorkerResult | undefined;
  for (const line of readFileSync(logPath, "utf8").split("\n")) {
    if (!line.startsWith("{")) continue;
    try {
      const record = JSON.parse(line) as { phase?: string } & RenderWorkerResult;
      if (record.phase === "result") result = record;
    } catch { /* progress output */ }
  }
  if (!result) throw new Error(`render worker produced no result; see ${logPath}`);
  return result;
};

const tracedPidFromEncoders = async (path: string): Promise<number> => {
  const tally = new Map<number, number>();
  for await (const row of readTraceRows(path)) {
    if (!String(row["encoder-label"] ?? "").includes("Sparse voxel")) continue;
    const match = /\((\d+)\)$/.exec(String(row.process ?? ""));
    if (!match) continue;
    const pid = Number(match[1]);
    tally.set(pid, (tally.get(pid) ?? 0) + 1);
  }
  const pid = [...tally].sort((left, right) => right[1] - left[1])[0]?.[0];
  if (pid === undefined) throw new Error("could not recover the render worker pid from encoder labels");
  return pid;
};

const writeReport = async (
  tables: Record<string, string>,
  policy: Pick<CounterExtractionPolicy,
    "sourceCounterCount" | "retainedCounterCount" | "timestampStride">,
  result: RenderWorkerResult, tracedPid: number,
): Promise<FrameReport> => {
  const report = await buildFrameReport({
    tables,
    lane: "svo-render",
    workUnit: "frame",
    title: `Hose tank SVO rendering — ${variant} GPU frame profile`,
    frameStartLabels: ["Sparse voxel primary visibility", "Sparse voxel cone-lighting prepass",
      "Sparse voxel compact cone visibility", "Sparse voxel dry scene"],
    occupancyCounterName: "Fragment Occupancy",
    environment: {
      FLUID_SCENE: scene,
      FLUID_EXPECT_GRID: `${width}x${height}`,
      FLUID_GPU_ISOLATE_PASS_LABELS: "1",
      FLUID_GPU_ISOLATE_PASS_LABEL_PREFIXES: "Sparse voxel",
      FLUID_GPU_ISOLATE_PASS_ENCODERS: timingOnly ? "0" : "1",
      FLUID_SVO_DRY_FRAME_TRAVERSAL: traversal,
      FLUID_SVO_DRY_FRAME_SHADING: shading,
      FLUID_SVO_DRY_FRAME_CONE_SCALE: String(coneScale),
      FLUID_SVO_DRY_FRAME_CONE_TRACING: coneTracing,
      FLUID_SVO_DRY_FRAME_RADIANCE_RECONSTRUCTION: radianceReconstruction,
    },
    tracedPid,
    traced: { simulationWall_ms: result.renderWall_ms, steps: result.frames },
    counterExtraction: {
      sourceCounterCount: policy.sourceCounterCount,
      retainedCounterCount: policy.retainedCounterCount,
      timestampStride: policy.timestampStride,
    },
  });
  assertRenderReport(report);
  await writeFile(resolve(outputDirectory, "summary.json"), `${JSON.stringify(report, null, 2)}\n`);
  await writeFile(resolve(outputDirectory, "report.html"), renderFrameReportHtml(report));
  return report;
};

const main = async (): Promise<void> => {
  mkdirSync(outputDirectory, { recursive: true });
  const source = sourceProvenance();
  const capturePath = resolve(outputDirectory, "capture.json");
  const priorCapture = existsSync(capturePath)
    ? JSON.parse(readFileSync(capturePath, "utf8")) as Record<string, unknown>
    : undefined;
  if (reuseTables) {
    console.log("rebuilding report from retained render tables...");
    const tables = Object.fromEntries(TABLES.filter((candidate) => !timingOnly || !candidate.file.startsWith("counter-")).map((table) => {
      const path = resolve(outputDirectory, `${table.file}.ndjson`);
      if (!existsSync(path)) throw new Error(`--reuse-tables requires ${path}`);
      return [table.file, path];
    }));
    const extraction = timingOnly
      ? { sourceCounterCount: 0, retainedCounterCount: 0, timestampStride: 1 }
      : JSON.parse(readFileSync(resolve(outputDirectory, "counter-extraction.json"), "utf8")) as Pick<
        CounterExtractionPolicy, "sourceCounterCount" | "retainedCounterCount" | "timestampStride">;
    const result = workerResultFromLog();
    const tracedPid = await tracedPidFromEncoders(tables.encoders);
    const report = await writeReport(tables, extraction, result, tracedPid);
    await writeFile(capturePath, `${JSON.stringify({
      variant, traversal, shading, coneScale, coneTracing, radianceReconstruction, warmups, scene, resolution: { width, height }, counterSeconds, counterReduction,
      ...priorCapture,
      source: priorCapture?.source ?? source,
      worker: result, tracePath, rebuiltAt: new Date().toISOString(),
    }, null, 2)}\n`);
    for (const line of report.console) console.log(line);
    console.log(`\nreport:  ${resolve(outputDirectory, "report.html")}`);
    console.log(`summary: ${resolve(outputDirectory, "summary.json")}`);
    console.log(`trace:   ${tracePath}`);
    return;
  }
  if (reuseTrace) {
    if (!existsSync(tracePath)) throw new Error(`--reuse-trace requires ${tracePath}`);
    console.log("re-exporting and rebuilding report from retained render trace...");
    const result = workerResultFromLog();
    const { tables, policy } = await exportTables();
    const tracedPid = await tracedPidFromEncoders(tables.encoders);
    const report = await writeReport(tables, policy, result, tracedPid);
    await writeFile(capturePath, `${JSON.stringify({
      variant, traversal, shading, coneScale, coneTracing, radianceReconstruction, warmups, scene, resolution: { width, height }, counterSeconds, counterReduction,
      ...priorCapture,
      source: priorCapture?.source ?? source,
      worker: result, tracePath, rebuiltAt: new Date().toISOString(),
    }, null, 2)}\n`);
    if (!keepTables) for (const path of Object.values(tables)) rmSync(path, { force: true });
    for (const line of report.console) console.log(line);
    console.log(`\nreport:  ${resolve(outputDirectory, "report.html")}`);
    console.log(`summary: ${resolve(outputDirectory, "summary.json")}`);
    console.log(`trace:   ${tracePath}`);
    return;
  }
  rmSync(tracePath, { recursive: true, force: true });
  for (const name of ["summary.json", "report.html"]) {
    rmSync(resolve(outputDirectory, name), { force: true });
  }
  // Persist identity before launching anything. If xctrace export is later
  // interrupted, `--reuse-trace` must retain the source fingerprint from the
  // capture rather than accidentally claiming the source state at rebuild time.
  await writeFile(capturePath, `${JSON.stringify({
    state: "capturing", variant, traversal, shading, coneScale, coneTracing, radianceReconstruction, warmups, scene,
    resolution: { width, height }, counterSeconds, counterReduction, source,
    tracePath, startedAt: new Date().toISOString(),
  }, null, 2)}\n`);
  const holder = await readWebGPUExclusiveLockHolder();
  if (holder) {
    throw new Error(`${WEBGPU_EXCLUSIVE_LOCK} is held by ${holder.description}`);
  }
  await acquireWebGPUExclusiveLock("svo-render-xctrace", `${scene} ${width}x${height}`);
  let lockHeld = true;
  let child: ReturnType<typeof spawn> | undefined;
  const stop = (): void => { if (child && child.exitCode === null) child.kill("SIGTERM"); };
  const onSignal = (): void => {
    stop();
    if (lockHeld) { releaseWebGPUExclusiveLockSync(); lockHeld = false; }
  };
  process.once("SIGINT", onSignal); process.once("SIGTERM", onSignal);
  try {
    const logPath = resolve(outputDirectory, "worker.log");
    const log = createWriteStream(logPath);
    child = spawn(process.execPath, ["--import", "tsx", worker], {
      cwd: root,
      env: {
        ...process.env,
        WEBGPU_NODE_MODULE: resolve(root, "node_modules/webgpu/index.js"),
        FLUID_WEBGPU_DAWN_FEATURES: "use_user_defined_labels_in_backend",
        FLUID_SVO_DRY_FRAME_SCENE: scene,
        FLUID_SVO_DRY_FRAME_WIDTH: String(width),
        FLUID_SVO_DRY_FRAME_HEIGHT: String(height),
        FLUID_SVO_DRY_FRAME_WARMUPS: String(warmups),
        FLUID_SVO_DRY_FRAME_CONE_SCALE: String(coneScale),
        FLUID_SVO_DRY_FRAME_CONE_TRACING: coneTracing,
        FLUID_SVO_DRY_FRAME_RADIANCE_RECONSTRUCTION: radianceReconstruction,
        FLUID_SVO_DRY_FRAME_TRAVERSAL: traversal,
        FLUID_SVO_DRY_FRAME_SHADING: shading,
        FLUID_SVO_DRY_FRAME_PROFILE_SECONDS: String(counterSeconds + 9),
        FLUID_SVO_DRY_FRAME_ISOLATE_PASS_ENCODERS: timingOnly ? "0" : "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    if (!child.stdout || !child.stderr || child.pid === undefined) throw new Error("render worker did not start");
    let constructedDone: () => void = () => {};
    let constructedFail: (error: Error) => void = () => {};
    const constructed = new Promise<void>((done, fail) => { constructedDone = done; constructedFail = fail; });
    let result: RenderWorkerResult | undefined;
    const tail: string[] = [];
    const consume = (line: string): void => {
      log.write(`${line}\n`); tail.push(line); if (tail.length > 20) tail.shift();
      if (!line.startsWith("{")) return;
      try {
        const record = JSON.parse(line) as { phase?: string; frames?: number; renderWall_ms?: number;
          medianFrame_ms?: number; p95Frame_ms?: number };
        if (record.phase === "constructed") constructedDone();
        if (record.phase === "result") result = record as RenderWorkerResult;
      } catch { /* progress output */ }
    };
    createInterface({ input: child.stdout }).on("line", consume);
    createInterface({ input: child.stderr }).on("line", consume);
    const finished = new Promise<void>((done, fail) => {
      child!.once("error", (error) => { constructedFail(error); fail(error); });
      child!.once("close", (code, signal) => {
        log.end();
        if (code === 0) { constructedDone(); done(); }
        else {
          const error = new Error(`render worker exited ${signal ?? code}: ${tail.join("\n")}`);
          constructedFail(error); fail(error);
        }
      });
    });
    finished.catch(() => {}); constructed.catch(() => {});
    console.log(`building ${scene} render workload at ${width}x${height}...`);
    await constructed;

    const developer = execFileSync("xcode-select", ["-p"]).toString().trim();
    const blankTemplate = resolve(developer,
      "../Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate");
    if (!existsSync(blankTemplate)) throw new Error(`Instruments Blank template not found at ${blankTemplate}`);
    console.log(`capturing ${counterSeconds}s of render-only Metal counters from node ${child.pid}...`);
    await run(["xcrun", "xctrace", "record",
      "--template", blankTemplate,
      "--instrument", "Metal Application",
      "--instrument", "GPU",
      "--instrument", "Metal GPU Counters",
      "--output", tracePath,
      "--no-prompt",
      "--time-limit", counterTimeLimit,
      "--attach", String(child.pid),
    ]);
    await finished;
    if (!result) throw new Error(`render worker produced no result; see ${logPath}`);

    // Counter-table export is CPU/disk reduction over an immutable completed
    // trace and can take several minutes. Release the process-wide GPU lock as
    // soon as the measured worker exits so unrelated shader compilation is not
    // blocked by report generation. `lockHeld` prevents the finalizer from
    // deleting a newer owner's lock.
    await releaseWebGPUExclusiveLock();
    lockHeld = false;

    console.log("exporting and reducing trace tables...");
    const { tables, policy } = await exportTables();
    const report = await writeReport(tables, policy, result, child.pid);
    await writeFile(capturePath, `${JSON.stringify({
      state: "complete",
      variant, traversal, shading, coneScale, coneTracing, radianceReconstruction, warmups, scene, resolution: { width, height }, counterSeconds, counterReduction,
      source,
      worker: result, tracePath, capturedAt: new Date().toISOString(),
    }, null, 2)}\n`);
    if (!keepTables) for (const path of Object.values(tables)) rmSync(path, { force: true });

    console.log("");
    for (const line of report.console) console.log(line);
    console.log(`\nreport:  ${resolve(outputDirectory, "report.html")}`);
    console.log(`summary: ${resolve(outputDirectory, "summary.json")}`);
    console.log(`trace:   ${tracePath}`);
  } finally {
    stop();
    process.removeListener("SIGINT", onSignal); process.removeListener("SIGTERM", onSignal);
    if (lockHeld) await releaseWebGPUExclusiveLock();
  }
};

await main().catch((error: unknown) => {
  console.error(`\nrender profile aborted: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
