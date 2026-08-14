#!/usr/bin/env node
/**
 * Capture ALU, occupancy, bandwidth, and Metal stage timing for production SVO
 * rendering. xctrace *launches* the render-only worker, then the established
 * frame report reducer keeps every complete frame and a representative
 * per-frame timeline.
 *
 * It launches rather than attaches because attaching cannot work here: the pid
 * this process sees for a child it spawned is namespace-local, so the
 * Instruments daemon answers "Cannot find process for provided pid" and the
 * capture retains only other processes' GPU work. Launching costs the window
 * the ~100 s of Metal pipeline creation an attached capture skipped, which is
 * what {@link CONSTRUCTION_ALLOWANCE_S} pays for.
 *
 * Usage:
 *   node --import tsx tools/profile-svo-render-xctrace.ts
 *     [--scene=hose-tank] [--scene-module=tools/…-scene.ts] [--resolution=660x662]
 *     [--refinement=0..3] [--construction-seconds=200]
 *     [--screen-space-pixels=3] [--cone-fanout=1]
 *     [--variant=baseline] [--traversal=hybrid|canonical|canonical-parametric|compact|wide|raster-primary]
 *     [--shading=inline|split]
 *     [--cone-scale=0.5|0.25|0.125] [--cone-tracing=cones|exact|off] [--warmups=4]
 *     [--radiance-reconstruction=nearest|gated-linear|joint-bilateral|wide-relight|full-res-relight]
 *     [--counter-seconds=3] [--counter-reduction=100] [--out=DIR]
 *     [--timing-only] [--reuse-trace] [--reuse-tables]
 */
// These lanes render without a solver, but they construct the renderer, and
// a renderer resolves a method by id on any path that reaches a scene.
import "../lib/methods";
import { execFileSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { resolve } from "node:path";

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
} from "../lib/harness/webgpu-smoke-isolation";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lib/svo/svo-screen-space-termination";
import { SVO_DRY_TRAVERSAL_MODES, type SvoDryTraversalMode } from "../lib/svo/webgpu-svo-dry-scene";
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
/**
 * A scene from a module instead of the catalog, forwarded to the worker's
 * `FLUID_SVO_DRY_FRAME_SCENE_MODULE`. This is how a camera framing gets
 * profiled — the catalog id fixes the scene, not where you are standing.
 */
const sceneModule = flag("scene-module");
if (sceneModule !== undefined && !existsSync(resolve(root, sceneModule))) {
  throw new Error(`--scene-module ${sceneModule} does not exist`);
}
/**
 * Screen-space termination and cone fan-out default to what
 * `webgpu-renderer.ts` builds the production pipeline with, not to the
 * worker's own historical defaults (0 and off). An unflagged run of the
 * benchmark measures the pre-W1 path with a pass production does not run,
 * which is a good way to profile a frame the app never ships.
 */
const screenSpacePixels = Number(flag("screen-space-pixels")
  ?? (traversal === "raster-primary" ? SVO_SCREEN_SPACE_TERMINATION_CONTRACT.defaultThresholdPixels : 0));
if (!Number.isFinite(screenSpacePixels) || screenSpacePixels < 0) {
  throw new Error("--screen-space-pixels must be a non-negative finite number");
}
const coneFanout = (flag("cone-fanout") ?? "1") === "1";
/**
 * Authored-environment refinement depth, forwarded to the worker's
 * `FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT`.
 *
 * Omitted keeps the worker's depth-0 tree, which is what every calibrated lane
 * here is based on. A depth is a *document* rebuild there, not a tree option —
 * the scene's own factory re-runs at `cell / 2^depth` — so this profiles the
 * frame production draws at that rung rather than a finer tree over a coarse
 * set. See `lib/svo-render-tuning.ts:SVO_ENVIRONMENT_REFINEMENT_DEPTH_DEFAULT`.
 */
const refinementRaw = flag("refinement");
if (refinementRaw !== undefined && !/^[0-3]$/.test(refinementRaw)) {
  throw new Error("--refinement must be an integer in 0..3");
}
const refinement = refinementRaw === undefined ? undefined : Number(refinementRaw);
const keepTables = process.argv.includes("--keep-tables");
const reuseTrace = process.argv.includes("--reuse-trace");
const reuseTables = process.argv.includes("--reuse-tables");
const timingOnly = process.argv.includes("--timing-only");
const outputDirectory = resolve(root, flag("out") ?? "artifacts/xctrace-hose-render-utilization");
const tracePath = resolve(outputDirectory, "svo-render.trace");
/**
 * Seconds allowed for device creation, scene assembly and pipeline compilation
 * before the render loop starts, because `--launch` records from process start.
 *
 * Measured at 100-120 s for `hero-garden-hose` at 1600x1240 — nearly all of it
 * Metal pipeline creation, which an attached capture never sees. The window has
 * to outlast it or the trace holds construction and nothing else.
 *
 * `--construction-seconds` raises it, which a refined scene needs: the CPU plan
 * alone is ~28 s at depth 3 on the hero garden, before pipeline creation and the
 * voxelization of a set expanded at a 0.78 mm leaf.
 */
const CONSTRUCTION_ALLOWANCE_S = Number(flag("construction-seconds") ?? 200);
if (!Number.isFinite(CONSTRUCTION_ALLOWANCE_S) || CONSTRUCTION_ALLOWANCE_S < 1) {
  throw new Error("--construction-seconds must be a positive number of seconds");
}

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
  /** The depth the tree was actually built at, which a degraded allocation lowers. */
  readonly environmentRefinementDepth?: number;
  readonly detailCellSize_mm?: number;
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
  // `raster-primary` does not encode "Sparse voxel primary visibility" at all:
  // the megakernel is replaced by the brick coverage/resolve pair plus one
  // exact per-primitive proxy pass, so demanding that label rejects every
  // capture of the shipping raster path rather than catching anything.
  // The live-scene primitive stage has two encodings and the shipping one is
  // the conservative pair: `Sparse voxel exact live-scene primitive visibility`
  // is the *fallback* the dry scene takes only when the coverage pipelines are
  // unbuilt, the record count exceeds the 16-bit candidate key, or
  // `scenePrimitiveDirect` is set. Naming just that label rejected every
  // capture of the production path — the identical mistake the comment above
  // records for `Sparse voxel primary visibility`, one stage later. Either
  // form satisfies the gate; requiring both would fail whichever did not run.
  const expectedLabels: readonly (string | readonly string[])[] = split
    ? (traversal === "raster-primary"
      ? [["Sparse voxel exact live-scene primitive visibility",
        "Sparse voxel conservative live-scene primitive resolve"], "Sparse voxel deferred dry lighting"]
      : ["Sparse voxel primary visibility", "Sparse voxel deferred dry lighting"])
    : coneTracing === "cones"
      ? ["Sparse voxel cone-lighting prepass", "Sparse voxel dry scene"]
      : ["Sparse voxel dry scene"];
  for (const expected of expectedLabels) {
    const alternatives = typeof expected === "string" ? [expected] : expected;
    const attributed = alternatives.some((label) => {
      const pass = report.passes.find((candidate) => candidate.label === label);
      return pass !== undefined
        && (timingOnly || (pass.counterSamples > 0 && pass.occupancy !== undefined && pass.alu !== undefined));
    });
    if (!attributed) {
      failures.push(`${alternatives.join(" / ")} has no attributed occupancy/ALU samples`);
    }
  }
  const compactConeLabel = "Sparse voxel compact cone visibility";
  const compactCone = report.passes.some((candidate) => candidate.label === compactConeLabel);
  // The cone stages are demand-driven: on a scene whose cache is already warm
  // for the captured window they can legitimately not fire, so their absence is
  // reported rather than fatal. Their *presence* under a mode that withholds
  // them stays fatal, because that is a configuration error rather than a
  // scene-dependent one.
  if (splitRelight && coneTracing === "cones" && !compactCone) {
    console.warn(`note: ${compactConeLabel} encoded no GPU interval in this window`);
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
  // The worker degrades an unaffordable depth rather than throwing, so the rung
  // it *built* is the one this profile is about. Its own assertion already fails
  // the run; this keeps the report from ever being labelled with the request.
  if (refinement !== undefined && result.environmentRefinementDepth !== refinement) {
    throw new Error(`requested refinement depth ${refinement} but the worker built`
      + ` ${result.environmentRefinementDepth ?? "an unreported depth"}; see ${logPath}`);
  }
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
    title: `${scene} SVO rendering — ${variant} GPU frame profile`
      + `${refinement === undefined ? "" : ` at refinement depth ${refinement}`}`,
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
      ...(refinement === undefined
        ? {} : { FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT: String(refinement) }),
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
      variant, traversal, shading, coneScale, coneTracing, radianceReconstruction, warmups, scene, refinement, resolution: { width, height }, counterSeconds, counterReduction,
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
      variant, traversal, shading, coneScale, coneTracing, radianceReconstruction, warmups, scene, refinement, resolution: { width, height }, counterSeconds, counterReduction,
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
    sceneModule, refinement, screenSpacePixels, coneFanout,
    resolution: { width, height }, counterSeconds, counterReduction, source,
    tracePath, startedAt: new Date().toISOString(),
  }, null, 2)}\n`);
  const holder = await readWebGPUExclusiveLockHolder();
  if (holder) {
    throw new Error(`${WEBGPU_EXCLUSIVE_LOCK} is held by ${holder.description}`);
  }
  await acquireWebGPUExclusiveLock("svo-render-xctrace", `${scene} ${width}x${height}`);
  let lockHeld = true;
  const onSignal = (): void => {
    if (lockHeld) { releaseWebGPUExclusiveLockSync(); lockHeld = false; }
  };
  process.once("SIGINT", onSignal); process.once("SIGTERM", onSignal);
  try {
    const logPath = resolve(outputDirectory, "worker.log");
    const developer = execFileSync("xcode-select", ["-p"]).toString().trim();
    const blankTemplate = resolve(developer,
      "../Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate");
    if (!existsSync(blankTemplate)) throw new Error(`Instruments Blank template not found at ${blankTemplate}`);

    // `--launch`, never `--attach`. The pid this process sees for a child it
    // spawned is namespace-local, so the Instruments daemon answers "Cannot
    // find process for provided pid" and the capture holds nothing but other
    // processes' GPU work. Launching hands xctrace the process itself.
    //
    // The environment travels in a wrapper script rather than in `--env`
    // flags: `--env` takes one KEY=VALUE per flag, and a comma-separated list
    // silently collapses into the first variable, which produces a worker
    // running some other scene at some other resolution with no error at all.
    // `use_user_defined_labels_in_backend` is load-bearing — without it Metal
    // never sees the WebGPU pass names and every encoder exports as
    // "Render Command N", leaving nothing to attribute per-pass time to.
    const wrapperPath = resolve(outputDirectory, "render-worker.sh");
    await writeFile(wrapperPath, `#!/bin/sh
set -e
cd ${JSON.stringify(root)}
export WEBGPU_NODE_MODULE=${JSON.stringify(resolve(root, "node_modules/webgpu/index.js"))}
export FLUID_WEBGPU_DAWN_FEATURES=skip_validation,use_user_defined_labels_in_backend
export FLUID_SVO_DRY_FRAME_SCENE=${JSON.stringify(scene)}
${sceneModule === undefined ? "" : `export FLUID_SVO_DRY_FRAME_SCENE_MODULE=${JSON.stringify(sceneModule)}\n`}${refinement === undefined ? "" : `export FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT=${refinement}\n`}export FLUID_SVO_DRY_FRAME_SCREEN_SPACE_PIXELS=${screenSpacePixels}
export FLUID_SVO_DRY_FRAME_CONE_FANOUT=${coneFanout ? 1 : 0}
export FLUID_SVO_DRY_FRAME_WIDTH=${width}
export FLUID_SVO_DRY_FRAME_HEIGHT=${height}
export FLUID_SVO_DRY_FRAME_WARMUPS=${warmups}
export FLUID_SVO_DRY_FRAME_CONE_SCALE=${coneScale}
export FLUID_SVO_DRY_FRAME_CONE_TRACING=${JSON.stringify(coneTracing)}
export FLUID_SVO_DRY_FRAME_RADIANCE_RECONSTRUCTION=${JSON.stringify(radianceReconstruction)}
export FLUID_SVO_DRY_FRAME_TRAVERSAL=${JSON.stringify(traversal)}
export FLUID_SVO_DRY_FRAME_SHADING=${JSON.stringify(shading)}
export FLUID_SVO_DRY_FRAME_PROFILE_SECONDS=${counterSeconds + 9}
export FLUID_SVO_DRY_FRAME_ISOLATE_PASS_ENCODERS=${timingOnly ? 0 : 1}
# xctrace does not reliably forward the launched process's stdout, so the
# worker's own result JSON is recovered from this log after the capture.
exec ${JSON.stringify(process.execPath)} --import tsx ${JSON.stringify(worker)} > ${JSON.stringify(logPath)} 2>&1
`, { mode: 0o755 });

    console.log(`building ${scene} at ${width}x${height} and capturing under xctrace...`);
    // The launched run is construction plus the render loop, so the limit has
    // to cover both; the reducer's own window selection trims the front.
    const launchTimeLimit = `${CONSTRUCTION_ALLOWANCE_S + counterSeconds + 20}s`;
    await run(["xcrun", "xctrace", "record",
      "--template", blankTemplate,
      "--instrument", "Metal Application",
      "--instrument", "GPU",
      ...(timingOnly ? [] : ["--instrument", "Metal GPU Counters"]),
      "--output", tracePath,
      "--no-prompt",
      "--time-limit", launchTimeLimit,
      "--launch", "--", wrapperPath,
    ]);

    const result = workerResultFromLog();

    // Counter-table export is CPU/disk reduction over an immutable completed
    // trace and can take several minutes. Release the process-wide GPU lock as
    // soon as the measured worker exits so unrelated shader compilation is not
    // blocked by report generation. `lockHeld` prevents the finalizer from
    // deleting a newer owner's lock.
    await releaseWebGPUExclusiveLock();
    lockHeld = false;

    console.log("exporting and reducing trace tables...");
    const { tables, policy } = await exportTables();
    // Under `--launch` the traced pid belongs to the wrapper's exec'd worker,
    // which this process never learns; the encoders table is the authority.
    const report = await writeReport(tables, policy, result, await tracedPidFromEncoders(tables.encoders));
    await writeFile(capturePath, `${JSON.stringify({
      state: "complete",
      variant, traversal, shading, coneScale, coneTracing, radianceReconstruction, warmups, scene, refinement, sceneModule, screenSpacePixels, coneFanout, resolution: { width, height }, counterSeconds, counterReduction,
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
    // No child to stop: xctrace owns the launched worker and its own
    // `--time-limit` ends the run.
    process.removeListener("SIGINT", onSignal); process.removeListener("SIGTERM", onSignal);
    if (lockHeld) await releaseWebGPUExclusiveLock();
  }
};

await main().catch((error: unknown) => {
  console.error(`\nrender profile aborted: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
