/** Reduce an attached Sparse CM12 stage-cost trace to the shared frame report. */
import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { CounterRowSelector, makeCounterExtractionPolicy,
  type CounterExtractionPolicy } from "./profile-mini-dam-xctrace";
import { buildFrameReport, renderFrameReportHtml } from "./xctrace-frame-report";
import { parseTraceTable, readTraceRows } from "./xctrace-trace-tables";

const value = (name: string): string | undefined => process.argv.slice(2)
  .find((entry) => entry.startsWith(`--${name}=`))?.slice(name.length + 3);
const trace = value("trace");
const output = resolve(value("out") ?? "artifacts/xctrace-sparse-cm12-stage-report");
const tracedPid = Number(value("pid"));
const frameLimit = Number(value("frames") ?? "1");
const counterReduction = Number(value("counter-reduction") ?? "100");
const includeCounters = value("counters") !== "0";
if (!trace || !Number.isSafeInteger(tracedPid) || tracedPid <= 0) {
  throw new Error("--trace and a positive --pid are required");
}
if (!Number.isSafeInteger(frameLimit) || frameLimit <= 0) {
  throw new Error("--frames must be a positive integer");
}
await mkdir(output, { recursive: true });

const specifications = ([
  ["metal-gpu-intervals", "gpu-intervals",
    ["start", "duration", "channel-name", "event-label", "process", "encoder-id"]],
  ["metal-application-encoders-list", "encoders",
    ["encoder-id", "encoder-label", "process"]],
  ["metal-application-command-buffer-submissions", "command-buffer-submissions",
    ["start", "duration", "num-encoders", "process", "cmdbuffer-id"]],
  ["metal-command-buffer-completed", "command-buffer-completed",
    ["timestamp", "cmdbuffer-id"]],
  ["gpu-counter-info", "counter-info",
    ["counter-id", "name", "type", "sample-interval"]],
  ["gpu-counter-value", "counter-value",
    ["timestamp", "counter-id", "value", "ring-buffer-index"]],
  ["metal-shader-profiler-intervals", "shader-profile",
    ["start", "label", "pso-label"]],
] as const).filter(([schema]) => includeCounters || !schema.startsWith("gpu-counter-"));

const tables: Record<string, string> = {};
let counterPolicy: CounterExtractionPolicy | undefined;
for (const [schema, name, columns] of specifications) {
  const path = `${output}/${name}.ndjson`;
  const child = spawn("xcrun", ["xctrace", "export", "--input", trace,
    "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${schema}"]`],
  { stdio: ["ignore", "pipe", "pipe"] });
  if (!child.stdout || !child.stderr) throw new Error(`missing export pipes for ${schema}`);
  child.stdout.setEncoding("utf8");
  let stderr = "";
  child.stderr.on("data", (chunk: Buffer) => { stderr += chunk.toString(); });
  const finished = new Promise<void>((done, fail) => {
    child.once("exit", (code) => code === 0 ? done()
      : fail(new Error(`${schema} export failed (${code}): ${stderr.slice(0, 400)}`)));
    child.once("error", fail);
  });
  const sink = createWriteStream(path);
  const selector = name === "counter-value" && counterPolicy
    ? new CounterRowSelector(counterPolicy) : undefined;
  let sourceRows = 0;
  let rows = 0;
  for await (const row of parseTraceTable(child.stdout as AsyncIterable<string>, { columns })) {
    sourceRows += 1;
    if (selector && !selector.keep(row)) continue;
    sink.write(`${JSON.stringify(row)}\n`);
    rows += 1;
  }
  await finished;
  await new Promise<void>((done) => sink.end(done));
  tables[name] = path;
  console.log(`${schema}: ${rows}${rows === sourceRows ? "" : `/${sourceRows}`}`);
  if (name === "counter-info") {
    const counters = new Map<string, string>();
    for await (const row of readTraceRows(path)) {
      counters.set(String(row["counter-id"]), String(row.name ?? ""));
    }
    counterPolicy = makeCounterExtractionPolicy(counters, counterReduction);
  }
}

const report = await buildFrameReport({
  tables,
  lane: "sparse-cm12-stage-cost",
  environment: {
    FLUID_GPU_ISOLATE_PASS_LABELS: "1",
    // This standalone probe owns semantic PassBroker splitting, but unlike the
    // smoke harness it does not install timestamp-audit encoder scratch.
    FLUID_GPU_ISOLATE_PASS_ENCODERS: "0",
  },
  tracedPid,
  frameLimit,
  counterExtraction: counterPolicy && {
    sourceCounterCount: counterPolicy.sourceCounterCount,
    retainedCounterCount: counterPolicy.retainedCounterCount,
    timestampStride: counterPolicy.timestampStride,
  },
});
await writeFile(`${output}/summary.json`, `${JSON.stringify(report, null, 2)}\n`);
await writeFile(`${output}/report.html`, renderFrameReportHtml(report));
for (const line of report.console) console.log(line);
