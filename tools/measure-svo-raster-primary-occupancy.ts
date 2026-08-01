/**
 * Attribute Metal GPU counters to render passes in an Instruments trace.
 *
 * The handoff gates the raster-primary work on fragment occupancy, which only
 * Instruments reports. `tools/profile-svo-render-xctrace.ts` captures by
 * attaching to a running worker, and attach-by-pid fails when the profiler and
 * the worker do not share a pid namespace — the Instruments daemon cannot see
 * the child. `xctrace record --launch` has no such problem, so captures are
 * taken that way and reduced here:
 *
 *   xcrun xctrace record --template Blank.tracetemplate \
 *     --instrument "Metal Application" --instrument "GPU" \
 *     --instrument "Metal GPU Counters" --output ARM.trace \
 *     --no-prompt --time-limit 45s --launch -- ./run-arm.sh
 *   node --import tsx tools/measure-svo-raster-primary-occupancy.ts \
 *     --trace=ARM.trace --out=artifacts/render-raster-primary/occupancy-ARM.json
 *
 * Counter samples carry a timestamp but no pass identity, so each sample is
 * attributed to whichever `metal-gpu-intervals` interval contains it. Samples
 * landing in no interval, or in more than one, are reported rather than
 * silently folded into a neighbour — a pass whose counters are mostly
 * unattributed is not evidence of anything.
 */
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { parseTraceTable } from "./xctrace-trace-tables";

const flag = (name: string): string | undefined =>
  process.argv.find((argument) => argument.startsWith(`--${name}=`))?.split("=").slice(1).join("=");

const tracePath = flag("trace");
if (!tracePath) throw new Error("--trace=PATH.trace is required");
const outPath = flag("out") ?? "artifacts/render-raster-primary/occupancy.json";
/** Counters worth reporting per pass; everything else in the set is dropped. */
const WANTED = ["Fragment Occupancy", "Vertex Occupancy", "Compute Occupancy",
  "ALU Utilization", "ALU Limiter", "F32 Utilization", "F16 Utilization",
  "GPU Last Level Cache Limiter", "Buffer Read Limiter", "Partial Renders Count"] as const;

/**
 * The counter-value table holds one row per counter per sample interval — tens
 * of millions of rows for a 45-second capture — so rows are consumed as they
 * arrive and never collected. Buffering them backpressures the export pipe and
 * turns a slow export into an unbounded one.
 *
 * `parseTraceTable` treats a plain string as a *file path*, so the child's
 * stdout is handed over as an async iterable of chunks instead.
 */
const scan = async (
  schema: string,
  columns: readonly string[],
  visit: (row: Record<string, string>) => void,
): Promise<void> => {
  const child = spawn("xcrun", ["xctrace", "export", "--input", tracePath,
    "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${schema}"]`],
  { stdio: ["ignore", "pipe", "ignore"] });
  child.stdout.setEncoding("utf8");
  let failure: Error | undefined;
  child.once("error", (error) => { failure = error; });
  for await (const row of parseTraceTable(child.stdout as AsyncIterable<string>, { columns })) {
    visit(Object.fromEntries(Object.entries(row)
      .map(([key, value]) => [key, Array.isArray(value) ? value.join(" ") : String(value ?? "")])));
  }
  if (failure) throw failure;
};

const rows = async (schema: string, columns: readonly string[]) => {
  const collected: Record<string, string>[] = [];
  await scan(schema, columns, (row) => { collected.push(row); });
  return collected;
};

/**
 * Instruments renders every time column as display text, and the three forms
 * all appear in one trace: a bare nanosecond count, a `MM:SS.mmm.uuu` clock
 * (timestamps), and a unit-suffixed magnitude such as `9.96 µs` (durations).
 * Rejecting any one of them silently empties the interval list, so all three
 * are parsed here rather than assumed.
 */
const UNIT_NS: Record<string, number> = {
  ns: 1, "µs": 1e3, us: 1e3, ms: 1e6, s: 1e9, sec: 1e9, min: 6e10,
};
const nanoseconds = (value: string): number => {
  const text = value.trim().replace(/,/g, "");
  const plain = Number(text);
  if (text !== "" && Number.isFinite(plain)) return plain;
  const clock = /^(?:(\d+):)?(\d+)\.(\d{3})\.(\d{3})(?:\.(\d{3}))?$/.exec(text);
  if (clock) {
    const [, minutes = "0", seconds, milli, micro, nano = "0"] = clock;
    return ((Number(minutes) * 60 + Number(seconds)) * 1e9)
      + Number(milli) * 1e6 + Number(micro) * 1e3 + Number(nano);
  }
  const scaled = /^([0-9.]+)\s*([a-zµ]+)$/i.exec(text);
  if (scaled && UNIT_NS[scaled[2]] !== undefined) return Number(scaled[1]) * UNIT_NS[scaled[2]];
  return NaN;
};

const counterInfo = await rows("gpu-counter-info", ["counter-id", "name"]);
const nameById = new Map(counterInfo.map((row) => [row["counter-id"], row.name]));
const wanted = new Set(WANTED as readonly string[]);
const trackedIds = new Map([...nameById].filter(([, name]) => wanted.has(name)));
if (trackedIds.size === 0) {
  throw new Error(`trace carries none of the tracked counters; saw ${[...new Set(nameById.values())].join(", ")}`);
}

/**
 * Instruments renders an interval label as
 * `Command Buffer 0:Dawn_RenderPassEncoder_<our label>` and sometimes appends
 * the owning process and an encoder address. Strip all of it so the pass name
 * that reaches the report is the one the renderer wrote.
 */
const passLabel = (raw: string): string => raw
  .replace(/^Command Buffer \d+:/, "")
  .replace(/^Dawn_(?:Render|Compute|Blit)PassEncoder_/, "")
  .replace(/\s{2,}\(.*$/, "")
  .trim();

// The trace is system-wide: WindowServer's compositing shows up as
// `Read Surface` / `Write Surface` intervals and unlabelled `Render Command N`
// encoders, and folding those into the report makes another process's GPU work
// look like the renderer's. Keep the owning process on every interval so the
// report can separate them.
const processFilter = flag("process");
const intervals = (await rows("metal-gpu-intervals", ["start", "duration", "event-label", "process"]))
  .map((row) => ({
    label: passLabel(row["event-label"] ?? ""),
    process: row.process ?? "",
    start: nanoseconds(row.start ?? ""),
    duration: nanoseconds(row.duration ?? ""),
  }))
  .filter((interval) => Number.isFinite(interval.start) && Number.isFinite(interval.duration) && interval.label
    && (!processFilter || interval.process.includes(processFilter)))
  .sort((left, right) => left.start - right.start);
if (intervals.length === 0) {
  throw new Error("no usable GPU intervals — the trace has no labelled encoders, or a time column did not parse");
}
// Anonymous encoder labels mean the capture ran without Dawn's
// use_user_defined_labels_in_backend toggle, so pass names never reached Metal
// and nothing here can be attributed to a render pass.
if (intervals.every((interval) => /^(Render|Compute|Blit) Command \d+$/.test(interval.label))) {
  throw new Error("GPU intervals carry only generic encoder names; re-capture with "
    + "FLUID_WEBGPU_DAWN_FEATURES=use_user_defined_labels_in_backend");
}

interface Accumulator { total: number; count: number }
const perPass = new Map<string, { process: string; label: string; occupied_ns: number;
  instances: number; counters: Map<string, Accumulator> }>();
for (const interval of intervals) {
  const key = `${interval.process}\u0000${interval.label}`;
  const entry = perPass.get(key)
    ?? { process: interval.process, label: interval.label, occupied_ns: 0, instances: 0,
         counters: new Map<string, Accumulator>() };
  entry.occupied_ns += interval.duration;
  entry.instances += 1;
  perPass.set(key, entry);
}

// Intervals are sorted by start, so the containing one is found by seeking the
// last interval that starts at or before the sample and walking back over the
// few that can still be open. Intervals on different GPU channels do overlap,
// so the walk is bounded by the longest interval rather than stopping at the
// first miss.
const starts = intervals.map((interval) => interval.start);
const longest = intervals.reduce((maximum, interval) => Math.max(maximum, interval.duration), 0);
const containing = (timestamp: number): typeof intervals[number] | undefined => {
  let low = 0, high = starts.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (starts[middle] <= timestamp) low = middle + 1; else high = middle;
  }
  for (let index = low - 1; index >= 0 && starts[index] >= timestamp - longest; index -= 1) {
    const interval = intervals[index];
    if (timestamp < interval.start + interval.duration) return interval;
  }
  return undefined;
};

let seen = 0, attributed = 0, unattributed = 0;
await scan("gpu-counter-value", ["timestamp", "counter-id", "value"], (row) => {
  const counter = nameById.get(row["counter-id"] ?? "") ?? "";
  if (!wanted.has(counter)) return;
  const timestamp = nanoseconds(row.timestamp ?? "");
  const value = Number((row.value ?? "").replace(/,/g, ""));
  if (!Number.isFinite(timestamp) || !Number.isFinite(value)) return;
  seen += 1;
  const hit = containing(timestamp);
  if (!hit) { unattributed += 1; return; }
  attributed += 1;
  const entry = perPass.get(`${hit.process}\u0000${hit.label}`)!;
  const accumulator = entry.counters.get(counter) ?? { total: 0, count: 0 };
  accumulator.total += value; accumulator.count += 1;
  entry.counters.set(counter, accumulator);
});

const passes = [...perPass.values()]
  .map((entry) => ({
    label: entry.label,
    process: entry.process,
    instances: entry.instances,
    occupied_ms: entry.occupied_ns / 1e6,
    counterSamples: [...entry.counters.values()].reduce((sum, value) => sum + value.count, 0),
    counters: Object.fromEntries([...entry.counters]
      .map(([name, value]) => [name, value.total / value.count])),
  }))
  .sort((left, right) => right.occupied_ms - left.occupied_ms);

const report = {
  tool: "measure-svo-raster-primary-occupancy",
  trace: tracePath,
  intervals: intervals.length,
  counterSamples: seen,
  attributedSamples: attributed,
  unattributedSamples: unattributed,
  trackedCounters: [...trackedIds.values()],
  passes,
};
mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ ...report, passes: passes.slice(0, 12) }, null, 2));
