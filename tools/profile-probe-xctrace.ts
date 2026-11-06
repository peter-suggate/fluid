/**
 * Frame structure of a probe-reservoir-bulk run, observed from outside the
 * process with Instruments' Metal System Trace (the gpu-frame-profile
 * skill's clean regime): GPU busy and idle gaps per advance, the largest GPU
 * tasks, and where the process's CPU time goes. No counters, no in-process
 * probes: pass the probe no --kernels or --dispatches.
 *
 *   node --import tsx tools/profile-probe-xctrace.ts --out=DIR [--record-only|--report-only]
 *     [--anchor=LABEL] -- <probe-reservoir-bulk arguments>
 *
 * Record under tools/gpu/fluid-gpu-run.sh timing (--record-only); the report
 * (export and reduction) needs no GPU. Following the skill's traps: the whole
 * run is recorded (never --window), GPU intervals are joined to the encoder
 * list by encoder-id and named by its label, and no counters are taken.
 *
 * An advance is the span between consecutive starts of the anchor encoder: by
 * default the label that recurs once an advance (its count closest to the
 * probe's --steps).
 */
import { execFileSync, spawn } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { durationMicroseconds, parseTraceTable, readTraceRows, timestampMicroseconds, type TraceRow } from "./xctrace-trace-tables";

const root = resolve(import.meta.dirname, "..");
const split = process.argv.indexOf("--");
const own = split < 0 ? process.argv.slice(2) : process.argv.slice(2, split);
const probeArguments = split < 0 ? [] : process.argv.slice(split + 1);
const flag = (name: string) => own.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const out = resolve(flag("out") ?? "artifacts/xctrace-probe");
const trace = `${out}/probe.trace`;
const recordOnly = own.includes("--record-only"), reportOnly = own.includes("--report-only");
mkdirSync(out, { recursive: true });

const run = (argv: string[], log?: string) => new Promise<void>((done, fail) => {
  const child = spawn(argv[0]!, argv.slice(1), { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
  const sink = log ? createWriteStream(log) : undefined;
  child.stdout.on("data", (c: Buffer) => sink?.write(c));
  let stderr = ""; child.stderr.on("data", (c: Buffer) => { stderr += c; sink?.write(c); });
  child.once("exit", code => { sink?.end(); code === 0 ? done() : fail(new Error(`${argv.slice(0, 4).join(" ")} failed (${code}): ${stderr.slice(-600)}`)); });
  child.once("error", fail);
});

if (!reportOnly) {
  if (probeArguments.some(a => a.startsWith("--kernels") || a === "--dispatches")) throw new Error("the clean regime takes no in-process probes (--kernels, --dispatches)");
  if (existsSync(trace)) execFileSync("rm", ["-rf", trace]);
  const environment = {
    WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? `${root}/node_modules/webgpu/index.js`,
    // Pass labels reach the Metal encoders; validation stays as the probe runs it.
    FLUID_WEBGPU_DAWN_FEATURES: "use_user_defined_labels_in_backend",
  };
  await run(["xcrun", "xctrace", "record", "--template", "Metal System Trace", "--output", trace, "--no-prompt", "--target-stdout", "-",
    ...Object.entries(environment).flatMap(([k, v]) => ["--env", `${k}=${v}`]),
    "--launch", "--", process.execPath, "--import", "tsx", "tools/probe-reservoir-bulk.ts", ...probeArguments], `${out}/traced.log`);
  writeFileSync(`${out}/probe-arguments.json`, `${JSON.stringify(probeArguments)}\n`);
  console.log(`recorded ${trace}`);
}
if (recordOnly) process.exit(0);

const TABLES = [
  { schema: "metal-gpu-intervals", file: "gpu-intervals", columns: ["start", "duration", "channel-name", "process", "encoder-id"] },
  { schema: "metal-application-encoders-list", file: "encoders", columns: ["encoder-id", "encoder-label", "process"] },
  { schema: "metal-application-command-buffer-submissions", file: "submissions", columns: ["start", "duration", "num-encoders", "process"] },
  { schema: "time-profile", file: "time-profile", stacks: true, columns: ["time", "process", "thread", "thread-state", "weight"] },
] as const;
for (const t of TABLES) {
  const ndjson = `${out}/${t.file}.ndjson`;
  if (existsSync(ndjson)) continue;
  const child = spawn("xcrun", ["xctrace", "export", "--input", trace, "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${t.schema}"]`], { cwd: root, stdio: ["ignore", "pipe", "pipe"] });
  child.stdout.setEncoding("utf8");
  const exited = new Promise<number | null>(done => child.once("exit", done));
  const sink = createWriteStream(ndjson);
  for await (const row of parseTraceTable(child.stdout as AsyncIterable<string>, { stacks: "stacks" in t, columns: t.columns })) sink.write(`${JSON.stringify(row)}\n`);
  await new Promise<void>(done => sink.end(done));
  if (await exited !== 0) throw new Error(`xctrace export ${t.schema} failed`);
}
const rows = async (file: string) => { const all: TraceRow[] = []; for await (const r of readTraceRows(`${out}/${file}.ndjson`)) all.push(r); return all; };
const text = (v: TraceRow[string]) => (typeof v === "string" ? v : "");

// The probe is the traced node process; Instruments names it "node (pid)".
const encoders = await rows("encoders");
const ours = (p: TraceRow[string]) => /^node\b/.test(text(p));
const labels = new Map(encoders.filter(e => ours(e.process)).map(e => [text(e["encoder-id"]), text(e["encoder-label"]) || "(unlabelled)"]));
const intervals = (await rows("gpu-intervals")).filter(r => ours(r.process))
  .map(r => ({ start: timestampMicroseconds(text(r.start)), duration: durationMicroseconds(text(r.duration)), label: labels.get(text(r["encoder-id"])) ?? "(no encoder)" }))
  .sort((a, b) => a.start - b.start);
const steps = Number((JSON.parse(existsSync(`${out}/probe-arguments.json`) ? execFileSync("cat", [`${out}/probe-arguments.json`]).toString() : "[]") as string[])
  .find(a => a.startsWith("--steps="))?.slice(8) ?? 60);
const counts = new Map<string, number>();
for (const i of intervals) counts.set(i.label, (counts.get(i.label) ?? 0) + 1);
const anchor = flag("anchor") ?? [...counts.entries()].filter(([l]) => l !== "(no encoder)" && l !== "(unlabelled)")
  .sort((a, b) => Math.abs(a[1] - steps) - Math.abs(b[1] - steps) || b[1] - a[1])[0]?.[0];
if (!anchor) throw new Error("no labelled GPU intervals of the node process: was use_user_defined_labels_in_backend applied?");
const starts = intervals.filter(i => i.label === anchor).map(i => i.start);
// Advances: consecutive anchor starts. The first is dropped (warm-up).
const frames: { from: number; to: number }[] = [];
for (let k = 1; k + 1 < starts.length; k++) frames.push({ from: starts[k]!, to: starts[k + 1]! });

const percentile = (values: number[], p: number) => { const s = [...values].sort((a, b) => a - b); return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))]! : 0; };
const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / Math.max(1, values.length);
const GAP_FLOOR_US = 2;
const perFrame = frames.map(f => {
  const inside = intervals.filter(i => i.start >= f.from && i.start < f.to);
  let busy = 0, end = f.from, gaps = 0, gapUs = 0, longGaps = 0, longGapUs = 0;
  for (const i of inside) {
    if (i.start > end) { const g = i.start - end; if (g >= GAP_FLOOR_US) { gaps++; gapUs += g; } if (g >= 50) { longGaps++; longGapUs += g; } }
    const e = Math.min(f.to, i.start + i.duration);
    if (e > Math.max(end, i.start)) busy += e - Math.max(end, i.start);
    end = Math.max(end, e);
  }
  if (f.to > end) { const g = f.to - end; if (g >= GAP_FLOOR_US) { gaps++; gapUs += g; } if (g >= 50) { longGaps++; longGapUs += g; } }
  return { wall: f.to - f.from, busy, gaps, gapUs, longGaps, longGapUs, intervals: inside.length };
});
const tasks = new Map<string, { us: number; n: number }>();
for (const f of frames) for (const i of intervals) if (i.start >= f.from && i.start < f.to) {
  const t = tasks.get(i.label) ?? { us: 0, n: 0 }; t.us += i.duration; t.n++; tasks.set(i.label, t);
}
const window = { from: frames[0]?.from ?? 0, to: frames.at(-1)?.to ?? 0 };

// CPU: the node process's running samples inside the advances, by thread and
// by the first frame of each stack that is not a system leaf.
const samples = (await rows("time-profile")).filter(r => ours(r.process));
const byThread = new Map<string, number>(), byLeaf = new Map<string, number>(), byModule = new Map<string, number>();
let cpuUs = 0, waitUs = 0;
// A main thread that polls the device (paceDawnEventPolling) spins in
// Atomics.load: running, but waiting on the GPU, not encoding.
const WAIT = /^(Builtins_Atomics(Load|Wait).*|__psynch_.*|semaphore_.*wait.*|mach_msg2?_trap|__ulock_wait.*)$/;
const SYSTEM = /^(0x[0-9a-f]+|__.*|_platform_.*|mach_msg.*|semaphore_.*|kevent.*|_pthread.*|pthread_.*)$/;
for (const s of samples) {
  const t = timestampMicroseconds(text(s.time));
  if (t < window.from || t >= window.to) continue;
  if (!/Running/i.test(text(s["thread-state"]) || "Running")) continue;
  const w = durationMicroseconds(text(s.weight)) || 1000;
  cpuUs += w;
  const thread = text(s.thread).replace(/0x[0-9a-f]+/g, "").replace(/\(\d+\)/g, "").trim() || "(thread)";
  byThread.set(thread, (byThread.get(thread) ?? 0) + w);
  const stack = (s.frames ?? []) as readonly string[];
  if (stack[0] !== undefined && WAIT.test(stack[0])) { waitUs += w; continue; }
  const leaf = stack.find(f => !SYSTEM.test(f)) ?? stack[0] ?? "(no stack)";
  byLeaf.set(leaf, (byLeaf.get(leaf) ?? 0) + w);
  const module = stack.some(f => /dawn|Dawn|wgpu|webgpu|tint|Tint/.test(f)) ? "dawn/tint" : stack.some(f => /AGX|MTL|Metal|IOGPU|IOKit/.test(f)) ? "metal driver" : stack.some(f => /^(Builtins_|v8::|node::|uv_)/.test(f) || /\.(ts|js|mjs)\b/.test(f)) ? "js/v8/node" : "other";
  byModule.set(module, (byModule.get(module) ?? 0) + w);
}
const n = Math.max(1, frames.length);
const top = (m: Map<string, number>, k: number) => [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, k).map(([name, us]) => ({ name, ms_per_advance: +(us / n / 1e3).toFixed(3) }));
const summary = {
  anchor, advances: frames.length, anchorCount: starts.length, intervals: intervals.length,
  wall_ms: { mean: +(mean(perFrame.map(f => f.wall)) / 1e3).toFixed(3), median: +(percentile(perFrame.map(f => f.wall), 0.5) / 1e3).toFixed(3) },
  gpuBusy_ms: { mean: +(mean(perFrame.map(f => f.busy)) / 1e3).toFixed(3), median: +(percentile(perFrame.map(f => f.busy), 0.5) / 1e3).toFixed(3) },
  gaps: { floor_us: GAP_FLOOR_US, count_mean: +mean(perFrame.map(f => f.gaps)).toFixed(1), ms_mean: +(mean(perFrame.map(f => f.gapUs)) / 1e3).toFixed(3),
    over50us_count_mean: +mean(perFrame.map(f => f.longGaps)).toFixed(1), over50us_ms_mean: +(mean(perFrame.map(f => f.longGapUs)) / 1e3).toFixed(3) },
  gpuIntervals_per_advance: +mean(perFrame.map(f => f.intervals)).toFixed(1),
  topTasks: [...tasks.entries()].sort((a, b) => b[1].us - a[1].us).slice(0, 15).map(([label, t]) => ({ label, ms_per_advance: +(t.us / n / 1e3).toFixed(3), per_advance: +(t.n / n).toFixed(1) })),
  cpu: { ms_per_advance: +(cpuUs / n / 1e3).toFixed(3), wait_ms_per_advance: +(waitUs / n / 1e3).toFixed(3), work_ms_per_advance: +((cpuUs - waitUs) / n / 1e3).toFixed(3), byThread: top(byThread, 6), byModule: top(byModule, 6), topLeaves: top(byLeaf, 15) },
};
writeFileSync(`${out}/summary.json`, `${JSON.stringify(summary, null, 1)}\n`);
console.log(JSON.stringify(summary, null, 1));
