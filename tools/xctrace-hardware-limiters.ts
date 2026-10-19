/** Attribute hardware counters only to unambiguous, uncontended stage interiors.
 * Counter values are device-wide, not process counters. Frame labels are emitted
 * by profile-narrow-band-worker --frame-labels. Diagnostic encoder separation
 * changes scheduling: these durations are not acceptance benchmark timings.
 */
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseEncoderLabel } from "./xctrace-frame-report";
import { readTraceRows, timestampMicroseconds, durationMicroseconds } from "./xctrace-trace-tables";

export interface HardwareInterval {
  start: number; end: number; ours: boolean; labels: readonly string[];
}
export interface HardwareSpan { start: number; end: number; label?: string }

/** Sweep all GPU activity, including other processes. Never credit a mixed
 * encoder or concurrent foreign workload to its first stage label. */
export function exclusiveHardwareSpans(intervals: readonly HardwareInterval[]): HardwareSpan[] {
  const events: { time: number; index: number; add: boolean }[] = [];
  intervals.forEach((v, index) => {
    if (v.end > v.start) events.push({ time: v.start, index, add: true }, { time: v.end, index, add: false });
  });
  events.sort((a, b) => a.time - b.time);
  const active = new Set<number>(); const spans: HardwareSpan[] = [];
  let previous = events[0]?.time ?? 0, cursor = 0;
  while (cursor < events.length) {
    const time = events[cursor].time;
    if (time > previous) {
      const rows = [...active].map(i => intervals[i]);
      const labels = new Set(rows.flatMap(v => [...v.labels]));
      const label = rows.length > 0 && rows.every(v => v.ours && v.labels.length === 1)
        && labels.size === 1 ? [...labels][0] : undefined;
      const last = spans.at(-1);
      if (last && last.label === label && last.end === previous) last.end = time;
      else spans.push({ start: previous, end: time, label });
    }
    while (cursor < events.length && events[cursor].time === time) {
      const event = events[cursor++];
      if (event.add) active.add(event.index); else active.delete(event.index);
    }
    previous = time;
  }
  return spans;
}

/** Guard both sides by a full hardware sample period. Export order is not
 * timestamp sorted, so lookup cannot depend on the previously visited row. */
export function hardwareSampleStage(spans: readonly HardwareSpan[], time: number, guardUs: number): string | undefined {
  let low = 0, high = spans.length - 1, hit = -1;
  while (low <= high) { const mid = (low + high) >>> 1;
    if (spans[mid].start <= time) { hit = mid; low = mid + 1; } else high = mid - 1;
  }
  const span = spans[hit];
  return span && time - guardUs >= span.start && time + guardUs <= span.end ? span.label : undefined;
}

async function main(): Promise<void> {
  const flag = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
  const artifact = flag("artifact");
  const pid = flag("pid"); if (!pid || !artifact) throw new Error("--pid and --artifact required");
  const directory = resolve(artifact);
  const frame = flag("frame") ? Number(flag("frame")) : undefined;
  const throughFrame = Number(flag("through-frame") ?? frame);
  if (frame !== undefined && (!Number.isInteger(frame) || frame < 1
    || !Number.isInteger(throughFrame) || throughFrame < frame)) throw new Error("invalid physical frame range");
  if (frame === undefined && flag("through-frame")) throw new Error("--through-frame requires --frame");
  let start = Number(flag("start-us") ?? 0), end = Number(flag("end-us") ?? Infinity);
  const encoderLabels = new Map<string, string[]>();
  for await (const row of readTraceRows(resolve(directory, "encoders.ndjson"))) {
    encoderLabels.set(String(row["encoder-id"]), parseEncoderLabel(row["encoder-label"] as string).encoders);
  }
  const intervals: HardwareInterval[] = []; const frameIntervals: HardwareInterval[] = [];
  const observedFrames = new Set<number>();
  for await (const row of readTraceRows(resolve(directory, "gpu-intervals.ndjson"))) {
    const begin = timestampMicroseconds(row.start as string), duration = durationMicroseconds(row.duration as string);
    const ours = String(row.process).endsWith(`(${pid})`);
    const rawLabels = encoderLabels.get(String(row["encoder-id"])) ?? ["(unlabelled)"];
    const frames = rawLabels.map(label => /^\[frame (\d+)\] /.exec(label)?.[1]);
    const labels = rawLabels.map(label => label.replace(/^\[frame \d+\] /, ""));
    const interval = { start: begin, end: begin + duration, ours,
      labels: labels.includes("(unlabelled)") ? [] : [...new Set(labels)] };
    intervals.push(interval);
    if (ours && frame !== undefined && frames.length && frames.every(value => value !== undefined
      && Number(value) >= frame && Number(value) <= throughFrame)) {
      frameIntervals.push(interval);
      frames.forEach(value => observedFrames.add(Number(value)));
    }
  }
  if (frame !== undefined) {
    if (!frameIntervals.length) throw new Error(`No encoder labels for physical frame ${frame}`);
    for (let requested = frame; requested <= throughFrame; requested++) {
      if (!observedFrames.has(requested)) throw new Error(`Missing physical frame ${requested} in requested range`);
    }
    start = Math.min(...frameIntervals.map(v => v.start)); end = Math.max(...frameIntervals.map(v => v.end));
  }
  const spans = exclusiveHardwareSpans(intervals.filter(v => v.end >= start && v.start <= end));
  const ownSpans = exclusiveHardwareSpans(intervals.filter(v => v.ours && v.end >= start && v.start <= end));
  const meta = new Map<string, { name: string; periodUs: number; type: string }>();
  for await (const row of readTraceRows(resolve(directory, "counter-info.ndjson"))) {
    meta.set(String(row["counter-id"]), { name: String(row.name), type: String(row.type),
      periodUs: Number(String(row["sample-interval"]).replaceAll(",", "")) / 1000 });
  }
  type Stat = { sum: number; count: number; min: number; max: number };
  const stats = new Map<string, Map<string, Map<string, Stat>>>();
  let retainedRows = 0, rejectedRows = 0;
  for await (const row of readTraceRows(resolve(directory, flag("counters") ?? "counter-value.ndjson"))) {
    const time = timestampMicroseconds(row.timestamp as string); if (time < start || time > end) continue;
    const counter = meta.get(String(row["counter-id"])); if (!counter) continue;
    const label = hardwareSampleStage(spans, time, counter.periodUs);
    if (!label) { rejectedRows++; continue; }
    const value = Number(String(row.value).replaceAll(",", "")); if (!Number.isFinite(value)) continue;
    retainedRows++;
    const stage = stats.get(label) ?? new Map(); stats.set(label, stage);
    const rings = stage.get(counter.name) ?? new Map(); stage.set(counter.name, rings);
    const ring = String(row["ring-buffer-index"]);
    const stat = rings.get(ring) ?? { sum: 0, count: 0, min: Infinity, max: -Infinity };
    stat.sum += value; stat.count++; stat.min = Math.min(stat.min, value); stat.max = Math.max(stat.max, value); rings.set(ring, stat);
  }
  const duration = (source: readonly HardwareSpan[], label: string) => source.filter(s => s.label === label)
    .reduce((sum, s) => sum + Math.max(0, Math.min(s.end, end) - Math.max(s.start, start)), 0);
  const stages = [...stats].map(([label, counters]) => ({ label,
    stageUs: duration(ownSpans, label), exclusiveUs: duration(spans, label),
    exclusiveFraction: duration(spans, label) / duration(ownSpans, label),
    counters: Object.fromEntries([...counters].map(([name, rings]) => {
      const perRing = [...rings].map(([ring, s]) => ({ ring, mean: s.sum / s.count, samples: s.count, min: s.min, max: s.max }));
      return [name, { meanAcrossRings: perRing.reduce((sum, s) => sum + s.mean, 0) / perRing.length, perRing }];
    })),
  })).sort((a, b) => b.exclusiveUs - a.exclusiveUs);
  const report = { status: "Diagnostic only; mixed encoders, observed foreign GPU overlap and sample boundaries excluded", directory, pid, frame, throughFrame: frame === undefined ? undefined : throughFrame, startUs: start, endUs: end,
    units: "Percentages and bandwidth retain Instruments export units; ring means equally weighted, not summed", retainedRows, rejectedRows, stages };
  const output = resolve(directory, flag("output") ?? "hardware-limiters.json");
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ output, start, end, stages: stages.length, retainedRows, rejectedRows }));
}
if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) {
  await main().catch(error => { console.error(error); process.exitCode = 1; });
}
