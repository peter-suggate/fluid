import { readFileSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

interface RedundancySample {
  readonly sample: number;
  readonly family: string;
  readonly identicalFraction: number;
  readonly epsilonIdenticalFraction: number;
  readonly totalPages: number;
  readonly missingPriorPages: number;
}

const median = (values: readonly number[]): number => {
  if (values.length === 0) return 0;
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0
    ? (ordered[middle - 1]! + ordered[middle]!) / 2 : ordered[middle]!;
};

const windowMedian = (samples: readonly RedundancySample[], begin: number, end: number) => {
  const first = Math.floor(samples.length * begin), after = Math.ceil(samples.length * end);
  const window = samples.slice(first, Math.max(first + 1, after));
  return {
    samples: window.length,
    exact: median(window.map((sample) => sample.identicalFraction)),
    epsilon: median(window.map((sample) => sample.epsilonIdenticalFraction)),
  };
};

export function analyzePowerRedundancy(records: readonly Record<string, unknown>[]) {
  const samples = records.filter((record) => record.phase === "frame-redundancy-census")
    .map((record): RedundancySample => ({
      sample: Number(record.sample), family: String(record.family),
      identicalFraction: Number(record.identicalFraction),
      epsilonIdenticalFraction: Number(record.epsilonIdenticalFraction),
      totalPages: Number(record.totalPages), missingPriorPages: Number(record.missingPriorPages),
    })).filter((sample) => Number.isSafeInteger(sample.sample) && sample.sample > 0
      && sample.family.length > 0 && Object.values(sample).every((value) =>
        typeof value === "string" || Number.isFinite(value)));
  const byFamily = new Map<string, RedundancySample[]>();
  for (const sample of samples) {
    const family = byFamily.get(sample.family) ?? [];
    family.push(sample); byFamily.set(sample.family, family);
  }
  const families = [...byFamily].sort(([left], [right]) => left.localeCompare(right))
    .map(([family, values]) => {
      values.sort((left, right) => left.sample - right.sample);
      const overall = windowMedian(values, 0, 1);
      const exactDecision = overall.exact > 0.7 ? "fund-exact-delta-repair"
        : overall.exact < 0.3 ? "kill-exact-delta-repair" : "inconclusive";
      const epsilonDecision = overall.epsilon > 0.7 ? "fund-quantized-delta-repair"
        : overall.epsilon < 0.3 ? "kill-quantized-delta-repair" : "inconclusive";
      return { family, generations: values.length,
        early: windowMedian(values, 0, 0.2), middle: windowMedian(values, 0.4, 0.6),
        late: windowMedian(values, 0.8, 1), overall,
        maximumMissingPriorPages: Math.max(0, ...values.map((value) => value.missingPriorPages)),
        exactDecision, epsilonDecision };
    });
  return { schemaVersion: 1, experiment: "X-2-frame-redundancy-census", families };
}

const parseLines = (value: string): Record<string, unknown>[] => value.split(/\r?\n/)
  .filter(Boolean).flatMap((line) => {
    try { return [JSON.parse(line) as Record<string, unknown>]; } catch { return []; }
  });

if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);
  let records: Record<string, unknown>[];
  if (args.includes("--run")) {
    const lane = args.find((value) => value.startsWith("--lane="))?.slice(7) ?? "large";
    if (!new Set(["large", "mini"]).has(lane)) throw new Error("--lane must be large or mini");
    const steps = Number(args.find((value) => value.startsWith("--steps="))?.slice(8) ?? 500);
    if (!Number.isSafeInteger(steps) || steps < 2) throw new Error("--steps must be at least 2");
    const root = fileURLToPath(new URL("..", import.meta.url));
    const benchmark = fileURLToPath(new URL("./benchmark-power-dam.ts", import.meta.url));
    const child = spawn(process.execPath, ["--import", "tsx", benchmark,
      `--lane=${lane}`, `--steps=${steps}`, "--forward-ndjson", "--json"], {
      cwd: root, env: { ...process.env, FLUID_REDUNDANCY_CENSUS: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    child.stdout.setEncoding("utf8"); child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => { output += chunk; });
    child.stderr.on("data", (chunk: string) => { output += chunk; });
    const code = await new Promise<number>((resolve, reject) => {
      child.once("error", reject); child.once("exit", (value) => resolve(value ?? 1));
    });
    if (code !== 0) {
      const tail = output.split(/\r?\n/).filter(Boolean).slice(-30).join("\n");
      throw new Error(`redundancy capture exited ${code}${tail ? `\n${tail}` : ""}`);
    }
    records = parseLines(output);
  } else {
    if (args.length === 0) throw new Error(
      "usage: analyze-power-liquids-redundancy.ts LOG.ndjson [...] | --run [--lane=large|mini] [--steps=N]");
    records = args.flatMap((path) => parseLines(readFileSync(path, "utf8")));
  }
  console.log(JSON.stringify(analyzePowerRedundancy(records), null, 2));
}
