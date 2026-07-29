#!/usr/bin/env node
/** Compare two render-only hose-scene xctrace artifacts.
 *
 * Usage:
 *   node --import tsx tools/compare-svo-render-profiles.ts \
 *     --baseline=artifacts/render-ab/baseline \
 *     --candidate=artifacts/render-ab/wide-only \
 *     --baseline-benchmark=artifacts/render-ab/baseline/benchmark.json \
 *     --candidate-benchmark=artifacts/render-ab/wide-only/benchmark.json \
 *     [--out=artifacts/render-ab/comparison]
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const flag = (name: string): string | undefined => process.argv
  .find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);

interface Capture {
  readonly variant?: string;
  readonly traversal?: string;
  readonly coneScale?: number;
  readonly scene?: string;
  readonly resolution?: { readonly width: number; readonly height: number };
  readonly worker?: { readonly frames?: number; readonly medianFrame_ms?: number; readonly p95Frame_ms?: number };
  readonly source?: { readonly commit?: string; readonly dirty?: boolean; readonly changedFiles?: number;
    readonly fingerprint?: string; readonly renderFingerprint?: string; readonly renderFiles?: number };
}

interface Pass {
  readonly label: string;
  readonly gpuMsPerFrame?: number;
  readonly occupancy?: number;
  readonly alu?: number;
  readonly readGBs?: number;
  readonly writeGBs?: number;
  readonly counterSamples?: number;
  readonly exactAttribution?: boolean;
}

interface Summary {
  readonly frames: { readonly count: number; readonly medianMs: number; readonly p10Ms: number; readonly p90Ms: number };
  readonly passes: readonly Pass[];
  readonly counters: { readonly exclusiveCoverage?: number; readonly meanOccupancy?: number; readonly meanAlu?: number };
  readonly contention?: readonly { readonly process?: string; readonly gpuMs?: number }[];
}

interface Benchmark {
  readonly traversalMode?: string;
  readonly resolution: { readonly width: number; readonly height: number };
  readonly timing: { readonly median_ms: number; readonly p95_ms: number };
  readonly scene: { readonly presetId: string };
  readonly fingerprint: { readonly imageHashFnv1a32: string; readonly samples?: readonly unknown[] };
}

interface Artifact { readonly directory: string; readonly capture: Capture; readonly summary: Summary }

const artifact = (input: string | undefined, role: string): Artifact => {
  if (!input) throw new Error(`--${role}=DIR is required`);
  const absolute = resolve(root, input);
  const directory = absolute.endsWith(".json") ? dirname(absolute) : absolute;
  const summaryPath = absolute.endsWith(".json") ? absolute : resolve(directory, "summary.json");
  const capturePath = resolve(directory, "capture.json");
  if (!existsSync(summaryPath)) throw new Error(`${role} summary not found: ${summaryPath}`);
  if (!existsSync(capturePath)) throw new Error(`${role} capture manifest not found: ${capturePath}`);
  return {
    directory,
    summary: JSON.parse(readFileSync(summaryPath, "utf8")) as Summary,
    capture: JSON.parse(readFileSync(capturePath, "utf8")) as Capture,
  };
};

const baseline = artifact(flag("baseline"), "baseline");
const candidate = artifact(flag("candidate"), "candidate");
const benchmarkPath = (name: string): string | undefined => {
  const value = flag(name);
  return value ? resolve(root, value) : undefined;
};
const baselineBenchmarkPath = benchmarkPath("baseline-benchmark");
const candidateBenchmarkPath = benchmarkPath("candidate-benchmark");
if (Boolean(baselineBenchmarkPath) !== Boolean(candidateBenchmarkPath)) {
  throw new Error("--baseline-benchmark and --candidate-benchmark must be supplied together");
}
const readBenchmark = (path: string | undefined): Benchmark | undefined => path
  ? JSON.parse(readFileSync(path, "utf8")) as Benchmark : undefined;
const baselineBenchmark = readBenchmark(baselineBenchmarkPath);
const candidateBenchmark = readBenchmark(candidateBenchmarkPath);
const finite = (value: number | undefined): number | undefined => Number.isFinite(value) ? value : undefined;
const deltaPercent = (before: number | undefined, after: number | undefined): number | undefined => {
  const finiteBefore = finite(before);
  const finiteAfter = finite(after);
  return finiteBefore !== undefined && finiteAfter !== undefined && finiteBefore !== 0
    ? 100 * (finiteAfter - finiteBefore) / finiteBefore : undefined;
};
const speedup = (before: number | undefined, after: number | undefined): number | undefined => {
  const finiteBefore = finite(before);
  const finiteAfter = finite(after);
  return finiteBefore !== undefined && finiteAfter !== undefined && finiteAfter !== 0
    ? finiteBefore / finiteAfter : undefined;
};
const pointDelta = (before: number | undefined, after: number | undefined): number | undefined => {
  const finiteBefore = finite(before);
  const finiteAfter = finite(after);
  return finiteBefore !== undefined && finiteAfter !== undefined ? 100 * (finiteAfter - finiteBefore) : undefined;
};

const mismatches: string[] = [];
if (baseline.capture.scene !== candidate.capture.scene) mismatches.push("scene differs");
if (JSON.stringify(baseline.capture.resolution) !== JSON.stringify(candidate.capture.resolution)) mismatches.push("resolution differs");
if (baseline.capture.coneScale !== candidate.capture.coneScale) mismatches.push("cone scale differs");
if (baselineBenchmark && candidateBenchmark) {
  if (baselineBenchmark.scene.presetId !== candidateBenchmark.scene.presetId) mismatches.push("benchmark scene differs");
  if (JSON.stringify(baselineBenchmark.resolution) !== JSON.stringify(candidateBenchmark.resolution)) {
    mismatches.push("benchmark resolution differs");
  }
  if (JSON.stringify(baselineBenchmark.resolution) !== JSON.stringify(baseline.capture.resolution)) {
    mismatches.push("benchmark and xctrace resolutions differ");
  }
}
if (mismatches.length > 0) throw new Error(`profiles are not comparable: ${mismatches.join("; ")}`);

const warnings: string[] = [];
const baselineSource = baseline.capture.source;
const candidateSource = candidate.capture.source;
if (baselineSource?.renderFingerprint && candidateSource?.renderFingerprint) {
  if (baselineSource.renderFingerprint !== candidateSource.renderFingerprint) {
    warnings.push("render-source fingerprints differ; this is not a strict same-source feature-flag A/B");
  }
} else if (baselineSource?.fingerprint && candidateSource?.fingerprint
  && baselineSource.fingerprint !== candidateSource.fingerprint) {
  warnings.push("working-tree fingerprints differ; verify intervening edits do not affect the render runtime");
}
for (const [role, current] of [["baseline", baseline], ["candidate", candidate]] as const) {
  if (current.summary.frames.count < 10) warnings.push(`${role} has fewer than 10 complete xctrace frames`);
  if ((current.summary.counters.exclusiveCoverage ?? 0) < 0.05) {
    warnings.push(`${role} counter attribution coverage is below 5%; treat ALU/occupancy as directional`);
  }
  if ((current.summary.contention?.length ?? 0) > 0) {
    warnings.push(`${role} capture overlaps other GPU processes; prefer worker medians and repeat before accepting a small delta`);
  }
  for (const pass of current.summary.passes) {
    if (!pass.exactAttribution) warnings.push(`${role} ${pass.label} timing attribution is not exact`);
    if ((pass.counterSamples ?? 0) < 10) warnings.push(`${role} ${pass.label} has fewer than 10 counter samples`);
  }
}

const passNames = [...new Set([...baseline.summary.passes, ...candidate.summary.passes].map((pass) => pass.label))];
const passes = passNames.map((label) => {
  const before = baseline.summary.passes.find((pass) => pass.label === label);
  const after = candidate.summary.passes.find((pass) => pass.label === label);
  return {
    label,
    baselineMs: finite(before?.gpuMsPerFrame),
    candidateMs: finite(after?.gpuMsPerFrame),
    deltaPercent: deltaPercent(before?.gpuMsPerFrame, after?.gpuMsPerFrame),
    speedup: speedup(before?.gpuMsPerFrame, after?.gpuMsPerFrame),
    occupancyPointDelta: pointDelta(before?.occupancy, after?.occupancy),
    aluPointDelta: pointDelta(before?.alu, after?.alu),
    baselineReadGBs: finite(before?.readGBs), candidateReadGBs: finite(after?.readGBs),
    baselineWriteGBs: finite(before?.writeGBs), candidateWriteGBs: finite(after?.writeGBs),
    baselineCounterSamples: before?.counterSamples ?? 0,
    candidateCounterSamples: after?.counterSamples ?? 0,
  };
});

const comparison = {
  baseline: { directory: baseline.directory, capture: baseline.capture },
  candidate: { directory: candidate.directory, capture: candidate.capture },
  frame: {
    baselineMedianMs: baseline.summary.frames.medianMs,
    candidateMedianMs: candidate.summary.frames.medianMs,
    deltaPercent: deltaPercent(baseline.summary.frames.medianMs, candidate.summary.frames.medianMs),
    speedup: speedup(baseline.summary.frames.medianMs, candidate.summary.frames.medianMs),
    baselineP10Ms: baseline.summary.frames.p10Ms, candidateP10Ms: candidate.summary.frames.p10Ms,
    baselineP90Ms: baseline.summary.frames.p90Ms, candidateP90Ms: candidate.summary.frames.p90Ms,
    baselineFrames: baseline.summary.frames.count, candidateFrames: candidate.summary.frames.count,
  },
  worker: {
    baselineMedianMs: baseline.capture.worker?.medianFrame_ms,
    candidateMedianMs: candidate.capture.worker?.medianFrame_ms,
    deltaPercent: deltaPercent(baseline.capture.worker?.medianFrame_ms, candidate.capture.worker?.medianFrame_ms),
    speedup: speedup(baseline.capture.worker?.medianFrame_ms, candidate.capture.worker?.medianFrame_ms),
  },
  visualParity: baselineBenchmark && candidateBenchmark ? {
    resolution: baselineBenchmark.resolution,
    baselineHash: baselineBenchmark.fingerprint.imageHashFnv1a32,
    candidateHash: candidateBenchmark.fingerprint.imageHashFnv1a32,
    exact: baselineBenchmark.fingerprint.imageHashFnv1a32 === candidateBenchmark.fingerprint.imageHashFnv1a32,
    baselineMedianMs: baselineBenchmark.timing.median_ms,
    candidateMedianMs: candidateBenchmark.timing.median_ms,
    timingDeltaPercent: deltaPercent(baselineBenchmark.timing.median_ms, candidateBenchmark.timing.median_ms),
  } : undefined,
  counters: {
    occupancyPointDelta: pointDelta(baseline.summary.counters.meanOccupancy, candidate.summary.counters.meanOccupancy),
    aluPointDelta: pointDelta(baseline.summary.counters.meanAlu, candidate.summary.counters.meanAlu),
  },
  passes,
  warnings,
};

const number = (value: number | undefined, digits = 2): string => value === undefined ? "—" : value.toFixed(digits);
const variantName = (capture: Capture, fallback: string): string =>
  `${capture.variant ?? fallback} (${capture.traversal ?? "unknown traversal"})`;
const markdown = [
  "# SVO render A/B comparison",
  "",
  `Baseline: **${variantName(baseline.capture, "baseline")}**`,
  "",
  `Candidate: **${variantName(candidate.capture, "candidate")}**`,
  "",
  "| Measurement | Baseline | Candidate | Change | Speedup |",
  "| --- | ---: | ---: | ---: | ---: |",
  `| xctrace frame median | ${number(comparison.frame.baselineMedianMs)} ms | ${number(comparison.frame.candidateMedianMs)} ms | ${number(comparison.frame.deltaPercent)}% | ${number(comparison.frame.speedup)}x |`,
  `| submit/fence median | ${number(comparison.worker.baselineMedianMs)} ms | ${number(comparison.worker.candidateMedianMs)} ms | ${number(comparison.worker.deltaPercent)}% | ${number(comparison.worker.speedup)}x |`,
  ...(comparison.visualParity ? [
    `| visual-parity benchmark | ${number(comparison.visualParity.baselineMedianMs)} ms | ${number(comparison.visualParity.candidateMedianMs)} ms | ${number(comparison.visualParity.timingDeltaPercent)}% | — |`,
  ] : []),
  ...passes.map((pass) => `| ${pass.label} | ${number(pass.baselineMs)} ms | ${number(pass.candidateMs)} ms | ${number(pass.deltaPercent)}% | ${number(pass.speedup)}x |`),
  "",
  "| Pass | Occupancy change | ALU change | Read GB/s | Write GB/s | Counter samples |",
  "| --- | ---: | ---: | ---: | ---: | ---: |",
  ...passes.map((pass) => `| ${pass.label} | ${number(pass.occupancyPointDelta)} pp | ${number(pass.aluPointDelta)} pp | ${number(pass.baselineReadGBs)} → ${number(pass.candidateReadGBs)} | ${number(pass.baselineWriteGBs)} → ${number(pass.candidateWriteGBs)} | ${pass.baselineCounterSamples} → ${pass.candidateCounterSamples} |`),
  "",
  ...(comparison.visualParity ? [
    "## Visual parity",
    "",
    `At ${comparison.visualParity.resolution.width}×${comparison.visualParity.resolution.height}: ${comparison.visualParity.baselineHash} → ${comparison.visualParity.candidateHash} (**${comparison.visualParity.exact ? "exact" : "DIFFERENT"}**).`,
    "",
  ] : []),
  ...(warnings.length > 0 ? ["## Warnings", "", ...warnings.map((warning) => `- ${warning}`), ""] : []),
  comparison.visualParity
    ? "The hash comparison is keyed to this campaign's exact resolution; the benchmark's separate built-in 1280×720 reference is not used."
    : "Timing equality does not prove visual equality. Run the dry-frame benchmark for both variants and compare its reference fingerprint and captured PNG before accepting a change.",
  "",
].join("\n");

const output = flag("out");
if (output) {
  const outputDirectory = resolve(root, output);
  mkdirSync(outputDirectory, { recursive: true });
  writeFileSync(resolve(outputDirectory, "comparison.json"), `${JSON.stringify(comparison, null, 2)}\n`);
  writeFileSync(resolve(outputDirectory, "comparison.md"), markdown);
}
process.stdout.write(markdown);
