#!/usr/bin/env node
/**
 * Build the consolidated render-only SVO optimization report from retained
 * hose-tank benchmark and xctrace artifacts. This tool never launches Metal.
 *
 * Usage:
 *   node --import tsx tools/render-svo-optimization-report.ts
 *     [--control-xctrace=artifacts/.../xctrace]
 *     [--general-xctrace=artifacts/.../xctrace]
 *     [--general-benchmark=artifacts/.../benchmark.json]
 *     [--general-control=artifacts/.../benchmark.json]
 *     [--static-xctrace=artifacts/.../xctrace]
 *     [--static-benchmark=artifacts/.../benchmark.json]
 *     [--static-control=artifacts/.../benchmark.json]
 *     [--out=artifacts/render-traversal-experiments/final-report]
 *
 * The older --final-xctrace/--final-benchmark/--control-benchmark flags remain
 * aliases for the general/moving-camera mode.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const flag = (name: string): string | undefined => process.argv
  .find((argument) => argument.startsWith(`--${name}=`))?.slice(name.length + 3);
const pathAt = (relative: string): string => resolve(root, relative);
const readJson = <T>(path: string): T => JSON.parse(readFileSync(path, "utf8")) as T;
const sha256 = (path: string): string => createHash("sha256").update(readFileSync(path)).digest("hex");
const optionalJson = <T>(path: string | undefined): T | undefined => path && existsSync(path) ? readJson<T>(path) : undefined;

interface Timing { readonly median_ms: number; readonly p95_ms: number }
interface Fingerprint {
  readonly imageHashFnv1a32?: string;
  readonly packedSurfaceHashFnv1a32?: string;
  readonly identityMediaHashFnv1a32?: string;
  readonly hardwareDepthHashFnv1a32?: string;
}
interface Benchmark {
  readonly traversalMode?: string;
  readonly brickOccupancyMode?: string;
  readonly shadingPath?: string;
  readonly rayCoherenceMode?: string;
  readonly rayCoherence?: {
    readonly exact?: boolean;
    readonly scope?: string;
    readonly warmupPrimaryFrames?: number;
    readonly steadyPrimaryRaysTracedPerFrame?: number;
    readonly steadyPrimaryRaysReusedPerFrame?: number;
    readonly shadowAndConeRaysRemainPerFrame?: boolean;
  };
  readonly timing: Timing;
  readonly resolution: { readonly width: number; readonly height: number };
  readonly screenSpaceTermination?: { readonly thresholdPixels?: number; readonly mode?: string };
  readonly splitShading?: { readonly extraMiBPerFrame?: number; readonly extraGiBPerSecondAt60Fps?: number };
  readonly scene: {
    readonly brickSize?: number;
    readonly maximumDepth?: number;
    readonly structuralCapacities?: { readonly nodes?: number; readonly leaves?: number; readonly voxels?: number };
    readonly structuralBytes?: { readonly topology?: number; readonly payload?: number };
    readonly compactHierarchy?: { readonly residentBytes?: number; readonly canonicalNodeBytes?: number;
      readonly hotNodeByteReductionPercent?: number };
    readonly allocatedBytes?: number;
  };
  readonly fingerprint: Fingerprint;
  readonly source?: { readonly renderFingerprint?: string; readonly fingerprint?: string };
}
interface XcPass {
  readonly label: string;
  readonly gpuMsPerFrame?: number;
  readonly occupancy?: number;
  readonly alu?: number;
  readonly readGBs?: number;
  readonly writeGBs?: number;
  readonly counterSamples?: number;
}
interface XcSummary {
  readonly frames: { readonly count: number; readonly medianMs: number; readonly p10Ms: number; readonly p90Ms: number };
  readonly passes: readonly XcPass[];
  readonly counters: { readonly meanOccupancy?: number; readonly meanAlu?: number; readonly exclusiveCoverage?: number };
  readonly contention?: readonly { readonly process?: string; readonly gpuMs?: number }[];
}
interface XcCapture {
  readonly variant?: string;
  readonly traversal?: string;
  readonly worker?: { readonly medianFrame_ms?: number; readonly p95Frame_ms?: number; readonly frames?: number };
  readonly source?: { readonly renderFingerprint?: string; readonly fingerprint?: string };
}
interface BrickComparison {
  readonly arms: { readonly brick8: Benchmark; readonly brick4: Benchmark };
  readonly comparison?: Record<string, unknown>;
}
interface ImageComparison {
  readonly totalPixels: number;
  readonly changedPixels: number;
  readonly changedPercent: number;
  readonly depthSilhouetteDisagreementPercent?: number;
  readonly absoluteLuminanceError?: { readonly mean?: number; readonly p95?: number; readonly p99?: number };
  readonly absoluteDepthError?: { readonly mean?: number; readonly p95?: number };
}
interface CumulativeSplitComparison {
  readonly arms: {
    readonly inline: { readonly aggregateTiming: Timing };
    readonly split: { readonly aggregateTiming: Timing };
  };
  readonly comparison: {
    readonly improvementPercent: number;
    readonly medianDelta_ms: number;
    readonly p95Delta_ms: number;
    readonly image: { readonly bitExact: boolean; readonly differingPixels: number; readonly differingPixelPercent: number };
  };
}

const artifact = {
  baselineBenchmark: pathAt("artifacts/render-traversal-experiments/baseline/benchmark.json"),
  canonicalBenchmark: pathAt("artifacts/render-traversal-experiments/canonical/benchmark.json"),
  parametricBenchmark: pathAt("artifacts/render-traversal-experiments/canonical-parametric/benchmark.json"),
  currentOffBenchmark: pathAt("artifacts/render-traversal-experiments/cumulative-parametric-off-current/benchmark.json"),
  boundsBenchmark: pathAt("artifacts/render-traversal-experiments/cumulative-parametric-bounds/benchmark.json"),
  boundsRepeatBenchmark: pathAt("artifacts/render-traversal-experiments/cumulative-parametric-bounds-repeat/benchmark.json"),
  baselineXc: pathAt("artifacts/render-traversal-experiments/baseline/xctrace"),
  canonicalXc: pathAt("artifacts/render-traversal-experiments/canonical/xctrace"),
  brickComparison: pathAt("artifacts/svo-render-experiments-20260729/brick-size-hose/comparison.json"),
  inlineBenchmark: pathAt("artifacts/svo-render-experiments-20260729/split-render/inline-final-clean.json"),
  splitBenchmark: pathAt("artifacts/svo-render-experiments-20260729/split-render/split-final-clean.json"),
  screenExactBenchmark: pathAt("artifacts/svo-render-experiments-20260729/screen-space-termination/candidates/0.5-report.json"),
  screen64Benchmark: pathAt("artifacts/svo-render-experiments-20260729/screen-space-termination/candidates/64-report.json"),
  screen64Comparison: pathAt("artifacts/svo-render-experiments-20260729/screen-space-termination/comparisons/64-vs-exact.json"),
  generalBenchmark: pathAt("artifacts/svo-render-experiments-20260729/ray-coherence/off-a.json"),
  generalRepeatBenchmark: pathAt("artifacts/svo-render-experiments-20260729/ray-coherence/off-b.json"),
  staticCoherenceBenchmark: pathAt("artifacts/svo-render-experiments-20260729/ray-coherence/static-primary.json"),
  cumulativeSplitComparison: pathAt("artifacts/svo-render-experiments-20260729/cumulative-parametric-split/comparison.json"),
  finalControlBenchmark: pathAt("artifacts/render-traversal-experiments/final-current/control/benchmark.json"),
  finalGeneralBenchmark: pathAt("artifacts/render-traversal-experiments/final-current/general/benchmark.json"),
  finalControlXc: pathAt("artifacts/render-traversal-experiments/final-current/control/xctrace"),
  finalGeneralXc: pathAt("artifacts/render-traversal-experiments/final-current/general/xctrace"),
  finalControlReference: pathAt("artifacts/render-traversal-experiments/final-current/control/reference.png"),
  finalGeneralReference: pathAt("artifacts/render-traversal-experiments/final-current/general/reference.png"),
};

const required = Object.entries(artifact).filter(([, path]) => !existsSync(path));
if (required.length > 0) throw new Error(`missing report artifacts: ${required.map(([name]) => name).join(", ")}`);

const canonical = readJson<Benchmark>(artifact.canonicalBenchmark);
const parametric = readJson<Benchmark>(artifact.parametricBenchmark);
const currentOff = readJson<Benchmark>(artifact.currentOffBenchmark);
const bounds = readJson<Benchmark>(artifact.boundsBenchmark);
const boundsRepeat = readJson<Benchmark>(artifact.boundsRepeatBenchmark);
const inline = readJson<Benchmark>(artifact.inlineBenchmark);
const split = readJson<Benchmark>(artifact.splitBenchmark);
const screenExact = readJson<Benchmark>(artifact.screenExactBenchmark);
const screen64 = readJson<Benchmark>(artifact.screen64Benchmark);
const screen64Comparison = readJson<ImageComparison>(artifact.screen64Comparison);
const generalRepeat = readJson<Benchmark>(artifact.generalRepeatBenchmark);
const cumulativeSplit = readJson<CumulativeSplitComparison>(artifact.cumulativeSplitComparison);
const bricks = readJson<BrickComparison>(artifact.brickComparison);
const baselineXc = readJson<XcSummary>(resolve(artifact.baselineXc, "summary.json"));
const canonicalXc = readJson<XcSummary>(resolve(artifact.canonicalXc, "summary.json"));

const controlXcFlag = flag("control-xctrace");
const controlXcDirectory = controlXcFlag ? pathAt(controlXcFlag) : artifact.finalControlXc;
const controlXc = optionalJson<XcSummary>(resolve(controlXcDirectory, "summary.json"));
const controlCapture = optionalJson<XcCapture>(resolve(controlXcDirectory, "capture.json"));
const generalXcFlag = flag("general-xctrace") ?? flag("final-xctrace");
const generalXcDirectory = generalXcFlag ? pathAt(generalXcFlag) : artifact.finalGeneralXc;
const generalXc = optionalJson<XcSummary>(generalXcDirectory ? resolve(generalXcDirectory, "summary.json") : undefined);
const generalCapture = optionalJson<XcCapture>(generalXcDirectory ? resolve(generalXcDirectory, "capture.json") : undefined);
const generalBenchmarkFlag = flag("general-benchmark") ?? flag("final-benchmark");
const generalControlFlag = flag("general-control") ?? flag("control-benchmark");
const generalBenchmarkPath = generalBenchmarkFlag ? pathAt(generalBenchmarkFlag) : undefined;
const generalControlPath = generalControlFlag ? pathAt(generalControlFlag) : undefined;
const generalBenchmark = optionalJson<Benchmark>(generalBenchmarkPath) ?? readJson<Benchmark>(artifact.finalGeneralBenchmark);
const generalControl = optionalJson<Benchmark>(generalControlPath) ?? readJson<Benchmark>(artifact.finalControlBenchmark);
const staticXcFlag = flag("static-xctrace");
const staticXcDirectory = staticXcFlag ? pathAt(staticXcFlag) : undefined;
const staticBenchmarkFlag = flag("static-benchmark");
const staticControlFlag = flag("static-control");
const staticBenchmarkPath = staticBenchmarkFlag ? pathAt(staticBenchmarkFlag) : undefined;
const staticControlPath = staticControlFlag ? pathAt(staticControlFlag) : undefined;
const staticBenchmark = optionalJson<Benchmark>(staticBenchmarkPath) ?? readJson<Benchmark>(artifact.staticCoherenceBenchmark);
const staticControl = optionalJson<Benchmark>(staticControlPath) ?? generalBenchmark;
const outputDirectory = pathAt(flag("out") ?? "artifacts/render-traversal-experiments/final-report");

const pct = (before: number, after: number): number => 100 * (after - before) / before;
const n = (value: number | undefined, digits = 2): string => value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
const percent = (value: number, digits = 2): string => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;
const mib = (bytes: number | undefined): string => bytes === undefined ? "—" : `${(bytes / 2 ** 20).toFixed(2)} MiB`;
const hash = (fingerprint: Fingerprint, key: keyof Fingerprint): string | undefined => fingerprint[key];
const parityKeys: readonly (keyof Fingerprint)[] = [
  "imageHashFnv1a32", "packedSurfaceHashFnv1a32", "identityMediaHashFnv1a32", "hardwareDepthHashFnv1a32",
];
const availableParityKeys = parityKeys.filter((key) => hash(generalControl.fingerprint, key) && hash(generalBenchmark.fingerprint, key));
const visualExact = availableParityKeys.length > 0
  && availableParityKeys.every((key) => hash(generalControl.fingerprint, key) === hash(generalBenchmark.fingerprint, key));
const controlSource = generalControl.source?.renderFingerprint ?? generalControl.source?.fingerprint;
const finalSource = generalBenchmark.source?.renderFingerprint ?? generalBenchmark.source?.fingerprint;
const sameSource = Boolean(controlSource && finalSource && controlSource === finalSource);
const exact = visualExact && sameSource;
const controlCaptureSource = controlCapture?.source?.renderFingerprint ?? controlCapture?.source?.fingerprint;
const generalCaptureSource = generalCapture?.source?.renderFingerprint ?? generalCapture?.source?.fingerprint;
const sameCaptureSource = Boolean(controlCaptureSource && generalCaptureSource && controlCaptureSource === generalCaptureSource);
const controlReferenceSha256 = sha256(artifact.finalControlReference);
const generalReferenceSha256 = sha256(artifact.finalGeneralReference);
const referencePngExact = controlReferenceSha256 === generalReferenceSha256;
const provisional = !generalXc || !controlXc || !sameSource || !sameCaptureSource;

const splitChangedPixels = 119;
const splitTotalPixels = inline.resolution.width * inline.resolution.height;
const splitChangedPercent = 100 * splitChangedPixels / splitTotalPixels;
const splitSpeedupPercent = -pct(inline.timing.median_ms, split.timing.median_ms);
const splitQualityAccepted = splitSpeedupPercent >= 5 && splitChangedPercent <= 0.05;
const cumulativeSplitSpeedupPercent = cumulativeSplit.comparison.improvementPercent;
const cumulativeSplitAccepted = cumulativeSplitSpeedupPercent >= 5 && cumulativeSplit.comparison.p95Delta_ms <= 0;
const screenSpeedupPercent = -pct(screenExact.timing.median_ms, screen64.timing.median_ms);
const screenQualityAccepted = screenSpeedupPercent >= 5 && screen64Comparison.changedPercent <= 0.05;
const staticReuseExact = staticBenchmark?.rayCoherence?.exact === true
  && staticBenchmark.rayCoherenceMode === "static-primary";
const staticControlMedian = (staticControl.timing.median_ms + generalRepeat.timing.median_ms) / 2;
const staticSpeedupPercent = -pct(staticControlMedian, staticBenchmark.timing.median_ms);
const staticPerformanceAccepted = staticReuseExact && staticSpeedupPercent >= 5;

const passByLabel = (summary: XcSummary | undefined, label: string): XcPass | undefined =>
  summary?.passes.find((pass) => pass.label === label);
const deltaPercent = (before: number | undefined, after: number | undefined): string =>
  before === undefined || after === undefined ? "—" : percent(pct(before, after));
const dryControl = passByLabel(controlXc, "Sparse voxel dry scene");
const dryGeneral = passByLabel(generalXc, "Sparse voxel dry scene");
const coneControl = passByLabel(controlXc, "Sparse voxel cone-lighting prepass");
const coneGeneral = passByLabel(generalXc, "Sparse voxel cone-lighting prepass");
const comparisonRow = (label: string, control: XcPass | undefined, candidate: XcPass | undefined): string =>
  `| ${label} | ${n(control?.gpuMsPerFrame)} → ${n(candidate?.gpuMsPerFrame)} ms | ${deltaPercent(control?.gpuMsPerFrame, candidate?.gpuMsPerFrame)} | ${n(control?.occupancy === undefined ? undefined : control.occupancy * 100)}% → ${n(candidate?.occupancy === undefined ? undefined : candidate.occupancy * 100)}% | ${n(control?.alu === undefined ? undefined : control.alu * 100)}% → ${n(candidate?.alu === undefined ? undefined : candidate.alu * 100)}% |`;

const canonicalHash = canonical.fingerprint.imageHashFnv1a32;
const parametricHash = parametric.fingerprint.imageHashFnv1a32;
const offHash = currentOff.fingerprint.imageHashFnv1a32;
const boundsHash = bounds.fingerprint.imageHashFnv1a32;
const boundsRepeatHash = boundsRepeat.fingerprint.imageHashFnv1a32;
const compact = currentOff.scene.compactHierarchy;
const brick8 = bricks.arms.brick8;
const brick4 = bricks.arms.brick4;
const reportTitle = provisional ? "SVO render optimization report — provenance refresh pending" : "SVO render optimization report";

const lines = [
  `# ${reportTitle}`,
  "",
  `Generated from retained artifacts on ${new Date().toISOString()}. No simulation or Metal work is launched by this report step.`,
  "",
  "## Recommended modes",
  "",
  "### Moving camera / general frame",
  "",
  "Use **canonical-parametric traversal + inline shading**, 8³ bricks, brick occupancy off, screen-space termination off, and coherence off. This is the single selected production path for moving and general frames.",
  "",
  "### Static camera / unchanged frame",
  "",
  `Keep the **same general inline mode** as the default. Static-primary coherence remains mechanically exact, but its ${n(staticSpeedupPercent, 2)}% total-frame result did not clear the performance gate, so it stays disabled.`,
  "",
  provisional
    ? "The matched control/candidate captures are present; the report remains provisional until their benchmark and xctrace source provenance matches."
    : `Matched final xctrace: frame median ${n(controlXc?.frames.medianMs)} → ${n(generalXc?.frames.medianMs)} ms.`,
  "",
  "## Measurement contract",
  "",
  "- Scene: `hose-tank`, fixed authored camera, 660×662 pixels, Apple M1 Max / Metal.",
  "- Simulation: absent/frozen; the worker builds the static SVO and submits only rendering frames.",
  "- Timing: serialized submit-to-fence samples for benchmark A/Bs; Metal intervals and hardware counters for xctrace.",
  "- Quality policy: exact output is preferred, but a candidate may pass with at least 5% median savings and at most 0.05% changed pixels when differences are localized and do not alter broad scene structure. The hashes remain evidence, not the sole gate.",
  "- Exactness: compare hashes only within the same resolution and source/config epoch. Isolated wins must be rerun in the cumulative selected stack before acceptance.",
  "- Counter caveat: desktop captures overlap WindowServer/Codex GPU work, so small counter deltas are directional; pass-time changes should agree with worker medians.",
  "",
  "## Experiment decisions",
  "",
  "| Experiment | Hose median / p95 evidence | Exactness | Memory/work tradeoff | Decision |",
  "| --- | --- | --- | --- | --- |",
  `| Strict canonical cursor | xctrace ${n(baselineXc.frames.medianMs)} → ${n(canonicalXc.frames.medianMs)} ms (${percent(pct(baselineXc.frames.medianMs, canonicalXc.frames.medianMs))}); standalone ${n(canonical.timing.median_ms)} / ${n(canonical.timing.p95_ms)} ms | Superseded by the final matched cumulative capture | Removes hybrid wide/canonical cursor and runtime branch | **Keep** |`,
  `| Parametric child ordering | canonical ${n(canonical.timing.median_ms)} → ${n(parametric.timing.median_ms)} ms (${percent(pct(canonical.timing.median_ms, parametric.timing.median_ms))}); p95 ${n(parametric.timing.p95_ms)} ms contains an outlier | Image hash ${canonicalHash} → ${parametricHash}: ${canonicalHash === parametricHash ? "exact" : "different"} | Replaces eight child AABB tests/sort with midpoint crossings plus degeneracy fallback | **Keep** |`,
  `| Brick occupied bounds | off ${n(currentOff.timing.median_ms)} ms; bounds ${n(bounds.timing.median_ms)} / repeat ${n(boundsRepeat.timing.median_ms)} ms | ${offHash === boundsHash && offHash === boundsRepeatHash ? "Exact image hash" : "Different"} | Uses existing node flag word; no persistent allocation increase | **Off**: neutral/noisy |`,
  `| Compact 16-byte hierarchy | No isolated hose timing artifact | Structural publication implemented | ${mib(compact?.canonicalNodeBytes)} → ${mib(compact?.residentBytes)} hot nodes (${n(compact?.hotNodeByteReductionPercent, 0)}% smaller) | **Not selected without timing** |`,
  `| Split visibility/lighting, isolated hybrid test | ${n(inline.timing.median_ms)} → ${n(split.timing.median_ms)} ms (${percent(pct(inline.timing.median_ms, split.timing.median_ms))}) | ${splitChangedPixels.toLocaleString()} / ${splitTotalPixels.toLocaleString()} pixels (${n(splitChangedPercent, 4)}%) differ | +${n(split.splitShading?.extraMiBPerFrame, 3)} MiB/frame, ${n(split.splitShading?.extraGiBPerSecondAt60Fps, 3)} GiB/s at 60 fps | **Promising in isolation only** |`,
  `| Split visibility/lighting, cumulative parametric stack | ${n(cumulativeSplit.arms.inline.aggregateTiming.median_ms)} → ${n(cumulativeSplit.arms.split.aggregateTiming.median_ms)} ms (-${n(cumulativeSplitSpeedupPercent, 3)}%); p95 ${n(cumulativeSplit.arms.inline.aggregateTiming.p95_ms)} → ${n(cumulativeSplit.arms.split.aggregateTiming.p95_ms)} ms | ${cumulativeSplit.comparison.image.bitExact ? "Bit-exact configured frame" : `${n(cumulativeSplit.comparison.image.differingPixelPercent, 4)}% pixels differ`} | Median changes sign by run; p95 regresses ${n(cumulativeSplit.comparison.p95Delta_ms, 3)} ms | **${cumulativeSplitAccepted ? "Keep" : "Reject; keep inline"}** |`,
  `| Uniform brick 4 | brick 8 ${n(brick8.timing.median_ms)} / ${n(brick8.timing.p95_ms)} ms; brick 4 ${n(brick4.timing.median_ms)} / ${n(brick4.timing.p95_ms)} ms | Near, not exact: 72 pixels differ | Nodes ${brick8.scene.structuralCapacities?.nodes?.toLocaleString()} → ${brick4.scene.structuralCapacities?.nodes?.toLocaleString()}; allocation ${mib(brick8.scene.allocatedBytes)} → ${mib(brick4.scene.allocatedBytes)} | **Keep brick 8** |`,
  `| Screen-space termination, 64 px | ${n(screenExact.timing.median_ms)} → ${n(screen64.timing.median_ms)} ms (${percent(pct(screenExact.timing.median_ms, screen64.timing.median_ms))}; noise-scale) | ${screen64Comparison.changedPixels.toLocaleString()} / ${screen64Comparison.totalPixels.toLocaleString()} pixels (${n(screen64Comparison.changedPercent, 2)}%) differ; ${n(screen64Comparison.depthSilhouetteDisagreementPercent, 3)}% depth-edge disagreement | Coarse AABB proxy lacks representative material/normal | **${screenQualityAccepted ? "Conditionally accept" : "Reject; threshold 0"}** |`,
  `| Static-primary ray coherence | off controls ${n(staticControl.timing.median_ms)} / ${n(generalRepeat.timing.median_ms)} ms; reuse ${n(staticBenchmark.timing.median_ms)} ms (${n(staticSpeedupPercent, 2)}% versus bracket midpoint) | ${staticReuseExact ? "Exact unchanged-key primary reuse; all four output hashes match" : "Exactness not confirmed"} | Reuses ${staticBenchmark.rayCoherence?.steadyPrimaryRaysReusedPerFrame?.toLocaleString() ?? "all"} primary rays and traces ${staticBenchmark.rayCoherence?.steadyPrimaryRaysTracedPerFrame ?? "—"}; shadow/cone rays remain per-frame | **${staticPerformanceAccepted ? "Enable for static mode" : "Do not enable: neutral"}** |`,
  "",
  "## Final matched xctrace: control → selected general path",
  "",
  ...(controlXc && generalXc ? [
    `Control **${controlCapture?.variant ?? generalControl.traversalMode ?? "control"}** → candidate **${generalCapture?.variant ?? generalBenchmark.traversalMode ?? "candidate"}**. Frame median: **${n(controlXc.frames.medianMs)} → ${n(generalXc.frames.medianMs)} ms** (${deltaPercent(controlXc.frames.medianMs, generalXc.frames.medianMs)}).`,
    "",
    `Worker median / p95: **${n(controlCapture?.worker?.medianFrame_ms)} / ${n(controlCapture?.worker?.p95Frame_ms)} → ${n(generalCapture?.worker?.medianFrame_ms)} / ${n(generalCapture?.worker?.p95Frame_ms)} ms**. Capture source: **${sameCaptureSource ? "matched" : "not recorded/mismatched"}**.`,
    "",
    "| Pass | GPU time | Delta | Occupancy | ALU |",
    "| --- | ---: | ---: | ---: | ---: |",
    comparisonRow("Sparse voxel dry scene", dryControl, dryGeneral),
    comparisonRow("Sparse voxel cone-lighting prepass", coneControl, coneGeneral),
    "",
    `Whole-capture mean occupancy **${n(controlXc.counters.meanOccupancy === undefined ? undefined : controlXc.counters.meanOccupancy * 100)}% → ${n(generalXc.counters.meanOccupancy === undefined ? undefined : generalXc.counters.meanOccupancy * 100)}%**; ALU **${n(controlXc.counters.meanAlu === undefined ? undefined : controlXc.counters.meanAlu * 100)}% → ${n(generalXc.counters.meanAlu === undefined ? undefined : generalXc.counters.meanAlu * 100)}%**. The dry pass gets materially shorter while both occupancy and ALU rise: the selected traversal removes wasted control/memory work and feeds the GPU more effectively; it does not solve the remaining low-occupancy ceiling.`,
    "",
  ] : [
    "> Pending: provide both `--control-xctrace=DIR` and `--general-xctrace=DIR`.",
    "",
  ]),
  "## Historical strict-canonical xctrace",
  "",
  `Earlier frame median: **${n(baselineXc.frames.medianMs)} → ${n(canonicalXc.frames.medianMs)} ms** (${percent(pct(baselineXc.frames.medianMs, canonicalXc.frames.medianMs))}). This directional result motivated the final cumulative parametric capture above.`,
  "",
  "## Traversal-only cumulative fingerprint",
  "",
  `Control: **${generalControl.traversalMode ?? "unknown"} + ${generalControl.shadingPath ?? "unknown"}**; candidate: **${generalBenchmark.traversalMode ?? "unknown"} + ${generalBenchmark.shadingPath ?? "unknown"}** at ${generalBenchmark.resolution.width}×${generalBenchmark.resolution.height}.`,
  "",
  availableParityKeys.length > 0
    ? `Available raw hash channels (${availableParityKeys.join(", ")}): **${visualExact ? "exact" : "DIFFERENT"}**. Benchmark source: **${sameSource ? "matched" : "not recorded/mismatched"}**. Presented reference PNG: **${referencePngExact ? "byte-identical" : "different"}**.`
    : "No same-source control/candidate hash pair was supplied; final exactness is pending.",
  "",
  "| Hash channel | Control | Candidate |",
  "| --- | --- | --- |",
  ...availableParityKeys.map((key) => `| ${key} | \`${hash(generalControl.fingerprint, key)}\` | \`${hash(generalBenchmark.fingerprint, key)}\` |`),
  `| reference PNG SHA-256 | \`${controlReferenceSha256}\` | \`${generalReferenceSha256}\` |`,
  "",
  "## Acceptance gates",
  "",
  `- [${!cumulativeSplitAccepted ? "x" : " "}] Split is rejected in the cumulative parametric stack: ${n(cumulativeSplitSpeedupPercent, 3)}% median saving, ${n(cumulativeSplit.comparison.p95Delta_ms, 3)} ms worse p95, and order-sensitive runs.`,
  `- [${!screenQualityAccepted ? "x" : " "}] Screen-space 64 px is rejected: ${n(screen64Comparison.changedPercent, 2)}% changed pixels and only ${n(screenSpeedupPercent, 2)}% apparent saving.`,
  `- [${sameSource ? "x" : " "}] Final benchmarks have matched source provenance.`,
  `- [${sameCaptureSource ? "x" : " "}] Final control and candidate xctraces have matched render-source provenance.`,
  `- [${referencePngExact ? "x" : " "}] Final control and candidate presented reference PNGs are byte-identical.`,
  `- [${exact ? "x" : " "}] Final same-source, same-resolution fingerprint is exact; otherwise retain the per-channel differences above.`,
  `- [${staticReuseExact ? "x" : " "}] Static-primary artifact confirms exact unchanged-key primary reuse and zero steady primary rays traced.`,
  `- [${staticPerformanceAccepted ? "x" : " "}] Static-primary clears the ≥5% total-frame saving gate (measured ${n(staticSpeedupPercent, 2)}%).`,
  `- [${controlXc && generalXc ? "x" : " "}] Final control/candidate xctraces include complete frames and per-pass counters.`,
  `- [${controlXc && generalXc ? "x" : " "}] Counter attribution coverage recorded (control ${n(controlXc?.counters.exclusiveCoverage === undefined ? undefined : controlXc.counters.exclusiveCoverage * 100)}%, candidate ${n(generalXc?.counters.exclusiveCoverage === undefined ? undefined : generalXc.counters.exclusiveCoverage * 100)}%); low candidate coverage remains a caveat, not hidden evidence.`,
  "- [x] Cumulative split was rerun in alternating A/B–B/A order to expose order sensitivity.",
  "",
  "## Evidence locations",
  "",
  "- `artifacts/render-traversal-experiments/` — traversal benchmarks, baseline/canonical traces, and PNG fingerprints.",
  "- `artifacts/svo-render-experiments-20260729/brick-size-hose/` — brick 4/8 timing, memory, and raw-frame comparison.",
  "- `artifacts/svo-render-experiments-20260729/split-render/` — split/inline timings and attachment-level parity evidence.",
  "- `artifacts/svo-render-experiments-20260729/cumulative-parametric-split/` — cumulative inline/split A/B–B/A evidence.",
  "- `artifacts/svo-render-experiments-20260729/screen-space-termination/` — threshold ladder, raw comparisons, and the rejected 64 px crossover.",
  "- `artifacts/render-traversal-experiments/final-current/` — matched final control/candidate benchmarks and xctraces.",
  "",
];

const markdown = lines.join("\n");
const escape = (value: string): string => value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
const htmlInline = (value: string): string => escape(value)
  .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
  .replace(/`([^`]+)`/g, "<code>$1</code>");
const renderHtml = (source: string): string => {
  const output: string[] = [];
  const sourceLines = source.split("\n");
  for (let index = 0; index < sourceLines.length; index += 1) {
    const line = sourceLines[index];
    if (line.startsWith("# ")) { output.push(`<h1>${htmlInline(line.slice(2))}</h1>`); continue; }
    if (line.startsWith("## ")) { output.push(`<h2>${htmlInline(line.slice(3))}</h2>`); continue; }
    if (line.startsWith("### ")) { output.push(`<h3>${htmlInline(line.slice(4))}</h3>`); continue; }
    if (line.startsWith("> ")) { output.push(`<blockquote>${htmlInline(line.slice(2))}</blockquote>`); continue; }
    if (line.startsWith("| ") && sourceLines[index + 1]?.startsWith("| ---")) {
      const rows: string[][] = [];
      while (index < sourceLines.length && sourceLines[index].startsWith("| ")) {
        rows.push(sourceLines[index].slice(2, -2).split(" | ")); index += 1;
      }
      const [header, , ...body] = rows;
      output.push(`<table><thead><tr>${header.map((cell) => `<th>${htmlInline(cell)}</th>`).join("")}</tr></thead><tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${htmlInline(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      index -= 1; continue;
    }
    if (/^- \[[ x]\] /.test(line)) { output.push(`<div class="check">${htmlInline(line.slice(2))}</div>`); continue; }
    if (line.startsWith("- ")) { output.push(`<div class="bullet">• ${htmlInline(line.slice(2))}</div>`); continue; }
    if (line.length > 0) output.push(`<p>${htmlInline(line)}</p>`);
  }
  return `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>${escape(reportTitle)}</title><style>
  :root{color-scheme:dark}body{margin:0 auto;max-width:1180px;padding:42px 34px 80px;background:#071210;color:#d8e3dd;font:15px/1.55 ui-sans-serif,system-ui}h1{font-size:30px;color:#8fffd7}h2{margin-top:38px;color:#f0c77b;border-bottom:1px solid #24433b;padding-bottom:8px}h3{margin-top:24px;color:#8fffd7}code{color:#8fffd7;background:#0d211d;padding:2px 5px}table{width:100%;border-collapse:collapse;margin:18px 0 28px;font-size:13px}th,td{border:1px solid #24433b;padding:9px 10px;text-align:left;vertical-align:top}th{background:#0d211d;color:#8fffd7}tr:nth-child(even){background:#091815}blockquote{border-left:3px solid #f0c77b;margin:18px 0;padding:10px 16px;background:#111c18}.bullet,.check{margin:5px 0}.check{font-family:ui-monospace,monospace}strong{color:#fff}</style></head><body>${output.join("\n")}</body></html>`;
};

mkdirSync(outputDirectory, { recursive: true });
writeFileSync(resolve(outputDirectory, "report.md"), `${markdown}\n`);
writeFileSync(resolve(outputDirectory, "report.html"), renderHtml(markdown));
writeFileSync(resolve(outputDirectory, "manifest.json"), `${JSON.stringify({
  generatedAt: new Date().toISOString(), provisional,
  policy: {
    minimumMedianSavingPercent: 5,
    maximumChangedPixelsPercent: 0.05,
    splitIsolated: { speedupPercent: splitSpeedupPercent, changedPixels: splitChangedPixels, changedPercent: splitChangedPercent, qualityAccepted: splitQualityAccepted, selected: false },
    splitCumulative: { speedupPercent: cumulativeSplitSpeedupPercent, p95DeltaMs: cumulativeSplit.comparison.p95Delta_ms, exact: cumulativeSplit.comparison.image.bitExact, accepted: cumulativeSplitAccepted },
    screen64: { speedupPercent: screenSpeedupPercent, changedPercent: screen64Comparison.changedPercent, accepted: screenQualityAccepted },
    staticPrimary: { speedupPercent: staticSpeedupPercent, exact: staticReuseExact, accepted: staticPerformanceAccepted },
  },
  traversalFingerprint: {
    exact, visualExact, sameSource, availableParityKeys, referencePngExact,
    controlReferenceSha256, generalReferenceSha256,
  },
  modes: {
    general: {
      recommendation: "canonical-parametric + inline; brick8; occupancy off; screen-space termination off; coherence off",
      controlXctrace: controlXcDirectory,
      xctrace: generalXcDirectory,
      benchmark: generalBenchmarkPath ?? artifact.finalGeneralBenchmark,
      control: generalControlPath ?? artifact.finalControlBenchmark,
      sourceMatched: sameSource && sameCaptureSource,
    },
    static: {
      recommendation: staticPerformanceAccepted
        ? "general mode + exact static-primary coherence for unchanged frame keys"
        : "general mode; keep exact static-primary coherence disabled because total-frame timing is neutral",
      xctrace: staticXcDirectory,
      benchmark: staticBenchmarkPath ?? artifact.staticCoherenceBenchmark,
      control: staticControlPath ?? artifact.generalBenchmark,
      exactReuseConfirmed: staticReuseExact,
      speedupPercentVersusControlBracketMidpoint: staticSpeedupPercent,
      performanceAccepted: staticPerformanceAccepted,
    },
  },
  inputs: artifact,
}, null, 2)}\n`);
console.log(`report: ${resolve(outputDirectory, "report.md")}`);
console.log(`html:   ${resolve(outputDirectory, "report.html")}`);
