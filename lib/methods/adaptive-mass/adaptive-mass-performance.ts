import type { PerformanceTrace } from "../../core/performance-trace";

export const SPARSE_CM12_PERFORMANCE_ACCEPTANCE = Object.freeze({
  sceneId: "symmetric-expansion",
  finestDimensions: [32, 16, 32] as const,
  dt_s: 0.004,
  maximumMedianFrameRatio: 1.30,
  targetMedianFrameRatio: 1.20,
  minimumTimedFrames: 30,
  constructionExcluded: true,
});

export interface SparseCM12PerformanceAcceptance {
  readonly sceneId: string;
  readonly finestDimensions: readonly [number, number, number];
  readonly dt_s: number;
  readonly maximumMedianFrameRatio: number;
  readonly targetMedianFrameRatio: number;
  readonly minimumTimedFrames: number;
  readonly constructionExcluded: boolean;
}

export const SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE = Object.freeze({
  ...SPARSE_CM12_PERFORMANCE_ACCEPTANCE,
  sceneId: "minimal-power-dam-break-32",
  finestDimensions: [32, 32, 32] as const,
  // Rung A protects a moving interface/receiver set, not one historical seam
  // seed. Require useful mixed topology without freezing a scene-specific count.
  minimumInitialFineBrickCount: 1,
  minimumInitialCoarseBrickCount: 1,
  minimumInitialFineCoarseFaceConnectedPairCount: 1,
  minimumInitialMixedSeamRows: 1,
  minimumEvolvedMixedSeamRows: 1,
});

export const SPARSE_CM12_LONG_DAM_PERFORMANCE_ACCEPTANCE = Object.freeze({
  ...SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  sceneId: "sparse-cm12-long-dam-break",
  finestDimensions: [96, 48, 16] as const,
});

export interface SparseCM12TimingSummary {
  readonly samples: number;
  readonly median_ms: number;
  readonly p95_ms: number;
}

export interface SparseCM12BenchmarkArm {
  readonly methodId: "uniform" | "adaptive-mass";
  readonly sceneId: string;
  readonly finestDimensions: readonly [number, number, number];
  readonly dt_s: number;
  readonly constructionExcluded: boolean;
  /** Serialized advance through queue completion, one value per timed frame. */
  readonly endToEndFrame_ms: readonly number[];
  /** CPU authority/encoding samples, kept separate from GPU observations. */
  readonly cpuTraces: readonly PerformanceTrace[];
  /** Hardware or queue-wall samples; never merged with CPU observations. */
  readonly gpuTraces: readonly PerformanceTrace[];
  readonly initialTopology?: SparseCM12TopologySample;
  readonly evolvedTopology?: readonly SparseCM12TopologySample[];
}

export interface SparseCM12TopologySample {
  readonly fineBricks: number;
  readonly coarseBricks: number;
  readonly fineCoarseFaceConnectedPairs: number;
  readonly mixedSeamRows: number;
}

export interface SparseCM12BenchmarkArmSummary {
  readonly methodId: SparseCM12BenchmarkArm["methodId"];
  readonly frame: SparseCM12TimingSummary;
  readonly cpu: SparseCM12TimingSummary | null;
  readonly gpu: SparseCM12TimingSummary | null;
  readonly cpuStages: Readonly<Record<string, SparseCM12TimingSummary>>;
  readonly gpuStages: Readonly<Record<string, SparseCM12TimingSummary>>;
  readonly gpuMeasurementSources: readonly string[];
}

export interface SparseCM12PerformanceVerdict {
  readonly passed: boolean;
  readonly targetMet: boolean;
  readonly medianFrameRatio: number;
  readonly maximumMedianFrameRatio: number;
  readonly targetMedianFrameRatio: number;
  readonly uniform: SparseCM12BenchmarkArmSummary;
  readonly sparse: SparseCM12BenchmarkArmSummary;
  readonly failures: readonly string[];
}

function finiteSamples(values: readonly number[]): number[] {
  return values.filter((value) => Number.isFinite(value) && value >= 0);
}

function quantile(values: readonly number[], fraction: number): number {
  const ordered = finiteSamples(values).sort((left, right) => left - right);
  if (ordered.length === 0) return Number.NaN;
  return ordered[Math.min(
    ordered.length - 1,
    Math.max(0, Math.ceil(fraction * ordered.length) - 1),
  )]!;
}

export function summarizeSparseCM12Timing(
  values: readonly number[],
): SparseCM12TimingSummary {
  const samples = finiteSamples(values);
  return {
    samples: samples.length,
    median_ms: quantile(samples, 0.5),
    p95_ms: quantile(samples, 0.95),
  };
}

function summarizeTraceStages(
  traces: readonly PerformanceTrace[],
): Readonly<Record<string, SparseCM12TimingSummary>> {
  const samples = new Map<string, number[]>();
  for (const trace of traces) {
    for (const phase of trace.phases) {
      const key = `${phase.id}: ${phase.label}`;
      const bucket = samples.get(key) ?? [];
      bucket.push(phase.duration_ms);
      samples.set(key, bucket);
    }
  }
  return Object.fromEntries([...samples.entries()].flatMap(([label, values]) => {
    const summary = summarizeSparseCM12Timing(values);
    return summary.samples > 0 ? [[label, summary] as const] : [];
  }));
}

export function summarizeSparseCM12BenchmarkArm(
  arm: SparseCM12BenchmarkArm,
): SparseCM12BenchmarkArmSummary {
  return {
    methodId: arm.methodId,
    frame: summarizeSparseCM12Timing(arm.endToEndFrame_ms),
    cpu: arm.cpuTraces.length > 0
      ? summarizeSparseCM12Timing(arm.cpuTraces.map((trace) => trace.total_ms))
      : null,
    gpu: arm.gpuTraces.length > 0
      ? summarizeSparseCM12Timing(arm.gpuTraces.map((trace) => trace.total_ms))
      : null,
    cpuStages: summarizeTraceStages(arm.cpuTraces),
    gpuStages: summarizeTraceStages(arm.gpuTraces),
    gpuMeasurementSources: [...new Set(arm.gpuTraces.map((trace) =>
      trace.measurementSource ?? "unspecified"))],
  };
}

/**
 * Hard matched-frame gate. Only serialized end-to-end wall observations are
 * ratioed. Host command-encoding and GPU execution traces remain separate
 * evidence; both methods publish `hostFluidAuthority = gpu-resident` and no
 * simulation-sized host work is permitted. Construction is forbidden from
 * both arms.
 */
export function evaluateSparseCM12Performance(
  uniform: SparseCM12BenchmarkArm,
  sparse: SparseCM12BenchmarkArm,
  acceptance: SparseCM12PerformanceAcceptance = SPARSE_CM12_PERFORMANCE_ACCEPTANCE,
): SparseCM12PerformanceVerdict {
  const failures: string[] = [];
  const expectedDimensions = acceptance.finestDimensions.join("x");
  for (const arm of [uniform, sparse]) {
    if (arm.sceneId !== acceptance.sceneId) {
      failures.push(`${arm.methodId} scene ${arm.sceneId} is not ${acceptance.sceneId}`);
    }
    if (arm.finestDimensions.join("x") !== expectedDimensions) {
      failures.push(`${arm.methodId} finest lattice ${arm.finestDimensions.join("x")} is not ${expectedDimensions}`);
    }
    if (Math.abs(arm.dt_s - acceptance.dt_s) > 1e-12) {
      failures.push(`${arm.methodId} dt ${arm.dt_s} is not ${acceptance.dt_s}`);
    }
    if (!arm.constructionExcluded) {
      failures.push(`${arm.methodId} timing includes one-time construction`);
    }
    if (finiteSamples(arm.endToEndFrame_ms).length < acceptance.minimumTimedFrames) {
      failures.push(`${arm.methodId} has fewer than ${acceptance.minimumTimedFrames} timed frames`);
    }
  }
  if (uniform.methodId !== "uniform" || sparse.methodId !== "adaptive-mass") {
    failures.push("performance arms must be uniform then adaptive-mass");
  }
  if ("minimumInitialFineBrickCount" in acceptance) {
    const mixedAcceptance = acceptance as typeof SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE;
    const initial = sparse.initialTopology;
    if (!initial) {
      failures.push("adaptive-mass did not publish its initial mixed-fineness topology");
    } else {
      if (initial.fineBricks < mixedAcceptance.minimumInitialFineBrickCount) {
        failures.push("adaptive-mass initial topology has no fine brick");
      }
      if (initial.coarseBricks < mixedAcceptance.minimumInitialCoarseBrickCount) {
        failures.push("adaptive-mass initial topology has no coarse brick");
      }
      if (initial.fineCoarseFaceConnectedPairs
        < mixedAcceptance.minimumInitialFineCoarseFaceConnectedPairCount) {
        failures.push("adaptive-mass has no fine brick face-connected to a coarse brick");
      }
      if (initial.mixedSeamRows < mixedAcceptance.minimumInitialMixedSeamRows) {
        failures.push("adaptive-mass initial composite grid has no live mixed seam row");
      }
    }
    const evolved = sparse.evolvedTopology ?? [];
    if (evolved.length === 0 || evolved.some((sample) =>
      sample.mixedSeamRows < mixedAcceptance.minimumEvolvedMixedSeamRows
      || sample.fineCoarseFaceConnectedPairs
        < mixedAcceptance.minimumInitialFineCoarseFaceConnectedPairCount)) {
      failures.push("adaptive-mass did not retain a face-connected live mixed seam throughout timed evolution");
    }
  }
  const uniformSummary = summarizeSparseCM12BenchmarkArm(uniform);
  const sparseSummary = summarizeSparseCM12BenchmarkArm(sparse);
  const ratio = sparseSummary.frame.median_ms / uniformSummary.frame.median_ms;
  if (!Number.isFinite(ratio)) failures.push("median frame ratio is not finite");
  else if (ratio > acceptance.maximumMedianFrameRatio) {
    failures.push(`Sparse CM12 median frame ratio ${ratio.toFixed(3)} exceeds ${acceptance.maximumMedianFrameRatio.toFixed(2)}`);
  }
  return {
    passed: failures.length === 0,
    targetMet: failures.length === 0 && ratio <= acceptance.targetMedianFrameRatio,
    medianFrameRatio: ratio,
    maximumMedianFrameRatio: acceptance.maximumMedianFrameRatio,
    targetMedianFrameRatio: acceptance.targetMedianFrameRatio,
    uniform: uniformSummary,
    sparse: sparseSummary,
    failures,
  };
}
