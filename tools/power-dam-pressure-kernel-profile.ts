import type {
  PowerDamFineTimestampBucket,
  PowerDamPassTimestampReport,
} from "./power-dam-performance-report";

/** Prefixes which contain the complete recurring pressure solve. Keeping this
 * list here gives the benchmark, the PassBroker isolation scope, and the JSON
 * report one shared definition of "pressure". */
export const POWER_DAM_PRESSURE_KERNEL_LABEL_PREFIXES = [
  "Pipelined MGPCG",
  "Section 4.3",
  "Factor-1 dense M1",
  "SPGrid V-cycle",
  "SPGrid accurate A2",
  "SPGrid Section 6.3",
] as const;

export interface PowerDamPressureKernelBucket {
  readonly samples: number;
  readonly total_ms: number;
  readonly totalPerAdvance_ms: number;
  readonly mean_ms: number;
  readonly minimum_ms: number;
  readonly maximum_ms: number;
  readonly shareOfPressure: number;
}

export interface PowerDamPressureKernelLabelBucket extends PowerDamPressureKernelBucket {
  readonly region: string;
}

export interface PowerDamPressureKernelProfile {
  readonly schemaVersion: 1;
  /** True means every pressure label owned its own pass and every requested
   * bracket produced a valid timestamp without exhausting the query set. */
  readonly exact: boolean;
  readonly completePressureScope: boolean;
  readonly attributionMode: "targeted-label-isolated";
  readonly capturedAdvances: number;
  readonly measuredPasses: number;
  readonly capacityOverflows: number;
  readonly invalidPasses: number;
  /** Sum of pressure-pass brackets divided by captured advances. This is an
   * instrumented ranking number, not the clean production wall clock. */
  readonly instrumentedPressurePerAdvance_ms: number;
  readonly byRegion: Readonly<Record<string, PowerDamPressureKernelBucket>>;
  readonly byLabel: Readonly<Record<string, PowerDamPressureKernelLabelBucket>>;
  readonly unclassifiedPressureLabels: readonly string[];
  readonly warnings: readonly string[];
}

const safeLabel = (label: string): string => label
  .replaceAll("·", "-")
  .replace(/\s+/g, " ")
  .trim();

export function isPowerDamPressureKernelLabel(label: string): boolean {
  const value = safeLabel(label);
  return POWER_DAM_PRESSURE_KERNEL_LABEL_PREFIXES.some((prefix) =>
    value.startsWith(prefix));
}

/** Stable optimization taxonomy over source-facing labels. It deliberately
 * describes numerical work rather than individual entry-point names so a
 * harmless shader rename does not destroy longitudinal profiles. */
export function powerDamPressureKernelRegion(label: string): string | undefined {
  const value = safeLabel(label);
  const lower = value.toLowerCase();
  const level = /\blevel\s+(\d+)\b/i.exec(value)?.[1];

  if (lower.startsWith("factor-1 dense m1")) {
    if (/convergence gate/.test(lower)) return "preconditioner.dense.gate";
    if (/initialize correction/.test(lower)) return "preconditioner.dense.initialize";
    if (/pre-smooth/.test(lower)) {
      return `preconditioner.dense.smoothing.pre.level-${level ?? "unknown"}`;
    }
    if (/post-smooth/.test(lower)) {
      return `preconditioner.dense.smoothing.post.level-${level ?? "unknown"}`;
    }
    if (/\brestrict\b/.test(lower)) {
      return `preconditioner.dense.transfer.restrict.level-${level ?? "unknown"}`;
    }
    if (/\bprolong\b/.test(lower)) {
      return `preconditioner.dense.transfer.prolong.level-${level ?? "unknown"}`;
    }
    if (/tail levels/.test(lower)) return "preconditioner.dense.coarse-tail";
    if (/publish correction/.test(lower)) return "preconditioner.dense.publish";
    return "preconditioner.dense.other";
  }

  if (lower.startsWith("spgrid v-cycle")) {
    if (/convergence-gated/.test(lower)) return "vcycle.gate";
    if (/setup|capture|candidate|inactive exact|validated exact/.test(lower)) {
      return "vcycle.setup";
    }
    if (/one-pass symmetric correction/.test(lower)) return "vcycle.initialize";
    if (/pre-smooth/.test(lower)) return `vcycle.smoothing.pre.level-${level ?? "unknown"}`;
    if (/post-smooth/.test(lower)) return `vcycle.smoothing.post.level-${level ?? "unknown"}`;
    if (/\brestrict\b/.test(lower)) return `vcycle.transfer.restrict.level-${level ?? "unknown"}`;
    if (/\bprolong\b/.test(lower)) return `vcycle.transfer.prolong.level-${level ?? "unknown"}`;
    if (/coarse v-cycle tail/.test(lower)) return "vcycle.coarse-tail";
    if (/publish correction/.test(lower)) return "vcycle.publish";
    return "vcycle.other";
  }

  if (lower.startsWith("spgrid accurate a2")) {
    if (/convergence-gated/.test(lower)) return "operator.full.gate";
    if (/direct terms/.test(lower)) return "operator.full.direct-terms";
    if (/adjoint children/.test(lower)) return "operator.full.adjoint-children";
    if (/row fold/.test(lower)) return "operator.full.row-fold";
    return "operator.full.other";
  }

  if (lower.startsWith("spgrid section 6.3")) {
    if (/direct terms/.test(lower)) return "operator.band.direct-terms";
    if (/adjoint children/.test(lower)) return "operator.band.adjoint-children";
    if (/row fold/.test(lower)) return "operator.band.row-fold";
    return "operator.band.other";
  }

  if (lower.startsWith("section 4.3")) {
    if (/convergence-gated|accepted-row|boundary-band publication/.test(lower)) {
      return "preconditioner.gates-worksets";
    }
    if (/smooth/.test(lower)) return "preconditioner.band-smoothing";
    if (/inner residual/.test(lower)) return "preconditioner.inner-residual";
    if (/correction publication/.test(lower)) return "preconditioner.publish";
    return "preconditioner.other";
  }

  if (lower.startsWith("pipelined mgpcg")) {
    if (/authority gate|fail-closed publication/.test(lower)) return "outer.gates-publication";
    if (/reduction|curvature/.test(lower)) return "outer.reductions";
    if (/initial residual/.test(lower)) return "outer.initial-residual";
    if (/state initialization|direction|outer iteration/.test(lower)) {
      return "outer.vector-updates";
    }
    return "outer.other";
  }
  return undefined;
}

interface MutableBucket {
  samples: number;
  total_ms: number;
  minimum_ms: number;
  maximum_ms: number;
}

const add = (target: Map<string, MutableBucket>, key: string,
  source: PowerDamFineTimestampBucket): void => {
  const bucket = target.get(key) ?? {
    samples: 0,
    total_ms: 0,
    minimum_ms: Number.POSITIVE_INFINITY,
    maximum_ms: 0,
  };
  bucket.samples += source.samples;
  bucket.total_ms += source.total_ms;
  bucket.minimum_ms = Math.min(bucket.minimum_ms, source.minimum_ms);
  bucket.maximum_ms = Math.max(bucket.maximum_ms, source.maximum_ms);
  target.set(key, bucket);
};

const finish = (
  bucket: MutableBucket,
  captures: number,
  pressureTotal: number,
): PowerDamPressureKernelBucket => ({
  ...bucket,
  totalPerAdvance_ms: bucket.total_ms / captures,
  mean_ms: bucket.total_ms / Math.max(1, bucket.samples),
  shareOfPressure: pressureTotal > 0 ? bucket.total_ms / pressureTotal : 0,
});

export function buildPowerDamPressureKernelProfile(
  timestamps: PowerDamPassTimestampReport,
): PowerDamPressureKernelProfile | undefined {
  const captures = Math.max(1, timestamps.capturedCommandBuffers);
  const regions = new Map<string, MutableBucket>();
  const labels = new Map<string, MutableBucket>();
  const labelRegions = new Map<string, string>();
  const unclassified = new Set<string>();
  let pressureTotal = 0;
  let pressureSamples = 0;

  for (const [label, bucket] of Object.entries(timestamps.byLabel)) {
    if (!isPowerDamPressureKernelLabel(label)) continue;
    const region = powerDamPressureKernelRegion(label);
    if (region === undefined) {
      unclassified.add(label);
      continue;
    }
    pressureTotal += bucket.total_ms;
    pressureSamples += bucket.samples;
    add(regions, region, bucket);
    add(labels, label, bucket);
    labelRegions.set(label, region);
  }
  if (pressureSamples === 0) return undefined;

  const selectedPrefixes = (timestamps.labelPrefixes ?? []).map(safeLabel);
  const completePressureScope = selectedPrefixes.length === 0
    || POWER_DAM_PRESSURE_KERNEL_LABEL_PREFIXES.every((required) =>
      selectedPrefixes.some((selected) =>
        required.startsWith(selected) || selected.startsWith(required)));
  const warnings: string[] = [
    "Per-region GPU time is measured on a diagnostic command stream with pressure labels split into passes; use a clean --steps=N run for production wall-clock savings.",
  ];
  if (!timestamps.labelIsolated) {
    warnings.push("Pressure labels did not own isolated passes; region attribution is composite.");
  }
  if (!completePressureScope) {
    warnings.push("The timestamp prefix filter did not cover every pressure-solver label family; the reported pressure total is partial.");
  }
  if (timestamps.capacityOverflows > 0) {
    warnings.push(`${timestamps.capacityOverflows} pressure timestamp brackets were dropped because the query set filled.`);
  }
  if (timestamps.invalidPasses > 0) {
    warnings.push(`${timestamps.invalidPasses} pressure timestamp brackets were invalid.`);
  }
  if (unclassified.size > 0) {
    warnings.push(`${unclassified.size} pressure labels did not map to the stable region taxonomy.`);
  }
  const exact = timestamps.labelIsolated === true
    && completePressureScope
    && timestamps.capacityOverflows === 0
    && timestamps.invalidPasses === 0
    && unclassified.size === 0;
  return {
    schemaVersion: 1,
    exact,
    completePressureScope,
    attributionMode: "targeted-label-isolated",
    capturedAdvances: timestamps.capturedCommandBuffers,
    measuredPasses: pressureSamples,
    capacityOverflows: timestamps.capacityOverflows,
    invalidPasses: timestamps.invalidPasses,
    instrumentedPressurePerAdvance_ms: pressureTotal / captures,
    byRegion: Object.fromEntries(Array.from(regions.entries())
      .sort((left, right) => right[1].total_ms - left[1].total_ms
        || left[0].localeCompare(right[0]))
      .map(([region, bucket]) => [region, finish(bucket, captures, pressureTotal)])),
    byLabel: Object.fromEntries(Array.from(labels.entries())
      .sort((left, right) => right[1].total_ms - left[1].total_ms
        || left[0].localeCompare(right[0]))
      .map(([label, bucket]) => [label, {
        ...finish(bucket, captures, pressureTotal),
        region: labelRegions.get(label)!,
      }])),
    unclassifiedPressureLabels: [...unclassified].sort(),
    warnings,
  };
}
