export type AdaptiveMassM1Axis = "x" | "y" | "z";
export type AdaptiveMassM1Resolution = 4 | 8;

export interface AdaptiveMassM1ProbeVariant {
  readonly axis: 0 | 1 | 2;
  readonly negativeResolution: AdaptiveMassM1Resolution;
  readonly positiveResolution: AdaptiveMassM1Resolution;
  readonly cellCount: number;
  readonly regularFaceCount: number;
  readonly seamPortCount: number;
  readonly seamTermCountMinimum: number;
  readonly seamTermCountMaximum: number;
  readonly constantGradientMaxAbs: number;
  readonly linearGradientMaxAbs: number;
  readonly constantNormalSeamDivergenceMaxAbs: number;
  readonly transposeError: number;
  readonly symmetryMaxAbs: number;
  /** x^T A x for the deterministic manufactured pressure vector. */
  readonly quadraticEnergy: number;
  /** Smallest deterministic sampled x^T A x / x^T x. */
  readonly minimumRayleigh: number;
}

export interface AdaptiveMassM1ReflectionProbe {
  readonly axis: 0 | 1 | 2;
  readonly maximumAbsoluteError: number;
}

export interface AdaptiveMassM1ProbeReceipt {
  readonly variants: readonly AdaptiveMassM1ProbeVariant[];
  readonly reflections: readonly AdaptiveMassM1ReflectionProbe[];
  readonly sourceFailures?: readonly string[];
}

export interface AdaptiveMassM1ABThresholds {
  readonly algebraAbsolute: number;
  readonly minimumEnergy: number;
  readonly minimumRayleigh: number;
}

export interface AdaptiveMassM1ABReport {
  readonly schemaVersion: 1;
  readonly generatedAt: string;
  readonly milestone: "M1.0-static-two-tile-seam";
  readonly execution: {
    readonly backend: "cpu-f64-algebra-oracle";
    readonly limitation: string;
  };
  readonly sharedAlgorithm: {
    readonly paperNumericsSource: string;
    readonly pressureOracleSource: string;
    readonly uniformArm: string;
    readonly adaptiveArm: string;
    readonly onlyMixedDifference: string;
  };
  readonly thresholds: AdaptiveMassM1ABThresholds;
  readonly axes: readonly AdaptiveMassM1AxisReport[];
  readonly missingMilestoneWork: readonly string[];
  readonly failures: readonly string[];
  readonly passed: boolean;
}

export interface AdaptiveMassM1ABArmReport {
  readonly id: "A-uniform-fine" | "A-uniform-coarse" | "B-adaptive-fine-negative"
    | "B-adaptive-fine-positive";
  readonly role: "uniform-control" | "adaptive-candidate";
  readonly topology: `${AdaptiveMassM1Resolution}+${AdaptiveMassM1Resolution}`;
  readonly variant: AdaptiveMassM1ProbeVariant;
  readonly passed: boolean;
}

export interface AdaptiveMassM1AxisReport {
  readonly axis: AdaptiveMassM1Axis;
  readonly explanation: string;
  readonly arms: readonly AdaptiveMassM1ABArmReport[];
  readonly reflection: AdaptiveMassM1ReflectionProbe;
  readonly passed: boolean;
}

const AXES = ["x", "y", "z"] as const;

function expectedShape(negative: AdaptiveMassM1Resolution,
  positive: AdaptiveMassM1Resolution) {
  if (negative === 8 && positive === 8) {
    return { cells: 1024, seams: 64, seamTerms: 2 };
  }
  if (negative === 4 && positive === 4) {
    return { cells: 128, seams: 16, seamTerms: 2 };
  }
  return { cells: 576, seams: 16, seamTerms: 5 };
}

function variantFailures(variant: AdaptiveMassM1ProbeVariant,
  thresholds: AdaptiveMassM1ABThresholds): string[] {
  const name = `${AXES[variant.axis]} ${variant.negativeResolution}+${variant.positiveResolution}`;
  const expected = expectedShape(variant.negativeResolution, variant.positiveResolution);
  const failures: string[] = [];
  if (variant.cellCount !== expected.cells) {
    failures.push(`${name}: ${variant.cellCount} cells, expected ${expected.cells}`);
  }
  if (variant.seamPortCount !== expected.seams) {
    failures.push(`${name}: ${variant.seamPortCount} seam ports, expected ${expected.seams}`);
  }
  if (variant.seamTermCountMinimum !== expected.seamTerms
    || variant.seamTermCountMaximum !== expected.seamTerms) {
    failures.push(`${name}: seam rows have ${variant.seamTermCountMinimum}..${variant.seamTermCountMaximum} terms, expected ${expected.seamTerms}`);
  }
  for (const [metric, value] of [
    ["constant gradient", variant.constantGradientMaxAbs],
    ["linear gradient", variant.linearGradientMaxAbs],
    ["constant normal seam divergence", variant.constantNormalSeamDivergenceMaxAbs],
    ["D=-G^T", variant.transposeError],
    ["matrix symmetry", variant.symmetryMaxAbs],
  ] as const) {
    if (!Number.isFinite(value) || value > thresholds.algebraAbsolute) {
      failures.push(`${name}: ${metric} error ${value} exceeds ${thresholds.algebraAbsolute}`);
    }
  }
  if (!Number.isFinite(variant.quadraticEnergy)
    || variant.quadraticEnergy < thresholds.minimumEnergy) {
    failures.push(`${name}: manufactured pressure energy ${variant.quadraticEnergy} is below ${thresholds.minimumEnergy}`);
  }
  if (!Number.isFinite(variant.minimumRayleigh)
    || variant.minimumRayleigh < thresholds.minimumRayleigh) {
    failures.push(`${name}: sampled Rayleigh quotient ${variant.minimumRayleigh} is below ${thresholds.minimumRayleigh}`);
  }
  return failures;
}

function requireVariant(receipt: AdaptiveMassM1ProbeReceipt, axis: 0 | 1 | 2,
  negativeResolution: AdaptiveMassM1Resolution,
  positiveResolution: AdaptiveMassM1Resolution): AdaptiveMassM1ProbeVariant {
  const variant = receipt.variants.find((candidate) => candidate.axis === axis
    && candidate.negativeResolution === negativeResolution
    && candidate.positiveResolution === positiveResolution);
  if (!variant) {
    throw new Error(`M1 A/B probe omitted ${AXES[axis]} ${negativeResolution}+${positiveResolution}`);
  }
  return variant;
}

/**
 * Builds one deliberately symmetric receipt. A and B are not separate formula
 * implementations: both consume the same row assembler and operator. Arm A
 * supplies the two uniform topologies; arm B changes only the topology data to
 * the 2:1 seam, then repeats it after reflecting the fine side.
 */
export function buildAdaptiveMassM1ABReport(receipt: AdaptiveMassM1ProbeReceipt,
  thresholds: AdaptiveMassM1ABThresholds = {
    algebraAbsolute: 1e-11,
    minimumEnergy: -1e-11,
    minimumRayleigh: -1e-11,
  }): AdaptiveMassM1ABReport {
  const failures: string[] = [...(receipt.sourceFailures ?? []).map(
    (failure) => `core oracle: ${failure}`,
  )];
  const axes = AXES.map((axisName, axisIndex): AdaptiveMassM1AxisReport => {
    const axis = axisIndex as 0 | 1 | 2;
    const variants = [
      ["A-uniform-fine", "uniform-control", 8, 8],
      ["A-uniform-coarse", "uniform-control", 4, 4],
      ["B-adaptive-fine-negative", "adaptive-candidate", 8, 4],
      ["B-adaptive-fine-positive", "adaptive-candidate", 4, 8],
    ] as const;
    const arms = variants.map(([id, role, negative, positive]): AdaptiveMassM1ABArmReport => {
      const variant = requireVariant(receipt, axis, negative, positive);
      const armFailures = variantFailures(variant, thresholds);
      failures.push(...armFailures);
      return { id, role, topology: `${negative}+${positive}`, variant,
        passed: armFailures.length === 0 };
    });
    const reflection = receipt.reflections.find((candidate) => candidate.axis === axis);
    if (!reflection) throw new Error(`M1 A/B probe omitted the ${axisName} reflection`);
    const reflectionPassed = Number.isFinite(reflection.maximumAbsoluteError)
      && reflection.maximumAbsoluteError <= thresholds.algebraAbsolute;
    if (!reflectionPassed) {
      failures.push(`${axisName}: reflected 8+4/4+8 error ${reflection.maximumAbsoluteError} exceeds ${thresholds.algebraAbsolute}`);
    }
    return {
      axis: axisName,
      explanation: "A uses uniform cell spacing on both tiles. B uses the same gradient-row and transpose code with one five-cell 2:1 seam row per coarse face patch; swapping 8+4 to 4+8 must be a reflection, not a second discretization.",
      arms,
      reflection,
      passed: arms.every((arm) => arm.passed) && reflectionPassed,
    };
  });
  return {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    milestone: "M1.0-static-two-tile-seam",
    execution: {
      backend: "cpu-f64-algebra-oracle",
      limitation: "This receipt validates CPU topology and composite pressure algebra only. It does not read back the production uniform shader, run WebGPU, time kernels, solve a complete projection, or advance a CM12 liquid step.",
    },
    sharedAlgorithm: {
      paperNumericsSource: "lib/core/cm12-numerics.ts",
      pressureOracleSource: "lib/methods/adaptive-mass/two-tile-composite-grid.ts",
      uniformArm: "The production uniform shaders and the new method consume the same core CM12 formulas. In this CPU pressure receipt, 4+4 and 8+8 configure the common row builder with equal widths, so every seam row algebraically becomes the ordinary two-cell uniform row.",
      adaptiveArm: "8+4 and 4+8 configure the same row builder with a 2:1 transition; only those 16 rows expand to one coarse plus four fine pressure terms.",
      onlyMixedDifference: "topology, center distance, dual weight, and the pressure terms supplied to the common operator",
    },
    thresholds,
    axes,
    missingMilestoneWork: [
      "GPU-authored characteristic/CM12 row construction and a GPU pressure solve",
      "arbitrary-velocity transport and velocity advection/extension across the seam",
      "a fully coupled multi-step free-surface liquid step and production render",
    ],
    failures: Object.freeze(failures),
    passed: failures.length === 0,
  };
}

function scientific(value: number): string {
  return Number.isFinite(value) ? value.toExponential(3) : String(value);
}

export function formatAdaptiveMassM1ABMarkdown(report: AdaptiveMassM1ABReport): string {
  const lines = [
    "# Adaptive mass M1 symmetric A/B",
    "",
    `Result: **${report.passed ? "PASS" : "FAIL"}**`,
    "",
    "A is the uniform-topology control at both useful resolutions. B is the 2:1 adaptive topology in both orientations. The CPU pressure arms execute the same row assembly, gradient, transpose-divergence, and operator code; configuration data is the only fork. The CM12 paper formulas used by the production uniform shader are separately shared from `lib/core/cm12-numerics.ts`.",
    "",
    `Scope: ${report.execution.limitation}`,
    "",
    "| axis | arm | topology | cells | seam rows | terms/row | const G | linear G | const div | transpose | symmetry | min Rayleigh | result |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
  ];
  for (const axis of report.axes) {
    for (const arm of axis.arms) {
      const v = arm.variant;
      lines.push(`| ${axis.axis} | ${arm.id} | ${arm.topology} | ${v.cellCount} | ${v.seamPortCount} | ${v.seamTermCountMinimum} | ${scientific(v.constantGradientMaxAbs)} | ${scientific(v.linearGradientMaxAbs)} | ${scientific(v.constantNormalSeamDivergenceMaxAbs)} | ${scientific(v.transposeError)} | ${scientific(v.symmetryMaxAbs)} | ${scientific(v.minimumRayleigh)} | ${arm.passed ? "PASS" : "FAIL"} |`);
    }
    lines.push(`| ${axis.axis} | B reflection | 8+4 ↔ 4+8 | — | — | — | — | — | — | — | ${scientific(axis.reflection.maximumAbsoluteError)} | — | ${axis.reflection.maximumAbsoluteError <= report.thresholds.algebraAbsolute ? "PASS" : "FAIL"} |`);
  }
  lines.push("", "## What differs", "", report.sharedAlgorithm.onlyMixedDifference + ".",
    "", "## Still outside this receipt", "");
  for (const item of report.missingMilestoneWork) lines.push(`- ${item}`);
  if (report.failures.length > 0) {
    lines.push("", "## Failures", "");
    for (const failure of report.failures) lines.push(`- ${failure}`);
  }
  return `${lines.join("\n")}\n`;
}
