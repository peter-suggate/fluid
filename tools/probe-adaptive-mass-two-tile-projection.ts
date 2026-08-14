#!/usr/bin/env node

import {
  probeTwoTilePressureProjection,
  type TwoTileProjectionProbeReceipt,
} from "../lib/methods/adaptive-mass/two-tile-pressure-projection-probe";

function scientific(value: number): string {
  return value.toExponential(3);
}

function markdown(receipt: TwoTileProjectionProbeReceipt): string {
  const lines = [
    "# Adaptive mass M1 pressure projection",
    "",
    `Result: **${receipt.passed ? "PASS" : "FAIL"}**`,
    "",
    `Scope: ${receipt.limitation}`,
    "",
    "| axis | topology | iterations | pre div L2(V) | post div L2(V) | reduction | solver residual | energy change | result |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---|",
  ];
  for (const variant of receipt.variants) {
    const passed = variant.rhsCompatibilityAbs <= receipt.thresholds.compatibilityAbsolute
      && variant.solverRelativeResidualL2 <= receipt.thresholds.solverRelative
      && (variant.postDivergenceVolumeL2 <= receipt.thresholds.postDivergenceAbsolute
        || variant.divergenceReduction <= receipt.thresholds.postDivergenceRelative)
      && variant.energyNonIncreasing;
    lines.push(`| ${"xyz"[variant.axis]} | ${variant.negativeResolution}+${variant.positiveResolution} | ${variant.iterations} | ${scientific(variant.preDivergenceVolumeL2)} | ${scientific(variant.postDivergenceVolumeL2)} | ${scientific(variant.divergenceReduction)} | ${scientific(variant.solverRelativeResidualL2)} | ${scientific(variant.kineticEnergyAfter - variant.kineticEnergyBefore)} | ${passed ? "PASS" : "FAIL"} |`);
  }
  lines.push("", "Reflected 8+4 ↔ 4+8 maximum errors:", "");
  for (const reflection of receipt.reflections) {
    lines.push(`- ${"xyz"[reflection.axis]}: pressure ${scientific(reflection.pressureMaxAbsError)}, projected velocity ${scientific(reflection.projectedVelocityMaxAbsError)}, post divergence ${scientific(reflection.postDivergenceMaxAbsError)}`);
  }
  if (receipt.failures.length > 0) {
    lines.push("", "Failures:", "", ...receipt.failures.map((failure) => `- ${failure}`));
  }
  return `${lines.join("\n")}\n`;
}

const json = process.argv.includes("--json");
const receipt = probeTwoTilePressureProjection();
process.stdout.write(json ? `${JSON.stringify(receipt, null, 2)}\n` : markdown(receipt));
process.exitCode = receipt.passed ? 0 : 1;
