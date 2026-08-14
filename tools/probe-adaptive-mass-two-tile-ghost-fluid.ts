#!/usr/bin/env node

import {
  probeTwoTileGhostFluidProjection,
  type GhostFluidProjectionProbeReceipt,
} from "../lib/methods/adaptive-mass/two-tile-ghost-fluid-projection-probe";

function e(value: number): string {
  return value.toExponential(3);
}

function markdown(receipt: GhostFluidProjectionProbeReceipt): string {
  const lines = [
    "# Adaptive mass M1 ghost-fluid projection",
    "",
    `Result: **${receipt.passed ? "PASS" : "FAIL"}**`,
    "",
    receipt.formulation.mixedRowWeighting,
    "",
    `Limitation: ${receipt.formulation.limitation}`,
    "",
    "| case | axis | topology | liquid | cut rows | cut 5-term seam | theta range | clamps | min Rayleigh | CG | rel residual | equation max | post div L2(V) | energy error | hydro p error |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ];
  for (const variant of receipt.variants) {
    const minimumRayleigh = Math.min(variant.sampledMinimumRayleigh, variant.minimumCgRayleigh);
    const hydrostaticError = variant.hydrostaticPressureMaxAbsError === undefined
      ? "—" : e(variant.hydrostaticPressureMaxAbsError);
    lines.push(`| ${variant.case} | ${"xyz"[variant.axis]} | ${variant.negativeResolution}+${variant.positiveResolution} | ${variant.liquidCellCount} | ${variant.cutRowCount} | ${variant.cutFiveTermSeamRowCount} | ${e(variant.thetaMinimum)}..${e(variant.thetaMaximum)} | ${variant.thetaClampCount} | ${e(minimumRayleigh)} | ${variant.iterations} | ${e(variant.solverRelativeResidualL2)} | ${e(variant.pressureEquationMaxAbsResidual)} | ${e(variant.postLiquidDivergenceVolumeL2)} | ${e(variant.energyIdentityAbsError)} | ${hydrostaticError} |`);
  }
  lines.push("", "Reflections:", "");
  for (const reflection of receipt.reflections) {
    lines.push(`- ${reflection.case} ${"xyz"[reflection.axis]}: theta ${e(reflection.thetaMaxAbsError)}, pressure ${e(reflection.pressureMaxAbsError)}, projected velocity ${e(reflection.projectedVelocityMaxAbsError)}`);
  }
  if (receipt.failures.length > 0) {
    lines.push("", "Failures:", "", ...receipt.failures.map((failure) => `- ${failure}`));
  }
  return `${lines.join("\n")}\n`;
}

const receipt = probeTwoTileGhostFluidProjection();
process.stdout.write(process.argv.includes("--json")
  ? `${JSON.stringify(receipt, null, 2)}\n`
  : markdown(receipt));
process.exitCode = receipt.passed ? 0 : 1;
