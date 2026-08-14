import {
  probeTwoTileSurfaceConditioning,
  type SurfaceConditioningVariantReceipt,
} from "../lib/methods/adaptive-mass/two-tile-surface-conditioning";

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  return process.argv.slice(2).find((value) => value.startsWith(prefix))?.slice(prefix.length);
};

const tolerance = Number(argument("tolerance") ?? "1e-10");
const steps = Number(argument("steps") ?? "8");
const result = probeTwoTileSurfaceConditioning(tolerance, steps);

function maximum(
  variants: readonly SurfaceConditioningVariantReceipt[],
  select: (variant: SurfaceConditioningVariantReceipt) => number,
): number {
  return Math.max(...variants.map(select));
}

if (process.argv.includes("--json")) {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} else {
  const equal = result.variants.filter((variant) =>
    variant.negativeResolution === variant.positiveResolution);
  const adaptive = result.variants.filter((variant) =>
    variant.negativeResolution !== variant.positiveResolution);
  process.stdout.write([
    "Adaptive-mass M1 surface conditioning",
    `status: ${result.passed ? "PASS" : "FAIL"}`,
    `variants: ${result.variants.length} (uniform controls ${equal.length}, adaptive ${adaptive.length})`,
    `steps per variant: ${result.steps}`,
    `uniform max final mass error: ${maximum(equal, (value) => value.finalMassAbsoluteError)}`,
    `adaptive max final mass error: ${maximum(adaptive, (value) => value.finalMassAbsoluteError)}`,
    `adaptive min density: ${Math.min(...adaptive.map((value) => value.minimumDensity))}`,
    `adaptive gamma range: ${Math.min(...adaptive.map((value) => value.minimumGamma))}`
      + ` .. ${maximum(adaptive, (value) => value.maximumGamma)}`,
    `adaptive total returned mass: ${adaptive.reduce((sum, value) => sum + value.totalSharpeningReturnMass, 0)}`,
    `adaptive cross-resolution return mass: ${adaptive.reduce((sum, value) => sum + value.totalCrossResolutionReturnMass, 0)}`,
    `reflection max density difference: ${Math.max(...result.reflections.map((value) => value.maximumDensityDifference))}`,
    `reflection max gamma difference: ${Math.max(...result.reflections.map((value) => value.maximumGammaDifference))}`,
    ...result.failures.map((failure) => `failure: ${failure}`),
  ].join("\n") + "\n");
}
if (!result.passed) process.exitCode = 1;
