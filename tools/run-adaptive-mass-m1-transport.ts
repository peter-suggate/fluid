import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  probeTwoTileConservativeTransport,
  type TwoTileTransportVariantReceipt,
} from "../lib/methods/adaptive-mass/two-tile-conservative-transport";

const argument = (name: string): string | undefined => {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  return inline?.slice(prefix.length);
};

const tolerance = Number(argument("tolerance") ?? "1e-11");
const json = process.argv.includes("--json");
const result = probeTwoTileConservativeTransport(tolerance);

function maximum(
  variants: readonly TwoTileTransportVariantReceipt[],
  select: (variant: TwoTileTransportVariantReceipt) => number,
): number {
  return Math.max(...variants.map(select));
}

if (json) {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} else {
  const equal = result.variants.filter((variant) =>
    variant.negativeResolution === variant.positiveResolution);
  const adaptive = result.variants.filter((variant) =>
    variant.negativeResolution !== variant.positiveResolution);
  process.stdout.write([
    "Adaptive-mass M1 conservative transport",
    `status: ${result.passed ? "PASS" : "FAIL"}`,
    `variants: ${result.variants.length} (uniform controls ${equal.length}, adaptive ${adaptive.length})`,
    `uniform max |beta-1|: ${maximum(equal, (variant) => variant.finalBetaMaximumAbsoluteError)}`,
    `adaptive max |beta-1|: ${maximum(adaptive, (variant) => variant.finalBetaMaximumAbsoluteError)}`,
    `uniform max pulse mass error: ${maximum(equal, (variant) => variant.pulseMassAbsoluteError)}`,
    `adaptive max pulse mass error: ${maximum(adaptive, (variant) => variant.pulseMassAbsoluteError)}`,
    `adaptive max deficit: ${maximum(adaptive, (variant) => variant.maximumDeficit)}`,
    `adaptive min coefficient: ${Math.min(...adaptive.map((variant) => variant.minimumCoefficient))}`,
    `reflection max density difference: ${Math.max(...result.reflections.map((value) => value.maximumDensityDifference))}`,
    `reflection max gamma difference: ${Math.max(...result.reflections.map((value) => value.maximumGammaDifference))}`,
    `persistent-gamma soak: ${result.soaks.length} x ${result.soaks[0]?.steps ?? 0} steps`,
    `soak max mass error: ${Math.max(...result.soaks.map((value) => value.maximumMassAbsoluteError))}`,
    `soak density range: ${Math.min(...result.soaks.map((value) => value.minimumDensity))}..${Math.max(...result.soaks.map((value) => value.maximumDensity))}`,
    `soak gamma range: ${Math.min(...result.soaks.map((value) => value.minimumGamma))}..${Math.max(...result.soaks.map((value) => value.maximumGamma))}`,
    ...result.failures.map((failure) => `failure: ${failure}`),
  ].join("\n") + "\n");
}

// Keep this file directly executable from any cwd while making the resolved
// repository location visible in stack traces produced by tsx.
void resolve(fileURLToPath(new URL("..", import.meta.url)));
if (!result.passed) process.exitCode = 1;
