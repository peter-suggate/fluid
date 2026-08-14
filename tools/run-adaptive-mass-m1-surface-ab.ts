import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { buildAdaptiveMassM1SurfaceAB } from
  "../lib/methods/adaptive-mass/two-tile-surface-ab";

function argument(name: string): string | undefined {
  const prefix = `--${name}=`;
  return process.argv.slice(2).find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

const result = buildAdaptiveMassM1SurfaceAB();
const svgPath = resolve(argument("svg-out") ?? "artifacts/adaptive-mass-m1-surface-ab.svg");
const jsonPath = resolve(argument("json-out") ?? "artifacts/adaptive-mass-m1-surface-ab.json");
await mkdir(dirname(svgPath), { recursive: true });
await mkdir(dirname(jsonPath), { recursive: true });
await Promise.all([
  writeFile(svgPath, result.svg),
  writeFile(jsonPath, `${JSON.stringify(result.receipt, null, 2)}\n`),
]);
process.stdout.write([
  "Adaptive-mass M1 symmetric CM12 surface-stage A/B",
  `status: ${result.receipt.passed ? "PASS" : "FAIL"}`,
  ...result.receipt.directions.map((direction) =>
    `${direction.adaptiveTopology}: adaptive L1 ${direction.adaptiveL1}, coarse L1 `
      + `${direction.coarseControlL1}, improvement ${100 * direction.adaptiveErrorReductionFromCoarse}%`),
  `reflection L1: ${result.receipt.reflectionL1}`,
  `SVG artifact: ${svgPath}`,
  `JSON artifact: ${jsonPath}`,
  ...result.receipt.failures.map((failure) => `failure: ${failure}`),
].join("\n") + "\n");
process.exitCode = result.receipt.passed ? 0 : 1;
