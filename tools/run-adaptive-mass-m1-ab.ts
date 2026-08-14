import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  buildAdaptiveMassM1ABReport,
  formatAdaptiveMassM1ABMarkdown,
  type AdaptiveMassM1ProbeReceipt,
  type AdaptiveMassM1ProbeVariant,
} from "../lib/harness/adaptive-mass-m1-ab-report";
import {
  probeTwoTileCompositeGrid,
} from "../lib/methods/adaptive-mass/two-tile-composite-grid";

const argument = (name: string): string | undefined => {
  const args = process.argv.slice(2);
  const prefix = `--${name}=`;
  const inline = args.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : undefined;
};

const here = fileURLToPath(new URL(".", import.meta.url));
const root = resolve(here, "..");
const tolerance = Number(argument("tolerance") ?? "1e-11");
if (!Number.isFinite(tolerance) || tolerance <= 0) {
  throw new RangeError("--tolerance must be finite and positive");
}

const outputPath = argument("out") ?? "artifacts/adaptive-mass-m1-ab.json";
const markdownOutputPath = argument("markdown-out");
const jsonOnly = process.argv.includes("--json");

const source = probeTwoTileCompositeGrid(tolerance);
const variants: AdaptiveMassM1ProbeVariant[] = source.variants.map((variant) => {
  return {
    ...variant,
    seamTermCountMinimum: variant.seamTermCountMin,
    seamTermCountMaximum: variant.seamTermCountMax,
  };
});

const receipt: AdaptiveMassM1ProbeReceipt = {
  variants,
  reflections: source.swapSymmetryByAxis.map((reflection) => ({
    axis: reflection.axis,
    maximumAbsoluteError: reflection.maximumMatrixDifference,
  })),
  sourceFailures: source.failures,
};
const report = buildAdaptiveMassM1ABReport(receipt, {
  algebraAbsolute: tolerance,
  minimumEnergy: -tolerance,
  minimumRayleigh: -tolerance,
});
const markdown = formatAdaptiveMassM1ABMarkdown(report);
const json = `${JSON.stringify(report, null, 2)}\n`;

async function writeArtifact(path: string, contents: string): Promise<string> {
  const absolute = resolve(root, path);
  await mkdir(dirname(absolute), { recursive: true });
  await writeFile(absolute, contents, "utf8");
  return absolute;
}

if (outputPath === "-") {
  process.stdout.write(json);
} else {
  const absoluteOutput = await writeArtifact(outputPath, json);
  if (!jsonOnly) process.stderr.write(`JSON artifact: ${absoluteOutput}\n`);
}
if (markdownOutputPath) {
  const absoluteMarkdown = await writeArtifact(markdownOutputPath, markdown);
  if (!jsonOnly) process.stderr.write(`Markdown artifact: ${absoluteMarkdown}\n`);
}
if (outputPath !== "-" && jsonOnly) process.stdout.write(json);
else if (!jsonOnly && outputPath !== "-") process.stdout.write(markdown);

if (!source.passed || !report.passed) process.exitCode = 1;
