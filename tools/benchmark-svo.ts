#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

import { SVO_BASELINE_CASES, type SVOBaselineQuality } from "./svo-baseline-cases";
import {
  aggregateSVOBenchmarkObservations,
  buildSVOBenchmarkPlan,
  type SVOBenchmarkObservationBundle,
  type SVOBenchmarkResolution,
} from "./svo-benchmark-contract";

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function requiredArgument(name: string): string {
  const value = argument(name);
  if (!value) throw new Error(`benchmark:svo requires ${name}`);
  return value;
}

function integerArgument(name: string, minimum: number): number {
  const value = Number(requiredArgument(name));
  if (!Number.isSafeInteger(value) || value < minimum) throw new Error(`${name} must be an integer of at least ${minimum}`);
  return value;
}

function parseResolution(value: string, label: string): SVOBenchmarkResolution {
  const match = /^(\d+)x(\d+)$/.exec(value);
  if (!match) throw new Error(`${label} must be WIDTHxHEIGHT`);
  const width = Number(match[1]), height = Number(match[2]);
  if (width < 1 || height < 1) throw new Error(`${label} must contain positive dimensions`);
  return { width, height };
}

const quality = (argument("--quality") ?? "balanced") as SVOBaselineQuality;
if (!["balanced", "high", "ultra"].includes(quality)) throw new Error("--quality must be balanced, high, or ultra");
const caseId = argument("--case");
const cases = caseId === undefined ? SVO_BASELINE_CASES : SVO_BASELINE_CASES.filter(({ id }) => id === caseId);
if (caseId !== undefined && cases.length === 0) throw new Error(`Unknown --case ${caseId}`);

const plan = buildSVOBenchmarkPlan({
  revision: requiredArgument("--revision"),
  adapterId: requiredArgument("--adapter-id"),
  resetToken: requiredArgument("--reset-token"),
  captureNotBeforeUnixMs: integerArgument("--not-before-ms", 0),
  pairCount: argument("--pairs") === undefined ? 4 : integerArgument("--pairs", 1),
  cases,
  quality,
  internalResolution: {
    raster: parseResolution(argument("--raster-resolution") ?? "1280x720", "Raster resolution"),
    svo: parseResolution(argument("--svo-resolution") ?? "1280x720", "SVO resolution"),
  },
});

const observationPath = argument("--observation");
const output = observationPath
  ? aggregateSVOBenchmarkObservations(
    plan,
    JSON.parse(readFileSync(resolve(observationPath), "utf8")) as SVOBenchmarkObservationBundle,
  )
  : plan;
const serialized = `${JSON.stringify(output, null, 2)}\n`;
const outputPath = argument(observationPath ? "--write-report" : "--write-plan");
if (outputPath) {
  const absolute = resolve(outputPath);
  mkdirSync(dirname(absolute), { recursive: true });
  writeFileSync(absolute, serialized);
  console.log(`Wrote ${observationPath ? "validated SVO benchmark report" : "external SVO capture plan"} to ${absolute}`);
} else {
  process.stdout.write(serialized);
}
