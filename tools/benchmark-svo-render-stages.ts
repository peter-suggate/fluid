#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

import { buildSVORenderStageBenchmarkPlan } from "./svo-render-stage-benchmark-contract";

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index < 0 ? undefined : process.argv[index + 1];
}

function required(name: string): string {
  const value = argument(name);
  if (!value) throw new Error(`benchmark:svo-render-stages requires ${name}`);
  return value;
}

function resolution(value: string): { width: number; height: number } {
  const match = /^(\d+)x(\d+)$/.exec(value);
  if (!match) throw new Error("--resolution must be WIDTHxHEIGHT");
  return { width: Number(match[1]), height: Number(match[2]) };
}

const plan = buildSVORenderStageBenchmarkPlan({
  revision: required("--revision"),
  baseUrl: argument("--base-url") ?? "http://localhost:3000/",
  resolution: resolution(argument("--resolution") ?? "1280x720"),
  cycles: argument("--cycles") === undefined ? undefined : Number(argument("--cycles")),
});
const serialized = `${JSON.stringify(plan, null, 2)}\n`, outputPath = argument("--write-plan");
if (outputPath) {
  const absolute = resolve(outputPath); mkdirSync(dirname(absolute), { recursive: true }); writeFileSync(absolute, serialized);
  console.log(`Wrote fixed-resolution SVO render-stage plan to ${absolute}`);
} else process.stdout.write(serialized);

