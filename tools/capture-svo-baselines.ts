#!/usr/bin/env node
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { canonicalScene } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import type { PerformanceSnapshot } from "../lib/stores/diagnostics-store";
import {
  SVO_BASELINE_ADAPTER_ASSUMPTIONS,
  SVO_BASELINE_REQUIRED_LIMITS,
  SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS,
  SVO_BASELINE_TOLERANCES,
  buildSVOBaselineCaptureJobs,
  buildSVOBaselineManifest,
  summarizeSVOBaselineTimings,
  type SVOBaselineAdapterObservation,
} from "./svo-baseline-contract";
import { createSmokeScenario } from "./webgpu-smoke-scenarios";

interface CaptureObservation {
  jobId: string;
  adapter: SVOBaselineAdapterObservation;
  performanceSamples: Pick<PerformanceSnapshot, "cpuFrame_ms" | "gpuRender_ms" | "gpuDryScene_ms" | "gpuRenderTimingAvailable">[];
  rendererOwnedBytes: number;
  colorFile: string;
  signalFiles?: Partial<Record<"depth" | "geometricNormal" | "identityMedia" | "energy", string>>;
}

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function resolution(value: string | undefined, fallback: readonly [number, number]) {
  const match = value?.match(/^(\d+)x(\d+)$/);
  const width = match ? Number(match[1]) : fallback[0], height = match ? Number(match[2]) : fallback[1];
  if (width < 1 || height < 1) throw new RangeError("Capture resolution must be WIDTHxHEIGHT with positive integers");
  return { width, height };
}

const revision = argument("--revision") ?? process.env.SVO_BASELINE_REVISION ?? "working-tree";
const adapterSlug = argument("--adapter") ?? process.env.SVO_BASELINE_ADAPTER ?? "unrecorded-adapter";
const baseUrl = argument("--base-url") ?? process.env.SVO_BASELINE_URL ?? "http://localhost:3000/";
const jobs = buildSVOBaselineCaptureJobs({
  baseUrl,
  revision,
  adapter: adapterSlug,
  internalResolution: {
    raster: resolution(argument("--raster-resolution"), [1280, 720]),
    svo: resolution(argument("--svo-resolution"), [1280, 720]),
  },
});

const observationPath = argument("--observation");
if (!observationPath) {
  const plan = {
    schemaVersion: 1,
    purpose: "Raster versus default SVO-dry plus raster-water deterministic capture plan",
    requiredLimits: SVO_BASELINE_REQUIRED_LIMITS,
    sparseLayoutAssumptions: SVO_BASELINE_SPARSE_LAYOUT_ASSUMPTIONS,
    adapterAssumptions: SVO_BASELINE_ADAPTER_ASSUMPTIONS,
    tolerances: SVO_BASELINE_TOLERANCES,
    jobs,
  };
  const output = `${JSON.stringify(plan, null, 2)}\n`;
  const outputPath = argument("--write-plan");
  if (outputPath) {
    const absolute = resolve(outputPath);
    mkdirSync(dirname(absolute), { recursive: true });
    writeFileSync(absolute, output);
    console.log(`Wrote ${jobs.length} deterministic capture jobs to ${absolute}`);
  } else {
    process.stdout.write(output);
  }
} else {
  const observation = JSON.parse(readFileSync(resolve(observationPath), "utf8")) as CaptureObservation;
  const job = jobs.find((candidate) => candidate.id === observation.jobId);
  if (!job) throw new Error(`Unknown baseline capture job ${observation.jobId}`);
  const timing = summarizeSVOBaselineTimings(observation.performanceSamples, job.warmupFrames);
  if (timing.measuredFrames < job.measuredFrames) {
    throw new Error(`Capture has ${timing.measuredFrames} measured frames; ${job.measuredFrames} are required`);
  }
  const scene = job.baseline.source.kind === "smoke-scenario"
    ? createSmokeScenario(job.baseline.source.id).scene
    : getScenePreset(job.baseline.source.id).create();
  const writeArtifact = (path: string, contents: string) => {
    const absolute = resolve(path);
    mkdirSync(dirname(absolute), { recursive: true });
    writeFileSync(absolute, contents);
  };
  writeArtifact(job.artifacts.scene, `${canonicalScene(scene)}\n`);
  writeArtifact(job.artifacts.camera, `${JSON.stringify({ camera: job.baseline.camera, checkpoint: job.baseline.checkpoint }, null, 2)}\n`);
  writeArtifact(job.artifacts.timings, `${JSON.stringify({ summary: timing, samples: observation.performanceSamples }, null, 2)}\n`);
  const copyArtifact = (source: string, target: string) => {
    const absolute = resolve(target);
    mkdirSync(dirname(absolute), { recursive: true });
    copyFileSync(resolve(source), absolute);
  };
  copyArtifact(observation.colorFile, job.artifacts.color);
  const available = new Set<"color" | "timings" | "scene" | "camera" | "depth" | "geometricNormal" | "identityMedia" | "energy">([
    "color", "timings", "scene", "camera",
  ]);
  for (const [key, source] of Object.entries(observation.signalFiles ?? {}) as ["depth" | "geometricNormal" | "identityMedia" | "energy", string][]) {
    copyArtifact(source, job.artifacts[key]);
    available.add(key);
  }
  const manifest = buildSVOBaselineManifest({
    job,
    adapter: observation.adapter,
    status: "captured",
    timing,
    rendererOwnedBytes: observation.rendererOwnedBytes,
    availableArtifacts: [...available],
  });
  writeArtifact(job.artifacts.manifest, `${JSON.stringify(manifest, null, 2)}\n`);
  console.log(`Captured ${job.id} at ${job.artifacts.manifest}`);
}
