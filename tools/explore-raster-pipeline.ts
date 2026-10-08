/** Paired whole-frame trials. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
const root = process.env.FLUID_RASTER_TRIAL_OUT ?? "/tmp/fluid-raster-pipeline";
mkdirSync(root, { recursive: true });
const trials = [
  { name: "baseline", specialized: "0", reuse: "0", opacity: "0" },
  { name: "specialized", specialized: "1", reuse: "0", opacity: "0" },
  { name: "shadow-reuse", specialized: "0", reuse: "1", opacity: "0" },
  { name: "opacity-only", specialized: "0", reuse: "0", opacity: "1" },
  { name: "combined", specialized: "1", reuse: "1", opacity: "1" },
];
const results = [];
let baseline: Buffer | undefined;
for (const trial of trials) {
  const directory = join(root, trial.name);
  const log: Buffer[] = [];
  const code = await new Promise<number | null>((resolve, reject) => {
    const child = spawn(process.execPath, ["--expose-gc", "--import", "tsx", "tools/explore-hero-render-budget.ts"], {
      env: { ...process.env, WEBGPU_NODE_MODULE: process.env.WEBGPU_NODE_MODULE ?? join(process.cwd(), "node_modules/webgpu/index.js"),
        FLUID_PROBE_DEPTH: "1", FLUID_EXPLORE_SURFACE_STYLE: "smooth", FLUID_EXPLORE_RASTER_AO: "1",
        FLUID_EXPLORE_ARMS: "production-raster", FLUID_EXPLORE_VIEWS: "hero", FLUID_EXPLORE_CYCLES: "24",
        FLUID_EXPLORE_DERIVED_TRIALS: trial.name === "baseline" ? "1" : "0",
        FLUID_EXPLORE_OUT: directory, FLUID_EXPLORE_SPECIALIZED: trial.specialized,
        FLUID_EXPLORE_SHADOW_REUSE: trial.reuse, FLUID_EXPLORE_OPACITY_ONLY: trial.opacity },
      stdio: ["ignore", "pipe", "pipe"],
    });
    for (const stream of [child.stdout, child.stderr]) stream.on("data", data => { log.push(data); process.stdout.write(data); });
    child.once("error", reject); child.once("exit", resolve);
  });
  writeFileSync(join(root, `${trial.name}.log`), Buffer.concat(log));
  assert.equal(code, 0, `${trial.name} failed`);
  const pixels = readFileSync(join(directory, "hero-production-raster.rgba16f"));
  baseline ??= pixels;
  let differentBytes = 0;
  assert.equal(pixels.length, baseline.length);
  for (let i = 0; i < pixels.length; i++) if (pixels[i] !== baseline[i]) differentBytes++;
  const frames = JSON.parse(readFileSync(join(directory, "results.json"), "utf8"));
  const maintenance = Buffer.concat(log).toString().split("\n").find(line => line.includes('"phase":"maintenance-total"'));
  const rebuildTrials = Buffer.concat(log).toString().split("\n").find(line => line.includes('"phase":"derived-rebuild-trials"'));
  results.push({ ...trial, rebuildTrials: rebuildTrials && JSON.parse(rebuildTrials), differentBytes, maintenance: maintenance && JSON.parse(maintenance),
    gpuMedian_ms: frames.map((frame: { gpuMedian_ms: number }) => frame.gpuMedian_ms),
    wallMedian_ms: frames.map((frame: { wallMedian_ms: number }) => frame.wallMedian_ms) });
  writeFileSync(join(root, "summary.json"), JSON.stringify(results, null, 2));
}
console.log(JSON.stringify({ phase: "raster-pipeline-summary", results }));
