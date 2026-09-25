import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
interface Sample {
  frame: number;
  closedMass: number;
  highBackMass: number;
  regions: { highBack: { wet: number; unsupportedDeepest: number } };
}

dawnTest("voxel trough back-wall surface dries when its liquid drains", { timeout: 240_000 }, () => {
  const arm = "regression-trough";
  const run = spawnSync(process.execPath, ["--import", "tsx", "tools/probe-uniform-trough-contact-dawn.ts",
    `--arm=${arm}`, "--frames=180"],
  { cwd: root, env: process.env, encoding: "utf8", timeout: 230_000, maxBuffer: 8 * 1024 * 1024 });
  assert.equal(run.status, 0, `${run.error ?? ""}\n${run.stdout}\n${run.stderr}`);
  const result = JSON.parse(readFileSync(`${root}/artifacts/trough-contact/${arm}.json`, "utf8")) as {
    h: number; samples: Sample[];
  };
  assert.ok(result.samples[0]!.regions.highBack.wet > 2000, "exercise the initially wetted voxel wall");
  let drySamples = 0;
  for (const sample of result.samples) {
    assert.equal(sample.closedMass, 0, "contact correction must not move water through solids");
    assert.ok(sample.regions.highBack.unsupportedDeepest > -result.h,
      `frame ${sample.frame}: unsupported wall phi must not grow into a deep false liquid region`);
    if (sample.frame >= 30 && sample.highBackMass < .05) {
      drySamples++;
      assert.equal(sample.regions.highBack.wet, 0, `frame ${sample.frame}: drained back wall must expose air`);
    }
  }
  assert.ok(drySamples >= 3, "exercise sustained release, including after the returning wave");
});
