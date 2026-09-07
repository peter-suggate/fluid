import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

// Deliberately a NEW correctness gate, not an assertion that today's defect is
// the expected result. Run explicitly while investigating; not in the short
// canonical suite until the reconstruction satisfies the contract.
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
const root = fileURLToPath(new URL("..", import.meta.url));
function capture(arm: string, profile: string) {
  const output = `artifacts/coarse-surface-grid-imprint/${profile}-${arm}`;
  const run = spawnSync(process.execPath, ["--import", "tsx", "tools/probe-coarse-surface-grid-imprint-dawn.ts",
    "--wait", `--arm=${arm}`, `--profile=${profile}`, `--output=${output}`],
  { cwd: root, env: process.env, encoding: "utf8", timeout: 240_000, maxBuffer: 8 * 1024 * 1024 });
  assert.equal(run.status, 0, `${arm}/${profile} fixture failed: ${run.error ?? ""} signal=${run.signal}\n${run.stdout}\n${run.stderr}`);
  return JSON.parse(readFileSync(`${root}/${output}/summary.json`, "utf8")) as {
    heightRMSError_mm: number; slopeRMSError: number; curvatureRMSError: number;
  };
}

dawnTest("coarse stationary flat and tilted surfaces remain affine", () => {
  for (const profile of ["flat", "tilt"]) {
    const result = capture("fixed4", profile);
    assert.ok(result.heightRMSError_mm < .05, `${profile}: ${JSON.stringify(result)}`);
    assert.ok(result.curvatureRMSError < .0003, `${profile}: no spurious curvature`);
  }
});

dawnTest("coarse-first loses detail without printing its grid into a broad curved surface", () => {
  const fine = capture("fixed1", "bowl");
  const candidates = ["fixed4", "mixed", "adaptive"].map(arm => ({ arm, result: capture(arm, "bowl") }));
  // The analytic second difference is .006 everywhere, on either topology.
  // Allow 25% RMS error, far above f16 publication noise, but reject recurring
  // zero-curvature panels and doubled-curvature knots. An offset in mean height
  // is deliberately not a failure; neither is exact pixel parity demanded.
  const budget = .25 * .006;
  assert.ok(fine.curvatureRMSError < budget / 4, `fine control: ${JSON.stringify(fine)}`);
  const failing = candidates.filter(({ result }) => result.curvatureRMSError > budget);
  assert.deepEqual(failing, [], `Grid-imprint curvature budget ${budget}; ${JSON.stringify(candidates)}`);
});
