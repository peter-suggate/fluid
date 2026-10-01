import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
interface Sample {
  frame: number; missing: number; maxSpeed: number; sum: number; excess: number;
  residual:number; recovery:number; full:number; vcycles:number;
  surface: Record<"all" | "interior", { mean_mm: number; rms_mm: number; max_mm: number; range_mm: number }>;
}

// Deliberately pins the reported UI configuration: hose off and tolerance zero
// (the complete configured solve). The loose global default is a separate
// pressure-quality question, not a reason to weaken this still-water oracle.
dawnTest("filled hero pond stays visually at rest with the hose disabled", { timeout: 360_000 }, () => {
  for (const [arm, args, frames] of [
    ["regression-no-forces", ["--gravity=0"], 3],
    ["regression-full", [], 90],
  ] as const) {
    const out = `artifacts/pond-rest/${arm}.json`;
    const run = spawnSync(process.execPath, ["--import", "tsx", "tools/probe-uniform-pond-rest-dawn.ts",
      `--arm=${arm}`, `--frames=${frames}`, `--out=${out}`, ...args],
    { cwd: root, env: process.env, encoding: "utf8", timeout: 170_000, maxBuffer: 8 * 1024 * 1024 });
    assert.equal(run.status, 0, `${arm}: ${run.error ?? ""}\n${run.stdout}\n${run.stderr}`);
    const result = JSON.parse(readFileSync(`${root}/${out}`, "utf8")) as { samples: Sample[] };
    const [initial, first] = result.samples;
    assert.ok(initial && first);
    assert.ok(initial.excess < .003, "no inaccessible seed mass; allow float32 terrain rounding");
    assert.ok(first.surface.all.max_mm < .001, "zero-velocity transport cannot move the authored plane");
    assert.ok(first.maxSpeed < .001, "flat shore must not generate a capillary/contact impulse");
    for (const sample of result.samples) {
      assert.ok(sample.residual < 1e-4, `frame ${sample.frame}: damp the pressure checkerboard (${sample.residual})`);
      assert.equal(sample.recovery,0,`frame ${sample.frame}: converged fixed-budget pond needs no recovery`);
      assert.equal(sample.missing, 0, `frame ${sample.frame}: preserve the whole sampled surface`);
      assert.ok(Math.abs(sample.sum / initial.sum - 1) < 1e-6, `frame ${sample.frame}: dust must not drain the pond`);
      assert.ok(sample.surface.interior.rms_mm < .025, `frame ${sample.frame}: interior RMS ${sample.surface.interior.rms_mm} mm`);
      assert.ok(sample.surface.all.max_mm < .5, `frame ${sample.frame}: shoreline error ${sample.surface.all.max_mm} mm`);
    }
  }
});
