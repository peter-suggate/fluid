import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
const execute = promisify(execFile);

dawnTest("topology-only rung changes preserve planar height and bound non-flat representation loss", {
  timeout: 240_000,
}, async () => {
  const output = process.env.SURFACE_TEST_OUTPUT ?? await mkdtemp(join(tmpdir(), "cm12-surface-rungs-"));
  try {
    for (const profile of ["flat", "bowl"]) {
      const arm = join(output, profile);
      await execute(process.execPath, ["--max-old-space-size=12288", "--import", "tsx",
        "tools/probe-surface-rung-continuity-dawn.ts"], {
        cwd: process.cwd(), env: { ...process.env, SURFACE_OUTPUT: arm, SURFACE_PROFILE: profile },
        timeout: 100_000, maxBuffer: 4 * 1024 * 1024,
      });
      let previousHeight: Float64Array | undefined;
      let previousMass: number | undefined;
      let previousGeneration: number | undefined;
      let maximumHeightChange = 0;
      for (let step = 1; step <= 5; step++) {
        const directory = join(arm, `step-${step}`);
        const floats = async (name: string) => {
          const bytes = await readFile(join(directory, name));
          return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
        };
        const density = await floats("density.bin"), phi = await floats("phi.bin");
        const activity = JSON.parse(await readFile(join(directory, "activity.json"), "utf8"));
        const active = activity.bricks.filter((b: { active: boolean }) => b.active);
        const resolution = step % 2 === 0 ? 8 : 4;
        assert.ok(active.length > 0 && active.every((b: { acceptedResolution: number }) => b.acceptedResolution === resolution),
          `${profile} step ${step} must actually commit the requested rung`);
        if (previousGeneration !== undefined)
          assert.notEqual(activity.acceptedTopologyGeneration, previousGeneration);
        previousGeneration = activity.acceptedTopologyGeneration;
        const height = new Float64Array(16 * 16).fill(NaN);
        for (let z = 0; z < 16; z++) for (let x = 0; x < 16; x++) {
          for (let y = 0; y < 15; y++) {
            const at = x + 16 * (y + 16 * z), lo = phi[at]!, hi = phi[at + 16]!;
            if (lo <= 0 && hi > 0) height[x + 16 * z] = y + .5 - lo / (hi - lo);
          }
        }
        assert.ok(height.every(Number.isFinite), "every column must retain its surface");
        const mass = density.reduce((sum, value) => sum + value, 0);
        if (previousMass !== undefined) assert.ok(Math.abs(mass - previousMass) < .001,
          "topology-only changes must preserve total accepted liquid volume");
        if (previousHeight) for (let i = 0; i < height.length; i++)
          maximumHeightChange = Math.max(maximumHeightChange, Math.abs(height[i]! - previousHeight[i]!));
        previousHeight = height; previousMass = mass;
        if (profile === "flat") assert.ok(height.every(h => Math.abs(h - 9.6) < .005),
          `${profile} step ${step}: height range ${Math.min(...height)}..${Math.max(...height)} must retain9.6 through both transitions`);
      }
      // A curved field loses subcell information when restricted; its accepted
      // child-column volumes also change. Bound that spatial reconstruction
      // difference separately from the exact planar invariant above. This is
      // eight times tighter than the default one-cell coarsening error budget.
      assert.ok(maximumHeightChange < (profile === "flat" ? .005 : .125),
        `${profile} topology-only height change was ${maximumHeightChange} finest cells`);
    }
  } finally {
    if (!process.env.SURFACE_TEST_OUTPUT) await rm(output, { recursive: true, force: true });
  }
});
