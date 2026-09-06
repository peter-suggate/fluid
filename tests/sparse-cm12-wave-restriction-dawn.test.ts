import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

// This evolved wave produces a nonconstant gamma field. The former serial
// 512-child restriction failed its conservation receipt and rolled back all
// 32 requested width-8 cells, even though the native face receipts passed.
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "a resolved gravity wave can coarsen directly from width 1 to width 8",
  { timeout: 120_000 }, async t => {
    const directory = await mkdtemp(join(tmpdir(), "cm12-wave-restriction-"));
    try {
      await new Promise<void>((resolve, reject) => {
        const child = spawn(process.execPath, ["--import", "tsx",
          "tools/probe-sparse-gravity-wave-dawn.ts", "--arm=fixed8",
          "--fork-step=70", "--steps=71", "--audit=70,71", `--output=${directory}`,
        ], { cwd: process.cwd(), env: process.env, stdio: ["ignore", "pipe", "pipe"] });
        let log = "";
        child.stdout.on("data", chunk => { log += chunk; });
        child.stderr.on("data", chunk => { log += chunk; });
        child.on("error", reject);
        child.on("exit", code => code === 0 ? resolve() : reject(new Error(`${code}\n${log}`)));
      });
      const activity = JSON.parse(await readFile(join(directory, "70-branch-activity.json"), "utf8"));
      assert.equal(activity.faultFlags, 0);
      assert.equal(activity.commitFailed, false);
      assert.equal(activity.preparedBrickCount, 32);
      assert.equal(activity.committedBrickCount, 32);
      let maximumGammaError = 0;
      for (const brick of activity.bricks) {
        assert.equal(brick.acceptedResolution, 1, "each brick contains one width-8 cell");
        assert.equal(brick.transferStatus, 1);
        assert.equal(brick.faceTransferStatus, 1);
        assert.ok(Math.abs(brick.transferMassErrorFineCells)
          <= Math.max(1e-4, 1e-6 * Math.abs(brick.transferMassBeforeFineCells)));
        maximumGammaError = Math.max(maximumGammaError, Math.abs(brick.transferGammaErrorFineCells));
      }
      assert.ok(maximumGammaError <= 1e-3, `gamma receipt error ${maximumGammaError}`);
      const stages: { stage: string; mass: number }[] = JSON.parse(
        await readFile(join(directory, "70-stages.json"), "utf8"));
      const before = stages.find(row => row.stage === "velocity-projection")!.mass;
      const after = stages.find(row => row.stage === "candidate-transfer")!.mass;
      assert.ok(Math.abs(after - before) / before < 1e-6, "physical mass survives restriction");
      t.diagnostic(JSON.stringify({ maximumGammaError, massChange_m3: after - before }));
    } finally { await rm(directory, { recursive: true, force: true }); }
  },
);
