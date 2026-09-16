import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import test from "node:test";

// The production probe owns the exclusive WebGPU lease in its child process.
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)(
  "Figure 2 coarsens through free fall instead of refining with falling speed",
  { timeout: 120_000 }, async () => {
    const directory = await mkdtemp(join(tmpdir(), "fluid-figure2-coarsening-"));
    const output = join(directory, "receipt.json");
    try {
      await promisify(execFile)(process.execPath, ["--import", "tsx",
        "tools/probe-sparse-geometric-sharpening-dawn.ts", "--scene=cm12-figure-2",
        "--steps=20", `--output=${output}`], { timeout: 110_000 });
      const receipt = JSON.parse(await readFile(output, "utf8")) as {
        completed: boolean; validationErrors: string[];
        checkpoints: Array<{ step: number; phiNegativeFineCells: number;
          volume: { acceptedCells: number; volumeFine3: number };
          adaptivity: { resolutions: Record<string, number> } }>;
      };
      assert.equal(receipt.completed, true);
      assert.deepEqual(receipt.validationErrors, []);
      const initial = receipt.checkpoints.find(c => c.step === 0)!;
      const middle = receipt.checkpoints.find(c => c.step === 10)!;
      const falling = receipt.checkpoints.find(c => c.step === 20)!;
      assert.ok(middle.volume.acceptedCells < initial.volume.acceptedCells,
        "an undeformed falling circle must recover a coarser grid");
      assert.ok(falling.volume.acceptedCells <= middle.volume.acceptedCells,
        "increasing free-fall speed must not erase recovered coarsening");
      assert.equal(falling.adaptivity.resolutions["8"], 0,
        "a broad circle in free fall must not be held at the finest rung");
      assert.ok(Math.abs(falling.volume.volumeFine3 / initial.volume.volumeFine3 - 1) < 1e-4,
        "coarsening must preserve transported volume");
      assert.ok(falling.phiNegativeFineCells > 0.9 * initial.phiNegativeFineCells,
        "coarsening must not obtain its savings by deleting the circle");
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  },
);
