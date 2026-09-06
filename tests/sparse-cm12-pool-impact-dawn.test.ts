import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("max1 half-pool impact has no late column-switch ridges", { timeout: 600_000 }, async () => {
  const output = await mkdtemp(join(tmpdir(), "cm12-pool-impact-"));
  try {
    // The child owns the repository WebGPU lease and retains its native GPU.
    await promisify(execFile)(process.execPath, ["--max-old-space-size=12288", "--import", "tsx",
      "tools/probe-pool-impact-ab-dawn.ts"], {
      env: { ...process.env, POOL_MAX_CELL: "1", POOL_STEPS: "197", POOL_DT: String(1 / 60),
        POOL_OVERRIDES: "{}", POOL_OUTPUT: output, POOL_VERIFY_SURFACE: "1" },
      timeout: 540_000, maxBuffer: 4 * 1024 * 1024,
    });
    const receipt = JSON.parse(await readFile(join(output, "surface-regression.json"), "utf8"));
    assert.equal(receipt.step, 197);
    assert.ok(receipt.maxReferenceErrorFineCells < 1 / 1024);
  } finally {
    await rm(output, { recursive: true, force: true });
  }
});
