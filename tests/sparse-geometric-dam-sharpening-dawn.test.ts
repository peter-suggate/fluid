import assert from "node:assert/strict";
import test from "node:test";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

// The child probe owns the WebGPU lease; this test must not acquire it twice.
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("dam-break retains volume and a bounded halo through 5.3 seconds", { timeout: 180_000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), "fluid-dam-sharpening-"));
  try {
    const output = join(directory, "receipt.json");
    await promisify(execFile)(process.execPath, ["--import", "tsx",
      "tools/probe-sparse-geometric-sharpening-dawn.ts", "--scene=water-box-dam-break",
      "--steps=159", `--output=${output}`], { timeout: 170_000, maxBuffer: 4 * 1024 * 1024 });
    const report = JSON.parse(await readFile(output, "utf8"));
    assert.equal(report.completed, true); assert.deepEqual(report.validationErrors, []);
    const mass = report.checkpoints[0].volume.volumeFine3;
    for (const frame of report.checkpoints) {
      assert.equal(frame.transportFault, 0);
      assert.equal(frame.volume.nonfiniteCells, 0); assert.equal(frame.volume.invalidCells, 0);
      assert.ok(Math.abs(frame.volume.volumeFine3 - mass) / mass < 1e-5, `mass at ${frame.step}`);
    }
    const final = report.checkpoints.at(-1);
    assert.equal(final.step, 159);
    // The reviewed budget-only version had 43% air-side mass and 18% beyond
    // phi=h. These bounds require a material improvement, not bitwise replay.
    assert.ok(final.airSideVolume / mass < 0.25, `air-side fraction ${final.airSideVolume / mass}`);
    assert.ok(final.deepAirVolume / mass < 0.075, `deep-air fraction ${final.deepAirVolume / mass}`);
    assert.ok(Math.abs(final.phiNegativeFineCells - mass) / mass < 0.05,
      `phi encloses ${final.phiNegativeFineCells} centre samples for volume ${mass}`);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
