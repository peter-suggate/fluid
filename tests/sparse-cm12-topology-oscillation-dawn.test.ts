import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

interface Row {
  step: number; mass: number; symmetry: number; changed: number; cells: number;
  fault: number; commitFailed: boolean;
  stages: Record<string, { faceKinetic: number; kinetic: number; momentum: number[]; mass: number }>;
}
// Each child owns the ordinary exclusive WebGPU lease. Keeping the arms in
// separate processes prevents device/compiler history from contaminating A/B.
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("repeated topology round trips retain a representable staggered mode", { timeout: 180_000 }, async t => {
  const directory = await mkdtemp(join(tmpdir(), "cm12-topology-roundtrip-"));
  try {
    const run = async (arm: string): Promise<Row[]> => {
      const output = join(directory, arm);
      await new Promise<void>((resolve, reject) => {
        const child = spawn(process.execPath, ["--import", "tsx", "tools/probe-topology-oscillation-dawn.ts"], {
          cwd: process.cwd(), env: { ...process.env, TOPOLOGY_ARM: arm, TOPOLOGY_STEPS: "16",
            TOPOLOGY_TRANSFER_ONLY: "1", TOPOLOGY_OUTPUT: output }, stdio: ["ignore", "pipe", "pipe"],
        });
        let log = "";
        child.stdout.on("data", chunk => { log += chunk; });
        child.stderr.on("data", chunk => { log += chunk; });
        child.on("error", reject);
        child.on("exit", code => code === 0 ? resolve() : reject(new Error(`arm ${arm}: ${code}\n${log}`)));
      });
      return JSON.parse(await readFile(join(output, "trace.json"), "utf8"));
    };
    const fixed = await run("fixed"), oscillating = await run("oscillate");
    assert.deepEqual(oscillating.slice(0, 3), fixed.slice(0, 3), "identical initial state and warm-up");
    assert.ok(oscillating.slice(3).every(row => row.changed > 0), "every oscillation commits");
    assert.ok(fixed.every(row => row.changed === 0), "control topology stays fixed");
    assert.equal(oscillating.at(-1)!.cells, fixed.at(-1)!.cells, "compare the same final discretization");
    for (const row of oscillating) {
      assert.equal(row.mass, 4096); assert.equal(row.symmetry, 0);
      assert.equal(row.fault, 0); assert.equal(row.commitFailed, false);
    }
    const reference = fixed.at(-1)!.stages["candidate-transfer"]!;
    const final = oscillating.at(-1)!.stages["candidate-transfer"]!;
    const retention = final.faceKinetic / reference.faceKinetic;
    // Remaining differences include ordinary transport/projection on alternate
    // grids. The former collocated transfer retains only 0.65% in this test.
    assert.ok(retention > .98 && retention < 1.02, `staggered energy retention ${retention}`);
    assert.ok(Math.abs(final.kinetic / reference.kinetic - 1) < .02);
    t.diagnostic(JSON.stringify({ reference: reference.faceKinetic, oscillating: final.faceKinetic, retention }));
  } finally { await rm(directory, { recursive: true, force: true }); }
});
