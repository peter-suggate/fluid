import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
dawnTest("coarse-first expansion retains curved features and matches the 1³ physical control", { timeout: 60_000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), "cm12-expansion-ab-"));
  const run = async (arm: string, maximum: number) => {
    const child = spawn(process.execPath, ["--import", "tsx", "tools/probe-symmetric-coarse-first-ab-dawn.ts"], {
      env: { ...process.env, SYMMETRIC_MAX_CELL: String(maximum), SYMMETRIC_STEPS: "13",
        SYMMETRIC_DT: String(1 / 30), SYMMETRIC_OVERRIDES: "{}", SYMMETRIC_OUTPUT: join(directory, arm) },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let output = "";
    child.stdout.on("data", data => { output += data; });
    child.stderr.on("data", data => { output += data; });
    const code = await new Promise<number | null>((resolve, reject) => {
      child.once("error", reject); child.once("exit", resolve);
    });
    assert.equal(code, 0, output);
  };
  const read = async (arm: string, step: number, name: string) => {
    const bytes = await readFile(join(directory, arm, `${step}-${name}.bin`));
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  };
  try {
    // Sequential isolated processes: never hold two Dawn devices concurrently.
    await run("fine", 1); await run("adaptive", 0);
    assert.deepEqual(await read("adaptive", 0, "density"), await read("fine", 0, "density"));
    const activity = JSON.parse(await readFile(join(directory, "adaptive", "1-activity.json"), "utf8"));
    const corners = activity.bricks.filter((b: { meanDensity: number }) => b.meanDensity > .5);
    assert.equal(corners.length, 4);
    assert.ok(corners.every((b: { reasons: number }) => ((b.reasons >>> 16) & 31) === 8),
      "accepted curvature must see the collapsing block's corners; an unstaged transport cache reports B1");
    const comparisons = [];
    for (const step of [3, 5, 8, 13]) {
      const reference = await read("fine", step, "density"), actual = await read("adaptive", step, "density");
      const heightReference = await read("fine", step, "height"), heightActual = await read("adaptive", step, "height");
      const mass = reference.reduce((a, b) => a + b, 0);
      const densityL1 = actual.reduce((a, b, i) => a + Math.abs(b - reference[i]!), 0) / mass;
      const heightL1 = heightActual.reduce((a, b, i) => a + Math.abs(b - heightReference[i]!), 0) / mass;
      const heightRms = Math.sqrt(heightActual.reduce((a, b, i) => a + (b - heightReference[i]!) ** 2, 0) / heightActual.length);
      comparisons.push({ step, densityL1, heightL1, heightRms });
      assert.ok(densityL1 < .025 && heightL1 < .02, JSON.stringify(comparisons));
      if (step === 13) {
        assert.ok(heightRms < .1, JSON.stringify(comparisons));
        assert.ok(Math.abs(actual.reduce((a, b) => a + b, 0) / 2048 - 1) < .002);
      }
    }
    console.log(JSON.stringify({ comparisons }));
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
