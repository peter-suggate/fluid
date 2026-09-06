import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";

const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;
const execute = promisify(execFile);

dawnTest("mixed coarse-first half-pool represents the same flat waterline at reset", {
  timeout: 180_000,
}, async () => {
  // Child owns the repository-wide GPU lease; the parent only reads its receipt.
  const output = await mkdtemp(join(tmpdir(), "cm12-pool-reset-"));
  try {
    await execute(process.execPath, ["--max-old-space-size=12288", "--import", "tsx",
      "tools/probe-pool-reset-surface-dawn.ts"], {
      cwd: process.cwd(), env: { ...process.env, RESET_OUTPUT: output },
      timeout: 160_000, maxBuffer: 4 * 1024 * 1024,
    });
    const directory = join(output, "step-0");
    const plan = JSON.parse(await readFile(join(directory, "source.json"), "utf8"));
    const [nx, ny, nz] = plan.sampleDimensions as [number, number, number];
    const floats = async (name: string) => {
      const bytes = await readFile(join(directory, name));
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    };
    const phi = await floats("phi.bin");
    const density = await floats("density.bin");
    const width = await readFile(join(directory, "width.bin"));
    const waterline = 16;
    const at = (x: number, y: number, z: number) => x + nx * (y + ny * z);
    const widths = new Set<number>();
    let maximumError = 0;
    let maximumDistanceError = 0;
    for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
      const below = at(x, waterline - 1, z), above = at(x, waterline, z);
      assert.ok(Number.isFinite(phi[below]) && Number.isFinite(phi[above]),
        `the complete pool surface must be published at ${x},${z}`);
      assert.equal(density[below], 1, "pool has the same full cell below every crossing");
      assert.equal(density[above], 0, "pool has the same empty cell above every crossing");
      widths.add(width[below]!); widths.add(width[above]!);
      assert.ok(phi[below]! < 0 && phi[above]! > 0);
      const crossing = waterline - .5 - phi[below]! / (phi[above]! - phi[below]!);
      maximumError = Math.max(maximumError, Math.abs(crossing - waterline));
      maximumDistanceError = Math.max(maximumDistanceError,
        Math.abs(phi[below]! + .025), Math.abs(phi[above]! - .025));
    }
    assert.ok(widths.size > 1 && widths.has(1),
      `must exercise resolved and coarse representations: ${[...widths]}`);
    assert.ok(maximumError < .01,
      `mixed representation displaced the reset surface by ${maximumError} finest cells`);
    assert.ok(maximumDistanceError < .0001,
      `mixed representation changed the planar signed-distance scale by ${maximumDistanceError} m`);
  } finally {
    await rm(output, { recursive: true, force: true });
  }
});
