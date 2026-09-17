/** Native-equivalent split-dam receipts for all browser Wasm artifacts.
 * node --import tsx tools/wasm/split-resolution-levelset-probe.ts [output.json]
 */
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";
import { decodePhysicsPublication, PhysicsPlane } from "../../lib/physics-wasm/publication";
import { parsePhysicsReceipt } from "../../lib/physics-wasm/protocol";
import { loadFluidWasmForNode } from "./load-module.mjs";

const seed = JSON.parse(await readFile(new URL(
  "../../rust/core/testdata/split-resolution-ladder-seed.json", import.meta.url), "utf8"));
const results: Record<string, Record<string, {
  frame: number; front: number[]; volume: number; overcapacity: number; phiArea: number;
}[]>> = {};
for (const artifact of ["scalar", "simd", "threaded"] as const) {
  const wasm = await loadFluidWasmForNode(undefined, { artifact });
  const lanes: typeof results[string] = {};
  for (const [left, right] of [[0, 0], [2, 1], [1, 2], [1, 1], [2, 2]]) {
    const world = wasm.FluidWorld.from_scene(JSON.stringify(seed.scene), JSON.stringify({
      runEpoch: 1, commandSequence: 0, pressureIterations: 256,
      pressureRelativeTolerance: 1e-6, transportExperiment: "level-set-volume",
      production: { dtS: 1 / 30, timeStep: "paper" },
    }));
    const frames: typeof lanes[string] = [];
    try {
      if (left !== 0 || right !== 0) world.apply_command(JSON.stringify({ type: "set-refinement-regions",
        runEpoch: 1, commandSequence: 1, regions: [
          { minimumFine: [0, 0], maximumFine: [16, 16], minimumCellWidth: left, maximumCellWidth: left },
          { minimumFine: [16, 0], maximumFine: [32, 16], minimumCellWidth: right, maximumCellWidth: right },
        ] }));
      const lastFrame = left === 0 ? 60 : left !== right ? 120 : 20;
      for (let frame = 0; frame <= lastFrame; frame++) {
        const receipt = parsePhysicsReceipt(frame === 0 ? world.receipt() : world.advance(frame + 1, 1 / 30));
        assert.equal(receipt.fault, null);
        const decoded = decodePhysicsPublication({ id: frame, revision: receipt,
          bytes: world.snapshot(2).slice(), release() {} });
        try {
          const segments = decoded.plane(PhysicsPlane.RdfSegments) as Float32Array;
          let minimum = Infinity, maximum = -Infinity;
          for (let i = 0; i < segments.length; i += 2) {
            minimum = Math.min(minimum, segments[i]!);
            maximum = Math.max(maximum, segments[i]!);
          }
          assert.ok(Number.isFinite(minimum) && Number.isFinite(maximum));
          const lsv = receipt.levelSetVolume as Readonly<Record<string, unknown>> | undefined;
          const volume = Number(receipt.liquidMeasure);
          const overcapacity = Number(lsv?.totalVolumeOverCapacity ?? 0);
          assert.ok(Math.abs(volume - 128) < 1e-4, `${artifact} ${left}/${right} frame ${frame}: volume drift`);
          if (left !== right && [20, 60, 120].includes(frame)) {
            assert.ok(overcapacity < 0.1, `${artifact} ${left}/${right} frame ${frame}: persistent overload ${overcapacity}`);
          }
          frames.push({ frame, front: [minimum, maximum], volume, overcapacity,
            phiArea: Number(lsv?.phiImpliedLiquidVolume ?? 128) });
        } finally { decoded.release(); }
      }
    } finally { world.free(); }
    lanes[`${left}${right}`] = frames;
  }
  results[artifact] = lanes;
  if (artifact !== "scalar") assert.deepEqual(lanes, results.scalar,
    `${artifact} and scalar split-dam publications differ`);
}
const output = `${JSON.stringify(results, null, 2)}\n`;
if (process.argv[2]) await writeFile(process.argv[2], output);
else process.stdout.write(output);
// The threaded artifact owns persistent worker-pool threads.
process.exit(0);
