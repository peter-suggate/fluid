/** The 16h bulk tier's gates on reservoir-seep (W4, docs/plans/tiered-adaptivity-huge-grids-2026-10-24.md).
 * Seep discharge over time (liquid volume past the wall), the jet's landing
 * (the furthest liquid x on the catch floor), volume drift, max speed, region
 * tiers and the bulk owner counts.
 * node --import tsx tools/probe-reservoir-bulk.ts [--cell=0.005] [--steps=1200] [--every=60]
 *   [--values={"bulkTier":"16h"}] [--closed] [--out=PATH.json]
 * --closed fills the throat: the hydrostatic rest gate. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import "../lib/methods";
import { createUniformSolver, readUniformFields, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocumentAtLattice } from "../lib/core/scene-definition";
import { RESERVOIR_SEEP_ID } from "../lib/core/reservoir-seep-scene";
import type { MethodParamValues } from "../lib/core/method-contract";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const cell = Number(option("cell") ?? 0.005), steps = Number(option("steps") ?? 1200), every = Number(option("every") ?? 60);
const closed = process.argv.includes("--closed");
const values: MethodParamValues = JSON.parse(option("values") ?? "{}");
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const definition = getSceneDefinition(RESERVOIR_SEEP_ID);
const { scene } = sceneDocumentAtLattice(definition, { cellSize_m: cell });
if (closed) {
  // The rest gate: the throat and its mouths filled, the passage's wetted
  // start dropped (it would sit inside the wall).
  scene.solidVoxels = scene.solidVoxels!.filter(v => v.operation === "fill");
  scene.fluid.initialLiquidVolumes = scene.fluid.initialLiquidVolumes!.slice(0, 1);
}
scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
const tag = Object.entries(values).map(([k, v]) => `${k}-${v}`).join("_") || "default";
const out = resolve(option("out") ?? `artifacts/reservoir-bulk/${closed ? "rest" : "seep"}-${cell * 1000}mm-${tag}.json`);
const rows: Record<string, unknown>[] = [];
await withUniformDevice("reservoir bulk probe", async device => {
  device.addEventListener("uncapturederror", event => console.error("GPU validation:", event.error.message));
  const solver = await createUniformSolver(device, scene, { ...definition.methodProfile!.overrides, ...values });
  const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
  const { nx, ny, nz } = solver.info;
  const high = 16 * Math.floor(nx / 32);
  const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const words = async (buffer: GPUBuffer, offset: number) => {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, offset, read, 0, 32);
    device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
    const w = [...new Uint32Array(read.getMappedRange()).slice(0, 8)]; read.unmap(); return w;
  };
  let wall = 0;
  try {
    for (let step = 1; step <= steps; step++) {
      const t = performance.now();
      assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, []));
      await solver.awaitFrameCompletion();
      wall += performance.now() - t;
      assert.equal(solver.info.simulationPipelineError, undefined);
      if (step % every !== 0 && step !== steps) continue;
      const tiers = (await words(frame.heat.tiers, 4 * frame.heat.regions)).slice(0, 3);
      const bulk = frame.bulk ? await frame.bulk.readCounts() : undefined;
      const stats = await solver.readStats();
      const { density: volume } = await readUniformFields(device, solver);
      let escaped = 0, front = -1, total = 0, moment = 0;
      const floor = Math.max(2, Math.round(0.02 / cell));
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const v = volume[x + nx * (y + ny * z)]!;
        total += v;
        if (x >= high) { escaped += v; moment += v * x; if (y < floor && v > 0.5 && x > front) front = x; }
      }
      const row = { step, t_s: step * scene.numerics.fixedDt_s, wall_ms: wall / step, tiers, bulk,
        escaped_L: escaped * cell ** 3 * 1000, front_m: front < 0 ? null : (front + 0.5) * cell, mean_x_m: escaped > 0 ? (moment / escaped + 0.5) * cell : null,
        volume: total, drift: stats.volumeDrift, maxSpeed: stats.maxSpeed_m_s, fine: solver.info.uniformMixedFineTiles,
        residual: stats.uniformPressureAcceptedResidual };
      rows.push(row); console.log(JSON.stringify(row));
    }
  } finally { read.destroy(); solver.destroy(); }
});
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), cell, steps, closed, values, rows }, null, 2) + "\n");
console.log(JSON.stringify({ out }));
