/** Real production frames, with and without the opt-in spatial sampler.
 * node --import tsx tools/benchmark-spatial-performance.ts [--steps=48] [--enabled]
 * WEBGPU_NODE_MODULE defaults to the repository's Dawn install.
 * Saves raw observations and the compiled, cross-validated attribution report. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { createMinimalPowerDamBreak32Scene } from "../lib/core/scenes";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { SpatialCostSnapshot } from "../lib/core/spatial-performance";
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.split("=")[1];
const steps = Number(option("steps") ?? 48), enabled = process.argv.includes("--enabled");
const out = option("out") ?? "artifacts/spatial-performance";
assert.ok(Number.isInteger(steps) && steps > 0);
await withUniformDevice("Spatial performance", async device => {
  const scene = createMinimalPowerDamBreak32Scene();scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
  if (process.argv.includes("--settled")) { scene.fluid.initialCondition = "tank-fill"; scene.container.fillFraction = 1 / 3; scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 }, max_m: { x: scene.container.width_m / 2, y: scene.container.height_m / 3, z: scene.container.depth_m / 2 } }]; }
  if (process.argv.includes("--quiet")) scene.fluid.gravity_m_s2 = {x:0,y:0,z:0};
  const build = performance.now();const solver = await createUniformSolver(device, scene);
  console.log(JSON.stringify({ construction_ms: performance.now() - build, enabled, steps }));
  solver.setSpatialPerformanceEnabled(enabled, process.argv.includes("--diagnostic"));
  const rows: object[] = [];let last: SpatialCostSnapshot | undefined;
  try {
    for (let frame = 1; frame <= steps; frame++) {
      // Identical pacing for A/B, permits one diagnostic sample each step.
      await new Promise(r => setTimeout(r, 360));
      const start = performance.now();assert.ok(solver.advanceTo(frame * scene.numerics.fixedDt_s, []));await solver.awaitFrameCompletion();
      const wall_ms = performance.now() - start;
      assert.equal(solver.info.simulationPipelineError, undefined);
      if (enabled) for (let n = 0; n < 200 && (solver.spatialPerformanceSnapshot?.report.frame ?? -1) < frame && !solver.info.spatialPerformanceError; n++) await new Promise(r => setTimeout(r, 5));
      const snapshot = solver.spatialPerformanceSnapshot;
      if (enabled) {
        assert.equal(solver.info.spatialPerformanceError, undefined, "every sampled frame must publish valid timing");
        assert.equal(snapshot?.report.frame, frame, "same-frame snapshot must arrive");
      }
      if (snapshot && snapshot !== last) {
        const report = snapshot.report;
        assert.ok(Math.abs(report.wall_ms - report.attributed_ms - report.shared_ms) < 1e-6);
        assert.ok(snapshot.cells.every(c => [...c.origin, ...c.size].every(Number.isFinite)), "finite world-space snapshot");
        assert.ok(snapshot.costs[0]!.every(v => Number.isFinite(v) && v >= 0));
        const byTier = [0, 0, 0];snapshot.cells.forEach((c, i) => byTier[c.tier]! += snapshot.costs[0]![i]!);
        rows.push({ frame, wall_ms, report, byTier, featureTotals: Object.fromEntries(Object.keys(snapshot.cells[0]?.features ?? {}).map(k => [k, snapshot.cells.reduce((s, c) => s + (c.features[k] ?? 0), 0)])) });last = snapshot;
      } else rows.push({ frame, wall_ms, error: solver.info.spatialPerformanceError });
      if (frame % 8 === 0 || frame === steps) console.log(JSON.stringify(rows.at(-1)));
    }
    if (enabled) assert.ok(last, solver.info.spatialPerformanceError ?? "No spatial capture published");
    mkdirSync(out, { recursive: true });writeFileSync(`${out}/${enabled ? "sampled" : "plain"}${process.argv.includes("--settled") ? "-settled" : ""}.json`, JSON.stringify({ enabled, steps, scene: scene.sceneId, profile: solver.spatialPerformanceProfile, rows, snapshot: last ? { report: last.report, cells: last.cells, costs: last.costs.map(c => [...c]) } : undefined }, null, 2));
  } finally { solver.destroy(); }
}, ["timestamp-query"]);
