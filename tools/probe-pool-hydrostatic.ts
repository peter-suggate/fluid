/** Diagnose the existing pool-rest threshold with explicit pressure stop policies.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-pool-hydrostatic.ts
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { createUniformSolver, readUniformFields, readUniformLiveAddress, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";

paceDawnEventPolling();
const output = "artifacts/svo-build-edit/pool-hydrostatic.json";
const report = { scene: "backyard-pool-party", harness: "direct production solver with stationary initial rigid poses, matching pool-rest test",
  frames: 120, dt_s: 1 / 60, originalRestVelocityThreshold_m_s: 1e-4,
  arms: [] as Record<string, unknown>[], gpuErrors: [] as string[], completed: false };
const save = () => { mkdirSync("artifacts/svo-build-edit", { recursive: true }); writeFileSync(output, JSON.stringify(report, null, 2) + "\n"); };
save();
try {
  await withUniformDevice("pool hydrostatic pressure comparison", async device => {
    device.addEventListener("uncapturederror", event => report.gpuErrors.push(event.error.message));
    for (const bandExit of ["returns", "strict"] as const) {
      const values = resolveMethodValues(uniformNarrowBandMethod, "balanced", { ...uniformNarrowBandMethod.appDefaults, timeStep: "scene", bandExit });
      const rows: Record<string, unknown>[] = [];
      const arm: Record<string, unknown> = { bandExit, methodValues: values, rows, completed: false };
      report.arms.push(arm); save();
      console.log(JSON.stringify({ phase: "initialize", bandExit }));
      const began = performance.now(), scene = createBackyardPoolPartyScene(), bodies = initializeRigidBodies(scene.rigidBodies);
      const solver = await createUniformSolver(device, scene, { bandExit });
      arm.initialization_ms = performance.now() - began;
      async function capture(frame: number) {
        const address = await readUniformLiveAddress(device, solver);
        const velocity = await readMixedTexture(device, solver.velocityTexture, address);
        let peak = 0, peakIndex = 0;
        for (let i = 0; i < velocity.length; i++) if (i % 4 !== 3) {
          assert.ok(Number.isFinite(velocity[i]), `frame ${frame}: finite velocity`);
          if (Math.abs(velocity[i]!) > peak) { peak = Math.abs(velocity[i]!); peakIndex = i; }
        }
        const fields = await readUniformFields(device, solver), { nx, ny } = solver.info;
        let volumeCells = 0;
        for (const value of fields.density) { assert.ok(Number.isFinite(value)); volumeCells += value; }
        let low = Infinity, high = -Infinity, maxWaterlineError = 0, sampledColumns = 0;
        for (const x of [81, 84, 120, 160, 200, 236, 239]) for (const z of [81, 84, 100, 120, 156, 159]) {
          for (let y = 28; y < 44; y++) {
            const a = fields.vertex(x, y, z), b = fields.vertex(x, y + 1, z);
            if (a <= 0 && b >= 0 && a < b) {
              const height = .05 * (y + a / (a - b));
              low = Math.min(low, height); high = Math.max(high, height);
              maxWaterlineError = Math.max(maxWaterlineError, Math.abs(height - 1.8)); sampledColumns++; break;
            }
          }
        }
        const cell = Math.floor(peakIndex / 4);
        const row = { frame, time_s: frame / 60, peakVelocityComponent_m_s: peak,
          peakCell: [cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny))], peakAxis: peakIndex % 4,
          passesOriginalRestThreshold: peak < 1e-4, representedVolume_m3: volumeCells * .05 ** 3,
          waterlineMinimum_m: low, waterlineMaximum_m: high, maximumWaterlineError_m: maxWaterlineError,
          sampledColumns, particles: solver.narrowBandFlipInfo?.particles,
          pipelineError: solver.info.simulationPipelineError ?? null };
        rows.push(row); save(); console.log(JSON.stringify({ bandExit, ...row }));
      }
      try {
        await capture(0);
        for (let step = 1; step <= 120; step++) {
          assert.ok(solver.advanceTo(step / 60, bodies)); await solver.awaitFrameCompletion();
          assert.equal(solver.info.simulationPipelineError, undefined);
          if ([1, 10, 60, 120].includes(step)) await capture(step);
        }
        arm.completed = true; save();
      } finally { solver.destroy(); }
    }
  });
  report.completed = true;
} finally { save(); }
