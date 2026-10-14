/** Cold build, teardown, warm rebuild on one device; validate actual solver
 * output as well as zero native compilation on the second scene opening.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/check-solver-compilation-cache-dawn.ts [scene-id]
 * FLUID_COMPILATION_REPORT=/tmp/compilation.json writes native call timings
 * (including any driver cache hits). FLUID_CACHE_CHECK_STEPS=30 exercises a
 * longer trajectory after each opening; the default is one step.
 */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { advanceUniform, createUniformSolver, readUniformFields, withUniformDevice } from "../tests/helpers/uniform-geometric";

const sceneId = process.argv[2] ?? "hydrostatic-power-large-offset";
const steps = Number(process.env.FLUID_CACHE_CHECK_STEPS ?? 1);
assert.ok(Number.isInteger(steps) && steps >= 1 && steps <= 2400, "FLUID_CACHE_CHECK_STEPS must be an integer in [1, 2400]");
const scene = sceneDocument(getSceneDefinition(sceneId));
await withUniformDevice("solver compilation cache", async device => {
  const manager = gpuCompilationManagerFor(device);
  let initial: Float32Array | undefined;
  for (const phase of ["cold", "warm"] as const) {
    const before = manager.snapshot().cacheStats;
    const started = performance.now();
    const solver = await createUniformSolver(device, structuredClone(scene));
    const elapsed_ms = performance.now() - started;
    try {
      await manager.whenIdle();
      const after = manager.snapshot().cacheStats;
      if (phase === "cold" && process.env.FLUID_COMPILATION_REPORT) {
        await writeFile(process.env.FLUID_COMPILATION_REPORT, JSON.stringify(manager.compilationTimings(), null, 2));
      }
      const compilations = after.compilationAttempts - before.compilationAttempts;
      if (phase === "warm") assert.equal(compilations, 0, "reopening the scene must reuse every solver pipeline");
      const fields = await readUniformFields(device, solver);
      if (initial) assert.deepEqual(fields.density, initial, "cached pipelines must publish the same t=0 liquid");
      else initial = fields.density;
      for (let step = 1; step <= steps; step++) {
        await advanceUniform(solver, step * (scene.numerics?.fixedDt_s ?? 1 / 60));
      }
      const advanced = await readUniformFields(device, solver);
      assert.ok(advanced.density.every(Number.isFinite));
      assert.ok(advanced.pressure.every(Number.isFinite));
      assert.ok(advanced.phi.every(Number.isFinite));
      console.log(JSON.stringify({ sceneId, phase, elapsed_ms, compilations, steps, cache: after }));
    } finally {
      await manager.whenIdle();
      await device.queue.onSubmittedWorkDone();
      solver.destroy();
    }
  }
});
