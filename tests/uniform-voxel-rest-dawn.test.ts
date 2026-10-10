import assert from "node:assert/strict";
import test from "node:test";
import { createUniformTroughScene } from "../lib/core/uniform-trough-scenes";
import { advanceUniform, createUniformSolver, readUniformFields, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedTexture } from "./helpers/uniform-mixed-native-fields";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("a resting pool over a voxel step does not generate edge currents", { timeout: 420_000 }, async () => {
  await withUniformDevice("Voxel step hydrostatics", async device => {
    const scene = createUniformTroughScene("settled-tank", 0.05);
    const waterline = 0.431;
    scene.container.width_m = scene.container.height_m = scene.container.depth_m = 0.8;
    scene.container.fillFraction = waterline / 0.8;
    scene.fluid.initialLiquidVolumes = [];
    scene.solidVoxels = [
      { operation: "fill", minimum: [0, 0, 0], maximumExclusive: [16, 4, 16] },
      { operation: "fill", minimum: [8, 4, 0], maximumExclusive: [16, 12, 16] },
    ];
    // The riser crosses the waterline. A submerged tread alone misses the
    // bug: its false normal is parallel to gravity and Newton can reject it.
    // Here the old stencil finds a zero inside the wall, producing ~4 mm of
    // surface error and ~0.055 m/s of motion in an initially motionless pool.
    // Isolate the boundary discretization from tolerance-based early exit.
    const solver = await createUniformSolver(device, scene, { pressureResidualTolerance: 0 });
    try {
      const initial = await readUniformFields(device, solver);
      const mass = initial.density.reduce((a, b) => a + b, 0);
      for (let frame = 1; frame <= 30; frame++) {
        await advanceUniform(solver, frame / 30);
        if (frame !== 1 && frame !== 30) continue;
        const fields = await readUniformFields(device, solver);
        const velocity = await readMixedTexture(device, solver.velocityTexture);
        let maximum = 0;
        for (let i = 0; i < fields.density.length; i++) {
          if (fields.density[i]! > 0.05) maximum = Math.max(maximum, Math.hypot(...velocity.subarray(4*i, 4*i+3)));
        }
        assert.ok(maximum < 1e-4, `frame ${frame}: unforced edge current ${maximum} m/s`);
        assert.ok(Math.abs(fields.density.reduce((a, b) => a + b, 0) - mass) < 1e-3, "rest preserves liquid volume");
        // The vertical riser meets the free surface; its boundary vertices must retain the plane.
        for (const [x, y, z] of [[8, 7, 8], [8, 8, 8], [8, 9, 8], [7, 8, 8]] as const) {
          assert.ok(Math.abs(fields.vertex(x, y, z) - (y * 0.05 - waterline)) < 1e-4,
            `frame ${frame}: preserve the liquid distance at step vertex ${x},${y},${z}`);
        }
      }
    } finally { solver.destroy(); }
  });
});
