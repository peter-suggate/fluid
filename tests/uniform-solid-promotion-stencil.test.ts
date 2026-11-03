import assert from "node:assert/strict";
import test from "node:test";
import { createRigidCouplingOracleScene } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { createUniformSolver, readUniformFields, withUniformDevice } from "./helpers/uniform-geometric";

const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
// Liquid-conditional solid promotion (uniform-mixed-dynamic.ts solidPromote) sized to what the coupling reads.
// A thin rigid plate, two cells thick and spanning the tank, holds liquid on its -x side and moves away from it
// (+x) with the velocity its pose changes by. The dry side must stay dry: neither a cut tile nor a coarse
// neighbour sharing its vertices may interpolate liquid across the plate. A body's support reaches the tiles its
// swept shape can (markTiles); a wet tile two tiles from the plate and from every static solid is not coupled to
// it and may run 4h: it carries no coupling term, and the vertices it shares hold no wall.
test("a thin moving plate keeps its dry side dry while uncoupled water beside it coarsens", gpu, async t => {
  await withUniformDevice("solid promotion stencil", async device => {
    const scene = createRigidCouplingOracleScene("rigid-hydrostatic"), h = .025, speed = .1;
    scene.voxelDomain.finestCellSize_m = h;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
    scene.duration_s = 2;
    scene.container.fillFraction = 0;
    scene.fluid.dynamicViscosity_Pa_s = 0;
    scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -.4, y: 0, z: -.4 }, max_m: { x: -.15, y: .425, z: .4 } }];
    // Cells 10.5..12.5 along x at t = 0: a partial cut on each side, straddling no tile plane yet.
    scene.rigidBodies = [{ ...scene.rigidBodies[0]!, id: "plate", name: "plate", shape: "box", motion: "static",
      dimensions_m: { x: 2 * h, y: 1.2, z: 1.2 }, position_m: { x: -.1125, y: .4, z: 0 } }];
    const bodies = initializeRigidBodies(scene.rigidBodies), plate = bodies[0]!;
    plate.linearVelocity_m_s = { x: speed, y: 0, z: 0 };
    const solver = await createUniformSolver(device, scene);
    try {
      let coarsened = 0;
      for (let step = 1; step <= 90; step++) {
        plate.position_m = { x: -.1125 + speed * step / 60, y: .4, z: 0 };
        assert.ok(solver.advanceTo(step / 60, bodies)); await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        if (step % 10 !== 0) continue;
        const f = await readUniformFields(device, solver), { nx, ny, nz } = solver.info;
        // The plate's +x face, in cells; one cell of slack for its partial cell.
        const face = (plate.position_m.x + h + .4) / h, first = Math.ceil(face) + 1;
        let inside = 0, outside = 0;
        for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
          const v = f.density[x + nx * (y + ny * z)]!; assert.ok(Number.isFinite(v));
          if (x >= first) outside += v; else if (x < Math.floor(face) - 3) inside += v;
        }
        assert.ok(inside > 3000, `step ${step}: the liquid is still there (${inside} cells)`);
        assert.ok(outside < 1e-4, `step ${step}: ${outside} cell volumes crossed the moving plate`);
        // Tile (1, 1, 4): deep water, a tile and more from the plate, no static solid near it.
        if (f.widthAt(5, 5, 17) === 4) coarsened++;
        t.diagnostic(`step ${step}: plate face ${face.toFixed(2)} cells, dry side ${outside}, deep uncoupled tile ${f.widthAt(5, 5, 17) === 4 ? "4h" : "h"}`);
      }
      assert.ok(coarsened >= 6, `the uncoupled deep tile ran 4h on ${coarsened} of 9 reads: the promotion stencil reaches past the body's support`);
    } finally { solver.destroy(); }
  });
});
