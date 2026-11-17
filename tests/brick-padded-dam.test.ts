// LANE E OWNS THIS FILE. A padded dam (M3 padding, lane E 2026-11-17): a 24 x 16 x 24 container in a 32^3 lattice.
// Its low walls are at the storage edge's 0 and its high walls at the box, 8 cells short of the lattice's edge.
//
// The liquid is a slab across the middle of x, cells 8..16, at full depth z and 12 cells tall, so the scene is symmetric
// under x -> 24 - x. It collapses onto both x walls and reaches them by step 20. A wall at the box that differs from
// the wall at 0 breaks the mirror: a wall twice, a wall at the storage edge, or the padding open.
//
// Each step (30 steps, under the 60-step probe), the test checks:
// - no FAIL latch;
// - assertPaddedPressure (P's observables, not `still`): no pressure row past the box, and every box face of a liquid
//   h row is WALL, with no other WALL face;
// - the presentation volume past the box is 0;
// - the x mirror: the column sums V(x) and V(23 - x) differ by at most MIRROR of the liquid in all;
// - the volume drift is under DRIFT. This bound is loose: the moving-liquid volume gain is lane F's named open fail,
//   an under-resolved sheet on the floor. It is not the padding: the same slab unpadded (32^3, x 12..20) gains 2.0%
//   by step 4 and 5.9% at its peak. So the padding is judged by the mirror and the box, not by the drift.
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { solidVoxelsOnMovedLattice } from "../lib/core/scene-lattice";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { BRICK_ENGINE_SCENES } from "../lib/methods/narrow-band/brick-scene";
import { paddedPressure } from "./brick-padded-pressure";
import { withUniformDevice } from "./helpers/uniform-geometric";

const STEPS = 30, H_M = 0.025, MIRROR = Number(process.env.PADDED_DAM_MIRROR ?? 0.02), DRIFT = Number(process.env.PADDED_DAM_DRIFT ?? 0.1);

test("brick engine: a padded dam (24x16x24 in 32^3) keeps its walls at the box: none past it, the x mirror held, 30 steps", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("brick padded dam", async device => {
    const definition = getSceneDefinition("minimal-power-dam-break-32"), base = structuredClone(sceneDocument(definition)), scene = structuredClone(base);
    scene.container = { ...scene.container, width_m: 24 * H_M, height_m: 16 * H_M, depth_m: 24 * H_M, fillFraction: 0 };
    scene.voxelDomain.finestCellSize_m = H_M; scene.solidVoxels = solidVoxelsOnMovedLattice(base, scene);
    scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -4 * H_M, y: 0, z: -12 * H_M }, max_m: { x: 4 * H_M, y: 12 * H_M, z: 12 * H_M } }];
    const profile = definition.methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
    const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
    const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
    const enabled = BRICK_ENGINE_SCENES as string[], had = enabled.includes(scene.sceneId!);
    if (!had) enabled.push(scene.sceneId!);
    const solver = await createBrickSolver(device, scene, values, quality).finally(() => { if (!had) enabled.splice(enabled.indexOf(scene.sceneId!), 1); });
    const e = solver.brickEngine;
    // The presentation's volume fractions over the lattice, as column sums along x and the volume past the box.
    const columns = async () => {
      await e.fence();
      const tex = solver.brickPresentation.volume, [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
      const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
      await buf.mapAsync(GPUMapMode.READ); const f = new Float32Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy();
      const col = new Float64Array(24); let past = 0, total = 0;
      for (let z = 0; z < dep; z++) for (let y = 0; y < hgt; y++) for (let x = 0; x < w; x++) {
        const v = f[(z * hgt + y) * bpr / 4 + x]!; total += v;
        if (x >= 24 || y >= 16 || z >= 24) past += v; else col[x] += v;
      }
      return { col, past, total, dims: [w, hgt, dep] };
    };
    try {
      assert.deepEqual([...e.layout.lattice], [32, 32, 32]); assert.deepEqual([...e.layout.box], [24, 16, 24]);
      const c0 = await columns();
      assert.deepEqual(c0.dims, [32, 32, 32], "the presentation covers the lattice");
      t.diagnostic(`step 0: V ${c0.total.toFixed(1)}, past the box ${c0.past}, columns ${[...c0.col].map(v => v.toFixed(0)).join(" ")}`);
      assert.equal(c0.past, 0, "the boot puts no liquid past the box");
      assert.ok(Math.abs(c0.total - 8 * 12 * 24) <= 0.02 * 8 * 12 * 24, `the slab's volume ${c0.total}`);
      // The boot's own asymmetry (the old engine's t = 0 phi of the slab, brickInitialPhi, at its two faces) is the
      // mirror's floor: what the walls may add is judged past it.
      let mirror0 = 0;
      for (let x = 0; x < 12; x++) mirror0 += Math.abs(c0.col[x]! - c0.col[23 - x]!);
      mirror0 /= c0.total;
      let worst = 0, reached = 0;
      for (let step = 1; step <= STEPS; step++) {
        assert.ok(solver.advanceTo(step * dt, []), `step ${step} admitted`);
        await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.();
        const m = await paddedPressure(e, H_M), c = await columns();
        let mirror = 0;
        for (let x = 0; x < 12; x++) mirror += Math.abs(c.col[x]! - c.col[23 - x]!);
        mirror = mirror / c.total - mirror0; worst = Math.max(worst, mirror);
        const drift = (c.total - c0.total) / c0.total, wall = c.col[0]! + c.col[23]!;
        if (wall > 1) reached = reached || step;
        t.diagnostic(`step ${step}: rows h ${m.hRows} native ${m.nativeRows} past ${m.pastBox}, box faces ${m.boxFaces}, stray walls ${m.strayWalls}, peak |U| ${m.peak.toExponential(2)} m/s; V past the box ${c.past.toExponential(2)}, mirror ${mirror.toExponential(2)}, walls' columns ${c.col[0]!.toFixed(1)} / ${c.col[23]!.toFixed(1)}, drift ${drift.toExponential(2)}`);
        assert.equal(m.pastBox, 0, `step ${step}: rows past the box`);
        assert.equal(m.openBoxFace, undefined, `step ${step}: a box face of a liquid h row is not WALL`);
        assert.equal(m.strayWalls, 0, `step ${step}: WALL faces off the box planes`);
        assert.equal(c.past, 0, `step ${step}: liquid past the box`);
        assert.ok(mirror <= MIRROR, `step ${step}: the x mirror is off by ${mirror} of the liquid past the boot's`);
        assert.ok(Math.abs(drift) <= DRIFT, `step ${step}: volume drift ${drift}`);
      }
      assert.ok(reached > 0, "the slab reaches both x walls");
      t.diagnostic(`worst mirror past the boot's ${worst.toExponential(2)} (the boot's ${mirror0.toExponential(2)}); the walls reached at step ${reached}`);
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
