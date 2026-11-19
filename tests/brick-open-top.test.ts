// LANE E OWNS THIS FILE. The open top (M3, lane E 2026-11-17; brick-scene.ts BRICK_OPEN_TOP_HEADROOM): a container
// 32 x 16 x 32 cells with an open top runs in a 32 x 64 x 32 lattice. The 48 cells above the container are air in the
// box. The x and z walls rise with that air, and the lattice's top is a lid that liquid must never meet.
//
// 1. The run-up. A dam column the container's full height (x 0..8, y 0..16, full z) collapses onto the far wall. Its
//    run-up climbs past y = 16, which a closed top's lid would stop. Each step (40, under the 60-step probe), the test
//    checks: no FAIL latch; no liquid in the lattice's last brick layer; the volume drift is under DRIFT, which is loose
//    because of F's floor-sheet gain (tests/brick-padded-dam.test.ts). Over the run, liquid must stand above y = 16.
// 2. The headroom's latch. A blob thrown upward at 8 m/s reaches the lattice's last brick layer (y >= 56, 1.4 m)
//    within 20 steps. The engine must latch FAIL.REACH with info bit 31 and name the open top, never meet the lid
//    silently.
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { solidVoxelsOnMovedLattice } from "../lib/core/scene-lattice";
import type { SceneDescription } from "../lib/core/model";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import { BRICK_OPEN_TOP_HEADROOM } from "../lib/methods/narrow-band/brick-scene";
import { BrickPresentation } from "../lib/methods/narrow-band/brick-present";
import { withUniformDevice } from "./helpers/uniform-geometric";

const H_M = 0.025, DRIFT = Number(process.env.OPEN_TOP_DRIFT ?? 0.1);

function openTopScene(liquid: { lo: [number, number, number]; hi: [number, number, number] }, velocity?: { x: number; y: number; z: number }) {
  const definition = getSceneDefinition("minimal-power-dam-break-32"), base = structuredClone(sceneDocument(definition)), scene: SceneDescription = structuredClone(base);
  scene.container = { ...scene.container, width_m: 32 * H_M, height_m: 16 * H_M, depth_m: 32 * H_M, fillFraction: 0, top: "open" };
  scene.voxelDomain.finestCellSize_m = H_M; scene.solidVoxels = solidVoxelsOnMovedLattice(base, scene);
  // Cells to metres: x and z from the container's centre, y from its floor.
  const m = (c: readonly number[]) => ({ x: (c[0]! - 16) * H_M, y: c[1]! * H_M, z: (c[2]! - 16) * H_M });
  scene.fluid.initialLiquidVolumes = [{ shape: "box", min_m: m(liquid.lo), max_m: m(liquid.hi) }];
  if (velocity) scene.fluid.initialVelocity_m_s = velocity;
  const profile = definition.methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
  const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
  return { scene, values, quality, dt };
}

async function solverFor(device: GPUDevice, s: ReturnType<typeof openTopScene>) {
  return createBrickSolver(device, s.scene, s.values, s.quality);
}

// The volume fractions over the whole lattice: the total, the liquid above the container's top, and in the lattice's
// last brick layer. The solver's own presentation spans the container (the renderer's extent), so the headroom is read
// through a presentation of the lattice, made once a solver.
const lattices = new WeakMap<object, BrickPresentation>();
async function volumes(device: GPUDevice, solver: Awaited<ReturnType<typeof createBrickSolver>>, top: number) {
  const e = solver.brickEngine;
  let full = lattices.get(solver);
  if (!full) { full = await BrickPresentation.create(e, (solver as unknown as { options: { farPhi_m: number } }).options.farPhi_m); lattices.set(solver, full); }
  await e.fence();
  const encoder = device.createCommandEncoder(); full.encode(encoder); device.queue.submit([encoder.finish()]);
  const tex = full.volume, [w, hgt, dep] = [tex.width, tex.height, tex.depthOrArrayLayers], bpr = Math.ceil(4 * w / 256) * 256;
  const buf = device.createBuffer({ size: bpr * hgt * dep, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = device.createCommandEncoder(); enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr, rowsPerImage: hgt }, [w, hgt, dep]); device.queue.submit([enc.finish()]);
  await buf.mapAsync(GPUMapMode.READ); const f = new Float32Array(buf.getMappedRange().slice(0)); buf.unmap(); buf.destroy();
  let total = 0, above = 0, last = 0, peak = 0;
  for (let z = 0; z < dep; z++) for (let y = 0; y < hgt; y++) for (let x = 0; x < w; x++) {
    const v = f[(z * hgt + y) * bpr / 4 + x]!; total += v;
    if (y >= top) above += v;
    if (y >= hgt - 8) last += v;
    if (v > 0.01) peak = Math.max(peak, y + 1);
  }
  return { total, above, last, peak, dims: [w, hgt, dep] };
}

test("brick engine: an open top's run-up climbs past the container's top into the headroom, with no lid there, 40 steps", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("brick open top", async device => {
    // 15 tall, a cell under the container's top: the old engine's φ continues a box that reaches the top face up past
    // it (as through a wall), and brickInitialPhi reads that φ verbatim, so a full-height column would start taller.
    const s = openTopScene({ lo: [0, 0, 0], hi: [8, 15, 32] });
    const solver = await solverFor(device, s), e = solver.brickEngine;
    try {
      assert.deepEqual([...e.layout.lattice], [32, 32 + BRICK_OPEN_TOP_HEADROOM, 32]);
      assert.deepEqual([...e.layout.box], [...e.layout.lattice], "the box rises to the lattice's top");
      assert.equal(e.layout.openTop, true);
      const v0 = await volumes(device, solver, 16);
      assert.deepEqual(v0.dims, [32, 64, 32], "the lattice's presentation covers the headroom");
      const shown = solver.brickPresentation.volume;
      assert.deepEqual([shown.width, shown.height, shown.depthOrArrayLayers], [32, 16, 32], "the renderer's presentation spans the container");
      assert.deepEqual(solver.info.tankCells, [32, 16, 32]);
      assert.ok(Math.abs(v0.total - 8 * 15 * 32) <= 0.02 * 8 * 15 * 32, `the column's volume ${v0.total}`);
      let above = 0, peak = 0, at = 0;
      for (let step = 1; step <= 40; step++) {
        assert.ok(solver.advanceTo(step * s.dt, []), `step ${step} admitted`);
        await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.();
        const v = await volumes(device, solver, 16), drift = (v.total - v0.total) / v0.total;
        if (v.above > above) { above = v.above; at = step; }
        peak = Math.max(peak, v.peak);
        t.diagnostic(`step ${step}: V ${v.total.toFixed(1)}, above y = 16 ${v.above.toFixed(2)}, the last layer ${v.last.toExponential(2)}, top ${v.peak}, drift ${drift.toExponential(2)}`);
        assert.equal(v.last, 0, `step ${step}: liquid in the lattice's last brick layer`);
        assert.ok(Math.abs(drift) <= DRIFT, `step ${step}: volume drift ${drift}`);
      }
      t.diagnostic(`peak volume above the container's top ${above.toFixed(2)} cells at step ${at}; the highest wet cell row ${peak}`);
      assert.ok(above > 1, `the run-up stands above the container's top (${above} cells): no lid at y = 16`);
    } finally { lattices.get(solver)?.destroy(); solver.destroy(); }
  }, ["subgroups"]);
});

test("brick engine: liquid thrown into an open top's last brick layer latches FAIL.REACH (info bit 31), never meets the lid", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("brick open top latch", async device => {
    const s = openTopScene({ lo: [12, 0, 12], hi: [20, 6, 20] }, { x: 0, y: 8, z: 0 });
    // Pools to spare (the lattice's 5x9x5 bricks with the skirt, its 2x3x2 pages): the pool rule sizes a blob at rest,
    // and this one is thrown.
    s.scene.fluid.brickPools = { f: 225, c: 12, b: 16, k: 0, particles: 65536 };
    const solver = await solverFor(device, s);
    try {
      let failure: Error | undefined, step = 0;
      while (!failure && step < 30) {
        step++;
        assert.ok(solver.advanceTo(step * s.dt, []), `step ${step} admitted`);
        try { await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.(); }
        catch (err) { failure = err as Error; }
        if (!failure) { const v = await volumes(device, solver, 16); t.diagnostic(`step ${step}: top ${v.peak}, above y = 16 ${v.above.toFixed(2)}`); }
      }
      assert.ok(failure, "the blob latched before step 30");
      t.diagnostic(`step ${step}: ${failure.message}`);
      assert.match(failure.message, /REACH/);
      assert.match(failure.message, /open top/);
    } finally { lattices.get(solver)?.destroy(); solver.destroy(); }
  }, ["subgroups"]);
});
