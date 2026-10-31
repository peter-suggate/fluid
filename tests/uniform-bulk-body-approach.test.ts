import assert from "node:assert/strict";
import test from "node:test";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { UNIFORM_BULK_HOLD_FRAMES } from "../lib/methods/uniform/uniform-bulk-tier-reference";
import { uniformDetailField } from "../lib/methods/uniform/uniform-detail-fields";
import { UNIFORM_KIND, uniformPageKind } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
import { STONE_TANK, uniformStoneTank } from "./helpers/uniform-stone-tank";

/* A body that reaches a BULK16 page faster than a promote takes effect.
 * A 16h region holds one pressure, so it cannot carry a concentrated load
 * a few cells away: the band's bottom layer carries it sideways instead.
 * Measured: the stone (r 0.15 m at 3 m/s, 10.7 cells a frame) came within
 * 3 cells of the BULK16 plane in frame 5. The scan asked for the pages under
 * it at frame 5's end, and frame 6's commit would have applied that. But
 * frame 5's band projection put 242 faces at up to 74 m/s, tangential, in
 * the bottom tile layer over the BULK16 pages. Frame 6's census read them,
 * and the departure closure (65 tiles) hit its cap. With the tier off, the
 * fastest tile stays at 4-7 m/s. A body's ballistic bound over the latency
 * now fails the scan (bulkPromote) and a region's membership (bulkSelect,
 * so no page demotes within its reach).
 *
 * The census's fastest tile travel (its word 18, h a step: what the
 * departure closure covers) is the criterion: on 16h, at most 1.5x its max
 * with the tier off over the same frames. That is a regression bound against
 * the reference, not physics: one run of each, and the solver is
 * nondeterministic. Measured with the tier off: at most 24 h a step
 * (7.3 m/s). The fastest face is no criterion: a sphere's water entry throws
 * a splash jet faster than the sphere (Wagner), 13 m/s at frame 1 with the
 * tier off. */
const gpu = { skip: !process.env.WEBGPU_NODE_MODULE };
/** The regression bound over the tier-off reference's fastest tile travel. */
const REFERENCE_MARGIN = 1.5;

/** Runs the stone tank for `held` frames with the stone held still, then
 * `moving` frames moving down at 3 m/s; each moving frame checked. Returns
 * the fastest tile travel of each moving frame, m/s. */
async function stone(t: { diagnostic(message: string): void }, tier: "16h" | "off", depth_m: number | undefined, held: number, moving = 12): Promise<number[]> {
  const { scene, values } = uniformStoneTank(tier, depth_m);
  let travel: number[] = [];
  await withUniformDevice("BULK16 body approach", async device => {
    const solver: any = await createUniformSolver(device, scene, values), a = solver.address;
    try {
      const [px, py, pz] = a.pages as number[];
      let D = 0;
      for (let z = 0; z < pz!; z++) for (let y = 0; y < py!; y++) for (let x = 0; x < px!; x++) if (uniformPageKind(a, x, y, z) === UNIFORM_KIND.bulk) D++;
      if (tier === "16h") assert.ok(D > 0, "the census start holds deep pages in BULK16");
      const bodies = initializeRigidBodies(scene.rigidBodies ?? []), body = bodies[0]!, push = { ...body.linearVelocity_m_s };
      const rows: GPUBuffer = solver.negativeBoundaryVelocityBuffer;
      const dt = scene.numerics.fixedDt_s!, work: GPUBuffer = solver.mixedDynamic.work;
      // The velocity's physical storage: its h texels and its 4h base.
      const velocity: GPUTexture = solver.velocityTexture, field: any = uniformDetailField(velocity);
      const stores: GPUTexture[] = field ? [field.storage.physical(velocity), field.storage.baseOf?.(field.handle)].filter(Boolean) : [velocity];
      if (held > 0) { body.held = true; body.linearVelocity_m_s = { x: 0, y: 0, z: 0 }; }
      const faces: number[] = []; travel = [];
      for (let frame = 1; frame <= held + moving; frame++) {
        if (held > 0 && frame === held + 1) { body.held = false; body.linearVelocity_m_s = { ...push }; }
        assert.ok(solver.advanceTo(frame * dt, bodies)); await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined, `frame ${frame}`);
        if (frame <= held) continue;
        const v = await readMixedBuffer(device, rows);
        let bad = 0; for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i]!)) bad++;
        assert.equal(bad, 0, `frame ${frame}: every low-face row is finite`);
        const fastest = (await readMixedBuffer(device, work))[18]! * STONE_TANK.cell_m / dt;
        let top = 0;
        for (const store of stores) {
          const u = await readMixedTexture(device, store);
          for (let i = 0; i < u.length; i++) if ((i & 3) !== 3) top = Math.max(top, Math.abs(u[i]!));
        }
        travel.push(fastest); faces.push(top);
      }
      // The slot rule holds the body's ring and the surface band (uniformFineSlotRule's sources).
      const info = solver.info;
      if (info.uniformMixedSlotCapacity !== undefined) assert.ok(info.uniformMixedSlotPeak <= info.uniformMixedSlotCapacity, `slot peak ${info.uniformMixedSlotPeak} within the rule's ${info.uniformMixedSlotCapacity}`);
      t.diagnostic(`${tier}: slots peak ${info.uniformMixedSlotPeak} of ${info.uniformMixedSlotCapacity}`);
      t.diagnostic(`${tier}: D ${D}; fastest tile travel by moving frame (m/s): ${travel.map(s => s.toFixed(2)).join(" ")}`);
      t.diagnostic(`fastest face by moving frame (m/s): ${faces.map(s => s.toFixed(2)).join(" ")}`);
    } finally { solver.destroy(); }
  });
  return travel;
}

/** The tier-off reference, then 16h: each moving frame's fastest tile within the margin of the reference's. */
async function approach(t: { diagnostic(message: string): void }, depth_m: number | undefined, held: number): Promise<void> {
  const reference = Math.max(...await stone(t, "off", depth_m, held)), travel = await stone(t, "16h", depth_m, held);
  travel.forEach((fastest, i) => assert.ok(fastest <= REFERENCE_MARGIN * reference,
    `moving frame ${i + 1}: the fastest tile travels ${fastest.toFixed(2)} m/s, over ${REFERENCE_MARGIN} x the tier-off run's ${reference.toFixed(2)}`));
}

test("a body approaching deep water promotes its BULK16 pages before it arrives", gpu, async t => {
  await approach(t, undefined, 0);
});

// The slot rule (uniformFineSlotRule) holds a body's census ball on top of
// the surface band: the stone resting 0.4 m deep at t=0 builds, and is held
// there past the hold (UNIFORM_BULK_HOLD_FRAMES) within the rule's slots.
// Red on 639b9128: "the first layout's detail ring needs 112 h slots; the
// slot rule holds 111". Its release is not tested here: pushed from rest to
// 3 m/s, the tier-off run stops at frame 48 ("Departure travel gives a
// closure of 33 tiles, over the cap of 32").
test("a body resting deep in the water at t=0 builds within the slot rule", gpu, async t => {
  await stone(t, "16h", 0.4, UNIFORM_BULK_HOLD_FRAMES + 10, 0);
});
