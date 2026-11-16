/**
 * The particle banks' counts and pJoin's travel latch (the GPU wedge of 2026-11-17, sparse-cm12-long-dam-break):
 * - A move empties its destination bank in pArgs, not in the move's first lane. A move over no records dispatches no
 *   workgroup, and the destination kept the records of two steps before, whose before holds pDelta's deltas
 *   (NB_DELTA_NONE, 3e38); pJoin read those as a travel of 4e38 cells and walked 2048^3 tiles a record.
 * - pJoin latches PARTICLES (code 4 a moved record, 3 a fresh seed) on a travel at or past the lattice extent, and
 *   bounds its join box by the lattice's tiles.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { BrickLayout, FAIL, H } from "./brick-abi";
import { BrickEngine, type BrickEngineOptions } from "./brick-engine";
import type { BrickPlacement } from "./brick-kernel";
import { fluidStagesAt } from "./brick-fluid-stages";
import { compositeMGPCG } from "./brick-pressure-solver";
import type { BrickStageContext } from "./brick-stages";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const label = (p: BrickPlacement) => p.kernel.label;

function methodValues(): Record<string, unknown> {
  const method = getMethod("uniform-narrow-band-flip");
  return resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
}

test("every move is preceded by its own arguments (pArgs, w even, the same z), which empty its destination bank", () => {
  const L = new BrickLayout({ lattice: [32, 32, 32], f: 200, c: 16, b: 16, k: 0, particles: 1 << 16 });
  const context = { layout: L, dt_s: 1 / 60, h_m: 0.0125, gravity_mps2: [0, -9.81, 0], values: methodValues() } as BrickStageContext;
  for (const tiered of [false, true]) {
    const stages = fluidStagesAt(tiered)(context);
    let moves = 0;
    for (const list of [stages.move, stages.kick?.seed ?? []]) for (let i = 0; i < list.length; i++) {
      if (label(list[i]!) !== "pMove") continue;
      moves++;
      const before = list[i - 1];
      assert.ok(before && label(before) === "pArgs", `tiered ${tiered}: pMove at ${i} follows ${before ? label(before) : "nothing"}, not pArgs`);
      assert.equal((before.w ?? 0) & 1, 0, `tiered ${tiered}: the move's pArgs is a move's arguments (w even)`);
      assert.equal(before.z ?? 0, list[i]!.z ?? 0, `tiered ${tiered}: the move's pArgs reads the move's banks`);
    }
    assert.equal(moves, 2, `tiered ${tiered}: the step's move and the seeding's file`);
    const code = stages.move.find(p => label(p) === "pArgs")!.kernel.code;
    assert.ok(code.includes("if((cfg.w&1u)==0u){atomicStore(&nbSet[NB_H_PCOUNT0+(pSrc()^1u)],0u);}"), "pArgs empties the move's destination");
    assert.ok(!stages.move.find(p => label(p) === "pMove")!.kernel.code.includes("atomicStore(&nbSet[NB_H_PCOUNT0+dst],0u)"), "the move's first lane no longer resets it");
  }
});

function tinyDam(): BrickEngineOptions {
  return {
    lattice: [32, 32, 32], h_m: 0.0125, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0], capacities: { f: 200, c: 16, b: 16, k: 0, particles: 1 << 16 },
    liquid: [{ lo: [-64, -64, -64], hi: [16, 12, 96] }], solver: compositeMGPCG(), stages: fluidStagesAt(true), values: methodValues(),
    fluid: { density_kg_m3: 1000, viscosity_Pa_s: 0, surfaceTension_N_m: 0 }, label: "banks",
  };
}

test("brick particle banks (GPU): a move over an empty source bank empties its destination", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick particle banks", async device => {
    const e = await BrickEngine.create(device, tinyDam());
    try {
      await e.fence();
      const part1 = e.schedule.part1, at = part1.findIndex(p => label(p) === "pMove" && (p.z ?? 0) === 0);
      assert.ok(at > 0 && label(part1[at - 1]!) === "pArgs");
      // run() binds ring slot 0: pSrc = 0 at z = 0, so bank 0 is the source and bank 1 the destination.
      e.write("set", H.PCOUNT0, Uint32Array.of(0, 777));
      await e.run([part1[at - 1]!, part1[at]!]);
      const h = await e.read("set", 0, 256);
      assert.equal(h[H.PCOUNT0], 0, "the source stays empty");
      assert.equal(h[H.PCOUNT1], 0, "the destination's stale count is gone though the move dispatched no workgroup");
      assert.equal(h[H.FAIL], 0);
    } finally { e.destroy(); }
  });
});

test("brick particle banks (GPU): pJoin latches a travel past the lattice and joins nothing", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick particle banks join", async device => {
    const e = await BrickEngine.create(device, tinyDam());
    try {
      await e.fence();
      const part2 = e.schedule.part2, j = part2.findIndex(p => label(p) === "pJoin");
      assert.ok(j > 0 && label(part2[j - 1]!) === "pArgs" && ((part2[j - 1]!.w ?? 0) & 1) === 1);
      // One coupled record at the dam's free surface in bank 1 (pJoin's bank at slot 0), its before pDelta's
      // NB_DELTA_NONE on z: the stale record of the wedge.
      const S = e.layout.particleStride, P = e.layout.cap.particles, rec = new Float32Array(S);
      rec.set([15.5, 11.5, 16.5, 1, 0, 0, 0, 0, 0, 0, 3e38, 0]);
      e.write("parts", S * P, rec);
      e.write("set", H.PCOUNT0, Uint32Array.of(0, 1));
      e.write("set", H.FAIL, Uint32Array.of(0)); e.write("set", H.FAIL_INFO, Uint32Array.of(0));
      const t0 = performance.now();
      await e.run([part2[j - 1]!, part2[j]!]);
      const ms = performance.now() - t0, h = await e.read("set", 0, 256);
      assert.ok((h[H.FAIL]! & FAIL.PARTICLES) !== 0, `PARTICLES latched (fail 0x${h[H.FAIL]!.toString(16)})`);
      if (h[H.FAIL] === FAIL.PARTICLES) assert.equal(h[H.FAIL_INFO], 0x84000000, "code 4, a moved record, index 0");
      assert.ok(ms < 5000, `the join returned (${ms.toFixed(0)} ms)`);
    } finally { e.destroy(); }
  });
});
