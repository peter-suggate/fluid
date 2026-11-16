/**
 * Lane F's loop bounds (E's audit after the GPU wedge of 2026-11-17): every loop of a fluid kernel is bounded by a
 * buffer or the lattice, and a configuration that would leave one unbounded fails on the host.
 * - lNear's cfg.z (the y and z passes' loop bound and a vector index) is checked against 0..2 by validateBrickStages.
 * - pScan stores a bin's count only up to the bank (a run past an overflow stays inside it), and fgRun holds every
 *   run to its slot's bin. The trace's own bound is in velocity-formulas.test.ts (the GPU check of a NaN dt).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { BrickLayout } from "./brick-abi";
import { fluidStagesAt } from "./brick-fluid-stages";
import { validateBrickStages, type BrickStageContext, type BrickStages } from "./brick-stages";
import { FG_RUN_WGSL } from "./fluid-grid.wgsl";
import "../index";

function context(): BrickStageContext {
  const method = getMethod("uniform-narrow-band-flip");
  const values = resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
  const layout = new BrickLayout({ lattice: [32, 32, 32], f: 200, c: 16, b: 16, k: 0, particles: 1 << 16 });
  return { layout, dt_s: 1 / 60, h_m: 0.0125, gravity_mps2: [0, -9.81, 0], values } as BrickStageContext;
}

test("lNear's passes are 0..2, and a placement outside them throws on the host", () => {
  for (const tiered of [false, true]) {
    const stages = fluidStagesAt(tiered)(context());
    const near = stages.levelSet.filter(p => p.kernel.label === "lNear");
    assert.deepEqual(near.map(p => p.z ?? 0), [0, 1, 2]);
    assert.deepEqual(near[0]!.kernel.cfgZ, [0, 2]);
    validateBrickStages(stages);
    for (const z of [3, -1, 1.5, 0xffffffff]) {
      const bad: BrickStages = { ...stages, levelSet: stages.levelSet.map(p => p === near[2] ? { ...p, z } : p) };
      assert.throws(() => validateBrickStages(bad), /lNear is placed with cfg\.z .* outside its range 0\.\.2/, `z ${z}`);
    }
  }
});

test("pScan's bin count and fgRun's runs stay inside the bank", () => {
  const stages = fluidStagesAt(true)(context());
  const scan = [...stages.rebin, ...stages.move, ...(stages.kick?.seed ?? [])].find(p => p.kernel.label === "pScan");
  assert.ok(scan, "the rebin scans");
  assert.ok(scan.kernel.code.includes("atomicStore(&nbSet[NB_S_BIN_COUNT+slot],min(total,NB_PCAP-min(base,NB_PCAP)));"), "pScan clamps the count to the bank");
  assert.ok(FG_RUN_WGSL.includes("let bs=min(nbSetWord(NB_S_BIN_START+s),NB_PCAP);let be=bs+min(count,NB_PCAP-bs);"), "fgRun's bin is inside the bank");
  assert.ok(FG_RUN_WGSL.includes("clamp(fcLoad(FC_RUN+512u*s+l),bs,be)") && FG_RUN_WGSL.includes("clamp(fcLoad(FC_RUN+512u*s+l+1u),bs,be)"), "both ends of a run are held to it");
});
