/**
 * LANE P OWNS THIS FILE. The brick engine's monolithic body coupling in the MG-PCG (pressure-mgpcg.ts options.bodies,
 * brick-pressure-solver.ts compositeMGPCG; lane P 2026-11-17, the coordinator's order: A' = A + (1/h) J M~ J^T for the
 * dynamic bodies, the MG preconditioner on A). CPU only (kernel text).
 * - Without a bodies frame the solver is the one before the change: no body kernel, no pmgB buffer, no mb text.
 * - With a bodies frame: J (setup), Z and the per-body fold (setup and every slot), and the operator's B term in setupF,
 *   direction and update; at most 8 storage buffers a kernel.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout } from "./brick-abi";
import { compositeMGPCG } from "./brick-pressure-solver";
import type { BrickPlacement } from "./brick-kernel";

function placements(bodies: boolean, monolithic = true): { all: BrickPlacement[]; buffers: Record<string, number> } {
  const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 16, k: 18, particles: 1024 });
  const s = compositeMGPCG({ monolithic })(L, bodies ? { bodies: { frame_m: [-0.8, 0, -0.4], h_m: 0.025 } } : {});
  const all: BrickPlacement[] = [...(s.before ?? []), ...s.setup, ...s.finish];
  for (let i = 0; i < s.slots; i++) all.push(...s.slot(i));
  return { all, buffers: { ...(s.buffers as Record<string, number> | undefined) } };
}
const labels = (all: BrickPlacement[]) => new Set(all.map(p => p.kernel.label.split(" / ").pop()!.split(":").pop()!));

test("monolithic coupling: no bodies frame, no body term", () => {
  for (const { all, buffers } of [placements(false), placements(true, false)]) {
    for (const p of all) {
      assert.ok(!/pmgBody/.test(p.kernel.label), `no body kernel (${p.kernel.label})`);
      assert.ok(!/\bmbApplyW?\b|\bmbStage\b|\bpmgB\b/.test(p.kernel.code), `no body term in ${p.kernel.label}`);
    }
    assert.ok(!("pmgB" in buffers), "no pmgB buffer");
  }
});

test("monolithic coupling: a bodies frame adds J, Z, the fold and the operator's B term", () => {
  const { all, buffers } = placements(true), names = labels(all);
  for (const k of ["pmgBodyJ", "pmgBodyZ0", "pmgBodyFold0", "pmgBodyZ", "pmgBodyFold"]) assert.ok(names.has(k), `kernel ${k}`);
  assert.ok((buffers.pmgB ?? 0) > 0, "the pmgB buffer");
  const code = all.map(p => p.kernel.code).join("\n");
  assert.ok(code.includes("-mbApply(mbBrick(slot),l,c,MB_WX,0.0,MB_WX)"), "setupF: r = b - A x - B x");
  assert.ok(code.includes("+mbApply(mbBrick(slot),l,c,MB_WZ,0.0,MB_WZ)"), "direction: q = A z + B z");
  assert.ok(code.includes("mbStage(lid,pAlpha);") && code.includes("let mb=mbApplyW(mbBrick(slot),l,c);") && code.includes("-ax-mb;"), "update: r -= B (x + alpha d), the bodies staged");
  for (const p of all) {
    const storage = p.kernel.bindings.filter(b => b.kind !== "uniform").length;
    assert.ok(storage <= 8, `${p.kernel.label}: ${storage} storage buffers`);
  }
});
