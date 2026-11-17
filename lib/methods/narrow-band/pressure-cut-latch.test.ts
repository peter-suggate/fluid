/**
 * LANE P OWNS THIS FILE. The composite operator's cut-tile latches (pressure-composite.wgsl.ts, pressure-mgpcg.ts
 * pressureRowsList; lane P 2026-11-17, for lane E's M3 static rule). CPU only (kernel text).
 * - With cuts (K slots): a cut brick's tile that is not h-owned latches only when it is a body's (BMASK != 0 in the
 *   operator, NEAR_BODY / BODY_WAS in the live lists' cold path: §4's h ring keeps a body's tiles h). A static cut's
 *   4h tile may stay (the old solidPromote leaves dry cut bricks at 4h); a liquid native row in a cut brick latches in
 *   rowsC (the native rows never read a cut: the old band certificate's failure).
 * - Without cuts, any cut brick latches, as before, and rowsC has no such check.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout, F_CLASS } from "./brick-abi";
import { brickPressureSolver } from "./brick-pressure-solver";
import { pressureOperatorKernels } from "./pressure-operator.wgsl";
import type { BrickPlacement } from "./brick-kernel";

function codes(k: number): string {
  const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 16, k, particles: 1024 });
  const s = brickPressureSolver!(L), all: BrickPlacement[] = [...(s.before ?? []), ...s.setup, ...s.finish];
  for (let i = 0; i < s.slots; i++) all.push(...s.slot(i));
  // The stages' operator (rowsList with the live lists: the cold path's cut latch).
  const op = pressureOperatorKernels(L, { faces: true });
  return [...new Set([...all.map(p => p.kernel.code), op.operator.code])].join("\n");
}

test("pressure cut latch: a cut brick's 4h tile latches only for a body; a liquid native row in a cut brick latches in rowsC", () => {
  const on = codes(8), body = (F_CLASS.NEAR_BODY | F_CLASS.BODY_WAS) >>> 0;
  assert.ok(on.includes("if(bmask!=0u&&lid<8u){let T=(nbFOrigin(slot)>>vec3u(2u))+poCorner(lid);if(nbTileInLattice(T)&&!nbTileOwnerH(T)){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(slot,NB_REC_KEY));}}"), "the operator's body tile latch");
  assert.ok(on.includes(`if(nbFRec(slot,NB_REC_CUT)!=NB_NONE&&(nbFRec(slot,NB_REC_CLASS)&(${body}u))!=0u){nbFail(NB_FAIL_UNSUPPORTED,key);}`), "the live lists' cold path: a body's");
  assert.ok(on.includes("if(liquid){let fs=nbFSlotOfBrick(T>>vec3u(1u));if(fs<NB_SEAM){if(nbFRec(fs,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,nbFRec(fs,NB_REC_KEY));}}}"), "rowsC: a liquid native row in a cut brick");
  assert.ok(!on.includes("if(kslot!=NB_NONE&&lid<8u)"), "no static all-h latch");
  const off = codes(0);
  assert.ok(off.includes("if(nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,key);}"), "without cuts any cut brick latches");
  assert.ok(!off.includes("if(liquid){let fs=nbFSlotOfBrick(T>>vec3u(1u));"), "without cuts rowsC is unchanged");
});
