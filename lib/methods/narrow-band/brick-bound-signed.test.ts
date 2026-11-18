/**
 * BOUND_F is signed and per axis (2026-11-19). CPU only (kernel text and the residency oracle).
 * The head's residency grew every side of a source box by one isotropic flow, the brick's fastest face plus |g| dt. On
 * hero-garden-hose-x10 a cut sliver's step-1 face (θ 0.03 at aperture 0.023, 14 m/s down into the terrain) then claimed
 * 9 bricks every way about each shore brick: 19105 F bricks entered at step 2 and left at step 3. The old localSpeed's
 * reach is signed per axis (umSignedFaceExtent), as seep's CFL band (S2j); the bound now is too.
 * - the layout gives BOUND_F six words a slot;
 * - project and pCount raise a side's word with the speed toward it; setMark and setClaimBricks clear all six;
 * - resMark grows each side by its own word and gravity's component that way (rmFlowSide);
 * - the oracle: a fast -y face grows the box downward only, and gravity only lowers the low y side.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { ACT, BOUND_WORDS, BrickLayout, brickKey, brickKeyCoord } from "./brick-abi";
import { residencyOracle } from "./brick-oracle";
import type { BrickSnapshot } from "./brick-readback";
import { brickResidencyKernels } from "./brick-residency.wgsl";
import { brickSetKernels } from "./brick-set.wgsl";
import { brickPressureSolver } from "./brick-pressure-solver";
import type { BrickPlacement } from "./brick-kernel";

const base = { lattice: [256, 256, 256] as const, f: 64, c: 8, b: 16, k: 8, particles: 1024 };

test("brick bound: six words an F slot", () => {
  const L = new BrickLayout(base);
  assert.equal(BOUND_WORDS, 6);
  const after = Object.entries(L.set).filter(([n, v]) => typeof v === "number" && n !== "words" && v > L.set.boundF).map(([, v]) => v as number);
  assert.ok(Math.min(...after) >= L.set.boundF + BOUND_WORDS * L.cap.f, "the next region starts past six words a slot");
});

test("brick bound: the kernels raise, clear and read a side's word", () => {
  const L = new BrickLayout(base);
  const s = brickPressureSolver!(L), all: BrickPlacement[] = [...(s.before ?? []), ...s.setup, ...s.finish];
  for (let i = 0; i < s.slots; i++) all.push(...s.slot(i));
  const project = [...new Set(all.map(p => p.kernel.code))].join("\n");
  assert.ok(project.includes("bound[a]=max(bound[a],-u);bound[3u+a]=max(bound[3u+a],u);"), "project: the face's speed toward each side");
  assert.ok(project.includes("atomicMax(&nbSet[NB_S_BOUND_F+NB_BOUND_WORDS*slot+lid],atomicLoad(&wBound[lid]))"));
  const set = brickSetKernels(L);
  for (const k of [set.setMark, set.setClaimBricks]) assert.ok(k.code.includes("for(var k=0u;k<NB_BOUND_WORDS;k++){atomicStore(&nbSet[NB_S_BOUND_F+NB_BOUND_WORDS*"), k.label);
  const res = Object.values(brickResidencyKernels(L)).map(k => (k as { code: string }).code).join("\n");
  assert.ok(res.includes("return (bitcast<f32>(atomicLoad(&nbSet[NB_S_BOUND_F+NB_BOUND_WORDS*s+3u*side+a]))+gs*dt)*dt/h;"), "rmFlowSide");
  assert.ok(!/NB_S_BOUND_F\+s\]/.test(res), "no scalar read");
});

test("brick bound: a fast -y face grows the residency box downward only", () => {
  const L = new BrickLayout(base), key = brickKey(16, 16, 16), slot = 0;
  const run = (bound: number[]) => {
    const set = new Uint32Array(L.set.words), f = new Float32Array(1), u = new Uint32Array(f.buffer);
    set[L.set.act + slot] = ACT.RES;
    bound.forEach((v, k) => { f[0] = v; set[L.set.boundF + BOUND_WORDS * slot + k] = u[0]!; });
    const before = { header: new Uint32Array(0), set, dir: new Uint32Array(L.dir.words), bricks: new Map([[key, slot]]), pages: new Map(), list: () => [] } as unknown as BrickSnapshot;
    const o = residencyOracle(L, before, new Uint32Array(L.set.words), false, { dt: 1 / 60, h: 0.01, gravity: [0, -9.8, 0] });
    const q = [...o.bricks].map(k => brickKeyCoord(k));
    return [0, 1, 2].map(a => [Math.min(...q.map(c => c[a]!)), Math.max(...q.map(c => c[a]!))]);
  };
  // All-h owners: the vertex box [128, 136] grown by 20 cells and the side's flow, (bound + max(±g_a, 0) dt) dt / h.
  assert.deepEqual(run([0, 0, 0, 0, 0, 0]), [[13, 19], [13, 19], [13, 19]], "at rest: gravity's 0.27 cells on the low y side only");
  // 14 m/s down: (14 + 9.8 / 60) / 60 / 0.01 = 23.6 cells below, nothing more on the other five sides.
  assert.deepEqual(run([0, 14, 0, 0, 0, 0]), [[13, 19], [10, 19], [13, 19]]);
  assert.deepEqual(run([0, 0, 0, 14, 0, 0]), [[13, 22], [13, 19], [13, 19]], "14 m/s in +x: the high x side");
});
