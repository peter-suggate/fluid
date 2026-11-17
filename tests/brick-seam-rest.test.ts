// A tiered still-water tank holds still across its h / 4h seams (lane P, 2026-11-17). Body-free: every tiered scene.
//
// rigid-hydrostatic at h = 0.025 with its body removed: a still tank, 32 cells a side and 16 deep. A flat still
// surface is 4h everywhere (no rule bands it), so the census's test hold (BrickCensusHold) forces an h block in the
// deep liquid, tiles [2, 1, 2] .. [6, 3, 6): seams (an h row's face to a liquid native 4h row) on all six of its sides.
// That is where the kick's defect struck: the seam head (poHeadS, the start's tangential slope) read the kick's cold
// start, then half the pressure (brick X is in dt units, the kick solves at dt/2), and the current it injected
// persisted (0.086 m/s with the old body version of this test).
//
// - **Seams present:** at least 512 seam faces on liquid h rows every step, so the test cannot pass vacuously.
// - **Seam balance:** each seam face's ghost (the 4h row's X plus the cached seam head) less the h row's X is within
//   0.05 cell heads (g dt h) of its at-rest value: 0 across y, the seam's sample distance (2.5 cells, the sign by
//   side) along y.
// - **Every liquid face, 15 steps:** peak |U| over every face of every liquid h row (seams, band; not WALL, NEUMANN or
//   CLOSED) and between liquid native rows is at most 2 MG_TOLERANCE h, the solve's own stopping bound (max |div| <=
//   MG_TOLERANCE, carried in U from step to step).
// The fixes: brick-engine's BRICK_KICK_RESOLVES (the old arena's 23 kick re-solves) and pressureKickScaleKernel.
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { solidVoxelsOnMovedLattice } from "../lib/core/scene-lattice";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { createBrickSolver } from "../lib/methods/narrow-band/brick-method";
import type { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { BRICK_ENGINE_SCENES } from "../lib/methods/narrow-band/brick-scene";
import { BRICK_CELLS, BRICK_STRIDE, CMASK_WORDS, H, LIST, RECORD_WORDS, REC, SEAM } from "../lib/methods/narrow-band/brick-abi";
import { MG_TOLERANCE, mgLayout } from "../lib/methods/narrow-band/pressure-mgpcg";
import { NP_K } from "../lib/methods/narrow-band/pressure-rows.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";

const STEPS = 15, H_M = 0.025, HOLD = { lo: [2, 1, 2], hi: [6, 3, 6] } as const;

test("brick engine: a still tiered tank's h / 4h seams hold still water (rigid-hydrostatic without its body, a held h block, h = 0.025, 15 steps)", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  await withUniformDevice("brick seam rest", async device => {
    const definition = getSceneDefinition("rigid-hydrostatic"), base = structuredClone(sceneDocument(definition)), scene = structuredClone(base);
    scene.voxelDomain.finestCellSize_m = H_M; scene.solidVoxels = solidVoxelsOnMovedLattice(base, scene);
    scene.fluid.brickPools = { f: 125, c: 8, b: 16, k: 0, particles: 86016 }; scene.rigidBodies = [];
    const profile = definition.methodProfile, quality = profile?.methodId === method.id ? profile.quality : "balanced";
    const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine: "brick" });
    const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
    const g = Math.hypot(scene.fluid.gravity_m_s2.x, scene.fluid.gravity_m_s2.y, scene.fluid.gravity_m_s2.z), unit = g * dt * H_M;
    const enabled = BRICK_ENGINE_SCENES as string[], had = enabled.includes(scene.sceneId!);
    if (!had) enabled.push(scene.sceneId!);
    const solver = await createBrickSolver(device, scene, values, quality, { censusHold: HOLD }).finally(() => { if (!had) enabled.splice(enabled.indexOf(scene.sceneId!), 1); });
    try {
      const e = solver.brickEngine, L = e.layout, M = mgLayout(L);
      assert.ok(L.lattice.every(n => n <= 32), "one page: page 0's word is the C slot");
      const f32 = (w: Uint32Array) => new Float32Array(w.buffer, w.byteOffset, w.length);
      const measure = async (step: number) => {
        const head = await e.read("set", 0, 64), n = Math.min(head[H.LIST_COUNT + LIST.F_ALL]!, L.set.listCap[LIST.F_ALL]!);
        const slots = await e.read("set", L.set.lists[LIST.F_ALL]!, Math.max(n, 1));
        const F = L.cap.f * BRICK_STRIDE, C = L.cap.c * BRICK_STRIDE;
        const x = f32(await e.read("pres", L.base("pres", "F", "X"), F)), diag = f32(await e.read("pres", L.base("pres", "F", "DIAG"), F));
        const cx = f32(await e.read("pres", L.base("pres", "C", "X"), C)), cdiag = f32(await e.read("pres", L.base("pres", "C", "DIAG"), C));
        const fc = f32(await e.read("pmgF", M.FC, 7 * F)), u = f32(await e.read("vel", L.base("vel", "F", "U0"), 3 * F)), uc = f32(await e.read("vel", L.base("vel", "C", "U0"), 3 * C));
        const cs = (await e.read("dir", L.dir.page, 1))[0]!; assert.ok(cs < SEAM, "page 0 holds a C slot");
        const cmask = await e.read("dir", L.dir.cmask + CMASK_WORDS * cs, CMASK_WORDS), tileH = (i: number) => ((cmask[i >> 5]! >>> (i & 31)) & 1) !== 0;
        const slotOf = new Map<number, number>();
        for (let i = 0; i < n; i++) slotOf.set((await e.read("dir", L.dir.frec + RECORD_WORDS * slots[i]! + REC.KEY, 1))[0]!, slots[i]!);
        const fSlot = (c: number[]) => slotOf.get(((c[0]! >> 3) | ((c[1]! >> 3) << 10) | ((c[2]! >> 3) << 20)) >>> 0);
        const local = (c: number[]) => (c[0]! & 7) | ((c[1]! & 7) << 3) | ((c[2]! & 7) << 6), tile = (T: number[]) => (T[0]! & 7) | ((T[1]! & 7) << 3) | ((T[2]! & 7) << 6);
        let seams = 0, worstSeam = 0, peak = 0, where = "", hTiles = 0;
        for (let i = 0; i < BRICK_CELLS; i++) if (tileH(i)) hTiles++;
        for (const [key, s] of slotOf) {
          const b = [key & 1023, (key >>> 10) & 1023, (key >>> 20) & 1023];
          for (let l = 0; l < BRICK_CELLS; l++) {
            if (!(diag[BRICK_STRIDE * s + l]! > 0)) continue;
            const c = [8 * b[0]! + (l & 7), 8 * b[1]! + ((l >> 3) & 7), 8 * b[2]! + (l >> 6)], kinds = fc[BRICK_STRIDE * (7 * s + 6) + l]!;
            for (let f = 0; f < 6; f++) {
              const kind = ((kinds >>> (4 * f)) & 15) - 1, a = f >> 1, sign = f & 1 ? 1 : -1;
              if (kind === NP_K.NEUMANN || kind === NP_K.CLOSED || kind === NP_K.WALL) continue;
              // The face's U: f = 2a is c's low face (this slot), f = 2a + 1 the low face of c + e (its brick's slot).
              const q = [...c]; if (sign > 0) q[a]! += 1;
              const qs = sign > 0 ? fSlot(q) : s;
              if (qs !== undefined) { const v = Math.abs(u[BRICK_STRIDE * (qs + a * L.cap.f) + local(q)]!); if (v > peak) { peak = v; where = `h face f${f} of ${c} kind ${kind}`; } }
              if (kind !== NP_K.SEAM) continue;
              seams++;
              const across = [...c]; across[a]! += sign; const T = across.map(v => v >> 2);
              const r = (cx[BRICK_STRIDE * cs + tile(T)]! + fc[BRICK_STRIDE * (7 * s + f) + l]! - x[BRICK_STRIDE * s + l]!) / unit;
              const rest = a === 1 ? -sign * 2.5 : 0;
              worstSeam = Math.max(worstSeam, Math.abs(r - rest));
            }
          }
        }
        // The native rows' faces between two liquid native rows (C U, a tile's low face along a).
        const T = L.lattice.map(v => v >> 2);
        for (let i = 0; i < BRICK_CELLS; i++) {
          const p = [i & 7, (i >> 3) & 7, i >> 6]; if (p.some((v, a) => v >= T[a]!) || tileH(i) || !(cdiag[BRICK_STRIDE * cs + i]! > 0)) continue;
          for (let a = 0; a < 3; a++) {
            if (p[a] === 0) continue; const m = [...p]; m[a]! -= 1; const j = tile(m);
            if (tileH(j) || !(cdiag[BRICK_STRIDE * cs + j]! > 0)) continue;
            const v = Math.abs(uc[BRICK_STRIDE * (cs + a * L.cap.c) + i]!); if (v > peak) { peak = v; where = `4h face a${a} of tile ${p}`; }
          }
        }
        t.diagnostic(`step ${step}: ${hTiles} h tiles, ${seams} seam faces, worst seam ${worstSeam.toFixed(4)} cell heads, peak |U| ${peak.toExponential(2)} m/s at ${where}`);
        return { seams, worstSeam, peak, where };
      };
      for (let step = 1; step <= STEPS; step++) {
        assert.ok(solver.advanceTo(step * dt, []), `step ${step} admitted`);
        await solver.awaitFrameCompletion?.(); await solver.assertSimulationHealthy?.();
        const m = await measure(step);
        assert.ok(m.seams >= 512, `step ${step}: ${m.seams} seam faces in the liquid (the held block's)`);
        assert.ok(m.worstSeam <= 0.05, `step ${step}: a seam is ${m.worstSeam} cell heads off hydrostatic balance`);
        assert.ok(m.peak <= 2 * MG_TOLERANCE * H_M, `step ${step}: peak |U| ${m.peak} m/s at ${m.where}, over the solve's bound ${2 * MG_TOLERANCE * H_M}`);
      }
    } finally { solver.destroy(); }
  }, ["subgroups"]);
});
