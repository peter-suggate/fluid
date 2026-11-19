/**
 * A ball dropped into a running brick scene (2026-11-19, injectLiquidBall and editFluid): a live drop in source form,
 * added by the next step's ring words as a scheduled drop is, and refused before anything is queued when the fixed
 * pools may not take it. CPU only (the GPU run: tests/brick-live-drop.test.ts).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { H, RING_WORDS } from "./brick-abi";
import type { Vec3 } from "./brick-engine";
import { brickDropRefusal } from "./brick-scene";
import { SRC_RING, brickLiveDrop, brickNoSources, brickSourceRing } from "./brick-source";

const frame = { lattice: [32, 32, 32] as Vec3, h_m: 0.01, origin_m: [-0.16, 0, -0.16] as Vec3 };

test("brick live drop: a ball in cells, its volume and its box held to the lattice", () => {
  const d = brickLiveDrop({ centre_m: { x: 0, y: 0.2, z: 0.05 }, radius_m: 0.04 }, 0.5, frame);
  assert.equal(d.kind, 1);
  assert.deepEqual(d.centre.map(v => +v.toFixed(9)), [16, 20, 21]);
  assert.ok(Math.abs(d.radius - 4) < 1e-9);
  assert.ok(Math.abs(d.volume - (4 / 3) * Math.PI * 64) < 1e-6);
  assert.deepEqual(d.box, [[8, 12, 13], [24, 28, 29]], "the bounds grown by the pin margin");
  const edge = brickLiveDrop({ centre_m: { x: 0.15, y: 0.3, z: 0 }, radius_m: 0.04 }, 0.5, frame);
  assert.deepEqual(edge.box[1], [32, 32, 24], "held to the lattice");
  const disk = brickLiveDrop({ centre_m: { x: 0, y: 0.2, z: 0 }, radius_m: 0.03, halfHeight_m: 0.02 }, 0.5, frame);
  assert.equal(disk.kind, 2);
  assert.ok(Math.abs(disk.volume - Math.PI * 9 * 4) < 1e-6, "a cylinder along z");
  assert.throws(() => brickLiveDrop({ centre_m: { x: 0, y: 0, z: 0 }, radius_m: 0 }, 0.5, frame), /radius/);
});

test("brick live drop: the step it is due on adds it (its kind, volume and box), the step before claims nothing of it", () => {
  const dt = 1 / 60, s = brickNoSources(dt), d = brickLiveDrop({ centre_m: { x: 0, y: 0.2, z: 0 }, radius_m: 0.04 }, 2 * dt, frame);
  const merged = { ...s, drops: [d] }, words = () => { const b = new ArrayBuffer(4 * RING_WORDS); return { u: new Uint32Array(b), f: new Float32Array(b), i: new Int32Array(b) }; };
  const before = words(); brickSourceRing(merged, 0, before.u, before.f);
  assert.equal(before.u[SRC_RING.KIND], 0); assert.equal(before.f[SRC_RING.ADDED], 0);
  assert.equal(before.u[SRC_RING.BOXES], 2, "the next step's box (AHEAD) only");
  const due = words(); brickSourceRing(merged, dt, due.u, due.f);
  assert.equal(due.u[SRC_RING.KIND], 1); assert.equal(due.u[SRC_RING.BOXES], 1);
  assert.ok(Math.abs(due.f[SRC_RING.ADDED]! - d.volume) < 1e-3);
  assert.deepEqual([...due.i.subarray(SRC_RING.BOX, SRC_RING.BOX + 6)], [...d.box[0], ...d.box[1]]);
  const after = words(); brickSourceRing(merged, 2 * dt, after.u, after.f);
  assert.equal(after.u[SRC_RING.KIND], 0, "added once");
});

test("brick live drop: refused when the pools may not take it, or with no receipt", () => {
  const d = brickLiveDrop({ centre_m: { x: 0, y: 0.2, z: 0 }, radius_m: 0.04 }, 0.5, frame);
  // The box [8,12,8]..[24,28,24]: bricks 1..2 x 1..3 x 1..2 (12), pages 0 x 0 x 0 (1).
  const header = (f: number, c: number, p: number) => { const h = new Uint32Array(256); h[H.LIVE_F] = f; h[H.LIVE_C] = c; h[H.PCOUNT0] = p; return h; };
  const cap = { f: 100, c: 10, particles: 20000 }, W = [4, 4, 4];
  assert.match(brickDropRefusal(d, undefined, cap, true, 3, W)!, /receipt/);
  assert.equal(brickDropRefusal(d, header(88, 9, 0), cap, true, 3, W), undefined);
  assert.match(brickDropRefusal(d, header(89, 0, 0), cap, true, 3, W)!, /12 bricks/);
  assert.match(brickDropRefusal(d, header(0, 10, 0), cap, true, 3, W)!, /1 pages/);
  assert.equal(brickDropRefusal(d, header(0, 10, 0), cap, false, 3, W), undefined, "no pages on a flat engine");
  assert.match(brickDropRefusal(d, header(0, 0, 19900), cap, true, 3, W)!, /particles/);
  // A full window: every page and brick resident, so the box claims nothing new.
  assert.equal(brickDropRefusal(d, header(64, 1, 0), { f: 64, c: 1, particles: 20000 }, true, 3, [1, 1, 1]), undefined);
});
