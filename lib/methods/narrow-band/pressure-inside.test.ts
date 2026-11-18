/**
 * FLAGS.INSIDE (umCellInsideSolid) and npBandFace's θ rule on the brick engine (2026-11-18). CPU only (kernel text).
 * The old rows (uniform-pressure-arena.ts pRows) take θ = 1 on a wet-dry face between two cells whose centres lie in
 * the solid and whose open fractions are both < 0.1. The composite passed inside = false, so terrain slivers (FRAC
 * 0.002-0.05) took the face-φ θ (0.001-0.04) and the projection put 20-90 m/s on their faces (hero-garden-hose-x10 at
 * step 1, the water slide from step 13; the old engine 1-4 m/s on the same initial table).
 * - the bit is its own (no other FLAGS bit);
 * - solidStamp, solidRestamp and bodyStamp write it with STATIC_SHUT (bodyStamp also for a body of BMASK);
 * - the composite's band face reads it for both cells, only where the rule can take it, and passes it to npBandFace.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout, FLAG } from "./brick-abi";
import { brickPressureSolver } from "./brick-pressure-solver";
import { brickSolidStampKernels } from "./brick-solid-stamp.wgsl";
import { brickStaticMaskWalls } from "./brick-static.wgsl";
import { brickBodyKernels } from "./bodies-stage.wgsl";
import type { BrickPlacement } from "./brick-kernel";

test("pressure inside: FLAGS.INSIDE is a bit of its own", () => {
  const others = Object.entries(FLAG).filter(([n]) => n !== "INSIDE" && n !== "KICK0").map(([, v]) => v);
  for (const v of others) assert.equal(v & FLAG.INSIDE, 0);
  for (let a = 0; a < 3; a++) assert.equal((FLAG.KICK0 << a) & FLAG.INSIDE, a === 2 ? FLAG.INSIDE : 0, "KICK2 shares it on C tiles only");
});

test("pressure inside: the stamps write FLAGS.INSIDE with STATIC_SHUT", () => {
  const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 16, k: 8, particles: 1024 });
  const s = brickSolidStampKernels(L, { walls: brickStaticMaskWalls([64, 32, 32], false) });
  assert.ok(s.solidStamp.code.includes("if(nbStaticSolidAt(vec3f(c)+0.5)){phi[f]=phi[f]|NB_FLAG_INSIDE;}"), "solidStamp");
  assert.ok(s.solidRestamp.code.includes("&~(NB_FLAG_STATIC_SHUT|NB_FLAG_INSIDE)"), "solidRestamp clears it");
  assert.ok(s.solidRestamp.code.includes("if(nbStaticSolidAt(vec3f(c)+0.5)){phi[f]=phi[f]|NB_FLAG_INSIDE;}"), "solidRestamp sets it");
  const body = brickBodyKernels(L, { frame_m: [0, 0, 0], h_m: 0.01 }, { openTop: false }).bodyStamp.code;
  assert.ok(body.includes("let inside=nbStaticSolidAt(world)||pbBodyAtCandidates(world,head.w)>=0;"), "bodyStamp: static or a body");
});

test("pressure inside: the composite's band face passes umCellInsideSolid to npBandFace", () => {
  const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 16, k: 8, particles: 1024 });
  const s = brickPressureSolver!(L), all: BrickPlacement[] = [...(s.before ?? []), ...s.setup, ...s.finish];
  for (let i = 0; i < s.slots; i++) all.push(...s.slot(i));
  const code = [...new Set(all.map(p => p.kernel.code))].join("\n");
  assert.ok(code.includes("if(across==NP_ACROSS_RESIDENT&&liquid!=n.wet&&open<0.1&&qOpen<0.1){inside=poInside(c);qInside=poInside(q);}"));
  assert.ok(code.includes("npBandFace(across,own,liquid,1.0,1.0,band,native,n,fp,volume,false,open<=1e-5,inside,open,qOpen>1e-5,qInside,qOpen)"));
  assert.ok(code.includes("else if(inside&&qInside&&open<0.1&&qOpen<0.1){theta=1.0;}"), "npBandFace's rule");
});
