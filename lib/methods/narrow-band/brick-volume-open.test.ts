/**
 * The brick engine's volume measures count open cells only, as the old engine's (uniform-mixed-remap.ts tileCapacity,
 * measureInitial over the open occupancy): phiSolid extends phi into a static solid (the nbflip paper scenes' voxel
 * wall layer, wallLayer), and counting it raised the oscillating surface's gauge 17.6% by step 3 (1507 of 8577 cells
 * in the wall shell, 2026-11-18), which the volume control then drained from the real liquid. CPU only (kernel text).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BrickLayout } from "./brick-abi";
import { WGSL_VOLUME } from "./brick-present";
import { fluidLayout } from "./fluid-grid.wgsl";
import { velocityKernels } from "./velocity-kernels.wgsl";

const physics = { viscosity_Pa_s: 1e-3, density_kg_m3: 1000, volumeControl_1_s: 10 };
const CLOSED_GAUGE = /if\(\(phi\[NB_PHI_F_FLAGS\+NB_STRIDE\*slot\+lid\+64u\*j\]&NB_FLAG_OPEN\)==0u\)\{continue;\}\n\s*let open=select\(cut\[NB_CUT_FRAC\+NB_STRIDE\*kslot\+lid\+64u\*j\],1\.0,kslot==NB_NONE\);\n\s*let f=select\(nbfCellFill\(v\),1\.0,negative==8u\)\*open;/;

// A cut cell holds fill times its open fraction (the old gauge's umCapacity, umCellOpen): a terrain or body cut is
// fractional, where a voxel's is 0 or 1. The K slot is the brick's (nbFRec CUT), NONE in an uncut brick (weight 1).
test("the volume gauge skips closed cells and weighs a cut cell by its open fraction, tiered and untiered", () => {
  const L = new BrickLayout({ lattice: [32, 32, 32], f: 125, c: 8, b: 0, k: 0, particles: 1024 });
  for (const tiered of [false, true]) {
    const v = velocityKernels(L, fluidLayout(L, { tiered }), physics) as unknown as Record<string, { code: string }>;
    assert.match(v.vGauge!.code, CLOSED_GAUGE, `tiered ${tiered}`);
    assert.match(v.vGauge!.code, /let kslot=nbFRec\(slot,NB_REC_CUT\);/, `tiered ${tiered}: the brick's K slot`);
  }
});

test("the presentation volume is 0 in a closed F cell", () => {
  const L = new BrickLayout({ lattice: [32, 32, 32], f: 125, c: 8, b: 0, k: 0, particles: 1024 });
  assert.match(WGSL_VOLUME(L), /let fs=nbFSlotOfBrick\(b\);\n\s*if\(f>0\.0&&fs<NB_SEAM&&\(phi\[NB_PHI_F_FLAGS\+NB_STRIDE\*fs\+nbLocal\(v\)\]&NB_FLAG_OPEN\)==0u\)\{f=0\.0;\}/);
});
