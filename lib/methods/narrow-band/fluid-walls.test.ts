/**
 * LANE F OWNS THIS FILE. F's walls at the box hook (brick-abi.md §2, M3 padding), on a padded layout: the lattice 32^3
 * of storage, the box 24x16x24 the container. CPU only (kernel text).
 * - Every F kernel resolves the wall tokens (fluidKernel's resolveBox).
 * - F's wall planes and its cell domain read NB_BOX: the owner readers' walls (foU/foP/foS), fgFaceOpen, fgHeld and
 *   the samplers' holds, the move's endpoint box and ballistic stop, the level set's wall planes and vertex domain,
 *   and the C faces' range. Storage (slots, pages, the vertex storage, the window) stays at NB_LATTICE.
 * - No F kernel walls at the storage edge (the engine guard's test), except lAdvect's E-owned release readers
 *   (levelset-release.wgsl.ts, prefix nbr), which switch with lane E's change.
 * - Unpadded, no kernel names NB_BOX: the dam's text is unchanged (byte-identical, checked by the kernel dump).
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BOX_TOKEN, BrickLayout } from "./brick-abi";
import { fluidLayout } from "./fluid-grid.wgsl";
import { levelSetKernels } from "./levelset-kernels.wgsl";
import { particleKernels } from "./particles-kernels.wgsl";
import { velocityKernels } from "./velocity-kernels.wgsl";

// brick-engine.ts STORAGE_EDGE_WALLS: an owner reader's walls at the storage edge.
const EDGE = [/fn \w*OnWall\(c:vec3i,a:u32\)->bool\{return c\[a\]<=0\|\|c\[a\]>=NB_LATTICE\[a\];\}/, /fn \w*FaceInLattice\(c:vec3i,a:u32\)->bool\{var hi=NB_LATTICE;/];
const physics = { viscosity_Pa_s: 1e-3, density_kg_m3: 1000, volumeControl_1_s: 10 };

function kernels(L: BrickLayout, tiered: boolean) {
  const F = fluidLayout(L, { tiered });
  return ([...Object.values(particleKernels(L, F)), ...Object.values(levelSetKernels(L, F)), ...Object.values(velocityKernels(L, F, physics))] as unknown[])
    .filter((k): k is { label: string; code: string } => typeof (k as { code?: unknown })?.code === "string");
}

test("fluid walls: on a padded lattice F's walls and cell domain are at the box, its storage at the lattice", () => {
  const P = new BrickLayout({ lattice: [32, 32, 32], box: [24, 16, 24], f: 125, c: 8, b: 0, k: 0, particles: 1024 });
  for (const tiered of [false, true]) {
    const ks = kernels(P, tiered), code = (label: string) => ks.find(k => k.label === label)!.code;
    assert.ok(ks.length > 20);
    for (const k of ks) {
      assert.ok(!k.code.includes(BOX_TOKEN) && !k.code.includes("nbInWallBox("), `${k.label}: a wall token unresolved`);
      // Only E's release readers (lAdvect's nbr owner) are left at the storage edge.
      const edge = k.code.split("\n").filter(l => EDGE.some(re => re.test(l)) && !/^fn nbr/.test(l));
      assert.deepEqual(edge, [], `${k.label}: F walls at the storage edge`);
      // The owner readers F instantiates (the tile part and the three face channels) wall at the box.
      for (const p of ["foU", "foP", "foS"]) if (k.code.includes(`fn ${p}OnWall(`)) {
        assert.ok(k.code.includes(`fn ${p}OnWall(c:vec3i,a:u32)->bool{return c[a]<=0||c[a]>=NB_BOX[a];}`), `${k.label}: ${p}OnWall`);
        assert.ok(k.code.includes(`fn ${p}FaceInLattice(c:vec3i,a:u32)->bool{var hi=NB_BOX;`), `${k.label}: ${p}FaceInLattice`);
      }
      if (k.code.includes("fn fgFaceOpen(")) assert.ok(k.code.includes("return nbInBox(t)&&c[a]>0&&c[a]<NB_BOX[a];}"), `${k.label}: fgFaceOpen`);
      if (k.code.includes("fn fgHeld(")) assert.ok(k.code.includes("fn fgHeld(p:vec3f)->vec3f{return clamp(p,vec3f(0.0),vec3f(NB_BOX));}"), `${k.label}: fgHeld`);
      if (k.code.includes("fn fgCellCoarse(")) assert.ok(k.code.includes("return FG_TIERED&&nbInBox(c)&&!foCellOwnerH(c);"), `${k.label}: fgCellCoarse`);
      if (k.code.includes("fn fgLiquidCell(")) assert.ok(k.code.includes("fn fgLiquidCell(c:vec3i)->bool{if(!nbInBox(c)){return false;}"), `${k.label}: fgLiquidCell`);
      // Storage stays at the lattice: the window and the vertex words.
      if (k.code.includes("fn fgWinCovers(v:vec3i)->bool{return fgWinOn&&")) assert.match(k.code, /fn fgWinCovers\([^\n]*all\(v<=NB_LATTICE\);\}/, `${k.label}: fgWinCovers`);
    }
    assert.match(code("pMove"), /nbfBoxEndpoint\(end,vec3f\(0\.0\),vec3f\(NB_BOX\),nbSampleRadius\(m\)\)/);
    assert.match(code("pMove"), /let solid=ahead\[axis\]<0\.0\|\|ahead\[axis\]>f32\(NB_BOX\[axis\]\);/);
    assert.match(code("pMove"), /fn pCellOf\(q:vec3f\)->vec3i\{return clamp\(vec3i\(floor\(q\)\),vec3i\(0\),NB_BOX-vec3i\(1\)\);\}/);
    assert.match(code("pJoin"), /if\(!nbInBox\(4\*t\)\)\{continue;\}/);
    assert.match(code("lAdvect"), /fn lWallWithin\(v:vec3i,a:u32,reach:i32\)->bool\{return v\[a\]<reach\|\|v\[a\]>NB_BOX\[a\]-reach;\}/);
    assert.match(code("lAdvect"), /fn lOnWall\(v:vec3i\)->bool\{return any\(v==vec3i\(0\)\)\|\|any\(v==NB_BOX\);\}/);
    assert.match(code("lUnion"), /fn lVertexIn\(v:vec3i\)->bool\{return all\(v>=vec3i\(0\)\)&&all\(v<=NB_BOX\);\}/);
    assert.match(code("vGauge"), /if\(!nbInBox\(o\+c\)\|\|!fgCellOwned\(o\+c\)\)\{continue;\}/);
    if (tiered) assert.match(code("uAdvect"), /fn vcInLattice\(c:vec3i,a:u32\)->bool\{var hi=NB_BOX;hi\[a\]\+=1;/);
  }
});

test("fluid walls: unpadded, no F kernel names the box", () => {
  const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 0, k: 0, particles: 1024 });
  for (const tiered of [false, true]) for (const k of kernels(L, tiered)) {
    assert.ok(!/NB_BOX|nbInBox|NB_WALL_BOX|nbInWallBox/.test(k.code), `${k.label}: names the box unpadded`);
  }
});
