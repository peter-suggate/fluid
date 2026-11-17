/**
 * LANE P OWNS THIS FILE. The pressure walls at the box hook (brick-abi.md §2, M3 padding; pressure-composite.wgsl.ts
 * pressureResolveBox), on a padded layout: the lattice 32^3 of storage, the box 24x16x24 the container. CPU only
 * (kernel text), the composite MGPCG the engine registers, with and without cuts (K slots).
 * - Every pressure kernel resolves the wall tokens (no NB_WALL_BOX, nbInWallBox or nbTileInWallBox left).
 * - The rows see the box as their lattice: poAcross's WALL, the wall face's volume, the cuts' solid, the h and native
 *   rows' domain, the wall planes (the separated state and its release, pmgWallsF / pmgWallsC), mgpcg's tiles, and the
 *   owners' walls (nbOnWall, nbFaceInLattice) read NB_BOX.
 * - Storage stays at the lattice: the staged vertex φ (poVertexS, nbVertexPhi's), the residency guard and the page
 *   clamp (poGuardVertexG, poPageHeld), pOwnerT (nbTileOwnerH's).
 * - Unpadded, no pressure kernel names the box (the dam's text is byte-identical: kernel-hashes over the enabled scenes).
 * - A padded tiered box that is not whole 4h tiles is refused: a native row cannot straddle a box plane.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { BOX_TOKEN, BrickLayout } from "./brick-abi";
import { brickPressureSolver } from "./brick-pressure-solver";
import type { BrickPlacement } from "./brick-kernel";

const EDGE = [/fn \w*OnWall\(c:vec3i,a:u32\)->bool\{return c\[a\]<=0\|\|c\[a\]>=NB_LATTICE\[a\];\}/, /fn \w*FaceInLattice\(c:vec3i,a:u32\)->bool\{var hi=NB_LATTICE;/];

function kernels(L: BrickLayout) {
  const s = brickPressureSolver!(L), all: BrickPlacement[] = [...(s.before ?? []), ...s.setup, ...s.finish];
  for (let k = 0; k < s.slots; k++) all.push(...s.slot(k));
  const byLabel = new Map<string, string>();
  for (const p of all) byLabel.set(p.kernel.label, p.kernel.code);
  return byLabel;
}

test("pressure walls: on a padded lattice the rows' walls and domain are at the box, storage at the lattice", () => {
  for (const k of [0, 8]) {
    const P = new BrickLayout({ lattice: [32, 32, 32], box: [24, 16, 24], f: 125, c: 8, b: 16, k, particles: 1024 });
    const ks = kernels(P), code = (label: string) => { const c = ks.get(label); assert.ok(c, `no kernel ${label}`); return c; };
    assert.ok(ks.size > 20);
    for (const [label, c] of ks) {
      assert.ok(!c.includes(BOX_TOKEN) && !c.includes("WallBox("), `${label}: a wall token unresolved`);
      const edge = c.split("\n").filter(l => EDGE.some(re => re.test(l)));
      assert.deepEqual(edge, [], `${label}: pressure walls at the storage edge`);
      if (c.includes("fn nbOnWall(")) assert.ok(c.includes("fn nbOnWall(c:vec3i,a:u32)->bool{return c[a]<=0||c[a]>=NB_BOX[a];}"), `${label}: nbOnWall`);
      if (c.includes("fn poAcross(")) assert.ok(c.includes("if(!nbInBox(q)){return NP_ACROSS_WALL;}"), `${label}: poAcross`);
      if (c.includes("fn poTileOpen(")) assert.ok(c.includes("fn poTileOpen(t:vec3i)->bool{return poTileInBox(t);}") && c.includes("fn poTileInBox(t:vec3i)->bool{return all(t>=vec3i(0))&&all(4*t<NB_BOX);}"), `${label}: poTileOpen`);
      if (c.includes("fn poCutK(")) assert.ok(c.includes("fn poCutK(c:vec3i)->u32{if(!nbInBox(c))") && c.includes("fn poFrac(c:vec3i)->f32{if(!nbInBox(c))"), `${label}: the cuts' solid`);
      // Storage: the staged vertex φ, the residency guard, the page clamp.
      if (c.includes("fn poVertexS(")) assert.ok(c.includes("if(any(v<vec3i(0))||any(v>NB_LATTICE)){return NB_PHI_FAR;}"), `${label}: poVertexS`);
      if (c.includes("fn poPageHeld(")) assert.ok(c.includes("fn poPageHeld(p:vec3i)->bool{return any(p<vec3i(0))||any(32*p>=NB_LATTICE);}"), `${label}: poPageHeld`);
      if (c.includes("fn pTileIn(")) assert.ok(c.includes("fn pTileIn(t:vec3i)->bool{return all(t>=vec3i(0))&&all(4*t<NB_BOX);}"), `${label}: pTileIn`);
    }
    const rows = [...ks.entries()].find(([, c]) => c.includes("let owned=nbInBox(c)&&poOwnerH(c);"));
    assert.ok(rows, "the h rows' domain is the box");
    assert.ok([...ks.values()].some(c => c.includes("let owned=poTileInBox(T)&&!nbTileH(cs,i);") && c.includes("if(!poTileInBox(Tn)){")), "the native rows' domain and wall slot are the box's");
    assert.match(code("pmgWallsF"), /let side=pWallFace\(c,a,NB_BOX\[a\]\);/);
    assert.match(code("pmgWallsF"), /let edge=any\(o==vec3i\(0\)\)\|\|any\(o\+vec3i\(8\)>=NB_BOX\);/);
    assert.match(code("pmgWallsC"), /let side=pWallFace\(T,a,\(NB_BOX\[a\]\+3\)\/4\);/);
    assert.ok([...ks.values()].some(c => c.includes("let wall=c[a]==0||c[a]==NB_BOX[a];")), "the h faces' wall planes");
    assert.ok([...ks.values()].some(c => c.includes("let wall=T[a]==0||4*T[a]==NB_BOX[a];")), "the native faces' wall planes");
  }
});

test("pressure walls: unpadded, no pressure kernel names the box", () => {
  for (const k of [0, 8]) {
    const L = new BrickLayout({ lattice: [64, 32, 32], f: 125, c: 8, b: 16, k, particles: 1024 });
    for (const [label, c] of kernels(L)) assert.ok(!/NB_BOX|nbInBox|poTileInBox|WallBox/.test(c), `${label}: names the box unpadded`);
  }
});

test("pressure walls: a padded box that is not whole 4h tiles is refused", () => {
  const P = new BrickLayout({ lattice: [32, 32, 32], box: [26, 16, 24], f: 125, c: 8, b: 16, k: 0, particles: 1024 });
  assert.throws(() => kernels(P), /need whole 4h tiles/);
});
