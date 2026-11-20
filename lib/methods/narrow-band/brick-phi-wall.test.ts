// lWall (levelset-kernels.wgsl.ts): the grid's wall view, PHI1 of every NEAR_SOLID brick, the liquid's φ continued
// through the particles' contact layer into a terrain or procedural solid. The pressure reads it there in place of
// PHI (brickOwnerWGSL's view: nbFPhiBase); the level set, the seeding and the gauge keep PHI.
//
// CPU: the layout's wallView turns it on (the engine sets it where the scene binds terrain or a field and no deferred
// mask); lWall then runs after the seeding over F_ALL, gates on NEAR_SOLID, writes PHI1 for every owned vertex (PHI,
// or the lesser of PHI and the contact layer's φ in the band) and never PHI; the pressure's owner reads choose PHI1 on
// NEAR_SOLID bricks only with the view, and the PHI1 owner copy (fo1) renames the choice with it. A view with no wall
// fails explicitly.
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { ARGS, BrickLayout, LIST } from "./brick-abi";
import { fluidStagesAt } from "./brick-fluid-stages";
import { brickOwnerWGSL } from "./brick-owner.wgsl";
import { pressureCompositeKernels } from "./pressure-composite.wgsl";
import { validateBrickStages, type BrickStageContext, type BrickStages } from "./brick-stages";
import { brickStaticMaskWalls, type BrickStaticWalls } from "./brick-static.wgsl";
import "../index";

const LATTICE = [32, 32, 32] as const;
const layoutOf = (wallView: boolean) => new BrickLayout({ lattice: LATTICE, f: 200, c: 16, b: 16, k: 16, particles: 1 << 16, ...(wallView ? { wallView: true } : {}) });
function context(walls?: BrickStaticWalls, wallView = walls?.terrain !== undefined || walls?.boundary !== undefined): BrickStageContext {
  const method = getMethod("uniform-narrow-band-flip");
  const values = resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
  return { layout: layoutOf(wallView), dt_s: 1 / 60, h_m: 0.0125, gravity_mps2: [0, -9.81, 0], values, statics: true, ...(walls ? { staticWalls: walls } : {}) } as BrickStageContext;
}
const mask = brickStaticMaskWalls(LATTICE, false);
const field: BrickStaticWalls = { ...mask, boundary: { base: 4096 } };
const ground: BrickStaticWalls = { ...mask, terrain: { base: 4096, columns: [32, 32] } };
const labels = (s: BrickStages) => s.levelSet.map(p => p.kernel.label);

test("lWall: the wall view runs last in the level set, after the seeding, over F_ALL", () => {
  for (const tiered of [false, true]) {
    for (const walls of [field, ground]) {
      const on = fluidStagesAt(tiered)(context(walls));
      validateBrickStages(on);
      const ls = labels(on), at = ls.indexOf("lWall");
      assert.equal(at, ls.length - 1, `tiered ${tiered}: the level set's last pass`);
      assert.equal(ls[at - 1], "pSeed", "after the seeding (which reads PHI)");
      assert.deepEqual(on.levelSet[at]!.dispatch, { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.G1 });
      assert.ok(on.levelSet[at]!.kernel.bindings.some(b => b.name === "solid"), "it reads the static buffer");
    }
    // Voxels alone (no view): no lWall, the plain level set.
    for (const walls of [undefined, mask]) assert.ok(!labels(fluidStagesAt(tiered)(context(walls))).includes("lWall"));
    // A view with no terrain or field to continue through fails explicitly.
    assert.throws(() => fluidStagesAt(tiered)(context(mask, true)), /wall view needs the statics' terrain or procedural boundary/);
  }
});

test("lWall: the rule (NEAR_SOLID bricks, PHI1 of every vertex, never drier than PHI, the contact layer's φ)", () => {
  const code = (walls: BrickStaticWalls) => fluidStagesAt(true)(context(walls)).levelSet.find(p => p.kernel.label === "lWall")!.kernel.code;
  const f = code(field), g = code(ground);
  for (const c of [f, g]) {
    assert.match(c, /if\(\(nbFRec\(slot,NB_REC_CLASS\)&NB_CLASS_NEAR_SOLID\)==0u\)\{return;\}/);
    assert.match(c, /const LW_CONTACT:f32=0\.25;/, "the contact layer: a unit sample's radius (NB_SAMPLE_RADIUS[1])");
    assert.match(c, /if\(!\(wall\.w<LW_CONTACT&&wall\.w>-LW_DEPTH\)\)\{return own;\}/, "the band: in the layer or inside the wall within LW_DEPTH");
    assert.match(c, /let foot=clamp\(vec3f\(v\)\+\(LW_CONTACT-wall\.w\)\*wall\.xyz,/, "the foot: the layer's point along the normal");
    assert.match(c, /if\(lwWall\(vec3f\(u\)\)\.w<LW_CONTACT\)\{continue;\}/, "only the foot cell's vertices beyond the layer");
    assert.match(c, /return min\(own,sum\/total\);/, "never drier than PHI, renormalised");
    assert.match(c, /lSetPhi\(NB_PHI_F_PHI1,slot,l,view\);/, "PHI1 for every vertex (an owned one: lSetPhi)");
    assert.doesNotMatch(c, /lSetPhi\(NB_PHI_F_PHI,/, "PHI is never written");
  }
  // Each reads its own solid: the field's samples (no record: none) or the ground's.
  assert.match(f, /let b=nbBoundarySample\(p\);let g=length\(b\.xyz\);if\(abs\(b\.w\)<1e5&&/); assert.doesNotMatch(f, /nbTerrainBoundary\(p\)/);
  assert.match(g, /let t=nbTerrainBoundary\(p\);/); assert.doesNotMatch(g, /nbBoundarySample\(p\)/);
});

test("lWall: the pressure reads the view on NEAR_SOLID bricks only with the layout's wallView", () => {
  const view = "fn nbFPhiBase(s:u32)->u32{return select(NB_PHI_F_PHI,NB_PHI_F_PHI1,(nbFRec(s,NB_REC_CLASS)&NB_CLASS_NEAR_SOLID)!=0u);}";
  const plain = "fn nbFPhiBase(s:u32)->u32{return NB_PHI_F_PHI;}";
  assert.ok(brickOwnerWGSL({ phi: "phi", vel: "vel", view: true }).includes(view));
  assert.ok(brickOwnerWGSL({ phi: "phi", vel: "vel" }).includes(plain));
  assert.ok(brickOwnerWGSL({ phi: "phi", vel: "vel" }).includes("return bitcast<f32>(phi[nbFPhiBase(s)+NB_STRIDE*s+nbLocal(v)]);"), "nbFVertex reads through the choice");
  const pressure = (wallView: boolean) => Object.values(pressureCompositeKernels(layoutOf(wallView))).flatMap(k => (k && typeof k === "object" && "code" in k ? [(k as { code: string }).code] : []));
  const on = pressure(true), off = pressure(false);
  assert.ok(on.length > 0 && on.some(c => c.includes(view)) && !on.some(c => c.includes(plain)), "the view's reads");
  assert.ok(off.some(c => c.includes(plain)) && !off.some(c => c.includes(view)), "PHI everywhere without it");
  // The level set's PHI1 owner copy (fluid-grid fo1) takes its own copy of the choice, so both declare once.
  const ls = fluidStagesAt(true)(context(field)).levelSet.map(p => p.kernel.code).find(c => c.includes("fn fo1FVertex"));
  if (ls) { assert.ok(ls.includes("fn fo1FPhiBase")); assert.equal(ls.split("fn foFPhiBase(").length, 2); }
});
