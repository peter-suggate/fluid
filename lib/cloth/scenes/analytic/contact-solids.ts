/**
 * The analytic cloth scenes' solid authoring for cloth contact: the friction
 * of each box fill (the Peirce ledge) and the analytic solids behind the
 * procedural bake (the incline wedges), in scene metres. The container shell
 * (the floor every scene lands on) is the contact lane's own, frictionless.
 *
 * `clothSolidSourceForScene` (lib/cloth/contact/scene-solids.ts, lane C2)
 * dispatches scenes whose id starts with `CLOTH_ANALYTIC_SCENE_PREFIX` here.
 */
import type { SceneDescription } from "../../../core/model";
import type { ClothSceneSolidAuthoring } from "../../contact/scene-solids";
import type { ClothSolidPart, V3 } from "../../contact/solid-field";
import { clothAnalyticLatticeOrigin, clothAnalyticSolidBounds, clothAnalyticSolidDistance, clothAnalyticSolidIsFill } from "./scene-description";
import { CLOTH_ANALYTIC_SCENES } from "./scenes";

export const CLOTH_ANALYTIC_SCENE_PREFIX = "cloth-analytic-";

export function clothAnalyticSceneSolids(scene: SceneDescription): ClothSceneSolidAuthoring {
  const source = CLOTH_ANALYTIC_SCENES.map(make => make()).find(s => s.id === scene.sceneId);
  if (!source) throw new Error(`Cloth contact: no analytic cloth scene "${scene.sceneId}"`);
  const h = source.cell_m, o = clothAnalyticLatticeOrigin(source);
  const boxes: { lo: V3; hi: V3; mu: number }[] = [], procedural: ClothSolidPart[] = [];
  for (const solid of source.solids) {
    const b = clothAnalyticSolidBounds(solid.shape);
    if (clothAnalyticSolidIsFill(solid, source)) { boxes.push({ lo: b.lo, hi: b.hi, mu: solid.friction }); continue; }
    procedural.push({ kind: "sdf", lo: b.lo, hi: b.hi, mu: solid.friction,
      sdf: p => clothAnalyticSolidDistance(solid.shape, { x: p[0], y: p[1], z: p[2] }) });
  }
  return {
    ...(procedural.length ? { procedural } : {}),
    boxFriction(patch, box) {
      const same = (u: V3, v: V3) => u.every((x, a) => Math.abs(x - v[a]!) <= 1e-6 * h);
      const owner = boxes.find(b => same(b.lo, box.lo) && same(b.hi, box.hi));
      if (!owner) throw new Error(`Cloth contact: ${scene.sceneId} has no solid for the box fill ${patch.minimum}..${patch.maximumExclusive} (lattice origin ${o.join(", ")})`);
      return owner.mu;
    },
  };
}
