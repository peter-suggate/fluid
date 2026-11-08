/**
 * The Bridson 2002 scenes' cloth solid authoring: the friction of each baked
 * box (the ground slab) and the analytic solids behind the procedural bake
 * (legs, frustum, and the curtain ball and spinning sphere while they hold
 * still), in scene metres, at their t = 0 pose. Moving solids are not here:
 * the cloth system stops at each one's pending time (lane C2, M6).
 */
import type { SceneDescription } from "../../../core/model";
import type { ClothSceneSolidAuthoring } from "../../contact/scene-solids";
import type { ClothSolidPart, V3 } from "../../contact/solid-field";
import { bridson2002SceneLayout } from "./scene-description";
import { BRIDSON_2002_SCENES } from "./scenes";
import { bridsonShapeBox, bridsonShapeReach, bridsonSolidDistance, bridsonSolidPose, type BridsonSolid } from "./solids";

const IDENTITY_TOLERANCE = 1e-12, ALIGNED_TOLERANCE_CELLS = 1e-6;

/** A solid's scene-space box at t = 0. */
function sceneBox(solid: BridsonSolid, t: V3): { lo: V3; hi: V3; rotated: boolean } {
  const { position_m: c, orientation: q } = bridsonSolidPose(solid, 0);
  const rotated = Math.abs(Math.abs(q.w) - 1) > IDENTITY_TOLERANCE;
  const r = bridsonShapeReach(solid.shape), b = bridsonShapeBox(solid.shape);
  const lo = rotated ? [c.x - r, c.y - r, c.z - r] : [c.x - b.halfX, c.y + b.yMin, c.z - b.halfZ];
  const hi = rotated ? [c.x + r, c.y + r, c.z + r] : [c.x + b.halfX, c.y + b.yMax, c.z + b.halfZ];
  return { lo: lo.map((v, a) => v + t[a]!) as unknown as V3, hi: hi.map((v, a) => v + t[a]!) as unknown as V3, rotated };
}

export function bridson2002ClothSceneSolids(scene: SceneDescription): ClothSceneSolidAuthoring {
  const source = BRIDSON_2002_SCENES.map(make => make()).find(s => s.id === scene.sceneId);
  if (!source) throw new Error(`Cloth contact: no Bridson 2002 scene "${scene.sceneId}"`);
  const layout = bridson2002SceneLayout(source), h = layout.cell_m;
  const t: V3 = [layout.translation_m.x, layout.translation_m.y, layout.translation_m.z];
  const origin: V3 = [layout.lower_m.x + t[0], layout.lower_m.y + t[1], layout.lower_m.z + t[2]];
  const top = layout.dimensions.map((n, a) => origin[a]! + n * h);
  const onLattice = (v: number, a: number) => Math.abs((v - origin[a]!) / h - Math.round((v - origin[a]!) / h)) < ALIGNED_TOLERANCE_CELLS;
  const held = source.solids.filter(s => layout.held.includes(s.id));
  // The bake's rule: an unrotated box with every face on a lattice plane is a fill; everything else is sampled.
  const boxes: { solid: BridsonSolid; lo: V3; hi: V3 }[] = [], procedural: ClothSolidPart[] = [];
  for (const solid of held) {
    const box = sceneBox(solid, t);
    if (solid.shape.kind === "box" && !box.rotated && [...box.lo, ...box.hi].every((v, i) => onLattice(v, i % 3))) {
      const clip = (v: number, a: number) => Math.max(origin[a]!, Math.min(top[a]!, v));
      boxes.push({ solid, lo: box.lo.map(clip) as unknown as V3, hi: box.hi.map(clip) as unknown as V3 });
      continue;
    }
    procedural.push({ kind: "sdf", lo: box.lo, hi: box.hi, mu: solid.friction,
      sdf: p => bridsonSolidDistance(solid, { x: p[0] - t[0], y: p[1] - t[1], z: p[2] - t[2] }, 0) });
  }
  return {
    procedural,
    boxFriction(patch, box) {
      const same = (u: V3, v: V3) => u.every((x, a) => Math.abs(x - v[a]!) <= ALIGNED_TOLERANCE_CELLS * h);
      const owner = boxes.find(b => same(b.lo, box.lo) && same(b.hi, box.hi));
      if (!owner) throw new Error(`Cloth contact: ${scene.sceneId} has no solid for the box fill ${patch.minimum}..${patch.maximumExclusive}`);
      return owner.solid.friction;
    },
  };
}
