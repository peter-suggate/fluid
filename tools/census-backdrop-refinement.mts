// CPU census of the backdrop's stored detail rings at each refinement reach
// (rings refined one level, innermost first): leaves per ring as the planner's
// own classifier admits them, and classify calls. No GPU.
//   node --import tsx tools/census-backdrop-refinement.mts [scene-id]
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { sceneCellSizes_m } from "../lib/core/scene-lattice-dimensions";
import { backdropSeamForScene, compileBackdropField } from "../lib/svo/features/backdrop/backdrop-field";
import { planBackdropTiles } from "../lib/svo/features/backdrop/backdrop-terrain-tiles";
import { backdropDetailCentreLattice, backdropDetailFromPlan, createBackdropDetailClassifier } from "../lib/svo/features/backdrop/backdrop-detail";

const scene = sceneDocument(getSceneDefinition(process.argv[2] ?? "hero-garden-hose"));
const field = compileBackdropField(scene.backdrop!, backdropSeamForScene(scene));
const rings = scene.backdrop!.detailRings ?? 0;
const h = sceneCellSizes_m(scene)[0];
const lattice = backdropDetailCentreLattice(h, rings);
const probe = backdropDetailFromPlan(field, planBackdropTiles(field, { origin_m: [0, 0, 0], cellSize_m: h, firstLevel: rings, centreLattice_m: lattice }), rings);
// The scene-cell level L, with one level below it for refined rings.
let L = rings - 1;
while (8 * h * 2 ** L < 2 * probe.outerHalf_m + 4 * lattice) L += 1;
const E0 = 8 * h * 2 ** L;
const origin = [Math.round(-E0 / 2 / lattice) * lattice, Math.round((probe.anchor_m - E0 / 2) / h) * h, Math.round(-E0 / 2 / lattice) * lattice] as const;
const plan = planBackdropTiles(field, { origin_m: origin, cellSize_m: h, firstLevel: rings, centreLattice_m: lattice });
const nodeEdge_m = Array.from({ length: L + 2 }, (_, level) => { const e = 8 * h * 2 ** (L - level); return [e, e, e]; });
console.log(`scene cell ${h * 1000} mm, ${rings} rings to ${probe.outerHalf_m} m`);
for (let refined = 0; refined <= rings; refined += 1) {
  const detail = backdropDetailFromPlan(field, plan, rings, refined);
  const classifier = createBackdropDetailClassifier({ field, detail, worldOrigin_m: origin, nodeEdge_m, solverLevel: L });
  const t0 = performance.now();
  const stack = [{ level: 0, x: 0, y: 0, z: 0 }];
  while (stack.length) {
    const node = stack.pop()!;
    if (classifier.classify(node.level, node) !== 2 || node.level > L) continue;
    for (let k = 0; k < 8; k += 1) stack.push({ level: node.level + 1, x: node.x * 2 + (k & 1), y: node.y * 2 + ((k >> 1) & 1), z: node.z * 2 + (k >> 2) });
  }
  const { census } = classifier;
  console.log(JSON.stringify({ refined, leaves: census.leaves, perRing: census.leavesPerRing, classified: census.classified, plan_ms: Math.round(performance.now() - t0) }));
}
