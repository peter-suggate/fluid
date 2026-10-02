/** Foliage through the production renderer. FOLIAGE_SPECIMEN=oak|bough|bonsai.
 * FOLIAGE_ISOLATED=1 removes terrain and backdrop for a bounded high-resolution specimen. */
import { createScene as oakScene, camera as oakCamera } from "./oak-v2";
import { bonsaiNodes, BONSAI_POND_CANOPY } from "../../lib/core/voxel-scenery/bonsai";
import { isSceneryShellNode, type SceneryGroupNode } from "../../lib/core/scenery-graph";
import { terrainHeightAt } from "../../lib/core/terrain";

export function createScene() {
  const scene = oakScene();
  const specimen = process.env.FOLIAGE_SPECIMEN ?? "oak";
  const tree = scene.scenery!.nodes.find(n => n.id === "tree") as SceneryGroupNode;
  const replacement = specimen === "bonsai" ? bonsaiNodes({ ...BONSAI_POND_CANOPY,
    key: "tree", seed: 4258, at_m: [.53, -.15], lean: [1, 0],
    groundHeightAt: (x, z) => terrainHeightAt(scene.terrain, x, z),
  })[0] : specimen === "bough" ? { ...tree, children: tree.children.map(group => ({
    ...group, children: (group as SceneryGroupNode).children.filter(n => n.id.includes("/bough-0/") || n.id.includes("/trunk-")),
  })) } : tree;
  const isolated = process.env.FOLIAGE_ISOLATED === "1";
  return { ...scene, ...(isolated ? { terrain: undefined, backdrop: undefined } : {}), scenery: { ...scene.scenery!, nodes: scene.scenery!.nodes
    .filter(n => n.id === "tree" || isSceneryShellNode(n))
    .map(n => n.id === "tree" ? replacement : n) } };
}
export const camera = process.env.FOLIAGE_SPECIMEN === "bonsai"
  ? { ...oakCamera, distance_m: 1.8, target_m: { x: .6, y: .36, z: -.15 } }
  : oakCamera;
