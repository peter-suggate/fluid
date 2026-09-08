/** Production-path oak comparison. OAK_TREE_JSON can select a saved baseline. */
import { readFileSync } from "node:fs";
import { DEFAULT_FINEST_CELL_SIZE_M } from "../../lib/core/model";
import { getSceneDefinition } from "../../lib/core/scenes";
import { sceneDocumentAtLattice } from "../../lib/core/scene-definition";
import { heroGardenCloudTree } from "../../lib/core/hero-garden-tree";
import type { SceneryNode } from "../../lib/core/scenery-graph";
import { heroPreviewCamera } from "./hero-still";
export const createScene = () => {
  const depth = Number(process.env.FLUID_SVO_DRY_FRAME_ENVIRONMENT_REFINEMENT ?? 3);
  const scene = sceneDocumentAtLattice(getSceneDefinition("hero-garden-hose"), {
    cellSize_m: DEFAULT_FINEST_CELL_SIZE_M, detailCellSize_m: DEFAULT_FINEST_CELL_SIZE_M / 2 ** depth,
  }).scene;
  const tree: SceneryNode = process.env.OAK_TREE_JSON
    ? JSON.parse(readFileSync(process.env.OAK_TREE_JSON, "utf8")) : heroGardenCloudTree();
  return { ...scene, scenery: { ...scene.scenery!, nodes: scene.scenery!.nodes.map(node => node.id === "tree" ? tree : node) } };
};
export const camera = {
  ...heroPreviewCamera(),
  azimuth_rad: Number(process.env.OAK_AZIMUTH ?? heroPreviewCamera().azimuth_rad),
  elevation_rad: 0.18, distance_m: 2.2, tanHalfFov: 0.34,
  target_m: { x: 0.48, y: 0.60, z: -0.15 },
};
