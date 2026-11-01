import type { InitialLiquidVolume, SceneDescription } from "../../lib/core/model";
import { createBackyardPoolPartyScene } from "../../lib/core/backyard-pool-party-scene";

export type CoarseVolumeCase = "film" | "drop" | "medium-drop" | "small-drop" | "slab" | "resolved-slab";
export function coarseVolumeScene(kind: CoarseVolumeCase, coarse = true, rest = false): SceneDescription {
  const scene = createBackyardPoolPartyScene();
  scene.container = { ...scene.container, width_m: 3.2, height_m: 3.2, depth_m: 3.2 };
  scene.solidBoundary = undefined; scene.rigidBodies = []; scene.solidVoxels = [];
  let liquid: InitialLiquidVolume;
  if (kind === "drop" || kind === "medium-drop" || kind === "small-drop") liquid = { shape: "sphere", center_m: { x: 0, y: 2.2, z: 0 }, radius_m: kind === "drop" ? 0.5 : kind === "medium-drop" ? 0.3 : 0.2 };
  else if (kind === "film") liquid = { shape: "box", min_m: { x: -0.6, y: 0, z: -0.6 }, max_m: { x: 0.6, y: 0.08, z: 0.6 } };
  // 0.3 m is SIX h cells, but only 1.5 coarse cells. Keep a genuinely
  // six-coarse-cell (1.2 m) case alongside the handoff's thin slab.
  else liquid = { shape: "box", min_m: { x: -0.6, y: 0, z: -0.6 }, max_m: { x: kind === "slab" ? -0.3 : 0.6, y: 1.2, z: 0.6 } };
  scene.fluid.initialLiquidVolumes = [liquid];
  if (rest) scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
  scene.fluid.refinementRegions = coarse ? [{ id: "all-4h", min_m: { x: -10, y: -10, z: -10 }, max_m: { x: 10, y: 10, z: 10 }, rule: "minimum-cell-size", minimumCellSize_cells: 4, maximumCellSize_cells: 4 }] : [];
  return scene;
}
