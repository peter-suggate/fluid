import { getSceneDefinition } from "../../lib/core/scenes";
import { sceneDocumentAtLattice } from "../../lib/core/scene-definition";
import { createReservoirSeepScene, RESERVOIR_SEEP_ID } from "../../lib/core/reservoir-seep-scene";
import { createBackyardPoolPartyScene } from "../../lib/core/backyard-pool-party-scene";

/** The stone tank: the reservoir seep's box at pages [6, 10, 4] and 5 mm,
 * solids cleared, still water 280 cells (1.4 m) deep, a census start. The
 * backyard ball is a free body (two-way coupled), r 0.15 m and 2500 kg/m3,
 * resting on the surface (or with its centre `depth_m` under it) and moving
 * down at 3 m/s. */
export const STONE_TANK = { pages: [6, 10, 4], water: 280, cell_m: 0.005, radius_m: 0.15, speed_m_s: 3 } as const;

export function uniformStoneTank(bulkTier: "16h" | "off", depth_m?: number) {
  const { pages, water, cell_m, radius_m: R, speed_m_s } = STONE_TANK;
  const base = getSceneDefinition(RESERVOIR_SEEP_ID);
  const { scene } = sceneDocumentAtLattice({ ...base, buildAt: (l: { cellSize_m: number }) => createReservoirSeepScene(l.cellSize_m, { widthScale: pages[0] / 16, heightScale: pages[1] / 8, depthScale: pages[2] / 2 }) }, { cellSize_m: cell_m });
  const s = scene as any, w = s.container.width_m, d = s.container.depth_m, top = water * cell_m;
  s.solidVoxels = [];
  s.fluid.initialLiquidVolumes = [{ shape: "box", min_m: { x: -w / 2, y: 0, z: -d / 2 }, max_m: { x: w / 2, y: top, z: d / 2 } }];
  s.fluid.initialVelocity_m_s = { x: 0, y: 0, z: 0 };
  delete s.fluid.pagePool; s.fluid.refinementRegions = [];
  s.fluid.dynamicViscosity_Pa_s = 0; s.fluid.surfaceTension_N_m = 0;
  s.fluid.tiers = { start: "census" };
  const ball = createBackyardPoolPartyScene().rigidBodies[0]!;
  s.rigidBodies = [{ ...ball, id: "stone", dimensions_m: { x: R, y: R, z: R }, density_kg_m3: 2500, position_m: { x: 0, y: depth_m === undefined ? top + R : top - depth_m, z: 0 }, linearVelocity_m_s: { x: 0, y: -speed_m_s, z: 0 } }];
  const values = { ...base.methodProfile!.overrides, bulkTier, volumeControlSeconds: 0 } as never;
  return { scene, values };
}
