/** Export shared composition/physics datums; Blender authors the actual meshes. */
import { mkdirSync, writeFileSync } from "node:fs";
import { HERO_GARDEN_VESSEL, HERO_GARDEN_SET_SEED, HERO_GARDEN_WATERLINE_M, HERO_GARDEN_HOSE_MOUTH_M, HERO_GARDEN_HOSE_IMPACT_M } from "../../lib/core/hero-garden-scene";
import { heroGardenCloudTree } from "../../lib/core/hero-garden-tree";
import { heroGardenBlenderCamera } from "../../lib/core/hero-garden-blender-scene";
import { heroGardenStressStands } from "../../lib/core/hero-garden-stress-scene";
import { heroGardenLayout } from "../../lib/core/voxel-scenery/hero-layout";
import { pondVesselHeightAt, pondVesselPlanCurve } from "../../lib/core/voxel-scenery/pond-vessel";
import { stoneSetBoulderStations, stoneSetSteppingStations } from "../../lib/core/voxel-scenery/stone-set";
const spec = { vessel: HERO_GARDEN_VESSEL, seed: HERO_GARDEN_SET_SEED, waterline_m: HERO_GARDEN_WATERLINE_M };
const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL);
const nx = 451, nz = 301, spacing = .004;
const heights = Array.from({ length: nz }, (_, z) => Array.from({ length: nx }, (_, x) => pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, -.9 + x * spacing, -.6 + z * spacing)));
const layout = { version: 1, oak: heroGardenCloudTree(), curve, vessel: HERO_GARDEN_VESSEL, waterline: HERO_GARDEN_WATERLINE_M,
  mouth: HERO_GARDEN_HOSE_MOUTH_M, impact: HERO_GARDEN_HOSE_IMPACT_M, camera: heroGardenBlenderCamera,
  terrain: { nx, nz, spacing, heights }, stands: heroGardenStressStands(81),
  boulders: stoneSetBoulderStations(spec), stepping: stoneSetSteppingStations(spec),
  layout: heroGardenLayout() };
mkdirSync("assets/hero-garden-blender", { recursive: true });
writeFileSync("assets/hero-garden-blender/layout.json", JSON.stringify(layout));
