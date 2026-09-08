import type { SceneryGroupNode } from "./scenery-graph";
import { planOakV2 } from "./voxel-scenery/oak-v2";

/** Scene placement/materials; the reusable v2 planner owns only tree geometry. */
export function heroGardenCloudTree(): SceneryGroupNode {
  const { node } = planOakV2({
    key: "tree", seed: 0x10a2, scale_m: 0.85,
    bark: { palette: "clay", value: 0.955, surface: "architectural" },
    foliage: { palette: "clay", value: 0.975, surface: "foliage" },
  });
  return {
    ...node,
    place: {
      units: "metres", anchor: "terrain", ground: [0.53, -0.15],
      position: { x: 0.53, y: 0, z: -0.15 },
    },
  };
}
