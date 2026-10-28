import type { SceneryGraph, SceneryNode } from "./scenery-graph";
import { ALPINE_TREELINE } from "./alpine-treeline";
import type { TerrainDescription } from "./terrain";
import { terrainHeightAt } from "./terrain";
import { meshAssetHeader } from "./imported-assets/asset-library";
import { quaternionRotate } from "./rigid-body";

/** Place authored Blender geometry at its original origin. Sampled FVX fields
 * are centred on their bounds; compensate that centre before rotation. */
function imported(id: string, asset: string, x: number, y: number, z: number, yaw = 0): SceneryNode {
  const header = meshAssetHeader(asset);
  const orientation = { w: Math.cos(yaw / 2), x: 0, y: Math.sin(yaw / 2), z: 0 };
  const center = quaternionRotate(orientation, {
    x: header.origin[0]! + header.dimensions[0]! * header.cellSize / 2,
    y: header.origin[1]! + header.dimensions[1]! * header.cellSize / 2,
    z: header.origin[2]! + header.dimensions[2]! * header.cellSize / 2,
  });
  const foliage = (asset.includes("cedar") || asset.startsWith("alpine-fir-"));
  return { kind: "field-program", id, tags: ["blender-alpine", foliage ? "tree" : "static-environment"],
    place: { units: "metres", position: { x: x + center.x, y: y + center.y, z: z + center.z }, orientation },
    program: { ops: [], result: 0, sampledVolume: { asset } },
    material: { colorLinear: header.materials[0]!.baseColorLinear, surface: foliage ? "foliage" : "stone" },
  };
}

export function alpineLakeScenery(terrain: TerrainDescription): SceneryGraph {
  const nodes: SceneryNode[] = [{ kind: "terrain-shell", id: "alpine/shore", materialModel: "forest-terrain", waterline_m: 2.3 }];
  nodes.push(imported("alpine/glaciated-ridge", "alpine-mountain-ridge", 0, 3.8, -43));
  // Closed mineral cobbles authored against the exact dry-bank height samples.
  // They add visible shore relief without replacing the physical terrain bed.
  nodes.push(imported("alpine/dry-shore-scree", "alpine-shore-scree", 0, 0, -36));
  let seed = 20261028;
  const random = () => { seed = Math.imul(seed ^ seed >>> 15, 1 | seed); seed ^= seed + Math.imul(seed ^ seed >>> 7, 61 | seed); return ((seed ^ seed >>> 14) >>> 0) / 4294967296; };
  // Preserve the existing mineral placement sequence while the independently
  // authored treeline changes. Five irregular groups overlap in depth.
  for (let i = 0; i < 42 * 3; i++) random();
  ALPINE_TREELINE.forEach(([x, y, z, yaw, variant], i) => {
    nodes.push(imported(`alpine/fir-${i}`, `alpine-fir-${variant}`, x, y, z, yaw));
  });
  for (const [i, x, z] of [[0, -31, 8], [1, 32, 3], [2, -33, -16], [3, 35, -21]]) {
    nodes.push(imported(`alpine/near-fir-${i}`, "alpine-fir-1", x!, terrainHeightAt(terrain, x!, z!), z!, i!));
  }
  // Rounded, real Blender stones lie both above and below the waterline.
  // They are decorative pebbles, not fluid barriers; the continuous terrain
  // remains the authoritative collision bed.
  for (let i = 0; i < 340; i++) {
    const x = -28 + random() * 56, z = -36 + random() * 72;
    const r = Math.hypot(x / 30, z / 38);
    if (r < .66 || r > 1.08) continue;
    nodes.push(imported(`alpine/shore-stone-${i}`, "forest-rock-0", x,
      terrainHeightAt(terrain, x, z) + .035, z, random() * Math.PI * 2));
  }
  // Dry-bank boulders break the smooth shore into a natural mineral edge.
  // Every origin lies above the lake level, outside its physical wet basin.
  for (let i = 0; i < 48; i++) {
    const angle = (random() - .5) * 2.6, radius = 1.025 + random() * .075;
    const x = Math.sin(angle) * 30 * radius, z = -Math.cos(angle) * 38 * radius;
    nodes.push(imported(`alpine/bank-boulder-${i}`, i % 4 ? "forest-rock-1" : "forest-rock-2", x,
      terrainHeightAt(terrain, x, z) - .08, z, random() * Math.PI * 2));
  }
  return { palettes: {}, nodes };
}
