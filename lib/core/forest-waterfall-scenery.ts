import type { SceneryGraph, SceneryNode } from "./scenery-graph";
import { terrainHeightAt, type TerrainDescription } from "./terrain";
import { meshAssetHeader } from "./imported-assets/asset-library";
import { quaternionRotate } from "./rigid-body";
import instances from "./imported-assets/forest-waterfall-instances.json";

/** A shared native volume is centred on its bake bounds, whereas Blender
 * instances are planted at their mesh origins. Rotate that offset too. */
export function forestWaterfallImportedNodes(terrain: TerrainDescription): SceneryNode[] {
  return instances.map(instance => {
    const header = meshAssetHeader(instance.asset);
    const [w, x, y, z] = instance.orientation;
    const orientation = { w: w!, x: x!, y: y!, z: z! };
    const center = header.origin.map((value, axis) => value + header.dimensions[axis]! * header.cellSize / 2);
    const offset = quaternionRotate(orientation, { x: center[0]!, y: center[1]!, z: center[2]! });
    const [px, py, pz] = instance.position;
    const tree = /^forest-(tree|alder|cedar)-/.test(instance.asset);
    const groundPlant = /forest-(fern|broadleaf|sedge)/.test(instance.asset)
      && !/cliff|crevice|wall/i.test(instance.sourceName);
    const foliage = tree || /forest-(fern|broadleaf|sedge|ivy)/.test(instance.asset);
    const fallback = (foliage ? header.materials.find(material => /leaf|blade|foliage/i.test(material.name)) : undefined)
      ?? header.materials[0]!;
    // Native terrain is the shared visual/physical authority. Quantized
    // botanical prototypes retain roots at local zero and follow that surface.
    const height = tree || groundPlant ? terrainHeightAt(terrain, px!, pz!) : py!;
    return {
      kind: "field-program", id: instance.id,
      tags: ["blender-forest", tree ? "tree" : "static-environment"],
      place: { units: "metres", position: { x: px! + offset.x, y: height + offset.y, z: pz! + offset.z }, orientation },
      program: { ops: [], result: 0, sampledVolume: { asset: instance.asset } },
      material: { colorLinear: fallback.baseColorLinear,
        surface: foliage ? "foliage" : instance.asset === "forest-fallen-log" ? "wood" : "stone" },
    } satisfies SceneryNode;
  });
}

/** Static surroundings; all visible liquid belongs to the live solver. */
export function forestWaterfallScenery(terrain: TerrainDescription): SceneryGraph {
  return { palettes: {}, nodes: [
    { kind: "terrain-shell", id: "forest/ground", materialModel: "forest-terrain", waterline_m: 3.1 },
    ...forestWaterfallImportedNodes(terrain),
  ] };
}
