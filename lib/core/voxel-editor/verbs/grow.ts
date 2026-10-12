import { materialControl } from "../controls";
import { sceneLatticeDimensions } from "../../scene-lattice";
import type { SolidWorldCoordinate } from "../../solid-world";
import type { VoxelRegionVerbPlugin } from "../verb";
import { blockOfRegion, contentsReplacementPatches, grownBlock, solidRegionOfBlock } from "../voxel-grid";
import { blockSolidCount, plural, transformUnavailable } from "./region";

/**
 * Add one layer of voxels on every exposed side of the selection's solids.
 *
 * Face neighbours only, in the chosen material; a side against something
 * already solid is not exposed and gains nothing. The layer stops at the edge
 * of the lattice — the cells beyond are the vessel's shell — and the selection
 * afterwards is exactly the thickened solids, so growing again grows those.
 */
export const growVerb: VoxelRegionVerbPlugin = {
  id: "grow",
  version: 1,
  ui: {
    label: "Grow", hint: "Add one layer of voxels on every exposed side, in the chosen material",
    group: "Edit", order: 4, menu: "Shape",
    icon: "M8 8h8v8H8zM12 5V2M12 22v-3M5 12H2M22 12h-3",
  },
  controls: [materialControl],
  unavailable: transformUnavailable,
  run: ({ scene, reader, region, materialId }) => {
    const contents = blockOfRegion(reader, region);
    const minimum: SolidWorldCoordinate = [region.minimum[0] - 1, region.minimum[1] - 1, region.minimum[2] - 1];
    const halo = reader.read(minimum, [region.maximumExclusive[0] + 1, region.maximumExclusive[1] + 1,
      region.maximumExclusive[2] + 1]);
    // The lattice's edge is a wall to a growing piece, whether or not a shell stands there: read as solid, nothing is added past it.
    const dimensions = sceneLatticeDimensions(scene), [sx, sy, sz] = halo.size;
    const outside = (value: number, axis: number) => minimum[axis]! + value < 0 || minimum[axis]! + value >= dimensions[axis]!;
    if (minimum.some((value, axis) => value < 0 || value + halo.size[axis]! > dimensions[axis]!)) {
      for (let z = 0, cell = 0; z < sz; z += 1) for (let y = 0; y < sy; y += 1) for (let x = 0; x < sx; x += 1, cell += 1) {
        if (outside(x, 0) || outside(y, 1) || outside(z, 2)) halo.solidFraction[cell] = 255;
      }
    }
    const grown = grownBlock(contents, halo, materialId);
    const added = blockSolidCount(grown) - blockSolidCount(contents);
    if (added === 0) throw new Error("Every side of these voxels is against something solid: there is nowhere to grow.");
    return {
      patches: contentsReplacementPatches(reader, region, { minimum, block: grown }),
      region: solidRegionOfBlock(grown, minimum)!,
      caption: `Grew by ${plural(added, "voxel")}`,
    };
  },
};
