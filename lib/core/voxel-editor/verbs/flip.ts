import type { VoxelRegionVerbPlugin } from "../verb";
import { blockOfRegion, contentsReplacementPatches, flipBlock, flippedRegion } from "../voxel-grid";
import { blockSolidCount, plural, transformUnavailable } from "./region";

const AXES = ["X", "Y", "Z"] as const;

/** The axis a variant of an axis verb fixed. A run without one is a wiring mistake: there is no default axis. */
export function verbAxis(values: Readonly<Record<string, number>>, verb: string): 0 | 1 | 2 {
  const axis = values.axis;
  if (axis !== 0 && axis !== 1 && axis !== 2) throw new Error(`${verb} needs an axis: run it as one of its X, Y or Z variants.`);
  return axis;
}

/**
 * Mirror the selection's solids inside the selection's own box.
 *
 * The box does not move: the cell at one end of the chosen axis trades places
 * with the cell at the other. A masked selection's mask is mirrored with it,
 * so the selection is still holding the same voxels afterwards.
 */
export const flipVerb: VoxelRegionVerbPlugin = {
  id: "flip",
  version: 1,
  ui: {
    label: "Flip", hint: "Mirror the selection's voxels inside its box", group: "Transform", order: 10,
    icon: "M12 3v18M8 7l-4 5l4 5zM16 7l4 5l-4 5z",
    variants: AXES.map((name, axis) => ({
      id: name.toLowerCase(), label: name, hint: `Mirror the selection's voxels along ${name}`, values: { axis },
    })),
  },
  unavailable: transformUnavailable,
  run: ({ reader, region, values }) => {
    const axis = verbAxis(values, "Flip");
    const contents = blockOfRegion(reader, region);
    return {
      patches: contentsReplacementPatches(reader, region, { minimum: region.minimum, block: flipBlock(contents, axis) }),
      region: flippedRegion(region, axis),
      caption: `Flipped ${plural(blockSolidCount(contents), "voxel")} along ${AXES[axis]}`,
    };
  },
};
