import type { SolidWorldCoordinate } from "../../solid-world";
import type { VoxelRegionVerbPlugin, VoxelVerbAxis, VoxelVerbVariant, VoxelVerbView } from "../verb";
import { blockOfRegion, contentsReplacementPatches, translatedRegion } from "../voxel-grid";
import { assertInsideLattice, blockSolidCount, plural, transformUnavailable } from "./region";

/** The far step: what Shift makes of an arrow. */
const FAR_STEP = 8;

interface Direction {
  readonly id: string;
  readonly label: string;
  readonly key: string;
  readonly icon: string;
  readonly toward: string;
  /** The lattice axis this direction lies along, as the reader is looking at it. */
  readonly along: (view: VoxelVerbView) => VoxelVerbAxis;
}

const against = (axis: VoxelVerbAxis): VoxelVerbAxis => ({ axis: axis.axis, sign: axis.sign > 0 ? -1 : 1 });

/**
 * Six directions, named by what they are on the screen. The four arrows move
 * across the floor — left and right along whichever of X and Z lies nearer the
 * screen's own horizontal, up and down along the other, away from the camera
 * and toward it — and the page keys move along Y.
 */
const DIRECTIONS: readonly Direction[] = [
  { id: "left", label: "←", key: "ArrowLeft", icon: "M19 12H5M11 6l-6 6l6 6", toward: "left", along: (view) => against(view.right) },
  { id: "right", label: "→", key: "ArrowRight", icon: "M5 12h14M13 6l6 6l-6 6", toward: "right", along: (view) => view.right },
  { id: "away", label: "↑", key: "ArrowUp", icon: "M12 19V5M6 11l6-6l6 6", toward: "away from the camera", along: (view) => view.forward },
  { id: "near", label: "↓", key: "ArrowDown", icon: "M12 5v14M6 13l6 6l6-6", toward: "toward the camera", along: (view) => against(view.forward) },
  { id: "up", label: "⇞", key: "PageUp", icon: "M12 20V8M6 14l6-6l6 6M5 4h14", toward: "up", along: () => ({ axis: 1, sign: 1 }) },
  { id: "down", label: "⇟", key: "PageDown", icon: "M12 4v12M6 10l6 6l6-6M5 20h14", toward: "down", along: () => ({ axis: 1, sign: -1 }) },
];

const variants: readonly VoxelVerbVariant[] = DIRECTIONS.flatMap((direction, index) => [
  { id: direction.id, label: direction.label, icon: direction.icon, keys: [direction.key],
    hint: `Move the selection's voxels one voxel ${direction.toward}  ·  ⇧ for ${FAR_STEP}`,
    values: { direction: index, step: 1 } },
  // The same move, far: a key only, so the bar stays six marks wide.
  { id: `${direction.id}-far`, label: `${direction.label}${FAR_STEP}`, icon: direction.icon, keys: [`Shift+${direction.key}`],
    values: { direction: index, step: FAR_STEP }, surface: "keys" as const },
]);

/**
 * Move the selection's solids, and the selection with them.
 *
 * A moved voxel overwrites whatever it lands on; the cell it left is empty.
 * One move is one edit — the cleared trailing cells and the written leading
 * ones are a single replacement — so one undo puts a nudge back exactly.
 */
export const nudgeVerb: VoxelRegionVerbPlugin = {
  id: "nudge",
  version: 1,
  ui: {
    label: "Nudge", hint: "Move the selection's voxels one voxel; hold Shift for eight", group: "Transform", order: 12,
    icon: "M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M21 12l-3-3M21 12l-3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3",
    variants,
    // Six one-voxel moves are keys and marks, not wedges: a ring that closes after every step is the wrong instrument.
    ring: false,
  },
  unavailable: transformUnavailable,
  run: ({ scene, reader, region, values, view }) => {
    const direction = DIRECTIONS[values.direction ?? -1];
    const step = values.step;
    if (!direction || !Number.isSafeInteger(step) || step! < 1) {
      throw new Error("Nudge needs a direction and a step: run it as one of its variants.");
    }
    const { axis, sign } = direction.along(view);
    const delta = [0, 0, 0];
    delta[axis] = sign * step!;
    const moved = translatedRegion(region, delta as unknown as SolidWorldCoordinate);
    const contents = blockOfRegion(reader, region);
    const placed = { minimum: moved.minimum, block: contents };
    assertInsideLattice({ scene }, placed);
    return {
      patches: contentsReplacementPatches(reader, region, placed),
      region: moved,
      caption: `Moved ${plural(blockSolidCount(contents), "voxel")} ${plural(step!, "voxel")} along ${sign > 0 ? "+" : "−"}${"XYZ"[axis]}`,
    };
  },
};
