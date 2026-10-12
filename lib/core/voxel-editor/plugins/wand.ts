import { containerShellContains } from "../../editor-solid-voxel";
import { voxelRegionHighlight } from "../../editor-voxel-region";
import { sceneLatticeDimensions } from "../../scene-lattice";
import type { SolidWorldCoordinate as Cell } from "../../solid-world";
import { byMaterialControl, selectionModeCaption, selectionModeControl, shellControl, toolSelectionMode } from "../controls";
import { boxHighlight, toolSurface } from "../geometry";
import { toolReader, type ToolContext, type ToolGesture, type ToolUpdate, type VoxelToolPlugin } from "../plugin";
import type { VoxelReader } from "../voxel-access";
import { DEFAULT_REGION_REACH, connectedRegion } from "../voxel-grid";

/**
 * Select the piece under the pointer: every solid cell joined face to face to
 * the one clicked, optionally only those of its material.
 *
 * It proposes a selection and no patches. The search runs once, at the press —
 * the cell is the question, so where the pointer goes afterwards changes
 * nothing — and not at all for the ghost under a bare pointer, which only
 * marks the cell a click would start from.
 */
function beginWand(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const surface = toolSurface(context, ray);
  if (!surface?.picked) return undefined;
  const seed = surface.coordinate;
  // Everything standing in the tank touches its floor, so a search that walked
  // into the shell would answer "the floor, and everything on it" for every
  // click — and the next Delete would take the floor. With the walls switch
  // off the shell is not part of any piece, and is not a piece to click.
  //
  // The same switch keeps the search inside the lattice. A selection is for the
  // verbs, and they refuse to set a voxel down past the lattice's edge: one
  // stray cell out there would make the whole piece immovable.
  const walls = values[shellControl.id] === 1;
  const dimensions = sceneLatticeDimensions(scene);
  const piece = (coordinate: Cell) => !containerShellContains(scene, coordinate)
    && coordinate.every((value, axis) => value >= 0 && value < dimensions[axis]!);
  if (!walls && !piece(seed)) return undefined;
  const all = toolReader(context);
  const reader: VoxelReader = walls ? all : {
    scene: all.scene,
    fraction: (coordinate) => all.fraction(coordinate),
    solid: (coordinate) => piece(coordinate) && all.solid(coordinate),
    material: (coordinate) => all.material(coordinate),
    pick: (input, options) => all.pick(input, options),
    read: (minimum, maximumExclusive) => all.read(minimum, maximumExclusive),
  };
  const cell = { minimum: seed, maximumExclusive: seed.map((v) => v + 1) as unknown as Cell };
  if (context.hover) {
    return { surface: true, update: () => ({ patches: [], tone: "region", highlight: boxHighlight(scene, cell),
      caption: "WAND · click to select the connected piece" }) };
  }
  let searched = false, found: ToolUpdate | undefined;
  return { surface: true, update(): ToolUpdate | undefined {
    // A search that failed has said so once; the release sample must not say it again.
    if (searched) return found;
    searched = true;
    const piece = connectedRegion(reader, seed, { byMaterial: values[byMaterialControl.id] === 1 });
    const selection = { minimum: piece.minimum, maximumExclusive: piece.maximumExclusive, mask: piece.mask };
    // Drawn as the standing selection will be, cell outline and all, so the release changes nothing on screen.
    const mode = toolSelectionMode(values);
    found = { patches: [], selection, selectionMode: mode, tone: "region", highlight: voxelRegionHighlight(scene, selection).highlight,
      caption: `${selectionModeCaption(mode)}WAND · ${piece.cellCount.toLocaleString("en-US")} voxels`
        + (piece.clipped ? ` · stopped ${DEFAULT_REGION_REACH} voxels from the click; the piece goes on` : "") };
    return found;
  } };
}

export const wandTool: VoxelToolPlugin = {
  id: "wand", version: 1,
  ui: { label: "Wand", hint: "Click a voxel to select the whole piece it belongs to.", group: "Select", order: 11,
    icon: "M4 20 14 10M13 4v4M11 6h4M19 9v4M17 11h4M18.5 3v2M17.5 4h2",
    controls: [selectionModeControl, byMaterialControl, shellControl] },
  unavailable: () => undefined,
  begin: beginWand,
};
