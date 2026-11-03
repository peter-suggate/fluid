import { containerShellContains } from "../../editor-solid-voxel";
import { voxelRegionHighlight } from "../../editor-voxel-region";
import { sceneLatticeDimensions } from "../../scene-lattice";
import type { SolidWorldCoordinate as Cell } from "../../solid-world";
import { byMaterialControl, selectionModeCaption, selectionModeControl, shellControl, toolSelectionMode } from "../controls";
import { boxHighlight, faceBox, faceCell, toolSurface } from "../geometry";
import { toolReader, type ToolContext, type ToolGesture, type ToolNumberControl, type ToolUpdate,
  type VoxelToolPlugin } from "../plugin";
import type { VoxelReader } from "../voxel-access";
import { DEFAULT_REGION_REACH, connectedRegion } from "../voxel-grid";

/** How deep a dragged box reaches behind the face it is drawn on. */
export const selectDepthControl: ToolNumberControl = {
  id: "depth", label: "Box depth · voxels", short: "Depth", adjust: "primary", min: 1, max: 128, step: 1, initial: 1,
};

/**
 * Select solid voxels, in one gesture: click a voxel for the whole piece it
 * belongs to, or drag across a face for a box.
 *
 * A press that is released in the cell it started on is a click, and selects
 * every solid cell joined face to face to that one — optionally only those of
 * its material. A drag that leaves the cell draws the box's footprint on the
 * face; the box reaches `depth` layers into the solid behind it, so it is
 * placed with one drag and no second click. On the tank's own shell, with the
 * shell switch off, the box stands in front of the face instead: the room, not
 * the vessel. Add and subtract are the shared selection mode.
 *
 * It proposes a selection and no patches, so it authors nothing, records no
 * history and runs under any solver.
 */
function beginSelect(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const surface = toolSurface(context, ray);
  if (!surface?.picked) return undefined;
  const seed = surface.coordinate;
  const start = faceCell(scene, surface, ray) ?? seed;
  const mode = toolSelectionMode(values);
  // Everything standing in the tank touches its floor, so a search that walked
  // into the shell would answer "the floor, and everything on it" for every
  // click — and the next Delete would take the floor. With the shell switch
  // off the shell is not part of any piece, and is not a piece to click. The
  // same switch keeps the search inside the lattice: one stray cell past its
  // edge would make the whole piece immovable for the verbs.
  const walls = values[shellControl.id] === 1;
  const onShell = !walls && containerShellContains(scene, seed);
  const dimensions = sceneLatticeDimensions(scene);
  const inPiece = (coordinate: Cell) => !containerShellContains(scene, coordinate)
    && coordinate.every((value, axis) => value >= 0 && value < dimensions[axis]!);
  const cell = { minimum: seed, maximumExclusive: seed.map((v) => v + 1) as unknown as Cell };
  if (context.hover) {
    return { surface: true, update: () => ({ patches: [], tone: "region", highlight: boxHighlight(scene, cell),
      caption: `${selectionModeCaption(mode)}SELECT · click a piece, drag a box` }) };
  }
  let piece: ToolUpdate | undefined;
  /** The click's answer: searched once, the first time the pointer is let go — or rests — in the pressed cell. */
  const click = (): ToolUpdate => {
    if (piece) return piece;
    if (onShell || (!walls && !inPiece(seed))) {
      return piece = { patches: [], tone: "region", highlight: boxHighlight(scene, cell),
        caption: "SELECT · the tank shell is not a piece: drag a box in front of it, or turn on the shell" };
    }
    const all = toolReader(context);
    const reader: VoxelReader = walls ? all : {
      scene: all.scene,
      fraction: (coordinate) => all.fraction(coordinate),
      solid: (coordinate) => inPiece(coordinate) && all.solid(coordinate),
      material: (coordinate) => all.material(coordinate),
      pick: (input, options) => all.pick(input, options),
      read: (minimum, maximumExclusive) => all.read(minimum, maximumExclusive),
    };
    const found = connectedRegion(reader, seed, { byMaterial: values[byMaterialControl.id] === 1 });
    const selection = { minimum: found.minimum, maximumExclusive: found.maximumExclusive, mask: found.mask };
    // Drawn as the standing selection will be, so the release changes nothing on screen.
    return piece = { patches: [], selection, selectionMode: mode, tone: "region",
      highlight: voxelRegionHighlight(scene, selection).highlight,
      caption: `${selectionModeCaption(mode)}SELECT · ${found.cellCount.toLocaleString("en-US")} voxels in this piece`
        + (found.clipped ? ` · stopped ${DEFAULT_REGION_REACH} voxels from the click; the piece goes on` : "") };
  };
  let shown: { key: string; update: ToolUpdate } | undefined;
  return { surface: true, update(input): ToolUpdate | undefined {
    const end = faceCell(scene, surface, input);
    if (!end || end.every((v, a) => v === start[a])) return click();
    const lo = start.map((v, a) => Math.min(v, end[a]!)), hi = start.map((v, a) => Math.max(v, end[a]!));
    const key = [...lo, ...hi].join(",");
    if (shown?.key === key) return shown.update;
    const depth = values[selectDepthControl.id] ?? selectDepthControl.initial;
    const box = faceBox(surface, lo, hi, depth, onShell);
    const size = box.minimum.map((v, a) => box.maximumExclusive[a]! - v);
    const update: ToolUpdate = { patches: [], selection: box, selectionMode: mode, tone: "region",
      highlight: boxHighlight(scene, box), caption: `${selectionModeCaption(mode)}SELECT · ${size.join(" × ")}` };
    shown = { key, update };
    return update;
  } };
}

export const selectTool: VoxelToolPlugin = {
  id: "select-voxels", version: 1,
  ui: { label: "Select voxels", hint: "Click a voxel to select its whole piece, or drag across a face to select a box. Brackets set the box depth.",
    group: "Select", order: 10, key: "w",
    icon: "M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M11 4h2M11 20h2M4 11v2M20 11v2",
    hints: {
      armed: { keys: ["adjust"], text: "click a piece · drag a box" },
      active: { keys: ["cancel"], text: "release to select" },
      after: { text: "the verbs act on it · Add or Subtract to refine it" },
    },
    controls: [selectionModeControl, selectDepthControl, byMaterialControl, shellControl] },
  unavailable: () => undefined,
  begin: beginSelect,
};
