import { containerShellContains } from "../../editor-solid-voxel";
import { selectionModeCaption, selectionModeControl, shellControl, toolSelectionMode } from "../controls";
import { beginFootprintGesture, boxHighlight, faceBox } from "../geometry";
import type { VoxelToolPlugin } from "../plugin";

/**
 * Select a box of cells: drag its footprint on a face, let go, then push into
 * the solid or pull out of it for depth, and click.
 *
 * It proposes a selection and no patches, so it authors nothing, records no
 * history and runs under any solver. Pushed in, the box holds the solid behind
 * the face; pulled out, the empty cells in front of it — which is what a fill
 * wants. A click that did not move selects the face's own layer.
 */
export const marqueeTool: VoxelToolPlugin = {
  id: "marquee", version: 1,
  ui: { label: "Marquee", hint: "Drag a footprint on a face and let go. Then push in or pull out for depth, and click to select the box.", group: "Select", order: 10,
    icon: "M4 8V4h4M16 4h4v4M20 16v4h-4M8 20H4v-4M11 4h2M11 20h2M4 11v2M20 11v2",
    controls: [selectionModeControl, shellControl] },
  unavailable: () => undefined,
  begin: (context) => beginFootprintGesture(context, ({ surface, lo, hi, depth: travel, extruding }) => {
    // Behind the tank's own floor there is nothing to select but the tank, and a
    // verb on that would cut the vessel open. With the walls switch off, a
    // footprint drawn on the shell selects the room in front of it instead.
    const onShell = context.values[shellControl.id] !== 1 && containerShellContains(context.scene, surface.coordinate);
    const depth = onShell ? Math.max(1, travel) : travel;
    const box = faceBox(surface, lo, hi, Math.max(1, Math.abs(depth)), depth > 0);
    const size = box.minimum.map((v, a) => box.maximumExclusive[a]! - v);
    const mode = toolSelectionMode(context.values);
    return { patches: [], selection: box, selectionMode: mode, tone: "region", highlight: boxHighlight(context.scene, box),
      caption: selectionModeCaption(mode) + (extruding ? `SELECT · ${size.join(" × ")} · click to select`
        : `FOOTPRINT · ${size.filter((_, a) => a !== surface.faceAxis).join(" × ")} · release, then push or pull for depth`) };
  }),
};
