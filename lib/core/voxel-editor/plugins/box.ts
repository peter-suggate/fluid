import type { ToolContext, ToolGesture, VoxelToolPlugin } from "../plugin";
import { PAINT_SAMPLES, materialControl, mirrorControls, operationCaption, operationControl, operationTone, shellControl,
  toolMaterialId, toolOperation } from "../controls";
import { assertStrokeCaps, beginFootprintGesture, boxHighlight, faceBox, mirroredPatches, shapePatches,
  voxelToolUnavailable } from "../geometry";

/**
 * A box, by push and pull: drag its footprint on a face, let go, then pull it
 * out of the face or push it in, and click.
 *
 * The direction is the operation, whichever of attach and erase is lit —
 * pulled out the box is built, pushed in it is cut — because the hand has
 * already said which by where it went, and for the same reason the "do the
 * opposite" modifier has nothing to swap. Paint has no direction: the footprint
 * is repainted into the solid as deep as it was dragged either way, and one
 * layer deep on a click that did not move.
 */
function beginBox(context: ToolContext): ToolGesture | undefined {
  const { scene, values } = context;
  const paints = toolOperation(values) === "paint";
  const materialId = toolMaterialId(values);
  return beginFootprintGesture(context, ({ surface, lo, hi, depth, extruding }) => {
    const axis = surface.faceAxis;
    const operation = paints ? "paint" : depth < 0 ? "clear" : "fill";
    const layers = Math.max(1, Math.abs(depth));
    const box = faceBox(surface, lo, hi, layers, operation === "fill");
    // Back at the face there is nothing to build or cut; a repaint still has the face itself.
    const patches = !extruding || (depth === 0 && !paints) ? []
      : mirroredPatches(context, shapePatches(box.minimum, box.maximumExclusive, operation, "box", axis, { materialId }));
    assertStrokeCaps(patches, "Use a smaller box, or fewer mirror planes.");
    const size = box.minimum.map((v, a) => box.maximumExclusive[a]! - v);
    return { patches, tone: operationTone(operation), highlight: boxHighlight(scene, box),
      caption: !extruding ? `FOOTPRINT · ${size.filter((_, a) => a !== axis).join(" × ")} · release, then pull or push`
        : patches.length === 0 ? "PULL OUT OR PUSH IN"
          : `${operationCaption(operation)} · ${size.join(" × ")} · click to commit` };
  });
}

export const boxTool: VoxelToolPlugin = {
  id: "box", version: 1,
  ui: { label: "Box", hint: "Drag a footprint on a face and let go. Then pull out to build or push in to cut, and click.", group: "Brush", order: 2,
    icon: "M4 9h11v11H4ZM9 9V4h11v11h-5", opposite: PAINT_SAMPLES,
    controls: [operationControl, materialControl, ...mirrorControls, shellControl] },
  unavailable: voxelToolUnavailable,
  begin: beginBox,
};
