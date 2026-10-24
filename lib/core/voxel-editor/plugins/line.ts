import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../../solid-world";
import { OPERATION_OPPOSITE, depthControl, materialControl, mirrorControls, operationCaption, operationControl,
  operationTone, shellControl, sizeControl, toolMaterialId, toolPatchOperation } from "../controls";
import { assertStrokeCaps, faceCell, interpolateCells, mirroredPatches, patchesSize, shapePatches, strokeHighlight,
  toolSurface, voxelToolUnavailable, type ToolSurface } from "../geometry";
import type { ToolContext, ToolGesture, ToolNumberControl, ToolUpdate, VoxelToolPlugin } from "../plugin";

const heightControl: ToolNumberControl = { ...depthControl, label: "Height · voxels", short: "H" };

/**
 * A straight run between two points of the scene: a wall when attaching, a
 * channel when erasing, a stripe when repainting.
 *
 * Both ends land on whatever is under them, read from the scene the stroke
 * began on, and the run between is a straight line through the lattice — it
 * bridges a dip rather than following it, which is the freehand brush's job.
 * Its section is the width across and the height along the normal of the face
 * the press met: up from the surface when attaching, down into it otherwise.
 * Over nothing, the far end stays on the plane of that first face.
 */
function beginLine(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const start = toolSurface(context, ray);
  if (!start) return undefined;
  const operation = toolPatchOperation(context);
  const materialId = toolMaterialId(values);
  const width = values[sizeControl.id]!, height = values[heightControl.id]!;
  const axis = start.faceAxis;
  // The way the section grows from the line: out of the solid when attaching, back into it otherwise.
  const grows = operation === "fill" ? start.faceSign : -start.faceSign;
  /** Where the line passes at a surface: the cell against the face when attaching, the picked cell otherwise. */
  const anchor = (surface: Pick<ToolSurface, "coordinate" | "faceAxis" | "faceSign">): Cell =>
    surface.coordinate.map((v, a) => a === surface.faceAxis && operation === "fill" ? v + surface.faceSign : v) as unknown as Cell;
  const from = anchor(start);
  const extent = [0, 1, 2].map((a) => a === axis ? height : width);
  let shown: { to: Cell; update: ToolUpdate } | undefined;
  return { surface: start.picked, update(input): ToolUpdate | undefined {
    const under = toolSurface(context, input, false);
    const onFace = under?.picked ? undefined : faceCell(scene, start, input);
    const to = under?.picked ? anchor(under) : onFace && anchor({ ...start, coordinate: onFace });
    if (!to) return undefined;
    if (shown && to.every((v, a) => v === shown!.to[a])) return shown.update;
    const patches = new Map<string, Patch>();
    for (const at of interpolateCells(from, to)) {
      const minimum = at.map((v, a) => a === axis ? (grows > 0 ? v : v - (height - 1)) : v - Math.floor((width - 1) / 2)) as unknown as Cell;
      const section = shapePatches(minimum, minimum.map((v, a) => v + extent[a]!) as unknown as Cell, operation, "box", axis, { materialId });
      // Keyed by corner, since every section has the same extent: a line across its own mirror plane writes each once.
      for (const patch of mirroredPatches(context, section)) patches.set(patch.minimum.join(","), patch);
    }
    const proposal = [...patches.values()];
    assertStrokeCaps(proposal);
    const update: ToolUpdate = { patches: proposal, tone: operationTone(operation), highlight: strokeHighlight(scene, proposal),
      caption: `${operationCaption(operation)} LINE · ${patchesSize(proposal)}` };
    shown = { to, update }; return update;
  } };
}

export const lineTool: VoxelToolPlugin = {
  id: "line", version: 1,
  ui: { label: "Line", hint: "Drag from one point to another: a wall when attaching, a channel when erasing. Width and height set its section.", group: "Brush", order: 3,
    opposite: OPERATION_OPPOSITE,
    icon: "M4 20 20 4M4 14 14 4",
    controls: [operationControl, sizeControl, heightControl, materialControl, ...mirrorControls, shellControl] },
  unavailable: voxelToolUnavailable,
  begin: beginLine,
};
