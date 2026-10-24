import type { SolidWorldCoordinate as Cell } from "../../solid-world";
import { subcellEllipsoidPatch } from "../../solid-patch/ellipsoid-bounds";
import { OPERATION_OPPOSITE, depthControl, materialControl, mirrorControls, operationCaption, operationControl,
  operationTone, shapeControl, shellControl, toolMaterialId, toolPatchOperation } from "../controls";
import { assertStrokeCaps, faceBox, faceCell, facePoint, mirroredPatches, patchesSize, shapePatches, strokeHighlight,
  toolSurface, voxelToolUnavailable } from "../geometry";
import type { ToolContext, ToolGesture, ToolNumberControl, ToolUpdate, VoxelToolPlugin } from "../plugin";

const shape = shapeControl(["sphere", "cylinder", "box"]);
const lengthControl: ToolNumberControl = { ...depthControl, label: "Cylinder depth · voxels" };

/**
 * A shape grown from where the press lands: the press is its centre, and the
 * drag across the face is its radius.
 *
 * A sphere is centred on the pressed face point, with subcell radius steps;
 * a box uses the adjacent cell when attaching and the picked cell otherwise.
 * Attaching raises a dome and erasing scoops a bowl. A cylinder stands along the face normal with
 * the radius as its section and its own depth as its length: a post out of the
 * face, or a drilled hole into it.
 */
function beginCenter(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const surface = toolSurface(context, ray);
  if (!surface) return undefined;
  const operation = toolPatchOperation(context);
  const materialId = toolMaterialId(values);
  const which = shape.of(values), axis = surface.faceAxis;
  const centre = surface.coordinate.map((v, a) => a === axis && operation === "fill" ? v + surface.faceSign : v);
  const pressed = facePoint(scene, surface, ray);
  const roundCentre = pressed?.map(v => Math.round(v * 8) / 8);
  let shown: { radius: number; update: ToolUpdate } | undefined;
  return { surface: surface.picked, update(input): ToolUpdate | undefined {
    const at = faceCell(scene, surface, input);
    if (!at) return undefined;
    const reach = at.map((v, a) => Math.abs(v - surface.coordinate[a]!));
    // Round shapes take the straight distance; a box takes the farther of its two sides.
    const point = which === "sphere" ? facePoint(scene, surface, input) : undefined;
    const radius = point && roundCentre ? Math.max(.5, Math.round(Math.hypot(...point.map((v, a) => v - roundCentre[a]!)) * 8) / 8)
      : which === "box" ? Math.max(...reach) : Math.round(Math.hypot(...reach));
    if (shown?.radius === radius) return shown.update;
    const boxRadius = which === "sphere" && radius === .5 ? 0 : radius;
    const box = which === "cylinder"
      ? faceBox(surface, surface.coordinate.map((v) => v - radius), surface.coordinate.map((v) => v + radius),
        values[lengthControl.id]!, operation === "fill")
      : { minimum: centre.map((v) => v - boxRadius) as unknown as Cell, maximumExclusive: centre.map((v) => v + boxRadius + 1) as unknown as Cell };
    // A radius of nothing is the one cell, whole, whatever the shape.
    const patches = mirroredPatches(context, which === "sphere" && roundCentre && radius > .5
      ? [{ ...subcellEllipsoidPatch(roundCentre.map(v => v - radius) as unknown as Cell,
        roundCentre.map(v => v + radius) as unknown as Cell, operation), ...(operation === "clear" ? {} : { materialId }) }]
      : shapePatches(box.minimum, box.maximumExclusive, operation, radius <= .5 ? "box" : which, axis, { materialId }));
    assertStrokeCaps(patches, "Use a smaller radius, or fewer mirror planes.");
    const update: ToolUpdate = { patches, tone: operationTone(operation), highlight: strokeHighlight(scene, patches),
      caption: `${operationCaption(operation)} ${which.toUpperCase()} · radius ${radius} · ${patchesSize(patches)}` };
    shown = { radius, update }; return update;
  } };
}

export const centerTool: VoxelToolPlugin = {
  id: "center", version: 1,
  ui: { label: "Center", hint: "Press where the centre goes and drag out to the radius: a sphere, a box, or a cylinder along the face normal.", group: "Brush", order: 4,
    opposite: OPERATION_OPPOSITE,
    icon: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M12 11a1 1 0 1 0 0 2a1 1 0 1 0 0-2M12 12h9",
    controls: [operationControl, lengthControl, shape.control, materialControl, ...mirrorControls, shellControl] },
  unavailable: voxelToolUnavailable,
  begin: beginCenter,
};
