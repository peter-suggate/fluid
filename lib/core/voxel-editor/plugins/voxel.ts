import type { EditorRay } from "../../editor-entity";
import type { SolidWorldVoxelPatch as Patch } from "../../solid-world";
import { OPERATION_OPPOSITE, depthControl, materialControl, mirrorControls, operationCaption, operationControl,
  operationTone, shapeControl, shellControl, sizeControl, toolMaterialId, toolPatchOperation } from "../controls";
import { MAX_STROKE_PATCHES, MAX_STROKE_VOXELS, faceBox, mirroredPatches, patchesSize, shapePatches,
  strokeHighlight, toolSurface, voxelToolUnavailable, type ToolSurface } from "../geometry";
import type { ToolContext, ToolGesture, ToolNumberControl, ToolUpdate, VoxelToolPlugin } from "../plugin";

const shape = shapeControl(["sphere", "cube"]);
/** How far a stamp reaches from the face. Nothing means as far as it is wide: a ball, a cube. */
const reachControl: ToolNumberControl = { ...depthControl, label: "Depth · voxels · 0 follows the width", min: 0, initial: 0 };
/** Cells a single sample may jump before the stroke is refused as one stroke. */
const MAX_SAMPLE_SPAN = 1024;

/**
 * The freehand brush: a stamp at every cell the pointer crosses.
 *
 * Each sample is picked afresh against the scene the stroke began on, so the
 * stroke follows whatever stands under the pointer — over a step, down a slope,
 * across terrain — where a plane locked at the press would cut through it or
 * hang above it. Reading the base scene, never the proposal, is what keeps the
 * stroke from climbing its own stamps.
 *
 * A stamp is the brush's width across the picked face and its depth along the
 * normal: out from the face when attaching, into it from the picked cell when
 * erasing or repainting.
 */
function beginVoxelStroke(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const first = toolSurface(context, ray);
  if (!first) return undefined;
  const operation = toolPatchOperation(context);
  const materialId = toolMaterialId(values);
  const size = values[sizeControl.id]!, depth = values[reachControl.id] || size;
  // One cell wide is that cell whichever shape is chosen: a ball inscribed in a single voxel is a fraction of one.
  const stampShape = size === 1 || shape.of(values) === "cube" ? "box" : "sphere";
  const stamp = (surface: ToolSurface): Patch[] => {
    const lo = surface.coordinate.map((v) => v - Math.floor((size - 1) / 2)), hi = lo.map((v) => v + size - 1);
    const box = faceBox(surface, lo, hi, depth, operation === "fill");
    return mirroredPatches(context, shapePatches(box.minimum, box.maximumExclusive, operation, stampShape,
      surface.faceAxis, { materialId }));
  };
  /**
   * The stroke as it grows: one entry per distinct stamp, keyed by its box.
   * Mirror images are entries like any other, so a stroke that crosses its own
   * mirror plane does not write a stamp twice.
   */
  const stroke = { keys: new Set<string>(), patches: [] as Patch[], voxels: 0 };
  let last: { readonly ray: EditorRay; readonly surface: ToolSurface } | undefined;
  /** The surfaces between the last sample and this one, found by sweeping the ray across the gap. */
  const crossed = (input: EditorRay, surface: ToolSurface): ToolSurface[] => {
    if (!last) return [surface];
    const span = Math.max(...surface.coordinate.map((v, a) => Math.abs(v - last!.surface.coordinate[a]!)));
    if (span > MAX_SAMPLE_SPAN) throw new Error("Stroke span exceeds 1,024 voxels; use shorter strokes.");
    const from = last.ray, landed: ToolSurface[] = [];
    const mix = (a: number, b: number, t: number) => a + (b - a) * t;
    for (let step = 1; step < span; step += 1) {
      const t = step / span;
      const between = toolSurface(context, {
        origin: { x: mix(from.origin.x, input.origin.x, t), y: mix(from.origin.y, input.origin.y, t), z: mix(from.origin.z, input.origin.z, t) },
        direction: { x: mix(from.direction.x, input.direction.x, t), y: mix(from.direction.y, input.direction.y, t), z: mix(from.direction.z, input.direction.z, t) },
      }, false);
      if (between) landed.push(between);
    }
    return [...landed, surface];
  };
  return { surface: first.picked, update(input): ToolUpdate | undefined {
    const surface = toolSurface(context, input, false);
    // Over nothing at all there is nowhere to stamp; the stroke keeps what it has.
    if (surface) {
      const accepted = { patches: stroke.patches.length, voxels: stroke.voxels }, added: string[] = [];
      for (const landed of crossed(input, surface)) for (const patch of stamp(landed)) {
        const key = `${patch.minimum.join(",")}|${patch.maximumExclusive.join(",")}`;
        if (stroke.keys.has(key)) continue;
        stroke.keys.add(key); added.push(key);
        stroke.patches.push(patch);
        stroke.voxels += patch.minimum.reduce((volume, v, a) => volume * (patch.maximumExclusive[a]! - v), 1);
      }
      if (stroke.patches.length > MAX_STROKE_PATCHES || stroke.voxels > MAX_STROKE_VOXELS) {
        // A rejected sample is taken back out, so the stroke keeps exactly the geometry it had already been shown.
        stroke.patches.length = accepted.patches; stroke.voxels = accepted.voxels;
        for (const key of added) stroke.keys.delete(key);
        throw new Error("Stroke is full; release and start another stroke.");
      }
      last = { ray: input, surface };
    }
    if (stroke.patches.length === 0) return undefined;
    const patches = stroke.patches.slice();
    return { patches, tone: operationTone(operation), highlight: strokeHighlight(scene, patches),
      caption: `${operationCaption(operation)} · ${patchesSize(patches)}` };
  } };
}

export const voxelBrushTool: VoxelToolPlugin = {
  id: "voxel", version: 1,
  ui: { label: "Voxel", hint: "Drag across solids to add to, cut into or repaint the surface under the pointer.", group: "Brush", order: 0,
    opposite: OPERATION_OPPOSITE,
    icon: "M12 3 20 7.5v9L12 21l-8-4.5v-9ZM4 7.5 12 12l8-4.5M12 12v9",
    controls: [operationControl, sizeControl, reachControl, shape.control, materialControl, ...mirrorControls, shellControl] },
  unavailable: voxelToolUnavailable,
  begin: beginVoxelStroke,
};
