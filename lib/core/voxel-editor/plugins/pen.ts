import type { EditorRay } from "../../editor-entity";
import { scenerySurfaceAt } from "../../editor-scenery";
import type { Vec3 } from "../../model";
import { sceneCellSizes_m } from "../../scene-lattice";
import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../../solid-world";
import { subcellEllipsoidPatch } from "../../solid-patch/ellipsoid-bounds";
import { SVO_SCULPT_CUT_MAXIMUM_RECORDS } from "../../../svo/features/scene-publication/svo-sculpt-cuts";
import { materialControl, operationTone, shellControl, toolMaterialId } from "../controls";
import { MAX_STROKE_PATCHES, MAX_STROKE_VOXELS, shapePatches, strokeHighlight, toolPassesThrough,
  voxelToolUnavailable } from "../geometry";
import { toolReader, type ToolChoiceControl, type ToolContext, type ToolGesture, type ToolNumberControl,
  type ToolUpdate, type VoxelToolPlugin } from "../plugin";

export const penModeControl: ToolChoiceControl = {
  id: "mode", kind: "choice", label: "Pen", initial: 0,
  options: [
    { value: 0, label: "Draw", hint: "Add a rounded stroke along the surface as you drag.", shortcut: "t", tone: "body" },
    { value: 1, label: "Carve", hint: "Cut a rounded stroke into solids, terrain or scenery as you drag.", shortcut: "r", tone: "inflow" },
  ],
};
export const penDiameterControl: ToolNumberControl = {
  id: "size", label: "Diameter · voxels", short: "Ø", adjust: "primary", min: 1, max: 64, step: 1, initial: 5,
};
interface PenSurface { readonly point: Vec3; readonly normal: Vec3 }

/** A real, nearest surface. The pen never invents a ground plane after a miss. */
function surfaceAt(context: ToolContext, ray: EditorRay, press: boolean): PenSurface | undefined {
  const voxel = toolReader(context).pick(ray, { rejectBudgetExhaustion: press,
    skip: coordinate => toolPassesThrough(context, coordinate) });
  const scenery = scenerySurfaceAt(context.scene, ray, voxel?.distance_m);
  if (scenery) return { point: scenery.position_m, normal: scenery.normal };
  if (!voxel) return undefined;
  const normal = { x: 0, y: 0, z: 0 };
  normal[(["x", "y", "z"] as const)[voxel.faceAxis]] = voxel.faceSign;
  return { point: voxel.point_m, normal };
}

function beginPen(context: ToolContext): ToolGesture | undefined {
  const carve = (context.values[penModeControl.id] === 1) !== (context.invert === true);
  const operation = carve ? "clear" : "fill";
  const first = surfaceAt(context, context.ray, true);
  if (!first) return undefined;
  const diameter = context.values[penDiameterControl.id] ?? penDiameterControl.initial;
  const cell = sceneCellSizes_m(context.scene);
  const origin = [-context.scene.container.width_m / 2, 0, -context.scene.container.depth_m / 2];
  const axes = ["x", "y", "z"] as const;
  const stamp = (surface: PenSurface): Patch => {
    if (diameter > 1) {
      // Eighth-cell centres remove whole-voxel stair steps without recording
      // every tiny pointer fluctuation as a new CSG operand.
      const centre = axes.map((axis, a) => Math.round((surface.point[axis] - origin[a]!) / cell[a]! * 8) / 8);
      return { ...subcellEllipsoidPatch(centre.map(v => v - diameter / 2) as unknown as Cell,
        centre.map(v => v + diameter / 2) as unknown as Cell, operation),
        ...(carve ? { affectsScenery: true } : { materialId: toolMaterialId(context.values) }) };
    }
    // Centred on the hit, a rounded stamp overlaps the surface instead of
    // balancing an outward sphere on a single point. The tiny normal offset
    // resolves half-cell ties outward for Draw and inward for Carve.
    const minimum = axes.map((axis, a) => {
      const position = (surface.point[axis] - origin[a]!) / cell[a]!;
      const bias = surface.normal[axis] * (carve ? -1 : 1) * 1e-5;
      return Math.round(position - diameter / 2 + bias);
    }) as unknown as Cell;
    const patch = shapePatches(minimum, minimum.map(v => v + diameter) as unknown as Cell, operation,
      diameter === 1 ? "box" : "sphere", 1, { materialId: toolMaterialId(context.values) })[0]!;
    return carve ? { ...patch, affectsScenery: true } : patch;
  };
  const existingCuts = context.scene.solidVoxels.filter(p => p.operation === "clear" && p.affectsScenery === true).length;
  const patches: Patch[] = [], keys = new Set<string>();
  let voxels = 0;
  let last: { readonly ray: EditorRay; readonly surface: PenSurface } | undefined;
  let published: readonly Patch[] = patches;
  let shown: { key: string; update: ToolUpdate } | undefined;
  const caption = `${carve ? "CARVE" : "DRAW"} · ${diameter} voxel diameter`;
  return { surface: true, update(ray): ToolUpdate | undefined {
    const surface = surfaceAt(context, ray, false);
    if (!surface) {
      // A later re-entry begins a fresh segment, even when two endpoints would
      // otherwise span an editable surface behind the off-surface excursion.
      last = undefined;
      shown = undefined;
      return patches.length ? { patches: published, highlight: { kind: "paths", paths: [] },
        tone: operationTone(operation), caption } : undefined;
    }
    const landed: PenSurface[] = [];
    if (last) {
      const delta = axes.map((axis, a) => (surface.point[axis] - last!.surface.point[axis]) / cell[a]!);
      const distance = diameter > 1 ? Math.hypot(...delta) : Math.max(...delta.map(Math.abs));
      if (distance > 1024) throw new Error("Stroke span exceeds 1,024 voxels; use shorter strokes.");
      const span = Math.ceil(distance / (diameter > 1 ? .5 : 1));
      const mix = (a: Vec3, b: Vec3, t: number): Vec3 => ({
        x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t,
      });
      for (let i = 1; i < span; i++) {
        const t = i / span;
        const crossed = surfaceAt(context, { origin: mix(last.ray.origin, ray.origin, t),
          direction: mix(last.ray.direction, ray.direction, t) }, false);
        if (crossed) landed.push(crossed);
      }
    }
    landed.push(surface);
    const nib = stamp(surface), nibKey = `${nib.minimum.join(",")}:${nib.payload ?? ""}`;
    const additions: Patch[] = [], addedKeys = new Set<string>();
    let additionalVoxels = 0;
    for (const hit of landed) {
      const patch = hit === surface ? nib : stamp(hit), key = `${patch.minimum.join(",")}:${patch.payload ?? ""}`;
      if (keys.has(key) || addedKeys.has(key)) continue;
      additions.push(patch); addedKeys.add(key);
      additionalVoxels += patch.minimum.reduce((volume, v, a) => volume * (patch.maximumExclusive[a]! - v), 1);
    }
    if (carve && existingCuts + patches.length + additions.length > SVO_SCULPT_CUT_MAXIMUM_RECORDS) {
      throw new Error(`Scenery sculpt capacity reached (${SVO_SCULPT_CUT_MAXIMUM_RECORDS} cuts); undo cuts before adding more.`);
    }
    if (patches.length + additions.length > MAX_STROKE_PATCHES || voxels + additionalVoxels > MAX_STROKE_VOXELS) {
      throw new Error("Stroke is full; release and start another stroke.");
    }
    patches.push(...additions); for (const key of addedKeys) keys.add(key); voxels += additionalVoxels;
    if (additions.length) published = patches.slice();
    last = { ray, surface };
    if (!additions.length && shown?.key === nibKey) return shown.update;
    // Only the nib is outlined. Accepted geometry is already live, and no
    // accumulated region or standing selection is part of this interaction.
    const update: ToolUpdate = { patches: published, highlight: strokeHighlight(context.scene, [nib]),
      tone: operationTone(operation), caption };
    shown = { key: nibKey, update }; return update;
  } };
}

export const penTool: VoxelToolPlugin = {
  id: "pen", version: 1, execution: "live",
  ui: { label: "Pen", hint: "Draw or carve directly on a surface while dragging. Brackets adjust the diameter.",
    opposite: "draw ↔ carve",
    group: "Brush", order: -1, icon: "m4 20 4-1 12-12-3-3L5 16Zm10-13 3 3M4 20l1-4 3 3",
    controls: [penModeControl, penDiameterControl, materialControl, shellControl] },
  unavailable: voxelToolUnavailable,
  begin: beginPen,
};
