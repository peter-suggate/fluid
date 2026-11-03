import type { EditorRay } from "../../editor-entity";
import { visibleToolSurface } from "../surface";
import { focusOrPick } from "../../editor-focus/focus";
import type { Vec3 } from "../../model";
import { sceneCellSizes_m } from "../../scene-lattice";
import { sceneWithSolidStroke, type SolidWorldCoordinate as Cell, type SolidWorldVoxelPatch as Patch } from "../../solid-world";
import { subcellEllipsoidPatch } from "../../solid-patch/ellipsoid-bounds";
import { OPERATION_OPPOSITE, PAINT_KEEPS_SCENERY, materialControl, mirrorControls, operationCaption, operationControl, operationTone,
  toolMaterialId, toolOperation, toolPatchOperation } from "../controls";
import { MAX_STROKE_PATCHES, MAX_STROKE_VOXELS, mirroredPatches, shapePatches, strokeHighlight,
  voxelToolUnavailable } from "../geometry";
import { type ToolContext, type ToolGesture, type ToolNumberControl, type ToolUpdate,
  type VoxelToolPlugin } from "../plugin";

export const penDiameterControl: ToolNumberControl = {
  id: "size", label: "Diameter · voxels", short: "Ø", adjust: "primary", min: 1, max: 64, step: 1, initial: 5,
};
interface PenSurface { readonly point: Vec3; readonly normal: Vec3; readonly scenery?: string }

/**
 * A real, nearest surface. The pen never invents a ground plane after a miss.
 * With a focus lock up the stamp stands on the lock instead (scene-focus.md):
 * in open air, Draw builds there and Erase cuts only where it meets solid.
 */
function surfaceAt(context: ToolContext, ray: EditorRay, press: boolean): PenSurface | undefined {
  return focusOrPick(context.focus, ray, (r: EditorRay) => visibleToolSurface(context.scene, r, press));
}

/**
 * The pen: a rounded nib dragged over whatever surface is under the pointer,
 * and the edit itself is what you see (`execution: "live"`).
 *
 * It does what the rail's operation says. Attach builds, erase carves —
 * through voxels, terrain and scenery alike — and paint recolours the solids
 * the nib covers and adds none. Alt swaps attach and erase for one stroke.
 *
 * Passes accumulate: a stamp stays out of picking while the nib is over it, so
 * one sweep stays level, and once the nib has left it the next pass lands on
 * the surface it made — higher for attach, deeper for erase.
 */
function beginPen(context: ToolContext): ToolGesture | undefined {
  const operation = toolPatchOperation(context);
  const carve = operation === "clear", paint = operation === "paint";
  const materialId = toolMaterialId(context.values);
  const first = surfaceAt(context, context.ray, true);
  if (!first) return undefined;
  // A repaint writes voxel materials; scenery keeps its own. Said at the press rather than painted into nothing.
  if (paint && first.scenery !== undefined) throw new Error(PAINT_KEEPS_SCENERY);
  const diameter = context.values[penDiameterControl.id] ?? penDiameterControl.initial;
  const cell = sceneCellSizes_m(context.scene);
  const origin = [-context.scene.container.width_m / 2, 0, -context.scene.container.depth_m / 2];
  const axes = ["x", "y", "z"] as const;
  /** The nib at a surface: the stamp itself first, then its images under the mirror planes that are on. */
  const stamp = (surface: PenSurface): Patch[] => {
    if (diameter > 1) {
      // Eighth-cell centres remove whole-voxel stair steps without recording
      // every tiny pointer fluctuation as a new CSG operand.
      const centre = axes.map((axis, a) => Math.round((surface.point[axis] - origin[a]!) / cell[a]! * 8) / 8);
      return mirroredPatches(context, [{ ...subcellEllipsoidPatch(centre.map(v => v - diameter / 2) as unknown as Cell,
        centre.map(v => v + diameter / 2) as unknown as Cell, operation),
        ...(carve ? { affectsScenery: true } : { materialId }) }]);
    }
    // Centred on the hit, a rounded stamp overlaps the surface instead of
    // balancing an outward sphere on a single point. The tiny normal offset
    // resolves half-cell ties outward for attach and inward for erase and paint,
    // which act on what is already there.
    const minimum = axes.map((axis, a) => {
      const position = (surface.point[axis] - origin[a]!) / cell[a]!;
      const bias = surface.normal[axis] * (operation === "fill" ? 1 : -1) * 1e-5;
      return Math.round(position - diameter / 2 + bias);
    }) as unknown as Cell;
    return mirroredPatches(context, shapePatches(minimum, minimum.map(v => v + diameter) as unknown as Cell, operation,
      "box", 1, { materialId }));
  };
  const keyOf = (nib: readonly Patch[]) => `${nib[0]!.minimum.join(",")}:${nib[0]!.payload ?? ""}`;
  const patches: Patch[] = [], keys = new Set<string>();
  // Stamps under the brush stay out of picking until the brush leaves them.
  // This keeps a sweep level, but lets a later pass work on the edited surface.
  let pickingContext = context;
  let active: { readonly patches: readonly Patch[]; readonly point: Vec3 }[] = [];
  let voxels = 0;
  let last: { readonly ray: EditorRay; readonly surface: PenSurface } | undefined;
  let published: readonly Patch[] = patches;
  let shown: { key: string; update: ToolUpdate } | undefined;
  const caption = `${operationCaption(operation)} · ${diameter} voxel diameter`;
  return { surface: true, update(ray): ToolUpdate | undefined {
    // Stage picking changes as well as geometry: a rejected update must leave
    // the previous accepted stroke usable, including its brush footprint.
    let nextContext = pickingContext, nextActive = active.slice();
    const additions: Patch[] = [], addedKeys = new Set<string>();
    let additionalVoxels = 0;
    // Paint moves no surface, so a departed stamp has nothing to change about picking.
    const retire = (departed: typeof active): void => {
      if (departed.length && !paint) nextContext = { ...nextContext,
        scene: sceneWithSolidStroke(nextContext.scene, departed.flatMap(entry => entry.patches)) };
    };
    const sample = (sampleRay: EditorRay): PenSurface | undefined => {
      // Measure across the ray in voxel units, so a changing depth alone never
      // counts as leaving. One diameter separates overlapping brush footprints.
      const direction = axes.map((axis, a) => sampleRay.direction[axis] / cell[a]!);
      const lengthSquared = direction.reduce((sum, v) => sum + v * v, 0);
      const departed: typeof active = [];
      nextActive = nextActive.filter(entry => {
        const offset = axes.map((axis, a) => (entry.point[axis] - sampleRay.origin[axis]) / cell[a]!);
        const t = lengthSquared > 0
          ? Math.max(0, offset.reduce((sum, v, a) => sum + v * direction[a]!, 0) / lengthSquared) : 0;
        const distance = Math.hypot(...offset.map((v, a) => v - t * direction[a]!));
        if (distance > diameter) { departed.push(entry); return false; }
        return true;
      });
      retire(departed);
      const hit = surfaceAt(nextContext, sampleRay, false);
      if (!hit) { retire(nextActive); nextActive = []; return undefined; }
      if (paint && hit.scenery !== undefined) return hit;
      const nib = stamp(hit), key = keyOf(nib);
      if (!keys.has(key) && !addedKeys.has(key)) {
        additions.push(...nib); addedKeys.add(key);
        nextActive.push({ patches: nib, point: hit.point });
        for (const patch of nib) {
          additionalVoxels += patch.minimum.reduce((volume, v, a) => volume * (patch.maximumExclusive[a]! - v), 1);
        }
        if (patches.length + additions.length > MAX_STROKE_PATCHES || voxels + additionalVoxels > MAX_STROKE_VOXELS) {
          throw new Error("Stroke is full; release and start another stroke.");
        }
      }
      return hit;
    };
    const target = surfaceAt(pickingContext, ray, false);
    if (last && target) {
      const delta = axes.map((axis, a) => (target.point[axis] - last!.surface.point[axis]) / cell[a]!);
      const distance = diameter > 1 ? Math.hypot(...delta) : Math.max(...delta.map(Math.abs));
      if (distance > 1024) throw new Error("Stroke span exceeds 1,024 voxels; use shorter strokes.");
      const span = Math.ceil(distance / (diameter > 1 ? .5 : 1));
      const mix = (a: Vec3, b: Vec3, t: number): Vec3 => ({
        x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, z: a.z + (b.z - a.z) * t,
      });
      for (let i = 1; i < span; i++) {
        const t = i / span;
        sample({ origin: mix(last.ray.origin, ray.origin, t),
          direction: mix(last.ray.direction, ray.direction, t) });
      }
    }
    const surface = sample(ray);
    patches.push(...additions); for (const key of addedKeys) keys.add(key); voxels += additionalVoxels;
    if (additions.length) published = patches.slice();
    pickingContext = nextContext; active = nextActive;
    if (!surface) {
      // A miss breaks interpolation and arms all accepted stamps for re-entry.
      last = undefined; shown = undefined;
      return patches.length ? { patches: published, highlight: { kind: "paths", paths: [] },
        tone: operationTone(operation), caption } : undefined;
    }
    const nib = stamp(surface), unpainted = paint && surface.scenery !== undefined;
    const nibKey = `${keyOf(nib)}${unpainted ? ":scenery" : ""}`;
    last = { ray, surface };
    if (!additions.length && shown?.key === nibKey) return shown.update;
    // Only the nib is outlined. Accepted geometry is already live, and no
    // accumulated region or standing selection is part of this interaction.
    const update: ToolUpdate = { patches: published, highlight: strokeHighlight(context.scene, nib),
      tone: operationTone(operation),
      caption: unpainted ? `${caption} · scenery keeps its own colours` : caption };
    shown = { key: nibKey, update }; return update;
  } };
}

/** What the pen's next pass does, in the words a hint uses. */
const passAgain = (values: Parameters<typeof toolOperation>[0]): string => {
  const operation = toolOperation(values);
  return operation === "erase" ? "pass again to carve deeper"
    : operation === "paint" ? "drag over solids to recolour them" : "pass again to build higher";
};

export const penTool: VoxelToolPlugin = {
  id: "pen", version: 1, execution: "live",
  ui: { label: "Pen", hint: "Attach, erase or paint directly on any surface while dragging. Brackets set the diameter.",
    opposite: OPERATION_OPPOSITE,
    group: "Brush", order: -1, key: "p", icon: "m4 20 4-1 12-12-3-3L5 16Zm10-13 3 3M4 20l1-4 3 3",
    hints: {
      armed: { keys: ["operation", "adjust"], text: "drag across any surface" },
      active: (values) => ({ keys: ["lock", "cancel"], text: toolOperation(values) === "paint"
        ? "recolours what the nib covers" : "leave a spot and come back to go further" }),
      after: (values) => ({ keys: ["adjust", "opposite"], text: passAgain(values) }),
    },
    controls: [operationControl, penDiameterControl, materialControl, ...mirrorControls] },
  unavailable: voxelToolUnavailable,
  begin: beginPen,
};
