import type { EditorRay } from "../../editor-entity";
import { focusOrPick } from "../../editor-focus/focus";
import type { Vec3 } from "../../model";
import { sceneCellSizes_m } from "../../scene-lattice";
import { subcellPatch } from "../../solid-patch/ellipsoid-bounds";
import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../../solid-world";
import { OPERATION_OPPOSITE, PAINT_KEEPS_SCENERY, materialControl, mirrorControls, operationCaption, operationControl,
  operationTone, shapeControl, toolMaterialId, toolOperation, toolPatchOperation } from "../controls";
import { assertStrokeCaps, latticePoint, mirroredPatches, patchesSize, strokeHighlight,
  voxelToolUnavailable } from "../geometry";
import type { ToolContext, ToolGesture, ToolNumberControl, ToolUpdate, VoxelToolPlugin } from "../plugin";
import { visibleToolSurface } from "../surface";

export const shapeKindControl = shapeControl(["sphere", "box", "cylinder"]);
const CYLINDER = 2;
/** A cylinder's length along the surface normal. Nothing set, it is as long as it is wide. */
export const shapeLengthControl: ToolNumberControl = {
  id: "length", label: "Cylinder length · voxels (0 follows the diameter)", short: "Length", tier: "advanced",
  min: 0, max: 128, step: 1, initial: 0, when: (values) => values[shapeKindControl.control.id] === CYLINDER,
};
/** Centre and radius move in eighths of a cell: fine enough to follow the hand, coarse enough to settle. */
const STEPS_PER_CELL = 8;
const quantized = (value: number) => Math.round(value * STEPS_PER_CELL) / STEPS_PER_CELL;
const KINDS = { sphere: "ellipsoid", box: "cuboid", cylinder: "cylinder" } as const;

/**
 * Where the drag ray says the rim is: on the plane through the pressed point
 * across its normal, or — looking along that plane, or away from it — the
 * ray's nearest approach to the pressed point.
 */
function rimPoint(at: Vec3, normal: Vec3, ray: EditorRay): readonly number[] {
  const o = [ray.origin.x, ray.origin.y, ray.origin.z], d = [ray.direction.x, ray.direction.y, ray.direction.z];
  const p = [at.x, at.y, at.z], n = [normal.x, normal.y, normal.z];
  const dot = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  const to = p.map((v, a) => v - o[a]!), dn = dot(d, n), dd = dot(d, d);
  let t = Math.abs(dn) > 1e-6 * Math.sqrt(dd) ? dot(to, n) / dn : Number.NaN;
  if (!(t > 0)) t = Math.max(0, dot(to, d) / dd);
  return o.map((v, a) => v + t * d[a]!);
}

/**
 * A sphere, a box or a cylinder grown from where the press lands.
 *
 * The press is the centre — the exact point of whatever surface is under the
 * pointer, to an eighth of a cell, never the nearest cell — and the drag is
 * the radius. The edit is live: what stands while dragging is the edit itself,
 * resized in place as the radius changes, one undo entry on release and gone
 * again on cancel. A cylinder stands along the surface's normal axis. With a
 * focus lock up (X/Y/Z) the centre stands on the lock instead; the radius
 * still follows the hand freely.
 *
 * Attach builds the shape, erase cuts it out of voxels, terrain and scenery
 * alike, and paint recolours the solids it covers.
 */
function beginShape(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const hit: { readonly point: Vec3; readonly normal: Vec3; readonly scenery?: string } | undefined =
    focusOrPick(context.focus, ray, (input: EditorRay) => visibleToolSurface(scene, input, true));
  if (!hit) return undefined;
  const operation = toolPatchOperation(context);
  if (operation === "paint" && hit.scenery !== undefined) throw new Error(PAINT_KEEPS_SCENERY);
  const which = shapeKindControl.of(values), materialId = toolMaterialId(values);
  const cell = sceneCellSizes_m(scene);
  const centre = latticePoint(scene, hit.point).map(quantized);
  const normal = [hit.normal.x, hit.normal.y, hit.normal.z].map(Math.abs);
  const axis = normal.indexOf(Math.max(...normal)) as 0 | 1 | 2;
  const length = values[shapeLengthControl.id] ?? 0;
  let shown: { radius: number; update: ToolUpdate } | undefined;
  return { surface: true, update(input): ToolUpdate | undefined {
    const rim = rimPoint(hit.point, hit.normal, input);
    const reach = Math.hypot(rim[0]! - hit.point.x, rim[1]! - hit.point.y, rim[2]! - hit.point.z) / Math.min(...cell);
    // Half a cell is the smallest radius: one voxel across.
    const radius = Math.max(.5, quantized(reach));
    if (shown?.radius === radius) return shown.update;
    const half = centre.map((_, a) => which === "cylinder" && a === axis && length > 0 ? length / 2 : radius);
    const patch: Patch = {
      ...subcellPatch(centre.map((v, a) => v - half[a]!) as unknown as Cell,
        centre.map((v, a) => v + half[a]!) as unknown as Cell, operation, KINDS[which]),
      ...(which === "cylinder" ? { axis } : {}),
      ...(operation === "clear" ? { affectsScenery: true } : { materialId }),
    };
    const patches = mirroredPatches(context, [patch]);
    assertStrokeCaps(patches, "Use a smaller radius, or fewer mirror planes.");
    const update: ToolUpdate = { patches, replaces: true, tone: operationTone(operation),
      highlight: strokeHighlight(scene, patches),
      caption: `${operationCaption(operation)} ${which.toUpperCase()} · radius ${radius} · ${patchesSize(patches)}` };
    shown = { radius, update };
    return update;
  } };
}

export const shapeTool: VoxelToolPlugin = {
  id: "shape", version: 1, execution: "live",
  ui: { label: "Shape", hint: "Press where the centre goes and drag out the radius: a sphere, a box, or a cylinder along the surface normal.",
    opposite: OPERATION_OPPOSITE,
    group: "Brush", order: 0, key: "u",
    icon: "M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18M12 11a1 1 0 1 0 0 2a1 1 0 1 0 0-2M12 12h9",
    hints: {
      armed: { keys: ["operation"], text: "press the centre, drag out the radius" },
      active: { keys: ["cancel"], text: "release to keep it" },
      after: (values) => ({ keys: ["lock", "undo"], text: toolOperation(values) === "erase"
        ? "erase cuts voxels, terrain and scenery" : "press again for another" }),
    },
    controls: [operationControl, shapeKindControl.control, materialControl, shapeLengthControl, ...mirrorControls] },
  unavailable: voxelToolUnavailable,
  begin: beginShape,
};
