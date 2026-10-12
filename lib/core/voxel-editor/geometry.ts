import type { EditorRay } from "../editor-entity";
import { containerShellContains, solidVoxelWorldBox } from "../editor-solid-voxel";
import { takesLiveSolidEdits } from "../method-contract";
import { sceneCellSizes_m } from "../scene-lattice";
import type { EditorHighlight } from "../editor-target";
import type { SceneDescription, Vec3 } from "../model";
import { decodeVoxelBlock, encodeVoxelBlock } from "../solid-patch/block";
import { SOLID_PATCH_BLOCK_KIND, SOLID_PATCH_CYLINDER_KIND, SOLID_PATCH_ELLIPSOID_KIND,
  solidPatchKindId } from "../solid-patch/kind";
import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../solid-world";
import { DEFAULT_VOXEL_MATERIAL_ID } from "../voxel-palette";
import { depthControl, materialControl, operationControl, shellControl, toolMirrorAxes, toolOperation } from "./controls";
import { toolReader, type ToolContext, type ToolGesture, type ToolUpdate, type VoxelToolPlugin } from "./plugin";
import { flipBlock } from "./voxel-grid";

/**
 * Limits are checked before accepting an update, never silently truncated.
 *
 * A stamped shape is one patch whatever its size — a box, or the ellipsoid or
 * cylinder inscribed in one — so the two bounds are about different costs. The
 * patch count bounds the *document*: every stamp of a stroke is a record that
 * is saved, cloned and replayed by each bake. The voxel count bounds the *bake*:
 * a patch is composed over the cells of its box, so the work a stroke asks for
 * is the summed volume of its patches' bounds, overlaps included. The runtime's
 * page capacity is a third bound, and release preflights it.
 */
export const MAX_STROKE_VOXELS = 128 ** 3;
export const MAX_STROKE_PATCHES = 4096;
/** Outlines drawn for an unreleased stroke; the rest are implied by its bounds. */
const MAX_PREVIEW_BOXES = 96;
/** A curved outline is tens of segments where a box is twelve, so fewer are drawn. */
const MAX_PREVIEW_CURVES = 12;
export function interpolateCells(a: Cell, b: Cell): Cell[] {
  const steps = Math.max(...a.map((v, i) => Math.abs(v - b[i]!)));
  if (steps > 1024) throw new Error("Stroke span exceeds 1,024 voxels; use shorter strokes.");
  return Array.from({ length: steps + 1 }, (_, i) => a.map((v, axis) =>
    Math.round(v + (b[axis]! - v) * (steps ? i / steps : 0))) as unknown as Cell);
}
/**
 * The flipped payloads of the block patches last mirrored. A stamp that
 * follows the pointer mirrors the same block at every sample, and flipping one
 * is a decode, a pass over its cells and an encode.
 */
const flippedPayloads: { readonly payload: string; readonly axis: number; readonly flipped: string }[] = [];
function flippedPayload(patch: Patch, axis: 0 | 1 | 2): string {
  const known = flippedPayloads.find((entry) => entry.axis === axis && entry.payload === patch.payload);
  if (known) return known.flipped;
  const size = patch.minimum.map((v, a) => patch.maximumExclusive[a]! - v) as unknown as Cell;
  const flipped = encodeVoxelBlock(flipBlock(decodeVoxelBlock(patch.payload!, size), axis));
  if (flippedPayloads.length === 16) flippedPayloads.shift();
  flippedPayloads.push({ payload: patch.payload!, axis, flipped });
  return flipped;
}
/**
 * A patch's image in the plane `2 x = centre` across one axis.
 *
 * A box, and the ellipsoid or cylinder inscribed in one, are symmetric about
 * every axis plane, so only their box moves: kind and axis are unchanged by any
 * reflection. A block's cells are not, so its payload is flipped along the
 * axis as well.
 */
export function mirrorPatch(patch: Patch, axis: 0 | 1 | 2, centre: number): Patch {
  const minimum = [...patch.minimum], maximumExclusive = [...patch.maximumExclusive];
  minimum[axis] = centre - patch.maximumExclusive[axis];
  maximumExclusive[axis] = centre - patch.minimum[axis];
  return { ...patch, minimum: minimum as unknown as Cell, maximumExclusive: maximumExclusive as unknown as Cell,
    ...(solidPatchKindId(patch) === SOLID_PATCH_BLOCK_KIND ? { payload: flippedPayload(patch, axis) } : {}) };
}
export const mirrorPatchX = (patch: Patch, centreX: number): Patch => mirrorPatch(patch, 0, centreX);
/** Where each axis' mirror plane stands, as the `centre` of `mirrorPatch`: the container's middle. */
export function mirrorCentres(scene: SceneDescription): readonly [number, number, number] {
  const cell = sceneCellSizes_m(scene);
  return [Math.round(scene.container.width_m / cell[0]), Math.round(scene.container.height_m / cell[1]),
    Math.round(scene.container.depth_m / cell[2])];
}
/**
 * A proposal and its images under the mirror planes that are on: each patch,
 * then up to seven reflections of it. A symmetric shape standing across a
 * plane is its own image there and is not written twice; a block is never
 * assumed to be, because its cells need not be symmetric though its box is.
 */
export function mirroredPatches(context: Pick<ToolContext, "scene" | "values">, patches: readonly Patch[]): Patch[] {
  const centres = mirrorCentres(context.scene);
  let images = [...patches];
  for (const axis of toolMirrorAxes(context.values)) {
    images = images.flatMap((patch) => {
      const image = mirrorPatch(patch, axis, centres[axis]);
      return image.minimum[axis] === patch.minimum[axis] && solidPatchKindId(patch) !== SOLID_PATCH_BLOCK_KIND
        ? [patch] : [patch, image];
    });
  }
  return images;
}
/**
 * Refuse a proposal past either stroke bound, before it is shown: more patches
 * than one stroke may record, or more voxels under their boxes than one bake
 * should be asked to compose.
 */
export function assertStrokeCaps(patches: readonly Patch[],
  message = "Stroke is full; release and start another stroke."): void {
  let voxels = 0;
  for (const patch of patches) {
    voxels += patch.minimum.reduce((volume, v, a) => volume * (patch.maximumExclusive[a]! - v), 1);
  }
  if (patches.length > MAX_STROKE_PATCHES || voxels > MAX_STROKE_VOXELS) throw new Error(message);
}
export type Shape = "box" | "sphere" | "cylinder";
export interface ShapeOptions {
  /** The material a fill or a paint writes. Defaults to the palette's default. */
  readonly materialId?: number;
  /** Wall thickness in voxels of a hollow shape; absent or 0 is solid. Only an attached shape can be hollow. */
  readonly hollow?: number;
}
/**
 * One stamped shape as patches: one, or two when it is hollow.
 *
 * The box is the patch's bounds and the kind says what stands inside it, so a
 * sphere costs the same record as a cube. A hollow shape is its outer shape
 * filled and the same shape, `hollow` voxels smaller on every side, cleared out
 * of it — in that order, which is why the pair is returned together.
 */
export function shapePatches(minimum: Cell, maximumExclusive: Cell, operation: Patch["operation"], shape: Shape,
  normalAxis: number = 1, options: ShapeOptions = {}): Patch[] {
  const materialId = options.materialId ?? DEFAULT_VOXEL_MATERIAL_ID;
  const kind = shape === "sphere" ? { kind: SOLID_PATCH_ELLIPSOID_KIND }
    : shape === "cylinder" ? { kind: SOLID_PATCH_CYLINDER_KIND, axis: normalAxis as 0 | 1 | 2 } : {};
  const patch = (min: Cell, max: Cell, apply: Patch["operation"]): Patch => ({ operation: apply, minimum: min,
    maximumExclusive: max, ...(apply === "clear" ? {} : { materialId }), ...kind });
  const extent = minimum.map((v, axis) => maximumExclusive[axis]! - v);
  if (extent.some((v) => v < 1 || !Number.isSafeInteger(v))
    || extent.reduce((a, b) => a * b, 1) > MAX_STROKE_VOXELS) {
    throw new Error(`Use a smaller region (at most ${MAX_STROKE_VOXELS.toLocaleString()} voxels).`);
  }
  if (shape === "cylinder" && normalAxis !== 0 && normalAxis !== 1 && normalAxis !== 2) {
    throw new Error(`A cylinder needs an axis of 0, 1 or 2, not ${normalAxis}.`);
  }
  const outer = patch(minimum, maximumExclusive, operation);
  const wall = options.hollow ?? 0;
  if (wall === 0) return [outer];
  if (operation !== "fill") throw new Error("Only an attached shape can be hollow.");
  if (!Number.isSafeInteger(wall) || wall < 0) throw new Error(`A hollow shape needs a whole wall thickness, not ${wall}.`);
  // Walls that meet leave nothing to clear: the shape is solid.
  if (extent.some((v) => v <= 2 * wall)) return [outer];
  return [outer, patch(minimum.map((v) => v + wall) as unknown as Cell,
    maximumExclusive.map((v) => v - wall) as unknown as Cell, "clear")];
}

const worldBox = (scene: SceneDescription, patch: Pick<Patch, "minimum" | "maximumExclusive">) => ({
  min: solidVoxelWorldBox(scene, patch.minimum).min,
  max: solidVoxelWorldBox(scene, patch.maximumExclusive.map((v) => v - 1) as unknown as Cell).max,
});
/**
 * A curved patch as wire loops: the ellipsoid's three great ellipses, or the
 * cylinder's two rims and four of its rulings. Sampled from the same inscribed
 * shape the bake evaluates, so the ghost is the solid that release will write.
 */
function curveOutline(scene: SceneDescription, patch: Patch): Vec3[][] {
  const box = worldBox(scene, patch);
  const lo = [box.min.x, box.min.y, box.min.z], hi = [box.max.x, box.max.y, box.max.z];
  const centre = lo.map((v, a) => (v + hi[a]!) / 2), radius = lo.map((v, a) => (hi[a]! - v) / 2);
  const point = (q: readonly number[]): Vec3 => ({ x: q[0]!, y: q[1]!, z: q[2]! });
  /** The ellipse in the plane normal to `normal`, at `offset` along it from the centre. */
  const loop = (normal: number, offset: number): Vec3[] => Array.from({ length: 33 }, (_, i) => {
    const angle = 2 * Math.PI * i / 32, u = (normal + 1) % 3, v = (normal + 2) % 3;
    const q = [...centre];
    q[normal] = centre[normal]! + offset;
    q[u] = centre[u]! + radius[u]! * Math.cos(angle);
    q[v] = centre[v]! + radius[v]! * Math.sin(angle);
    return point(q);
  });
  if (solidPatchKindId(patch) === SOLID_PATCH_ELLIPSOID_KIND) return [loop(0, 0), loop(1, 0), loop(2, 0)];
  const axis = patch.axis ?? 1, u = (axis + 1) % 3, v = (axis + 2) % 3;
  const rulings = [[1, 0], [-1, 0], [0, 1], [0, -1]].map(([su, sv]) => [-1, 1].map((end) => {
    const q = [...centre];
    q[axis] = centre[axis]! + end * radius[axis]!;
    q[u] = centre[u]! + su! * radius[u]!;
    q[v] = centre[v]! + sv! * radius[v]!;
    return point(q);
  }));
  return [loop(axis, -radius[axis]!), loop(axis, radius[axis]!), ...rulings];
}
/**
 * What an unreleased stroke draws. Nothing is voxelized until release, so the
 * outline is the whole preview: the stroke's bounds first (it anchors the
 * caption), then its newest pieces — boxes as boxes, and a curved piece as the
 * curve it is, because its bounding box would promise a cube.
 */
export function strokeHighlight(scene: SceneDescription, patches: readonly Patch[]): EditorHighlight {
  const bounds = worldBox(scene, {
    minimum: [0, 1, 2].map((a) => Math.min(...patches.map((p) => p.minimum[a]!))) as unknown as Cell,
    maximumExclusive: [0, 1, 2].map((a) => Math.max(...patches.map((p) => p.maximumExclusive[a]!))) as unknown as Cell,
  });
  const curved = (patch: Patch) => {
    const kind = solidPatchKindId(patch);
    return kind === SOLID_PATCH_ELLIPSOID_KIND || kind === SOLID_PATCH_CYLINDER_KIND;
  };
  if (!patches.some(curved)) {
    return patches.length === 1 ? { kind: "box", box: bounds }
      : { kind: "boxes", boxes: [bounds, ...patches.slice(-MAX_PREVIEW_BOXES).map((p) => worldBox(scene, p))],
        truncated: patches.length > MAX_PREVIEW_BOXES };
  }
  const corner = (box: { min: Vec3; max: Vec3 }, i: number): Vec3 =>
    ({ x: i & 1 ? box.max.x : box.min.x, y: i & 2 ? box.max.y : box.min.y, z: i & 4 ? box.max.z : box.min.z });
  // A box among curves is drawn as its edges, since one highlight is one kind.
  const boxEdges = (box: { min: Vec3; max: Vec3 }): Vec3[][] => [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3],
    [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]].map(([a, b]) => [corner(box, a!), corner(box, b!)]);
  // The corner a box highlight hangs its caption on, so the caption does not jump between kinds.
  return { kind: "paths", anchor_m: { x: bounds.min.x, y: bounds.max.y, z: bounds.max.z },
    paths: patches.slice(-MAX_PREVIEW_CURVES).flatMap((p) => curved(p) ? curveOutline(scene, p) : boxEdges(worldBox(scene, p))) };
}

/** A face a stroke can stand on: a solid cell, and which of its six sides the ray met. */
export interface ToolSurface {
  /** The solid cell — or, where nothing was under the ray, the cell under the ground plane there. */
  readonly coordinate: Cell;
  readonly faceAxis: 0 | 1 | 2;
  readonly faceSign: -1 | 1;
  /** False when the ray met nothing and the surface is the ground plane. */
  readonly picked: boolean;
}
const worldOrigin = (scene: SceneDescription): readonly [number, number, number] =>
  [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
/**
 * Where a ray meets the plane of a surface's face, in cells and fractions of
 * one. Undefined when the ray runs along the plane or away from it.
 */
export function facePoint(scene: SceneDescription, surface: Pick<ToolSurface, "coordinate" | "faceAxis" | "faceSign">,
  ray: EditorRay): number[] | undefined {
  const cell = sceneCellSizes_m(scene), origin = worldOrigin(scene), axis = surface.faceAxis;
  const plane = surface.coordinate[axis] + (surface.faceSign > 0 ? 1 : 0);
  const o = [ray.origin.x, ray.origin.y, ray.origin.z];
  const d = [ray.direction.x, ray.direction.y, ray.direction.z];
  if (Math.abs(d[axis]!) < 1e-8) return undefined;
  const t = (origin[axis] + plane * cell[axis] - o[axis]!) / d[axis]!;
  if (!(t > 0)) return undefined;
  const q = o.map((v, a) => (v + t * d[a]! - origin[a]!) / cell[a]!);
  q[axis] = plane;
  return q.every((v) => Number.isFinite(v) && Math.abs(v) < 1_000_000) ? q : undefined;
}
/** The cell of a surface's own layer under that point. */
export function faceCell(scene: SceneDescription, surface: Pick<ToolSurface, "coordinate" | "faceAxis" | "faceSign">,
  ray: EditorRay): Cell | undefined {
  const point = facePoint(scene, surface, ray);
  return point && point.map((v, a) => a === surface.faceAxis ? surface.coordinate[a]! : Math.floor(v)) as unknown as Cell;
}
/**
 * Whether a tool's ray passes through a cell as if it were not there: the
 * tank's own walls, unless the shared walls switch is on.
 *
 * A surface reached that way may stand flush against the wall it was reached
 * through, so whoever asks what is in front of it has to ask with this too.
 */
export function toolPassesThrough(context: Pick<ToolContext, "scene" | "values">, coordinate: Cell): boolean {
  return context.values[shellControl.id] !== 1 && coordinate[1] >= 0
    && containerShellContains(context.scene, coordinate);
}
/**
 * What a ray lands on: the first solid cell along it, read from the scene the
 * gesture began on — so a stroke never chases the surface it is itself
 * proposing — and, where it meets nothing, the ground plane.
 *
 * The tank's own walls are passed through unless the shared walls switch is
 * on; the floor is always a surface. `press` is the sample a gesture begins
 * on: a ray too long to walk is refused there, and merely lands on nothing at
 * a later sample, which a drag that swings past the horizon produces freely.
 */
export function toolSurface(context: Pick<ToolContext, "scene" | "values">, ray: EditorRay,
  press = true): ToolSurface | undefined {
  const { scene } = context;
  const hit = toolReader(context).pick(ray, {
    rejectBudgetExhaustion: press,
    skip: (coordinate) => toolPassesThrough(context, coordinate),
  });
  if (hit) return { coordinate: hit.coordinate, faceAxis: hit.faceAxis, faceSign: hit.faceSign, picked: true };
  const ground = { coordinate: [0, -1, 0], faceAxis: 1, faceSign: 1 } as const;
  const coordinate = faceCell(scene, ground, ray);
  return coordinate && { ...ground, coordinate, picked: false };
}
/**
 * What the "do the opposite" modifier samples while the hand paints: the
 * palette material of the voxel under the pointer, and the ghost that says so.
 *
 * Undefined when the tool is not painting, lists no material to set, or the
 * pointer is on nothing the palette holds — the press is then the tool's own.
 */
export function sampledMaterial(plugin: VoxelToolPlugin, context: Pick<ToolContext, "scene" | "values">, ray: EditorRay):
  { readonly materialId: number; readonly preview: ToolUpdate } | undefined {
  const { controls } = plugin.ui;
  if (!controls.includes(materialControl) || !controls.includes(operationControl)
    || toolOperation(context.values) !== "paint") return undefined;
  const surface = toolSurface(context, ray, false);
  if (!surface?.picked) return undefined;
  const materialId = toolReader(context).material(surface.coordinate);
  const entry = materialControl.options.find((option) => option.value === materialId);
  if (!entry) return undefined;
  const cell = { minimum: surface.coordinate, maximumExclusive: surface.coordinate.map((v) => v + 1) as unknown as Cell };
  return { materialId, preview: { patches: [], tone: "prop", highlight: boxHighlight(context.scene, cell),
    caption: `PICK MATERIAL · ${entry.label.toUpperCase()}` } };
}
/**
 * How far along a surface's outward normal a ray points, in whole layers:
 * measured where the ray passes closest to the normal through `through`, a
 * point of the face as `facePoint` gives it. Positive is out of the solid.
 * Undefined looking straight down the normal, where the pointer says nothing
 * about depth.
 *
 * `through` is where the hand last touched the face, not the middle of what it
 * drew there: the ray that put it there then measures exactly zero, so a click
 * that does not move is a depth of nothing from any angle.
 */
export function normalTravel(scene: SceneDescription, surface: Pick<ToolSurface, "coordinate" | "faceAxis" | "faceSign">,
  through: readonly number[], ray: EditorRay, limit: number): number | undefined {
  const cell = sceneCellSizes_m(scene), origin = worldOrigin(scene), axis = surface.faceAxis, sign = surface.faceSign;
  const plane = origin[axis] + (surface.coordinate[axis] + (sign > 0 ? 1 : 0)) * cell[axis];
  const centre = [0, 1, 2].map((a) => a === axis ? plane : origin[a]! + through[a]! * cell[a]!);
  const o = [ray.origin.x, ray.origin.y, ray.origin.z];
  const d = [ray.direction.x, ray.direction.y, ray.direction.z];
  const w = centre.map((v, a) => v - o[a]!);
  const dd = d.reduce((sum, v) => sum + v * v, 0), dn = d[axis]! * sign;
  const denominator = dd - dn * dn;
  if (!(denominator > 1e-6 * dd)) return undefined;
  const dw = d.reduce((sum, v, a) => sum + v * w[a]!, 0), nw = w[axis]! * sign;
  const along_m = (dn * dw - dd * nw) / denominator;
  return Math.max(-limit, Math.min(limit, Math.round(along_m / cell[axis])));
}
/**
 * The box `layers` deep that stands on a face over a tangential span of cells
 * (`lo` and `hi` inclusive): out from the solid, or back into it starting at
 * the face's own layer.
 */
export function faceBox(surface: Pick<ToolSurface, "coordinate" | "faceAxis" | "faceSign">, lo: readonly number[],
  hi: readonly number[], layers: number, outward: boolean): { readonly minimum: Cell; readonly maximumExclusive: Cell } {
  const axis = surface.faceAxis, sign = surface.faceSign, layer = surface.coordinate[axis];
  // Outward layers start beyond the face; inward layers start at it.
  const from = outward ? layer + sign : layer;
  const near = outward === sign > 0 ? from : from - (layers - 1);
  return { minimum: lo.map((v, a) => a === axis ? near : v) as unknown as Cell,
    maximumExclusive: hi.map((v, a) => a === axis ? near + layers : v + 1) as unknown as Cell };
}
/** A patch list's bounds as the " × "-joined size a caption states. */
export function patchesSize(patches: readonly Pick<Patch, "minimum" | "maximumExclusive">[]): string {
  return [0, 1, 2].map((a) => Math.max(...patches.map((p) => p.maximumExclusive[a]!))
    - Math.min(...patches.map((p) => p.minimum[a]!))).join(" × ");
}

/** A drag's footprint on a face and how far it has since been pulled or pushed. */
export interface Footprint {
  readonly surface: ToolSurface;
  /** The dragged span of the face's layer, inclusive at both corners. */
  readonly lo: readonly number[];
  readonly hi: readonly number[];
  /** Layers along the outward normal: positive pulled out, negative pushed in, 0 at the face. */
  readonly depth: number;
  /** False while the footprint is still being dragged. */
  readonly extruding: boolean;
}
/**
 * The two-phase gesture: drag a footprint across a face, let go, then move the
 * bare pointer along the face normal and click to commit.
 *
 * Depth is the one dimension a single drag on a plane cannot express. Splitting
 * the gesture at the release gives it a drag of its own: it is measured along
 * the normal through the point of the face where the drag let go, so it starts
 * at nothing and follows the cursor from any view except straight down that
 * normal.
 *
 * What a footprint and a depth *mean* — a box built or cut, a box selected —
 * is the caller's `propose`; this owns only the hand.
 */
export function beginFootprintGesture(context: ToolContext,
  propose: (footprint: Footprint) => ToolUpdate): ToolGesture | undefined {
  const { scene, ray } = context;
  const surface = toolSurface(context, ray);
  const start = surface && faceCell(scene, surface, ray);
  let released = surface && facePoint(scene, surface, ray);
  if (!surface || !start || !released) return undefined;
  let lo: readonly number[] = start, hi: readonly number[] = start, extruding = false, depth = 0;
  return {
    surface: surface.picked,
    advance() {
      if (extruding) return false;
      extruding = true;
      return true;
    },
    update(input): ToolUpdate | undefined {
      if (!extruding) {
        const end = faceCell(scene, surface, input);
        if (!end) return undefined;
        released = facePoint(scene, surface, input)!;
        lo = start.map((v, a) => Math.min(v, end[a]!)); hi = start.map((v, a) => Math.max(v, end[a]!));
      } else depth = normalTravel(scene, surface, released!, input, depthControl.max) ?? depth;
      return propose({ surface, lo, hi, depth, extruding });
    },
  };
}
/** The outline of one box of cells. */
export function boxHighlight(scene: SceneDescription, box: Pick<Patch, "minimum" | "maximumExclusive">): EditorHighlight {
  return { kind: "box", box: worldBox(scene, box) };
}

/** The declaration is shared by this family, but each plugin owns its availability. */
export function voxelToolUnavailable({ scene, methodId }: {
  scene: ToolContext["scene"]; methodId: string;
}): string | undefined {
  if (scene.systems?.fluid !== false && !takesLiveSolidEdits(methodId)) return "This fluid method cannot take solid edits while water runs.";
  return undefined;
}
