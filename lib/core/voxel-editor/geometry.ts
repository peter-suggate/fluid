import type { EditorRay } from "../editor-entity";
import { containerShellContains, pickSolidVoxel, solidVoxelWorldBox } from "../editor-solid-voxel";
import { takesLiveSolidEdits } from "../method-contract";
import { sceneCellSizes_m } from "../scene-lattice";
import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../solid-world";
import type { ToolContext, ToolControl, ToolGesture, ToolUpdate } from "./plugin";

export const sizeControl: ToolControl = { id: "size", presentation: "primary", label: "Width · voxels", min: 1, max: 64, step: 1, initial: 1 };
export const depthControl: ToolControl = { id: "depth", presentation: "primary", label: "Depth · voxels", min: 1, max: 128, step: 1, initial: 1 };
export const planeControl: ToolControl = { id: "plane", presentation: "advanced", label: "Empty-space height · voxels", min: -64, max: 128, step: 1, initial: 0 };
export const shellControl: ToolControl = { id: "shell", kind: "toggle", presentation: "advanced", label: "Edit tank walls", min: 0, max: 1, step: 1, initial: 0 };
export const mirrorControl: ToolControl = { id: "mirror", kind: "toggle", presentation: "advanced", label: "Mirror X", min: 0, max: 1, step: 1, initial: 0 };
/**
 * Limits are checked before accepting an update, never silently truncated.
 * A box is one patch applied a page at a time, so its voxel count is cheap; the
 * bound that matters is the runtime's page capacity, which release preflights.
 */
export const MAX_STROKE_VOXELS = 128 ** 3;
export const MAX_STROKE_PATCHES = 4096;
/** Outlines drawn for an unreleased stroke; the rest are implied by its bounds. */
const MAX_PREVIEW_BOXES = 96;
export function interpolateCells(a: Cell, b: Cell): Cell[] {
  const steps = Math.max(...a.map((v, i) => Math.abs(v - b[i]!)));
  if (steps > 1024) throw new Error("Stroke span exceeds 1,024 voxels; use shorter strokes.");
  return Array.from({ length: steps + 1 }, (_, i) => a.map((v, axis) =>
    Math.round(v + (b[axis]! - v) * (steps ? i / steps : 0))) as unknown as Cell);
}
export function mirrorPatchX(patch: Patch, centreX: number): Patch {
  return { ...patch,
    minimum: [centreX - patch.maximumExclusive[0], patch.minimum[1], patch.minimum[2]],
    maximumExclusive: [centreX - patch.minimum[0], patch.maximumExclusive[1], patch.maximumExclusive[2]] };
}
export type Shape = "box" | "sphere" | "cylinder";
export function shapePatches(minimum: Cell, maximumExclusive: Cell, operation: Patch["operation"], shape: Shape, normalAxis: number = 1): Patch[] {
  const patch = (min: Cell, max: Cell): Patch => ({ operation, minimum: min, maximumExclusive: max,
    ...(operation === "fill" ? { materialId: 2 } : {}) });
  const extent = minimum.map((v, axis) => maximumExclusive[axis]! - v);
  if (extent.some((v) => v < 1 || !Number.isSafeInteger(v))
    || extent.reduce((a, b) => a * b, 1) > MAX_STROKE_VOXELS) {
    throw new Error(`Use a smaller region (at most ${MAX_STROKE_VOXELS.toLocaleString()} voxels).`);
  }
  if (shape === "box") return [patch(minimum, maximumExclusive)];
  const result: Patch[] = [];
  const centre = minimum.map((v, axis) => (v + maximumExclusive[axis]!) / 2);
  for (let z = minimum[2]; z < maximumExclusive[2]; z++) {
    for (let y = minimum[1]; y < maximumExclusive[1]; y++) {
      let first: number | undefined;
      let last = 0;
      for (let x = minimum[0]; x < maximumExclusive[0]; x++) {
        const q = [x, y, z].map((v, axis) => (v + 0.5 - centre[axis]!) / (extent[axis]! / 2));
        if (q.reduce((sum, v, a) => sum + (shape === "cylinder" && a === normalAxis ? 0 : v * v), 0) <= 1) {
          first ??= x; last = x + 1;
        }
      }
      if (first !== undefined) result.push(patch([first, y, z], [last, y + 1, z + 1]));
    }
  }
  return result;
}

/** A stable face plane prevents the current stroke from chasing its own surface. */
export function beginShapeGesture(context: ToolContext, authored: Patch["operation"],
  mode: "brush" | "box" | "stamp" | "line", shape: Shape = "box"): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const operation: Patch["operation"] = context.invert ? (authored === "fill" ? "clear" : "fill") : authored;
  const cell = sceneCellSizes_m(scene);
  const origin = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const hit = pickSolidVoxel(scene, ray, undefined, {
    rejectBudgetExhaustion: true,
    skip: (coordinate) => values.shell !== 1 && coordinate[1] >= 0
      && containerShellContains(scene, coordinate),
  });
  const axis = hit?.faceAxis ?? 1;
  const sign = hit?.faceSign ?? 1;
  const plane = hit ? origin[axis]! + (hit.coordinate[axis] + (sign > 0 ? 1 : 0)) * cell[axis]!
    : (values.plane ?? 0) * cell[1];
  const project = (input: EditorRay): Cell | undefined => {
    const o = [input.origin.x, input.origin.y, input.origin.z];
    const d = [input.direction.x, input.direction.y, input.direction.z];
    if (Math.abs(d[axis]!) < 1e-8) return undefined;
    const t = (plane - o[axis]!) / d[axis]!;
    if (!(t > 0)) return undefined;
    const q = o.map((v, a) => Math.floor((v + t * d[a]! - origin[a]!) / cell[a]!));
    q[axis] = hit ? hit.coordinate[axis] + (operation === "fill" ? sign : 0) : values.plane ?? 0;
    return q.every((v) => Number.isSafeInteger(v) && Math.abs(v) < 1_000_000) ? q as unknown as Cell : undefined;
  };
  const start = project(ray);
  if (!start) return undefined;
  let last = start;
  const accumulated = new Map<string, Patch>();
  const size = values.size ?? 1;
  const depth = values.depth ?? size;
  const mirrorCentre = Math.round(scene.container.width_m / cell[0]);
  const stamp = (at: Cell) => {
    const min = at.map((v, a) => a === axis ? v - (sign < 0 ? depth - 1 : 0) : v - Math.floor((size - 1) / 2));
    const max = min.map((v, a) => v + (a === axis ? depth : size));
    // Erasing extends INTO the picked face, adding extends OUT from it.
    if (operation === "clear" && hit) {
      min[axis] = at[axis] - (sign > 0 ? depth - 1 : 0);
      max[axis] = min[axis]! + depth;
    }
    return shapePatches(min as unknown as Cell, max as unknown as Cell, operation, shape, axis);
  };
  return { update(input): ToolUpdate | undefined {
    const end = project(input);
    if (!end) return undefined;
    let patches: Patch[];
    if (mode === "box") {
      const base = stamp(start)[0]!;
      const endBox = stamp(end)[0]!;
      patches = shapePatches(base.minimum.map((v, a) => Math.min(v, endBox.minimum[a]!)) as unknown as Cell,
        base.maximumExclusive.map((v, a) => Math.max(v, endBox.maximumExclusive[a]!)) as unknown as Cell,
        operation, shape, axis);
    } else if (mode === "stamp") patches = stamp(end);
    else {
      const next = new Map(mode === "brush" ? accumulated : []);
      for (const at of interpolateCells(mode === "brush" ? last : start, end)) {
        for (const p of stamp(at)) next.set(JSON.stringify(p), p);
      }
      patches = [...next.values()];
    }
    if (values.mirror) patches = patches.flatMap((p) => [p, mirrorPatchX(p, mirrorCentre)]);
    patches = [...new Map(patches.map((p) => [JSON.stringify(p), p])).values()];
    const work = patches.reduce((sum, p) => sum + p.minimum.reduce((n, v, a) => n * (p.maximumExclusive[a]! - v), 1), 0);
    if (patches.length > MAX_STROKE_PATCHES || work > MAX_STROKE_VOXELS) throw new Error("Stroke is full; release and start another stroke.");
    if (mode === "brush") {
      accumulated.clear();
      // Retain the accepted stroke; mirrored stamps are deduplicated above.
      for (const p of patches) accumulated.set(JSON.stringify(p), p);
    }
    last = end;
    const minimum = [0, 1, 2].map((a) => Math.min(...patches.map((p) => p.minimum[a]!))) as unknown as Cell;
    const maximum = [0, 1, 2].map((a) => Math.max(...patches.map((p) => p.maximumExclusive[a]!)) - 1) as unknown as Cell;
    const worldBox = (lo: Cell, hi: Cell) => ({ min: solidVoxelWorldBox(scene, lo).min, max: solidVoxelWorldBox(scene, hi).max });
    const bounds = worldBox(minimum, maximum);
    // Nothing is voxelized until release, so the outline is the whole preview:
    // the stroke's bounds first (it anchors the caption), then its newest pieces.
    const members = patches.slice(-MAX_PREVIEW_BOXES).map((p) =>
      worldBox(p.minimum, p.maximumExclusive.map((v) => v - 1) as unknown as Cell));
    return { patches, highlight: patches.length === 1 ? { kind: "box", box: bounds }
      : { kind: "boxes", boxes: [bounds, ...members], truncated: patches.length > MAX_PREVIEW_BOXES },
      caption: `${operation === "fill" ? "BUILD" : "CARVE"} · ${minimum.map((v, a) => maximum[a]! - v + 1).join(" × ")}` };
  } };
}

/**
 * Push/pull: drag a footprint across a face, let go, then move the bare pointer
 * along the face normal — outward builds, inward carves — and click to commit.
 *
 * Depth is the one dimension a single drag on a plane cannot express, which is
 * why every other tool here takes it as a typed number. Splitting the gesture
 * at the release gives it a drag of its own. The extrusion is measured where
 * the pointer ray passes closest to the normal through the footprint's centre,
 * so it follows the cursor from any view except straight down that normal.
 */
export function beginPushPullGesture(context: ToolContext): ToolGesture | undefined {
  const { scene, values, ray } = context;
  const cell = sceneCellSizes_m(scene);
  const origin = [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2];
  const hit = pickSolidVoxel(scene, ray, undefined, {
    rejectBudgetExhaustion: true,
    skip: (coordinate) => values.shell !== 1 && coordinate[1] >= 0
      && containerShellContains(scene, coordinate),
  });
  const axis = hit?.faceAxis ?? 1;
  const sign = hit?.faceSign ?? 1;
  // The layer of cells the face belongs to; empty space stands on the construction plane.
  const surface = hit ? hit.coordinate[axis] : (values.plane ?? 0) - 1;
  const plane = origin[axis]! + (surface + (sign > 0 ? 1 : 0)) * cell[axis]!;
  const project = (input: EditorRay): number[] | undefined => {
    const o = [input.origin.x, input.origin.y, input.origin.z];
    const d = [input.direction.x, input.direction.y, input.direction.z];
    if (Math.abs(d[axis]!) < 1e-8) return undefined;
    const t = (plane - o[axis]!) / d[axis]!;
    if (!(t > 0)) return undefined;
    const q = o.map((v, a) => Math.floor((v + t * d[a]! - origin[a]!) / cell[a]!));
    return q.every((v) => Number.isSafeInteger(v) && Math.abs(v) < 1_000_000) ? q : undefined;
  };
  const start = project(ray);
  if (!start) return undefined;
  let lo = start, hi = start, extruding = false, depth = 0;
  const mirrorCentre = Math.round(scene.container.width_m / cell[0]);
  const limit = depthControl.max;
  const measure = (input: EditorRay): number | undefined => {
    const centre = [0, 1, 2].map((a) => a === axis ? plane : origin[a]! + (lo[a]! + hi[a]! + 1) / 2 * cell[a]!);
    const o = [input.origin.x, input.origin.y, input.origin.z];
    const d = [input.direction.x, input.direction.y, input.direction.z];
    const w = centre.map((v, a) => v - o[a]!);
    const dd = d.reduce((sum, v) => sum + v * v, 0), dn = d[axis]! * sign;
    const denominator = dd - dn * dn;
    // Looking straight down the normal, the pointer says nothing about depth.
    if (!(denominator > 1e-6 * dd)) return undefined;
    const dw = d.reduce((sum, v, a) => sum + v * w[a]!, 0), nw = w[axis]! * sign;
    const along_m = (dn * dw - dd * nw) / denominator;
    return Math.max(-limit, Math.min(limit, Math.round(along_m / cell[axis]!)));
  };
  return {
    advance() {
      if (extruding) return false;
      extruding = true;
      return true;
    },
    update(input): ToolUpdate | undefined {
      if (!extruding) {
        const end = project(input);
        if (!end) return undefined;
        lo = start.map((v, a) => Math.min(v, end[a]!)); hi = start.map((v, a) => Math.max(v, end[a]!));
      } else depth = measure(input) ?? depth;
      const build = depth >= 0;
      const layers = Math.max(1, Math.abs(depth));
      // Outward layers start beyond the face; inward layers start at it.
      const from = build ? surface + sign : surface;
      const near = build === sign > 0 ? from : from - (layers - 1);
      const minimum = lo.map((v, a) => a === axis ? near : v) as unknown as Cell;
      const maximumExclusive = hi.map((v, a) => a === axis ? near + layers : v + 1) as unknown as Cell;
      let patches = extruding && depth === 0 ? []
        : shapePatches(minimum, maximumExclusive, build ? "fill" : "clear", "box", axis);
      if (values.mirror) patches = patches.flatMap((p) => [p, mirrorPatchX(p, mirrorCentre)]);
      const last = maximumExclusive.map((v) => v - 1) as unknown as Cell;
      const size = minimum.map((v, a) => a === axis ? (extruding ? Math.abs(depth) : 0) : maximumExclusive[a]! - v);
      return { patches: extruding ? patches : [], tone: build ? undefined : "region",
        highlight: { kind: "box", box: { min: solidVoxelWorldBox(scene, minimum).min, max: solidVoxelWorldBox(scene, last).max } },
        caption: !extruding ? `FOOTPRINT · ${size.filter((_, a) => a !== axis).join(" × ")} · release, then pull or push`
          : `${depth === 0 ? "PULL OR PUSH" : build ? "PULL" : "PUSH"} · ${size.join(" × ")} · click to commit` };
    },
  };
}

/** The declaration is shared by this family, but each plugin owns its availability. */
export function voxelToolUnavailable({ scene, methodId }: {
  scene: ToolContext["scene"]; methodId: string;
}): string | undefined {
  if (scene.systems?.fluid !== false && !takesLiveSolidEdits(methodId)) return "This fluid method cannot take solid edits while water runs.";
  return undefined;
}
