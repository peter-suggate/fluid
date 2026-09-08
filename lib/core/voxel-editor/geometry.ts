import type { EditorRay } from "../editor-entity";
import { containerShellContains, pickSolidVoxel, solidVoxelWorldBox } from "../editor-solid-voxel";
import { sceneCellSizes_m } from "../scene-lattice";
import type { SolidWorldCoordinate as Cell, SolidWorldVoxelPatch as Patch } from "../solid-world";
import type { ToolContext, ToolControl, ToolGesture, ToolUpdate } from "./plugin";

export const sizeControl: ToolControl = { id: "size", presentation: "primary", label: "Width · voxels", min: 1, max: 16, step: 1, initial: 1 };
export const depthControl: ToolControl = { id: "depth", presentation: "primary", label: "Depth · voxels", min: 1, max: 32, step: 1, initial: 1 };
export const planeControl: ToolControl = { id: "plane", presentation: "advanced", label: "Empty-space height · voxels", min: -64, max: 128, step: 1, initial: 0 };
export const shellControl: ToolControl = { id: "shell", kind: "toggle", presentation: "advanced", label: "Edit tank walls", min: 0, max: 1, step: 1, initial: 0 };
export const mirrorControl: ToolControl = { id: "mirror", kind: "toggle", presentation: "advanced", label: "Mirror X", min: 0, max: 1, step: 1, initial: 0 };
/** Limits are checked before accepting an update, never silently truncated. */
export const MAX_STROKE_VOXELS = 32768;
export const MAX_STROKE_PATCHES = 4096;
export function interpolateCells(a: Cell, b: Cell): Cell[] {
  const steps = Math.max(...a.map((v, i) => Math.abs(v - b[i]!)));
  if (steps > 256) throw new Error("Stroke span exceeds 256 voxels; use shorter strokes.");
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
export function beginShapeGesture(context: ToolContext, operation: Patch["operation"],
  mode: "brush" | "box" | "stamp" | "line", shape: Shape = "box"): ToolGesture | undefined {
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
    return { patches, highlight: { kind: "box", box: {
      min: solidVoxelWorldBox(scene, minimum).min, max: solidVoxelWorldBox(scene, maximum).max } },
      caption: `${operation === "fill" ? "BUILD" : "CARVE"} · ${minimum.map((v, a) => maximum[a]! - v + 1).join(" × ")}` };
  } };
}

/** The declaration is shared by this family, but each plugin owns its availability. */
export function voxelToolUnavailable({ scene, methodId }: {
  scene: ToolContext["scene"]; methodId: string;
}): string | undefined {
  if (scene.systems?.fluid !== false && methodId !== "adaptive-mass") return "Choose Sparse CM12 to edit solids while water runs.";
  return undefined;
}
