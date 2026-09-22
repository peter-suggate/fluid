import type { EditorHighlight } from "../editor-target";
import type { EditorRay } from "../editor-entity";
import type { InitialLiquidVolume, SceneDescription, Vec3 } from "../model";
import type { LiveFluidEdit } from "../live-fluid-edit";
import { sceneCellSizes_m } from "../scene-lattice";
import type { ToolContext, ToolControl, ToolGesture, ToolUpdate, ToolValues } from "./plugin";

export const fluidSizeControl: ToolControl = { id: "size", presentation: "primary", label: "Diameter · voxels", min: 2, max: 24, step: 1, initial: 6 };
export const fluidHeightControl: ToolControl = { id: "height", presentation: "primary", label: "Height above floor · voxels", min: 0, max: 128, step: 1, initial: 2 };
export const fluidRemoveControl: ToolControl = { id: "remove", kind: "toggle", presentation: "primary", label: "Remove water", min: 0, max: 1, step: 1, initial: 0 };
export const fluidTubeControl: ToolControl = { id: "thickness", presentation: "primary", label: "Tube thickness · voxels", min: 1, max: 10, step: 1, initial: 2 };

/** Start above the initial pool when there is room, while keeping the entire drop in the tank. */
export function fluidToolDefaults(scene: SceneDescription, values: ToolValues, shape: LiveFluidEdit["shape"]): ToolValues {
  const voxel = Math.min(...sceneCellSizes_m(scene));
  const height = scene.container.height_m;
  if (!(voxel > 0) || !Number.isFinite(voxel) || !(height > 0) || !Number.isFinite(height)) return { height: 2 };
  const shapeHeight = (shape === "torus" ? values.thickness : values.size) * voxel;
  const fill = Number.isFinite(scene.container.fillFraction) ? Math.max(0, Math.min(1, scene.container.fillFraction)) : 0;
  const desired = fill > 0 ? Math.ceil((fill * height + 2 * voxel) / voxel) : 2;
  const highestBottom = Math.floor((height - shapeHeight) / voxel + 1e-8);
  return { height: Math.max(0, Math.min(128, desired, highestBottom)) };
}

export function fluidToolUnavailable({ scene, methodId }: { scene: SceneDescription; methodId: string },
  shape: LiveFluidEdit["shape"] = "cube"): string | undefined {
  if (scene.systems?.fluid === false) return "Enable water from Scene to use fluid tools.";
  if (methodId === "adaptive-mass" || methodId === "adaptive-volume") return undefined;
  // The uniform solvers own one water source shape, a ball dropped on the next step.
  if ((methodId === "uniform" || methodId === "uniform-volume") && shape === "ball") return undefined;
  return "Choose Sparse Geometric to edit moving water with this shape.";
}

/** The descriptor is shared by preview and the live command, never rasterized twice by the UI. */
export function fluidShapeBounds(edit: LiveFluidEdit): { min: Vec3; max: Vec3 } {
  const r = edit.radius_m;
  const y = edit.shape === "torus" ? edit.tubeRadius_m! : r;
  return { min: { x: edit.center_m.x - r, y: edit.center_m.y - y, z: edit.center_m.z - r },
    max: { x: edit.center_m.x + r, y: edit.center_m.y + y, z: edit.center_m.z + r } };
}

/** Useful for exact shape checks and authored previews; the live action itself is transient. */
export function fluidShapeVolume(edit: LiveFluidEdit): InitialLiquidVolume {
  if (edit.shape === "cube") { const box = fluidShapeBounds(edit); return { shape: "box", min_m: box.min, max_m: box.max }; }
  if (edit.shape === "torus") return { shape: "torus", center_m: edit.center_m,
    radius_m: edit.radius_m - edit.tubeRadius_m!, tubeRadius_m: edit.tubeRadius_m! };
  return { shape: "sphere", center_m: edit.center_m, radius_m: edit.radius_m };
}

/** Wire loops are sampled from the very same analytic surface sent to the solver. */
export function fluidShapeHighlight(edit: LiveFluidEdit): EditorHighlight {
  if (edit.shape === "cube") return { kind: "box", box: fluidShapeBounds(edit) };
  const c = edit.center_m;
  const circle = (at: (angle: number) => Vec3) => Array.from({ length: 49 }, (_, i) => at(2 * Math.PI * i / 48));
  const r = edit.radius_m;
  const paths: Vec3[][] = [];
  if (edit.shape === "ball") {
    paths.push(circle(a => ({ x: c.x + r * Math.cos(a), y: c.y + r * Math.sin(a), z: c.z })),
      circle(a => ({ x: c.x + r * Math.cos(a), y: c.y, z: c.z + r * Math.sin(a) })),
      circle(a => ({ x: c.x, y: c.y + r * Math.cos(a), z: c.z + r * Math.sin(a) })));
  } else {
    const tube = edit.tubeRadius_m!;
    const major = r - tube;
    // Outer/inner silhouettes and upper/lower tube circles preserve the visible hole.
    for (const phi of [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2]) {
      paths.push(circle(a => ({ x: c.x + (major + tube * Math.cos(phi)) * Math.cos(a),
        y: c.y + tube * Math.sin(phi), z: c.z + (major + tube * Math.cos(phi)) * Math.sin(a) })));
    }
    for (const theta of [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2]) {
      paths.push(circle(a => ({ x: c.x + (major + tube * Math.cos(a)) * Math.cos(theta),
        y: c.y + tube * Math.sin(a), z: c.z + (major + tube * Math.cos(a)) * Math.sin(theta) })));
    }
  }
  return { kind: "paths", paths, anchor_m: c };
}

/** Drop stamps move over a frozen horizontal plane; releasing applies exactly one command. */
export function beginFluidShapeGesture(context: ToolContext, shape: LiveFluidEdit["shape"]): ToolGesture | undefined {
  const { scene, values } = context;
  const voxel = Math.min(...sceneCellSizes_m(scene));
  const radius = values.size * voxel / 2;
  const tube = (values.thickness ?? 2) * voxel / 2;
  if (!(radius > 0) || !Number.isFinite(radius)) throw new Error("Choose a positive fluid size.");
  if (shape === "torus" && (!(tube > 0) || tube >= radius / 2)) {
    throw new Error("Tube thickness must be less than half the outer diameter to keep the ring open.");
  }
  const bottom = (values.height ?? 2) * voxel;
  const halfHeight = shape === "torus" ? tube : radius;
  const centerY = bottom + halfHeight;
  const project = (ray: EditorRay): Vec3 | undefined => {
    if (Math.abs(ray.direction.y) < 1e-8) return undefined;
    const t = (centerY - ray.origin.y) / ray.direction.y;
    if (!(t > 0)) return undefined;
    const snap = (value: number) => Math.round(value / voxel) * voxel;
    return { x: snap(ray.origin.x + t * ray.direction.x), y: centerY, z: snap(ray.origin.z + t * ray.direction.z) };
  };
  if (!project(context.ray)) return undefined;
  return { update(ray): ToolUpdate | undefined {
    const center = project(ray);
    if (!center) return undefined;
    const edit: LiveFluidEdit = { operation: values.remove === 1 ? "remove" : "add", shape,
      center_m: center, radius_m: radius, ...(shape === "torus" ? { tubeRadius_m: tube } : {}) };
    const bounds = fluidShapeBounds(edit);
    const epsilon = voxel * 1e-6;
    if (bounds.min.x < -scene.container.width_m / 2 - epsilon || bounds.max.x > scene.container.width_m / 2 + epsilon
      || bounds.min.z < -scene.container.depth_m / 2 - epsilon || bounds.max.z > scene.container.depth_m / 2 + epsilon
      || bounds.min.y < -epsilon || bounds.max.y > scene.container.height_m + epsilon) {
      throw new Error("Keep the whole fluid shape inside the tank. Reduce its size or drop height.");
    }
    return { patches: [], tone: "fluid", action: { kind: "fluid", edit }, highlight: fluidShapeHighlight(edit),
      caption: `RELEASE TO ${edit.operation === "add" ? "ADD" : "REMOVE"} WATER · ${shape.toUpperCase()}` };
  } };
}
