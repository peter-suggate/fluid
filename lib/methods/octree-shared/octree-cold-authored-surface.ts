import type { SceneDescription } from "../../core/model";
import { initialFluidBrickComponentBounds, sceneDamBreakBox } from "../../core/initial-fluid";
import { sceneHasTerrain } from "../../core/terrain";

/** Construction-only analytic boxes carried beside the cold topology params. */
export const OCTREE_COLD_AUTHORED_SURFACE_BOX_CAPACITY = 8;
export const OCTREE_COLD_AUTHORED_SURFACE_PARAMS_BYTES =
  16 + OCTREE_COLD_AUTHORED_SURFACE_BOX_CAPACITY * 2 * 16;

export interface OctreeColdAuthoredSurfaceBox {
  readonly minimum: readonly [number, number, number];
  readonly maximum: readonly [number, number, number];
}

/**
 * Resolve the same rectangular brick components used by the one-time direct
 * nodal SDF bootstrap, but in exact finest-node coordinates.  Returning no
 * boxes is deliberate when the authored shape cannot be represented exactly:
 * the cold topology must not replace an L-shaped body with its bounding box.
 */
export function octreeColdAuthoredSurfaceBoxes(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  brickSize: number,
): readonly OctreeColdAuthoredSurfaceBox[] {
  if (scene.fluid.initialBrickSeedsAdditive || sceneHasTerrain(scene)) return [];
  if ((scene.fluid.initialBrickSeeds_m?.length ?? 0) === 0
      && scene.fluid.initialCondition === "dam-break") {
    const dam = sceneDamBreakBox(scene);
    return [{ minimum: [dam.min.x * dimensions[0], dam.min.y * dimensions[1],
      dam.min.z * dimensions[2]], maximum: [dam.max.x * dimensions[0],
      dam.max.y * dimensions[1], dam.max.z * dimensions[2]] }];
  }
  const components = initialFluidBrickComponentBounds(scene, dimensions, brickSize);
  if (!components || components.length > OCTREE_COLD_AUTHORED_SURFACE_BOX_CAPACITY) return [];
  const origin = [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m] as const;
  const spacing = [scene.container.width_m / dimensions[0],
    scene.container.height_m / dimensions[1],
    scene.container.depth_m / dimensions[2]] as const;
  const axis = ["x", "y", "z"] as const;
  return components.map((component) => ({
    minimum: axis.map((name, index) => Math.round(
      (component.minimum[name] - origin[index]!) / spacing[index]!,
    )) as [number, number, number],
    maximum: axis.map((name, index) => Math.round(
      (component.maximum[name] - origin[index]!) / spacing[index]!,
    )) as [number, number, number],
  }));
}

export function packOctreeColdAuthoredSurfaceBoxes(
  boxes: readonly OctreeColdAuthoredSurfaceBox[],
): ArrayBuffer {
  const data = new ArrayBuffer(OCTREE_COLD_AUTHORED_SURFACE_PARAMS_BYTES);
  new Uint32Array(data, 0, 4)[0] = Math.min(
    OCTREE_COLD_AUTHORED_SURFACE_BOX_CAPACITY, boxes.length,
  );
  const packed = new Float32Array(data, 16);
  boxes.slice(0, OCTREE_COLD_AUTHORED_SURFACE_BOX_CAPACITY).forEach((box, index) => {
    packed.set([...box.minimum, 0], index * 8);
    packed.set([...box.maximum, 0], index * 8 + 4);
  });
  return data;
}

export interface OctreeColdAuthoredSurfaceInterval {
  readonly minimumPhi: number;
  readonly maximumPhi: number;
  readonly crossesOrTouchesSurface: boolean;
}

function boxSignedDistanceAtNode(
  point: readonly [number, number, number],
  box: OctreeColdAuthoredSurfaceBox,
  spacing: readonly [number, number, number],
): number {
  const q = point.map((value, axis) => (
    Math.abs(value - 0.5 * (box.minimum[axis]! + box.maximum[axis]!))
      - 0.5 * (box.maximum[axis]! - box.minimum[axis]!)
  ) * spacing[axis]!) as [number, number, number];
  return Math.hypot(Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0))
    + Math.min(Math.max(q[0], q[1], q[2]), 0);
}

/** CPU mirror of the cold WGSL classifier, used by focused topology tests. */
export function octreeColdAuthoredSurfaceInterval(
  boxes: readonly OctreeColdAuthoredSurfaceBox[],
  origin: readonly [number, number, number],
  size: number,
  spacing: readonly [number, number, number],
): OctreeColdAuthoredSurfaceInterval | undefined {
  if (boxes.length === 0) return undefined;
  let minimumPhi = Number.POSITIVE_INFINITY;
  let maximumPhi = Number.NEGATIVE_INFINITY;
  for (let corner = 0; corner < 8; corner += 1) {
    const point = [origin[0] + ((corner & 1) ? size : 0),
      origin[1] + ((corner & 2) ? size : 0),
      origin[2] + ((corner & 4) ? size : 0)] as const;
    let phi = Number.POSITIVE_INFINITY;
    for (const box of boxes) phi = Math.min(phi, boxSignedDistanceAtNode(point, box, spacing));
    minimumPhi = Math.min(minimumPhi, phi);
    maximumPhi = Math.max(maximumPhi, phi);
  }
  const high = origin.map((value) => value + size) as [number, number, number];
  const crossesOrTouchesSurface = boxes.some((box) => {
    const overlapsClosure = origin.every((value, axis) =>
      high[axis]! >= box.minimum[axis]! && value <= box.maximum[axis]!);
    const strictlyInside = origin.every((value, axis) =>
      value > box.minimum[axis]! && high[axis]! < box.maximum[axis]!);
    return overlapsClosure && !strictlyInside;
  });
  return {
    minimumPhi: crossesOrTouchesSurface ? Math.min(minimumPhi, 0) : minimumPhi,
    maximumPhi: crossesOrTouchesSurface ? Math.max(maximumPhi, 0) : maximumPhi,
    crossesOrTouchesSurface,
  };
}

export function octreeColdAuthoredSurfaceWouldRefine(
  boxes: readonly OctreeColdAuthoredSurfaceBox[],
  origin: readonly [number, number, number],
  size: number,
  spacing: readonly [number, number, number],
  finestSurfaceCellSize = 1,
): boolean {
  if (size <= finestSurfaceCellSize) return false;
  const interval = octreeColdAuthoredSurfaceInterval(boxes, origin, size, spacing);
  return interval?.crossesOrTouchesSurface ?? false;
}
