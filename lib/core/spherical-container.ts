import type { SceneDescription, Vec3 } from "./model";

export interface SphericalContainerGeometry {
  readonly center_m: Vec3;
  readonly radius_m: number;
}

/** A spherical world container expressed in finest-lattice coordinates. */
export interface SphericalContainerFineGeometry {
  readonly kind: "sphere";
  readonly centerFine: readonly [number, number, number];
  /** Per-axis lattice radii. A metric sphere is generally an ellipsoid here. */
  readonly radiiFine: readonly [number, number, number];
}

export function sceneHasSphericalContainer(scene: Pick<SceneDescription, "container">): boolean {
  return scene.container.shape === "sphere";
}

/** The largest centred sphere that fits the authored domain extents. */
export function sphericalContainerGeometry(
  scene: Pick<SceneDescription, "container">,
): SphericalContainerGeometry | undefined {
  if (!sceneHasSphericalContainer(scene)) return undefined;
  const c = scene.container;
  const radius_m = 0.5 * Math.min(c.width_m, c.height_m, c.depth_m);
  return { center_m: { x: 0, y: 0.5 * c.height_m, z: 0 }, radius_m };
}

export function sphericalContainerFineGeometry(
  scene: Pick<SceneDescription, "container">,
  dimensions: readonly [number, number, number],
): SphericalContainerFineGeometry | undefined {
  const sphere = sphericalContainerGeometry(scene);
  if (!sphere) return undefined;
  const c = scene.container;
  const cell = [c.width_m / dimensions[0], c.height_m / dimensions[1],
    c.depth_m / dimensions[2]] as const;
  return {
    kind: "sphere",
    centerFine: [
      (sphere.center_m.x + 0.5 * c.width_m) / cell[0],
      sphere.center_m.y / cell[1],
      (sphere.center_m.z + 0.5 * c.depth_m) / cell[2],
    ],
    radiiFine: [sphere.radius_m / cell[0], sphere.radius_m / cell[1],
      sphere.radius_m / cell[2]],
  };
}

export function pointInsideSphericalContainerFine(
  geometry: SphericalContainerFineGeometry,
  point: readonly [number, number, number],
): boolean {
  let distance2 = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const radius = geometry.radiiFine[axis];
    if (!(radius > 0)) return false;
    const offset = (point[axis] - geometry.centerFine[axis]) / radius;
    distance2 += offset * offset;
  }
  return distance2 <= 1;
}

/** Eight-sample open-volume estimate, matching the uniform CM12 quadrature. */
export function sphericalContainerOpenFractionAtFineBox(
  geometry: SphericalContainerFineGeometry | undefined,
  center: readonly [number, number, number],
  widths: readonly [number, number, number],
): number {
  if (!geometry) return 1;
  let open = 0;
  for (let sample = 0; sample < 8; sample += 1) {
    const point = [0, 0, 0] as [number, number, number];
    for (let axis = 0; axis < 3; axis += 1) {
      point[axis] = center[axis]
        + ((sample & (1 << axis)) ? 0.4 : -0.4) * widths[axis];
    }
    if (pointInsideSphericalContainerFine(geometry, point)) open += 1;
  }
  return open / 8;
}

/** Four-sample transport aperture estimate on an axis-aligned face. */
export function sphericalContainerOpenFractionAtFineFace(
  geometry: SphericalContainerFineGeometry | undefined,
  center: readonly [number, number, number],
  axis: 0 | 1 | 2,
  tangentialWidths: readonly [number, number],
): number {
  if (!geometry) return 1;
  const tangents = axis === 0 ? [1, 2] as const
    : axis === 1 ? [0, 2] as const : [0, 1] as const;
  let open = 0;
  for (let sample = 0; sample < 4; sample += 1) {
    const point = [...center] as [number, number, number];
    point[tangents[0]] += ((sample & 1) ? 0.35 : -0.35) * tangentialWidths[0];
    point[tangents[1]] += ((sample & 2) ? 0.35 : -0.35) * tangentialWidths[1];
    if (pointInsideSphericalContainerFine(geometry, point)) open += 1;
  }
  return open / 4;
}

export type SphericalContainerBoxClassification = "inside" | "cut" | "outside";

/** Exact ellipsoid/AABB inside-outside classification in lattice coordinates. */
export function classifyFineBoxAgainstSphericalContainer(
  geometry: SphericalContainerFineGeometry | undefined,
  minimum: readonly [number, number, number],
  maximum: readonly [number, number, number],
): SphericalContainerBoxClassification {
  if (!geometry) return "inside";
  let nearest2 = 0, farthest2 = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    const center = geometry.centerFine[axis], radius = geometry.radiiFine[axis];
    const nearest = center < minimum[axis] ? minimum[axis] - center
      : center > maximum[axis] ? center - maximum[axis] : 0;
    const farthest = Math.max(Math.abs(minimum[axis] - center),
      Math.abs(maximum[axis] - center));
    nearest2 += (nearest / radius) ** 2;
    farthest2 += (farthest / radius) ** 2;
  }
  if (nearest2 > 1) return "outside";
  return farthest2 <= 1 ? "inside" : "cut";
}

export function pointInsideSceneContainer(
  scene: Pick<SceneDescription, "container">,
  point_m: Vec3,
): boolean {
  const sphere = sphericalContainerGeometry(scene);
  if (!sphere) return point_m.x >= -0.5 * scene.container.width_m
    && point_m.x <= 0.5 * scene.container.width_m
    && point_m.y >= 0 && point_m.y <= scene.container.height_m
    && point_m.z >= -0.5 * scene.container.depth_m
    && point_m.z <= 0.5 * scene.container.depth_m;
  return Math.hypot(
    point_m.x - sphere.center_m.x,
    point_m.y - sphere.center_m.y,
    point_m.z - sphere.center_m.z,
  ) <= sphere.radius_m;
}

/**
 * Eight-point cut-cell estimate matching the uniform shader's solid fraction
 * quadrature. Box containers have no embedded solid and therefore return one.
 */
export function sphericalContainerOpenFractionAtCell(
  scene: Pick<SceneDescription, "container">,
  x: number,
  y: number,
  z: number,
  dimensions: readonly [number, number, number],
): number {
  return sphericalContainerOpenFractionAtFineBox(
    sphericalContainerFineGeometry(scene, dimensions),
    [x + 0.5, y + 0.5, z + 0.5], [1, 1, 1],
  );
}
