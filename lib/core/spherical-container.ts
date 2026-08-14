import type { SceneDescription, Vec3 } from "./model";

export interface SphericalContainerGeometry {
  readonly center_m: Vec3;
  readonly radius_m: number;
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
  const sphere = sphericalContainerGeometry(scene);
  if (!sphere) return 1;
  const c = scene.container;
  const h = [c.width_m / dimensions[0], c.height_m / dimensions[1], c.depth_m / dimensions[2]] as const;
  const center = {
    x: -0.5 * c.width_m + (x + 0.5) * h[0],
    y: (y + 0.5) * h[1],
    z: -0.5 * c.depth_m + (z + 0.5) * h[2],
  };
  let open = 0;
  for (let corner = 0; corner < 8; corner += 1) {
    const point = {
      x: center.x + ((corner & 1) ? 0.4 : -0.4) * h[0],
      y: center.y + ((corner & 2) ? 0.4 : -0.4) * h[1],
      z: center.z + ((corner & 4) ? 0.4 : -0.4) * h[2],
    };
    if (pointInsideSceneContainer(scene, point)) open += 1;
  }
  return open / 8;
}
