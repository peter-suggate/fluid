import type { SceneDescription } from "./model";

/** A construction-time liquid surface in world metres, not a simulation constraint. */
export interface InitialLiquidHeightField {
  kind: "quadratic";
  baseHeight_m: number;
  center_m: { x: number; z: number };
  curvatureX_mInv: number;
  curvatureZ_mInv: number;
}

export function initialHeightFieldHeight(field: InitialLiquidHeightField, x: number, z: number): number {
  return field.baseHeight_m + field.curvatureX_mInv * (x - field.center_m.x) ** 2
    + field.curvatureZ_mInv * (z - field.center_m.z) ** 2;
}

/** Bounds for a convex quadratic over an axis-aligned horizontal footprint. */
export function initialHeightFieldRange(field: InitialLiquidHeightField,
  x0: number, x1: number, z0: number, z1: number): readonly [number, number] {
  const cx = Math.max(x0, Math.min(x1, field.center_m.x));
  const cz = Math.max(z0, Math.min(z1, field.center_m.z));
  return [initialHeightFieldHeight(field, cx, cz), Math.max(
    initialHeightFieldHeight(field, x0, z0), initialHeightFieldHeight(field, x0, z1),
    initialHeightFieldHeight(field, x1, z0), initialHeightFieldHeight(field, x1, z1))];
}

/** Same 8×8 area quadrature as the stationary-bowl Dawn fixture. Native coarse
 * cells are restrictions of these finest-cell volumes, never point samples. */
export function initialHeightFieldFractionAtCell(scene: SceneDescription,
  x: number, y: number, z: number, dimensions: readonly [number, number, number]): number | undefined {
  const field = scene.fluid.initialHeightField;
  if (!field) return undefined;
  if (scene.systems?.fluid === false || x < 0 || y < 0 || z < 0
    || x >= dimensions[0] || y >= dimensions[1] || z >= dimensions[2]) return 0;
  const c = scene.container;
  const hx = c.width_m / dimensions[0], hy = c.height_m / dimensions[1], hz = c.depth_m / dimensions[2];
  const x0 = -c.width_m / 2 + x * hx, z0 = -c.depth_m / 2 + z * hz, y0 = y * hy;
  const [low, high] = initialHeightFieldRange(field, x0, x0 + hx, z0, z0 + hz);
  if (y0 + hy <= low) return 1;
  if (y0 >= high) return 0;
  let fraction = 0;
  for (let iz = 0; iz < 8; iz++) for (let ix = 0; ix < 8; ix++) {
    const height = initialHeightFieldHeight(field, x0 + (ix + .5) * hx / 8, z0 + (iz + .5) * hz / 8);
    fraction += Math.max(0, Math.min(1, (height - y0) / hy)) / 64;
  }
  return fraction;
}
