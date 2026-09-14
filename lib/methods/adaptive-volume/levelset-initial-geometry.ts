import type { InitialLiquidVolume, SceneDescription, Vec3 } from "../../core/model";
import { initialFluidBrickComponentBounds, sceneDamBreakBox } from "../../core/initial-fluid";

const number = (value: number): string => {
  if (!Number.isFinite(value)) throw new RangeError("Initial level-set geometry must be finite");
  const literal = Math.fround(value).toString();
  return /[.e]/i.test(literal) ? literal : `${literal}.0`;
};
const vector = (value: Vec3): string =>
  `vec3f(${number(value.x)},${number(value.y)},${number(value.z)})`;

function halfSpaceBoxExpression(
  minimum: Vec3,
  maximum: Vec3,
  omitWallPlane: (axis: keyof Vec3, side: "minimum" | "maximum", value: number) => boolean,
  far: number,
): string {
  const distances: string[] = [];
  for (const axis of ["x", "y", "z"] as const) {
    if (!omitWallPlane(axis, "minimum", minimum[axis])) {
      distances.push(`(${number(minimum[axis])}-point.${axis})`);
    }
    if (!omitWallPlane(axis, "maximum", maximum[axis])) {
      distances.push(`(point.${axis}-${number(maximum[axis])})`);
    }
  }
  if (distances.length === 0) return number(-far);
  const maximumDistance = distances.reduce((a, b) => `max(${a},${b})`);
  const squaredOutside = distances
    .map(distance => `(max(${distance},0.0)*max(${distance},0.0))`).join("+");
  return `(sqrt(${squaredOutside})+min(${maximumDistance},0.0))`;
}

function volumeExpression(
  volume: InitialLiquidVolume,
  container: SceneDescription["container"],
): string {
  if (volume.shape === "box") {
    const limits = {
      x: [-container.width_m / 2, container.width_m / 2],
      y: [0, container.height_m],
      z: [-container.depth_m / 2, container.depth_m / 2],
    } as const;
    const far = Math.max(container.width_m, container.height_m, container.depth_m);
    const tolerance = 1e-9 * Math.max(1, far);
    return halfSpaceBoxExpression(volume.min_m, volume.max_m, (axis, side, value) => {
      if ((container.shape ?? "box") !== "box") return false;
      if (axis === "y" && side === "maximum" && container.top === "open") return false;
      const wall = limits[axis][side === "minimum" ? 0 : 1];
      return Math.abs(value - wall) <= tolerance;
    }, far);
  }
  const delta = `(point-${vector(volume.center_m)})`;
  if (volume.shape === "sphere") return `(length(${delta})-${number(volume.radius_m)})`;
  if (volume.shape === "cylinder") {
    return `lsvInitialCylinder(${delta},${number(volume.radius_m)},${number(volume.halfHeight_m)})`;
  }
  if (volume.shape === "torus") {
    return `(length(vec2f(length(${delta}.xz)-${number(volume.radius_m)},${delta}.y))-${number(volume.tubeRadius_m)})`;
  }
  const n = volume.outwardNormal;
  const magnitude = Math.hypot(n.x, n.y, n.z);
  const normal = magnitude > 1e-12
    ? { x: n.x / magnitude, y: n.y / magnitude, z: n.z / magnitude }
    : { x: 0, y: 1, z: 0 };
  return `max(length(${delta})-${number(volume.radius_m)},dot(${delta},${vector(normal)}))`;
}

/** Authored geometry is used only when the first adaptive phi generation is seeded.
 * No full-domain field is allocated, and evolving phi never reads this function. */
export function createInitialLevelSetGeometryWGSL(
  scene: SceneDescription,
  dimensions: readonly [number, number, number],
  finestCellSize_m: number,
): string {
  if (!(finestCellSize_m > 0) || !Number.isFinite(finestCellSize_m)
    || dimensions.some(n => !Number.isSafeInteger(n) || n <= 0)) {
    throw new RangeError("Initial level-set lattice must have positive finite dimensions and spacing");
  }
  const c = scene.container;
  const far = Math.max(c.width_m, c.height_m, c.depth_m);
  let base = number(far);
  if (c.fillFraction > 0) {
    if (scene.fluid.initialCondition === "tank-fill") {
      base = `(point.y-${number(c.fillFraction * c.height_m)})`;
    } else {
      const box = sceneDamBreakBox(scene);
      const extents = { x: c.width_m, y: c.height_m, z: c.depth_m };
      const origin = { x: -c.width_m / 2, y: 0, z: -c.depth_m / 2 };
      const distances: string[] = [];
      for (const axis of ["x", "y", "z"] as const) {
        if (box.min[axis] > 1e-9) distances.push(`(${number(origin[axis] + box.min[axis] * extents[axis])}-point.${axis})`);
        if (box.max[axis] < 1 - 1e-9 || (axis === "y" && c.top !== "closed")) {
          distances.push(`(point.${axis}-${number(origin[axis] + box.max[axis] * extents[axis])})`);
        }
      }
      if (!distances.length) base = number(-far);
      else {
        const maximum = distances.reduce((a,b) => `max(${a},${b})`);
        const squared = distances.map(d => `(max(${d},0.0)*max(${d},0.0))`).join("+");
        base = `(sqrt(${squared})+min(${maximum},0.0))`;
      }
    }
  }
  const height = scene.fluid.initialHeightField;
  if (height) {
    const level = height.kind === "cosine"
      ? `(${number(height.baseHeight_m)}+${number(height.amplitude_m)}*cos(${number(2 * Math.PI / height.wavelength_m)}*(point.x-${number(height.originX_m)})))`
      : `(${number(height.baseHeight_m)}+${number(height.curvatureX_mInv)}*(point.x-${number(height.center_m.x)})*(point.x-${number(height.center_m.x)})+${number(height.curvatureZ_mInv)}*(point.z-${number(height.center_m.z)})*(point.z-${number(height.center_m.z)}))`;
    base = `(point.y-${level})`;
  }
  const bricks = initialFluidBrickComponentBounds(scene, dimensions, scene.voxelDomain.brickSize_cells);
  if (bricks !== undefined) {
    const expression = bricks.map(b => `lsvInitialBox(point,${vector(b.minimum)},${vector(b.maximum)})`)
      .reduce((a,b) => `min(${a},${b})`, number(far));
    base = scene.fluid.initialBrickSeedsAdditive ? `min(${base},${expression})` : expression;
  }
  for (const volume of scene.fluid.initialLiquidVolumes ?? []) {
    base = `min(${base},${volumeExpression(volume, c)})`;
  }
  if (scene.systems?.fluid === false) base = number(far);
  return /* wgsl */ `
fn lsvInitialBox(point:vec3f,minimum:vec3f,maximum:vec3f)->f32{
  let q=max(minimum-point,point-maximum);
  return length(max(q,vec3f(0.0)))+min(max(q.x,max(q.y,q.z)),0.0);
}
fn lsvInitialCylinder(delta:vec3f,radius:f32,halfHeight:f32)->f32{
  let q=vec2f(length(delta.xy)-radius,abs(delta.z)-halfHeight);
  return length(max(q,vec2f(0.0)))+min(max(q.x,q.y),0.0);
}
fn lsvAuthoredPhi(positionFine:vec3f)->f32{
  let point=vec3f(${number(-c.width_m / 2)},0.0,${number(-c.depth_m / 2)})
    +positionFine*vec3f(${number(c.width_m / dimensions[0])},${number(c.height_m / dimensions[1])},${number(c.depth_m / dimensions[2])});
  return ${base}/${number(finestCellSize_m)};
}
`;
}
