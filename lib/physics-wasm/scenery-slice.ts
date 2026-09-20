import type { SceneDescription } from "../core/model";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../core/voxel-environments";
import { svoDescriptorForEnvironmentProxy } from "../svo/features/scene-publication/svo-scene-primitives";
import { sampleSvoPrimitive } from "../svo/contracts/svo-primitive-abi";

/** Static scenery occupancy on an XY plane. Bounds cull work; the renderer's
 * actual shape sampler decides occupancy, including rotated and procedural props.
 * Sample bits are unioned before counting so overlapping props cannot overfill.
 */
export function scenerySliceFraction(
  scene: SceneDescription, dimensions: readonly [number, number, number], z_m: number,
): Float32Array {
  const [nx, ny] = dimensions;
  const hx = scene.container.width_m / nx, hy = scene.container.height_m / ny;
  const left = -scene.container.width_m / 2;
  const bits = new Uint8Array(nx * ny);
  const catalog = buildEnvironmentProxyCatalog(scene, scene.environment ?? "default");
  for (const primitive of environmentProxyPrimitives(catalog, false)) {
    const { min, max } = primitive.aabb_m;
    if (z_m < min.z || z_m > max.z) continue;
    const x0 = Math.max(0, Math.floor((min.x - left) / hx));
    const x1 = Math.min(nx, Math.ceil((max.x - left) / hx));
    const y0 = Math.max(0, Math.floor(min.y / hy));
    const y1 = Math.min(ny, Math.ceil(max.y / hy));
    const descriptor = svoDescriptorForEnvironmentProxy(primitive);
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
      const i = x + nx * y;
      for (let s = 0; s < 4; s++) {
        const bit = 1 << s;
        if (bits[i]! & bit) continue;
        const point = { x: left + (x + ((s & 1) ? 0.75 : 0.25)) * hx,
          y: (y + ((s & 2) ? 0.75 : 0.25)) * hy, z: z_m };
        if (sampleSvoPrimitive(descriptor, point).signedDistance_m <= 0) bits[i]! |= bit;
      }
    }
  }
  return Float32Array.from(bits, (mask) =>
    ((mask & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1)) / 4);
}

/** Centre by default; the garden's authored tree is the useful initial cut. */
export function uniformLabSlice(scene: SceneDescription, nz: number, requested_m?: number) {
  const tree = scene.scenery?.nodes.find((node) => node.id === "tree");
  const preferred = requested_m ?? (tree?.place?.units === "metres" ? tree.place.position?.z : undefined) ?? 0;
  if (!Number.isFinite(preferred)) throw new Error("Slice depth must be finite");
  const depth = scene.container.depth_m;
  const index = Math.max(0, Math.min(nz - 1, Math.floor((preferred / depth + 0.5) * nz)));
  return { index, z_m: ((index + 0.5) / nz - 0.5) * depth };
}
