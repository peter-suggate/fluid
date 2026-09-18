import { initialHeightFieldHeight } from "../../core/initial-height-field";
import { damBreakSignedDistanceAtNode, initialFluidBrickSignedDistanceAtNode,
  initialLiquidVolumesSignedDistance } from "../../core/initial-fluid";
import type { SceneDescription } from "../../core/model";

/** Construction-only analytic samples, in metres, on shared lattice vertices. */
export function uniformVolumeInitialPhi(scene: SceneDescription, dimensions: readonly [number, number, number]): Float32Array {
  const [nx, ny, nz] = dimensions;
  const c = scene.container;
  const phi = new Float32Array((nx+1)*(ny+1)*(nz+1));
  for (let z=0; z<=nz; z++) for (let y=0; y<=ny; y++) for (let x=0; x<=nx; x++) {
    const point = { x: (x/nx-0.5)*c.width_m, y: y/ny*c.height_m, z: (z/nz-0.5)*c.depth_m };
    const empty = Math.max(c.width_m,c.height_m,c.depth_m);
    const height = scene.fluid.initialHeightField;
    let base = height ? point.y-initialHeightFieldHeight(height, point.x, point.z)
      : c.fillFraction > 0 ? damBreakSignedDistanceAtNode(scene,x,y,z,dimensions)
        ?? (point.y-c.fillFraction*c.height_m) : empty;
    const brick = initialFluidBrickSignedDistanceAtNode(scene,x,y,z,dimensions);
    if (brick !== undefined) base = scene.fluid.initialBrickSeedsAdditive ? Math.min(base,brick) : brick;
    const value = scene.systems?.fluid === false ? empty
      : Math.min(base,initialLiquidVolumesSignedDistance(scene,point) ?? empty);
    phi[x+(nx+1)*(y+(ny+1)*z)] = Number.isFinite(value) ? value : Math.max(c.width_m,c.height_m,c.depth_m);
  }
  return phi;
}
