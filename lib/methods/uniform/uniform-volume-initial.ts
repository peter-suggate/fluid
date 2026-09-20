import { initialHeightFieldHeight } from "../../core/initial-height-field";
import { damBreakSignedDistanceAtNode, initialFluidBrickSignedDistanceAtNode,
  initialLiquidVolumesSignedDistance, baseInitialLiquidFractionAtCell, damBreakBoxContains, initialLiquidFractionAtCell, sceneDamBreakBox } from "../../core/initial-fluid";
import { terrainColumnHeights } from "../../core/terrain";
import { sampleSolidWorld, solidWorldForScene } from "../../core/solid-world";
import type { SceneDescription } from "../../core/model";

/** Construction-only analytic samples, in metres, on shared lattice vertices. */
export function uniformVolumeInitialPhi(scene: SceneDescription, dimensions: readonly [number, number, number], sliceZ?: number): Float32Array {
  const [nx, ny, nz] = dimensions;
  const c = scene.container;
  const phi = new Float32Array((nx+1)*(ny+1)*(sliceZ === undefined ? nz+1 : 1));
  for (let outputZ=0; outputZ<(sliceZ === undefined ? nz+1 : 1); outputZ++) for (let y=0; y<=ny; y++) for (let x=0; x<=nx; x++) {
    const z = sliceZ ?? outputZ;
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
    phi[x+(nx+1)*(y+(ny+1)*outputZ)] = Number.isFinite(value) ? value : Math.max(c.width_m,c.height_m,c.depth_m);
  }
  return phi;
}

/** Shared initial conservative field; keeps the density comparison lane's seed rule. */
export function uniformInitialVolume(scene: SceneDescription, dimensions: readonly [number, number, number], geometric = true, sliceZ?: number) {
  const [nx,ny,nz]=dimensions;
  const c=scene.container;
  const terrain=terrainColumnHeights(scene,nx,nz);
  const cellHeight=c.height_m/ny;
  const volume=new Float32Array(nx*ny*(sliceZ === undefined ? nz : 1));
  const solidWorld=solidWorldForScene(scene);
  const dam=sceneDamBreakBox(scene);
  let initial=0;
  const wetMinimum=[nx,ny,nz];
  const wetMaximum=[0,0,0];
  for(let outputZ=0;outputZ<(sliceZ === undefined ? nz : 1);outputZ++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++) {
    const z=sliceZ ?? outputZ;
    const aboveGround=(y+0.5)*cellHeight>terrain[x+nx*z];
    const solidOpen=1-sampleSolidWorld(solidWorld,[x,y,z]).solidFraction;
    const base=geometric?baseInitialLiquidFractionAtCell(scene,x,y,z,dimensions):scene.fluid.initialCondition==="dam-break"
      ?damBreakBoxContains(dam,(x+0.5)/nx,(y+0.5)/ny,(z+0.5)/nz):(y+0.5)/ny<=c.fillFraction;
    const liquidFraction=aboveGround?initialLiquidFractionAtCell(scene,x,y,z,dimensions,base):0;
    const density=Math.min(solidOpen,liquidFraction);
    volume[x+nx*(y+ny*outputZ)]=density;
    initial+=density;
    if(density>1e-5){
      wetMinimum[0]=Math.min(wetMinimum[0]!,x);wetMinimum[1]=Math.min(wetMinimum[1]!,y);wetMinimum[2]=Math.min(wetMinimum[2]!,z);
      wetMaximum[0]=Math.max(wetMaximum[0]!,x+1);wetMaximum[1]=Math.max(wetMaximum[1]!,y+1);wetMaximum[2]=Math.max(wetMaximum[2]!,z+1);
    }
  }
  return {volume,terrain,initial,wetMinimum,wetMaximum,dam};
}
