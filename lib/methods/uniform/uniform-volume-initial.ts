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
  const buried = buriedVertices(scene, dimensions, sliceZ);
  for (let outputZ=0; outputZ<(sliceZ === undefined ? nz+1 : 1); outputZ++) for (let y=0; y<=ny; y++) for (let x=0; x<=nx; x++) {
    const z = sliceZ ?? outputZ;
    const point = { x: (x/nx-0.5)*c.width_m, y: y/ny*c.height_m, z: (z/nz-0.5)*c.depth_m };
    const empty = Math.max(c.width_m,c.height_m,c.depth_m);
    // phi inside a solid is not state (uvBuried in uniform-volume.wgsl.ts): a
    // vertex with no open incident cell is air by construction, whatever
    // seeded body the analytic samples would put there.
    if (buried[x+(nx+1)*(y+(ny+1)*outputZ)]) { phi[x+(nx+1)*(y+(ny+1)*outputZ)] = empty; continue; }
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

/**
 * Vertices whose every in-range incident cell is closed by the solid world,
 * under the occupancy mask's rule (any solid fraction closes the cell). A
 * z slice reads the one cell layer the slice plane lies in.
 */
function buriedVertices(scene: SceneDescription, dimensions: readonly [number, number, number], sliceZ?: number): Uint8Array {
  const [nx, ny, nz] = dimensions;
  const layers = sliceZ === undefined ? nz+1 : 1;
  const buried = new Uint8Array((nx+1)*(ny+1)*layers);
  const world = solidWorldForScene(scene);
  const closed = new Uint8Array(nx*ny*nz);
  for (let z=0; z<nz; z++) for (let y=0; y<ny; y++) for (let x=0; x<nx; x++)
    closed[x+nx*(y+ny*z)] = sampleSolidWorld(world,[x,y,z]).solidFraction > 0 ? 1 : 0;
  for (let outputZ=0; outputZ<layers; outputZ++) for (let y=0; y<=ny; y++) for (let x=0; x<=nx; x++) {
    const zs = sliceZ === undefined ? [outputZ-1, outputZ] : [Math.floor(sliceZ)];
    let allClosed = true, any = false;
    for (const z of zs) for (const cy of [y-1, y]) for (const cx of [x-1, x]) {
      if (cx<0 || cy<0 || z<0 || cx>=nx || cy>=ny || z>=nz) continue;
      any = true;
      if (!closed[cx+nx*(cy+ny*z)]) { allClosed = false; }
    }
    buried[x+(nx+1)*(y+(ny+1)*outputZ)] = any && allClosed ? 1 : 0;
  }
  return buried;
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
